"""One local subprocess at a time, with durable logs and real manifest progress."""
import asyncio
import codecs
from contextlib import ExitStack, suppress
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import sys
import tempfile
import threading
from uuid import uuid4

from scripts.next import pipeline
from scripts.next.dashboard.data_status import inspect_data
from scripts.next.dashboard.models import DataStatusRequest, RunRequest
from scripts.next.dashboard.schema import resolve_settings

TERMINAL = {'complete', 'failed', 'cancelled'}
MAX_LOG_LINES = 500


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.tmp',
                                         dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class BusyError(ValueError):
    pass


class RunManager:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root.resolve()
        self.cache_root = self.repo_root / 'cache'
        self.storage = self.cache_root / '.dashboard'
        self.jobs: dict[str, dict] = {}
        self.processes: dict[str, asyncio.subprocess.Process] = {}
        self.tasks: dict[str, asyncio.Task] = {}
        self.lock = asyncio.Lock()
        self.persistence_lock = threading.RLock()
        self._restore()

    def _restore(self):
        if (self.cache_root.resolve() != self.cache_root
                or not self.storage.is_dir() or self.storage.is_symlink()):
            return
        for path in sorted(self.storage.glob('*.json')):
            try:
                job = json.loads(path.read_text())
                job_id = job['id']
                if not re.fullmatch(r'[0-9a-f]{32}', job_id) or path.stem != job_id:
                    continue
                if job.get('status') not in TERMINAL:
                    # Restart time is when the dashboard discovers the failure,
                    # not when analysis stopped. Never include offline time in
                    # an unfinished stage's elapsed duration.
                    for stage in job.get('stages', []):
                        if not isinstance(stage, dict):
                            continue
                        seconds = stage.get('seconds')
                        measured = (isinstance(seconds, (int, float))
                                    and not isinstance(seconds, bool)
                                    and math.isfinite(seconds) and seconds >= 0)
                        if (stage.get('status') == 'running' and not measured
                                and not stage.get('finished_at')):
                            stage['elapsed_unavailable'] = True
                    job.update(status='failed', finished_at=now(),
                               error='Dashboard stopped before this job finished. Check the run manifest before restarting.')
                    self._save(job)
                log = self.storage / f'{job_id}.log'
                if log.exists():
                    # Tail logs without reading a potentially huge decoding log into memory.
                    with log.open('rb') as handle:
                        handle.seek(max(0, log.stat().st_size - 128 * 1024))
                        job['logs'] = handle.read().decode('utf-8', errors='replace').splitlines()[-MAX_LOG_LINES:]
                self.jobs[job_id] = job
            except (OSError, ValueError, KeyError, TypeError):
                continue

    def _local_path(self, value):
        try:
            path = Path(value).expanduser()
        except RuntimeError as exc:
            raise ValueError(f'Cannot resolve the home directory in {value!r}. '
                             'Choose a local path or an existing user.') from exc
        return (path if path.is_absolute() else self.repo_root / path).resolve()

    def validate(self, request: RunRequest, job_id='preview'):
        cache_dir, data_dir = self._local_path(request.cache_dir), self._local_path(request.data_dir)
        # Cache writes must stay in this repository, including through existing symlinks.
        if self.cache_root.resolve() != self.cache_root or not cache_dir.is_relative_to(self.cache_root):
            raise ValueError('cache_dir must be inside this repository’s cache directory.')
        relative = cache_dir.relative_to(self.cache_root)
        if len(relative.parts) != 1 or any(part.startswith('.') for part in relative.parts):
            raise ValueError('Choose a single named run directory: cache/<run-name>, outside hidden dashboard files.')
        if cache_dir.exists() and not cache_dir.is_dir():
            raise ValueError('cache_dir already exists and is not a directory.')
        if cache_dir.exists() and any(cache_dir.iterdir()) and not request.allow_existing:
            raise ValueError('This cache directory is not empty. Enable reuse of existing outputs to run additional stages or resume.')
        if request.trust_unverified_legacy_results:
            if not request.allow_existing:
                raise ValueError('Trusting unverified legacy results requires reuse of existing outputs.')
            if not (cache_dir / 'decode' / 'decoding_confidence.pkl').is_file():
                raise ValueError('Trusting legacy results requires an existing run with decode/decoding_confidence.pkl.')
        # Existing symlinks can redirect scientific writers that predate this dashboard.
        if cache_dir.exists() and any(path.is_symlink() for path in cache_dir.rglob('*')):
            raise ValueError('The dashboard cannot write a run directory containing symlinks.')
        session_file = request.session_list_file
        if session_file is not None:
            session_path = self._local_path(session_file)
        resolved = resolve_settings(request, self.repo_root, cache_dir, data_dir)
        data_status = inspect_data(self.repo_root, DataStatusRequest(**request.model_dump(include={
            'data_dir', 'stages', 'session_list_file', 'settings', 'trust_unverified_legacy_results',
        })))
        if data_status['blocking']:
            raise ValueError(data_status['message'])
        settings_path = self.repo_root / 'configs' / 'next' / '.dashboard' / f'{job_id}.json'
        argv = [sys.executable, '-u', str(self.repo_root / 'scripts/next/pipeline.py'),
                '--settings', str(settings_path), '--data-dir', str(data_dir),
                '--cache-dir', str(cache_dir), '--stages', *request.stages,
                '--n-jobs', str(request.n_jobs), '--figure-font', request.figure_font,
                '--figure-formats', *request.figure_formats]
        if request.max_sessions_to_run is not None:
            argv.extend(['--max-sessions-to-run', str(request.max_sessions_to_run)])
        if session_file is not None:
            argv.extend(['--session-list-file', str(session_path)])
        if request.trust_unverified_legacy_results:
            argv.append('--trust-unverified-legacy-results')
        return {'valid': True, 'command': shlex.join(argv), 'resolved': resolved,
                'argv': argv, 'cache_dir': str(cache_dir), 'settings_path': settings_path}

    def _save(self, job):
        # REST polling uses worker threads while the subprocess uses the event
        # loop. Serialize the pair of writes so both records stay aligned.
        with self.persistence_lock:
            value = {k: v for k, v in job.items() if k != 'logs'}
            atomic_json(self.storage / f"{job['id']}.json", value)
            # Older jobs remain untouched in their run directories. Every newly
            # launched invocation has its own record, even when reusing a run folder.
            if job.get('run_record_path'):
                atomic_json(self._run_directory(job) / f"{job['id']}.json", value)

    def _run_directory(self, job):
        cache_dir = Path(job['cache_dir'])
        if (self.cache_root.resolve() != self.cache_root
                or not cache_dir.is_relative_to(self.cache_root)
                or len(cache_dir.relative_to(self.cache_root).parts) != 1
                or cache_dir.name.startswith('.')
                or cache_dir.resolve() != cache_dir):
            raise ValueError('Run record directory must stay inside its original cache folder.')
        directory = cache_dir / 'dashboard'
        if directory.resolve() != directory:
            raise ValueError('Run record directory must not redirect through a symlink.')
        return directory

    def log_path(self, job_id):
        """Return only a known job's regular full log, never a supplied path."""
        if job_id not in self.jobs:
            raise KeyError(job_id)
        path = self.storage / f'{job_id}.log'
        if (not re.fullmatch(r'[0-9a-f]{32}', job_id)
                or self.cache_root.resolve() != self.cache_root
                or self.storage.resolve() != self.storage
                or path.is_symlink() or not path.is_file()):
            raise FileNotFoundError('The full log file is not available for this job.')
        return path

    def _refresh_manifest(self, job):
        with self.persistence_lock:
            if job['status'] in TERMINAL:
                return
            self._read_manifest(job)

    def _read_manifest(self, job):
        try:
            path = Path(job['cache_dir']) / 'pipeline_manifest.json'
            manifest = json.loads(path.read_text())
            # A partial rerun starts with an older latest manifest: never show it as new progress.
            if manifest.get('invocation', {}).get('argv') != job.get('argv'):
                return
            manifest_id = manifest.get('run_id')
            entries = {entry['stage']: entry for entry in manifest.get('stages', [])}
            stages = [entries.get(stage, {'stage': stage, 'status': 'pending'})
                      for stage in job['requested_stages']]
            legacy_trust = manifest.get('legacy_trust')
            if (manifest_id != job.get('manifest_id') or stages != job['stages']
                    or legacy_trust != job.get('legacy_trust')):
                job.update(manifest_id=manifest_id, stages=stages, legacy_trust=legacy_trust)
                self._save(job)
        except (OSError, ValueError, KeyError, TypeError):
            pass

    def snapshot(self, job_id):
        if job_id not in self.jobs:
            raise KeyError(job_id)
        job = self.jobs[job_id]
        if job['status'] not in TERMINAL:
            self._refresh_manifest(job)
        return {key: value for key, value in job.items() if key not in {'argv'}}

    def list_jobs(self):
        return [self.snapshot(job['id']) for job in
                sorted(self.jobs.values(), key=lambda value: value['created_at'], reverse=True)]

    async def start(self, request: RunRequest):
        async with self.lock:
            if any(job['status'] not in TERMINAL for job in self.jobs.values()):
                raise BusyError('A pipeline is already running. Wait for it to finish or cancel it.')
            job_id = uuid4().hex
            plan = self.validate(request, job_id)
            # Store the exact sparse settings used by this invocation; manifests store resolved values.
            plan['settings_path'].parent.mkdir(parents=True, exist_ok=True)
            if not plan['settings_path'].parent.resolve().is_relative_to(self.repo_root / 'configs/next'):
                raise ValueError('Dashboard settings directory must not redirect outside configs/next.')
            if self.storage.is_symlink():
                raise ValueError('Dashboard storage must not be a symlink.')
            self.storage.mkdir(parents=True, exist_ok=True)
            run_directory = Path(plan['cache_dir']) / 'dashboard'
            job = dict(id=job_id, name=request.name, cache_dir=plan['cache_dir'],
                       status='queued', created_at=now(), started_at=None, finished_at=None,
                       command=plan['command'], argv=plan['argv'],
                       requested_stages=list(request.stages),
                       stages=[{'stage': stage, 'status': 'pending'} for stage in request.stages],
                       logs=[], exit_code=None, error=None, manifest_id=None,
                       request=request.model_dump(),
                       trust_unverified_legacy_results=request.trust_unverified_legacy_results,
                       run_record_path=str(run_directory / f'{job_id}.json'),
                       run_log_path=str(run_directory / f'{job_id}.log'),
                       run_settings_path=str(run_directory / f'{job_id}.settings.json'))
            run_directory = self._run_directory(job)
            # Create the empty full log before the job is visible, including for
            # jobs cancelled while queued or unable to launch a subprocess.
            new_files = [plan['settings_path'], self.storage / f'{job_id}.json',
                         self.storage / f'{job_id}.log',
                         *(Path(job[key]) for key in ('run_record_path', 'run_log_path', 'run_settings_path'))]
            if any(path.exists() or path.is_symlink() for path in new_files):
                raise ValueError('A dashboard job with this identifier already exists.')
            try:
                run_directory.mkdir(parents=True, exist_ok=True)
                atomic_json(plan['settings_path'], request.settings)
                atomic_json(Path(job['run_settings_path']), request.settings)
                (self.storage / f'{job_id}.log').touch(exist_ok=False)
                Path(job['run_log_path']).touch(exist_ok=False)
                self._save(job)
            except BaseException:
                for path in new_files:
                    with suppress(OSError):
                        path.unlink(missing_ok=True)
                for directory in (run_directory, run_directory.parent):
                    with suppress(OSError):
                        directory.rmdir()  # Remove only empty directories.
                raise
            self.jobs[job_id] = job
            self.tasks[job_id] = asyncio.create_task(self._execute(job_id), name=f'pipeline-{job_id}')
            return self.snapshot(job_id)

    def _append_log(self, job, line):
        # Keep the live channel bounded; the full stream remains in the .log file.
        job['logs'].append(line[-16000:])
        del job['logs'][:-MAX_LOG_LINES]

    async def _execute(self, job_id):
        job = self.jobs[job_id]
        try:
            if job['status'] == 'cancelling':
                return
            environment = dict(os.environ, PYTHONUNBUFFERED='1', MPLBACKEND='Agg')
            environment.setdefault('MPLCONFIGDIR', str(self.storage / 'matplotlib'))
            process = await asyncio.create_subprocess_exec(
                *job['argv'], cwd=self.repo_root, env=environment,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
            self.processes[job_id] = process
            job.update(started_at=now())
            if job['status'] != 'cancelling':
                job['status'] = 'running'
            self._save(job)
            # Cancellation can arrive while the subprocess is being created.
            if job['status'] == 'cancelling':
                self._signal_group(process, signal.SIGINT)
            pending, decoder = '', codecs.getincrementaldecoder('utf-8')(errors='replace')
            with ExitStack() as stack:
                log_paths = [self.storage / f'{job_id}.log',
                             self._run_directory(job) / f'{job_id}.log']
                logs = [stack.enter_context(path.open('ab')) for path in log_paths]
                while chunk := await process.stdout.read(8192):
                    for log in logs:
                        log.write(chunk)
                        log.flush()
                    pending += decoder.decode(chunk).replace('\r', '\n')
                    lines = pending.split('\n')
                    pending = lines.pop()
                    for line in lines:
                        if line:
                            self._append_log(job, line)
                    if len(pending) > 16000:
                        self._append_log(job, pending)
                        pending = ''
                pending += decoder.decode(b'', final=True)
                if pending:
                    self._append_log(job, pending)
            job['exit_code'] = await process.wait()
            self._refresh_manifest(job)
            if job['status'] != 'cancelling':
                job['status'] = 'complete' if process.returncode == 0 else 'failed'
                if process.returncode != 0:
                    job['error'] = f'Pipeline exited with code {process.returncode}. See the processing log.'
        except BaseException as exc:
            # A logging/metadata failure must not leave untracked analysis workers running.
            process = self.processes.get(job_id)
            if process is not None:
                self._signal_group(process, signal.SIGTERM)
                try:
                    await asyncio.wait_for(process.communicate(), timeout=3)
                except asyncio.TimeoutError:
                    self._signal_group(process, signal.SIGKILL)
                    await process.communicate()
                job['exit_code'] = process.returncode
            if isinstance(exc, asyncio.CancelledError):
                job.update(status='cancelling', error='Dashboard stopped before this job finished.')
            elif job['status'] != 'cancelling':
                job.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            message = f"Dashboard: {type(exc).__name__}: {exc}"
            self._append_log(job, message)
            # Persist launch/metadata failures as well as subprocess output.
            # A failed log destination must not prevent recording the other.
            for path in (self.storage / f'{job_id}.log', Path(job['run_log_path'])):
                with suppress(OSError):
                    with path.open('ab') as log:
                        log.write((message + '\n').encode('utf-8', errors='replace'))
                        log.flush()
        finally:
            if job['status'] == 'cancelling':
                job['status'] = 'cancelled'
            job['finished_at'] = now()
            try:
                self._save(job)
            except OSError as exc:
                # Keep the failure visible through the API even when disk persistence fails.
                job['error'] = f"{job.get('error') or ''} Job state could not be saved: {exc}".strip()
            finally:
                self.processes.pop(job_id, None)

    @staticmethod
    def _signal_group(process, sig):
        with suppress(ProcessLookupError):
            os.killpg(process.pid, sig)

    async def cancel(self, job_id):
        if job_id not in self.jobs:
            raise KeyError(job_id)
        job = self.jobs[job_id]
        if job['status'] in TERMINAL:
            return self.snapshot(job_id)
        job.update(status='cancelling', error='Cancelled by the user.')
        process = self.processes.get(job_id)
        task = self.tasks.get(job_id)
        if process is not None:
            self._signal_group(process, signal.SIGINT)
        if task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=5)
            except asyncio.TimeoutError:
                process = self.processes.get(job_id)
                if process is not None:
                    self._signal_group(process, signal.SIGTERM)
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=3)
                except asyncio.TimeoutError:
                    if process is not None:
                        self._signal_group(process, signal.SIGKILL)
                    await asyncio.shield(task)
        job['finished_at'] = job['finished_at'] or now()
        self._save(job)
        return self.snapshot(job_id)

    async def close(self):
        for job_id in list(self.tasks):
            if self.jobs[job_id]['status'] not in TERMINAL:
                await self.cancel(job_id)
