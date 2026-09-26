"""One local subprocess at a time, with durable logs and real manifest progress."""
import asyncio
import codecs
from contextlib import suppress
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import signal
import sys
from uuid import uuid4

from scripts.next import pipeline
from scripts.next.dashboard.models import RunRequest
from scripts.next.dashboard.schema import resolve_settings

TERMINAL = {'complete', 'failed', 'cancelled'}
MAX_LOG_LINES = 500


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


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
                    job.update(status='failed', finished_at=now(),
                               error='Dashboard stopped before this job finished. Check the run manifest before restarting.')
                    atomic_json(path, job)
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
        path = Path(value).expanduser()
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
        # Existing symlinks can redirect scientific writers that predate this dashboard.
        if cache_dir.exists() and any(path.is_symlink() for path in cache_dir.rglob('*')):
            raise ValueError('The dashboard cannot write a run directory containing symlinks.')
        if not data_dir.is_dir():
            raise ValueError(f'data_dir does not exist or is not a directory: {data_dir}')
        session_file = request.session_list_file
        if session_file is not None:
            session_path = self._local_path(session_file)
            if not session_path.is_file():
                raise ValueError(f'session_list_file does not exist: {session_path}')
        # Validate session selection without loading potentially large MAT files.
        if any(stage in request.stages for stage in ('select', 'decode')):
            sessions = {path.stem for path in data_dir.glob('*.mat')}
            if not sessions:
                raise ValueError(f'No .mat session files in {data_dir}.')
            if session_file is not None:
                requested = {line.split('#', 1)[0].strip() for line in session_path.read_text().splitlines()}
                if not (sessions & requested):
                    raise ValueError('Session list does not select any available .mat sessions.')
        resolved = resolve_settings(request, self.repo_root, cache_dir, data_dir)
        settings_path = self.repo_root / 'configs' / 'next' / '.dashboard' / f'{job_id}.json'
        argv = [sys.executable, '-u', str(self.repo_root / 'scripts/next/pipeline.py'),
                '--settings', str(settings_path), '--data-dir', str(data_dir),
                '--cache-dir', str(cache_dir), '--stages', *request.stages,
                '--n-jobs', str(request.n_jobs), '--figure-formats', *request.figure_formats]
        if request.max_sessions_to_run is not None:
            argv.extend(['--max-sessions-to-run', str(request.max_sessions_to_run)])
        if session_file is not None:
            argv.extend(['--session-list-file', str(session_path)])
        return {'valid': True, 'command': shlex.join(argv), 'resolved': resolved,
                'argv': argv, 'cache_dir': str(cache_dir), 'settings_path': settings_path}

    def _save(self, job):
        atomic_json(self.storage / f"{job['id']}.json", {k: v for k, v in job.items() if k != 'logs'})

    def _refresh_manifest(self, job):
        try:
            path = Path(job['cache_dir']) / 'pipeline_manifest.json'
            manifest = json.loads(path.read_text())
            # A partial rerun starts with an older latest manifest: never show it as new progress.
            if manifest.get('invocation', {}).get('argv') != job.get('argv'):
                return
            job['manifest_id'] = manifest.get('run_id')
            entries = {entry['stage']: entry for entry in manifest.get('stages', [])}
            job['stages'] = [entries.get(stage, {'stage': stage, 'status': 'pending'})
                             for stage in job['requested_stages']]
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
            atomic_json(plan['settings_path'], request.settings)
            self.storage.mkdir(parents=True, exist_ok=True)
            job = dict(id=job_id, name=request.name, cache_dir=plan['cache_dir'],
                       status='queued', created_at=now(), started_at=None, finished_at=None,
                       command=plan['command'], argv=plan['argv'],
                       requested_stages=list(request.stages),
                       stages=[{'stage': stage, 'status': 'pending'} for stage in request.stages],
                       logs=[], exit_code=None, error=None, manifest_id=None)
            self._save(job)
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
            with (self.storage / f'{job_id}.log').open('ab') as log:
                while chunk := await process.stdout.read(8192):
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
            self._append_log(job, f"Dashboard: {type(exc).__name__}: {exc}")
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
