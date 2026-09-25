"""Persistent invocation records plus an atomic latest-run view."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sys
import tempfile
from uuid import uuid4

from scripts.next.common import json_value


def utc_now():
    return datetime.now(timezone.utc)


def invocation_context(argv=None):
    """Keep CLI argument values exactly, without claiming to recover shell syntax.

    The CLI passes sys.orig_argv to retain interpreter flags and -m invocation.
    Python API calls have no pipeline command; never record their host's argv.
    """
    arguments = list(argv) if argv is not None else None
    return {
        'source': 'cli' if arguments is not None else 'programmatic',
        'argv': arguments,
        'command': shlex.join(arguments) if arguments is not None else None,
        'cwd': str(Path.cwd()),
        'python_executable': sys.executable,
    }


def _atomic_write(path, payload):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class RunManifest:
    """Update only this invocation's history file, preserving earlier records."""

    def __init__(self, cache_dir, settings, runner_config, *, invocation=None):
        self.latest = Path(cache_dir) / 'pipeline_manifest.json'
        history = Path(cache_dir) / 'manifests'
        history.mkdir(parents=True, exist_ok=True)
        self._preserve_previous(history)
        started = utc_now()
        run_id = f'{started:%Y%m%dT%H%M%S.%fZ}_{uuid4().hex}'
        self.path = history / f'{run_id}.json'
        self.record = {
            'run_id': run_id,
            'started_at': started.isoformat(),
            'finished_at': None,
            'status': 'running',
            'runner_config': runner_config,
            'invocation': invocation if invocation is not None else invocation_context(),
            'settings': settings,
            'stages': [],
        }
        self.save()

    def _preserve_previous(self, history):
        """Archive old-format or orphaned latest manifests before replacing them."""
        if not self.latest.exists():
            return
        payload = self.latest.read_bytes()
        try:
            previous = json.loads(payload)
        except (ValueError, UnicodeDecodeError):
            previous = {}
        run_id = previous.get('run_id') if isinstance(previous, dict) else None
        if isinstance(run_id, str) and re.fullmatch(r'\d{8}T\d{6}\.\d{6}Z_[0-9a-f]{32}', run_id):
            archived = history / f'{run_id}.json'
            if archived.exists() and archived.read_bytes() == payload:
                return
        # Keep the original bytes, including old records without timestamp fields.
        archived = history / f'prior-{hashlib.sha256(payload).hexdigest()}.json'
        if not archived.exists():
            _atomic_write(archived, payload)

    def save(self):
        payload = (json.dumps(self.record, indent=2, default=json_value) + '\n').encode()
        # Preserve the record before updating the replaceable convenience view.
        _atomic_write(self.path, payload)
        _atomic_write(self.latest, payload)

    def finish(self, status):
        self.record.update(status=status, finished_at=utc_now().isoformat())
        self.save()
