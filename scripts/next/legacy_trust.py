"""Explicit, run-scoped manual trust for unverified legacy decoding results.

An audit entry records the user's decision, not a claim that legacy estimates
were generated with current settings, source, or package versions. No cache key
is rewritten and no recorded trust decision authorizes a later invocation.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import re
from typing import Callable
import warnings


_ACTIVE_TRUST: ContextVar[LegacyTrustState | None] = ContextVar('next_legacy_trust', default=None)


def _input_checksum(path):
    path = Path(path).expanduser().resolve()
    digest, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
            size += len(block)
        after = os.fstat(stream.fileno())
    if (size != before.st_size
            or (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            != (after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
        raise OSError(f'Input changed while recording manual legacy trust: {path}')
    return {'path': str(path), 'sha256': digest.hexdigest(), 'size_bytes': size}


class LegacyTrustState:
    def __init__(self, cache_dir, enabled=False, on_update: Callable[[dict], None] | None = None):
        if not isinstance(enabled, bool):
            raise ValueError('Manual legacy trust must be explicitly true or false.')
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.enabled = enabled
        self.on_update = on_update
        self.stage: str | None = None
        self._events = []
        self._seen = set()
        self._warned = set()

    def set_stage(self, stage: str):
        if not isinstance(stage, str) or not stage.strip():
            raise ValueError('Manual legacy trust requires an explicit analysis stage.')
        self.stage = stage.strip()

    @property
    def audit(self):
        return {'enabled': self.enabled, 'manual_trust_used': bool(self._events),
                'events': deepcopy(self._events)}

    def record(self, original_fingerprint, selection_path, data_path, reason: Exception) -> bool:
        """Audit an eligible exception before allowing its legacy result to be used."""
        if (not self.enabled or not isinstance(original_fingerprint, str)
                or not re.fullmatch(r'[0-9a-f]{64}', original_fingerprint)):
            return False
        expected = self.cache_dir / 'select' / 'cell_screening.pkl'
        # Do not let a nested invocation or a redirected selection cache extend
        # this explicit decision to another run's inputs.
        if (expected.resolve() != expected
                or Path(selection_path).expanduser().resolve() != expected):
            return False
        if not isinstance(self.stage, str) or not self.stage.strip():
            raise ValueError('Set the analysis stage before using manual legacy trust.')
        selection, data = _input_checksum(selection_path), _input_checksum(data_path)
        session = Path(data['path']).stem
        reason_text, reason_type = str(reason), type(reason).__name__
        identity = (self.stage, session, original_fingerprint,
                    selection['path'], selection['sha256'], data['path'], data['sha256'],
                    reason_type, reason_text)
        if identity in self._seen:
            return True
        event = {
            'stage': self.stage, 'session': session,
            'original_fingerprint': original_fingerprint,
            'reason': reason_text, 'reason_type': reason_type,
            'recorded_at': datetime.now(timezone.utc).isoformat(),
            'manual_trust': True, 'verification_status': 'unverified-legacy',
            'original_runtime_versions': 'unknown',
            'current_input_checksums': {'selection': selection, 'data': data},
        }
        proposed = {'enabled': True, 'manual_trust_used': True,
                    'events': deepcopy([*self._events, event])}
        if self.on_update is not None:
            # Persistence failure must abort the decision, not silently authorize
            # scientific use without the promised manifest audit.
            self.on_update(proposed)
        self._events.append(event)
        self._seen.add(identity)
        warning_identity = (self.stage, session)
        if warning_identity not in self._warned:
            warnings.warn(
                f'{self.stage}: manually trusting unverified legacy decoding results for '
                f'{session}. Original fingerprints are retained; compatibility and original '
                f'package versions are not established. Verification failed: {reason_text}',
                RuntimeWarning, stacklevel=3,
            )
            self._warned.add(warning_identity)
        return True


def current_legacy_trust() -> LegacyTrustState | None:
    """Return this invocation's context, never a decision persisted by an older run."""
    return _ACTIVE_TRUST.get()


def record_legacy_trust(original_fingerprint, selection_path, data_path, reason: Exception) -> bool:
    state = current_legacy_trust()
    return state.record(original_fingerprint, selection_path, data_path, reason) if state is not None else False


@contextmanager
def legacy_trust_context(cache_dir, enabled=False, on_update=None):
    """Isolate manual trust and its audit to one invocation and selected run root."""
    state = LegacyTrustState(cache_dir, enabled, on_update)
    token = _ACTIVE_TRUST.set(state)
    try:
        yield state
    finally:
        _ACTIVE_TRUST.reset(token)
