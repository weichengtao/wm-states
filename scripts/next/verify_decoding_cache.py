"""Read-only verification of decoding provenance, checkpoints, and state links."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "scripts.next"

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import tyro

from scripts.next import cache_io
from scripts.next.cache_paths import primary_cache, stage_path
from scripts.next.common import validate_state_provenance
from scripts.next.decoding_provenance import verify_decoding_fingerprint
from scripts.next.legacy_trust import legacy_trust_context


@dataclass
class Config:
    data_dir: Path = Path('data/nature')
    cache_dir: Path = Path('cache/next_run')


def _session(record, context):
    if not isinstance(record, dict):
        raise ValueError(f'{context}: expected a result object.')
    session = record.get('session')
    if (not isinstance(session, str) or not session or session in ('.', '..')
            or any(character in session for character in ('/', '\\', '\x00'))):
        raise ValueError(f'{context}: invalid session identifier {session!r}.')
    return session


def _records(value, context, *, allow_empty=False):
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError(f'{context}: expected {"a" if allow_empty else "a nonempty"} session result list.')
    records = {}
    for record in value:
        session = _session(record, context)
        if session in records:
            raise ValueError(f'{context}: duplicate session {session}.')
        records[session] = record
    return records


def _verify(record, config, selection_path, context):
    session = _session(record, context)
    settings, fingerprint = record.get('config'), record.get('fingerprint')
    if not isinstance(settings, dict) or not settings:
        raise ValueError(f'{context}: session {session} is missing decoding settings.')
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError(f'{context}: session {session} is missing a decoding fingerprint.')
    try:
        verified = verify_decoding_fingerprint(fingerprint, settings, selection_path,
                                              config.data_dir / f'{session}.mat')
    except (ValueError, OSError) as error:
        raise ValueError(f'{context}: session {session}: {error}') from error
    return verified


def _different(left, right, path='result'):
    """Return the first differing payload field, treating paired NaNs equally."""
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
            return path
        if left.shape != right.shape or left.dtype != right.dtype:
            return path
        equal_nan = left.dtype.kind in 'fc'
        return None if np.array_equal(left, right, equal_nan=equal_nan) else path
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict) or left.keys() != right.keys():
            return path + '.keys'
        for key in left:
            mismatch = _different(left[key], right[key], f'{path}.{key}')
            if mismatch:
                return mismatch
        return None
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        if type(left) is not type(right) or len(left) != len(right):
            return path
        for index, (first, second) in enumerate(zip(left, right, strict=True)):
            mismatch = _different(first, second, f'{path}[{index}]')
            if mismatch:
                return mismatch
        return None
    if isinstance(left, (float, np.floating)) and isinstance(right, (float, np.floating)):
        if np.isnan(left) and np.isnan(right):
            return None
    return None if left == right else path


def _summary(session, verified):
    return {'session': session, 'scheme': verified.scheme,
            'source_revision': verified.source_revision,
            'current_fingerprint': verified.current_fingerprint}


def _verify_run(config: Config):
    """Verify saved artifacts without fitting models, plotting, or writing files."""
    selection_path = primary_cache(config.cache_dir, 'cell_screening.pkl')
    decoded = _records(cache_io.read(primary_cache(config.cache_dir, 'decoding_confidence.pkl')),
                       'Primary decoding cache')
    verifications, sessions = {}, []
    for session, record in decoded.items():
        verified = _verify(record, config, selection_path, 'Primary decoding cache')
        verifications[session] = verified
        sessions.append(_summary(session, verified))

    checkpoints, extra_sessions = [], []
    for candidate in sorted(stage_path(config.cache_dir, 'decode', 'checkpoints').glob('*.pkl')):
        # Resolve each candidate through the stage helper so a symlink cannot
        # silently substitute a checkpoint outside the selected run's stage.
        path = stage_path(config.cache_dir, 'decode', 'checkpoints', candidate.name)
        checkpoint = cache_io.read(path)
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get('result'), dict):
            raise ValueError(f'{path}: expected a checkpoint envelope and result.')
        record = checkpoint['result']
        session = _session(record, str(path))
        if candidate.stem != session:
            raise ValueError(f'{path}: checkpoint filename does not match session {session}.')
        if not record.get('fingerprint') or checkpoint.get('fingerprint') != record['fingerprint']:
            raise ValueError(f'{path}: outer and result fingerprints differ or are missing.')
        verified = _verify(record, config, selection_path, str(path))
        primary = decoded.get(session)
        if primary is not None:
            if record['fingerprint'] != primary['fingerprint']:
                raise ValueError(f'{path}: checkpoint and primary fingerprints differ.')
            if verified.current_fingerprint != verifications[session].current_fingerprint:
                raise ValueError(f'{path}: checkpoint and primary scientific settings differ.')
            # Operational settings may differ. Provenance above validates the
            # scientific settings; all remaining data and metadata must match.
            omitted = {'config', 'fingerprint'}
            mismatch = _different({k: v for k, v in record.items() if k not in omitted},
                                  {k: v for k, v in primary.items() if k not in omitted})
            if mismatch:
                raise ValueError(f'{path}: checkpoint and primary differ at {mismatch}.')
        else:
            extra_sessions.append(session)
        checkpoints.append(_summary(session, verified))

    states_path = primary_cache(config.cache_dir, 'on_off_states.pkl')
    state_sessions = []
    if states_path.exists():
        states = cache_io.read(states_path)
        state_sessions = list(_records(states, 'State cache', allow_empty=True))
        validate_state_provenance(states, config.cache_dir, config.data_dir)

    all_verified = sessions + checkpoints
    return {
        'valid': True,
        'cache_dir': str(config.cache_dir.resolve()),
        'data_dir': str(config.data_dir.resolve()),
        'primary_sessions': len(sessions),
        'primary_schemes': {scheme: sum(row['scheme'] == scheme for row in sessions)
                            for scheme in ('current', 'legacy')},
        'checkpoint_count': len(checkpoints),
        'checkpoint_schemes': {scheme: sum(row['scheme'] == scheme for row in checkpoints)
                               for scheme in ('current', 'legacy')},
        'extra_checkpoint_sessions': extra_sessions,
        'states_present': states_path.exists(),
        'state_sessions': len(state_sessions),
        'legacy_source_revisions': sorted({row['source_revision'] for row in all_verified
                                           if row['source_revision'] is not None}),
        'sessions': sessions,
        'checkpoints': checkpoints,
    }


def verify(config: Config):
    # An explicit verification must never turn an enclosing manual acceptance
    # policy into a report claiming that unverified results are valid.
    with legacy_trust_context(config.cache_dir, enabled=False):
        return _verify_run(config)


def main(config: Config):
    result = verify(config)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    main(tyro.cli(Config))
