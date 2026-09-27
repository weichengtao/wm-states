"""Check cached decoder identities before manually trusted plotting/consumers."""
from pathlib import Path

from scripts.next import cache_io
from scripts.next.cache_paths import primary_cache
from scripts.next.decoding_provenance import verify_decoding_fingerprint


def audit_cached_decoding(cache_dir, data_dir):
    """Apply the invocation's trust policy to every primary decoding session.

    Decode fitting/resume checks its actual checkpoints instead. Plot-only and
    downstream-only invocations otherwise could consume a legacy primary cache
    without recording that manual trust was necessary. Shape and state linkage
    checks in each consumer continue to apply independently of this policy.
    """
    results = cache_io.read(primary_cache(cache_dir, 'decoding_confidence.pkl'))
    if not isinstance(results, list) or not results:
        raise ValueError('Legacy trust requires a nonempty decoding result list.')
    seen = set()
    for result in results:
        if not isinstance(result, dict):
            raise ValueError('Legacy trust requires decoding result objects.')
        session = result.get('session')
        if (not isinstance(session, str) or not session or session in ('.', '..')
                or any(character in session for character in ('/', '\\', '\x00'))):
            raise ValueError('Legacy trust requires valid decoding session identifiers.')
        if session in seen:
            raise ValueError(f'Duplicate decoding session: {session}.')
        seen.add(session)
        config = result.get('config')
        if not isinstance(config, dict) or not config:
            raise ValueError(f'{session}: decoding settings are missing; manual trust cannot repair them.')
        verify_decoding_fingerprint(result.get('fingerprint'), config,
                                   primary_cache(cache_dir, 'cell_screening.pkl'),
                                   Path(data_dir) / f'{session}.mat')
