"""Scientific decoding identities and verified, read-only legacy compatibility.

Legacy fingerprints have no version tag. They hashed all top-level next scripts;
compatibility therefore needs the exact historical bytes, not a trusted flag in
a cache. No cached fingerprint is rewritten, preserving existing state links.
"""
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re

from scripts.next.decoding_signature import ScientificSourceError, scientific_source_digest
from scripts.next.legacy_decoding_provenance import (
    HistoricalSourceUnavailable, load_legacy_source_snapshots,
)


FINGERPRINT_PREFIX = 'decode-v2:'
NON_ANALYSIS_SETTINGS = (
    'n_jobs', 'par_verbose', 'resume', 'plot_only', 'save_figures',
    'plot_actual_trial_id', 'session_list_file', 'max_sessions_to_run',
)
RUNTIME_PACKAGES = ('numpy', 'scipy', 'scikit-learn', 'joblib', 'threadpoolctl')


class DecodingCacheMismatch(ValueError):
    """The cached analysis cannot be reproduced from these settings/inputs/code."""


@dataclass(frozen=True)
class FingerprintVerification:
    scheme: str
    source_revision: str | None
    current_fingerprint: str | None


def _runtime_versions():
    return {package: version(package) for package in RUNTIME_PACKAGES}


def _settings_bytes(config, exclude):
    from scripts.next.common import json_value

    settings = asdict(config) if is_dataclass(config) else dict(config)
    settings = {key: value for key, value in settings.items() if key not in exclude}
    return json.dumps(settings, sort_keys=True, default=json_value).encode()


def _framed_update(digest, value):
    digest.update(len(value).to_bytes(8, 'big'))
    digest.update(value)


def _compute(config, paths, *, exclude, source_digest, snapshots=()):
    """Read each input once for both the new and candidate historical hashes."""
    settings = _settings_bytes(config, exclude)
    digest = hashlib.sha256()
    _framed_update(digest, json.dumps({
        'scheme': FINGERPRINT_PREFIX,
        'scientific_source': source_digest,
        'packages': _runtime_versions(),
    }, sort_keys=True).encode())
    _framed_update(digest, settings)
    legacy = []
    for snapshot in snapshots:
        previous = hashlib.sha256(settings)
        for filename, contents in sorted(snapshot.sources.items()):
            if filename.endswith('.py'):
                previous.update(contents)
        legacy.append((snapshot.revision, previous))
    for path in paths:
        path = Path(path)
        resolved = str(path.resolve()).encode()
        content_hash = hashlib.sha256()
        for _, previous in legacy:
            previous.update(resolved)
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                content_hash.update(block)
                for _, previous in legacy:
                    previous.update(block)
        _framed_update(digest, resolved)
        _framed_update(digest, content_hash.digest())
    return FINGERPRINT_PREFIX + digest.hexdigest(), {
        revision: previous.hexdigest() for revision, previous in legacy
    }


def fingerprint(config, paths=(), *, exclude=()):
    """Hash scientific source, effective settings, runtime versions and inputs."""
    return _compute(config, paths, exclude=exclude,
                    source_digest=scientific_source_digest())[0]


def decoding_fingerprint(config, selection_path, data_path):
    return fingerprint(config, (selection_path, data_path), exclude=NON_ANALYSIS_SETTINGS)


def decoding_settings_match(first, second):
    """Compare the effective analysis settings using the fingerprint serializer."""
    return _settings_bytes(first, NON_ANALYSIS_SETTINGS) == _settings_bytes(second, NON_ANALYSIS_SETTINGS)


def _verify_decoding_fingerprint(saved_fingerprint, config, selection_path, data_path):
    """Verify a current key or reproduce a scientifically equivalent legacy key.

    Raises DecodingCacheMismatch for stale/unknown keys. Missing historical Git
    objects raise HistoricalSourceUnavailable instead of turning an unverifiable
    legacy checkpoint into an unexpected multi-hour refit.
    """
    if not isinstance(saved_fingerprint, str):
        raise DecodingCacheMismatch('Missing decoding fingerprint.')
    source_digest = scientific_source_digest()
    paths = (selection_path, data_path)
    if saved_fingerprint.startswith(FINGERPRINT_PREFIX):
        current, _ = _compute(config, paths, exclude=NON_ANALYSIS_SETTINGS,
                              source_digest=source_digest)
        if saved_fingerprint != current:
            raise DecodingCacheMismatch(
                'Decoding cache is stale for the current settings, inputs, scientific code, or package versions.')
        return FingerprintVerification('current', None, current)
    if not re.fullmatch(r'[0-9a-f]{64}', saved_fingerprint):
        raise DecodingCacheMismatch('Unknown or malformed decoding fingerprint; compatibility cannot be verified.')
    history = load_legacy_source_snapshots()
    compatible = tuple(snapshot for snapshot in history.snapshots
                       if scientific_source_digest(snapshot.sources) == source_digest)
    current, previous = _compute(config, paths, exclude=NON_ANALYSIS_SETTINGS,
                                 source_digest=source_digest, snapshots=compatible)
    for revision, legacy_key in previous.items():
        if saved_fingerprint == legacy_key:
            return FingerprintVerification('legacy', revision, current)
    if history.unavailable_refs:
        raise HistoricalSourceUnavailable(
            history.unavailable_refs,
            reason='Run scripts/next/verify_decoding_cache.py before resuming. '
                   'To intentionally recompute, disable resume.')
    raise DecodingCacheMismatch(
        'Legacy decoding cache is stale or unsupported: its fingerprint does not match '
        'the verified historical implementation with the current settings and inputs.')


def verify_decoding_fingerprint(saved_fingerprint, config, selection_path, data_path):
    """Verify normally, allowing only an explicit run-scoped legacy trust fallback.

    Corrupt historical objects, unreadable inputs, malformed/missing keys and
    native v2 mismatches remain errors. Trusted legacy results retain their old
    identities and never receive a fabricated verified current fingerprint.
    """
    try:
        return _verify_decoding_fingerprint(saved_fingerprint, config, selection_path, data_path)
    except (DecodingCacheMismatch, HistoricalSourceUnavailable, ScientificSourceError) as exc:
        if isinstance(saved_fingerprint, str) and re.fullmatch(r'[0-9a-f]{64}', saved_fingerprint):
            from scripts.next.legacy_trust import record_legacy_trust
            if record_legacy_trust(saved_fingerprint, selection_path, data_path, exc):
                return FingerprintVerification('trusted-legacy', None, None)
        raise
