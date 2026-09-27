"""Read pinned historical fingerprint sources without executing or fetching them.

This registry is intentionally finite. A cache cannot nominate a revision or
assert its own compatibility. The caller must additionally compare historical
and current scientific signatures and reproduce the original cache fingerprint.
New scientific fingerprints do not need this module or a Git checkout.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import re
import subprocess
from types import MappingProxyType
from typing import Mapping


PINNED_LEGACY_REVISIONS = (
    '4843a39b529d9b9649981fb09b97356fda630c21',
    'f6fb50e04734a3d47648508a5ca0f7f26748cc2d',
)

# Python 3.12 AST signatures for the exact original algorithm and serializer.
# Keep these independent of current code: changing current fingerprint logic
# must never silently change what is accepted as a verified historical cache.
_ALGORITHM_SIGNATURES = {
    'fingerprint': 'a64fb899f51e54253c7d7546054960133090733217b2dba3db7494a0cf14744d',
    'decoding_fingerprint': '16df7de196986252a1b0bae356fcc33ede36cacdd6712097f0e7902a6421cc46',
    'json_value': '91efbf47b802b4955fd2c994e886fe81d9c555b03e6598df2a1882ddd86009ba',
}


class HistoricalSourceUnavailable(ValueError):
    """A legacy key cannot be verified because required local history is missing."""

    def __init__(self, revisions, *, reason=None):
        self.revisions = tuple(revisions)
        references = ', '.join(revision[:12] for revision in self.revisions)
        message = (
            f'Historical decoding sources are unavailable locally ({references}). '
            'Restore repository Git history containing these revisions and retry '
            'verification, or explicitly choose to recompute decoding. '
            'The legacy cache cannot be verified from the available source history.'
        )
        if reason:
            message += f' {reason}'
        super().__init__(message)


class HistoricalSourceInvalid(ValueError):
    """Available historical objects do not match the supported source contract."""


@dataclass(frozen=True)
class LegacySourceSnapshot:
    revision: str
    sources: Mapping[str, bytes]


@dataclass(frozen=True)
class LegacySourceHistory:
    snapshots: tuple[LegacySourceSnapshot, ...]
    unavailable_refs: tuple[str, ...]

    def require_complete(self):
        """Call only after no available snapshot has verified the saved key."""
        if self.unavailable_refs:
            raise HistoricalSourceUnavailable(self.unavailable_refs)


def _git(root: Path, arguments, *, input_bytes=None):
    # Disable replace refs and partial-clone lazy fetching: these must be exact
    # local immutable objects, with no prompts, remote access, or source execution.
    environment = dict(os.environ, GIT_NO_REPLACE_OBJECTS='1', GIT_NO_LAZY_FETCH='1',
                       GIT_TERMINAL_PROMPT='0', GIT_OPTIONAL_LOCKS='0')
    try:
        completed = subprocess.run(
            ['git', '--no-pager', '-C', str(root), *arguments],
            input=input_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=environment, check=False, timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise HistoricalSourceUnavailable(PINNED_LEGACY_REVISIONS,
                                          reason='Local Git source lookup failed.') from exc
    if completed.returncode:
        raise HistoricalSourceUnavailable(PINNED_LEGACY_REVISIONS)
    return completed.stdout


def validate_legacy_algorithm(sources: Mapping[str, bytes]):
    """Reject missing, altered, or ambiguous historical fingerprint functions."""
    try:
        tree = ast.parse(sources['common.py'])
    except (KeyError, SyntaxError, UnicodeDecodeError) as exc:
        raise HistoricalSourceInvalid('Historical common.py is missing or invalid.') from exc
    for name, expected in _ALGORITHM_SIGNATURES.items():
        definitions = [node for node in tree.body
                       if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                       and node.name == name]
        if len(definitions) != 1:
            raise HistoricalSourceInvalid(f'Historical fingerprint function {name!r} is missing or ambiguous.')
        actual = hashlib.sha256(ast.dump(definitions[0], include_attributes=False).encode()).hexdigest()
        if actual != expected:
            raise HistoricalSourceInvalid(f'Unsupported historical fingerprint algorithm: {name}.')


@lru_cache(maxsize=8)
def _load_snapshot(root: Path, revision: str):
    if revision not in PINNED_LEGACY_REVISIONS:
        raise HistoricalSourceInvalid('Only pinned historical source revisions may be loaded.')
    tree = _git(root, ['ls-tree', '-z', f'{revision}:scripts/next'])
    blobs = {}
    try:
        for entry in tree.split(b'\0'):
            if not entry:
                continue
            metadata, filename = entry.split(b'\t', 1)
            name = filename.decode('utf-8')
            if not name.endswith('.py'):
                continue
            mode, kind, object_id = metadata.decode('ascii').split()
            if (mode not in ('100644', '100755') or kind != 'blob'
                    or '/' in name or '\\' in name
                    or not re.fullmatch(r'(?:[0-9a-f]{40}|[0-9a-f]{64})', object_id)
                    or name in blobs):
                raise ValueError('Unexpected Python source entry')
            blobs[name] = object_id
    except (ValueError, UnicodeDecodeError) as exc:
        raise HistoricalSourceInvalid(f'Invalid historical Python source tree at {revision[:12]}.') from exc
    if not blobs:
        raise HistoricalSourceInvalid(f'Historical Python source tree is empty at {revision[:12]}.')
    names = sorted(blobs)
    payload = _git(root, ['cat-file', '--batch'],
                   input_bytes=''.join(blobs[name] + '\n' for name in names).encode('ascii'))
    sources, offset = {}, 0
    for name in names:
        end = payload.find(b'\n', offset)
        if end < 0:
            raise HistoricalSourceUnavailable((revision,))
        header = payload[offset:end].split()
        if header == [blobs[name].encode('ascii'), b'missing']:
            raise HistoricalSourceUnavailable((revision,))
        if (len(header) != 3 or header[0] != blobs[name].encode('ascii')
                or header[1] != b'blob' or not header[2].isdigit()):
            raise HistoricalSourceInvalid(f'Invalid historical source blob for {name}.')
        size = int(header[2])
        start, stop = end + 1, end + 1 + size
        if stop >= len(payload) or payload[stop:stop + 1] != b'\n':
            raise HistoricalSourceInvalid(f'Truncated historical source blob for {name}.')
        source = payload[start:stop]
        digest = hashlib.sha1() if len(blobs[name]) == 40 else hashlib.sha256()
        digest.update(f'blob {size}\0'.encode('ascii'))
        digest.update(source)
        if digest.hexdigest() != blobs[name]:
            raise HistoricalSourceInvalid(f'Historical source object checksum mismatch for {name}.')
        sources[name] = source
        offset = stop + 1
    if offset != len(payload):
        raise HistoricalSourceInvalid('Unexpected content after historical source blobs.')
    validate_legacy_algorithm(sources)
    return LegacySourceSnapshot(revision, MappingProxyType(sources))


def load_legacy_source_snapshots(repo_root: Path | None = None) -> LegacySourceHistory:
    """Load available pinned snapshots; report missing history without fetching it.

    The caller may accept a positively verified cache from any available
    snapshot. If none match, ``require_complete()`` distinguishes genuinely
    incompatible caches from caches whose historical sources are unavailable.
    """
    root = (repo_root or Path(__file__).resolve().parents[2]).resolve()
    try:
        top = Path(_git(root, ['rev-parse', '--show-toplevel']).decode('utf-8').strip()).resolve()
    except (UnicodeDecodeError, ValueError) as exc:
        if isinstance(exc, HistoricalSourceUnavailable):
            raise
        raise HistoricalSourceUnavailable(PINNED_LEGACY_REVISIONS) from exc
    if top != root:
        raise HistoricalSourceUnavailable(PINNED_LEGACY_REVISIONS,
                                          reason='The installed scripts are not at this Git repository root.')
    snapshots, unavailable = [], []
    for revision in PINNED_LEGACY_REVISIONS:
        try:
            snapshots.append(_load_snapshot(root, revision))
        except HistoricalSourceUnavailable:
            unavailable.append(revision)
    if not snapshots:
        raise HistoricalSourceUnavailable(unavailable)
    return LegacySourceHistory(tuple(snapshots), tuple(unavailable))
