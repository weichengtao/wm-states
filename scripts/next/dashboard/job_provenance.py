"""Attach immutable, invocation-matched dashboard request metadata to history."""
from collections import Counter
import json
from pathlib import Path
import re

from scripts.next.dashboard.models import RunRequest


def _records(directory: Path, *, run_root: Path, repo_root: Path, local: bool):
    if (not directory.is_dir() or directory.is_symlink()
            or directory.resolve() != directory):
        return []
    records = []
    for path in sorted(directory.glob('*.json')):
        if (not re.fullmatch(r'[0-9a-f]{32}\.json', path.name)
                or path.is_symlink() or not path.is_file()):
            continue
        try:
            value = json.loads(path.read_text())
            if not isinstance(value, dict) or value.get('id') != path.stem:
                continue
            if not local:
                # Run-local records travel with copied/restored runs. Central
                # records require an exact directory match to avoid borrowing
                # metadata from another run that happens to have similar args.
                cache_dir = value.get('cache_dir')
                if not isinstance(cache_dir, str):
                    continue
                cache_dir = Path(cache_dir)
                if not cache_dir.is_absolute():
                    cache_dir = repo_root / cache_dir
                if cache_dir.resolve() != run_root:
                    continue
            records.append(value)
        except (OSError, ValueError, TypeError):
            continue
    return records


def _matches(job: dict, manifest: dict) -> bool:
    manifest_id = job.get('manifest_id')
    if manifest_id is not None:
        return isinstance(manifest_id, str) and bool(manifest_id) and manifest_id == manifest['id']
    invocation = manifest.get('invocation')
    argv = invocation.get('argv') if isinstance(invocation, dict) else None
    return (isinstance(argv, list) and bool(argv) and all(isinstance(item, str) for item in argv)
            and job.get('argv') == argv)


def _argv_key(manifest: dict):
    invocation = manifest.get('invocation')
    argv = invocation.get('argv') if isinstance(invocation, dict) else None
    if isinstance(argv, list) and argv and all(isinstance(item, str) for item in argv):
        return tuple(argv)
    return None


def attach_job_provenance(repo_root: Path, cache_root: Path, run_root: Path,
                          manifests: list[dict], errors: list[str]) -> None:
    """Never infer a source template from the latest job, name, or current files."""
    local = _records(run_root / 'dashboard', run_root=run_root, repo_root=repo_root, local=True)
    central = _records(cache_root / '.dashboard', run_root=run_root, repo_root=repo_root, local=False)
    argv_counts = Counter(key for manifest in manifests if (key := _argv_key(manifest)) is not None)
    for manifest in manifests:
        manifest.update(source_template=None, original_request=None)
        ambiguous_argv = False
        candidates = []
        for records in (local, central):
            for job in records:
                if not _matches(job, manifest):
                    continue
                if job.get('manifest_id') is None and argv_counts[_argv_key(manifest)] != 1:
                    # The exact same command can be replayed after settings
                    # files change. Arguments alone cannot identify that run.
                    ambiguous_argv = True
                    continue
                candidates.append(job)
            if candidates:
                break
        if not candidates:
            if ambiguous_argv:
                errors.append(f"Invocation {manifest['id']} shares its exact command with another invocation; "
                              'an unassigned dashboard record cannot identify its source-template metadata.')
            continue
        if len(candidates) != 1:
            errors.append(f"Multiple dashboard records match invocation {manifest['id']}; "
                          'source-template metadata was not attached.')
            continue
        try:
            payload = candidates[0].get('request')
            if not isinstance(payload, dict):
                continue
            request = RunRequest.model_validate(payload).model_dump(exclude_unset=True)
            json.dumps(request, allow_nan=False)
        except (ValueError, TypeError):
            errors.append(f"Invalid saved dashboard request for invocation {manifest['id']}; "
                          'source-template metadata was not attached.')
            continue
        manifest.update(source_template=request.get('source_template'), original_request=request)
