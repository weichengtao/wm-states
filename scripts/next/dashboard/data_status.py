"""Read-only recording discovery without opening MAT files or scientific caches."""
import os
from pathlib import Path

from scripts.next.pipeline import STAGES
from scripts.next.dashboard.models import DataStatusRequest


def local_path(root: Path, value: str) -> Path:
    if not isinstance(value, str) or not value.strip() or '\x00' in value:
        raise ValueError('Choose a nonempty path without NUL characters.')
    try:
        path = Path(value).expanduser()
        return (path if path.is_absolute() else root / path).resolve()
    except RuntimeError as exc:
        raise ValueError(f'Cannot resolve the home directory in {value!r}. '
                         'Choose a local path or an existing user.') from exc


def inspect_data(root: Path, request: DataStatusRequest) -> dict:
    """Count regular, top-level .mat files and the allowlists actually used."""
    directory = request.data_dir
    required = []
    counts = []
    file_count = 0

    def result(status, message, *, blocking=None):
        if blocking is None:
            blocking = bool(required) and status != 'ready'
        if not required and status not in ('ready', 'invalid'):
            message += ' The selected stages use cached outputs; this does not prevent launch.'
        return {'status': status, 'data_dir': str(directory), 'file_count': file_count,
                'blocking': blocking, 'message': message, 'stages': counts}

    if not request.stages or len(set(request.stages)) != len(request.stages):
        return result('invalid', 'Choose at least one stage, without duplicates.', blocking=True)
    if unknown := set(request.stages) - set(STAGES):
        return result('invalid', f'Unknown stages: {sorted(unknown)}.', blocking=True)
    for stage in request.stages:
        if request.trust_unverified_legacy_results:
            # The pipeline audits original data files before manual cache reuse,
            # including otherwise cache-only plotting and downstream stages.
            required.append(stage)
            continue
        if stage == 'decode':
            plot_only = request.settings.get(stage, {}).get('plot_only', False)
            if type(plot_only) is not bool:
                return result('invalid', 'decode.plot_only must be true or false.', blocking=True)
            if not plot_only:
                required.append(stage)
        elif stage in ('select', 'activity', 'prepare', 'criticality'):
            required.append(stage)
    try:
        directory = local_path(root, request.data_dir)
    except (ValueError, OSError) as exc:
        return result('invalid', str(exc), blocking=True)
    try:
        with os.scandir(directory) as entries:
            sessions = {Path(entry.name).stem for entry in entries
                        if entry.name.endswith('.mat') and entry.is_file()}
    except (FileNotFoundError, NotADirectoryError):
        return result('missing', f'Data directory does not exist or is not a directory: {directory}')
    except OSError:
        return result('unreadable', f'Data directory cannot be read: {directory}')
    file_count = len(sessions)
    counts.extend({'stage': stage, 'eligible_count': file_count} for stage in required)
    if not sessions:
        return result('empty', f'No regular .mat session files in {directory}. Subfolders are not searched.')
    missing = []
    for count in counts:
        stage = count['stage']
        if stage not in ('select', 'decode'):
            continue
        if stage == 'decode' and request.settings.get(stage, {}).get('plot_only', False):
            continue
        value = request.settings.get(stage, {}).get('session_list_file', request.session_list_file)
        if value is None:
            continue
        try:
            path = local_path(root, value)
            requested = {line.split('#', 1)[0].strip() for line in path.read_text().splitlines()}
        except (ValueError, FileNotFoundError, IsADirectoryError) as exc:
            return result('invalid', f'{stage} session list file is invalid or missing: {value!r}. {exc}',
                          blocking=True)
        except (OSError, UnicodeError):
            return result('unreadable', f'{stage} session list file cannot be read: {value!r}.', blocking=True)
        count['eligible_count'] = len(sessions & requested)
        if not count['eligible_count']:
            missing.append(stage)
    if missing:
        return result('no_matches', f'Session list does not select any available .mat sessions for: '
                      f'{", ".join(missing)}.', blocking=True)
    message = f'{file_count} .mat session file{"s" if file_count != 1 else ""} available.'
    if not required:
        message += ' The selected stages use cached outputs.'
    else:
        message += ' Session counts reflect file names and allowlists, before screening or cache checks.'
    return result('ready', message, blocking=False)
