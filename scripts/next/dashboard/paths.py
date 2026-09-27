"""Bounded, read-only completion of paths on the dashboard server's filesystem."""
from collections.abc import Callable
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request


MAX_SCANNED_ENTRIES = 10_000


def complete_path(root: Path, value: str, mode: str = 'any', extensions: str = '',
                  limit: int = 30):
    """Keep the user's relative/absolute/home notation in returned completions."""
    result = {'entries': [], 'warning': None, 'truncated': False}
    if '\x00' in value:
        result['warning'] = 'Paths cannot contain a null character.'
        return result
    if value.startswith('~') and value != '~' and not value.startswith('~/'):
        result['warning'] = 'Use ~/ for your home directory; named-user shortcuts are not supported.'
        return result
    if value == '~':
        result['entries'] = [{'name': '~/', 'path': '~/', 'kind': 'directory'}]
        return result
    if value in ('.', '..'):
        value += '/'
    parent_text, separator, prefix = value.rpartition('/')
    visible_parent = parent_text + separator
    suffixes = tuple(suffix.lower() if suffix.startswith('.') else '.' + suffix.lower()
                     for suffix in (part.strip() for part in extensions.split(',')) if suffix)
    entries = []
    try:
        directory = Path(visible_parent or '.').expanduser()
        if not directory.is_absolute():
            directory = root / directory
        with os.scandir(directory) as children:
            for scanned, entry in enumerate(children):
                if scanned >= MAX_SCANNED_ENTRIES:
                    result['truncated'] = True
                    break
                if not entry.name.casefold().startswith(prefix.casefold()):
                    continue
                if entry.name.startswith('.') and not prefix.startswith('.'):
                    continue
                try:
                    is_directory = entry.is_dir()
                    if not is_directory and (mode == 'directory' or not entry.is_file()):
                        continue
                except OSError:
                    continue  # A concurrently removed/unreadable child is not a candidate.
                if not is_directory and suffixes and not entry.name.lower().endswith(suffixes):
                    continue
                entries.append({'name': entry.name + ('/' if is_directory else ''),
                                'path': visible_parent + entry.name + ('/' if is_directory else ''),
                                'kind': 'directory' if is_directory else 'file'})
    except FileNotFoundError:
        result['warning'] = 'This parent folder does not exist yet. You can still type a path manually.'
    except NotADirectoryError:
        result['warning'] = 'The parent path is a file. Choose a folder to browse its contents.'
    except PermissionError:
        result['warning'] = 'The dashboard does not have permission to list this folder.'
    except (OSError, RuntimeError):
        result['warning'] = 'This folder could not be listed. You can still type a path manually.'
    entries.sort(key=lambda item: (item['kind'] != 'directory', item['name'].casefold(), item['name']))
    result['entries'] = entries[:limit]
    result['truncated'] = result['truncated'] or len(entries) > limit
    if result['truncated']:
        result['warning'] = 'More matches are available. Type more of the name to narrow the list.'
    return result


def create_paths_router(root: Path, trusted_origin: Callable[[str | None, str | None], bool]):
    router = APIRouter()

    @router.get('/api/paths/complete')
    def complete(request: Request, path: str = Query('', max_length=4096),
                 mode: str = Query('any', pattern='^(directory|any)$'),
                 extensions: str = Query('', max_length=300),
                 limit: int = Query(30, ge=1, le=50)):
        # Unlike other read-only routes, this reveals local filenames. Check both
        # Origin and Fetch Metadata: browser navigations can omit Origin.
        if (request.headers.get('sec-fetch-site') == 'cross-site'
                or not trusted_origin(request.headers.get('origin'), request.headers.get('host'))):
            raise HTTPException(403, 'Cross-origin path browsing is not allowed.')
        return complete_path(root, path, mode, extensions, limit)

    return router
