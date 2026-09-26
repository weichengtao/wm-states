"""Reusable dashboard configurations, independent of jobs and recording access."""
from contextlib import contextmanager
from datetime import datetime, timezone
import errno
import fcntl
import json
import os
from pathlib import Path
import stat
from uuid import uuid4

from scripts.next.dashboard.models import RunRequest, SavedTemplate, TemplateConfig, TemplateRequest
from scripts.next.dashboard.schema import resolve_settings


class TemplateConflict(ValueError):
    """A template with this display name already exists."""


class TemplateStore:
    """Read built-ins and publish new, immutable saved-template files."""

    directory_parts = ('configs', 'next', 'templates')
    shared_fields = {'stages', 'n_jobs', 'max_sessions_to_run', 'figure_formats'}
    builtin_details = {
        'example': ('Example pipeline', 'Example scientific analysis settings.'),
        'smoke': ('Smoke test', 'Reduced computation for an integration check.'),
    }

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root.resolve()

    @contextmanager
    def _directory(self, parts, *, create=False):
        """Walk from the repository with descriptor-relative, no-symlink opens."""
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        descriptor = os.open(self.repo_root, flags)
        try:
            for name in parts:
                if create:
                    try:
                        os.mkdir(name, dir_fd=descriptor)
                        os.fsync(descriptor)
                    except FileExistsError:
                        pass
                try:
                    child = os.open(name, flags, dir_fd=descriptor)
                except FileNotFoundError:
                    if create:
                        raise
                    yield None
                    return
                except OSError as exc:
                    if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                        raise ValueError('Template directories must not contain symlinks or non-directory paths.') from exc
                    raise
                os.close(descriptor)
                descriptor = child
            yield descriptor
        finally:
            os.close(descriptor)

    @staticmethod
    def _read(descriptor, filename):
        handle = os.open(filename, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor)
        with os.fdopen(handle, encoding='utf-8') as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError('Template must be a regular JSON file.')
            def reject_constant(value):
                raise ValueError(f'Nonfinite JSON value {value!r} is not supported.')
            return json.load(stream, parse_constant=reject_constant)

    def _validate_config(self, config: TemplateConfig):
        # Validate the real stage dataclasses but neither inspect recordings nor
        # materialize resolved defaults: sparse settings and explicit nulls matter.
        values = config.model_dump(exclude_unset=True)
        request = RunRequest(**values)
        # A reusable path may refer to another computer or user. Expanding ~user
        # here would require that user to exist on the machine saving it.
        data_dir = Path(request.data_dir)
        if not data_dir.is_absolute():
            data_dir = self.repo_root / data_dir
        resolve_settings(request, self.repo_root, self.repo_root / 'cache/template-validation', data_dir,
                         validate_external_files=False)
        json.dumps(values, allow_nan=False)

    def _builtins(self, warnings):
        result = []
        try:
            with self._directory(('configs', 'next')) as directory:
                if directory is None:
                    warnings.append('Built-in templates are unavailable: configs/next is missing.')
                    return result
                defaults = RunRequest().model_dump(include=self.shared_fields)
                for identifier, (name, description) in self.builtin_details.items():
                    filename = f'{identifier}_pipeline.json'
                    try:
                        settings = self._read(directory, filename)
                        config = TemplateConfig(**defaults, settings=settings)
                        self._validate_config(config)
                        result.append({
                            'id': identifier, 'name': name, 'description': description,
                            'builtin': True, 'path': f'configs/next/{filename}',
                            'config': config.model_dump(exclude_unset=True),
                        })
                    except (OSError, ValueError, TypeError) as exc:
                        warnings.append(f'Could not load built-in template {filename}: {exc}')
        except (OSError, ValueError) as exc:
            warnings.append(f'Could not load built-in templates: {exc}')
        return result

    def _saved(self, directory, warnings, names):
        result = []
        for filename in sorted(os.listdir(directory)):
            if filename.startswith('.') or not filename.endswith('.json'):
                continue
            try:
                value = self._read(directory, filename)
                if not isinstance(value, dict) or type(value.get('schema_version')) is not int or value['schema_version'] != 1:
                    raise ValueError('Unsupported template schema_version; expected 1.')
                envelope = SavedTemplate.model_validate(value)
                if filename != f'{envelope.id}.json':
                    raise ValueError('Template ID must match its filename.')
                self._validate_config(envelope.config)
                name_key = envelope.name.strip().casefold()
                if name_key in names:
                    raise ValueError(f'A template named {envelope.name!r} is already listed.')
                names.add(name_key)
                result.append(self._record(envelope))
            except (OSError, ValueError, TypeError) as exc:
                warnings.append(f'Skipped saved template {filename}: {exc}')
        return sorted(result, key=lambda item: (datetime.fromisoformat(item['created_at']), item['id']), reverse=True)

    @staticmethod
    def _record(envelope):
        return {
            'id': envelope.id, 'name': envelope.name, 'description': envelope.description,
            'created_at': envelope.created_at,
            'config': envelope.config.model_dump(exclude_unset=True),
            'builtin': False,
            'path': f'configs/next/templates/{envelope.id}.json',
        }

    def list(self):
        warnings = []
        builtins = self._builtins(warnings)
        saved = []
        names = {name.strip().casefold() for name, _ in self.builtin_details.values()}
        try:
            with self._directory(self.directory_parts) as directory:
                if directory is not None:
                    saved = self._saved(directory, warnings, names)
        except (OSError, ValueError) as exc:
            warnings.append(f'Could not read saved templates: {exc}')
        return {'templates': [*builtins, *saved], 'warnings': warnings}

    @staticmethod
    def _publish(directory, filename, value):
        """Fsync a new file, then link it into place without replacing any file."""
        temporary = f'.{uuid4().hex}.tmp'
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600, dir_fd=directory)
        try:
            with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
                json.dump(value, stream, indent=2, allow_nan=False, ensure_ascii=False)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, filename, src_dir_fd=directory, dst_dir_fd=directory,
                    follow_symlinks=False)
        finally:
            os.unlink(temporary, dir_fd=directory)
            os.fsync(directory)

    def save(self, request: TemplateRequest):
        self._validate_config(request.config)
        # Lock across processes too, so concurrent saves cannot pass the same
        # duplicate-name check. This lock never blocks the analysis runner.
        with self._directory(self.directory_parts, create=True) as directory:
            lock_flags = os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK
            try:
                lock = os.open('.templates.lock', lock_flags | os.O_CREAT | os.O_EXCL, 0o600,
                               dir_fd=directory)
            except FileExistsError:
                lock = os.open('.templates.lock', lock_flags, dir_fd=directory)
            try:
                if not stat.S_ISREG(os.fstat(lock).st_mode):
                    raise ValueError('Template lock must be a regular file.')
                fcntl.flock(lock, fcntl.LOCK_EX)
                warnings = []
                names = {name.strip().casefold() for name, _ in self.builtin_details.values()}
                self._saved(directory, warnings, names)
                if request.name.strip().casefold() in names:
                    raise TemplateConflict('A template with this name already exists. Choose a different name.')
                for _ in range(8):
                    envelope = SavedTemplate(
                        **{**request.model_dump(exclude_unset=True), 'description': request.description},
                        schema_version=1, id=uuid4().hex,
                        created_at=datetime.now(timezone.utc).isoformat(),
                    )
                    try:
                        self._publish(directory, f'{envelope.id}.json', envelope.model_dump(exclude_unset=True))
                    except FileExistsError:
                        continue
                    return self._record(envelope)
                raise ValueError('Could not allocate a unique template file. Try saving again.')
            finally:
                os.close(lock)
