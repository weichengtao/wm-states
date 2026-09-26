"""Expose and validate the actual pipeline dataclasses without duplicating them."""
from dataclasses import asdict, fields
from enum import Enum
from functools import lru_cache
import importlib
import inspect
import json
import math
from pathlib import Path
import types
from typing import Literal, Union, get_args, get_origin, get_type_hints

from scripts.next import pipeline
from scripts.next.common import json_value
from scripts.next.dashboard.models import RunRequest
from scripts.next.diagnostic_config import load_diagnostic_config

STAGE_INFO = {
    'select': ('Cell screening', 'Screen full sessions with explicit cell-quality checks.'),
    'decode': ('Decode confidence', 'Fit one observed estimate and training-label null shuffles.'),
    'evaluate': ('Evaluate decoding', 'Summarize confidence and decoding performance by session.'),
    'states': ('Identify states', 'Detect on/off states with null-based cluster correction.'),
    'activity': ('Compare activity', 'Inspect population activity and state-dependent structure.'),
    'prepare': ('Prepare features', 'Build trial-level outcomes and fold-safe model features.'),
    'models': ('Model families', 'Compare mixed-effects models for off-state duration.'),
    'nested-count': ('Cell count models', 'Test nested contributions of population cell counts.'),
    'nested-activity': ('Activity models', 'Test nested contributions of normalized activity.'),
    'criticality': ('Active-cell thresholds', 'Scan activity thresholds and compare model performance.'),
    'interactions': ('Period interactions', 'Test activity interactions across task periods.'),
}
SHARED_FIELDS = {'cache_dir', 'data_dir', 'n_jobs', 'n_jobs_session', 'cv_n_jobs',
                 'session_list_file', 'max_sessions_to_run'}


def jsonable(value):
    return json.loads(json.dumps(value, default=json_value, allow_nan=False))


@lru_cache(maxsize=None)
def stage_module(stage):
    return importlib.import_module(f'scripts.next.{pipeline.STAGES[stage]}')


def _type_description(hint):
    origin, args = get_origin(hint), get_args(hint)
    if origin in (Union, types.UnionType):
        result = _type_description(next(item for item in args if item is not type(None)))
        return {**result, 'nullable': type(None) in args}
    if origin is Literal:
        return {'type': 'string', 'choices': list(args)}
    if isinstance(hint, type) and issubclass(hint, Enum):
        return {'type': 'string', 'choices': [item.value for item in hint]}
    if origin in (list, tuple):
        return {'type': 'array', 'items': _type_description(args[0])}
    return {'type': {bool: 'boolean', int: 'integer', float: 'number', str: 'string',
                     Path: 'string'}.get(hint, 'string')}


def _field_help(config_class):
    source = inspect.getsource(config_class).splitlines()
    comments, descriptions = [], {}
    for line in source:
        stripped = line.strip()
        if stripped.startswith('#'):
            comments.append(stripped[1:].strip())
            continue
        if ':' in stripped and not stripped.startswith(('class ', 'def ', '"')):
            name = stripped.split(':', 1)[0]
            if name.isidentifier():
                inline = stripped.split('#', 1)[1].strip() if '#' in stripped else ''
                descriptions[name] = ' '.join([*comments, inline]).strip()
        comments = []
    return descriptions


def get_schema(repo_root):
    presets = {name: json.loads((repo_root / f'configs/next/{name}_pipeline.json').read_text())
               for name in ('example', 'smoke')}
    stages = []
    for stage, (label, description) in STAGE_INFO.items():
        module = stage_module(stage)
        config = module.Config()
        hints, help_text = get_type_hints(module.Config), _field_help(module.Config)
        stage_fields = []
        for field in fields(config):
            if field.name in SHARED_FIELDS:
                continue
            stage_fields.append({'name': field.name, **_type_description(hints[field.name]),
                                 'nullable': type(None) in get_args(hints[field.name]),
                                 'default': jsonable(getattr(config, field.name)),
                                 'description': help_text.get(field.name, '')})
        stages.append({'id': stage, 'label': label, 'description': description, 'fields': stage_fields})
    return {'stages': stages, 'presets': presets, 'defaults': {**RunRequest().model_dump(), 'settings': presets['example']}}


def _validate_type(value, hint, field):
    origin, args = get_origin(hint), get_args(hint)
    if origin in (Union, types.UnionType):
        if value is None and type(None) in args:
            return
        for choice in args:
            if choice is type(None):
                continue
            try:
                _validate_type(value, choice, field)
                return
            except ValueError:
                pass
        raise ValueError(f'{field}: value does not match its required type.')
    if origin is Literal:
        if not any(type(value) is type(choice) and value == choice for choice in args):
            raise ValueError(f'{field}: choose one of {list(args)}.')
    elif isinstance(hint, type) and issubclass(hint, Enum):
        if not isinstance(value, str) or value not in {*hint.__members__, *(item.value for item in hint)}:
            raise ValueError(f'{field}: choose one of {[item.value for item in hint]}.')
    elif origin in (list, tuple):
        if not isinstance(value, list):
            raise ValueError(f'{field}: expected an array.')
        for index, item in enumerate(value):
            _validate_type(item, args[0], f'{field}[{index}]')
    elif hint is float:
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f'{field}: expected a finite number.')
    elif hint is Path:
        if not isinstance(value, str) or not value.strip() or '\x00' in value:
            raise ValueError(f'{field}: expected a nonempty path string.')
    elif hint in (str, bool, int):
        if type(value) is not hint:
            raise ValueError(f'{field}: expected {hint.__name__}.')
        if hint is str and '\x00' in value:
            raise ValueError(f'{field}: NUL characters are not allowed.')
    else:
        raise ValueError(f'{field}: unsupported parameter type {hint}.')


def _validate_limits(config, stage):
    """Check shared model/plot options before their main functions read caches."""
    values = vars(config)
    for name, value in values.items():
        if value is None:
            continue
        if name.endswith('_subdir'):
            path = Path(value)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError(f'{stage}.{name}: must stay within the owning stage directory.')
        if name.endswith('_filename') and (Path(value).name != value or value in ('', '.', '..')):
            raise ValueError(f'{stage}.{name}: must be a filename, not a path.')
        if name in ('n_jobs', 'n_jobs_session', 'cv_n_jobs') and value == 0:
            raise ValueError(f'{stage}.{name}: must be nonzero.')
        if name in ('cv_seed', 'seed', 'cv_prediction_sample_per_model') and value < 0:
            raise ValueError(f'{stage}.{name}: must be nonnegative.')
        if name in ('cv_shuffles', 'figure_dpi', 'max_iterations', 'max_points_per_color_group',
                    'max_points_per_max_off_state', 'activity_bin_width_ms',
                    'marginal_histogram_bin_width') and value <= 0:
            raise ValueError(f'{stage}.{name}: must be positive.')
        if name in ('cv_holdout_fraction', 'significance_alpha') and not 0 < value < 1:
            raise ValueError(f'{stage}.{name}: must be in (0, 1).')
        if name == 'history_alpha' and not 0 < value <= 1:
            raise ValueError(f'{stage}.{name}: must be in (0, 1].')
        if name == 'marginal_histogram_bin_offset_fraction' and not 0 <= value <= .5:
            raise ValueError(f'{stage}.{name}: must be in [0, 0.5].')


def resolve_settings(request: RunRequest, repo_root: Path, cache_dir: Path, data_dir: Path):
    unknown = set(request.stages) - set(pipeline.STAGES)
    if unknown:
        raise ValueError(f'Unknown stages: {sorted(unknown)}.')
    if request.stages != [stage for stage in pipeline.STAGES if stage in request.stages]:
        raise ValueError('Stages must follow pipeline order; choose any ordered subset.')
    unknown = set(request.settings) - set(pipeline.STAGES)
    if unknown:
        raise ValueError(f'Unknown settings stages: {sorted(unknown)}.')
    shared = dict(cache_dir=cache_dir, data_dir=data_dir, n_jobs=request.n_jobs,
                  n_jobs_session=request.n_jobs, cv_n_jobs=request.n_jobs,
                  max_sessions_to_run=request.max_sessions_to_run,
                  session_list_file=request.session_list_file)
    result = {}
    # Validate disabled stages too: spelling mistakes must never silently disappear.
    for stage in dict.fromkeys([*request.stages, *request.settings]):
        module, overrides = stage_module(stage), request.settings.get(stage, {})
        hints = get_type_hints(module.Config)
        for name, value in overrides.items():
            if name in {'cache_dir', 'data_dir'}:
                raise ValueError(f'{stage}.{name}: use the shared run settings.')
            if name not in hints:
                raise ValueError(f'{stage}: unknown setting {name!r}.')
            _validate_type(value, hints[name], f'{stage}.{name}')
        try:
            config = pipeline.resolve_config(module, shared, overrides)
            _validate_limits(config, stage)
            if hasattr(module, '_validate_config'):
                module._validate_config(config)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{stage}: {exc}') from exc
        if stage in request.stages:
            if (stage == 'select' and config.save_extended_diagnostics
                    and config.diagnostics_figure_config is not None):
                path = config.diagnostics_figure_config
                if not path.is_absolute():
                    path = repo_root / path
                try:
                    load_diagnostic_config(path)
                except (OSError, ValueError) as exc:
                    raise ValueError(f'select.diagnostics_figure_config: {exc}') from exc
            result[stage] = jsonable(asdict(config))
    if 'decode' in result and 'states' in result and result['decode']['n_decode_shuffle'] < 2:
        raise ValueError('State detection needs at least two decoding null shuffles.')
    return result
