"""Run the full-session pipeline from one command and optional JSON settings."""

# Use one module namespace for direct CLI and package execution.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "scripts.next"

from dataclasses import asdict, dataclass, fields
from enum import Enum
import importlib
import json
from pathlib import Path
import sys
import time
from typing import get_args, get_type_hints

import tyro
from threadpoolctl import threadpool_limits

from scripts.next.common import json_value
from scripts.next.cache_paths import primary_cache
from scripts.next.diagnostic_config import load_diagnostic_config
from scripts.next.legacy_run_checks import audit_cached_decoding
from scripts.next.legacy_trust import legacy_trust_context
from scripts.next.figure_exports import (
    DEFAULT_FIGURE_FONT, FigureFormat, figure_font_context, figure_format_context,
    validate_figure_font, validate_figure_formats,
)
from scripts.next.run_manifest import RunManifest, invocation_context, utc_now

STAGES = {
    'select': 'cell_screening',
    'decode': 'decoding_confidence',
    'evaluate': 'eval_confidence',
    'states': 'on_off_states',
    'activity': 'compare_activity_across_states',
    'prepare': 'prepare_data_for_mixedlm',
    'models': 'compare_mixed_effect_models',
    'nested-count': 'nested_model_comparison_cell_counts',
    'nested-activity': 'nested_model_comparison_mean_norm_activity',
    'criticality': 'find_active_cell_criticality',
    'interactions': 'test_interactions_across_periods',
}
FULL_SESSION = ('select', 'decode', 'evaluate', 'states', 'activity')
MIXED = ('prepare', 'models', 'nested-count', 'nested-activity', 'criticality', 'interactions')


@dataclass
class Config:
    cache_dir: Path = Path('cache/next_run')
    data_dir: Path = Path('data/nature')
    settings: Path | None = None  # JSON object keyed by stage names; values use dataclass field names.
    stages: tuple[str, ...] = FULL_SESSION
    n_jobs: int = 1  # Session workers for selection, trial workers for decoding.
    session_list_file: Path | None = None
    max_sessions_to_run: int | None = None
    figure_formats: tuple[FigureFormat, ...] = ('png',)
    figure_font: str = DEFAULT_FIGURE_FONT  # Font family for all figures; unavailable fonts warn and fall back to DejaVu Sans.
    trust_unverified_legacy_results: bool = False  # Manually accept unverifiable legacy decoding keys for this invocation; recorded in the manifest.
    dry_run: bool = False  # Print resolved settings without running any analysis.


def resolve_config(module, shared, overrides, *, expand_paths=True):
    names = {field.name for field in fields(module.Config)}
    unknown = set(overrides) - names
    if unknown:
        raise ValueError(f'{module.__name__}: unknown settings {sorted(unknown)}.')
    values = {k: v for k, v in shared.items() if k in names}
    values.update(overrides)
    hints = get_type_hints(module.Config)
    for key, value in list(values.items()):
        hint = hints[key]
        alternatives = (hint,) + get_args(hint)
        if value is not None and Path in alternatives:
            path = Path(value)
            if expand_paths:
                try:
                    path = path.expanduser()
                except RuntimeError as exc:
                    raise ValueError(f'{key}: cannot resolve the home directory in {str(value)!r}. '
                                     'Choose a local path or an existing user.') from exc
            values[key] = path
        elif isinstance(hint, type) and issubclass(hint, Enum):
            # JSON accepts enum CLI names (SIGMOID) as well as values (sigmoid).
            values[key] = hint.__members__.get(str(value), None) or hint(value)
    return module.Config(**values)


def main(config: Config, *, argv=None):
    invocation = invocation_context(argv)
    validate_figure_formats(config.figure_formats)
    validate_figure_font(config.figure_font)
    if not isinstance(config.trust_unverified_legacy_results, bool):
        raise ValueError('trust_unverified_legacy_results must be true or false.')
    if (config.trust_unverified_legacy_results
            and not primary_cache(config.cache_dir, 'decoding_confidence.pkl').is_file()):
        raise ValueError('Trusting legacy results requires an existing run with decode/decoding_confidence.pkl.')
    stages = config.stages
    if stages == ('all',):
        stages = tuple(STAGES)
    elif stages == ('mixed',):
        stages = MIXED
    unknown_stages = set(stages) - set(STAGES)
    if unknown_stages:
        raise ValueError(f'Unknown stages: {sorted(unknown_stages)}. Choose from {tuple(STAGES)}, or all / mixed.')
    if not stages or len(set(stages)) != len(stages):
        raise ValueError(f'Choose stages from {tuple(STAGES)}, or all / mixed.')
    settings = json.loads(config.settings.read_text()) if config.settings else {}
    if not isinstance(settings, dict):
        raise ValueError(f'Settings must be an object keyed by {tuple(STAGES)}.')
    unknown_settings_stages = set(settings) - set(STAGES)
    if unknown_settings_stages:
        raise ValueError(f'Unknown settings stages: {sorted(unknown_settings_stages)}. Choose from {tuple(STAGES)}.')
    shared = {
        'cache_dir': config.cache_dir, 'data_dir': config.data_dir,
        'n_jobs': config.n_jobs, 'n_jobs_session': config.n_jobs, 'cv_n_jobs': config.n_jobs,
        'session_list_file': config.session_list_file,
        'max_sessions_to_run': config.max_sessions_to_run,
    }
    resolved = []
    for stage in stages:
        module = importlib.import_module(f'{__package__}.{STAGES[stage]}' if __package__ else STAGES[stage])
        overrides = settings.get(stage, {})
        if not isinstance(overrides, dict):
            raise ValueError(f'Settings for {stage} must be an object.')
        if 'cache_dir' in overrides or 'data_dir' in overrides:
            raise ValueError('Set cache_dir and data_dir on the pipeline command to keep stages aligned.')
        stage_config = resolve_config(module, shared, overrides)
        if stage == 'select' and stage_config.save_extended_diagnostics:
            load_diagnostic_config(stage_config.diagnostics_figure_config)
        resolved.append((stage, module, stage_config))
    if config.dry_run:
        print(json.dumps({s: asdict(c) for s, _, c in resolved}, indent=2, default=json_value))
        return
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    with figure_format_context(config.figure_formats), figure_font_context(config.figure_font) as resolved_font:
        manifest = RunManifest(config.cache_dir, {s: asdict(c) for s, _, c in resolved},
                               {**asdict(config), 'resolved_figure_font': resolved_font}, invocation=invocation)
        def persist_trust(audit):
            manifest.record['legacy_trust'] = audit
            manifest.save()

        try:
            with legacy_trust_context(config.cache_dir, enabled=config.trust_unverified_legacy_results,
                                      on_update=persist_trust) as trust:
                persist_trust(trust.audit)
                for stage, module, stage_config in resolved:
                    trust.set_stage(stage)
                    started = time.monotonic()
                    entry = {'stage': stage, 'status': 'running',
                             'started_at': utc_now().isoformat(), 'finished_at': None}
                    manifest.record['stages'].append(entry)
                    manifest.save()
                    print(f'\n[{stage}]', flush=True)
                    try:
                        # Fitting mode checks the checkpoints it actually reuses.
                        # Plotting/downstream-only stages need an explicit audit.
                        if (config.trust_unverified_legacy_results and stage != 'select'
                                and (stage != 'decode' or stage_config.plot_only)):
                            audit_cached_decoding(config.cache_dir, config.data_dir)
                        with threadpool_limits(limits=1):
                            module.main(stage_config)
                    except BaseException as exc:
                        entry.update(status='failed' if isinstance(exc, Exception) else 'interrupted',
                                     error=f'{type(exc).__name__}: {exc}')
                        raise
                    else:
                        entry['status'] = 'complete'
                    finally:
                        entry['seconds'] = round(time.monotonic() - started, 3)
                        entry['finished_at'] = utc_now().isoformat()
                        manifest.save()
        except BaseException as exc:
            manifest.finish('failed' if isinstance(exc, Exception) else 'interrupted')
            raise
        else:
            manifest.finish('complete')


if __name__ == '__main__':
    main(tyro.cli(Config), argv=sys.orig_argv)
