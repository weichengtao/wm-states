"""Screen complete sessions with independently enabled, validated cell checks."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "scripts.next"

from scripts.next.cache_paths import primary_cache, stage_path
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import pandas as pd
import tyro
from joblib import Parallel, delayed

from scripts.next import cache_io, screening_checks as checks
from scripts.next.common import json_value, load_session, session_files, worker_context
from scripts.next.selection_diagnostics import save_diagnostics


@dataclass
class Config:
    """Full-session screening. Disable checks with --no-check-<name>, not thresholds.

    Times are milliseconds relative to cue onset. Disabled checks do not reject
    unavailable measurements. Input validity and cue metadata remain required.
    """

    data_dir: Path = Path('data/nature')
    cache_dir: Path = Path('cache/next_run')
    n_jobs_session: int = 1
    session_list_file: Path | None = None
    max_sessions_to_run: int | None = None

    check_min_trials: bool = True  # Gate on total session trials (correct + incorrect).
    min_trial_per_session: int = 320
    check_firing_rate: bool = False  # Require mean correct-trial test-period rate >= min_fr_test.
    min_fr_test: float = 0.0  # Nonnegative firing-rate threshold in Hz.
    check_presence_ratio: bool = True  # Require a spike on this fraction of correct trials.
    min_presence_ratio: float = 0.9
    presence_start: int = -400
    presence_end: int = 1400

    check_delay_variance: bool = False  # Require delay/baseline variance ratio above the cutoff.
    var_ratio_threshold_delay_over_baseline: float = 1.0
    check_baseline_variance: bool = False  # Require sliding/global baseline variance ratio above the cutoff.
    var_ratio_threshold_sliding_over_all: float = 0.5
    min_trial_for_temp_check: int = 50  # Minimum correct trials, also the sliding window length.
    temp_check_baseline_start: int = -500
    temp_check_baseline_end: int = 0
    temp_check_delay_start: int = 500
    temp_check_delay_end: int = 1000

    check_baseline_drift: bool = True  # Require |Pearson r| <= baseline drift cutoff on correct trials.
    temp_dep_r_threshold_baseline: float = 0.3
    baseline_drift_start: int = -400
    baseline_drift_end: int = 0
    check_selectivity: bool = True  # Reject cells without a sufficiently long above-threshold PEV run.
    sig_pev_threshold: float = 2.5  # PEV percentage in [0, 100], strict lower cutoff.
    sig_pev_duration: int = 100  # Required contiguous duration in ms (bin count times stride).
    pev_clip_at: float = 0.0
    check_preferred_drift: bool = False  # Require |Pearson r| <= cutoff on all preferred-cue trials.
    temp_dep_r_threshold: float = 0.3

    # Test period for rate and preferred-cue drift; bin-start grid for PEV metadata.
    t_test_start: int = 500
    t_test_end: int = 1400
    t_test_window: int = 50
    t_test_step: int = 10

    save_extended_diagnostics: bool = False
    diagnostics_figure_config: Path | None = None
    skip_not_applicable_reasons_in_diagnostics_figure: bool = True

    def __post_init__(self):
        for name, value in vars(self).items():
            if name.startswith('check_') and not isinstance(value, bool):
                raise ValueError(f'{name} must be true or false.')
        for name in ('min_trial_per_session', 'min_trial_for_temp_check', 't_test_window',
                     't_test_step', 'sig_pev_duration'):
            value = getattr(self, name)
            minimum = 2 if name == 'min_trial_for_temp_check' else 1
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}.')
        if isinstance(self.n_jobs_session, bool) or not isinstance(self.n_jobs_session, Integral) or self.n_jobs_session == 0:
            raise ValueError('n_jobs_session must be a nonzero integer.')
        if self.max_sessions_to_run is not None and (
            isinstance(self.max_sessions_to_run, bool) or not isinstance(self.max_sessions_to_run, Integral)
            or self.max_sessions_to_run < 1
        ):
            raise ValueError('max_sessions_to_run must be a positive integer.')
        for start, end in (('t_test_start', 't_test_end'), ('presence_start', 'presence_end'),
                           ('baseline_drift_start', 'baseline_drift_end'),
                           ('temp_check_baseline_start', 'temp_check_baseline_end'),
                           ('temp_check_delay_start', 'temp_check_delay_end')):
            for name in (start, end):
                value = getattr(self, name)
                if isinstance(value, bool) or not isinstance(value, Integral):
                    raise ValueError(f'{name} must be a finite integer in milliseconds.')
            if getattr(self, end) <= getattr(self, start):
                raise ValueError(f'{end} must follow {start}.')
        limits = {'min_fr_test': (0, np.inf), 'min_presence_ratio': (0, 1),
                  'var_ratio_threshold_delay_over_baseline': (0, np.inf),
                  'var_ratio_threshold_sliding_over_all': (0, np.inf),
                  'temp_dep_r_threshold': (0, 1), 'temp_dep_r_threshold_baseline': (0, 1),
                  'sig_pev_threshold': (0, 100), 'pev_clip_at': (0, 100)}
        for name, (lower, upper) in limits.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f'{name} must be finite and in [{lower}, {upper}]. Use the corresponding --no-check-* flag to disable a check.')


# Check order defines diagnostic columns and the pre-selectivity stationary pool.
CHECK_NAMES = ('firing_rate', 'presence_ratio', 'delay_variance', 'baseline_variance',
               'baseline_drift', 'selectivity', 'preferred_drift')
REASONS = dict(zip(CHECK_NAMES, (
    'fail_min_fr_test', 'fail_min_presence_ratio', 'fail_temp_dep_stage1',
    'fail_temp_dep_stage2', 'fail_temp_dep_stage3_baseline', 'fail_sig_pev',
    'fail_temp_dep_stage3',
)))


class ScreeningResults:
    """Accumulate each enabled check's decision without short-circuiting cells."""

    def __init__(self, size, config):
        self.selected = np.ones(size, dtype=bool)
        self.reasons = [[] for _ in range(size)]
        self.status = {name: np.full(size, 'disabled', dtype=object) for name in CHECK_NAMES}
        self.enabled = {name: getattr(config, f'check_{name}') for name in CHECK_NAMES}

    def apply(self, name, values, passes):
        if not self.enabled[name]:
            return
        finite = np.isfinite(values)
        self.status[name][:] = np.where(~finite, 'not_applicable', np.where(passes, 'pass', 'fail'))
        self.selected &= finite & passes
        for cell in np.flatnonzero(~finite | ~passes):
            self.reasons[cell].append(REASONS[name] + ('_not_applicable' if not finite[cell] else ''))

    def diagnostics(self, session, measurements):
        return [dict(session=session, cell_idx=cell, is_rejected=bool(reasons),
                     rejection_reason='|'.join(reasons) or 'pass',
                     **{f'check_{name}': status[cell] for name, status in self.status.items()},
                     **{name: values[cell] for name, values in measurements.items()})
                for cell, reasons in enumerate(self.reasons)]


def process_session(data_file, config):
    session = data_file.stem
    data = checks.SessionMeasurements(*load_session(data_file))
    num_trials = data.spikes.shape[0]
    if config.check_min_trials and num_trials < config.min_trial_per_session:
        print(f'{session}: skipped; {num_trials} trials < {config.min_trial_per_session}', flush=True)
        return None, []
    labels = np.unique(data.cues[data.correct])
    num_correct = int(data.correct.sum())
    # Required for cue metadata, even if PEV rejection is disabled.
    if labels.size < 2 or num_correct <= labels.size:
        print(f'{session}: skipped; cue metadata requires at least two correct-trial cues and residual degrees of freedom', flush=True)
        return None, []
    result = ScreeningResults(data.num_cells, config)
    measurements = {}
    specifications = (
        ('firing_rate', checks.firing_rate, 'mean_fr_test', lambda x: x >= config.min_fr_test),
        ('presence_ratio', checks.presence_ratio, 'presence_ratio', lambda x: x >= config.min_presence_ratio),
        ('delay_variance', checks.delay_variance, 'temp_dep_var_ratio_stage1', lambda x: x > config.var_ratio_threshold_delay_over_baseline),
        ('baseline_variance', checks.baseline_variance, 'temp_dep_sliding_ratio_stage2', lambda x: x > config.var_ratio_threshold_sliding_over_all),
        ('baseline_drift', checks.baseline_drift, 'temp_dep_r_baseline', lambda x: np.abs(x) <= config.temp_dep_r_threshold_baseline),
    )
    for name, measure, column, passes in specifications:
        if result.enabled[name]:
            values = measure(data, config)
            measurements[column] = values
            result.apply(name, values, passes(values))
    before_pev = result.selected.copy()
    # Cue/PEV metadata is required by decoding regardless of the PEV rejection switch.
    pev = checks.selectivity(data, config)
    measurements['mean_pev_test'] = pev.mean_pev
    result.apply('selectivity', np.where(pev.applicable, pev.mean_pev, np.nan), pev.passes)
    if config.check_preferred_drift:
        values = checks.preferred_drift(data, config, pev.preferred_cue)
        measurements['temp_dep_r'] = values
        result.apply('preferred_drift', values, np.abs(values) <= config.temp_dep_r_threshold)
    rows = result.diagnostics(session, measurements)
    if not result.selected.any():
        return None, rows
    selected = np.flatnonzero(result.selected)
    if not np.all(np.isfinite(pev.preferred_cue[selected])):
        raise ValueError(f'{session}: selected cells have unavailable cue metadata; check input spike data and test windows.')
    preferred_status = result.status['preferred_drift']
    stationary = np.flatnonzero(before_pev & ((preferred_status == 'pass') | (preferred_status == 'disabled')))
    preferences = pev.preferred_cue[selected].astype(np.int64)
    mean_pev = pev.mean_pev[selected]
    groups = preferences[None, :] == labels[:, None]
    group_counts = groups.sum(axis=1)
    group_pev = (groups * mean_pev[None, :]).sum(axis=1)
    # A disabled presence gate admits all cells in the corresponding decoder mode.
    presence_status = result.status['presence_ratio']
    passed_presence = np.flatnonzero((presence_status == 'pass') | (presence_status == 'disabled'))
    properties = {name: values[selected] for name, values in measurements.items()}
    properties.update(cell_idx=selected, mean_pref_test=preferences)
    if config.check_selectivity:
        properties['num_sig_pev_bins'] = pev.significant_bins[selected].sum(axis=1)
    out = {
        'session': session, 'num_trials': num_trials, 'num_trials_selected': num_correct,
        'num_cells_selected': selected.size, 'cell_idx_selected': selected,
        'trial_idx_selected': np.flatnonzero(data.correct), 'labels_set_idx': labels,
        'labels_set_deg': (labels - 1) * 45 - 135,
        'num_cells_per_group': group_counts, 'total_pev_per_group': group_pev,
        'max_num_cells_per_group': group_counts.max(), 'max_total_pev_per_group': group_pev.max(),
        'cell_idx_stationary': stationary, 'cell_idx_passed_presence_ratio': passed_presence,
        'cell_properties': properties,
        'screening_checks': {'min_trials': config.check_min_trials, **result.enabled},
        'screening_config': {key: json_value(value) if isinstance(value, Path) else value
                             for key, value in asdict(config).items()},
    }
    return out, rows


def main(config: Config):
    files = session_files(config.data_dir, config.session_list_file, config.max_sessions_to_run)
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    results, diagnostic_rows = [], []
    with worker_context(config.n_jobs_session):
        completed = Parallel(return_as='generator')(delayed(process_session)(path, config) for path in files)
        for path, (result, rows) in zip(files, completed):
            print(f'{path.stem}: {0 if result is None else result["num_cells_selected"]} selected cells', flush=True)
            if result is not None:
                results.append(result)
            diagnostic_rows.extend(rows)
    if not results:
        if config.save_extended_diagnostics and diagnostic_rows:
            save_diagnostics(diagnostic_rows, files, config)
        raise ValueError('No sessions passed cell screening; inspect data and enabled checks.')
    cache_io.save(results, primary_cache(config.cache_dir, 'cell_screening.pkl'))
    table_path = stage_path(config.cache_dir, 'select', 'tables', 'cell_screening.csv')
    table_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(table_path, index=False)
    if config.save_extended_diagnostics:
        save_diagnostics(diagnostic_rows, files, config)
    return results


if __name__ == '__main__':
    main(tyro.cli(Config))
