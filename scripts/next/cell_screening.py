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
from scripts.next.diagnostic_config import load_diagnostic_config
from scripts.next.screening_names import CHECK_NAMES, MEASUREMENT_COLUMNS, REASONS
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
    min_trials_per_session: int = 320
    check_firing_rate: bool = False  # Require mean correct-trial test-period rate >= min_test_firing_rate_hz.
    min_test_firing_rate_hz: float = 0.0  # Nonnegative firing-rate threshold in Hz.
    check_presence_ratio: bool = True  # Require a spike on this fraction of correct trials.
    min_presence_ratio: float = 0.9
    presence_start_ms: int = -400
    presence_end_ms: int = 1400

    check_delay_variance: bool = False  # Require delay/baseline variance ratio above the cutoff.
    min_delay_to_baseline_variance_ratio: float = 1.0
    check_baseline_variance: bool = False  # Require sliding/global baseline variance ratio above the cutoff.
    min_baseline_window_variance_ratio: float = 0.5
    variance_window_trials: int = 50  # Minimum correct trials for both variance checks; baseline sliding-window length.
    variance_baseline_start_ms: int = -500
    variance_baseline_end_ms: int = 0
    variance_delay_start_ms: int = 500
    variance_delay_end_ms: int = 1000

    check_baseline_drift: bool = True  # Require |Pearson r| <= baseline drift cutoff on correct trials.
    max_abs_baseline_drift_r: float = 0.3
    baseline_drift_start_ms: int = -400
    baseline_drift_end_ms: int = 0
    check_selectivity: bool = True  # Reject cells without a sufficiently long above-threshold PEV run.
    selectivity_pev_threshold_pct: float = 2.5  # PEV percentage in [0, 100], strict lower cutoff.
    selectivity_min_duration_ms: int = 100  # Required contiguous duration in ms (bin count times stride).
    selectivity_pev_floor_pct: float = 0.0
    check_preferred_cue_drift: bool = False  # Require |Pearson r| <= cutoff on all preferred-cue trials.
    max_abs_preferred_cue_drift_r: float = 0.3

    # Test period for rate and preferred-cue drift; bin-start grid for PEV metadata.
    test_start_ms: int = 500
    test_end_ms: int = 1400
    selectivity_bin_width_ms: int = 50
    selectivity_bin_step_ms: int = 10

    save_extended_diagnostics: bool = False  # Save per-cell measurements and optional configured plots.
    diagnostics_figure_config: Path | None = None  # Next diagnostic JSON; None saves the CSV without cell figures.

    def __post_init__(self):
        if not isinstance(self.save_extended_diagnostics, bool):
            raise ValueError('save_extended_diagnostics must be true or false.')
        for name, value in vars(self).items():
            if name.startswith('check_') and not isinstance(value, bool):
                raise ValueError(f'{name} must be true or false.')
        for name in ('min_trials_per_session', 'variance_window_trials', 'selectivity_bin_width_ms',
                     'selectivity_bin_step_ms', 'selectivity_min_duration_ms'):
            value = getattr(self, name)
            minimum = 2 if name == 'variance_window_trials' else 1
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}.')
        if isinstance(self.n_jobs_session, bool) or not isinstance(self.n_jobs_session, Integral) or self.n_jobs_session == 0:
            raise ValueError('n_jobs_session must be a nonzero integer.')
        if self.max_sessions_to_run is not None and (
            isinstance(self.max_sessions_to_run, bool) or not isinstance(self.max_sessions_to_run, Integral)
            or self.max_sessions_to_run < 1
        ):
            raise ValueError('max_sessions_to_run must be a positive integer.')
        for start, end in (('test_start_ms', 'test_end_ms'), ('presence_start_ms', 'presence_end_ms'),
                           ('baseline_drift_start_ms', 'baseline_drift_end_ms'),
                           ('variance_baseline_start_ms', 'variance_baseline_end_ms'),
                           ('variance_delay_start_ms', 'variance_delay_end_ms')):
            for name in (start, end):
                value = getattr(self, name)
                if isinstance(value, bool) or not isinstance(value, Integral):
                    raise ValueError(f'{name} must be a finite integer in milliseconds.')
            if getattr(self, end) <= getattr(self, start):
                raise ValueError(f'{end} must follow {start}.')
        limits = {'min_test_firing_rate_hz': (0, np.inf), 'min_presence_ratio': (0, 1),
                  'min_delay_to_baseline_variance_ratio': (0, np.inf),
                  'min_baseline_window_variance_ratio': (0, np.inf),
                  'max_abs_preferred_cue_drift_r': (0, 1), 'max_abs_baseline_drift_r': (0, 1),
                  'selectivity_pev_threshold_pct': (0, 100), 'selectivity_pev_floor_pct': (0, 100)}
        for name, (lower, upper) in limits.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f'{name} must be finite and in [{lower}, {upper}]. Use the corresponding --no-check-* flag to disable a check.')


class ScreeningResults:
    """Accumulate each enabled check's decision without short-circuiting cells."""

    def __init__(self, num_cells, config):
        self.selected = np.ones(num_cells, dtype=bool)
        self.reasons = [[] for _ in range(num_cells)]
        self.status = {check_name: np.full(num_cells, 'disabled', dtype=object) for check_name in CHECK_NAMES}
        self.enabled = {check_name: getattr(config, f'check_{check_name}') for check_name in CHECK_NAMES}

    def apply(self, check_name, measurements, passes_threshold):
        if not self.enabled[check_name]:
            return
        has_measurement = np.isfinite(measurements)
        self.status[check_name][:] = np.where(
            ~has_measurement, 'not_applicable', np.where(passes_threshold, 'pass', 'fail'))
        self.selected &= has_measurement & passes_threshold
        for cell_index in np.flatnonzero(~has_measurement | ~passes_threshold):
            unavailable_suffix = '_not_applicable' if not has_measurement[cell_index] else ''
            self.reasons[cell_index].append(REASONS[check_name] + unavailable_suffix)

    def diagnostics(self, session, measurements):
        return [dict(session=session, cell_idx=cell_index, is_rejected=bool(failed_checks),
                     rejection_reason='|'.join(failed_checks) or 'pass',
                     **{f'check_{check_name}': status[cell_index]
                        for check_name, status in self.status.items()},
                     **{column: cell_measurements[cell_index]
                        for column, cell_measurements in measurements.items()})
                for cell_index, failed_checks in enumerate(self.reasons)]


def process_session(data_file, config):
    session = data_file.stem
    session_data = checks.SessionMeasurements(*load_session(data_file))
    num_trials = session_data.spikes.shape[0]
    if config.check_min_trials and num_trials < config.min_trials_per_session:
        print(f'{session}: skipped; {num_trials} trials < {config.min_trials_per_session}', flush=True)
        return None, []
    correct_trial_cues = np.unique(session_data.cues[session_data.correct])
    num_correct_trials = int(session_data.correct.sum())
    # Required for cue metadata, even if PEV rejection is disabled.
    if correct_trial_cues.size < 2 or num_correct_trials <= correct_trial_cues.size:
        print(f'{session}: skipped; cue metadata requires at least two correct-trial cues and residual degrees of freedom', flush=True)
        return None, []
    screening_results = ScreeningResults(session_data.num_cells, config)
    measurements = {}
    stationary_check_specs = (
        ('firing_rate', checks.firing_rate,
         lambda firing_rates: firing_rates >= config.min_test_firing_rate_hz),
        ('presence_ratio', checks.presence_ratio,
         lambda presence_ratios: presence_ratios >= config.min_presence_ratio),
        ('delay_variance', checks.delay_variance,
         lambda variance_ratios: variance_ratios > config.min_delay_to_baseline_variance_ratio),
        ('baseline_variance', checks.baseline_variance,
         lambda variance_ratios: variance_ratios > config.min_baseline_window_variance_ratio),
        ('baseline_drift', checks.baseline_drift,
         lambda drift_correlations: np.abs(drift_correlations) <= config.max_abs_baseline_drift_r),
    )
    for check_name, measure_check, passes_threshold in stationary_check_specs:
        if screening_results.enabled[check_name]:
            check_measurements = measure_check(session_data, config)
            measurements[MEASUREMENT_COLUMNS[check_name]] = check_measurements
            screening_results.apply(check_name, check_measurements, passes_threshold(check_measurements))
    passes_stationary_checks = screening_results.selected.copy()
    # Cue/PEV metadata is required by decoding regardless of the PEV rejection switch.
    selectivity = checks.selectivity(session_data, config)
    measurements[MEASUREMENT_COLUMNS['selectivity']] = selectivity.mean_pev_pct
    screening_results.apply(
        'selectivity', np.where(selectivity.has_finite_pev, selectivity.mean_pev_pct, np.nan),
        selectivity.passes_duration_check)
    if config.check_preferred_cue_drift:
        preferred_cue_drift_r = checks.preferred_cue_drift(session_data, config, selectivity.preferred_cue)
        measurements[MEASUREMENT_COLUMNS['preferred_cue_drift']] = preferred_cue_drift_r
        screening_results.apply(
            'preferred_cue_drift', preferred_cue_drift_r,
            np.abs(preferred_cue_drift_r) <= config.max_abs_preferred_cue_drift_r)
    diagnostic_rows = screening_results.diagnostics(session, measurements)
    if not screening_results.selected.any():
        return None, diagnostic_rows
    selected_cell_indices = np.flatnonzero(screening_results.selected)
    if not np.all(np.isfinite(selectivity.preferred_cue[selected_cell_indices])):
        raise ValueError(f'{session}: selected cells have unavailable cue metadata; check input spike data and test windows.')
    preferred_cue_drift_status = screening_results.status['preferred_cue_drift']
    stationary_cell_indices = np.flatnonzero(passes_stationary_checks & (
        (preferred_cue_drift_status == 'pass') | (preferred_cue_drift_status == 'disabled')))
    selected_preferred_cues = selectivity.preferred_cue[selected_cell_indices].astype(np.int64)
    selected_mean_pev_pct = selectivity.mean_pev_pct[selected_cell_indices]
    preferred_cue_groups = selected_preferred_cues[None, :] == correct_trial_cues[:, None]
    cell_count_per_cue = preferred_cue_groups.sum(axis=1)
    total_pev_per_cue = (preferred_cue_groups * selected_mean_pev_pct[None, :]).sum(axis=1)
    # A disabled presence gate admits all cells in the corresponding decoder mode.
    presence_status = screening_results.status['presence_ratio']
    presence_passed_cell_indices = np.flatnonzero((presence_status == 'pass') | (presence_status == 'disabled'))
    selected_cell_properties = {
        column: cell_measurements[selected_cell_indices] for column, cell_measurements in measurements.items()}
    selected_cell_properties.update(cell_idx=selected_cell_indices, preferred_cue=selected_preferred_cues)
    if config.check_selectivity:
        selected_cell_properties['qualifying_selectivity_bin_count'] = (
            selectivity.qualifying_bin_mask[selected_cell_indices].sum(axis=1))
    session_record = {
        'session': session, 'num_trials': num_trials, 'num_trials_selected': num_correct_trials,
        'num_cells_selected': selected_cell_indices.size, 'cell_idx_selected': selected_cell_indices,
        'trial_idx_selected': np.flatnonzero(session_data.correct), 'labels_set_idx': correct_trial_cues,
        'labels_set_deg': (correct_trial_cues - 1) * 45 - 135,
        'num_cells_per_group': cell_count_per_cue, 'total_pev_per_group': total_pev_per_cue,
        'max_num_cells_per_group': cell_count_per_cue.max(), 'max_total_pev_per_group': total_pev_per_cue.max(),
        'cell_idx_stationary': stationary_cell_indices, 'cell_idx_passed_presence_ratio': presence_passed_cell_indices,
        'cell_properties': selected_cell_properties,
        'screening_checks': {'min_trials': config.check_min_trials, **screening_results.enabled},
        'screening_config': {key: json_value(value) if isinstance(value, Path) else value
                             for key, value in asdict(config).items()},
    }
    return session_record, diagnostic_rows


def main(config: Config):
    # Fail on malformed diagnostic settings before loading recordings or screening.
    diagnostic_config = (load_diagnostic_config(config.diagnostics_figure_config)
                         if config.save_extended_diagnostics else None)
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
            save_diagnostics(diagnostic_rows, files, config, diagnostic_config)
        raise ValueError('No sessions passed cell screening; inspect data and enabled checks.')
    cache_io.save(results, primary_cache(config.cache_dir, 'cell_screening.pkl'))
    table_path = stage_path(config.cache_dir, 'select', 'tables', 'cell_screening.csv')
    table_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(table_path, index=False)
    if config.save_extended_diagnostics:
        save_diagnostics(diagnostic_rows, files, config, diagnostic_config)
    return results


if __name__ == '__main__':
    main(tyro.cli(Config))
