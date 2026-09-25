"""Select stable, cue-selective cells using every trial in each session."""

# Use one module namespace for direct CLI and package execution.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "scripts.next"

import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import tyro
from joblib import Parallel, delayed
from scripts.next.selection_math import get_periods_and_mask, pev_and_preferred_cue
from scipy.stats import circmean, linregress, spearmanr
from scripts.next.common import compute_binned_rates, load_session, session_files, worker_context
from scripts.next import cache_io
from scripts.next.selection_diagnostics import save_diagnostics


def check_temporal_stability_preferred_trials(
    spikes: np.ndarray,
    cue_labels: np.ndarray,
    preferred_cues: np.ndarray,
    active_cell_idx: np.ndarray,
    t: np.ndarray,
    config: 'Config',
):
    """
    Correlation-based temporal stability check (stage 3) on trials with each cell's preferred cue.

    Returns
    -------
    keep : np.ndarray
        Boolean mask over `preferred_cues` indicating cells that pass the stability check.
    slopes, intercepts, r_values : np.ndarray
        Regression parameters for downstream analyses.
    """
    t_test_mask = (t >= config.t_test_start) & (t < config.t_test_end)
    bin_width_s = (t[1] - t[0]) / 1000.0
    test_duration_s = np.sum(t_test_mask) * bin_width_s

    cue_trials = {cue: np.flatnonzero(cue_labels == cue) for cue in np.unique(preferred_cues)}

    # test-period firing rate for every trial (including incorrect) and selected cell
    spikes_window = spikes[:, :, active_cell_idx]
    test_rates = spikes_window[:, t_test_mask, :].sum(axis=1) / test_duration_s

    num_cells = len(active_cell_idx)
    keep = np.ones(num_cells, dtype=np.bool_)
    slopes = np.full(num_cells, np.nan)
    intercepts = np.full(num_cells, np.nan)
    r_values = np.full(num_cells, np.nan)

    for i_cell, cue in enumerate(preferred_cues):
        trial_idx = cue_trials[cue]
        if len(trial_idx) < 2:
            keep[i_cell] = False
            continue
        fr = test_rates[trial_idx, i_cell]
        res = linregress(trial_idx, fr)
        slopes[i_cell] = res.slope
        intercepts[i_cell] = res.intercept
        r_values[i_cell] = res.rvalue
        keep[i_cell] = np.abs(res.rvalue) <= config.temp_dep_r_threshold

    return keep, slopes, intercepts, r_values


def cue_to_deg(cue):
    """
    Map cue indices 1-8 to degrees:
        1 -> -135, 2 -> -90, 3 -> -45, 4 -> 0, 5 -> 45, 6 -> 90, 7 -> 135, 8 -> 180.
    """
    cue = np.asarray(cue)
    cue = (cue - 1) % 8 + 1 # normalize cue index to [1, 8]
    return (cue - 1) * 45 - 135 # map to degrees


def deg_to_cue(deg):
    """
    Map degrees back to cue indices:
        -135 -> 1, -90 -> 2, -45 -> 3, 0 -> 4, 45 -> 5, 90 -> 6, 135 -> 7, 180 -> 8.
    """
    deg = np.asarray(deg)
    cue = (deg + 135) / 45 + 1 # map degrees to cue index
    return (cue - 1) % 8 + 1 # normalize cue index to [1, 8]


def circular_mean_cue(cue):
    """Compute circular mean of cue indices using their angular representation."""
    rad = np.deg2rad(cue_to_deg(cue))
    circular_mean_rad = circmean(rad, high=np.pi, low=-np.pi)
    res = deg_to_cue(np.rad2deg(circular_mean_rad))
    return (res.round().astype(np.int64) - 1) % 8 + 1 # round to nearest cue index and normalize to [1, 8]


@dataclass
class Config:
    """
    Configuration for selecting stable, cue-selective cells.

    All time values are in milliseconds relative to cue onset; directory paths are
    resolved relative to the project root.
    """
    n_jobs_session: int = 1 # Parallel CPU workers across sessions
    seed: int = 42  # Random seed for any stochastic routines

    data_dir: Path = Path('data/nature') # Folder with {session}.mat files containing spks/isCorr/cueAngIdx/tc
    cache_dir: Path = Path('cache/next_run') # Output directory for pkl/csv summaries


    t_test_start: int = 500 # Analysis window start for selectivity tests (delay)
    t_test_end: int = 1400 # Analysis window end for selectivity tests
    t_test_window: int = 50 # Test bin width in ms
    t_test_step: int = 10 # Test bin stride in ms

    min_cell_per_group: int = 1 # Minimum cells per cue location to keep for downstream analyses
    min_fr_test: float = -1.0 # Hz threshold on mean firing rate during the test window
    min_presence_ratio: float = 0.9 # Fraction of selected-correct trials where a cell must fire at least once in [-400, 1400) ms
    temp_dep_detection: bool = True # Drop cells that show strong temporal dependence
    min_trial_for_temp_check: int = 50 # Require this many trials before running temporal checks
    var_ratio_threshold_delay_over_baseline: float = -1 # Delay-vs-baseline variance ratio cutoff (stage 1)
    var_ratio_threshold_sliding_over_all: float = -1 # Sliding-window-vs-global baseline variance ratio cutoff (stage 2)
    temp_dep_r_threshold: float = 2 # Minimum correlation coefficient to flag temporal dependence
    temp_dep_r_threshold_baseline: float = 0.3 # |Pearson r| cutoff for baseline-count drift over selected-correct trials (stage 3 baseline)
    sig_pev_threshold: float = 2.5 # Percent explained variance threshold to call a bin selective
    sig_pev_duration: int = 100 # Minimum contiguous duration (ms) a cell must stay selective
    pev_clip_at: float = 0 # Lower bound when clipping PEV values

    temp_check_baseline_start: int = -500 # Baseline window start for temporal dependence checks
    temp_check_baseline_end: int = 0 # Baseline window end for temporal dependence checks
    temp_check_delay_start: int = 500 # Delay window start for temporal dependence checks
    temp_check_delay_end: int = 1000 # Delay window end for temporal dependence checks

    min_trial_per_session: int = 320 # Minimum total trials (correct + incorrect) required to process a session

    save_extended_diagnostics: bool = False # Save additional diagnostic measures and figures
    diagnostics_figure_config: Path | None = None # Optional JSON config listing which per-cell diagnostic figures to save
    skip_not_applicable_reasons_in_diagnostics_figure: bool = True # Hide *_not_applicable reasons in diagnostics figure text unless they are the only reasons

    session_list_file: Path | None = None
    max_sessions_to_run: int | None = None

def process_session(data_file, config):
    session = data_file.stem
    spikes, t, cue_labels, trial_boo_correct = load_session(data_file)
    num_trials, _, num_cells_total = spikes.shape
    if num_trials < config.min_trial_per_session:
        return None, []
    dt = t[1] - t[0]
    trial_boo_selected = trial_boo_correct
    num_trials_selected = int(trial_boo_selected.sum())
    if num_trials_selected <= np.unique(cue_labels[trial_boo_selected]).size or np.unique(cue_labels[trial_boo_selected]).size < 2:
        return None, []
    cue_labels_selected = cue_labels[trial_boo_selected]
    labels_set = np.unique(cue_labels_selected)
    trial_filtered_spikes = spikes[trial_boo_selected]
    diag_presence_mask = (t >= -400) & (t < 1400)
    diag_baseline_mask = (t >= -400) & (t < 0)
    rejection_reasons = [[] for _ in range(num_cells_total)]

    def add_rejection_reason(mask, reason):
        for index in np.flatnonzero(mask):
            rejection_reasons[index].append(reason)

    def diagnostics():
        return [{'session': session, 'cell_idx': i, 'is_rejected': bool(reasons),
                 'rejection_reason': '|'.join(reasons) or 'pass'}
                for i, reasons in enumerate(rejection_reasons)]

    # Evaluate all criteria on all cells; final selection keeps cells with zero failures.
    cell_boo_selected = np.ones(num_cells_total, dtype=np.bool_)

    # criterion 1: minimum firing rate during the test period (selected-correct trials)
    t_test_mask = (t >= config.t_test_start) & (t < config.t_test_end)
    bin_width_s = dt / 1000.0 # convert ms to seconds
    mean_firing_rate_hz = np.full(num_cells_total, np.nan, dtype=np.float64)
    if np.sum(t_test_mask) == 0 or not np.isfinite(bin_width_s) or bin_width_s <= 0:
        fail_not_app = np.ones(num_cells_total, dtype=np.bool_)
        add_rejection_reason(fail_not_app, 'fail_min_fr_test_not_applicable')
        cell_boo_selected &= ~fail_not_app
    else:
        spikes_test_period = trial_filtered_spikes[:, t_test_mask, :]
        total_spike_counts = np.sum(spikes_test_period, axis=(0, 1))
        total_time_s = num_trials_selected * np.sum(t_test_mask) * bin_width_s
        if not np.isfinite(total_time_s) or total_time_s <= 0:
            fail_not_app = np.ones(num_cells_total, dtype=np.bool_)
            add_rejection_reason(fail_not_app, 'fail_min_fr_test_not_applicable')
            cell_boo_selected &= ~fail_not_app
        else:
            mean_firing_rate_hz = total_spike_counts / total_time_s
            finite_fr = np.isfinite(mean_firing_rate_hz)
            fail_not_app = ~finite_fr
            fail_min_fr = finite_fr & (mean_firing_rate_hz < config.min_fr_test)
            add_rejection_reason(fail_not_app, 'fail_min_fr_test_not_applicable')
            add_rejection_reason(fail_min_fr, 'fail_min_fr_test')
            cell_boo_selected &= ~(fail_not_app | fail_min_fr)

    # criterion 2: minimum presence ratio in [-400, 1400) ms (selected-correct trials)
    presence_ratio_selection = np.full(num_cells_total, np.nan, dtype=np.float64)
    if not np.any(diag_presence_mask):
        fail_not_app = np.ones(num_cells_total, dtype=np.bool_)
        add_rejection_reason(fail_not_app, 'fail_min_presence_ratio_not_applicable')
        cell_boo_selected &= ~fail_not_app
    else:
        presence_counts = np.sum(trial_filtered_spikes[:, diag_presence_mask, :], axis=1)
        presence_ratio_selection = np.mean(presence_counts > 0, axis=0).astype(np.float64)
        finite_presence = np.isfinite(presence_ratio_selection)
        fail_not_app = ~finite_presence
        fail_min_presence = finite_presence & (presence_ratio_selection < config.min_presence_ratio)
        add_rejection_reason(fail_not_app, 'fail_min_presence_ratio_not_applicable')
        add_rejection_reason(fail_min_presence, 'fail_min_presence_ratio')
        cell_boo_selected &= ~(fail_not_app | fail_min_presence)
    passed_presence_ratio = (
        np.isfinite(presence_ratio_selection)
        & (presence_ratio_selection >= config.min_presence_ratio)
    )

    var_ratio_stage1 = None
    sliding_ratio_stage2 = None
    slopes_stage3 = None
    intercepts_stage3 = None
    r_stage3 = None
    r_stage3_baseline = None

    # temporal dependency criteria (stages 1 and 2) are skipped when temp_dep_detection=False
    if config.temp_dep_detection:
        var_ratio_stage1 = np.full(num_cells_total, np.nan, dtype=np.float64)
        sliding_ratio_stage2 = np.full(num_cells_total, np.nan, dtype=np.float64)

        temp_baseline_mask = (t >= config.temp_check_baseline_start) & (t < config.temp_check_baseline_end)
        temp_delay_mask = (t >= config.temp_check_delay_start) & (t < config.temp_check_delay_end)

        baseline_rates_temp = None
        baseline_var_temp = None
        if np.any(temp_baseline_mask) and num_trials_selected >= 2 and np.isfinite(bin_width_s) and bin_width_s > 0:
            baseline_counts_temp = np.sum(trial_filtered_spikes[:, temp_baseline_mask, :], axis=1)
            baseline_rates_temp = baseline_counts_temp / (np.sum(temp_baseline_mask) * bin_width_s)
            baseline_var_temp = np.var(baseline_rates_temp, axis=0, ddof=1)

        stage1_global_ok = (
            num_trials_selected >= config.min_trial_for_temp_check
            and num_trials_selected >= 2
            and np.any(temp_baseline_mask)
            and np.any(temp_delay_mask)
            and np.isfinite(bin_width_s)
            and bin_width_s > 0
        )
        if not stage1_global_ok:
            fail_not_app = np.ones(num_cells_total, dtype=np.bool_)
            add_rejection_reason(fail_not_app, 'fail_temp_dep_stage1_not_applicable')
            cell_boo_selected &= ~fail_not_app
        else:
            delay_counts_temp = np.sum(trial_filtered_spikes[:, temp_delay_mask, :], axis=1)
            delay_rates_temp = delay_counts_temp / (np.sum(temp_delay_mask) * bin_width_s)
            delay_var_temp = np.var(delay_rates_temp, axis=0, ddof=1)
            with np.errstate(divide='ignore', invalid='ignore'):
                var_ratio = delay_var_temp / baseline_var_temp
            var_ratio_stage1[:] = var_ratio
            finite_stage1 = np.isfinite(var_ratio)
            fail_not_app = ~finite_stage1
            fail_stage1 = finite_stage1 & (var_ratio <= config.var_ratio_threshold_delay_over_baseline)
            add_rejection_reason(fail_not_app, 'fail_temp_dep_stage1_not_applicable')
            add_rejection_reason(fail_stage1, 'fail_temp_dep_stage1')
            cell_boo_selected &= ~(fail_not_app | fail_stage1)

        stage2_global_ok = (
            num_trials_selected >= config.min_trial_for_temp_check
            and config.min_trial_for_temp_check >= 2
            and np.any(temp_baseline_mask)
            and np.isfinite(bin_width_s)
            and bin_width_s > 0
        )
        if not stage2_global_ok:
            fail_not_app = np.ones(num_cells_total, dtype=np.bool_)
            add_rejection_reason(fail_not_app, 'fail_temp_dep_stage2_not_applicable')
            cell_boo_selected &= ~fail_not_app
        else:
            windows = np.lib.stride_tricks.sliding_window_view(
                baseline_rates_temp,
                window_shape=config.min_trial_for_temp_check,
                axis=0,
            )
            window_var = np.var(windows, axis=-1, ddof=1)
            sliding_var = np.mean(window_var, axis=0)
            with np.errstate(divide='ignore', invalid='ignore'):
                sliding_ratio = sliding_var / baseline_var_temp
            sliding_ratio_stage2[:] = sliding_ratio
            finite_stage2 = np.isfinite(sliding_ratio)
            fail_not_app = ~finite_stage2
            fail_stage2 = finite_stage2 & (sliding_ratio <= config.var_ratio_threshold_sliding_over_all)
            add_rejection_reason(fail_not_app, 'fail_temp_dep_stage2_not_applicable')
            add_rejection_reason(fail_stage2, 'fail_temp_dep_stage2')
            cell_boo_selected &= ~(fail_not_app | fail_stage2)

        # criterion: baseline activity trend over selected-correct trials (Pearson r)
        r_stage3_baseline = np.full(num_cells_total, np.nan, dtype=np.float64)
        stage3_baseline_global_ok = num_trials_selected >= 2 and np.any(diag_baseline_mask)
        if not stage3_baseline_global_ok:
            fail_not_app = np.ones(num_cells_total, dtype=np.bool_)
            add_rejection_reason(fail_not_app, 'fail_temp_dep_stage3_baseline_not_applicable')
            cell_boo_selected &= ~fail_not_app
        else:
            baseline_counts_stage3 = np.sum(trial_filtered_spikes[:, diag_baseline_mask, :], axis=1).astype(np.float64)
            x = np.arange(num_trials_selected, dtype=np.float64)
            x_centered = x - x.mean()
            x_norm = np.sqrt(np.sum(x_centered ** 2))
            if not np.isfinite(x_norm) or x_norm <= 0:
                fail_not_app = np.ones(num_cells_total, dtype=np.bool_)
                add_rejection_reason(fail_not_app, 'fail_temp_dep_stage3_baseline_not_applicable')
                cell_boo_selected &= ~fail_not_app
            else:
                fail_not_app = np.zeros(num_cells_total, dtype=np.bool_)
                fail_stage3_baseline = np.zeros(num_cells_total, dtype=np.bool_)
                for i_cell in range(num_cells_total):
                    y = baseline_counts_stage3[:, i_cell]
                    if not np.all(np.isfinite(y)):
                        fail_not_app[i_cell] = True
                        continue
                    y_centered = y - y.mean()
                    y_norm = np.sqrt(np.sum(y_centered ** 2))
                    if not np.isfinite(y_norm) or y_norm <= 0:
                        fail_not_app[i_cell] = True
                        continue
                    r_val = float(np.dot(x_centered, y_centered) / (x_norm * y_norm))
                    r_stage3_baseline[i_cell] = r_val
                    if not np.isfinite(r_val):
                        fail_not_app[i_cell] = True
                    elif np.abs(r_val) > config.temp_dep_r_threshold_baseline:
                        fail_stage3_baseline[i_cell] = True
                add_rejection_reason(fail_not_app, 'fail_temp_dep_stage3_baseline_not_applicable')
                add_rejection_reason(fail_stage3_baseline, 'fail_temp_dep_stage3_baseline')
                cell_boo_selected &= ~(fail_not_app | fail_stage3_baseline)
    else:
        var_ratio_stage1 = None
        sliding_ratio_stage2 = None

    # Keep cells that pass every applicable criterion before the PEV check.
    # The decoder's "stationary" mode can also use these cells when they fail PEV.
    cell_boo_before_pev = cell_boo_selected.copy()

    # criterion: significant PEV duration in the test period
    if config.t_test_step > 0:
        t_bin_start = np.arange(config.t_test_start, config.t_test_end + 1, config.t_test_step)
    else:
        t_bin_start = np.asarray([], dtype=np.int64)
    num_test_bins = len(t_bin_start)
    pev_mat = np.full((num_cells_total, num_test_bins), np.nan, dtype=np.float64)
    pref_mat = np.full((num_cells_total, num_test_bins), np.nan, dtype=np.float64)
    bin_boo_pev = np.zeros((num_cells_total, num_test_bins), dtype=np.bool_)
    mean_pev_full = np.full(num_cells_total, np.nan, dtype=np.float64)
    mean_pref_full = np.full(num_cells_total, np.nan, dtype=np.float64)

    if num_test_bins == 0:
        fail_not_app = np.ones(num_cells_total, dtype=np.bool_)
        add_rejection_reason(fail_not_app, 'fail_sig_pev_not_applicable')
        cell_boo_selected &= ~fail_not_app
    else:
        t_bin_masks = [(t >= t_min) & (t < (t_min + config.t_test_window)) for t_min in t_bin_start]
        valid_bins = np.asarray([np.any(mask) for mask in t_bin_masks], dtype=np.bool_)
        if not np.any(valid_bins):
            fail_not_app = np.ones(num_cells_total, dtype=np.bool_)
            add_rejection_reason(fail_not_app, 'fail_sig_pev_not_applicable')
            cell_boo_selected &= ~fail_not_app
        else:
            pev_rates = compute_binned_rates(trial_filtered_spikes, t, t_bin_start, config.t_test_window, dtype=np.float64)
            pev_mat, pref_mat = pev_and_preferred_cue(pev_rates, cue_labels_selected, labels_set)

            pev_mat = np.clip(pev_mat, config.pev_clip_at, 100)
            sig_pev_applicable = np.any(np.isfinite(pev_mat), axis=1)
            fail_not_app = ~sig_pev_applicable
            fail_sig_pev = np.zeros(num_cells_total, dtype=np.bool_)
            min_sig_duration_bins = config.sig_pev_duration / config.t_test_step
            for i_cell in range(num_cells_total):
                if not sig_pev_applicable[i_cell]:
                    continue
                _, sig_mask = get_periods_and_mask(
                    pev_mat[i_cell],
                    min_sig_duration_bins,
                    config.sig_pev_threshold,
                )
                bin_boo_pev[i_cell] = sig_mask
                if not np.any(sig_mask):
                    fail_sig_pev[i_cell] = True
                    continue
                mean_pev_full[i_cell] = np.mean(pev_mat[i_cell, sig_mask])
                pref_vals = pref_mat[i_cell, sig_mask]
                if np.all(np.isfinite(pref_vals)):
                    mean_pref_full[i_cell] = circular_mean_cue(pref_vals.astype(np.int64))

            add_rejection_reason(fail_not_app, 'fail_sig_pev_not_applicable')
            add_rejection_reason(fail_sig_pev, 'fail_sig_pev')
            cell_boo_selected &= ~(fail_not_app | fail_sig_pev)
    cell_boo_after_pev = cell_boo_selected.copy()

    # criterion: preferred-cue temporal stability (stage 3)
    if config.temp_dep_detection:
        slopes_stage3 = np.full(num_cells_total, np.nan, dtype=np.float64)
        intercepts_stage3 = np.full(num_cells_total, np.nan, dtype=np.float64)
        r_stage3 = np.full(num_cells_total, np.nan, dtype=np.float64)

        stage3_applicable = np.any(bin_boo_pev, axis=1) & np.isfinite(mean_pref_full)
        fail_stage3_not_app = ~stage3_applicable
        fail_stage3 = np.zeros(num_cells_total, dtype=np.bool_)
        if np.any(stage3_applicable):
            active_idx_stage3 = np.nonzero(stage3_applicable)[0]
            keep_stage3, slopes_tmp, intercepts_tmp, r_tmp = check_temporal_stability_preferred_trials(
                spikes,
                cue_labels,
                mean_pref_full[active_idx_stage3].astype(np.int64),
                active_idx_stage3,
                t,
                config,
            )
            slopes_stage3[active_idx_stage3] = slopes_tmp
            intercepts_stage3[active_idx_stage3] = intercepts_tmp
            r_stage3[active_idx_stage3] = r_tmp
            finite_stage3 = np.isfinite(r_tmp)
            fail_stage3_not_app[active_idx_stage3] |= ~finite_stage3
            fail_stage3_idx = active_idx_stage3[finite_stage3 & (~keep_stage3)]
            fail_stage3[fail_stage3_idx] = True

        add_rejection_reason(fail_stage3_not_app, 'fail_temp_dep_stage3_not_applicable')
        add_rejection_reason(fail_stage3, 'fail_temp_dep_stage3')
        cell_boo_selected &= ~(fail_stage3_not_app | fail_stage3)

    # keep only cells that pass all active criteria
    if not np.any(cell_boo_selected):
        return None, diagnostics()

    cell_idx_selected = np.nonzero(cell_boo_selected)[0]
    cell_boo_stationary = cell_boo_selected | (cell_boo_before_pev & ~cell_boo_after_pev)
    cell_idx_stationary = np.nonzero(cell_boo_stationary)[0]
    mean_pev_test = mean_pev_full[cell_idx_selected]
    mean_pref_test = mean_pref_full[cell_idx_selected].astype(np.int64)
    bin_boo_pev_selected = bin_boo_pev[cell_idx_selected]

    group_boo = np.asarray([mean_pref_test == l for l in labels_set])
    # count number of cells selective to each cue location
    num_cells_per_group = np.sum(group_boo, axis=1)
    # total PEV per group
    total_pev_per_group = np.asarray([mean_pev_test[group_boo[i]].sum() for i in range(len(labels_set))])


    trial_idx_selected = np.nonzero(trial_boo_selected)[0]
    cell_properties = {
        'cell_idx': cell_idx_selected,
        'mean_fr_test': mean_firing_rate_hz[cell_idx_selected],
        'mean_pev_test': mean_pev_test,
        'mean_pref_test': mean_pref_test,
        'num_sig_pev_bins': bin_boo_pev_selected.sum(axis=1),
    }
    if var_ratio_stage1 is not None:
        cell_properties.update({
            'temp_dep_var_ratio_stage1': var_ratio_stage1[cell_idx_selected],
            'temp_dep_sliding_ratio_stage2': sliding_ratio_stage2[cell_idx_selected],
        })
    if slopes_stage3 is not None:
        cell_properties.update({
            'temp_dep_slope': slopes_stage3[cell_idx_selected],
            'temp_dep_intercept': intercepts_stage3[cell_idx_selected],
            'temp_dep_r': r_stage3[cell_idx_selected],
        })

    out = {
        'session': session,
        'num_trials': num_trials,
        'num_trials_selected': num_trials_selected,
        'num_cells_selected': np.sum(cell_boo_selected),
        'cell_idx_selected': cell_idx_selected,
        'trial_idx_selected': trial_idx_selected,
        'labels_set_idx': labels_set,
        'labels_set_deg': cue_to_deg(labels_set),
        'num_cells_per_group': num_cells_per_group,
        'total_pev_per_group': total_pev_per_group,
        'max_num_cells_per_group': np.max(num_cells_per_group),
        'max_total_pev_per_group': np.max(total_pev_per_group),
        'cell_idx_stationary': cell_idx_stationary,
        'cell_idx_passed_presence_ratio': np.nonzero(passed_presence_ratio)[0],
        'cell_properties': cell_properties,
    }

    return out, diagnostics()


def main(config: Config):
    if config.t_test_window <= 0 or config.t_test_step <= 0 or config.t_test_end <= config.t_test_start:
        raise ValueError('Selectivity window/step must be positive and end must follow start.')
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
        raise ValueError('No sessions passed cell screening; inspect data and selection thresholds.')
    cache_io.save(results, config.cache_dir / 'cell_trial_selection.pkl')
    pd.DataFrame(results).to_csv(config.cache_dir / 'cell_trial_selection.csv', index=False)
    if config.save_extended_diagnostics:
        save_diagnostics(diagnostic_rows, files, config)
    return results


if __name__ == '__main__':
    main(tyro.cli(Config))
