import builtins
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tyro
from joblib import Parallel, delayed
from numba import njit
from scipy.io import loadmat
from scipy.stats import circmean, linregress


@njit(cache=True)
def get_pev(samples, tags, conditions) -> float | None:
    """Compute percent explained variance (omega squared) across categorical conditions."""
    samples = np.asarray(samples)
    tags = np.asarray(tags)
    conditions = np.asarray(list(conditions))
    sst = np.sum((samples - samples.mean()) ** 2)
    if sst == 0:
        return 0
    sse = np.zeros(len(conditions))
    for i, cond in enumerate(conditions):
        arr = samples[tags == cond]
        sse[i] = np.sum((arr - arr.mean()) ** 2)
    sse = np.sum(sse)
    ssb = sst - sse
    dfe = len(samples) - len(conditions)
    dfb = len(conditions) - 1
    mse = sse / dfe
    omega_squared = (ssb - dfb * mse) / (mse + sst)
    return omega_squared * 100 # percent

@njit(cache=True)
def get_preferred_cue(samples, cue_labels, cue_label_set):
    """Return the cue label with the highest mean firing rate."""
    samples = np.asarray(samples)
    cue_labels = np.asarray(cue_labels)
    cue_label_set = np.asarray(list(cue_label_set))
    firing_rates = np.zeros(len(cue_label_set))
    for i, cue in enumerate(cue_label_set):
        firing_rates[i] = samples[cue_labels == cue].mean()
    return cue_label_set[np.argmax(firing_rates)]

@njit(cache=True)
def get_periods(sig: np.ndarray, min_window_width: int | float, sig_threshold: int | float | np.ndarray, greater_than: bool = True):
    """Find contiguous periods where `sig` crosses `sig_threshold` for at least `min_window_width` samples."""
    if greater_than:
        above_thresh = np.nonzero(sig > sig_threshold)[0]
    else:
        above_thresh = np.nonzero(sig < sig_threshold)[0]
    above_thresh_extended = np.zeros(len(above_thresh) + 2).astype(np.int64)
    above_thresh_extended[0] = -2
    above_thresh_extended[1:-1] = above_thresh # above_thresh is in range [0, len(sig)-1]
    above_thresh_extended[-1] = len(sig) + 1
    left_idx = np.nonzero(np.diff(above_thresh_extended) > 1)[0] # find the index of the left edge (inclusive)
    right_idx = left_idx - 1 # find the index of the right edge (inclusive)
    left_idx = left_idx[:-1]
    right_idx = right_idx[1:]
    res = np.zeros((len(left_idx), 3)).astype(np.int64)
    for i in range(len(left_idx)):
        left = above_thresh[left_idx[i]]
        right = above_thresh[right_idx[i]]
        w = right - left + 1 # because both left and right are inclusive, add 1 to the width
        res[i] = left, right, w
    return res[res[:, -1] >= min_window_width]

@njit(cache=True)
def get_periods_and_mask(sig: np.ndarray, min_window_width: int | float, sig_threshold: int | float | np.ndarray, greater_than: bool = True):
    """Return threshold-crossing periods and a mask marking those samples."""
    if greater_than:
        above_thresh = np.nonzero(sig > sig_threshold)[0]
    else:
        above_thresh = np.nonzero(sig < sig_threshold)[0]
    above_thresh_extended = np.zeros(len(above_thresh) + 2).astype(np.int64)
    above_thresh_extended[0] = -2
    above_thresh_extended[1:-1] = above_thresh
    above_thresh_extended[-1] = len(sig) + 1
    left_idx = np.nonzero(np.diff(above_thresh_extended) > 1)[0]
    right_idx = left_idx - 1
    left_idx = left_idx[:-1]
    right_idx = right_idx[1:]
    res = np.zeros((len(left_idx), 3)).astype(np.int64)
    for i in range(len(left_idx)):
        left = above_thresh[left_idx[i]]
        right = above_thresh[right_idx[i]]
        w = right - left + 1
        res[i] = left, right, w
    res = res[res[:, -1] >= min_window_width]
    mask = np.zeros_like(sig) # prepare a mask with the same shape as sig
    for left, right, w in res:
        mask[left:left + w] = 1
    return res, mask.astype(np.bool_)

def check_temporal_stability_preferred_trials(
    spikes: np.ndarray,
    cue_labels: np.ndarray,
    preferred_cues: np.ndarray,
    active_cell_idx: np.ndarray,
    t: np.ndarray,
    trial_start: int,
    trial_end: int,
    config: 'Config',
    trial_holdout: int | None = None,
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

    # trials inside the sliding window
    trial_boo_window = np.zeros_like(cue_labels, dtype=np.bool_)
    trial_boo_window[trial_start:trial_end] = True

    # precompute preferred-cue trial indices relative to the window start
    cue_trials = {}
    for cue in np.unique(preferred_cues):
        idx = np.nonzero((cue_labels == cue) & trial_boo_window)[0]
        if trial_holdout is not None:
            idx = idx[idx != trial_holdout]
        cue_trials[cue] = idx - trial_start

    # test-period firing rate for every trial (including incorrect) and selected cell
    spikes_window = spikes[trial_start:trial_end][:, :, active_cell_idx]
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
    n_jobs_partition: int = 8 # Parallel CPU workers across partitions within a session
    seed: int = 42  # Random seed for any stochastic routines

    data_dir: Path = Path('data/nature') # Folder with {session}.mat files containing spks/isCorr/cueAngIdx/tc
    cache_dir: Path = Path('cache/run_001') # Output directory for pkl/csv summaries
    log_messages: bool = True # Capture print output per session and persist to a log file
    console_messages: bool = False # Whether to also print per-session messages to the console
    log_filename: str = 'cell_trial_selection.log' # Log file name written to cache_dir when log_messages=True

    loo_cell_selection: bool = False # Enable leave-one-out cell selection variants
    loo_cue_labels: Path = Path('configs/loo_cue_labels.json') # JSON mapping sessions to candidate cue labels

    t_plot_start: int = -200 # PSTH start bin (inclusive) used for plotting
    t_plot_end: int = 1400 # PSTH final bin start (exclusive of window width)
    t_plot_window: int = 50 # PSTH bin width in ms
    t_plot_step: int = 10 # PSTH bin stride in ms

    t_test_start: int = 500 # Analysis window start for selectivity tests (delay)
    t_test_end: int = 1400 # Analysis window end for selectivity tests
    t_test_window: int = 50 # Test bin width in ms
    t_test_step: int = 10 # Test bin stride in ms

    min_cell_per_group: int = 12 # Minimum cells per cue location to keep for downstream analyses
    min_fr_test: float = 1.0 # Hz threshold on mean firing rate during the test window
    temp_dep_detection: bool = True # Drop cells that show strong temporal dependence
    min_trial_for_temp_check: int = 50 # Require this many trials before running temporal checks
    var_ratio_threshold_delay_over_baseline: float = 1.2 # Delay-vs-baseline variance ratio cutoff (stage 1)
    var_ratio_threshold_sliding_over_all: float = 0.8 # Sliding-window-vs-global baseline variance ratio cutoff (stage 2)
    temp_dep_r_threshold: float = 0.5 # Minimum correlation coefficient to flag temporal dependence
    sig_pev_threshold: float = 2.5 # Percent explained variance threshold to call a bin selective
    sig_pev_duration: int = 100 # Minimum contiguous duration (ms) a cell must stay selective
    pev_clip_at: float = 0 # Lower bound when clipping PEV values

    temp_check_baseline_start: int = -500 # Baseline window start for temporal dependence checks
    temp_check_baseline_end: int = 0 # Baseline window end for temporal dependence checks
    temp_check_delay_start: int = 500 # Delay window start for temporal dependence checks
    temp_check_delay_end: int = 1000 # Delay window end for temporal dependence checks

    trial_selection_window_size: int = 320 # Size of sliding trial window when subsampling
    trial_selection_step_size: int = 5 # Step size between consecutive trial windows

def load_loo_cue_labels(path: Path) -> dict[str, set[int]]:
    """Return a mapping from session string -> set of candidate cue labels."""
    mapping: dict[str, set[int]] = {}
    if not path.exists():
        return mapping
    with open(path, 'r') as f:
        data = json.load(f)
    for item in data:
        session = item.get('session')
        cues = item.get('cue_labels', [])
        if session is None:
            continue
        mapping[str(session)] = set(int(c) for c in cues)
    return mapping

def process_session(
    data_file: Path,
    config: Config,
    loo_cue_map: dict[str, set[int]] | None = None,
    n_jobs_partition: int = 1,
):
    session = data_file.stem
    session_loo_cues: set[int] = set()
    if config.loo_cell_selection and loo_cue_map is not None:
        session_loo_cues = loo_cue_map.get(session, set())
    use_loo = config.loo_cell_selection and len(session_loo_cues) > 0
    log_lines = [] if config.log_messages else None
    builtin_print = builtins.print
    if log_lines is not None or config.console_messages:
        def session_print(*args, **kwargs):
            sep = kwargs.pop('sep', ' ')
            end = kwargs.pop('end', '\n')
            message = sep.join(str(a) for a in args) + end
            if log_lines is not None:
                log_lines.append(message)
            if config.console_messages:
                builtin_print(*args, sep=sep, end=end, **kwargs)
    else:
        def session_print(*args, **kwargs):
            return None
    print = session_print
    print(f'Processing session: {session}')
    outs = []

    # load data
    data = loadmat(data_file)

    # boo for correct trials
    trial_boo_correct = np.asarray(data['isCorr']).flatten().astype(np.bool_)
    num_trials = len(trial_boo_correct)
    if num_trials < config.trial_selection_window_size:
        print(f'  Skipping session {session} due to insufficient trials ({num_trials} < {config.trial_selection_window_size})')
        return session, outs, log_lines
    num_trials_correct = np.sum(trial_boo_correct)
    print(f'  Total trials: {num_trials}, Correct trials: {num_trials_correct}')

    # load cue labels
    cue_labels = np.asarray(data['cueAngIdx']).flatten().astype(np.int64)
    cue_labels_correct = cue_labels[trial_boo_correct]
    labels_set, labels_counts = np.unique(cue_labels_correct, return_counts=True)
    print(f'  Cue labels distribution w/ percent (correct trials):')
    for lbl, cnt in zip(labels_set, labels_counts):
        print(f'    Label {lbl}: {cnt} trials ({cnt / num_trials_correct * 100:.2f}%)')

    # load spike data
    spikes = np.asarray(data['spks']) # shape: (trial, time, cell)
    spikes_correct = spikes[trial_boo_correct]
    print(f'  Spike data shape (trials, time, cells)')
    print(f'  Spike data shape (correct trials): {spikes_correct.shape}')

    # load timestamps
    t = np.asarray(data['tc']).flatten()
    dt = t[1] - t[0]  # ms
    print(f'  Timestamps (ms): start {t[0]}, end {t[-1]}, step {dt}')

    tasks: list[tuple[int, int, int | None]] = []
    for trial_start in range(0, num_trials - config.trial_selection_window_size + 1, config.trial_selection_step_size):
        trial_end = trial_start + config.trial_selection_window_size
        # baseline partition
        tasks.append((trial_start, trial_end, None))
        if use_loo:
            trial_boo_window = np.zeros(num_trials, dtype=np.bool_)
            trial_boo_window[trial_start:trial_end] = True
            candidates = np.nonzero(
                trial_boo_window
                & trial_boo_correct
                & np.isin(cue_labels, np.asarray(list(session_loo_cues), dtype=np.int64))
            )[0]
            for trial_holdout in candidates:
                tasks.append((trial_start, trial_end, int(trial_holdout)))

    def run_partition(trial_start: int, trial_end: int, trial_holdout: int | None):
        label = 'baseline' if trial_holdout is None else f'LOO trial {trial_holdout}'
        partition_logs = [] if log_lines is not None else None

        def partition_print(*args, **kwargs):
            sep = kwargs.pop('sep', ' ')
            end = kwargs.pop('end', '\n')
            message = sep.join(str(a) for a in args) + end
            if partition_logs is not None:
                partition_logs.append(message)
            if config.console_messages:
                builtin_print(*args, sep=sep, end=end, **kwargs)

        partition_print(f'  Trial window: {trial_start} to {trial_end} (size: {config.trial_selection_window_size}) [{label}]')

        # boolean array for selected trials in the current window
        trial_boo_window = np.zeros(num_trials, dtype=np.bool_)
        trial_boo_window[trial_start:trial_end] = True

        # combine with correct trials and drop holdout trial if requested
        trial_boo_selected = trial_boo_correct & trial_boo_window
        if trial_holdout is not None and 0 <= trial_holdout < num_trials:
            trial_boo_selected[trial_holdout] = False
        num_trials_selected = np.sum(trial_boo_selected)
        if num_trials_selected == 0:
            partition_print(f'    Skipping {label}: no correct trials after holdout')
            return None, partition_logs, trial_start

        cue_labels_selected = cue_labels[trial_boo_selected]
        if cue_labels_selected.size == 0:
            partition_print(f'    Skipping {label}: no cue labels after selection')
            return None, partition_logs, trial_start
        labels_set_sel, labels_counts_sel = np.unique(cue_labels_selected, return_counts=True)
        partition_print(f'    {label} - correct trials: {num_trials_selected}')
        partition_print(f'    {label} - cue labels distribution w/ percent (selected trials):')
        for lbl, cnt in zip(labels_set_sel, labels_counts_sel):
            partition_print(f'      Label {lbl}: {cnt} trials ({cnt / num_trials_selected * 100:.2f}%)')
        
        trial_filtered_spikes = spikes[trial_boo_selected]
        partition_print(f'    {label} - spike data shape (selected trials): {trial_filtered_spikes.shape}')

        # select cells based on firing rate during test period (selected trials)
        t_test_mask = (t >= config.t_test_start) & (t < config.t_test_end)
        bin_width_s = dt / 1000.0 # convert ms to seconds
        spikes_test_period = trial_filtered_spikes[:, t_test_mask, :]
        total_spike_counts = np.sum(spikes_test_period, axis=(0, 1))
        total_time_s = num_trials_selected * np.sum(t_test_mask) * bin_width_s
        mean_firing_rate_hz = total_spike_counts / total_time_s
        cell_boo_selected = mean_firing_rate_hz >= config.min_fr_test

        var_ratio_stage1 = None
        sliding_ratio_stage2 = None
        # select cells based on temporal dependency check (stage 1 and 2)
        if config.temp_dep_detection and num_trials_selected >= config.min_trial_for_temp_check and np.any(cell_boo_selected):
            num_cells_total = mean_firing_rate_hz.shape[0]
            var_ratio_stage1 = np.full(num_cells_total, np.nan)
            sliding_ratio_stage2 = np.full(num_cells_total, np.nan)
            # stage 1: delay vs. baseline variance ratio estimated using all selected trials (higher is better)
            baseline_mask = (t >= config.temp_check_baseline_start) & (t < config.temp_check_baseline_end)
            delay_mask = (t >= config.temp_check_delay_start) & (t < config.temp_check_delay_end)
            active_idx = np.nonzero(cell_boo_selected)[0] # indices of active cells after fr check
            # compute trial-averaged firing rates for active cells in baseline and delay windows
            cell_filtered_spikes = trial_filtered_spikes[:, :, active_idx] # shape: (trial, time, cell)
            baseline_counts = np.sum(cell_filtered_spikes[:, baseline_mask, :], axis=1) # shape: (trial, cell)
            delay_counts = np.sum(cell_filtered_spikes[:, delay_mask, :], axis=1)
            baseline_rates = baseline_counts / (np.sum(baseline_mask) * bin_width_s) # shape: (trial, cell)
            delay_rates = delay_counts / (np.sum(delay_mask) * bin_width_s)
            baseline_var = np.var(baseline_rates, axis=0, ddof=1) # shape: (cell,)
            delay_var = np.var(delay_rates, axis=0, ddof=1)
            with np.errstate(divide='ignore', invalid='ignore'):
                var_ratio = delay_var / baseline_var
            var_ratio_stage1[active_idx] = var_ratio
            keep_stage1 = (var_ratio > config.var_ratio_threshold_delay_over_baseline) & ~np.isnan(var_ratio)
            cell_boo_selected[active_idx] = keep_stage1 # update selected cells after stage 1
            if np.any(keep_stage1):
                # stage 2: baseline variance ratio estimated using sliding windows vs. all selected trials (higher is better)
                active_idx_stage2 = active_idx[keep_stage1] # indices of active cells after stage 1
                baseline_rates = baseline_rates[:, keep_stage1]
                baseline_var = baseline_var[keep_stage1]
                windows = np.lib.stride_tricks.sliding_window_view(
                    baseline_rates,
                    window_shape=config.min_trial_for_temp_check,
                    axis=0,
                ) # shape: (num_windows, cell, window_size)
                window_var = np.var(windows, axis=-1, ddof=1) # shape: (num_windows, cell)
                sliding_var = np.mean(window_var, axis=0) # shape: (cell,)
                with np.errstate(divide='ignore', invalid='ignore'):
                    sliding_ratio = sliding_var / baseline_var
                sliding_ratio_stage2[active_idx_stage2] = sliding_ratio
                keep_stage2 = (sliding_ratio > config.var_ratio_threshold_sliding_over_all) & ~np.isnan(sliding_ratio)
                cell_boo_selected[active_idx_stage2] = keep_stage2 # update selected cells after stage 2
            partition_print(f'    {label} - cells remaining after temporal dependency check: {np.sum(cell_boo_selected)}')
        
        # select cells based on PEV during test period (selected cells and trials)
        if np.any(cell_boo_selected):
            active_idx = np.nonzero(cell_boo_selected)[0] # indices of active cells after fr and temp dep check (stage 1 and 2)
            num_cells_selected = len(active_idx)
            cell_filtered_spikes = trial_filtered_spikes[:, :, active_idx] # (trial, time, cell)
            
            fr_list = [] # time-resolved firing rate of each cell
            pev_list = [] # time-resolved pev of each cell
            pref_list = [] # time-resolved preferred cue of each cell
            t_bin_start = np.arange(config.t_test_start, config.t_test_end + 1, config.t_test_step)
            for i_cell in range(num_cells_selected):
                for t_min in t_bin_start:
                    t_max = t_min + config.t_test_window
                    t_boo = (t >= t_min) & (t < t_max)
                    spikes_bin = cell_filtered_spikes[:, t_boo, i_cell].mean(axis=1)
                    fr_list.append(spikes_bin * 1000) # spikes/ms -> spikes/second
                    pev_list.append(get_pev(spikes_bin, cue_labels_selected, labels_set))
                    pref_list.append(get_preferred_cue(spikes_bin, cue_labels_selected, labels_set))
            fr_mat = np.asarray(fr_list).reshape((num_cells_selected, -1, num_trials_selected)) # (cell, bin, trial)
            pev_mat = np.array(pev_list).reshape((num_cells_selected, -1)).clip(config.pev_clip_at, 100) # (cell, bin)
            pref_mat = np.array(pref_list).reshape((num_cells_selected, -1)) # (cell, bin)

            # select cells based on significant PEV duration
            keep_pev = np.asarray([
                get_periods(pev_mat[i_cell], config.sig_pev_duration / config.t_test_step, config.sig_pev_threshold).shape[0] > 0 
                for i_cell in range(num_cells_selected)]) # pev consistently higher than sig_pev_threshold for at least sig_pev_duration
            fr_mat = fr_mat[keep_pev]
            pev_mat = pev_mat[keep_pev]
            pref_mat = pref_mat[keep_pev]
            cell_boo_selected[active_idx] = keep_pev # update selected cells after significant pev check
            partition_print(f'    {label} - cells remaining after significant PEV check: {np.sum(cell_boo_selected)}')

        # group cells by preferred cue location
        if np.any(cell_boo_selected):
            # only periods with sig pev are used to estimate mean pev and selectivity
            bin_boo_pev = np.asarray([
                get_periods_and_mask(p, config.sig_pev_duration / config.t_test_step, config.sig_pev_threshold)[1]
                for p in pev_mat])
            mean_pev_test = np.asarray([p[b].mean() for p, b in zip(pev_mat, bin_boo_pev)])
            mean_pref_test = np.asarray([circular_mean_cue(p[b]) for p, b in zip(pref_mat, bin_boo_pev)])

            slopes_stage3 = None
            intercepts_stage3 = None
            r_stage3 = None
            if config.temp_dep_detection:
                active_idx = np.nonzero(cell_boo_selected)[0]
                keep_stage3, slopes_stage3, intercepts_stage3, r_stage3 = check_temporal_stability_preferred_trials(
                    spikes,
                    cue_labels,
                    mean_pref_test,
                    active_idx,
                    t,
                    trial_start,
                    trial_end,
                    config,
                    trial_holdout=trial_holdout,
                )
                if not np.all(keep_stage3):
                    fr_mat = fr_mat[keep_stage3]
                    pev_mat = pev_mat[keep_stage3]
                    pref_mat = pref_mat[keep_stage3]
                    bin_boo_pev = bin_boo_pev[keep_stage3]
                    mean_pev_test = mean_pev_test[keep_stage3]
                    mean_pref_test = mean_pref_test[keep_stage3]
                    slopes_stage3 = slopes_stage3[keep_stage3]
                    intercepts_stage3 = intercepts_stage3[keep_stage3]
                    r_stage3 = r_stage3[keep_stage3]
                    cell_boo_selected[active_idx] = keep_stage3
                partition_print(f'    {label} - cells remaining after temporal stability check (stage 3): {np.sum(cell_boo_selected)}')

            group_boo = np.asarray([mean_pref_test == l for l in labels_set])
            partition_print(f'    {label} - group_boo.shape (label, cell): {group_boo.shape}')
            # count number of cells selective to each cue location
            num_cells_per_group = np.sum(group_boo, axis=1)
            # total PEV per group
            total_pev_per_group = np.asarray([mean_pev_test[group_boo[i]].sum() for i in range(len(labels_set))])

            partition_print(f'    {label} - number of cells per group (preferred cue location):')
            for i, l in enumerate(labels_set):
                partition_print(f'      Label {l}: {num_cells_per_group[i]} cells, Total PEV: {total_pev_per_group[i]:.2f}')

            if np.any(num_cells_per_group >= config.min_cell_per_group):
                partition_print(f'    {label} - found a good session window from {session} with at least {config.min_cell_per_group} cells in one group.')
                partition_print(f'    {label} - session window is {trial_start} to {trial_end} (size: {config.trial_selection_window_size})')

            cell_idx_selected = np.nonzero(cell_boo_selected)[0]
            trial_idx_selected = np.nonzero(trial_boo_selected)[0]
            cell_properties = {
                'cell_idx': cell_idx_selected,
                'mean_fr_test': mean_firing_rate_hz[cell_idx_selected],
                'mean_pev_test': mean_pev_test,
                'mean_pref_test': mean_pref_test,
                'num_sig_pev_bins': bin_boo_pev.sum(axis=1),
            }
            if var_ratio_stage1 is not None:
                cell_properties.update({
                    'temp_dep_var_ratio_stage1': var_ratio_stage1[cell_idx_selected],
                    'temp_dep_sliding_ratio_stage2': sliding_ratio_stage2[cell_idx_selected],
                })
            if slopes_stage3 is not None:
                cell_properties.update({
                    'temp_dep_slope': slopes_stage3,
                    'temp_dep_intercept': intercepts_stage3,
                    'temp_dep_r': r_stage3,
                })
            out = {
                'session': session,
                'trial_start': trial_start,
                'trial_end': trial_end,
                'trial_holdout': trial_holdout,
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
                'cell_properties': cell_properties,
            }
            return out, partition_logs, trial_start
        return None, partition_logs, trial_start

    if n_jobs_partition > 1:
        partition_results = Parallel(n_jobs=n_jobs_partition, verbose=5)(
            delayed(run_partition)(ts, te, th) for ts, te, th in tasks
        )
    else:
        partition_results = [run_partition(ts, te, th) for ts, te, th in tasks]

    partition_log_rows = []
    for res_out, res_logs, res_trial_start in partition_results:
        if res_out is not None:
            outs.append(res_out)
        if res_logs is not None:
            partition_log_rows.append((res_trial_start, res_logs))

    if log_lines is not None and partition_log_rows:
        partition_log_rows.sort(key=lambda x: x[0])
        for _, logs in partition_log_rows:
            log_lines.extend(logs)

    return session, outs, log_lines

def main(config: Config):
    data_files = sorted(config.data_dir.glob('*.mat'))
    cache_dir = config.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    # enforce single-level parallelism; prefer partition-level if both are requested
    jobs_session = config.n_jobs_session
    jobs_partition = config.n_jobs_partition
    if jobs_session > 1 and jobs_partition > 1:
        jobs_session = 1
    elif jobs_session > 1:
        jobs_partition = 1

    loo_cue_map: dict[str, set[int]] | None = None
    if config.loo_cell_selection:
        loo_cue_map = load_loo_cue_labels(config.loo_cue_labels)
        print(f'Loaded LOO cue labels for {len(loo_cue_map)} sessions from {config.loo_cue_labels}')

    session_results = Parallel(n_jobs=jobs_session, verbose=10)(
        delayed(process_session)(data_file, config, loo_cue_map, jobs_partition)
        for data_file in data_files
    )
    outs = []
    session_logs: list[tuple[str, str]] = []
    for session, session_outs, log_lines in session_results:
        outs.extend(session_outs)
        if config.log_messages and log_lines is not None:
            session_logs.append((session, ''.join(log_lines)))

    if config.log_messages:
        log_file = cache_dir / config.log_filename
        with open(log_file, 'w') as f:
            for session, log_text in session_logs:
                f.write(f'[{session}]\n')
                f.write(log_text)
                if not log_text.endswith('\n'):
                    f.write('\n')
                f.write('\n')
        print(f'Saved processing logs to {log_file}')

    # save outs to cache
    cache_file = cache_dir / 'cell_trial_selection.pkl'
    with open(cache_file, 'wb') as f:
        pickle.dump(outs, f)
    print(f'Saved cell trial selection results to {cache_file}')
    # save outs as csv
    df_out = pd.DataFrame(outs)
    csv_file = cache_dir / 'cell_trial_selection.csv'
    df_out.to_csv(csv_file, index=False)
    print(f'Saved cell trial selection results to {csv_file}')


if __name__ == '__main__':
    config = tyro.cli(Config)
    main(config)
