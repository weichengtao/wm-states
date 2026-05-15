import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import tyro
from joblib import Parallel, delayed
from scipy.io import loadmat

matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from scripts.figure_exports import configure_figure_style, save_figure_all_formats
except ModuleNotFoundError:
    from figure_exports import configure_figure_style, save_figure_all_formats

configure_figure_style(matplotlib)


def cue_to_deg(cue):
    """Convert cue indices (1-8) to degrees on a -135..180 scale."""
    cue = np.asarray(cue)
    cue = (cue - 1) % 8 + 1
    return (cue - 1) * 45 - 135


def get_opposite_cue(cue):
    """Return the cue index opposite to the given cue (1-8)."""
    return (cue + 3) % 8 + 1


def preferred_cue_from_cells(pref_cues):
    """Pick the most frequent preferred cue across cells."""
    cues, counts = np.unique(pref_cues, return_counts=True)
    if cues.size == 0:
        return None
    best = np.argmax(counts)
    return int(cues[best])


def preferred_cue_from_partitions(partition_cues):
    """Pick the most frequent preferred cue across partitions."""
    cues, counts = np.unique(partition_cues, return_counts=True)
    if cues.size == 0:
        return None
    best = np.argmax(counts)
    return int(cues[best])


def total_unique_trials(partitions):
    """Count unique trials covered by a list of partitions."""
    if not partitions:
        return 0
    max_end = max(int(p['trial_end']) for p in partitions)
    covered = np.zeros(max_end, dtype=np.bool_)
    for p in partitions:
        covered[int(p['trial_start']):int(p['trial_end'])] = True
    return int(covered.sum())


def covered_trial_mask(partitions, num_trials):
    """Return a boolean mask for trials covered by the no-holdout partitions."""
    covered = np.zeros(num_trials, dtype=np.bool_)
    for p in partitions:
        start = max(0, int(p['trial_start']))
        end = min(num_trials, int(p['trial_end']))
        if start < end:
            covered[start:end] = True
    return covered


def repeated_spike_times(spike_counts, t_values):
    """Expand per-bin spike counts into spike times, preserving simultaneous spikes."""
    spike_counts = np.asarray(spike_counts, dtype=np.int64)
    if spike_counts.size == 0:
        return np.empty(0, dtype=np.float64)
    return np.repeat(np.asarray(t_values, dtype=np.float64), spike_counts)


def preferred_cell_intersection(no_holdout_parts, pref_cue):
    """Intersect preferred-cue cells across no-holdout partitions and average PEV."""
    cell_idx_set: set[int] | None = None
    pev_by_cell: dict[int, list[float]] = {}
    for p in no_holdout_parts:
        cell_props = p['cell_properties']
        pref_cues = np.asarray(cell_props['mean_pref_test'])
        cells = np.asarray(cell_props['cell_idx'], dtype=np.int64)
        pevs = np.asarray(cell_props['mean_pev_test'], dtype=np.float64)
        part_mask = pref_cues == pref_cue
        part_cells = set(cells[part_mask].tolist())
        if cell_idx_set is None:
            cell_idx_set = part_cells
        else:
            cell_idx_set &= part_cells
        for cell, pev in zip(cells[part_mask], pevs[part_mask], strict=False):
            pev_by_cell.setdefault(int(cell), []).append(float(pev))

    if not cell_idx_set:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    cell_idx = np.asarray(sorted(cell_idx_set), dtype=np.int64)
    mean_pev = np.asarray([np.mean(pev_by_cell[int(cell)]) for cell in cell_idx], dtype=np.float64)
    return cell_idx, mean_pev


def population_spike_times(spikes, trial_idx, cell_idx, time_mask, t):
    """Merge spike trains across cells for one trial."""
    counts_by_time = np.asarray(spikes[trial_idx][:, cell_idx][time_mask].sum(axis=1), dtype=np.int64)
    return repeated_spike_times(counts_by_time, t[time_mask])


def cell_spike_times(spikes, trial_idx, cell_idx, time_mask, t):
    """Return spike times for one cell on one trial."""
    counts = np.asarray(spikes[trial_idx][:, cell_idx][time_mask], dtype=np.int64)
    return repeated_spike_times(counts, t[time_mask])


def compute_pisi_by_trial(spikes, t, trial_idx, cell_idx, t_start, t_end):
    """Compute pISI intervals for each trial after injecting delay-edge spikes."""
    delay_mask = (t >= t_start) & (t < t_end)
    trial_isis = []
    for trial in trial_idx:
        pop_times = population_spike_times(spikes, int(trial), cell_idx, delay_mask, t)
        pop_times_with_edges = np.sort(np.concatenate(([float(t_start)], pop_times, [float(t_end)])))
        trial_isis.append(np.diff(pop_times_with_edges))
    return trial_isis


def trial_with_max_pisi(trial_idx, trial_isis):
    """Find the earliest trial containing the maximum population ISI."""
    max_values = np.asarray(
        [np.max(isis) if isis.size else np.nan for isis in trial_isis],
        dtype=np.float64,
    )
    if max_values.size == 0 or not np.any(np.isfinite(max_values)):
        return None, np.nan
    max_pisi = float(np.nanmax(max_values))
    candidate_idx = np.nonzero(np.isclose(max_values, max_pisi, rtol=0.0, atol=1e-9))[0]
    if candidate_idx.size == 0:
        return None, max_pisi
    candidate_trials = np.asarray(trial_idx, dtype=np.int64)[candidate_idx]
    return int(candidate_trials.min()), max_pisi


def max_pisi_from_population_activity(population_activity, t_values, t_start, t_end):
    """Return per-trial max pISI from population activity over time."""
    max_values = np.empty(population_activity.shape[0], dtype=np.float64)
    for i_trial, activity in enumerate(population_activity):
        spike_times = t_values[activity]
        spike_times_with_edges = np.concatenate(([float(t_start)], spike_times, [float(t_end)]))
        max_values[i_trial] = np.max(np.diff(spike_times_with_edges))
    return max_values


def shuffled_null_max_pisi(spikes, t, trial_idx, cell_idx, t_start, t_end, n_shuffle, seed):
    """Generate null max-pISI values by independently shuffling trials per cell."""
    if n_shuffle <= 0:
        return np.empty(0, dtype=np.float64)

    trial_idx = np.asarray(trial_idx, dtype=np.int64)
    cell_idx = np.asarray(cell_idx, dtype=np.int64)
    delay_mask = (t >= t_start) & (t < t_end)
    t_delay = t[delay_mask]
    delay_spikes = spikes[trial_idx][:, delay_mask, :][:, :, cell_idx] > 0
    num_trials = trial_idx.size
    num_cells = cell_idx.size
    cell_axis = np.arange(num_cells)
    rng = np.random.default_rng(seed)
    null_max = np.empty(n_shuffle, dtype=np.float64)

    for i_shuffle in range(n_shuffle):
        shuffled_trial_idx = np.empty((num_trials, num_cells), dtype=np.int64)
        for i_cell in range(num_cells):
            shuffled_trial_idx[:, i_cell] = rng.permutation(num_trials)
        shuffled_cell_spikes = delay_spikes[shuffled_trial_idx, :, cell_axis]
        population_activity = np.any(shuffled_cell_spikes, axis=1)
        null_max[i_shuffle] = np.max(
            max_pisi_from_population_activity(population_activity, t_delay, t_start, t_end)
        )
    return null_max


def stable_session_seed(seed, session):
    """Make a deterministic per-session seed independent of Python hash state."""
    session_text = str(session)
    session_offset = sum((i + 1) * ord(ch) for i, ch in enumerate(session_text))
    return int(seed + session_offset)


def plot_population_isi(
    fig_dir,
    session,
    pref_cue,
    cell_idx,
    mean_pev,
    top_cell_idx,
    trial_idx,
    max_pisi,
    spikes,
    t,
    t_plot_start,
    t_plot_end,
    t_test_start,
    t_test_end,
    null_max_pisi,
    null_pisi_99,
):
    """Save the per-session pISI raster figure."""
    plot_mask = (t >= t_plot_start) & (t <= t_plot_end)
    num_cell_rows = top_cell_idx.size
    num_rows = num_cell_rows + 1
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.5, max(3.5, 0.28 * num_rows + 1.4)),
        width_ratios=(2.2, 1.0),
        layout='constrained',
    )
    ax_left, ax_right = axes

    row_y = np.arange(num_rows)
    ax_left.axvspan(0, 50, color='tab:red', alpha=0.1, lw=0, zorder=0)
    ax_left.axvspan(t_test_start, t_test_end, color='tab:green', alpha=0.1, lw=0, zorder=0)

    delay_mask = (t >= t_test_start) & (t < t_test_end)
    delay_pop_times = population_spike_times(spikes, trial_idx, cell_idx, delay_mask, t)
    pisi_edges = np.sort(
        np.concatenate(([float(t_test_start)], delay_pop_times, [float(t_test_end)]))
    )
    pisi_intervals = np.diff(pisi_edges)
    if pisi_intervals.size:
        max_interval_idx = int(np.argmax(pisi_intervals))
        ax_left.axvspan(
            pisi_edges[max_interval_idx],
            pisi_edges[max_interval_idx + 1],
            color='gray',
            alpha=0.2,
            lw=0,
            zorder=0.5,
        )

    for y in row_y:
        ax_left.hlines(y, t_plot_start, t_plot_end, color='black', linewidth=0.6, zorder=2)

    for i_row, cell in enumerate(top_cell_idx):
        spike_times = cell_spike_times(spikes, trial_idx, int(cell), plot_mask, t)
        if spike_times.size:
            ax_left.vlines(
                spike_times,
                i_row - 0.35,
                i_row + 0.35,
                color='black',
                linewidth=0.7,
                zorder=3,
            )

    pop_times = population_spike_times(spikes, trial_idx, cell_idx, plot_mask, t)
    pop_row = num_rows - 1
    if pop_times.size:
        ax_left.vlines(
            pop_times,
            pop_row - 0.35,
            pop_row + 0.35,
            color='black',
            linewidth=0.7,
            zorder=3,
        )

    ax_left.set_xlim(t_plot_start, t_plot_end)
    ax_left.set_ylim(num_rows - 0.5, -0.5)
    ax_left.set_xlabel('Time (ms)')
    ax_left.set_ylabel(
        f'{session} ({int(cue_to_deg(pref_cue))}$\\degree$), '
        f'{len(cell_idx)} cells, Max pISI {max_pisi:.0f} ms'
    )

    rank_by_cell = {int(cell): i for i, cell in enumerate(cell_idx)}
    labels = []
    for cell in top_cell_idx:
        i_cell = rank_by_cell[int(cell)]
        labels.append(f'Cell {int(cell)}')
    labels.append('Population')
    ax_left.set_yticks(row_y)
    ax_left.set_yticklabels(labels)
    ax_left.set_title('Spike Train')

    if null_max_pisi.size:
        bins = np.arange(0, 201, 25)
        weights = np.full(null_max_pisi.size, 100.0 / null_max_pisi.size)
        ax_right.hist(
            null_max_pisi,
            bins=bins,
            weights=weights,
            color='tab:blue',
            edgecolor='white',
            linewidth=0.8,
        )
        ax_right.axvline(
            null_pisi_99,
            color='tab:blue',
            linestyle='--',
            linewidth=1.1,
            label='Null 99th',
        )
        ax_right.axvline(
            max_pisi,
            color='black',
            linestyle='--',
            linewidth=1.1,
            label='Actual',
        )
        ax_right.set_xlim(0, 200)
        ax_right.set_xticks(bins)
        ax_right.set_xlabel('Max pISI (ms)')
        ax_right.set_ylabel('%')
        ax_right.legend(frameon=False, loc='upper right')
    else:
        ax_right.axis('off')
    save_figure_all_formats(fig, fig_dir / f'population_isi_{session}_{pref_cue}.png', dpi=300)
    plt.close(fig)


@dataclass
class Config:
    """CLI configuration for pISI analysis."""
    n_jobs: int = 1 # number of parallel jobs for pISI analysis
    seed: int = 42 # reserved for future trial shuffling
    data_dir: Path = Path('data/nature') # directory with {session}.mat files
    cache_dir: Path = Path('cache/run_001') # directory for cached results and figures
    min_cell_per_group: int = 12 # a good partition has at least one group with this many cells
    min_trials_good_session: int = 320 # a good session has at least this many trials in good partitions
    t_plot_start: int = -200 # start time for pISI plotting (ms relative to trial onset)
    t_plot_end: int = 1400 # end time for pISI plotting (ms relative to trial onset)
    t_test_start: int = 500 # start time for delay (ms relative to trial onset)
    t_test_end: int = 1400 # end time for delay (ms relative to trial onset)
    n_trial_shuffle: int = 2000 # number of per-cell trial shuffles for null max pISI
    max_sessions_to_run: int | None = None # max number of good sessions to process (None to run all)


def main(config: Config):
    cache_dir = config.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    selection_pkl = cache_dir / 'cell_trial_selection.pkl'
    fig_dir = cache_dir / 'population_isi'
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not selection_pkl.exists():
        raise FileNotFoundError(f'Missing selection file: {selection_pkl}')
    with open(selection_pkl, 'rb') as f:
        selection_outs = pickle.load(f)

    # Force no-holdout behavior: ignore all LOO selection entries.
    good_partitions = []
    for out in selection_outs:
        if out.get('trial_holdout') is not None:
            continue
        if out.get('max_num_cells_per_group', 0) < config.min_cell_per_group:
            continue
        good_partitions.append(out)

    partitions_by_session = {}
    for out in good_partitions:
        partitions_by_session.setdefault(out['session'], []).append(out)

    good_sessions = {}
    for session, no_holdout_parts in partitions_by_session.items():
        covered_trial_count = total_unique_trials(no_holdout_parts)
        if covered_trial_count < config.min_trials_good_session:
            continue
        partition_pref = []
        for p in no_holdout_parts:
            pref_cues = np.asarray(p['cell_properties']['mean_pref_test'])
            pref_cue = preferred_cue_from_cells(pref_cues)
            if pref_cue is not None:
                partition_pref.append(pref_cue)
        session_pref = preferred_cue_from_partitions(np.asarray(partition_pref))
        if session_pref is not None:
            good_sessions[session] = {
                'preferred_cue': session_pref,
                'no_holdout_partitions': no_holdout_parts,
            }

    if not good_sessions:
        print('No good sessions found for pISI analysis')
        return

    sessions = list(good_sessions.keys())
    if config.max_sessions_to_run is not None:
        sessions = sessions[:config.max_sessions_to_run]

    def process_session(session: str):
        session_info = good_sessions[session]
        pref_cue = session_info['preferred_cue']
        no_holdout_parts = session_info['no_holdout_partitions']

        print(f'Processing session {session}')
        data_file = config.data_dir / f'{session}.mat'
        if not data_file.exists():
            print(f'  Skipping: missing data file {data_file}')
            return None

        data = loadmat(data_file)
        spikes = np.asarray(data['spks'])
        cue_labels = np.asarray(data['cueAngIdx']).flatten().astype(np.int64)
        trial_boo_correct = np.asarray(data['isCorr']).flatten().astype(np.bool_)
        t = np.asarray(data['tc']).flatten()

        cell_idx, mean_pev = preferred_cell_intersection(no_holdout_parts, pref_cue)
        if cell_idx.size < config.min_cell_per_group:
            print(
                f'  Skipping: preferred-cue cell intersection has {cell_idx.size} cells '
                f'(< {config.min_cell_per_group})'
            )
            return None

        covered_trials = covered_trial_mask(no_holdout_parts, cue_labels.size)
        trial_mask = covered_trials & trial_boo_correct & (cue_labels == pref_cue)
        trial_idx_pref = np.nonzero(trial_mask)[0]
        if trial_idx_pref.size == 0:
            print('  Skipping: no covered correct preferred-cue trials')
            return None

        trial_isis = compute_pisi_by_trial(
            spikes,
            t,
            trial_idx_pref,
            cell_idx,
            config.t_test_start,
            config.t_test_end,
        )
        max_trial, max_pisi = trial_with_max_pisi(trial_idx_pref, trial_isis)
        if max_trial is None:
            print('  Skipping: unable to compute pISI')
            return None

        sort_idx = np.lexsort((cell_idx, -mean_pev))
        sorted_cell_idx = cell_idx[sort_idx]
        sorted_mean_pev = mean_pev[sort_idx]
        top_cell_idx = sorted_cell_idx[:config.min_cell_per_group]
        null_max_pisi = shuffled_null_max_pisi(
            spikes=spikes,
            t=t,
            trial_idx=trial_idx_pref,
            cell_idx=sorted_cell_idx,
            t_start=config.t_test_start,
            t_end=config.t_test_end,
            n_shuffle=config.n_trial_shuffle,
            seed=stable_session_seed(config.seed, session),
        )
        null_pisi_99 = (
            float(np.percentile(null_max_pisi, 99)) if null_max_pisi.size else np.nan
        )

        plot_population_isi(
            fig_dir=fig_dir,
            session=session,
            pref_cue=pref_cue,
            cell_idx=sorted_cell_idx,
            mean_pev=sorted_mean_pev,
            top_cell_idx=top_cell_idx,
            trial_idx=max_trial,
            max_pisi=max_pisi,
            spikes=spikes,
            t=t,
            t_plot_start=config.t_plot_start,
            t_plot_end=config.t_plot_end,
            t_test_start=config.t_test_start,
            t_test_end=config.t_test_end,
            null_max_pisi=null_max_pisi,
            null_pisi_99=null_pisi_99,
        )
        print(
            f'  Saved pISI figure: cue {pref_cue}, {cell_idx.size} cells, '
            f'{trial_idx_pref.size} trials, max pISI {max_pisi:.1f} ms on trial {max_trial}, '
            f'null 99th {null_pisi_99:.1f} ms'
        )
        return {
            'session': session,
            'cue': int(pref_cue),
            'cue_deg': int(cue_to_deg(pref_cue)),
            'trial_idx': trial_idx_pref,
            'cell_idx': sorted_cell_idx,
            'mean_pev': sorted_mean_pev,
            'max_pisi_trial': int(max_trial),
            'max_pisi': float(max_pisi),
            'null_max_pisi': null_max_pisi,
            'null_pisi_99': null_pisi_99,
            'n_trial_shuffle': int(config.n_trial_shuffle),
        }

    if config.n_jobs == 1:
        results = [process_session(session) for session in sessions]
    else:
        results = Parallel(n_jobs=config.n_jobs, verbose=5)(
            delayed(process_session)(session) for session in sessions
        )
    results = [res for res in results if res is not None]
    with open(cache_dir / 'population_isi.pkl', 'wb') as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    config = tyro.cli(Config)
    main(config)
