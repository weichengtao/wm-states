import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tyro
from joblib import Parallel, delayed
from scipy.io import loadmat
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

matplotlib.use('Agg')

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

def compute_binned_rates(spikes, t, bin_starts, window_ms):
    """Compute firing rates per trial, time bin, and cell."""
    dt = t[1] - t[0]
    num_trials, _, num_cells = spikes.shape
    num_bins = len(bin_starts)
    rates = np.empty((num_trials, num_bins, num_cells), dtype=np.float32)
    for i, start in enumerate(bin_starts):
        # Build a mask for this time window and convert spike counts to Hz.
        mask = (t >= start) & (t < start + window_ms)
        if not np.any(mask):
            rates[:, i, :] = 0.0
            continue
        duration_s = mask.sum() * dt / 1000.0
        counts = spikes[:, mask, :].sum(axis=1)
        rates[:, i, :] = counts / duration_s
    return rates

def decode_one_trial(
    test_idx,
    binned_rates,
    labels,
    seed,
    n_shuffle,
):
    """Decode a single test trial across all bins with optional shuffles."""
    rng = np.random.default_rng(seed + int(test_idx))
    num_trials = binned_rates.shape[0]
    num_bins = binned_rates.shape[1]
    # Leave-one-out split for the current test trial.
    train_mask = np.ones(num_trials, dtype=np.bool_)
    train_mask[test_idx] = False
    train_idx = np.nonzero(train_mask)[0]
    y_train_full = labels[train_idx]
    pref_idx = train_idx[y_train_full == 1]
    opp_idx = train_idx[y_train_full == 0]
    if pref_idx.size == 0 or opp_idx.size == 0:
        conf = np.full(num_bins, np.nan, dtype=np.float32)
        null_conf = None if n_shuffle <= 0 else np.full((num_bins, n_shuffle), np.nan, dtype=np.float32)
        return conf, null_conf

    # Balance classes so the decoder isn't biased by class counts.
    n_train = min(pref_idx.size, opp_idx.size)
    pref_sel = rng.choice(pref_idx, size=n_train, replace=False)
    opp_sel = rng.choice(opp_idx, size=n_train, replace=False)
    train_balanced = np.concatenate([pref_sel, opp_sel])
    y_bal = labels[train_balanced]

    conf = np.empty(num_bins, dtype=np.float32)
    null_conf = None
    if n_shuffle > 0:
        null_conf = np.empty((num_bins, n_shuffle), dtype=np.float32)

    for b in range(num_bins):
        # Train a per-bin decoder to isolate the time-resolved confidence.
        X_train = binned_rates[train_balanced, b, :]
        X_test = binned_rates[test_idx, b, :][None, :]
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", probability=True, random_state=seed)),
        ])
        model.fit(X_train, y_bal)
        proba = model.predict_proba(X_test)[0]
        class_index = 1 if model.classes_[1] == 1 else 0
        conf[b] = proba[class_index]

        if n_shuffle > 0:
            # Shuffle labels to estimate a null confidence distribution.
            for s in range(n_shuffle):
                y_shuf = rng.permutation(y_bal)
                model.fit(X_train, y_shuf)
                proba = model.predict_proba(X_test)[0]
                class_index = 1 if model.classes_[1] == 1 else 0
                null_conf[b, s] = proba[class_index]

    return conf, null_conf

def plot_decoding_heatmap(
    fig_dir,
    session,
    pref_cue,
    cue_angle,
    trial_idx_pref,
    bin_starts,
    decode_confidence,
    plot_actual_trial_id,
    num_cells,
):
    """Save a heatmap of decoding confidence over time and trials."""
    fig, ax = plt.subplots(1, 1, figsize=(5, 4), layout='constrained')
    sns.heatmap(
        decode_confidence,
        ax=ax,
        vmin=0.5,
        vmax=1.0,
        cmap=None,
        cbar_kws={'label': 'Decoding confidence'},
    )
    xticks = [i for i, t_val in enumerate(bin_starts) if t_val % 200 == 0]
    ax.set_xticks([x + 0.5 for x in xticks])
    ax.set_xticklabels([str(bin_starts[x]) for x in xticks], rotation=0)
    ytick_positions = np.arange(9, trial_idx_pref.size, 10)
    ax.set_yticks(ytick_positions + 0.5)
    if plot_actual_trial_id:
        ytick_labels = [str(trial_idx_pref[i]) for i in ytick_positions]
    else:
        ytick_labels = [str(i + 1) for i in ytick_positions]
    ax.set_yticklabels(ytick_labels, rotation=0)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Trial')
    ax.set_title(
        f'{session} ({cue_angle}$\\degree$), {num_cells} cells, {trial_idx_pref.size} trials'
    )
    fig.savefig(fig_dir / f'{session}_{pref_cue}.png', dpi=300)
    plt.close(fig)

def total_unique_trials(partitions):
    """Count unique trials covered by a list of partitions."""
    if not partitions:
        return 0
    max_end = max(p['trial_end'] for p in partitions)
    covered = np.zeros(max_end, dtype=np.bool_)
    for p in partitions:
        covered[p['trial_start']:p['trial_end']] = True
    return int(covered.sum())

@dataclass
class Config:
    """CLI configuration for decoding confidence analysis."""
    n_jobs: int = 1 # numer of parallel jobs for single-trial decoding
    seed: int = 42 # random seed for cue label balancing and shuffling
    data_dir: Path = Path('data/nature') # directory with {session}.mat files
    cache_dir: Path = Path('cache/run_001') # directory for cached results and figures
    loo_cell_selection: bool = False # use leave-one-out cell selection entries when available
    min_cell_per_group: int = 12 # a good partition has at least one group with this many cells
    min_trials_good_session: int = 320 # a good session has at least this many trials in good partitions
    t_decode_start: int = -200
    t_decode_end: int = 1400
    t_decode_window: int = 50
    t_decode_step: int = 10
    n_decode_shuffle: int = 0 # number of label shuffles for null distribution of decoding confidence (0 to skip)
    plot_only: bool = False # if True, only generate plots from cached decoding results
    plot_actual_trial_id: bool = False # if True, y-axis shows actual trial ids instead of 1 to N
    max_sessions_to_run: int | None = None # max number of good sessions to process (None to run all)

def main(config: Config):
    """Run decoding confidence analysis or generate plots from cached results.

    This analysis filters cached partition analyses to find sessions with enough
    preferred-cue cells and sufficient stable trials. For each eligible session,
    it keeps correct trials from the preferred cue and its opposite, bins spike
    rates over sliding time windows, and decodes each preferred-cue trial in
    parallel to produce a time-by-trial confidence map. Optional label shuffles
    generate a null distribution. Results are cached and per-session heatmaps
    are saved; when plot-only mode is enabled, the cached results are used to
    regenerate figures without recomputing decoding.
    """
    cache_dir = config.cache_dir
    plot_actual_trial_id = config.plot_actual_trial_id

    cache_dir.mkdir(parents=True, exist_ok=True)
    selection_pkl = cache_dir / 'cell_trial_selection.pkl'
    decode_pkl = cache_dir / 'decoding_confidence.pkl'
    fig_dir = cache_dir / 'decoding_confidence'
    fig_dir.mkdir(parents=True, exist_ok=True)

    if config.plot_only:
        if not decode_pkl.exists():
            raise FileNotFoundError(f'Missing decoding file: {decode_pkl}')
        with open(decode_pkl, 'rb') as f:
            results = pickle.load(f)
        for res in results:
            trial_idx_pref = np.asarray(res.get('trial_idx', []), dtype=np.int64)
            bin_starts = np.asarray(res.get('time_bins', []))
            decode_confidence = np.asarray(res.get('decoding_confidence', []))
            if decode_confidence.size == 0:
                continue
            plot_decoding_heatmap(
                fig_dir,
                res.get('session', 'unknown_session'),
                res.get('cue', 0),
                res.get('cue_deg', 0),
                trial_idx_pref,
                bin_starts,
                decode_confidence,
                plot_actual_trial_id,
                int(res.get('num_cells', 0)),
            )
        return
    
    if not selection_pkl.exists():
        raise FileNotFoundError(f'Missing selection file: {selection_pkl}')
    with open(selection_pkl, 'rb') as f:
        selection_outs = pickle.load(f)

    # Filter partitions to those with enough cells per group.
    good_partitions = []
    for out in selection_outs:
        if out.get('max_num_cells_per_group', 0) < config.min_cell_per_group:
            continue
        # When LOO decoding is off, ignore LOO selection entries.
        trial_holdout = out.get('trial_holdout')
        if not config.loo_cell_selection and trial_holdout is not None:
            continue
        good_partitions.append(out)
    partitions_by_session = {}
    for out in good_partitions:
        partitions_by_session.setdefault(out['session'], []).append(out)

    # Identify good sessions with enough trials in good partitions.
    # Prefer baseline partitions (w/o LOO) for trial coverage and preferred cue estimation.
    good_sessions = {}
    for session, parts in partitions_by_session.items():
        baseline_parts = [p for p in parts if p.get('trial_holdout') is None]
        parts_for_trials = baseline_parts if baseline_parts else parts
        # Count unique trials covered by good partitions
        covered_trial_count = total_unique_trials(parts_for_trials)
        if covered_trial_count < config.min_trials_good_session:
            continue
        partition_pref = []
        # Session-level preferred cue estimation
        pref_parts = baseline_parts if baseline_parts else parts
        for p in pref_parts:
            pref_cues = np.asarray(p['cell_properties']['mean_pref_test'])
            pref_cue = preferred_cue_from_cells(pref_cues)
            if pref_cue is not None:
                partition_pref.append(pref_cue)
        session_pref = preferred_cue_from_partitions(np.asarray(partition_pref))
        if session_pref is not None:
            good_sessions[session] = {
                'preferred_cue': session_pref,
                'partitions': parts,
                'baseline_partitions': baseline_parts,
            }

    if not good_sessions:
        return

    results = []
    sessions = list(good_sessions.keys())
    if config.max_sessions_to_run is not None:
        sessions = sessions[:config.max_sessions_to_run]
    for idx_session, session in enumerate(sessions, start=1):
        session_info = good_sessions[session]
        print(f'Processing session {session} ({idx_session}/{len(sessions)})')
        data_file = config.data_dir / f'{session}.mat'
        if not data_file.exists():
            print(f'  Skipping: missing data file {data_file}')
            continue
        data = loadmat(data_file)
        spikes = np.asarray(data['spks'])
        cue_labels = np.asarray(data['cueAngIdx']).flatten().astype(np.int64)
        trial_boo_correct = np.asarray(data['isCorr']).flatten().astype(np.bool_)
        t = np.asarray(data['tc']).flatten()

        pref_cue = session_info['preferred_cue']
        opposite_cue = get_opposite_cue(pref_cue)

        partitions = session_info['partitions']
        baseline_parts = session_info.get('baseline_partitions', [])
        if not baseline_parts:
            baseline_parts = [p for p in partitions if p.get('trial_holdout') is None]
        # group partitions by their holdout trial (if any) for LOO cell selection
        holdout_map: dict[int, list] = {}
        if config.loo_cell_selection:
            for p in partitions:
                th = p.get('trial_holdout')
                if th is None:
                    continue
                holdout_map.setdefault(int(th), []).append(p)

        # quick sanity check: cell pool for preferred cue (from baseline partitions)
        baseline_cell_set = set()
        for p in baseline_parts:
            cell_props = p['cell_properties']
            pref_cues = np.asarray(cell_props['mean_pref_test'])
            cells = np.asarray(cell_props['cell_idx'])
            baseline_cell_set.update(cells[pref_cues == pref_cue].tolist())
        if not baseline_cell_set:
            print('  Skipping: no preferred-cue cells found')
            continue

        # Restrict to correct trials for preferred vs. opposite cue decoding.
        trial_mask = trial_boo_correct & ((cue_labels == pref_cue) | (cue_labels == opposite_cue))
        selected_trial_idx = np.nonzero(trial_mask)[0]
        if selected_trial_idx.size == 0:
            print('  Skipping: no trials for preferred/opposite cue')
            continue
        labels_sel = (cue_labels[selected_trial_idx] == pref_cue).astype(np.int64)
        test_sel_indices = np.nonzero(labels_sel == 1)[0]
        if test_sel_indices.size == 0:
            print('  Skipping: no preferred-cue trials to test')
            continue
        print(
            f'  Cue {pref_cue} vs {opposite_cue}, '
            f'~{len(baseline_cell_set)} cells, {selected_trial_idx.size} trials '
            f'({test_sel_indices.size} test trials)'
        )

        bin_starts = np.arange(config.t_decode_start, config.t_decode_end + 1, config.t_decode_step)

        def decode_trial(idx_test: int):
            trial_abs = int(selected_trial_idx[idx_test])
            if config.loo_cell_selection:
                baseline_non_overlap = [
                    p for p in baseline_parts
                    if not (trial_abs >= p['trial_start'] and trial_abs < p['trial_end'])
                ]
                holdout_parts = holdout_map.get(trial_abs, [])
                if not holdout_parts and not baseline_non_overlap:
                    return ('warn_no_partitions', trial_abs, None, None, None)
                applicable_parts = baseline_non_overlap + holdout_parts
            else:
                applicable_parts = baseline_parts

            cell_idx_set = set()
            for p in applicable_parts:
                cell_props = p['cell_properties']
                pref_cues = np.asarray(cell_props['mean_pref_test'])
                cells = np.asarray(cell_props['cell_idx'])
                cell_idx_set.update(cells[pref_cues == pref_cue].tolist())

            if not cell_idx_set:
                return ('warn_no_cells', trial_abs, None, None, None)

            cell_idx = np.asarray(sorted(cell_idx_set), dtype=np.int64)
            spikes_sel = spikes[selected_trial_idx][:, :, cell_idx]
            binned_rates = compute_binned_rates(spikes_sel, t, bin_starts, config.t_decode_window)

            conf, null_conf = decode_one_trial(
                idx_test,
                binned_rates,
                labels_sel,
                config.seed,
                config.n_decode_shuffle,
            )
            return ('ok', trial_abs, conf, null_conf, int(cell_idx.size))

        decoded = Parallel(n_jobs=config.n_jobs, verbose=10)(
            delayed(decode_trial)(idx) for idx in test_sel_indices
        )

        conf_list = []
        null_list = []
        trial_idx_pref = []
        num_cells_per_trial: list[int] = []
        for status, trial_abs, conf, null_conf, n_cell in decoded:
            if status == 'warn_no_partitions':
                print(f'  Skipping test trial {trial_abs}: no matching LOO partition and all baseline partitions overlap this trial')
                continue
            if status == 'warn_no_cells':
                print(f'  Skipping test trial {trial_abs}: no preferred-cue cells from applicable partitions')
                continue
            conf_list.append(conf)
            if null_conf is not None:
                null_list.append(null_conf)
            trial_idx_pref.append(trial_abs)
            num_cells_per_trial.append(n_cell)

        if not conf_list:
            print('  Skipping: no decodable preferred-cue trials with available cells')
            continue

        decode_confidence = np.stack(conf_list, axis=0)
        decode_confidence_null = None
        if config.n_decode_shuffle > 0:
            decode_confidence_null = np.stack(null_list, axis=0) if null_list else np.empty((0, len(bin_starts), config.n_decode_shuffle))

        cue_angle = int(cue_to_deg(pref_cue))
        results.append({
            'session': session,
            'cue': int(pref_cue),
            'cue_deg': cue_angle,
            'trial_idx': np.asarray(trial_idx_pref, dtype=np.int64),
            'time_bins': bin_starts,
            'decoding_confidence': decode_confidence,
            'decoding_confidence_null': decode_confidence_null,
            'num_cells': int(max(num_cells_per_trial)),
            'num_cells_per_trial': num_cells_per_trial,
            'num_trials': int(len(trial_idx_pref)),
        })

        plot_decoding_heatmap(
            fig_dir,
            session,
            pref_cue,
            cue_angle,
            np.asarray(trial_idx_pref, dtype=np.int64),
            bin_starts,
            decode_confidence,
            plot_actual_trial_id,
            int(max(num_cells_per_trial)),
        )

    with open(decode_pkl, 'wb') as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    config = tyro.cli(Config)
    main(config)
