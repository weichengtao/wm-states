import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tyro
from scipy.ndimage import label

try:
    from scripts.figure_exports import configure_figure_style, save_figure_all_formats
except ModuleNotFoundError:
    from figure_exports import configure_figure_style, save_figure_all_formats

matplotlib.use('Agg')
configure_figure_style(matplotlib)

def show_spines(ax, lw=1, color='black'):
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(lw)
        spine.set_color(color)

def add_off_cluster_cutoff_lines(ax, cutoffs):
    cutoffs = np.asarray(cutoffs, dtype=float)
    cutoffs = cutoffs[np.isfinite(cutoffs)]
    for cutoff_idx, cutoff in enumerate(cutoffs):
        cutoff_label = (
            'Off-state CC cutoff'
            if cutoffs.size == 1
            else f'Off-state CC cutoff {cutoff_idx + 1}'
        )
        ax.axvline(
            cutoff,
            color='black',
            linestyle='--',
            linewidth=1.5,
            label=cutoff_label,
        )
    if cutoffs.size:
        ax.relim()
        ax.autoscale_view()

def cue_to_deg(cue):
    '''
    {
        1: -135,
        2: -90,
        3: -45,
        4: 0,
        5: 45,
        6: 90,
        7: 135,
        8: 180,
    }
    '''
    cue = np.asarray(cue)
    cue = (cue - 1) % 8 + 1
    return (cue - 1) * 45 - 135

def get_off_candidate_mask(z_map, z_threshold: float = 1.645, method: str = 'one_tailed'):
    '''
    Get off-state candidate mask from z_map using specified method.
    method:
        'two_tailed': find clusters where z-scores falling in between -z_threshold and z_threshold
        'one_tailed': find clusters where z-scores below z_threshold
    '''
    if method == 'two_tailed':
        off_candidate_mask = np.abs(z_map) <= z_threshold
    elif method == 'one_tailed':
        off_candidate_mask = z_map <= z_threshold
    else:
        raise ValueError(f'Unknown method: {method}')
    return off_candidate_mask


def infer_time_bin_step(bin_starts):
    """Return the uniform time-bin step in milliseconds."""
    bin_starts = np.asarray(bin_starts, dtype=float)
    if bin_starts.size < 2:
        raise ValueError('At least two time bins are required to infer t_decode_step.')
    steps = np.diff(bin_starts)
    if not np.all(np.isfinite(steps)) or not np.all(steps > 0):
        raise ValueError('time_bins must be finite and strictly increasing.')
    if not np.allclose(steps, steps[0]):
        raise ValueError('time_bins must have a uniform t_decode_step.')
    return float(steps[0])


def off_state_duration_per_trial(
    off_state_mask,
    bin_starts,
    t_decode_step,
    delay_start,
    delay_end,
):
    """Count delay-period off-state bins and multiply by ``t_decode_step``."""
    bin_starts = np.asarray(bin_starts, dtype=float)
    delay_bins = (bin_starts >= delay_start) & (bin_starts <= delay_end)
    return (
        np.asarray(off_state_mask[:, delay_bins], dtype=float).sum(axis=1)
        * t_decode_step
    )


def state_durations(
    state_mask,
    state_ids,
    state_labeled,
    bin_starts,
    t_decode_step,
    delay_start,
    delay_end,
):
    """Return cluster durations as delay-bin counts times ``t_decode_step``."""
    if state_mask is None or not state_ids.size or state_labeled is None:
        return np.array([], dtype=float)

    bin_starts = np.asarray(bin_starts, dtype=float)
    delay_bins = (bin_starts >= delay_start) & (bin_starts <= delay_end)
    if not np.any(delay_bins):
        return np.array([], dtype=float)

    max_label = int(np.max(state_labeled))
    bin_counts = np.bincount(
        state_labeled[:, delay_bins].ravel(),
        minlength=max_label + 1,
    )
    durations = bin_counts[np.asarray(state_ids, dtype=int)] * t_decode_step
    return durations[durations > 0]


def _selected_repeat_indices(repeat_count: int, list_of_repeats):
    """Validate and return the requested zero-based repeat indices."""
    if list_of_repeats is None:
        return np.arange(repeat_count, dtype=np.int64)
    if len(list_of_repeats) == 0:
        raise ValueError('list_of_repeats must not be empty when subset selection is enabled.')
    if any(
        isinstance(repeat_idx, (bool, np.bool_))
        or not isinstance(repeat_idx, (int, np.integer))
        for repeat_idx in list_of_repeats
    ):
        raise ValueError('list_of_repeats must contain only integer indices.')
    repeat_indices = np.asarray(list_of_repeats, dtype=np.int64)
    if np.unique(repeat_indices).size != repeat_indices.size:
        raise ValueError('list_of_repeats must not contain duplicate indices.')
    if np.any(repeat_indices < 0) or np.any(repeat_indices >= repeat_count):
        raise IndexError(
            f'list_of_repeats indices must be between 0 and {repeat_count - 1}.'
        )
    return repeat_indices


def _decoding_confidence_for_analysis(out_dict, config):
    """Return decoding confidence, optionally averaged over selected repeats."""
    if not config.use_decoding_estimates_from_subset_of_repeats:
        confidence = out_dict.get('decoding_confidence', None)
        return None if confidence is None else np.asarray(confidence)

    confidence_repeats = out_dict.get('decoding_confidence_repeats')
    if confidence_repeats is None:
        raise ValueError(
            'The decoding cache is missing decoding_confidence_repeats; '
            'subset selection requires a repeat-format decoding cache.'
        )
    confidence_repeats = np.asarray(confidence_repeats)
    if confidence_repeats.ndim != 3:
        raise ValueError(
            'decoding_confidence_repeats must have shape (trial, repeat, bin).'
        )
    repeat_indices = _selected_repeat_indices(
        confidence_repeats.shape[1],
        config.list_of_repeats,
    )
    return np.nanmean(confidence_repeats[:, repeat_indices, :], axis=1)


@dataclass
class Config:
    cache_dir: Path = Path('cache/run_001') # directory for cached results and figures
    use_decoding_estimates_from_subset_of_repeats: bool = False
    list_of_repeats: list[int] | None = None
    z_threshold_on: float = 1.645
    z_threshold_off: float = 0.842
    cp_method_off: Literal['two_tailed', 'one_tailed'] = 'one_tailed'
    cluster_size_threshold_off: int = 5
    cc_method_on: Literal['one_tailed', 'two_tailed', 'skipped'] = 'one_tailed'
    cc_method_off: Literal['one_tailed', 'two_tailed', 'skipped'] = 'one_tailed'
    cc_alpha_on: float = 0.05
    cc_alpha_off: float = 0.05
    compare_with_cc_skipped_on: bool = False
    compare_with_cc_skipped_off: bool = False
    on_duration_xmax: float = 1000.0
    off_duration_xmax: float = 1000.0

def main(config: Config):
    cache_dir = config.cache_dir
    z_threshold_on = config.z_threshold_on
    z_threshold_off = config.z_threshold_off
    cp_method_off = config.cp_method_off
    cluster_size_threshold_off = config.cluster_size_threshold_off
    cc_method_on = config.cc_method_on
    cc_method_off = config.cc_method_off
    cc_alpha_on = config.cc_alpha_on
    cc_alpha_off = config.cc_alpha_off
    compare_with_cc_skipped_on = config.compare_with_cc_skipped_on
    compare_with_cc_skipped_off = config.compare_with_cc_skipped_off
    on_duration_xmax = max(100.0, config.on_duration_xmax)
    off_duration_xmax = max(100.0, config.off_duration_xmax)

    # for cluster identification
    CONNECTIVITY_STRUCTURE = np.zeros((3, 3), dtype=int)
    CONNECTIVITY_STRUCTURE[1, :] = 1

    # load decoding confidence cache
    with open(cache_dir / 'decoding_confidence.pkl', 'rb') as f:
        outs = pickle.load(f)

    # prepare figure dir for this analysis
    fig_dir = cache_dir / 'on_off_states'
    fig_dir.mkdir(parents=True, exist_ok=True)
    state_results = []

    # loop through outs and get decoding confidence and null distribution
    for out_dict in outs:
        decoding_confidence = _decoding_confidence_for_analysis(out_dict, config) # (trial, bin)
        decoding_confidence_null = out_dict.get('decoding_confidence_null', None) # (trial, bin, shuffle)
        session = out_dict.get('session', 'unknown_session')
        cue = out_dict.get('cue', 'unknown_cue')
        bin_starts = np.asarray(out_dict.get('time_bins', None))
        
        # define delay period
        delay_start = 500 # first bin start
        delay_end = 1400 # last bin start

        if decoding_confidence is not None:
            t_decode_step = infer_time_bin_step(bin_starts)
            # prepare placeholders
            on_state_mask = None
            on_state_ids = np.array([], dtype=int)
            on_state_labeled = None
            off_state_mask = None
            off_state_ids = np.array([], dtype=int)
            off_state_labeled = None
            null_cluster_masses = None
            candidate_ids = np.array([], dtype=int)
            off_cluster_masses = None
            off_cluster_mass_cutoffs = np.array([], dtype=float)
            on_state_mask_cc_skipped = None
            on_state_ids_cc_skipped = np.array([], dtype=int)
            on_state_labeled_cc_skipped = None
            off_state_mask_cc_skipped = None
            off_state_ids_cc_skipped = np.array([], dtype=int)
            off_state_labeled_cc_skipped = None

            if decoding_confidence_null is not None and decoding_confidence_null.shape[2] > 0:
                # get on-state mask using cluster mass approach
                # 1. convert decoding_confidence to z-score using null distribution mean and std
                # 2. find clusters where z-scores exceed z_threshold_on and compute their cluster masses (sum of z-scores of a cluster)
                # 3. repeat 1 - 2 for each shuffle in null distribution of decoding to get null distribution of max cluster masses
                # 4. apply the requested cluster-based correction
                # 5. create on-state mask

                # 1: convert to z-score
                null_mean = np.mean(decoding_confidence_null, axis=2)
                null_std = np.std(decoding_confidence_null, axis=2)
                safe_std = null_std.copy()
                safe_std[safe_std == 0] = np.nan
                z_map = (decoding_confidence - null_mean) / safe_std
                z_map = np.nan_to_num(z_map)

                # Standardized shuffled maps are needed by either non-skipped
                # cluster-based correction.  Compute them once for reuse below.
                z_null = None
                if cc_method_on != 'skipped' or cc_method_off != 'skipped':
                    z_null = (decoding_confidence_null - null_mean[:, :, None]) / safe_std[:, :, None]
                    z_null = np.nan_to_num(z_null)

                # 2. get on-state candidate clusters and compute their cluster masses
                on_candidate_mask = z_map > z_threshold_on
                # the on_state_labeled is a 2d array with same shape as z_map
                # each cluster is labeled with an integer starting from 1; background is labeled as 0
                on_state_labeled, _ = label(on_candidate_mask, structure=CONNECTIVITY_STRUCTURE)
                # compute cluster masses for each labeled on-state cluster (on_cluster_masses[i] is the mass of cluster i)
                on_cluster_masses = np.bincount(on_state_labeled.ravel(), weights=z_map.ravel())
                # if there is no cluster, ensure on_cluster_masses has at least one element (0 for background)
                if on_cluster_masses.size == 0:
                    on_cluster_masses = np.zeros(1, dtype=float)
                # set background mass to 0 (on_cluster_masses[0] corresponds to background)
                on_cluster_masses[0] = 0.0

                if cc_method_on == 'skipped':
                    # Skip cluster-based correction but retain all thresholded clusters.
                    on_state_ids = np.flatnonzero(on_cluster_masses > 0)
                else:
                    # 3. get null distribution of max cluster masses (one per shuffle)
                    null_max_masses = np.zeros(decoding_confidence_null.shape[2], dtype=float)
                    for shuffle_idx in range(decoding_confidence_null.shape[2]):
                        z_null_slice = z_null[:, :, shuffle_idx]
                        supra_null = z_null_slice > z_threshold_on
                        labeled_null, num_null_clusters = label(supra_null, structure=CONNECTIVITY_STRUCTURE)
                        if num_null_clusters:
                            null_masses = np.bincount(labeled_null.ravel(), weights=z_null_slice.ravel())
                            if null_masses.size > 1:
                                null_max_masses[shuffle_idx] = null_masses[1:].max()

                    # 4. determine the cluster-mass cutoff based on the requested tail
                    if cc_method_on == 'one_tailed':
                        cutoff_percentile = 100 * (1 - cc_alpha_on)
                    else:  # two_tailed
                        cutoff_percentile = 100 * (1 - cc_alpha_on / 2)
                    cluster_cutoff = np.percentile(null_max_masses, cutoff_percentile) if null_max_masses.size else np.inf
                    on_state_ids = np.where(on_cluster_masses > cluster_cutoff)[0]

                # 5. create on-state mask
                if on_state_ids.size:
                    on_state_mask = np.isin(on_state_labeled, on_state_ids)

                # get off-state mask using cluster mass approach
                # 1. convert decoding_confidence to z-score using null distribution mean and std
                # 2. find off-state candidate clusters using the requested candidate-point method
                # 3. keep clusters with size >= cluster_size_threshold_off as off-state candidates
                # 4. compute cluster masses for all off-state candidate clusters
                # 5. repeat 1 - 3 for each shuffle in null distribution of decoding to get null distribution of cluster masses
                # 6. apply the requested cluster-based correction
                # 7. create off-state mask

                # 1: z_map already computed above

                # 2 - 3: get off-state candidate clusters with size thresholding
                off_candidate_mask = get_off_candidate_mask(z_map, z_threshold_off, method=cp_method_off)
                off_state_labeled, num_off_clusters = label(off_candidate_mask, structure=CONNECTIVITY_STRUCTURE)
                if num_off_clusters:
                    off_cluster_sizes = np.bincount(off_state_labeled.ravel())
                    if off_cluster_sizes.size == 0:
                        off_cluster_sizes = np.zeros(1, dtype=int)
                    off_cluster_sizes[0] = 0
                    candidate_ids = np.where(off_cluster_sizes >= cluster_size_threshold_off)[0]
                    if candidate_ids.size:
                        # 4. compute cluster masses for all labeled off-state clusters (not just candidate clusters)
                        off_cluster_masses = np.bincount(off_state_labeled.ravel(), weights=z_map.ravel())
                        if off_cluster_masses.size == 0:
                            off_cluster_masses = np.zeros(1, dtype=float)
                        if cc_method_off == 'skipped':
                            # Skip cluster-based correction but retain size-qualified candidates.
                            keep_ids = candidate_ids.tolist()
                        else:
                            # 5. get null distribution of off-state cluster masses (n valid clusters per shuffle)
                            null_cluster_masses = []
                            for shuffle_idx in range(decoding_confidence_null.shape[2]):
                                z_null_slice = z_null[:, :, shuffle_idx]
                                off_null_mask = get_off_candidate_mask(z_null_slice, z_threshold_off, method=cp_method_off)
                                labeled_null, num_null_clusters = label(off_null_mask, structure=CONNECTIVITY_STRUCTURE)
                                if num_null_clusters:
                                    null_sizes = np.bincount(labeled_null.ravel())
                                    if null_sizes.size == 0:
                                        null_sizes = np.zeros(1, dtype=int)
                                    null_sizes[0] = 0
                                    valid_null_ids = np.where(null_sizes >= cluster_size_threshold_off)[0]
                                    if valid_null_ids.size:
                                        # compute cluster masses for valid null clusters
                                        masses = np.bincount(labeled_null.ravel(), weights=z_null_slice.ravel())
                                        if masses.size == 0:
                                            masses = np.zeros(1, dtype=float)
                                        null_cluster_masses.append(masses[valid_null_ids])
                            if null_cluster_masses:
                                null_cluster_masses = np.concatenate(null_cluster_masses)
                            else:
                                null_cluster_masses = np.zeros(1, dtype=float)
                            null_cluster_masses = null_cluster_masses[np.isfinite(null_cluster_masses)]
                            if null_cluster_masses.size == 0:
                                null_cluster_masses = np.zeros(1, dtype=float)
                            # 6. keep off-state candidates according to the requested correction tail
                            if cc_method_off == 'one_tailed':
                                upper_mass_cutoff = np.percentile(null_cluster_masses, 100 * (1 - cc_alpha_off))
                                off_cluster_mass_cutoffs = np.array([upper_mass_cutoff])
                                keep_ids = [cid for cid in candidate_ids if off_cluster_masses[cid] <= upper_mass_cutoff]
                            else:  # two_tailed
                                lower_mass_cutoff = np.percentile(null_cluster_masses, 100 * (cc_alpha_off / 2))
                                upper_mass_cutoff = np.percentile(null_cluster_masses, 100 * (1 - cc_alpha_off / 2))
                                off_cluster_mass_cutoffs = np.array([lower_mass_cutoff, upper_mass_cutoff])
                                keep_ids = [
                                    cid for cid in candidate_ids
                                    if lower_mass_cutoff <= off_cluster_masses[cid] <= upper_mass_cutoff
                                ]
                        # 7. create off-state mask
                        if keep_ids:
                            off_state_ids = np.array(keep_ids, dtype=int)
                            off_state_mask = np.isin(off_state_labeled, off_state_ids)

                # Reuse the same candidate clusters with correction skipped for
                # optional duration-histogram comparisons. This changes only
                # the relevant cluster-correction step.
                if compare_with_cc_skipped_on and cc_method_on != 'skipped':
                    on_state_labeled_cc_skipped = on_state_labeled
                    on_state_ids_cc_skipped = np.flatnonzero(on_cluster_masses > 0)
                    if on_state_ids_cc_skipped.size:
                        on_state_mask_cc_skipped = np.isin(
                            on_state_labeled_cc_skipped,
                            on_state_ids_cc_skipped,
                        )

                if compare_with_cc_skipped_off and cc_method_off != 'skipped':
                    off_state_labeled_cc_skipped = off_state_labeled
                    off_state_ids_cc_skipped = candidate_ids.copy()
                    if off_state_ids_cc_skipped.size:
                        off_state_mask_cc_skipped = np.isin(
                            off_state_labeled_cc_skipped,
                            off_state_ids_cc_skipped,
                        )

            # save decoding confidence heatmap
            fig, ax = plt.subplots(1, 1, figsize=(5, 4), layout='constrained')
            sns.heatmap(decoding_confidence, vmin=0.5, vmax=1.0, ax=ax)
            show_spines(ax)
            plt.xlabel('Time (ms)')
            plt.ylabel('Trial')
            plt.title(f'Decoding Confidence\nSession: {session}, Cue: {cue_to_deg(cue)}°')
            # set xticks and xticklabels; rotate to horizontal
            xticks = np.arange(0, len(bin_starts), 20) # every 200ms
            xticklabels = bin_starts[xticks]
            plt.xticks(xticks, xticklabels, rotation=0)
            # set yticks and yticklabels
            yticks = np.arange(10, decoding_confidence.shape[0], 10) # every 10 trials starting from trial 10
            yticklabels = yticks
            plt.yticks(yticks, yticklabels)
            # set limits and invert y axis
            plt.xlim(0, len(bin_starts)) # ensure all time bins are shown
            plt.ylim(decoding_confidence.shape[0], 0) # ensure all trials are shown
            # save figure to fig_dir with session and cue in filename
            save_figure_all_formats(fig, fig_dir / f'decoding_confidence_{session}_{cue}.png', dpi=300)
            plt.close(fig)

            # save on-state mask if exists
            if on_state_mask is not None:
                fig, ax = plt.subplots(1, 1, figsize=(5, 4), layout='constrained')
                sns.heatmap(on_state_mask.astype(float), vmin=0, vmax=1, ax=ax)
                show_spines(ax)
                plt.xlabel('Time (ms)')
                plt.ylabel('Trial')
                plt.title(f'On-State Mask\nSession: {session}, Cue: {cue_to_deg(cue)}°')
                # set xticks and xticklabels; rotate to horizontal
                xticks = np.arange(0, len(bin_starts), 20) # every 200ms
                xticklabels = bin_starts[xticks]
                plt.xticks(xticks, xticklabels, rotation=0)
                # set yticks and yticklabels
                yticks = np.arange(10, decoding_confidence.shape[0], 10) # every 10 trials starting from trial 10
                yticklabels = yticks
                plt.yticks(yticks, yticklabels)
                # set limits and invert y axis
                plt.xlim(0, len(bin_starts)) # ensure all time bins are shown
                plt.ylim(decoding_confidence.shape[0], 0) # ensure all trials are shown
                # save figure to fig_dir with session and cue in filename
                save_figure_all_formats(fig, fig_dir / f'on_state_mask_{session}_{cue}.png', dpi=300)
                plt.close(fig)

            # save off-state mask if exists
            if off_state_mask is not None and off_state_ids.size:
                fig, ax = plt.subplots(1, 1, figsize=(5, 4), layout='constrained')
                sns.heatmap(off_state_mask.astype(float), vmin=0, vmax=1, ax=ax)
                show_spines(ax)
                plt.xlabel('Time (ms)')
                plt.ylabel('Trial')
                plt.title(f'Off-State Mask\nSession: {session}, Cue: {cue_to_deg(cue)}°')
                xticks = np.arange(0, len(bin_starts), 20)
                xticklabels = bin_starts[xticks]
                plt.xticks(xticks, xticklabels, rotation=0)
                yticks = np.arange(10, decoding_confidence.shape[0], 10)
                yticklabels = yticks
                plt.yticks(yticks, yticklabels)
                plt.xlim(0, len(bin_starts))
                plt.ylim(decoding_confidence.shape[0], 0)
                save_figure_all_formats(fig, fig_dir / f'off_state_mask_{session}_{cue}.png', dpi=300)
                plt.close(fig)

            # save on off state duration histograms
            # params for duration histograms
            bin_size = 50 
            on_bins = np.arange(0, on_duration_xmax + bin_size, bin_size) # for on-state plot
            off_bins = np.arange(0, off_duration_xmax + bin_size, bin_size) # for off-state plot
            on_xlim = (0, on_duration_xmax)
            off_xlim = (0, off_duration_xmax)

            compare_on = compare_with_cc_skipped_on and cc_method_on != 'skipped'
            on_durations = state_durations(
                on_state_mask,
                on_state_ids,
                on_state_labeled,
                bin_starts,
                t_decode_step,
                delay_start,
                delay_end,
            )
            on_durations_cc_skipped = state_durations(
                on_state_mask_cc_skipped if compare_on else None,
                on_state_ids_cc_skipped,
                on_state_labeled_cc_skipped,
                bin_starts,
                t_decode_step,
                delay_start,
                delay_end,
            )
            if on_durations.size or on_durations_cc_skipped.size:
                fig, ax = plt.subplots(1, 1, figsize=(5, 4), layout='constrained')
                if compare_on:
                    if on_durations_cc_skipped.size:
                        sns.histplot(
                            on_durations_cc_skipped,
                            bins=on_bins,
                            ax=ax,
                            color='tab:orange',
                            label='CC skipped',
                        )
                    if on_durations.size:
                        sns.histplot(
                            on_durations,
                            bins=on_bins,
                            ax=ax,
                            color='tab:blue',
                            label='CC applied',
                        )
                    if on_durations.size and on_durations_cc_skipped.size:
                        ax.legend(frameon=False)
                else:
                    sns.histplot(on_durations, bins=on_bins, ax=ax)
                show_spines(ax)
                plt.xlabel('Duration (ms)')
                plt.ylabel('Count')
                plt.title(f'On-State Duration\nSession: {session}, Cue: {cue_to_deg(cue)}°')
                plt.xlim(*on_xlim)
                save_figure_all_formats(fig, fig_dir / f'on_state_duration_{session}_{cue}.png', dpi=300)
                plt.close(fig)

            compare_off = compare_with_cc_skipped_off and cc_method_off != 'skipped'
            off_durations = state_durations(
                off_state_mask,
                off_state_ids,
                off_state_labeled,
                bin_starts,
                t_decode_step,
                delay_start,
                delay_end,
            )
            off_durations_cc_skipped = state_durations(
                off_state_mask_cc_skipped if compare_off else None,
                off_state_ids_cc_skipped,
                off_state_labeled_cc_skipped,
                bin_starts,
                t_decode_step,
                delay_start,
                delay_end,
            )
            off_duration_per_trial = np.zeros(
                decoding_confidence.shape[0], dtype=float
            )
            if off_state_mask is not None:
                off_duration_per_trial = off_state_duration_per_trial(
                    off_state_mask,
                    bin_starts,
                    t_decode_step,
                    delay_start,
                    delay_end,
                )

            off_duration_per_trial_cc_skipped = np.array([], dtype=float)
            if compare_off:
                off_duration_per_trial_cc_skipped = np.zeros(
                    decoding_confidence.shape[0], dtype=float
                )
                if off_state_mask_cc_skipped is not None:
                    off_duration_per_trial_cc_skipped = (
                        off_state_duration_per_trial(
                            off_state_mask_cc_skipped,
                            bin_starts,
                            t_decode_step,
                            delay_start,
                            delay_end,
                        )
            )
            if off_durations.size or off_durations_cc_skipped.size:
                fig, ax = plt.subplots(1, 1, figsize=(5, 4), layout='constrained')
                if compare_off:
                    if off_durations.size:
                        sns.histplot(
                            off_durations,
                            bins=off_bins,
                            ax=ax,
                            color='tab:blue',
                            label='CC applied',
                        )
                    if off_durations_cc_skipped.size:
                        sns.histplot(
                            off_durations_cc_skipped,
                            bins=off_bins,
                            ax=ax,
                            element='step',
                            fill=False,
                            linewidth=2,
                            color='tab:orange',
                            label='CC skipped',
                        )
                    if off_durations.size and off_durations_cc_skipped.size:
                        ax.legend(frameon=False)
                else:
                    sns.histplot(off_durations, bins=off_bins, ax=ax)
                show_spines(ax)
                plt.xlabel('Duration (ms)')
                plt.ylabel('Count')
                plt.title(f'Off-State Duration\nSession: {session}, Cue: {cue_to_deg(cue)}°')
                plt.xlim(*off_xlim)
                save_figure_all_formats(fig, fig_dir / f'off_state_duration_{session}_{cue}.png', dpi=300)
                plt.close(fig)

            # Unlike the state-level histogram, this includes one value for
            # every trial, including trials with zero delay-period duration.
            fig, ax = plt.subplots(1, 1, figsize=(5, 4), layout='constrained')
            if compare_off:
                sns.histplot(
                    off_duration_per_trial,
                    bins=off_bins,
                    ax=ax,
                    color='tab:blue',
                    label='CC applied',
                )
                sns.histplot(
                    off_duration_per_trial_cc_skipped,
                    bins=off_bins,
                    ax=ax,
                    element='step',
                    fill=False,
                    linewidth=2,
                    color='tab:orange',
                    label='CC skipped',
                )
                ax.legend(frameon=False)
            else:
                sns.histplot(off_duration_per_trial, bins=off_bins, ax=ax)
            show_spines(ax)
            plt.xlabel('Total duration per trial (ms)')
            plt.ylabel('Count')
            plt.title(
                f'Trial-Level Off-State Duration\n'
                f'Session: {session}, Cue: {cue_to_deg(cue)}°'
            )
            plt.xlim(*off_xlim)
            save_figure_all_formats(
                fig,
                fig_dir / f'off_state_duration_per_trial_{session}_{cue}.png',
                dpi=300,
            )
            plt.close(fig)

            # save histgram of off-state null cluster masses
            if isinstance(null_cluster_masses, np.ndarray) and null_cluster_masses.size:
                masses = null_cluster_masses[np.isfinite(null_cluster_masses)]
                if masses.size:
                    bins_mass = np.histogram_bin_edges(masses, bins='auto')
                    fig, ax = plt.subplots(1, 1, figsize=(5, 4), layout='constrained')
                    sns.histplot(masses, bins=bins_mass, ax=ax)
                    add_off_cluster_cutoff_lines(ax, off_cluster_mass_cutoffs)
                    show_spines(ax)
                    plt.xlabel('Cluster Mass')
                    plt.ylabel('Count')
                    plt.title(f'Off-State Null Cluster Masses\nSession: {session}, Cue: {cue_to_deg(cue)}°')
                    if off_cluster_mass_cutoffs.size:
                        ax.legend()
                    save_figure_all_formats(fig, fig_dir / f'off_state_null_cluster_masses_{session}_{cue}.png', dpi=300)
                    plt.close(fig)

            # save histgram of off-state candidate cluster masses
            if isinstance(off_cluster_masses, np.ndarray) and off_cluster_masses.size:
                masses = off_cluster_masses[np.isfinite(off_cluster_masses)]
                # masses is a 1D array with length equal to number of labeled off-state clusters
                # it includes masses of all off-state clusters including those not passing size threshold
                # we want to plot the distribution of masses of all off-state candidate clusters that passed size threshold
                # so we filter masses with candidate_ids
                if candidate_ids.size:
                    masses = masses[candidate_ids]
                if masses.size:
                    delay_masses = np.array([], dtype=float)
                    if candidate_ids.size and off_state_labeled is not None:
                        off_rows, off_cols = np.nonzero(off_state_labeled)
                        if off_rows.size:
                            off_labels = off_state_labeled[off_rows, off_cols]
                            max_label = off_labels.max()
                            min_col = np.full(max_label + 1, off_state_labeled.shape[1], dtype=int)
                            max_col = np.zeros(max_label + 1, dtype=int)
                            np.minimum.at(min_col, off_labels, off_cols)
                            np.maximum.at(max_col, off_labels, off_cols)
                            candidate_start_idx = min_col[candidate_ids]
                            candidate_end_idx = max_col[candidate_ids]
                            start_ms = bin_starts[candidate_start_idx]
                            end_ms = bin_starts[candidate_end_idx]
                            overlap = (end_ms >= delay_start) & (start_ms <= delay_end)
                            if np.any(overlap):
                                delay_masses = masses[overlap]
                    bins_mass = np.histogram_bin_edges(masses, bins='auto')
                    fig, ax = plt.subplots(1, 1, figsize=(5, 4), layout='constrained')
                    sns.histplot(masses, bins=bins_mass, ax=ax)
                    add_off_cluster_cutoff_lines(ax, off_cluster_mass_cutoffs)
                    if delay_masses.size:
                        ax.hist(delay_masses, bins=bins_mass, histtype='step', linewidth=2, color='C1', label='Delay')
                    if delay_masses.size or off_cluster_mass_cutoffs.size:
                        ax.legend()
                    show_spines(ax)
                    plt.xlabel('Cluster Mass')
                    plt.ylabel('Count')
                    plt.title(f'Off-State Candidate Cluster Masses\nSession: {session}, Cue: {cue_to_deg(cue)}°')
                    save_figure_all_formats(fig, fig_dir / f'off_state_candidate_cluster_masses_{session}_{cue}.png', dpi=300)
                    plt.close(fig)

            state_results.append({
                'session': session,
                'cue': cue,
                'trial_idx': np.asarray(out_dict.get('trial_idx', []), dtype=np.int64),
                'off_state_duration_per_trial': off_duration_per_trial,
                'off_state_duration_per_state': np.asarray(off_durations, dtype=float),
                'off_state_duration_correction': 'applied',
                'off_state_duration_delay_start': delay_start,
                'off_state_duration_delay_end': delay_end,
                't_decode_step': t_decode_step,
            })

    with open(cache_dir / 'on_off_states.pkl', 'wb') as f:
        pickle.dump(state_results, f)

if __name__ == '__main__':
    config = tyro.cli(Config)
    main(config)
