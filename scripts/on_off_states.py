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

matplotlib.use('Agg')

def show_spines(ax, lw=1, color='black'):
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(lw)
        spine.set_color(color)

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

def get_off_candidate_mask(z_map, z_threshold: float = 1.96, method: str = 'two_tailed'):
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

@dataclass
class Config:
    cache_dir: Path = Path('cache/run_001') # directory for cached results and figures
    z_threshold_on: float = 1.96
    z_threshold_off: float = 1.96
    cp_method_off: Literal['two_tailed', 'one_tailed'] = 'two_tailed'
    on_duration_xmax: float = 1000.0
    off_duration_xmax: float = 1000.0

def main(config: Config):
    cache_dir = config.cache_dir
    z_threshold_on = config.z_threshold_on
    z_threshold_off = config.z_threshold_off
    cp_method_off = config.cp_method_off
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

    # loop through outs and get decoding confidence and null distribution
    for out_dict in outs:
        decoding_confidence = out_dict.get('decoding_confidence', None) # (trial, bin)
        decoding_confidence_null = out_dict.get('decoding_confidence_null', None) # (trial, bin, shuffle)
        session = out_dict.get('session', 'unknown_session')
        cue = out_dict.get('cue', 'unknown_cue')
        bin_starts = np.asarray(out_dict.get('time_bins', None))
        
        # define delay period
        delay_start = 500 # first bin start
        delay_end = 1400 # last bin start

        if decoding_confidence is not None:
            # prepare placeholders
            on_state_mask = None
            on_state_ids = np.array([], dtype=int)
            on_state_labeled = None
            off_state_mask = None
            off_state_ids = np.array([], dtype=int)
            off_state_labeled = None
            null_cluster_masses = None

            if decoding_confidence_null is not None and decoding_confidence_null.shape[2] > 0:
                # get on-state mask using cluster mass approach
                # 1. convert decoding_confidence to z-score using null distribution mean and std
                # 2. find clusters where z-scores > 1.96 (97.5th percentile of standard normal distribution) and compute their cluster masses (sum of z-scores of a cluster)
                # 3. repeat 1 - 2 for each shuffle in null distribution of decoding to get null distribution of max cluster masses
                # 4. keep on-state clusters if their cluster masses are significantly different from null distribution (e.g., p < 0.05)
                # 5. create on-state mask

                # 1: convert to z-score
                null_mean = np.mean(decoding_confidence_null, axis=2)
                null_std = np.std(decoding_confidence_null, axis=2)
                safe_std = null_std.copy()
                safe_std[safe_std == 0] = np.nan
                z_map = (decoding_confidence - null_mean) / safe_std
                z_map = np.nan_to_num(z_map)

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

                # 3. get null distribution of max cluster masses (one per shuffle)
                z_null = (decoding_confidence_null - null_mean[:, :, None]) / safe_std[:, :, None]
                z_null = np.nan_to_num(z_null)
                null_max_masses = np.zeros(decoding_confidence_null.shape[2], dtype=float)
                for shuffle_idx in range(decoding_confidence_null.shape[2]):
                    z_null_slice = z_null[:, :, shuffle_idx]
                    supra_null = z_null_slice > 1.96
                    labeled_null, num_null_clusters = label(supra_null, structure=CONNECTIVITY_STRUCTURE)
                    if num_null_clusters:
                        null_masses = np.bincount(labeled_null.ravel(), weights=z_null_slice.ravel())
                        if null_masses.size > 1:
                            null_max_masses[shuffle_idx] = null_masses[1:].max()

                # 4. determine cluster mass cutoff at 95th percentile of null distribution
                # and keep on-state clusters above this cutoff
                cluster_cutoff = np.percentile(null_max_masses, 95) if null_max_masses.size else np.inf
                on_state_ids = np.where(on_cluster_masses > cluster_cutoff)[0]

                # 5. create on-state mask
                if on_state_ids.size:
                    on_state_mask = np.isin(on_state_labeled, on_state_ids)

                # get off-state mask using cluster mass approach
                # 1. convert decoding_confidence to z-score using null distribution mean and std
                # 2. find clusters where z-scores falling in between -1.96 and 1.96 (2.5th and 97.5th percentiles of standard normal distribution)
                # 3. keep clusters with size >= 5 as off-state candidates
                # 4. compute cluster masses for all off-state candidate clusters
                # 5. repeat 1 - 3 for each shuffle in null distribution of decoding to get null distribution of cluster masses
                # 6. keep off-state candidates if their cluster masses are not significantly different from null distribution (e.g., p >= 0.05)
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
                    candidate_ids = np.where(off_cluster_sizes >= 5)[0]
                    if candidate_ids.size:
                        # 4. compute cluster masses for all labeled off-state clusters (not just candidate clusters)
                        off_cluster_masses = np.bincount(off_state_labeled.ravel(), weights=z_map.ravel())
                        if off_cluster_masses.size == 0:
                            off_cluster_masses = np.zeros(1, dtype=float)
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
                                valid_null_ids = np.where(null_sizes >= 5)[0]
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
                        # 6. keep off-state candidates if their cluster masses are not significantly different from null distribution
                        lower_mass_cutoff = np.percentile(null_cluster_masses, 2.5)
                        upper_mass_cutoff = np.percentile(null_cluster_masses, 97.5)
                        keep_ids = []
                        for cid in candidate_ids:
                            mass = off_cluster_masses[cid]
                            if mass < lower_mass_cutoff or mass > upper_mass_cutoff:
                                continue
                            keep_ids.append(cid)
                        # 7. create off-state mask
                        if keep_ids:
                            off_state_ids = np.array(keep_ids, dtype=int)
                            off_state_mask = np.isin(off_state_labeled, off_state_ids)

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
            plt.savefig(fig_dir / f'decoding_confidence_{session}_{cue}.png', dpi=300)
            plt.close()

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
                plt.savefig(fig_dir / f'on_state_mask_{session}_{cue}.png', dpi=300)
                plt.close()

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
                plt.savefig(fig_dir / f'off_state_mask_{session}_{cue}.png', dpi=300)
                plt.close()

            # save on off state duration histograms
            # params for duration histograms
            bin_size = 50 
            on_bins = np.arange(0, on_duration_xmax + bin_size, bin_size) # for on-state plot
            off_bins = np.arange(0, off_duration_xmax + bin_size, bin_size) # for off-state plot
            on_xlim = (0, on_duration_xmax)
            off_xlim = (0, off_duration_xmax)
            ylim = (0, 100)

            if on_state_mask is not None and on_state_ids.size:
                # find non-zero entries in on_state_mask
                sig_rows, sig_cols = np.nonzero(on_state_mask)
                # find cluster ids for these entries
                sig_cluster_ids = on_state_labeled[sig_rows, sig_cols]
                valid = sig_cluster_ids > 0
                if np.any(valid):
                    # filter out background entries
                    sig_cluster_ids = sig_cluster_ids[valid]
                    sig_cols = sig_cols[valid]
                    # make room to store min and max col for each cluster
                    max_sig_label = sig_cluster_ids.max()
                    min_col = np.full(max_sig_label + 1, on_state_mask.shape[1], dtype=int)
                    max_col = np.zeros(max_sig_label + 1, dtype=int)
                    # group sig_cols by sig_cluster_ids (clusters) to get min and max col for each cluster
                    np.minimum.at(min_col, sig_cluster_ids, sig_cols)
                    np.maximum.at(max_col, sig_cluster_ids, sig_cols)
                    # filter min and max col with on_state_ids (valid on-state clusters)
                    start_idx = min_col[on_state_ids]
                    end_idx = max_col[on_state_ids]
                    start_ms = bin_starts[start_idx]
                    end_ms = bin_starts[end_idx]
                    # identify on-states that overlap with delay periods
                    overlap = (end_ms >= delay_start) & (start_ms <= delay_end)
                    if np.any(overlap):
                        # clip on-state if it extends beyond delay period
                        clipped_start = np.maximum(start_ms[overlap], delay_start)
                        clipped_end = np.minimum(end_ms[overlap], delay_end)
                        durations = clipped_end - clipped_start
                        durations = durations[durations > 0]
                        if durations.size:
                            fig, ax = plt.subplots(1, 1, figsize=(5, 4), layout='constrained')
                            sns.histplot(durations, bins=on_bins, ax=ax)
                            show_spines(ax)
                            plt.xlabel('Duration (ms)')
                            plt.ylabel('Count')
                            plt.title(f'On-State Duration\nSession: {session}, Cue: {cue_to_deg(cue)}°')
                            plt.xlim(*on_xlim)
                            plt.ylim(*ylim)
                            plt.savefig(fig_dir / f'on_state_duration_{session}_{cue}.png', dpi=300)
                            plt.close()

            if off_state_ids.size:
                off_rows, off_cols = np.nonzero(off_state_mask)
                # off_state_labels is a 1D array with length equal to number of non-zero entries in off_state_mask
                off_state_labels = off_state_labeled[off_rows, off_cols]
                # make room to store min and max col for each cluster
                max_off_label = off_state_labels.max()
                min_col = np.full(max_off_label + 1, off_state_mask.shape[1], dtype=int)
                max_col = np.zeros(max_off_label + 1, dtype=int)
                # group off_cols by off_state_labels (clusters) to get min and max col for each cluster
                np.minimum.at(min_col, off_state_labels, off_cols)
                np.maximum.at(max_col, off_state_labels, off_cols)
                # filter min and max col with off_state_ids (valid off-state clusters)
                off_start_idx = min_col[off_state_ids]
                off_end_idx = max_col[off_state_ids]
                off_start_ms = bin_starts[off_start_idx]
                off_end_ms = bin_starts[off_end_idx]
                # identify off-states that overlap with delay periods
                overlap = (off_end_ms >= delay_start) & (off_start_ms <= delay_end)
                if np.any(overlap):
                    # clip off-state if it extends beyond delay period
                    clipped_start = np.maximum(off_start_ms[overlap], delay_start)
                    clipped_end = np.minimum(off_end_ms[overlap], delay_end)
                    off_durations = clipped_end - clipped_start
                    off_durations = off_durations[off_durations > 0]
                    if off_durations.size:
                        fig, ax = plt.subplots(1, 1, figsize=(5, 4), layout='constrained')
                        sns.histplot(off_durations, bins=off_bins, ax=ax)
                        show_spines(ax)
                        plt.xlabel('Duration (ms)')
                        plt.ylabel('Count')
                        plt.title(f'Off-State Duration\nSession: {session}, Cue: {cue_to_deg(cue)}°')
                        plt.xlim(*off_xlim)
                        plt.ylim(*ylim)
                        plt.savefig(fig_dir / f'off_state_duration_{session}_{cue}.png', dpi=300)
                        plt.close()

            # save histgram of off-state null cluster masses
            if isinstance(null_cluster_masses, np.ndarray) and null_cluster_masses.size:
                masses = null_cluster_masses[np.isfinite(null_cluster_masses)]
                if masses.size:
                    bins_mass = np.arange(-10, 11, 1) # from -10 to 10 with bin size 1
                    fig, ax = plt.subplots(1, 1, figsize=(5, 4), layout='constrained')
                    sns.histplot(masses, bins=bins_mass, ax=ax)
                    show_spines(ax)
                    plt.xlabel('Cluster Mass')
                    plt.ylabel('Count')
                    plt.title(f'Off-State Null Cluster Masses\nSession: {session}, Cue: {cue_to_deg(cue)}°')
                    plt.savefig(fig_dir / f'off_state_null_cluster_masses_{session}_{cue}.png', dpi=300)
                    plt.close()

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
                    bins_mass = np.arange(-10, 11, 1) # from -10 to 10 with bin size 1
                    fig, ax = plt.subplots(1, 1, figsize=(5, 4), layout='constrained')
                    sns.histplot(masses, bins=bins_mass, ax=ax)
                    if delay_masses.size:
                        ax.hist(delay_masses, bins=bins_mass, histtype='step', linewidth=2, color='C1', label='Delay')
                        ax.legend()
                    show_spines(ax)
                    plt.xlabel('Cluster Mass')
                    plt.ylabel('Count')
                    plt.title(f'Off-State Candidate Cluster Masses\nSession: {session}, Cue: {cue_to_deg(cue)}°')
                    plt.savefig(fig_dir / f'off_state_candidate_cluster_masses_{session}_{cue}.png', dpi=300)
                    plt.close()

if __name__ == '__main__':
    config = tyro.cli(Config)
    main(config)
