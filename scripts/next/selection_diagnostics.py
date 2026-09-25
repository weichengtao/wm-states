"""Optional full-session screening diagnostics, outside the screening hot path."""
from scripts.next.cache_paths import stage_path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scripts.next.common import load_session
from scripts.next.figure_exports import save_figure_png_only


def save_diagnostics(rows, files, config):
    directory = stage_path(config.cache_dir, 'select', 'diagnostics')
    directory.mkdir(parents=True, exist_ok=True)
    targets = {}
    if config.diagnostics_figure_config is not None:
        for entry in json.loads(config.diagnostics_figure_config.read_text()).get('figures', []):
            if entry.get('trial_start', 0) != 0 or entry.get('trial_holdout') is not None or entry.get('trial_end') is not None:
                raise ValueError('Diagnostic targets must specify full sessions; omit trial bounds and holdouts.')
            targets[str(entry['session'])] = entry
    frame = pd.DataFrame(rows)
    for path in files:
        mask = frame.session == path.stem
        if not mask.any():
            continue
        spikes, times, _, correct = load_session(path)
        baseline = spikes[:, (times >= config.baseline_drift_start) & (times < config.baseline_drift_end)].sum(axis=1)
        delay = spikes[:, (times >= config.t_test_start) & (times < config.t_test_end)].sum(axis=1)
        presence_mask = (times >= config.presence_start) & (times < config.presence_end)
        presence = np.full(spikes.shape[2], np.nan)
        if correct.any() and presence_mask.any():
            presence = (spikes[correct][:, presence_mask].sum(axis=1) > 0).mean(axis=0)
        correlations = [spearmanr(np.arange(len(baseline)), baseline[:, c]).statistic
                        if np.ptp(baseline[:, c]) else np.nan for c in range(spikes.shape[2])]
        frame.loc[mask, 'presence_ratio'] = presence
        frame.loc[mask, 'r_s_baseline'] = correlations
        entry = targets.get(path.stem)
        if entry is None:
            continue
        cells = set(entry.get('cells', []))
        if 'cell_start' in entry or 'cell_end' in entry:
            cells.update(range(entry.get('cell_start', 0), entry.get('cell_end', spikes.shape[2] - 1) + 1))
        for cell in sorted(cells):
            if not 0 <= cell < spikes.shape[2]:
                raise ValueError(f'{path.stem}: diagnostic cell {cell} is out of range.')
            fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 5), layout='constrained')
            axes[0].plot(baseline[:, cell]); axes[0].set_ylabel('Baseline spikes')
            axes[1].plot(delay[:, cell]); axes[1].set_ylabel('Delay spikes')
            axes[1].set_xlabel('Trial')
            reason = frame.loc[mask & (frame.cell_idx == cell), 'rejection_reason'].iloc[0]
            if config.skip_not_applicable_reasons_in_diagnostics_figure:
                reason = '|'.join(r for r in reason.split('|') if not r.endswith('_not_applicable')) or reason
            fig.suptitle(f'{path.stem}, cell {cell}: {reason}')
            save_figure_png_only(fig, directory / 'figures' / 'cells' / f'{path.stem}_{cell}.png')
            plt.close(fig)
    frame.to_csv(directory / 'cell_rejection_diagnostics.csv', index=False)
