"""Optional full-session screening diagnostics, outside the screening hot path."""
import json
import warnings
from textwrap import fill

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from scripts.next.cache_paths import stage_path
from scripts.next.common import load_session
from scripts.next.diagnostic_config import load_diagnostic_config, select_cells
from scripts.next.figure_exports import save_figure
from scripts.next.screening_names import rejection_label


def _figure_targets(frame, files, settings):
    """Resolve every target before exporting, without restricting screening or CSV rows."""
    if not settings.plots.enabled:
        return {}
    available = {path.stem for path in files}
    requested = (available if settings.targets.sessions == 'available'
                 else set(settings.targets.sessions))
    missing = (requested | set(settings.targets.cells_by_session)) - available
    if missing:
        warnings.warn('Diagnostic figure sessions unavailable in the selected input files: '
                      f'{", ".join(sorted(missing))}. Skipping their figures.', stacklevel=2)
    targets = {}
    for session in sorted(requested & available):
        rows = frame[frame.session == session] if not frame.empty else frame
        if rows.empty:
            warnings.warn(f'{session}: no per-cell screening diagnostics (session was skipped); '
                          'skipping diagnostic figures.', stacklevel=2)
            continue
        selector = settings.targets.cells_by_session.get(session, settings.targets.cells)
        cells = select_cells(selector, len(rows), session)
        limit = settings.plots.max_cells_per_session
        if limit is not None and len(cells) > limit:
            warnings.warn(f'{session}: limiting diagnostic figures to the first {limit} of '
                          f'{len(cells)} requested cells by ascending cell index. '
                          'Set plots.max_cells_per_session to null to export all.', stacklevel=2)
            cells = cells[:limit]
        targets[session] = list(cells)
    return targets


def save_diagnostics(rows, files, config, diagnostic_config=None):
    settings = diagnostic_config or load_diagnostic_config(config.diagnostics_figure_config)
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=['session', 'cell_idx', 'rejection_reason',
                                      'presence_ratio', 'baseline_all_trials_spearman_r'])
    targets = _figure_targets(frame, files, settings)
    directory = stage_path(config.cache_dir, 'select', 'diagnostics')
    directory.mkdir(parents=True, exist_ok=True)
    # Keep the effective settings and targets even if the source JSON is later edited.
    snapshot = {'configuration': settings.to_dict(), 'resolved_targets': targets}
    (directory / 'figure_config.json').write_text(json.dumps(snapshot, indent=2) + '\n')
    for path in files:
        mask = frame.session == path.stem
        if not mask.any():
            continue
        spikes, times, _, correct = load_session(path)
        baseline = spikes[:, (times >= config.baseline_drift_start_ms) & (times < config.baseline_drift_end_ms)].sum(axis=1)
        test = spikes[:, (times >= config.test_start_ms) & (times < config.test_end_ms)].sum(axis=1)
        presence_mask = (times >= config.presence_start_ms) & (times < config.presence_end_ms)
        presence = np.full(spikes.shape[2], np.nan)
        if correct.any() and presence_mask.any():
            presence = (spikes[correct][:, presence_mask].sum(axis=1) > 0).mean(axis=0)
        correlations = np.array([spearmanr(np.arange(len(baseline)), baseline[:, c]).statistic
                                 if np.ptp(baseline[:, c]) else np.nan for c in range(spikes.shape[2])])
        indices = frame.loc[mask, 'cell_idx'].to_numpy(dtype=int)
        frame.loc[mask, 'presence_ratio'] = presence[indices]
        frame.loc[mask, 'baseline_all_trials_spearman_r'] = correlations[indices]
        for cell in targets.get(path.stem, ()):
            fig, axes = plt.subplots(2, 1, sharex=True, figsize=settings.plots.size_inches,
                                     layout='constrained')
            try:
                axes[0].plot(baseline[:, cell])
                axes[0].set_ylabel('Baseline spikes')
                axes[0].set_title(f'[{config.baseline_drift_start_ms}, {config.baseline_drift_end_ms}) ms', fontsize=10)
                axes[1].plot(test[:, cell])
                axes[1].set_ylabel('Test-period spikes')
                axes[1].set_title(f'[{config.test_start_ms}, {config.test_end_ms}) ms', fontsize=10)
                axes[1].set_xlabel('Trial index (full session, correct + incorrect)')
                reasons = frame.loc[mask & (frame.cell_idx == cell), 'rejection_reason'].iloc[0]
                description = rejection_label(reasons,
                                              show_not_applicable=settings.plots.show_not_applicable_reasons)
                title_width = max(30, int(settings.plots.size_inches[0] * 12))
                fig.suptitle(f'{path.stem}, cell {cell}\n{fill(description, width=title_width)}')
                save_figure(fig, directory / 'figures' / 'cells' / f'{path.stem}_{cell}.png',
                            dpi=settings.plots.dpi)
            finally:
                plt.close(fig)
    frame.to_csv(directory / 'cell_rejection_diagnostics.csv', index=False)
