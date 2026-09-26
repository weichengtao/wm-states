"""Observed and null decoding plots, independent of model fitting."""
from scripts.next.cache_paths import stage_path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scripts.next.figure_exports import save_figure


def plot_session(result, cache_dir, actual_trial_id=False):
    directory = stage_path(cache_dir, 'decode', 'figures')
    for key, title in [('decoding_confidence', 'Observed confidence'),
                       ('decoding_confidence_null', 'Mean null confidence'),
                       ('decoding_classifier_c', 'Observed log10(C)'),
                       ('decoding_classifier_c_null', 'Mean null log10(C)')]:
        values = np.asarray(result[key])
        if not values.size:
            continue
        if 'classifier_c' in key:
            values = np.log10(values)
        if values.ndim == 3:
            values = values.mean(axis=2)
        fig, axes = plt.subplots(2, 1, figsize=(7, 6), layout='constrained', sharex=True)
        times = result['time_bins']
        mesh = axes[0].pcolormesh(times, np.arange(len(values)), values, shading='nearest')
        axes[0].invert_yaxis()
        if actual_trial_id:
            ticks = np.linspace(0, len(values) - 1, min(8, len(values)), dtype=int)
            axes[0].set_yticks(ticks, np.asarray(result['trial_idx'])[ticks])
        axes[0].set_ylabel('Trial ID' if actual_trial_id else 'Trial row')
        fig.colorbar(mesh, ax=axes[0])
        axes[1].plot(times, values.T, alpha=0.12, color='tab:blue')
        axes[1].plot(times, values.mean(axis=0), color='black', label='Mean')
        if key == 'decoding_confidence':
            axes[1].plot(times, result['decoding_accuracy'], label='Accuracy')
        axes[1].legend(); axes[1].set_xlabel('Time (ms)')
        fig.suptitle(f'{result["session"]}: {title}')
        save_figure(fig, directory / f'{result["session"]}_{key}.png')
        plt.close(fig)
