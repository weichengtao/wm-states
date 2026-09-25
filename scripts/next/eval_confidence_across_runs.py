"""Compare cached confidence evaluation scores across runs."""

# Use one module namespace for direct CLI and package execution.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "scripts.next"


from scripts.next import cache_io as pickle
from scripts.next.common import full_session_selection
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tyro
from scipy.stats import t

from scripts.next.figure_exports import configure_figure_style, save_figure_png_only

configure_figure_style(matplotlib)

METRICS = (
    ('brier_score', 'Brier score'),
    ('log_loss', 'Log loss'),
    ('accuracy', 'Accuracy'),
    ('decoding_confidence', 'Decoding confidence'),
)


@dataclass
class Config:
    """Compare session scores from two or more eval_confidence.pkl caches."""

    cache_dirs: list[Path]  # ordered run directories; figures go in every run cache
    run_aliases: list[str] | None = None  # legend names, in cache_dirs order
    line_colors: list[str] | None = None  # matplotlib colors, in cache_dirs order
    null_shading: Literal['confidence_intervals', 'percentiles', 'none'] = 'percentiles'
    # Null bands: 95% CI of the mean, 2.5–97.5 percentiles, or no shading.


def load_runs(cache_dirs):
    if len(cache_dirs) < 2:
        raise ValueError('Provide at least two --cache-dirs.')
    paths = [Path(path).resolve() for path in cache_dirs]
    if len(set(paths)) != len(paths):
        raise ValueError('Run cache directories must be distinct.')
    names = [path.name for path in paths]
    if len(set(names)) != len(names):
        raise ValueError('Run cache directories must have distinct folder names.')
    runs = []
    for path in paths:
        cache_file = path / 'eval_confidence.pkl'
        if not cache_file.exists():
            raise FileNotFoundError(
                f'Missing {cache_file}. Run scripts/next/eval_confidence.py '
                f'--cache-dir {path} first.'
            )
        with cache_file.open('rb') as stream:
            results = pickle.load(stream)
        sessions = {}
        for result in results:
            session = str(result['session'])
            for kind in ('observed', 'null'):
                scores = result.get(kind)
                if scores is not None and any(
                    f'{metric}_by_time_bin' not in scores for metric, _ in METRICS
                ):
                    raise ValueError(f'{cache_file}, session {session}: missing metrics. '
                                     f'Rerun scripts/next/eval_confidence.py --cache-dir {path}.')
            if session in sessions:
                raise ValueError(f'Duplicate session {session} in {cache_file}.')
            sessions[session] = result
        if not sessions:
            raise ValueError(f'No evaluated sessions in {cache_file}.')
        runs.append(sessions)
    # Preserve the first run's order and align by ID, never by cache row.
    common = [session for session in runs[0] if all(session in run for run in runs)]
    if not common:
        raise ValueError('The runs have no common evaluated sessions.')
    for name, run in zip(names, runs):
        omitted = [session for session in run if session not in common]
        if omitted:
            print(f'{name}: omitting sessions absent from another run: {", ".join(omitted)}')
    return names, runs, common


def score_curve(result, kind, metric, num_bins, null_shading='percentiles'):
    """Return observed scores or null means with optional bands."""
    if null_shading not in ('confidence_intervals', 'percentiles', 'none'):
        raise ValueError(f'Unknown null shading mode: {null_shading}')
    scores = result.get(kind)
    if kind == 'observed':
        if scores is None:
            raise ValueError('Observed scores are missing; rerun evaluation.')
        curve = np.asarray(scores[f'{metric}_by_time_bin'], dtype=float)
        if curve.shape != (num_bins,):
            raise ValueError('Observed scores do not match the time axis.')
        return curve, None
    if scores is None:
        return np.full(num_bins, np.nan), None
    samples = scores.get(f'{metric}_by_time_bin_and_sample')
    if samples is None:
        return np.asarray(scores[f'{metric}_by_time_bin'], dtype=float), None
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[0] != num_bins or samples.shape[1] == 0:
        raise ValueError(f'{kind} {metric} samples must have shape (time bin, sample).')
    valid = np.isfinite(samples)
    counts = valid.sum(axis=1)
    means = np.divide(
        np.where(valid, samples, 0).sum(axis=1), counts,
        out=np.full(num_bins, np.nan), where=counts > 0,
    )
    if null_shading == 'none' or not np.any(counts > 1):
        return means, None
    if null_shading == 'percentiles':
        bounds = np.full((2, num_bins), np.nan)
        for bin_idx in np.flatnonzero(counts > 1):
            bounds[:, bin_idx] = np.percentile(
                samples[bin_idx, valid[bin_idx]], [2.5, 97.5]
            )
        return means, (bounds[0], bounds[1])
    squared_errors = np.where(valid, samples - means[:, None], 0) ** 2
    variance = np.divide(
        squared_errors.sum(axis=1), counts - 1,
        out=np.full(num_bins, np.nan), where=counts > 1,
    )
    standard_error = np.sqrt(np.divide(
        variance, counts, out=np.full(num_bins, np.nan), where=counts > 1,
    ))
    half_width = t.ppf(0.975, np.where(counts > 1, counts - 1, np.nan)) * standard_error
    return means, (means - half_width, means + half_width)


def resolve_line_colors(line_colors, num_runs):
    if line_colors is None:
        return [plt.get_cmap('tab10')(index % 10) for index in range(num_runs)]
    if len(line_colors) != num_runs:
        raise ValueError('Provide one --line-colors value per cache directory.')
    for color in line_colors:
        if not matplotlib.colors.is_color_like(color):
            raise ValueError(f'Invalid --line-colors value: {color!r}. Use a matplotlib color name or hex code.')
    return line_colors


def plot_session(names, runs, session, run_aliases=None, null_shading='percentiles', line_colors=None):
    """Metric rows and observed/null columns, with a line per run."""
    labels = names if run_aliases is None else run_aliases
    if len(labels) != len(runs) or any(not label.strip() for label in labels):
        raise ValueError('Provide one nonempty --run-aliases value per cache directory.')
    colors = resolve_line_colors(line_colors, len(runs))
    fig, axes = plt.subplots(
        4, 2, figsize=(14, 15),
        sharex=True, sharey='row', layout='constrained',
    )
    for row, (metric, title) in enumerate(METRICS):
        for ax, kind in zip(axes[row], ('observed', 'null')):
            for index, (name, run) in enumerate(zip(names, runs)):
                result = run[session]
                x = np.asarray(result['time_bins'], dtype=float)
                if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
                    raise ValueError(f'{name}, session {session}: invalid time bins.')
                if np.any(np.diff(x) <= 0):
                    raise ValueError(f'{name}, session {session}: time bins must increase.')
                values, interval = score_curve(result, kind, metric, x.size, null_shading)
                if values.shape != x.shape:
                    raise ValueError(f'{name}, session {session}: {kind} {metric} shape '
                                     'does not match time bins.')
                ax.plot(x, values, linewidth=1.5,
                        label=labels[index], color=colors[index],
                        linestyle=('-', '--', '-.', ':')[(index // 10) % 4])
                if interval is not None:
                    ax.fill_between(x, *interval, color=colors[index],
                                    alpha=0.2, linewidth=0, label='_nolegend_')
                if not np.any(np.isfinite(values)):
                    print(f'{name}, session {session}: {kind} {metric} unavailable')
            if row == 0:
                ax.set_title(kind.capitalize())
            ax.set_ylabel(title)
            ax.grid(axis='y', alpha=0.25)
    axes[0, 0].legend(loc='best')
    for ax in axes[-1]:
        ax.set_xlabel('Time bin start (ms)')
    shading_labels = {
        'confidence_intervals': 'pointwise 95% CI of the null mean',
        'percentiles': '2.5–97.5 percentiles across null shuffles',
        'none': 'none',
    }
    fig.suptitle(f'Session {session} — decoding metrics across runs\n'
                 f'Shading: {shading_labels[null_shading]}')
    return fig


def main(config: Config):
    resolve_line_colors(config.line_colors, len(config.cache_dirs))
    if config.run_aliases is not None and (
        len(config.run_aliases) != len(config.cache_dirs)
        or any(not alias.strip() for alias in config.run_aliases)
    ):
        raise ValueError('Provide one nonempty --run-aliases value per cache directory.')
    names, runs, sessions = load_runs(config.cache_dirs)
    comparison = '_vs_'.join(
        re.sub(r'[^A-Za-z0-9_.-]+', '_', name) for name in names
    )
    output_dirs = [
        Path(cache_dir) / 'eval_confidence_across_runs' / comparison
        for cache_dir in config.cache_dirs
    ]
    output_paths = []
    for session in sessions:
        safe_session = re.sub(r'[^A-Za-z0-9_.-]+', '_', session)
        fig = plot_session(names, runs, session, config.run_aliases, config.null_shading,
                           config.line_colors)
        try:
            for output_dir in output_dirs:
                output_path = output_dir / f'{safe_session}_confidence_scores.png'
                save_figure_png_only(fig, output_path)
                output_paths.append(output_path)
                print(f'Saved {output_path}')
        finally:
            plt.close(fig)
    return output_paths


if __name__ == '__main__':
    main(tyro.cli(Config))
