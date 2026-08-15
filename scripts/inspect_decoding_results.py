import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tyro

try:
    from scripts.figure_exports import configure_figure_style, save_figure_png_only
except ModuleNotFoundError:
    from figure_exports import configure_figure_style, save_figure_png_only

configure_figure_style(matplotlib)


@dataclass
class Config:
    """Configuration for inspecting one cached decoding trial and time bin."""

    session: str
    trial: list[int]  # one value or inclusive first/last zero-based cache rows
    time_bin_start: list[float]  # one value or inclusive first/last requested times
    cache_dir: Path = Path('cache/run_001')
    verbose: int = 1  # 0: no output, 1: summary, 2: summary and each figure


def _matching_result(results, session: str):
    matches = [result for result in results if str(result.get('session')) == str(session)]
    if not matches:
        available = sorted({str(result.get('session')) for result in results})
        raise ValueError(
            f'No decoding result found for session {session!r}. '
            f'Available sessions: {available}'
        )
    if len(matches) > 1:
        raise ValueError(
            f'Found {len(matches)} decoding results for session {session!r}; '
            'the cache must contain one result per session for inspection.'
        )
    return matches[0]


def _safe_filename_part(value) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '-', str(value)).strip('-') or 'value'


def _format_number(value) -> str:
    return f'{float(value):g}'


def _parse_range(values, name: str):
    if len(values) not in (1, 2):
        raise ValueError(f'{name} must contain one value or two inclusive endpoints.')
    first = values[0]
    last = values[-1]
    if last < first:
        raise ValueError(f'{name} range must be increasing: got {values}.')
    return first, last


def _load_selected_values(config: Config):
    cache_path = config.cache_dir / 'decoding_confidence.pkl'
    if not cache_path.exists():
        raise FileNotFoundError(f'Missing decoding cache: {cache_path}')

    with cache_path.open('rb') as f:
        results = pickle.load(f)
    result = _matching_result(results, config.session)

    required_keys = (
        'trial_idx',
        'time_bins',
        'decoding_confidence_repeats',
        'decoding_accuracy_per_trial_repeats',
    )
    missing_keys = [key for key in required_keys if key not in result]
    if missing_keys:
        raise ValueError(
            f'Cached result for session {config.session!r} is missing '
            f'{missing_keys}; rerun decoding_confidence.py with the repeat-format cache.'
        )

    trial_idx = np.asarray(result['trial_idx'])
    time_bins = np.asarray(result['time_bins'], dtype=float)
    confidence_repeats = np.asarray(result['decoding_confidence_repeats'])
    accuracy_repeats = np.asarray(result['decoding_accuracy_per_trial_repeats'])

    if trial_idx.ndim != 1:
        raise ValueError('Cached trial_idx must be one-dimensional.')
    if time_bins.ndim != 1 or time_bins.size == 0:
        raise ValueError('Cached time_bins must be a non-empty one-dimensional array.')
    if confidence_repeats.ndim != 3:
        raise ValueError(
            'decoding_confidence_repeats must have shape (trial, repeat, bin).'
        )
    if accuracy_repeats.shape != confidence_repeats.shape:
        raise ValueError(
            'decoding_accuracy_per_trial_repeats and '
            'decoding_confidence_repeats must have the same shape.'
        )
    if confidence_repeats.shape[0] != trial_idx.size:
        raise ValueError(
            'The trial dimension of decoding_confidence_repeats does not match trial_idx.'
        )
    if confidence_repeats.shape[2] != time_bins.size:
        raise ValueError(
            'The bin dimension of decoding_confidence_repeats does not match time_bins.'
        )
    trial_first, trial_last = _parse_range(config.trial, 'trial')
    if trial_first < 0 or trial_last >= trial_idx.size:
        raise IndexError(
            f'trial must be between 0 and {trial_idx.size - 1}; '
            f'got {config.trial}.'
        )
    selected_trial_rows = np.arange(trial_first, trial_last + 1, dtype=int)

    time_first, time_last = _parse_range(config.time_bin_start, 'time_bin_start')
    if not np.isfinite(time_first) or not np.isfinite(time_last):
        raise ValueError('time_bin_start values must be finite.')
    first_bin_index = int(np.argmin(np.abs(time_bins - time_first)))
    last_bin_index = int(np.argmin(np.abs(time_bins - time_last)))
    if last_bin_index < first_bin_index:
        raise ValueError(
            'The nearest time-bin range is decreasing; check time_bin_start values.'
        )
    selected_bin_indices = np.arange(first_bin_index, last_bin_index + 1, dtype=int)

    return (
        result,
        trial_idx,
        selected_trial_rows,
        selected_bin_indices,
        time_bins,
        confidence_repeats,
        accuracy_repeats,
    )


def save_inspection_figure(
    output_dir: Path,
    session: str,
    trial: int,
    time_bin_start: float,
    accuracy_values: np.ndarray,
    confidence_values: np.ndarray,
) -> Path:
    """Save side-by-side repeat histograms as a PNG and return its path."""
    session_part = _safe_filename_part(session)
    time_part = _safe_filename_part(_format_number(time_bin_start))
    output_path = output_dir / (
        f'decoding_hist_session-{session_part}'
        f'_trial-row-{trial}'
        f'_time-bin-start-{time_part}.png'
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9, 4),
        gridspec_kw={'width_ratios': (1, 3)},
        layout='constrained',
    )
    axes[0].hist(
        accuracy_values,
        bins=np.array([-0.5, 0.5, 1.5]),
        color='tab:blue',
        edgecolor='white',
        rwidth=0.8,
    )
    axes[0].set_xticks([0, 1])
    axes[0].set_xlim(-0.75, 1.75)
    axes[0].set_xlabel('Decoding accuracy')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'Accuracy ({accuracy_values.size} repeats)')

    axes[1].hist(
        confidence_values,
        bins=np.linspace(0.0, 1.0, 21),
        color='tab:orange',
        edgecolor='white',
    )
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_xlabel('Decoding confidence')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'Confidence ({confidence_values.size} repeats)')
    axes[1].axvline(0.5, color='black', linestyle='--', linewidth=1)

    fig.suptitle(
        f'Session {session}, trial row {trial}, '
        f'time bin start {time_bin_start:g} ms'
    )
    save_figure_png_only(fig, output_path, dpi=300)
    plt.close(fig)
    return output_path


def main(config: Config):
    if config.verbose not in (0, 1, 2):
        raise ValueError('verbose must be 0, 1, or 2.')
    (
        result,
        trial_idx,
        selected_trial_rows,
        selected_bin_indices,
        time_bins,
        confidence_repeats,
        accuracy_repeats,
    ) = _load_selected_values(config)
    output_dir = config.cache_dir / 'inspect_decoding_results'
    session = str(result.get('session', config.session))
    saved_paths = []
    for trial_row in selected_trial_rows:
        for bin_index in selected_bin_indices:
            confidence_values = confidence_repeats[trial_row, :, bin_index]
            accuracy_values = accuracy_repeats[trial_row, :, bin_index]
            confidence_values = confidence_values[np.isfinite(confidence_values)]
            accuracy_values = accuracy_values[np.isfinite(accuracy_values)]
            time_bin_start = time_bins[bin_index]
            if confidence_values.size == 0:
                raise ValueError(
                    f'Trial row {trial_row}, time bin {time_bin_start:g} ms '
                    'contains no finite confidence repeats.'
                )
            if accuracy_values.size == 0:
                raise ValueError(
                    f'Trial row {trial_row}, time bin {time_bin_start:g} ms '
                    'contains no finite accuracy repeats.'
                )
            saved_paths.append(
                save_inspection_figure(
                    output_dir,
                    session,
                    int(trial_row),
                    time_bin_start,
                    accuracy_values,
                    confidence_values,
                )
            )
    if config.verbose >= 1:
        print(f'Saved {len(saved_paths)} inspection figures to {output_dir}')
    if config.verbose >= 2:
        for path, trial_row, bin_index in zip(
            saved_paths,
            np.repeat(selected_trial_rows, selected_bin_indices.size),
            np.tile(selected_bin_indices, selected_trial_rows.size),
        ):
            print(
                f'  {path} '
                f'(cached trial id: {trial_idx[trial_row]}, '
                f'time bin: {time_bins[bin_index]:g} ms)'
            )


if __name__ == '__main__':
    main(tyro.cli(Config))
