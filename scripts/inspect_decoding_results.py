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

DEFAULT_Z_THRESHOLD_OFF = 0.842
DEFAULT_Z_THRESHOLD_ON = 1.645


@dataclass
class Config:
    """Configuration for inspecting one cached decoding trial and time bin."""

    session: str
    trial: list[int]  # one value or inclusive first/last zero-based cache rows
    time_bin_start: list[float]  # one value or inclusive first/last requested times
    cache_dir: Path = Path('cache/run_001')
    with_null: bool = False
    with_state: bool = False
    use_decoding_estimates_from_subset_of_repeats: bool = False
    list_of_repeats: list[int] | None = None
    compare_with_repeat_idx: int | None = None
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


def _matching_state_result(results, session: str, cue):
    """Return the state result corresponding to a decoding session and cue."""
    session_matches = [
        result for result in results if str(result.get('session')) == str(session)
    ]
    if cue is not None:
        matches = [result for result in session_matches if result.get('cue') == cue]
    else:
        matches = session_matches
    if not matches:
        available = sorted(
            {
                (str(result.get('session')), str(result.get('cue')))
                for result in results
            }
        )
        raise ValueError(
            f'No on/off-state result found for session {session!r}, cue {cue!r}. '
            f'Available session/cue pairs: {available}'
        )
    if len(matches) > 1:
        raise ValueError(
            f'Found {len(matches)} on/off-state results for session {session!r}, '
            f'cue {cue!r}; the cache must contain one matching result.'
        )
    return matches[0]


def _safe_filename_part(value) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '-', str(value)).strip('-') or 'value'


def _format_number(value) -> str:
    return f'{float(value):g}'


def _format_time_filename_part(value) -> str:
    """Format a time value without losing the sign in a filename."""
    formatted = _format_number(value)
    if formatted.startswith('-'):
        return f'neg{formatted[1:]}'
    return formatted


def _comparison_label(repeat_idx: int) -> str:
    """Return the display label for a comparison repeat."""
    return 'Observed' if repeat_idx == 0 else f'Repeat {repeat_idx}'


def _parse_range(values, name: str):
    if len(values) not in (1, 2):
        raise ValueError(f'{name} must contain one value or two inclusive endpoints.')
    first = values[0]
    last = values[-1]
    if last < first:
        raise ValueError(f'{name} range must be increasing: got {values}.')
    return first, last


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


def _validated_compare_repeat_idx(repeat_count: int, repeat_idx):
    """Validate an optional zero-based comparison repeat index."""
    if repeat_idx is None:
        return None
    if isinstance(repeat_idx, (bool, np.bool_)) or not isinstance(
        repeat_idx, (int, np.integer)
    ):
        raise ValueError('compare_with_repeat_idx must be an integer index or None.')
    repeat_idx = int(repeat_idx)
    if repeat_idx < 0 or repeat_idx >= repeat_count:
        raise IndexError(
            'compare_with_repeat_idx must be between '
            f'0 and {repeat_count - 1}; got {repeat_idx}.'
        )
    return repeat_idx


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
    compare_repeat_idx = _validated_compare_repeat_idx(
        confidence_repeats.shape[1],
        config.compare_with_repeat_idx,
    )
    compare_confidence = (
        None
        if compare_repeat_idx is None
        else confidence_repeats[:, compare_repeat_idx, :]
    )
    compare_accuracy = (
        None
        if compare_repeat_idx is None
        else accuracy_repeats[:, compare_repeat_idx, :]
    )
    if config.use_decoding_estimates_from_subset_of_repeats:
        repeat_indices = _selected_repeat_indices(
            confidence_repeats.shape[1],
            config.list_of_repeats,
        )
        confidence_repeats = confidence_repeats[:, repeat_indices, :]
        accuracy_repeats = accuracy_repeats[:, repeat_indices, :]
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
        compare_repeat_idx,
        compare_confidence,
        compare_accuracy,
    )


def _load_null_values(result, trial_count: int, bin_count: int):
    """Return cached null confidence and per-trial null accuracy values.

    ``decoding_confidence.py`` caches the null confidence for the preferred
    class, but older cache formats do not include null accuracy or the test
    labels.  The inspector only examines preferred-cue test trials, so null
    accuracy can be derived by thresholding the preferred-class confidence at
    0.5.  If a cache already contains ``decoding_accuracy_null``, use it
    instead.
    """
    confidence_null = result.get('decoding_confidence_null')
    if confidence_null is None:
        raise ValueError(
            f'Cached result for session {result.get("session", "unknown")} '
            'does not contain decoding_confidence_null; rerun '
            'decoding_confidence.py with n_decode_shuffle > 0.'
        )
    confidence_null = np.asarray(confidence_null, dtype=float)
    expected_shape = (trial_count, bin_count)
    if confidence_null.ndim != 3 or confidence_null.shape[:2] != expected_shape:
        raise ValueError(
            'decoding_confidence_null must have shape '
            f'(trial, bin, shuffle) with first dimensions {expected_shape}; '
            f'got {confidence_null.shape}.'
        )
    if confidence_null.shape[2] == 0:
        raise ValueError('decoding_confidence_null does not contain any shuffles.')

    accuracy_null = result.get('decoding_accuracy_null')
    if accuracy_null is not None:
        accuracy_null = np.asarray(accuracy_null, dtype=float)
        if accuracy_null.shape != confidence_null.shape:
            raise ValueError(
                'decoding_accuracy_null must have the same shape as '
                'decoding_confidence_null; '
                f'got {accuracy_null.shape} and {confidence_null.shape}.'
            )
    else:
        test_labels = result.get('decoding_test_labels')
        if test_labels is None:
            # The null confidence is the probability of the preferred class;
            # inspect_decoding_results only selects preferred-cue test trials.
            test_labels = np.ones(trial_count, dtype=bool)
        else:
            test_labels = np.asarray(test_labels)
            if test_labels.shape != (trial_count,):
                raise ValueError(
                    'decoding_test_labels must have one label per cached trial; '
                    f'got {test_labels.shape} for {trial_count} trials.'
                )
            test_labels = test_labels.astype(bool)
        null_predicted_preferred = confidence_null >= 0.5
        accuracy_null = np.where(
            np.isfinite(confidence_null),
            null_predicted_preferred == test_labels[:, None, None],
            np.nan,
        )
        accuracy_null = accuracy_null.astype(float)

    return confidence_null, accuracy_null


def _matching_indices(source_values, requested_values, name: str):
    """Return source positions for requested values, requiring unique matches."""
    source_values = np.asarray(source_values)
    requested_values = np.asarray(requested_values)
    positions = []
    for value in requested_values:
        matches = np.flatnonzero(source_values == value)
        if matches.size != 1:
            raise ValueError(
                f'On/off-state cache must contain exactly one {name} matching '
                f'{value!r}; found {matches.size}.'
            )
        positions.append(int(matches[0]))
    return np.asarray(positions, dtype=int)


def _load_state_masks(
    cache_dir: Path,
    decoding_result,
    trial_idx: np.ndarray,
    time_bins: np.ndarray,
):
    """Load on/off masks and align them to the decoding cache coordinates."""
    cache_path = cache_dir / 'on_off_states.pkl'
    if not cache_path.exists():
        raise FileNotFoundError(
            f'Missing on/off-state cache: {cache_path}; run on_off_states.py first.'
        )
    with cache_path.open('rb') as f:
        results = pickle.load(f)
    if not isinstance(results, list):
        raise ValueError('on_off_states.pkl must contain a list of session results.')

    session = str(decoding_result.get('session'))
    state_result = _matching_state_result(
        results,
        session,
        decoding_result.get('cue'),
    )
    required_keys = ('trial_idx', 'time_bins', 'on_state_mask', 'off_state_mask')
    missing_keys = [key for key in required_keys if key not in state_result]
    if missing_keys:
        raise ValueError(
            f'On/off-state result for session {session!r} is missing {missing_keys}; '
            'rerun on_off_states.py.'
        )

    state_trial_idx = np.asarray(state_result['trial_idx']).ravel()
    state_time_bins = np.asarray(state_result['time_bins'], dtype=float).ravel()
    on_state_mask = np.asarray(state_result['on_state_mask'], dtype=bool)
    off_state_mask = np.asarray(state_result['off_state_mask'], dtype=bool)
    expected_shape = (state_trial_idx.size, state_time_bins.size)
    if on_state_mask.shape != expected_shape or off_state_mask.shape != expected_shape:
        raise ValueError(
            'On/off-state masks must both have shape '
            f'{expected_shape}; got {on_state_mask.shape} and {off_state_mask.shape}.'
        )
    if np.any(on_state_mask & off_state_mask):
        raise ValueError('On/off-state cache contains bins marked as both states.')

    trial_positions = _matching_indices(state_trial_idx, trial_idx, 'trial ID')
    bin_positions = _matching_indices(state_time_bins, time_bins, 'time bin')
    selection = np.ix_(trial_positions, bin_positions)
    return on_state_mask[selection], off_state_mask[selection]


def _state_duration_ms(
    state_mask: np.ndarray,
    trial_row: int,
    bin_index: int,
    time_bins: np.ndarray,
) -> float:
    """Return the duration of the contiguous state containing one bin."""
    state_mask = np.asarray(state_mask, dtype=bool)
    time_bins = np.asarray(time_bins, dtype=float)
    if time_bins.size < 2:
        raise ValueError('At least two time bins are required to calculate state duration.')
    time_bin_steps = np.diff(time_bins)
    if (
        not np.all(np.isfinite(time_bin_steps))
        or np.any(time_bin_steps <= 0)
        or not np.allclose(time_bin_steps, time_bin_steps[0])
    ):
        raise ValueError(
            'Time bins must be finite, strictly increasing, and uniformly spaced '
            'to calculate state duration.'
        )
    if not state_mask[trial_row, bin_index]:
        raise ValueError('The requested bin does not belong to this state.')

    trial_mask = state_mask[trial_row]
    state_start = bin_index
    while state_start > 0 and trial_mask[state_start - 1]:
        state_start -= 1
    state_end = bin_index
    while state_end + 1 < trial_mask.size and trial_mask[state_end + 1]:
        state_end += 1
    return float((state_end - state_start + 1) * time_bin_steps[0])


def _state_label(
    is_on_state: bool,
    is_off_state: bool,
    state_duration_ms: float | None = None,
) -> str:
    """Return a display label for one mutually exclusive state assignment."""
    if is_on_state and is_off_state:
        raise ValueError('A time bin cannot belong to both on-state and off-state.')
    if is_on_state:
        state_name = 'on-state'
    elif is_off_state:
        state_name = 'off-state'
    else:
        return 'not in any state'
    if state_duration_ms is None:
        raise ValueError('State duration is required for a bin assigned to a state.')
    if not np.isfinite(state_duration_ms) or state_duration_ms <= 0:
        raise ValueError('State duration must be finite and positive.')
    return f'{state_name} ({_format_number(state_duration_ms)} ms)'


def _repeat_title(label: str, repeat_count: int) -> str:
    """Include the repeat count only when a distribution has multiple repeats."""
    if repeat_count == 1:
        return label
    return f'{label} ({repeat_count} repeats)'


def save_inspection_figure(
    output_dir: Path,
    session: str,
    trial: int,
    time_bin_start: float,
    accuracy_values: np.ndarray,
    confidence_values: np.ndarray,
    null_accuracy_values: np.ndarray | None = None,
    null_confidence_values: np.ndarray | None = None,
    *,
    bin_index: int | None = None,
    compare_repeat_idx: int | None = None,
    compare_accuracy_value: float | None = None,
    compare_confidence_value: float | None = None,
    state_label: str | None = None,
) -> Path:
    """Save side-by-side repeat histograms as a PNG and return its path."""
    if compare_repeat_idx is not None and (
        compare_accuracy_value is None
        or compare_confidence_value is None
        or not np.isfinite(compare_accuracy_value)
        or not np.isfinite(compare_confidence_value)
    ):
        raise ValueError(
            'Finite comparison accuracy and confidence values are required '
            'when compare_repeat_idx is provided.'
        )
    session_part = _safe_filename_part(session)
    time_part = _format_time_filename_part(time_bin_start)
    bin_part = 'unknown' if bin_index is None else f'{int(bin_index):04d}'
    output_path = output_dir / (
        f'decoding_hist_session-{session_part}'
        f'_trial-row-{trial}'
        f'_bin-{bin_part}'
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
    if null_accuracy_values is not None:
        axes[0].hist(
            null_accuracy_values,
            bins=np.array([-0.5, 0.5, 1.5]),
            color='tab:green',
            histtype='step',
            linewidth=1.5,
            label='Null',
        )
    if compare_repeat_idx is not None:
        axes[0].axvline(
            compare_accuracy_value,
            color='lightgray',
            linestyle='--',
            linewidth=1.5,
            label=_comparison_label(compare_repeat_idx),
            zorder=4,
        )
    axes[0].set_xticks([0, 1])
    axes[0].set_xlim(-0.75, 1.75)
    axes[0].set_xlabel('Decoding accuracy')
    axes[0].set_ylabel('Count')
    axes[0].set_title(_repeat_title('Accuracy', accuracy_values.size))

    axes[1].hist(
        confidence_values,
        bins=np.linspace(0.0, 1.0, 21),
        color='tab:orange',
        edgecolor='white',
    )
    if null_confidence_values is not None:
        axes[1].hist(
            null_confidence_values,
            bins=np.linspace(0.0, 1.0, 21),
            color='tab:green',
            histtype='step',
            linewidth=1.5,
            label='Null',
        )
        null_mean = np.mean(null_confidence_values)
        null_std = np.std(null_confidence_values)
        axes[1].axvline(
            null_mean + DEFAULT_Z_THRESHOLD_OFF * null_std,
            color='lightgreen',
            linestyle='--',
            linewidth=1.5,
            label=f'Off-state threshold (+{DEFAULT_Z_THRESHOLD_OFF:g} SD)',
            zorder=4,
        )
        axes[1].axvline(
            null_mean + DEFAULT_Z_THRESHOLD_ON * null_std,
            color='darkgreen',
            linestyle='--',
            linewidth=1.5,
            label=f'On-state threshold (+{DEFAULT_Z_THRESHOLD_ON:g} SD)',
            zorder=4,
        )
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_xlabel('Decoding confidence')
    axes[1].set_ylabel('Count')
    axes[1].set_title(_repeat_title('Confidence', confidence_values.size))
    axes[1].axvline(0.5, color='black', linestyle='--', linewidth=1)
    if compare_repeat_idx is not None:
        axes[1].axvline(
            compare_confidence_value,
            color='lightgray',
            linestyle='--',
            linewidth=1.5,
            label=_comparison_label(compare_repeat_idx),
            zorder=4,
        )
    if null_confidence_values is not None or compare_repeat_idx is not None:
        axes[1].legend(loc='upper left', frameon=False)

    figure_title = (
        f'Session {session}, trial row {trial}, '
        f'time bin start {time_bin_start:g} ms'
    )
    if state_label is not None:
        figure_title += f', {state_label}'
    fig.suptitle(figure_title)
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
        compare_repeat_idx,
        compare_confidence,
        compare_accuracy,
    ) = _load_selected_values(config)
    null_confidence = null_accuracy = None
    if config.with_null:
        null_confidence, null_accuracy = _load_null_values(
            result,
            trial_idx.size,
            time_bins.size,
        )
    on_state_mask = off_state_mask = None
    if config.with_state:
        on_state_mask, off_state_mask = _load_state_masks(
            config.cache_dir,
            result,
            trial_idx,
            time_bins,
        )
    output_dir = config.cache_dir / 'inspect_decoding_results'
    session = str(result.get('session', config.session))
    saved_paths = []
    for trial_row in selected_trial_rows:
        for bin_index in selected_bin_indices:
            confidence_values = confidence_repeats[trial_row, :, bin_index]
            accuracy_values = accuracy_repeats[trial_row, :, bin_index]
            compare_confidence_value = (
                None
                if compare_confidence is None
                else float(compare_confidence[trial_row, bin_index])
            )
            compare_accuracy_value = (
                None
                if compare_accuracy is None
                else float(compare_accuracy[trial_row, bin_index])
            )
            null_confidence_values = (
                None
                if null_confidence is None
                else null_confidence[trial_row, bin_index, :]
            )
            null_accuracy_values = (
                None
                if null_accuracy is None
                else null_accuracy[trial_row, bin_index, :]
            )
            confidence_values = confidence_values[np.isfinite(confidence_values)]
            accuracy_values = accuracy_values[np.isfinite(accuracy_values)]
            if null_confidence_values is not None:
                null_confidence_values = null_confidence_values[
                    np.isfinite(null_confidence_values)
                ]
                null_accuracy_values = null_accuracy_values[
                    np.isfinite(null_accuracy_values)
                ]
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
            if compare_repeat_idx is not None and (
                not np.isfinite(compare_confidence_value)
                or not np.isfinite(compare_accuracy_value)
            ):
                raise ValueError(
                    f'Comparison repeat {compare_repeat_idx} has a non-finite '
                    f'result for trial row {trial_row}, time bin '
                    f'{time_bin_start:g} ms.'
                )
            if config.with_null and null_confidence_values.size == 0:
                raise ValueError(
                    f'Trial row {trial_row}, time bin {time_bin_start:g} ms '
                    'contains no finite null confidence values.'
                )
            if config.with_null and null_accuracy_values.size == 0:
                raise ValueError(
                    f'Trial row {trial_row}, time bin {time_bin_start:g} ms '
                    'contains no finite null accuracy values.'
                )
            state_label = None
            if on_state_mask is not None:
                is_on_state = bool(on_state_mask[trial_row, bin_index])
                is_off_state = bool(off_state_mask[trial_row, bin_index])
                state_duration_ms = None
                if is_on_state or is_off_state:
                    state_duration_ms = _state_duration_ms(
                        on_state_mask if is_on_state else off_state_mask,
                        trial_row,
                        bin_index,
                        time_bins,
                    )
                state_label = _state_label(
                    is_on_state,
                    is_off_state,
                    state_duration_ms,
                )
            saved_paths.append(
                save_inspection_figure(
                    output_dir,
                    session,
                    int(trial_row),
                    time_bin_start,
                    accuracy_values,
                    confidence_values,
                    null_accuracy_values=null_accuracy_values,
                    null_confidence_values=null_confidence_values,
                    bin_index=int(bin_index),
                    compare_repeat_idx=compare_repeat_idx,
                    compare_accuracy_value=compare_accuracy_value,
                    compare_confidence_value=compare_confidence_value,
                    state_label=state_label,
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
