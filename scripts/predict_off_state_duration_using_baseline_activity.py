import csv
import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tyro
from scipy.io import loadmat
from scipy.stats import t as student_t

try:
    from scripts.decoding_confidence import compute_binned_rates
    from scripts.figure_exports import configure_figure_style, save_figure_png_only
except ModuleNotFoundError:
    from decoding_confidence import compute_binned_rates
    from figure_exports import configure_figure_style, save_figure_png_only

configure_figure_style(matplotlib)


BASELINE_START = -400
BASELINE_END = 0
DELAY_START = 500
DELAY_END = 1400
GROUP_NAMES = (
    'preferred',
    'selective_nonpreferred',
    'stationary_nonselective',
)


@dataclass
class Config:
    """Fit activity regressions for every cached session."""

    data_dir: Path = Path('data/nature')
    cache_dir: Path = Path('cache/run_001')
    skip_group_level_norm_before_fit: bool = False
    z_threshold_for_active_cell: float = -0.842
    compare_with_delay: bool = False


def _same_session(left, right) -> bool:
    return str(left) == str(right)


def _load_pickle(path: Path):
    if not path.exists():
        raise FileNotFoundError(f'Missing cache file: {path}')
    with path.open('rb') as f:
        return pickle.load(f)


def _find_full_session_selection(selection_results, session, num_trials):
    matches = [
        result
        for result in selection_results
        if _same_session(result.get('session'), session)
        and result.get('trial_holdout') is None
        and int(result.get('trial_start', -1)) == 0
        and int(result.get('trial_end', -1)) == num_trials
    ]
    if len(matches) != 1:
        raise ValueError(
            f'Expected exactly one full-session selection entry for session '
            f'{session!r}; found {len(matches)}.'
        )
    return matches[0]


def _cell_groups(selection_result, preferred_cue):
    cell_properties = selection_result['cell_properties']
    selective_cells = np.asarray(cell_properties['cell_idx'], dtype=np.int64)
    preferred_cues = np.asarray(cell_properties['mean_pref_test'])
    if preferred_cues.shape != selective_cells.shape:
        raise ValueError('cell_idx and mean_pref_test must have matching shapes.')

    preferred_cells = selective_cells[preferred_cues == preferred_cue]
    selective_nonpreferred = selective_cells[preferred_cues != preferred_cue]
    stationary_cells = np.asarray(
        selection_result['cell_idx_stationary'], dtype=np.int64
    )
    stationary_nonselective = stationary_cells[
        ~np.isin(stationary_cells, selective_cells)
    ]
    return {
        'preferred': preferred_cells,
        'selective_nonpreferred': selective_nonpreferred,
        'stationary_nonselective': stationary_nonselective,
    }


def _standardize_cells(activity_rates, cell_indices):
    """Z-score usable cells across trials and return the normalized values."""
    cell_indices = np.asarray(cell_indices, dtype=np.int64)
    if cell_indices.size == 0:
        return None, 0, 'no_cells'

    values = np.asarray(activity_rates[:, cell_indices], dtype=float)
    finite_cells = np.all(np.isfinite(values), axis=0)
    if not np.any(finite_cells):
        return None, 0, 'no_finite_cells'
    values = values[:, finite_cells]
    cell_mean = np.mean(values, axis=0)
    cell_std = np.std(values, axis=0)
    variable_cells = np.isfinite(cell_std) & (cell_std > 0)
    if not np.any(variable_cells):
        if np.allclose(values, 0.0):
            return None, 0, 'zero_activity'
        return None, 0, 'no_variable_cells'

    standardized = (values[:, variable_cells] - cell_mean[variable_cells]) / cell_std[
        variable_cells
    ]
    return standardized, int(np.sum(variable_cells)), 'ok'


def _standardize_cells_then_average(activity_rates, cell_indices):
    """Z-score each cell across trials, then average the usable cells."""
    standardized, n_cells, status = _standardize_cells(activity_rates, cell_indices)
    if standardized is None:
        return np.zeros(activity_rates.shape[0], dtype=float), n_cells, status
    return np.mean(standardized, axis=1), n_cells, status


def _count_active_cells(activity_rates, cell_indices, z_threshold):
    """Count usable cells whose activity z-score exceeds the threshold."""
    standardized, n_cells, status = _standardize_cells(activity_rates, cell_indices)
    if standardized is None:
        return np.zeros(activity_rates.shape[0], dtype=float), n_cells, status
    active_counts = np.sum(standardized > z_threshold, axis=1, dtype=np.int64)
    return active_counts.astype(float), n_cells, status


def _standardize_across_trials(values):
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError('Cannot standardize values containing non-finite entries.')
    mean = float(np.mean(values))
    std = float(np.std(values))
    if not np.isfinite(std) or std == 0:
        return np.zeros(values.shape, dtype=float), False
    return (values - mean) / std, True


def _fit_multiple_regression(predictors, response):
    """Fit OLS with intercept and return coefficients, p-values, and R-squared."""
    predictors = np.asarray(predictors, dtype=float)
    response = np.asarray(response, dtype=float)
    design = np.column_stack([np.ones(response.size), predictors])
    if not np.all(np.isfinite(design)) or not np.all(np.isfinite(response)):
        raise ValueError('Regression design and response must be finite.')
    coefficients, _, rank, _ = np.linalg.lstsq(design, response, rcond=None)
    if not np.all(np.isfinite(coefficients)):
        raise ValueError('OLS returned non-finite coefficients.')
    try:
        with np.errstate(over='raise', invalid='raise'):
            fitted = np.sum(design * coefficients, axis=1)
    except FloatingPointError as error:
        raise ValueError('OLS fitted values overflowed or became non-finite.') from error
    residual = response - fitted
    residual_sum_squares = float(np.sum(residual**2))
    total_sum_squares = float(np.sum((response - np.mean(response)) ** 2))
    if total_sum_squares == 0:
        raise ValueError('The standardized response has zero variance.')
    r_squared = 1.0 - residual_sum_squares / total_sum_squares

    p_values = np.full(coefficients.shape, np.nan, dtype=float)
    degrees_of_freedom = response.size - rank
    if degrees_of_freedom > 0:
        residual_variance = residual_sum_squares / degrees_of_freedom
        covariance = residual_variance * np.linalg.pinv(design.T @ design)
        standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        estimable = standard_errors > 0
        t_values = np.divide(
            coefficients,
            standard_errors,
            out=np.full(coefficients.shape, np.nan),
            where=estimable,
        )
        p_values[estimable] = 2 * student_t.sf(
            np.abs(t_values[estimable]), degrees_of_freedom
        )

    return coefficients, p_values, float(r_squared), int(rank)


def _significance_stars(p_value):
    if not np.isfinite(p_value):
        return ''
    if p_value < 0.001:
        return '***'
    if p_value < 0.01:
        return '**'
    if p_value < 0.05:
        return '*'
    return ''


def _save_scatter_plot(
    output_dir,
    session,
    predictors,
    response,
    coefficients,
    p_values,
    r_squared,
    group_level_norm_applied,
    *,
    activity_label='baseline',
):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained')
    activity_display = activity_label.title()
    if group_level_norm_applied:
        x_label = f'{activity_display} activity (z-score)'
        normalization_title = 'group-level normalized'
    else:
        x_label = f'{activity_display} activity (mean cell z-score)'
        normalization_title = 'group-level normalization skipped'
    y_label = 'Off-state duration (ms)'
    for index, group_name in enumerate(GROUP_NAMES):
        x = predictors[:, index]
        ax = axes[index]
        ax.scatter(x, response, color=f'C{index}', alpha=0.7, s=24)
        if np.ptp(x) > 0:
            x_line = np.linspace(np.min(x), np.max(x), 100)
            marginal_coefficients = np.polyfit(x, response, 1)
            ax.plot(
                x_line,
                marginal_coefficients[0] * x_line + marginal_coefficients[1],
                color='black',
                linewidth=1.5,
                label='Marginal fit',
            )
            ax.legend(frameon=False)
        else:
            ax.text(
                0.05,
                0.95,
                'Constant predictor',
                transform=ax.transAxes,
                va='top',
            )
        ax.axhline(0, color='0.8', linewidth=1)
        ax.set_title(group_name.replace('_', ' ').title())
        ax.set_xlabel(x_label)
        if index == 0:
            ax.set_ylabel(y_label)
        ax.text(
            0.05,
            0.05,
            f'β = {coefficients[index + 1]:.3g}'
            f'{_significance_stars(p_values[index + 1])}\n'
            f'R² = {r_squared:.3g}',
            transform=ax.transAxes,
            va='bottom',
        )

    fig.suptitle(
        f'Session {session}: {activity_label} activity and off-state duration '
        f'({normalization_title})'
    )
    output_path = output_dir / (
        f'off_state_duration_vs_{activity_label}_activity_session_{session}.png'
    )
    save_figure_png_only(fig, output_path, dpi=300)
    plt.close(fig)
    return output_path


def _save_active_cell_count_scatter_plot(
    output_dir,
    session,
    predictors,
    response,
    coefficients,
    p_values,
    r_squared,
    z_threshold_for_active_cell,
    *,
    title_label=None,
    filename_label=None,
    activity_label='baseline',
):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained')
    x_label = (
        f'Active {activity_label} cells '
        f'(cell z-score > {z_threshold_for_active_cell:g})'
    )
    y_label = 'Off-state duration (ms)'
    for index, group_name in enumerate(GROUP_NAMES):
        x = predictors[:, index]
        ax = axes[index]
        ax.scatter(x, response, color=f'C{index}', alpha=0.7, s=24)
        if np.ptp(x) > 0:
            x_line = np.linspace(np.min(x), np.max(x), 100)
            marginal_coefficients = np.polyfit(x, response, 1)
            ax.plot(
                x_line,
                marginal_coefficients[0] * x_line + marginal_coefficients[1],
                color='black',
                linewidth=1.5,
                label='Marginal fit',
            )
            ax.legend(frameon=False)
        else:
            ax.text(
                0.05,
                0.95,
                'Constant predictor',
                transform=ax.transAxes,
                va='top',
            )
        ax.axhline(0, color='0.8', linewidth=1)
        ax.set_title(group_name.replace('_', ' ').title())
        ax.set_xlabel(x_label)
        if index == 0:
            ax.set_ylabel(y_label)
        ax.text(
            0.05,
            0.05,
            f'β = {coefficients[index + 1]:.3g}'
            f'{_significance_stars(p_values[index + 1])}\n'
            f'R² = {r_squared:.3g}',
            transform=ax.transAxes,
            va='bottom',
        )

    title_label = title_label or f'Session {session}'
    filename_label = filename_label or f'session_{session}'
    fig.suptitle(
        f'{title_label}: {activity_label} active cell count and off-state duration'
    )
    output_path = output_dir / (
        f'off_state_duration_vs_{activity_label}_active_cell_count_{filename_label}.png'
    )
    save_figure_png_only(fig, output_path, dpi=300)
    plt.close(fig)
    return output_path


def _save_active_inactive_cell_count_scatter_plot(
    output_dir,
    predictors,
    response,
    coefficients,
    p_values,
    r_squared,
    z_threshold_for_active_cell,
    *,
    title_label,
    filename_label,
    activity_label='baseline',
):
    """Save six-IV active/inactive cell-count plots in two rows."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), layout='constrained')
    y_label = 'Off-state duration (ms)'
    row_labels = (
        (
            'Active cells',
            f'Active {activity_label} cells '
            f'(cell z-score > {z_threshold_for_active_cell:g})',
        ),
        ('Inactive cells', f'Inactive usable {activity_label} cells'),
    )
    for row_index, (row_label, x_label) in enumerate(row_labels):
        for column_index, group_name in enumerate(GROUP_NAMES):
            predictor_index = row_index * len(GROUP_NAMES) + column_index
            x = predictors[:, predictor_index]
            ax = axes[row_index, column_index]
            ax.scatter(x, response, color=f'C{column_index}', alpha=0.7, s=24)
            if np.ptp(x) > 0:
                x_line = np.linspace(np.min(x), np.max(x), 100)
                marginal_coefficients = np.polyfit(x, response, 1)
                ax.plot(
                    x_line,
                    marginal_coefficients[0] * x_line + marginal_coefficients[1],
                    color='black',
                    linewidth=1.5,
                    label='Marginal fit',
                )
                ax.legend(frameon=False)
            else:
                ax.text(
                    0.05,
                    0.95,
                    'Constant predictor',
                    transform=ax.transAxes,
                    va='top',
                )
            ax.axhline(0, color='0.8', linewidth=1)
            ax.set_title(f'{row_label}: {group_name.replace("_", " ").title()}')
            ax.set_xlabel(x_label)
            if column_index == 0:
                ax.set_ylabel(y_label)
            ax.text(
                0.05,
                0.05,
                f'β = {coefficients[predictor_index + 1]:.3g}'
                f'{_significance_stars(p_values[predictor_index + 1])}\n'
                f'R² = {r_squared:.3g}',
                transform=ax.transAxes,
                va='bottom',
            )

    fig.suptitle(
        f'{title_label}: {activity_label} active/inactive cell counts and '
        'off-state duration'
    )
    output_path = output_dir / (
        f'off_state_duration_vs_{activity_label}_active_inactive_cell_count_'
        f'{filename_label}.png'
    )
    save_figure_png_only(fig, output_path, dpi=300)
    plt.close(fig)
    return output_path


def _empty_row(session, description):
    return {
        'session': session,
        'description': description,
        'n_trials': '',
        'n_preferred_cells': '',
        'n_selective_nonpreferred_cells': '',
        'n_stationary_nonselective_cells': '',
        'intercept': '',
        'coefficient_preferred': '',
        'coefficient_selective_nonpreferred': '',
        'coefficient_stationary_nonselective': '',
        'p_value_intercept': '',
        'p_value_preferred': '',
        'p_value_selective_nonpreferred': '',
        'p_value_stationary_nonselective': '',
        'r_squared': '',
    }


def _empty_active_cell_count_row(session, description):
    return {
        'session': session,
        'description': description,
        'n_trials': '',
        'z_threshold_for_active_cell': '',
        'n_usable_preferred_cells': '',
        'n_usable_selective_nonpreferred_cells': '',
        'n_usable_stationary_nonselective_cells': '',
        'intercept': '',
        'coefficient_preferred': '',
        'coefficient_selective_nonpreferred': '',
        'coefficient_stationary_nonselective': '',
        'p_value_intercept': '',
        'p_value_preferred': '',
        'p_value_selective_nonpreferred': '',
        'p_value_stationary_nonselective': '',
        'r_squared': '',
    }


def _empty_pooled_active_cell_count_row(description):
    return {
        'session': 'all_sessions',
        'description': description,
        'n_sessions': '',
        'n_trials': '',
        'z_threshold_for_active_cell': '',
        'intercept': '',
        'coefficient_preferred': '',
        'coefficient_selective_nonpreferred': '',
        'coefficient_stationary_nonselective': '',
        'p_value_intercept': '',
        'p_value_preferred': '',
        'p_value_selective_nonpreferred': '',
        'p_value_stationary_nonselective': '',
        'r_squared': '',
    }


def _empty_pooled_active_inactive_cell_count_row(description):
    return {
        'session': 'all_sessions',
        'description': description,
        'n_sessions': '',
        'n_trials': '',
        'z_threshold_for_active_cell': '',
        'intercept': '',
        'coefficient_active_preferred': '',
        'coefficient_active_selective_nonpreferred': '',
        'coefficient_active_stationary_nonselective': '',
        'coefficient_inactive_preferred': '',
        'coefficient_inactive_selective_nonpreferred': '',
        'coefficient_inactive_stationary_nonselective': '',
        'p_value_intercept': '',
        'p_value_active_preferred': '',
        'p_value_active_selective_nonpreferred': '',
        'p_value_active_stationary_nonselective': '',
        'p_value_inactive_preferred': '',
        'p_value_inactive_selective_nonpreferred': '',
        'p_value_inactive_stationary_nonselective': '',
        'r_squared': '',
    }


def _run_activity_regressions(
    config: Config,
    *,
    activity_label,
    activity_start,
    activity_end,
):
    z_threshold_for_active_cell = float(config.z_threshold_for_active_cell)
    if not np.isfinite(z_threshold_for_active_cell):
        raise ValueError('z_threshold_for_active_cell must be finite.')
    if activity_end <= activity_start:
        raise ValueError('activity_end must be greater than activity_start.')

    decoding_results = _load_pickle(config.cache_dir / 'decoding_confidence.pkl')
    off_state_results = _load_pickle(config.cache_dir / 'on_off_states.pkl')
    selection_results = _load_pickle(config.cache_dir / 'cell_trial_selection.pkl')
    group_level_norm_applied = not config.skip_group_level_norm_before_fit
    output_dir = (
        config.cache_dir
        / f'predict_off_state_duration_using_{activity_label}_activity'
    )
    active_cell_count_output_dir = (
        config.cache_dir
        / f'predict_off_state_duration_using_{activity_label}_active_cell_count'
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    active_cell_count_output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    active_cell_count_rows = []
    pooled_active_cell_count_predictors = []
    pooled_active_inactive_cell_count_predictors = []
    pooled_responses = []
    pooled_sessions = set()
    pooled_active_inactive_sessions = set()
    for off_state_result in off_state_results:
        session = off_state_result.get('session', 'unknown_session')
        row = _empty_row(session, '')
        active_cell_count_row = _empty_active_cell_count_row(session, '')
        try:
            if off_state_result.get('off_state_duration_correction') != 'applied':
                raise ValueError('The off-state cache does not contain CC-applied durations.')

            decoding_matches = [
                result
                for result in decoding_results
                if _same_session(result.get('session'), session)
            ]
            if len(decoding_matches) != 1:
                raise ValueError(
                    f'Expected exactly one decoding result for session; found '
                    f'{len(decoding_matches)}.'
                )
            decoding_result = decoding_matches[0]
            data_path = config.data_dir / f'{session}.mat'
            if not data_path.exists():
                raise FileNotFoundError(f'Missing data file: {data_path}')
            data = loadmat(data_path)
            spikes = np.asarray(data['spks'])
            t = np.asarray(data['tc']).flatten()
            num_trials = spikes.shape[0]
            selection_result = _find_full_session_selection(
                selection_results,
                session,
                num_trials,
            )

            trial_idx = np.asarray(off_state_result['trial_idx'], dtype=np.int64)
            off_duration = np.asarray(
                off_state_result['off_state_duration_per_trial'], dtype=float
            )
            if trial_idx.ndim != 1 or off_duration.shape != trial_idx.shape:
                raise ValueError('Off-state trial_idx and duration arrays must align.')
            if np.any(trial_idx < 0) or np.any(trial_idx >= num_trials):
                raise ValueError('Off-state trial_idx contains out-of-range trials.')

            decoding_trial_idx = np.asarray(
                decoding_result.get('trial_idx', []), dtype=np.int64
            )
            if not np.array_equal(trial_idx, decoding_trial_idx):
                raise ValueError('Off-state and decoding trial_idx arrays do not match.')
            preferred_cue = int(decoding_result['cue'])
            groups = _cell_groups(selection_result, preferred_cue)

            activity_rates = compute_binned_rates(
                spikes,
                t,
                np.asarray([activity_start]),
                activity_end - activity_start,
            )[:, 0, :]
            activity_trial_rates = activity_rates[trial_idx]
            activity_group_values = []
            active_cell_count_values = []
            group_status = []
            active_group_status = []
            group_cell_counts = []
            active_group_cell_counts = []
            for group_name in GROUP_NAMES:
                group_values, n_cells, status = _standardize_cells_then_average(
                    activity_trial_rates,
                    groups[group_name],
                )
                active_counts, active_n_cells, active_status = _count_active_cells(
                    activity_trial_rates,
                    groups[group_name],
                    z_threshold_for_active_cell,
                )
                if group_level_norm_applied:
                    group_values, group_is_variable = _standardize_across_trials(group_values)
                    if not group_is_variable and status == 'ok':
                        status = 'zero_group_variance'
                activity_group_values.append(group_values)
                active_cell_count_values.append(active_counts)
                group_status.append(f'{group_name}:{status}')
                active_group_status.append(f'{group_name}:{active_status}')
                group_cell_counts.append(n_cells)
                active_group_cell_counts.append(active_n_cells)

            predictors = np.column_stack(activity_group_values)
            active_cell_count_predictors = np.column_stack(active_cell_count_values)
            usable_cell_counts = np.asarray(active_group_cell_counts, dtype=float)
            inactive_cell_count_predictors = (
                usable_cell_counts[None, :] - active_cell_count_predictors
            )
            if np.any(inactive_cell_count_predictors < 0):
                raise ValueError('Inactive cell counts must be non-negative.')
            response = off_duration
            valid_pooled_rows = np.isfinite(response) & np.all(
                np.isfinite(active_cell_count_predictors), axis=1
            )
            if np.any(valid_pooled_rows):
                pooled_active_cell_count_predictors.append(
                    active_cell_count_predictors[valid_pooled_rows]
                )
                pooled_responses.append(response[valid_pooled_rows])
                pooled_sessions.add(str(session))
                pooled_active_inactive_cell_count_predictors.append(
                    np.column_stack((
                        active_cell_count_predictors[valid_pooled_rows],
                        inactive_cell_count_predictors[valid_pooled_rows],
                    ))
                )
                pooled_active_inactive_sessions.add(str(session))
            active_cell_count_row.update({
                'n_trials': int(trial_idx.size),
                'z_threshold_for_active_cell': z_threshold_for_active_cell,
                'n_usable_preferred_cells': active_group_cell_counts[0],
                'n_usable_selective_nonpreferred_cells': active_group_cell_counts[1],
                'n_usable_stationary_nonselective_cells': active_group_cell_counts[2],
            })

            try:
                response_is_variable = np.ptp(response) > 0
                if not np.all(np.isfinite(response)):
                    raise ValueError('Off-state duration contains non-finite values.')
                if not response_is_variable:
                    raise ValueError('Off-state duration has zero variance across trials.')
                if not np.all(np.isfinite(predictors)):
                    raise ValueError('Activity predictors contain non-finite values.')

                coefficients, p_values, r_squared, rank = _fit_multiple_regression(
                    predictors,
                    response,
                )
                plot_path = _save_scatter_plot(
                    output_dir,
                    session,
                    predictors,
                    response,
                    coefficients,
                    p_values,
                    r_squared,
                    group_level_norm_applied,
                    activity_label=activity_label,
                )
                row.update({
                    'description': (
                        f'ok; rank={rank}; '
                        f'group_level_norm={"applied" if group_level_norm_applied else "skipped"}; '
                        'dv=raw_ms; '
                        + ', '.join(group_status)
                        + f'; plot={plot_path.name}'
                    ),
                    'n_trials': int(trial_idx.size),
                    'n_preferred_cells': group_cell_counts[0],
                    'n_selective_nonpreferred_cells': group_cell_counts[1],
                    'n_stationary_nonselective_cells': group_cell_counts[2],
                    'intercept': coefficients[0],
                    'coefficient_preferred': coefficients[1],
                    'coefficient_selective_nonpreferred': coefficients[2],
                    'coefficient_stationary_nonselective': coefficients[3],
                    'p_value_intercept': p_values[0],
                    'p_value_preferred': p_values[1],
                    'p_value_selective_nonpreferred': p_values[2],
                    'p_value_stationary_nonselective': p_values[3],
                    'r_squared': r_squared,
                })
            except Exception as error:
                row['description'] = f'skipped: {type(error).__name__}: {error}'

            try:
                if not np.all(np.isfinite(response)):
                    raise ValueError('Off-state duration contains non-finite values.')
                if not np.all(np.isfinite(active_cell_count_predictors)):
                    raise ValueError(
                        'Active-cell-count predictors contain non-finite values.'
                    )

                active_coefficients, active_p_values, active_r_squared, active_rank = (
                    _fit_multiple_regression(
                        active_cell_count_predictors,
                        response,
                    )
                )
                active_plot_path = _save_active_cell_count_scatter_plot(
                    active_cell_count_output_dir,
                    session,
                    active_cell_count_predictors,
                    response,
                    active_coefficients,
                    active_p_values,
                    active_r_squared,
                    z_threshold_for_active_cell,
                    activity_label=activity_label,
                )
                active_cell_count_row.update({
                    'description': (
                        f'ok; rank={active_rank}; '
                        f'z_threshold_for_active_cell={z_threshold_for_active_cell:g}; '
                        'iv=raw_active_cell_count; dv=raw_ms; '
                        + ', '.join(active_group_status)
                        + f'; plot={active_plot_path.name}'
                    ),
                    'intercept': active_coefficients[0],
                    'coefficient_preferred': active_coefficients[1],
                    'coefficient_selective_nonpreferred': active_coefficients[2],
                    'coefficient_stationary_nonselective': active_coefficients[3],
                    'p_value_intercept': active_p_values[0],
                    'p_value_preferred': active_p_values[1],
                    'p_value_selective_nonpreferred': active_p_values[2],
                    'p_value_stationary_nonselective': active_p_values[3],
                    'r_squared': active_r_squared,
                })
            except Exception as error:
                active_cell_count_row['description'] = (
                    f'skipped: {type(error).__name__}: {error}'
                )
        except Exception as error:
            row['description'] = f'skipped: {type(error).__name__}: {error}'
            active_cell_count_row['description'] = (
                f'skipped: {type(error).__name__}: {error}'
            )
        rows.append(row)
        active_cell_count_rows.append(active_cell_count_row)

    csv_path = output_dir / f'off_state_duration_{activity_label}_regression.csv'
    fieldnames = list(rows[0]) if rows else list(_empty_row('', ''))
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved regression statistics to {csv_path}')
    print(f'Saved session scatter plots to {output_dir}')

    active_cell_count_csv_path = (
        active_cell_count_output_dir
        / f'off_state_duration_{activity_label}_active_cell_count_regression.csv'
    )
    active_cell_count_fieldnames = (
        list(active_cell_count_rows[0])
        if active_cell_count_rows
        else list(_empty_active_cell_count_row('', ''))
    )
    with active_cell_count_csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=active_cell_count_fieldnames)
        writer.writeheader()
        writer.writerows(active_cell_count_rows)
    print(f'Saved active-cell-count regression statistics to {active_cell_count_csv_path}')
    print(
        'Saved active-cell-count regression scatter plots to '
        f'{active_cell_count_output_dir}'
    )

    pooled_row = _empty_pooled_active_cell_count_row('')
    if pooled_active_cell_count_predictors:
        pooled_predictors = np.concatenate(
            pooled_active_cell_count_predictors,
            axis=0,
        )
        pooled_response = np.concatenate(pooled_responses, axis=0)
        try:
            pooled_coefficients, pooled_p_values, pooled_r_squared, pooled_rank = (
                _fit_multiple_regression(
                    pooled_predictors,
                    pooled_response,
                )
            )
            pooled_plot_path = _save_active_cell_count_scatter_plot(
                active_cell_count_output_dir,
                'all_sessions',
                pooled_predictors,
                pooled_response,
                pooled_coefficients,
                pooled_p_values,
                pooled_r_squared,
                z_threshold_for_active_cell,
                title_label='All sessions',
                filename_label='all_sessions',
                activity_label=activity_label,
            )
            pooled_row.update({
                'description': (
                    f'ok; rank={pooled_rank}; '
                    f'z_threshold_for_active_cell={z_threshold_for_active_cell:g}; '
                    'iv=raw_active_cell_count; dv=raw_ms; '
                    'session_effects=none; '
                    f'plot={pooled_plot_path.name}'
                ),
                'n_sessions': int(len(pooled_sessions)),
                'n_trials': int(pooled_response.size),
                'z_threshold_for_active_cell': z_threshold_for_active_cell,
                'intercept': pooled_coefficients[0],
                'coefficient_preferred': pooled_coefficients[1],
                'coefficient_selective_nonpreferred': pooled_coefficients[2],
                'coefficient_stationary_nonselective': pooled_coefficients[3],
                'p_value_intercept': pooled_p_values[0],
                'p_value_preferred': pooled_p_values[1],
                'p_value_selective_nonpreferred': pooled_p_values[2],
                'p_value_stationary_nonselective': pooled_p_values[3],
                'r_squared': pooled_r_squared,
            })
        except Exception as error:
            pooled_row['description'] = (
                f'skipped: {type(error).__name__}: {error}'
            )
    else:
        pooled_row['description'] = 'skipped: no valid trial rows to pool'

    pooled_csv_path = (
        active_cell_count_output_dir
        / f'off_state_duration_{activity_label}_active_cell_count_pooled_regression.csv'
    )
    pooled_fieldnames = list(pooled_row)
    with pooled_csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=pooled_fieldnames)
        writer.writeheader()
        writer.writerow(pooled_row)
    print(f'Saved pooled active-cell-count regression to {pooled_csv_path}')

    pooled_active_inactive_row = _empty_pooled_active_inactive_cell_count_row('')
    if pooled_active_inactive_cell_count_predictors:
        pooled_active_inactive_predictors = np.concatenate(
            pooled_active_inactive_cell_count_predictors,
            axis=0,
        )
        pooled_active_inactive_response = np.concatenate(pooled_responses, axis=0)
        try:
            (
                pooled_active_inactive_coefficients,
                pooled_active_inactive_p_values,
                pooled_active_inactive_r_squared,
                pooled_active_inactive_rank,
            ) = _fit_multiple_regression(
                pooled_active_inactive_predictors,
                pooled_active_inactive_response,
            )
            pooled_active_inactive_plot_path = (
                _save_active_inactive_cell_count_scatter_plot(
                    active_cell_count_output_dir,
                    pooled_active_inactive_predictors,
                    pooled_active_inactive_response,
                    pooled_active_inactive_coefficients,
                    pooled_active_inactive_p_values,
                    pooled_active_inactive_r_squared,
                    z_threshold_for_active_cell,
                    title_label='All sessions',
                    filename_label='all_sessions',
                    activity_label=activity_label,
                )
            )
            pooled_active_inactive_row.update({
                'description': (
                    f'ok; rank={pooled_active_inactive_rank}; '
                    f'z_threshold_for_active_cell={z_threshold_for_active_cell:g}; '
                    'iv=raw_active_and_inactive_cell_counts; dv=raw_ms; '
                    'intercept=common; session_effects=none; '
                    f'plot={pooled_active_inactive_plot_path.name}'
                ),
                'n_sessions': int(len(pooled_active_inactive_sessions)),
                'n_trials': int(pooled_active_inactive_response.size),
                'z_threshold_for_active_cell': z_threshold_for_active_cell,
                'intercept': pooled_active_inactive_coefficients[0],
                'coefficient_active_preferred': pooled_active_inactive_coefficients[1],
                'coefficient_active_selective_nonpreferred': (
                    pooled_active_inactive_coefficients[2]
                ),
                'coefficient_active_stationary_nonselective': (
                    pooled_active_inactive_coefficients[3]
                ),
                'coefficient_inactive_preferred': pooled_active_inactive_coefficients[4],
                'coefficient_inactive_selective_nonpreferred': (
                    pooled_active_inactive_coefficients[5]
                ),
                'coefficient_inactive_stationary_nonselective': (
                    pooled_active_inactive_coefficients[6]
                ),
                'p_value_intercept': pooled_active_inactive_p_values[0],
                'p_value_active_preferred': pooled_active_inactive_p_values[1],
                'p_value_active_selective_nonpreferred': (
                    pooled_active_inactive_p_values[2]
                ),
                'p_value_active_stationary_nonselective': (
                    pooled_active_inactive_p_values[3]
                ),
                'p_value_inactive_preferred': pooled_active_inactive_p_values[4],
                'p_value_inactive_selective_nonpreferred': (
                    pooled_active_inactive_p_values[5]
                ),
                'p_value_inactive_stationary_nonselective': (
                    pooled_active_inactive_p_values[6]
                ),
                'r_squared': pooled_active_inactive_r_squared,
            })
        except Exception as error:
            pooled_active_inactive_row['description'] = (
                f'skipped: {type(error).__name__}: {error}'
            )
    else:
        pooled_active_inactive_row['description'] = (
            'skipped: no valid trial rows to pool'
        )

    pooled_active_inactive_csv_path = (
        active_cell_count_output_dir
        / f'off_state_duration_{activity_label}_active_inactive_cell_count_pooled_regression.csv'
    )
    pooled_active_inactive_fieldnames = list(pooled_active_inactive_row)
    with pooled_active_inactive_csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=pooled_active_inactive_fieldnames,
        )
        writer.writeheader()
        writer.writerow(pooled_active_inactive_row)
    print(
        'Saved pooled active/inactive-cell-count regression to '
        f'{pooled_active_inactive_csv_path}'
    )


def main(config: Config):
    _run_activity_regressions(
        config,
        activity_label='baseline',
        activity_start=BASELINE_START,
        activity_end=BASELINE_END,
    )
    if config.compare_with_delay:
        _run_activity_regressions(
            config,
            activity_label='delay',
            activity_start=DELAY_START,
            activity_end=DELAY_END,
        )


if __name__ == '__main__':
    main(tyro.cli(Config))
