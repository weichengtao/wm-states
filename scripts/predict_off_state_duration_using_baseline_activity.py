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
GROUP_NAMES = (
    'preferred',
    'selective_nonpreferred',
    'stationary_nonselective',
)


@dataclass
class Config:
    """Fit one multiple regression for every cached session."""

    data_dir: Path = Path('data/nature')
    cache_dir: Path = Path('cache/run_001')
    skip_group_level_norm_before_fit: bool = False


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


def _standardize_cells_then_average(baseline_rates, cell_indices):
    """Z-score each cell across trials, then average the usable cells."""
    cell_indices = np.asarray(cell_indices, dtype=np.int64)
    if cell_indices.size == 0:
        return np.zeros(baseline_rates.shape[0], dtype=float), 0, 'no_cells'

    values = np.asarray(baseline_rates[:, cell_indices], dtype=float)
    finite_cells = np.all(np.isfinite(values), axis=0)
    if not np.any(finite_cells):
        return np.zeros(baseline_rates.shape[0], dtype=float), 0, 'no_finite_cells'
    values = values[:, finite_cells]
    cell_mean = np.mean(values, axis=0)
    cell_std = np.std(values, axis=0)
    variable_cells = np.isfinite(cell_std) & (cell_std > 0)
    if not np.any(variable_cells):
        if np.allclose(values, 0.0):
            return np.zeros(baseline_rates.shape[0], dtype=float), 0, 'zero_activity'
        return np.zeros(baseline_rates.shape[0], dtype=float), 0, 'no_variable_cells'

    standardized = (values[:, variable_cells] - cell_mean[variable_cells]) / cell_std[
        variable_cells
    ]
    return np.mean(standardized, axis=1), int(np.sum(variable_cells)), 'ok'


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
    coefficients, _, rank, _ = np.linalg.lstsq(design, response, rcond=None)
    fitted = design @ coefficients
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
):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained')
    if group_level_norm_applied:
        x_label = 'Baseline activity (z-score)'
        normalization_title = 'group-level normalized'
    else:
        x_label = 'Baseline activity (mean cell z-score)'
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
        f'Session {session}: baseline activity and off-state duration '
        f'({normalization_title})'
    )
    output_path = output_dir / (
        f'off_state_duration_vs_baseline_activity_session_{session}.png'
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


def main(config: Config):
    decoding_results = _load_pickle(config.cache_dir / 'decoding_confidence.pkl')
    off_state_results = _load_pickle(config.cache_dir / 'on_off_states.pkl')
    selection_results = _load_pickle(config.cache_dir / 'cell_trial_selection.pkl')
    group_level_norm_applied = not config.skip_group_level_norm_before_fit
    output_dir = config.cache_dir / 'predict_off_state_duration_using_baseline_activity'
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for off_state_result in off_state_results:
        session = off_state_result.get('session', 'unknown_session')
        row = _empty_row(session, '')
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

            baseline_rates = compute_binned_rates(
                spikes,
                t,
                np.asarray([BASELINE_START]),
                BASELINE_END - BASELINE_START,
            )[:, 0, :]
            baseline_group_values = []
            group_status = []
            group_cell_counts = []
            for group_name in GROUP_NAMES:
                group_values, n_cells, status = _standardize_cells_then_average(
                    baseline_rates[trial_idx],
                    groups[group_name],
                )
                if group_level_norm_applied:
                    group_values, group_is_variable = _standardize_across_trials(group_values)
                    if not group_is_variable and status == 'ok':
                        status = 'zero_group_variance'
                baseline_group_values.append(group_values)
                group_status.append(f'{group_name}:{status}')
                group_cell_counts.append(n_cells)

            predictors = np.column_stack(baseline_group_values)
            response = off_duration
            response_is_variable = np.ptp(response) > 0
            if not np.all(np.isfinite(response)):
                raise ValueError('Off-state duration contains non-finite values.')
            if not response_is_variable:
                raise ValueError('Off-state duration has zero variance across trials.')
            if not np.all(np.isfinite(predictors)):
                raise ValueError('Baseline predictors contain non-finite values.')

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
        rows.append(row)

    csv_path = output_dir / 'off_state_duration_baseline_regression.csv'
    fieldnames = list(rows[0]) if rows else list(_empty_row('', ''))
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved regression statistics to {csv_path}')
    print(f'Saved session scatter plots to {output_dir}')


if __name__ == '__main__':
    main(tyro.cli(Config))
