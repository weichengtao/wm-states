import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tyro

try:
    from scripts.figure_exports import configure_figure_style, save_figure_png_only
    from scripts.predict_off_state_duration_using_baseline_activity import (
        GROUP_NAMES,
        _cell_groups,
        _find_full_session_selection,
        _fit_multiple_regression,
        _same_session,
        _significance_stars,
        _results_root,
    )
except ModuleNotFoundError:
    from figure_exports import configure_figure_style, save_figure_png_only
    from predict_off_state_duration_using_baseline_activity import (
        GROUP_NAMES,
        _cell_groups,
        _find_full_session_selection,
        _fit_multiple_regression,
        _same_session,
        _significance_stars,
        _results_root,
    )

configure_figure_style(matplotlib)


DV_SPECS = (
    (
        'mean_off_state_duration_across_states',
        'Mean off-state duration across states',
    ),
    (
        'max_off_state_duration_across_states',
        'Maximum off-state duration across states',
    ),
    (
        'mean_trial_total_off_state_duration',
        'Mean trial-level total off-state duration',
    ),
    (
        'max_trial_total_off_state_duration',
        'Maximum trial-level total off-state duration',
    ),
)


@dataclass
class Config:
    """Fit cross-session regressions using raw cell counts and durations."""

    data_dir: Path = Path('data/nature')
    cache_dir: Path = Path('cache/run_001')
    output_subdir: str = 'fixedlm'


def _load_pickle(path: Path):
    if not path.exists():
        raise FileNotFoundError(f'Missing cache file: {path}')
    with path.open('rb') as f:
        return pickle.load(f)


def _format_value(value):
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    value = float(value)
    if not np.isfinite(value):
        return 'nan'
    return f'{value:.8g}'


def _session_observation(off_state_result, selection_results, data_dir):
    session = off_state_result.get('session', 'unknown_session')
    if off_state_result.get('off_state_duration_correction') != 'applied':
        raise ValueError('The off-state cache does not contain CC-applied durations.')

    state_durations = np.asarray(
        off_state_result.get('off_state_duration_per_state', []),
        dtype=float,
    )
    if state_durations.ndim != 1 or state_durations.size == 0:
        raise ValueError('No CC-applied off-state durations were detected.')
    if not np.all(np.isfinite(state_durations)):
        raise ValueError('Per-state off-state durations contain non-finite values.')

    trial_durations = np.asarray(
        off_state_result.get('off_state_duration_per_trial', []),
        dtype=float,
    )
    if trial_durations.ndim != 1 or trial_durations.size == 0:
        raise ValueError('No preferred-cue trial-level durations were cached.')
    if not np.all(np.isfinite(trial_durations)):
        raise ValueError('Trial-level off-state durations contain non-finite values.')

    data_path = data_dir / f'{session}.mat'
    if not data_path.exists():
        raise FileNotFoundError(f'Missing data file: {data_path}')
    from scipy.io import loadmat

    data = loadmat(data_path)
    num_trials = np.asarray(data['spks']).shape[0]
    selection_result = _find_full_session_selection(
        selection_results,
        session,
        num_trials,
    )
    groups = _cell_groups(selection_result, int(off_state_result['cue']))
    cell_counts = {
        group_name: int(np.asarray(groups[group_name]).size)
        for group_name in GROUP_NAMES
    }

    return {
        'session': session,
        'preferred_cue': int(off_state_result['cue']),
        'n_trials': int(trial_durations.size),
        'n_states': int(state_durations.size),
        'n_preferred_cells': cell_counts['preferred'],
        'n_selective_nonpreferred_cells': cell_counts['selective_nonpreferred'],
        'n_stationary_nonselective_cells': cell_counts['stationary_nonselective'],
        'mean_off_state_duration_across_states': float(np.mean(state_durations)),
        'max_off_state_duration_across_states': float(np.max(state_durations)),
        'mean_trial_total_off_state_duration': float(np.mean(trial_durations)),
        'max_trial_total_off_state_duration': float(np.max(trial_durations)),
        'description': (
            'ok; preferred-cue trials; CC-applied off-states; '
            f'{state_durations.size} states and {trial_durations.size} trials'
        ),
    }


def _save_scatter_plot(
    output_dir,
    dv_name,
    dv_label,
    predictors,
    response,
    coefficients,
    p_values,
    r_squared,
):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained')
    for index, group_name in enumerate(GROUP_NAMES):
        x = predictors[:, index]
        ax = axes[index]
        ax.scatter(x, response, color=f'C{index}', alpha=0.75, s=32)
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
        ax.set_title(group_name.replace('_', ' ').title())
        ax.set_xlabel('Number of cells')
        if index == 0:
            ax.set_ylabel(f'{dv_label} (ms)')
        ax.text(
            0.05,
            0.05,
            f'β = {coefficients[index + 1]:.3g}'
            f'{_significance_stars(p_values[index + 1])}\n'
            f'R² = {r_squared:.3g}',
            transform=ax.transAxes,
            va='bottom',
        )

    fig.suptitle(f'{dv_label} predicted by session cell counts')
    output_path = output_dir / f'{dv_name}.png'
    save_figure_png_only(fig, output_path, dpi=300)
    plt.close(fig)
    return output_path


def _write_log(log_path, session_entries, regression_entries):
    lines = [
        'Predict off-state duration using session cell counts',
        'IVs and DVs are not normalized.',
        '',
        'Session observations:',
    ]
    for entry in session_entries:
        lines.append(
            f"session={entry['session']}; status={entry['status']}; "
            f"description={entry['description']}"
        )
        if entry['status'] != 'ok':
            continue
        observation = entry['observation']
        lines.append(
            '  '
            + '; '.join(
                f'{key}={_format_value(observation[key])}'
                for key in (
                    'preferred_cue',
                    'n_trials',
                    'n_states',
                    'n_preferred_cells',
                    'n_selective_nonpreferred_cells',
                    'n_stationary_nonselective_cells',
                    'mean_off_state_duration_across_states',
                    'max_off_state_duration_across_states',
                    'mean_trial_total_off_state_duration',
                    'max_trial_total_off_state_duration',
                )
            )
        )

    lines.extend(['', 'Regression statistics:'])
    for entry in regression_entries:
        lines.append(f"DV={entry['dv_name']}; description={entry['description']}")
        if entry['status'] != 'ok':
            continue
        result = entry['result']
        coefficients, p_values, r_squared, rank = result
        lines.append(f"  n_sessions={entry['n_sessions']}; rank={rank}")
        lines.append(
            '  '
            + '; '.join(
                f'{name}={_format_value(value)}'
                for name, value in zip(
                    (
                        'intercept',
                        'coefficient_preferred',
                        'coefficient_selective_nonpreferred',
                        'coefficient_stationary_nonselective',
                    ),
                    coefficients,
                )
            )
        )
        lines.append(
            '  '
            + '; '.join(
                f'{name}={_format_value(value)}'
                for name, value in zip(
                    (
                        'p_value_intercept',
                        'p_value_preferred',
                        'p_value_selective_nonpreferred',
                        'p_value_stationary_nonselective',
                    ),
                    p_values,
                )
            )
            + f'; r_squared={_format_value(r_squared)}'
        )

    log_path.write_text('\n'.join(lines) + '\n')


def main(config: Config):
    off_state_results = _load_pickle(config.cache_dir / 'on_off_states.pkl')
    selection_results = _load_pickle(config.cache_dir / 'cell_trial_selection.pkl')
    output_dir = (
        _results_root(config.cache_dir, config.output_subdir)
        / 'predict_off_state_duration_using_cell_count'
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    session_entries = []
    observations = []
    seen_sessions = set()
    for off_state_result in off_state_results:
        session = off_state_result.get('session', 'unknown_session')
        session_key = str(session)
        if session_key in seen_sessions:
            session_entries.append({
                'session': session,
                'status': 'skipped',
                'description': 'multiple on/off cache entries for this session',
            })
            continue
        seen_sessions.add(session_key)
        try:
            observation = _session_observation(
                off_state_result,
                selection_results,
                config.data_dir,
            )
        except Exception as error:
            session_entries.append({
                'session': session,
                'status': 'skipped',
                'description': f'{type(error).__name__}: {error}',
            })
            continue
        observations.append(observation)
        session_entries.append({
            'session': session,
            'status': 'ok',
            'description': observation['description'],
            'observation': observation,
        })

    regression_entries = []
    if observations:
        predictors = np.asarray([
            [
                observation['n_preferred_cells'],
                observation['n_selective_nonpreferred_cells'],
                observation['n_stationary_nonselective_cells'],
            ]
            for observation in observations
        ], dtype=float)
        for dv_name, dv_label in DV_SPECS:
            response = np.asarray(
                [observation[dv_name] for observation in observations],
                dtype=float,
            )
            try:
                if not np.all(np.isfinite(predictors)) or not np.all(np.isfinite(response)):
                    raise ValueError('Predictors or response contains non-finite values.')
                if response.size < 2:
                    raise ValueError('At least two sessions are required.')
                if np.ptp(response) == 0:
                    raise ValueError('Response has zero variance across sessions.')
                result = _fit_multiple_regression(predictors, response)
                plot_path = _save_scatter_plot(
                    output_dir,
                    dv_name,
                    dv_label,
                    predictors,
                    response,
                    result[0],
                    result[1],
                    result[2],
                )
                description = f'ok; plot={plot_path.name}'
                regression_entries.append({
                    'dv_name': dv_name,
                    'status': 'ok',
                    'description': description,
                    'n_sessions': len(observations),
                    'result': result,
                })
            except Exception as error:
                regression_entries.append({
                    'dv_name': dv_name,
                    'status': 'skipped',
                    'description': f'{type(error).__name__}: {error}',
                })
    else:
        for dv_name, _ in DV_SPECS:
            regression_entries.append({
                'dv_name': dv_name,
                'status': 'skipped',
                'description': 'no usable sessions',
            })

    log_path = output_dir / 'cell_count_regression.log'
    _write_log(log_path, session_entries, regression_entries)
    print(f'Saved regression log to {log_path}')
    print(f'Saved session scatter plots to {output_dir}')


if __name__ == '__main__':
    main(tyro.cli(Config))
