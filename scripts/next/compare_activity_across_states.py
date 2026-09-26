"""Compare balanced correct-trial activity across states and cue groups.

Preparation, typed cell/PC metadata, and rendering live in dedicated activity
modules. This entry point validates run inputs and writes stage-owned figures.
"""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "scripts.next"

from pathlib import Path
from typing import Any

import numpy as np
import tyro

from scripts.next import cache_io as pickle
from scripts.next.cache_paths import primary_cache, stage_path
from scripts.next.common import validate_state_provenance
from scripts.next.activity_weighting import weighting_subdir
from scripts.next.activity_types import (
    Config, CellActivityDimensions, PrincipalComponentActivity,
    PrincipalComponentDimensions, SessionActivity, get_opposite_cue, cue_to_deg,
)
from scripts.next.screening_metadata import preferred_pev_cells
from scripts.next.activity_preparation import (
    find_full_session_selection,
    top_preferred_pev_cells,
    session_cell_groups,
    balance_trial_groups,
    maximum_delay_off_state_mask,
    compute_binned_firing_rates,
    normalize_balanced_activity,
    balanced_activity_normalization_parameters,
    apply_activity_normalization,
    compute_preferred_cell_principal_components,
    principal_component_session_activity,
    prepare_session_activity,
)
from scripts.next.activity_plots import (
    save_session_activity_figures,
    activity_point_categories,
    population_mean_point_categories,
    _category_legend_label,
    fixed_width_bin_edges,
    centered_histogram_bin_offsets,
    _activity_dimension_axis_label,
    _activity_dimension_title,
    _session_activity_figure_title,
    _placeholder_figure,
    _category_zorder,
    _plot_single_cell_strip,
    plot_session_activity,
    plot_session_activity_pairwise,
    _plot_marginal_histogram,
    _plot_ecdf,
    plot_session_activity_marginal_histograms,
)


def _load_pickle(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing cache file: {path}")
    with path.open("rb") as handle:
        return pickle.load(handle)


def main(config: Config):
    """Generate one activity-state comparison plot for every cached session."""
    output_subdir = Path(config.output_subdir)
    if output_subdir.is_absolute() or ".." in output_subdir.parts:
        raise ValueError("output_subdir must stay within its owning stage directory.")
    if config.figure_dpi <= 0:
        raise ValueError("figure_dpi must be positive.")
    if config.max_points_per_color_group is not None and (
        isinstance(config.max_points_per_color_group, (bool, np.bool_))
        or config.max_points_per_color_group <= 0
    ):
        raise ValueError("max_points_per_color_group must be positive when set.")
    if config.max_points_per_max_off_state is not None and (
        isinstance(config.max_points_per_max_off_state, (bool, np.bool_))
        or config.max_points_per_max_off_state <= 0
    ):
        raise ValueError("max_points_per_max_off_state must be positive when set.")
    if (
        not np.isfinite(config.marginal_histogram_bin_width)
        or config.marginal_histogram_bin_width <= 0
    ):
        raise ValueError(
            "marginal_histogram_bin_width must be finite and positive."
        )
    if (
        not np.isfinite(config.marginal_histogram_bin_offset_fraction)
        or not 0 <= config.marginal_histogram_bin_offset_fraction <= 0.5
    ):
        raise ValueError(
            "marginal_histogram_bin_offset_fraction must be between 0 and 0.5."
        )

    selection_results = _load_pickle(primary_cache(config.cache_dir, "cell_screening.pkl"))
    state_results = _load_pickle(primary_cache(config.cache_dir, "on_off_states.pkl"))
    if not isinstance(selection_results, list) or not isinstance(state_results, list):
        raise TypeError("Both input cache files must contain lists of results.")
    if not state_results:
        raise ValueError("The on/off-state cache contains no session results.")

    sessions = [str(result.get("session", "unknown_session")) for result in state_results]
    if len(set(sessions)) != len(sessions):
        raise ValueError("The on/off-state cache contains duplicate session entries.")
    validate_state_provenance(state_results, config.cache_dir, config.data_dir)

    figure_dir = weighting_subdir(
        stage_path(config.cache_dir, "activity", output_subdir, "figures"),
        config.pev_weighted_average,
    )
    figure_dir.mkdir(parents=True, exist_ok=True)
    for session_idx, state_result in enumerate(state_results):
        prepared = prepare_session_activity(
            state_result,
            selection_results,
            config,
            session_seed=config.seed + session_idx,
        )
        plotting_views = [(prepared, "")]
        if config.show_principal_components:
            plotting_views.append(
                (
                    principal_component_session_activity(prepared),
                    "principal_components_",
                )
            )
        for plotting_activity, filename_prefix in plotting_views:
            save_session_activity_figures(
                plotting_activity,
                figure_dir,
                config,
                point_seed=config.seed + session_idx,
                filename_prefix=filename_prefix,
            )
        principal_component_summary = (
            "with principal components " if config.show_principal_components else ""
        )
        print(
            "Saved state and cue activity, pairwise, and marginal comparisons "
            f"{principal_component_summary}"
            "for session "
            f"{prepared.session} "
            f"({prepared.preferred_trial_ids.size} trials per cue; "
            f"{prepared.max_off_state_activity.shape[0]} maximum off-state bins)."
        )


if __name__ == "__main__":
    main(tyro.cli(Config))
