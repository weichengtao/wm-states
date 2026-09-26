"""Figures and reproducible display sampling for prepared activity results."""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
from textwrap import fill
from typing import Literal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.next.activity_types import (
    Config, PrincipalComponentDimensions, SessionActivity,
    POPULATION_GROUP_KEYS, cue_to_deg,
)
from scripts.next.figure_exports import configure_figure_style, save_figure
from scripts.next.screening_metadata import population_labels

configure_figure_style(matplotlib)


def activity_point_categories(
    session_activity: SessionActivity,
    comparison: Literal["state", "cue"] = "state",
    compare_with_max_off_state: bool = False,
    hide_opposite_cue_points: bool = False,
    hide_all_preferred_cue_points: bool = False,
    max_points_per_color_group: int | None = None,
    max_points_per_max_off_state: int | None = None,
    seed: int = 42,
):
    """Return optionally subsampled points for one comparison figure."""
    if max_points_per_color_group is not None and (
        isinstance(max_points_per_color_group, (bool, np.bool_))
        or max_points_per_color_group <= 0
    ):
        raise ValueError("max_points_per_color_group must be positive when set.")
    if max_points_per_max_off_state is not None and (
        isinstance(max_points_per_max_off_state, (bool, np.bool_))
        or max_points_per_max_off_state <= 0
    ):
        raise ValueError("max_points_per_max_off_state must be positive when set.")
    if comparison not in ("state", "cue"):
        raise ValueError("comparison must be either 'state' or 'cue'.")

    num_cells = session_activity.dimension_count
    preferred_point_count = int(np.prod(session_activity.preferred_activity.shape[:2]))
    opposite_point_count = int(np.prod(session_activity.opposite_activity.shape[:2]))
    preferred_points = session_activity.preferred_activity.reshape(
        preferred_point_count,
        num_cells,
    )
    opposite_points = session_activity.opposite_activity.reshape(
        opposite_point_count,
        num_cells,
    )
    on_mask = session_activity.on_state_mask.ravel()
    off_mask = session_activity.off_state_mask.ravel()
    categories = []
    if comparison == "state":
        categories.extend(
            [
                (preferred_points[on_mask], "tab:blue", "Preferred cue: on state"),
                (
                    preferred_points[off_mask],
                    "tab:orange",
                    "Preferred cue: off state",
                ),
            ]
        )
        if (
            compare_with_max_off_state
            and session_activity.max_off_state_activity is not None
        ):
            max_off_state_points = np.asarray(
                session_activity.max_off_state_activity,
                dtype=float,
            )
            if (
                max_off_state_points.ndim != 2
                or max_off_state_points.shape[1] != num_cells
            ):
                raise ValueError(
                    "Maximum off-state activity must have shape (bin, cell)."
                )
            categories.append(
                (
                    max_off_state_points,
                    "tab:red",
                    "Preferred cue: maximum off state",
                )
            )
    else:
        if not hide_all_preferred_cue_points:
            categories.append(
                (preferred_points, "tab:green", "Preferred cue: all delay bins")
            )
        if not hide_opposite_cue_points:
            categories.append(
                (opposite_points, "tab:gray", "Opposite cue: all delay bins")
            )

    rng = np.random.default_rng(seed)
    max_off_state_rng = np.random.default_rng(
        np.random.SeedSequence([seed, 1])
    )
    displayed_categories = []
    for points, color, label in categories:
        total_count = points.shape[0]
        point_limit = (
            max_points_per_max_off_state
            if color == "tab:red"
            else max_points_per_color_group
        )
        if (
            point_limit is not None
            and total_count > point_limit
        ):
            category_rng = max_off_state_rng if color == "tab:red" else rng
            point_indices = np.sort(
                category_rng.choice(
                    total_count,
                    size=point_limit,
                    replace=False,
                )
            )
            points = points[point_indices]
        displayed_categories.append((points, color, label, total_count))
    return displayed_categories


def population_mean_point_categories(
    session_activity: SessionActivity,
    population_group: str = "preferred",
    comparison: Literal["state", "cue"] = "state",
    compare_with_max_off_state: bool = False,
    hide_opposite_cue_points: bool = False,
    hide_all_preferred_cue_points: bool = False,
    max_points_per_max_off_state: int | None = None,
    seed: int = 42,
):
    """Return state groups for one cell population's mean normalized activity."""
    if population_group in session_activity.population_mean_activities:
        (
            preferred_mean,
            opposite_mean,
            population_cell_count,
        ) = session_activity.population_mean_activities[population_group]
    elif population_group == "preferred":
        preferred_mean = session_activity.preferred_population_mean_activity
        opposite_mean = session_activity.opposite_population_mean_activity
        population_cell_count = session_activity.preferred_population_cell_count
        if preferred_mean is None or opposite_mean is None:
            if (session_activity.activity_space == "principal_components"
                    or session_activity.dimension_count == 0):
                return None, 0
            preferred_mean = np.mean(session_activity.preferred_activity, axis=2)
            opposite_mean = np.mean(session_activity.opposite_activity, axis=2)
            population_cell_count = int(session_activity.dimension_count)
    else:
        return None, 0
    if preferred_mean is None or opposite_mean is None or population_cell_count == 0:
        return None, 0
    if comparison not in ("state", "cue"):
        raise ValueError("comparison must be either 'state' or 'cue'.")
    if max_points_per_max_off_state is not None and (
        isinstance(max_points_per_max_off_state, (bool, np.bool_))
        or max_points_per_max_off_state <= 0
    ):
        raise ValueError("max_points_per_max_off_state must be positive when set.")

    preferred_points = np.asarray(preferred_mean, dtype=float).ravel()
    opposite_points = np.asarray(opposite_mean, dtype=float).ravel()
    on_mask = session_activity.on_state_mask.ravel()
    off_mask = session_activity.off_state_mask.ravel()
    if preferred_points.size != on_mask.size or preferred_points.size != off_mask.size:
        raise ValueError("Preferred population mean activity does not match state masks.")
    categories = []
    if comparison == "state":
        categories.extend(
            [
                (
                    preferred_points[on_mask, None],
                    "tab:blue",
                    "Preferred cue: on state",
                    int(np.count_nonzero(on_mask)),
                ),
                (
                    preferred_points[off_mask, None],
                    "tab:orange",
                    "Preferred cue: off state",
                    int(np.count_nonzero(off_mask)),
                ),
            ]
        )
        max_off_state_mean = None
        if compare_with_max_off_state:
            max_off_state_mean = (
                session_activity.max_off_state_population_mean_activities.get(
                    population_group
                )
            )
        if (
            compare_with_max_off_state
            and max_off_state_mean is None
            and population_group == "preferred"
            and session_activity.activity_space == "cells"
        ):
            max_off_state_activity = session_activity.max_off_state_activity
            if max_off_state_activity is not None and max_off_state_activity.shape[1]:
                max_off_state_mean = np.mean(max_off_state_activity, axis=1)
        if max_off_state_mean is not None:
            max_off_state_points = np.asarray(
                max_off_state_mean,
                dtype=float,
            ).ravel()
            total_count = int(max_off_state_points.size)
            if (
                max_points_per_max_off_state is not None
                and total_count > max_points_per_max_off_state
            ):
                rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
                selected = np.sort(
                    rng.choice(
                        total_count,
                        size=max_points_per_max_off_state,
                        replace=False,
                    )
                )
                max_off_state_points = max_off_state_points[selected]
            categories.append(
                (
                    max_off_state_points[:, None],
                    "tab:red",
                    "Preferred cue: maximum off state",
                    total_count,
                )
            )
    else:
        if not hide_all_preferred_cue_points:
            categories.append(
                (
                    preferred_points[:, None],
                    "tab:green",
                    "Preferred cue: all delay bins",
                    int(preferred_points.size),
                )
            )
        if not hide_opposite_cue_points:
            categories.append(
                (
                    opposite_points[:, None],
                    "tab:gray",
                    "Opposite cue: all delay bins",
                    int(opposite_points.size),
                )
            )
    return categories, population_cell_count


def _category_legend_label(label: str, displayed_count: int, total_count: int) -> str:
    if displayed_count == total_count:
        return f"{label} (n={total_count})"
    return f"{label} (shown={displayed_count}, total={total_count})"


def fixed_width_bin_edges(values: np.ndarray, bin_width: float = 0.25) -> np.ndarray:
    """Return bin edges aligned to integer multiples of a fixed width."""
    values = np.asarray(values, dtype=float).ravel()
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Histogram values must be non-empty and finite.")
    if not np.isfinite(bin_width) or bin_width <= 0:
        raise ValueError("marginal_histogram_bin_width must be finite and positive.")
    lower = np.floor(np.min(values) / bin_width) * bin_width
    upper = np.ceil(np.max(values) / bin_width) * bin_width
    if upper <= lower:
        upper = lower + bin_width
    bin_count = max(1, int(np.ceil((upper - lower) / bin_width - 1e-12)))
    return lower + np.arange(bin_count + 1, dtype=float) * bin_width


def centered_histogram_bin_offsets(
    category_count: int,
    bin_width: float,
    offset_fraction: float = 0.2,
) -> np.ndarray:
    """Return symmetric display offsets for overlaid histogram outlines."""
    if (
        not isinstance(category_count, (int, np.integer))
        or isinstance(category_count, (bool, np.bool_))
        or category_count < 0
    ):
        raise ValueError("category_count must be a non-negative integer.")
    if not np.isfinite(bin_width) or bin_width <= 0:
        raise ValueError("bin_width must be finite and positive.")
    if not np.isfinite(offset_fraction) or not 0 <= offset_fraction <= 0.5:
        raise ValueError("offset_fraction must be between 0 and 0.5.")
    centered_positions = np.arange(category_count, dtype=float)
    centered_positions -= (category_count - 1) / 2
    return centered_positions * offset_fraction * bin_width


def _activity_dimension_axis_label(
    session_activity: SessionActivity,
    dimension_position: int,
) -> str:
    """Return the full axis label for a selected cell or principal component."""
    dimensions = session_activity.dimensions
    if isinstance(dimensions, PrincipalComponentDimensions):
        return (
            f"PC{dimensions.component_numbers[dimension_position]} score\n"
            f"(explained variance={dimensions.explained_variance_ratio[dimension_position] * 100:.2f}%)"
        )
    return (
        f"Cell {dimensions.cell_ids[dimension_position]} normalized activity\n"
        f"(selectivity PEV={dimensions.selectivity_pev_pct[dimension_position]:.2f}%)"
    )


def _activity_dimension_title(
    session_activity: SessionActivity,
    dimension_position: int,
) -> str:
    """Return a compact panel title without reusing cell IDs for PC numbers."""
    dimensions = session_activity.dimensions
    if isinstance(dimensions, PrincipalComponentDimensions):
        return f"PC{dimensions.component_numbers[dimension_position]}"
    return f"Cell {dimensions.cell_ids[dimension_position]}"



def _session_activity_figure_title(
    session_activity: SessionActivity,
    comparison_description: str,
    marginal: bool = False,
) -> str:
    """Return a representation-aware session figure title."""
    base = (
        f"Session {session_activity.session}: preferred cue "
        f"{cue_to_deg(session_activity.preferred_cue)}°"
    )
    if session_activity.activity_space == "principal_components":
        marginal_label = " marginal" if marginal else ""
        return (
            f"{base} principal-component{marginal_label} activity\n"
            f"(PCA of {session_activity.activity_source_cell_count} preferred cells; "
            f"{comparison_description})"
        )
    if marginal:
        return f"{base} marginal activity ({comparison_description})"
    return f"{base} ({comparison_description})"


def _placeholder_figure(session_activity: SessionActivity, plot_name: str):
    """Return a session figure explaining that no preferred cells are available."""
    fig, ax = plt.subplots(figsize=(6.5, 4), layout="constrained")
    if session_activity.activity_space == "principal_components":
        message = "No finite-PEV preferred cells available for PCA"
    else:
        message = "No finite-PEV cells preferred for this session cue"
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=ax.transAxes,
    )
    ax.set_title(
        f"Session {session_activity.session}: preferred cue "
        f"{cue_to_deg(session_activity.preferred_cue)}° {plot_name}"
    )
    ax.set_axis_off()
    return fig


def _category_zorder(color: str) -> int:
    """Keep state-specific marks above the all-bin comparison groups."""
    return {
        "tab:gray": 1,
        "tab:green": 2,
        "tab:orange": 3,
        "tab:blue": 4,
        "tab:red": 5,
    }[color]


def _plot_single_cell_strip(ax, session_activity: SessionActivity, categories):
    """Plot one activity dimension against categorical state rows."""
    y_positions = np.arange(len(categories), dtype=float)
    y_labels = []
    for category_idx, (points, color, label, total_count) in enumerate(categories):
        ax.scatter(
            points[:, 0],
            np.full(points.shape[0], y_positions[category_idx]),
            s=12,
            alpha=0.55,
            color=color,
            edgecolors="none",
            zorder=_category_zorder(color),
        )
        y_labels.append(
            _category_legend_label(label, points.shape[0], total_count)
        )
    ax.set_yticks(y_positions, y_labels)
    ax.set_xlabel(_activity_dimension_axis_label(session_activity, 0))
    ax.set_title(_activity_dimension_title(session_activity, 0))
    ax.spines[["top", "right"]].set_visible(False)


def plot_session_activity(
    session_activity: SessionActivity,
    comparison: Literal["state", "cue"] = "state",
    compare_with_max_off_state: bool = False,
    hide_opposite_cue_points: bool = False,
    hide_all_preferred_cue_points: bool = False,
    max_points_per_color_group: int | None = None,
    max_points_per_max_off_state: int | None = None,
    point_seed: int = 42,
):
    """Create a session scatter plot using every available preferred cell."""
    num_cells = session_activity.dimension_count
    if num_cells == 0:
        return _placeholder_figure(session_activity, f"{comparison} activity")
    categories = activity_point_categories(
        session_activity,
        comparison=comparison,
        compare_with_max_off_state=compare_with_max_off_state,
        hide_opposite_cue_points=hide_opposite_cue_points,
        hide_all_preferred_cue_points=hide_all_preferred_cue_points,
        max_points_per_color_group=max_points_per_color_group,
        max_points_per_max_off_state=max_points_per_max_off_state,
        seed=point_seed,
    )

    if num_cells == 3:
        fig = plt.figure(figsize=(6.5, 5.5), layout="constrained")
        ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
        for points, color, label, total_count in categories:
            ax.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                s=12,
                alpha=0.55,
                color=color,
                edgecolors="none",
                zorder=_category_zorder(color),
                label=_category_legend_label(label, points.shape[0], total_count),
            )
        for axis_idx, axis_name in enumerate(
            ("set_xlabel", "set_ylabel", "set_zlabel")
        ):
            getattr(ax, axis_name)(
                _activity_dimension_axis_label(session_activity, axis_idx)
            )
        ax.legend(loc="best", frameon=False)
        ax.view_init(elev=24, azim=42)
    elif num_cells == 2:
        fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
        for points, color, label, total_count in categories:
            ax.scatter(
                points[:, 0],
                points[:, 1],
                s=12,
                alpha=0.55,
                color=color,
                edgecolors="none",
                zorder=_category_zorder(color),
                label=_category_legend_label(label, points.shape[0], total_count),
            )
        ax.set_xlabel(_activity_dimension_axis_label(session_activity, 0))
        ax.set_ylabel(_activity_dimension_axis_label(session_activity, 1))
        ax.legend(loc="best", frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    else:
        fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
        _plot_single_cell_strip(ax, session_activity, categories)
    ax.set_title(
        _session_activity_figure_title(
            session_activity,
            "on/off states" if comparison == "state" else "all delay bins by cue",
        )
    )
    return fig


def plot_session_activity_pairwise(
    session_activity: SessionActivity,
    comparison: Literal["state", "cue"] = "state",
    compare_with_max_off_state: bool = False,
    hide_opposite_cue_points: bool = False,
    hide_all_preferred_cue_points: bool = False,
    max_points_per_color_group: int | None = None,
    max_points_per_max_off_state: int | None = None,
    point_seed: int = 42,
):
    """Create every available pairwise 2D projection for one session."""
    num_cells = session_activity.dimension_count
    if num_cells == 0:
        return _placeholder_figure(
            session_activity,
            f"{comparison} pairwise activity",
        )
    categories = activity_point_categories(
        session_activity,
        comparison=comparison,
        compare_with_max_off_state=compare_with_max_off_state,
        hide_opposite_cue_points=hide_opposite_cue_points,
        hide_all_preferred_cue_points=hide_all_preferred_cue_points,
        max_points_per_color_group=max_points_per_color_group,
        max_points_per_max_off_state=max_points_per_max_off_state,
        seed=point_seed,
    )
    cell_pairs = list(combinations(range(num_cells), 2))

    if not cell_pairs:
        fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
        _plot_single_cell_strip(ax, session_activity, categories)
        dimension_description = (
            "principal component"
            if session_activity.activity_space == "principal_components"
            else "preferred cell"
        )
        fig.suptitle(
            f"Session {session_activity.session}: preferred cue "
            f"{cue_to_deg(session_activity.preferred_cue)}° "
            f"(one {dimension_description}; no pair available)"
        )
        return fig

    fig, axes = plt.subplots(
        1,
        len(cell_pairs),
        figsize=(4 * len(cell_pairs), 4),
        layout="constrained",
        squeeze=False,
    )
    axes = axes.ravel()
    for pair_idx, (ax, (x_idx, y_idx)) in enumerate(zip(axes, cell_pairs)):
        for points, color, label, total_count in categories:
            ax.scatter(
                points[:, x_idx],
                points[:, y_idx],
                s=8,
                alpha=0.45,
                color=color,
                edgecolors="none",
                zorder=_category_zorder(color),
                label=_category_legend_label(label, points.shape[0], total_count),
            )
        ax.set_xlabel(_activity_dimension_axis_label(session_activity, x_idx))
        ax.set_ylabel(_activity_dimension_axis_label(session_activity, y_idx))
        ax.set_title(
            f"{_activity_dimension_title(session_activity, x_idx)} and "
            f"{_activity_dimension_title(session_activity, y_idx)}"
        )
        ax.spines[["top", "right"]].set_visible(False)
        if pair_idx == 0:
            ax.legend(loc="best", frameon=False, fontsize="small")

    fig.suptitle(
        _session_activity_figure_title(
            session_activity,
            "on/off states" if comparison == "state" else "all delay bins by cue",
        )
    )
    return fig


def _plot_marginal_histogram(
    ax,
    categories,
    value_position: int,
    bin_width: float,
    bin_offset_fraction: float,
    xlabel: str,
    title: str,
    show_legend: bool,
):
    """Plot one overlaid full-distribution marginal histogram."""
    ax.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=1,
        zorder=0,
    )
    nonempty_values = [
        points[:, value_position]
        for points, _, _, _ in categories
        if points.shape[0] > 0
    ]
    if not nonempty_values:
        raise ValueError("At least one activity color group must contain points.")
    bin_edges = fixed_width_bin_edges(
        np.concatenate(nonempty_values),
        bin_width=bin_width,
    )
    bin_offsets = centered_histogram_bin_offsets(
        len(categories),
        bin_width,
        offset_fraction=bin_offset_fraction,
    )
    for category_idx, (points, color, label, total_count) in enumerate(categories):
        legend_label = _category_legend_label(
            label,
            points.shape[0],
            total_count,
        )
        if points.shape[0] == 0:
            ax.plot(
                [],
                [],
                color=color,
                linewidth=1.5,
                label=legend_label,
                zorder=_category_zorder(color),
            )
            continue
        density, _ = np.histogram(
            points[:, value_position],
            bins=bin_edges,
            density=True,
        )
        ax.stairs(
            density,
            bin_edges + bin_offsets[category_idx],
            linewidth=1.5,
            color=color,
            label=legend_label,
            zorder=_category_zorder(color),
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    if show_legend:
        ax.legend(loc="best", frameon=False, fontsize="small")


def _plot_ecdf(
    ax,
    categories,
    value_position: int,
    xlabel: str,
):
    """Plot one empirical cumulative distribution per activity group."""
    ax.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=1,
        zorder=0,
    )
    for points, color, label, total_count in categories:
        legend_label = _category_legend_label(
            label,
            points.shape[0],
            total_count,
        )
        if points.shape[0] == 0:
            ax.plot(
                [],
                [],
                color=color,
                linewidth=1.5,
                label=legend_label,
                zorder=_category_zorder(color),
            )
            continue
        values = np.sort(points[:, value_position])
        cumulative_probability = np.arange(1, values.size + 1) / values.size
        ax.step(
            values,
            cumulative_probability,
            where="post",
            linewidth=1.5,
            color=color,
            label=legend_label,
            zorder=_category_zorder(color),
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("ECDF")
    ax.set_ylim(0, 1.02)
    ax.spines[["top", "right"]].set_visible(False)


def plot_session_activity_marginal_histograms(
    session_activity: SessionActivity,
    comparison: Literal["state", "cue"] = "state",
    compare_with_max_off_state: bool = False,
    hide_opposite_cue_points: bool = False,
    hide_all_preferred_cue_points: bool = False,
    max_points_per_max_off_state: int | None = None,
    point_seed: int = 42,
    bin_width: float = 0.25,
    bin_offset_fraction: float = 0.2,
):
    """Plot selected-cell marginals plus three cell-population means."""
    num_cells = session_activity.dimension_count
    categories = None
    if num_cells:
        categories = activity_point_categories(
            session_activity,
            comparison=comparison,
            compare_with_max_off_state=compare_with_max_off_state,
            hide_opposite_cue_points=hide_opposite_cue_points,
            hide_all_preferred_cue_points=hide_all_preferred_cue_points,
            max_points_per_max_off_state=max_points_per_max_off_state,
            seed=point_seed,
        )
    population_data = []
    group_labels = population_labels(session_activity.screening_checks)
    for group_name in POPULATION_GROUP_KEYS:
        group_title = group_labels[group_name]
        group_categories, group_cell_count = population_mean_point_categories(
            session_activity,
            population_group=group_name,
            comparison=comparison,
            compare_with_max_off_state=compare_with_max_off_state,
            hide_opposite_cue_points=hide_opposite_cue_points,
            hide_all_preferred_cue_points=hide_all_preferred_cue_points,
            max_points_per_max_off_state=max_points_per_max_off_state,
            seed=point_seed,
        )
        population_data.append(
            (group_name, group_title, group_categories, group_cell_count)
        )
    num_panels = num_cells + len(POPULATION_GROUP_KEYS)
    fig, axes = plt.subplots(
        2,
        num_panels,
        figsize=(4 * num_panels, 7),
        layout="constrained",
        sharex="col",
        squeeze=False,
    )
    histogram_axes = axes[0]
    ecdf_axes = axes[1]
    legend_shown = False
    for cell_position in range(num_cells):
        xlabel = _activity_dimension_axis_label(session_activity, cell_position)
        _plot_marginal_histogram(
            histogram_axes[cell_position],
            categories,
            value_position=cell_position,
            bin_width=bin_width,
            bin_offset_fraction=bin_offset_fraction,
            xlabel="",
            title=_activity_dimension_title(session_activity, cell_position),
            show_legend=not legend_shown,
        )
        legend_shown = True
        _plot_ecdf(
            ecdf_axes[cell_position],
            categories,
            value_position=cell_position,
            xlabel=xlabel,
        )
    for group_idx, (group_name, group_title, group_categories, group_count) in enumerate(
        population_data
    ):
        panel_idx = num_cells + group_idx
        histogram_ax = histogram_axes[panel_idx]
        ecdf_ax = ecdf_axes[panel_idx]
        if group_categories is None:
            histogram_ax.set_title(fill(group_title, width=32))
            histogram_ax.text(
                0.5,
                0.5,
                "No cells in group",
                ha="center",
                va="center",
                transform=histogram_ax.transAxes,
            )
            histogram_ax.set_axis_off()
            ecdf_ax.text(
                0.5,
                0.5,
                "No ECDF available",
                ha="center",
                va="center",
                transform=ecdf_ax.transAxes,
            )
            ecdf_ax.set_axis_off()
            continue
        count_label = f"{group_count} {group_title.lower()}"
        _plot_marginal_histogram(
            histogram_ax,
            group_categories,
            value_position=0,
            bin_width=bin_width,
            bin_offset_fraction=bin_offset_fraction,
            xlabel="",
            title=fill(group_title, width=32),
            show_legend=not legend_shown,
        )
        legend_shown = True
        _plot_ecdf(
            ecdf_ax,
            group_categories,
            value_position=0,
            xlabel="Mean normalized activity\n" + fill(f"({count_label})", width=38),
        )

    fig.suptitle(
        _session_activity_figure_title(
            session_activity,
            "on/off states" if comparison == "state" else "all delay bins by cue",
            marginal=True,
        )
    )
    return fig


def save_session_activity_figures(
    session_activity: SessionActivity,
    figure_dir: Path,
    config: Config,
    *,
    point_seed: int,
    filename_prefix: str = "",
) -> None:
    """Save all state/cue figure families through the shared export helper."""
    for comparison, filename_group in (
        ("state", "states"),
        ("cue", "cues"),
    ):
        fig = plot_session_activity(
            session_activity,
            comparison=comparison,
            compare_with_max_off_state=config.compare_with_max_off_state,
            hide_opposite_cue_points=config.hide_opposite_cue_points,
            hide_all_preferred_cue_points=(
                config.hide_all_preferred_cue_points
            ),
            max_points_per_color_group=config.max_points_per_color_group,
            max_points_per_max_off_state=(
                config.max_points_per_max_off_state
            ),
            point_seed=point_seed,
        )
        save_figure(
            fig,
            figure_dir
            / ("principal_components" if filename_prefix else "activity")
            / filename_group
            / (
                f"{filename_prefix}activity_across_{filename_group}_"
                f"{session_activity.session}.png"
            ),
            dpi=config.figure_dpi,
        )
        plt.close(fig)

        pairwise_fig = plot_session_activity_pairwise(
            session_activity,
            comparison=comparison,
            compare_with_max_off_state=config.compare_with_max_off_state,
            hide_opposite_cue_points=config.hide_opposite_cue_points,
            hide_all_preferred_cue_points=(
                config.hide_all_preferred_cue_points
            ),
            max_points_per_color_group=config.max_points_per_color_group,
            max_points_per_max_off_state=(
                config.max_points_per_max_off_state
            ),
            point_seed=point_seed,
        )
        save_figure(
            pairwise_fig,
            figure_dir
            / ("principal_components" if filename_prefix else "activity")
            / filename_group
            / (
                f"{filename_prefix}activity_across_{filename_group}_"
                f"pairwise_{session_activity.session}.png"
            ),
            dpi=config.figure_dpi,
        )
        plt.close(pairwise_fig)

        marginal_fig = plot_session_activity_marginal_histograms(
            session_activity,
            comparison=comparison,
            compare_with_max_off_state=config.compare_with_max_off_state,
            hide_opposite_cue_points=config.hide_opposite_cue_points,
            hide_all_preferred_cue_points=(
                config.hide_all_preferred_cue_points
            ),
            max_points_per_max_off_state=(
                config.max_points_per_max_off_state
            ),
            point_seed=point_seed,
            bin_width=config.marginal_histogram_bin_width,
            bin_offset_fraction=(
                config.marginal_histogram_bin_offset_fraction
            ),
        )
        save_figure(
            marginal_fig,
            figure_dir
            / ("principal_components" if filename_prefix else "activity")
            / filename_group
            / (
                f"{filename_prefix}activity_across_{filename_group}_"
                f"marginals_{session_activity.session}.png"
            ),
            dpi=config.figure_dpi,
        )
        plt.close(marginal_fig)
