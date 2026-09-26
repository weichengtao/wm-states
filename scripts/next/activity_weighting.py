"""Shared policies for averaging normalized activity across cell groups."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scripts.next.screening_metadata import (
    SELECTED_POPULATION_KEYS,
    ScreeningMetadata,
    validate_cell_ids,
)


PEV_WEIGHTED_SUBDIR = "pev_weighted"
STATIONARY_NONSELECTIVE = "stationary_nonselective"


def weighting_mode(pev_weighted_average: bool) -> str:
    """Return the stable name recorded in paths and metadata."""
    return "pev_weighted" if pev_weighted_average else "equal"


def weighting_policy(pev_weighted_average: bool) -> dict[str, str]:
    """Describe how each population is averaged."""
    if not pev_weighted_average:
        return {
            "preferred": "equal",
            "selective_nonpreferred": "equal",
            STATIONARY_NONSELECTIVE: "equal",
        }
    return {
        "preferred": "mean_selectivity_pev_pct",
        "selective_nonpreferred": "mean_selectivity_pev_pct",
        STATIONARY_NONSELECTIVE: "equal",
    }


def weighting_subdir(path: str | Path, pev_weighted_average: bool) -> Path:
    """Append the isolated weighted-results directory when requested."""
    resolved = Path(path)
    if pev_weighted_average:
        resolved /= PEV_WEIGHTED_SUBDIR
    return resolved


def cell_group_activity_weights(
    selection_result: Mapping[str, Any],
    cell_groups: Mapping[str, np.ndarray],
    pev_weighted_average: bool,
) -> dict[str, np.ndarray | None]:
    """Return PEV weights for selected groups and equal weights otherwise.

    A ``None`` value represents equal weighting. Empty selected groups receive
    an empty weight vector so callers can retain their existing zero-cell logic.
    """
    weights: dict[str, np.ndarray | None] = {
        group_name: None for group_name in cell_groups
    }
    if not pev_weighted_average:
        return weights

    metadata = ScreeningMetadata(selection_result)
    pev_by_cell = dict(zip(
        metadata.selected_cell_ids.tolist(), metadata.selectivity_pev_pct.tolist()
    ))

    for group_name in SELECTED_POPULATION_KEYS:
        cell_ids = validate_cell_ids(cell_groups.get(group_name, []), group_name)
        missing = [
            int(cell_id)
            for cell_id in cell_ids
            if int(cell_id) not in pev_by_cell
        ]
        if missing:
            raise ValueError(
                f"Missing mean_selectivity_pev_pct for {group_name} cells: {missing}"
            )
        group_weights = np.asarray(
            [pev_by_cell[int(cell_id)] for cell_id in cell_ids], dtype=float
        )
        if group_weights.size:
            if not np.all(np.isfinite(group_weights)):
                raise ValueError(
                    f"Non-finite mean_selectivity_pev_pct values in {group_name} group."
                )
            if np.any(group_weights < 0):
                raise ValueError(
                    f"Negative mean_selectivity_pev_pct values in {group_name} group."
                )
            if float(np.sum(group_weights)) <= 0:
                raise ValueError(
                    f"mean_selectivity_pev_pct weights sum to zero in {group_name} group."
                )
        weights[group_name] = group_weights

    # PEV estimates are deliberately not used for cells outside the selected pool.
    if STATIONARY_NONSELECTIVE in cell_groups:
        weights[STATIONARY_NONSELECTIVE] = None
    return weights


def mean_cell_activity(
    activity: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    axis: int = -1,
) -> np.ndarray:
    """Average a cell axis equally or with validated nonnegative weights."""
    activity = np.asarray(activity, dtype=float)
    if activity.ndim == 0 or not -activity.ndim <= axis < activity.ndim:
        raise ValueError(f"Invalid cell axis {axis} for shape {activity.shape}.")
    resolved_axis = axis % activity.ndim
    num_cells = activity.shape[resolved_axis]
    output_shape = activity.shape[:resolved_axis] + activity.shape[resolved_axis + 1 :]
    if num_cells == 0:
        return np.zeros(output_shape, dtype=float)
    if weights is None:
        return np.mean(activity, axis=resolved_axis)

    weights = np.asarray(weights, dtype=float).ravel()
    if weights.size != num_cells:
        raise ValueError(
            f"Activity has {num_cells} cells but received {weights.size} weights."
        )
    if not np.all(np.isfinite(weights)):
        raise ValueError("Activity weights must be finite.")
    if np.any(weights < 0):
        raise ValueError("Activity weights must be nonnegative.")
    if float(np.sum(weights)) <= 0:
        raise ValueError("Activity weights must have a positive sum.")
    return np.average(activity, axis=resolved_axis, weights=weights)
