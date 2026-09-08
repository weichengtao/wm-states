"""Shared policies for averaging normalized activity across cell groups."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np


PEV_WEIGHTED_SUBDIR = "pev_weighted"
SELECTIVE_GROUPS = ("preferred", "selective_nonpreferred")
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
        "preferred": "mean_pev_test",
        "selective_nonpreferred": "mean_pev_test",
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
    """Return PEV weights for selective groups and equal weights otherwise.

    A ``None`` value represents equal weighting. Empty selective groups receive
    an empty weight vector so callers can retain their existing zero-cell logic.
    """
    weights: dict[str, np.ndarray | None] = {
        group_name: None for group_name in cell_groups
    }
    if not pev_weighted_average:
        return weights

    properties = selection_result.get("cell_properties", {})
    selective_cells = np.asarray(
        properties.get("cell_idx", []), dtype=np.int64
    ).ravel()
    selective_pev = np.asarray(
        properties.get("mean_pev_test", []), dtype=float
    ).ravel()
    if selective_cells.shape != selective_pev.shape:
        raise ValueError(
            "cell_idx and mean_pev_test must have matching shapes for "
            "PEV-weighted activity."
        )
    if np.unique(selective_cells).size != selective_cells.size:
        raise ValueError("Selective cell IDs must be unique for PEV weighting.")
    pev_by_cell = dict(zip(selective_cells.tolist(), selective_pev.tolist()))

    for group_name in SELECTIVE_GROUPS:
        cell_ids = np.asarray(cell_groups.get(group_name, []), dtype=np.int64).ravel()
        missing = [
            int(cell_id)
            for cell_id in cell_ids
            if int(cell_id) not in pev_by_cell
        ]
        if missing:
            raise ValueError(
                f"Missing mean_pev_test for {group_name} cells: {missing}"
            )
        group_weights = np.asarray(
            [pev_by_cell[int(cell_id)] for cell_id in cell_ids], dtype=float
        )
        if group_weights.size:
            if not np.all(np.isfinite(group_weights)):
                raise ValueError(
                    f"Non-finite mean_pev_test values in {group_name} group."
                )
            if np.any(group_weights < 0):
                raise ValueError(
                    f"Negative mean_pev_test values in {group_name} group."
                )
            if float(np.sum(group_weights)) <= 0:
                raise ValueError(
                    f"mean_pev_test weights sum to zero in {group_name} group."
                )
        weights[group_name] = group_weights

    # PEV estimates are deliberately not used for stationary-nonselective cells.
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
