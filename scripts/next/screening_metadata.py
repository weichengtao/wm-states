"""Validated access to screening metadata and shared downstream populations.

Persisted population keys retain their established names. Their membership is
defined by screening results, even when the selectivity gate is disabled; use
``population_labels`` for descriptions that reflect the recorded checks.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np


POPULATION_KEYS = ("preferred", "selective_nonpreferred", "stationary_nonselective")
SELECTED_POPULATION_KEYS = POPULATION_KEYS[:2]


def _numeric_vector(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain numeric values.")
    return array


def validate_cell_ids(
    values: Any, name: str, *, num_cells_total: int | None = None
) -> np.ndarray:
    """Validate cell IDs without silently truncating or wrapping invalid values."""
    cell_ids = _numeric_vector(values, name)
    if (
        not np.all(np.isfinite(cell_ids))
        or np.any(cell_ids < 0)
        or np.any(cell_ids >= 2**63)
        or np.any(cell_ids != np.floor(cell_ids))
    ):
        raise ValueError(f"{name} must contain nonnegative integer cell IDs.")
    cell_ids = cell_ids.astype(np.int64, copy=False)
    if np.unique(cell_ids).size != cell_ids.size:
        raise ValueError(f"{name} cell IDs must be unique.")
    if num_cells_total is not None and np.any(cell_ids >= num_cells_total):
        raise ValueError(
            f"{name} contains cell IDs outside the session's {num_cells_total} cells."
        )
    return cell_ids


def _cue_indices(values: Any, name: str) -> np.ndarray:
    cue_indices = _numeric_vector(values, name)
    if (
        not np.all(np.isfinite(cue_indices))
        or np.any(cue_indices < 1)
        or np.any(cue_indices > 8)
        or np.any(cue_indices != np.floor(cue_indices))
    ):
        raise ValueError(f"{name} must contain integer cue indices from 1 through 8.")
    return cue_indices.astype(np.int64, copy=False)


def validate_cue(cue: int, name: str = "preferred_cue") -> int:
    """Validate a single encoded cue before matching population metadata."""
    return int(_cue_indices([cue], name)[0])


@dataclass(frozen=True)
class ScreeningMetadata:
    """Access only the fields required by an analysis, validating their alignment.

    Decoder pools that do not use cue preferences or PEV do not require those
    properties. Likewise, weighting requires PEV but not cue preferences. PEV
    may be nonfinite here: the consuming analysis decides whether to filter it
    (ranked activity views) or reject it (weighted averages).
    """

    selection: Mapping[str, Any]
    num_cells_total: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.selection, Mapping):
            raise ValueError("Screening metadata must be a mapping.")
        if self.num_cells_total is not None and (
            isinstance(self.num_cells_total, (bool, np.bool_))
            or not isinstance(self.num_cells_total, (int, np.integer))
            or self.num_cells_total < 0
        ):
            raise ValueError("num_cells_total must be a nonnegative integer.")

    def _required(self, source: Mapping[str, Any], name: str) -> Any:
        if name not in source:
            raise ValueError(
                f"The screening cache does not contain {name}. "
                "Rerun scripts/next/cell_screening.py."
            )
        return source[name]

    @cached_property
    def _properties(self) -> Mapping[str, Any]:
        properties = self._required(self.selection, "cell_properties")
        if not isinstance(properties, Mapping):
            raise ValueError("cell_properties must be a mapping.")
        return properties

    @cached_property
    def selected_cell_ids(self) -> np.ndarray:
        cell_ids = validate_cell_ids(
            self._required(self._properties, "cell_idx"), "cell_idx",
            num_cells_total=self.num_cells_total,
        )
        if "cell_idx_selected" in self.selection:
            recorded_ids = validate_cell_ids(
                self.selection["cell_idx_selected"], "cell_idx_selected",
                num_cells_total=self.num_cells_total,
            )
            if not np.array_equal(cell_ids, recorded_ids):
                raise ValueError("cell_idx and cell_idx_selected must contain the same ordered cell IDs.")
        return cell_ids

    def _aligned_property(self, name: str) -> np.ndarray:
        values = _numeric_vector(self._required(self._properties, name), name)
        if values.shape != self.selected_cell_ids.shape:
            raise ValueError(f"cell_idx and {name} must have matching shapes.")
        return values

    @cached_property
    def preferred_cues(self) -> np.ndarray:
        return _cue_indices(self._aligned_property("preferred_cue"), "preferred_cue")

    @cached_property
    def selectivity_pev_pct(self) -> np.ndarray:
        return self._aligned_property("mean_selectivity_pev_pct").astype(float, copy=False)

    @cached_property
    def stationary_cell_ids(self) -> np.ndarray:
        return validate_cell_ids(
            self._required(self.selection, "cell_idx_stationary"), "cell_idx_stationary",
            num_cells_total=self.num_cells_total,
        )

    @cached_property
    def presence_passed_cell_ids(self) -> np.ndarray:
        return validate_cell_ids(
            self._required(self.selection, "cell_idx_passed_presence_ratio"),
            "cell_idx_passed_presence_ratio", num_cells_total=self.num_cells_total,
        )


def _ranked_preferred_cells(
    metadata: ScreeningMetadata, preferred_cue: int
) -> tuple[np.ndarray, np.ndarray]:
    eligible = (
        (metadata.preferred_cues == preferred_cue)
        & np.isfinite(metadata.selectivity_pev_pct)
    )
    cell_ids = metadata.selected_cell_ids[eligible]
    pev_pct = metadata.selectivity_pev_pct[eligible]
    order = np.argsort(-pev_pct, kind="stable")
    return cell_ids[order], pev_pct[order]


def preferred_pev_cells(
    selection: Mapping[str, Any], preferred_cue: int, *, num_cells_total: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return preferred selected cells with finite PEV, stably ranked by PEV."""
    return _ranked_preferred_cells(
        ScreeningMetadata(selection, num_cells_total=num_cells_total), validate_cue(preferred_cue)
    )


def cell_groups(
    selection: Mapping[str, Any], preferred_cue: int, *, rank_preferred_by_pev: bool = False,
    num_cells_total: int | None = None,
) -> dict[str, np.ndarray]:
    """Build established populations without changing stage-specific ordering.

    The default keeps screening order and does not require PEV, as trial-table
    preparation has always done. Activity views opt into finite-PEV filtering
    and stable descending PEV order for the preferred group only. The remaining
    groups retain screening order in either mode.
    """
    preferred_cue = validate_cue(preferred_cue)
    metadata = ScreeningMetadata(selection, num_cells_total=num_cells_total)
    selected_cell_ids = metadata.selected_cell_ids
    stationary_cell_ids = metadata.stationary_cell_ids
    if not np.all(np.isin(selected_cell_ids, stationary_cell_ids)):
        raise ValueError("Selected cell_idx must be a subset of cell_idx_stationary.")
    is_preferred = metadata.preferred_cues == preferred_cue
    preferred_cell_ids = (
        _ranked_preferred_cells(metadata, preferred_cue)[0]
        if rank_preferred_by_pev else selected_cell_ids[is_preferred]
    )
    return {
        "preferred": preferred_cell_ids,
        "selective_nonpreferred": selected_cell_ids[~is_preferred],
        "stationary_nonselective": stationary_cell_ids[
            ~np.isin(stationary_cell_ids, selected_cell_ids)
        ],
    }


def population_labels(screening_checks: Mapping[str, Any] | None) -> dict[str, str]:
    """Describe recorded selection without inferring disabled or unknown checks."""
    if screening_checks is not None and not isinstance(screening_checks, Mapping):
        raise ValueError("screening_checks must be a mapping or None.")
    selectivity_enabled = None if screening_checks is None else screening_checks.get("selectivity")
    if selectivity_enabled is not None and not isinstance(selectivity_enabled, (bool, np.bool_)):
        raise ValueError("screening_checks.selectivity must be a boolean.")
    if selectivity_enabled:
        return {
            "preferred": "Selective preferred cells",
            "selective_nonpreferred": "Selective non-preferred cells",
            "stationary_nonselective": "Cells failing selectivity, passing other checks",
        }
    return {
        "preferred": "Selected preferred cells",
        "selective_nonpreferred": "Selected non-preferred cells",
        "stationary_nonselective": "Other cells passing enabled checks",
    }
