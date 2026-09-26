"""Typed settings and activity metadata shared by preparation and figures."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from scripts.next.screening_metadata import POPULATION_KEYS as POPULATION_GROUP_KEYS


@dataclass
class Config:
    """Input locations and plotting settings."""

    data_dir: Path = Path("data/nature")
    cache_dir: Path = Path("cache/next_run")
    output_subdir: str = ""  # relative to this stage under cache_dir
    activity_bin_width_ms: float = 50.0
    seed: int = 42
    figure_dpi: int = 300
    hide_opposite_cue_points: bool = False
    hide_all_preferred_cue_points: bool = False
    compare_with_max_off_state: bool = False
    # Add parallel plots of up to three PCs fitted across all preferred cells.
    show_principal_components: bool = False
    # This general color-group cap never applies to the red maximum-state points.
    max_points_per_color_group: int | None = None
    # The red points have their own independent, per-session cap; None keeps all.
    max_points_per_max_off_state: int | None = None
    # Use this fixed normalized-activity bin width for marginal histograms.
    marginal_histogram_bin_width: float = 0.25
    # Horizontally separate adjacent histogram outlines by this bin-width fraction.
    marginal_histogram_bin_offset_fraction: float = 0.2
    # Use PEV weights for selected-group means; the remaining group stays equal.
    pev_weighted_average: bool = False


@dataclass(frozen=True)
class CellActivityDimensions:
    """Cell IDs paired only with their screening selectivity effect sizes."""

    cell_ids: np.ndarray
    selectivity_pev_pct: np.ndarray

    def __post_init__(self):
        if (self.cell_ids.ndim != 1
                or self.cell_ids.shape != self.selectivity_pev_pct.shape):
            raise ValueError(
                "Cell IDs and selectivity PEV must be aligned one-dimensional arrays."
            )

    @property
    def count(self) -> int:
        return int(self.cell_ids.size)


@dataclass(frozen=True)
class PrincipalComponentDimensions:
    """PC numbers paired with PCA explained variance, never cell selectivity."""

    component_numbers: np.ndarray
    explained_variance_ratio: np.ndarray
    source_cell_count: int

    def __post_init__(self):
        if (self.component_numbers.ndim != 1
                or self.component_numbers.shape != self.explained_variance_ratio.shape):
            raise ValueError(
                "PC numbers and explained variance must be aligned one-dimensional arrays."
            )

    @property
    def count(self) -> int:
        return int(self.component_numbers.size)


@dataclass
class PrincipalComponentActivity:
    """A shared PCA projection of one session's preferred-cell activity."""

    preferred_activity: np.ndarray
    opposite_activity: np.ndarray
    max_off_state_activity: np.ndarray
    components: np.ndarray
    center: np.ndarray
    explained_variance_ratio: np.ndarray
    source_cell_count: int


@dataclass
class SessionActivity:
    """Normalized activity and state labels prepared for one session."""

    session: str
    preferred_cue: int
    opposite_cue: int
    dimensions: CellActivityDimensions | PrincipalComponentDimensions
    delay_bin_starts: np.ndarray
    preferred_activity: np.ndarray
    opposite_activity: np.ndarray
    on_state_mask: np.ndarray
    off_state_mask: np.ndarray
    preferred_trial_ids: np.ndarray
    opposite_trial_ids: np.ndarray
    preferred_population_mean_activity: np.ndarray | None = None
    opposite_population_mean_activity: np.ndarray | None = None
    preferred_population_cell_count: int = 0
    population_mean_activities: dict[
        str,
        tuple[np.ndarray | None, np.ndarray | None, int],
    ] = field(default_factory=dict)
    max_off_state_activity: np.ndarray | None = None
    max_off_state_population_mean_activities: dict[
        str,
        np.ndarray | None,
    ] = field(default_factory=dict)
    max_off_state_trial_id: int | None = None
    max_off_state_delay_bin_starts: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=float)
    )
    principal_component_activity: PrincipalComponentActivity | None = None
    screening_checks: dict[str, bool] | None = None

    @property
    def dimension_count(self) -> int:
        return self.dimensions.count

    @property
    def activity_space(self) -> Literal["cells", "principal_components"]:
        if isinstance(self.dimensions, PrincipalComponentDimensions):
            return "principal_components"
        return "cells"

    @property
    def activity_source_cell_count(self) -> int:
        if isinstance(self.dimensions, PrincipalComponentDimensions):
            return self.dimensions.source_cell_count
        return self.dimensions.count


def get_opposite_cue(cue: int) -> int:
    """Return the cue index opposite to a cue numbered from 1 through 8."""
    return (int(cue) + 3) % 8 + 1


def cue_to_deg(cue: int) -> int:
    """Map a cue index from 1 through 8 to its displayed angle."""
    return int(((int(cue) - 1) % 8) * 45 - 135)
