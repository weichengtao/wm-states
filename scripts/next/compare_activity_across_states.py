"""Compare preferred-cell activity across states and cue groups.

For each session, this script selects up to three preferred-cue cells with the
largest cached delay-period PEV, balances correct preferred- and opposite-cue
trials, and z-normalizes each cell across the balanced trials independently at
every delay-bin start. Separate figures compare preferred-cue on/off states and
preferred/opposite-cue all-delay-bin populations. The state figures highlight
the delay bins in each session's longest off state. Optionally, it also plots
the three highest-variance principal components of all preferred cells using a
single PCA basis fitted to the pooled balanced cue groups.
"""

from __future__ import annotations

# Use one module namespace for direct CLI and package execution.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "scripts.next"


from scripts.next import cache_io as pickle
from scripts.next.common import full_session_selection
from dataclasses import dataclass, field, replace
from itertools import combinations
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tyro
from scipy.io import loadmat
from sklearn.decomposition import PCA

from scripts.next.activity_weighting import (
    cell_group_activity_weights,
    mean_cell_activity,
    weighting_subdir,
)
from scripts.next.figure_exports import configure_figure_style, save_figure_all_formats


configure_figure_style(matplotlib)


POPULATION_GROUPS = (
    ("preferred", "All preferred cells"),
    ("selective_nonpreferred", "Selective non-preferred cells"),
    ("stationary_nonselective", "Stationary non-selective cells"),
)


@dataclass
class Config:
    """Input locations and plotting settings."""

    data_dir: Path = Path("data/nature")
    cache_dir: Path = Path("cache/next_run")
    output_subdir: str = "compare_activity_across_states"
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
    # Use PEV weights for selective means; stationary-nonselective stays equal.
    pev_weighted_average: bool = False


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
    cell_ids: np.ndarray
    cell_pev: np.ndarray
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
    activity_space: Literal["cells", "principal_components"] = "cells"
    activity_source_cell_count: int = 0


def _load_pickle(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing cache file: {path}")
    with path.open("rb") as handle:
        return pickle.load(handle)


def get_opposite_cue(cue: int) -> int:
    """Return the cue index opposite to a cue numbered from 1 through 8."""
    return (int(cue) + 3) % 8 + 1


def cue_to_deg(cue: int) -> int:
    """Map a cue index from 1 through 8 to its displayed angle."""
    return int(((int(cue) - 1) % 8) * 45 - 135)


def find_full_session_selection(selection_results, session, num_trials):
    return full_session_selection(selection_results, session, num_trials)


def preferred_pev_cells(
    selection_result: dict[str, Any],
    preferred_cue: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return all finite-PEV cells preferred for the session cue, PEV-ranked."""
    properties = selection_result.get("cell_properties", {})
    cell_ids = np.asarray(properties.get("cell_idx", []), dtype=np.int64).ravel()
    preferred_cues = np.asarray(properties.get("mean_pref_test", [])).ravel()
    mean_pev = np.asarray(properties.get("mean_pev_test", []), dtype=float).ravel()
    if not (cell_ids.shape == preferred_cues.shape == mean_pev.shape):
        raise ValueError(
            "cell_idx, mean_pref_test, and mean_pev_test must have matching shapes."
        )

    eligible = (preferred_cues == preferred_cue) & np.isfinite(mean_pev)
    eligible_ids = cell_ids[eligible]
    eligible_pev = mean_pev[eligible]
    order = np.argsort(-eligible_pev, kind="stable")
    return eligible_ids[order], eligible_pev[order]


def top_preferred_pev_cells(
    selection_result: dict[str, Any],
    preferred_cue: int,
    count: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Return up to ``count`` highest-PEV cells preferred for the session cue."""
    cell_ids, mean_pev = preferred_pev_cells(selection_result, preferred_cue)
    return cell_ids[:count], mean_pev[:count]


def session_cell_groups(
    selection_result: dict[str, Any],
    preferred_cue: int,
) -> dict[str, np.ndarray]:
    """Return preferred, selective non-preferred, and stationary groups."""
    properties = selection_result.get("cell_properties", {})
    selective_cells = np.asarray(
        properties.get("cell_idx", []),
        dtype=np.int64,
    ).ravel()
    preferred_cues = np.asarray(properties.get("mean_pref_test", [])).ravel()
    if selective_cells.shape != preferred_cues.shape:
        raise ValueError("cell_idx and mean_pref_test must have matching shapes.")
    preferred_cells, _ = preferred_pev_cells(selection_result, preferred_cue)
    stationary_cells = np.asarray(
        selection_result.get("cell_idx_stationary", []),
        dtype=np.int64,
    ).ravel()
    return {
        "preferred": preferred_cells,
        "selective_nonpreferred": selective_cells[
            preferred_cues != preferred_cue
        ],
        "stationary_nonselective": stationary_cells[
            ~np.isin(stationary_cells, selective_cells)
        ],
    }


def balance_trial_groups(
    preferred_trial_ids: np.ndarray,
    opposite_trial_ids: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subsample both groups equally and return preferred row positions too."""
    preferred_trial_ids = np.asarray(preferred_trial_ids, dtype=np.int64).ravel()
    opposite_trial_ids = np.asarray(opposite_trial_ids, dtype=np.int64).ravel()
    if preferred_trial_ids.size == 0 or opposite_trial_ids.size == 0:
        raise ValueError("Both preferred- and opposite-cue trial groups must be non-empty.")
    if np.unique(preferred_trial_ids).size != preferred_trial_ids.size:
        raise ValueError("Preferred-cue trial IDs must be unique.")
    if np.unique(opposite_trial_ids).size != opposite_trial_ids.size:
        raise ValueError("Opposite-cue trial IDs must be unique.")

    balanced_count = min(preferred_trial_ids.size, opposite_trial_ids.size)
    rng = np.random.default_rng(seed)
    if preferred_trial_ids.size == balanced_count:
        preferred_positions = np.arange(balanced_count, dtype=np.int64)
    else:
        preferred_positions = np.sort(
            rng.choice(preferred_trial_ids.size, size=balanced_count, replace=False)
        )
    if opposite_trial_ids.size == balanced_count:
        opposite_positions = np.arange(balanced_count, dtype=np.int64)
    else:
        opposite_positions = np.sort(
            rng.choice(opposite_trial_ids.size, size=balanced_count, replace=False)
        )
    return (
        preferred_positions,
        preferred_trial_ids[preferred_positions],
        opposite_trial_ids[opposite_positions],
    )


def maximum_delay_off_state_mask(
    off_state_mask: np.ndarray,
    delay_bins: np.ndarray,
) -> np.ndarray:
    """Return only the longest contiguous off-state overlap with the delay."""
    off_state_mask = np.asarray(off_state_mask, dtype=bool)
    delay_bins = np.asarray(delay_bins, dtype=bool).ravel()
    if off_state_mask.ndim != 2:
        raise ValueError("off_state_mask must be two-dimensional.")
    if off_state_mask.shape[1] != delay_bins.size:
        raise ValueError("delay_bins must match the off-state mask time dimension.")

    maximum_mask = np.zeros_like(off_state_mask, dtype=bool)
    maximum_overlap_count = 0
    for trial_position, trial_mask in enumerate(off_state_mask):
        transitions = np.diff(np.pad(trial_mask.astype(np.int8), (1, 1)))
        run_starts = np.flatnonzero(transitions == 1)
        run_ends = np.flatnonzero(transitions == -1)
        for run_start, run_end in zip(run_starts, run_ends):
            run_bins = np.arange(run_start, run_end, dtype=np.int64)
            delay_run_bins = run_bins[delay_bins[run_bins]]
            if delay_run_bins.size > maximum_overlap_count:
                maximum_mask.fill(False)
                maximum_mask[trial_position, delay_run_bins] = True
                maximum_overlap_count = int(delay_run_bins.size)

    return maximum_mask


def compute_binned_firing_rates(
    spikes: np.ndarray,
    times_ms: np.ndarray,
    trial_ids: np.ndarray,
    cell_ids: np.ndarray,
    bin_starts: np.ndarray,
    bin_width_ms: float,
) -> np.ndarray:
    """Return firing rates shaped as trial by bin by cell."""
    spikes = np.asarray(spikes)
    times_ms = np.asarray(times_ms, dtype=float).ravel()
    trial_ids = np.asarray(trial_ids, dtype=np.int64).ravel()
    cell_ids = np.asarray(cell_ids, dtype=np.int64).ravel()
    bin_starts = np.asarray(bin_starts, dtype=float).ravel()
    if spikes.ndim != 3 or spikes.shape[1] != times_ms.size:
        raise ValueError("Spike data must have shape (trial, time, cell).")
    if not np.isfinite(bin_width_ms) or bin_width_ms <= 0:
        raise ValueError("activity_bin_width_ms must be finite and positive.")
    if np.any(trial_ids < 0) or np.any(trial_ids >= spikes.shape[0]):
        raise ValueError("Trial IDs are outside the spike array.")
    if np.any(cell_ids < 0) or np.any(cell_ids >= spikes.shape[2]):
        raise ValueError("Cell IDs are outside the spike array.")

    rates = np.empty((trial_ids.size, bin_starts.size, cell_ids.size), dtype=float)
    duration_seconds = bin_width_ms / 1000.0
    for bin_idx, bin_start in enumerate(bin_starts):
        time_ids = np.flatnonzero(
            (times_ms >= bin_start) & (times_ms < bin_start + bin_width_ms)
        )
        if time_ids.size == 0:
            raise ValueError(
                f"No samples found in activity bin [{bin_start:g}, "
                f"{bin_start + bin_width_ms:g}) ms."
            )
        rates[:, bin_idx, :] = (
            spikes[np.ix_(trial_ids, time_ids, cell_ids)].sum(axis=1)
            / duration_seconds
        )
    if not np.all(np.isfinite(rates)):
        raise ValueError("Binned firing rates contain non-finite values.")
    return rates


def normalize_balanced_activity(
    preferred_rates: np.ndarray,
    opposite_rates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Z-normalize each bin/cell across the combined balanced trial groups."""
    means, stds = balanced_activity_normalization_parameters(
        preferred_rates,
        opposite_rates,
    )
    preferred_normalized = apply_activity_normalization(preferred_rates, means, stds)
    opposite_normalized = apply_activity_normalization(opposite_rates, means, stds)
    return preferred_normalized, opposite_normalized


def balanced_activity_normalization_parameters(
    preferred_rates: np.ndarray,
    opposite_rates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-bin/cell normalization parameters to balanced cue groups."""
    preferred_rates = np.asarray(preferred_rates, dtype=float)
    opposite_rates = np.asarray(opposite_rates, dtype=float)
    if preferred_rates.shape != opposite_rates.shape:
        raise ValueError("Balanced preferred and opposite activity must match in shape.")
    if preferred_rates.ndim != 3:
        raise ValueError("Activity must have shape (trial, bin, cell).")
    combined = np.concatenate([preferred_rates, opposite_rates], axis=0)
    means = np.mean(combined, axis=0)
    stds = np.std(combined, axis=0, ddof=0)
    return means, stds


def apply_activity_normalization(
    rates: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
) -> np.ndarray:
    """Apply fitted per-bin/cell normalization, mapping zero variance to zero."""
    rates = np.asarray(rates, dtype=float)
    means = np.asarray(means, dtype=float)
    stds = np.asarray(stds, dtype=float)
    if rates.ndim != 3 or rates.shape[1:] != means.shape or means.shape != stds.shape:
        raise ValueError("Rates and normalization parameters have incompatible shapes.")
    normalized = np.zeros_like(rates, dtype=float)
    usable = np.isfinite(means) & np.isfinite(stds) & (stds > 0)
    normalized[:, usable] = (rates[:, usable] - means[usable]) / stds[usable]
    return normalized


def compute_preferred_cell_principal_components(
    preferred_activity: np.ndarray,
    opposite_activity: np.ndarray,
    max_off_state_activity: np.ndarray,
    count: int = 3,
) -> PrincipalComponentActivity:
    """Fit pooled balanced-cue PCA and project each preferred-cell point set."""
    preferred_activity = np.asarray(preferred_activity, dtype=float)
    opposite_activity = np.asarray(opposite_activity, dtype=float)
    max_off_state_activity = np.asarray(max_off_state_activity, dtype=float)
    if preferred_activity.shape != opposite_activity.shape:
        raise ValueError(
            "Balanced preferred and opposite activity must match in shape."
        )
    if preferred_activity.ndim != 3:
        raise ValueError("Activity must have shape (trial, bin, cell).")
    if (
        max_off_state_activity.ndim != 2
        or max_off_state_activity.shape[1] != preferred_activity.shape[2]
    ):
        raise ValueError(
            "Maximum off-state activity must have shape (bin, cell)."
        )
    if (
        isinstance(count, (bool, np.bool_))
        or not isinstance(count, (int, np.integer))
        or count <= 0
    ):
        raise ValueError("Principal-component count must be positive.")
    if not (
        np.all(np.isfinite(preferred_activity))
        and np.all(np.isfinite(opposite_activity))
        and np.all(np.isfinite(max_off_state_activity))
    ):
        raise ValueError("PCA activity inputs must contain only finite values.")

    source_cell_count = int(preferred_activity.shape[2])
    observation_count_per_cue = int(np.prod(preferred_activity.shape[:2]))
    if source_cell_count and observation_count_per_cue == 0:
        raise ValueError("PCA requires at least one activity observation per cue.")
    component_count = min(
        count,
        source_cell_count,
        observation_count_per_cue * 2,
    )
    if component_count == 0:
        empty_preferred = np.empty((*preferred_activity.shape[:2], 0), dtype=float)
        empty_opposite = np.empty((*opposite_activity.shape[:2], 0), dtype=float)
        empty_max_off_state = np.empty((max_off_state_activity.shape[0], 0))
        return PrincipalComponentActivity(
            preferred_activity=empty_preferred,
            opposite_activity=empty_opposite,
            max_off_state_activity=empty_max_off_state,
            components=np.empty((0, source_cell_count), dtype=float),
            center=np.empty(source_cell_count, dtype=float),
            explained_variance_ratio=np.empty(0, dtype=float),
            source_cell_count=source_cell_count,
        )

    preferred_points = preferred_activity.reshape(-1, source_cell_count)
    opposite_points = opposite_activity.reshape(-1, source_cell_count)
    pooled_points = np.concatenate([preferred_points, opposite_points], axis=0)
    center = np.mean(pooled_points, axis=0)
    if not np.any(pooled_points != center):
        components = np.eye(source_cell_count, dtype=float)[:component_count]
        pooled_scores = np.zeros(
            (pooled_points.shape[0], component_count),
            dtype=float,
        )
        max_off_state_scores = (max_off_state_activity - center) @ components.T
        explained_variance_ratio = np.zeros(component_count, dtype=float)
        preferred_point_count = preferred_points.shape[0]
        return PrincipalComponentActivity(
            preferred_activity=pooled_scores[:preferred_point_count].reshape(
                *preferred_activity.shape[:2],
                component_count,
            ),
            opposite_activity=pooled_scores[preferred_point_count:].reshape(
                *opposite_activity.shape[:2],
                component_count,
            ),
            max_off_state_activity=max_off_state_scores,
            components=components,
            center=center,
            explained_variance_ratio=explained_variance_ratio,
            source_cell_count=source_cell_count,
        )
    pca = PCA(n_components=component_count, svd_solver="full")
    pooled_scores = pca.fit_transform(pooled_points)
    preferred_point_count = preferred_points.shape[0]
    explained_variance_ratio = np.nan_to_num(
        pca.explained_variance_ratio_,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return PrincipalComponentActivity(
        preferred_activity=pooled_scores[:preferred_point_count].reshape(
            *preferred_activity.shape[:2],
            component_count,
        ),
        opposite_activity=pooled_scores[preferred_point_count:].reshape(
            *opposite_activity.shape[:2],
            component_count,
        ),
        max_off_state_activity=pca.transform(max_off_state_activity),
        components=pca.components_.copy(),
        center=pca.mean_.copy(),
        explained_variance_ratio=explained_variance_ratio,
        source_cell_count=source_cell_count,
    )


def principal_component_session_activity(
    session_activity: SessionActivity,
) -> SessionActivity:
    """Return a plotting view backed by a session's PCA scores."""
    projection = session_activity.principal_component_activity
    if projection is None:
        raise ValueError("Session activity does not contain a PCA projection.")
    component_count = projection.preferred_activity.shape[2]
    return replace(
        session_activity,
        cell_ids=np.arange(1, component_count + 1, dtype=np.int64),
        cell_pev=projection.explained_variance_ratio * 100.0,
        preferred_activity=projection.preferred_activity,
        opposite_activity=projection.opposite_activity,
        max_off_state_activity=projection.max_off_state_activity,
        activity_space="principal_components",
        activity_source_cell_count=projection.source_cell_count,
    )


def prepare_session_activity(
    state_result: dict[str, Any],
    selection_results: list[dict[str, Any]],
    config: Config,
    session_seed: int,
) -> SessionActivity:
    """Load, validate, balance, bin, and normalize one session."""
    session = str(state_result.get("session", "unknown_session"))
    required_state_fields = ("time_bins", "on_state_mask", "off_state_mask")
    missing = [field for field in required_state_fields if field not in state_result]
    if missing:
        raise ValueError(
            f"Session {session} on/off cache is missing {missing}; rerun "
            "scripts/next/on_off_states.py to generate state-mask cache fields."
        )

    data_path = config.data_dir / f"{session}.mat"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing session data: {data_path}")
    data = loadmat(
        data_path,
        variable_names=["spks", "tc", "cueAngIdx", "isCorr"],
    )
    spikes = np.asarray(data["spks"])
    times_ms = np.asarray(data["tc"], dtype=float).ravel()
    cue_labels = np.asarray(data["cueAngIdx"], dtype=np.int64).ravel()
    correct_trials = np.asarray(data["isCorr"]).ravel().astype(bool)
    if spikes.ndim != 3 or spikes.shape[1] != times_ms.size:
        raise ValueError(f"Unexpected spike/time shape for session {session}.")
    if cue_labels.size != spikes.shape[0] or correct_trials.size != spikes.shape[0]:
        raise ValueError(f"Trial metadata does not match spike data for session {session}.")

    preferred_cue = int(state_result["cue"])
    opposite_cue = get_opposite_cue(preferred_cue)
    preferred_trial_ids = np.asarray(state_result.get("trial_idx", []), dtype=np.int64).ravel()
    time_bins = np.asarray(state_result["time_bins"], dtype=float).ravel()
    on_state_mask = np.asarray(state_result["on_state_mask"], dtype=bool)
    off_state_mask = np.asarray(state_result["off_state_mask"], dtype=bool)
    expected_mask_shape = (preferred_trial_ids.size, time_bins.size)
    if on_state_mask.shape != expected_mask_shape or off_state_mask.shape != expected_mask_shape:
        raise ValueError(
            f"Session {session} state masks must have shape {expected_mask_shape}."
        )
    if np.any(on_state_mask & off_state_mask):
        raise ValueError(f"Session {session} has bins marked as both on and off state.")
    if np.any(preferred_trial_ids < 0) or np.any(preferred_trial_ids >= spikes.shape[0]):
        raise ValueError(f"Preferred-cue trial IDs are out of range for session {session}.")
    if not np.all(correct_trials[preferred_trial_ids]):
        raise ValueError(f"Cached preferred-cue trials are not all correct in {session}.")
    if not np.all(cue_labels[preferred_trial_ids] == preferred_cue):
        raise ValueError(f"Cached trials do not all use the preferred cue in {session}.")

    delay_start = float(state_result.get("off_state_duration_delay_start", 500))
    delay_end = float(state_result.get("off_state_duration_delay_end", 1400))
    delay_bins = (time_bins >= delay_start) & (time_bins <= delay_end)
    if not np.any(delay_bins):
        raise ValueError(f"Session {session} has no cached bins in the delay period.")
    delay_bin_starts = time_bins[delay_bins]
    max_off_state_mask = maximum_delay_off_state_mask(off_state_mask, delay_bins)
    max_off_state_rows, _ = np.nonzero(max_off_state_mask)
    max_off_state_trial_id = None
    max_off_state_delay_mask = np.zeros(delay_bins.sum(), dtype=bool)
    if max_off_state_rows.size:
        max_off_state_trial_position = int(max_off_state_rows[0])
        max_off_state_trial_id = int(preferred_trial_ids[max_off_state_trial_position])
        max_off_state_delay_mask = max_off_state_mask[max_off_state_trial_position, delay_bins]

    selection = find_full_session_selection(selection_results, session, spikes.shape[0])
    all_preferred_cell_ids, all_preferred_cell_pev = preferred_pev_cells(
        selection,
        preferred_cue,
    )
    group_cell_ids = session_cell_groups(selection, preferred_cue)
    group_activity_weights = cell_group_activity_weights(
        selection,
        group_cell_ids,
        config.pev_weighted_average,
    )
    analysis_cell_ids = np.concatenate(
        [group_cell_ids[group_name] for group_name, _ in POPULATION_GROUPS]
    )
    if np.unique(analysis_cell_ids).size != analysis_cell_ids.size:
        raise ValueError(f"Cell population groups overlap in session {session}.")
    cell_ids = all_preferred_cell_ids[:3]
    cell_pev = all_preferred_cell_pev[:3]
    opposite_trial_ids = np.flatnonzero(correct_trials & (cue_labels == opposite_cue))
    preferred_positions, preferred_ids, opposite_ids = balance_trial_groups(
        preferred_trial_ids,
        opposite_trial_ids,
        session_seed,
    )

    preferred_rates = compute_binned_firing_rates(
        spikes,
        times_ms,
        preferred_ids,
        analysis_cell_ids,
        delay_bin_starts,
        config.activity_bin_width_ms,
    )
    opposite_rates = compute_binned_firing_rates(
        spikes,
        times_ms,
        opposite_ids,
        analysis_cell_ids,
        delay_bin_starts,
        config.activity_bin_width_ms,
    )
    normalization_means, normalization_stds = (
        balanced_activity_normalization_parameters(
            preferred_rates,
            opposite_rates,
        )
    )
    all_preferred_activity = apply_activity_normalization(
        preferred_rates,
        normalization_means,
        normalization_stds,
    )
    all_opposite_activity = apply_activity_normalization(
        opposite_rates,
        normalization_means,
        normalization_stds,
    )
    max_off_state_all_activity = np.empty((0, analysis_cell_ids.size))
    if max_off_state_trial_id is not None:
        max_off_state_rates = compute_binned_firing_rates(
            spikes,
            times_ms,
            np.asarray([max_off_state_trial_id], dtype=np.int64),
            analysis_cell_ids,
            delay_bin_starts,
            config.activity_bin_width_ms,
        )
        max_off_state_all_activity = apply_activity_normalization(
            max_off_state_rates,
            normalization_means,
            normalization_stds,
        )[0, max_off_state_delay_mask, :]
    if max_off_state_all_activity.shape[0] != np.count_nonzero(
        max_off_state_delay_mask
    ):
        raise RuntimeError("Maximum off-state activity extraction lost delay bins.")
    principal_component_activity = None
    if config.show_principal_components:
        preferred_cell_count = int(group_cell_ids["preferred"].size)
        principal_component_activity = compute_preferred_cell_principal_components(
            all_preferred_activity[:, :, :preferred_cell_count],
            all_opposite_activity[:, :, :preferred_cell_count],
            max_off_state_all_activity[:, :preferred_cell_count],
        )
    max_off_state_activity = max_off_state_all_activity[:, :cell_ids.size]
    preferred_activity = all_preferred_activity[:, :, :cell_ids.size]
    opposite_activity = all_opposite_activity[:, :, :cell_ids.size]
    population_mean_activities = {}
    max_off_state_population_mean_activities = {}
    cell_offset = 0
    for group_name, _ in POPULATION_GROUPS:
        group_count = int(group_cell_ids[group_name].size)
        group_slice = slice(cell_offset, cell_offset + group_count)
        preferred_group_mean = None
        opposite_group_mean = None
        if group_count:
            preferred_group_mean = mean_cell_activity(
                all_preferred_activity[:, :, group_slice],
                group_activity_weights[group_name],
                axis=2,
            )
            opposite_group_mean = mean_cell_activity(
                all_opposite_activity[:, :, group_slice],
                group_activity_weights[group_name],
                axis=2,
            )
        population_mean_activities[group_name] = (
            preferred_group_mean,
            opposite_group_mean,
            group_count,
        )
        max_off_state_group_mean = None
        if group_count:
            max_off_state_group_mean = mean_cell_activity(
                max_off_state_all_activity[:, group_slice],
                group_activity_weights[group_name],
                axis=1,
            )
        max_off_state_population_mean_activities[group_name] = (
            max_off_state_group_mean
        )
        cell_offset += group_count
    (
        preferred_population_mean_activity,
        opposite_population_mean_activity,
        preferred_population_cell_count,
    ) = population_mean_activities["preferred"]
    return SessionActivity(
        session=session,
        preferred_cue=preferred_cue,
        opposite_cue=opposite_cue,
        cell_ids=cell_ids,
        cell_pev=cell_pev,
        delay_bin_starts=delay_bin_starts,
        preferred_activity=preferred_activity,
        opposite_activity=opposite_activity,
        on_state_mask=on_state_mask[preferred_positions][:, delay_bins],
        off_state_mask=off_state_mask[preferred_positions][:, delay_bins],
        preferred_trial_ids=preferred_ids,
        opposite_trial_ids=opposite_ids,
        preferred_population_mean_activity=preferred_population_mean_activity,
        opposite_population_mean_activity=opposite_population_mean_activity,
        preferred_population_cell_count=preferred_population_cell_count,
        population_mean_activities=population_mean_activities,
        max_off_state_activity=max_off_state_activity,
        max_off_state_population_mean_activities=(
            max_off_state_population_mean_activities
        ),
        max_off_state_trial_id=max_off_state_trial_id,
        max_off_state_delay_bin_starts=delay_bin_starts[
            max_off_state_delay_mask
        ],
        principal_component_activity=principal_component_activity,
    )


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

    num_cells = session_activity.cell_ids.size
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
            if session_activity.cell_ids.size == 0:
                return None, 0
            preferred_mean = np.mean(session_activity.preferred_activity, axis=2)
            opposite_mean = np.mean(session_activity.opposite_activity, axis=2)
            population_cell_count = int(session_activity.cell_ids.size)
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
    dimension_id = int(session_activity.cell_ids[dimension_position])
    dimension_percent = session_activity.cell_pev[dimension_position]
    if session_activity.activity_space == "principal_components":
        return (
            f"PC{dimension_id} score\n"
            f"(explained variance={dimension_percent:.2f}%)"
        )
    return (
        f"Cell {dimension_id} normalized activity\n"
        f"(delay PEV={dimension_percent:.2f}%)"
    )


def _activity_dimension_title(
    session_activity: SessionActivity,
    dimension_position: int,
) -> str:
    """Return a compact panel title for one plotted activity dimension."""
    dimension_id = int(session_activity.cell_ids[dimension_position])
    if session_activity.activity_space == "principal_components":
        return f"PC{dimension_id}"
    return f"Cell {dimension_id}"


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
    num_cells = session_activity.cell_ids.size
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
    num_cells = session_activity.cell_ids.size
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
        if session_activity.activity_space == "principal_components":
            ax.set_title(
                f"PC{session_activity.cell_ids[x_idx]} and "
                f"PC{session_activity.cell_ids[y_idx]}"
            )
        else:
            ax.set_title(
                f"Cells {session_activity.cell_ids[x_idx]} and "
                f"{session_activity.cell_ids[y_idx]}"
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
    num_cells = session_activity.cell_ids.size
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
    for group_name, group_title in POPULATION_GROUPS:
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
    num_panels = num_cells + len(POPULATION_GROUPS)
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
            histogram_ax.set_title(group_title)
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
        if group_name == "preferred":
            count_label = f"all {group_count} preferred cells"
        elif group_name == "selective_nonpreferred":
            count_label = f"{group_count} selective non-preferred cells"
        else:
            count_label = f"{group_count} stationary non-selective cells"
        _plot_marginal_histogram(
            histogram_ax,
            group_categories,
            value_position=0,
            bin_width=bin_width,
            bin_offset_fraction=bin_offset_fraction,
            xlabel="",
            title=group_title,
            show_legend=not legend_shown,
        )
        legend_shown = True
        _plot_ecdf(
            ecdf_ax,
            group_categories,
            value_position=0,
            xlabel=f"Mean normalized activity\n({count_label})",
        )

    fig.suptitle(
        _session_activity_figure_title(
            session_activity,
            "on/off states" if comparison == "state" else "all delay bins by cue",
            marginal=True,
        )
    )
    return fig


def main(config: Config):
    """Generate one activity-state comparison plot for every cached session."""
    output_subdir = Path(config.output_subdir)
    if output_subdir.is_absolute() or ".." in output_subdir.parts:
        raise ValueError("output_subdir must stay within cache_dir.")
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

    selection_results = _load_pickle(config.cache_dir / "cell_trial_selection.pkl")
    state_results = _load_pickle(config.cache_dir / "on_off_states.pkl")
    if not isinstance(selection_results, list) or not isinstance(state_results, list):
        raise TypeError("Both input cache files must contain lists of results.")
    if not state_results:
        raise ValueError("The on/off-state cache contains no session results.")

    sessions = [str(result.get("session", "unknown_session")) for result in state_results]
    if len(set(sessions)) != len(sessions):
        raise ValueError("The on/off-state cache contains duplicate session entries.")

    figure_dir = weighting_subdir(
        config.cache_dir / output_subdir,
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
            for comparison, filename_group in (
                ("state", "states"),
                ("cue", "cues"),
            ):
                fig = plot_session_activity(
                    plotting_activity,
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
                    point_seed=config.seed + session_idx,
                )
                save_figure_all_formats(
                    fig,
                    figure_dir
                    / (
                        f"{filename_prefix}activity_across_{filename_group}_"
                        f"{prepared.session}.png"
                    ),
                    dpi=config.figure_dpi,
                )
                plt.close(fig)

                pairwise_fig = plot_session_activity_pairwise(
                    plotting_activity,
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
                    point_seed=config.seed + session_idx,
                )
                save_figure_all_formats(
                    pairwise_fig,
                    figure_dir
                    / (
                        f"{filename_prefix}activity_across_{filename_group}_"
                        f"pairwise_{prepared.session}.png"
                    ),
                    dpi=config.figure_dpi,
                )
                plt.close(pairwise_fig)

                marginal_fig = plot_session_activity_marginal_histograms(
                    plotting_activity,
                    comparison=comparison,
                    compare_with_max_off_state=config.compare_with_max_off_state,
                    hide_opposite_cue_points=config.hide_opposite_cue_points,
                    hide_all_preferred_cue_points=(
                        config.hide_all_preferred_cue_points
                    ),
                    max_points_per_max_off_state=(
                        config.max_points_per_max_off_state
                    ),
                    point_seed=config.seed + session_idx,
                    bin_width=config.marginal_histogram_bin_width,
                    bin_offset_fraction=(
                        config.marginal_histogram_bin_offset_fraction
                    ),
                )
                save_figure_all_formats(
                    marginal_fig,
                    figure_dir
                    / (
                        f"{filename_prefix}activity_across_{filename_group}_"
                        f"marginals_{prepared.session}.png"
                    ),
                    dpi=config.figure_dpi,
                )
                plt.close(marginal_fig)
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
