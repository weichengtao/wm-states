"""Load and prepare balanced, normalized activity without plotting dependencies.

Each cell is normalized across the same balanced correct preferred/opposite
trials, independently for each delay bin. PCA uses all eligible preferred cells
and the same pooled balanced observations. Display limits never change fitting.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any
import warnings

import numpy as np
from sklearn.decomposition import PCA

from scripts.next.activity_types import (
    Config, CellActivityDimensions, PrincipalComponentActivity,
    PrincipalComponentDimensions, SessionActivity, POPULATION_GROUP_KEYS,
    get_opposite_cue,
)
from scripts.next.activity_weighting import cell_group_activity_weights, mean_cell_activity
from scripts.next.common import full_session_selection
from scripts.next.screening_metadata import cell_groups, preferred_pev_cells, validate_cue
from scripts.next.session_inputs import load_session_inputs, validate_state_trial_ids


def find_full_session_selection(selection_results, session, num_trials):
    return full_session_selection(selection_results, session, num_trials)


def top_preferred_pev_cells(
    selection_result: dict[str, Any],
    preferred_cue: int,
    count: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Return up to ``count`` highest-PEV cells preferred for the session cue."""
    cell_ids, selectivity_pev_pct = preferred_pev_cells(selection_result, preferred_cue)
    return cell_ids[:count], selectivity_pev_pct[:count]


def session_cell_groups(
    selection_result: dict[str, Any],
    preferred_cue: int,
    *,
    num_cells_total: int | None = None,
) -> dict[str, np.ndarray]:
    """Build shared populations, preserving finite-PEV preferred-cell ranking."""
    return cell_groups(
        selection_result, preferred_cue, rank_preferred_by_pev=True,
        num_cells_total=num_cells_total,
    )


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
        max_off_state_activity=(
            pca.transform(max_off_state_activity)
            if max_off_state_activity.shape[0]
            else np.empty((0, component_count), dtype=float)
        ),
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
        dimensions=PrincipalComponentDimensions(
            component_numbers=np.arange(1, component_count + 1, dtype=np.int64),
            explained_variance_ratio=projection.explained_variance_ratio.copy(),
            source_cell_count=projection.source_cell_count,
        ),
        preferred_activity=projection.preferred_activity,
        opposite_activity=projection.opposite_activity,
        max_off_state_activity=projection.max_off_state_activity,
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
    session_inputs = load_session_inputs(data_path, session=session)
    spikes = session_inputs.spikes
    times_ms = session_inputs.times_ms
    cue_labels = session_inputs.cue_labels
    correct_trials = session_inputs.correct_trials

    preferred_cue = validate_cue(state_result["cue"])
    opposite_cue = get_opposite_cue(preferred_cue)
    preferred_trial_ids = validate_state_trial_ids(
        session_inputs, state_result.get("trial_idx", []), preferred_cue,
    )
    time_bins = np.asarray(state_result["time_bins"], dtype=float).ravel()
    if (not np.all(np.isfinite(time_bins)) or np.any(np.diff(time_bins) <= 0)):
        raise ValueError(f"Session {session} time bins must be finite and strictly increasing.")
    on_state_mask = np.asarray(state_result["on_state_mask"])
    off_state_mask = np.asarray(state_result["off_state_mask"])
    expected_mask_shape = (preferred_trial_ids.size, time_bins.size)
    if on_state_mask.shape != expected_mask_shape or off_state_mask.shape != expected_mask_shape:
        raise ValueError(
            f"Session {session} state masks must have shape {expected_mask_shape}."
        )
    for state_name, state_mask in (("on", on_state_mask), ("off", off_state_mask)):
        if state_mask.dtype.kind not in "biuf" or not np.all(np.isin(state_mask, [0, 1])):
            raise ValueError(
                f"Session {session} {state_name}-state mask must contain only "
                "finite Boolean or 0/1 values."
            )
    on_state_mask = on_state_mask.astype(bool)
    off_state_mask = off_state_mask.astype(bool)
    if np.any(on_state_mask & off_state_mask):
        raise ValueError(f"Session {session} has bins marked as both on and off state.")
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
    all_preferred_cell_ids, all_preferred_selectivity_pev_pct = preferred_pev_cells(
        selection,
        preferred_cue,
        num_cells_total=spikes.shape[2],
    )
    group_cell_ids = session_cell_groups(
        selection, preferred_cue, num_cells_total=spikes.shape[2],
    )
    group_activity_weights = cell_group_activity_weights(
        selection,
        group_cell_ids,
        config.pev_weighted_average,
    )
    analysis_cell_ids = np.concatenate(
        [group_cell_ids[group_name] for group_name in POPULATION_GROUP_KEYS]
    )
    if np.unique(analysis_cell_ids).size != analysis_cell_ids.size:
        raise ValueError(f"Cell population groups overlap in session {session}.")
    cell_ids = all_preferred_cell_ids[:3]
    cell_selectivity_pev_pct = all_preferred_selectivity_pev_pct[:3]
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
    if max_off_state_trial_id is None:
        warnings.warn(
            f"Session {session}: no cached off-state bins occur in the delay period. "
            "Activity and PCA comparisons continue without maximum-off-state points; "
            "review the states output and thresholds if this is unexpected.",
            RuntimeWarning,
            stacklevel=2,
        )
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
    for group_name in POPULATION_GROUP_KEYS:
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
        dimensions=CellActivityDimensions(
            cell_ids=cell_ids,
            selectivity_pev_pct=cell_selectivity_pev_pct,
        ),
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
        screening_checks=selection.get("screening_checks"),
    )
