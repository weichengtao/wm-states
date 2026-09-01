"""Compare top preferred-cell activity across on/off and opposite-cue states.

For each session, this script selects up to three preferred-cue cells with the
largest cached delay-period PEV, balances correct preferred- and opposite-cue
trials, and z-normalizes each cell across the balanced trials independently at
every delay-bin start. It adapts the plot dimensionality to the number of
available preferred cells, coloring preferred-cue bins by their cached on/off
state and all opposite-cue bins as a comparison population.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tyro
from scipy.io import loadmat

try:
    from scripts.figure_exports import configure_figure_style, save_figure_all_formats
except ModuleNotFoundError:
    from figure_exports import configure_figure_style, save_figure_all_formats


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
    cache_dir: Path = Path("cache/run_001")
    output_subdir: str = "compare_activity_across_states"
    activity_bin_width_ms: float = 50.0
    seed: int = 42
    figure_dpi: int = 300
    hide_opposite_cue_points: bool = False
    hide_all_preferred_cue_points: bool = False
    max_points_per_color_group: int | None = None
    marginal_histogram_bin_width: float = 0.25


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


def find_full_session_selection(
    selection_results: list[dict[str, Any]],
    session: str,
    num_trials: int,
) -> dict[str, Any]:
    """Find the unique non-holdout selection spanning the full session."""
    matches = [
        result
        for result in selection_results
        if str(result.get("session")) == session
        and result.get("trial_holdout") is None
        and int(result.get("trial_start", -1)) == 0
        and int(result.get("trial_end", -1)) == num_trials
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one full-session cell-selection entry for "
            f"session {session}; found {len(matches)}."
        )
    return matches[0]


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
    preferred_rates = np.asarray(preferred_rates, dtype=float)
    opposite_rates = np.asarray(opposite_rates, dtype=float)
    if preferred_rates.shape != opposite_rates.shape:
        raise ValueError("Balanced preferred and opposite activity must match in shape.")
    if preferred_rates.ndim != 3:
        raise ValueError("Activity must have shape (trial, bin, cell).")
    combined = np.concatenate([preferred_rates, opposite_rates], axis=0)
    means = np.mean(combined, axis=0)
    stds = np.std(combined, axis=0, ddof=0)
    normalized = np.zeros_like(combined, dtype=float)
    usable = np.isfinite(means) & np.isfinite(stds) & (stds > 0)
    normalized[:, usable] = (combined[:, usable] - means[usable]) / stds[usable]
    split = preferred_rates.shape[0]
    return normalized[:split], normalized[split:]


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
            "scripts/on_off_states.py to generate state-mask cache fields."
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

    selection = find_full_session_selection(selection_results, session, spikes.shape[0])
    all_preferred_cell_ids, all_preferred_cell_pev = preferred_pev_cells(
        selection,
        preferred_cue,
    )
    group_cell_ids = session_cell_groups(selection, preferred_cue)
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
    all_preferred_activity, all_opposite_activity = normalize_balanced_activity(
        preferred_rates,
        opposite_rates,
    )
    preferred_activity = all_preferred_activity[:, :, :cell_ids.size]
    opposite_activity = all_opposite_activity[:, :, :cell_ids.size]
    population_mean_activities = {}
    cell_offset = 0
    for group_name, _ in POPULATION_GROUPS:
        group_count = int(group_cell_ids[group_name].size)
        group_slice = slice(cell_offset, cell_offset + group_count)
        preferred_group_mean = None
        opposite_group_mean = None
        if group_count:
            preferred_group_mean = np.mean(
                all_preferred_activity[:, :, group_slice],
                axis=2,
            )
            opposite_group_mean = np.mean(
                all_opposite_activity[:, :, group_slice],
                axis=2,
            )
        population_mean_activities[group_name] = (
            preferred_group_mean,
            opposite_group_mean,
            group_count,
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
    )


def activity_point_categories(
    session_activity: SessionActivity,
    hide_opposite_cue_points: bool = False,
    hide_all_preferred_cue_points: bool = False,
    max_points_per_color_group: int | None = None,
    seed: int = 42,
):
    """Return optionally subsampled activity points for each displayed color."""
    if max_points_per_color_group is not None and (
        isinstance(max_points_per_color_group, (bool, np.bool_))
        or max_points_per_color_group <= 0
    ):
        raise ValueError("max_points_per_color_group must be positive when set.")

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
    categories = [
        (preferred_points[on_mask], "tab:blue", "Preferred cue: on state"),
        (preferred_points[off_mask], "tab:orange", "Preferred cue: off state"),
    ]
    if not hide_all_preferred_cue_points:
        categories.append(
            (preferred_points, "tab:green", "Preferred cue: all delay bins")
        )
    if not hide_opposite_cue_points:
        categories.append((opposite_points, "tab:gray", "Opposite cue"))

    rng = np.random.default_rng(seed)
    displayed_categories = []
    for points, color, label in categories:
        total_count = points.shape[0]
        if (
            max_points_per_color_group is not None
            and total_count > max_points_per_color_group
        ):
            point_indices = np.sort(
                rng.choice(
                    total_count,
                    size=max_points_per_color_group,
                    replace=False,
                )
            )
            points = points[point_indices]
        displayed_categories.append((points, color, label, total_count))
    return displayed_categories


def population_mean_point_categories(
    session_activity: SessionActivity,
    population_group: str = "preferred",
    hide_opposite_cue_points: bool = False,
    hide_all_preferred_cue_points: bool = False,
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

    preferred_points = np.asarray(preferred_mean, dtype=float).ravel()
    opposite_points = np.asarray(opposite_mean, dtype=float).ravel()
    on_mask = session_activity.on_state_mask.ravel()
    off_mask = session_activity.off_state_mask.ravel()
    if preferred_points.size != on_mask.size or preferred_points.size != off_mask.size:
        raise ValueError("Preferred population mean activity does not match state masks.")
    categories = [
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
                "Opposite cue",
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


def _placeholder_figure(session_activity: SessionActivity, plot_name: str):
    """Return a session figure explaining that no preferred cells are available."""
    fig, ax = plt.subplots(figsize=(6.5, 4), layout="constrained")
    ax.text(
        0.5,
        0.5,
        "No finite-PEV cells preferred for this session cue",
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
    ax.set_xlabel(
        f"Cell {session_activity.cell_ids[0]} normalized activity\n"
        f"(delay PEV={session_activity.cell_pev[0]:.2f}%)"
    )
    ax.set_title(f"Cell {session_activity.cell_ids[0]}")
    ax.spines[["top", "right"]].set_visible(False)


def plot_session_activity(
    session_activity: SessionActivity,
    hide_opposite_cue_points: bool = False,
    hide_all_preferred_cue_points: bool = False,
    max_points_per_color_group: int | None = None,
    point_seed: int = 42,
):
    """Create a session scatter plot using every available preferred cell."""
    num_cells = session_activity.cell_ids.size
    if num_cells == 0:
        return _placeholder_figure(session_activity, "activity")
    categories = activity_point_categories(
        session_activity,
        hide_opposite_cue_points=hide_opposite_cue_points,
        hide_all_preferred_cue_points=hide_all_preferred_cue_points,
        max_points_per_color_group=max_points_per_color_group,
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
                f"Cell {session_activity.cell_ids[axis_idx]} normalized activity\n"
                f"(delay PEV={session_activity.cell_pev[axis_idx]:.2f}%)"
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
        ax.set_xlabel(
            f"Cell {session_activity.cell_ids[0]} normalized activity\n"
            f"(delay PEV={session_activity.cell_pev[0]:.2f}%)"
        )
        ax.set_ylabel(
            f"Cell {session_activity.cell_ids[1]} normalized activity\n"
            f"(delay PEV={session_activity.cell_pev[1]:.2f}%)"
        )
        ax.legend(loc="best", frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    else:
        fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
        _plot_single_cell_strip(ax, session_activity, categories)
    ax.set_title(
        f"Session {session_activity.session}: preferred cue "
        f"{cue_to_deg(session_activity.preferred_cue)}°"
    )
    return fig


def plot_session_activity_pairwise(
    session_activity: SessionActivity,
    hide_opposite_cue_points: bool = False,
    hide_all_preferred_cue_points: bool = False,
    max_points_per_color_group: int | None = None,
    point_seed: int = 42,
):
    """Create every available pairwise 2D projection for one session."""
    num_cells = session_activity.cell_ids.size
    if num_cells == 0:
        return _placeholder_figure(session_activity, "pairwise activity")
    categories = activity_point_categories(
        session_activity,
        hide_opposite_cue_points=hide_opposite_cue_points,
        hide_all_preferred_cue_points=hide_all_preferred_cue_points,
        max_points_per_color_group=max_points_per_color_group,
        seed=point_seed,
    )
    cell_pairs = list(combinations(range(num_cells), 2))

    if not cell_pairs:
        fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
        _plot_single_cell_strip(ax, session_activity, categories)
        fig.suptitle(
            f"Session {session_activity.session}: preferred cue "
            f"{cue_to_deg(session_activity.preferred_cue)}° "
            "(one preferred cell; no pair available)"
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
        ax.set_xlabel(
            f"Cell {session_activity.cell_ids[x_idx]} normalized activity\n"
            f"(delay PEV={session_activity.cell_pev[x_idx]:.2f}%)"
        )
        ax.set_ylabel(
            f"Cell {session_activity.cell_ids[y_idx]} normalized activity\n"
            f"(delay PEV={session_activity.cell_pev[y_idx]:.2f}%)"
        )
        ax.set_title(
            f"Cells {session_activity.cell_ids[x_idx]} and "
            f"{session_activity.cell_ids[y_idx]}"
        )
        ax.spines[["top", "right"]].set_visible(False)
        if pair_idx == 0:
            ax.legend(loc="best", frameon=False, fontsize="small")

    fig.suptitle(
        f"Session {session_activity.session}: preferred cue "
        f"{cue_to_deg(session_activity.preferred_cue)}°"
    )
    return fig


def _plot_marginal_histogram(
    ax,
    categories,
    value_position: int,
    bin_width: float,
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
        ax.hist(
            points[:, value_position],
            bins=bin_edges,
            density=True,
            histtype="step",
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
    hide_opposite_cue_points: bool = False,
    hide_all_preferred_cue_points: bool = False,
    bin_width: float = 0.25,
):
    """Plot selected-cell marginals plus three cell-population means."""
    num_cells = session_activity.cell_ids.size
    categories = None
    if num_cells:
        categories = activity_point_categories(
            session_activity,
            hide_opposite_cue_points=hide_opposite_cue_points,
            hide_all_preferred_cue_points=hide_all_preferred_cue_points,
        )
    population_data = []
    for group_name, group_title in POPULATION_GROUPS:
        group_categories, group_cell_count = population_mean_point_categories(
            session_activity,
            population_group=group_name,
            hide_opposite_cue_points=hide_opposite_cue_points,
            hide_all_preferred_cue_points=hide_all_preferred_cue_points,
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
        xlabel = (
            f"Cell {session_activity.cell_ids[cell_position]} normalized activity\n"
            f"(delay PEV={session_activity.cell_pev[cell_position]:.2f}%)"
        )
        _plot_marginal_histogram(
            histogram_axes[cell_position],
            categories,
            value_position=cell_position,
            bin_width=bin_width,
            xlabel="",
            title=f"Cell {session_activity.cell_ids[cell_position]}",
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
        f"Session {session_activity.session}: preferred cue "
        f"{cue_to_deg(session_activity.preferred_cue)}° marginal activity"
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
    if (
        not np.isfinite(config.marginal_histogram_bin_width)
        or config.marginal_histogram_bin_width <= 0
    ):
        raise ValueError(
            "marginal_histogram_bin_width must be finite and positive."
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

    figure_dir = config.cache_dir / output_subdir
    figure_dir.mkdir(parents=True, exist_ok=True)
    for session_idx, state_result in enumerate(state_results):
        prepared = prepare_session_activity(
            state_result,
            selection_results,
            config,
            session_seed=config.seed + session_idx,
        )
        fig = plot_session_activity(
            prepared,
            hide_opposite_cue_points=config.hide_opposite_cue_points,
            hide_all_preferred_cue_points=config.hide_all_preferred_cue_points,
            max_points_per_color_group=config.max_points_per_color_group,
            point_seed=config.seed + session_idx,
        )
        save_figure_all_formats(
            fig,
            figure_dir / f"activity_across_states_{prepared.session}.png",
            dpi=config.figure_dpi,
        )
        plt.close(fig)

        pairwise_fig = plot_session_activity_pairwise(
            prepared,
            hide_opposite_cue_points=config.hide_opposite_cue_points,
            hide_all_preferred_cue_points=config.hide_all_preferred_cue_points,
            max_points_per_color_group=config.max_points_per_color_group,
            point_seed=config.seed + session_idx,
        )
        save_figure_all_formats(
            pairwise_fig,
            figure_dir / f"activity_across_states_pairwise_{prepared.session}.png",
            dpi=config.figure_dpi,
        )
        plt.close(pairwise_fig)

        marginal_fig = plot_session_activity_marginal_histograms(
            prepared,
            hide_opposite_cue_points=config.hide_opposite_cue_points,
            hide_all_preferred_cue_points=config.hide_all_preferred_cue_points,
            bin_width=config.marginal_histogram_bin_width,
        )
        save_figure_all_formats(
            marginal_fig,
            figure_dir / f"activity_across_states_marginals_{prepared.session}.png",
            dpi=config.figure_dpi,
        )
        plt.close(marginal_fig)
        print(
            f"Saved activity, pairwise, and marginal comparisons for session "
            f"{prepared.session} "
            f"({prepared.preferred_trial_ids.size} trials per cue)."
        )


if __name__ == "__main__":
    main(tyro.cli(Config))
