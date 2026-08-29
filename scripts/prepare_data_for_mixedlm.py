"""Prepare trial-level predictors and off-state durations for MixedLM analyses.

The output contains one row for every preferred-cue trial that has at least one
earlier preferred-cue trial in the same session.  Cell activity is normalized
cell by cell, within session and period, across all cached preferred-cue trials.
No normalization is applied to cell counts, active fractions, EMA histories, or
off-state durations.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tyro
from scipy.io import loadmat


PERIODS = {
    "baseline": (-400, 0),
    "encoding": (100, 300),
    "pre_delay": (300, 500),
    "delay": (500, 1400),
}

GROUP_NAMES = (
    "preferred",
    "selective_nonpreferred",
    "stationary_nonselective",
)


@dataclass
class Config:
    """Input locations and feature-extraction settings."""

    data_dir: Path = Path("data/nature")
    cache_dir: Path = Path("cache/run_001")
    output_subdir: str = "prepare_data_for_mixedlm"
    output_filename: str = "mixedlm_data.pkl"
    active_threshold: float = 0.0
    history_alpha: float = 0.2


def _load_pickle(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing cache file: {path}")
    with path.open("rb") as handle:
        return pickle.load(handle)


def _find_full_session_selection(
    selection_results: list[dict[str, Any]],
    session: str,
    num_trials: int,
) -> dict[str, Any]:
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


def _cell_groups(
    selection_result: dict[str, Any],
    preferred_cue: int,
) -> dict[str, np.ndarray]:
    properties = selection_result["cell_properties"]
    selective_cells = np.asarray(properties["cell_idx"], dtype=np.int64).ravel()
    preferred_cues = np.asarray(properties["mean_pref_test"]).ravel()
    if selective_cells.shape != preferred_cues.shape:
        raise ValueError("cell_idx and mean_pref_test must have matching shapes.")

    stationary_cells = np.asarray(
        selection_result["cell_idx_stationary"], dtype=np.int64
    ).ravel()
    return {
        "preferred": selective_cells[preferred_cues == preferred_cue],
        "selective_nonpreferred": selective_cells[
            preferred_cues != preferred_cue
        ],
        "stationary_nonselective": stationary_cells[
            ~np.isin(stationary_cells, selective_cells)
        ],
    }


def _period_firing_rates(
    spikes: np.ndarray,
    trial_ids: np.ndarray,
    times_ms: np.ndarray,
    cell_ids: np.ndarray,
    start_ms: int,
    end_ms: int,
) -> np.ndarray:
    """Return firing rates with shape (target trial, cell) for [start, end)."""
    time_mask = (times_ms >= start_ms) & (times_ms < end_ms)
    if not np.any(time_mask):
        raise ValueError(f"No samples found in period [{start_ms}, {end_ms}) ms.")
    if cell_ids.size == 0:
        return np.empty((trial_ids.size, 0), dtype=float)

    time_ids = np.flatnonzero(time_mask)
    duration_seconds = (end_ms - start_ms) / 1000.0
    rates = (
        spikes[np.ix_(trial_ids, time_ids, cell_ids)].sum(axis=1)
        / duration_seconds
    )
    rates = np.asarray(rates, dtype=float)
    if not np.all(np.isfinite(rates)):
        raise ValueError("Raw firing rates contain non-finite values.")
    return rates


def _normalize_cells(raw_rates: np.ndarray) -> np.ndarray:
    """Normalize each cell across trials, mapping zero mean/std cells to zero."""
    raw_rates = np.asarray(raw_rates, dtype=float)
    normalized = np.zeros_like(raw_rates, dtype=float)
    if raw_rates.shape[1] == 0:
        return normalized

    cell_means = np.mean(raw_rates, axis=0)
    cell_stds = np.std(raw_rates, axis=0, ddof=0)
    usable = (
        np.isfinite(cell_means)
        & np.isfinite(cell_stds)
        & (cell_means != 0)
        & (cell_stds != 0)
    )
    normalized[:, usable] = (
        raw_rates[:, usable] - cell_means[usable]
    ) / cell_stds[usable]
    return normalized


def _group_features(
    normalized_activity: np.ndarray,
    active_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return group mean normalized activity and active-cell fraction."""
    num_trials, num_cells = normalized_activity.shape
    if num_cells == 0:
        zeros = np.zeros(num_trials, dtype=float)
        return zeros.copy(), zeros
    mean_activity = np.mean(normalized_activity, axis=1)
    active_fraction = np.mean(normalized_activity > active_threshold, axis=1)
    return mean_activity, active_fraction


def _causal_history_ema(values: np.ndarray, alpha: float) -> np.ndarray:
    """Return the EMA before each trial; the first trial has no history."""
    values = np.asarray(values, dtype=float)
    history = np.full(values.shape, np.nan, dtype=float)
    if values.size == 0:
        return history

    state = float(values[0])
    for trial_position in range(1, values.size):
        history[trial_position] = state
        state = alpha * float(values[trial_position]) + (1.0 - alpha) * state
    return history


def _validate_off_state_result(
    state_result: dict[str, Any],
    session: str,
) -> tuple[np.ndarray, np.ndarray]:
    if state_result.get("off_state_duration_correction") != "applied":
        raise ValueError(
            f"Session {session} does not contain CC-applied off-state durations."
        )
    if (
        int(state_result.get("off_state_duration_delay_start", -1)) != 500
        or int(state_result.get("off_state_duration_delay_end", -1)) != 1400
    ):
        raise ValueError(
            f"Session {session} has unexpected off-state delay metadata."
        )

    trial_ids = np.asarray(state_result["trial_idx"], dtype=np.int64).ravel()
    durations = np.asarray(
        state_result["off_state_duration_per_trial"], dtype=float
    ).ravel()
    if trial_ids.shape != durations.shape:
        raise ValueError(
            f"Off-state trial IDs and durations do not align for session {session}."
        )
    if trial_ids.size < 2:
        raise ValueError(
            f"Session {session} needs at least two preferred-cue trials for history."
        )
    if not np.all(np.isfinite(durations)):
        raise ValueError(f"Off-state durations are non-finite for session {session}.")

    order = np.argsort(trial_ids, kind="stable")
    trial_ids = trial_ids[order]
    durations = durations[order]
    if np.any(np.diff(trial_ids) <= 0):
        raise ValueError(
            f"Preferred-cue trial IDs are not unique for session {session}."
        )
    return trial_ids, durations


def _prepare_session_rows(
    state_result: dict[str, Any],
    selection_results: list[dict[str, Any]],
    config: Config,
) -> list[dict[str, Any]]:
    session = str(state_result.get("session", "unknown_session"))
    trial_ids, off_state_durations = _validate_off_state_result(
        state_result, session
    )
    preferred_cue = int(state_result["cue"])

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
    if np.any(trial_ids < 0) or np.any(trial_ids >= spikes.shape[0]):
        raise ValueError(f"Trial IDs are out of range for session {session}.")
    if not np.all(cue_labels[trial_ids] == preferred_cue):
        raise ValueError(f"Cached trials do not all use the preferred cue in {session}.")
    if not np.all(correct_trials[trial_ids]):
        raise ValueError(f"Cached trials are not all correct in session {session}.")

    selection = _find_full_session_selection(
        selection_results, session, spikes.shape[0]
    )
    groups = _cell_groups(selection, preferred_cue)
    counts = {
        f"{group_name}_cell_count": int(groups[group_name].size)
        for group_name in GROUP_NAMES
    }

    trial_features: dict[str, np.ndarray] = {}
    for period_name, (start_ms, end_ms) in PERIODS.items():
        for group_name in GROUP_NAMES:
            raw_rates = _period_firing_rates(
                spikes,
                trial_ids,
                times_ms,
                groups[group_name],
                start_ms,
                end_ms,
            )
            normalized = _normalize_cells(raw_rates)
            mean_activity, active_fraction = _group_features(
                normalized, config.active_threshold
            )

            mean_column = (
                f"{period_name}_mean_normalized_activity_{group_name}"
            )
            fraction_column = f"{period_name}_active_fraction_{group_name}"
            trial_features[mean_column] = mean_activity
            trial_features[fraction_column] = active_fraction
            trial_features[f"history_ema_{mean_column}"] = _causal_history_ema(
                mean_activity, config.history_alpha
            )
            trial_features[f"history_ema_{fraction_column}"] = (
                _causal_history_ema(active_fraction, config.history_alpha)
            )

    rows: list[dict[str, Any]] = []
    # Position zero supplies the initial history state and is not saved.
    for trial_position in range(1, trial_ids.size):
        row: dict[str, Any] = {
            "session": session,
            "trial_id": int(trial_ids[trial_position]),
            "preferred_cue": preferred_cue,
            "off_state_duration_ms": float(
                off_state_durations[trial_position]
            ),
            **counts,
        }
        row.update(
            {
                column: float(values[trial_position])
                for column, values in trial_features.items()
            }
        )
        rows.append(row)
    return rows


def prepare_data(config: Config) -> pd.DataFrame:
    """Build and save the complete trial-level DataFrame."""
    if not np.isfinite(config.active_threshold):
        raise ValueError("active_threshold must be finite.")
    if not np.isfinite(config.history_alpha) or not 0 < config.history_alpha <= 1:
        raise ValueError("history_alpha must be finite and in (0, 1].")
    if Path(config.output_subdir).is_absolute():
        raise ValueError("output_subdir must be relative to cache_dir.")
    if Path(config.output_filename).name != config.output_filename:
        raise ValueError("output_filename must be a filename, not a path.")

    selection_results = _load_pickle(config.cache_dir / "cell_trial_selection.pkl")
    off_state_results = _load_pickle(config.cache_dir / "on_off_states.pkl")
    if not isinstance(selection_results, list) or not isinstance(
        off_state_results, list
    ):
        raise TypeError("Both input cache files must contain lists of results.")

    seen_sessions: set[str] = set()
    rows: list[dict[str, Any]] = []
    for state_result in off_state_results:
        session = str(state_result.get("session", "unknown_session"))
        if session in seen_sessions:
            raise ValueError(f"Duplicate on/off-state entry for session {session}.")
        seen_sessions.add(session)
        rows.extend(_prepare_session_rows(state_result, selection_results, config))

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No rows were produced.")
    if frame.duplicated(["session", "trial_id"]).any():
        raise ValueError("The output contains duplicate session/trial IDs.")
    numeric_columns = frame.columns.difference(["session"])
    if not np.all(np.isfinite(frame[numeric_columns].to_numpy(dtype=float))):
        raise ValueError("The output contains non-finite numeric values.")

    output_dir = config.cache_dir / config.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / config.output_filename
    frame.to_pickle(output_path)
    print(
        f"Saved {len(frame)} trials from {frame['session'].nunique()} sessions "
        f"with {len(frame.columns)} columns to {output_path}"
    )
    return frame


def main(config: Config) -> None:
    prepare_data(config)


if __name__ == "__main__":
    main(tyro.cli(Config))
