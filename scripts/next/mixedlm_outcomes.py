"""Shared outcome definitions and output paths for MixedLM analyses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


OutcomeSelection = Literal["both", "total", "maximum"]


@dataclass(frozen=True)
class OutcomeSpec:
    """One trial-level delay-period outcome analyzed by the MixedLM pipeline."""

    name: Literal["total", "maximum"]
    column: str
    cache_key: str
    slug: str
    label: str


TOTAL_OUTCOME = OutcomeSpec(
    name="total",
    column="total_off_state_duration_ms",
    cache_key="off_state_duration_per_trial",
    slug="total_off_state_duration",
    label="Total off-state duration during delay",
)
MAXIMUM_OUTCOME = OutcomeSpec(
    name="maximum",
    column="maximum_off_state_duration_ms",
    cache_key="max_off_state_duration_per_trial",
    slug="maximum_off_state_duration",
    label="Maximum off-state duration during delay",
)
ALL_OUTCOMES = (TOTAL_OUTCOME, MAXIMUM_OUTCOME)


def select_outcomes(selection: OutcomeSelection) -> tuple[OutcomeSpec, ...]:
    """Resolve a CLI outcome selection in a stable output order."""
    if selection == "both":
        return ALL_OUTCOMES
    selected = tuple(
        outcome for outcome in ALL_OUTCOMES if outcome.name == selection
    )
    if not selected:
        raise ValueError("outcome must be 'both', 'total', or 'maximum'.")
    return selected


def analysis_output_dir(
    cache_dir: Path,
    output_subdir: str,
    outcome: OutcomeSpec,
    analysis_name: str,
) -> Path:
    """Return the outcome-first directory for one analysis."""
    return (
        cache_dir
        / output_subdir
        / "outcomes"
        / outcome.slug
        / analysis_name
    )
