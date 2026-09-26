"""Validated session arrays and cached trial alignment for downstream stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts.next.common import load_session
from scripts.next.screening_metadata import validate_cue


@dataclass(frozen=True)
class SessionInputs:
    """Recorded arrays; spike axes are trial, sample, and cell respectively."""

    session: str
    spikes: np.ndarray
    times_ms: np.ndarray
    cue_labels: np.ndarray
    correct_trials: np.ndarray


def load_session_inputs(
    path: str | Path, *, session: str | None = None,
) -> SessionInputs:
    """Load the shared MAT contract without changing the recorded spike values."""
    path = Path(path)
    session = str(session) if session is not None else path.stem
    if not path.exists():
        raise FileNotFoundError(f"Session {session}: missing session data: {path}")
    try:
        spikes, times_ms, cue_labels, correct_trials = load_session(path)
    except KeyError as error:
        raise ValueError(
            f"Session {session}: missing required MAT variable {error.args[0]!r}."
        ) from error
    except ValueError as error:
        raise ValueError(f"Session {session}: {error}") from error
    return SessionInputs(session, spikes, times_ms, cue_labels, correct_trials)


def validate_trial_ids(
    trial_ids: np.ndarray, *, session: str, num_trials: int | None = None,
) -> np.ndarray:
    """Return unique integer trial IDs in their original cached row order.

    Integral floats and MATLAB row/column vectors are accepted. Fractions,
    non-finite values, booleans, and matrices are rejected before conversion so
    a malformed cache cannot silently point to a different recorded trial.
    """
    values = np.asarray(trial_ids)
    if values.ndim > 2 or (values.ndim == 2 and 1 not in values.shape):
        raise ValueError(f"Session {session}: cached trial IDs must be a vector.")
    values = values.ravel()
    if values.dtype.kind not in "iuf" or not np.all(np.isfinite(values)):
        raise ValueError(f"Session {session}: cached trial IDs must be finite integers.")
    if values.dtype.kind == "f" and np.any(values != np.floor(values)):
        raise ValueError(f"Session {session}: cached trial IDs must be finite integers.")
    if (
        np.any(values < 0)
        or (values.dtype.kind == "f" and np.any(values >= 2**63))
        or (values.dtype.kind == "u" and np.any(values > np.iinfo(np.int64).max))
        or (num_trials is not None and np.any(values >= num_trials))
    ):
        raise ValueError(f"Session {session}: cached trial IDs are out of range.")
    indices = np.asarray(values, dtype=np.int64)
    if np.unique(indices).size != indices.size:
        raise ValueError(f"Session {session}: cached trial IDs are not unique.")
    return indices


def validate_state_trial_ids(
    session_inputs: SessionInputs,
    trial_ids: np.ndarray,
    preferred_cue: int,
) -> np.ndarray:
    """Require cached state rows to reference correct preferred-cue trials.

    This helper does not sort: state masks and other row-aligned values must
    continue to use the cache's original order. Analyses that require temporal
    sorting, such as trial-history models, own that operation explicitly.
    """
    session = session_inputs.session
    preferred_cue = validate_cue(preferred_cue, f"Session {session}: preferred cue")
    indices = validate_trial_ids(
        trial_ids, session=session, num_trials=session_inputs.spikes.shape[0],
    )
    if not np.all(session_inputs.correct_trials[indices]):
        raise ValueError(f"Session {session}: cached trials are not all correct.")
    if not np.all(session_inputs.cue_labels[indices] == preferred_cue):
        raise ValueError(f"Session {session}: cached trials do not all use the preferred cue.")
    return indices
