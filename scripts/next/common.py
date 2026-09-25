"""Shared session validation, binning, provenance and bounded parallelism."""
from dataclasses import asdict
from enum import Enum
import hashlib
import json
from pathlib import Path

import numpy as np
from joblib import parallel_config
from scipy.io import loadmat


def worker_context(n_jobs):
    if n_jobs == 0:
        raise ValueError('n_jobs must be positive or a negative joblib CPU count.')
    return parallel_config(backend='loky', n_jobs=n_jobs, inner_max_num_threads=1)


def load_session(path):
    data = loadmat(path, variable_names=['spks', 'tc', 'cueAngIdx', 'isCorr'])
    spikes = np.asarray(data['spks'])
    times = np.asarray(data['tc'], dtype=float).ravel()
    cues = np.asarray(data['cueAngIdx']).ravel().astype(np.int64)
    correct = np.asarray(data['isCorr']).ravel().astype(bool)
    if spikes.ndim != 3 or spikes.shape[:2] != (cues.size, times.size) or correct.shape != cues.shape:
        raise ValueError(f'{path}: inconsistent trial/time/cell dimensions.')
    if times.size < 2 or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0) or not np.allclose(np.diff(times), times[1] - times[0]):
        raise ValueError(f'{path}: timestamps must be finite, increasing and uniformly sampled.')
    if not np.all(np.isin(cues, np.arange(1, 9))):
        raise ValueError(f'{path}: cue labels must be integers from 1 to 8.')
    return spikes, times, cues, correct


def compute_binned_rates(spikes, times, starts, window_ms, *, dtype=np.float32):
    """Overlapping rectangular bins in O(trials * (samples + bins) * cells).

    Chunk trials to cap the float64 prefix-sum workspace. Duration follows the
    number of recorded samples in each half-open window, including edge bins.
    """
    spikes = np.asarray(spikes)
    times = np.asarray(times, dtype=float)
    starts = np.asarray(starts, dtype=float)
    if spikes.ndim != 3 or times.shape != (spikes.shape[1],) or times.size < 2:
        raise ValueError('Expected spikes (trial, time, cell) and matching timestamps.')
    if window_ms <= 0 or starts.ndim != 1 or not np.all(np.isfinite(starts)):
        raise ValueError('Bin width must be positive and starts must be finite.')
    dt = times[1] - times[0]
    if dt <= 0 or not np.all(np.isfinite(times)) or not np.allclose(np.diff(times), dt):
        raise ValueError('Timestamps must be finite, increasing and uniform.')
    left = np.searchsorted(times, starts)
    right = np.searchsorted(times, starts + window_ms)
    duration = (right - left) * dt / 1000
    if np.any(duration <= 0):
        raise ValueError('A requested bin has no recorded samples.')
    out = np.empty((spikes.shape[0], starts.size, spikes.shape[2]), dtype=dtype)
    chunk = max(1, 32 * 1024**2 // max(8 * (times.size + 1) * spikes.shape[2], 1))
    for begin in range(0, spikes.shape[0], chunk):
        values = spikes[begin:begin + chunk]
        if not np.all(np.isfinite(values)):
            raise ValueError('Spike counts must be finite.')
        cumulative = np.empty((values.shape[0], times.size + 1, values.shape[2]), dtype=np.float64)
        cumulative[:, 0] = 0
        np.cumsum(values, axis=1, dtype=np.float64, out=cumulative[:, 1:])
        out[begin:begin + chunk] = (cumulative[:, right] - cumulative[:, left]) / duration[None, :, None]
    return out


def full_session_selection(results, session, num_trials):
    matches = [r for r in results if str(r['session']) == str(session)]
    if len(matches) != 1 or matches[0].get('num_trials') != num_trials:
        raise ValueError(f'Expected one full-session selection with {num_trials} trials for {session}.')
    return matches[0]


def json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def fingerprint(config, paths=(), *, exclude=()):
    settings = {k: v for k, v in asdict(config).items() if k not in exclude}
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True, default=json_value).encode())
    # Include code so checkpoints cannot outlive a changed implementation.
    for path in sorted(Path(__file__).parent.glob('*.py')):
        digest.update(path.read_bytes())
    for path in paths:
        path = Path(path)
        digest.update(str(path.resolve()).encode())
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
    return digest.hexdigest()


def session_files(data_dir, session_list_file=None, max_sessions_to_run=None):
    files = sorted(Path(data_dir).glob('*.mat'))
    if not files:
        raise FileNotFoundError(f'No .mat session files in {data_dir}.')
    if session_list_file is not None:
        requested = {line.split('#', 1)[0].strip() for line in Path(session_list_file).read_text().splitlines()}
        requested.discard('')
        unknown = requested - {p.stem for p in files}
        if unknown:
            import warnings
            warnings.warn(f'Unknown sessions: {sorted(unknown)}', stacklevel=2)
        files = [p for p in files if p.stem in requested]
    if max_sessions_to_run is not None:
        if max_sessions_to_run < 1:
            raise ValueError('max_sessions_to_run must be positive.')
        files = files[:max_sessions_to_run]
    if not files:
        raise ValueError('Session filter selected no data files.')
    return files
