"""Numerical screening kernels with a stable import name for Numba caches."""
import numpy as np
from numba import njit


def pev_and_preferred_cue(rates, labels, conditions):
    """Compute omega-squared and preferred cue for every bin/cell at once."""
    values = np.asarray(rates, dtype=np.float64)
    labels = np.asarray(labels)
    conditions = np.asarray(conditions)
    if values.ndim != 3 or labels.shape != (values.shape[0],):
        raise ValueError('Expected rates (trial, bin, cell) and labels (trial,).')
    if conditions.size < 2 or values.shape[0] <= conditions.size:
        raise ValueError('PEV requires at least two conditions and positive residual degrees of freedom.')
    total = np.sum((values - values.mean(axis=0)) ** 2, axis=0)
    within = np.zeros_like(total)
    means = []
    for condition in conditions:
        group = values[labels == condition]
        if not group.shape[0]:
            raise ValueError('Every condition must have at least one trial.')
        mean = group.mean(axis=0)
        means.append(mean)
        within += np.sum((group - mean) ** 2, axis=0)
    mse = within / (values.shape[0] - conditions.size)
    omega = np.divide(total - within - (conditions.size - 1) * mse,
                      mse + total, out=np.zeros_like(total), where=total != 0) * 100
    preferred = conditions[np.argmax(np.stack(means), axis=0)]
    return omega.T, preferred.T


@njit(cache=True)
def get_periods_and_mask(sig: np.ndarray, min_window_width: int | float, sig_threshold: int | float | np.ndarray, greater_than: bool = True):
    """Return threshold-crossing periods and a mask marking those samples."""
    if greater_than:
        above_thresh = np.nonzero(sig > sig_threshold)[0]
    else:
        above_thresh = np.nonzero(sig < sig_threshold)[0]
    above_thresh_extended = np.zeros(len(above_thresh) + 2).astype(np.int64)
    above_thresh_extended[0] = -2
    above_thresh_extended[1:-1] = above_thresh
    above_thresh_extended[-1] = len(sig) + 1
    left_idx = np.nonzero(np.diff(above_thresh_extended) > 1)[0]
    right_idx = left_idx - 1
    left_idx = left_idx[:-1]
    right_idx = right_idx[1:]
    res = np.zeros((len(left_idx), 3)).astype(np.int64)
    for i in range(len(left_idx)):
        left = above_thresh[left_idx[i]]
        right = above_thresh[right_idx[i]]
        w = right - left + 1
        res[i] = left, right, w
    res = res[res[:, -1] >= min_window_width]
    mask = np.zeros_like(sig) # prepare a mask with the same shape as sig
    for left, right, w in res:
        mask[left:left + w] = 1
    return res, mask.astype(np.bool_)
