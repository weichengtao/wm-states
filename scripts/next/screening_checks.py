"""Independent screening measurements; policy and enable switches live in Config."""
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import circmean

from scripts.next.common import compute_binned_rates
from scripts.next.selection_math import get_periods_and_mask, pev_and_preferred_cue


@dataclass
class SessionMeasurements:
    spikes: np.ndarray
    times: np.ndarray
    cues: np.ndarray
    correct: np.ndarray
    _rates: dict = field(default_factory=dict, init=False)

    @property
    def num_cells(self):
        return self.spikes.shape[2]

    def period_rates(self, start, end, *, correct_only=True):
        key = (start, end, correct_only)
        if key not in self._rates:
            mask = (self.times >= start) & (self.times < end)
            trials = np.flatnonzero(self.correct) if correct_only else np.arange(len(self.cues))
            if not mask.any():
                rates = np.full((len(trials), self.num_cells), np.nan)
            else:
                duration = mask.sum() * (self.times[1] - self.times[0]) / 1000
                rates = self.spikes[np.ix_(trials, np.flatnonzero(mask), np.arange(self.num_cells))].sum(axis=1) / duration
            self._rates[key] = rates
        return self._rates[key]


def firing_rate(data, config):
    return data.period_rates(config.t_test_start, config.t_test_end).mean(axis=0)


def presence_ratio(data, config):
    rates = data.period_rates(config.presence_start, config.presence_end)
    result = (rates > 0).mean(axis=0)
    result[~np.all(np.isfinite(rates), axis=0)] = np.nan
    return result


def delay_variance(data, config):
    baseline = data.period_rates(config.temp_check_baseline_start, config.temp_check_baseline_end)
    delay = data.period_rates(config.temp_check_delay_start, config.temp_check_delay_end)
    if len(baseline) < config.min_trial_for_temp_check:
        return np.full(data.num_cells, np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.var(delay, axis=0, ddof=1) / np.var(baseline, axis=0, ddof=1)
    result[~np.isfinite(result)] = np.nan
    return result


def baseline_variance(data, config):
    baseline = data.period_rates(config.temp_check_baseline_start, config.temp_check_baseline_end)
    if len(baseline) < config.min_trial_for_temp_check:
        return np.full(data.num_cells, np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(baseline, config.min_trial_for_temp_check, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.var(windows, axis=-1, ddof=1).mean(axis=0) / np.var(baseline, axis=0, ddof=1)
    result[~np.isfinite(result)] = np.nan
    return result


def temporal_correlation(values, indices):
    """Pearson r for each cell; constant or unavailable measurements stay NaN."""
    result = np.full(values.shape[1], np.nan)
    if len(indices) < 2:
        return result
    x = np.asarray(indices, dtype=float)
    x = x - x.mean()
    centered = values - values.mean(axis=0)
    denominator = np.sqrt(np.sum(x ** 2) * np.sum(centered ** 2, axis=0))
    np.divide(np.sum(x[:, None] * centered, axis=0), denominator,
              out=result, where=np.isfinite(denominator) & (denominator > 0))
    return np.clip(result, -1, 1)


def baseline_drift(data, config):
    baseline = data.period_rates(config.baseline_drift_start, config.baseline_drift_end)
    return temporal_correlation(baseline, np.arange(len(baseline)))


@dataclass
class Selectivity:
    mean_pev: np.ndarray
    preferred_cue: np.ndarray
    significant_bins: np.ndarray
    passes: np.ndarray
    applicable: np.ndarray


def selectivity(data, config):
    """Measure PEV and cue preference even when their rejection gate is disabled.

    With the gate enabled, qualifying cells use their above-threshold runs for
    metadata. With it disabled, all finite bins are used and no run test is made.
    Nonqualifying cells also use all finite bins for optional preferred drift.
    """
    starts = np.arange(config.t_test_start, config.t_test_end + 1, config.t_test_step)
    rates = compute_binned_rates(data.spikes[data.correct], data.times, starts,
                                config.t_test_window, dtype=np.float64)
    labels = data.cues[data.correct]
    pev, preferences = pev_and_preferred_cue(rates, labels, np.unique(labels))
    pev = np.clip(pev, config.pev_clip_at, 100)
    masks = np.zeros(pev.shape, dtype=bool)
    means = np.full(data.num_cells, np.nan)
    cues = np.full(data.num_cells, np.nan)
    applicable = np.isfinite(pev).any(axis=1)
    for cell in np.flatnonzero(applicable):
        if config.check_selectivity:
            _, masks[cell] = get_periods_and_mask(pev[cell], config.sig_pev_duration / config.t_test_step,
                                                  config.sig_pev_threshold)
        bins = masks[cell] if masks[cell].any() else np.isfinite(pev[cell])
        means[cell] = pev[cell, bins].mean()
        angles = np.deg2rad((preferences[cell, bins] - 1) * 45 - 135)
        angle = np.rad2deg(circmean(angles, high=np.pi, low=-np.pi))
        cues[cell] = (np.round((angle + 135) / 45 + 1).astype(int) - 1) % 8 + 1
    return Selectivity(means, cues, masks, masks.any(axis=1), applicable)


def preferred_drift(data, config, preferred_cues):
    rates = data.period_rates(config.t_test_start, config.t_test_end, correct_only=False)
    result = np.full(data.num_cells, np.nan)
    for cue in np.unique(preferred_cues[np.isfinite(preferred_cues)]):
        cells = np.flatnonzero(preferred_cues == cue)
        trials = np.flatnonzero(data.cues == cue)
        result[cells] = temporal_correlation(rates[np.ix_(trials, cells)], trials)
    return result
