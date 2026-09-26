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

    def period_rates(self, start_ms, end_ms, *, correct_only=True):
        cache_key = (start_ms, end_ms, correct_only)
        if cache_key not in self._rates:
            time_mask = (self.times >= start_ms) & (self.times < end_ms)
            trial_indices = np.flatnonzero(self.correct) if correct_only else np.arange(len(self.cues))
            if not time_mask.any():
                firing_rates_hz = np.full((len(trial_indices), self.num_cells), np.nan)
            else:
                duration_seconds = time_mask.sum() * (self.times[1] - self.times[0]) / 1000
                firing_rates_hz = self.spikes[np.ix_(
                    trial_indices, np.flatnonzero(time_mask), np.arange(self.num_cells)
                )].sum(axis=1) / duration_seconds
            self._rates[cache_key] = firing_rates_hz
        return self._rates[cache_key]


def firing_rate(data, config):
    return data.period_rates(config.test_start_ms, config.test_end_ms).mean(axis=0)


def presence_ratio(data, config):
    presence_period_rates_hz = data.period_rates(config.presence_start_ms, config.presence_end_ms)
    trial_presence_ratios = (presence_period_rates_hz > 0).mean(axis=0)
    trial_presence_ratios[~np.all(np.isfinite(presence_period_rates_hz), axis=0)] = np.nan
    return trial_presence_ratios


def delay_variance(data, config):
    baseline_rates_hz = data.period_rates(config.variance_baseline_start_ms, config.variance_baseline_end_ms)
    delay_rates_hz = data.period_rates(config.variance_delay_start_ms, config.variance_delay_end_ms)
    if len(baseline_rates_hz) < config.variance_window_trials:
        return np.full(data.num_cells, np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        delay_to_baseline_ratios = np.var(delay_rates_hz, axis=0, ddof=1) / np.var(baseline_rates_hz, axis=0, ddof=1)
    delay_to_baseline_ratios[~np.isfinite(delay_to_baseline_ratios)] = np.nan
    return delay_to_baseline_ratios


def baseline_variance(data, config):
    baseline_rates_hz = data.period_rates(config.variance_baseline_start_ms, config.variance_baseline_end_ms)
    if len(baseline_rates_hz) < config.variance_window_trials:
        return np.full(data.num_cells, np.nan)
    baseline_trial_windows = np.lib.stride_tricks.sliding_window_view(
        baseline_rates_hz, config.variance_window_trials, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        window_to_global_variance_ratios = (
            np.var(baseline_trial_windows, axis=-1, ddof=1).mean(axis=0)
            / np.var(baseline_rates_hz, axis=0, ddof=1))
    window_to_global_variance_ratios[~np.isfinite(window_to_global_variance_ratios)] = np.nan
    return window_to_global_variance_ratios


def temporal_correlation(firing_rates_hz, trial_indices):
    """Pearson r for each cell; constant or unavailable measurements stay NaN."""
    drift_correlations = np.full(firing_rates_hz.shape[1], np.nan)
    if len(trial_indices) < 2:
        return drift_correlations
    trial_positions = np.asarray(trial_indices, dtype=float)
    centered_trial_positions = trial_positions - trial_positions.mean()
    centered_firing_rates = firing_rates_hz - firing_rates_hz.mean(axis=0)
    correlation_denominator = np.sqrt(
        np.sum(centered_trial_positions ** 2) * np.sum(centered_firing_rates ** 2, axis=0))
    np.divide(np.sum(centered_trial_positions[:, None] * centered_firing_rates, axis=0),
              correlation_denominator, out=drift_correlations,
              where=np.isfinite(correlation_denominator) & (correlation_denominator > 0))
    return np.clip(drift_correlations, -1, 1)


def baseline_drift(data, config):
    baseline_rates_hz = data.period_rates(config.baseline_drift_start_ms, config.baseline_drift_end_ms)
    return temporal_correlation(baseline_rates_hz, np.arange(len(baseline_rates_hz)))


@dataclass
class Selectivity:
    mean_pev_pct: np.ndarray
    preferred_cue: np.ndarray
    qualifying_bin_mask: np.ndarray
    passes_duration_check: np.ndarray
    has_finite_pev: np.ndarray


def circular_preferred_cue(preferred_cues_by_bin):
    """Round the circular mean to a cue, or return NaN when it is undefined.

    Cancellation of opposing directions leaves only floating-point roundoff.
    The tolerance is the standard accumulation bound gamma_n = n*u/(1-n*u),
    plus four rounding units for forming each two-component unit vector. It is
    a numerical-zero test, not a minimum directional-selectivity threshold.
    Defined means retain the existing circular-mean and cue-rounding policy.
    """
    preferred_cues_by_bin = np.asarray(preferred_cues_by_bin, dtype=np.float64)
    if preferred_cues_by_bin.size == 0 or not np.all(np.isfinite(preferred_cues_by_bin)):
        return np.nan
    preferred_angles_radians = np.deg2rad((preferred_cues_by_bin - 1) * 45 - 135)
    resultant_length = abs(np.exp(1j * preferred_angles_radians).mean())
    rounding_unit = np.finfo(np.float64).eps / 2
    accumulated_roundoff = preferred_angles_radians.size * rounding_unit
    cancellation_tolerance = (
        accumulated_roundoff / (1 - accumulated_roundoff) + 4 * rounding_unit
    )
    if resultant_length <= cancellation_tolerance:
        return np.nan
    mean_preferred_angle_degrees = np.rad2deg(
        circmean(preferred_angles_radians, high=np.pi, low=-np.pi)
    )
    return (np.round((mean_preferred_angle_degrees + 135) / 45 + 1).astype(int) - 1) % 8 + 1


def selectivity(data, config):
    """Measure PEV and cue preference even when their rejection gate is disabled.

    With the gate enabled, qualifying cells use their above-threshold runs for
    metadata. With it disabled, all finite bins are used and no run test is made.
    Nonqualifying cells also use all finite bins for optional preferred drift.
    """
    bin_start_times_ms = np.arange(config.test_start_ms, config.test_end_ms + 1, config.selectivity_bin_step_ms)
    correct_trial_bin_rates_hz = compute_binned_rates(
        data.spikes[data.correct], data.times, bin_start_times_ms,
        config.selectivity_bin_width_ms, dtype=np.float64)
    correct_trial_cues = data.cues[data.correct]
    pev_pct_by_bin, preferred_cues_by_bin = pev_and_preferred_cue(
        correct_trial_bin_rates_hz, correct_trial_cues, np.unique(correct_trial_cues))
    pev_pct_by_bin = np.clip(pev_pct_by_bin, config.selectivity_pev_floor_pct, 100)
    qualifying_bin_mask = np.zeros(pev_pct_by_bin.shape, dtype=bool)
    mean_pev_pct = np.full(data.num_cells, np.nan)
    preferred_cues = np.full(data.num_cells, np.nan)
    has_finite_pev = np.isfinite(pev_pct_by_bin).any(axis=1)
    for cell_index in np.flatnonzero(has_finite_pev):
        if config.check_selectivity:
            _, qualifying_bin_mask[cell_index] = get_periods_and_mask(
                pev_pct_by_bin[cell_index], config.selectivity_min_duration_ms / config.selectivity_bin_step_ms,
                config.selectivity_pev_threshold_pct)
        metadata_bin_mask = (qualifying_bin_mask[cell_index] if qualifying_bin_mask[cell_index].any()
                             else np.isfinite(pev_pct_by_bin[cell_index]))
        mean_pev_pct[cell_index] = pev_pct_by_bin[cell_index, metadata_bin_mask].mean()
        preferred_cues[cell_index] = circular_preferred_cue(
            preferred_cues_by_bin[cell_index, metadata_bin_mask]
        )
    return Selectivity(mean_pev_pct, preferred_cues, qualifying_bin_mask,
                       qualifying_bin_mask.any(axis=1), has_finite_pev)


def preferred_cue_drift(data, config, preferred_cues):
    test_period_rates_hz = data.period_rates(config.test_start_ms, config.test_end_ms, correct_only=False)
    preferred_cue_drift_r = np.full(data.num_cells, np.nan)
    for cue in np.unique(preferred_cues[np.isfinite(preferred_cues)]):
        cell_indices = np.flatnonzero(preferred_cues == cue)
        cue_trial_indices = np.flatnonzero(data.cues == cue)
        preferred_cue_drift_r[cell_indices] = temporal_correlation(
            test_period_rates_hz[np.ix_(cue_trial_indices, cell_indices)], cue_trial_indices)
    return preferred_cue_drift_r
