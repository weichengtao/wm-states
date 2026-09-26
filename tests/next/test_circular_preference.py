"""Undefined circular cue means must never create an arbitrary preferred cue."""

import dataclasses
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from scripts.next import cell_screening as screening
from scripts.next import screening_checks as checks


class CircularPreferredCueTest(unittest.TestCase):
    def test_opposing_and_symmetric_preferences_are_unavailable(self):
        examples = ([1, 5], [2, 6], [3, 7], [4, 8], [1, 3, 5, 7],
                    list(range(1, 9)), [1, 5] * 1000)
        for preferred_cues in examples:
            with self.subTest(preferred_cues=preferred_cues[:8]):
                self.assertTrue(np.isnan(checks.circular_preferred_cue(preferred_cues)))

    def test_direction_wrap_boundary_retains_valid_preferences(self):
        for preferred_cues, expected in (
            ([8, 8, 1], 8), ([8, 8, 7], 8), ([8, 1, 1], 1), ([8, 7, 7], 7),
            ([1], 1), ([8], 8), ([1, 3], 2),
        ):
            with self.subTest(preferred_cues=preferred_cues):
                self.assertEqual(checks.circular_preferred_cue(preferred_cues), expected)

    def test_small_but_defined_resultant_is_not_a_new_rejection_threshold(self):
        nearly_opposing = [1, 5] * 1000 + [1]
        self.assertEqual(checks.circular_preferred_cue(nearly_opposing), 1)

    def test_unavailable_inputs_do_not_create_a_direction(self):
        for preferred_cues in ([], [np.nan], [np.inf], [1, np.nan]):
            with self.subTest(preferred_cues=preferred_cues):
                self.assertTrue(np.isnan(checks.circular_preferred_cue(preferred_cues)))

    def test_selectivity_preserves_pev_and_gate_result_when_direction_is_undefined(self):
        data = checks.SessionMeasurements(
            np.ones((4, 4, 2)), np.arange(500, 700, 50),
            np.array([1, 5, 1, 5]), np.ones(4, dtype=bool),
        )
        config = screening.Config(
            test_start_ms=500, test_end_ms=550, selectivity_bin_step_ms=50,
            selectivity_min_duration_ms=100,
        )
        with patch.object(checks, 'pev_and_preferred_cue', return_value=(
            np.array([[10., 20.], [10., 20.]]), np.array([[1, 5], [8, 8]])
        )):
            result = checks.selectivity(data, config)
        np.testing.assert_array_equal(result.mean_pev_pct, [15, 15])
        np.testing.assert_array_equal(result.passes_duration_check, [True, True])
        np.testing.assert_array_equal(result.has_finite_pev, [True, True])
        np.testing.assert_array_equal(result.preferred_cue, [np.nan, 8])

    def test_only_qualifying_bins_define_preference_when_selectivity_is_enabled(self):
        data = checks.SessionMeasurements(
            np.ones((4, 5, 1)), np.arange(500, 750, 50),
            np.array([1, 5, 1, 5]), np.ones(4, dtype=bool),
        )
        config = screening.Config(
            test_start_ms=500, test_end_ms=650, selectivity_bin_step_ms=50,
            selectivity_min_duration_ms=100,
        )
        with patch.object(checks, 'pev_and_preferred_cue', return_value=(
            np.array([[10., 10., 0., 0.]]), np.array([[1, 1, 5, 5]])
        )):
            enabled = checks.selectivity(data, config)
            disabled = checks.selectivity(data, dataclasses.replace(config, check_selectivity=False))
        self.assertEqual(enabled.preferred_cue[0], 1)
        self.assertTrue(np.isnan(disabled.preferred_cue[0]))


class UnavailableCueSessionTest(unittest.TestCase):
    def setUp(self):
        self.config = screening.Config(
            **{f'check_{name}': False for name in ('min_trials', *screening.CHECK_NAMES)},
            test_start_ms=500, test_end_ms=550, selectivity_bin_step_ms=50,
        )
        self.data = (
            np.ones((8, 4, 2)), np.arange(500, 700, 50),
            np.array([1, 5] * 4), np.ones(8, dtype=bool),
        )
        self.selectivity = checks.Selectivity(
            mean_pev_pct=np.array([15., 15.]), preferred_cue=np.array([np.nan, 1.]),
            qualifying_bin_mask=np.ones((2, 2), dtype=bool),
            passes_duration_check=np.ones(2, dtype=bool),
            has_finite_pev=np.ones(2, dtype=bool),
        )

    def run_session(self, config):
        with patch.object(screening, 'load_session', return_value=self.data), \
                patch.object(checks, 'selectivity', return_value=self.selectivity):
            return screening.process_session(Path('session_ambiguous.mat'), config)

    def test_selected_undefined_cue_fails_with_session_cell_and_action_context(self):
        for preferred_drift in (False, True):
            config = dataclasses.replace(self.config, check_preferred_cue_drift=preferred_drift)
            with self.subTest(preferred_drift=preferred_drift), \
                    patch.object(checks, 'preferred_cue_drift') as drift, \
                    self.assertRaisesRegex(ValueError, r'session_ambiguous: selected cells \[0\].*Inspect'):
                self.run_session(config)
            drift.assert_not_called()

    def test_already_rejected_undefined_cue_warns_without_dropping_valid_selected_cells(self):
        self.selectivity.passes_duration_check[0] = False
        config = dataclasses.replace(self.config, check_selectivity=True)
        with self.assertWarnsRegex(RuntimeWarning, r'session_ambiguous: cells \[0\].*already rejected'):
            result, rows = self.run_session(config)
        np.testing.assert_array_equal(result['cell_idx_selected'], [1])
        np.testing.assert_array_equal(result['cell_idx_stationary'], [0, 1])
        np.testing.assert_array_equal(result['cell_properties']['preferred_cue'], [1])
        self.assertEqual(rows[0]['rejection_reason'], 'fail_selectivity')

    def test_undefined_cue_on_cell_rejected_by_another_check_also_warns(self):
        config = dataclasses.replace(self.config, check_presence_ratio=True)
        with patch.object(checks, 'presence_ratio', return_value=np.array([0., 1.])), \
                self.assertWarnsRegex(RuntimeWarning, r'cells \[0\].*already rejected'):
            result, rows = self.run_session(config)
        np.testing.assert_array_equal(result['cell_idx_selected'], [1])
        self.assertEqual(rows[0]['rejection_reason'], 'fail_presence_ratio')

    def test_preferred_drift_leaves_undefined_cell_unavailable(self):
        data = checks.SessionMeasurements(*self.data)
        trial_rates = np.column_stack([np.arange(8), np.arange(8)])
        with patch.object(data, 'period_rates', return_value=trial_rates):
            correlations = checks.preferred_cue_drift(data, self.config, self.selectivity.preferred_cue)
        np.testing.assert_allclose(correlations, [np.nan, 1])

    def test_no_selected_cells_warns_and_returns_diagnostics(self):
        self.selectivity.passes_duration_check[:] = False
        config = dataclasses.replace(self.config, check_selectivity=True)
        with self.assertWarnsRegex(RuntimeWarning, r'cells \[0\].*already rejected'):
            result, rows = self.run_session(config)
        self.assertIsNone(result)
        self.assertEqual(len(rows), 2)


if __name__ == '__main__':
    unittest.main()
