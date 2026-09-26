import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat

from scripts.next.compare_mixed_effect_models import ModelSpec
from scripts.next.mixedlm_outcomes import (
    MAXIMUM_OUTCOME,
    TOTAL_OUTCOME,
    select_outcomes,
)
from scripts.next.prepare_data_for_mixedlm import (
    Config,
    _cell_groups,
    _prepare_session_rows,
    _validate_off_state_result,
)


class OffStateOutcomeValidationTest(unittest.TestCase):
    def state_result(self):
        return {
            "off_state_duration_correction": "applied",
            "off_state_duration_delay_start": 500,
            "off_state_duration_delay_end": 1400,
            "trial_idx": np.asarray([8, 2, 5]),
            "off_state_duration_per_trial": np.asarray([40.0, 0.0, 70.0]),
            "max_off_state_duration_per_trial": np.asarray([30.0, 0.0, 50.0]),
        }

    def test_sorts_both_outcomes_with_trial_ids(self):
        trial_ids, total, maximum = _validate_off_state_result(
            self.state_result(), "example"
        )

        np.testing.assert_array_equal(trial_ids, [2, 5, 8])
        np.testing.assert_array_equal(total, [0.0, 70.0, 40.0])
        np.testing.assert_array_equal(maximum, [0.0, 50.0, 30.0])

    def test_rejects_maximum_larger_than_total(self):
        state_result = self.state_result()
        state_result["max_off_state_duration_per_trial"][0] = 50.0

        with self.assertRaisesRegex(ValueError, "exceeds total duration"):
            _validate_off_state_result(state_result, "example")

    def test_missing_maximum_requests_upstream_cache_regeneration(self):
        state_result = self.state_result()
        del state_result["max_off_state_duration_per_trial"]

        with self.assertRaisesRegex(ValueError, "rerun on_off_states.py"):
            _validate_off_state_result(state_result, "example")

    def test_rejects_fractional_ids_before_sorting_or_truncation(self):
        state_result = self.state_result()
        state_result["trial_idx"] = np.asarray([8, 2.5, 5])
        with self.assertRaisesRegex(ValueError, "Session example:.*finite integers"):
            _validate_off_state_result(state_result, "example")

    def test_rejects_duplicate_ids(self):
        state_result = self.state_result()
        state_result["trial_idx"] = np.asarray([8, 2, 8])
        with self.assertRaisesRegex(ValueError, "Session example:.*not unique"):
            _validate_off_state_result(state_result, "example")


class SharedPreparationInputsTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config = Config(data_dir=Path(self.directory.name))
        self.times_ms = np.arange(-400, 1500, 100)
        # Each cell's rate increases across trials. Group order and history
        # alignment can be checked independently from the raw cache values.
        self.spikes = np.ones((3, self.times_ms.size, 4)) * np.arange(1, 4)[:, None, None]
        savemat(self.config.data_dir / "example.mat", {
            "spks": self.spikes,
            "tc": self.times_ms,
            "cueAngIdx": np.ones(3),
            "isCorr": np.ones(3),
        })
        self.selection = {
            "session": "example",
            "num_trials": 3,
            "cell_idx_stationary": np.asarray([0, 1, 2, 3]),
            "screening_checks": {"selectivity": False},
            "cell_properties": {
                "cell_idx": np.asarray([2, 0, 1]),
                "preferred_cue": np.asarray([1, 1, 5]),
                "mean_selectivity_pev_pct": np.asarray([2.0, 8.0, 4.0]),
            },
        }
        self.state = {
            "session": "example",
            "cue": 1,
            "off_state_duration_correction": "applied",
            "off_state_duration_delay_start": 500,
            "off_state_duration_delay_end": 1400,
            "trial_idx": np.asarray([2, 0, 1]),
            "off_state_duration_per_trial": np.asarray([30.0, 10.0, 20.0]),
            "max_off_state_duration_per_trial": np.asarray([15.0, 5.0, 10.0]),
        }

    def test_grouping_preserves_selected_order_without_pev_ranking(self):
        groups = _cell_groups(self.selection, 1)
        np.testing.assert_array_equal(groups["preferred"], [2, 0])
        np.testing.assert_array_equal(groups["selective_nonpreferred"], [1])
        np.testing.assert_array_equal(groups["stationary_nonselective"], [3])

    def test_preparation_preserves_history_and_records_population_meaning(self):
        rows, cv_session = _prepare_session_rows(self.state, [self.selection], self.config)

        self.assertEqual([row["trial_id"] for row in rows], [1, 2])
        self.assertEqual([row["total_off_state_duration_ms"] for row in rows], [20, 30])
        self.assertEqual([row["maximum_off_state_duration_ms"] for row in rows], [10, 15])
        np.testing.assert_array_equal(cv_session["trial_ids"], [0, 1, 2])
        np.testing.assert_array_equal(
            cv_session["raw_firing_rates_hz"]["delay"]["preferred"],
            [[10, 10], [20, 20], [30, 30]],
        )
        self.assertAlmostEqual(rows[0]["delay_mean_normalized_activity_preferred"], 0)
        self.assertAlmostEqual(
            rows[0]["history_ema_delay_mean_normalized_activity_preferred"],
            -np.sqrt(1.5),
        )
        self.assertAlmostEqual(
            rows[1]["history_ema_delay_mean_normalized_activity_preferred"],
            -0.8 * np.sqrt(1.5),
        )
        self.assertEqual(cv_session["screening_checks"], {"selectivity": False})
        self.assertEqual(cv_session["population_labels"]["preferred"], "Selected preferred cells")
        self.assertNotIn("screening_checks", rows[0])

    def test_preparation_rejects_state_ids_outside_recorded_trials(self):
        self.state["trial_idx"] = np.asarray([3, 0, 1])
        with self.assertRaisesRegex(ValueError, "Session example:.*out of range"):
            _prepare_session_rows(self.state, [self.selection], self.config)

    def test_preparation_rejects_cell_ids_outside_recorded_cells(self):
        self.selection["cell_idx_stationary"] = np.asarray([0, 1, 2, 4])
        with self.assertRaisesRegex(ValueError, "outside the session's 4 cells"):
            _prepare_session_rows(self.state, [self.selection], self.config)

    def test_preparation_rejects_fractional_preferred_cue(self):
        self.state["cue"] = 1.5
        with self.assertRaisesRegex(ValueError, "Session example: preferred cue.*integer cue"):
            _prepare_session_rows(self.state, [self.selection], self.config)


class OutcomeSelectionTest(unittest.TestCase):
    def test_both_preserves_total_then_maximum_order(self):
        self.assertEqual(
            select_outcomes("both"),
            (TOTAL_OUTCOME, MAXIMUM_OUTCOME),
        )

    def test_model_formula_uses_selected_outcome(self):
        spec = ModelSpec(
            name="M0",
            description="test",
            predictors=(),
            parent=None,
            outcome=MAXIMUM_OUTCOME.column,
        )

        self.assertEqual(
            spec.formula,
            "maximum_off_state_duration_ms ~ 1",
        )


if __name__ == "__main__":
    unittest.main()
