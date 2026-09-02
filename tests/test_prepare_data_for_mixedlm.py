import unittest

import numpy as np

from scripts.compare_mixed_effect_models import ModelSpec
from scripts.mixedlm_outcomes import (
    MAXIMUM_OUTCOME,
    TOTAL_OUTCOME,
    select_outcomes,
)
from scripts.prepare_data_for_mixedlm import _validate_off_state_result


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
