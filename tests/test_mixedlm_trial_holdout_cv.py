import unittest

import numpy as np
import pandas as pd

from scripts.compare_mixed_effect_models import (
    ModelSpec,
    _fit_model,
    _predictions_and_r2,
)
from scripts.mixedlm_outcomes import TOTAL_OUTCOME
from scripts.mixedlm_trial_holdout_cv import (
    _rank_models_by_marginal_rmse,
    _test_predictions,
)


class TestPredictionsTest(unittest.TestCase):
    def test_ranks_models_by_held_out_marginal_rmse(self):
        summary = pd.DataFrame(
            {
                "model": ["conditional-winner", "marginal-winner", "failed"],
                "n_successful_fits": [5, 5, 0],
                "fixed_rmse_ms_mean": [2.0, 1.0, 0.1],
                "conditional_rmse_ms_mean": [0.5, 3.0, 0.1],
            }
        )

        ranked = _rank_models_by_marginal_rmse(summary)

        self.assertEqual(
            ranked["model"].tolist(),
            ["marginal-winner", "conditional-winner"],
        )

    def test_predicts_held_out_rows_with_fixed_and_random_effects(self):
        rng = np.random.default_rng(123)
        sessions = np.repeat([f"session-{index}" for index in range(6)], 20)
        predictor = rng.normal(size=sessions.size)
        session_intercepts = np.repeat(
            np.asarray([-2.0, -1.2, -0.4, 0.4, 1.2, 2.0]), 20
        )
        frame = pd.DataFrame(
            {
                TOTAL_OUTCOME.column: (
                    10.0
                    + 2.5 * predictor
                    + session_intercepts
                    + rng.normal(scale=0.2, size=sessions.size)
                ),
                "session": sessions,
                "predictor": predictor,
            }
        )
        position_within_session = frame.groupby("session").cumcount()
        train = frame[position_within_session < 15]
        test = frame[position_within_session >= 15]
        spec = ModelSpec(
            name="test-model",
            description="Regression test model",
            predictors=("predictor",),
            parent=None,
        )

        result, _ = _fit_model(train, spec, max_iterations=1000)
        fixed, conditional = _test_predictions(result, test)

        expected_fixed = np.asarray(result.predict(test), dtype=float)
        expected_offsets = np.asarray(
            [
                np.asarray(result.random_effects[session], dtype=float).ravel()[0]
                for session in test["session"].astype(str)
            ]
        )
        np.testing.assert_allclose(fixed, expected_fixed)
        np.testing.assert_allclose(conditional, expected_fixed + expected_offsets)

        train_metrics = _predictions_and_r2(result, train)
        observed = train[TOTAL_OUTCOME.column].to_numpy(dtype=float)
        np.testing.assert_allclose(
            train_metrics["marginal_mae_ms"],
            np.mean(np.abs(observed - train_metrics["fixed_prediction"])),
        )
        np.testing.assert_allclose(
            train_metrics["conditional_mae_ms"],
            np.mean(
                np.abs(observed - train_metrics["conditional_prediction"])
            ),
        )


if __name__ == "__main__":
    unittest.main()
