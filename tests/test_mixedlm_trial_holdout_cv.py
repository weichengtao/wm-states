import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.compare_mixed_effect_models import (
    ModelSpec,
    _fit_model,
    _predictions_and_r2,
)
from scripts.mixedlm_outcomes import TOTAL_OUTCOME
from scripts.mixedlm_trial_holdout_cv import (
    CVModelRequest,
    TrialHoldoutConfig,
    _rank_models_by_marginal_rmse,
    _test_predictions,
    build_fold_frame,
    run_trial_holdout_cv,
)
from scripts.prepare_data_for_mixedlm import (
    CV_CACHE_SCHEMA_VERSION,
    GROUP_NAMES,
    PERIODS,
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


class FoldActivityWeightingTest(unittest.TestCase):
    def test_uses_pev_for_selective_groups_but_not_stationary_group(self):
        preferred_rates = np.asarray(
            [[0.0, 10.0], [1.0, 14.0], [3.0, 12.0], [5.0, 14.0]]
        )
        stationary_rates = np.asarray(
            [[10.0, 0.0], [14.0, 1.0], [12.0, 3.0], [14.0, 5.0]]
        )
        raw_by_period = {
            period: {
                "preferred": preferred_rates,
                "selective_nonpreferred": preferred_rates[:, :1],
                "stationary_nonselective": stationary_rates,
            }
            for period in PERIODS
        }
        cache = {
            "sessions": [
                {
                    "session": "example",
                    "preferred_cue": 1,
                    "trial_ids": np.asarray([0, 1, 2, 3]),
                    "total_off_state_duration_ms": np.asarray([0, 10, 20, 30]),
                    "maximum_off_state_duration_ms": np.asarray([0, 5, 10, 15]),
                    "cell_counts": {
                        f"{group}_cell_count": 2 for group in GROUP_NAMES
                    },
                    "raw_firing_rates_hz": raw_by_period,
                    "activity_weights": {
                        "preferred": np.asarray([1.0, 3.0]),
                        "selective_nonpreferred": np.asarray([1.0]),
                        "stationary_nonselective": None,
                    },
                }
            ]
        }
        split = {"test_trial_ids_by_session": {"example": np.asarray([3])}}

        equal = build_fold_frame(cache, split, 0.0, 0.2, False)
        weighted = build_fold_frame(cache, split, 0.0, 0.2, True)

        self.assertNotEqual(
            equal.loc[2, "baseline_mean_normalized_activity_preferred"],
            weighted.loc[2, "baseline_mean_normalized_activity_preferred"],
        )
        np.testing.assert_allclose(
            equal["baseline_mean_normalized_activity_stationary_nonselective"],
            weighted["baseline_mean_normalized_activity_stationary_nonselective"],
        )

    def test_rejects_a_cache_from_the_other_weighting_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cv.pkl"
            with cache_path.open("wb") as handle:
                pickle.dump(
                    {
                        "schema_version": CV_CACHE_SCHEMA_VERSION,
                        "sessions": [{}],
                        "pev_weighted_average": True,
                    },
                    handle,
                )
            request = CVModelRequest(
                spec=ModelSpec(
                    name="M0",
                    description="test",
                    predictors=(),
                    parent=None,
                )
            )

            with self.assertRaisesRegex(ValueError, "weighting mode"):
                run_trial_holdout_cv(
                    cache_path,
                    [request],
                    Path(tmpdir) / "output",
                    TrialHoldoutConfig(pev_weighted_average=False),
                )


if __name__ == "__main__":
    unittest.main()
