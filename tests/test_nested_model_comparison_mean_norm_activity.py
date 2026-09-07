import unittest

import pandas as pd

from scripts.nested_model_comparison_mean_norm_activity import (
    ACTIVITY_PREDICTORS,
    COUNT_PREDICTORS,
    Config,
    _model_specs,
    _nested_contrast_table,
)


class NestedMeanNormalizedActivityModelTest(unittest.TestCase):
    def test_model_sequence_matches_requested_cumulative_blocks(self):
        specs = _model_specs("outcome")

        self.assertEqual([spec.name for spec in specs], [
            "M0",
            "M1",
            "M2",
            "M3",
            "M4",
            "M5",
        ])
        self.assertEqual(specs[0].predictors, ())
        self.assertEqual(specs[1].predictors, COUNT_PREDICTORS)
        for stage in range(2, 6):
            self.assertEqual(
                specs[stage].predictors,
                (*COUNT_PREDICTORS, *ACTIVITY_PREDICTORS[: stage - 1]),
            )
            self.assertEqual(specs[stage].parent, f"M{stage - 1}")
        self.assertTrue(all(spec.outcome == "outcome" for spec in specs))

    def test_history_alpha_is_not_public_configuration(self):
        self.assertFalse(hasattr(Config(), "history_alpha"))

    def test_contrast_table_uses_full_minus_parent_r2(self):
        specs = _model_specs("outcome")
        rows = []
        for stage, spec in enumerate(specs):
            rows.append(
                {
                    "model": spec.name,
                    "marginal_r2": stage * 0.02,
                    "conditional_r2": 0.3 + stage * 0.01,
                    "log_likelihood": -100 + stage,
                    "aic": 220 - stage,
                    "bic": 230 + stage,
                    "marginal_rmse_ms": 10 - stage * 0.2,
                    "conditional_rmse_ms": 9 - stage * 0.2,
                    "marginal_mae_ms": 8 - stage * 0.1,
                    "conditional_mae_ms": 7 - stage * 0.1,
                    "random_intercept_variance": 3 - stage * 0.1,
                    "residual_variance": 4 - stage * 0.1,
                    "likelihood_ratio_vs_parent": 2.0,
                    "likelihood_ratio_df": len(spec.predictors)
                    - len(specs[stage - 1].predictors)
                    if stage
                    else float("nan"),
                    "likelihood_ratio_p_value": 0.1,
                }
            )

        contrasts = _nested_contrast_table(pd.DataFrame(rows), specs)

        self.assertEqual(contrasts["contrast"].tolist(), [
            "M1-vs-M0",
            "M2-vs-M1",
            "M3-vs-M2",
            "M4-vs-M3",
            "M5-vs-M4",
        ])
        self.assertAlmostEqual(contrasts.iloc[1]["delta_marginal_r2"], 0.02)
        self.assertAlmostEqual(contrasts.iloc[1]["delta_conditional_r2"], 0.01)
        self.assertEqual(
            contrasts.iloc[1]["added_predictors"], ACTIVITY_PREDICTORS[0]
        )
        self.assertAlmostEqual(
            contrasts.iloc[1]["conditional_rmse_improvement_ms"], 0.2
        )


if __name__ == "__main__":
    unittest.main()
