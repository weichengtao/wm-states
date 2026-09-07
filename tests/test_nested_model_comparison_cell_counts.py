import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from scripts.compare_mixed_effect_models import COUNT_PREDICTORS
from scripts.nested_model_comparison_cell_counts import (
    _contrast_specs,
    _cv_contrast_repeat_metrics,
    _model_specs,
    _nested_contrast_rows,
    _summarize_cv_contrasts,
)


class NestedCellCountModelTest(unittest.TestCase):
    def test_model_specs_match_requested_drop_one_models(self):
        specs = {spec.name: spec for spec in _model_specs("outcome")}

        self.assertEqual(set(specs), {
            "M0",
            "M1",
            "M1-drop-preferred",
            "M1-drop-non-preferred",
            "M1-drop-non-selective",
        })
        self.assertEqual(specs["M0"].predictors, ())
        self.assertEqual(specs["M1"].predictors, COUNT_PREDICTORS)
        self.assertNotIn(
            "preferred_cell_count",
            specs["M1-drop-preferred"].predictors,
        )
        self.assertNotIn(
            "selective_nonpreferred_cell_count",
            specs["M1-drop-non-preferred"].predictors,
        )
        self.assertNotIn(
            "stationary_nonselective_cell_count",
            specs["M1-drop-non-selective"].predictors,
        )
        self.assertTrue(
            all(spec.outcome == "outcome" for spec in specs.values())
        )

    def test_nested_contrasts_use_full_minus_reduced_direction(self):
        model_metrics = {
            "M0": (0.00, 0.25, -110.0, 224.0, 230.0, 8.0, 6.0),
            "M1": (0.20, 0.45, -100.0, 210.0, 220.0, 6.0, 4.0),
            "M1-drop-preferred": (
                0.15,
                0.40,
                -103.0,
                214.0,
                223.0,
                7.0,
                5.0,
            ),
            "M1-drop-non-preferred": (
                0.18,
                0.43,
                -101.0,
                210.0,
                219.0,
                6.5,
                4.5,
            ),
            "M1-drop-non-selective": (
                0.19,
                0.44,
                -100.5,
                209.0,
                218.0,
                6.2,
                4.2,
            ),
        }
        rows = []
        results = {}
        specs = {spec.name: spec for spec in _model_specs("outcome")}
        for name, values in model_metrics.items():
            marginal_r2, conditional_r2, llf, aic, bic, rmse, mae = values
            rows.append(
                {
                    "model": name,
                    "marginal_r2": marginal_r2,
                    "conditional_r2": conditional_r2,
                    "log_likelihood": llf,
                    "aic": aic,
                    "bic": bic,
                    "marginal_rmse_ms": rmse + 1,
                    "conditional_rmse_ms": rmse,
                    "marginal_mae_ms": mae + 1,
                    "conditional_mae_ms": mae,
                    "random_intercept_variance": 2.0,
                    "residual_variance": 3.0,
                }
            )
            exog_names = ["Intercept", *specs[name].predictors]
            results[name] = SimpleNamespace(
                llf=llf,
                df_modelwc=2 + len(specs[name].predictors),
                model=SimpleNamespace(exog_names=exog_names),
            )

        contrasts = _nested_contrast_rows(
            pd.DataFrame(rows), results, _contrast_specs(), 0.05
        ).set_index("contrast")

        preferred = contrasts.loc["preferred-count"]
        self.assertAlmostEqual(preferred["delta_marginal_r2"], 0.05)
        self.assertAlmostEqual(preferred["delta_conditional_r2"], 0.05)
        self.assertAlmostEqual(preferred["likelihood_ratio"], 6.0)
        self.assertEqual(preferred["likelihood_ratio_df"], 1)
        self.assertAlmostEqual(
            preferred["conditional_rmse_improvement_ms"], 1.0
        )
        self.assertAlmostEqual(preferred["conditional_mae_improvement_ms"], 1.0)
        self.assertEqual(contrasts.loc["M1-vs-M0", "likelihood_ratio_df"], 3)

    def test_cv_contrasts_are_paired_within_repeat(self):
        specs = _model_specs("outcome")
        rows = []
        for repeat in range(2):
            for spec in specs:
                if spec.name == "M0":
                    score, df, llf = 0.10, 2, -110.0
                elif spec.name == "M1":
                    score, df, llf = 0.30, 5, -100.0
                else:
                    score, df, llf = 0.25, 4, -102.0
                row = {
                    "repeat": repeat,
                    "model": spec.name,
                    "fit_success": True,
                    "fit_error": "",
                    "train_df_modelwc": df,
                    "train_log_likelihood": llf,
                    "train_aic": -2 * llf + 2 * df,
                    "train_bic": -2 * llf + np.log(100) * df,
                    "train_marginal_r2": score,
                    "train_conditional_r2": score + 0.2,
                    "train_random_intercept_variance": 2.0,
                    "train_residual_variance": 3.0,
                    "train_marginal_rmse_ms": 10.0 - score,
                    "train_conditional_rmse_ms": 9.0 - score,
                    "train_marginal_mae_ms": 8.0 - score,
                    "train_conditional_mae_ms": 7.0 - score,
                }
                for prefix, offset in (("fixed", 0.0), ("conditional", 0.1)):
                    row[f"{prefix}_r2"] = score + offset
                    row[f"{prefix}_session_centered_r2"] = score + offset - 0.05
                    row[f"{prefix}_rmse_ms"] = 10.0 - score - offset
                    row[f"{prefix}_mae_ms"] = 8.0 - score - offset
                    row[f"{prefix}_pearson_r"] = score + offset + 0.2
                rows.append(row)

        repeat_metrics = _cv_contrast_repeat_metrics(
            pd.DataFrame(rows), _contrast_specs(), 0.05
        )
        self.assertEqual(len(repeat_metrics), 8)
        preferred = repeat_metrics[
            repeat_metrics["contrast"] == "preferred-count"
        ]
        np.testing.assert_allclose(
            preferred["heldout_delta_marginal_r2"], 0.05
        )
        np.testing.assert_allclose(
            preferred["heldout_delta_conditional_r2"], 0.05
        )
        np.testing.assert_allclose(
            preferred["conditional_rmse_ms_improvement_reduced_minus_full"],
            0.05,
        )

        summary = _summarize_cv_contrasts(repeat_metrics).set_index("contrast")
        self.assertEqual(summary.loc["preferred-count", "n_successful_pairs"], 2)
        self.assertAlmostEqual(
            summary.loc[
                "preferred-count", "heldout_delta_conditional_r2_mean"
            ],
            0.05,
        )


if __name__ == "__main__":
    unittest.main()
