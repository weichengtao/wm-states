import io
import tempfile
import unittest
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from scripts.next import cache_io
from scripts.next import compare_mixed_effect_models as models
from scripts.next import mixedlm_trial_holdout_cv as cv
from scripts.next import nested_model_comparison_cell_counts as counts
from scripts.next import nested_model_comparison_mean_norm_activity as activity
from scripts.next import find_active_cell_criticality as criticality
from scripts.next import test_interactions_across_periods as interactions
from scripts.next.find_active_cell_criticality import _criticality_summary
from scripts.next.prepare_data_for_mixedlm import CV_CACHE_SCHEMA_VERSION, GROUP_NAMES, PERIODS


class InferenceValidityTest(unittest.TestCase):
    def result(self):
        return SimpleNamespace(
            converged=True, llf=-10.0, scale=1.0,
            fe_params=pd.Series([10.0, 2.0], index=["Intercept", "x"]),
            bse_fe=pd.Series([0.1, 0.2], index=["Intercept", "x"]),
            cov_params=Mock(return_value=np.diag([0.01, 0.04, 0.1])),
        )

    def test_boundary_and_retry_warnings_do_not_invalidate_sound_covariance(self):
        self.assertEqual(models._inference_errors(self.result(), [
            "The MLE may be on the boundary of the parameter space.",
            "Retrying MixedLM optimization with bfgs", "Random effects covariance is singular",
        ]), [])

    def test_non_positive_definite_covariance_invalidates_inference(self):
        result = self.result()
        result.cov_params.return_value = np.asarray([[1, 2, 0], [2, 1, 0], [0, 0, 1]])
        self.assertTrue(any("covariance" in error for error in models._inference_errors(result, [])))

    def test_missing_nonfinite_or_zero_standard_errors_invalidate_inference(self):
        for standard_errors in ([0.1, np.nan], [0.1, 0.0], None):
            with self.subTest(standard_errors=standard_errors):
                result = self.result()
                if standard_errors is None:
                    del result.bse_fe
                else:
                    result.bse_fe = np.asarray(standard_errors)
                self.assertTrue(any("standard errors" in error for error in models._inference_errors(result, [])))

    def test_final_hessian_warning_is_recorded_and_inference_is_withheld(self):
        result = self.result()
        model = Mock(exog=np.asarray([[1, 0], [1, 1], [1, 2]]))
        def fitted(**kwargs):
            warnings.warn("The Hessian matrix at the estimated parameter values is not positive definite.")
            return result
        model.fit.side_effect = fitted
        spec = models.ModelSpec("M", "test", ("x",), None)
        with patch.object(models.smf, "mixedlm", return_value=model), self.assertWarnsRegex(RuntimeWarning, "inference withheld"):
            fitted_result, messages = models._fit_model(pd.DataFrame({"session": ["a", "a", "b"]}), spec, 100)
        self.assertIs(fitted_result, result)
        self.assertFalse(result.inference_valid)
        self.assertTrue(any("final Hessian" in message for message in messages))
        rows = models._fixed_effect_rows(result, spec, 0.05)
        self.assertEqual(rows[1]["coefficient"], 2.0)
        for row in rows:
            self.assertFalse(row["inference_valid"])
            self.assertFalse(row["significant"])
            for key in ("p_value", "z_value", "std_error", "ci_95_lower", "ci_95_upper"):
                self.assertTrue(np.isnan(row[key]), key)

    def test_nonidentifiable_design_is_rejected_before_optimization(self):
        model = Mock(exog=np.asarray([[1, 1], [1, 1], [1, 1]]))
        with patch.object(models.smf, "mixedlm", return_value=model), self.assertRaisesRegex(RuntimeError, "rank-deficient"):
            models._fit_model(pd.DataFrame({"session": ["a", "a", "b"]}), models.ModelSpec("M", "test", ("x",), None), 100)
        model.fit.assert_not_called()

    def test_failed_model_is_recorded_without_preventing_independent_fit(self):
        frame = pd.DataFrame({"session": ["a", "b"]})
        config = models.Config()
        specs = [models.ModelSpec(name, "test", (), None) for name in ("bad", "good")]
        good_result = object()
        with patch.object(models, "_fit_model", side_effect=[RuntimeError("rank deficient"), (good_result, [])]), \
                patch.object(models, "_predictions_and_r2", return_value={}), \
                patch.object(models, "_fixed_effect_rows", return_value=[]), \
                patch.object(models, "_comparison_row", return_value={"fit_success": True}), \
                self.assertWarnsRegex(RuntimeWarning, "bad: model failed"):
            bad = models._fit_model_summary(frame, specs[0], {}, config)
            good = models._fit_model_summary(frame, specs[1], {}, config)
        self.assertIsNone(bad[0])
        self.assertFalse(bad[1]["fit_success"])
        self.assertFalse(bad[1]["inference_valid"])
        self.assertIn("rank deficient", bad[1]["fit_error"])
        self.assertEqual(bad[2][0]["term"], "Intercept")
        self.assertIs(good[0], good_result)


class NestedLikelihoodGuardTest(unittest.TestCase):
    def test_correct_improvement_and_tiny_roundoff(self):
        statistic, p_value = models._nested_likelihood_ratio(-10, -12, 1, context="test")
        self.assertEqual(statistic, 4)
        self.assertAlmostEqual(p_value, 0.04550026389635857)
        with self.assertWarnsRegex(RuntimeWarning, "roundoff"):
            statistic, p_value = models._nested_likelihood_ratio(-10.000000001, -10, 1, context="test")
        self.assertEqual((statistic, p_value), (0.0, 1.0))

    def test_materially_negative_improvement_is_an_error(self):
        with self.assertRaisesRegex(RuntimeError, "materially lower log likelihood"):
            models._nested_likelihood_ratio(-12, -10, 1, context="test")

    def test_nonfinite_statistics_and_invalid_degrees_fail(self):
        for values in ((np.nan, -12, 1), (-10, -12, 0)):
            with self.subTest(values=values), self.assertRaisesRegex(RuntimeError, "invalid nested"):
                models._nested_likelihood_ratio(*values, context="test")

    def test_invalid_inference_is_not_counted_as_nonsignificant_cv_contrast(self):
        repeats = pd.DataFrame([
            {"repeat": 0, "contrast": "a", "description": "a", "full_model": "full", "reduced_model": "reduced",
             "tested_predictors": "x", "paired_fit_success": True, "train_lrt_valid": False,
             "train_lrt_significant": False},
        ])
        summary = counts._summarize_cv_contrasts(repeats)
        self.assertEqual(summary.iloc[0]["n_valid_train_lrt_pairs"], 0)
        self.assertTrue(np.isnan(summary.iloc[0]["train_lrt_significant_fraction"]))


class CVRankingGuardTest(unittest.TestCase):
    def test_partial_cv_with_no_successful_contrast_pairs_still_exports_log(self):
        metrics = pd.DataFrame([
            {"repeat": 0, "model": "M0", "fit_success": True, "fit_error": ""},
            {"repeat": 0, "model": "M1", "fit_success": False, "fit_error": "invalid design"},
        ])
        paired = counts._cv_contrast_repeat_metrics(metrics, counts._contrast_specs()[:1], 0.05)
        summary = counts._summarize_cv_contrasts(paired)
        self.assertEqual(summary.iloc[0]["n_successful_pairs"], 0)
        self.assertEqual(summary.iloc[0]["n_failed_pairs"], 1)
        self.assertTrue(np.isnan(summary.iloc[0]["heldout_delta_marginal_r2_mean"]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contrasts.log"
            counts._write_cv_contrast_log(path, paired, summary)
            self.assertIn("successful: 0", path.read_text())
            self.assertIsNone(counts._plot_cv_contrasts(summary, Path(directory), 50, "Outcome"))

    def test_excludes_an_easy_fold_winner_with_missing_fits(self):
        summary = pd.DataFrame({
            "model": ["incomplete", "complete"], "n_successful_fits": [1, 5],
            "n_shuffles_requested": [5, 5], "fixed_rmse_ms_mean": [0.01, 2.0],
        })
        with self.assertWarnsRegex(RuntimeWarning, "incomplete"):
            ranked = cv._rank_models_by_marginal_rmse(summary)
        self.assertEqual(ranked["model"].tolist(), ["complete"])

    def test_all_failed_cv_retains_exports_and_renders_empty_ranking(self):
        sessions = []
        for name in ("a", "b"):
            sessions.append({
                "session": name, "preferred_cue": 1, "trial_ids": np.arange(4),
                "total_off_state_duration_ms": np.arange(4, dtype=float),
                "maximum_off_state_duration_ms": np.arange(4, dtype=float),
                "cell_counts": {f"{group}_cell_count": 1 for group in GROUP_NAMES},
                "raw_firing_rates_hz": {period: {group: np.arange(4, dtype=float)[:, None]
                    for group in GROUP_NAMES} for period in PERIODS},
            })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.pkl"
            cache_io.save({"schema_version": CV_CACHE_SCHEMA_VERSION, "sessions": sessions}, path)
            request = cv.CVModelRequest(models.ModelSpec("bad", "test", (), None))
            with patch.object(cv, "_fit_cv_model", return_value=(None, [], "invalid design")), \
                    warnings.catch_warnings(record=True) as caught, redirect_stdout(io.StringIO()), \
                    self.assertRaisesRegex(RuntimeError, "All cross-validation fits failed"):
                cv.run_trial_holdout_cv(
                    path, [request], Path(directory) / "output",
                    cv.TrialHoldoutConfig(n_shuffles=2, figure_dpi=50),
                )
            metrics = pd.read_csv(Path(directory) / "output/tables/cv_repeat_metrics.csv")
            summary = pd.read_csv(Path(directory) / "output/tables/cv_model_summary.csv")
            predictions = pd.read_pickle(Path(directory) / "output/tables/cv_prediction_sample.pkl")
            self.assertEqual(len(metrics), 2)
            self.assertFalse(metrics["fit_success"].any())
            self.assertEqual(summary.iloc[0]["n_failed_fits"], 2)
            self.assertFalse(summary.iloc[0]["rank_eligible"])
            self.assertTrue(predictions.empty)
            self.assertTrue((Path(directory) / "output/tables/cv_model_summary.csv").exists())
            self.assertTrue((Path(directory) / "output/figures/cv_model_performance.png").exists())
            self.assertTrue(any("Excluded models" in str(warning.message) for warning in caught))

    def test_all_failed_threshold_scan_has_no_spurious_winner(self):
        frame = pd.DataFrame([{ "base_model": "M", "active_percentile": 50,
            **dict.fromkeys(("aic", "bic", "marginal_r2", "conditional_r2", "conditional_rmse_ms", "active_z_threshold"), np.nan)}])
        summary = _criticality_summary(frame)
        self.assertTrue(np.isnan(summary.iloc[0]["best_aic_percentile"]))


class AllFailedStageTest(unittest.TestCase):
    def test_all_failed_stages_save_diagnostics_before_actionable_error(self):
        outcome = models.TOTAL_OUTCOME
        columns = {column for spec in models._model_specs(outcome.column) for column in spec.predictors}
        frame = pd.DataFrame({column: np.arange(6, dtype=float) for column in columns})
        frame["session"] = ["a"] * 3 + ["b"] * 3
        frame["trial_id"] = [1, 2, 3] * 2
        frame[outcome.column] = np.arange(6, dtype=float)
        for module, stage, filename in (
            (models, "models", "model_family_comparison.csv"),
            (counts, "nested-count", "model_comparison.csv"),
            (activity, "nested-activity", "model_comparison.csv"),
            (criticality, "criticality", "active_cell_model_comparison.csv"),
            (interactions, "interactions", "period_model_comparison.csv"),
        ):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "prepare").mkdir()
                input_path = root / "prepare/trial_table.pkl"
                frame.to_pickle(input_path)
                config = module.Config(cache_dir=root, run_cv=False, figure_dpi=30)
                args = (config, outcome)
                if module is criticality:
                    args += ([50], {50: frame}, {50: 0.0}, {50: input_path})
                with patch.object(models, "_fit_model", side_effect=RuntimeError("invalid fixed design")), \
                        warnings.catch_warnings(), redirect_stdout(io.StringIO()), \
                        self.assertRaisesRegex(RuntimeError, "All model fits failed"):
                    warnings.simplefilter("ignore", RuntimeWarning)
                    module._run_outcome(*args)
                table = pd.read_csv(root / stage / "outcomes" / outcome.slug / "tables" / filename)
                self.assertFalse(table["fit_success"].any())
                self.assertTrue(table["fit_error"].str.contains("invalid fixed design").all())


if __name__ == "__main__":
    unittest.main()
