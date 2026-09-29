"""Session-level replication, holdout isolation, and resampling contracts."""
from copy import deepcopy
import unittest

import numpy as np
import pandas as pd
import statsmodels.api as sm

from scripts.next.validate_m1_robustness import (
    COUNTS, NONSELECTIVE, OUTCOMES, _design, aggregate_sessions,
    align_decoder_quality, analyze_session_model, binary_training_pool_size, permutation_omnibus,
    require_shared_decoder_hashes, attach_animal_mapping, leave_one_animal_out,
)


def session_frame():
    rng = np.random.default_rng(3)
    counts = rng.integers(1, 25, size=(20, 2))
    return pd.DataFrame({"session": [str(i) for i in range(20)],
                         COUNTS[0]: counts[:, 0], COUNTS[1]: counts[:, 1],
                         "outcome": 100 - 2 * counts[:, 0] + rng.normal(size=20) * 10})


def trial_frame():
    rows = []
    for session, n_trials in (("210101", 2), ("220101", 4)):
        for trial in range(n_trials):
            rows.append({"session": session, "trial_id": trial, "preferred_cue": 1, COUNTS[0]: n_trials,
                         COUNTS[1]: 1, NONSELECTIVE: 3,
                         OUTCOMES[0]: 10 * trial, OUTCOMES[1]: 20 * trial})
    return pd.DataFrame(rows)


class M1RobustnessTests(unittest.TestCase):
    def test_aggregation_is_equal_session_and_counts_are_not_trial_replicates(self):
        actual = aggregate_sessions(trial_frame())
        self.assertEqual(len(actual), 2)
        self.assertEqual(actual.n_prepared_trials.tolist(), [2, 4])
        self.assertEqual(actual[OUTCOMES[0]].tolist(), [5, 15])
        self.assertEqual(actual.decoder_cell_count.tolist(), [6, 8])
        self.assertEqual(actual.recording_year.tolist(), ["2021", "2022"])

    def test_invalid_trial_data_rejected(self):
        for problem in ["duplicate", "changing_count", "negative", "nan", "fractional_count"]:
            with self.subTest(problem=problem):
                frame = trial_frame()
                if problem == "duplicate":
                    frame = pd.concat([frame, frame.iloc[[0]]])
                elif problem == "changing_count":
                    frame.loc[0, COUNTS[0]] += 1
                elif problem == "negative":
                    frame.loc[0, OUTCOMES[0]] = -1
                elif problem == "nan":
                    frame.loc[0, OUTCOMES[0]] = np.nan
                else:
                    frame[COUNTS[0]] = frame[COUNTS[0]].astype(float)
                    frame.loc[0, COUNTS[0]] += 0.5
                with self.assertRaises(ValueError):
                    aggregate_sessions(frame)

    def test_training_pool_counts_binary_correct_trials_and_excludes_heldout(self):
        self.assertEqual(binary_training_pool_size([1, 1, 5, 5, 2, 3], [1, 1, 1, 0, 1, 1], 1), 2)
        with self.assertRaises(ValueError):
            binary_training_pool_size([1, 1, 2], [1, 1, 1], 1)

    def test_loso_prediction_excludes_heldout_outcome_and_matches_manual_refit(self):
        frame = session_frame()
        first = analyze_session_model(frame, "outcome", n_bootstrap=0)
        changed = frame.copy()
        changed.loc[0, "outcome"] += 1000
        second = analyze_session_model(changed, "outcome", n_bootstrap=0)
        self.assertEqual(first["loso"]["rows"][0]["prediction_ms"], second["loso"]["rows"][0]["prediction_ms"])
        self.assertEqual(first["loso"]["rows"][0]["m0_prediction_ms"], second["loso"]["rows"][0]["m0_prediction_ms"])
        x = _design(frame, COUNTS)
        manual = x[0] @ np.linalg.lstsq(x[1:], frame.outcome.iloc[1:], rcond=None)[0]
        self.assertAlmostEqual(first["loso"]["rows"][0]["prediction_ms"], manual)
        rows = first["loso"]["rows"]
        sse = sum((r["observed_mean_ms"] - r["prediction_ms"])**2 for r in rows)
        base = sum((r["observed_mean_ms"] - r["m0_prediction_ms"])**2 for r in rows)
        self.assertAlmostEqual(first["loso"]["r2_vs_heldout_training_mean"], 1 - sse / base)

    def test_quality_covariates_align_by_id_and_reject_missing_or_wrong_methods(self):
        frame = session_frame()
        frame["preferred_cue"] = 1
        frame["decoder_cell_count"] = 10
        frame["recorded_cell_count"] = 20
        evidence = {"design": {"methods": {"weighted_search_calibrated": {"weighted": True, "search": True, "calibrate": True}},
                               "sessions": [{"session": session, "cached_cue": 1, "cached_stationary_cells": 10, "all_cells": 20} for session in reversed(frame.session)]},
                    "summary": {"cached_population": {"n_sessions": len(frame), "per_session": {
                        session: {"weighted_search_calibrated": {"brier": .2 + i * .001, "balanced_accuracy": .6, "auc": .7}}
                        for i, session in enumerate(frame.session)}}}}
        aligned = align_decoder_quality(frame, evidence)
        np.testing.assert_allclose(aligned.decoder_cv_brier, .2 + np.arange(len(frame)) * .001)
        self.assertNotIn("decoder_cv_brier", frame)
        for problem in ("missing_session", "extra_session", "wrong_method", "wrong_population", "wrong_cue", "duplicate_design", "nan"):
            with self.subTest(problem=problem):
                invalid = deepcopy(evidence)
                scores = invalid["summary"]["cached_population"]["per_session"]
                if problem == "missing_session":
                    scores.pop("0")
                elif problem == "extra_session":
                    scores["extra"] = scores["0"]
                elif problem == "wrong_method":
                    invalid["design"]["methods"]["weighted_search_calibrated"]["calibrate"] = False
                elif problem == "wrong_cue":
                    invalid["design"]["sessions"][0]["cached_cue"] = 5
                elif problem == "wrong_population":
                    invalid["design"]["sessions"][0]["all_cells"] = 99
                elif problem == "duplicate_design":
                    invalid["design"]["sessions"].append(invalid["design"]["sessions"][0])
                else:
                    scores["0"]["weighted_search_calibrated"]["brier"] = np.nan
                with self.assertRaises(ValueError):
                    align_decoder_quality(frame, invalid)

    def test_quality_source_hashes_reject_missing_changed_and_aliased_conflicts(self):
        from scripts.next.validate_m1_robustness import ROOT
        path = "data/nature/session.mat"
        evidence = {"design": {"sources_sha256": {path: "abc"}}}
        require_shared_decoder_hashes(evidence, {str(ROOT / path): "abc"})
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            require_shared_decoder_hashes(evidence, {path: "changed"})
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            require_shared_decoder_hashes(evidence, {"missing.mat": "abc"})
        evidence["design"]["sources_sha256"][str(ROOT / path)] = "changed"
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            require_shared_decoder_hashes(evidence, {path: "abc"})

    def test_confirmed_animal_mapping_requires_exact_explicit_coverage(self):
        frame = session_frame()
        assignments = {s: "A" if i < 7 else "H" if i < 14 else "J" for i, s in enumerate(frame.session)}
        mapping = {"sessions": assignments, "expected_session_counts": {"A": 7, "H": 7, "J": 6}}
        actual = attach_animal_mapping(frame, mapping)
        self.assertEqual(actual.animal.tolist(), ["A"] * 7 + ["H"] * 7 + ["J"] * 6)
        missing = deepcopy(mapping)
        missing["sessions"].pop("0")
        with self.assertRaises(ValueError):
            attach_animal_mapping(frame, missing)
        wrong_counts = deepcopy(mapping)
        wrong_counts["expected_session_counts"]["A"] = 8
        with self.assertRaises(ValueError):
            attach_animal_mapping(frame, wrong_counts)

    def test_animal_holdout_excludes_all_heldout_outcomes_and_scores_both_weightings(self):
        frame = session_frame()
        frame["animal"] = ["A"] * 7 + ["H"] * 7 + ["J"] * 6
        frame.loc[frame.animal == "A", COUNTS[0]] += 100
        actual = leave_one_animal_out(frame, "outcome")
        changed = frame.copy()
        changed.loc[changed.animal == "A", "outcome"] += 1000
        modified = leave_one_animal_out(changed, "outcome")
        first = actual["folds"][0]
        self.assertEqual(first["training_animals"], ["H", "J"])
        self.assertFalse(set(first["heldout_sessions"]) & set(first["training_sessions"]))
        self.assertEqual(first["coefficients"], modified["folds"][0]["coefficients"])
        self.assertEqual(first["n_sessions_with_any_count_outside_training_range"], 7)
        for before, after in zip(actual["session_predictions"][:7], modified["session_predictions"][:7]):
            self.assertEqual(before["prediction_ms"], after["prediction_ms"])
            self.assertEqual(before["baseline_prediction_ms"], after["baseline_prediction_ms"])
        folds = actual["folds"]
        sse, baseline_sse = sum(f["sse_ms2"] for f in folds), sum(f["baseline_sse_ms2"] for f in folds)
        self.assertAlmostEqual(actual["session_equal"]["r2_vs_training_session_mean"], 1 - sse / baseline_sse)
        mse = np.mean([f["sse_ms2"] / f["n_heldout_sessions"] for f in folds])
        base_mse = np.mean([f["baseline_sse_ms2"] / f["n_heldout_sessions"] for f in folds])
        self.assertAlmostEqual(actual["animal_equal"]["r2_vs_training_session_mean"], 1 - mse / base_mse)

    def test_adjusted_loso_has_a_matching_reduced_covariate_baseline(self):
        frame = session_frame()
        frame["covariate"] = np.arange(len(frame))
        result = analyze_session_model(frame, "outcome", (*COUNTS, "covariate"), n_bootstrap=0, reduced_predictors=("covariate",))
        reduced = _design(frame, ("covariate",))
        manual = reduced[0] @ np.linalg.lstsq(reduced[1:], frame.outcome.iloc[1:], rcond=None)[0]
        self.assertAlmostEqual(result["loso"]["rows"][0]["reduced_model_prediction_ms"], manual)
        rows = result["loso"]["rows"]
        full_sse = sum((r["observed_mean_ms"] - r["prediction_ms"])**2 for r in rows)
        reduced_sse = sum((r["observed_mean_ms"] - r["reduced_model_prediction_ms"])**2 for r in rows)
        self.assertAlmostEqual(result["loso"]["r2_vs_reduced_model"], 1 - full_sse / reduced_sse)
        with self.assertRaises(ValueError):
            analyze_session_model(frame, "outcome", n_bootstrap=0, reduced_predictors=("covariate",))

    def test_hc3_finite_session_df_and_bootstrap_reproducibility(self):
        frame = session_frame()
        actual = analyze_session_model(frame, "outcome", n_bootstrap=80)
        repeat = analyze_session_model(frame, "outcome", n_bootstrap=80)
        reference = sm.OLS(frame.outcome, _design(frame, COUNTS)).fit(cov_type="HC3", use_t=True)
        self.assertEqual(actual, repeat)
        self.assertEqual(actual["residual_df"], 17)
        term = actual["terms"][COUNTS[0]]
        np.testing.assert_allclose(term["hc3_t_ci95"], reference.conf_int().iloc[1])
        self.assertAlmostEqual(term["hc3_t_p_value"], reference.pvalues.iloc[1])
        self.assertEqual(actual["bootstrap"]["valid"] + actual["bootstrap"]["rank_deficient"], 80)

    def test_invalid_session_regression_rejected(self):
        for problem in ["duplicate_session", "rank_deficient", "nonfinite", "negative_bootstrap"]:
            with self.subTest(problem=problem):
                frame = session_frame()
                if problem == "duplicate_session":
                    frame.loc[0, "session"] = frame.loc[1, "session"]
                elif problem == "rank_deficient":
                    frame[COUNTS[1]] = frame[COUNTS[0]]
                elif problem == "nonfinite":
                    frame.loc[0, "outcome"] = np.nan
                with self.assertRaises(ValueError):
                    analyze_session_model(frame, "outcome", n_bootstrap=-1 if problem == "negative_bootstrap" else 0)

    def test_permutation_plus_one_reproducibility_and_strong_association(self):
        frame = session_frame()
        x = _design(frame, COUNTS)
        y = x[:, 1] * 5 + np.arange(len(x)) * 0.001
        first = permutation_omnibus(x, y, (1, 2), n_permutations=99, seed=2)
        second = permutation_omnibus(x, y, (1, 2), n_permutations=99, seed=2)
        self.assertEqual(first, second)
        self.assertEqual(first["p_value_plus_one"], (first["exceedances"] + 1) / 100)
        self.assertEqual(first["p_value_plus_one"], 0.01)

    def test_blocked_permutation_requires_block_controls(self):
        frame = session_frame()
        blocks = np.repeat([0, 1], 10)
        with self.assertRaisesRegex(ValueError, "block fixed effects"):
            permutation_omnibus(_design(frame, COUNTS), frame.outcome.to_numpy(), (1, 2), n_permutations=9, seed=2, blocks=blocks)
        frame["block"] = blocks
        actual = permutation_omnibus(_design(frame, (*COUNTS, "block")), frame.outcome.to_numpy(), (1, 2), n_permutations=9, seed=2, blocks=blocks)
        self.assertEqual(actual["exchangeability"], "within recording year")


if __name__ == "__main__":
    unittest.main()
