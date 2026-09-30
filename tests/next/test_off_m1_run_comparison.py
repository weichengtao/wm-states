"""Downstream run matching and M0-relative CV denominators."""
from copy import deepcopy
import unittest

import numpy as np
import pandas as pd

from scripts.next.compare_off_m1_runs import (
    METRICS, align_cv_features, align_trials, state_trials, summarize_cv,
)
from scripts.next.validate_m1_robustness import OUTCOMES


def cv_rows():
    rows = []
    for repeat, n in ((0, 2), (1, 8)):
        for model in ("M0", "M1"):
            rmse = {(0, "M0"): 2, (0, "M1"): 1, (1, "M0"): 4, (1, "M1"): 3}[repeat, model]
            row = {"repeat": repeat, "model": model, "n_test": n,
                   "fit_success": True, "converged": True, "inference_valid": True}
            row.update({metric: rmse if metric.endswith("rmse_ms") else 0.1 for metric in METRICS})
            rows.append(row)
    return pd.DataFrame(rows)


class OffM1ComparisonTests(unittest.TestCase):
    def test_trial_alignment_allows_changed_outcomes_but_rejects_identity_or_predictor_changes(self):
        first = pd.DataFrame({"session": ["a", "a", "b"], "trial_id": [1, 2, 1],
                              "count": [3, 3, 4], OUTCOMES[0]: [1, 2, 3], OUTCOMES[1]: [4, 5, 6]})
        second = first.iloc[::-1].copy()
        second[OUTCOMES[0]] += 10
        a, b = align_trials(first, second)
        np.testing.assert_array_equal(b[OUTCOMES[0]] - a[OUTCOMES[0]], [10, 10, 10])
        for column in ("trial_id", "count"):
            bad = second.copy()
            bad.iloc[0, bad.columns.get_loc(column)] = 90
            with self.assertRaises(ValueError):
                align_trials(first, bad)
        with self.assertRaises(ValueError):
            align_trials(first, pd.concat([second, second.iloc[[0]]]))

    def test_cached_folds_and_raw_features_match_exactly_while_outcomes_may_differ(self):
        first = {"sessions": [{"trial_ids": np.array([1, 2]), "raw": np.array([[1., np.nan], [2., 3.]]), OUTCOMES[0]: [10., 20.]}],
                 "splits": [{"repeat": 0, "test_trial_ids_by_session": {"a": np.array([2])}}]}
        second = deepcopy(first)
        second["sessions"][0][OUTCOMES[0]] = [30., 40.]
        self.assertEqual(align_cv_features(first, second), {0: 1})
        for field in ("raw", "splits"):
            bad = deepcopy(second)
            if field == "raw":
                bad["sessions"][0]["raw"][0, 0] = 10
            else:
                bad["splits"][0]["test_trial_ids_by_session"]["a"][0] = 1
            with self.assertRaises(ValueError):
                align_cv_features(first, bad)

    def test_relative_gain_uses_pooled_sse_and_each_models_actual_test_count(self):
        result = summarize_cv(cv_rows(), {0: 2, 1: 8})
        gain = result["m1_vs_m0"]["fixed"]
        self.assertEqual(gain["m0_sse_ms2"], 2 * 2**2 + 8 * 4**2)
        self.assertEqual(gain["m1_sse_ms2"], 2 * 1**2 + 8 * 3**2)
        self.assertAlmostEqual(gain["relative_sse_reduction"], 1 - 74 / 136)
        self.assertNotAlmostEqual(gain["relative_sse_reduction"], np.mean([1 - 1/4, 1 - 9/16]))
        self.assertEqual(gain["n_scored_trial_appearances"], 10)
        self.assertAlmostEqual(gain["pooled_test_rmse_m0_ms"], np.sqrt(13.6))

    def test_invalid_or_rank_ineligible_fit_excludes_the_whole_model_pair(self):
        rows = cv_rows()
        rows.loc[(rows.repeat == 1) & (rows.model == "M1"), "inference_valid"] = False
        result = summarize_cv(rows, {0: 2, 1: 8})
        self.assertEqual(result["usable_repeats"], [0])
        self.assertEqual(result["flags"]["M0"]["usable"], 2)
        self.assertEqual(result["flags"]["M1"]["usable"], 1)
        self.assertEqual(result["m1_vs_m0"]["fixed"]["relative_sse_reduction"], .75)

    def test_cv_identity_and_boolean_errors_cannot_silently_pass(self):
        for problem in ("count", "missing", "duplicate", "string_flag", "nonfinite", "zero_baseline"):
            rows = cv_rows()
            if problem == "count":
                rows.loc[0, "n_test"] = 99
            elif problem == "missing":
                rows = rows.iloc[1:]
            elif problem == "duplicate":
                rows = pd.concat([rows, rows.iloc[[0]]])
            elif problem == "string_flag":
                rows["inference_valid"] = rows.inference_valid.astype(object)
                rows.loc[0, "inference_valid"] = "False"
            elif problem == "zero_baseline":
                rows.loc[rows.model == "M0", "fixed_rmse_ms"] = 0
            else:
                rows.loc[0, "fixed_rmse_ms"] = np.nan
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                summarize_cv(rows, {0: 2, 1: 8})

    def test_state_outcomes_check_physical_ordering_and_duplicate_ids(self):
        cache = {"schema": "wm-states-next", "version": 1, "results": [{
            "session": "a", "cue": 1, "trial_idx": np.array([2, 1]),
            "off_state_duration_delay_start": 500, "off_state_duration_delay_end": 1400,
            "max_off_state_duration_per_trial": np.array([20., 30.]),
            "off_state_duration_per_trial": np.array([50., 40.]),
        }]}
        self.assertEqual(state_trials(cache).trial_id.tolist(), [1, 2])
        bad = deepcopy(cache)
        bad["results"][0]["max_off_state_duration_per_trial"][0] = 100
        with self.assertRaises(ValueError):
            state_trials(bad)
        bad = deepcopy(cache)
        bad["results"][0]["trial_idx"][0] = 1
        with self.assertRaises(ValueError):
            state_trials(bad)


if __name__ == "__main__":
    unittest.main()
