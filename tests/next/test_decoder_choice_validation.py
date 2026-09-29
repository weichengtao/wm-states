"""Statistical and holdout contracts of the supplementary validation."""
import unittest
from unittest.mock import patch

import numpy as np

from scripts.next import validate_decoder_choices as validation


class DecoderChoiceValidationTest(unittest.TestCase):

    def test_equal_prior_metrics_are_invariant_to_class_replication(self):
        labels = np.array([0, 0, 1, 1])
        p = np.array([.1, .3, .7, .9])
        first = validation.probability_metrics(labels, p)
        second = validation.probability_metrics(np.r_[labels, labels[:2], labels[:2]], np.r_[p, p[:2], p[:2]])
        for key in validation.METRICS:
            np.testing.assert_allclose(first[key], second[key], atol=1e-15)
        weights = validation.equal_prior_weights(np.array([0, 0, 0, 1]))
        self.assertAlmostEqual(weights[:3].sum(), .5)
        assert weights[3] == .5

    def test_constant_baseline_has_known_proper_scores(self):
        result = validation.probability_metrics(np.array([0, 0, 0, 1]), np.full(4, .5))
        assert result['brier'] == .25
        np.testing.assert_allclose(result['log_loss'], np.log(2))
        assert result['balanced_accuracy'] == .5
        assert result['auc'] == .5
        assert result['mean_probability_minus_prevalence'] == 0
        assert result['ece_10_equal_width'] == 0

    def test_invalid_metrics_fail(self):
        for labels, probabilities in [([0, 0], [.5, .5]), ([0, 1], [.5, np.nan]), ([0, 1], [.5, 1.1]), ([0, 2], [.5, .5])]:
            with self.assertRaises(ValueError):
                validation.probability_metrics(labels, probabilities)

    def test_downsampling_uses_only_outer_training_indices(self):
        labels = np.r_[np.zeros(15, dtype=int), np.ones(10, dtype=int)]
        training = np.r_[np.arange(12), np.arange(15, 22)]
        chosen = validation.choose_training_indices(labels, training, weighted=False, seed=123)
        assert set(chosen) <= set(training)
        assert len(np.unique(chosen)) == len(chosen) == 14
        np.testing.assert_array_equal(np.bincount(labels[chosen]), [7, 7])
        np.testing.assert_array_equal(chosen, validation.choose_training_indices(labels, training, weighted=False, seed=123))
        np.testing.assert_array_equal(training, validation.choose_training_indices(labels, training, weighted=True, seed=123))

    def test_heldout_activity_and_labels_do_not_enter_training(self):
        rng = np.random.default_rng(8)
        X = rng.normal(size=(30, 3))
        y = np.tile([0, 1], 15)
        training, test = np.arange(24), np.arange(24, 30)
        calls = []

        class FakeModel:
            classes_ = np.array([0, 1])

            def fit(self, x, labels):
                calls.append(('fit', x.copy(), labels.copy()))
                return self

            def predict_proba(self, x):
                return np.tile([.4, .6], (len(x), 1))

        def choose(x, labels, groups, *args, **kwargs):
            calls.append(('search', x.copy(), labels.copy()))
            return .1

        def calibrate(base, x, labels, method, splits, **kwargs):
            calls.append(('calibrate', x.copy(), labels.copy()))
            for fit, heldout in splits:
                assert set(fit).isdisjoint(heldout)
                assert max(np.r_[fit, heldout]) < len(training)
            return base
        with patch.object(validation, 'select_classifier_c', choose), patch.object(validation, 'create_base_decoder', lambda *a, **k: FakeModel()), patch.object(validation, 'fit_calibrated_decoder', calibrate):
            validation.fit_predict(X, y, training, test, method='weighted_search_calibrated', seed=2)
            initial = [(name, a.copy(), b.copy()) for name, a, b in calls]
            calls.clear()
            X[test] = 1e8
            y[test] = 1 - y[test]
            validation.fit_predict(X, y, training, test, method='weighted_search_calibrated', seed=2)
            assert [c[0] for c in calls] == [c[0] for c in initial]
            for (_, a, b), (_, c, d) in zip(calls, initial):
                np.testing.assert_array_equal(a, c)
                np.testing.assert_array_equal(b, d)

    def test_session_bootstrap_does_not_treat_trials_as_replicates(self):
        rows = []
        for session, score, tasks in [('a', .1, 1), ('b', .3, 3)]:
            for _ in range(tasks):
                methods = {}
                for method in (*validation.METHODS, 'constant_half'):
                    metrics = {name: score if method == 'weighted_search_calibrated' else .25 for name in validation.METRICS}
                    methods[method] = {'seed_metrics': [metrics] * 3, 'probability_sd_across_outer_seeds': 0}
                rows.append({'scope': 'test', 'session': session, 'methods': methods})
        result = validation.aggregate_results(rows, bootstrap=100)['test']
        np.testing.assert_allclose(result['aggregate_equal_session']['weighted_search_calibrated']['brier'], .2)
        comparison = result['paired_comparisons']['constant_half']['brier']
        assert comparison['n_sessions'] == 2
        np.testing.assert_allclose(comparison['default_minus_comparator'], -.05)
        assert comparison['sessions_favoring_default'] == 1

    def test_common_outer_folds_and_heldout_ensemble(self):
        labels = np.tile([0, 1], 15)
        task = {'X': np.arange(60).reshape(30, 2), 'labels': labels,
                'scope': 'test', 'session': 's'}
        calls = []

        def fake_fit(X, y, training, test, *, method, seed):
            self.assertFalse(set(training) & set(test))
            calls.append((method, tuple(test), seed))
            p = np.full(len(test), .25 + .5 * ((seed % 101) / 100))
            return p, 1.0, training

        with patch.object(validation, 'fit_predict', fake_fit):
            result = validation.evaluate_task(task, seeds=(11, 13, 17))
        fold_ids = [[entry[1:] for entry in calls if entry[0] == method] for method in validation.METHODS]
        for folds in fold_ids:
            self.assertEqual(folds, fold_ids[0])
            for seed_block in range(3):
                self.assertEqual(sorted(i for test, _ in folds[seed_block * 5:(seed_block + 1) * 5] for i in test), list(range(30)))
        for method in validation.METHODS:
            value = result['methods'][method]
            expected = validation.probability_metrics(labels, value['predictions'].mean(axis=0))
            for metric in ('brier', 'log_loss'):
                self.assertAlmostEqual(value['ensemble_metrics'][metric], expected[metric])
                self.assertLessEqual(expected[metric], np.mean([v[metric] for v in value['seed_metrics']]) + 1e-15)


if __name__ == "__main__":
    unittest.main()
