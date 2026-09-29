"""Probability algebra, outer-holdout isolation, and aggregation contracts."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts.next import validate_regularization_path as validation
from scripts.next.decoder_models import DecoderModel, SVMKernel, create_base_decoder


class RegularizationPathValidationTests(unittest.TestCase):
    def test_brier_decomposition_and_skill_use_equal_class_prior(self):
        labels = np.array([0, 0, 0, 0, 0, 1, 1])
        probability = np.array([.1, .2, .4, .6, .8, .7, .9])
        weights = np.array([.1] * 5 + [.25] * 2)
        actual = validation.score(labels, probability)
        brier = np.sum(weights * (probability - labels)**2)
        spread = np.sum(weights * (probability - .5)**2)
        alignment = 2 * np.sum(weights * (probability - .5) * (labels - .5))
        self.assertAlmostEqual(actual['brier'], brier)
        self.assertAlmostEqual(actual['squared_probability_spread'], spread)
        self.assertAlmostEqual(actual['label_alignment_term'], alignment)
        self.assertAlmostEqual(brier, .25 + spread - alignment)
        self.assertAlmostEqual(actual['brier_skill_vs_half'], 1 - 4 * brier)
        self.assertAlmostEqual(actual['information_gain_nats_vs_half'], np.log(2) - actual['log_loss'])
        # Replicating one cue does not change an equal-prior estimand.
        repeated = validation.score(np.r_[labels, labels[:5]], np.r_[probability, probability[:5]])
        for name in validation.SCORE_KEYS:
            self.assertAlmostEqual(actual[name], repeated[name], places=14)

    def test_constant_perfect_and_reversed_predictions_have_known_brier_terms(self):
        labels = np.array([0, 0, 0, 1])
        for p, brier, spread, alignment, skill in [
            (np.full(4, .5), .25, 0., 0., 0.),
            (labels, 0., .25, .5, 1.),
            (1 - labels, 1., .25, -.5, -3.),
        ]:
            with self.subTest(probability=p.tolist()):
                actual = validation.score(labels, p)
                self.assertAlmostEqual(actual['brier'], brier)
                self.assertAlmostEqual(actual['squared_probability_spread'], spread)
                self.assertAlmostEqual(actual['label_alignment_term'], alignment)
                self.assertAlmostEqual(actual['brier_skill_vs_half'], skill)
        constant = validation.score(labels, np.full(4, .5))
        self.assertEqual(constant['auc'], .5)
        self.assertEqual(constant['balanced_accuracy'], .5)
        self.assertAlmostEqual(constant['information_gain_nats_vs_half'], 0.)
        self.assertEqual(constant['exact_half_fraction'], 1.)

    def test_shrinkage_endpoints_are_exact_and_interior_preserves_auc_and_decisions(self):
        endpoint_values = np.array([0., 1e-30, .1, .3, .5, .7, .9, 1.])
        np.testing.assert_array_equal(validation.shrink(endpoint_values, 0), np.full(8, .5))
        np.testing.assert_array_equal(validation.shrink(endpoint_values, 1), endpoint_values)
        labels = np.array([0, 0, 0, 0, 1, 1, 1])
        probability = np.array([.07, .2, .49, .5, .51, .78, .94])
        baseline = validation.score(labels, probability)
        for alpha in (.1, .25, .5, .75, 1.):
            shrunk = validation.shrink(probability, alpha)
            np.testing.assert_array_equal(shrunk >= .5, probability >= .5)
            actual = validation.score(labels, shrunk)
            self.assertEqual(actual['auc'], baseline['auc'])
            self.assertEqual(actual['balanced_accuracy'], baseline['balanced_accuracy'])
            self.assertAlmostEqual(actual['squared_probability_spread'], alpha**2 * baseline['squared_probability_spread'])
            self.assertAlmostEqual(actual['label_alignment_term'], alpha * baseline['label_alignment_term'])

    def test_base_margin_auc_is_distinct_and_missing_when_no_margins_available(self):
        labels = np.array([0, 0, 1, 1])
        probabilities = np.array([.1, .2, .8, .9])
        self.assertIsNone(validation.score(labels, probabilities)['base_margin_auc'])
        diagnostic = validation.score(labels, probabilities, margins=np.array([2., 1., -1., -2.]))
        self.assertEqual(diagnostic['auc'], 1.)
        self.assertEqual(diagnostic['base_margin_auc'], 0.)

    def test_shrinkage_rejects_invalid_probability_or_alpha(self):
        for probabilities, alpha in [([.2, np.nan], .5), ([-.1, .4], .5), ([.2, 1.1], .5),
                                     ([.2, .4], np.nan), ([.2, .4], -.1), ([.2, .4], 1.1)]:
            with self.subTest(probabilities=probabilities, alpha=alpha), self.assertRaises(ValueError):
                validation.shrink(probabilities, alpha)

    def test_all_c_values_reuse_outer_folds_and_each_trial_is_scored_once_per_seed(self):
        labels = np.tile([0, 1], 20)
        task = {'session': 's', 'scope': 'test', 'X': np.column_stack([np.arange(40), np.ones(40)]), 'labels': labels}
        calls = []
        cs, seeds = (1., .01, .0001), (11, 13)

        def fake_fit(X, y, training, test, *, c, seed):
            self.assertFalse(set(training) & set(test))
            calls.append((c, tuple(training), tuple(test), seed))
            probability = .2 + .6 * X[test, 0] / 39
            return {'raw': probability, 'calibrated': .5 + .8 * (probability - .5),
                    'margin': X[test, 0], 'diagnostics': {'training_class_counts': np.bincount(y[training], minlength=2).tolist()}}

        with patch.object(validation, 'fit_fold', fake_fit):
            actual = validation.evaluate_task(task, np.full((2, 40), .6), cs=cs, seeds=seeds)
        partitions = [[entry[1:] for entry in calls if entry[0] == c] for c in cs]
        for partition in partitions:
            self.assertEqual(partition, partitions[0])
            for seed_index in range(len(seeds)):
                selected = partition[seed_index * 5:(seed_index + 1) * 5]
                self.assertEqual(sorted(i for _, test, _ in selected for i in test), list(range(40)))
        expected = .2 + .6 * np.arange(40) / 39
        for c in cs:
            np.testing.assert_array_equal(actual['methods'][validation.method_name(c, False)]['predictions'], np.tile(expected, (2, 1)))
        with self.assertRaises(ValueError):
            validation.evaluate_task(task, np.full((1, 40), .6), cs=cs, seeds=seeds)

    def test_heldout_activity_and_labels_never_enter_fitting_or_calibration(self):
        rng = np.random.default_rng(7)
        X, labels = rng.normal(size=(36, 3)), np.tile([0, 1], 18)
        training, test = np.arange(30), np.arange(30, 36)
        calls = []

        class FakeEstimator:
            classes_ = np.array([0, 1])
            named_steps = {'classifier': SimpleNamespace(coef_=np.array([[1., 0., 0.]]))}

            def decision_function(self, x):
                return x[:, 0]

            def predict_proba(self, x):
                return np.tile([.4, .6], (len(x), 1))

        class FakeCalibrated:
            classes_ = np.array([0, 1])
            calibrated_classifiers_ = [SimpleNamespace(estimator=FakeEstimator(), calibrators=[SimpleNamespace(a_=-1., b_=0.)])]

            def predict_proba(self, x):
                return np.tile([.3, .7], (len(x), 1))

        def calibrate(base, x, y, method, splits, *, balanced_class_weights):
            self.assertEqual(method, 'sigmoid')
            self.assertTrue(balanced_class_weights)
            calls.append((x.copy(), y.copy()))
            for fit, heldout in splits:
                self.assertFalse(set(fit) & set(heldout))
                self.assertLess(max(np.r_[fit, heldout]), len(training))
            return FakeCalibrated()

        with patch.object(validation, 'fit_calibrated_decoder', calibrate):
            validation.fit_fold(X, labels, training, test, c=.01, seed=9)
            X[test] = 1e9
            labels[test] = 1 - labels[test]
            validation.fit_fold(X, labels, training, test, c=.01, seed=9)
        self.assertEqual(len(calls), 2)
        for before, after in zip(calls[0], calls[1]):
            np.testing.assert_array_equal(before, after)
        np.testing.assert_array_equal(calls[0][0], X[training])
        np.testing.assert_array_equal(calls[0][1], labels[training])

    def test_final_raw_base_matches_a_separate_weighted_fit(self):
        rng = np.random.default_rng(41)
        X = rng.normal(size=(72, 4))
        labels = np.r_[np.zeros(42, dtype=int), np.ones(30, dtype=int)]
        rng.shuffle(labels)
        X[:, 0] += labels * .8
        training, test = np.arange(60), np.arange(60, 72)
        c, seed = .01, 19
        actual = validation.fit_fold(X, labels, training, test, c=c, seed=seed)
        independent = create_base_decoder(c, DecoderModel.LOGISTIC_REGRESSION, SVMKernel.LINEAR, seed, class_weight='balanced')
        independent.fit(X[training], labels[training])
        positive = np.flatnonzero(independent.classes_ == 1)[0]
        np.testing.assert_array_equal(actual['raw'], independent.predict_proba(X[test])[:, positive])
        np.testing.assert_array_equal(actual['margin'], independent.decision_function(X[test]))
        self.assertEqual(actual['diagnostics']['training_class_counts'], np.bincount(labels[training], minlength=2).tolist())

    def test_reference_alignment_rejects_task_metadata_seeds_and_labels_mismatches(self):
        metadata = {'session': 's', 'scope': 'cached_population', 'cue': 1, 'opposite_cue': 5,
                    'bin_start_ms': 500, 'trial_ids': [2, 4, 6, 8], 'class_counts': [2, 2], 'n_cells': 3}
        task = {**metadata, 'labels': np.array([0, 1, 0, 1]), 'X': np.zeros((4, 3))}
        old = {'design': {'outer_seeds': list(validation.OUTER_SEEDS)}, 'tasks': [metadata]}
        archive = {'task_0_labels': task['labels'].copy()}
        validation.validate_alignment([task], old, archive)
        for key in metadata:
            changed = deepcopy(old)
            value = changed['tasks'][0][key]
            changed['tasks'][0][key] = value[::-1] if isinstance(value, list) else str(value) + '_changed'
            # Reversing an equal class-count vector is not a change.
            if key == 'class_counts':
                changed['tasks'][0][key] = [1, 3]
            with self.subTest(key=key), self.assertRaises(ValueError):
                validation.validate_alignment([task], changed, archive)
        changed = deepcopy(old)
        changed['design']['outer_seeds'][0] += 1
        with self.assertRaises(ValueError):
            validation.validate_alignment([task], changed, archive)
        with self.assertRaises(ValueError):
            validation.validate_alignment([task], old, {'task_0_labels': 1 - task['labels']})
        with self.assertRaises(ValueError):
            validation.validate_alignment([task, task], old, archive)

    def test_aggregation_weights_sessions_equally_and_retains_animal_groups(self):
        methods = (validation.REFERENCE, validation.method_name(.01), 'shrink_alpha_0')
        rows = []
        for session, value, n_tasks in [('a', .1, 1), ('b', .3, 3), ('c', .2, 2), ('d', .4, 1)]:
            for _ in range(n_tasks):
                row_methods = {}
                for method in methods:
                    base = value if method == validation.method_name(.01) else .25
                    row_methods[method] = {'seed_metrics': [{k: base + delta for k in validation.SCORE_KEYS} for delta in (-.01, 0., .01)]}
                rows.append({'scope': 'test', 'session': session, 'methods': row_methods})
        mapping = {'a': 'A', 'b': 'A', 'c': 'H', 'd': 'J'}
        actual = validation.aggregate(rows, mapping, bootstrap=100)['test']
        method = validation.method_name(.01)
        self.assertEqual(actual['n_sessions'], 4)
        self.assertEqual(actual['n_tasks'], 7)
        self.assertAlmostEqual(actual['aggregate_equal_session'][method]['brier'], .25)
        self.assertAlmostEqual(actual['per_animal_equal_session']['A'][method]['brier'], .2)
        self.assertAlmostEqual(actual['per_animal_equal_session']['H'][method]['brier'], .2)
        self.assertAlmostEqual(actual['per_animal_equal_session']['J'][method]['brier'], .4)
        without_margins = deepcopy(rows)
        for row in without_margins:
            for missing_method in (validation.REFERENCE, 'shrink_alpha_0'):
                for metrics in row['methods'][missing_method]['seed_metrics']:
                    metrics['base_margin_auc'] = None
        missing = validation.aggregate(without_margins, mapping, bootstrap=10)['test']
        self.assertIsNone(missing['aggregate_equal_session'][validation.REFERENCE]['base_margin_auc'])
        self.assertIsNone(missing['per_animal_equal_session']['A']['shrink_alpha_0']['base_margin_auc'])
        without_margins[0]['methods'][validation.REFERENCE]['seed_metrics'][0]['base_margin_auc'] = .6
        with self.assertRaisesRegex(ValueError, 'Partially missing'):
            validation.aggregate(without_margins, mapping, bootstrap=10)
        comparison = actual['paired_comparisons'][validation.REFERENCE][method]['brier']
        self.assertAlmostEqual(comparison['method_minus_reference'], 0.)
        self.assertEqual(comparison['sessions_favoring_method'], 2)
        with self.assertRaises(ValueError):
            validation.aggregate(rows, {'a': 'A', 'b': 'A', 'c': 'H'}, bootstrap=10)
        with self.assertRaises(ValueError):
            validation.aggregate(rows, {**mapping, 'extra': 'J'}, bootstrap=10)


if __name__ == '__main__':
    unittest.main()
