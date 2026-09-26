import dataclasses
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from joblib import Parallel, delayed

from scripts.next import decoding_confidence as decoding
from scripts.next import cache_io
from scripts.next.common import worker_context
from scripts.next.eval_confidence import evaluate_session
from scripts.next.eval_confidence_across_runs import load_runs


class NullTimeStructureTest(unittest.TestCase):
    def setUp(self):
        self.labels = np.tile([0, 1], 10)
        self.rates = np.repeat(np.random.default_rng(123).normal(size=(20, 1, 3)), 3, axis=1)
        self.times = np.array([0, 10, 20])
        self.config = decoding.Config(n_decode_shuffle=3, logistic_calibration_method='none')

    def decode(self, config, test=1):
        return decoding.decode_one_trial(test, self.rates, self.labels, self.times, config)

    def test_default_keeps_independent_bins_and_option_preserves_identical_bins(self):
        independent = self.decode(self.config)
        preserved = self.decode(dataclasses.replace(self.config, preserve_null_time_structure=True))
        self.assertFalse(self.config.preserve_null_time_structure)
        self.assertGreater(np.ptp(independent[3][:, 0]), 0)
        np.testing.assert_array_equal(preserved[3], np.repeat(preserved[3][:1], 3, axis=0))
        # The new option leaves every observed estimate and the old first-bin
        # null RNG stream unchanged.
        for index in (0, 1, 2):
            np.testing.assert_array_equal(independent[index], preserved[index])
        np.testing.assert_array_equal(independent[3][0], preserved[3][0])

    def test_shared_labels_work_with_c_search_and_calibration(self):
        config = dataclasses.replace(self.config, preserve_null_time_structure=True,
                                     n_decode_shuffle=1, grid_search_for_c=True,
                                     logistic_calibration_method='sigmoid')
        result = self.decode(config)
        np.testing.assert_array_equal(result[3], np.repeat(result[3][:1], 3, axis=0))
        np.testing.assert_array_equal(result[4], np.repeat(result[4][:1], 3, axis=0))
        self.assertEqual(result[-1], (5,))

    def test_prefix_zero_nulls_and_parallel_reproducibility(self):
        config = dataclasses.replace(self.config, preserve_null_time_structure=True)
        serial = [self.decode(config, trial) for trial in (1, 3)]
        longer = self.decode(dataclasses.replace(config, n_decode_shuffle=5))
        empty = self.decode(dataclasses.replace(config, n_decode_shuffle=0))
        np.testing.assert_array_equal(serial[0][3], longer[3][:, :3])
        np.testing.assert_array_equal(serial[0][0], empty[0])
        self.assertEqual(empty[3].shape, (3, 0))
        with worker_context(2):
            parallel = Parallel()(delayed(decoding.decode_one_trial)(
                trial, self.rates, self.labels, self.times, config) for trial in (1, 3))
        for left, right in zip(serial, parallel):
            for a, b in zip(left[:5], right[:5]):
                np.testing.assert_array_equal(a, b)

    def test_option_keeps_holdout_and_balancing_before_permutation(self):
        fits = []
        class Recorder:
            classes_ = np.array([0, 1])
            def fit(self, x, y):
                fits.append((x.copy(), y.copy()))
                return self
            def predict_proba(self, x):
                return np.tile([.4, .6], (len(x), 1))
            def predict(self, x):
                return np.ones(len(x), dtype=int)
        rates = self.rates.copy()
        rates[:, :, 0] = np.arange(20)[:, None]
        with patch.object(decoding, 'create_base_decoder', side_effect=lambda *a: Recorder()):
            decoding.decode_one_trial(1, rates, self.labels, self.times,
                dataclasses.replace(self.config, preserve_null_time_structure=True))
        for estimate in range(4):
            block = fits[estimate * 3:(estimate + 1) * 3]
            for x, y in block:
                self.assertNotIn(1, x[:, 0])
                np.testing.assert_array_equal(x[:, 0], fits[0][0][:, 0])
                np.testing.assert_array_equal(np.bincount(y), [9, 9])
                np.testing.assert_array_equal(y, block[0][1])

    def test_rejects_non_boolean_switch(self):
        with self.assertRaisesRegex(ValueError, 'must be true or false'):
            decoding.Config(preserve_null_time_structure='false')

    def test_cli_switch_enables_and_disables(self):
        import tyro
        self.assertTrue(tyro.cli(decoding.Config, args=['--preserve-null-time-structure']).preserve_null_time_structure)
        self.assertFalse(tyro.cli(decoding.Config, args=['--no-preserve-null-time-structure']).preserve_null_time_structure)


class DecoderEligibilityTest(unittest.TestCase):
    def test_class_count_failure_precedes_binning_or_workers(self):
        source = (np.zeros((10, 3, 1)), np.arange(3), np.array([1] * 5 + [5] * 5), np.ones(10, dtype=bool))
        selection = {'num_trials': 10, 'cell_idx_selected': [0], 'cell_idx_stationary': [0],
                     'cell_properties': {'cell_idx': [0], 'preferred_cue': [1]}}
        with patch.object(decoding, 'load_session', return_value=source), patch.object(decoding, 'compute_binned_rates') as binning:
            with self.assertRaisesRegex(ValueError, 'example:.*6 correct.*5 correct.*found 5 and 5'):
                decoding.decode_session(Path('example.mat'), selection, decoding.Config(grid_search_for_c=True))
            binning.assert_not_called()

    def test_reduced_calibration_warns_but_feasible_training_continues(self):
        with self.assertWarnsRegex(RuntimeWarning, 'reducing logistic calibration from 5 to 2'):
            decoding.validate_training_class_counts(np.array([1, 1, 1, 0, 0]), decoding.Config(), context='example')
        with self.assertRaisesRegex(ValueError, '3 correct.*2 correct'):
            decoding.validate_training_class_counts(np.array([1, 1, 0, 0]), decoding.Config(), context='example')


class ComparableAccuracyTest(unittest.TestCase):
    def test_native_svm_disagreement_uses_same_threshold_for_both_estimates(self):
        source = dict(session='svm', trial_idx=[0], time_bins=[500],
                      decoding_test_labels=[1], decoding_confidence=np.array([[.6275]]),
                      decoding_predicted_labels=np.array([[0]]),
                      decoding_confidence_null=np.array([[[.6275]]]))
        with self.assertWarnsRegex(RuntimeWarning, 'cached native predictions differ'):
            result = evaluate_session(source)
        self.assertEqual(result['observed']['accuracy'], 1)
        self.assertEqual(result['observed']['accuracy'], result['null']['accuracy'])
        self.assertEqual(result['observed_accuracy_source'], result['null_accuracy_source'])
        self.assertEqual(source['decoding_predicted_labels'][0, 0], 0)

    def test_null_policy_survives_evaluation_and_cross_run_comparison_warns(self):
        source = dict(session='example', cue=1, trial_idx=[0], time_bins=[500],
                      decoding_test_labels=[1], decoding_confidence=np.array([[.7]]),
                      decoding_predicted_labels=np.array([[1]]),
                      decoding_confidence_null=np.array([[[.4, .6]]]))
        independent = evaluate_session(source)
        preserved = evaluate_session({**source, 'preserve_null_time_structure': True,
                                      'null_policy': 'shared across time bins'})
        self.assertTrue(preserved['preserve_null_time_structure'])
        self.assertEqual(preserved['null_policy'], 'shared across time bins')
        with tempfile.TemporaryDirectory() as directory:
            roots = [Path(directory) / name for name in ('independent', 'preserved')]
            for root, result in zip(roots, (independent, preserved)):
                cache_io.save([result], root / 'evaluate/eval_confidence.pkl')
            with self.assertWarnsRegex(UserWarning, 'null time-structure policies differ'):
                names, runs, sessions = load_runs(roots)
            self.assertEqual(sessions, ['example'])


if __name__ == '__main__':
    unittest.main()
