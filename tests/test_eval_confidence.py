import contextlib
import csv
import io
import pickle
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss

from scripts import eval_confidence


class ConfidenceEvaluationTest(unittest.TestCase):
    def test_matches_sklearn_for_both_labels(self):
        labels = np.array([0, 1, 1])
        probabilities = np.array([
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.9, 0.7], [0.6, 0.8]],
            [[0.5, 0.8], [0.9, 0.7]],
        ])
        scores = eval_confidence.score_probabilities(probabilities, labels)
        flat_labels = np.repeat(labels, 4)
        self.assertAlmostEqual(scores['brier_score'], brier_score_loss(flat_labels, probabilities.ravel()))
        self.assertAlmostEqual(scores['log_loss'], log_loss(flat_labels, probabilities.ravel()))
        for time_bin in range(2):
            for sample in range(2):
                self.assertAlmostEqual(
                    scores['log_loss_by_time_bin_and_sample'][time_bin, sample],
                    log_loss(labels, probabilities[:, time_bin, sample]),
                )

    def test_single_class_endpoints_and_missing_predictions(self):
        probabilities = np.array([[[0.0, 1.0], [np.nan, np.nan]]])
        with self.assertWarnsRegex(RuntimeWarning, '2 missing probabilities'):
            scores = eval_confidence.score_probabilities(probabilities, [1])
        self.assertEqual(scores['brier_score'], 0.5)
        self.assertTrue(np.isfinite(scores['log_loss']))
        self.assertEqual(scores['n_valid'], 2)
        self.assertTrue(np.isnan(scores['log_loss_by_time_bin'][1]))
        np.testing.assert_array_equal(scores['n_valid_by_time_bin'], [2, 0])

    def test_rejects_invalid_probabilities_and_labels(self):
        for value in (-0.01, 1.01, np.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                eval_confidence.score_probabilities([[[value]]], [1])
        for labels in ([2], [np.nan], [0, 1]):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                eval_confidence.score_probabilities([[[0.5]]], labels)

    def test_missing_data_warns_and_preserves_metric_specific_counts(self):
        source = self.source()
        source['decoding_confidence_repeats'][0, 0, 0] = np.nan
        source['decoding_predicted_labels'] = np.array([[[-1, -1], [1, 1]]])
        source['decoding_confidence_null'][0, 0, 0] = np.nan
        with self.assertWarnsRegex(RuntimeWarning, 'Session example, observed repeat 0'):
            result = eval_confidence.evaluate_session(source)
        scores = result['observed']
        self.assertEqual(scores['n_valid'], 1)
        self.assertEqual(scores['brier_score_n_valid'], 1)
        self.assertEqual(scores['accuracy_n_valid'], 0)
        self.assertTrue(np.isnan(scores['accuracy']))
        np.testing.assert_array_equal(scores['accuracy_n_valid_by_time_bin'], [0, 0])
        with self.assertWarnsRegex(RuntimeWarning, 'Session example, null'):
            eval_confidence.evaluate_session(source)

    def test_finite_probability_missing_label_warns_and_counts_accuracy_only(self):
        with self.assertWarnsRegex(RuntimeWarning, '1 additional predictions from accuracy'):
            scores = eval_confidence.score_probabilities(
                np.array([[[0.8]], [[0.2]]]), [1, 0], np.array([[[1.0]], [[np.nan]]]),
            )
        self.assertEqual(scores['accuracy'], 1.0)
        self.assertEqual(scores['accuracy_n_valid'], 1)
        self.assertEqual(scores['brier_score_n_valid'], 2)

    def source(self):
        return {
            'session': 'example', 'cue': 1, 'trial_idx': np.array([4]),
            'time_bins': np.array([0, 50]), 'decoding_test_labels': np.array([1]),
            'decoding_confidence': np.array([[0.5, 0.5]]),
            'decoding_confidence_repeats': np.array([[[0.1, 0.2], [0.9, 0.8]]]),
            'decoding_confidence_null': np.array([[[0.1, 0.9], [0.2, 0.8]]]),
        }

    def test_scores_each_null_before_averaging(self):
        result = eval_confidence.evaluate_session(self.source())
        self.assertAlmostEqual(result['observed']['brier_score'], 0.725)
        self.assertAlmostEqual(result['null']['brier_score'], 0.375)
        np.testing.assert_allclose(result['null']['brier_score_by_sample'], [0.725, 0.025])
        np.testing.assert_allclose(
            result['observed']['log_loss_by_sample'],
            result['null']['log_loss_by_sample'][:1],
        )

    def test_optional_arrays_and_mismatched_axes(self):
        source = self.source()
        source['decoding_confidence_null'] = None
        result = eval_confidence.evaluate_session(source)
        self.assertIsNone(result['null'])
        self.assertNotIn('observed_repeats', result)
        self.assertEqual(result['observed_repeat_idx'], 0)
        self.assertEqual(result['num_repeats'], 1)
        source['decoding_confidence_null'] = np.zeros((1, 3, 2))
        with self.assertRaises(ValueError):
            eval_confidence.evaluate_session(source)

    def test_later_repeats_and_cached_average_are_ignored(self):
        source = self.source()
        source['decoding_confidence_repeats'][:, 1:, :] = np.nan
        source['decoding_confidence'][:] = 1.0
        result = eval_confidence.evaluate_session(source)
        self.assertAlmostEqual(result['observed']['brier_score'], 0.725)
        self.assertEqual(result['observed']['brier_score_by_time_bin_and_sample'].shape, (2, 1))
        self.assertNotIn('observed_repeats', result)
        self.assertFalse(any('observed_repeats' in key for key in eval_confidence.summary_row(result)))
        source.pop('decoding_confidence_repeats')
        with self.assertRaisesRegex(ValueError, 'repeat 0'):
            eval_confidence.evaluate_session(source)

    def test_accuracy_uses_observed_predictions_and_null_probability_threshold(self):
        source = self.source()
        source['decoding_predicted_labels'] = np.array([[[1, 0], [0, 1]]])
        result = eval_confidence.evaluate_session(source)
        np.testing.assert_allclose(result['observed']['accuracy_by_time_bin'], [1, 0])
        np.testing.assert_allclose(result['observed']['decoding_confidence_by_time_bin'], [0.1, 0.2])
        np.testing.assert_allclose(result['null']['accuracy_by_time_bin_and_sample'], [[0, 1], [0, 1]])
        np.testing.assert_allclose(result['null']['decoding_confidence_by_time_bin_and_sample'], [[0.1, 0.9], [0.2, 0.8]])
        with self.assertWarnsRegex(RuntimeWarning, '1 missing probabilities'):
            scores = eval_confidence.score_probabilities(np.array([[[0.5, np.nan]]]), [1])
        self.assertEqual(scores['accuracy'], 1.0)
        self.assertTrue(np.isnan(scores['accuracy_by_sample'][1]))

    def test_end_to_end_cache_and_csv(self):
        with TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            source_path = cache_dir / 'decoding_confidence.pkl'
            source_path.write_bytes(pickle.dumps([self.source()]))
            original = source_path.read_bytes()
            for _ in range(2):
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    eval_confidence.main(eval_confidence.Config(cache_dir=cache_dir))
                self.assertIn('example | observed: Brier=0.725000', output.getvalue())
                with (cache_dir / 'eval_confidence.pkl').open('rb') as stream:
                    results = pickle.load(stream)
                self.assertEqual(len(results), 1)
                with (cache_dir / 'eval_confidence.csv').open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 1)
                self.assertAlmostEqual(float(rows[0]['null_brier_score']), 0.375)
                self.assertEqual(source_path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
