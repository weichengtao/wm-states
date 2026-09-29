"""Controls for the bounded three-target temporal-null experiment."""
import tempfile
import unittest
import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.next import validate_focused_nulls as validation
from scripts.next.decoder_models import LogisticCalibrationMethod, TrainingBalance
from scripts.next.common import json_value


class FocusedNullValidationTest(unittest.TestCase):
    def test_target_selection_is_metadata_only_and_deterministic(self):
        rows = [{'session': '221024', 'trial_idx': np.array([136, 200, 300]), 'cell_idx': [0, 1]},
                {'session': '210921', 'trial_idx': np.array([5, 15, 25, 35]), 'cell_idx': [0]},
                {'session': '221021', 'trial_idx': np.array([50, 60, 70]), 'cell_idx': [0, 1, 2]}]
        self.assertEqual(validation.choose_targets(rows), [('221024', 136), ('210921', 25), ('221021', 60)])
        self.assertEqual(validation.choose_targets(rows[::-1]), validation.choose_targets(rows))

    def test_candidate_duration_counts_bins_times_stride(self):
        maximum, total, lengths = validation.candidate_durations([[1, 1, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0]], step=10)
        np.testing.assert_array_equal(maximum, [30, 0])
        np.testing.assert_array_equal(total, [50, 0])
        np.testing.assert_array_equal(lengths, [20, 30])

    def test_temporal_dependence_changes_null_runs_with_similar_marginals(self):
        rng = np.random.default_rng(124)
        starts = np.arange(500, 1401, 10)
        independent = .5 + .08 * rng.normal(size=(91, 2000))
        shared = .5 + .08 * rng.normal(size=(1, 2000)) + .005 * rng.normal(size=(91, 2000))
        observed = np.full(91, .55)
        first = validation.null_summary(observed, independent, starts)
        second = validation.null_summary(observed, shared, starts)
        self.assertLess(abs(first['null_adjacent_bin_correlation_delay']['mean']), .02)
        self.assertGreater(second['null_adjacent_bin_correlation_delay']['mean'], .95)
        self.assertLess(abs(first['null_candidate_fraction_delay'] - second['null_candidate_fraction_delay']), .03)
        self.assertGreater(second['null_candidate_max_off_ms']['mean'], first['null_candidate_max_off_ms']['mean'])

    def test_zero_variance_bins_are_excluded(self):
        result = validation.null_summary(np.array([.3, .6]), np.full((2, 5), .5), np.array([500, 510]))
        self.assertEqual(result['valid_null_delay_bins'], 0)
        self.assertEqual(result['observed_candidate_total_off_ms'], 0)
        self.assertEqual(result['null_cluster_count_delay'], 0)
        self.assertIsNone(result['null_adjacent_bin_correlation_delay'])

    def test_nonbinary_constant_null_is_excluded_despite_roundoff_sd(self):
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype):
                null = np.full((2, 100), .1, dtype=dtype)
                self.assertGreater(null.std(axis=1)[0], 0)
                result = validation.null_summary(np.array([.1, .05]), null, np.array([500, 510]))
                self.assertEqual(result['valid_null_delay_bins'], 0)
                self.assertEqual(result['constant_null_bin_count_full_grid'], 2)
                self.assertEqual(result['observed_candidate_total_off_ms'], 0)
                self.assertIsNone(result['null_adjacent_bin_correlation_delay'])

    def test_float32_inputs_are_standardized_as_float64(self):
        rng = np.random.default_rng(44)
        null = rng.uniform(.2, .8, size=(3, 100)).astype(np.float32)
        observed = np.array([.5, .55, .6], dtype=np.float32)
        starts = np.array([500, 510, 520])
        self.assertEqual(validation.null_summary(observed, null, starts),
                         validation.null_summary(observed.astype(float), null.astype(float), starts))

    def test_off_candidate_boundary_and_invalid_inputs(self):
        values = np.array([[.3, .5, .7], [.3, .5, .7]])
        # An exactly representable z=0 tests the inclusive boundary without
        # floating-point roundoff from reconstructing the .842 cutoff.
        threshold = values.mean(axis=1)
        result = validation.null_summary(threshold, values, np.array([500, 510]), z_off=0)
        self.assertEqual(result['observed_candidate_total_off_ms'], 20)
        for bad in [np.full((2, 1), .5), np.array([[.5, np.nan], [.4, .6]])]:
            with self.assertRaises(ValueError):
                validation.null_summary(np.array([.5, .5]), bad, np.array([500, 510]))

    def test_official_decoder_receives_the_prespecified_fitting_policy(self):
        calls = []

        def fake_decoder(index, rates, labels, starts, config):
            calls.append(config)
            self.assertEqual(index, 0)
            self.assertEqual(rates.shape[1], 3)
            n = len(starts)
            return np.zeros(n), np.zeros(n), np.full(n, .01), np.zeros((n, 2)), np.full((n, 2), .01), (5,)

        task = {'session': 's', 'trial': 9, 'c_policy': 'fixed_001', 'shared_time': True,
                'rates': np.zeros((4, 161, 2)), 'labels': np.array([1, 0, 1, 0]),
                'starts': np.arange(-200, 1401, 10), 'trial_ids': np.array([9, 10, 11, 12]),
                'test_index': 0, 'cached_observed': np.zeros(161)}
        with tempfile.TemporaryDirectory() as directory, patch.object(validation, 'decode_one_trial', fake_decoder):
            result = validation.fit_task(task, n_null=2, benchmark=True, output_dir=Path(directory))
            config = calls[0]
            self.assertEqual(config.seed, 42)
            self.assertEqual(config.training_balance, TrainingBalance.BALANCED_CLASS_WEIGHTS)
            self.assertEqual(config.classifier_c, .01)
            self.assertFalse(config.grid_search_for_c)
            self.assertTrue(config.preserve_null_time_structure)
            self.assertEqual(config.logistic_calibration_method, LogisticCalibrationMethod.SIGMOID)
            self.assertEqual(config.logistic_calibration_cv, 5)
            with np.load(result['artifact']) as arrays:
                np.testing.assert_array_equal(arrays['training_trial_ids'], [10, 11, 12])

    def test_policy_comparison_requires_identical_observed_fits(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / 'first.npz', Path(directory) / 'second.npz'
            observed = np.array([.5, .6])
            selected_c = np.array([1., .01])
            null = np.array([[.4, .5, .6], [.4, .5, .6]])
            np.savez(first, observed=observed, selected_c=selected_c, null=null)
            np.savez(second, observed=observed, selected_c=selected_c, null=null + np.array([[0], [.1]]))
            rows = [{'session': 's', 'trial': 1, 'c_policy': 'search', 'shared_time': shared,
                     'artifact': str(path), 'summary': {'observed_candidate_max_off_ms': maximum,
                                                       'observed_candidate_total_off_ms': total}}
                    for shared, path, maximum, total in [(False, first, 20, 40), (True, second, 30, 20)]]
            result = validation.compare_policies(rows)[0]
            self.assertTrue(result['observed_exact_match_between_null_policies'])
            self.assertTrue(result['null_first_bin_exact_match'])
            self.assertAlmostEqual(result['pointwise_null_mean_mean_abs_difference'], .05)
            self.assertEqual(result['candidate_max_off_shared_minus_independent_ms'], 10)
            self.assertEqual(result['candidate_total_off_shared_minus_independent_ms'], -20)
            for changed_observed, changed_c in [(observed + .01, selected_c), (observed, selected_c + .1)]:
                np.savez(second, observed=changed_observed, selected_c=changed_c, null=null)
                with self.assertRaises(ValueError):
                    validation.compare_policies(rows)

    def test_archive_validation_checks_digest_shapes_and_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trial.npz'
            config = validation.Config(seed=42, training_balance=TrainingBalance.BALANCED_CLASS_WEIGHTS,
                                       grid_search_for_c=False, classifier_c=.01,
                                       preserve_null_time_structure=True, n_decode_shuffle=3)
            arrays = {'observed': np.full(161, .5), 'predicted_labels': np.ones(161),
                      'selected_c': np.full(161, .01), 'null': np.full((161, 3), .5),
                      'null_c': np.full((161, 3), .01), 'bin_starts': np.arange(-200, 1401, 10),
                      'training_trial_ids': np.array([1, 2, 3]),
                      'configuration': json.dumps(asdict(config), default=json_value)}
            row = {'artifact': str(path), 'trial': 4, 'c_policy': 'fixed_001', 'shared_time': True}
            np.savez(path, **arrays)
            row['artifact_sha256'] = validation.file_hash(path)
            validation.load_verified_artifact(row, n_null=3, n_training=3)
            row['artifact_sha256'] = 'changed'
            with self.assertRaisesRegex(ValueError, 'digest'):
                validation.load_verified_artifact(row, n_null=3, n_training=3)
            for field, bad in [('null', np.full((160, 3), .5)), ('observed', np.full(161, 1.1)),
                               ('training_trial_ids', np.array([1, 2, 4])), ('selected_c', np.ones(161)),
                               ('configuration', '{}')]:
                np.savez(path, **{**arrays, field: bad})
                row['artifact_sha256'] = validation.file_hash(path)
                with self.assertRaises(ValueError):
                    validation.load_verified_artifact(row, n_null=3, n_training=3)

    def test_cached_null_reproduction_handles_different_bank_sizes(self):
        cached = np.arange(200).reshape(2, 100)
        short = validation.compare_cached_null_prefix(cached[:, :50], cached)
        self.assertTrue(short['null_matches_run005'])
        self.assertEqual(short['null_reproduction_compared_count'], 50)
        long = validation.compare_cached_null_prefix(np.c_[cached, np.zeros((2, 20))], cached)
        self.assertTrue(long['null_matches_run005'])
        self.assertEqual(long['null_reproduction_compared_count'], 100)
        self.assertEqual(long['null_reproduction_cached_count'], 100)
        self.assertFalse(validation.compare_cached_null_prefix(cached + 1, cached)['null_matches_run005'])
        with self.assertRaises(ValueError):
            validation.compare_cached_null_prefix(cached[:1], cached)


if __name__ == '__main__':
    unittest.main()
