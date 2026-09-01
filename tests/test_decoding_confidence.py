import pickle
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from scripts import decoding_confidence


class CacheCheckpointTest(unittest.TestCase):
    def test_atomic_save_replaces_only_with_complete_pickle(self):
        class Unpickleable:
            def __reduce__(self):
                raise RuntimeError('intentional pickle failure')

        with TemporaryDirectory() as temporary_directory:
            cache_path = Path(temporary_directory) / 'decoding_confidence.pkl'
            decoding_confidence.save_pickle_atomic(
                [{'session': 'first'}],
                cache_path,
            )
            with open(cache_path, 'rb') as cache_file:
                self.assertEqual(pickle.load(cache_file), [{'session': 'first'}])

            with self.assertRaisesRegex(RuntimeError, 'intentional pickle failure'):
                decoding_confidence.save_pickle_atomic(
                    [Unpickleable()],
                    cache_path,
                )

            with open(cache_path, 'rb') as cache_file:
                self.assertEqual(pickle.load(cache_file), [{'session': 'first'}])
            self.assertEqual(
                list(Path(temporary_directory).glob('*.tmp')),
                [],
            )


class CuePreservedTrainSetShuffleTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(123)
        self.binned_rates = rng.normal(size=(8, 3, 4)).astype(np.float32)
        self.labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
        self.test_idx = 4

    def decode(self, enabled, balance_training_trials=True):
        return decoding_confidence.decode_one_trial(
            self.test_idx,
            self.binned_rates,
            self.labels,
            seed=42,
            balance_decoder_training_trials=balance_training_trials,
            classifier_c=0.1,
            decoder_model=decoding_confidence.DecoderModel.LOGISTIC_REGRESSION,
            svm_kernel=decoding_confidence.SVMKernel.LINEAR,
            n_repeats_for_model_fit=3,
            n_shuffle=5,
            cue_preserved_train_set_shuffle=enabled,
        )

    def test_repeat_zero_and_null_are_unchanged(self):
        unshuffled = self.decode(enabled=False)
        shuffled = self.decode(enabled=True)

        for output_idx in range(3):
            np.testing.assert_array_equal(
                unshuffled[output_idx][0],
                shuffled[output_idx][0],
            )
        np.testing.assert_array_equal(unshuffled[3], shuffled[3])

    def test_only_repeats_after_zero_shuffle_balanced_training_set(self):
        with patch.object(
            decoding_confidence,
            'shuffle_trial_idx_within_labels',
            wraps=decoding_confidence.shuffle_trial_idx_within_labels,
        ) as shuffle:
            self.decode(enabled=True, balance_training_trials=True)

        self.assertEqual(shuffle.call_count, 2)
        for call in shuffle.call_args_list:
            shuffled_rates, shuffled_labels, _ = call.args
            self.assertEqual(shuffled_rates.shape, (6, 3, 4))
            np.testing.assert_array_equal(
                np.unique(shuffled_labels, return_counts=True)[1],
                np.asarray([3, 3]),
            )

    def test_shuffle_follows_imbalanced_training_set_preparation(self):
        with patch.object(
            decoding_confidence,
            'shuffle_trial_idx_within_labels',
            wraps=decoding_confidence.shuffle_trial_idx_within_labels,
        ) as shuffle:
            self.decode(enabled=True, balance_training_trials=False)

        self.assertEqual(shuffle.call_count, 2)
        for call in shuffle.call_args_list:
            shuffled_rates, shuffled_labels, _ = call.args
            self.assertEqual(shuffled_rates.shape, (7, 3, 4))
            np.testing.assert_array_equal(
                np.unique(shuffled_labels, return_counts=True)[1],
                np.asarray([4, 3]),
            )


class DelayTrainingBinPoolingTest(unittest.TestCase):
    def setUp(self):
        self.binned_rates = np.arange(3 * 5 * 2, dtype=np.float32).reshape(3, 5, 2)
        self.labels = np.asarray([0, 1, 0], dtype=np.int64)
        self.bin_starts = np.asarray([490, 500, 750, 1390, 1400])
        self.delay_bin_mask = (
            (self.bin_starts >= 500) & (self.bin_starts <= 1400)
        )

    def prepare(self, test_bin_idx, enabled=True):
        return decoding_confidence.prepare_decoder_training_samples(
            self.binned_rates,
            self.labels,
            test_bin_idx,
            self.delay_bin_mask,
            enabled,
        )

    def test_delay_boundaries_are_inclusive(self):
        for test_bin_idx in (1, 2, 3, 4):
            X_train, y_train = self.prepare(test_bin_idx)
            self.assertEqual(X_train.shape, (12, 2))
            np.testing.assert_array_equal(
                y_train,
                np.repeat(self.labels, 4),
            )

        X_train, y_train = self.prepare(test_bin_idx=0)
        np.testing.assert_array_equal(X_train, self.binned_rates[:, 0, :])
        np.testing.assert_array_equal(y_train, self.labels)

    def test_disabled_preserves_same_bin_training(self):
        X_train, y_train = self.prepare(test_bin_idx=2, enabled=False)
        np.testing.assert_array_equal(X_train, self.binned_rates[:, 2, :])
        np.testing.assert_array_equal(y_train, self.labels)

    def test_delay_pool_is_trial_major(self):
        X_train, y_train = self.prepare(test_bin_idx=2)
        expected = self.binned_rates[:, self.delay_bin_mask, :].reshape(12, 2)
        np.testing.assert_array_equal(X_train, expected)
        np.testing.assert_array_equal(y_train, np.repeat(self.labels, 4))

    def test_pooling_requires_bin_starts(self):
        with self.assertRaisesRegex(ValueError, 'bin_starts is required'):
            decoding_confidence.decode_one_trial(
                0,
                self.binned_rates,
                self.labels,
                seed=42,
                balance_decoder_training_trials=True,
                classifier_c=0.1,
                decoder_model=decoding_confidence.DecoderModel.LOGISTIC_REGRESSION,
                svm_kernel=decoding_confidence.SVMKernel.LINEAR,
                n_repeats_for_model_fit=1,
                n_shuffle=0,
                train_delay_decoder_using_all_delay_time_bins=True,
            )

    def test_all_repeats_and_null_pool_without_the_test_trial(self):
        rng = np.random.default_rng(123)
        binned_rates = rng.normal(size=(8, 5, 4)).astype(np.float32)
        labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
        test_idx = 4
        prepared = []
        original_prepare = decoding_confidence.prepare_decoder_training_samples

        def record_prepare(rates, trial_labels, bin_idx, delay_mask, enabled):
            X_train, y_train = original_prepare(
                rates,
                trial_labels,
                bin_idx,
                delay_mask,
                enabled,
            )
            prepared.append(
                (rates.copy(), bin_idx, enabled, X_train.shape, y_train.shape)
            )
            return X_train, y_train

        with patch.object(
            decoding_confidence,
            'prepare_decoder_training_samples',
            side_effect=record_prepare,
        ):
            decoding_confidence.decode_one_trial(
                test_idx,
                binned_rates,
                labels,
                seed=42,
                balance_decoder_training_trials=True,
                classifier_c=0.1,
                decoder_model=decoding_confidence.DecoderModel.LOGISTIC_REGRESSION,
                svm_kernel=decoding_confidence.SVMKernel.LINEAR,
                n_repeats_for_model_fit=3,
                n_shuffle=2,
                cue_preserved_train_set_shuffle=True,
                bin_starts=self.bin_starts,
                train_delay_decoder_using_all_delay_time_bins=True,
            )

        # Each of three repeats and the null path prepares one pooled delay set
        # plus the one non-delay bin.
        self.assertEqual(len(prepared), 8)
        for rates, _, enabled, X_shape, y_shape in prepared:
            self.assertFalse(
                np.any(np.all(rates == binned_rates[test_idx], axis=(1, 2)))
            )
            if enabled:
                self.assertEqual(X_shape, (24, 4))
                self.assertEqual(y_shape, (24,))
            else:
                self.assertEqual(X_shape, (6, 4))
                self.assertEqual(y_shape, (6,))

        # The final two preparations are the null path; its selected training
        # trials match repeat zero's first two preparations exactly.
        for repeat_zero_call, null_call in zip(prepared[:2], prepared[-2:]):
            np.testing.assert_array_equal(repeat_zero_call[0], null_call[0])


class LogisticCalibrationTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(321)
        self.binned_rates = rng.normal(size=(12, 3, 4)).astype(np.float32)
        self.labels = np.asarray([0] * 6 + [1] * 6, dtype=np.int64)
        self.test_idx = 6

    def decode(
        self,
        decoder_model=decoding_confidence.DecoderModel.LOGISTIC_REGRESSION,
        calibration_method=decoding_confidence.LogisticCalibrationMethod.SIGMOID,
        calibration_cv=5,
        n_shuffle=0,
        **kwargs,
    ):
        return decoding_confidence.decode_one_trial(
            self.test_idx,
            self.binned_rates,
            self.labels,
            seed=42,
            balance_decoder_training_trials=True,
            classifier_c=0.1,
            decoder_model=decoder_model,
            svm_kernel=decoding_confidence.SVMKernel.LINEAR,
            n_repeats_for_model_fit=1,
            n_shuffle=n_shuffle,
            logistic_calibration_method=calibration_method,
            logistic_calibration_cv=calibration_cv,
            **kwargs,
        )

    def test_grouped_folds_keep_pooled_samples_from_one_trial_together(self):
        trial_labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
        labels = np.repeat(trial_labels, 3)
        groups = decoding_confidence.decoder_training_sample_groups(6, 18)
        splits, effective_cv = (
            decoding_confidence.make_logistic_calibration_cv_splits(
                labels,
                groups,
                requested_cv_folds=5,
                seed=42,
            )
        )

        self.assertEqual(effective_cv, 3)
        validation_indices = []
        for train_indices, fold_validation_indices in splits:
            self.assertEqual(np.unique(labels[train_indices]).size, 2)
            self.assertEqual(np.unique(labels[fold_validation_indices]).size, 2)
            self.assertFalse(
                np.intersect1d(
                    groups[train_indices],
                    groups[fold_validation_indices],
                ).size
            )
            validation_indices.extend(fold_validation_indices.tolist())
        np.testing.assert_array_equal(
            np.sort(validation_indices),
            np.arange(labels.size),
        )

    def test_fold_count_is_reduced_to_smallest_class_trial_count(self):
        labels = np.asarray([0, 0, 1, 1, 1, 1], dtype=np.int64)
        groups = np.arange(labels.size)
        _, effective_cv = (
            decoding_confidence.make_logistic_calibration_cv_splits(
                labels,
                groups,
                requested_cv_folds=5,
                seed=42,
            )
        )
        self.assertEqual(effective_cv, 2)

    def test_sigmoid_calibration_is_finite_and_deterministic(self):
        first = self.decode()
        second = self.decode()
        uncalibrated = self.decode(
            calibration_method=(
                decoding_confidence.LogisticCalibrationMethod.NONE
            )
        )

        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        np.testing.assert_array_equal(first[2], second[2])
        self.assertFalse(np.allclose(first[0], uncalibrated[0]))
        self.assertEqual(first[4], (5,))
        self.assertTrue(np.all(np.isfinite(first[0])))
        self.assertTrue(np.all((first[0] >= 0.0) & (first[0] <= 1.0)))

    def test_calibration_never_fits_on_outer_test_trial(self):
        self.binned_rates[self.test_idx] = 999.0
        fitted_features = []
        original_fit = decoding_confidence.CalibratedClassifierCV.fit

        def record_fit(calibrator, X, y, *args, **kwargs):
            fitted_features.append(np.asarray(X).copy())
            return original_fit(calibrator, X, y, *args, **kwargs)

        with patch.object(
            decoding_confidence.CalibratedClassifierCV,
            'fit',
            new=record_fit,
        ):
            self.decode()

        self.assertTrue(fitted_features)
        for X_train in fitted_features:
            self.assertFalse(np.any(np.all(X_train == 999.0, axis=1)))

    def test_calibration_is_applied_to_label_shuffled_null(self):
        output = self.decode(
            n_shuffle=2,
            bin_starts=np.asarray([500, 600, 700]),
            train_delay_decoder_using_all_delay_time_bins=True,
        )
        self.assertEqual(output[3].shape, (3, 2))
        self.assertTrue(np.all(np.isfinite(output[3])))
        self.assertTrue(np.all((output[3] >= 0.0) & (output[3] <= 1.0)))
        self.assertTrue(output[4])

    def test_logistic_calibration_settings_do_not_change_svm(self):
        uncalibrated = self.decode(
            decoder_model=decoding_confidence.DecoderModel.SVM,
            calibration_method=decoding_confidence.LogisticCalibrationMethod.NONE,
        )
        ignored_calibration = self.decode(
            decoder_model=decoding_confidence.DecoderModel.SVM,
            calibration_method=decoding_confidence.LogisticCalibrationMethod.ISOTONIC,
            calibration_cv=1,
        )

        for output_idx in range(3):
            np.testing.assert_array_equal(
                uncalibrated[output_idx],
                ignored_calibration[output_idx],
            )
        self.assertEqual(uncalibrated[4], ())
        self.assertEqual(ignored_calibration[4], ())


if __name__ == '__main__':
    unittest.main()
