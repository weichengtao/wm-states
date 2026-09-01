import unittest
from unittest.mock import patch

import numpy as np

from scripts import decoding_confidence


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


if __name__ == '__main__':
    unittest.main()
