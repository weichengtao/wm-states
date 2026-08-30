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


if __name__ == '__main__':
    unittest.main()
