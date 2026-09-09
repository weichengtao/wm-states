import pickle
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from scripts import decoding_confidence


class SessionListFilterTest(unittest.TestCase):
    def test_loads_unique_uncommented_session_ids(self):
        with TemporaryDirectory() as temporary_directory:
            session_list = Path(temporary_directory) / 'sessions.txt'
            session_list.write_text(
                '\n'.join([
                    '# Disabled session',
                    '  221024  ',
                    '',
                    '   # 221025',
                    '221026',
                    '221024',
                ]),
                encoding='utf-8',
            )

            sessions = decoding_confidence.load_session_list(session_list)

        self.assertEqual(sessions, ['221024', '221026'])

    def test_missing_session_list_raises_clear_error(self):
        missing_path = Path('missing-session-list.txt')
        with self.assertRaisesRegex(
            FileNotFoundError,
            'Missing session list file: missing-session-list.txt',
        ):
            decoding_confidence.load_session_list(missing_path)

    def test_filters_membership_without_changing_eligible_order(self):
        good_sessions = {
            '221024': {'preferred_cue': 1},
            '221025': {'preferred_cue': 2},
            '221026': {'preferred_cue': 3},
        }

        filtered, warnings = decoding_confidence.filter_sessions_by_list(
            good_sessions,
            known_sessions={'221024', '221025', '221026'},
            requested_sessions=['221026', '221024'],
        )

        self.assertEqual(list(filtered), ['221024', '221026'])
        self.assertEqual(warnings, [])

    def test_warns_for_unknown_and_known_but_ineligible_sessions(self):
        filtered, warnings = decoding_confidence.filter_sessions_by_list(
            {'221024': {'preferred_cue': 1}},
            known_sessions={'221024', '221025'},
            requested_sessions=['unknown', '221025'],
        )

        self.assertEqual(filtered, {})
        self.assertEqual(len(warnings), 2)
        self.assertIn(
            'Session unknown is not present in cell_trial_selection.pkl',
            warnings[0],
        )
        self.assertIn(
            'Session 221025 is not eligible for decoding',
            warnings[1],
        )


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


class ClassifierCSelectionTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(987)
        self.binned_rates = rng.normal(size=(12, 2, 4)).astype(np.float32)
        self.labels = np.asarray([0] * 6 + [1] * 6, dtype=np.int64)
        self.test_idx = 6
        self.seed = 42

    def decode(
        self,
        *,
        balance_training_trials=True,
        grid_search_for_c=False,
        classifier_c=0.1,
        n_repeats_for_model_fit=2,
        n_shuffle=2,
        **kwargs,
    ):
        return decoding_confidence.decode_one_trial(
            self.test_idx,
            self.binned_rates,
            self.labels,
            seed=self.seed,
            balance_decoder_training_trials=balance_training_trials,
            classifier_c=classifier_c,
            decoder_model=decoding_confidence.DecoderModel.LOGISTIC_REGRESSION,
            svm_kernel=decoding_confidence.SVMKernel.LINEAR,
            n_repeats_for_model_fit=n_repeats_for_model_fit,
            n_shuffle=n_shuffle,
            grid_search_for_c=grid_search_for_c,
            **kwargs,
        )

    def _expected_prepared_training_sets(
        self,
        *,
        balance_training_trials,
        n_repeats_for_model_fit,
        n_shuffle,
        cue_preserved_train_set_shuffle=False,
    ):
        train_idx = np.delete(np.arange(self.labels.size), self.test_idx)
        preferred_idx = train_idx[self.labels[train_idx] == 1]
        opposite_idx = train_idx[self.labels[train_idx] == 0]

        repeat_rng = np.random.default_rng(self.seed + self.test_idx)
        repeat_training_indices = []
        for _ in range(n_repeats_for_model_fit):
            if balance_training_trials:
                n_train = min(preferred_idx.size, opposite_idx.size)
                selected_preferred = repeat_rng.choice(
                    preferred_idx,
                    size=n_train,
                    replace=False,
                )
                selected_opposite = repeat_rng.choice(
                    opposite_idx,
                    size=n_train,
                    replace=False,
                )
                repeat_training_indices.append(
                    np.concatenate([selected_preferred, selected_opposite])
                )
            else:
                repeat_training_indices.append(train_idx)

        expected = []
        train_shuffle_rng = np.random.default_rng(self.seed + self.test_idx)
        for repeat_idx, selected_indices in enumerate(repeat_training_indices):
            repeat_rates = self.binned_rates[selected_indices]
            repeat_labels = self.labels[selected_indices]
            if cue_preserved_train_set_shuffle and repeat_idx > 0:
                repeat_rates = (
                    decoding_confidence.shuffle_trial_idx_within_labels(
                        repeat_rates,
                        repeat_labels,
                        train_shuffle_rng,
                    )
                )
            for bin_idx in range(self.binned_rates.shape[1]):
                expected.append((
                    repeat_rates[:, bin_idx, :],
                    repeat_labels,
                ))

        null_rng = np.random.default_rng(self.seed + self.test_idx)
        if balance_training_trials:
            n_train = min(preferred_idx.size, opposite_idx.size)
            null_training_indices = np.concatenate([
                null_rng.choice(preferred_idx, size=n_train, replace=False),
                null_rng.choice(opposite_idx, size=n_train, replace=False),
            ])
        else:
            null_training_indices = train_idx
        null_labels = self.labels[null_training_indices]
        for bin_idx in range(self.binned_rates.shape[1]):
            null_features = self.binned_rates[
                null_training_indices,
                bin_idx,
                :,
            ]
            for _ in range(n_shuffle):
                expected.append((null_features, null_rng.permutation(null_labels)))
        return expected

    def test_fixed_c_is_cached_for_every_fit_and_bypasses_selection(self):
        with patch.object(decoding_confidence, 'select_classifier_c') as select_c:
            output = self.decode(
                grid_search_for_c=False,
                classifier_c=0.1,
                n_repeats_for_model_fit=2,
                n_shuffle=3,
            )

        select_c.assert_not_called()
        repeat_c = output[5]
        null_c = output[6]
        self.assertEqual(repeat_c.shape, output[0].shape)
        self.assertEqual(null_c.shape, output[3].shape)
        np.testing.assert_allclose(repeat_c, 0.1)
        np.testing.assert_allclose(null_c, 0.1)

    def test_missing_class_returns_nan_c_arrays(self):
        output = decoding_confidence.decode_one_trial(
            0,
            np.zeros((4, 2, 3), dtype=np.float32),
            np.ones(4, dtype=np.int64),
            seed=self.seed,
            balance_decoder_training_trials=True,
            classifier_c=0.1,
            decoder_model=decoding_confidence.DecoderModel.LOGISTIC_REGRESSION,
            svm_kernel=decoding_confidence.SVMKernel.LINEAR,
            n_repeats_for_model_fit=2,
            n_shuffle=3,
            grid_search_for_c=True,
        )

        self.assertEqual(output[5].shape, (2, 2))
        self.assertEqual(output[6].shape, (2, 3))
        self.assertTrue(np.all(np.isnan(output[5])))
        self.assertTrue(np.all(np.isnan(output[6])))

    def test_grid_search_uses_each_exact_balanced_or_imbalanced_fit_set(self):
        n_repeats = 2
        n_shuffle = 2
        num_bins = self.binned_rates.shape[1]
        num_fits = n_repeats * num_bins + num_bins * n_shuffle

        for balance_training_trials in (True, False):
            with self.subTest(balance_training_trials=balance_training_trials):
                calls = []
                selected_values = [
                    decoding_confidence.CLASSIFIER_C_GRID[
                        call_idx % len(decoding_confidence.CLASSIFIER_C_GRID)
                    ]
                    for call_idx in range(num_fits)
                ]

                def record_selection(
                    X_train,
                    y_train,
                    sample_groups,
                    decoder_model,
                    svm_kernel,
                    seed,
                    *,
                    fit_context='decoder',
                ):
                    calls.append((
                        np.asarray(X_train).copy(),
                        np.asarray(y_train).copy(),
                        np.asarray(sample_groups).copy(),
                        decoder_model,
                        svm_kernel,
                        seed,
                        fit_context,
                    ))
                    return selected_values[len(calls) - 1]

                with patch.object(
                    decoding_confidence,
                    'select_classifier_c',
                    side_effect=record_selection,
                ):
                    output = self.decode(
                        balance_training_trials=balance_training_trials,
                        grid_search_for_c=True,
                        n_repeats_for_model_fit=n_repeats,
                        n_shuffle=n_shuffle,
                        cue_preserved_train_set_shuffle=True,
                    )

                self.assertEqual(len(calls), num_fits)
                expected_training_sets = self._expected_prepared_training_sets(
                    balance_training_trials=balance_training_trials,
                    n_repeats_for_model_fit=n_repeats,
                    n_shuffle=n_shuffle,
                    cue_preserved_train_set_shuffle=True,
                )
                self.assertEqual(len(expected_training_sets), num_fits)
                for call, (expected_X, expected_y) in zip(
                    calls,
                    expected_training_sets,
                ):
                    X_train, y_train, groups, model, kernel, seed, context = call
                    np.testing.assert_array_equal(X_train, expected_X)
                    np.testing.assert_array_equal(y_train, expected_y)
                    np.testing.assert_array_equal(
                        groups,
                        np.arange(expected_y.size),
                    )
                    self.assertIs(
                        model,
                        decoding_confidence.DecoderModel.LOGISTIC_REGRESSION,
                    )
                    self.assertIs(kernel, decoding_confidence.SVMKernel.LINEAR)
                    self.assertEqual(seed, self.seed)
                    self.assertTrue(context)

                empirical_fit_count = n_repeats * num_bins
                np.testing.assert_allclose(
                    output[5],
                    np.asarray(
                        selected_values[:empirical_fit_count]
                    ).reshape(n_repeats, num_bins),
                )
                np.testing.assert_allclose(
                    output[6],
                    np.asarray(
                        selected_values[empirical_fit_count:]
                    ).reshape(num_bins, n_shuffle),
                )

    def test_selected_c_is_used_by_calibrated_logistic_estimator(self):
        selected_values = [1.0, 0.01]
        calibrated_estimator_cs = []
        original_calibrator = decoding_confidence.CalibratedClassifierCV

        def record_calibrator(*args, **kwargs):
            estimator = kwargs.get('estimator', args[0] if args else None)
            calibrated_estimator_cs.append(
                estimator.named_steps['classifier'].C
            )
            return original_calibrator(*args, **kwargs)

        with (
            patch.object(
                decoding_confidence,
                'select_classifier_c',
                side_effect=selected_values,
            ),
            patch.object(
                decoding_confidence,
                'CalibratedClassifierCV',
                side_effect=record_calibrator,
            ),
        ):
            output = self.decode(
                grid_search_for_c=True,
                n_repeats_for_model_fit=1,
                n_shuffle=0,
                logistic_calibration_method=(
                    decoding_confidence.LogisticCalibrationMethod.SIGMOID
                ),
                logistic_calibration_cv=5,
            )

        np.testing.assert_allclose(output[5], np.asarray([selected_values]))
        self.assertIsNone(output[6])
        np.testing.assert_allclose(calibrated_estimator_cs, selected_values)

    def test_selected_c_is_used_by_final_svm(self):
        original_builder = decoding_confidence.create_base_decoder
        with (
            patch.object(
                decoding_confidence,
                'select_classifier_c',
                return_value=0.01,
            ),
            patch.object(
                decoding_confidence,
                'create_base_decoder',
                wraps=original_builder,
            ) as create_decoder,
        ):
            output = decoding_confidence.decode_one_trial(
                self.test_idx,
                self.binned_rates,
                self.labels,
                seed=self.seed,
                balance_decoder_training_trials=True,
                classifier_c=0.1,
                decoder_model=decoding_confidence.DecoderModel.SVM,
                svm_kernel=decoding_confidence.SVMKernel.LINEAR,
                n_repeats_for_model_fit=1,
                n_shuffle=0,
                grid_search_for_c=True,
            )

        np.testing.assert_allclose(output[5], 0.01)
        self.assertEqual(create_decoder.call_count, self.binned_rates.shape[1])
        for call in create_decoder.call_args_list:
            self.assertEqual(call.args[0], 0.01)
            self.assertIs(call.args[1], decoding_confidence.DecoderModel.SVM)

    def test_grid_search_configuration_constants(self):
        self.assertEqual(
            decoding_confidence.CLASSIFIER_C_GRID,
            (1.0, 0.1, 0.01),
        )
        self.assertEqual(decoding_confidence.CLASSIFIER_C_GRID_SEARCH_CV, 5)

    def test_selector_uses_five_grouped_balanced_accuracy_folds(self):
        trial_labels = np.asarray([0] * 5 + [1] * 5, dtype=np.int64)
        y_train = np.repeat(trial_labels, 2)
        X_train = np.random.default_rng(123).normal(size=(20, 3))
        groups = np.repeat(np.arange(10), 2)
        captured = {}

        class RecordingGridSearch:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.best_params_ = {'classifier__C': 0.1}

            def fit(self, X, y):
                captured['fit_X'] = np.asarray(X).copy()
                captured['fit_y'] = np.asarray(y).copy()
                return self

        with patch.object(
            decoding_confidence,
            'GridSearchCV',
            RecordingGridSearch,
        ):
            selected_c = decoding_confidence.select_classifier_c(
                X_train,
                y_train,
                groups,
                decoding_confidence.DecoderModel.LOGISTIC_REGRESSION,
                decoding_confidence.SVMKernel.LINEAR,
                seed=42,
            )

        self.assertEqual(selected_c, 0.1)
        self.assertEqual(
            captured['param_grid'],
            {'classifier__C': [1.0, 0.1, 0.01]},
        )
        self.assertEqual(captured['scoring'], 'balanced_accuracy')
        self.assertEqual(captured['n_jobs'], 1)
        self.assertFalse(captured['refit'])
        self.assertEqual(len(captured['cv']), 5)
        np.testing.assert_array_equal(captured['fit_X'], X_train)
        np.testing.assert_array_equal(captured['fit_y'], y_train)
        validation_indices = []
        for train_indices, validation_fold_indices in captured['cv']:
            self.assertEqual(np.unique(y_train[train_indices]).size, 2)
            self.assertEqual(
                np.unique(y_train[validation_fold_indices]).size,
                2,
            )
            self.assertFalse(
                np.intersect1d(
                    groups[train_indices],
                    groups[validation_fold_indices],
                ).size
            )
            validation_indices.extend(validation_fold_indices.tolist())
        np.testing.assert_array_equal(
            np.sort(validation_indices),
            np.arange(y_train.size),
        )

    def test_selector_runs_real_grid_search_deterministically(self):
        rng = np.random.default_rng(456)
        trial_labels = np.asarray([0] * 5 + [1] * 5, dtype=np.int64)
        y_train = np.repeat(trial_labels, 2)
        groups = np.repeat(np.arange(trial_labels.size), 2)
        X_train = rng.normal(size=(y_train.size, 3))
        X_train[:, 0] += y_train * 0.75

        selected = [
            decoding_confidence.select_classifier_c(
                X_train,
                y_train,
                groups,
                decoding_confidence.DecoderModel.LOGISTIC_REGRESSION,
                decoding_confidence.SVMKernel.LINEAR,
                seed=42,
            )
            for _ in range(2)
        ]

        self.assertEqual(selected[0], selected[1])
        self.assertIn(selected[0], decoding_confidence.CLASSIFIER_C_GRID)

    def test_selector_prefers_first_candidate_on_exact_score_tie(self):
        y_train = np.asarray([0] * 5 + [1] * 5, dtype=np.int64)
        selected_c = decoding_confidence.select_classifier_c(
            np.zeros((y_train.size, 2)),
            y_train,
            np.arange(y_train.size),
            decoding_confidence.DecoderModel.LOGISTIC_REGRESSION,
            decoding_confidence.SVMKernel.LINEAR,
            seed=42,
        )
        self.assertEqual(selected_c, 1.0)

    def test_selector_requires_five_groups_containing_each_class(self):
        y_train = np.asarray([0] * 4 + [1] * 4, dtype=np.int64)
        groups = np.arange(y_train.size)
        X_train = np.arange(y_train.size * 2, dtype=float).reshape(-1, 2)

        with self.assertRaisesRegex(
            ValueError,
            'Empirical fit: Classifier C grid search requires 5',
        ):
            decoding_confidence.select_classifier_c(
                X_train,
                y_train,
                groups,
                decoding_confidence.DecoderModel.LOGISTIC_REGRESSION,
                decoding_confidence.SVMKernel.LINEAR,
                seed=42,
                fit_context='Empirical fit',
            )

    def test_five_grouped_folds_support_mixed_label_source_trials(self):
        labels = np.tile(np.asarray([0, 1], dtype=np.int64), 10)
        groups = np.repeat(np.arange(10), 2)
        splits, effective_folds = (
            decoding_confidence.make_grouped_stratified_cv_splits(
                labels,
                groups,
                requested_cv_folds=5,
                seed=42,
                purpose='Classifier C grid search',
                allow_fold_reduction=False,
            )
        )

        self.assertEqual(effective_folds, 5)
        validation_indices = []
        for train_indices, fold_validation_indices in splits:
            self.assertEqual(np.unique(labels[train_indices]).size, 2)
            self.assertEqual(
                np.unique(labels[fold_validation_indices]).size,
                2,
            )
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


class ClassifierCPlotTest(unittest.TestCase):
    def test_log10_transform_and_means_use_log_space(self):
        classifier_c = np.asarray([
            [[1.0, 0.1], [0.01, 1.0]],
            [[0.1, np.nan], [1.0, 0.01]],
        ])
        np.testing.assert_allclose(
            decoding_confidence.log10_classifier_c(classifier_c),
            np.asarray([
                [[0.0, -1.0], [-2.0, 0.0]],
                [[-1.0, np.nan], [0.0, -2.0]],
            ]),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            decoding_confidence.mean_log10_classifier_c(
                classifier_c,
                axis=1,
            ),
            np.asarray([[-1.0, -0.5], [-0.5, -2.0]]),
        )

    def test_grid_search_plot_limits_match_candidate_log10_values(self):
        vmin, vmax, ticks = decoding_confidence.classifier_c_plot_limits(
            np.asarray([[0.0, -1.0, -2.0]]),
            grid_search_for_c=True,
        )
        self.assertEqual(vmin, -2.0)
        self.assertEqual(vmax, 0.0)
        np.testing.assert_array_equal(ticks, np.asarray([-2.0, -1.0, 0.0]))

    def test_empirical_and_null_heatmaps_and_lineplots_are_generated(self):
        classifier_c_repeats = np.asarray([
            [[1.0, 0.1], [0.1, 0.01]],
            [[0.01, 1.0], [1.0, 0.1]],
        ])
        classifier_c_null = np.asarray([
            [[1.0, 0.1], [0.1, 0.01]],
            [[0.01, 0.1], [0.1, 1.0]],
        ])

        with (
            TemporaryDirectory() as temporary_directory,
            patch.object(
                decoding_confidence,
                'save_figure_all_formats',
            ) as save_figure,
        ):
            decoding_confidence.plot_decoding_classifier_c(
                Path(temporary_directory),
                session='session_a',
                pref_cue=1,
                cue_angle=-135,
                trial_idx_pref=np.asarray([3, 7]),
                bin_starts=np.asarray([0, 200]),
                classifier_c_repeats=classifier_c_repeats,
                classifier_c_null=classifier_c_null,
                plot_actual_trial_id=False,
                num_cells=4,
                grid_search_for_c=True,
            )

        output_names = [call.args[1].name for call in save_figure.call_args_list]
        self.assertCountEqual(
            output_names,
            [
                'session_a_1_classifier_c.png',
                'session_a_1_classifier_c_lineplot.png',
                'session_a_1_classifier_c_null.png',
                'session_a_1_classifier_c_null_lineplot.png',
            ],
        )

    def test_plot_only_skips_c_plots_for_legacy_cache(self):
        legacy_result = {
            'session': 'legacy_session',
            'cue': 1,
            'cue_deg': -135,
            'trial_idx': np.asarray([7]),
            'time_bins': np.asarray([0, 10]),
            'decoding_confidence': np.asarray([[0.6, 0.7]]),
            'decoding_predicted_labels': np.asarray([[[1, 1]]]),
            'decoding_accuracy_repeats': np.asarray([[1.0, 1.0]]),
            'decoding_test_labels': np.asarray([1]),
            'num_cells': 4,
        }
        with TemporaryDirectory() as temporary_directory:
            cache_dir = Path(temporary_directory)
            with open(cache_dir / 'decoding_confidence.pkl', 'wb') as handle:
                pickle.dump([legacy_result], handle)

            with (
                patch.object(
                    decoding_confidence,
                    'plot_decoding_heatmap',
                ) as plot_confidence_heatmap,
                patch.object(
                    decoding_confidence,
                    'plot_decoding_confidence_lineplot',
                ) as plot_confidence_line,
                patch.object(
                    decoding_confidence,
                    'plot_decoding_classifier_c',
                ) as plot_c,
                patch('builtins.print') as print_message,
            ):
                decoding_confidence.main(
                    decoding_confidence.Config(
                        cache_dir=cache_dir,
                        plot_only=True,
                    )
                )

        plot_confidence_heatmap.assert_called_once()
        plot_confidence_line.assert_called_once()
        plot_c.assert_not_called()
        self.assertIn(
            'does not contain decoder C values',
            ' '.join(str(call) for call in print_message.call_args_list),
        )


class ClassifierCCacheIntegrationTest(unittest.TestCase):
    def test_main_caches_and_plots_c_with_cue_shuffle_axes_aligned(self):
        rng = np.random.default_rng(3210)
        session = 'session_a'
        num_trials = 12
        num_cells = 4
        data = {
            'spks': rng.poisson(
                0.2,
                size=(num_trials, 10, num_cells),
            ).astype(np.float32),
            'cueAngIdx': np.asarray([5] * 6 + [1] * 6),
            'isCorr': np.ones(num_trials, dtype=np.bool_),
            'tc': np.arange(10, dtype=float),
        }
        selection = [{
            'session': session,
            'max_num_cells_per_group': num_cells,
            'trial_holdout': None,
            'trial_start': 0,
            'trial_end': num_trials,
            'cell_properties': {
                'cell_idx': np.arange(num_cells),
                'mean_pref_test': np.ones(num_cells, dtype=np.int64),
            },
        }]

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_dir = root / 'data'
            cache_dir = root / 'cache'
            data_dir.mkdir()
            cache_dir.mkdir()
            (data_dir / f'{session}.mat').touch()
            with open(cache_dir / 'cell_trial_selection.pkl', 'wb') as handle:
                pickle.dump(selection, handle)

            config = decoding_confidence.Config(
                n_jobs=1,
                par_verbose=0,
                seed=42,
                data_dir=data_dir,
                cache_dir=cache_dir,
                cells_used_for_decoder=(
                    decoding_confidence.CellsUsedForDecoder.ALL
                ),
                decoder_model=(
                    decoding_confidence.DecoderModel.LOGISTIC_REGRESSION
                ),
                svm_kernel=decoding_confidence.SVMKernel.LINEAR,
                classifier_c=0.1,
                grid_search_for_c=True,
                min_cell_per_group=1,
                min_trials_good_session=1,
                t_decode_start=0,
                t_decode_end=0,
                t_decode_window=10,
                t_decode_step=10,
                n_cue_preserved_trial_idx_shuffle=2,
                max_sessions_to_run=1,
            )

            with (
                patch.object(decoding_confidence, 'loadmat', return_value=data),
                patch('builtins.print'),
                patch.object(
                    decoding_confidence,
                    'select_classifier_c',
                    return_value=0.01,
                ) as select_c,
                patch.object(decoding_confidence, 'plot_decoding_heatmap'),
                patch.object(
                    decoding_confidence,
                    'plot_decoding_confidence_lineplot',
                ),
                patch.object(
                    decoding_confidence,
                    'plot_decoding_classifier_c',
                ) as plot_c,
            ):
                decoding_confidence.main(config)

            with open(cache_dir / 'decoding_confidence.pkl', 'rb') as handle:
                cached = pickle.load(handle)

        self.assertEqual(len(cached), 1)
        result = cached[0]
        self.assertEqual(
            result['decoding_classifier_c_repeats'].shape,
            result['decoding_confidence_repeats'].shape,
        )
        self.assertEqual(
            result['decoding_classifier_c_repeats'].shape,
            (6, 2, 1),
        )
        self.assertEqual(
            result['decoding_classifier_c_null'].shape,
            result['decoding_confidence_null'].shape,
        )
        self.assertEqual(result['decoding_classifier_c_null'].shape, (6, 1, 2))
        np.testing.assert_allclose(
            result['decoding_classifier_c_repeats'],
            0.01,
        )
        np.testing.assert_allclose(result['decoding_classifier_c_null'], 0.01)
        self.assertTrue(result['grid_search_for_c'])
        self.assertEqual(result['classifier_c_grid'], (1.0, 0.1, 0.01))
        self.assertEqual(result['classifier_c_grid_search_cv_folds'], 5)
        self.assertEqual(
            result['classifier_c_grid_search_scoring'],
            'balanced_accuracy',
        )
        self.assertEqual(select_c.call_count, 24)
        plot_c.assert_called_once()
        np.testing.assert_array_equal(
            plot_c.call_args.args[6],
            result['decoding_classifier_c_repeats'],
        )
        np.testing.assert_array_equal(
            plot_c.call_args.args[7],
            result['decoding_classifier_c_null'],
        )


if __name__ == '__main__':
    unittest.main()
