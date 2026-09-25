import dataclasses
from contextlib import ExitStack, redirect_stdout
import importlib
import io
import json
import pickle
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from joblib import Parallel, delayed

from scripts.next import cache_io
from scripts.next.cell_trial_selection import Config as SelectionConfig
from scripts.next.common import compute_binned_rates, fingerprint, full_session_selection, worker_context
from scripts.next.decoding_confidence import Config, decode_one_trial
from scripts.next.decoder_models import make_grouped_stratified_cv_splits
from scripts.next.eval_confidence import evaluate_session
from scripts.next.eval_confidence_across_runs import score_curve
from scripts.next.pipeline import Config as PipelineConfig, resolve_config


class CacheContractTest(unittest.TestCase):
    def test_resume_matches_settings_and_inputs_but_ignores_worker_count(self):
        from scripts.next import decoding_confidence as decoder
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'data'
            data.mkdir()
            source = data / 'session.mat'
            source.write_bytes(b'input version 1')
            cache = root / 'cache'
            cache_io.save([{'session': 'session', 'num_trials': 400, 'max_num_cells_per_group': 4}],
                          cache / 'cell_trial_selection.pkl')
            config = Config(data_dir=data, cache_dir=cache, save_figures=False)
            with patch.object(decoder, 'decode_session', return_value={'session': 'session'}) as fit:
                decoder.main(config)
                decoder.main(dataclasses.replace(config, n_jobs=2))
                self.assertEqual(fit.call_count, 1)
                source.write_bytes(b'input version 2')
                decoder.main(config)
                self.assertEqual(fit.call_count, 2)
                decoder.main(dataclasses.replace(config, n_decode_shuffle=2))
                self.assertEqual(fit.call_count, 3)

    def test_rejects_legacy_cache_and_roundtrips_new_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cell_trial_selection.pkl'
            path.write_bytes(pickle.dumps([{'session': 'old'}]))
            with self.assertRaisesRegex(ValueError, 'Incompatible cache'):
                cache_io.read(path)
            cache_io.save([{'session': 'new', 'num_trials': 400}], path)
            self.assertEqual(cache_io.read(path)[0]['session'], 'new')
            self.assertEqual(pickle.loads(path.read_bytes())['schema'], 'wm-states-next')

    def test_failed_atomic_write_preserves_existing_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'decoding_confidence.pkl'
            cache_io.save(['complete'], path)
            with patch.object(cache_io._pickle, 'dump', side_effect=RuntimeError('interrupted')):
                with self.assertRaises(RuntimeError):
                    cache_io.save(['partial'], path)
            self.assertEqual(cache_io.read(path), ['complete'])
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_full_session_selection_rejects_partial_or_duplicate_records(self):
        with self.assertRaises(ValueError):
            full_session_selection([{'session': 'a', 'trial_start': 0, 'trial_end': 50}], 'a', 100)
        with self.assertRaises(ValueError):
            full_session_selection([{'session': 'a', 'num_trials': 100}] * 2, 'a', 100)

    def test_removed_options_are_not_config_fields(self):
        forbidden = {'enable_trial_selection', 'trial_selection_window_size', 'trial_selection_step_size',
                     'n_jobs_partition', 'n_repeats_for_model_fit', 'cue_preserved_train_set_shuffle',
                     'n_cue_preserved_trial_idx_shuffle', 'use_decoding_estimates_from_subset_of_repeats',
                     'list_of_repeats', 'compare_with_repeat_idx',
                     'train_delay_decoder_using_all_delay_time_bins'}
        for name in ['cell_trial_selection', 'decoding_confidence', 'on_off_states', 'inspect_decoding_results']:
            module = importlib.import_module('scripts.next.' + name)
            self.assertFalse(forbidden & {f.name for f in dataclasses.fields(module.Config)})


class BinningTest(unittest.TestCase):
    def test_vectorized_pev_matches_scalar_omega_squared(self):
        from scripts.next.selection_math import pev_and_preferred_cue
        rng = np.random.default_rng(7)
        labels = np.tile(np.arange(1, 5), 8)
        rates = rng.normal(size=(32, 4, 3)) + labels[:, None, None]
        omega, preferred = pev_and_preferred_cue(rates, labels, np.arange(1, 5))
        for cell in range(3):
            for time_bin in range(4):
                sample = rates[:, time_bin, cell]
                total = ((sample - sample.mean()) ** 2).sum()
                groups = [sample[labels == cue] for cue in range(1, 5)]
                within = sum(((group - group.mean()) ** 2).sum() for group in groups)
                mse = within / 28
                expected = (total - within - 3 * mse) / (total + mse) * 100
                self.assertAlmostEqual(omega[cell, time_bin], expected)
                self.assertEqual(preferred[cell, time_bin], 1 + np.argmax([g.mean() for g in groups]))

    def test_prefix_sums_match_direct_sums_including_edge_windows(self):
        rng = np.random.default_rng(7)
        spikes = rng.poisson(0.5, (5, 210, 4))
        times = np.arange(-100, 320, 2)
        starts = np.array([-100, -91, 0, 15, 300])
        expected = np.stack([spikes[:, (times >= start) & (times < start + 50)].mean(axis=1) * 500
                             for start in starts], axis=1)
        np.testing.assert_allclose(compute_binned_rates(spikes, times, starts, 50), expected, rtol=1e-7)

    def test_invalid_bins_and_nonfinite_samples_are_errors(self):
        spikes = np.ones((2, 4, 3))
        with self.assertRaises(ValueError):
            compute_binned_rates(spikes, np.arange(4), [10], 2)
        spikes[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            compute_binned_rates(spikes, np.arange(4), [0], 2)


class StateTest(unittest.TestCase):
    def test_zero_variance_null_bins_are_unclassified(self):
        from scripts.next import on_off_states as states
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            source = {'session': 'example', 'cue': 1, 'trial_idx': [2, 4],
                      'time_bins': np.array([500, 550, 600]),
                      'decoding_confidence': np.full((2, 3), 0.8),
                      'decoding_confidence_null': np.full((2, 3, 3), 0.5)}
            cache_io.save([source], cache / 'decoding_confidence.pkl')
            with patch.object(states, 'save_figure_all_formats'):
                states.main(states.Config(cache_dir=cache, cluster_size_threshold_off=1))
            result = cache_io.read(cache / 'on_off_states.pkl')[0]
            self.assertFalse(result['on_state_mask'].any())
            self.assertFalse(result['off_state_mask'].any())
            np.testing.assert_array_equal(result['off_state_duration_per_trial'], [0, 0])

    def test_no_off_states_produces_empty_maximum_mask(self):
        from scripts.next.compare_activity_across_states import maximum_delay_off_state_mask
        result = maximum_delay_off_state_mask(np.zeros((3, 4), dtype=bool), np.ones(4, dtype=bool))
        self.assertEqual(result.shape, (3, 4))
        self.assertFalse(result.any())


class DecoderTest(unittest.TestCase):
    def setUp(self):
        self.labels = np.tile([0, 1], 10)
        rng = np.random.default_rng(7)
        self.rates = rng.normal(size=(20, 3, 4)) + self.labels[:, None, None]
        self.times = np.array([0, 500, 550])
        self.config = Config(n_decode_shuffle=3, logistic_calibration_method='none', min_trials_good_session=1)

    def test_shapes_finite_values_and_shuffle_prefix_invariance(self):
        result = decode_one_trial(1, self.rates, self.labels, self.times, self.config)
        self.assertEqual(result[0].shape, (3,))
        self.assertEqual(result[1].shape, (3,))
        self.assertEqual(result[3].shape, (3, 3))
        self.assertTrue(np.all((result[0] >= 0) & (result[0] <= 1)))
        larger = decode_one_trial(1, self.rates, self.labels, self.times,
                                  dataclasses.replace(self.config, n_decode_shuffle=5))
        np.testing.assert_array_equal(result[0], larger[0])
        np.testing.assert_array_equal(result[3], larger[3][:, :3])
        zero = decode_one_trial(1, self.rates, self.labels, self.times,
                                dataclasses.replace(self.config, n_decode_shuffle=0))
        np.testing.assert_array_equal(result[0], zero[0])
        self.assertEqual(zero[3].shape, (3, 0))

    def test_parallelism_does_not_change_estimates(self):
        serial = [decode_one_trial(i, self.rates, self.labels, self.times, self.config) for i in (1, 3)]
        with worker_context(2):
            parallel = Parallel()(delayed(decode_one_trial)(i, self.rates, self.labels, self.times, self.config) for i in (1, 3))
        for left, right in zip(serial, parallel):
            for a, b in zip(left[:5], right[:5]):
                np.testing.assert_array_equal(a, b)

    def test_fits_use_only_training_trials_and_the_current_bin(self):
        fits = []
        test_samples = []
        class Recorder:
            classes_ = np.array([0, 1])
            def fit(self, x, y):
                fits.append((x.copy(), y.copy()))
                return self
            def predict_proba(self, x):
                test_samples.append(x.copy())
                return np.tile([0.4, 0.6], (len(x), 1))
            def predict(self, x):
                return np.ones(len(x), dtype=int)
        rates = np.empty((20, 3, 2))
        rates[:, :, 0] = np.arange(20)[:, None]
        rates[:, :, 1] = np.arange(3)[None, :]
        with patch('scripts.next.decoding_confidence.create_base_decoder', side_effect=lambda *a: Recorder()):
            decode_one_trial(1, rates, self.labels, self.times, self.config)
        self.assertEqual(len(fits), 3 * (1 + self.config.n_decode_shuffle))
        for index, ((x, y), test_sample) in enumerate(zip(fits, test_samples)):
            self.assertNotIn(1, x[:, 0])
            self.assertEqual(len(x), 18)
            self.assertEqual(np.unique(x[:, 0]).size, len(x))
            np.testing.assert_array_equal(x[:, 0], fits[0][0][:, 0])
            np.testing.assert_array_equal(x[:, 1], np.full(len(x), index % 3))
            np.testing.assert_array_equal(test_sample, [[1, index % 3]])
            np.testing.assert_array_equal(np.bincount(y), [9, 9])
            if index < 3:
                np.testing.assert_array_equal(y, self.labels[x[:, 0].astype(int)])

    def test_grid_search_and_calibration_work_per_bin(self):
        config = dataclasses.replace(self.config, n_decode_shuffle=1, grid_search_for_c=True,
            logistic_calibration_method='sigmoid')
        result = decode_one_trial(1, self.rates, self.labels, self.times, config)
        self.assertTrue(np.all(np.isfinite(result[0])))
        self.assertTrue(np.all(np.isin(result[2], [1, 0.1, 0.01])))
        self.assertTrue(np.all(np.isin(result[4], [1, 0.1, 0.01])))
        self.assertEqual(result[-1], (5,))

    def test_optimized_c_search_agrees_with_sklearn_grid_search(self):
        from sklearn.model_selection import GridSearchCV
        from scripts.next.decoder_models import create_base_decoder, select_classifier_c
        x = self.rates[:, 0]
        groups = np.arange(len(x))
        splits, _ = make_grouped_stratified_cv_splits(self.labels, groups, 5, 42,
            purpose='Test', allow_fold_reduction=False)
        for model in ('svm', 'logistic_regression'):
            for kernel in ('linear', 'rbf'):
                for y in (self.labels, np.random.default_rng(19).permutation(self.labels)):
                    folds, _ = make_grouped_stratified_cv_splits(y, groups, 5, 42,
                        purpose='Test', allow_fold_reduction=False)
                    reference = GridSearchCV(create_base_decoder(1, model, kernel, 42, svm_probability=False),
                        {'classifier__C': [1, .1, .01]}, cv=folds, scoring='balanced_accuracy', refit=False)
                    reference.fit(x, y)
                    selected = select_classifier_c(x, y, groups, model, kernel, 42)
                    self.assertEqual(selected, reference.best_params_['classifier__C'])

    def test_grouped_folds_never_split_source_trials(self):
        labels = self.labels
        groups = np.arange(self.labels.size)
        splits, count = make_grouped_stratified_cv_splits(labels, groups, 5, 42,
            purpose='Test', allow_fold_reduction=False)
        self.assertEqual(count, 5)
        for train, test in splits:
            self.assertFalse(set(groups[train]) & set(groups[test]))
            self.assertEqual(set(labels[train]), {0, 1})
            self.assertEqual(set(labels[test]), {0, 1})


class EvaluationTest(unittest.TestCase):
    def test_observed_and_null_scores_and_cross_run_curve(self):
        source = {'session': 'example', 'cue': 1, 'trial_idx': [3, 5], 'time_bins': [0, 50],
                  'decoding_test_labels': np.ones(2),
                  'decoding_confidence': np.array([[0.8, 0.6], [0.4, 1.0]]),
                  'decoding_predicted_labels': np.array([[1, 1], [0, 1]]),
                  'decoding_confidence_null': np.full((2, 2, 3), 0.5)}
        result = evaluate_session(source)
        self.assertAlmostEqual(result['observed']['brier_score'], 0.14)
        self.assertEqual(result['observed']['accuracy'], 0.75)
        self.assertEqual(result['num_null_shuffles'], 3)
        curve, band = score_curve(result, 'observed', 'accuracy', 2)
        np.testing.assert_array_equal(curve, [0.5, 1])
        self.assertIsNone(band)


class RunnerTest(unittest.TestCase):
    def test_presets_dispatch_requested_stages_and_record_manifest(self):
        from scripts.next import pipeline
        core = ['select', 'decode', 'evaluate', 'states', 'activity']
        mixed = ['prepare', 'models', 'nested-count', 'nested-activity', 'criticality', 'interactions']
        config_dir = Path(__file__).resolve().parents[2] / 'configs' / 'next'
        for preset, null_count in [('example_pipeline.json', 100), ('smoke_pipeline.json', 3)]:
            for requested, expected in [(None, core), (('mixed',), mixed), (('all',), core + mixed)]:
                with self.subTest(preset=preset, stages=requested), tempfile.TemporaryDirectory() as directory:
                    cache = Path(directory) / 'cache'
                    config = PipelineConfig(settings=config_dir / preset, cache_dir=cache)
                    if requested is not None:
                        config = dataclasses.replace(config, stages=requested)
                    calls = []
                    with ExitStack() as stack:
                        stack.enter_context(redirect_stdout(io.StringIO()))
                        stack.enter_context(patch.dict('os.environ'))
                        for stage, name in pipeline.STAGES.items():
                            module = importlib.import_module('scripts.next.' + name)
                            stack.enter_context(patch.object(module, 'main',
                                side_effect=lambda stage_config, stage=stage: calls.append(stage)))
                        pipeline.main(config)
                    self.assertEqual(calls, expected)
                    manifest = json.loads((cache / 'pipeline_manifest.json').read_text())
                    self.assertEqual(list(manifest['settings']), expected)
                    self.assertEqual([item['stage'] for item in manifest['stages']], expected)
                    self.assertTrue(all(item['status'] == 'complete' for item in manifest['stages']))
                    if 'decode' in expected:
                        self.assertEqual(manifest['settings']['decode']['n_decode_shuffle'], null_count)

    def test_removed_stages_are_rejected_before_writing_outputs(self):
        from scripts.next import pipeline
        for stage in ('baseline', 'cell-count'):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                cache = Path(directory) / 'cache'
                with self.assertRaisesRegex(ValueError, f'Unknown stages:.*{stage}'):
                    pipeline.main(PipelineConfig(cache_dir=cache, stages=(stage,)))
                self.assertFalse(cache.exists())

    def test_removed_settings_stages_are_rejected_even_when_not_requested(self):
        from scripts.next import pipeline
        for stage in ('baseline', 'cell-count'):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                settings = root / 'settings.json'
                settings.write_text(json.dumps({stage: {}}))
                cache = root / 'cache'
                with self.assertRaisesRegex(ValueError, f'Unknown settings stages:.*{stage}'):
                    pipeline.main(PipelineConfig(cache_dir=cache, settings=settings, stages=('evaluate',)))
                self.assertFalse(cache.exists())

    def test_integer_and_float_c_settings_share_a_checkpoint_key(self):
        self.assertEqual(fingerprint(Config(classifier_c=1)), fingerprint(Config(classifier_c=1.0)))

    def test_typed_json_settings_and_removed_flags_rejected(self):
        module = importlib.import_module('scripts.next.decoding_confidence')
        result = resolve_config(module, {'cache_dir': 'cache/example'}, {'decoder_model': 'LOGISTIC_REGRESSION'})
        self.assertEqual(result.cache_dir, Path('cache/example'))
        self.assertEqual(result.decoder_model.value, 'logistic_regression')
        for key in ('n_repeats_for_model_fit', 'train_delay_decoder_using_all_delay_time_bins'):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'unknown settings'):
                resolve_config(module, {}, {key: 2})


if __name__ == '__main__':
    unittest.main()
