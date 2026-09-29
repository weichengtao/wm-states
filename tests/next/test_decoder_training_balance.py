"""Scientific regression checks for training membership, fold weights and calibration."""
from dataclasses import asdict, replace
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from joblib import Parallel, delayed
from sklearn.calibration import _SigmoidCalibration
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

from scripts.next import decoding_confidence as decoding
from scripts.next.common import fingerprint, worker_context
from scripts.next.decoder_models import (
    TrainingBalance, create_base_decoder, fit_calibrated_decoder,
    make_logistic_calibration_cv_splits, select_classifier_c,
)
from scripts.next.pipeline import resolve_config


class TrainingBalanceTests(unittest.TestCase):
    def setUp(self):
        self.y = np.array([1] * 12 + [0] * 17)
        self.x = np.random.default_rng(150).normal(size=(29, 2, 4))
        self.x[:, :, 0] += self.y[:, None] * .7
        self.times = np.array([500, 510])
        self.config = decoding.Config(training_balance='balanced_class_weights',
            n_decode_shuffle=2, grid_search_for_c=True)

    def test_membership_all_modes_and_seeds(self):
        for mode in TrainingBalance:
            selections = [decoding.training_trials(self.y, 0, seed, mode) for seed in (42, 43)]
            for train in selections:
                self.assertNotIn(0, train)
                self.assertEqual(len(train), len(set(train)))
                counts = np.bincount(self.y[train])
                if mode is TrainingBalance.BALANCED_TRAINING_TRIALS:
                    np.testing.assert_array_equal(counts, [11, 11])
                else:
                    np.testing.assert_array_equal(counts, [17, 11])
                    np.testing.assert_array_equal(train, np.arange(1, 29))
            if mode is TrainingBalance.BALANCED_TRAINING_TRIALS:
                self.assertFalse(np.array_equal(*selections))

    def reference(self, x, y, target, model_name, method='sigmoid', weighted=True):
        """Independent public estimators, manual OOF loop, and calibrator fit."""
        folds, _ = make_logistic_calibration_cv_splits(y, np.arange(len(y)), 5, 42)
        def base(c=1):
            kwargs = dict(C=c, random_state=42, class_weight='balanced' if weighted else None)
            classifier = (LogisticRegression(solver='liblinear', max_iter=1000, **kwargs)
                          if model_name == 'logistic_regression' else SVC(kernel='linear', **kwargs))
            return make_pipeline(StandardScaler(), classifier)
        step = 'logisticregression' if model_name == 'logistic_regression' else 'svc'
        search = GridSearchCV(base(), {f'{step}__C': [1, .1, .01]},
                              cv=folds, scoring='balanced_accuracy', refit=True)
        search.fit(x, y)
        c = search.best_params_[f'{step}__C']
        margins = np.empty(len(y))
        for train, validation in folds:
            margins[validation] = base(c).fit(x[train], y[train]).decision_function(x[validation])
        weights = compute_sample_weight('balanced', y) if weighted else None
        calibrator = (_SigmoidCalibration() if method == 'sigmoid'
                      else IsotonicRegression(out_of_bounds='clip'))
        calibrator.fit(margins, y, sample_weight=weights)
        probability = calibrator.predict(search.best_estimator_.decision_function(target))[0]
        return probability, c

    def test_full_observed_and_null_path_matches_independent_reference(self):
        for model in ('logistic_regression', 'svm'):
            for preserved in (False, True):
                config = replace(self.config, decoder_model=model, preserve_null_time_structure=preserved)
                result = decoding.decode_one_trial(0, self.x, self.y, self.times, config)
                for estimate in range(3):
                    for b in range(2):
                        y = self.y[1:]
                        if estimate:
                            rng = np.random.default_rng(np.random.SeedSequence([42, 0, 1, estimate, 0 if preserved else b]))
                            y = rng.permutation(y)
                        expected, c = self.reference(self.x[1:, b], y, self.x[0:1, b], model)
                        actual = result[0][b] if estimate == 0 else result[3][b, estimate - 1]
                        actual_c = result[2][b] if estimate == 0 else result[4][b, estimate - 1]
                        self.assertAlmostEqual(float(actual), expected, places=6)
                        self.assertEqual(actual_c, c)
                self.assertEqual(result[-1], (5,))

    def test_isotonic_and_uncalibrated_logistic_weighted_modes(self):
        config = replace(self.config, logistic_calibration_method='isotonic', n_decode_shuffle=0)
        result = decoding.decode_one_trial(0, self.x, self.y, self.times, config)
        expected, _ = self.reference(self.x[1:, 0], self.y[1:], self.x[0:1, 0],
                                      'logistic_regression', method='isotonic')
        self.assertAlmostEqual(float(result[0][0]), expected, places=6)
        config = replace(config, logistic_calibration_method='none', grid_search_for_c=False)
        result = decoding.decode_one_trial(0, self.x, self.y, self.times, config)
        expected = make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear',
            class_weight='balanced', random_state=42, max_iter=1000)).fit(self.x[1:, 0], self.y[1:])
        self.assertAlmostEqual(float(result[0][0]), expected.predict_proba(self.x[:1, 0])[0, 1], places=6)
        self.assertEqual(result[-1], ())

    def test_calibration_prior_and_no_double_weighting_or_global_fold_weights(self):
        x = np.zeros((len(self.y), 1))
        folds, _ = make_logistic_calibration_cv_splits(self.y, np.arange(len(self.y)), 5, 42)
        records = []
        original = LogisticRegression.fit
        def fit(model, x, y, sample_weight=None):
            records.append((len(y), np.bincount(y), model.class_weight, sample_weight))
            return original(model, x, y, sample_weight=sample_weight)
        with patch.object(LogisticRegression, 'fit', fit):
            model = fit_calibrated_decoder(create_base_decoder(1, 'logistic_regression', 'linear', 42,
                class_weight='balanced'), x, self.y, 'isotonic', folds, balanced_class_weights=True)
        self.assertEqual(len(records), 6)
        for (_, counts, class_weight, supplied), (train, _) in zip(records, [*folds, (np.arange(len(x)), None)]):
            np.testing.assert_array_equal(counts, np.bincount(self.y[train]))
            self.assertEqual(class_weight, 'balanced')
            self.assertIsNone(supplied)
            w = compute_sample_weight(class_weight, self.y[train])
            self.assertAlmostEqual(w[self.y[train] == 0].sum(), w[self.y[train] == 1].sum())
        self.assertAlmostEqual(model.predict_proba([[0]])[0, 1], .5)
        unweighted = fit_calibrated_decoder(create_base_decoder(1, 'logistic_regression', 'linear', 42),
            x, self.y, 'isotonic', folds)
        self.assertAlmostEqual(unweighted.predict_proba([[0]])[0, 1], np.mean(self.y))

    def test_c_search_matches_gridsearch_for_both_models_and_kernels(self):
        x, y = self.x[1:, 0], self.y[1:]
        for model in ('logistic_regression', 'svm'):
            for kernel in ('linear', 'rbf'):
                for weight in (None, 'balanced'):
                    for labels in (y, np.random.default_rng(9).permutation(y)):
                        folds, _ = make_logistic_calibration_cv_splits(labels, np.arange(len(y)), 5, 42)
                        reference = GridSearchCV(create_base_decoder(1, model, kernel, 42,
                            class_weight=weight, svm_probability=False), {'classifier__C': [1, .1, .01]},
                            cv=folds, scoring='balanced_accuracy', refit=False).fit(x, labels)
                        selected = select_classifier_c(x, labels, np.arange(len(y)), model, kernel, 42,
                            class_weight=weight)
                        self.assertEqual(selected, reference.best_params_['classifier__C'])

    def test_holdout_activity_cannot_change_training_or_fitted_calibration(self):
        records = []
        original = decoding.fit_calibrated_decoder
        def fit(base, x, y, method, splits, **kwargs):
            model = original(base, x, y, method, splits, **kwargs)
            records.append((x.copy(), y.copy(), model.predict_proba(x)))
            return model
        changed = self.x.copy()
        changed[0] = 1e6
        with patch.object(decoding, 'fit_calibrated_decoder', side_effect=fit):
            first = decoding.decode_one_trial(0, self.x, self.y, self.times, self.config)
            second = decoding.decode_one_trial(0, changed, self.y, self.times, self.config)
        for before, after in zip(records[:6], records[6:]):
            for a, b in zip(before, after):
                np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(first[2], second[2])
        np.testing.assert_array_equal(first[4], second[4])

    def test_seed_prefix_parallel_and_preserved_time_reproducibility(self):
        x = np.repeat(self.x[:, :1], 2, axis=1)
        config = replace(self.config, preserve_null_time_structure=True)
        serial = decoding.decode_one_trial(0, x, self.y, self.times, config)
        np.testing.assert_array_equal(serial[3][0], serial[3][1])
        empty = decoding.decode_one_trial(0, x, self.y, self.times, replace(config, n_decode_shuffle=0))
        np.testing.assert_array_equal(serial[0], empty[0])
        with worker_context(2):
            parallel = Parallel()(delayed(decoding.decode_one_trial)(0, x, self.y, self.times, config) for _ in range(2))
        for result in parallel:
            for a, b in zip(serial, result):
                np.testing.assert_array_equal(a, b)

    def test_json_migration_cli_choices_and_distinct_fingerprints(self):
        import tyro
        for previous, mode in [(True, 'balanced_training_trials'), (False, 'none')]:
            config = resolve_config(decoding, {}, {'balance_decoder_training_trials': previous})
            self.assertEqual(config.training_balance.value, mode)
            self.assertNotIn('balance_decoder_training_trials', asdict(config))
        for mode in TrainingBalance:
            cli = tyro.cli(decoding.Config, args=['--training-balance', mode.name])
            self.assertEqual(cli.training_balance, mode)
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            resolve_config(decoding, {}, {'balance_decoder_training_trials': False,
                                         'training_balance': 'BALANCED_CLASS_WEIGHTS'})
        for invalid in (False, True, 'unknown'):
            with self.assertRaises(ValueError):
                decoding.Config(training_balance=invalid)
        keys = {fingerprint(replace(self.config, training_balance=mode)) for mode in TrainingBalance}
        self.assertEqual(len(keys), 3)

    def test_weighted_svm_checks_calibration_class_counts(self):
        config = replace(self.config, decoder_model='svm', grid_search_for_c=False)
        with self.assertRaisesRegex(ValueError, '3 correct.*2 correct'):
            decoding.validate_training_class_counts(np.array([1, 1, 0, 0]), config, context='SVM')
        with self.assertWarnsRegex(RuntimeWarning, 'weighted SVM calibration from 5 to 2'):
            decoding.validate_training_class_counts(np.array([1, 1, 1, 0, 0]), config, context='SVM')

    def test_default_template_runs_selection_through_states_on_imbalanced_recording(self):
        from scipy.io import savemat
        from scripts.next import cache_io, cell_screening, on_off_states, pipeline
        from scripts.next.common import validate_state_provenance
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'data'
            data.mkdir()
            times = np.arange(-500, 1501, 10)
            rng = np.random.default_rng(17)
            spikes = rng.poisson(.2, size=(len(self.y), len(times), 3)).astype(float)
            spikes[self.y == 1, :, 0] += 1
            spikes[self.y == 1, :, 1] += 1
            savemat(data / 'session.mat', {'spks': spikes, 'tc': times,
                'cueAngIdx': np.where(self.y == 1, 1, 5), 'isCorr': np.ones(len(self.y))})
            settings = json.loads((Path(__file__).resolve().parents[2] / 'configs/next/default_pipeline.json').read_text())
            settings['select'].update({f'check_{name}': False for name in ('min_trials', *cell_screening.CHECK_NAMES)})
            settings['select']['selectivity_bin_step_ms'] = 50
            settings['decode'].update(min_trials_good_session=1, n_decode_shuffle=4,
                t_decode_start=500, t_decode_end=1400, t_decode_step=450,
                preserve_null_time_structure=True, save_figures=False)
            path = root / 'settings.json'
            path.write_text(json.dumps(settings))
            cache = root / 'cache'
            with redirect_stdout(io.StringIO()), patch.object(on_off_states, 'save_figure'):
                pipeline.main(pipeline.Config(data_dir=data, cache_dir=cache, settings=path,
                    stages=('select', 'decode', 'evaluate', 'states'), n_jobs=2))
            decoded = cache_io.read(cache / 'decode/decoding_confidence.pkl')[0]
            evaluated = cache_io.read(cache / 'evaluate/eval_confidence.pkl')[0]
            states = cache_io.read(cache / 'states/on_off_states.pkl')
            self.assertEqual(decoded['training_balance'], 'balanced_class_weights')
            self.assertEqual(decoded['probability_calibration_method'], 'sigmoid')
            self.assertEqual(decoded['calibration_effective_cv_folds'], [5])
            np.testing.assert_array_equal(decoded['decoding_classifier_c'], 0.01)
            np.testing.assert_array_equal(decoded['decoding_classifier_c_null'], 0.01)
            self.assertEqual(decoded['decoding_confidence_null'].shape, (12, 3, 4))
            self.assertEqual(evaluated['training_balance'], 'balanced_class_weights')
            self.assertEqual(states[0]['decoding_training_balance'], 'balanced_class_weights')
            self.assertTrue(np.all(np.isfinite(decoded['decoding_confidence_null'])))
            validate_state_provenance(states, cache, data)
            manifest = json.loads((cache / 'pipeline_manifest.json').read_text())
            self.assertTrue(all(stage['status'] == 'complete' for stage in manifest['stages']))
            self.assertEqual(manifest['settings']['decode']['training_balance'], 'balanced_class_weights')
            self.assertEqual(manifest['settings']['decode']['classifier_c'], 0.01)
            self.assertFalse(manifest['settings']['decode']['grid_search_for_c'])


if __name__ == '__main__':
    unittest.main()
