"""Controls for the dated, read-only completed-run comparison."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import re
import shlex
import tempfile
import unittest

import numpy as np
import tyro

from scripts.next import compare_statistical_choices as comparison
from scripts.next import decoding_confidence as decoder
from scripts.next.common import json_value
from scripts.next.pipeline import resolve_config


class StatisticalChoicesComparisonTest(unittest.TestCase):
    def test_historical_balance_is_translated_without_adopting_new_defaults(self):
        for previous, mode in ((True, 'balanced_training_trials'), (False, 'none')):
            with self.subTest(previous=previous):
                old = {'balance_decoder_training_trials': previous, 'seed': 42,
                       'n_jobs': 10, 'cache_dir': 'old'}
                new = {'training_balance': mode, 'seed': 42, 'n_jobs': 4, 'cache_dir': 'new'}
                self.assertEqual(comparison.scientific_settings(old), comparison.scientific_settings(new))
                self.assertIn('balance_decoder_training_trials', old)
        for invalid in ({'seed': 42}, {'balance_decoder_training_trials': 'false'},
                        {'balance_decoder_training_trials': True,
                         'training_balance': 'balanced_class_weights'}):
            with self.subTest(invalid=invalid), self.assertRaises(AssertionError):
                comparison.scientific_settings(invalid)

    def test_real_default_decoder_command_matches_template_scientific_settings(self):
        text = (comparison.ROOT / 'docs/next/pipeline.md').read_text()
        command = re.search(r'uv run python scripts/next/decoding_confidence\.py \\\n(?:.*\\\n)*.*', text)
        self.assertIsNotNone(command)
        args = shlex.split(command.group().replace('\\\n', ' '))[4:]
        cli = tyro.cli(decoder.Config, args=args)
        preset = json.loads((comparison.ROOT / 'configs/next/default_pipeline.json').read_text())
        template = resolve_config(decoder, {}, preset['decode'])
        def settings(config):
            return comparison.scientific_settings(json.loads(json.dumps(asdict(config), default=json_value)))
        self.assertEqual(settings(cli), settings(template))
        self.assertEqual(cli.training_balance.value, 'balanced_class_weights')

    def test_fitting_manifest_is_distinct_from_latest_plot_and_state_invocation(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            manifests = folder / 'manifests'
            manifests.mkdir()
            config = {'balance_decoder_training_trials': True, 'seed': 42, 'plot_only': False}
            state_config = {'z_threshold_off': .842, 'cache_dir': 'cache/old', 'off_duration_xmax': 500}
            def save(name, decode_config, stages):
                manifest = {'run_id': name, 'settings': {'decode': decode_config, 'states': state_config},
                            'stages': stages}
                (manifests / f'{name}.json').write_text(json.dumps(manifest))
            save('001-fit', config, [{'stage': 'decode', 'status': 'complete', 'seconds': 100},
                                     {'stage': 'states', 'status': 'complete'}])
            save('002-plot', {**config, 'plot_only': True},
                 [{'stage': 'decode', 'status': 'complete', 'seconds': 2},
                  {'stage': 'states', 'status': 'complete'}])
            save('003-failed', config, [{'stage': 'decode', 'status': 'failed'},
                                       {'stage': 'states', 'status': 'pending'}])
            cached = {'training_balance': 'balanced_training_trials', 'seed': 42, 'plot_only': False}
            fit_path, fit, state_path, state, settings = comparison.run_manifests(folder, cached)
            self.assertEqual(fit_path.name, '001-fit.json')
            self.assertEqual(comparison.completed_stage(fit, 'decode')['seconds'], 100)
            self.assertEqual(state_path.name, '002-plot.json')
            self.assertEqual(state['run_id'], '002-plot')
            self.assertEqual(settings, {'z_threshold_off': .842})
            with self.assertRaises(AssertionError):
                comparison.run_manifests(folder, {**cached, 'seed': 43})
            save('004-second-fit', config, [{'stage': 'decode', 'status': 'complete', 'seconds': 90}])
            with self.assertRaisesRegex(AssertionError, 'actual fitting invocation'):
                comparison.run_manifests(folder, cached)

    def test_completed_decode_without_completed_states_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'manifests').mkdir()
            config = {'training_balance': 'none', 'plot_only': False}
            manifest = {'settings': {'decode': config},
                        'stages': [{'stage': 'decode', 'status': 'complete'},
                                   {'stage': 'states', 'status': 'failed'}]}
            (folder / 'manifests/001.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(AssertionError, 'no completed state'):
                comparison.run_manifests(folder, config)

    @staticmethod
    def contrast_fixture():
        row = {'trial_idx': np.array([136, 150]), 'cell_idx': np.array([3, 9]),
               'times': np.array([500, 510]), 'cue': 1, 'p': np.full((2, 2), .6),
               'C': np.zeros((2, 2), dtype=np.uint8), 'C_null': np.zeros((2, 2, 3), dtype=np.uint8),
               'max_off': np.array([100., 200.]), 'off_mask': np.ones((2, 2), dtype=bool),
               'native_prediction': np.ones((2, 2))}
        summary = {'config': {'balance_decoder_training_trials': True, 'seed': 42},
                   'scientific_config': {'training_balance': 'balanced_training_trials', 'seed': 42},
                   'state_config': {'z_threshold_off': .842},
                   'aggregate': {'observed_delay': {'brier': .16, 'log_loss': .51}},
                   'per_session': {'s': {'observed_delay': {'brier': .16, 'log_loss': .51}}}}
        weighted = deepcopy(summary)
        weighted['config'] = {'training_balance': 'balanced_class_weights', 'seed': 42}
        weighted['scientific_config']['training_balance'] = 'balanced_class_weights'
        weighted['aggregate']['observed_delay'] = {'brier': .09, 'log_loss': .36}
        weighted['per_session']['s']['observed_delay'] = {'brier': .09, 'log_loss': .36}
        other_row = deepcopy(row)
        other_row['max_off'] = np.array([130., 130.])
        other_row['p'][:] = .7
        return {'a': summary, 'b': weighted}, {'a': {'s': row}, 'b': {'s': other_row}}

    def test_contrast_retains_raw_migration_and_paired_metric_direction(self):
        runs, datasets = self.contrast_fixture()
        result = comparison.compare_contrast('a', 'b', {'training_balance'}, 'weights', runs, datasets)
        self.assertTrue(result['structural_alignment_verified'])
        self.assertTrue(result['state_config_identical'])
        self.assertEqual(set(result['scientific_config_differences']), {'training_balance'})
        self.assertEqual(set(result['cached_config_differences']),
                         {'training_balance', 'balance_decoder_training_trials'})
        self.assertEqual(result['max_off_other_longer_count'], 1)
        self.assertEqual(result['max_off_other_shorter_count'], 1)
        self.assertEqual(result['max_off_mean_abs_change_ms'], 50)
        self.assertAlmostEqual(result['other_minus_baseline_brier_delay'], -.07)
        self.assertAlmostEqual(result['other_minus_baseline_log_loss_delay'], -.15)
        self.assertEqual(result['sessions_other_has_lower_preferred_only_brier'], 1)
        self.assertEqual(result['sessions_other_has_lower_preferred_only_log_loss'], 1)

    def test_uncontrolled_settings_or_population_differences_fail_comparison(self):
        for field in ('seed', 'state_threshold', 'sessions', 'trial_idx', 'cell_idx', 'times', 'cue'):
            runs, datasets = self.contrast_fixture()
            if field == 'seed':
                runs['b']['scientific_config']['seed'] = 43
            elif field == 'state_threshold':
                runs['b']['state_config']['z_threshold_off'] = 1
            elif field == 'sessions':
                datasets['b']['other'] = datasets['b'].pop('s')
            elif field == 'cue':
                datasets['b']['s']['cue'] = 2
            else:
                datasets['b']['s'][field][0] += 1
            with self.subTest(field=field), self.assertRaises(AssertionError):
                comparison.compare_contrast('a', 'b', {'training_balance'}, 'weights', runs, datasets)

    def test_factorial_contrasts_keep_calibration_and_C_effects_distinct(self):
        # Hand-chosen losses: calibration helps more at fixed C; C search helps
        # more without calibration. The fifth (weighted) run is not a 2x2 cell.
        runs = {run: {'aggregate': {'observed_delay': {'brier': value, 'log_loss': 2 * value}}}
                for run, value in zip(comparison.RUNS, (.1, .3, .2, .6, 99))}
        effects = comparison.calibration_c_effects(runs)
        expected = {'calibration_minus_none_with_C_search': -.2,
                    'calibration_minus_none_with_fixed_C': -.4,
                    'C_search_minus_fixed_with_calibration': -.1,
                    'C_search_minus_fixed_without_calibration': -.3,
                    'difference_of_calibration_effects_search_minus_fixed': .2}
        for name, value in expected.items():
            self.assertAlmostEqual(effects['brier'][name], value)
            self.assertAlmostEqual(effects['log_loss'][name], 2 * value)


if __name__ == '__main__':
    unittest.main()
