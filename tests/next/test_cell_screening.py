import dataclasses
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import tyro

from scripts.next import cell_screening as screening
from scripts.next import screening_checks as checks
from scripts.next.pipeline import resolve_config


class ScreeningConfigTest(unittest.TestCase):
    def test_cli_can_enable_and_disable_every_check(self):
        for name in ('min_trials', *screening.CHECK_NAMES):
            flag = name.replace('_', '-')
            with self.subTest(check=name):
                enabled = tyro.cli(screening.Config, args=[f'--check-{flag}'])
                disabled = tyro.cli(screening.Config, args=[f'--no-check-{flag}'])
                self.assertTrue(getattr(enabled, f'check_{name}'))
                self.assertFalse(getattr(disabled, f'check_{name}'))
        config = tyro.cli(screening.Config, args=[
            '--check-preferred-drift', '--temp-dep-r-threshold', '0.6',
            '--no-check-baseline-drift', '--presence-start', '-300', '--presence-end', '1200',
        ])
        self.assertEqual(config.temp_dep_r_threshold, .6)
        self.assertFalse(config.check_baseline_drift)
        self.assertEqual((config.presence_start, config.presence_end), (-300, 1200))

    def test_invalid_thresholds_are_rejected_even_for_disabled_checks(self):
        invalid = {
            'min_fr_test': [-1, np.nan, np.inf],
            'min_presence_ratio': [-.1, 1.1, np.nan],
            'var_ratio_threshold_delay_over_baseline': [-1, np.inf],
            'var_ratio_threshold_sliding_over_all': [-1, np.nan],
            'temp_dep_r_threshold': [2, -.1, np.nan],
            'temp_dep_r_threshold_baseline': [2, np.inf],
            'sig_pev_threshold': [-1, 101], 'pev_clip_at': [-1, 101],
        }
        for field, values in invalid.items():
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, field):
                    screening.Config(**{field: value})

    def test_invalid_counts_windows_and_json_switches_are_rejected(self):
        for fields in [dict(min_trial_per_session=0), dict(min_trial_for_temp_check=1),
                       dict(sig_pev_duration=0), dict(t_test_step=0), dict(t_test_window=-1),
                       dict(presence_end=-400), dict(baseline_drift_end=-500),
                       dict(temp_check_delay_end=500), dict(t_test_start=np.nan),
                       dict(check_firing_rate='false'), dict(n_jobs_session=0),
                       dict(max_sessions_to_run=0)]:
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                screening.Config(**fields)

    def test_presets_use_explicit_switches_and_resolve_all_stages(self):
        from scripts.next.pipeline import STAGES
        for path in Path('configs/next').glob('*pipeline.json'):
            settings = json.loads(path.read_text())
            for stage, module in STAGES.items():
                resolve_config(importlib.import_module(f'scripts.next.{module}'), {}, settings.get(stage, {}))
            resolved = resolve_config(screening, {}, settings['select'])
            for name in ('min_trials', *screening.CHECK_NAMES):
                self.assertIn(f'check_{name}', settings['select'])
                self.assertEqual(getattr(resolved, f'check_{name}'),
                                 name in ('min_trials', 'presence_ratio', 'baseline_drift', 'selectivity'))

    def test_removed_umbrella_and_unused_settings_fail_clearly(self):
        for key in ('temp_dep_detection', 'min_cell_per_group', 'seed'):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'unknown settings'):
                resolve_config(screening, {}, {key: 1})


class CellScreeningTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(13)
        self.times = np.arange(-500, 1501, 50)
        self.cues = np.tile([1, 5], 15)
        self.correct = np.ones(30, dtype=bool)
        self.spikes = rng.poisson(.3, (30, len(self.times), 3)).astype(float)
        self.spikes[self.cues == 1, :, 0] += 2
        self.config = screening.Config(
            **{f'check_{name}': False for name in ('min_trials', *screening.CHECK_NAMES)},
            t_test_end=600, t_test_step=50,
        )

    def run_session(self, config):
        with patch.object(screening, 'load_session', return_value=(
            self.spikes, self.times, self.cues, self.correct,
        )):
            return screening.process_session(Path('example.mat'), config)

    def test_disabled_measurements_do_not_execute_or_reject(self):
        from contextlib import ExitStack
        with ExitStack() as stack:
            for name in screening.CHECK_NAMES:
                if name != 'selectivity':  # PEV/cue metadata is still needed downstream.
                    stack.enter_context(patch.object(checks, name, side_effect=AssertionError(name)))
            result, rows = self.run_session(self.config)
        np.testing.assert_array_equal(result['cell_idx_selected'], [0, 1, 2])
        np.testing.assert_array_equal(result['cell_idx_passed_presence_ratio'], [0, 1, 2])
        self.assertTrue(all(row[f'check_{name}'] == 'disabled' for row in rows for name in screening.CHECK_NAMES))
        self.assertTrue(all(not row['is_rejected'] for row in rows))
        self.assertFalse(result['screening_checks']['min_trials'])

    def test_each_measurement_gate_rejects_failure_and_unavailable_values_independently(self):
        settings = {
            'firing_rate': ('min_fr_test', 1, [np.nan, .1, 2]),
            'presence_ratio': ('min_presence_ratio', .9, [np.nan, .2, 1]),
            'delay_variance': ('var_ratio_threshold_delay_over_baseline', 1, [np.nan, .1, 2]),
            'baseline_variance': ('var_ratio_threshold_sliding_over_all', 1, [np.nan, .1, 2]),
            'baseline_drift': ('temp_dep_r_threshold_baseline', .3, [np.nan, .9, .1]),
            'preferred_drift': ('temp_dep_r_threshold', .3, [np.nan, .9, .1]),
        }
        for name, (threshold, value, measurements) in settings.items():
            config = dataclasses.replace(self.config, **{f'check_{name}': True, threshold: value})
            with self.subTest(check=name), patch.object(checks, name, return_value=np.array(measurements)):
                result, rows = self.run_session(config)
            np.testing.assert_array_equal(result['cell_idx_selected'], [2])
            self.assertEqual([row[f'check_{name}'] for row in rows], ['not_applicable', 'fail', 'pass'])
            self.assertTrue(rows[0]['rejection_reason'].endswith('_not_applicable'))
            self.assertEqual(rows[1]['rejection_reason'], screening.REASONS[name])
            self.assertEqual(rows[2]['rejection_reason'], 'pass')

    def test_selectivity_can_be_disabled_without_invalid_cue_metadata(self):
        self.spikes[:, :, 1:] = 0
        enabled = dataclasses.replace(self.config, check_selectivity=True)
        selected, rows = self.run_session(enabled)
        np.testing.assert_array_equal(selected['cell_idx_selected'], [0])
        np.testing.assert_array_equal(selected['cell_idx_stationary'], [0, 1, 2])
        self.assertEqual(rows[1]['check_selectivity'], 'fail')
        unfiltered, rows = self.run_session(self.config)
        np.testing.assert_array_equal(unfiltered['cell_idx_selected'], [0, 1, 2])
        self.assertTrue(np.all(np.isin(unfiltered['cell_properties']['mean_pref_test'], [1, 5])))
        self.assertTrue(np.all(np.isfinite(unfiltered['cell_properties']['mean_pev_test'])))
        self.assertEqual(rows[1]['check_selectivity'], 'disabled')

    def test_preferred_drift_also_filters_the_stationary_pool(self):
        self.spikes[:, :, 1:] = 0
        config = dataclasses.replace(self.config, check_selectivity=True, check_preferred_drift=True)
        with patch.object(checks, 'preferred_drift', return_value=np.array([.1, .9, np.nan])):
            result, _ = self.run_session(config)
        np.testing.assert_array_equal(result['cell_idx_stationary'], [0])

    def test_disabled_selectivity_skips_run_test_and_ignores_its_cutoff(self):
        with patch.object(checks, 'get_periods_and_mask', side_effect=AssertionError('disabled check')):
            result, _ = self.run_session(self.config)
            changed, _ = self.run_session(dataclasses.replace(self.config, sig_pev_threshold=99, sig_pev_duration=1000))
        np.testing.assert_array_equal(result['cell_properties']['mean_pref_test'], changed['cell_properties']['mean_pref_test'])
        np.testing.assert_array_equal(result['cell_properties']['mean_pev_test'], changed['cell_properties']['mean_pev_test'])
        self.assertNotIn('num_sig_pev_bins', result['cell_properties'])

    def test_min_trial_gate_can_be_disabled_but_metadata_requirements_remain(self):
        result, _ = self.run_session(dataclasses.replace(self.config, check_min_trials=True))
        self.assertIsNone(result)
        result, _ = self.run_session(self.config)
        self.assertIsNotNone(result)
        self.cues[:] = 1
        result, _ = self.run_session(self.config)
        self.assertIsNone(result)

    def test_missing_disabled_check_windows_do_not_reject_cells(self):
        config = dataclasses.replace(self.config, presence_start=-900, presence_end=-800,
                                    baseline_drift_start=-900, baseline_drift_end=-800)
        result, _ = self.run_session(config)
        self.assertIsNotNone(result)
        for name in ('presence_ratio', 'baseline_drift'):
            result, rows = self.run_session(dataclasses.replace(config, **{f'check_{name}': True}))
            self.assertIsNone(result)
            self.assertTrue(all(row[f'check_{name}'] == 'not_applicable' for row in rows))

    def test_constant_baseline_only_fails_enabled_variance_and_drift_checks(self):
        self.spikes[:, self.times < 0] = 1
        result, _ = self.run_session(self.config)
        self.assertIsNotNone(result)
        for name in ('delay_variance', 'baseline_variance', 'baseline_drift'):
            config = dataclasses.replace(self.config, min_trial_for_temp_check=10, **{f'check_{name}': True})
            result, rows = self.run_session(config)
            self.assertIsNone(result)
            self.assertTrue(all(row[f'check_{name}'] == 'not_applicable' for row in rows))

    def test_presence_uses_correct_trials_and_configured_half_open_interval(self):
        data = checks.SessionMeasurements(self.spikes, self.times, self.cues, self.correct)
        data.spikes[:] = 0
        data.correct[2:] = False
        data.spikes[:2, self.times == 0, 0] = 1
        data.spikes[2:, self.times == 0, 1] = 1
        data.spikes[:2, self.times == 100, 2] = 1
        config = dataclasses.replace(self.config, presence_start=0, presence_end=100)
        np.testing.assert_array_equal(checks.presence_ratio(data, config), [1, 0, 0])
        # Extended diagnostics must honor the same custom window even with the gate off.
        import pandas as pd
        from scripts.next.selection_diagnostics import save_diagnostics
        with tempfile.TemporaryDirectory() as directory:
            config = dataclasses.replace(config, cache_dir=Path(directory))
            rows = [dict(session='example', cell_idx=i, rejection_reason='pass',
                         check_presence_ratio='disabled') for i in range(3)]
            with patch('scripts.next.selection_diagnostics.load_session', return_value=(
                data.spikes, data.times, data.cues, data.correct,
            )):
                save_diagnostics(rows, [Path('example.mat')], config)
            frame = pd.read_csv(config.cache_dir / 'select/diagnostics/cell_rejection_diagnostics.csv')
            np.testing.assert_array_equal(frame.presence_ratio, [1, 0, 0])
            self.assertTrue((frame.check_presence_ratio == 'disabled').all())

    def test_variance_ratios_and_correlation_match_known_values(self):
        data = checks.SessionMeasurements(self.spikes, self.times, self.cues, self.correct)
        baseline = np.arange(4.)[:, None]
        config = dataclasses.replace(self.config, min_trial_for_temp_check=2)
        with patch.object(data, 'period_rates', side_effect=[baseline, 2 * baseline]):
            np.testing.assert_allclose(checks.delay_variance(data, config), [4])
        with patch.object(data, 'period_rates', return_value=baseline):
            np.testing.assert_allclose(checks.baseline_variance(data, config), [.3])
        values = np.column_stack([np.arange(4), -np.arange(4), np.ones(4)])
        indices = np.arange(4, dtype=float)
        np.testing.assert_allclose(checks.temporal_correlation(values, indices), [1, -1, np.nan])
        np.testing.assert_array_equal(indices, np.arange(4))


if __name__ == '__main__':
    unittest.main()
