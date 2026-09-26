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
            '--check-preferred-cue-drift', '--max-abs-preferred-cue-drift-r', '0.6',
            '--no-check-baseline-drift', '--presence-start-ms', '-300', '--presence-end-ms', '1200',
        ])
        self.assertEqual(config.max_abs_preferred_cue_drift_r, .6)
        self.assertFalse(config.check_baseline_drift)
        self.assertEqual((config.presence_start_ms, config.presence_end_ms), (-300, 1200))

    def test_invalid_thresholds_are_rejected_even_for_disabled_checks(self):
        invalid = {
            'min_test_firing_rate_hz': [-1, np.nan, np.inf],
            'min_presence_ratio': [-.1, 1.1, np.nan],
            'min_delay_to_baseline_variance_ratio': [-1, np.inf],
            'min_baseline_window_variance_ratio': [-1, np.nan],
            'max_abs_preferred_cue_drift_r': [2, -.1, np.nan],
            'max_abs_baseline_drift_r': [2, np.inf],
            'selectivity_pev_threshold_pct': [-1, 101], 'selectivity_pev_floor_pct': [-1, 101],
        }
        for field, values in invalid.items():
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, field):
                    screening.Config(**{field: value})

    def test_invalid_counts_windows_and_json_switches_are_rejected(self):
        for fields in [dict(min_trials_per_session=0), dict(variance_window_trials=1),
                       dict(selectivity_min_duration_ms=0), dict(selectivity_bin_step_ms=0), dict(selectivity_bin_width_ms=-1),
                       dict(presence_end_ms=-400), dict(baseline_drift_end_ms=-500),
                       dict(variance_delay_end_ms=500), dict(test_start_ms=np.nan),
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
        removed_keys = (
            'temp_dep_detection', 'min_cell_per_group', 'seed', 'min_trial_per_session',
            'min_fr_test', 'presence_start', 'presence_end',
            'var_ratio_threshold_delay_over_baseline', 'var_ratio_threshold_sliding_over_all',
            'min_trial_for_temp_check', 'temp_check_baseline_start', 'temp_check_baseline_end',
            'temp_check_delay_start', 'temp_check_delay_end', 'temp_dep_r_threshold_baseline',
            'baseline_drift_start', 'baseline_drift_end', 'sig_pev_threshold', 'sig_pev_duration',
            'pev_clip_at', 'temp_dep_r_threshold', 'check_preferred_drift',
            't_test_start', 't_test_end', 't_test_window', 't_test_step',
        )
        for key in removed_keys:
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
            test_end_ms=600, selectivity_bin_step_ms=50,
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
            'firing_rate': ('min_test_firing_rate_hz', 1, [np.nan, .1, 2]),
            'presence_ratio': ('min_presence_ratio', .9, [np.nan, .2, 1]),
            'delay_variance': ('min_delay_to_baseline_variance_ratio', 1, [np.nan, .1, 2]),
            'baseline_variance': ('min_baseline_window_variance_ratio', 1, [np.nan, .1, 2]),
            'baseline_drift': ('max_abs_baseline_drift_r', .3, [np.nan, .9, .1]),
            'preferred_cue_drift': ('max_abs_preferred_cue_drift_r', .3, [np.nan, .9, .1]),
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

    def test_all_failed_checks_and_cached_measurements_have_descriptive_names(self):
        from contextlib import ExitStack

        config = dataclasses.replace(self.config, min_test_firing_rate_hz=1,
                                     **{f'check_{name}': True for name in screening.CHECK_NAMES})
        check_measurements = {
            'firing_rate': [np.nan, .1, 2],
            'presence_ratio': [np.nan, .2, 1],
            'delay_variance': [np.nan, .1, 2],
            'baseline_variance': [np.nan, .1, 2],
            'baseline_drift': [np.nan, .9, .1],
            'preferred_cue_drift': [np.nan, .9, .1],
        }
        selectivity = checks.Selectivity(
            mean_pev_pct=np.array([np.nan, .5, 20]), preferred_cue=np.array([1, 1, 1]),
            qualifying_bin_mask=np.array([[False, False], [False, False], [True, True]]),
            passes_duration_check=np.array([False, False, True]),
            has_finite_pev=np.array([False, True, True]),
        )
        with ExitStack() as stack:
            for name, measurements in check_measurements.items():
                stack.enter_context(patch.object(checks, name, return_value=np.array(measurements)))
            stack.enter_context(patch.object(checks, 'selectivity', return_value=selectivity))
            result, rows = self.run_session(config)

        expected_reasons = [
            'fail_firing_rate', 'fail_presence_ratio', 'fail_delay_variance',
            'fail_baseline_variance', 'fail_baseline_drift', 'fail_selectivity',
            'fail_preferred_cue_drift',
        ]
        self.assertEqual(rows[0]['rejection_reason'], '|'.join(
            reason + '_not_applicable' for reason in expected_reasons))
        self.assertEqual(rows[1]['rejection_reason'], '|'.join(expected_reasons))
        self.assertEqual(rows[2]['rejection_reason'], 'pass')
        self.assertEqual({key for key in rows[1] if key.startswith('check_')}, {
            'check_firing_rate', 'check_presence_ratio', 'check_delay_variance',
            'check_baseline_variance', 'check_baseline_drift', 'check_selectivity',
            'check_preferred_cue_drift',
        })
        expected_measurements = {
            'mean_test_firing_rate_hz', 'presence_ratio', 'delay_to_baseline_variance_ratio',
            'baseline_window_variance_ratio', 'baseline_drift_r', 'mean_selectivity_pev_pct',
            'preferred_cue_drift_r',
        }
        self.assertEqual(set(rows[1]) - {key for key in rows[1] if key.startswith('check_')}
                         - {'session', 'cell_idx', 'is_rejected', 'rejection_reason'}, expected_measurements)
        self.assertEqual(set(result['cell_properties']), expected_measurements | {
            'cell_idx', 'preferred_cue', 'qualifying_selectivity_bin_count'})
        np.testing.assert_array_equal(result['cell_idx_selected'], [2])
        np.testing.assert_array_equal(result['cell_idx_stationary'], [2])
        np.testing.assert_array_equal(result['cell_properties']['qualifying_selectivity_bin_count'], [2])
        self.assertEqual(result['screening_checks'], {
            'min_trials': False, 'firing_rate': True, 'presence_ratio': True,
            'delay_variance': True, 'baseline_variance': True, 'baseline_drift': True,
            'selectivity': True, 'preferred_cue_drift': True,
        })

    def test_selectivity_can_be_disabled_without_invalid_cue_metadata(self):
        self.spikes[:, :, 1:] = 0
        enabled = dataclasses.replace(self.config, check_selectivity=True)
        selected, rows = self.run_session(enabled)
        np.testing.assert_array_equal(selected['cell_idx_selected'], [0])
        np.testing.assert_array_equal(selected['cell_idx_stationary'], [0, 1, 2])
        self.assertEqual(rows[1]['check_selectivity'], 'fail')
        unfiltered, rows = self.run_session(self.config)
        np.testing.assert_array_equal(unfiltered['cell_idx_selected'], [0, 1, 2])
        self.assertTrue(np.all(np.isin(unfiltered['cell_properties']['preferred_cue'], [1, 5])))
        self.assertTrue(np.all(np.isfinite(unfiltered['cell_properties']['mean_selectivity_pev_pct'])))
        self.assertEqual(rows[1]['check_selectivity'], 'disabled')

    def test_preferred_cue_drift_also_filters_the_stationary_pool(self):
        self.spikes[:, :, 1:] = 0
        config = dataclasses.replace(self.config, check_selectivity=True, check_preferred_cue_drift=True)
        with patch.object(checks, 'preferred_cue_drift', return_value=np.array([.1, .9, np.nan])):
            result, _ = self.run_session(config)
        np.testing.assert_array_equal(result['cell_idx_stationary'], [0])

    def test_disabled_selectivity_skips_run_test_and_ignores_its_cutoff(self):
        with patch.object(checks, 'get_periods_and_mask', side_effect=AssertionError('disabled check')):
            result, _ = self.run_session(self.config)
            changed, _ = self.run_session(dataclasses.replace(self.config, selectivity_pev_threshold_pct=99, selectivity_min_duration_ms=1000))
        np.testing.assert_array_equal(result['cell_properties']['preferred_cue'], changed['cell_properties']['preferred_cue'])
        np.testing.assert_array_equal(result['cell_properties']['mean_selectivity_pev_pct'], changed['cell_properties']['mean_selectivity_pev_pct'])
        self.assertNotIn('qualifying_selectivity_bin_count', result['cell_properties'])

    def test_min_trial_gate_can_be_disabled_but_metadata_requirements_remain(self):
        result, _ = self.run_session(dataclasses.replace(self.config, check_min_trials=True))
        self.assertIsNone(result)
        result, _ = self.run_session(self.config)
        self.assertIsNotNone(result)
        self.cues[:] = 1
        result, _ = self.run_session(self.config)
        self.assertIsNone(result)

    def test_missing_disabled_check_windows_do_not_reject_cells(self):
        config = dataclasses.replace(self.config, presence_start_ms=-900, presence_end_ms=-800,
                                    baseline_drift_start_ms=-900, baseline_drift_end_ms=-800)
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
            config = dataclasses.replace(self.config, variance_window_trials=10, **{f'check_{name}': True})
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
        config = dataclasses.replace(self.config, presence_start_ms=0, presence_end_ms=100)
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
        config = dataclasses.replace(self.config, variance_window_trials=2)
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
