import copy
import dataclasses
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd

from scripts.next import cache_io
from scripts.next.common import decoding_fingerprint, json_value, validate_state_provenance
from scripts.next.decoding_confidence import Config as DecodeConfig
from scripts.next.eval_confidence_across_runs import load_runs, METRICS
from scripts.next.selection_diagnostics import save_diagnostics


class StateProvenanceTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.cache = self.root / 'cache'
        self.data = self.root / 'data'
        self.data.mkdir()
        self.source = self.data / 'example.mat'
        self.source.write_bytes(b'data version 1')
        self.selection_path = self.cache / 'select/cell_screening.pkl'
        cache_io.save([{'session': 'example', 'num_trials': 10, 'cells': [1]}], self.selection_path)
        config = DecodeConfig(data_dir=self.data, cache_dir=self.cache)
        self.decoded = {
            'session': 'example', 'cue': 1, 'trial_idx': np.array([2, 4]),
            'time_bins': np.array([500, 550]),
            'config': json.loads(json.dumps(dataclasses.asdict(config), default=json_value)),
            'fingerprint': decoding_fingerprint(config, self.selection_path, self.source),
        }
        self.state = {field: copy.deepcopy(self.decoded[field])
                      for field in ('session', 'cue', 'trial_idx', 'time_bins')}
        self.state['decoding_fingerprint'] = self.decoded['fingerprint']
        self.save_decoding()

    def save_decoding(self):
        cache_io.save([self.decoded], self.cache / 'decode/decoding_confidence.pkl')

    def validate(self):
        validate_state_provenance([self.state], self.cache, self.data)

    def test_matching_chain_and_serialized_settings_are_valid(self):
        self.validate()
        self.decoded['config']['n_jobs'] = 8
        self.save_decoding()
        self.validate()

    def test_changed_selection_with_same_trial_count_is_rejected(self):
        cache_io.save([{'session': 'example', 'num_trials': 10, 'cells': [2]}], self.selection_path)
        with self.assertRaisesRegex(ValueError, 'example.*stale.*Rerun decode'):
            self.validate()

    def test_changed_data_with_same_size_is_rejected(self):
        self.source.write_bytes(b'data version 2')
        with self.assertRaisesRegex(ValueError, 'stale'):
            self.validate()

    def test_changed_decoder_settings_are_rejected(self):
        self.decoded['config']['n_decode_shuffle'] += 1
        self.save_decoding()
        with self.assertRaisesRegex(ValueError, 'stale'):
            self.validate()

    def test_state_axes_and_cue_must_match(self):
        for field, value in [('cue', 5), ('trial_idx', [4, 2]), ('time_bins', [500, 600])]:
            with self.subTest(field=field):
                original = self.state[field]
                self.state[field] = value
                with self.assertRaisesRegex(ValueError, f'{field} mismatch'):
                    self.validate()
                self.state[field] = original

    def test_missing_and_stale_state_fingerprints_are_rejected(self):
        for value in (None, 'old-decoding-run'):
            with self.subTest(value=value):
                self.state['decoding_fingerprint'] = value
                with self.assertRaisesRegex(ValueError, 'fingerprint mismatch or missing provenance'):
                    self.validate()

    def test_missing_decoding_fingerprint_and_settings_are_rejected(self):
        self.decoded['fingerprint'] = None
        self.state['decoding_fingerprint'] = None
        self.save_decoding()
        with self.assertRaisesRegex(ValueError, 'missing provenance'):
            self.validate()
        self.decoded['fingerprint'] = self.state['decoding_fingerprint'] = 'matching-key'
        del self.decoded['config']
        self.save_decoding()
        with self.assertRaisesRegex(ValueError, 'missing decoding settings'):
            self.validate()

    def test_missing_or_duplicate_decoder_session_is_rejected(self):
        for decoded, message in [([], 'no matching decoding result'),
                                 ([self.decoded, self.decoded], 'Duplicate sessions')]:
            with self.subTest(message=message):
                cache_io.save(decoded, self.cache / 'decode/decoding_confidence.pkl')
                with self.assertRaisesRegex(ValueError, message):
                    self.validate()

    def test_missing_decoding_file_is_actionable(self):
        (self.cache / 'decode/decoding_confidence.pkl').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing decoding cache.*Rerun decode'):
            self.validate()

    def test_both_downstream_entrypoints_reject_stale_states_before_analysis(self):
        from scripts.next import compare_activity_across_states as activity
        from scripts.next import prepare_data_for_mixedlm as prepare

        self.state['decoding_fingerprint'] = 'old-decoding-run'
        cache_io.save([self.state], self.cache / 'states/on_off_states.pkl')
        for module, entrypoint, worker in [
            (activity, activity.main, 'prepare_session_activity'),
            (prepare, prepare.prepare_data, '_prepare_session_rows'),
        ]:
            with self.subTest(module=module.__name__), patch.object(module, worker) as analyze:
                with self.assertRaisesRegex(ValueError, 'fingerprint mismatch'):
                    entrypoint(module.Config(cache_dir=self.cache, data_dir=self.data))
                analyze.assert_not_called()


class ScreeningDiagnosticsTest(unittest.TestCase):
    def test_presence_uses_correct_trials_and_half_open_screening_window(self):
        spikes = np.zeros((10, 3, 2))
        spikes[:2, 0, 0] = 1  # cell 0 fires on both correct trials
        spikes[2:, 0, 1] = 1  # cell 1 fires only on incorrect trials
        spikes[:2, 2, 1] = 1  # 1400 ms is outside the screening window
        with tempfile.TemporaryDirectory() as directory:
            config = SimpleNamespace(cache_dir=Path(directory), diagnostics_figure_config=None,
                                     baseline_drift_start_ms=-400, baseline_drift_end_ms=0,
                                     test_start_ms=500, test_end_ms=1400,
                                     presence_start_ms=-400, presence_end_ms=1400)
            rows = [dict(session='example', cell_idx=i, rejection_reason='pass') for i in range(2)]
            with patch('scripts.next.selection_diagnostics.load_session', return_value=(
                spikes, np.array([-400, 500, 1400]), np.ones(10), np.arange(10) < 2,
            )):
                save_diagnostics(rows, [Path('example.mat')], config)
            frame = pd.read_csv(config.cache_dir / 'select/diagnostics/cell_rejection_diagnostics.csv')
            np.testing.assert_array_equal(frame.presence_ratio, [1, 0])


class CrossRunWarningTest(unittest.TestCase):
    def compare(self, changes):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = dict(session='example', cue=1, trial_idx=np.array([2, 4]),
                          observed={f'{metric}_by_time_bin': np.array([.5]) for metric, _ in METRICS},
                          null=None)
            other = dict(result, **changes)
            cache_io.save([result], root / 'run_a/evaluate/eval_confidence.pkl')
            cache_io.save([other], root / 'run_b/evaluate/eval_confidence.pkl')
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                names, runs, sessions = load_runs([root / 'run_a', root / 'run_b'])
            self.assertEqual(names, ['run_a', 'run_b'])
            self.assertEqual(sessions, ['example'])
            self.assertEqual(len(runs), 2)
            return [str(w.message) for w in caught]

    def test_matching_population_and_reordered_trials_do_not_warn(self):
        self.assertEqual(self.compare({}), [])
        self.assertEqual(self.compare({'trial_idx': np.array([4, 2])}), [])

    def test_different_cues_or_trials_warn_and_continue(self):
        for changes, expected in [({'cue': 5}, 'preferred cues differ'),
                                  ({'trial_idx': np.array([2, 6])}, 'trial IDs differ')]:
            with self.subTest(changes=changes):
                messages = self.compare(changes)
                self.assertEqual(len(messages), 1)
                self.assertIn('Session example, runs run_a and run_b', messages[0])
                self.assertIn(expected, messages[0])
                self.assertIn('Continuing comparison', messages[0])

    def test_both_differences_are_combined_in_one_warning(self):
        messages = self.compare({'cue': 5, 'trial_idx': np.array([1])})
        self.assertEqual(len(messages), 1)
        self.assertIn('preferred cues differ', messages[0])
        self.assertIn('trial IDs differ', messages[0])

    def test_missing_metadata_warns_instead_of_silently_claiming_comparability(self):
        messages = self.compare({'cue': None, 'trial_idx': None})
        self.assertEqual(len(messages), 1)
        self.assertIn('cue metadata is missing', messages[0])
        self.assertIn('trial ID metadata is missing', messages[0])


if __name__ == '__main__':
    unittest.main()
