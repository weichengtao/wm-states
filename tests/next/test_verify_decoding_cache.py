"""Whole-run verification is read-only and rejects inconsistent cache records."""
from contextlib import redirect_stdout
import copy
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts.next import cache_io, verify_decoding_cache as verifier
from scripts.next.common import decoding_fingerprint, json_value
from scripts.next.decoding_confidence import Config as DecodeConfig


class VerifyDecodingCacheTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.data = self.root / 'data'
        self.data.mkdir()
        self.cache = self.root / 'run'
        self.config = verifier.Config(data_dir=self.data, cache_dir=self.cache)
        self.selection = self.cache / 'select/cell_screening.pkl'
        self.primary = self.cache / 'decode/decoding_confidence.pkl'
        cache_io.save([{'session': 'session-1'}], self.selection)
        self.record = self.make_record('session-1')
        self.save_primary(self.record)

    def make_record(self, session):
        (self.data / f'{session}.mat').write_bytes(b'trusted fixture input bytes')
        settings = json.loads(json.dumps(asdict(DecodeConfig(data_dir=self.data, cache_dir=self.cache)),
                                         default=json_value))
        return {'session': session, 'config': settings,
                'fingerprint': decoding_fingerprint(settings, self.selection, self.data / f'{session}.mat'),
                'cue': 1, 'trial_idx': np.array([2, 5]), 'time_bins': np.array([0, 10]),
                'cell_idx': np.array([0]), 'decoding_confidence': np.array([[.8, .9], [.7, .85]]),
                'decoding_confidence_null': np.ones((2, 2, 3)) * .5}

    def save_primary(self, *records):
        cache_io.save(list(records), self.primary)

    def save_checkpoint(self, record, *, name=None, outer=None):
        path = self.cache / 'decode/checkpoints' / f'{name or record["session"]}.pkl'
        cache_io.save({'fingerprint': record['fingerprint'] if outer is None else outer,
                       'result': record}, path)
        return path

    def snapshot(self):
        return {str(path.relative_to(self.root)): path.read_bytes()
                for path in self.root.rglob('*') if path.is_file()}

    def test_current_cache_checkpoint_and_state_are_verified_without_writes(self):
        self.save_checkpoint(self.record)
        state = {key: self.record[key] for key in ('session', 'cue', 'trial_idx', 'time_bins')}
        state['decoding_fingerprint'] = self.record['fingerprint']
        cache_io.save([state], self.cache / 'states/on_off_states.pkl')
        before = self.snapshot()
        output = io.StringIO()
        with redirect_stdout(output):
            report = verifier.main(self.config)
        self.assertEqual(json.loads(output.getvalue()), report)
        self.assertTrue(report['valid'])
        self.assertEqual(report['primary_schemes'], {'current': 1, 'legacy': 0})
        self.assertEqual(report['checkpoint_count'], 1)
        self.assertEqual(report['state_sessions'], 1)
        self.assertEqual(self.snapshot(), before)

    def test_extra_checkpoint_from_partial_run_is_independently_verified(self):
        self.save_checkpoint(self.record)
        second = self.make_record('session-2')
        self.save_checkpoint(second)
        report = verifier.verify(self.config)
        self.assertEqual(report['checkpoint_count'], 2)
        self.assertEqual(report['extra_checkpoint_sessions'], ['session-2'])
        self.assertFalse(report['states_present'])
        (self.data / 'session-2.mat').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'session-2'):
            verifier.verify(self.config)

    def test_missing_checkpoints_are_allowed(self):
        report = verifier.verify(self.config)
        self.assertEqual(report['checkpoint_count'], 0)
        self.assertEqual(report['extra_checkpoint_sessions'], [])

    def test_stale_inputs_and_scientific_settings_fail(self):
        self.record['config']['n_decode_shuffle'] += 1
        self.save_primary(self.record)
        with self.assertRaisesRegex(ValueError, 'session-1'):
            verifier.verify(self.config)

    def test_checkpoint_nonanalysis_settings_may_differ(self):
        checkpoint = copy.deepcopy(self.record)
        checkpoint['config']['n_jobs'] = 4
        checkpoint['config']['save_figures'] = False
        self.save_checkpoint(checkpoint)
        self.assertTrue(verifier.verify(self.config)['valid'])

    def test_checkpoint_payload_and_axes_must_equal_primary(self):
        for field in ('trial_idx', 'time_bins', 'decoding_confidence', 'decoding_confidence_null'):
            with self.subTest(field=field):
                checkpoint = copy.deepcopy(self.record)
                checkpoint[field].flat[0] += 1
                self.save_checkpoint(checkpoint)
                with self.assertRaisesRegex(ValueError, field):
                    verifier.verify(self.config)

    def test_checkpoint_envelope_key_filename_and_duplicate_identity_are_rejected(self):
        path = self.save_checkpoint(self.record, outer='wrong')
        with self.assertRaisesRegex(ValueError, 'outer and result fingerprints'):
            verifier.verify(self.config)
        path.unlink()
        self.save_checkpoint(self.record, name='another-session')
        with self.assertRaisesRegex(ValueError, 'filename does not match'):
            verifier.verify(self.config)

    def test_primary_requires_unique_safe_sessions_settings_and_fingerprint(self):
        cases = [([], 'nonempty'), ([self.record, self.record], 'duplicate')]
        for field, value, message in [('session', '../escape', 'invalid session'),
                                      ('session', 1, 'invalid session'),
                                      ('config', {}, 'missing decoding settings'),
                                      ('fingerprint', None, 'missing a decoding fingerprint')]:
            record = copy.deepcopy(self.record)
            record[field] = value
            cases.append(([record], message))
        for records, message in cases:
            with self.subTest(message=message):
                cache_io.save(records, self.primary)
                with self.assertRaisesRegex(ValueError, message):
                    verifier.verify(self.config)

    def test_states_require_unique_sessions_and_matching_provenance(self):
        state = {key: self.record[key] for key in ('session', 'cue', 'trial_idx', 'time_bins')}
        state['decoding_fingerprint'] = 'wrong'
        path = self.cache / 'states/on_off_states.pkl'
        cache_io.save([state], path)
        with self.assertRaisesRegex(ValueError, 'fingerprint mismatch'):
            verifier.verify(self.config)
        cache_io.save([state, state], path)
        with self.assertRaisesRegex(ValueError, 'duplicate session'):
            verifier.verify(self.config)

    def test_legacy_verification_revisions_are_reported_and_missing_history_is_not_ignored(self):
        legacy = SimpleNamespace(scheme='legacy', source_revision='a' * 40,
                                 current_fingerprint='decode-v2:verified')
        with patch.object(verifier, 'verify_decoding_fingerprint', return_value=legacy):
            report = verifier.verify(self.config)
        self.assertEqual(report['primary_schemes'], {'current': 0, 'legacy': 1})
        self.assertEqual(report['legacy_source_revisions'], ['a' * 40])
        with patch.object(verifier, 'verify_decoding_fingerprint', side_effect=ValueError('Missing history')):
            with self.assertRaisesRegex(ValueError, 'Missing history'):
                verifier.verify(self.config)

    def test_payload_comparison_handles_nan_and_detects_missing_keys(self):
        self.assertIsNone(verifier._different({'x': np.array([np.nan]), 'y': [float('nan')]},
                                             {'x': np.array([np.nan]), 'y': [float('nan')]}))
        self.assertEqual(verifier._different({'a': 1}, {'b': 1}), 'result.keys')


if __name__ == '__main__':
    unittest.main()
