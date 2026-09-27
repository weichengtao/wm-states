import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.next import cache_io, decoding_confidence as decoder
from scripts.next.common import json_value, validate_state_provenance
from scripts.next.decoding_provenance import (
    DecodingCacheMismatch, FINGERPRINT_PREFIX, NON_ANALYSIS_SETTINGS,
    decoding_fingerprint, verify_decoding_fingerprint,
)
from scripts.next.legacy_decoding_provenance import HistoricalSourceUnavailable


class DecodingProvenanceTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.cache, self.data = self.root / 'cache', self.root / 'data'
        self.data.mkdir()
        self.source = self.data / 'session.mat'
        self.source.write_bytes(b'recording version 1')
        self.selection = self.cache / 'select/cell_screening.pkl'
        cache_io.save([{'session': 'session', 'num_trials': 400, 'max_num_cells_per_group': 4}], self.selection)
        self.config = decoder.Config(data_dir=self.data, cache_dir=self.cache, save_figures=False)
        self.source_directory = Path(decoder.__file__).parent
        sources = {path.name: path.read_bytes() for path in self.source_directory.glob('*.py')}
        self.snapshot = SimpleNamespace(revision='a' * 40, sources=sources)
        self.history = SimpleNamespace(snapshots=(self.snapshot,), unavailable_refs=())

    def legacy_key(self, config=None, sources=None):
        settings = asdict(config or self.config)
        settings = {key: value for key, value in settings.items() if key not in NON_ANALYSIS_SETTINGS}
        digest = hashlib.sha256(json.dumps(settings, sort_keys=True, default=json_value).encode())
        for _, contents in sorted((sources or self.snapshot.sources).items()):
            digest.update(contents)
        for path in (self.selection, self.source):
            digest.update(str(path.resolve()).encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()

    def verify(self, key, config=None):
        return verify_decoding_fingerprint(key, config or self.config, self.selection, self.source)

    def mock_history(self, history=None):
        return patch('scripts.next.decoding_provenance.load_legacy_source_snapshots',
                     return_value=history or self.history)

    def save_legacy_results(self):
        key = self.legacy_key()
        result = {
            'session': 'session', 'fingerprint': key, 'cue': 1,
            'trial_idx': np.array([1, 3]), 'time_bins': np.array([500, 550]),
            'decoding_confidence': np.array([[.7, .8], [.6, .9]]),
            'config': json.loads(json.dumps(asdict(self.config), default=json_value)),
        }
        self.checkpoint = self.cache / 'decode/checkpoints/session.pkl'
        self.primary = self.cache / 'decode/decoding_confidence.pkl'
        cache_io.save({'fingerprint': key, 'result': result}, self.checkpoint)
        cache_io.save([result], self.primary)
        self.state = {field: copy.deepcopy(result[field]) for field in ('session', 'cue', 'trial_idx', 'time_bins')}
        self.state['decoding_fingerprint'] = key
        return result

    def test_current_fingerprint_needs_no_git_and_survives_non_analysis_options(self):
        key = decoding_fingerprint(self.config, self.selection, self.source)
        self.assertTrue(key.startswith(FINGERPRINT_PREFIX))
        altered = replace(self.config, n_jobs=5, par_verbose=10, resume=False,
                          plot_only=True, save_figures=True, plot_actual_trial_id=True,
                          session_list_file=self.root / 'another-list.txt', max_sessions_to_run=1)
        with patch('scripts.next.decoding_provenance.load_legacy_source_snapshots',
                   side_effect=AssertionError('New fingerprints must work without Git')):
            verified = self.verify(key, altered)
        self.assertEqual(verified.scheme, 'current')
        self.assertIsNone(verified.source_revision)
        self.assertEqual(verified.current_fingerprint, key)

    def test_current_fingerprint_rejects_changed_runtime_versions(self):
        key = decoding_fingerprint(self.config, self.selection, self.source)
        with patch('scripts.next.decoding_provenance._runtime_versions', return_value={'numpy': 'changed'}):
            with self.assertRaisesRegex(DecodingCacheMismatch, 'package versions'):
                self.verify(key)

    def test_verified_legacy_cache_keeps_original_identity(self):
        key = self.legacy_key()
        with self.mock_history():
            verified = self.verify(key, replace(self.config, n_jobs=3, plot_only=True))
        self.assertEqual(verified.scheme, 'legacy')
        self.assertEqual(verified.source_revision, self.snapshot.revision)
        self.assertEqual(verified.current_fingerprint,
                         decoding_fingerprint(self.config, self.selection, self.source))

    def test_legacy_acceptance_requires_identical_scientific_implementation(self):
        key = self.legacy_key()
        sources = dict(self.snapshot.sources)
        sources['decoding_confidence.py'] = sources['decoding_confidence.py'].replace(
            b'np.delete(np.arange(labels.size), test_idx)',
            b'np.delete(np.arange(labels.size), 0)')
        self.assertNotEqual(sources['decoding_confidence.py'], self.snapshot.sources['decoding_confidence.py'])
        history = SimpleNamespace(snapshots=(SimpleNamespace(revision='b' * 40, sources=sources),),
                                  unavailable_refs=())
        # Even a perfectly reproduced old key cannot admit a different algorithm.
        key = self.legacy_key(sources=sources)
        with self.mock_history(history), self.assertRaises(DecodingCacheMismatch):
            self.verify(key)

    def test_both_schemes_reject_changed_analysis_settings_and_inputs(self):
        for scheme in ('current', 'legacy'):
            with self.subTest(scheme=scheme), self.mock_history():
                key = (self.legacy_key() if scheme == 'legacy'
                       else decoding_fingerprint(self.config, self.selection, self.source))
                for config in (replace(self.config, seed=2), replace(self.config, n_decode_shuffle=2),
                               replace(self.config, preserve_null_time_structure=True)):
                    with self.assertRaises(DecodingCacheMismatch):
                        self.verify(key, config)
                self.source.write_bytes(b'recording version 2')
                with self.assertRaises(DecodingCacheMismatch):
                    self.verify(key)
                self.source.write_bytes(b'recording version 1')
                original = self.selection.read_bytes()
                cache_io.save([{'session': 'session', 'num_trials': 400, 'max_num_cells_per_group': 3}],
                              self.selection)
                with self.assertRaises(DecodingCacheMismatch):
                    self.verify(key)
                self.selection.write_bytes(original)

    def test_missing_history_is_an_error_and_partial_history_is_not_blindly_accepted(self):
        key = self.legacy_key()
        unavailable = SimpleNamespace(snapshots=(), unavailable_refs=('b' * 40,))
        with self.mock_history(unavailable), self.assertRaisesRegex(HistoricalSourceUnavailable, 'history'):
            self.verify(key)
        incomplete = SimpleNamespace(snapshots=self.history.snapshots, unavailable_refs=('b' * 40,))
        with self.mock_history(incomplete):
            self.assertEqual(self.verify(key).scheme, 'legacy')
            with self.assertRaises(HistoricalSourceUnavailable):
                self.verify('0' * 64)

    def test_unknown_or_malformed_keys_are_rejected(self):
        with self.mock_history():
            for key in (None, '', 'not-a-key', '0' * 64, FINGERPRINT_PREFIX + '0' * 64):
                with self.subTest(key=key), self.assertRaises(DecodingCacheMismatch):
                    self.verify(key)

    def test_resume_and_downstream_reuse_legacy_without_fitting_or_rewriting_checkpoint(self):
        result = self.save_legacy_results()
        original = self.checkpoint.read_bytes()
        with self.mock_history(), patch.object(decoder, 'decode_session') as fit:
            resumed = decoder.main(replace(self.config, n_jobs=2))
            validate_state_provenance([self.state], self.cache, self.data)
        fit.assert_not_called()
        self.assertEqual(resumed[0]['fingerprint'], result['fingerprint'])
        np.testing.assert_array_equal(resumed[0]['decoding_confidence'], result['decoding_confidence'])
        self.assertEqual(self.checkpoint.read_bytes(), original)

    def test_unavailable_history_does_not_start_refitting_or_overwrite_caches(self):
        self.save_legacy_results()
        original = (self.checkpoint.read_bytes(), self.primary.read_bytes())
        with patch('scripts.next.decoding_provenance.load_legacy_source_snapshots',
                   side_effect=HistoricalSourceUnavailable(('a' * 40,), reason='Restore local Git history')), \
                patch.object(decoder, 'decode_session') as fit, \
                self.assertRaisesRegex(HistoricalSourceUnavailable, 'history'):
            decoder.main(self.config)
        fit.assert_not_called()
        self.assertEqual((self.checkpoint.read_bytes(), self.primary.read_bytes()), original)

    def test_later_checkpoint_failure_does_not_publish_a_partial_primary(self):
        self.source.with_name('zsession.mat').write_bytes(b'second session')
        cache_io.save([
            {'session': session, 'num_trials': 400, 'max_num_cells_per_group': 4}
            for session in ('session', 'zsession')
        ], self.selection)
        first = self.save_legacy_results()
        second = dict(first, session='zsession', fingerprint='b' * 64)
        cache_io.save({'fingerprint': second['fingerprint'], 'result': second},
                      self.checkpoint.with_name('zsession.pkl'))
        cache_io.save([first, second], self.primary)
        original = self.primary.read_bytes()
        with self.mock_history():
            first_verified = self.verify(first['fingerprint'])
        with patch.object(decoder, 'verify_decoding_fingerprint', side_effect=[
            first_verified, HistoricalSourceUnavailable(('b' * 40,)),
        ]), patch.object(decoder, 'decode_session') as fit, \
                self.assertRaises(HistoricalSourceUnavailable):
            decoder.main(self.config)
        fit.assert_not_called()
        self.assertEqual(self.primary.read_bytes(), original)

    def test_tampered_saved_result_settings_are_not_republished_as_verified(self):
        result = self.save_legacy_results()
        checkpoint = cache_io.read(self.checkpoint)
        checkpoint['result']['config']['seed'] = 123
        cache_io.save(checkpoint, self.checkpoint)
        with self.mock_history(), patch.object(decoder, 'decode_session', return_value=result) as fit:
            decoder.main(self.config)
        fit.assert_called_once()
        self.assertTrue(cache_io.read(self.checkpoint)['fingerprint'].startswith(FINGERPRINT_PREFIX))

    def test_checkpoint_envelope_and_result_keys_must_agree(self):
        self.save_legacy_results()
        checkpoint = cache_io.read(self.checkpoint)
        checkpoint['result']['fingerprint'] = '0' * 64
        cache_io.save(checkpoint, self.checkpoint)
        with patch.object(decoder, 'decode_session') as fit, self.assertRaisesRegex(ValueError, 'inconsistent'):
            decoder.main(self.config)
        fit.assert_not_called()


if __name__ == '__main__':
    unittest.main()
