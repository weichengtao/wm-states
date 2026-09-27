"""Run-level manual acceptance is explicit, auditable, and never contagious."""
from contextlib import ExitStack, redirect_stdout
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np

from scripts.next import cache_io, decoding_confidence as decoder, pipeline
from scripts.next.common import json_value, validate_state_provenance
from scripts.next.decoding_provenance import DecodingCacheMismatch, decoding_fingerprint
from scripts.next.legacy_trust import current_legacy_trust, legacy_trust_context
from scripts.next.verify_decoding_cache import Config as VerifyConfig, verify


class PipelineLegacyTrustTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.data, self.cache = self.root / 'data', self.root / 'cache'
        self.data.mkdir()
        self.source = self.data / 'session.mat'
        self.source.write_bytes(b'current inputs')
        self.selection = self.cache / 'select/cell_screening.pkl'
        cache_io.save([{'session': 'session', 'num_trials': 400, 'max_num_cells_per_group': 4}], self.selection)
        self.decode_config = decoder.Config(cache_dir=self.cache, data_dir=self.data, save_figures=False)
        self.result = dict(session='session', fingerprint='0' * 64,
                           config=json.loads(json.dumps(asdict(self.decode_config), default=json_value)),
                           cue=1, trial_idx=np.array([1, 3]), time_bins=np.array([500, 550]),
                           decoding_confidence=np.array([[.7, .8], [.8, .9]]))
        self.primary = self.cache / 'decode/decoding_confidence.pkl'
        self.checkpoint = self.cache / 'decode/checkpoints/session.pkl'
        self.state = {key: self.result[key] for key in ('session', 'cue', 'trial_idx', 'time_bins')}
        self.state['decoding_fingerprint'] = self.result['fingerprint']
        self.save_results()
        self.config = pipeline.Config(cache_dir=self.cache, data_dir=self.data, stages=('activity',))

    def save_results(self):
        cache_io.save([self.result], self.primary)
        cache_io.save({'fingerprint': self.result['fingerprint'], 'result': self.result}, self.checkpoint)
        cache_io.save([self.state], self.cache / 'states/on_off_states.pkl')

    def invoke(self, config, stage_action=None):
        from scripts.next import compare_activity_across_states as activity
        with redirect_stdout(io.StringIO()), warnings.catch_warnings(record=True) as caught, \
                patch('scripts.next.decoding_provenance._verify_decoding_fingerprint',
                      side_effect=DecodingCacheMismatch('Legacy source cannot be verified')), \
                patch.object(activity, 'main', side_effect=stage_action or (
                    lambda _: validate_state_provenance([self.state], self.cache, self.data))):
            warnings.simplefilter('always')
            pipeline.main(config)
        return caught

    def manifest(self):
        return json.loads((self.cache / 'pipeline_manifest.json').read_text())

    def test_default_rejects_unverified_and_does_not_record_manual_use(self):
        self.assertFalse(self.config.trust_unverified_legacy_results)
        with self.assertRaisesRegex(ValueError, 'stale'):
            self.invoke(self.config)
        record = self.manifest()
        self.assertEqual(record['status'], 'failed')
        self.assertEqual(record['legacy_trust'], {'enabled': False, 'manual_trust_used': False, 'events': []})

    def test_enabled_downstream_run_audits_before_use_and_preserves_keys(self):
        originals = (self.primary.read_bytes(), self.checkpoint.read_bytes())
        def consume(_):
            record = self.manifest()
            self.assertTrue(record['legacy_trust']['manual_trust_used'])
            validate_state_provenance([self.state], self.cache, self.data)
        caught = self.invoke(replace(self.config, trust_unverified_legacy_results=True), consume)
        record = self.manifest()
        self.assertEqual(record['status'], 'complete')
        self.assertTrue(record['runner_config']['trust_unverified_legacy_results'])
        events = record['legacy_trust']['events']
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['original_fingerprint'], self.result['fingerprint'])
        self.assertEqual(events[0]['stage'], 'activity')
        self.assertTrue(any('manually trusting' in str(warning.message) for warning in caught))
        self.assertEqual((self.primary.read_bytes(), self.checkpoint.read_bytes()), originals)
        self.assertIsNone(current_legacy_trust())
        history = (self.cache / 'manifests' / (record['run_id'] + '.json')).read_bytes()
        with self.assertRaisesRegex(ValueError, 'stale'):
            self.invoke(self.config)
        self.assertEqual((self.cache / 'manifests' / (record['run_id'] + '.json')).read_bytes(), history)

    def test_plot_only_audits_legacy_without_fitting_or_rewriting_results(self):
        settings = self.root / 'settings.json'
        settings.write_text(json.dumps({'decode': {'plot_only': True}}))
        original = self.primary.read_bytes()
        with redirect_stdout(io.StringIO()), warnings.catch_warnings(record=True), \
                patch('scripts.next.decoding_provenance._verify_decoding_fingerprint',
                      side_effect=DecodingCacheMismatch('Unverified source')), \
                patch.object(decoder, 'decode_session') as fit, patch.object(decoder, 'plot_session') as plot:
            pipeline.main(replace(self.config, stages=('decode',), settings=settings,
                                  trust_unverified_legacy_results=True))
        fit.assert_not_called()
        plot.assert_called_once()
        self.assertTrue(self.manifest()['legacy_trust']['manual_trust_used'])
        self.assertEqual(self.primary.read_bytes(), original)

    def test_resume_records_trust_for_reused_checkpoint(self):
        with redirect_stdout(io.StringIO()), warnings.catch_warnings(record=True), \
                patch('scripts.next.decoding_provenance._verify_decoding_fingerprint',
                      side_effect=DecodingCacheMismatch('Unverified source')), \
                patch.object(decoder, 'decode_session') as fit, patch.object(decoder, 'plot_session'):
            pipeline.main(replace(self.config, stages=('decode',), trust_unverified_legacy_results=True))
        fit.assert_not_called()
        self.assertEqual(self.manifest()['legacy_trust']['events'][0]['stage'], 'decode')
        self.assertEqual(cache_io.read(self.primary)[0]['fingerprint'], self.result['fingerprint'])

    def test_changed_settings_and_explicit_refit_do_not_record_unused_legacy_trust(self):
        for overrides in ({'resume': False}, {'seed': 7}):
            with self.subTest(overrides=overrides):
                self.save_results()
                settings = self.root / 'settings.json'
                settings.write_text(json.dumps({'decode': overrides}))
                fresh = {**self.result, 'config': {**self.result['config'], **overrides}}
                with redirect_stdout(io.StringIO()), patch.object(decoder, 'decode_session', return_value=fresh) as fit, \
                        patch.object(decoder, 'plot_session'), \
                        patch.object(decoder, 'verify_decoding_fingerprint', side_effect=AssertionError('No reused result')):
                    pipeline.main(replace(self.config, stages=('decode',), settings=settings,
                                          trust_unverified_legacy_results=True))
                fit.assert_called_once()
                self.assertFalse(self.manifest()['legacy_trust']['manual_trust_used'])

    def test_failed_stage_retains_acceptance_and_clears_scope(self):
        with self.assertRaisesRegex(RuntimeError, 'stage failed'):
            self.invoke(replace(self.config, trust_unverified_legacy_results=True),
                        lambda _: (_ for _ in ()).throw(RuntimeError('stage failed')))
        self.assertEqual(self.manifest()['status'], 'failed')
        self.assertTrue(self.manifest()['legacy_trust']['manual_trust_used'])
        self.assertIsNone(current_legacy_trust())

    def test_native_v2_and_state_link_mismatches_remain_errors(self):
        self.result['fingerprint'] = 'decode-v2:' + '0' * 64
        self.save_results()
        with self.assertRaises(DecodingCacheMismatch):
            self.invoke(replace(self.config, trust_unverified_legacy_results=True))
        self.assertFalse(self.manifest()['legacy_trust']['manual_trust_used'])
        self.result['fingerprint'] = '0' * 64
        self.state['decoding_fingerprint'] = '1' * 64
        self.save_results()
        with self.assertRaisesRegex(ValueError, 'fingerprint mismatch'):
            self.invoke(replace(self.config, trust_unverified_legacy_results=True))

    def test_flag_requires_existing_cache_and_strict_boolean(self):
        for config in (replace(self.config, trust_unverified_legacy_results='true'),
                       replace(self.config, cache_dir=self.root / 'missing', trust_unverified_legacy_results=True)):
            with self.subTest(config=config), self.assertRaises(ValueError):
                pipeline.main(config)

    def test_explicit_readonly_verification_stays_strict_inside_trust_context(self):
        with legacy_trust_context(self.cache, enabled=True) as trust:
            trust.set_stage('decode')
            with patch('scripts.next.decoding_provenance._verify_decoding_fingerprint',
                       side_effect=DecodingCacheMismatch('Unverified source')), \
                    self.assertRaisesRegex(ValueError, 'Unverified source'):
                verify(VerifyConfig(cache_dir=self.cache, data_dir=self.data))
            self.assertFalse(trust.audit['manual_trust_used'])


if __name__ == '__main__':
    unittest.main()
