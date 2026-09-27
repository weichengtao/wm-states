"""Manual trust is explicit, scoped, audited, and never rewrites cache identity."""
import hashlib
from pathlib import Path
import tempfile
import unittest
import warnings
from unittest.mock import patch

from scripts.next.decoding_confidence import Config
from scripts.next.decoding_provenance import (
    DecodingCacheMismatch, FingerprintVerification, decoding_fingerprint,
    verify_decoding_fingerprint,
)
from scripts.next.decoding_signature import ScientificSourceError
from scripts.next.legacy_decoding_provenance import HistoricalSourceInvalid, HistoricalSourceUnavailable
from scripts.next.legacy_trust import current_legacy_trust, legacy_trust_context


class LegacyTrustTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.cache = self.root / 'run'
        self.selection = self.cache / 'select/cell_screening.pkl'
        self.selection.parent.mkdir(parents=True)
        self.selection.write_bytes(b'current readable selection')
        self.data = self.root / 'session.mat'
        self.data.write_bytes(b'current readable recording')
        self.config = Config(cache_dir=self.cache, data_dir=self.root)
        self.key = 'b' * 64

    def verify(self, key=None, **paths):
        return verify_decoding_fingerprint(
            self.key if key is None else key, self.config,
            paths.get('selection', self.selection), paths.get('data', self.data),
        )

    def strict_error(self, error=None):
        return patch('scripts.next.decoding_provenance._verify_decoding_fingerprint',
                     side_effect=error or DecodingCacheMismatch('original inputs do not match'))

    def test_disabled_and_absent_context_never_bypass_verification(self):
        self.assertIsNone(current_legacy_trust())
        with self.strict_error(), self.assertRaises(DecodingCacheMismatch):
            self.verify()
        with legacy_trust_context(self.cache) as trust, self.strict_error():
            trust.set_stage('decode')
            with self.assertRaises(DecodingCacheMismatch):
                self.verify()
            self.assertEqual(trust.audit, {'enabled': False, 'manual_trust_used': False, 'events': []})
        self.assertIsNone(current_legacy_trust())

    def test_strict_success_does_not_use_manual_trust_even_when_enabled(self):
        expected = FingerprintVerification('legacy', 'a' * 40, 'decode-v2:' + 'c' * 64)
        with legacy_trust_context(self.cache, True) as trust, \
                patch('scripts.next.decoding_provenance._verify_decoding_fingerprint', return_value=expected), \
                warnings.catch_warnings(record=True) as caught:
            trust.set_stage('decode')
            self.assertEqual(self.verify(), expected)
            self.assertEqual(trust.audit['events'], [])
        self.assertEqual(caught, [])

    def test_eligible_failures_are_audited_before_acceptance_without_new_identity(self):
        errors = [DecodingCacheMismatch('settings or inputs differ'),
                  HistoricalSourceUnavailable(('a' * 40,)),
                  ScientificSourceError('unsupported dynamic lookup')]
        for error in errors:
            persisted = []
            def save_audit(audit):
                self.assertFalse(current_legacy_trust().audit['manual_trust_used'])
                persisted.append(audit)
            with self.subTest(error=type(error).__name__), \
                    legacy_trust_context(self.cache, True, save_audit) as trust, \
                    self.strict_error(error), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                trust.set_stage('activity')
                verified = self.verify()
                self.assertEqual(verified, FingerprintVerification('trusted-legacy', None, None))
                self.assertEqual(persisted, [trust.audit])
                event, = trust.audit['events']
                self.assertEqual(event['original_fingerprint'], self.key)
                self.assertEqual(event['stage'], 'activity')
                self.assertEqual(event['session'], 'session')
                self.assertEqual(event['reason'], str(error))
                self.assertEqual(event['reason_type'], type(error).__name__)
                self.assertEqual(event['verification_status'], 'unverified-legacy')
                self.assertEqual(event['original_runtime_versions'], 'unknown')
                self.assertTrue(event['manual_trust'])
                for kind, path in [('selection', self.selection), ('data', self.data)]:
                    expected = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                'size_bytes': path.stat().st_size}
                    self.assertEqual(event['current_input_checksums'][kind], expected)
                self.assertEqual(len(caught), 1)
                self.assertEqual(caught[0].category, RuntimeWarning)

    def test_native_v2_missing_and_malformed_fingerprints_cannot_be_trusted(self):
        current = decoding_fingerprint(self.config, self.selection, self.data)
        self.data.write_bytes(b'changed recording')
        with legacy_trust_context(self.cache, True) as trust:
            trust.set_stage('decode')
            for key in (current, '', 'invalid', 'A' * 64, 'c' * 63, 'c' * 65):
                with self.subTest(key=key), self.assertRaises(DecodingCacheMismatch):
                    self.verify(key)
            with self.assertRaises(DecodingCacheMismatch):
                verify_decoding_fingerprint(None, self.config, self.selection, self.data)
            self.assertEqual(trust.audit['events'], [])

    def test_historical_corruption_and_unreadable_inputs_remain_fatal(self):
        with legacy_trust_context(self.cache, True) as trust:
            trust.set_stage('decode')
            for error in (HistoricalSourceInvalid('blob checksum mismatch'),
                          OSError('cannot read input')):
                with self.subTest(error=type(error).__name__), self.strict_error(error), \
                        self.assertRaises(type(error)):
                    self.verify()
            with self.strict_error(), self.assertRaises(FileNotFoundError):
                self.verify(data=self.root / 'missing.mat')
            self.selection.unlink()
            with self.strict_error(), self.assertRaises(FileNotFoundError):
                self.verify()
            self.assertFalse(trust.audit['manual_trust_used'])

    def test_other_run_and_redirected_selection_do_not_share_the_override(self):
        other = self.root / 'other/select/cell_screening.pkl'
        other.parent.mkdir(parents=True)
        other.write_bytes(self.selection.read_bytes())
        with legacy_trust_context(self.cache, True) as trust, self.strict_error():
            trust.set_stage('decode')
            with self.assertRaises(DecodingCacheMismatch):
                self.verify(selection=other)
            self.selection.unlink()
            self.selection.symlink_to(other)
            with self.assertRaises(DecodingCacheMismatch):
                self.verify()
            self.assertEqual(trust.audit['events'], [])

    def test_audit_failure_aborts_before_acceptance_and_can_be_retried(self):
        def failed_save(audit):
            raise OSError('manifest storage unavailable')
        with legacy_trust_context(self.cache, True, failed_save) as trust, self.strict_error(), \
                warnings.catch_warnings(record=True) as caught:
            trust.set_stage('decode')
            with self.assertRaisesRegex(OSError, 'manifest storage unavailable'):
                self.verify()
            self.assertEqual(trust.audit['events'], [])
            self.assertEqual(caught, [])
            trust.on_update = lambda audit: None
            self.assertEqual(self.verify().scheme, 'trusted-legacy')
            self.assertEqual(len(trust.audit['events']), 1)

    def test_repeated_checks_deduplicate_audits_and_warn_once_per_stage_session(self):
        audits = []
        with legacy_trust_context(self.cache, True, audits.append) as trust, self.strict_error(), \
                warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            trust.set_stage('decode')
            self.verify()
            self.verify()
            self.assertEqual(len(audits), 1)
            self.data.write_bytes(b'another readable recording')
            self.verify()
            self.assertEqual(len(audits), 2)
            self.assertEqual(len(caught), 1)
            trust.set_stage('activity')
            self.verify()
            self.assertEqual(len(audits), 3)
            self.assertEqual(len(caught), 2)
            detached = trust.audit
            detached['events'][0]['original_fingerprint'] = 'changed externally'
            self.assertEqual(trust.audit['events'][0]['original_fingerprint'], self.key)

    def test_context_is_restored_after_nested_disabled_scope_and_failure(self):
        with legacy_trust_context(self.cache, True) as outer, self.strict_error():
            outer.set_stage('decode')
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                with legacy_trust_context(self.cache, False) as inner:
                    self.assertIs(current_legacy_trust(), inner)
                    with self.assertRaises(DecodingCacheMismatch):
                        self.verify()
                    raise RuntimeError('interrupted')
            self.assertIs(current_legacy_trust(), outer)
        self.assertIsNone(current_legacy_trust())

    def test_stage_and_boolean_are_explicit(self):
        with self.assertRaises(ValueError):
            with legacy_trust_context(self.cache, 'true'):
                pass
        with legacy_trust_context(self.cache, True), self.strict_error(), \
                self.assertRaisesRegex(ValueError, 'Set the analysis stage'):
            self.verify()


if __name__ == '__main__':
    unittest.main()
