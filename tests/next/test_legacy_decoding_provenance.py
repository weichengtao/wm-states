"""Historical verification loads only pinned local source objects, never code."""
import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts.next import legacy_decoding_provenance as legacy


LEGACY_COMMON = b'''def json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)

def fingerprint(config, paths=(), *, exclude=()):
    settings = asdict(config) if is_dataclass(config) else dict(config)
    settings = {k: v for k, v in settings.items() if k not in exclude}
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True, default=json_value).encode())
    for path in sorted(Path(__file__).parent.glob('*.py')):
        digest.update(path.read_bytes())
    for path in paths:
        path = Path(path)
        digest.update(str(path.resolve()).encode())
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
    return digest.hexdigest()

def decoding_fingerprint(config, selection_path, data_path):
    """Use the same provenance rules for decoder resume and downstream checks."""
    return fingerprint(config, [selection_path, data_path], exclude=(
        'n_jobs', 'par_verbose', 'resume', 'plot_only', 'save_figures',
        'plot_actual_trial_id', 'session_list_file', 'max_sessions_to_run',
    ))
'''


def blob_id(content):
    return hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest()


class LegacySourceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        legacy._load_snapshot.cache_clear()
        self.addCleanup(legacy._load_snapshot.cache_clear)
        self.files = {'common.py': LEGACY_COMMON,
                      'z_extra.py': b'raise RuntimeError("Never execute historical sources")\n'}
        self.missing = set()
        self.calls = []

    def git(self, root, arguments, *, input_bytes=None):
        self.calls.append((arguments, input_bytes))
        self.assertEqual(root, self.root)
        if arguments == ['rev-parse', '--show-toplevel']:
            return str(self.root).encode() + b'\n'
        if arguments[0] == 'ls-tree':
            revision = arguments[-1].split(':')[0]
            self.assertIn(revision, legacy.PINNED_LEGACY_REVISIONS)
            if revision in self.missing:
                raise legacy.HistoricalSourceUnavailable((revision,))
            return b''.join(f'100644 blob {blob_id(content)}\t{name}\0'.encode()
                            for name, content in reversed(list(self.files.items()))) + (
                b'040000 tree ' + b'0' * 40 + b'\tdashboard\0'
                b'100644 blob ' + b'1' * 40 + b'\tREADME.md\0')
        self.assertEqual(arguments, ['cat-file', '--batch'])
        by_id = {blob_id(content): content for content in self.files.values()}
        return b''.join(identifier + f' blob {len(by_id[identifier.decode()])}\n'.encode()
                        + by_id[identifier.decode()] + b'\n'
                        for identifier in input_bytes.splitlines())

    def test_loads_sorted_immutable_sources_only_from_pinned_local_revisions(self):
        with patch.object(legacy, '_git', side_effect=self.git):
            history = legacy.load_legacy_source_snapshots(self.root)
        self.assertEqual(tuple(snapshot.revision for snapshot in history.snapshots),
                         legacy.PINNED_LEGACY_REVISIONS)
        self.assertEqual(history.unavailable_refs, ())
        history.require_complete()
        for snapshot in history.snapshots:
            self.assertEqual(list(snapshot.sources), ['common.py', 'z_extra.py'])
            self.assertEqual(dict(snapshot.sources), self.files)
            with self.assertRaises(TypeError):
                snapshot.sources['common.py'] = b'altered'
        self.assertEqual(len(self.calls), 5)

    def test_missing_revision_is_visible_and_successful_other_history_is_retained(self):
        self.missing.add(legacy.PINNED_LEGACY_REVISIONS[0])
        with patch.object(legacy, '_git', side_effect=self.git):
            history = legacy.load_legacy_source_snapshots(self.root)
        self.assertEqual(len(history.snapshots), 1)
        self.assertEqual(history.unavailable_refs, (legacy.PINNED_LEGACY_REVISIONS[0],))
        with self.assertRaisesRegex(legacy.HistoricalSourceUnavailable, 'explicitly choose to recompute'):
            history.require_complete()

    def test_no_history_and_nonrepository_installation_fail_with_actionable_error(self):
        self.missing.update(legacy.PINNED_LEGACY_REVISIONS)
        with patch.object(legacy, '_git', side_effect=self.git), self.assertRaisesRegex(
                legacy.HistoricalSourceUnavailable, 'Restore repository Git history'):
            legacy.load_legacy_source_snapshots(self.root)
        with self.assertRaises(legacy.HistoricalSourceUnavailable):
            legacy.load_legacy_source_snapshots(self.root)

    def test_unpinned_revision_and_parent_repository_are_rejected(self):
        with patch.object(legacy, '_git') as git, self.assertRaisesRegex(
                legacy.HistoricalSourceInvalid, 'Only pinned'):
            legacy._load_snapshot(self.root, 'HEAD')
        git.assert_not_called()
        with patch.object(legacy, '_git', return_value=str(self.root.parent).encode()), \
                self.assertRaisesRegex(legacy.HistoricalSourceUnavailable, 'repository root'):
            legacy.load_legacy_source_snapshots(self.root)

    def test_all_original_algorithm_functions_and_serialization_are_checked(self):
        legacy.validate_legacy_algorithm({'common.py': LEGACY_COMMON})
        variants = [
            LEGACY_COMMON.replace(b'sha256(', b'sha1('),
            LEGACY_COMMON.replace(b"'max_sessions_to_run',", b"'seed',"),
            LEGACY_COMMON.replace(b'return value.value', b'return value.name'),
            LEGACY_COMMON + b'\ndef fingerprint(config):\n    return "trusted"\n',
            b'not valid Python {',
        ]
        for source in variants:
            with self.subTest(source=source[-80:]), self.assertRaises(legacy.HistoricalSourceInvalid):
                legacy.validate_legacy_algorithm({'common.py': source})
        with self.assertRaises(legacy.HistoricalSourceInvalid):
            legacy.validate_legacy_algorithm({})

    def test_invalid_available_algorithm_is_not_silently_treated_as_missing(self):
        self.files['common.py'] = LEGACY_COMMON.replace(b'sort_keys=True', b'sort_keys=False')
        with patch.object(legacy, '_git', side_effect=self.git), \
                self.assertRaises(legacy.HistoricalSourceInvalid):
            legacy.load_legacy_source_snapshots(self.root)

    def test_missing_blob_and_malformed_blob_are_distinguished(self):
        valid_git = self.git
        def missing_blob(root, arguments, *, input_bytes=None):
            if arguments[0] == 'cat-file':
                return input_bytes.splitlines()[0] + b' missing\n'
            return valid_git(root, arguments, input_bytes=input_bytes)
        with patch.object(legacy, '_git', side_effect=missing_blob), \
                self.assertRaises(legacy.HistoricalSourceUnavailable):
            legacy.load_legacy_source_snapshots(self.root)
        def malformed_blob(root, arguments, *, input_bytes=None):
            if arguments[0] == 'cat-file':
                return valid_git(root, arguments, input_bytes=input_bytes)[:-5]
            return valid_git(root, arguments, input_bytes=input_bytes)
        with patch.object(legacy, '_git', side_effect=malformed_blob), \
                self.assertRaises(legacy.HistoricalSourceInvalid):
            legacy.load_legacy_source_snapshots(self.root)

    def test_git_disables_lazy_fetch_replacement_refs_and_prompts(self):
        with patch.object(legacy.subprocess, 'run', return_value=subprocess.CompletedProcess(
                args=[], returncode=0, stdout=b'local bytes', stderr=b'')) as run:
            self.assertEqual(legacy._git(self.root, ['cat-file', '--batch'], input_bytes=b'abc\n'), b'local bytes')
        arguments, options = run.call_args
        self.assertEqual(arguments[0], ['git', '--no-pager', '-C', str(self.root), 'cat-file', '--batch'])
        self.assertEqual(options['env']['GIT_NO_LAZY_FETCH'], '1')
        self.assertEqual(options['env']['GIT_NO_REPLACE_OBJECTS'], '1')
        self.assertEqual(options['env']['GIT_TERMINAL_PROMPT'], '0')
        self.assertEqual(options['input'], b'abc\n')
        self.assertFalse(options.get('shell', False))

    def test_blob_payload_must_match_its_git_object_identifier(self):
        valid_git = self.git
        def changed_blob(root, arguments, *, input_bytes=None):
            payload = valid_git(root, arguments, input_bytes=input_bytes)
            if arguments[0] == 'cat-file':
                payload = payload.replace(b'Never execute', b'Never EXECUTE')
            return payload
        with patch.object(legacy, '_git', side_effect=changed_blob), \
                self.assertRaisesRegex(legacy.HistoricalSourceInvalid, 'checksum mismatch'):
            legacy.load_legacy_source_snapshots(self.root)


if __name__ == '__main__':
    unittest.main()
