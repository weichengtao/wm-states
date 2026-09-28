"""Recording availability stays cheap, stage-aware, and authoritative at launch."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from scripts.next.dashboard.app import create_app
from scripts.next.dashboard.data_status import inspect_data
from scripts.next.dashboard.models import DataStatusRequest, RunRequest
from scripts.next.dashboard.runner import RunManager
from tests.next.test_dashboard_runner import fixture_root


class DashboardDataStatusTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = fixture_root(directory.name)
        self.client = TestClient(create_app(self.root), base_url='http://127.0.0.1:8000')
        self.addCleanup(self.client.close)

    def status(self, **changes):
        response = self.client.post('/api/data-status', json={
            'data_dir': 'data', 'stages': ['select', 'decode'], **changes,
        })
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_endpoint_needs_no_unrelated_run_fields_and_never_loads_mat_or_cache(self):
        (self.root / 'data/sample.mat').write_bytes(b'not a MAT file; metadata only')
        with patch('scipy.io.loadmat', side_effect=AssertionError('loaded MAT')), \
                patch('scripts.next.cache_io.read', side_effect=AssertionError('loaded cache')):
            result = self.status()
        self.assertEqual(result['status'], 'ready')
        self.assertFalse(result['blocking'])
        self.assertEqual(result['file_count'], 1)
        self.assertEqual(result['data_dir'], str((self.root / 'data').resolve()))
        self.assertEqual(result['stages'], [
            {'stage': 'select', 'eligible_count': 1}, {'stage': 'decode', 'eligible_count': 1}])
        self.assertFalse((self.root / 'cache').exists())
        self.assertFalse((self.root / 'configs/next/.dashboard').exists())

    def test_counts_only_regular_top_level_lowercase_mat_files(self):
        (self.root / 'data/one.mat').touch()
        (self.root / 'data/two.MAT').touch()
        (self.root / 'data/pretend.mat').mkdir()
        (self.root / 'data/pretend.mat/nested.mat').touch()
        (self.root / 'data/readme.txt').touch()
        self.assertEqual(self.status()['file_count'], 1)

    def test_missing_non_directory_empty_and_unreadable_are_distinct(self):
        missing = self.status(data_dir='missing')
        self.assertEqual(missing['status'], 'missing')
        self.assertTrue(missing['blocking'])
        (self.root / 'file').touch()
        self.assertEqual(self.status(data_dir='file')['status'], 'missing')
        self.assertEqual(self.status()['status'], 'empty')
        with patch('scripts.next.dashboard.data_status.os.scandir', side_effect=PermissionError):
            unreadable = self.status()
        self.assertEqual(unreadable['status'], 'unreadable')
        self.assertTrue(unreadable['blocking'])

    def test_invalid_paths_and_unknown_stages_have_actionable_status(self):
        for value in ['', '  ', 'data\x00oops']:
            with self.subTest(value=value):
                self.assertEqual(self.status(data_dir=value)['status'], 'invalid')
        self.assertEqual(self.status(stages=['invented'])['status'], 'invalid')
        self.assertEqual(self.status(stages=[])['status'], 'invalid')

    def test_shared_and_stage_allowlists_follow_override_and_null_semantics(self):
        for session in ['one', 'two']:
            (self.root / f'data/{session}.mat').touch()
        (self.root / 'shared.txt').write_text('one # comment\nnot-downloaded\n\n')
        (self.root / 'decode.txt').write_text('two\n')
        result = self.status(session_list_file='shared.txt', settings={
            'select': {'session_list_file': None}, 'decode': {'session_list_file': 'decode.txt'},
        })
        self.assertEqual(result['status'], 'ready')
        self.assertEqual(result['stages'], [
            {'stage': 'select', 'eligible_count': 2}, {'stage': 'decode', 'eligible_count': 1}])
        result = self.status(session_list_file='shared.txt')
        self.assertEqual([item['eligible_count'] for item in result['stages']], [1, 1])

    def test_one_stage_with_zero_matching_sessions_blocks_the_whole_invocation(self):
        (self.root / 'data/one.mat').touch()
        (self.root / 'no-match.txt').write_text('other\n')
        result = self.status(settings={'decode': {'session_list_file': 'no-match.txt'}})
        self.assertEqual(result['status'], 'no_matches')
        self.assertTrue(result['blocking'])
        self.assertEqual(result['stages'][-1], {'stage': 'decode', 'eligible_count': 0})
        self.assertIn('decode', result['message'])

    def test_empty_missing_unreadable_allowlists_are_checked_without_data_loading(self):
        (self.root / 'data/one.mat').touch()
        self.assertEqual(self.status(session_list_file='missing.txt')['status'], 'invalid')
        (self.root / 'empty.txt').write_text('# no selected sessions\n')
        self.assertEqual(self.status(session_list_file='empty.txt')['status'], 'no_matches')
        with patch.object(Path, 'read_text', side_effect=PermissionError):
            result = inspect_data(self.root, DataStatusRequest(
                data_dir='data', stages=['select'], session_list_file='empty.txt'))
        self.assertEqual(result['status'], 'unreadable')

    def test_cache_only_work_is_not_blocked_by_missing_data_or_unused_allowlists(self):
        for stages, settings in [(['evaluate', 'states'], {}), (['models', 'nested-count'], {}),
                                 (['decode'], {'decode': {'plot_only': True,
                                                        'session_list_file': 'missing.txt'}})]:
            with self.subTest(stages=stages):
                result = self.status(data_dir='missing', stages=stages, settings=settings,
                                     session_list_file='missing.txt')
                self.assertEqual(result['status'], 'missing')
                self.assertFalse(result['blocking'])
                self.assertEqual(result['stages'], [])
                self.assertIn('does not prevent launch', result['message'])
                RunManager(self.root).validate(RunRequest(
                    data_dir='missing', cache_dir='cache/new', stages=stages,
                    settings=settings, session_list_file='missing.txt'))

    def test_activity_prepare_criticality_and_legacy_audits_require_recordings(self):
        for stages, settings, trust in [(['activity'], {}, False), (['prepare'], {}, False),
                                        (['criticality'], {}, False), (['evaluate'], {}, True),
                                        (['decode'], {'decode': {'plot_only': True}}, True)]:
            with self.subTest(stages=stages, trust=trust):
                result = self.status(data_dir='missing', stages=stages, settings=settings,
                                     trust_unverified_legacy_results=trust)
                self.assertTrue(result['blocking'])

    def test_plot_only_does_not_mask_selected_screening_or_apply_decode_allowlist(self):
        (self.root / 'data/one.mat').touch()
        result = self.status(settings={'decode': {'plot_only': True, 'session_list_file': 'missing.txt'}})
        self.assertEqual(result['status'], 'ready')
        self.assertEqual(result['stages'], [{'stage': 'select', 'eligible_count': 1}])

    def test_validate_and_start_recheck_stage_overrides_before_writing_any_job(self):
        (self.root / 'data/one.mat').touch()
        (self.root / 'none.txt').write_text('not-available\n')
        for stage in ['select', 'decode']:
            payload = RunRequest(data_dir='data', cache_dir='cache/new', stages=[stage],
                                 settings={stage: {'session_list_file': 'none.txt'}}).model_dump()
            for endpoint in ['/api/validate', '/api/jobs']:
                with self.subTest(stage=stage, endpoint=endpoint):
                    response = self.client.post(endpoint, json=payload)
                    self.assertEqual(response.status_code, 422, response.text)
                    self.assertIn('does not select', response.json()['detail'])
        self.assertFalse((self.root / 'cache').exists())
        self.assertFalse((self.root / 'configs/next/.dashboard').exists())

    def test_mutation_origin_guard_also_protects_read_only_directory_probe(self):
        response = self.client.post('/api/data-status', json={'data_dir': 'data', 'stages': ['select']},
                                    headers={'Origin': 'https://untrusted.example'})
        self.assertEqual(response.status_code, 403)


if __name__ == '__main__':
    unittest.main()
