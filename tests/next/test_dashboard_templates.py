"""Saved templates validate settings without touching jobs, inputs, or outputs."""
from concurrent.futures import ThreadPoolExecutor
import copy
import errno
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from scripts.next.dashboard.app import create_app
from scripts.next.dashboard.models import RunRequest


REPO = Path(__file__).resolve().parents[2]


def fixture_root(directory):
    root = Path(directory)
    (root / 'configs/next').mkdir(parents=True)
    for name in ('example', 'smoke'):
        shutil.copyfile(REPO / f'configs/next/{name}_pipeline.json',
                        root / f'configs/next/{name}_pipeline.json')
    return root


def payload(**overrides):
    return {
        'name': 'My analysis template',
        'description': 'Reusable settings, including explicit nulls.',
        'config': {
            'settings': {'activity': {'max_points_per_color_group': None}},
            'stages': ['evaluate'], 'n_jobs': -1, 'max_sessions_to_run': None,
            'figure_formats': ['png', 'pdf'],
        },
        **overrides,
    }


class DashboardTemplateTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = fixture_root(directory.name)
        self.app = create_app(self.root)
        self.client = TestClient(self.app, base_url='http://127.0.0.1:8000')
        self.addCleanup(self.client.close)
        self.storage = self.root / 'configs/next/templates'

    def post(self, value=None):
        return self.client.post('/api/templates', json=value or payload())

    def test_builtins_come_from_current_files_and_shared_run_defaults_without_paths(self):
        response = self.client.get('/api/templates')
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['warnings'], [])
        records = response.json()['templates']
        self.assertEqual([record['id'] for record in records], ['example', 'smoke'])
        defaults = RunRequest().model_dump()
        for record in records:
            self.assertTrue(record['builtin'])
            config = record['config']
            self.assertEqual(config['settings'], json.loads((self.root / record['path']).read_text()))
            for field in ('stages', 'n_jobs', 'max_sessions_to_run', 'figure_formats'):
                self.assertEqual(config[field], defaults[field])
            for field in ('data_dir', 'session_list_file', 'cache_dir', 'name', 'allow_existing'):
                self.assertNotIn(field, config)
        path = self.root / 'configs/next/smoke_pipeline.json'
        source = json.loads(path.read_text())
        source['decode']['seed'] = 71
        path.write_text(json.dumps(source))
        refreshed = self.client.get('/api/templates').json()['templates'][1]
        self.assertEqual(refreshed['config']['settings']['decode']['seed'], 71)
        self.assertFalse(self.storage.exists())
        self.assertFalse((self.root / 'cache').exists())

    def test_save_is_durable_and_preserves_sparse_config_nulls_and_enum_spelling(self):
        request = payload()
        request['config']['settings']['decode'] = {'decoder_model': 'LOGISTIC_REGRESSION', 'classifier_c': 1}
        response = self.post(request)
        self.assertEqual(response.status_code, 201, response.text)
        saved = response.json()
        self.assertRegex(saved['id'], r'^[0-9a-f]{32}$')
        self.assertEqual(saved['config'], request['config'])
        self.assertEqual(saved['description'], request['description'])
        self.assertFalse(saved['builtin'])
        self.assertTrue(saved['created_at'].endswith('+00:00'))
        self.assertEqual(saved['path'], f"configs/next/templates/{saved['id']}.json")
        envelope = json.loads((self.root / saved['path']).read_text())
        self.assertEqual(envelope['schema_version'], 1)
        self.assertEqual(envelope['config'], request['config'])
        self.assertNotIn('builtin', envelope)
        self.assertNotIn('path', envelope)
        reopened = TestClient(create_app(self.root), base_url='http://127.0.0.1:8000')
        self.addCleanup(reopened.close)
        listing = reopened.get('/api/templates').json()
        self.assertEqual(listing['warnings'], [])
        self.assertEqual(listing['templates'][2], saved)
        self.assertFalse((self.root / 'cache').exists())
        self.assertFalse((self.root / 'configs/next/.dashboard').exists())

    def test_save_does_not_require_recordings_diagnostics_or_an_idle_runner(self):
        request = payload()
        request['config'].update(stages=['select'], data_dir='/not-mounted/recordings',
                                 session_list_file='missing-sessions.txt')
        request['config']['settings']['select'] = {
            'save_extended_diagnostics': True,
            'diagnostics_figure_config': 'missing-diagnostics.json',
        }
        active = {'id': 'existing-job', 'status': 'running', 'unrelated': ['keep this']}
        self.app.state.runner.jobs['existing-job'] = copy.deepcopy(active)
        with patch.object(self.app.state.runner, 'validate', side_effect=AssertionError('Must not validate a job')), \
                patch.object(self.app.state.runner, 'start', side_effect=AssertionError('Must not start a job')):
            response = self.post(request)
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(response.json()['config'], request['config'])
        self.assertEqual(self.app.state.runner.jobs, {'existing-job': active})
        self.assertFalse(self.app.state.runner.processes)
        self.assertFalse(self.app.state.runner.tasks)
        self.assertFalse((self.root / 'data').exists())
        self.assertFalse((self.root / 'cache').exists())
        request['name'] = 'Recording path for another computer'
        request['config']['data_dir'] = '~wm_states_nonexistent_user_20260926/recordings'
        response = self.post(request)
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(response.json()['config']['data_dir'], request['config']['data_dir'])
        self.assertEqual(self.app.state.runner.jobs, {'existing-job': active})

    def test_omitted_recording_paths_and_explicit_null_session_list_stay_distinct(self):
        first = self.post().json()
        self.assertNotIn('data_dir', first['config'])
        self.assertNotIn('session_list_file', first['config'])
        request = payload(name='Clear the session list')
        request['config']['session_list_file'] = None
        request['config']['data_dir'] = 'data/my recordings'
        second = self.post(request)
        self.assertEqual(second.status_code, 201, second.text)
        self.assertEqual(second.json()['config']['data_dir'], 'data/my recordings')
        self.assertIn('session_list_file', second.json()['config'])
        self.assertIsNone(second.json()['config']['session_list_file'])
        minimal = payload(name='Default description')
        del minimal['description']
        response = self.post(minimal)
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(response.json()['description'], '')

    def test_invalid_request_settings_limits_and_stage_order_are_rejected(self):
        cases = [
            {'stages': []}, {'stages': ['evaluate', 'evaluate']},
            {'stages': ['decode', 'select']}, {'stages': ['unknown']},
            {'n_jobs': 0}, {'n_jobs': True}, {'n_jobs': '2'},
            {'max_sessions_to_run': 0}, {'max_sessions_to_run': False},
            {'figure_formats': []}, {'figure_formats': ['png', 'png']},
            {'figure_formats': ['svg']}, {'data_dir': None}, {'data_dir': ''},
            {'session_list_file': ' '}, {'session_list_file': 12},
            {'settings': {'unknown': {}}},
            {'settings': {'decode': {'n_decode_shuffle': True}}},
            {'settings': {'decode': {'n_decode_shuffles': 2}}},
            {'settings': {'decode': {'n_decode_shuffle': -1}}},
            {'settings': {'select': {'max_abs_baseline_drift_r': 2}}},
            {'settings': {'models': {'significance_alpha': 1}}},
            {'settings': {'models': {'output_subdir': '../outside'}}},
            {'settings': {'decode': {'cache_dir': 'cache/override'}}},
            {'stages': ['decode', 'states'], 'settings': {'decode': {'n_decode_shuffle': 1}}},
            {'cache_dir': 'cache/should-not-be-saved'},
            {'name': 'a run name'}, {'allow_existing': True},
        ]
        for overrides in cases:
            with self.subTest(overrides=overrides):
                request = payload()
                request['config'].update(overrides)
                response = self.post(request)
                self.assertEqual(response.status_code, 422, response.text)
        for field in ('settings', 'stages', 'n_jobs', 'max_sessions_to_run', 'figure_formats'):
            request = payload()
            del request['config'][field]
            self.assertEqual(self.post(request).status_code, 422)
        self.assertFalse(self.storage.exists())

    def test_duplicate_names_are_case_insensitive_and_builtins_remain_immutable(self):
        response = self.post()
        self.assertEqual(response.status_code, 201, response.text)
        saved = response.json()
        original = (self.root / saved['path']).read_bytes()
        builtins = {name: (self.root / f'configs/next/{name}_pipeline.json').read_bytes()
                    for name in ('example', 'smoke')}
        for name in ('MY ANALYSIS TEMPLATE', '  My analysis template  ', 'example PIPELINE', 'Smoke test'):
            with self.subTest(name=name):
                response = self.post(payload(name=name))
                self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual((self.root / saved['path']).read_bytes(), original)
        for name, value in builtins.items():
            self.assertEqual((self.root / f'configs/next/{name}_pipeline.json').read_bytes(), value)
        self.assertEqual(len(list(self.storage.glob('*.json'))), 1)

    def test_client_cannot_choose_ids_storage_paths_or_overwrite_metadata(self):
        for extra in ({'id': 'example'}, {'path': '../../outside.json'}, {'builtin': True},
                      {'schema_version': 1}, {'created_at': '2026-01-01T00:00:00Z'}):
            with self.subTest(extra=extra):
                self.assertEqual(self.post(payload(**extra)).status_code, 422)
        for name in ('', ' ', 'bad\x00name'):
            self.assertEqual(self.post(payload(name=name)).status_code, 422)
        response = self.post(payload(name='../../display name'))
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual((self.root / response.json()['path']).parent, self.storage)
        self.assertEqual(len(list(self.root.rglob('*.json'))), 3)

    def test_corrupt_unsupported_and_mismatched_saved_files_warn_and_are_skipped(self):
        saved = self.post().json()
        envelope = json.loads((self.root / saved['path']).read_text())
        corrupt_id, version_id, mismatch_id, invalid_id = [uuid4().hex for _ in range(4)]
        (self.storage / f'{corrupt_id}.json').write_text('{')
        (self.storage / f'{version_id}.json').write_text(json.dumps({**envelope, 'id': version_id, 'schema_version': 2}))
        (self.storage / f'{mismatch_id}.json').write_text(json.dumps(envelope))
        invalid = {**envelope, 'id': invalid_id, 'name': 'Invalid settings',
                   'config': {**envelope['config'], 'n_jobs': 0}}
        (self.storage / f'{invalid_id}.json').write_text(json.dumps(invalid))
        listing = self.client.get('/api/templates').json()
        self.assertEqual([record['id'] for record in listing['templates']], ['example', 'smoke', saved['id']])
        self.assertEqual(len(listing['warnings']), 4)
        for identifier in (corrupt_id, version_id, mismatch_id, invalid_id):
            self.assertTrue(any(identifier in warning for warning in listing['warnings']))
        self.assertEqual(self.post(payload(name='Still save valid settings')).status_code, 201)

    def test_symlinked_template_files_and_locks_are_never_followed(self):
        saved = self.post().json()
        envelope = json.loads((self.root / saved['path']).read_text())
        identifier = uuid4().hex
        outside = self.root / 'outside.json'
        outside.write_text(json.dumps({**envelope, 'id': identifier, 'name': 'Outside template'}))
        original = outside.read_bytes()
        (self.storage / f'{identifier}.json').symlink_to(outside)
        listing = self.client.get('/api/templates').json()
        self.assertEqual(len(listing['templates']), 3)
        self.assertTrue(any(identifier in warning for warning in listing['warnings']))
        (self.storage / '.templates.lock').unlink()
        (self.storage / '.templates.lock').symlink_to(outside)
        self.assertEqual(self.post(payload(name='Do not follow the lock')).status_code, 422)
        self.assertEqual(outside.read_bytes(), original)

    def test_symlinked_storage_ancestors_block_saves_without_writing_elsewhere(self):
        for index, relative in enumerate(('configs', 'configs/next', 'configs/next/templates')):
            with self.subTest(relative=relative):
                root = fixture_root(self.root / f'case-{index}')
                destination = self.root / f'outside-{index}'
                destination.mkdir()
                linked = root / relative
                if linked.exists():
                    shutil.rmtree(linked)
                linked.symlink_to(destination, target_is_directory=True)
                client = TestClient(create_app(root), base_url='http://127.0.0.1:8000')
                self.addCleanup(client.close)
                response = client.post('/api/templates', json=payload())
                self.assertEqual(response.status_code, 422, response.text)
                self.assertIn('symlinks', response.json()['detail'])
                self.assertEqual(list(destination.iterdir()), [])
                self.assertTrue(client.get('/api/templates').json()['warnings'])

    def test_id_collision_and_failed_publication_never_overwrite_a_saved_file(self):
        first = self.post().json()
        first_path = self.root / first['path']
        original = first_path.read_bytes()
        next_id = uuid4().hex
        identifiers = [UUID(first['id']), uuid4(), UUID(next_id), uuid4()]
        with patch('scripts.next.dashboard.templates.uuid4', side_effect=identifiers):
            response = self.post(payload(name='Second template'))
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(response.json()['id'], next_id)
        self.assertEqual(first_path.read_bytes(), original)
        with patch('scripts.next.dashboard.templates.os.link', side_effect=OSError(errno.ENOSPC, 'Disk full')):
            response = self.post(payload(name='No partial file'))
        self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(len(list(self.storage.glob('*.json'))), 2)
        self.assertFalse(list(self.storage.glob('.*.tmp')))
        self.assertEqual(first_path.read_bytes(), original)

    def test_concurrent_duplicate_saves_publish_only_one_complete_template(self):
        with ThreadPoolExecutor(max_workers=2) as executor:
            responses = list(executor.map(lambda name: self.post(payload(name=name)), ('Parallel', 'PARALLEL')))
        self.assertEqual(sorted(response.status_code for response in responses), [201, 409],
                         [response.json() for response in responses])
        self.assertEqual(len(list(self.storage.glob('*.json'))), 1)
        listing = self.client.get('/api/templates').json()
        self.assertEqual(len(listing['templates']), 3)
        self.assertEqual(listing['warnings'], [])
        self.assertFalse(list(self.storage.glob('.*.tmp')))

    def test_cross_origin_template_mutations_are_rejected(self):
        response = self.client.post('/api/templates', json=payload(), headers={'Origin': 'https://untrusted.example'})
        self.assertEqual(response.status_code, 403)
        self.assertFalse(self.storage.exists())


if __name__ == '__main__':
    unittest.main()
