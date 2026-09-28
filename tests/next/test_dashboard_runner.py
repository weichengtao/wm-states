"""Dashboard contracts: real subprocesses, strict settings and safe local writes."""
import asyncio
import json
from pathlib import Path
import shlex
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from pydantic import ValidationError

from scripts.next import pipeline
from scripts.next.dashboard.app import _FullLogResponse, create_app
from scripts.next.dashboard.models import RunRequest
from scripts.next.dashboard.runner import BusyError, RunManager
from scripts.next.dashboard.schema import get_schema

REPO = Path(__file__).resolve().parents[2]


def fixture_root(directory):
    root = Path(directory)
    (root / 'data').mkdir()
    (root / 'configs/next').mkdir(parents=True)
    for name in ('example', 'smoke'):
        (root / f'configs/next/{name}_pipeline.json').write_bytes(
            (REPO / f'configs/next/{name}_pipeline.json').read_bytes())
    (root / 'scripts/next').mkdir(parents=True)
    return root


def request(**overrides):
    return RunRequest(**dict(name='A quoted "run"', data_dir='data',
                             cache_dir='cache/first', stages=['evaluate'], **overrides))


class DashboardSchemaTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = fixture_root(directory.name)
        self.manager = RunManager(self.root)

    def test_schema_uses_real_fields_and_example_choices(self):
        schema = get_schema(self.root)
        self.assertEqual([stage['id'] for stage in schema['stages']], list(pipeline.STAGES))
        stages = {stage['id']: stage for stage in schema['stages']}
        fields = {field['name']: field for field in stages['decode']['fields']}
        self.assertEqual(fields['grid_search_for_c']['default'], False)
        self.assertTrue(schema['presets']['example']['decode']['grid_search_for_c'])
        self.assertEqual(fields['n_decode_shuffle']['default'], 100)
        self.assertEqual(fields['n_decode_shuffle']['type'], 'integer')
        self.assertEqual(fields['preserve_null_time_structure']['type'], 'boolean')
        self.assertFalse(fields['preserve_null_time_structure']['default'])
        self.assertIn('across time bins', fields['preserve_null_time_structure']['description'])
        self.assertFalse(schema['presets']['example']['decode']['preserve_null_time_structure'])
        self.assertIn('sigmoid', fields['logistic_calibration_method']['choices'])
        self.assertNotIn('cache_dir', fields)
        self.assertEqual(schema['defaults']['settings'], schema['presets']['example'])
        self.assertTrue(next(field for field in stages['activity']['fields']
                             if field['name'] == 'max_points_per_color_group')['nullable'])

    def test_example_resolves_and_command_quotes_paths_without_shell(self):
        (self.root / 'data' / 'sample.mat').touch()
        schema = get_schema(self.root)
        values = {**schema['defaults'], 'data_dir': 'data', 'cache_dir': "cache/a 'quoted' name"}
        plan = self.manager.validate(RunRequest(**values))
        self.assertEqual(list(plan['resolved']), list(pipeline.STAGES))
        self.assertTrue(plan['resolved']['decode']['grid_search_for_c'])
        self.assertEqual(shlex.split(plan['command']), plan['argv'])
        self.assertEqual(plan['argv'][1], '-u')
        self.assertIn(str((self.root / "cache/a 'quoted' name").resolve()), plan['argv'])
        self.assertFalse((self.root / 'cache').exists())
        self.assertFalse(plan['settings_path'].exists())

    def test_unknown_and_wrong_types_fail_even_for_unselected_stage(self):
        cases = [({'typo-stage': {}}, 'Unknown settings stages'),
                 ({'decode': {'n_decode_shuffles': 3}}, 'unknown setting'),
                 ({'decode': {'n_decode_shuffle': True}}, 'expected int'),
                 ({'decode': {'save_figures': 'false'}}, 'expected bool'),
                 ({'decode': {'preserve_null_time_structure': 'false'}}, 'expected bool'),
                 ({'decode': {'classifier_c': float('inf')}}, 'finite'),
                 ({'states': {'cc_method_on': 'invented'}}, 'choose one'),
                 ({'select': {'max_abs_preferred_cue_drift_r': 2}}, 'finite and in'),
                 ({'select': {'temp_dep_r_threshold': 0.3}}, 'unknown setting'),
                 ({'select': {'check_preferred_drift': False}}, 'unknown setting'),
                 ({'select': {'skip_not_applicable_reasons_in_diagnostics_figure': False}}, 'unknown setting'),
                 ({'criticality': {'active_percentiles': [10, 10]}}, 'duplicates'),
                 ({'models': {'history_alpha': 0}}, 'in (0, 1]'),
                 ({'models': {'output_subdir': '../escape'}}, 'owning stage'),
                 ({'models': {'input_filename': '../data.pkl'}}, 'filename'),
                 ({'decode': {'cache_dir': '/tmp/escape'}}, 'shared run')]
        for settings, message in cases:
            with self.subTest(settings=settings), self.assertRaisesRegex(ValueError, __import__('re').escape(message)):
                self.manager.validate(request(settings=settings))

    def test_diagnostic_file_is_required_only_for_selected_extended_screening(self):
        (self.root / 'data/sample.mat').touch()
        settings = {'select': {'save_extended_diagnostics': False,
                               'diagnostics_figure_config': 'configs/next/missing.json'}}
        payload = request(settings=settings).model_copy(update={'stages': ['select']})
        self.manager.validate(payload)
        settings['select']['save_extended_diagnostics'] = True
        self.manager.validate(request(settings=settings))
        with self.assertRaisesRegex(ValueError, 'select.diagnostics_figure_config'):
            self.manager.validate(request(settings=settings).model_copy(update={'stages': ['select']}))
        self.assertFalse((self.root / 'cache').exists())
        self.assertFalse((self.root / 'configs/next/.dashboard').exists())

    def test_invalid_diagnostic_json_is_rejected_before_a_job_can_start(self):
        (self.root / 'data/sample.mat').touch()
        path = self.root / 'configs/next/diagnostics.json'
        payload = request(settings={'select': {
            'save_extended_diagnostics': True,
            'diagnostics_figure_config': 'configs/next/diagnostics.json',
        }}).model_copy(update={'stages': ['select']})
        client = TestClient(create_app(self.root), base_url='http://127.0.0.1:8000')
        self.addCleanup(client.close)
        for content in ('{', json.dumps({'figures': [{'session': 'sample', 'trial_start': 0}]}),
                        json.dumps({'schema_version': 1, 'targets': {'trial_start': 0}}),
                        json.dumps({'schema_version': 1, 'plots': {'enabled': False, 'invented_option': True}})):
            with self.subTest(content=content):
                path.write_text(content)
                for endpoint in ('/api/validate', '/api/jobs'):
                    response = client.post(endpoint, json=payload.model_dump())
                    self.assertEqual(response.status_code, 422, response.text)
                    self.assertIn('select.diagnostics_figure_config', response.json()['detail'])
                self.assertEqual(client.get('/api/jobs').json(), {'jobs': []})
        self.assertFalse((self.root / 'cache').exists())
        self.assertFalse((self.root / 'configs/next/.dashboard').exists())

    def test_diagnostic_config_paths_resolve_from_repository_and_allow_csv_only(self):
        (self.root / 'data/sample.mat').touch()
        path = self.root / 'configs/next/diagnostics.json'
        variants = [
            {'schema_version': 1},
            {'schema_version': 1, 'plots': {'enabled': False}},
            {'schema_version': 1,
             'targets': {'sessions': ['sample'], 'cells': {'start': 0, 'stop': 4},
                         'cells_by_session': {'sample': [0, 2]}},
             'plots': {'enabled': True, 'max_cells_per_session': 2, 'size_inches': [9, 6],
                       'dpi': 150, 'show_not_applicable_reasons': True}},
        ]
        for content in variants:
            path.write_text(json.dumps(content))
            for configured_path in ('configs/next/diagnostics.json', str(path)):
                with self.subTest(content=content, configured_path=configured_path):
                    payload = request(settings={'select': {
                        'save_extended_diagnostics': True,
                        'diagnostics_figure_config': configured_path,
                    }}).model_copy(update={'stages': ['select']})
                    plan = self.manager.validate(payload)
                    self.assertEqual(plan['resolved']['select']['diagnostics_figure_config'], configured_path)
        self.assertFalse((self.root / 'cache').exists())

    def test_home_diagnostic_and_stage_session_paths_match_validation_and_execution(self):
        from scripts.next import cell_screening
        (self.root / 'data/sample.mat').touch()
        diagnostics = self.root / 'diagnostics.json'
        diagnostics.write_text(json.dumps({'schema_version': 1, 'plots': {'enabled': False}}))
        sessions = self.root / 'sessions.txt'
        sessions.write_text('sample\n')
        settings = {'select': {
            'save_extended_diagnostics': True,
            'diagnostics_figure_config': '~/diagnostics.json',
            'session_list_file': '~/sessions.txt',
        }}
        payload = request(settings=settings).model_copy(update={'stages': ['select']})
        original_expanduser = Path.expanduser

        def expand_fixture_home(path):
            return self.root / str(path)[2:] if str(path).startswith('~/') else original_expanduser(path)

        with patch.object(Path, 'expanduser', autospec=True, side_effect=expand_fixture_home):
            plan = self.manager.validate(payload)
            self.assertEqual(plan['resolved']['select']['diagnostics_figure_config'], str(diagnostics))
            self.assertEqual(plan['resolved']['select']['session_list_file'], str(sessions))
            saved_settings = self.root / 'settings.json'
            saved_settings.write_text(json.dumps(settings))
            with patch.object(cell_screening, 'main') as screening:
                pipeline.main(pipeline.Config(cache_dir=self.root / 'cache/home-paths',
                                              data_dir=self.root / 'data', settings=saved_settings,
                                              stages=('select',)))
            resolved = screening.call_args.args[0]
            self.assertEqual(resolved.diagnostics_figure_config, diagnostics)
            self.assertEqual(resolved.session_list_file, sessions)

        missing_home = '~wm_states_missing_user_for_test/recordings'

        def fail_missing_home(path):
            if str(path) == missing_home:
                raise RuntimeError('Could not determine home directory.')
            return original_expanduser(path)

        with patch.object(Path, 'expanduser', autospec=True, side_effect=fail_missing_home), \
                TestClient(create_app(self.root), base_url='http://127.0.0.1:8000') as client:
            response = client.post('/api/validate', json=request().model_copy(
                update={'data_dir': missing_home}).model_dump())
        self.assertEqual(response.status_code, 422, response.text)
        self.assertIn('Cannot resolve the home directory', response.json()['detail'])
        self.assertIn(missing_home, response.json()['detail'])

    def test_cache_confinement_existing_outputs_and_symlinks(self):
        invalid = ['/tmp/escape', 'cache', 'cache/.dashboard', 'cache/../outside', 'cache/run/nested']
        for cache in invalid:
            with self.subTest(cache=cache), self.assertRaises(ValueError):
                self.manager.validate(request().model_copy(update={'cache_dir': cache}))
        destination = self.root / 'cache/first'
        destination.mkdir(parents=True)
        (destination / 'existing.txt').write_text('keep me')
        with self.assertRaisesRegex(ValueError, 'not empty'):
            self.manager.validate(request())
        self.manager.validate(request(allow_existing=True))
        (destination / 'linked').symlink_to(self.root / 'data', target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlinks'):
            self.manager.validate(request(allow_existing=True))
        (destination / 'linked').unlink()
        (self.root / 'cache/escape').symlink_to(self.root / 'data', target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'inside'):
            self.manager.validate(request().model_copy(update={'cache_dir': 'cache/escape'}))

    def test_worker_stage_session_and_null_validation(self):
        for values in ({'n_jobs': 0}, {'n_jobs': True}, {'max_sessions_to_run': 0},
                       {'stages': []}, {'stages': ['select', 'select']}, {'extra': 'wrong'}):
            with self.subTest(values=values), self.assertRaises(ValidationError):
                RunRequest(**values)
        (self.root / 'data/sample.mat').touch()
        with self.assertRaisesRegex(ValueError, 'pipeline order'):
            self.manager.validate(request().model_copy(update={'stages': ['decode', 'select']}))
        with self.assertRaisesRegex(ValueError, 'Unknown stages'):
            self.manager.validate(request().model_copy(update={'stages': ['bad']}))
        with self.assertRaisesRegex(ValueError, 'at least two'):
            self.manager.validate(request(settings={'decode': {'n_decode_shuffle': 1}}).model_copy(
                update={'stages': ['decode', 'states']}))
        sessions = self.root / 'sessions.txt'
        sessions.write_text('missing\n')
        with self.assertRaisesRegex(ValueError, 'does not select'):
            self.manager.validate(request(session_list_file='sessions.txt').model_copy(update={'stages': ['select']}))
        sessions.write_text('sample # selected\nmissing\n')
        self.manager.validate(request(session_list_file='sessions.txt').model_copy(update={'stages': ['select']}))

    def test_legacy_trust_is_explicit_boolean_and_requires_existing_outputs(self):
        self.assertFalse(RunRequest().trust_unverified_legacy_results)
        self.assertFalse(get_schema(self.root)['defaults']['trust_unverified_legacy_results'])
        for value in ('true', 'false', 0, 1, None, []):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                RunRequest(trust_unverified_legacy_results=value)
        self.assertNotIn('--trust-unverified-legacy-results', self.manager.validate(request())['argv'])
        with self.assertRaisesRegex(ValueError, 'requires reuse'):
            self.manager.validate(request(trust_unverified_legacy_results=True))
        with self.assertRaisesRegex(ValueError, 'existing run with decode/decoding_confidence.pkl'):
            self.manager.validate(request(trust_unverified_legacy_results=True, allow_existing=True))
        decoded = self.root / 'cache/first/decode/decoding_confidence.pkl'
        decoded.parent.mkdir(parents=True)
        decoded.write_bytes(b'existing cache placeholder; validation does not unpickle')
        (self.root / 'data/sample.mat').touch()
        enabled = self.manager.validate(request(trust_unverified_legacy_results=True, allow_existing=True))
        self.assertEqual(enabled['argv'].count('--trust-unverified-legacy-results'), 1)
        self.assertIn('--trust-unverified-legacy-results', shlex.split(enabled['command']))
        disabled = self.manager.validate(request(allow_existing=True))
        self.assertNotIn('--trust-unverified-legacy-results', disabled['argv'])

    def test_restored_crashed_job_does_not_count_offline_time_as_stage_runtime(self):
        job_id = 'd' * 32
        self.manager.storage.mkdir(parents=True)
        record = {
            'id': job_id, 'status': 'running', 'created_at': '2026-09-27T01:00:00Z',
            'started_at': '2026-09-27T01:00:00Z', 'finished_at': None,
            'stages': [
                {'stage': 'select', 'status': 'complete', 'seconds': 3.125},
                {'stage': 'decode', 'status': 'running',
                 'started_at': '2026-09-27T01:00:03Z', 'finished_at': None},
            ],
        }
        path = self.manager.storage / f'{job_id}.json'
        path.write_text(json.dumps(record))
        with patch('scripts.next.dashboard.runner.now', return_value='2026-09-29T01:00:00Z'):
            restored = RunManager(self.root).snapshot(job_id)
        self.assertEqual(restored['status'], 'failed')
        self.assertEqual(restored['stages'][0], record['stages'][0])
        self.assertEqual(restored['stages'][1]['started_at'], record['stages'][1]['started_at'])
        self.assertTrue(restored['stages'][1]['elapsed_unavailable'])
        self.assertIsNone(restored['stages'][1]['finished_at'])
        self.assertNotIn('seconds', restored['stages'][1])
        self.assertTrue(json.loads(path.read_text())['stages'][1]['elapsed_unavailable'])
        # A subsequent restart must preserve the annotation and recovery time.
        again = RunManager(self.root).snapshot(job_id)
        self.assertEqual(again['stages'], restored['stages'])
        self.assertEqual(again['finished_at'], restored['finished_at'])


class DashboardApiTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = fixture_root(directory.name)
        self.app = create_app(self.root)
        self.client = TestClient(self.app, base_url='http://127.0.0.1:8000')
        self.addCleanup(self.client.close)

    def test_health_schema_validate_and_errors(self):
        self.assertEqual(self.client.get('/api/health').json()['status'], 'ok')
        self.assertEqual(len(self.client.get('/api/schema').json()['stages']), 11)
        response = self.client.post('/api/validate', json=request().model_dump())
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()['valid'])
        self.assertIn('evaluate', response.json()['resolved'])
        response = self.client.post('/api/validate', json={'n_jobs': '2'})
        self.assertEqual(response.status_code, 422)
        self.assertIsInstance(response.json()['detail'], str)
        self.assertEqual(self.client.get('/api/jobs').json(), {'jobs': []})
        self.assertEqual(self.client.get('/api/jobs/absent').status_code, 404)
        self.assertEqual(self.client.post('/api/jobs/absent/cancel').status_code, 404)
        self.assertEqual(self.client.get('/api/not-an-endpoint').status_code, 404)
        self.assertEqual(self.client.get('/').status_code, 503)

    def test_pdf_format_is_validated_and_forwarded_to_the_pipeline(self):
        for formats in (['pdf'], ['png', 'pdf'], ['png', 'tif', 'eps', 'pdf']):
            with self.subTest(formats=formats):
                response = self.client.post('/api/validate', json=request(figure_formats=formats).model_dump())
                self.assertEqual(response.status_code, 200, response.text)
                arguments = shlex.split(response.json()['command'])
                index = arguments.index('--figure-formats')
                self.assertEqual(arguments[index + 1:index + 1 + len(formats)], formats)
        for formats in ([], ['pdf', 'pdf'], ['svg'], ['PDF']):
            with self.subTest(formats=formats):
                payload = request().model_dump()
                payload['figure_formats'] = formats
                self.assertEqual(self.client.post('/api/validate', json=payload).status_code, 422)
        schema = self.client.get('/api/openapi.json').json()
        choices = schema['components']['schemas']['RunRequest']['properties']['figure_formats']['items']['enum']
        self.assertIn('pdf', choices)

    def test_cross_origin_mutations_and_dns_rebinding_are_rejected(self):
        payload = request().model_dump()
        for origin in ('https://untrusted.example', 'null', 'http://127.0.0.1:9999'):
            with self.subTest(origin=origin):
                response = self.client.post('/api/validate', json=payload, headers={'Origin': origin})
                self.assertEqual(response.status_code, 403)
        for origin in ('http://127.0.0.1:8000', 'http://localhost:5173'):
            response = self.client.post('/api/validate', json=payload, headers={'Origin': origin})
            self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.get('/api/health', headers={'Host': 'evil.example'}).status_code, 400)
        self.assertEqual(self.client.get('/api/health').headers['Cache-Control'], 'no-store')

    def test_completed_websocket_returns_real_snapshot(self):
        manager = self.app.state.runner
        manager.jobs['completed'] = dict(id='completed', status='complete', logs=['real log'],
                                        created_at='2026-01-01T00:00:00Z', stages=[])
        with self.client.websocket_connect('ws://127.0.0.1:8000/api/jobs/completed/events') as websocket:
            snapshot = websocket.receive_json()
            self.assertEqual(snapshot['status'], 'complete')
            self.assertEqual(snapshot['logs'], ['real log'])

    def test_full_log_download_includes_more_than_the_live_preview(self):
        manager = self.app.state.runner
        job_id = 'a' * 32
        manager.jobs[job_id] = dict(id=job_id, status='running', logs=['latest line'])
        manager.storage.mkdir(parents=True)
        path = manager.storage / f'{job_id}.log'
        payload = ''.join(f'line {index}\n' for index in range(650)).encode()
        path.write_bytes(payload)
        response = self.client.get(f'/api/jobs/{job_id}/log')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, payload)
        self.assertIn(f'{job_id}.log', response.headers['content-disposition'])
        self.assertEqual(response.headers['content-length'], str(len(payload)))
        head = self.client.head(f'/api/jobs/{job_id}/log')
        self.assertEqual(head.content, b'')
        self.assertEqual(head.headers['content-length'], str(len(payload)))
        path.write_bytes(b'')
        self.assertEqual(self.client.get(f'/api/jobs/{job_id}/log').content, b'')
        self.assertEqual(self.client.get('/api/jobs/unknown/log').status_code, 404)
        path.unlink()
        self.assertEqual(self.client.get(f'/api/jobs/{job_id}/log').status_code, 404)

    def test_built_frontend_and_spa_route(self):
        dist = self.root / 'dashboard/dist'
        dist.mkdir(parents=True)
        (dist / 'index.html').write_text('<html>Dashboard</html>')
        response = self.client.get('/runs/a-run')
        self.assertEqual(response.status_code, 200)
        self.assertIn('Dashboard', response.text)


class DashboardProcessTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = fixture_root(self.directory.name)
        self.manager = RunManager(self.root)

    async def asyncTearDown(self):
        await self.manager.close()
        self.directory.cleanup()

    def script(self, source):
        (self.root / 'scripts/next/pipeline.py').write_text(source)

    async def wait_running(self, job_id):
        for _ in range(100):
            if self.manager.snapshot(job_id)['status'] == 'running':
                return
            await asyncio.sleep(.01)
        self.fail('Job did not start')

    async def test_log_download_stops_at_initial_length_and_closes_on_disconnect(self):
        path = self.root / 'growing.log'
        payload = b'initial output\n' * 10_000
        path.write_bytes(payload)
        response = _FullLogResponse(path, 'a' * 32)
        messages = []

        async def send(message):
            messages.append(message)
            if message['type'] == 'http.response.start':
                with path.open('ab') as stream:
                    stream.write(b'output after download started\n')

        async def receive():
            return {'type': 'http.disconnect'}

        scope = {'type': 'http', 'method': 'GET', 'asgi': {'spec_version': '2.4'}}
        await response(scope, receive, send)
        self.assertEqual(b''.join(message.get('body', b'') for message in messages), payload)
        self.assertTrue(response.stream.closed)

        response = _FullLogResponse(path, 'a' * 32)
        async def disconnected(message):
            raise OSError('client left')
        with self.assertRaises(Exception):
            await response(scope, receive, disconnected)
        self.assertTrue(response.stream.closed)

    async def test_real_process_output_progress_and_restoration(self):
        self.script('''import json, pathlib, sys, time
args = sys.argv
cache = pathlib.Path(args[args.index('--cache-dir') + 1])
cache.mkdir(parents=True, exist_ok=True)
record = {'run_id': 'real-invocation', 'invocation': {'argv': sys.orig_argv},
          'stages': [{'stage':'evaluate','status':'running'}]}
(cache / 'pipeline_manifest.json').write_text(json.dumps(record))
print('evaluating sessions', flush=True)
time.sleep(.2)
record['stages'][0].update(status='complete', seconds=.2)
(cache / 'pipeline_manifest.json').write_text(json.dumps(record))
print('evaluation done', flush=True)
''')
        job = await self.manager.start(request())
        self.assertEqual(job['status'], 'queued')
        await asyncio.wait_for(self.manager.tasks[job['id']], 5)
        final = self.manager.snapshot(job['id'])
        self.assertEqual(final['status'], 'complete')
        self.assertEqual(final['exit_code'], 0)
        self.assertEqual(final['manifest_id'], 'real-invocation')
        self.assertEqual(final['stages'][0]['status'], 'complete')
        self.assertEqual(final['logs'], ['evaluating sessions', 'evaluation done'])
        self.assertEqual(json.loads((self.root / f"configs/next/.dashboard/{job['id']}.json").read_text()), {})
        self.assertEqual(Path(final['run_log_path']).read_bytes(), self.manager.log_path(job['id']).read_bytes())
        self.assertEqual(Path(final['run_record_path']).read_bytes(),
                         (self.manager.storage / f"{job['id']}.json").read_bytes())
        self.assertEqual(json.loads(Path(final['run_settings_path']).read_text()), {})
        self.assertEqual(json.loads(Path(final['run_record_path']).read_text())['request'], request().model_dump())
        restored = RunManager(self.root).snapshot(job['id'])
        self.assertEqual(restored['status'], 'complete')
        self.assertEqual(restored['logs'], final['logs'])

    async def test_initial_state_write_failure_does_not_leave_a_queued_job(self):
        with patch.object(self.manager, '_save', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError, 'disk full'):
                await self.manager.start(request())
        self.assertFalse(self.manager.jobs)
        self.assertFalse(self.manager.tasks)
        self.script('print("ok")\n')
        job = await self.manager.start(request())
        await self.manager.tasks[job['id']]
        self.assertEqual(self.manager.snapshot(job['id'])['status'], 'complete')

    async def test_start_failure_is_visible_and_does_not_lock_queue(self):
        with patch('asyncio.create_subprocess_exec', side_effect=OSError('launch unavailable')):
            job = await self.manager.start(request())
            await self.manager.tasks[job['id']]
        final = self.manager.snapshot(job['id'])
        self.assertEqual(final['status'], 'failed')
        self.assertIn('launch unavailable', final['error'])
        self.script('raise SystemExit(7)\n')
        self.assertIn('launch unavailable', Path(final['run_log_path']).read_text())
        second = await self.manager.start(request(allow_existing=True))
        await self.manager.tasks[second['id']]
        self.assertEqual(self.manager.snapshot(second['id'])['exit_code'], 7)

    async def test_one_active_process_and_cancel_entire_process_group(self):
        self.script('import time\nprint("processing", flush=True)\ntime.sleep(60)\n')
        job = await self.manager.start(request())
        await self.wait_running(job['id'])
        with self.assertRaises(BusyError):
            await self.manager.start(request())
        with patch.object(self.manager, '_signal_group', wraps=self.manager._signal_group) as signal_group:
            cancelled = await asyncio.wait_for(self.manager.cancel(job['id']), 8)
        self.assertTrue(signal_group.called)
        self.assertEqual(cancelled['status'], 'cancelled')
        self.assertIsNotNone(cancelled['finished_at'])
        self.assertNotIn(job['id'], self.manager.processes)
        self.assertTrue(self.manager.tasks[job['id']].done())
        self.assertEqual((await self.manager.cancel(job['id']))['status'], 'cancelled')

    async def test_post_spawn_storage_error_stops_and_reaps_process(self):
        self.script('import time\nprint("processing", flush=True)\ntime.sleep(60)\n')
        job = await self.manager.start(request())
        real_save = self.manager._save
        def fail_running(value):
            if value['status'] == 'running':
                raise OSError('disk unavailable')
            real_save(value)
        with patch.object(self.manager, '_save', side_effect=fail_running), \
             patch.object(self.manager, '_signal_group', wraps=self.manager._signal_group) as signal_group:
            await asyncio.wait_for(self.manager.tasks[job['id']], 5)
        final = self.manager.snapshot(job['id'])
        self.assertEqual(final['status'], 'failed')
        self.assertIn('disk unavailable', final['error'])
        self.assertIsNotNone(final['exit_code'])
        self.assertTrue(signal_group.called)
        self.assertFalse(self.manager.processes)

    async def test_full_log_is_mirrored_while_running_and_reruns_preserve_history(self):
        self.script('''import json, pathlib, sys, time
cache = pathlib.Path(sys.argv[sys.argv.index('--cache-dir') + 1])
record = {'run_id': 'current', 'invocation': {'argv': sys.orig_argv},
          'stages': [{'stage': 'evaluate', 'status': 'running',
                      'started_at': '2026-09-27T01:00:00Z', 'finished_at': None}]}
(cache / 'pipeline_manifest.json').write_text(json.dumps(record))
for number in range(650):
    print(f'line {number}', flush=True)
print('standard error', file=sys.stderr, flush=True)
while not (cache / 'continue').exists():
    time.sleep(.01)
''')
        settings = {'decode': {'n_decode_shuffle': 3}}
        job = await self.manager.start(request(settings=settings))
        await self.wait_running(job['id'])
        for _ in range(100):
            if self.manager.jobs[job['id']]['logs'][-1:] == ['standard error']:
                break
            await asyncio.sleep(.01)
        else:
            self.fail('Full process output did not arrive')
        live = self.manager.snapshot(job['id'])
        self.assertEqual(live['status'], 'running')
        self.assertEqual(len(live['logs']), 500)
        expected = ''.join(f'line {number}\n' for number in range(650)) + 'standard error\n'
        self.assertEqual(self.manager.log_path(job['id']).read_text(), expected)
        self.assertEqual(Path(live['run_log_path']).read_text(), expected)
        self.assertEqual(json.loads(Path(live['run_record_path']).read_text())['status'], 'running')
        self.assertEqual(json.loads(Path(live['run_record_path']).read_text())['stages'][0]['started_at'],
                         '2026-09-27T01:00:00Z')
        self.assertEqual(json.loads(Path(live['run_settings_path']).read_text()), settings)
        (Path(live['cache_dir']) / 'continue').touch()
        await asyncio.wait_for(self.manager.tasks[job['id']], 5)
        originals = {key: Path(live[key]).read_bytes()
                     for key in ('run_record_path', 'run_log_path', 'run_settings_path')}
        self.script('print("second invocation")\n')
        second = await self.manager.start(request(allow_existing=True))
        await asyncio.wait_for(self.manager.tasks[second['id']], 5)
        self.assertNotEqual(job['id'], second['id'])
        self.assertEqual(len(list((Path(live['cache_dir']) / 'dashboard').glob('*.log'))), 2)
        for key, payload in originals.items():
            self.assertEqual(Path(live[key]).read_bytes(), payload)

    async def test_manual_trust_flag_and_live_audit_are_mirrored_without_stage_change(self):
        (self.root / 'data/fixture.mat').touch()
        decoded = self.root / 'cache/first/decode/decoding_confidence.pkl'
        decoded.parent.mkdir(parents=True)
        decoded.write_bytes(b'existing fixture cache')
        self.script('''import json, pathlib, sys, time
cache = pathlib.Path(sys.argv[sys.argv.index('--cache-dir') + 1])
record = {'run_id': 'trusted-invocation', 'invocation': {'argv': sys.orig_argv},
          'stages': [{'stage': 'evaluate', 'status': 'running'}],
          'legacy_trust': {'enabled': True, 'manual_trust_used': False, 'events': []}}
manifest = cache / 'pipeline_manifest.json'
manifest.write_text(json.dumps(record))
print('initial audit', flush=True)
while not (cache / 'record_trust').exists():
    time.sleep(.01)
record['legacy_trust'].update(manual_trust_used=True, events=[{
    'stage': 'evaluate', 'session': 'fixture', 'manual_trust': True,
    'original_fingerprint': 'b' * 64, 'reason': 'fixture verification failure',
    'verification_status': 'unverified-legacy'}])
manifest.write_text(json.dumps(record))
print('trust recorded', flush=True)
while not (cache / 'finish').exists():
    time.sleep(.01)
record['stages'][0]['status'] = 'complete'
manifest.write_text(json.dumps(record))
''')
        job = await self.manager.start(request(allow_existing=True, trust_unverified_legacy_results=True))
        self.assertTrue(job['trust_unverified_legacy_results'])
        self.assertTrue(json.loads(Path(job['run_record_path']).read_text())['request']['trust_unverified_legacy_results'])
        for _ in range(100):
            if self.manager.jobs[job['id']]['logs'][-1:] == ['initial audit']:
                break
            await asyncio.sleep(.01)
        else:
            self.fail('Initial trust audit did not arrive')
        initial = self.manager.snapshot(job['id'])
        self.assertFalse(initial['legacy_trust']['manual_trust_used'])
        (Path(job['cache_dir']) / 'record_trust').touch()
        for _ in range(100):
            if self.manager.jobs[job['id']]['logs'][-1:] == ['trust recorded']:
                break
            await asyncio.sleep(.01)
        else:
            self.fail('Updated trust audit did not arrive')
        updated = self.manager.snapshot(job['id'])
        self.assertEqual(updated['stages'], initial['stages'])
        self.assertTrue(updated['legacy_trust']['manual_trust_used'])
        self.assertEqual(updated['legacy_trust']['events'][0]['original_fingerprint'], 'b' * 64)
        for record_path in (Path(updated['run_record_path']), self.manager.storage / f"{job['id']}.json"):
            persisted = json.loads(record_path.read_text())
            self.assertTrue(persisted['trust_unverified_legacy_results'])
            self.assertEqual(persisted['legacy_trust'], updated['legacy_trust'])
        (Path(job['cache_dir']) / 'finish').touch()
        await asyncio.wait_for(self.manager.tasks[job['id']], 5)
        final = self.manager.snapshot(job['id'])
        self.assertEqual(final['status'], 'complete')
        restored = RunManager(self.root).snapshot(job['id'])
        self.assertTrue(restored['trust_unverified_legacy_results'])
        self.assertEqual(restored['legacy_trust'], final['legacy_trust'])

    async def test_log_paths_reject_unknown_jobs_and_symlink_files(self):
        self.script('print("recorded")\n')
        job = await self.manager.start(request())
        await self.manager.tasks[job['id']]
        with self.assertRaises(KeyError):
            self.manager.log_path('../outside')
        log = self.manager.log_path(job['id'])
        log.unlink()
        log.symlink_to(Path(job['run_log_path']))
        with self.assertRaises(FileNotFoundError):
            self.manager.log_path(job['id'])

    async def test_queued_cancel_and_old_manifest_do_not_fake_progress(self):
        cache = self.root / 'cache/first'
        cache.mkdir(parents=True)
        (cache / 'pipeline_manifest.json').write_text(json.dumps({
            'run_id': 'previous', 'invocation': {'argv': ['old-command']},
            'stages': [{'stage': 'evaluate', 'status': 'complete'}],
        }))
        self.script('import time\ntime.sleep(60)\n')
        job = await self.manager.start(request(allow_existing=True))
        self.assertEqual(job['stages'][0]['status'], 'pending')
        self.assertIsNone(job['manifest_id'])
        self.assertIsNone(job.get('legacy_trust'))
        final = await self.manager.cancel(job['id'])
        self.assertEqual(final['status'], 'cancelled')
        self.assertEqual(final['stages'][0]['status'], 'pending')
        self.assertFalse(self.manager.processes)
        self.assertEqual(Path(final['run_log_path']).read_bytes(), b'')
        self.assertEqual(self.manager.log_path(job['id']).read_bytes(), b'')


if __name__ == '__main__':
    unittest.main()
