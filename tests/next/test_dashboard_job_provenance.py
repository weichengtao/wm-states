"""Source-template history follows the exact saved invocation, including restored runs."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError

from scripts.next.dashboard.models import RunRequest
from scripts.next.dashboard.results import ResultStore
from scripts.next.dashboard.runner import RunManager
from tests.next.test_dashboard_runner import fixture_root


def template():
    return {'id': 'example', 'name': 'Example pipeline', 'description': 'Original choices.',
            'builtin': True, 'path': 'configs/next/example_pipeline.json',
            'config': {'settings': {'decode': {'n_decode_shuffle': 100}}, 'stages': ['decode'],
                       'n_jobs': 1, 'max_sessions_to_run': None, 'figure_formats': ['png']}}


class SourceTemplateModelTests(unittest.TestCase):
    def test_snapshot_preserves_sparse_fields_without_accessing_external_templates(self):
        snapshot = template()
        request = RunRequest(source_template=snapshot)
        self.assertEqual(request.model_dump()['source_template'], snapshot)
        self.assertEqual(RunRequest.model_validate(request.model_dump()).model_dump()['source_template'], snapshot)
        self.assertNotIn('data_dir', request.model_dump()['source_template']['config'])
        self.assertNotIn('session_list_file', request.model_dump()['source_template']['config'])
        self.assertIsNone(RunRequest().source_template)

    def test_invalid_snapshots_are_rejected_and_cannot_include_run_permissions(self):
        cases = [None, 'example', {'id': 'example'}, {**template(), 'builtin': 'true'},
                 {**template(), 'name': '  '}, {**template(), 'id': ''},
                 {**template(), 'created_at': 'not-a-date'},
                 {**template(), 'created_at': '2026-09-28'},
                 {**template(), 'unexpected': 'field'}]
        # A missing/null snapshot remains valid for CLI-derived and older drafts.
        for value in cases[1:]:
            with self.subTest(value=value), self.assertRaises(ValidationError):
                RunRequest(source_template=value)
        for change in [{'n_jobs': 0}, {'stages': []}, {'figure_formats': ['invented']},
                       {'trust_unverified_legacy_results': True}, {'allow_existing': True},
                       {'settings': {'decode': {'value': float('nan')}}}]:
            value = template()
            value['config'].update(change)
            with self.subTest(change=change), self.assertRaises(ValidationError):
                RunRequest(source_template=value)


class DashboardJobProvenanceTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = fixture_root(directory.name)
        self.run = self.root / 'cache/restored'
        (self.run / 'select').mkdir(parents=True)
        self.store = ResultStore(self.root)
        self.argv = ['python', '-u', 'pipeline.py', '--cache-dir', '/old/cache/run']
        self.manifest('invocation-one')

    def manifest(self, identity, *, history=False, argv=None):
        path = self.run / (f'manifests/{identity}.json' if history else 'pipeline_manifest.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'run_id': identity, 'status': 'complete', 'stages': [],
                                    'invocation': {'argv': argv or self.argv}}))

    def job(self, *, local=True, identity='a', **changes):
        job_id = identity * 32
        value = {'id': job_id, 'manifest_id': 'invocation-one', 'argv': self.argv,
                 'cache_dir': str(self.run), 'request': RunRequest(
                     name='Original request', data_dir='data', cache_dir='cache/restored',
                     stages=['evaluate'], source_template=template()).model_dump(), **changes}
        directory = self.run / 'dashboard' if local else self.root / 'cache/.dashboard'
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f'{job_id}.json'
        path.write_text(json.dumps(value))
        return path, value

    def history(self):
        return self.store.detail('restored')

    def test_restored_run_uses_local_snapshot_without_central_storage_or_template_files(self):
        self.job(cache_dir='/a/former/machine/cache/original')
        record = self.history()['manifests'][0]
        self.assertEqual(record['source_template'], template())
        self.assertEqual(record['original_request']['name'], 'Original request')
        self.assertFalse((self.root / 'cache/.dashboard').exists())

    def test_local_record_wins_over_stale_central_copy(self):
        self.job()
        snapshot = template()
        snapshot['name'] = 'Stale central template'
        self.job(local=False, request=RunRequest(source_template=snapshot).model_dump())
        self.assertEqual(self.history()['manifests'][0]['source_template']['name'], 'Example pipeline')

    def test_central_fallback_requires_matching_run_directory(self):
        path, record = self.job(local=False, cache_dir=str(self.root / 'cache/unrelated'))
        self.assertIsNone(self.history()['manifests'][0]['source_template'])
        record['cache_dir'] = str(self.run)
        path.write_text(json.dumps(record))
        self.assertEqual(self.history()['manifests'][0]['source_template'], template())

    def test_unrelated_newer_jobs_cannot_change_an_earlier_manifest(self):
        self.manifest('invocation-one', history=True)
        self.manifest('invocation-two', argv=[*self.argv, '--stages', 'models'])
        self.job()
        other = template()
        other['name'] = 'New template'
        self.job(identity='b', manifest_id='invocation-two',
                 request=RunRequest(source_template=other).model_dump())
        records = {item['id']: item for item in self.history()['manifests']}
        self.assertEqual(records['invocation-one']['source_template'], template())
        self.assertEqual(records['invocation-two']['source_template']['name'], 'New template')

    def test_exact_argv_fallback_only_when_record_has_no_manifest_id(self):
        path, value = self.job(manifest_id=None)
        self.assertEqual(self.history()['manifests'][0]['source_template'], template())
        value['manifest_id'] = 'unrelated-id'
        path.write_text(json.dumps(value))
        self.assertIsNone(self.history()['manifests'][0]['source_template'])
        value['manifest_id'] = None
        value['argv'] = [*self.argv, '--other-option']
        path.write_text(json.dumps(value))
        self.assertIsNone(self.history()['manifests'][0]['source_template'])

    def test_argv_fallback_is_rejected_when_a_command_was_replayed(self):
        self.manifest('invocation-one', history=True)
        self.manifest('invocation-two')
        path, value = self.job(manifest_id=None)
        result = self.history()
        self.assertEqual(len(result['manifests']), 2)
        for record in result['manifests']:
            self.assertIsNone(record['source_template'])
            self.assertIsNone(record['original_request'])
        self.assertTrue(all('shares its exact command' in error for error in result['errors']))
        self.assertEqual(len(result['errors']), 2)
        # A recorded manifest ID still identifies the original invocation even
        # though a later invocation reused the same argv and settings path.
        value['manifest_id'] = 'invocation-one'
        path.write_text(json.dumps(value))
        records = {record['id']: record for record in self.history()['manifests']}
        self.assertEqual(records['invocation-one']['source_template'], template())
        self.assertIsNone(records['invocation-two']['source_template'])

    def test_explicit_central_id_is_usable_when_local_argv_fallback_is_ambiguous(self):
        self.manifest('invocation-one', history=True)
        self.manifest('invocation-two')
        self.job(manifest_id=None)
        self.job(local=False, identity='b', manifest_id='invocation-two')
        records = {record['id']: record for record in self.history()['manifests']}
        self.assertIsNone(records['invocation-one']['source_template'])
        self.assertEqual(records['invocation-two']['source_template'], template())

    def test_legacy_runs_without_snapshots_do_not_borrow_current_template(self):
        self.assertIsNone(self.history()['manifests'][0]['source_template'])
        self.job(request={'name': 'Older run', 'stages': ['evaluate']})
        record = self.history()['manifests'][0]
        self.assertIsNone(record['source_template'])
        self.assertEqual(record['original_request'], {'name': 'Older run', 'stages': ['evaluate']})

    def test_invalid_or_ambiguous_records_are_not_mislabeled(self):
        path, value = self.job()
        value['request']['source_template']['builtin'] = 'true'
        path.write_text(json.dumps(value))
        result = self.history()
        self.assertIsNone(result['manifests'][0]['source_template'])
        self.assertIn('Invalid saved dashboard request', result['errors'][0])
        self.job()
        self.job(identity='b')
        result = self.history()
        self.assertIsNone(result['manifests'][0]['source_template'])
        self.assertIn('Multiple dashboard records', result['errors'][0])

    def test_symlinked_job_records_are_not_followed(self):
        path, value = self.job()
        outside = self.root / 'outside.json'
        outside.write_text(json.dumps(value))
        path.unlink()
        path.symlink_to(outside)
        self.assertIsNone(self.history()['manifests'][0]['source_template'])

    def test_run_library_summary_does_not_read_job_provenance(self):
        with patch('scripts.next.dashboard.results.attach_job_provenance', side_effect=AssertionError):
            self.assertEqual(self.store.list_runs()[0]['id'], 'restored')


class SourceTemplatePersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_submitted_snapshot_is_copied_to_both_durable_records_without_resolving_template(self):
        with tempfile.TemporaryDirectory() as directory:
            root = fixture_root(directory)
            manager = RunManager(root)
            snapshot = template()
            payload = RunRequest(data_dir='data', cache_dir='cache/new', stages=['evaluate'],
                                 source_template=deepcopy(snapshot))
            with patch.object(manager, '_execute', new=AsyncMock()):
                job = await manager.start(payload)
                await manager.tasks[job['id']]
            payload.source_template.config.settings['decode']['n_decode_shuffle'] = 1
            for path in [manager.storage / f"{job['id']}.json", Path(job['run_record_path'])]:
                self.assertEqual(json.loads(path.read_text())['request']['source_template'], snapshot)
            self.assertEqual(manager.snapshot(job['id'])['request']['source_template'], snapshot)
            manager.jobs[job['id']]['status'] = 'complete'
            await manager.close()


if __name__ == '__main__':
    unittest.main()
