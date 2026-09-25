"""Manifest history must survive partial runs, failures, and upgrades."""
from contextlib import ExitStack, redirect_stdout
from dataclasses import replace
from datetime import datetime, timezone
import importlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.next import cache_io
from scripts.next import pipeline, run_manifest


class ManifestHistoryTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.config = pipeline.Config(cache_dir=self.root, stages=('all',))
        self.latest = self.root / 'pipeline_manifest.json'

    def invoke(self, config, failure=None):
        with ExitStack() as stack:
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(patch.dict('os.environ'))
            for stage, name in pipeline.STAGES.items():
                module = importlib.import_module('scripts.next.' + name)
                stack.enter_context(patch.object(module, 'main',
                    side_effect=failure if stage == 'evaluate' else None))
            pipeline.main(config)

    def history(self):
        return {p.name: p.read_bytes() for p in (self.root / 'manifests').glob('*.json')}

    def test_full_then_partial_runs_preserve_original_record_and_settings(self):
        # Same timestamp must still yield distinct files.
        instant = datetime(2026, 9, 26, tzinfo=timezone.utc)
        with patch.object(run_manifest, 'utc_now', return_value=instant):
            self.invoke(self.config)
            original = self.history()
            first = json.loads(self.latest.read_bytes())
            self.invoke(replace(self.config, stages=('evaluate',), n_jobs=2))
            self.invoke(replace(self.config, stages=('evaluate',), n_jobs=3))
        history = self.history()
        latest = json.loads(self.latest.read_bytes())
        self.assertEqual(len(history), 3)
        self.assertTrue(original.items() <= history.items())
        self.assertEqual(list(first['settings']), list(pipeline.STAGES))
        self.assertEqual(first['runner_config']['stages'], ['all'])
        self.assertEqual(list(latest['settings']), ['evaluate'])
        self.assertEqual(latest['runner_config']['n_jobs'], 3)
        self.assertEqual(history[latest['run_id'] + '.json'], self.latest.read_bytes())
        for payload in history.values():
            record = json.loads(payload)
            self.assertEqual(record['status'], 'complete')
            self.assertIsNotNone(record['finished_at'])
            self.assertTrue(all(s['status'] == 'complete' for s in record['stages']))

    def test_failures_and_interruptions_are_recorded_without_touching_prior_runs(self):
        self.invoke(replace(self.config, stages=('select',)))
        original = self.history()
        for error, status in [(RuntimeError('test failure'), 'failed'),
                              (KeyboardInterrupt(), 'interrupted')]:
            with self.subTest(status=status), self.assertRaises(type(error)):
                self.invoke(replace(self.config, stages=('select', 'evaluate', 'states')), error)
            latest = json.loads(self.latest.read_bytes())
            self.assertEqual(latest['status'], status)
            self.assertIsNotNone(latest['finished_at'])
            self.assertEqual([s['stage'] for s in latest['stages']], ['select', 'evaluate'])
            self.assertEqual(latest['stages'][0]['status'], 'complete')
            self.assertEqual(latest['stages'][1]['status'], status)
            self.assertIn(type(error).__name__, latest['stages'][1]['error'])
            self.assertIn('seconds', latest['stages'][1])
            self.assertEqual(self.history()[latest['run_id'] + '.json'], self.latest.read_bytes())
        self.assertTrue(original.items() <= self.history().items())
        self.assertEqual(len(self.history()), 3)

    def test_old_latest_is_archived_byte_for_byte_once(self):
        original = b'{"settings": {"select": {}}, "stages": []}\n'
        self.latest.write_bytes(original)
        self.invoke(replace(self.config, stages=('evaluate',)))
        self.invoke(replace(self.config, stages=('evaluate',)))
        history = self.history()
        self.assertEqual(len(history), 3)
        self.assertEqual([v for k, v in history.items() if k.startswith('prior-')], [original])

    def test_orphaned_and_unreadable_latest_records_are_preserved(self):
        self.invoke(replace(self.config, stages=('evaluate',)))
        original = self.latest.read_bytes()
        for path in (self.root / 'manifests').iterdir():
            path.unlink()
        self.invoke(replace(self.config, stages=('evaluate',)))
        self.assertIn(original, self.history().values())
        malformed = b'{"partially written old record":'
        self.latest.write_bytes(malformed)
        self.invoke(replace(self.config, stages=('evaluate',)))
        self.assertIn(malformed, self.history().values())

    def test_dry_run_and_invalid_config_leave_history_unchanged(self):
        self.invoke(replace(self.config, stages=('select',)))
        original, latest = self.history(), self.latest.read_bytes()
        self.invoke(replace(self.config, dry_run=True))
        with self.assertRaises(ValueError):
            self.invoke(replace(self.config, stages=('unknown',)))
        self.assertEqual(self.history(), original)
        self.assertEqual(self.latest.read_bytes(), latest)

    def test_running_status_is_saved_before_stage_executes(self):
        def inspect(config):
            record = json.loads(self.latest.read_bytes())
            self.assertEqual(record['status'], 'running')
            self.assertIsNone(record['finished_at'])
            self.assertEqual(record['stages'][-1]['status'], 'running')
            self.assertEqual(self.history()[record['run_id'] + '.json'], self.latest.read_bytes())
        self.invoke(replace(self.config, stages=('evaluate',)), inspect)

    def test_failed_atomic_replace_preserves_previous_file_and_cleans_temp(self):
        self.latest.write_bytes(b'original')
        with patch.object(Path, 'replace', side_effect=OSError('write failed')):
            with self.assertRaises(OSError):
                run_manifest._atomic_write(self.latest, b'replacement')
        self.assertEqual(self.latest.read_bytes(), b'original')
        self.assertEqual(list(self.root.iterdir()), [self.latest])


class InvocationCommandTest(unittest.TestCase):
    def test_quoting_preserves_exact_argument_values_and_copies_input(self):
        argv = ['python', '-X', 'utf8', '-m', 'scripts.next.pipeline', '--cache-dir',
                "cache/with space, 'quotes', $variable; $(literal) and \u03b1", '']
        invocation = run_manifest.invocation_context(argv)
        self.assertEqual(invocation['argv'], argv)
        self.assertEqual(shlex.split(invocation['command']), argv)
        self.assertEqual(invocation['cwd'], str(Path.cwd()))
        self.assertEqual(invocation['python_executable'], sys.executable)
        self.assertEqual(invocation['source'], 'cli')
        argv.append('later mutation')
        self.assertNotEqual(invocation['argv'], argv)

    def test_programmatic_run_does_not_claim_host_process_command(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), \
             patch.dict(os.environ), patch.object(sys, 'orig_argv', ['python', '-m', 'unittest']), \
             patch('scripts.next.eval_confidence.main'):
            root = Path(directory)
            pipeline.main(pipeline.Config(cache_dir=root, stages=('evaluate',)))
            invocation = json.loads((root / 'pipeline_manifest.json').read_text())['invocation']
            self.assertEqual(invocation['source'], 'programmatic')
            self.assertIsNone(invocation['argv'])
            self.assertIsNone(invocation['command'])
            self.assertEqual(invocation['cwd'], str(Path.cwd()))

    def test_direct_and_module_cli_capture_commands_and_working_directory(self):
        repo = Path(__file__).resolve().parents[2]
        launches = [
            [sys.executable, '-X', 'utf8', str(repo / 'scripts/next/pipeline.py')],
            [sys.executable, '-X', 'utf8', '-m', 'scripts.next.pipeline'],
        ]
        source = dict(session='example', cue=1, trial_idx=np.array([1, 3]),
                      time_bins=np.array([0, 50]), decoding_test_labels=np.ones(2),
                      decoding_confidence=np.full((2, 2), .8),
                      decoding_predicted_labels=np.ones((2, 2)),
                      decoding_confidence_null=np.full((2, 2, 3), .5))
        with tempfile.TemporaryDirectory() as directory:
            for index, launch in enumerate(launches):
                with self.subTest(launch=launch):
                    cwd = Path(directory) if index == 0 else repo
                    cache = Path(directory) / f"run {index} 'quoted'; $literal"
                    cache_io.save([source], cache / 'decode/decoding_confidence.pkl')
                    argv = launch + ['--stages', 'evaluate', '--cache-dir', str(cache),
                                     '--data-dir', 'data/example', '--n-jobs=2']
                    result = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    latest = (cache / 'pipeline_manifest.json').read_bytes()
                    record = json.loads(latest)
                    invocation = record['invocation']
                    self.assertEqual(record['status'], 'complete')
                    self.assertEqual(invocation['argv'], argv)
                    self.assertEqual(shlex.split(invocation['command']), argv)
                    self.assertEqual(invocation['cwd'], str(cwd.resolve()))
                    self.assertEqual((cache / 'manifests' / f"{record['run_id']}.json").read_bytes(), latest)

    def test_failed_cli_keeps_invocation_for_reference(self):
        repo = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / 'missing inputs'
            argv = [sys.executable, '-m', 'scripts.next.pipeline', '--cache-dir', str(cache),
                    '--stages', 'evaluate']
            result = subprocess.run(argv, cwd=repo, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            record = json.loads((cache / 'pipeline_manifest.json').read_text())
            self.assertEqual(record['status'], 'failed')
            self.assertEqual(record['invocation']['argv'], argv)
            self.assertEqual(shlex.split(record['invocation']['command']), argv)
            self.assertIn('FileNotFoundError', record['stages'][0]['error'])


if __name__ == '__main__':
    unittest.main()
