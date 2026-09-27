"""Path completion lists names only, preserves notation, and blocks foreign sites."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from scripts.next.dashboard.app import _trusted_origin
from scripts.next.dashboard.paths import complete_path, create_paths_router


class DashboardPathTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / 'recordings').mkdir()
        (self.root / 'results').mkdir()
        (self.root / '.hidden').mkdir()
        (self.root / 'sessions.json').write_text('contents are never read')
        (self.root / 'sessions.TXT').write_text('contents are never read')
        (self.root / 'session notes.csv').write_text('contents are never read')
        app = FastAPI()
        app.include_router(create_paths_router(self.root, _trusted_origin))
        self.client = TestClient(app, base_url='http://127.0.0.1:8000')
        self.addCleanup(self.client.close)

    def paths(self, result):
        return [entry['path'] for entry in result['entries']]

    def test_relative_directory_prefix_and_directory_only(self):
        self.assertEqual(self.paths(complete_path(self.root, 're', 'directory')),
                         ['recordings/', 'results/'])
        self.assertEqual(self.paths(complete_path(self.root, '', 'directory')),
                         ['recordings/', 'results/'])
        (self.root / 'recordings' / 'session 01').mkdir()
        self.assertEqual(self.paths(complete_path(self.root, 'recordings/')),
                         ['recordings/session 01/'])

    def test_extensions_filter_files_but_keep_directories_and_never_read_contents(self):
        with patch.object(Path, 'read_text', side_effect=AssertionError('No content reads')):
            result = complete_path(self.root, '', extensions='.json,txt')
        self.assertEqual(self.paths(result),
                         ['recordings/', 'results/', 'sessions.json', 'sessions.TXT'])
        self.assertEqual(result['entries'][2]['kind'], 'file')

    def test_absolute_home_and_dot_notation(self):
        self.assertEqual(self.paths(complete_path(self.root, str(self.root) + '/ses')),
                         [str(self.root / 'session notes.csv'), str(self.root / 'sessions.json'),
                          str(self.root / 'sessions.TXT')])
        with patch.dict('os.environ', {'HOME': str(self.root)}):
            self.assertEqual(self.paths(complete_path(self.root, '~/re')), ['~/recordings/', '~/results/'])
            self.assertEqual(self.paths(complete_path(self.root, '~')), ['~/'])
        self.assertEqual(self.paths(complete_path(self.root, './re')), ['./recordings/', './results/'])
        self.assertIn('./recordings/', self.paths(complete_path(self.root, '.')))
        self.assertIn('../', self.paths(complete_path(self.root / 'recordings', '..'))[0])

    def test_hidden_names_require_explicit_dot_prefix_and_case_insensitive_matching(self):
        self.assertEqual(self.paths(complete_path(self.root, '.h')), ['.hidden/'])
        self.assertEqual(self.paths(complete_path(self.root, 'REC')), ['recordings/'])

    def test_missing_parent_non_directory_and_permission_warnings(self):
        self.assertIn('does not exist', complete_path(self.root, 'missing/')['warning'])
        self.assertIn('is a file', complete_path(self.root, 'sessions.json/')['warning'])
        with patch('scripts.next.dashboard.paths.os.scandir', side_effect=PermissionError):
            result = complete_path(self.root, '')
        self.assertEqual(result['entries'], [])
        self.assertIn('permission', result['warning'])
        self.assertIn('null', complete_path(self.root, '\0')['warning'])
        self.assertIn('named-user', complete_path(self.root, '~someone/')['warning'])

    def test_bounds_and_scan_cap_return_narrowing_hint(self):
        result = complete_path(self.root, '', limit=2)
        self.assertEqual(len(result['entries']), 2)
        self.assertTrue(result['truncated'])
        self.assertIn('Type more', result['warning'])
        with patch('scripts.next.dashboard.paths.MAX_SCANNED_ENTRIES', 2):
            self.assertTrue(complete_path(self.root, '')['truncated'])

    def test_symlinks_list_as_paths_and_broken_links_are_ignored(self):
        (self.root / 'linked').symlink_to(self.root / 'recordings', target_is_directory=True)
        (self.root / 'broken').symlink_to(self.root / 'missing')
        self.assertEqual(self.paths(complete_path(self.root, 'link', 'directory')), ['linked/'])
        self.assertEqual(self.paths(complete_path(self.root, 'bro')), [])

    def test_api_validates_queries_and_preserves_literal_spaces(self):
        response = self.client.get('/api/paths/complete', params={'path': 'session ', 'mode': 'any'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.paths(response.json()), ['session notes.csv'])
        for params in ({'limit': 51}, {'limit': 0}, {'mode': 'everything'}, {'path': 'a' * 4097}):
            self.assertEqual(self.client.get('/api/paths/complete', params=params).status_code, 422)

    def test_api_blocks_cross_origin_and_cross_site_navigations(self):
        for headers in ({'Origin': 'https://example.com'}, {'Sec-Fetch-Site': 'cross-site'},
                        {'Origin': 'http://evil.localhost:5173'}):
            response = self.client.get('/api/paths/complete', headers=headers)
            self.assertEqual(response.status_code, 403)
        for headers in ({}, {'Origin': 'http://127.0.0.1:8000', 'Sec-Fetch-Site': 'same-origin'},
                        {'Origin': 'http://localhost:5173'}):
            self.assertEqual(self.client.get('/api/paths/complete', headers=headers).status_code, 200)


if __name__ == '__main__':
    unittest.main()
