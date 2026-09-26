"""Integrated documentation routes and the optional one-command site build."""
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import call, patch

from fastapi.testclient import TestClient

from scripts.next.dashboard.app import create_app
from scripts.next.dashboard.build import build_assets
from scripts.next.dashboard.__main__ import main


class DocumentationRoutesTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        frontend = self.root / 'dashboard/dist'
        frontend.mkdir(parents=True)
        (frontend / 'index.html').write_text('<html>React dashboard</html>')
        self.client = TestClient(create_app(self.root), base_url='http://127.0.0.1:8000')
        self.addCleanup(self.client.close)

    def build_docs(self):
        site = self.root / 'site'
        (site / 'next/methods').mkdir(parents=True)
        (site / 'assets').mkdir()
        (site / 'index.html').write_text('<html>Pipeline documentation</html>')
        (site / 'next/methods/index.html').write_text('<html>Analysis methods</html>')
        (site / 'assets/theme.css').write_text('body { color: teal; }')
        (site / '404.html').write_text('<html>Documentation page not found</html>')
        return site

    def test_missing_build_is_503_without_falling_through_to_spa(self):
        response = self.client.get('/docs', follow_redirects=False)
        self.assertEqual(response.status_code, 307)
        self.assertEqual(response.headers['location'], '/docs/')
        for path in ('/docs/', '/docs/next/methods/', '/docs/assets/theme.css'):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 503)
                self.assertIn('mkdocs build', response.json()['detail'])
                self.assertNotIn('React dashboard', response.text)
        self.assertIn('React dashboard', self.client.get('/runs').text)

    def test_missing_build_has_accessible_html_for_browsers_and_json_for_api_clients(self):
        response = self.client.get('/docs/next/methods/', headers={
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        })
        self.assertEqual(response.status_code, 503)
        self.assertTrue(response.headers['content-type'].startswith('text/html'))
        self.assertEqual(response.headers['cache-control'], 'no-store')
        self.assertEqual(response.headers['vary'], 'Accept')
        self.assertIn('<html lang="en">', response.text)
        self.assertIn('Documentation is not built yet</h1>', response.text)
        self.assertIn('uv run --locked --group dashboard --group docs python -m mkdocs build --strict', response.text)
        self.assertIn('href="/">Return to dashboard</a>', response.text)
        self.assertIn('aria-live="polite"', response.text)
        for accept in ('application/json', 'text/html;q=0,application/json'):
            with self.subTest(accept=accept):
                response = self.client.get('/docs/', headers={'Accept': accept})
                self.assertEqual(response.status_code, 503)
                self.assertIn('uv run --locked', response.json()['detail'])

    def test_docs_built_after_startup_become_available_with_real_assets(self):
        self.assertEqual(self.client.get('/docs/').status_code, 503)
        self.build_docs()
        self.assertIn('Pipeline documentation', self.client.get('/docs/').text)
        response = self.client.get('/docs/next/methods/', follow_redirects=False)
        self.assertEqual(response.status_code, 200)
        self.assertIn('Analysis methods', response.text)
        response = self.client.get('/docs/next/methods', follow_redirects=False)
        self.assertEqual(response.status_code, 307)
        self.assertTrue(response.headers['location'].endswith('/docs/next/methods/'))
        response = self.client.get('/docs/assets/theme.css')
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers['content-type'].startswith('text/css'))
        self.assertEqual(self.client.head('/docs/next/methods/').status_code, 200)

    def test_missing_pages_are_true_documentation_404s(self):
        self.build_docs()
        response = self.client.get('/docs/not-a-page/')
        self.assertEqual(response.status_code, 404)
        self.assertIn('Documentation page not found', response.text)
        self.assertNotIn('React dashboard', response.text)
        (self.root / 'site/404.html').unlink()
        self.assertEqual(self.client.get('/docs/not-a-page/').status_code, 404)

    def test_docs_do_not_serve_outside_files_through_symlinks(self):
        site = self.build_docs()
        secret = self.root / 'private.txt'
        secret.write_text('private local content')
        (site / 'outside.txt').symlink_to(secret)
        response = self.client.get('/docs/outside.txt')
        self.assertEqual(response.status_code, 404)
        self.assertNotIn('private local content', response.text)

    def test_api_docs_and_openapi_live_under_api_prefix(self):
        response = self.client.get('/api/openapi.json')
        self.assertEqual(response.status_code, 200)
        self.assertIn('/api/health', response.json()['paths'])
        swagger = self.client.get('/api/docs')
        self.assertEqual(swagger.status_code, 200)
        self.assertIn('/api/openapi.json', swagger.text)
        self.assertIn('/api/docs/oauth2-redirect', swagger.text)
        redoc = self.client.get('/api/redoc')
        self.assertEqual(redoc.status_code, 200)
        self.assertIn('/api/openapi.json', redoc.text)
        self.assertEqual(self.client.get('/api/not-an-endpoint').status_code, 404)


class DashboardBuildTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        frontend = self.root / 'dashboard'
        frontend.mkdir()
        (frontend / 'package.json').write_text('{}')
        (self.root / 'mkdocs.yml').write_text('site_name: Test\n')

    def test_builds_with_current_python_and_installs_missing_node_dependencies(self):
        with patch('shutil.which', return_value='/node/npm'), \
             patch('importlib.util.find_spec', return_value=object()), \
             patch('subprocess.run') as run, redirect_stdout(io.StringIO()):
            build_assets(self.root)
        self.assertEqual(run.call_args_list, [
            call(['/node/npm', 'ci'], cwd=self.root / 'dashboard', check=True),
            call(['/node/npm', 'run', 'build'], cwd=self.root / 'dashboard', check=True),
            call([sys.executable, '-m', 'mkdocs', 'build', '--strict',
                  '--config-file', str(self.root / 'mkdocs.yml'), '--site-dir', str(self.root / 'site')],
                 cwd=self.root, check=True),
        ])
        self.assertTrue(all('shell' not in item.kwargs for item in run.call_args_list))

    def test_existing_node_modules_skip_install_and_build_failure_stops(self):
        (self.root / 'dashboard/node_modules').mkdir()
        with patch('shutil.which', return_value='/node/npm'), \
             patch('importlib.util.find_spec', return_value=object()), \
             patch('subprocess.run', side_effect=subprocess.CalledProcessError(7, ['npm'])) as run, \
             redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'exit code 7'):
            build_assets(self.root)
        run.assert_called_once_with(['/node/npm', 'run', 'build'], cwd=self.root / 'dashboard', check=True)

    def test_missing_build_tools_fail_before_running_commands(self):
        with patch('shutil.which', return_value=None), patch('subprocess.run') as run:
            with self.assertRaisesRegex(RuntimeError, 'nvm use'):
                build_assets(self.root)
            run.assert_not_called()
        with patch('shutil.which', return_value='/node/npm'), \
             patch('importlib.util.find_spec', return_value=None), patch('subprocess.run') as run:
            with self.assertRaisesRegex(RuntimeError, 'uv sync --locked --group dashboard --group docs'):
                build_assets(self.root)
            run.assert_not_called()

    def test_launcher_build_flag_runs_before_server_and_preserves_loopback_options(self):
        events = []
        with patch('sys.argv', ['dashboard', '--build', '--host', 'localhost', '--port', '8020']), \
             patch('scripts.next.dashboard.build.build_assets', side_effect=lambda: events.append('build')), \
             patch('uvicorn.run', side_effect=lambda *a, **kw: events.append(('serve', kw))):
            main()
        self.assertEqual(events, ['build', ('serve', {'factory': True, 'host': 'localhost', 'port': 8020})])

    def test_build_failure_does_not_start_server_and_default_launch_does_not_build(self):
        with patch('sys.argv', ['dashboard', '--build']), \
             patch('scripts.next.dashboard.build.build_assets', side_effect=RuntimeError('Build failed')), \
             patch('uvicorn.run') as serve, redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main()
        serve.assert_not_called()
        with patch('sys.argv', ['dashboard']), \
             patch('scripts.next.dashboard.build.build_assets') as build, patch('uvicorn.run') as serve:
            main()
        build.assert_not_called()
        serve.assert_called_once_with('scripts.next.dashboard.app:create_app', factory=True,
                                      host='127.0.0.1', port=8000)


if __name__ == '__main__':
    unittest.main()
