"""Runtime mounts work without rebuilding assets or duplicating pipeline managers."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from scripts.next.dashboard.__main__ import main
from scripts.next.dashboard.app import create_app
from scripts.next.dashboard.runner import RunManager
from scripts.next.dashboard.urls import DashboardURLs, normalize_url_prefix
from tests.next.test_dashboard_runner import fixture_root, request


class DashboardPrefixTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = fixture_root(directory.name)
        dist = self.root / 'dashboard/dist'
        (dist / 'assets').mkdir(parents=True)
        (dist / 'index.html').write_text(
            '<html><head><script src="./assets/app.js"></script></head><body>Dashboard</body></html>')
        (dist / 'assets/app.js').write_text('/* bundled dashboard */')
        (dist / 'favicon.svg').write_text('<svg></svg>')
        site = self.root / 'site'
        (site / 'next/methods').mkdir(parents=True)
        (site / 'index.html').write_text('<html>Guide</html>')
        (site / 'next/methods/index.html').write_text('<html>Methods</html>')
        (site / '404.html').write_text('<html>Guide page missing</html>')
        run = self.root / 'cache/sample'
        (run / 'select/figures').mkdir(parents=True)
        (run / 'select/figures/a plot.pdf').write_bytes(b'%PDF test artifact')
        (run / 'pipeline_manifest.json').write_text(json.dumps({
            'run_id': 'fixture', 'status': 'complete', 'stages': [],
        }))

    def client(self, prefix='/wm-states'):
        app = create_app(self.root, url_prefix=prefix)
        client = TestClient(app, base_url='http://127.0.0.1:8000')
        self.addCleanup(client.close)
        return app, client

    def test_default_nested_and_root_prefix_work_with_one_asset_build(self):
        for prefix in ('/wm-states', '/lab/team/project', '/'):
            with self.subTest(prefix=prefix):
                app, client = self.client(prefix)
                urls = app.state.urls
                for path in ('/', '/runs/fixture', '/index.html'):
                    response = client.get(urls.dashboard + path)
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertIn(f'<base href="{urls.dashboard}/">', response.text)
                    self.assertIn(f'name="wm-states-dashboard-base" content="{urls.dashboard}/"', response.text)
                    self.assertIn(f'name="wm-states-docs-base" content="{urls.docs}/"', response.text)
                    self.assertLess(response.text.index('<base'), response.text.index('<script'))
                    self.assertEqual(response.headers['cache-control'], 'no-store')
                self.assertEqual(client.get(urls.dashboard + '/assets/app.js').text, '/* bundled dashboard */')
                self.assertEqual(client.get(urls.dashboard + '/favicon.svg').status_code, 200)
                self.assertEqual(client.get(urls.dashboard + '/api/health').json()['status'], 'ok')
                self.assertIn('Guide', client.get(urls.docs + '/').text)
                self.assertIn('Methods', client.get(urls.docs + '/next/methods/').text)
                self.assertEqual(client.get(urls.docs + '/missing/').status_code, 404)
                self.assertEqual(client.get(urls.dashboard + '/api/missing').status_code, 404)

    def test_roots_and_legacy_docs_redirect_to_canonical_mounts(self):
        _, client = self.client('/lab/project/')
        for path, location in (
            ('/', '/lab/project/dashboard/'),
            ('/lab/project', '/lab/project/dashboard/'),
            ('/lab/project/', '/lab/project/dashboard/'),
            ('/lab/project/dashboard', '/lab/project/dashboard/'),
            ('/lab/project/docs', '/lab/project/docs/'),
            ('/docs', '/lab/project/docs/'),
            ('/docs/next/methods/?q=one%20two', '/lab/project/docs/next/methods/?q=one%20two'),
        ):
            with self.subTest(path=path):
                response = client.get(path, follow_redirects=False)
                self.assertEqual(response.status_code, 307)
                self.assertEqual(response.headers['location'], location)
        response = client.get('/lab/project/docs/next/methods', follow_redirects=False)
        self.assertEqual(response.headers['location'], 'http://127.0.0.1:8000/lab/project/docs/next/methods/')

    def test_root_alias_uses_current_runtime_metadata_and_legacy_apis_work(self):
        _, client = self.client('/research')
        self.assertIn('<base href="/research/dashboard/">', client.get('/runs/fixture').text)
        self.assertEqual(client.get('/api/health').status_code, 200)
        self.assertEqual(client.get('/assets/app.js').status_code, 200)
        self.assertEqual(client.get('/api/jobs').json(), client.get('/research/dashboard/api/jobs').json())

    def test_api_docs_and_swagger_use_the_mount_including_openapi_server(self):
        _, client = self.client('/lab/project')
        base = '/lab/project/dashboard'
        # A prior legacy request must not freeze an unprefixed OpenAPI server.
        client.get('/api/openapi.json')
        swagger = client.get(base + '/api/docs')
        self.assertIn(base + '/api/openapi.json', swagger.text)
        self.assertIn(base + '/api/docs/oauth2-redirect', swagger.text)
        self.assertIn(base + '/api/openapi.json', client.get(base + '/api/redoc').text)
        schema = client.get(base + '/api/openapi.json').json()
        self.assertIn({'url': base}, schema['servers'])
        self.assertIn('/api/health', schema['paths'])

    def test_artifact_urls_and_log_downloads_follow_request_mount(self):
        app, client = self.client('/lab/project')
        base = '/lab/project/dashboard'
        for prefix in ('', base):
            with self.subTest(prefix=prefix):
                detail = client.get(prefix + '/api/runs/sample').json()
                artifact = next(item for item in detail['artifacts'] if item['kind'] == 'figure')
                self.assertEqual(artifact['url'], prefix + '/api/runs/sample/artifacts/select/figures/a%20plot.pdf')
                self.assertEqual(client.get(artifact['url']).content, b'%PDF test artifact')
                self.assertEqual(client.get(prefix + '/api/runs/sample').headers['cache-control'], 'no-store')
        job_id = 'b' * 32
        manager = app.state.runner
        manager.jobs[job_id] = {'id': job_id, 'status': 'complete', 'logs': ['last line']}
        manager.storage.mkdir(parents=True)
        (manager.storage / f'{job_id}.log').write_text('complete log\n')
        response = client.get(base + f'/api/jobs/{job_id}/log')
        self.assertEqual(response.text, 'complete log\n')
        self.assertEqual(client.head(base + f'/api/jobs/{job_id}/log').headers['content-length'], '13')

    def test_canonical_and_legacy_routes_share_one_manager_and_origin_checks(self):
        with patch('scripts.next.dashboard.app.RunManager', wraps=RunManager) as constructor:
            app, client = self.client('/lab/project')
        constructor.assert_called_once()
        app.state.runner.jobs['done'] = dict(id='done', status='complete', logs=['one manager'],
                                            created_at='2026-01-01T00:00:00Z', stages=[])
        for base in ('', '/lab/project/dashboard'):
            with self.subTest(base=base):
                self.assertEqual(client.get(base + '/api/jobs').json()['jobs'][0]['id'], 'done')
                response = client.post(base + '/api/validate', json=request().model_dump(),
                                       headers={'Origin': 'http://127.0.0.1:8000'})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(client.post(base + '/api/validate', json=request().model_dump(),
                                             headers={'Origin': 'https://evil.example'}).status_code, 403)
                with client.websocket_connect('ws://127.0.0.1:8000' + base + '/api/jobs/done/events',
                                              headers={'Origin': 'http://127.0.0.1:8000'}) as socket:
                    self.assertEqual(socket.receive_json()['logs'], ['one manager'])
                with self.assertRaises(WebSocketDisconnect):
                    with client.websocket_connect('ws://127.0.0.1:8000' + base + '/api/jobs/done/events',
                                                  headers={'Origin': 'https://evil.example'}):
                        pass

    def test_missing_docs_link_returns_to_configured_dashboard(self):
        (self.root / 'site/index.html').unlink()
        _, client = self.client('/research/lab')
        response = client.get('/research/lab/docs/next/methods/', headers={'Accept': 'text/html'})
        self.assertEqual(response.status_code, 503)
        self.assertIn('href="/research/lab/dashboard/">Return to dashboard', response.text)
        self.assertEqual(client.get('/docs/', headers={'Accept': 'text/html'}).status_code, 503)

    def test_missing_docs_pages_are_self_contained_for_every_runtime_and_build_prefix(self):
        for prefix in ('/wm-states', '/research/nested', '/'):
            for build_prefix in ('/', '/github-project/'):
                with self.subTest(prefix=prefix, build_prefix=build_prefix):
                    generated = (f'<html><head><link href="{build_prefix}assets/theme.css"></head>'
                                 f'<body>Generated 404<script src="{build_prefix}assets/search.js"></script></body></html>')
                    (self.root / 'site/404.html').write_text(generated)
                    app, client = self.client(prefix)
                    response = client.get(app.state.urls.docs + '/missing/deep/page/',
                                          headers={'Accept': 'text/html'})
                    self.assertEqual(response.status_code, 404)
                    self.assertTrue(response.headers['content-type'].startswith('text/html'))
                    self.assertEqual(response.headers['cache-control'], 'no-store')
                    self.assertEqual(response.headers['vary'], 'Accept')
                    self.assertIn('Documentation page not found', response.text)
                    self.assertIn(f'href="{app.state.urls.docs}/">Open guide', response.text)
                    self.assertIn(f'href="{app.state.urls.dashboard}/">Return to dashboard', response.text)
                    self.assertIn('<style>', response.text)
                    self.assertNotIn('<script', response.text)
                    self.assertNotIn('<link', response.text)
                    self.assertNotIn('Generated 404', response.text)
                    self.assertEqual((self.root / 'site/404.html').read_text(), generated)
        (self.root / 'site/404.html').unlink()
        app, client = self.client('/research')
        response = client.get('/research/docs/missing/', headers={'Accept': 'text/html'})
        self.assertEqual(response.status_code, 404)
        self.assertIn('href="/research/docs/">Open guide', response.text)

    def test_missing_docs_assets_and_non_html_clients_receive_true_json_404(self):
        _, client = self.client()
        for path, accept in (('/wm-states/docs/assets/missing.css', 'text/css,*/*;q=0.1'),
                             ('/wm-states/docs/missing/', 'application/json'),
                             ('/wm-states/docs/missing/', 'text/html;q=0,application/json')):
            with self.subTest(path=path, accept=accept):
                response = client.get(path, headers={'Accept': accept})
                self.assertEqual(response.status_code, 404)
                self.assertEqual(response.json()['detail'], 'Documentation page or asset not found.')
                self.assertNotIn('<html', response.text)
                self.assertEqual(response.headers['vary'], 'Accept')

    def test_invalid_prefixes_fail_before_a_run_manager_is_created(self):
        for value in ('', 'relative', '//host/path', '/a//b', '/..', '/a/./b', '/a/../b',
                      '/a?q=x', '/a#hash', '/a%2Fb', '/with space', '/a\\b', '/a//'):
            with self.subTest(value=value), patch('scripts.next.dashboard.app.RunManager') as manager:
                with self.assertRaisesRegex(ValueError, '--url-prefix'):
                    create_app(self.root, url_prefix=value)
                manager.assert_not_called()
        self.assertEqual(normalize_url_prefix('/one/two/'), '/one/two')
        self.assertEqual(DashboardURLs.from_prefix('/').dashboard, '/dashboard')

    def test_cli_passes_normalized_prefix_and_reports_both_urls(self):
        output = io.StringIO()
        with patch('sys.argv', ['dashboard', '--url-prefix', '/lab/project/']), \
                patch('scripts.next.dashboard.app.create_app') as factory, \
                patch('uvicorn.run') as run, redirect_stdout(output):
            main()
        factory.assert_called_once_with(url_prefix='/lab/project')
        self.assertIs(run.call_args.args[0], factory.return_value)
        self.assertIn('http://127.0.0.1:8000/lab/project/dashboard/', output.getvalue())
        self.assertIn('http://127.0.0.1:8000/lab/project/docs/', output.getvalue())

    def test_invalid_cli_prefix_fails_before_build_or_network_activity(self):
        with patch('sys.argv', ['dashboard', '--build', '--tailnet', '--url-prefix', '//invalid']), \
                patch('scripts.next.dashboard.build.build_assets') as build, \
                patch('scripts.next.dashboard.network.discover_tailnet_addresses') as discover, \
                redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main()
        build.assert_not_called()
        discover.assert_not_called()


if __name__ == '__main__':
    unittest.main()
