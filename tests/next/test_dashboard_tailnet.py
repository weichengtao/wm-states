"""Same-origin dashboard behavior over local, direct-IP and Serve connections."""
from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from scripts.next.dashboard.app import create_app
from scripts.next.dashboard.network import TailnetNetworkError
from tests.next.test_dashboard_runner import fixture_root, request


TAILNET_IPS = ('100.94.253.10', 'fd7a:115c:a1e0::2101:fd0d')
ORIGINS = ('http://127.0.0.1:8000', 'http://localhost:8000',
           'http://[::1]:8000', 'http://100.94.253.10:8000',
           'http://[fd7a:115c:a1e0::2101:fd0d]:8000',
           'https://workstation.example.ts.net:8443')


class DashboardTailnetTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = fixture_root(directory.name)
        frontend = self.root / 'dashboard/dist'
        (frontend / 'assets').mkdir(parents=True)
        (frontend / 'index.html').write_text('<html>Dashboard</html>')
        (frontend / 'assets/app.js').write_text('/* frontend */')
        guide = self.root / 'site/next/methods'
        guide.mkdir(parents=True)
        (guide / 'index.html').write_text('<html>Methods</html>')
        (self.root / 'site/index.html').write_text('<html>Guide</html>')
        self.app = create_app(self.root, tailnet_ips=TAILNET_IPS)
        self.app.state.runner.jobs['completed'] = dict(
            id='completed', status='complete', logs=['same manager'],
            created_at='2026-01-01T00:00:00Z', stages=[])

    def client(self, origin, app=None, **kwargs):
        client = TestClient(app or self.app, base_url=origin, **kwargs)
        self.addCleanup(client.close)
        return client

    def test_default_remains_local_only(self):
        app = create_app(self.root)
        for origin in ORIGINS:
            with self.subTest(origin=origin):
                status = self.client(origin, app).get('/api/health').status_code
                self.assertEqual(status, 200 if origin in ORIGINS[:3] else 400)

    def test_all_route_families_share_one_app_across_enabled_origins(self):
        for origin in ORIGINS:
            with self.subTest(origin=origin):
                client = self.client(origin)
                for path in ('/', '/runs', '/assets/app.js', '/wm-states/docs/',
                             '/wm-states/docs/next/methods/', '/api/health', '/api/schema',
                             '/wm-states/dashboard/', '/wm-states/dashboard/assets/app.js',
                             '/wm-states/dashboard/api/health', '/wm-states/dashboard/api/schema'):
                    self.assertEqual(client.get(path).status_code, 200, (origin, path))
                self.assertEqual(client.get('/api/jobs').json()['jobs'][0]['id'], 'completed')
                redirect = client.get('/wm-states/docs/next/methods', follow_redirects=False)
                self.assertEqual(redirect.headers['location'], origin + '/wm-states/docs/next/methods/')
                validated = client.post('/api/validate', json=request().model_dump(),
                                        headers={'Origin': origin})
                self.assertEqual(validated.status_code, 200, validated.text)
                completed = client.get('/api/paths/complete', params={'path': 'data/'},
                                       headers={'Origin': origin, 'Sec-Fetch-Site': 'same-origin'})
                self.assertEqual(completed.status_code, 200, completed.text)
                with client.websocket_connect(origin.replace('http', 'ws', 1) + '/wm-states/dashboard/api/jobs/completed/events',
                                              headers={'Origin': origin}) as websocket:
                    self.assertEqual(websocket.receive_json()['logs'], ['same manager'])

    def test_unsupported_hosts_are_rejected_even_in_tailnet_mode(self):
        client = self.client(ORIGINS[3])
        for host in ('100.94.253.11:8000', '192.168.1.20:8000', '0.0.0.0:8000',
                     '[fd7a:115c:a1e0::2]:8000', 'evil.example', 'ts.net',
                     'workstation.ts.net.evil.example', 'not-ts.net', 'workstation'):
            with self.subTest(host=host):
                self.assertEqual(client.get('/api/health', headers={'Host': host}).status_code, 400)

    def test_rest_websocket_and_path_origins_remain_restricted(self):
        for origin in ORIGINS[3:]:
            client = self.client(origin)
            for untrusted in ('https://evil.example', 'null', 'http://[invalid',
                              origin + ':9000', ORIGINS[0]):
                with self.subTest(origin=origin, untrusted=untrusted):
                    headers = {'Origin': untrusted}
                    response = client.post('/api/validate', json=request().model_dump(), headers=headers)
                    self.assertEqual(response.status_code, 403)
                    self.assertEqual(client.get('/api/paths/complete', headers=headers).status_code, 403)
                    with self.assertRaises(WebSocketDisconnect) as error:
                        with client.websocket_connect(origin.replace('http', 'ws', 1) + '/api/jobs/completed/events', headers=headers):
                            pass
                    self.assertEqual(error.exception.code, 1008)
            self.assertEqual(client.get('/api/paths/complete', headers={
                'Origin': origin, 'Sec-Fetch-Site': 'cross-site'}).status_code, 403)

    def test_trusted_loopback_proxy_preserves_https_redirect_and_websocket(self):
        app = ProxyHeadersMiddleware(self.app, trusted_hosts=['127.0.0.1', '::1'])
        for proxy in ('127.0.0.1', '::1'):
            with self.subTest(proxy=proxy):
                client = self.client('http://workstation.example.ts.net:8443', app,
                                     client=(proxy, 12345))
                headers = {'X-Forwarded-Proto': 'https', 'X-Forwarded-For': '100.64.0.2'}
                response = client.get('/wm-states/docs/next/methods', headers=headers, follow_redirects=False)
                self.assertEqual(response.headers['location'], ORIGINS[-1] + '/wm-states/docs/next/methods/')
                with client.websocket_connect('ws://workstation.example.ts.net:8443/wm-states/dashboard/api/jobs/completed/events', headers={
                    **headers, 'Origin': ORIGINS[-1],
                }) as websocket:
                    self.assertEqual(websocket.receive_json()['status'], 'complete')

    def test_direct_tailnet_peer_cannot_spoof_forwarded_scheme_or_host(self):
        app = ProxyHeadersMiddleware(self.app, trusted_hosts=['127.0.0.1', '::1'])
        client = self.client(ORIGINS[3], app, client=('100.64.0.2', 12345))
        headers = {'X-Forwarded-Proto': 'https', 'X-Forwarded-For': '127.0.0.1',
                   'X-Forwarded-Host': 'evil.example', 'Tailscale-User-Login': 'spoof@example.com'}
        response = client.get('/wm-states/docs/next/methods', headers=headers, follow_redirects=False)
        self.assertEqual(response.headers['location'], ORIGINS[3] + '/wm-states/docs/next/methods/')
        response = client.post('/api/validate', json=request().model_dump(),
                               headers={**headers, 'Origin': 'https://evil.example'})
        self.assertEqual(response.status_code, 403)

    def test_factory_rejects_non_tailnet_addresses(self):
        for addresses in (('0.0.0.0',), ('example.ts.net',), ('192.168.1.1',), ('::',), '100.64.0.1'):
            with self.subTest(addresses=addresses), self.assertRaises(TailnetNetworkError):
                create_app(Path(self.root), tailnet_ips=addresses)


if __name__ == '__main__':
    unittest.main()
