"""Tailnet startup discovers exact interfaces and fails without broadening access."""
from contextlib import redirect_stderr, redirect_stdout
import errno
import io
import json
import socket
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from scripts.next.dashboard.__main__ import main
from scripts.next.dashboard.network import (
    TAILSCALE_STATUS_TIMEOUT, TailnetNetworkError, bind_dashboard_sockets,
    discover_tailnet_addresses, validate_tailnet_addresses,
)


TAILNET_IPS = ('100.94.253.10', 'fd7a:115c:a1e0::2101:fd0d')
NETWORK_MODULE = 'scripts.next.dashboard.network'


class TailnetDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.which = patch(f'{NETWORK_MODULE}.shutil.which', return_value='/test/tailscale').start()
        self.run = patch(f'{NETWORK_MODULE}.subprocess.run').start()
        self.addCleanup(patch.stopall)
        self.status = {'BackendState': 'Running', 'Self': {'TailscaleIPs': list(TAILNET_IPS)},
                       'Peer': {'other': {'TailscaleIPs': ['100.64.0.2']}}}
        self.respond(self.status)

    def respond(self, status):
        self.run.return_value = SimpleNamespace(returncode=0, stdout=json.dumps(status))

    def test_reads_only_self_addresses_with_a_bounded_read_only_command(self):
        self.assertEqual(discover_tailnet_addresses(), TAILNET_IPS)
        self.which.assert_called_once_with('tailscale')
        self.run.assert_called_once_with(
            ['/test/tailscale', 'status', '--json'], capture_output=True,
            text=True, encoding='utf-8', errors='replace', timeout=TAILSCALE_STATUS_TIMEOUT, check=False)

    def test_normalizes_deduplicates_and_sorts_addresses(self):
        addresses = ['FD7A:115C:A1E0:0:0:0:2101:FD0D', TAILNET_IPS[0], TAILNET_IPS[1]]
        self.assertEqual(validate_tailnet_addresses(addresses), TAILNET_IPS)

    def test_accepts_ipv4_or_ipv6_only_installations(self):
        for address in TAILNET_IPS:
            with self.subTest(address=address):
                self.status['Self']['TailscaleIPs'] = [address]
                self.respond(self.status)
                self.assertEqual(discover_tailnet_addresses(), (address,))

    def test_validates_exact_tailscale_address_ranges(self):
        for value in ['0.0.0.0', '::', '127.0.0.1', '::1', '192.168.1.1', '100.63.255.255',
                      '100.128.0.0', 'fd7a:115c:a1e1::1', 'example.ts.net', '*.ts.net',
                      '[fd7a:115c:a1e0::1]', 'fd7a:115c:a1e0::1%utun4', None, 123]:
            with self.subTest(value=value), self.assertRaisesRegex(TailnetNetworkError, 'invalid local'):
                validate_tailnet_addresses([value])
        for value in ['100.64.0.0', '100.127.255.255', 'fd7a:115c:a1e0::1']:
            self.assertEqual(validate_tailnet_addresses([value]), (value,))

    def test_missing_cli_explains_local_fallback_without_running_anything(self):
        self.which.return_value = None
        with patch(f'{NETWORK_MODULE}.sys.platform', 'linux'), \
                self.assertRaisesRegex(TailnetNetworkError, 'omit --tailnet'):
            discover_tailnet_addresses()
        self.run.assert_not_called()

    def test_macos_can_use_the_standard_app_cli_without_a_path_entry(self):
        self.which.return_value = None
        with patch(f'{NETWORK_MODULE}.sys.platform', 'darwin'), \
                patch(f'{NETWORK_MODULE}.Path.is_file', return_value=True), \
                patch(f'{NETWORK_MODULE}.os.access', return_value=True):
            self.assertEqual(discover_tailnet_addresses(), TAILNET_IPS)
        self.assertEqual(self.run.call_args.args[0],
                         ['/Applications/Tailscale.app/Contents/MacOS/Tailscale', 'status', '--json'])

    def test_macos_fallback_requires_an_executable_file(self):
        self.which.return_value = None
        for is_file, executable in [(False, True), (True, False)]:
            with self.subTest(is_file=is_file, executable=executable), \
                    patch(f'{NETWORK_MODULE}.sys.platform', 'darwin'), \
                    patch(f'{NETWORK_MODULE}.Path.is_file', return_value=is_file), \
                    patch(f'{NETWORK_MODULE}.os.access', return_value=executable), \
                    self.assertRaisesRegex(TailnetNetworkError, 'CLI was not found'):
                discover_tailnet_addresses()
        self.run.assert_not_called()

    def test_timeout_does_not_retry_or_modify_tailscale(self):
        self.run.side_effect = subprocess.TimeoutExpired('tailscale', TAILSCALE_STATUS_TIMEOUT)
        with self.assertRaisesRegex(TailnetNetworkError, 'timed out'):
            discover_tailnet_addresses()
        self.assertEqual(self.run.call_count, 1)

    def test_unlaunchable_cli_is_actionable(self):
        self.run.side_effect = PermissionError('blocked')
        with self.assertRaisesRegex(TailnetNetworkError, 'could not be started'):
            discover_tailnet_addresses()

    def test_failed_cli_does_not_print_status_or_peer_information(self):
        self.run.return_value = SimpleNamespace(returncode=1, stdout='private peer info', stderr='secret')
        with self.assertRaisesRegex(TailnetNetworkError, 'status failed') as caught:
            discover_tailnet_addresses()
        self.assertNotIn('private', str(caught.exception))
        self.assertNotIn('secret', str(caught.exception))

    def test_disconnected_states_require_a_connection(self):
        for state in ['Stopped', 'NeedsLogin', 'Starting', 'NoState', None]:
            with self.subTest(state=state), self.assertRaisesRegex(TailnetNetworkError, 'not connected'):
                self.status['BackendState'] = state
                self.respond(self.status)
                discover_tailnet_addresses()

    def test_malformed_status_does_not_fall_back_to_peer_addresses(self):
        for status in [[], None, 'Running']:
            with self.subTest(status=status), self.assertRaisesRegex(TailnetNetworkError, 'invalid status'):
                self.respond(status)
                discover_tailnet_addresses()
        self.run.return_value = SimpleNamespace(returncode=0, stdout='not JSON')
        with self.assertRaisesRegex(TailnetNetworkError, 'unreadable status'):
            discover_tailnet_addresses()
        for local in [None, [], {}, {'TailscaleIPs': []}, {'TailscaleIPs': TAILNET_IPS[0]}]:
            with self.subTest(local=local), self.assertRaises(TailnetNetworkError):
                self.status['Self'] = local
                self.respond(self.status)
                discover_tailnet_addresses()


class DashboardBindingTests(unittest.TestCase):
    def setUp(self):
        self.listeners = [MagicMock(name=f'listener-{index}') for index in range(4)]
        self.socket = patch(f'{NETWORK_MODULE}.socket.socket', side_effect=self.listeners).start()
        self.addCleanup(patch.stopall)

    def test_prebinds_exact_loopback_and_all_tailnet_addresses_without_listening(self):
        sockets = bind_dashboard_sockets('127.0.0.1', 8020, TAILNET_IPS)
        self.assertEqual(sockets, self.listeners[:3])
        self.assertEqual(self.socket.call_args_list[0].args, (socket.AF_INET, socket.SOCK_STREAM))
        self.assertEqual(self.socket.call_args_list[2].args, (socket.AF_INET6, socket.SOCK_STREAM))
        self.listeners[0].bind.assert_called_once_with(('127.0.0.1', 8020))
        self.listeners[1].bind.assert_called_once_with((TAILNET_IPS[0], 8020))
        self.listeners[2].bind.assert_called_once_with((TAILNET_IPS[1], 8020, 0, 0))
        self.listeners[2].setsockopt.assert_any_call(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        for listener in sockets:
            listener.setblocking.assert_called_once_with(False)
            listener.listen.assert_not_called()
            listener.close.assert_not_called()

    def test_ipv6_local_selection_keeps_ipv4_loopback_for_serve(self):
        for host in ['::1', 'localhost']:
            with self.subTest(host=host):
                self.socket.side_effect = self.listeners
                for listener in self.listeners:
                    listener.reset_mock()
                sockets = bind_dashboard_sockets(host, 8000, (TAILNET_IPS[0],))
                self.assertEqual(len(sockets), 3)
                self.listeners[0].bind.assert_called_once_with(('127.0.0.1', 8000))
                self.listeners[1].bind.assert_called_once_with(('::1', 8000, 0, 0))

    def test_invalid_bind_request_never_opens_a_socket(self):
        for host, port, addresses in [('0.0.0.0', 8000, TAILNET_IPS),
                                      ('::', 8000, TAILNET_IPS),
                                      ('127.0.0.1', 0, TAILNET_IPS),
                                      ('127.0.0.1', True, TAILNET_IPS),
                                      ('127.0.0.1', 8000, ()),
                                      ('127.0.0.1', 8000, ('0.0.0.0',))]:
            with self.subTest(host=host, port=port, addresses=addresses), \
                    self.assertRaises(TailnetNetworkError):
                bind_dashboard_sockets(host, port, addresses)
        self.socket.assert_not_called()

    def test_late_bind_failure_closes_every_opened_socket_without_fallback(self):
        self.listeners[2].bind.side_effect = OSError(errno.EADDRNOTAVAIL, 'not assigned')
        with self.assertRaisesRegex(TailnetNetworkError, 'local-only access'):
            bind_dashboard_sockets('127.0.0.1', 8000, TAILNET_IPS)
        for listener in self.listeners[:3]:
            listener.close.assert_called_once_with()
        self.listeners[3].bind.assert_not_called()
        self.assertEqual(self.socket.call_count, 3)

    def test_port_conflict_and_permission_failures_are_actionable(self):
        for error, phrase in [(errno.EADDRINUSE, 'another --port'),
                              (errno.EACCES, 'permissions')]:
            with self.subTest(error=error):
                self.socket.side_effect = self.listeners
                self.listeners[0].bind.side_effect = OSError(error, 'failure')
                with self.assertRaisesRegex(TailnetNetworkError, phrase):
                    bind_dashboard_sockets('127.0.0.1', 8000, TAILNET_IPS)

    def test_socket_creation_failure_closes_earlier_sockets(self):
        self.socket.side_effect = [self.listeners[0], OSError(errno.EMFILE, 'too many files')]
        with self.assertRaises(TailnetNetworkError):
            bind_dashboard_sockets('127.0.0.1', 8000, TAILNET_IPS)
        self.listeners[0].close.assert_called_once_with()

    def test_interrupted_setup_closes_opened_sockets(self):
        self.listeners[1].bind.side_effect = KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            bind_dashboard_sockets('127.0.0.1', 8000, TAILNET_IPS)
        for listener in self.listeners[:2]:
            listener.close.assert_called_once_with()


class DashboardTailnetLauncherTests(unittest.TestCase):
    def setUp(self):
        self.discover = patch(f'{NETWORK_MODULE}.discover_tailnet_addresses', return_value=TAILNET_IPS).start()
        self.listeners = [MagicMock(), MagicMock()]
        self.listeners[0].getsockname.return_value = ('127.0.0.1', 8020)
        self.listeners[1].getsockname.return_value = (TAILNET_IPS[1], 8020, 0, 0)
        self.bind = patch(f'{NETWORK_MODULE}.bind_dashboard_sockets', return_value=self.listeners).start()
        self.create_app = patch('scripts.next.dashboard.app.create_app').start()
        self.config = patch('uvicorn.Config').start()
        self.server = patch('uvicorn.Server').start()
        self.run = patch('uvicorn.run').start()
        self.addCleanup(patch.stopall)

    def invoke(self, *args):
        output = io.StringIO()
        with patch('sys.argv', ['dashboard', '--tailnet', '--port', '8020', *args]), \
                redirect_stdout(output):
            main()
        return output.getvalue()

    def test_uses_one_app_and_server_for_all_listeners_and_reports_urls(self):
        output = self.invoke()
        self.bind.assert_called_once_with('127.0.0.1', 8020, TAILNET_IPS)
        self.create_app.assert_called_once_with(tailnet_ips=TAILNET_IPS, url_prefix='/wm-states')
        self.config.assert_called_once_with(self.create_app.return_value, host='127.0.0.1', port=8020,
                                            proxy_headers=True, forwarded_allow_ips='127.0.0.1,::1')
        self.server.assert_called_once_with(self.config.return_value)
        self.server.return_value.run.assert_called_once_with(sockets=self.listeners)
        self.run.assert_not_called()
        self.assertIn('http://127.0.0.1:8020/wm-states/docs/', output)
        self.assertIn(f'http://[{TAILNET_IPS[1]}]:8020/', output)
        for listener in self.listeners:
            listener.close.assert_called_once_with()

    def test_selected_local_host_is_passed_to_binding(self):
        self.invoke('--host', '::1')
        self.bind.assert_called_once_with('::1', 8020, TAILNET_IPS)

    def test_discovery_or_binding_failure_does_not_create_an_app(self):
        for operation in [self.discover, self.bind]:
            with self.subTest(operation=operation), redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                operation.side_effect = TailnetNetworkError('Actionable startup failure')
                self.invoke()
            operation.side_effect = None
        self.create_app.assert_not_called()
        self.server.assert_not_called()

    def test_app_creation_or_server_failure_closes_all_sockets(self):
        for operation in [self.create_app, self.server.return_value.run]:
            with self.subTest(operation=operation):
                operation.side_effect = RuntimeError('startup failed')
                with self.assertRaisesRegex(RuntimeError, 'startup failed'):
                    self.invoke()
                operation.side_effect = None
                for listener in self.listeners:
                    listener.close.assert_called_once_with()
                    listener.reset_mock()

    def test_without_flag_retains_local_only_launcher_and_skips_discovery(self):
        with patch('sys.argv', ['dashboard']):
            main()
        self.discover.assert_not_called()
        self.bind.assert_not_called()
        self.create_app.assert_called_once_with(url_prefix='/wm-states')
        self.run.assert_called_once_with(self.create_app.return_value, host='127.0.0.1', port=8000,
                                         proxy_headers=True, forwarded_allow_ips='127.0.0.1,::1')

    def test_keyboard_interrupt_is_a_quiet_shutdown_and_closes_all_sockets(self):
        self.server.return_value.run.side_effect = KeyboardInterrupt
        self.invoke()
        for listener in self.listeners:
            listener.close.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
