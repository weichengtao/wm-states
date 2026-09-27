"""Discover local Tailscale addresses and bind only the requested interfaces."""
import errno
import ipaddress
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys


TAILSCALE_STATUS_TIMEOUT = 5
MACOS_TAILSCALE_CLI = Path('/Applications/Tailscale.app/Contents/MacOS/Tailscale')
TAILSCALE_NETWORKS = (
    ipaddress.ip_network('100.64.0.0/10'),
    ipaddress.ip_network('fd7a:115c:a1e0::/48'),
)


class TailnetNetworkError(RuntimeError):
    """An actionable startup failure, without a broader network fallback."""


def validate_tailnet_addresses(values) -> tuple[str, ...]:
    """Normalize a nonempty list of exact numeric addresses in Tailscale's ranges."""
    if not isinstance(values, (list, tuple)) or not values:
        raise TailnetNetworkError('Tailscale did not report any local tailnet IP addresses. '
                                  'Connect Tailscale, then restart the dashboard.')
    addresses = set()
    for value in values:
        try:
            if not isinstance(value, str) or '%' in value:
                raise ValueError('Invalid address type or scope')
            address = ipaddress.ip_address(value)
            if not any(address.version == network.version and address in network
                       for network in TAILSCALE_NETWORKS):
                raise ValueError('Address is outside Tailscale ranges')
        except ValueError as exc:
            raise TailnetNetworkError('Tailscale reported an invalid local tailnet IP address. '
                                      'Check `tailscale status --json` before retrying.') from exc
        addresses.add(address)
    return tuple(str(address) for address in sorted(addresses, key=lambda item: (item.version, int(item))))


def discover_tailnet_addresses() -> tuple[str, ...]:
    """Read this machine's addresses; never change Tailscale or Serve settings."""
    executable = shutil.which('tailscale')
    if (executable is None and sys.platform == 'darwin' and MACOS_TAILSCALE_CLI.is_file()
            and os.access(MACOS_TAILSCALE_CLI, os.X_OK)):
        executable = str(MACOS_TAILSCALE_CLI)
    if executable is None:
        raise TailnetNetworkError('The Tailscale CLI was not found. Add `tailscale` to PATH, '
                                  'or omit --tailnet to serve only on localhost.')
    try:
        result = subprocess.run([executable, 'status', '--json'], capture_output=True,
                                text=True, encoding='utf-8', errors='replace',
                                timeout=TAILSCALE_STATUS_TIMEOUT, check=False)
    except subprocess.TimeoutExpired as exc:
        raise TailnetNetworkError('Tailscale status timed out. Check that Tailscale is running '
                                  'and connected, then retry.') from exc
    except OSError as exc:
        raise TailnetNetworkError('The Tailscale CLI could not be started. Check `tailscale status` '
                                  'in this terminal, then retry.') from exc
    if result.returncode:
        raise TailnetNetworkError('Tailscale status failed. Run `tailscale status` in this terminal '
                                  'and connect or sign in before retrying.')
    try:
        status = json.loads(result.stdout)
    except (ValueError, TypeError) as exc:
        raise TailnetNetworkError('Tailscale returned an unreadable status response. '
                                  'Check `tailscale status --json` before retrying.') from exc
    if not isinstance(status, dict):
        raise TailnetNetworkError('Tailscale returned an invalid status response. '
                                  'Check `tailscale status --json` before retrying.')
    if status.get('BackendState') != 'Running':
        raise TailnetNetworkError('Tailscale is not connected. Open Tailscale and connect or sign in, '
                                  'then restart the dashboard.')
    local = status.get('Self')
    if not isinstance(local, dict):
        raise TailnetNetworkError('Tailscale did not report this machine. '
                                  'Check `tailscale status --json` before retrying.')
    return validate_tailnet_addresses(local.get('TailscaleIPs'))


def bind_dashboard_sockets(host: str, port: int, tailnet_ips: tuple[str, ...]) -> list[socket.socket]:
    """Prebind all interfaces before creating the app; clean up completely on failure."""
    if host not in ('127.0.0.1', 'localhost', '::1'):
        raise TailnetNetworkError('The local dashboard host must be 127.0.0.1, localhost, or ::1.')
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise TailnetNetworkError('The dashboard port must be between 1 and 65535.')
    addresses = ['127.0.0.1']
    if host in ('localhost', '::1'):
        addresses.append('::1')
    addresses.extend(validate_tailnet_addresses(tailnet_ips))
    sockets = []
    address = addresses[0]
    try:
        for address in addresses:
            ipv6 = ':' in address
            listener = socket.socket(socket.AF_INET6 if ipv6 else socket.AF_INET, socket.SOCK_STREAM)
            sockets.append(listener)
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if ipv6:
                listener.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            listener.bind((address, port, 0, 0) if ipv6 else (address, port))
            listener.setblocking(False)
    except BaseException as exc:
        for listener in sockets:
            listener.close()
        if not isinstance(exc, OSError):
            raise
        endpoint = f'[{address}]:{port}' if ':' in address else f'{address}:{port}'
        if exc.errno == errno.EADDRINUSE:
            detail = 'This address/port is already in use. Choose another --port or stop its existing server.'
        elif exc.errno == errno.EADDRNOTAVAIL:
            detail = ('This address cannot be bound on this computer. Check that Tailscale is connected '
                      'and restart the dashboard. If your Tailscale installation does not support direct '
                      'IP binding, omit --tailnet for local-only access while resolving '
                      'the interface configuration.')
        else:
            detail = 'Check local networking permissions and Tailscale connectivity, then retry.'
        raise TailnetNetworkError(f'Cannot listen on {endpoint}. {detail} '
                                  'No broader network interface was used.') from exc
    return sockets
