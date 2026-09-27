"""Start the local dashboard: python -m scripts.next.dashboard."""
import argparse


def main():
    parser = argparse.ArgumentParser(description='Serve the next pipeline dashboard on this computer.')
    parser.add_argument('--host', choices=('127.0.0.1', 'localhost', '::1'), default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--tailnet', action='store_true',
                        help='Also listen on this computer\'s discovered Tailscale IP addresses; '
                             'requires a connected Tailscale CLI. Keeps localhost access enabled.')
    parser.add_argument('--build', action='store_true',
                        help='Build the dashboard and MkDocs documentation before starting the server.')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('--port must be between 1 and 65535')
    if args.build:
        from scripts.next.dashboard.build import build_assets
        try:
            build_assets()
        except RuntimeError as exc:
            parser.error(str(exc))
    import uvicorn
    if not args.tailnet:
        uvicorn.run('scripts.next.dashboard.app:create_app', factory=True, host=args.host, port=args.port)
        return

    from scripts.next.dashboard.network import (
        TailnetNetworkError, bind_dashboard_sockets, discover_tailnet_addresses,
    )
    try:
        tailnet_ips = discover_tailnet_addresses()
        sockets = bind_dashboard_sockets(args.host, args.port, tailnet_ips)
    except TailnetNetworkError as exc:
        parser.error(str(exc))
    try:
        from scripts.next.dashboard.app import create_app
        app = create_app(tailnet_ips=tailnet_ips)
        config = uvicorn.Config(app, host=args.host, port=args.port, proxy_headers=True,
                                forwarded_allow_ips='127.0.0.1,::1')
        for listener in sockets:
            address = listener.getsockname()[0]
            authority = f'[{address}]' if ':' in address else address
            print(f'Dashboard: http://{authority}:{args.port}/  '
                  f'Guide: http://{authority}:{args.port}/docs/', flush=True)
        uvicorn.Server(config).run(sockets=sockets)
    except KeyboardInterrupt:
        pass  # Match uvicorn.run: Ctrl+C is a normal shutdown, not a traceback.
    finally:
        for listener in sockets:
            listener.close()


if __name__ == '__main__':
    main()
