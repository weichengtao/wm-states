"""Start the local dashboard: python -m scripts.next.dashboard."""
import argparse


def main():
    parser = argparse.ArgumentParser(description='Serve the next pipeline dashboard on this computer.')
    parser.add_argument('--host', choices=('127.0.0.1', 'localhost', '::1'), default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('--port must be between 1 and 65535')
    import uvicorn
    uvicorn.run('scripts.next.dashboard.app:create_app', factory=True, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
