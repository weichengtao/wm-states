"""Same-origin REST/WebSocket API and production frontend for a local dashboard."""
import asyncio
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from scripts.next.dashboard.documentation import DocumentationFiles
from scripts.next.dashboard.models import RunRequest, TemplateRequest
from scripts.next.dashboard.runner import BusyError, RunManager, TERMINAL
from scripts.next.dashboard.schema import get_schema
from scripts.next.dashboard.templates import TemplateConflict, TemplateStore


def _trusted_origin(origin, host):
    if origin is None:
        return True  # Command-line clients do not send Origin.
    parsed = urlsplit(origin)
    return (parsed.scheme in ('http', 'https') and
            (parsed.netloc == host or parsed.netloc in ('localhost:5173', '127.0.0.1:5173')))


def create_app(repo_root: Path | None = None):
    root = (repo_root or Path(__file__).resolve().parents[3]).resolve()
    manager = RunManager(root)
    templates = TemplateStore(root)

    @asynccontextmanager
    async def lifespan(app):
        yield
        await manager.close()

    app = FastAPI(title='WM States · Next dashboard', version='1.0', lifespan=lifespan,
                  docs_url='/api/docs', redoc_url='/api/redoc',
                  openapi_url='/api/openapi.json',
                  swagger_ui_oauth2_redirect_url='/api/docs/oauth2-redirect')
    app.state.repo_root, app.state.runner = root, manager
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['localhost', '127.0.0.1', '[::1]'])

    @app.middleware('http')
    async def local_mutations(request: Request, call_next):
        if request.method not in ('GET', 'HEAD', 'OPTIONS') and not _trusted_origin(
                request.headers.get('origin'), request.headers.get('host')):
            return JSONResponse({'detail': 'Cross-origin mutations are not allowed.'}, status_code=403)
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'same-origin'
        if request.url.path.startswith('/api/'):
            response.headers['Cache-Control'] = 'no-store'
        return response

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        issues = '; '.join(f"{'.'.join(str(part) for part in error['loc'][1:])}: {error['msg']}"
                           for error in exc.errors())
        return JSONResponse({'detail': issues}, status_code=422)

    @app.get('/api/health')
    def health():
        return {'status': 'ok', 'service': 'wm-states-next', 'repo_root': str(root)}

    @app.get('/api/schema')
    def schema():
        return get_schema(root)

    @app.get('/api/templates')
    def list_templates():
        return templates.list()

    @app.post('/api/templates', status_code=201)
    def save_template(request: TemplateRequest):
        try:
            return templates.save(request)
        except TemplateConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post('/api/validate')
    def validate(request: RunRequest):
        try:
            result = manager.validate(request)
            return {key: result[key] for key in ('valid', 'command', 'resolved')}
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get('/api/jobs')
    def jobs():
        return {'jobs': manager.list_jobs()}

    @app.post('/api/jobs', status_code=201)
    async def start(request: RunRequest):
        try:
            return await manager.start(request)
        except BusyError as exc:
            raise HTTPException(409, str(exc)) from exc
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get('/api/jobs/{job_id}')
    def job(job_id: str):
        try:
            return manager.snapshot(job_id)
        except KeyError as exc:
            raise HTTPException(404, 'Job not found.') from exc

    @app.post('/api/jobs/{job_id}/cancel')
    async def cancel(job_id: str):
        try:
            return await manager.cancel(job_id)
        except KeyError as exc:
            raise HTTPException(404, 'Job not found.') from exc

    @app.websocket('/api/jobs/{job_id}/events')
    async def events(websocket: WebSocket, job_id: str):
        if not _trusted_origin(websocket.headers.get('origin'), websocket.headers.get('host')):
            await websocket.close(code=1008)
            return
        if job_id not in manager.jobs:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        try:
            while True:
                snapshot = manager.snapshot(job_id)
                await websocket.send_json(snapshot)
                if snapshot['status'] in TERMINAL:
                    break
                # Receive disconnect frames promptly, while publishing snapshots every second.
                try:
                    message = await asyncio.wait_for(websocket.receive(), timeout=1)
                    if message['type'] == 'websocket.disconnect':
                        break
                except asyncio.TimeoutError:
                    pass
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            with suppress(RuntimeError):
                await websocket.close()

    from scripts.next.dashboard.results import create_results_router
    app.include_router(create_results_router(root))

    @app.get('/docs', include_in_schema=False)
    def documentation_root():
        return RedirectResponse('/docs/', status_code=307)

    # Register before the frontend catch-all: documentation misses must remain 404s.
    app.mount('/docs', DocumentationFiles(root / 'site'), name='documentation')

    frontend = root / 'dashboard' / 'dist'
    if (frontend / 'assets').is_dir():
        app.mount('/assets', StaticFiles(directory=frontend / 'assets'), name='assets')

    @app.get('/{path:path}', include_in_schema=False)
    def frontend_route(path: str):
        if path.startswith('api/'):
            raise HTTPException(404, 'API endpoint not found.')
        asset = (frontend / path).resolve()
        if asset.is_relative_to(frontend.resolve()) and asset.is_file():
            return FileResponse(asset)
        if (frontend / 'index.html').is_file():
            return FileResponse(frontend / 'index.html')
        return JSONResponse({'detail': 'Frontend is not built. Run npm ci && npm run build in dashboard/, '
                                       'or use the Vite development server on localhost:5173.'}, status_code=503)

    return app
