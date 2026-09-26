"""Local MkDocs serving and an accessible first-build page."""
from pathlib import Path

from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import Headers


DOCS_BUILD_COMMAND = 'uv run --locked --group dashboard --group docs python -m mkdocs build --strict'
MISSING_DOCS_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Build the documentation · WM States</title>
<style>
:root { color-scheme: light dark; --bg:#f3f7f6; --card:#fff; --ink:#152f35;
  --muted:#50666c; --border:#dce8e5; --accent:#086664; --code:#edf4f2; }
* { box-sizing:border-box; } body { margin:0; min-height:100vh; display:grid;
  place-items:center; padding:32px 20px; background:var(--bg); color:var(--ink);
  font:16px/1.65 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
main { width:min(100%,680px); padding:clamp(24px,5vw,48px); background:var(--card);
  border:1px solid var(--border); border-radius:24px; box-shadow:0 18px 70px #153b3510; }
.eyebrow { color:var(--accent); font-size:12px; font-weight:750; letter-spacing:.14em; }
h1 { font-size:clamp(27px,5vw,36px); letter-spacing:-.035em; line-height:1.2; margin:18px 0; }
p { color:var(--muted); margin:0 0 24px; } .command-label { font-size:13px; font-weight:650;
  margin-bottom:9px; color:var(--ink); } .command { background:var(--code);
  padding:18px; border:1px solid var(--border); border-radius:12px; }
pre { margin:0; white-space:pre-wrap; overflow-wrap:anywhere; } code { font:13px/1.8
  ui-monospace,SFMono-Regular,Consolas,monospace; user-select:all; }
.copy-row { display:flex; align-items:center; gap:12px; margin-top:12px; }
button,a { font:inherit; } button { font-size:13px; font-weight:650; border:1px solid var(--border);
  border-radius:8px; padding:7px 12px; color:var(--accent); background:var(--card); cursor:pointer; }
#copy-status { font-size:12px; color:var(--muted); } .note { font-size:13px; margin:16px 0 28px; }
a { color:var(--accent); font-weight:650; text-underline-offset:4px; }
a:focus-visible,button:focus-visible,code:focus-visible { outline:3px solid var(--accent); outline-offset:4px; }
.actions { display:flex; flex-wrap:wrap; align-items:center; gap:18px; }
.primary { color:white; background:var(--accent); text-decoration:none; padding:10px 17px; border-radius:9px; }
@media(prefers-color-scheme:dark) { :root { --bg:#101c22; --card:#17292f; --ink:#e9f4f1;
  --muted:#b2c7c8; --border:#30474b; --accent:#82dbca; --code:#102127; } .primary { color:#102127; } }
</style>
</head>
<body><main aria-labelledby="docs-title">
<div class="eyebrow">WM STATES / PIPELINE GUIDE</div>
<h1 id="docs-title">Documentation is not built yet</h1>
<p>The dashboard is ready. Build the local guide to view the analysis methods and instructions here.</p>
<div class="command-label">Run this from the repository root</div>
<div class="command"><pre><code id="build-command" tabindex="0" aria-label="Documentation build command">__BUILD_COMMAND__</code></pre>
<div class="copy-row"><button id="copy-command" type="button" hidden>Copy command</button><span id="copy-status" role="status" aria-live="polite"></span></div></div>
<p class="note">This builds only the documentation. When it finishes, reload this page. Your running analyses can continue.</p>
<div class="actions"><a class="primary" href="/">Return to dashboard</a><a href="">Reload page</a></div>
</main>
<script>
const copy = document.getElementById('copy-command');
if (navigator.clipboard) {
  copy.hidden = false;
  copy.addEventListener('click', async () => {
    const status = document.getElementById('copy-status');
    try {
      await navigator.clipboard.writeText(document.getElementById('build-command').textContent);
      status.textContent = 'Command copied';
    } catch {
      status.textContent = 'Select the command above and copy it manually.';
    }
  });
}
</script>
</body></html>""".replace('__BUILD_COMMAND__', DOCS_BUILD_COMMAND)


def _accepts_html(scope):
    for item in Headers(scope=scope).get('accept', '').split(','):
        media, *parameters = item.strip().lower().split(';')
        if media != 'text/html':
            continue
        quality = next((value.strip()[2:] for value in parameters if value.strip().startswith('q=')), '1')
        try:
            return float(quality) > 0
        except ValueError:
            return False
    return False


class DocumentationFiles(StaticFiles):
    """Serve MkDocs pages, including sites built after the server starts."""

    def __init__(self, directory: Path):
        self.site_directory = directory
        super().__init__(directory=directory, html=True, check_dir=False)

    async def check_config(self):
        # Missing builds are an ordinary 503 response, not a startup/ASGI exception.
        # Check on every request so an independently completed build becomes available.
        return None

    async def get_response(self, path, scope):
        if not (self.site_directory / 'index.html').is_file():
            headers = {'Cache-Control': 'no-store', 'Vary': 'Accept'}
            if _accepts_html(scope):
                return HTMLResponse(MISSING_DOCS_HTML, status_code=503, headers=headers)
            return JSONResponse({
                'detail': f'Documentation is not built. From the repository root, run {DOCS_BUILD_COMMAND}, '
                          'or launch the dashboard with --build to build both sites.',
            }, status_code=503, headers=headers)
        return await super().get_response(path, scope)
