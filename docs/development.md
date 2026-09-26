# Development

## Source layout and tests

Analysis code belongs in `scripts/next/`, JSON presets in `configs/next/`, and
Python tests in `tests/next/`. Dashboard frontend code and TypeScript tests live
in `dashboard/src/`; the FastAPI backend lives in `scripts/next/dashboard/`.
The new implementation does not import historical
analysis scripts. Documentation lives in `docs/`.

```bash
uv run --group dashboard --group docs --locked python -m unittest discover -s tests/next -v
uv run --group dashboard --group docs --locked python -m unittest discover -s tests -v
```

The optional dashboard group supplies FastAPI and the HTTP test client. The
docs group enables integration checks against actual MkDocs builds; those
checks skip when that optional group is absent.
The second command includes both new and historical tests. Individual scripts
remain directly executable, and `python -m scripts.next.pipeline` is supported.
Record analysis validation in [the next pipeline log](validation/next.md), with
the date, exact commands, test counts, integration scope, and limitations.
Generated datasets and cache outputs stay outside version control.

## Downstream analysis modules

Use `scripts/next/screening_metadata.py` for validated screening fields,
population membership, preferred-cell ranking, and check-aware display labels.
Do not duplicate cell-group definitions in individual consumers. The activity
caller requests finite-PEV ranking explicitly; model preparation keeps the
screening order. Stable group keys are storage and formula identifiers, not
claims that screening checks were enabled.

`session_inputs.py` owns downstream MAT loading and correct preferred-cue
trial alignment. Stage-specific outcome validation stays with its analysis.
Activity code is divided into `activity_types.py` (records and configuration),
`activity_preparation.py` (numerical preparation), `activity_plots.py` (figures),
and `compare_activity_across_states.py` (CLI orchestration).

Keep selectivity PEV and PCA explained variance in separate typed metadata.
Shared loading and grouping must not merge normalization procedures: activity
plots use balanced cue groups, full-data model tables use cached preferred-cue
trials, and model CV estimates normalization from training rows only.

## Dashboard frontend

For everyday use, follow [first-time setup](next/dashboard.md#first-time-setup)
and [Open the dashboard again](next/dashboard.md#open-the-dashboard-again).
The backend serves the built interface at `/` and the built guide at `/docs/`
on port **8000**. The API reference is at `/api/docs`, ReDoc at `/api/redoc`, and
the OpenAPI schema at `/api/openapi.json`. A separate Vite server
is only needed while editing the frontend. The
[frontend development workflow](next/dashboard.md#frontend-development) uses
Vite on port **5173**, with API, WebSocket, and built guide requests proxied to
port 8000.
Frontend packages are locked in `dashboard/package-lock.json`.

```bash
cd dashboard
npm ci
npm test
npm run build
npm run format:check
```

The build checks TypeScript before bundling. Helper tests cover configuration
reuse and defaults, comparison warnings, and formatting; backend tests exercise
real subprocess control and result APIs. Use a smoke run and browser review when
changing the launch or progress workflow.
See [generated files and reusable presets](next/dashboard.md#generated-files-and-reusable-presets)
for the distinction between local build/job files and configurations to commit.

## Serve the documentation

For normal use, the dashboard's `--build` option builds both interfaces and
serves the guide at `/docs/` on the same server. MkDocs Material is provided
through the optional `docs` dependency group.

For **documentation editing with automatic reload**, run from the repository root:

```bash
uv run --group dashboard --group docs --locked mkdocs serve --dev-addr 127.0.0.1:8001
```

Open [http://127.0.0.1:8001/](http://127.0.0.1:8001/). Port 8001 keeps the docs
preview separate from the dashboard on port 8000, allowing both to run at once.
The preview reloads when Markdown or `mkdocs.yml` changes. Keep its terminal
running and stop it with Ctrl+C.
The command retains both optional groups in the shared Python environment.
During frontend editing, use
`VITE_DOCS_BASE_URL=http://127.0.0.1:8001/ npm --prefix dashboard run dev` to
open this live preview from the Help panel. The default Vite proxy serves the
backend's built guide instead. Rebuild the integrated guide after editing;
`mkdocs serve` does not update the production `site/` directory.

The analysis dependencies do not require the documentation group. A dedicated
docs-only environment can instead use `uv run --only-group docs --locked` before
the same MkDocs commands.

## Build static HTML

```bash
uv run --group dashboard --group docs --locked mkdocs build --strict
```

The generated site is written to `site/`, which is ignored by Git. Strict mode
fails on build warnings, including missing documentation links and anchors.
All pages are explicit entries in `mkdocs.yml`, and the built-in search plugin
indexes their content. The theme uses system fonts and bundled assets.

The dashboard mounts this directory at `/docs/`. It detects a build added after
startup; no backend restart is needed. Missing builds return a helpful 503,
while missing guide pages return 404, never the React app. The docs routes are
registered before the frontend fallback.

You can also serve `site/` with any static HTTP server. Relative documentation
links and assets work at a site root or project subpath. `docs/overrides/main.html`
adds a **Back to dashboard** bar only on a loopback host under `/docs/`; it is
hidden on standalone/public sites. The override source is excluded from build
output. No backend requests or recordings are needed to browse the guide.

## Publish the guide on GitHub Pages

The repository keeps one MkDocs source tree for local and public use. Publishing
is optional and has not been enabled by the dashboard integration. Set
`MKDOCS_SITE_URL` to your eventual public URL so canonical URLs and the sitemap
match its domain and repository prefix. For this repository, a project Pages
build would use:

```bash
MKDOCS_SITE_URL=https://weichengtao.github.io/wm-states/ \
  uv run --only-group docs --locked mkdocs build --strict
```

This **builds locally**; it does not publish. Later, configure GitHub Pages to
deploy the contents of `site/`, using GitHub Actions or MkDocs' `gh-deploy`
workflow. See [MkDocs deployment guidance](https://www.mkdocs.org/user-guide/deploying-your-docs/).
The public site only needs generated documentation; the FastAPI service,
recordings, caches, and React dashboard remain local.

All documentation-to-documentation links should stay relative `.md` links.
Do not prefix them with `/docs/` or `/wm-states/`. MkDocs rewrites them for the
built pages; see [its link guidance](https://www.mkdocs.org/user-guide/writing-your-docs/#linking-to-pages).
The default empty `MKDOCS_SITE_URL` keeps local builds independent of a fixed
host or port. Setting the public URL changes canonical metadata, not the
dashboard server's mount point.

The dashboard uses its own local guide by default. To deliberately link it to a
published guide, build its frontend with that guide's base URL:

```bash
VITE_DOCS_BASE_URL=https://weichengtao.github.io/wm-states/ \
  npm --prefix dashboard run build
```

The URL must include the repository prefix. This setting is compiled into the
frontend; changing it requires another build. Omit it to restore `/docs/`.
Stage anchors and topic paths are appended to the chosen base. Use a published
version that matches the local pipeline, since public docs can describe newer
code. API reference links always stay with the local backend at `/api/docs`.

## Edit a guide

1. Edit Markdown in `docs/next/`; preserve the validation history in
   `docs/validation/next.md`.
2. Add new pages to the `nav` section of `mkdocs.yml`.
3. Use relative `.md` links between documentation pages. Keep repository paths
   outside `docs/` in code spans or link to the corresponding source on GitHub.
4. Check analysis command examples against the current CLI and presets. A docs
   build validates page links; it does not execute analysis examples.
5. Build with `--strict` and inspect the local preview before submitting changes.
6. When changing a help destination or heading, check the dashboard's help topic
   links and run the documentation link tests. They build the guide under both
   local-style and GitHub Pages-style prefixes and check the actual anchors.
7. Keep method claims consistent with the stage implementation and resolved
   example preset, including inherited defaults. Link primary papers or official
   library documentation beside the choice they explain. Identify project-specific
   thresholds and rules explicitly; citing a general method does not validate
   a custom procedure. Prefer versioned API links when behavior is version-sensitive.

Dashboard fields and defaults come from the analysis `Config` dataclasses and
the selected preset. Inline parameter explanations come from those dataclasses'
comments/docstrings, so update them at the source. Contextual topics, guide
destinations, and optional external background references live in
`dashboard/src/lib/help-links.json`. Guide paths stay relative to the selected
documentation base; external references keep their full HTTPS URLs. Check both
surfaces when changing an analysis option, and retain the methods page's explicit
stage anchors so existing links keep working.

The README is a short entry point. Keep detailed workflow instructions in these
guides to avoid maintaining two copies. Dependency versions are recorded in
`uv.lock`; existing analysis-package versions should stay stable when updating
the documentation group.

For theme options, see the [Material documentation](https://squidfunk.github.io/mkdocs-material/).
For site settings and link validation, see the
[MkDocs configuration reference](https://www.mkdocs.org/user-guide/configuration/).
