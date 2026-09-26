# Development

## Source layout and tests

Analysis code belongs in `scripts/next/`, JSON presets in `configs/next/`, and
Python tests in `tests/next/`. Dashboard frontend code and TypeScript tests live
in `dashboard/src/`; the FastAPI backend lives in `scripts/next/dashboard/`.
The new implementation does not import historical
analysis scripts. Documentation lives in `docs/`.

```bash
uv run --group dashboard --locked python -m unittest discover -s tests/next -v
uv run --group dashboard --locked python -m unittest discover -s tests -v
```

The optional dashboard group supplies FastAPI and the HTTP test client.
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
The backend serves the built interface on port **8000**; a separate Vite server
is only needed while editing the frontend. The
[frontend development workflow](next/dashboard.md#frontend-development) uses
Vite on port **5173**, with API and WebSocket requests proxied to port 8000.
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

MkDocs Material is provided through the optional `docs` dependency group.
Run from the repository root:

```bash
uv sync --group docs --locked
uv run --group docs --locked mkdocs serve --dev-addr 127.0.0.1:8001
```

Open [http://127.0.0.1:8001/](http://127.0.0.1:8001/). Port 8001 keeps the docs
preview separate from the dashboard on port 8000, allowing both to run at once.
The preview reloads when Markdown or `mkdocs.yml` changes. Keep its terminal
running and stop it with Ctrl+C.
If the dashboard shares this Python environment, also pass `--group dashboard`
to these `uv` commands to retain both optional dependency groups.

The analysis dependencies do not require the documentation group. A dedicated
docs-only environment can instead use `uv run --only-group docs --locked` before
the same MkDocs commands.

## Build static HTML

```bash
uv run --group docs --locked mkdocs build --strict
```

The generated site is written to `site/`, which is ignored by Git. Strict mode
fails on build warnings, including missing documentation links and anchors.
All pages are explicit entries in `mkdocs.yml`, and the built-in search plugin
indexes their content. The theme uses system fonts and bundled assets.

Serve the generated `site/` directory with a static HTTP server. The MkDocs
preview server is for local development. To publish under a particular domain
or subpath, configure `site_url` in `mkdocs.yml` for that destination and rebuild.
No deployment service is required to build the site.

## Edit a guide

1. Edit Markdown in `docs/next/`; preserve the validation history in
   `docs/validation/next.md`.
2. Add new pages to the `nav` section of `mkdocs.yml`.
3. Use relative `.md` links between documentation pages. Keep repository paths
   outside `docs/` in code spans or link to the corresponding source on GitHub.
4. Check analysis command examples against the current CLI and presets. A docs
   build validates page links; it does not execute analysis examples.
5. Build with `--strict` and inspect the local preview before submitting changes.

The README is a short entry point. Keep detailed workflow instructions in these
guides to avoid maintaining two copies. Dependency versions are recorded in
`uv.lock`; existing analysis-package versions should stay stable when updating
the documentation group.

For theme options, see the [Material documentation](https://squidfunk.github.io/mkdocs-material/).
For site settings and link validation, see the
[MkDocs configuration reference](https://www.mkdocs.org/user-guide/configuration/).
