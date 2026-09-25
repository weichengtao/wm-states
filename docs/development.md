# Development

## Source layout and tests

Analysis code belongs in `scripts/next/`, JSON presets in `configs/next/`, and
new tests in `tests/next/`. The new implementation does not import historical
analysis scripts. Documentation lives in `docs/`.

```bash
uv run python -m unittest discover -s tests/next -v
uv run python -m unittest discover -s tests -v
```

The second command includes both new and historical tests. Individual scripts
remain directly executable, and `python -m scripts.next.pipeline` is supported.
Record analysis validation in [the next pipeline log](validation/next.md), with
the date, exact commands, test counts, integration scope, and limitations.
Generated datasets and cache outputs stay outside version control.

## Serve the documentation

MkDocs Material is provided through the optional `docs` dependency group.
Run from the repository root:

```bash
uv sync --group docs --locked
uv run --group docs --locked mkdocs serve
```

Open `http://127.0.0.1:8000/`. The preview server reloads when Markdown or
`mkdocs.yml` changes. Stop it with Ctrl+C. To use a different local port:

```bash
uv run --group docs --locked mkdocs serve --dev-addr 127.0.0.1:8001
```

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
