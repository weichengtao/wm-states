# WM States Analyses

Select neural populations, decode working-memory content, identify on/off
states, and fit mixed-effects models from full recording sessions.

The supported pipeline lives in `scripts/next/`. It produces one observed
estimate and N null estimates per tested trial and time bin. JSON presets are
in `configs/next/`, tests in `tests/next/`, and documentation in `docs/`.
Historical scripts remain available in their original locations; use a fresh
cache directory for the new pipeline.

## Run an analysis

Use Python 3.12 and run commands from the repository root:

```bash
uv sync --python 3.12 --locked
```

Download the recordings from
[Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.kkwh70sct) and put the
session `.mat` files in `data/nature/`. Then run the example preset:

```bash
uv run python scripts/next/pipeline.py \
  --settings configs/next/example_pipeline.json \
  --data-dir data/nature \
  --cache-dir cache/next_run_034_full_session \
  --session-list-file configs/decoding_sessions.txt \
  --n-jobs 10
```

This runs five stages: selection, decoding, evaluation, state detection, and
activity comparison. Add `--stages all` for all eleven stages, including
mixed-effects analyses. Choose a worker count appropriate for your machine.
The session list filters files present in the data directory; omit it to
consider all available sessions.

Set the null count in `decode.n_decode_shuffle` in the JSON preset. The example
uses 100; the smoke preset uses 3. Use `--dry-run` to inspect resolved settings
before running. The runner has no `--n-decode-shuffle` flag.

For a smaller integration run, follow the
[smoke-run setup](docs/next/getting-started.md). Data and generated outputs are
excluded from Git.

## Reuse and compare results

Keep selection, decoding, and state caches together. Activity comparison and
mixed-effects preparation verify that the state results match decoding and that
decoding matches the current selection cache, session data, and implementation
code. Stale inputs stop the analysis with rerun instructions. After updating
the code, rerun decoding and downstream stages; rerun selection first if data
or screening settings changed. See [checkpoint reuse](docs/next/configuration.md#resume-and-rerun).

Screening diagnostics report presence ratios over correct trials, matching the
selection criterion. Cross-run confidence comparisons warn when preferred cues
or trial sets differ and still generate comparison plots. See
[outputs and inspection](docs/next/outputs.md) for commands and interpretation.

Population ISI analysis is excluded from `next`. The
[migration guide](docs/next/migration.md) lists all removed analyses and options.

## Documentation

The [documentation home](docs/index.md) links to the full guides:

- [Getting started](docs/next/getting-started.md)
- [Configuration and checkpoint reuse](docs/next/configuration.md)
- [Pipeline stages and standalone commands](docs/next/pipeline.md)
- [Selection and decoding behavior](docs/next/selection-decoding.md)
- [Mixed-effects analyses](docs/next/mixed-effects.md)
- [Outputs and inspection](docs/next/outputs.md)
- [Migration](docs/next/migration.md) and [troubleshooting](docs/next/troubleshooting.md)
- [Validation record](docs/validation/next.md)

Serve these pages with MkDocs Material:

```bash
uv sync --group docs --locked
uv run --group docs --locked mkdocs serve
```

Open `http://127.0.0.1:8000/`. To build static HTML in `site/`:

```bash
uv run --group docs --locked mkdocs build --strict
```

The optional `docs` dependency group and its versions are recorded in
`pyproject.toml` and `uv.lock`. See [Development](docs/development.md) for
preview options and documentation maintenance.

## Tests

```bash
# New pipeline tests.
uv run python -m unittest discover -s tests/next -v

# All tests, including the historical scripts.
uv run python -m unittest discover -s tests -v
```

The [validation log](docs/validation/next.md) records the tested analysis scope,
results, and limitations.
