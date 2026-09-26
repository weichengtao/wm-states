# WM States: next pipeline

Analyze working-memory recordings from complete sessions: screen neural
populations, decode memory content, identify on/off states, and compare
mixed-effects models.

The supported implementation is in `scripts/next/`. It produces **one observed
estimate and N null estimates per tested trial and time bin**. The default run
has five stages; the full pipeline has eleven.

## Start here

To use the pipeline in your browser, follow the dashboard's
[first-time setup](next/dashboard.md#first-time-setup). Already set up? Use
[Open the dashboard again](next/dashboard.md#open-the-dashboard-again).
The same viewer can [find existing command-line runs](next/dashboard.md#find-existing-runs).
Its **Help** panel offers guidance without leaving your work, and **Pipeline
guide** opens these pages from the same server at `/docs/`.

| Task | Guide |
| --- | --- |
| Install dependencies and run a smoke analysis | [Getting started](next/getting-started.md) |
| Configure runs, follow progress, and compare sessions visually | [Dashboard](next/dashboard.md) |
| Choose sessions, null counts, and workers | [Configuration](next/configuration.md) |
| Run or resume analysis stages | [Pipeline stages](next/pipeline.md) |
| Review the example preset’s methods, populations, and validation design | [Analysis methods](next/methods.md) |
| Understand screening and decoder estimates | [Selection and decoding](next/selection-decoding.md) |
| Compare trial-level statistical models | [Mixed-effects analyses](next/mixed-effects.md) |
| Find caches, figures, scores, and invocation history | [Outputs and inspection](next/outputs.md) |
| Migrate historical settings | [Migration](next/migration.md) |
| Review tested behavior and limitations | [Validation record](validation/next.md) |

## Repository layout

| Location | Contents |
| --- | --- |
| `scripts/next/` | Analysis scripts and shared implementation, including the dashboard backend |
| `dashboard/` | React and TypeScript dashboard frontend |
| `configs/next/` | Example and smoke JSON presets |
| `tests/next/` | Tests for the new implementation |
| `docs/next/` | Pipeline guides |
| `docs/validation/next.md` | Dated validation results |
| `data/` | Local recordings, excluded from Git |
| `cache/` | Generated analysis outputs, excluded from Git |

Each analysis stage writes to its own subdirectory under the run root. Pass
that root, without a stage suffix, to `--cache-dir` on every command. See the
[output layout](next/outputs.md).

Historical scripts remain outside `scripts/next/`. Their caches are incompatible
with the new pipeline; begin with a fresh cache directory. See the
[migration guide](next/migration.md) for removed features.

## Consistent inputs and comparisons

Activity comparison and mixed-effects preparation check that state results,
decoding, selection, and session data belong together. Code changes also
invalidate decoder fingerprints. Follow the
[rerun guidance](next/configuration.md#resume-and-rerun) when updating an existing
run. Screening presence diagnostics use correct trials, and
[cross-run comparisons](next/outputs.md#inspect-and-compare-results) warn about
different preferred cues or trial sets while continuing to produce figures.

To build or serve these docs locally, see [Development](development.md#serve-the-documentation).
The dashboard and built guide share port **8000**. Port **8001** is only for the
optional documentation development preview. The guide can also be published as
a [standalone static site](development.md#publish-the-guide-on-github-pages).
