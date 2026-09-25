# WM States: next pipeline

Analyze working-memory recordings from complete sessions: screen neural
populations, decode memory content, identify on/off states, and compare
mixed-effects models.

The supported implementation is in `scripts/next/`. It produces **one observed
estimate and N null estimates per tested trial and time bin**. The default run
has five stages; the full pipeline has eleven.

## Start here

| Task | Guide |
| --- | --- |
| Install dependencies and run a smoke analysis | [Getting started](next/getting-started.md) |
| Choose sessions, null counts, and workers | [Configuration](next/configuration.md) |
| Run or resume analysis stages | [Pipeline stages](next/pipeline.md) |
| Understand screening and decoder estimates | [Selection and decoding](next/selection-decoding.md) |
| Compare trial-level statistical models | [Mixed-effects analyses](next/mixed-effects.md) |
| Find caches, figures, and scores | [Outputs and inspection](next/outputs.md) |
| Migrate historical settings | [Migration](next/migration.md) |
| Review tested behavior and limitations | [Validation record](validation/next.md) |

## Repository layout

| Location | Contents |
| --- | --- |
| `scripts/next/` | Analysis scripts and shared implementation |
| `configs/next/` | Example and smoke JSON presets |
| `tests/next/` | Tests for the new implementation |
| `docs/next/` | Pipeline guides |
| `docs/validation/next.md` | Dated validation results |
| `data/` | Local recordings, excluded from Git |
| `cache/` | Generated analysis outputs, excluded from Git |

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

To build or serve these docs locally, see [Development](development.md).
