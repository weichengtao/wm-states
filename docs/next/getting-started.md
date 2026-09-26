# Getting started

This page walks through running the pipeline from a terminal. To configure runs,
follow progress, and view results in your browser, start with the dashboard's
[first-time setup](dashboard.md#first-time-setup), or
[open it again](dashboard.md#open-the-dashboard-again) if it is already built.
Both interfaces use the same recordings, analysis stages, and cache layout;
[existing command-line runs](dashboard.md#find-existing-runs) can also appear in
the dashboard.

## Install the analysis environment

Use Python 3.12 and run all commands from the repository root:

```bash
uv sync --python 3.12 --locked
```

Alternatively, activate a Python 3.12 environment and run `pip install -e .`.
In that environment, replace `uv run python` below with `python`.

## Prepare the recordings

Download the dataset from
[Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.kkwh70sct) and place
session files at `data/nature/<session>.mat`.

The scripts read these MATLAB variables:

| Variable | Meaning |
| --- | --- |
| `spks` | Spike counts with axes `(trial, time, cell)` |
| `tc` | Uniformly sampled timestamps in milliseconds relative to cue onset |
| `cueAngIdx` | One cue ID from 1 through 8 per trial |
| `isCorr` | One correctness indicator per trial |

The time axis must be finite and strictly increasing. Trial counts in `spks`,
`cueAngIdx`, and `isCorr` must agree. The pipeline checks these constraints when
loading a session.

Data and caches are excluded from Git. The four-session `data/example` fixture
used during validation must be supplied locally. To create it from the downloaded
recordings:

```bash
mkdir -p data/example
cp data/nature/{210921,211015,221020,221024}.mat data/example/
```

## Run the pipeline

Run commands from the repository root. To check the complete pipeline on the
four full sessions in `data/example`, use the smoke preset:

```bash
uv run python scripts/next/pipeline.py \
  --settings configs/next/smoke_pipeline.json \
  --data-dir data/example \
  --cache-dir cache/test_run_034_next \
  --stages all --n-jobs 2
```

This runs all 11 stages with a 400 ms decoding stride, three null shuffles,
one mixed-effects holdout, one activity threshold, and a reduced optimization
budget. It checks integration; use the example preset for scientific analyses.
Small smoke runs can produce rank-deficient or nonconverged statistical fits.

For the full dataset, use the example preset:

```bash
uv run python scripts/next/pipeline.py \
  --settings configs/next/example_pipeline.json \
  --data-dir data/nature \
  --cache-dir cache/next_run_034_full_session \
  --session-list-file configs/decoding_sessions.txt \
  --n-jobs 10
```

This runs selection, decoding, evaluation, state detection, and activity comparison. Add `--stages all` to include the
mixed-effects analyses. The session-list file selects whole sessions by ID;
omit it to consider all `.mat` files in the data directory. Use one ID per line,
without `.mat`; blank lines and `#` comments are allowed. Only IDs also present
in the data directory are processed. Missing files produce warnings; no matching
sessions is an error.

See [Pipeline stages](pipeline.md) for stage order and prerequisites, and
[Configuration](configuration.md) to customize the presets. A successful smoke
run checks integration; it does not replace analysis of the full dataset.
