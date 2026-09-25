# Outputs and inspection

All run outputs live under the chosen cache directory:

| Path | Contents |
| --- | --- |
| `pipeline_manifest.json` | Resolved settings and status of the latest runner invocation |
| `cell_trial_selection.pkl` | Full-session screening results |
| `decoding_confidence.pkl` | Observed and null decoding estimates |
| `eval_confidence.pkl`, `eval_confidence.csv` | Decoding evaluation results |
| `on_off_states.pkl` | State masks and duration summaries |
| `checkpoints/decoding/` | Per-session decoding checkpoints |
| `mixedlm/` | Prepared data and mixed-effects results |

The four primary `.pkl` caches in the table use a versioned format. Load their
result lists with `scripts.next.cache_io.read(path)`. Read pickle caches only
from trusted sources.

Activity comparison includes preferred/opposite cue views, per-cell
and population plots, PCA, deterministic point sampling, and maximum off-state
highlights. PEV weighting applies to selective-cell population means; stationary
nonselective cells retain equal weights.

## Load a primary cache

Run this from the repository root in the analysis environment:

```python
from pathlib import Path
from scripts.next.cache_io import read

results = read(Path("cache/next_run_034_full_session/decoding_confidence.pkl"))
for session in results:
    print(session["session"], session["decoding_confidence"].shape)
```

The loader checks the primary cache schema. Historical caches must be regenerated
with the new scripts. Do not use a raw `pickle.load` result as if it were the
session list: the four primary cache files store a versioned envelope.

## Inspect and compare results

After state detection, inspect a session present in the decoding cache. `--trial`
uses zero-based cached trial rows; figures also show the original trial IDs.

```bash
uv run python scripts/next/inspect_decoding_results.py \
  --cache-dir cache/next_run_034_full_session \
  --session 221024 --trial 0 1 \
  --time-bin-start -200 1400 --with-null --with-state
```

To compare two completed runs, evaluate each run first with
`eval_confidence.py`. Replace the second cache path and aliases below with your
runs. Comparison aligns common sessions and supports null shading with
`percentiles`, `confidence_intervals`, or `none`.

For each shared session, comparison warns when another run's preferred cue or
trial-ID set differs from the first run, or that metadata is missing. Figures
are still generated, but the scores may describe different trial populations.
Reordering the same trials does not produce a warning.

```bash
uv run python scripts/next/eval_confidence_across_runs.py \
  --cache-dirs cache/next_run_034_full_session cache/next_run_035_full_session \
  --run-aliases "Run 034" "Run 035" \
  --line-colors "tab:blue" "tab:orange" \
  --null-shading percentiles
```

## Diagnostic tools

These optional scripts are separate from the eleven-stage runner:

| Script in `scripts/next/` | Purpose |
| --- | --- |
| `inspect_decoding_results.py` | Plot observed/null confidence and state assignments |
| `eval_confidence_across_runs.py` | Compare evaluated runs on common sessions |
| `reject_reason_histograms.py` | Summarize saved screening rejection diagnostics |

Selection diagnostics are opt-in through `select.save_extended_diagnostics` in
JSON or `--save-extended-diagnostics` on the selection script. Use each script's
`--help` for its required inputs and plotting options.
The diagnostic CSV's `presence_ratio` uses correct trials in the screening
window [−400, 1400) ms, matching the screening criterion. Activity traces and
the additional baseline Spearman correlation still describe all session trials.
