# Outputs and inspection

`--cache-dir` always names the **run root**, for both the runner and standalone
scripts. Each stage owns a directory named after its pipeline stage ID. Only
`pipeline_manifest.json` lives at the root; the runner creates stage directories
as they are needed.

| Path relative to the run root | Contents |
| --- | --- |
| `pipeline_manifest.json` | Resolved settings and status of the latest runner invocation |
| `select/cell_screening.pkl` | Full-session screening results |
| `select/tables/cell_screening.csv` | Screening summary, enabled checks, and settings |
| `select/diagnostics/` | Optional per-cell diagnostic CSV and rejection summary; plots in `figures/cells/` and `figures/reasons/` |
| `decode/decoding_confidence.pkl` | Observed and null decoding estimates |
| `decode/checkpoints/` | Per-session decoding checkpoints |
| `decode/figures/` | Confidence and classifier-C plots; trial inspection in `inspection/` |
| `evaluate/eval_confidence.pkl` | Decoding evaluation results |
| `evaluate/tables/eval_confidence.csv` | Evaluation summary |
| `evaluate/figures/across_runs/<comparison>/` | Cross-run evaluation plots, saved in each compared run |
| `states/on_off_states.pkl` | State masks and duration summaries |
| `states/figures/` | Plots grouped into `confidence/`, `masks/`, `durations/`, and `cluster_masses/` |
| `activity/figures/` | Activity plots grouped by `activity/` or `principal_components/`, then `states/` or `cues/` |
| `prepare/` | Shared `trial_table.pkl`, `cv_feature_cache.pkl`, and preparation `manifest.json` |
| `models/outcomes/<outcome>/` | Model-family comparison |
| `nested-count/outcomes/<outcome>/` | Nested cell-count comparisons |
| `nested-activity/outcomes/<outcome>/` | Nested activity comparisons |
| `criticality/outcomes/<outcome>/` | Active-cell threshold comparisons |
| `criticality/prepared/active_thresholds/` | Threshold-specific trial tables and manifests in `percentile_<NN>/`, plus `thresholds.csv` |
| `interactions/outcomes/<outcome>/` | Period interaction comparisons |

Model outcome directories contain `tables/`, `figures/`, `logs/`, and, when
enabled, `cross_validation/`. `<outcome>` is `total_off_state_duration` or
`maximum_off_state_duration`. Optional plots and diagnostics are created only
when requested and when the corresponding data are available.

PEV-weighted variants add `pev_weighted/` to the relevant output directory:
`prepare/pev_weighted/`, `activity/figures/pev_weighted/`, and
`<model-stage>/outcomes/<outcome>/pev_weighted/`. Criticality's weighted trial
tables live in `criticality/prepared/active_thresholds/percentile_<NN>/pev_weighted/`;
its threshold summary lives in `criticality/prepared/active_thresholds/pev_weighted/`.
The example preset weights activity plots but leaves mixed-effects inputs
unweighted. See [custom subdirectory settings](configuration.md#cache-directory-layout).

The four primary `.pkl` caches (screening, decoding, evaluation, and states) use
a versioned envelope. Load their result lists with `scripts.next.cache_io.read(path)`.
Read pickle caches only from trusted sources. Earlier flat next caches and the
shared `mixedlm/` layout must be regenerated in a fresh run directory; the new
scripts do not fall back to old locations.

Activity comparison includes preferred/opposite cue views, per-cell
and population plots, PCA, deterministic point sampling, and maximum off-state
highlights. PEV weighting applies to selective-cell population means; stationary
nonselective cells retain equal weights.

## Load a primary cache

Run this from the repository root in the analysis environment:

```python
from pathlib import Path
from scripts.next.cache_io import read

results = read(Path("cache/next_run_034_full_session/decode/decoding_confidence.pkl"))
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
The diagnostic CSV's `presence_ratio` uses correct trials in the configured
screening window ([−400, 1400) ms in the example), matching the screening criterion.
Per-check columns distinguish `disabled`, `pass`, `fail`, and `not_applicable`.
Activity traces and
the additional baseline Spearman correlation still describe all session trials.
