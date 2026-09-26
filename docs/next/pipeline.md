# Pipeline stages

Run stages from the repository root with one consistent data directory, cache
directory, and set of analysis settings.

## Stage order

| Stage | Analysis |
| --- | --- |
| [`select`](methods.md#select) | Screen cells across each full session |
| [`decode`](methods.md#decode) | Estimate observed and shuffled-null confidence |
| [`evaluate`](methods.md#evaluate) | Score decoding confidence and predictions |
| [`states`](methods.md#states) | Detect on/off states |
| [`activity`](methods.md#activity) | Compare activity across states and cue groups |
| [`prepare`](methods.md#prepare) | Prepare mixed-effects tables and CV features |
| [`models`](methods.md#models) | Compare mixed-effects model families |
| [`nested-count`](methods.md#nested-count) | Compare nested cell-count models |
| [`nested-activity`](methods.md#nested-activity) | Compare nested activity models |
| [`criticality`](methods.md#criticality) | Scan active-cell thresholds |
| [`interactions`](methods.md#interactions) | Test interactions across periods |

Each stage links to its method, including the implemented statistical procedure,
trial population, outputs, and interpretation. The methods guide also describes
[shared mixed-effects estimation and validation](methods.md#mixed-effects-estimation).

With no `--stages` flag the runner executes the first five stages.
`--stages all` executes all eleven; `--stages mixed` executes the last six.
Explicit stage lists run in the supplied order, and the runner does not add
prerequisites automatically. Each runner invocation keeps a separate
[history record](outputs.md#run-manifest-history), so a partial rerun preserves
the earlier full-run manifest.

| Stage | Required inputs |
| --- | --- |
| `select` | Session `.mat` files |
| `decode` | Session files and selection cache |
| `evaluate` | Decoding cache |
| `states` | Decoding cache with at least two null estimates |
| `activity` | Session files, selection, decoding, and state caches |
| `prepare` | Session files, selection, decoding, and state caches |
| `models`, `nested-count`, `nested-activity`, `interactions` | Prepared table and, when CV is enabled, raw CV feature cache |
| `criticality` | Session files, selection, decoding, and state caches; raw CV feature cache when CV is enabled |

After changing a stage's inputs or settings, rerun its dependents. See
[Resume and rerun](configuration.md#resume-and-rerun) for checkpoint behavior.

For a recorded partial rerun, select only the required stages, for example
`--stages evaluate` on the same pipeline command. Standalone commands below
produce the same stage outputs but do not create runner-history records.

## Default stages, one script at a time

These standalone commands reproduce the first five stages of the example
preset and use the same session list as the [getting-started command](getting-started.md). Run them in
order with the same data and cache directories.

```bash
# 1. Select decoder cells and cache the selection results.
uv run python scripts/next/cell_screening.py \
  --n-jobs-session 10 \
  --data-dir data/nature \
  --cache-dir cache/next_run_034_full_session \
  --session-list-file configs/decoding_sessions.txt \
  --selectivity-bin-width-ms 50 \
  --check-min-trials \
  --no-check-firing-rate \
  --check-presence-ratio \
  --min-presence-ratio 0.9 \
  --no-check-delay-variance \
  --no-check-baseline-variance \
  --check-baseline-drift \
  --max-abs-baseline-drift-r 0.3 \
  --check-selectivity \
  --selectivity-pev-threshold-pct 2.5 \
  --no-check-preferred-cue-drift \
  --no-save-extended-diagnostics \
  --diagnostics-figure-config configs/next/diagnostic_figures.json

# 2. Fit the observed decoder and estimate shuffled null confidence.
uv run python scripts/next/decoding_confidence.py \
  --data-dir data/nature \
  --cache-dir cache/next_run_034_full_session \
  --session-list-file configs/decoding_sessions.txt \
  --t-decode-window 50 \
  --min-cell-per-group 1 \
  --n-decode-shuffle 100 \
  --no-preserve-null-time-structure \
  --n-jobs 10 \
  --cells-used-for-decoder STATIONARY \
  --svm-kernel LINEAR \
  --decoder-model LOGISTIC_REGRESSION \
  --logistic-calibration-method SIGMOID \
  --logistic-calibration-cv 5 \
  --classifier-c 1 \
  --grid-search-for-c \
  --seed 42

# 3. Evaluate observed and shuffled null confidence.
uv run python scripts/next/eval_confidence.py \
  --cache-dir cache/next_run_034_full_session

# 4. Identify and summarize on/off states.
uv run python scripts/next/on_off_states.py \
  --cache-dir cache/next_run_034_full_session \
  --cc-method-on one_tailed \
  --cc-method-off one_tailed \
  --compare-with-cc-skipped-on \
  --compare-with-cc-skipped-off \
  --cluster-size-threshold-off 1

# 5. Compare top preferred-cell activity across states and cue groups.
uv run python scripts/next/compare_activity_across_states.py \
  --data-dir data/nature \
  --cache-dir cache/next_run_034_full_session \
  --activity-bin-width-ms 50 \
  --seed 42 \
  --max-points-per-color-group 50 \
  --show-principal-components \
  --compare-with-max-off-state \
  --pev-weighted-average
```


For the remaining six stages, follow [Mixed-effects analyses](mixed-effects.md).
[Outputs and inspection](outputs.md) covers optional result inspection and
cross-run comparison.
