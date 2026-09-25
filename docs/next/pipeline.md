# Pipeline stages

Run stages from the repository root with one consistent data directory, cache
directory, and set of analysis settings.

## Stage order

| Stage | Analysis |
| --- | --- |
| `select` | Screen cells across each full session |
| `decode` | Estimate observed and shuffled-null confidence |
| `evaluate` | Score decoding confidence and predictions |
| `states` | Detect on/off states |
| `activity` | Compare activity across states and cue groups |
| `prepare` | Prepare mixed-effects tables and CV features |
| `models` | Compare mixed-effects model families |
| `nested-count` | Compare nested cell-count models |
| `nested-activity` | Compare nested activity models |
| `criticality` | Scan active-cell thresholds |
| `interactions` | Test interactions across periods |

With no `--stages` flag the runner executes the first five stages.
`--stages all` executes all eleven; `--stages mixed` executes the last six.
Explicit stage lists run in the supplied order, and the runner does not add
prerequisites automatically.

| Stage | Required inputs |
| --- | --- |
| `select` | Session `.mat` files |
| `decode` | Session files and selection cache |
| `evaluate` | Decoding cache |
| `states` | Decoding cache with at least two null estimates |
| `activity` | Session files, selection, decoding, and state caches |
| `prepare` | Session files, selection, decoding, and state caches |
| Mixed-effects analyses | Prepared table and CV feature cache; criticality also reads session files |

After changing a stage's inputs or settings, rerun its dependents. See
[Resume and rerun](configuration.md#resume-and-rerun) for checkpoint behavior.

## Default stages, one script at a time

These standalone commands reproduce the first five stages of the example
preset and use the same session list as the [getting-started command](getting-started.md). Run them in
order with the same data and cache directories.

```bash
# 1. Select decoder cells and cache the selection results.
uv run python scripts/next/cell_trial_selection.py \
  --n-jobs-session 10 \
  --data-dir data/nature \
  --cache-dir cache/next_run_034_full_session \
  --session-list-file configs/decoding_sessions.txt \
  --t-test-window 50 \
  --min-cell-per-group 1 \
  --min-fr-test -1 \
  --min-presence-ratio 0.9 \
  --var-ratio-threshold-delay-over-baseline -1 \
  --var-ratio-threshold-sliding-over-all -1 \
  --temp-dep-r-threshold 2 \
  --temp-dep-r-threshold-baseline 0.3 \
  --sig-pev-threshold 2.5 \
  --no-save-extended-diagnostics \
  --diagnostics-figure-config configs/diagnostic_figure_config.json

# 2. Fit the observed decoder and estimate shuffled null confidence.
uv run python scripts/next/decoding_confidence.py \
  --data-dir data/nature \
  --cache-dir cache/next_run_034_full_session \
  --session-list-file configs/decoding_sessions.txt \
  --t-decode-window 50 \
  --min-cell-per-group 1 \
  --n-decode-shuffle 100 \
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
