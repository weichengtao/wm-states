# WM States Analyses

Select neural populations, decode working-memory content, identify on/off
states, and fit mixed-effects models. The supported pipeline lives in
`scripts/next/`; the original scripts remain available for historical runs.
New and historical caches are intentionally incompatible. Use a fresh cache directory.

The refactored pipeline is organized as follows:

| Location | Contents |
| --- | --- |
| `scripts/next/` | Pipeline runner, analysis scripts, and supporting modules |
| `configs/next/example_pipeline.json` | Example analysis settings |
| `configs/next/smoke_pipeline.json` | Reduced-cost integration settings |
| `tests/next/` | Tests for the refactored pipeline |

Selection and analysis use whole sessions. Decoding fits each time bin separately,
with one observed estimate and N null-shuffle estimates per tested trial/bin.
Session partition selection, label-preserving shuffles before the train/test
split or within the training set, decoder repeats across random seeds, and
pooled-delay decoding are no longer supported by the new scripts. The standalone
fixed-effects duration regressions and their `baseline` and `cell-count` pipeline
stages have also been removed. Mixed-effects analyses retain baseline activity
and cell-count predictors. Remove the `baseline` and `cell-count` objects from
older JSON settings before using them with the new runner.

## Setup

Python 3.12 is required. Install dependencies with:

```bash
uv sync --python 3.12
```

Alternatively, create a Python 3.12 environment and run `pip install -e .`.
Run the commands below from the repository root; with an activated pip-managed
environment, replace `uv run python` with `python`.
Download the dataset from
[Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.kkwh70sct) and put the
`.mat` files in `data/nature`.

## Quick start

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

This runs the first five stages below. Add `--stages all` to include the
mixed-effects analyses. The session-list file selects whole sessions by ID;
omit it to consider all `.mat` files in the data directory. Use one ID per line,
without `.mat`; blank lines and `#` comments are allowed. Only IDs also present
in the data directory are processed. Missing files produce warnings; no matching
sessions is an error.

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

Use `--stages mixed` to run the last six stages after completing the first five.
You can also name individual stages, such as `--stages evaluate states`.
Named stages run in the supplied order; prerequisites are not added automatically.
Keep the same settings, data directory, and cache directory when continuing a run.

## Configure a run

Copy a preset within `configs/next/`, edit its stage settings, and pass the file
with `--settings`. A settings file is a JSON object keyed by the stage names above.
Each stage accepts its script's dataclass field names with underscores;
standalone script CLI flags use hyphens. Unknown stage names and unknown settings
in selected stages are errors. Omitted settings use the script's defaults.
A custom settings file replaces the preset; it is not merged with it.

For example, edit this field inside the preset's existing `decode` object:

```json
{
  "decode": {
    "n_decode_shuffle": 100
  }
}
```

`decode.n_decode_shuffle` controls the number of null estimates. The example
preset uses 100, the smoke preset uses 3, and the decoder default is 100.
The pipeline runner has no `--n-decode-shuffle` flag. When running
`decoding_confidence.py` directly, use `--n-decode-shuffle 100`.

Set `--data-dir` and `--cache-dir` on the pipeline command; they are not allowed
inside stage settings. Other shared options, such as worker count and session
filters, supply defaults to applicable stages; stage-specific JSON values take
precedence. Paths in JSON are relative to the working directory. The presets use
shared diagnostic figure settings at `configs/diagnostic_figure_config.json`.

Preview every stage's resolved settings without running analyses:

```bash
uv run python scripts/next/pipeline.py \
  --settings configs/next/example_pipeline.json --stages all --dry-run
```

`--n-jobs` controls selection session workers, decoding trial workers, and
mixed-effects CV model workers. Start with a value suited to the available CPUs
and memory. Numerical library threads inside workers are limited to one.
Standalone mixed-effects commands expose `--cv-n-jobs`. Decoding bins activity
once per session using chunked cumulative sums and sends binned activity to fit
workers. The decoder seed controls balancing and null permutations without
creating repeated observed estimates.

PNG is the default figure format. The runner's `--figure-formats png tif eps`
enables all three formats for analyses using the shared exporter; plots that
only support PNG keep that format.

### Resume and rerun

The runner writes resolved settings, stage status, and timings to
`<cache>/pipeline_manifest.json`. It runs every requested stage on each invocation;
only decoding automatically reuses matching per-session checkpoints. A checkpoint
is reused when its analysis settings, source data, selection cache, and code
fingerprint match. Changing the worker count does not invalidate it.

To force decoding to refit, set `"resume": false` in the JSON's `decode` object,
or pass `--no-resume` to the standalone decoder. After changing decoding, rerun
evaluation, states, and downstream analyses. After changing screening, rerun
decoding and its downstream stages. Use a fresh cache directory when comparing
analysis settings or migrating from the historical scripts.

## Full-session pipeline, step by step

These standalone commands reproduce the first five stages of the example
preset and use the same session list as the quick-start command. Run them in
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

```bash
uv run python scripts/next/eval_confidence_across_runs.py \
  --cache-dirs cache/next_run_034_full_session cache/next_run_035_full_session \
  --run-aliases "Run 034" "Run 035" \
  --line-colors "tab:blue" "tab:orange" \
  --null-shading percentiles
```

## Selection and decoding behavior

- Screening always uses a full session. Each selection result contains
  `num_trials` and one set of selected, stationary, and presence-passing cells.
  There are no session partitions or leave-one-out cell-selection variants.
- Selection uses correct trials for PEV and presence filtering. The example's
  negative firing-rate/variance thresholds and preferred-cue correlation cutoff
  of 2 disable their numerical exclusion thresholds; presence ratio, baseline
  correlation, and PEV remain constrained. Unavailable statistics still fail
  their applicability checks.
- Decoding uses correct preferred- and opposite-cue trials, testing each
  preferred-cue trial once. Training, normalization, C search, and calibration
  exclude all time bins of the held-out trial. Cell screening itself remains
  a full-session procedure; it is not nested within decoder cross-validation.
- There is exactly one observed estimate per trial/bin. With N null shuffles,
  `decoding_confidence` and `decoding_classifier_c` have shape `(trial, bin)`;
  `decoding_confidence_null` and `decoding_classifier_c_null` have shape
  `(trial, bin, N)`. Observed predictions also have shape `(trial, bin)`.
  N=0 produces an empty null axis and supports evaluation, but state detection
  requires at least two null estimates. No repeat axis or repeat-selection option exists.
- By default, observed and null fits use the same once-balanced training trials.
  Each null estimate independently permutes training-trial labels for each bin,
  **after** the outer train/test split. Every observed and null fit uses only
  the current time bin, with one sample per training trial. Pooled-delay decoding
  and cell-wise label-preserving shuffles are not supported.
- `--grid-search-for-c` selects among C=(1, 0.1, 0.01) using balanced accuracy
  and exactly five source-trial-grouped folds for every distinct observed/null
  training problem. Calibration uses the selected C and grouped training-only
  folds. Calibration may reduce its fold count when necessary; C search requires
  five source-trial groups containing each class. Fold scaling is reused across C
  candidates without using validation data. Without search, `--classifier-c` is used directly.
- Unknown and ineligible sessions produce warnings. `--max-sessions-to-run`
  on the runner caps selection as well as decoding. Selection caps the candidate
  file list; decoding caps eligible sessions after screening.
- Evaluation does not refit models. It reports Brier score, natural-log loss,
  accuracy, confidence, and valid counts for observed data and individual null
  shuffles. These scores concern preferred-cue test trials only.
- State detection retains the existing cluster-correction methods and total/
  maximum contiguous delay-duration outcomes. Bins with zero null variance are
  unclassified. Activity plots support sessions with no detected off-states.

## Outputs

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

## Mixed-effects pipeline

Preparation reads full-session selection and on/off-state caches. Both total
and maximum contiguous off-state duration are analyzed by default. These
trial-holdout repetitions are independent of the removed decoder fit repeats.

```bash
# 1. Prepare the no-CV table and the separate fold-safe CV feature cache.
uv run python scripts/next/prepare_data_for_mixedlm.py \
  --data-dir data/nature \
  --cache-dir cache/next_run_034_full_session \
  --cv-shuffles 50 \
  --cv-holdout-fraction 0.2 \
  --cv-seed 42 \
  --active-threshold 0 \
  --history-alpha 0.2

# 2. Compare the M0--M5 and RM2--RM5 model families.
uv run python scripts/next/compare_mixed_effect_models.py \
  --cache-dir cache/next_run_034_full_session \
  --cv-shuffles 50 \
  --cv-holdout-fraction 0.2 \
  --cv-seed 42 \
  --history-alpha 0.2 \
  --cv-prediction-sample-per-model 1000 \
  --significance-alpha 0.05

# 3. Compare the full cell-count model with M0 and three drop-one models.
uv run python scripts/next/nested_model_comparison_cell_counts.py \
  --cache-dir cache/next_run_034_full_session \
  --cv-shuffles 50 \
  --cv-holdout-fraction 0.2 \
  --cv-seed 42 \
  --cv-prediction-sample-per-model 1000 \
  --significance-alpha 0.05

# 4. Add preferred-cell mean normalized activity period by period.
uv run python scripts/next/nested_model_comparison_mean_norm_activity.py \
  --cache-dir cache/next_run_034_full_session \
  --cv-shuffles 50 \
  --cv-holdout-fraction 0.2 \
  --cv-seed 42 \
  --cv-prediction-sample-per-model 1000 \
  --significance-alpha 0.05

# 5. Scan active-cell thresholds from the 10th to 90th percentiles.
uv run python scripts/next/find_active_cell_criticality.py \
  --data-dir data/nature \
  --cache-dir cache/next_run_034_full_session \
  --cv-shuffles 50 \
  --cv-holdout-fraction 0.2 \
  --cv-seed 42 \
  --cv-prediction-sample-per-model 1000 \
  --active-percentiles 10 20 30 40 50 60 70 80 90 \
  --history-alpha 0.2 \
  --significance-alpha 0.05

# 6. Test interactions across periods for each cell group.
uv run python scripts/next/test_interactions_across_periods.py \
  --cache-dir cache/next_run_034_full_session \
  --cv-shuffles 50 \
  --cv-holdout-fraction 0.2 \
  --cv-seed 42 \
  --history-alpha 0.2 \
  --cv-prediction-sample-per-model 1000 \
  --significance-alpha 0.05
```

Preparation creates reproducible within-session trial holdouts, rounding the
20% holdout count up. CV activity normalization is fitted using training trials
only. `--history-alpha`, weighting policy, and CV settings must match preparation.
The 1000-row prediction sample cap affects plotting samples, not CV metrics.
`--outcome total` or `--outcome maximum` selects one outcome.

For PEV-weighted mixed-effects analyses, add `--pev-weighted-average` to preparation,
model-family comparison, nested activity comparison, criticality, and interactions.
The cell-count-only comparison uses the unweighted prepared table; prepare both
variants if running that analysis alongside weighted models.

```text
<cache>/mixedlm/
├── prepared/
│   ├── trial_table.pkl
│   ├── cv_feature_cache.pkl
│   ├── manifest.json
│   ├── pev_weighted/
│   └── active_thresholds/
└── outcomes/
    └── <outcome>/
        ├── model_family/
        ├── nested_cell_count_comparison/
        ├── nested_mean_norm_activity_comparison/
        ├── active_cell_criticality/
        └── period_interactions/
```

`<outcome>` is `total_off_state_duration` or `maximum_off_state_duration`;
each has the same analysis subdirectory layout.

Check the statistical analyses' logs and result tables for failed or nonconverged
fits; a completed command does not imply that every model converged.

## Development

All new executable code and supporting modules are under `scripts/next/`,
with JSON presets under `configs/next/` and tests under `tests/next/`.
The new scripts do not import the original implementations.

```bash
# Run the new pipeline's tests.
uv run python -m unittest discover -s tests/next -v

# Run all tests, including the historical scripts' tests.
uv run python -m unittest discover -s tests -v

# Inspect available pipeline and decoder options.
uv run python scripts/next/pipeline.py --help
uv run python scripts/next/decoding_confidence.py --help
```

Individual scripts remain directly executable; `python -m scripts.next.pipeline`
is also supported. Use a fresh cache directory when migrating historical runs.
See [refactor validation](scripts/next/VALIDATION.md) for the recorded checks and
integration-run scope.
