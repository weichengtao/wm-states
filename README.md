# WM States Analyses

Standalone scripts for selecting neural populations, decoding working-memory
content, identifying on/off states, and fitting fixed- and mixed-effects models.

## Setup

Python 3.12 is required (`pyproject.toml` allows `>=3.12,<3.13`).

With conda:

```bash
conda create -n wm_states python=3.12
conda activate wm_states
pip install -e .
```

With uv:

```bash
uv sync --python 3.12
```

Download the dataset from
[Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.kkwh70sct) and place the
`.mat` files in `data/nature`.

## Full-session pipeline

Run the following steps in order with the same `--cache-dir`.

```bash
# 1. Select decoder cells and cache the selection results.
uv run python scripts/cell_trial_selection.py \
--n-jobs-session 10 \
--n-jobs-partition 1 \
--data-dir data/nature \
--cache-dir cache/run_034_full_session \
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

# 2. Fit repeated decoders and estimate shuffled null confidence.
uv run python scripts/decoding_confidence.py \
--data-dir data/nature \
--cache-dir cache/run_034_full_session \
--session-list-file configs/decoding_sessions.txt \
--t-decode-window 50 \
--min-cell-per-group 1 \
--n-repeats-for-model-fit 1 \
--cue-preserved-train-set-shuffle \
--n-decode-shuffle 100 \
--n-jobs 10 \
--cells-used-for-decoder STATIONARY \
--svm-kernel LINEAR \
--decoder-model LOGISTIC_REGRESSION \
--logistic-calibration-method SIGMOID \
--logistic-calibration-cv 5 \
--classifier-c 1 \
--grid-search-for-c \
--seed 42 \
--max-sessions-to-run 25

# 3. Evaluate observed and shuffled null confidence.
uv run python scripts/eval_confidence.py \
--cache-dir cache/run_034_full_session

# Compare evaluated runs (optional; run step 3 for each cache directory first).
uv run python scripts/eval_confidence_across_runs.py \
--cache-dirs cache/run_034_full_session cache/run_032_full_session cache/run_037_full_session cache/run_038_full_session \
--run-aliases "fixed c = 1; w/ calib." "fixed c = 1; w/o calib." "optimal c; w/ calib." "optimal c; w/o calib." \
--line-colors "tab:blue" "tab:orange"  "tab:green"  "tab:red" \
--null-shading percentiles

# 4. Identify and summarize on/off states.
uv run python scripts/on_off_states.py \
--cache-dir cache/run_034_full_session \
--cc-method-on one_tailed \
--cc-method-off one_tailed \
--compare-with-cc-skipped-on \
--compare-with-cc-skipped-off \
--use-decoding-estimates-from-subset-of-repeats \
--list-of-repeats 0 \
--cluster-size-threshold-off 1

# 5. Inspect repeat-level accuracy, confidence, and state assignments (optional).
uv run python scripts/inspect_decoding_results.py \
--cache-dir cache/run_034_full_session \
--session 221024 \
--trial 0 1 \
--time-bin-start -200 1400 \
--with-null \
--with-state \
--compare-with-repeat-idx 0

# 6. Compare top preferred-cell activity across states and cue groups.
uv run python scripts/compare_activity_across_states.py \
--data-dir data/nature \
--cache-dir cache/run_034_full_session \
--activity-bin-width-ms 50 \
--seed 42 \
--max-points-per-color-group 50 \
--show-principal-components \
--compare-with-max-off-state \
--pev-weighted-average

# 7. Regress CC-applied off-state duration on baseline, delay, and encoding activity.
uv run python scripts/predict_off_state_duration_using_baseline_activity.py \
--data-dir data/nature \
--cache-dir cache/run_034_full_session \
--compare-with-delay \
--compare-with-encoding

# 8. Regress CC-applied off-state duration on session cell counts.
uv run python scripts/predict_off_state_duration_using_cell_count.py \
--data-dir data/nature \
--cache-dir cache/run_034_full_session
```

### Essential settings

- Cell selection uses 50 ms PEV test windows and requires at least one selected
  cell in a cue group. The `-1` variance and firing-rate thresholds, together with
  `--temp-dep-r-threshold 2`, effectively disable those exclusion gates;
  presence ratio, baseline temporal dependence, and PEV remain constrained by
  the values shown.
- The decoder uses stationary cells and logistic regression. With
  `--grid-search-for-c`, every empirical repeat/bin and null shuffle/bin fit
  independently selects from `C=(1, 0.1, 0.01)` using balanced accuracy and
  exactly five source-trial-grouped folds. The search receives the same
  balanced or imbalanced prepared training set used by that fit; for null fits,
  it runs after the labels have been permuted. The selected `C` is then held
  fixed for the final fit and the subsequent `SIGMOID` or `ISOTONIC`
  calibration, without using the held-out decoding trial. `--classifier-c 1`
  is retained in the example as the fixed value used only when
  `--grid-search-for-c` is omitted. The 100 label shuffles estimate null
  confidence. Per-fit grid search can substantially increase decoding runtime,
  especially when many null shuffles are requested.
- Step 2 uses `configs/decoding_sessions.txt`, which lists all dataset sessions
  as an editable session-filter example. Keep one session ID per line and
  remove or comment out IDs to restrict decoding; blank lines and lines
  beginning with `#` are ignored. Unknown IDs produce a warning, as do listed
  sessions that do not pass the current decoding thresholds.
  `--max-sessions-to-run` is applied after this filter. A filtered run
  checkpoints only the selected sessions to `decoding_confidence.pkl`; use a
  separate pipeline cache directory with its own matching selection cache to
  preserve an existing all-session decoding cache.
- `--n-repeats-for-model-fit 1` produces only repeat 0. The cue-preserved
  training-set shuffle intentionally leaves repeat 0 unchanged, so it has no
  effect unless the repeat count is increased.
- Step 3 evaluates observed repeat 0 and every null shuffle without refitting.
  Scores cover preferred-cue test trials. Accuracy uses cached observed
  predictions (or a 0.5 probability threshold if unavailable); null accuracy
  uses the same threshold. Missing values emit warnings and are excluded,
  with valid counts stored per metric.
- Cross-run comparison accepts two or more `--cache-dirs` and plots common
  sessions. `--run-aliases` supplies legend names in the same order; otherwise
  folder names are used. Each figure has Brier score, log loss, accuracy, and
  decoding confidence rows, observed/null columns, and one legend.
- `--null-shading percentiles` (default) shows the 2.5th–97.5th percentile range
  across shuffle scores. Use `confidence_intervals` for a pointwise 95%
  t-interval of the null mean or `none` for no shading. Bands require at least
  two valid shuffles per bin; the mean line is unchanged.
- On/off-state detection uses one-tailed cluster correction, also generates
  uncorrected comparison summaries, and uses decoder repeat 0.
- The activity comparison generates separate figures for preferred-cue on/off
  states and for preferred- versus opposite-cue activity across all delay bins.
  It uses 50 ms activity bins. `--show-principal-components` adds parallel
  figures for the three highest-variance principal components of all
  finite-PEV preferred cells. Each session's PCA basis is fitted to the pooled,
  balanced preferred- and opposite-cue delay activity after per-bin/cell
  normalization; the same basis projects both cue groups and the maximum
  off-state points.
- `--max-points-per-color-group 50` deterministically samples at most 50 points
  from each ordinary blue, orange, green, or gray group. It never affects the
  red maximum-off-state points.
- Red maximum-off-state points are disabled by default. The example enables
  them with `--compare-with-max-off-state`. By default every bin in each
  session's maximum off-state is retained; use
  `--max-points-per-max-off-state N` to subsample each state independently.
- `--compare-with-delay` and `--compare-with-encoding` add delay- and
  encoding-activity regressions to the baseline regression.
- Add `--pev-weighted-average` to steps 6 and 7 to weight preferred and
  selective non-preferred cells by their cached `mean_pev_test` when computing
  population mean activity. Stationary non-selective cells remain equally
  weighted because their PEV estimates are noisy. The option does not affect
  active-cell counts, individual-cell plots, or PCA.

Primary caches are written directly under `cache/run_034_full_session/`,
including `cell_trial_selection.pkl`, `decoding_confidence.pkl`,
`eval_confidence.pkl`, and `on_off_states.pkl`. Step 4 must be rerun if an older
`on_off_states.pkl` lacks trial-level maximum off-state duration.
Fixed-effects results from steps 7 and 8 are grouped under `fixedlm/`.
PEV-weighted results from steps 6 and 7 are written to a `pev_weighted/`
subfolder inside the corresponding analysis
directory, leaving equal-weight results unchanged.

Each session in `decoding_confidence.pkl` records the exact regularization used
for each confidence estimate. `decoding_classifier_c_repeats` has shape
`(trial, repeat, bin)`, matching `decoding_confidence_repeats`, and
`decoding_classifier_c_null` has shape `(trial, bin, shuffle)`, matching
`decoding_confidence_null`. The decoding figure directory contains empirical
and, when null decoding is enabled, null heatmaps and line plots of
`log10(C)` alongside the confidence figures. The heatmaps average in log space
across repeats or null shuffles; line plots show those trial summaries and their
session mean. Plot-only mode regenerates these C figures from the cached
tensors; older caches without the C tensors still regenerate confidence figures
but require decoding to be rerun before C figures can be produced.

Step 3 writes `eval_confidence.csv` (session summaries) and
`eval_confidence.pkl` (observed/null scores by time bin and individual shuffle).
Rerun it to refresh older evaluation caches. Cross-run PNGs are saved in every
supplied run cache under
`eval_confidence_across_runs/<run_a>_vs_<run_b>[_vs_<run_c>...]/<session>_confidence_scores.png`.
Rerunning a comparison replaces its figures.

## Mixed-effects pipeline

This pipeline reads `cell_trial_selection.pkl` and `on_off_states.pkl`. It fits
both trial-level delay outcomes by default: total off-state duration and maximum
contiguous off-state duration.

```bash
# 1. Prepare the no-CV table and the separate fold-safe CV feature cache.
uv run python scripts/prepare_data_for_mixedlm.py \
--data-dir data/nature \
--cache-dir cache/run_034_full_session \
--cv-shuffles 50 \
--cv-holdout-fraction 0.2 \
--cv-seed 42 \
--active-threshold 0 \
--history-alpha 0.2

# 2. Compare the M0--M5 and RM2--RM5 model families.
uv run python scripts/compare_mixed_effect_models.py \
--cache-dir cache/run_034_full_session \
--cv-shuffles 50 \
--cv-holdout-fraction 0.2 \
--cv-seed 42 \
--history-alpha 0.2 \
--cv-prediction-sample-per-model 1000 \
--significance-alpha 0.05

# 3. Compare the full cell-count model with M0 and three drop-one models.
uv run python scripts/nested_model_comparison_cell_counts.py \
--cache-dir cache/run_034_full_session \
--cv-shuffles 50 \
--cv-holdout-fraction 0.2 \
--cv-seed 42 \
--cv-prediction-sample-per-model 1000 \
--significance-alpha 0.05

# 4. Add preferred-cell mean normalized activity period by period.
uv run python scripts/nested_model_comparison_mean_norm_activity.py \
--cache-dir cache/run_034_full_session \
--cv-shuffles 50 \
--cv-holdout-fraction 0.2 \
--cv-seed 42 \
--cv-prediction-sample-per-model 1000 \
--significance-alpha 0.05

# 5. Scan active-cell thresholds from the 10th to 90th percentiles.
uv run python scripts/find_active_cell_criticality.py \
--data-dir data/nature \
--cache-dir cache/run_034_full_session \
--cv-shuffles 50 \
--cv-holdout-fraction 0.2 \
--cv-seed 42 \
--cv-prediction-sample-per-model 1000 \
--active-percentiles 10 20 30 40 50 60 70 80 90 \
--history-alpha 0.2 \
--significance-alpha 0.05

# 6. Test interactions across periods for each cell group.
uv run python scripts/test_interactions_across_periods.py \
--cache-dir cache/run_034_full_session \
--cv-shuffles 50 \
--cv-holdout-fraction 0.2 \
--cv-seed 42 \
--history-alpha 0.2 \
--cv-prediction-sample-per-model 1000 \
--significance-alpha 0.05
```

### Essential settings

- Preparation creates 50 reproducible trial-level holdouts within each session,
  with the 20% holdout count rounded up. Activity z-scores used by CV are
  estimated from training trials only.
- `--active-threshold 0` defines an active cell as having activity above its
  training-trial mean. `--history-alpha 0.2` controls the exponential history
  features and must match across preparation and analyses that use those
  features.
- `--cv-prediction-sample-per-model 1000` caps stored plotting samples without
  changing the CV metrics. `--significance-alpha 0.05` sets the comparison
  threshold.
- The criticality scan converts the listed percentiles to active-cell
  thresholds and refits threshold-dependent models.
- Both outcomes are analyzed unless `--outcome total` or `--outcome maximum`
  is supplied.
- Activity means are equally weighted by default. For the PEV-weighted variant,
  append `--pev-weighted-average` to preparation and to steps 2, 4, 5, and 6.
  The flag must be used consistently so each analysis reads the matching
  prepared table and CV cache. Preferred and selective non-preferred cells are
  weighted by `mean_pev_test`; stationary non-selective cells remain equally
  weighted. Active-cell fractions, cell counts, and their model terms are
  unchanged, so the cell-count-only comparison in step 3 needs no flag.

Equal-weight prepared data stays directly under `mixedlm/prepared/`.
PEV-weighted preparation writes to `mixedlm/prepared/pev_weighted/`, and each
weighted analysis writes to a `pev_weighted/` subfolder inside its usual output
directory. Manifests and logs record the selected weighting policy.

Prepared data and results use the following layout:

```text
cache/run_034_full_session/mixedlm/
├── prepared/
│   ├── trial_table.pkl
│   ├── cv_feature_cache.pkl
│   ├── manifest.json
│   └── active_thresholds/
└── outcomes/
    ├── total_off_state_duration/
    │   ├── model_family/
    │   ├── nested_cell_count_comparison/
    │   ├── nested_mean_norm_activity_comparison/
    │   ├── active_cell_criticality/
    │   └── period_interactions/
    └── maximum_off_state_duration/
        ├── model_family/
        ├── nested_cell_count_comparison/
        ├── nested_mean_norm_activity_comparison/
        ├── active_cell_criticality/
        └── period_interactions/
```

Use `uv run python scripts/<script>.py --help` for the complete option list.
