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
--seed 42 \
--max-sessions-to-run 25

# 3. Inspect repeat-level accuracy and confidence distributions (optional).
uv run python scripts/inspect_decoding_results.py \
--cache-dir cache/run_034_full_session \
--session 221024 \
--trial 0 1 \
--time-bin-start -200 1400 \
--with-null \
--compare-with-repeat-idx 0

# 4. Identify and summarize on/off states.
uv run python scripts/on_off_states.py \
--cache-dir cache/run_034_full_session \
--cc-method-on one_tailed \
--cc-method-off one_tailed \
--compare-with-cc-skipped-on \
--compare-with-cc-skipped-off \
--use-decoding-estimates-from-subset-of-repeats \
--list-of-repeats 0

# 5. Compare top preferred-cell activity across on/off and opposite-cue states.
uv run python scripts/compare_activity_across_states.py \
--data-dir data/nature \
--cache-dir cache/run_034_full_session \
--activity-bin-width-ms 50 \
--seed 42 \
--max-points-per-color-group 50

# 6. Regress CC-applied off-state duration on baseline, delay, and encoding activity.
uv run python scripts/predict_off_state_duration_using_baseline_activity.py \
--data-dir data/nature \
--cache-dir cache/run_034_full_session \
--compare-with-delay \
--compare-with-encoding

# 7. Regress CC-applied off-state duration on session cell counts.
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
- The decoder uses stationary cells and logistic regression with `C=1`.
  `SIGMOID` requests nested five-fold calibration without using the held-out
  decoding trial. The 100 label shuffles estimate null confidence.
- `--n-repeats-for-model-fit 1` produces only repeat 0. The cue-preserved
  training-set shuffle intentionally leaves repeat 0 unchanged, so it has no
  effect unless the repeat count is increased.
- On/off-state detection uses one-tailed cluster correction, also generates
  uncorrected comparison summaries, and uses decoder repeat 0.
- The activity comparison uses 50 ms bins and displays at most 50
  deterministically sampled points per color group.
- `--compare-with-delay` and `--compare-with-encoding` add delay- and
  encoding-activity regressions to the baseline regression.

Primary caches are written directly under `cache/run_034_full_session/`,
including `cell_trial_selection.pkl`, `decoding_confidence.pkl`, and
`on_off_states.pkl`. Step 4 must be rerun if an older `on_off_states.pkl` lacks
trial-level maximum off-state duration. Fixed-effects results from steps 6 and
7 are grouped under `fixedlm/`.

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

# 3. Scan active-cell thresholds from the 10th to 90th percentiles.
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

# 4. Test interactions across periods for each cell group.
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
  features and must match across preparation and model fitting.
- `--cv-prediction-sample-per-model 1000` caps stored plotting samples without
  changing the CV metrics. `--significance-alpha 0.05` sets the comparison
  threshold.
- The criticality scan converts the listed percentiles to active-cell
  thresholds and refits threshold-dependent models.
- Both outcomes are analyzed unless `--outcome total` or `--outcome maximum`
  is supplied.

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
    │   ├── active_cell_criticality/
    │   └── period_interactions/
    └── maximum_off_state_duration/
        ├── model_family/
        ├── active_cell_criticality/
        └── period_interactions/
```

Use `uv run python scripts/<script>.py --help` for the complete option list.
