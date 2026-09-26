# Mixed-effects analyses

The methods guide describes [feature preparation](methods.md#prepare),
[shared estimation and CV](methods.md#mixed-effects-estimation), and the
procedures for [model families](methods.md#models),
[nested counts](methods.md#nested-count), [nested activity](methods.md#nested-activity),
[threshold scanning](methods.md#criticality), and [interactions](methods.md#interactions).
It includes references for statsmodels estimation, variance-component R², and
validation, with the assumptions needed to interpret each result.

After completing the default pipeline, run all six mixed-effects stages with
the same preset and cache:

```bash
uv run python scripts/next/pipeline.py \
  --settings configs/next/example_pipeline.json \
  --data-dir data/nature \
  --cache-dir cache/next_run_034_full_session \
  --stages mixed --n-jobs 10
```

## Standalone commands

Preparation reads full-session selection and on/off-state caches and validates
their provenance against the decoding cache and current session files. Both total
and maximum contiguous off-state duration are analyzed by default. These
trial-holdout repetitions are independent of the removed decoder fit repeats.

```bash
# 1. Prepare the full-data table and the separate raw CV feature cache.
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

# 5. Scan standard-normal active-cell cutoffs from the 10th to 90th percentiles.
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
only. Keep `--history-alpha`, weighting policy, and CV settings aligned with
preparation to reproduce this example. Weighting mismatches are rejected; if the
requested split count, seed, or holdout fraction cannot reuse the cached splits,
the CV helper generates splits from the raw cache and records their source.
Changing a model stage's history alpha does not rewrite the full-data table;
regenerate that table when changing the intended history definition.
The 1000-row prediction sample cap affects plotting samples, not CV metrics.
`--outcome total` or `--outcome maximum` selects one outcome.

For PEV-weighted mixed-effects analyses, add `--pev-weighted-average` to preparation,
model-family comparison, nested activity comparison, criticality, and interactions.
The cell-count-only comparison uses the unweighted prepared table; prepare both
variants if running that analysis alongside weighted models.

```text
<cache>/
├── prepare/
│   ├── trial_table.pkl
│   ├── cv_feature_cache.pkl
│   ├── manifest.json
│   └── pev_weighted/                 # when requested
├── models/outcomes/<outcome>/
├── nested-count/outcomes/<outcome>/
├── nested-activity/outcomes/<outcome>/
├── criticality/
│   ├── prepared/active_thresholds/
│   │   ├── percentile_<NN>/          # trial_table.pkl + manifest.json
│   │   └── thresholds.csv
│   └── outcomes/<outcome>/
└── interactions/outcomes/<outcome>/
```

Each model stage owns its result directory, with `tables/`, `figures/`, `logs/`,
and optional `cross_validation/` below each outcome. `<outcome>` is
`total_off_state_duration` or `maximum_off_state_duration`. Weighted model results
add `pev_weighted/` below the outcome directory. Shared feature caches belong to
`prepare/`; threshold-specific criticality tables belong to `criticality/`.
See [Outputs](outputs.md) for the full layout and weighted threshold paths.

Always pass the run root to `--cache-dir`. Subdirectory overrides are relative
to the owning stage, as described in [Configuration](configuration.md#cache-directory-layout).

Check the statistical analyses' logs and result tables for failed or nonconverged
fits; a completed command does not imply that every model converged.

Optimization and inference have separate status fields. `fit_success=false`
records a failed model and its `fit_error`; other models can continue.
`inference_valid=false` means usable point estimates may remain, but coefficient
standard errors, confidence intervals, and p-values are withheld. Consult
`inference_error` before interpreting a missing value or `significant=false`;
withheld inference is not evidence of no effect. Nested comparisons similarly
report validity and error reasons instead of silently accepting invalid fits
or materially negative likelihood improvements. General comparison tables use
`likelihood_ratio_valid` / `likelihood_ratio_error`; the two nested stages'
`nested_contrasts` tables use `inference_valid` / `inference_error` for the
contrast. CV repeat metrics use `train_likelihood_ratio_valid` /
`train_likelihood_ratio_error`. See the [diagnostic field table](methods.md#model-fitting).
If every model fails for an outcome, the stage saves diagnostic tables and logs,
then raises an error. The same rule applies when every requested CV fit fails.

CV rankings require every requested holdout to succeed with finite RMSE.
Incomplete models remain in the raw/summary outputs, with failure counts,
`rank_eligible=false`, and a reason. They do not compete against models evaluated
on all holdouts. A warning explains each exclusion. These safeguards do not
change the Gaussian model family, the random trial split, or multiplicity policy;
see [the estimation methods](methods.md#mixed-effects-estimation).


## Activity periods and groups

Feature preparation uses the following intervals, in milliseconds relative to
cue onset:

| Period | Interval |
| --- | --- |
| Baseline | [−400, 0) |
| Encoding | [100, 300) |
| Pre-delay | [300, 500) |
| Delay | [500, 1400) |

With the example screening settings, features distinguish preferred selective
cells, selective nonpreferred cells, and stationary nonselective cells. Group
membership comes from the same validated screening metadata used by decoding
and activity plots. Preparation preserves cached cell order; plotting can rank
preferred cells by finite PEV for display.

When selectivity screening is disabled, membership in the selected pool does
not establish selectivity. Recorded population labels and screening switches
describe that distinction. Existing column names and model identifiers retain
`selective_nonpreferred` and `stationary_nonselective` so formulas remain stable.
PEV weighting changes selected-population means; it does not change active-cell
fractions or weights for the remaining cells passing the other checks.

The removed standalone fixed-effects regressions are not part of this workflow.
Baseline activity and cell-count predictors remain available in these
mixed-effects model families.
