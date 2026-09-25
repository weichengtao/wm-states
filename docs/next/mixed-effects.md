# Mixed-effects analyses

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


## Activity periods and groups

Feature preparation uses the following intervals, in milliseconds relative to
cue onset:

| Period | Interval |
| --- | --- |
| Baseline | −400 to 0 |
| Encoding | 100 to 300 |
| Pre-delay | 300 to 500 |
| Delay | 500 to 1400 |

Features distinguish preferred selective cells, selective nonpreferred cells,
and stationary nonselective cells. PEV weighting changes selective-population
means; it does not change active-cell fractions or stationary-nonselective weights.

The removed standalone fixed-effects regressions are not part of this workflow.
Baseline activity and cell-count predictors remain available in these
mixed-effects model families.
