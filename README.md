# WM States Analyses

This repo contains standalone analysis scripts under `scripts/`. These scripts
are not meant to be installed as a package; the `pyproject.toml` exists to
capture dependencies for reproducible setup.

## Setup with conda

```bash
conda create -n wm_states python=3.10.12
conda activate wm_states
pip install -e .
```

## Setup with uv

```bash
uv sync --python 3.10.12
```

## Data

Download the dataset from `https://datadryad.org/dataset/doi:10.5061/dryad.kkwh70sct` and place the `.mat` files in
`data/nature` before running the scripts.

## Run scripts

Run the scripts with `uv run` so they use the environment defined by
`pyproject.toml`. For a complete analysis, run cell selection, decoding,
optional repeat inspection, on/off-state analysis, baseline-activity
regression, and cell-count regression in that order. Use the same
`--cache-dir` for all stages.

## New pipeline

The commands below run the full-session pipeline. Cell selection writes the
selection cache, decoding reuses it and writes repeat-level confidence and
accuracy results, and the final analysis uses the decoding cache to identify
on/off states.

In the decoding command, `--cue-preserved-train-set-shuffle` leaves repeat 0
unchanged. For every later model-fit repeat, after the balanced or imbalanced
training set is selected, it independently permutes each cell's trial indices
within cue labels. This shuffle is not applied to the null estimates.

The inspection script is optional and saves one PNG per
selected trial and time bin; `--with-null` overlays null confidence and
accuracy, two range endpoints are inclusive, and time-bin endpoints are
matched to the nearest cached bin.

```bash
# 1. Select decoder cells and cache the selection results.
uv run python scripts/cell_trial_selection.py \
--n-jobs-session 10 \
--n-jobs-partition 1 \
--data-dir data/nature \
--cache-dir cache/run_029_full_session \
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
--cache-dir cache/run_029_full_session \
--t-decode-window 50 \
--min-cell-per-group 1 \
--n-repeats-for-model-fit 500 \
--cue-preserved-train-set-shuffle \
--n-decode-shuffle 500 \
--n-jobs 10 \
--cells-used-for-decoder STATIONARY \
--svm-kernel LINEAR \
--decoder-model LOGISTIC_REGRESSION \
--classifier-c 0.01 \
--logistic-calibration-method SIGMOID \
--logistic-calibration-cv 5 \
--seed 42 \
--max-sessions-to-run 25

`SIGMOID` enables nested CV-based calibration of logistic-regression
probabilities. The held-out decoding trial remains excluded from calibration,
and pooled delay-bin samples from the same training trial stay in one inner CV
fold. Use `NONE` to reproduce the uncalibrated logistic probabilities or
`ISOTONIC` when the calibration set is large enough to support a nonparametric
mapping. Calibration is also applied to label-shuffled null fits, so runtime is
approximately multiplied by the requested number of calibration folds plus
the final full-training-set fit.

The decoding cache is atomically updated after each completed session, so an
interrupted multi-session run retains every completed session without exposing
a partially written pickle.

# 3. Inspect repeat-level accuracy and confidence distributions (optional).
uv run python scripts/inspect_decoding_results.py \
--cache-dir cache/run_029_full_session \
--session 210921 \
--trial 0 1 \
--time-bin-start -200 1400 \
--with-null \
--compare-with-repeat-idx 0

# 4. Identify and summarize on/off states.
uv run python scripts/on_off_states.py \
--cache-dir cache/run_029_full_session \
--cc-method-on one_tailed \
--cc-method-off one_tailed \
--compare-with-cc-skipped-on \
--compare-with-cc-skipped-off \
--use-decoding-estimates-from-subset-of-repeats \
--list-of-repeats 0

# 5. Compare top preferred-cell activity across on/off and opposite-cue states.
uv run python scripts/compare_activity_across_states.py \
--data-dir data/nature \
--cache-dir cache/run_029_full_session \
--activity-bin-width-ms 50 \
--max-points-per-color-group 50 \
--seed 42

# 6. Regress CC-applied off-state duration on baseline, delay, and encoding activity.
uv run python scripts/predict_off_state_duration_using_baseline_activity.py \
--data-dir data/nature \
--cache-dir cache/run_029_full_session \
--compare-with-delay \
--compare-with-encoding

# 7. Regress CC-applied off-state duration on session cell counts.
uv run python scripts/predict_off_state_duration_using_cell_count.py \
--data-dir data/nature \
--cache-dir cache/run_029_full_session
```

The activity comparison uses up to three highest-delay-PEV cells whose own
preferred cue matches the session preferred cue. It balances correct preferred-
and opposite-cue trials, normalizes each cell separately at every delay bin, and
saves an adaptive scatter plot, every available pairwise projection, and one
full-distribution marginal histogram per selected cell plus population-mean
histograms for preferred, selective non-preferred, and stationary non-selective
cells, with matching ECDF plots directly below every histogram, under
`compare_activity_across_states/`. Existing `on_off_states.pkl` files created
before state masks were cached must be regenerated by rerunning step 4. Pass
`--hide-opposite-cue-points` to omit the gray comparison points from both plot
types without changing trial balancing or normalization. The green group shows
all delay-bin points from balanced preferred-cue trials and can be omitted with
`--hide-all-preferred-cue-points`. Pass
`--max-points-per-color-group 100` to show at most 100 deterministically sampled
points from each displayed color group; full data are still used for balancing
and normalization. The scatter cap does not subsample the marginal histograms,
which use fixed 0.25-wide bins by default (`--marginal-histogram-bin-width`).
Blue marks are layered above orange, and orange above gray, wherever groups
overlap. Histograms and ECDFs include a dashed black zero-reference line below
all colored distributions.
Sessions with two preferred cells use a 2D scatter, sessions with one use a 1D
colored strip, and sessions with none still produce labeled placeholder figures.

## Mixed-effects analyses

The mixed-effects analyses use `cell_trial_selection.pkl` from step 1 and
`on_off_states.pkl` from step 4 above. First, prepare one trial-level table with
the off-state duration, session and trial IDs, raw cell counts, period-specific
activity features, and their history EMAs. The default active-cell threshold is
z = 0 (activity above the cell's across-trial mean), and the history EMA uses
alpha = 0.2.

```bash
# 1. Prepare the no-CV table and the separate fold-safe CV feature cache.
uv run python scripts/prepare_data_for_mixedlm.py \
--data-dir data/nature \
--cache-dir cache/run_029_full_session \
--output-subdir prepare_data_for_mixedlm \
--output-filename mixedlm_data.pkl \
--cv-output-subdir prepare_data_for_mixedlm_cv \
--cv-output-filename mixedlm_cv_data.pkl \
--cv-shuffles 50 \
--cv-holdout-fraction 0.2 \
--cv-seed 42 \
--active-threshold 0 \
--history-alpha 0.2
```

This writes the original no-CV table to
`cache/run_029_full_session/prepare_data_for_mixedlm/mixedlm_data.pkl` and raw
cell-by-trial firing rates plus seeded within-session holdouts to
`cache/run_029_full_session/prepare_data_for_mixedlm_cv/mixedlm_cv_data.pkl`.
The raw cache lets every fold estimate cell-wise means and standard deviations
from training trials only. Use these caches to run the standard forward and
reverse-order model comparison:

```bash
# 2. Compare the M0--M5 and RM2--RM5 model families.
uv run python scripts/compare_mixed_effect_models.py \
--cache-dir cache/run_029_full_session \
--input-subdir prepare_data_for_mixedlm \
--input-filename mixedlm_data.pkl \
--output-subdir compare_mixed_effect_models \
--cv-input-subdir prepare_data_for_mixedlm_cv \
--cv-input-filename mixedlm_cv_data.pkl \
--cv-shuffles 50 \
--cv-holdout-fraction 0.2 \
--cv-seed 42 \
--history-alpha 0.2 \
--cv-prediction-sample-per-model 1000 \
--significance-alpha 0.05 \
--max-iterations 1000 \
--figure-dpi 200
```

The model tables, detailed Statsmodels log, and marginal-effect figures are
saved under
`cache/run_029_full_session/compare_mixed_effect_models/`. Cross-validated
metrics, logs, prediction samples, and held-out plots are kept separately under
its `cross_validation/` subdirectory. Held-out fixed predictions use fixed
effects only; held-out conditional predictions additionally use the session
random intercept estimated from that session's training trials.

The active-cell criticality scan converts the 10th through 90th percentiles to
z thresholds, prepares threshold-specific data, and refits every model that
uses current or historical active-cell fraction. It reads the Run 029 cell
selection and on/off-state caches directly, so it does not overwrite the
shared table prepared above.

```bash
# 3. Scan active-cell thresholds from the 10th to 90th percentiles.
uv run python scripts/find_active_cell_criticality.py \
--data-dir data/nature \
--cache-dir cache/run_029_full_session \
--output-subdir find_active_cell_criticality \
--cv-input-subdir prepare_data_for_mixedlm_cv \
--cv-input-filename mixedlm_cv_data.pkl \
--cv-shuffles 50 \
--cv-holdout-fraction 0.2 \
--cv-seed 42 \
--cv-prediction-sample-per-model 1000 \
--active-percentiles 10 20 30 40 50 60 70 80 90 \
--history-alpha 0.2 \
--significance-alpha 0.05 \
--max-iterations 1000 \
--figure-dpi 200
```

Threshold-specific prepared data, comparison tables, fit logs, and criticality
figures are saved under
`cache/run_029_full_session/find_active_cell_criticality/`; its independently
saved CV results are under `cross_validation/`.

Finally, use the shared trial-level table to fit the IM1--IM9 model branches for
preferred, selective non-preferred, and stationary non-selective cells. These
models test two-way activity interactions across periods and interactions
between cell count and period activity.

```bash
# 4. Test interactions across periods for each cell group.
uv run python scripts/test_interactions_across_periods.py \
--cache-dir cache/run_029_full_session \
--input-subdir prepare_data_for_mixedlm \
--input-filename mixedlm_data.pkl \
--output-subdir test_interactions_across_periods \
--cv-input-subdir prepare_data_for_mixedlm_cv \
--cv-input-filename mixedlm_cv_data.pkl \
--cv-shuffles 50 \
--cv-holdout-fraction 0.2 \
--cv-seed 42 \
--history-alpha 0.2 \
--cv-prediction-sample-per-model 1000 \
--significance-alpha 0.05 \
--max-iterations 1000 \
--figure-dpi 200
```

The interaction-model comparison tables, coefficients, fit log, and figures
are saved under
`cache/run_029_full_session/test_interactions_across_periods/`; its CV outputs
are under `cross_validation/`.

For every analysis, `cv_repeat_metrics.csv` contains one row per model and
shuffle, while `cv_model_summary.csv` reports the mean, standard deviation,
median, and 2.5th/97.5th percentiles of held-out RMSE, MAE, R², within-session
centered R², Pearson correlation, and child-minus-parent metric differences.
Negative RMSE/MAE differences and positive R² differences favor the child
model. A negative held-out R² means the model predicts worse than the held-out
grand mean; the within-session centered R² isolates trial-level prediction
after removing session means. Per-shuffle rows also retain training AIC/BIC,
fixed-effect coefficients, convergence warnings, and any fit errors.
Plotting samples use deterministic uniform reservoir sampling across all
held-out predictions from every shuffle, with 1,000 points per model by
default.
