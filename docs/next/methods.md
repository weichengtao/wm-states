# Analysis methods: example pipeline

This page describes the analysis selected by **`configs/next/example_pipeline.json`**.
Each stage starts with its example choices, including inherited script defaults
where the JSON omits a setting. Alternative options are identified separately;
they are not part of this example analysis. Implementation filenames below
refer to `scripts/next/`.

The example command runs the first five stages; add `--stages all` to include
the six preparation and mixed-effects stages described here. See
[Pipeline stages](pipeline.md) for commands. Inspect effective settings with
`--settings configs/next/example_pipeline.json --stages all --dry-run`, and
retain the corresponding `manifests/<run_id>.json` records with reported results.

## Example design at a glance

| Analysis choice | Resolved example setting |
| --- | --- |
| Screening | At least 320 total trials; correct-trial presence ≥0.9, absolute baseline Pearson r ≤0.3, and PEV >2.5% for at least 100 ms |
| Decoding population | Stationary cells; correct preferred- and opposite-cue training trials, preferred-cue test trials only |
| Decoder | Logistic regression; training-class balancing; five-fold C search; sigmoid calibration; seed 42 |
| Decoding time grid | 50 ms windows starting −200 through 1400 ms, every 10 ms: 161 bins |
| Estimates | One observed estimate and 100 training-label null estimates per tested trial/bin |
| Null time structure | `preserve_null_time_structure=false`: independently permuted training labels at each bin |
| State detection | One-tailed on and off cluster rules, alpha 0.05; off clusters may contain one bin |
| Duration outcomes | Total and longest contiguous off-state duration from bin starts 500 through 1400 ms inclusive |
| Activity plots | PEV-weighted selective-population means; PCA and longest-off-state views enabled |
| Mixed-effects features | Equal cell weights; active threshold z >0; preceding-trial EMA alpha 0.2 |
| Mixed-effects validation | 50 random 20% within-session trial holdouts; seed 42 |
| Threshold scan | Standard-normal quantiles at percentiles 10, 20, …, 90 |

These are the resolved settings, including inherited defaults. In particular,
**PEV weighting in activity plots does not enable weighting in mixed-effects
models**. The 50 model holdouts are independent of the 100 decoding null
shuffles. The smoke preset uses fewer bins, null estimates, holdouts, and
thresholds for integration testing; its outputs do not implement the full
example analysis.

## Populations, time conventions, and normalization

Times are milliseconds relative to cue onset. Neural activity has axes
trial × time sample × cell. A complete session is the unit of screening and
processing; this does not mean every procedure uses every trial or cell.
Screening uses all correct cue conditions. Decoding trains on correct preferred
and opposite cues and tests only the preferred cue. Activity comparisons use
both cue groups; mixed-effects tables use cached preferred-cue trials.

The example distinguishes the following cell populations. The preferred cue
is defined separately for each session, as described in [Decoding](#decode).

| Population | Definition in this example |
| --- | --- |
| Selected/selective | Cells passing every enabled screening check, including PEV |
| Stationary | Cells passing the enabled non-PEV checks, whether or not they pass PEV; the decoder uses this population |
| Preferred selective | Selected cells preferring the session's preferred cue |
| Selective nonpreferred | Selected cells preferring any other cue, not only the opposite cue |
| Stationary nonselective | Stationary cells outside the selected population |

The last three groups are disjoint and partition the stationary population
in this example. “Stationary” names the screening rule; it does not establish
stationarity under every possible test. Presence-passing cells are also cached,
but that alternative decoder population is not selected by the example.

These population definitions are shared by decoding, activity comparison,
weighting, and model preparation. Cached cell IDs and aligned cue/PEV arrays
are validated before use. Activity plots rank preferred cells by finite PEV,
preserving screening order for ties; model preparation retains screening order.
The distinction changes neither the population definitions nor their weights.
Outside the example, disabled selectivity screening means “selected” does not
imply “selective”. Activity labels and preparation metadata describe the checks
actually enabled. Existing table keys such as `selective_nonpreferred` and
`stationary_nonselective` remain stable identifiers; interpret them using the
recorded population definitions rather than treating their names as evidence
that a disabled check passed.

Window start times and window coverage are distinct. A 50 ms window beginning
at 1400 ms includes samples in [1400, 1450) ms. Decoding and PEV include their
last configured start; preparation uses the half-open periods listed below.
State duration counts bin starts using the 10 ms stride, not the 50 ms window
width. Consequently, 91 off bins from 500 through 1400 ms represent **910 ms**
under the implemented duration convention.

Normalization also depends on the stage:

| Stage | Data used to estimate each cell's mean and standard deviation |
| --- | --- |
| Decoder fitting | Outer-training trials at one bin; inner C-search/calibration fits use their own training folds |
| Activity plots | Combined balanced preferred/opposite trials, separately for each bin |
| Full-data mixed-effects table | Cached preferred-cue trials, separately for each session and period |
| Mixed-effects CV | Training model rows only, separately for each session and period; apply those moments to all trials |

Screening and cue selection remain full-session procedures. Decoder holdouts
and model holdouts therefore assess predictions conditional on that selected
population; they do not cross-validate the complete cell-selection procedure.

All paths below are relative to the run root supplied as `--cache-dir`. Each
stage owns its outputs; see [Outputs](outputs.md) for tables, figures, diagnostics,
weighted variants, and custom subdirectory rules.

## 1. Cell and session screening {#select}

**Stage:** `select` · **Implementation:** `cell_screening.py`

**Example choices.** Explicitly enable the total-trial minimum, presence,
baseline-drift, and selectivity checks. Explicitly disable firing-rate,
delay-variance, baseline-variance, and preferred-cue drift checks. Use 50 ms
PEV windows, presence at least 0.9, absolute baseline correlation at most 0.3,
and PEV above 2.5%. Inherited defaults place PEV bin starts from 500 through
1400 ms every 10 ms and require 100 ms of contiguous selectivity. Extended
diagnostics are disabled. The supplied figure-config path takes effect when
`save_extended_diagnostics` is enabled: it requests all available sessions,
with a cap of 12 cell plots per session, 8 × 5 inch figures, and 150 DPI.

Session files are intersected with the optional allowlist. The enabled session
gate requires at least 320 total trials. Independently of that gate, cue metadata
requires at least two correct-trial cue conditions and positive residual degrees
of freedom. A session must also retain at least one selected cell to be cached. Each
retained session has one selection record.

The example applies three cell checks:

1. **Presence.** The fraction of correct trials with at least one spike in
    [−400, 1400) ms must be at least 0.9.
2. **Baseline drift.** Calculate Pearson correlation between baseline activity
    in [−400, 0) ms and ordinal position in the correct-trial sequence. Reject
    cells with absolute correlation above 0.3. Constant baseline activity has an
    undefined correlation and fails this enabled check's applicability rule.
3. **Cue selectivity.** Bin correct-trial firing rates and estimate a one-way
    ANOVA omega-squared statistic for cue condition:

    ```text
    MS_error = SS_within / (n_trials − K)
    PEV = 100 × [SS_total − SS_within − (K − 1) × MS_error] / [SS_total + MS_error]
    ```

    Here `K` is the number of cue conditions, `SS_total` is the total sum of
    squares, and `SS_within` is the within-cue sum of squares. Zero-total-variance
    bins receive zero. Clip PEV to [0, 100]. Require a contiguous run strictly above 2.5% lasting
    at least 100 ms, measured as bin count × 10 ms stride (at least ten
    consecutive bins). The highest-mean-rate cue defines each bin's preference. Summarize PEV over qualifying bins and
    define the cell's cue by their circular-mean preference, rounded to a cue
    index. PEV is an effect-size criterion, not a per-bin significance test.

Opposing or symmetric bin preferences can cancel the circular resultant. A
numerically undefined circular mean remains unavailable rather than being
rounded into an arbitrary cue. Otherwise selected cells with unavailable cue
metadata stop screening with a diagnostic error; cells already rejected by
enabled checks produce a warning. Inspect their cue responses and configured
test window before changing the analysis.

Bin starts include `test_end_ms`; each half-open window can extend beyond that
last start. Enabled checks reject unavailable statistics. Disabled checks
perform no rejection or applicability exclusion, replacing the old sentinel
behavior. All thresholds must be finite and in their documented ranges.

**Other available checks, disabled in this example:** minimum correct-trial
mean firing rate in the test period; correct-trial delay/baseline variance ratio;
mean sliding-window/global baseline variance ratio; and absolute Pearson
correlation between test-period firing rate and original trial index among all
preferred-cue trials, including incorrect trials. Their independent switches,
windows, and thresholds are listed in [Screening checks](configuration.md#screening-checks).

The cache records selected cells, presence-passing cells, and a stationary
population that bypasses PEV rejection but respects the other enabled checks.
The example's selected cells are cue-selective. With selectivity disabled in
another analysis, selection need not imply cue selectivity: cue/PEV metadata
uses all finite test bins without a threshold-run test. If preferred-cue drift
is enabled, it also filters the stationary pool; nonselective cells use cue
metadata summarized across all finite test bins for that check.

Caches include the resolved selection settings and enabled-check map. Optional
diagnostics distinguish disabled checks from passes, failures, and unavailable
statistics. Check identifiers describe the method: `baseline_drift` and
`preferred_cue_drift` name the two distinct correlations, while `delay_variance`
and `baseline_variance` name the two variance-ratio checks. A failed enabled
check records `fail_<check identifier>`; an unavailable required measurement
adds `_not_applicable`. These names do not change the statistics, thresholds,
check order, or selected populations. See the
[diagnostic field reference](outputs.md#screening-diagnostic-fields).
Diagnostic presence uses the same configured correct-trial window
as screening; traces and the additional baseline Spearman statistic describe
all trials, including incorrect trials. Diagnostic figure targets and cell caps
limit plots only; they do not change any screening check, cell population, or
CSV row. The default cap takes the first 12 sorted cell indices, so these plots
are an inspection aid rather than a representative sample. Set explicit cells
or remove the cap when a different review scope is needed; see
[diagnostic configuration](configuration.md#screening-diagnostics).
Screening remains a full-session procedure outside decoder CV.

**Outputs:** `select/cell_screening.pkl`, `select/tables/cell_screening.csv`, and
optional `select/diagnostics/`.

## 2. Observed and null decoding {#decode}

**Stage:** `decode` · **Implementation:** `decoding_confidence.py`

**Example choices.** Decode stationary cells with logistic regression,
five-fold C search, sigmoid calibration requesting five folds, and seed 42.
Fit one observed estimate and 100 null estimates per tested trial/bin, with
`preserve_null_time_structure=false` as explicitly set in the preset.
Windows are 50 ms wide; inherited defaults set bin starts from −200 through
1400 ms every 10 ms (161 bins) and enable balanced training classes.

The session's preferred cue is the most frequent preference among selected
cells; ties resolve to the lowest cue index. Its opposite is four positions
away among the eight cues. Correct trials from these two conditions form the
binary classification dataset, with preferred cue labeled 1. Use the cached
stationary population, including its nonselective cells. The example's
`min_cell_per_group=1` and inherited 320-total-trial minimum determine decoding
eligibility alongside the selection results.

Spike counts are converted to Hz in half-open windows at the configured bin
starts. Window duration is the number of recorded samples × sample interval;
the last start is included. Rates are computed once per session using chunked
cumulative sums. Every preferred-cue trial is held out once, and a separate
classifier is fitted at each time bin. No time bin from the held-out trial
enters fitting, scaling, C selection, or calibration. In this example, training cue
classes are subsampled without replacement to equal sizes once per held-out
trial and reused across bins and null estimates.

The example fits standardized logistic regression with the `liblinear` solver.
Each observed or null training problem selects C from `{1, 0.1, 0.01}` by
mean balanced accuracy in five stratified source-trial-grouped folds. Equal
scores choose the first candidate in that order. Scaling is fitted within those folds.
Although the JSON sets `classifier_c=1`, enabling `grid_search_for_c` means the
selected value is used for each fit. Sigmoid calibration uses only outer-training
trials and can reduce the requested five folds when necessary; the C search
still requires five folds. Calibration uses out-of-fold training scores and
then refits the base classifier on all outer-training trials (`ensemble=False`).
C is selected before calibration and shared across its folds, rather than
reselected within each calibration fold. The held-out test trial enters neither
step. `svm_kernel=LINEAR` is present in the preset but has no effect because the
selected decoder is logistic regression.

Before launching session fit workers, validate the correct-trial class counts:
the example's C search needs at least six preferred-cue and five opposite-cue
trials, leaving five of each after each preferred-trial holdout and balancing.
Calibration warns when available training counts reduce its requested folds;
insufficient counts for the configured procedure are an error.

The observed fit uses the original training labels. In the example's default
null policy, each null fit independently permutes those labels after the outer
split and training-trial balancing, for each bin and shuffle. It uses the same
training activity and repeats C selection and calibration as configured.

The optional `decode.preserve_null_time_structure=true` policy instead uses one
training-label permutation per held-out trial and shuffle across all time bins.
It also reuses that permutation's inner C-search and calibration fold indices
across bins. Scaling, C selection, model fitting, and calibration remain
separate at every bin; time bins are not pooled. Permutations remain independent
across held-out trials, whose training sets differ. This preserves the label
assignment through time within each trial's null trajectory; it does not create
a joint session-wide permutation test or correct for full-session selection.

With the seed and other settings fixed, increasing N preserves the existing
shuffle prefix in either policy and does not create additional observed fits.
Changing only this boolean preserves the observed calculation. It does not
restore pre-split shuffles, cell-wise label-preserving shuffles, or seed repeats.
The [configuration reference](configuration.md#null-shuffle-time-structure)
compares the policies and provides JSON and CLI examples.

Outputs include preferred-cue probabilities, observed class predictions,
selected C values, original test-trial IDs, and provenance. Caches also record
`preserve_null_time_structure`, the resolved `config`, and `null_policy`.
Changing the policy invalidates resume checkpoints and requires regenerated
decoding and dependent outputs. Observed arrays
have shape `(trial, 161)` and null arrays `(trial, 161, 100)` with the example's
resolved time grid. Since only preferred-cue trials are tested, the resulting scores describe that class;
they are not estimates of balanced two-class test performance.

**Alternatives, not used here:** other cell populations, SVM decoding, fixed C,
isotonic or disabled logistic calibration, and different null counts/time grids.

**Outputs:** `decode/decoding_confidence.pkl`, per-session checkpoints in
`decode/checkpoints/`, and plots in `decode/figures/`.

## 3. Confidence evaluation {#evaluate}

**Stage:** `evaluate` · **Implementation:** `eval_confidence.py`

**Example choices.** The JSON has no `evaluate` overrides. Use the evaluator's
defaults to score the single observed estimate and all 100 cached null estimates;
there is no additional fit or repetition setting.

This stage scores cached predictions without refitting. For preferred-cue
probability `p` and target `y`, it calculates Brier score `(p − y)²`, natural-log
loss `−y log(p) − (1 − y) log(1 − p)`, accuracy, and mean decoding confidence.
Probabilities are clipped to float64 epsilon bounds only for log loss.
Observed and null accuracy both threshold probabilities at `p >= 0.5`.
Because every tested label is 1, accuracy is the fraction of preferred-cue
trials whose preferred-cue probability reaches 0.5. Native classifier
predictions remain in the decoding cache; if they disagree with this rule,
evaluation warns and uses the probability threshold for both estimates.
Such disagreement can occur with SVM probability estimates. The cached native
predictions and native decoding accuracy are not overwritten.

Metrics are aggregated overall, by time bin, by estimate, and by time bin and
estimate. Missing probabilities are excluded with warnings and valid-entry
counts; aggregations with no valid entries are NaN. Each null shuffle remains
a separate estimate, while observed data have a single estimate. Results are
retained separately within each session; the stage does not compute a pooled
across-session performance estimate. Observed-only evaluation with N=0 is
supported as an alternative, but is not used by the example.

The separate cross-run plotting tool aligns common session IDs. It warns when
preferred cues or trial sets differ, or comparison metadata is missing, and
continues plotting. It also warns about different null time-structure policies
or observed-accuracy decision rules; rerun evaluation to align old accuracy
results with the current rule. Its percentile bands summarize null estimates, not
uncertainty across independently recorded sessions.

**Outputs:** `evaluate/eval_confidence.pkl` and `evaluate/tables/eval_confidence.csv`;
optional cross-run plots in `evaluate/figures/across_runs/`.

## 4. On/off-state detection {#states}

**Stage:** `states` · **Implementation:** `on_off_states.py`

**Example choices.** Apply one-tailed cluster correction to both on and off
states, with the inherited alpha of 0.05 for each. Inherited candidate rules
use `z > 1.645` for on states and `z <= 0.842` for off states. The preset sets
the minimum off-cluster size to **one bin**, overriding the script default of
five. Both correction-skipped comparison plots are enabled; the primary cached
outcomes still use correction.

At least two finite null estimates and uniformly spaced time-bin starts are
required. Each observed probability is standardized against the mean and
population standard deviation (`ddof=0`) of its trial/bin null estimates:
`z = (observed − null_mean) / null_std`. Each shuffled map is standardized
against the same null moments. Zero-null-variance bins are unclassified in
both masks and produce a warning. Nonfinite/out-of-range probabilities,
incompatible time axes, no bin starts in the delay interval, and overlapping
on/off candidate thresholds are errors.

The example's independent per-bin null permutations do not preserve the
permutation across adjacent time bins. Cluster correction warns about this
limitation. The alternative shared-across-time policy preserves that assignment
within each held-out trial, but trials still use independently generated
permutations and different training sets; full-session screening is unchanged.
It therefore does not by itself validate joint session-wide or
selection-corrected cluster inference.

Small null counts also warn. Under the current in-sample null standardization,
each standardized null value is bounded above by `sqrt(N − 1)`. With the smoke
preset's N=3, no null value can exceed the default on threshold 1.645, so its
on-cluster cutoff is necessarily zero. Those outputs check integration only.
Other small counts can have poor Monte Carlo precision in the configured tails;
increase N before interpreting the cluster results.

Candidate clusters connect adjacent time bins within a trial, never across
trials. Clustering uses the full decoded time grid, before extracting delay
outcomes. Cluster mass is the signed sum of z-scores in that cluster. The two state
definitions use different null summaries:

- **On states:** candidates exceed 1.645. Each
  shuffle contributes its maximum on-cluster mass across the session, or zero
  if it has no clusters. Keep observed masses strictly above the 95th
  percentile of these 100 null maximum masses.
- **Off states:** candidates satisfy `z <= 0.842`. Every nonempty candidate
  cluster meets the example's one-bin minimum. Pool all off-cluster masses
  from the 100 shuffles, then retain observed masses at or below the 95th
  percentile of this pooled distribution.

If observed off candidates need correction but there are no usable null off
clusters after the size filter, the stage raises an error with guidance to
review null count, thresholds, and cluster size. It does not substitute a
fabricated zero cutoff. An absence of observed off candidates can legitimately
produce an empty off-state mask.

Thus off states identify the implemented low/null-compatible confidence
criterion, not necessarily significantly below-chance decoding, and the
off-cluster procedure is not the on-state maximum-cluster test. Unclassified
bins can belong to neither state. The enabled correction-skipped comparisons
generate additional duration plots without replacing the corrected masks.

The state cache contains masks and two delay outcomes per preferred-cue trial:
total off-state bins and the longest contiguous off-state run, each multiplied
by the decoding stride. Delay membership uses bin starts **500 through 1400 ms
inclusive**. Duration is a discrete bin-count measure, not the union of window
coverage: the example's 91 delay bins at a 10 ms stride yield 910 ms if every
bin is off. This convention differs from the
half-open [500, 1400) feature window used in preparation.

**Alternatives, not used for the primary example outputs:** two-tailed
off candidates use `abs(z) <= z_threshold_off`; two-tailed on correction uses
the `(1 − alpha/2)` maximum-mass quantile; two-tailed off correction retains
pooled masses within the `[alpha/2, 1 − alpha/2]` interval. Skipping correction
retains candidate clusters (with the off-size criterion still applied).

**Outputs:** `states/on_off_states.pkl` and `states/figures/`, grouped into
confidence, masks, durations, and cluster-mass plots.

## 5. Activity comparisons {#activity}

**Stage:** `activity` · **Implementation:** `compare_activity_across_states.py`

The entry point coordinates numerical preparation in `activity_preparation.py`
and figure generation in `activity_plots.py`. Typed activity records in
`activity_types.py` keep cell selectivity and PCA metrics separate.

**Example choices.** Use 50 ms activity windows, seed 42, PEV-weighted selective
population means, PCA views, and longest-off-state comparisons. Set
`max_points_per_color_group=50` for point sampling in applicable plots.

After checking state/decoding/selection/data provenance, retrieve correct
preferred- and opposite-cue trials and subsample the larger group without
replacement to equal counts. The seed makes this balancing reproducible.
Compute firing rates at cached delay-bin starts using `activity_bin_width_ms`.
For each cell and bin, z-normalize across the combined balanced cue groups;
zero-variance activity maps to zero.

Compare preferred-cue on/off-state points and preferred/opposite-cue activity
using per-cell, population, pairwise, and marginal-distribution figures. Cell
views show up to the three preferred cells with highest cached mean PEV.
Population views distinguish preferred selective, other selective, and
stationary nonselective cells. The enabled PEV weighting affects selective-cell
population means only; nonselective means retain equal weights.

Cell-axis percentages report **selectivity PEV** from the configured screening
test windows. PCA-axis percentages report **explained variance of normalized
activity**. These are different quantities, stored separately; PCA does not
replace a cell's PEV or assign component numbers as cell IDs.

The enabled PCA fits one common basis to pooled normalized trial/bin points from
both balanced cue groups using all preferred cells, then projects the views
onto up to three components. Longest-off-state highlighting reuses
the fitted normalization and PCA transform. Plot sampling caps displayed
points; it does not refit decoders or alter cached state assignments. These
are descriptive comparisons, with normalization and PCA fitted to the plotted
cue groups, rather than a further held-out decoding evaluation. Activity-state
associations are descriptive: the state labels themselves were derived from
activity in the stationary population.

A session with no cached delay off-state bins emits a warning and continues
without maximum-off-state points. Its PCA still fits the balanced cue groups
and returns an empty maximum-off-state projection. Malformed/nonfinite masks
or nonfinite/non-increasing time bins are errors, rather than being treated as
an empty state population.

**Outputs:** `activity/figures/pev_weighted/` in this example, split into activity
and principal-component views, then state and cue comparisons.

## 6. Mixed-effects data preparation {#prepare}

**Stage:** `prepare` · **Implementation:** `prepare_data_for_mixedlm.py`

**Example choices.** Define active cells by normalized activity strictly above
zero and use EMA alpha 0.2. Prepare 50 within-session holdouts of 20% of eligible
trials with CV seed 42. `pev_weighted_average` is omitted here, so its inherited
value is **false**: mixed-effects features use equal cell weights even though
the activity plotting stage uses PEV weights. Both duration outcomes and the
separate raw CV cache are retained by default.

Validate cache provenance and require cluster-correction-applied off-state
outcomes with the expected delay metadata. Associate each correct preferred-cue
trial with its total and maximum off-state durations and three disjoint cell
groups: preferred selective, selective nonpreferred, and stationary nonselective.

Compute each cell's firing rate in four half-open periods:

| Period | Interval (ms) |
| --- | --- |
| Baseline | [−400, 0) |
| Encoding | [100, 300) |
| Pre-delay | [300, 500) |
| Delay | [500, 1400) |

For the full-data descriptive table, z-normalize each cell within its session
and period across cached preferred-cue trials, using `ddof=0`. Zero-mean or
zero-variance cells map to zero. Compute group mean normalized activity and
the fraction of cells strictly above zero. Empty groups produce zero features
and a session-level warning listing the absent populations. These zeros denote
an unavailable population, not measured average activity; inspect group counts
before interpreting models. Populated groups with zero activity do not trigger
that warning.
Group means use equal weights in this preset. PEV weighting is an alternative
that must be enabled consistently for preparation and compatible analyses;
it changes selective means, not active fractions or raw group cell counts.

For every activity mean and active fraction, construct a preceding-trial
exponential moving average. If `x_i` is the current feature, history satisfies
`h_1 = x_0` and `h_(i+1) = alpha × x_i + (1 − alpha) × h_i`. Trial zero has no
history and is omitted from all model tables; history follows the ordered
preferred-cue trials, not every original session trial. The example sets alpha to 0.2.

Write the descriptive table, preparation manifest, and a separate CV cache
containing raw per-cell rates, outcomes, cell counts/weights, and reproducible
within-session holdout assignments. The CV analyses reconstruct normalized
features from these raw rates rather than evaluating the full-data normalized
table on held-out rows.

**Outputs:** `prepare/trial_table.pkl`, `prepare/cv_feature_cache.pkl`, and
`prepare/manifest.json`. No `pev_weighted/` suffix is used in this example.

## Shared mixed-effects estimation and validation {#mixed-effects-estimation}

**Example choices shared by all five fitting stages.** Enable CV with 50
holdouts, a 0.2 holdout fraction, and seed 42. Set `significance_alpha=0.05` and
`cv_prediction_sample_per_model=1000`. Omitted settings retain both duration
outcomes, a 1000-iteration fitting budget, and unweighted activity features
where applicable. The 1000-row sample cap affects prediction plots, not metrics.

### Model fitting

The remaining five stages fit each selected duration outcome separately using
a Gaussian linear mixed model: `duration_is = X_is × beta + b_s + error_is`,
where `i` indexes trials and `s` indexes sessions, with a fixed intercept and
one session random intercept. Durations enter in milliseconds without a
transformation. Fits use maximum likelihood (`reml=False`) and the optimizer sequence L-BFGS, BFGS, Powell, and
Nelder–Mead as needed. Full-data fits require at least two sessions; fitting
failures and convergence information must be checked in the outputs.

Nested comparisons use `2 × (logLik_full − logLik_reduced)` against a chi-square
distribution with degrees of freedom equal to the parameter-count difference;
only tiny negative differences within floating-point tolerance are clipped to
zero, with a warning. A materially lower likelihood for the full nested model
invalidates the comparison and records an error instead of a p-value.
Outputs include coefficient estimates, standard errors, 95% intervals, AIC/BIC,
variance components, and marginal/conditional R². Marginal R² attributes
variance to fixed effects; conditional R² also includes random-intercept
variance. These variance-component R² values describe fitted models and are
different from the held-out predictive R² below. Reported coefficient and
model-comparison p-values are not adjusted across all fitted models, outcomes,
or thresholds.

Optimization success and inferential validity are recorded separately.
Rank-deficient fixed-effect designs, nonconvergence, or nonfinite point estimates
are model failures. A converged model with usable point estimates but an invalid
final Hessian, parameter covariance, or fixed-effect standard errors warns and
sets `inference_valid=false`. Its point estimates remain available, but Wald
standard errors, z statistics, p-values, and confidence intervals are withheld.
Comparisons involving such a model also withhold their likelihood-ratio p-value.
A withheld result does not establish a nonsignificant effect. Boundary warnings
alone do not invalidate inference when the final numerical checks pass.
Inspect `fit_success`, `fit_error`, `inference_valid`, `inference_error`,
`likelihood_ratio_valid`, and `likelihood_ratio_error` in the result tables.
Individual model failures retain diagnostic rows and allow other models to run.
An outcome with no usable model fits, or a CV analysis with no successful fits,
saves its diagnostics and then raises an error rather than reporting success.

### Repeated trial holdouts

In the example, CV repeats 50 random within-session holdouts of 20% of eligible
trials. The holdout size is rounded up, retaining at least one train and test
trial in every session. Models share each repeat's split; with the matched
example settings, all fitting stages reuse the prepared holdouts. Cell
normalization is fitted only on training model rows and applied to all trials; active
fractions and histories are then recomputed. Histories contain preceding
covariates from the trial sequence, including earlier held-out trials, but no
held-out duration outcomes. Cell selection and PEV weights remain full-session
quantities; they are not re-estimated within these folds.

### Performance and uncertainty

Held-out predictions are reported both from fixed effects alone and with
session intercepts estimated from training rows. Metrics include RMSE, MAE,
predictive R², session-centered R², and Pearson correlation. Metrics pool
held-out trial rows across sessions, so sessions with more held-out rows have
more influence. Predictive R² is `1 − SSE / SST`, using the mean held-out outcome
for SST, and may be negative. Session-centered R² centers observations and predictions
separately within each session’s held-out rows. This centering is part of
scoring only; it does not supply outcome information to model fitting.

These random trial holdouts assess prediction within observed sessions, not new-session
generalization or prospective forecasting. Full-delay activity is concurrent
with the duration outcome. Repeated holdouts are distinct from decoding null
shuffles and the removed observed decoder repeats. Prediction sampling affects
figures only, not CV metrics.

CV summaries aggregate successful fits and report failure counts. Their means,
standard deviations, medians, and 2.5th/97.5th percentiles describe variation
across overlapping random holdouts; they are not confidence intervals based on
50 independent datasets. Pairwise CV changes use matched model/parent fits
within a repeat. Check both fit-success counts and warnings before comparing
models. Ranked CV tables and plots admit only models with all requested fits
successful and finite held-out RMSE; excluded models warn and remain in the raw
and summary tables with `rank_eligible` and `rank_exclusion_reason`. This prevents
rankings based on different subsets of successful holdouts. Usable predictive
fits can remain in CV even when their coefficient inference is withheld.
A completed pipeline is not evidence that every requested model fitted
successfully.

## 7. Mixed-effects model families {#models}

**Stage:** `models` · **Implementation:** `compare_mixed_effect_models.py`

**Example choices.** Fit both duration outcomes using equal-weight activity
features, EMA alpha 0.2, and the shared 50-holdout CV settings above. The full
forward and reverse model families are built by the implementation; the preset
does not select a subset of them.

Use the prepared table and raw CV cache with the shared estimation procedure.
M0 contains the fixed and session random intercepts. M1 adds raw counts for all
three cell groups. For each of the four periods independently, build a forward
and reverse sequence of predictor blocks; every block contains one feature
per cell group. There are **34 models per outcome**: M0, M1, and eight
period-specific models for each of four periods. Periods are fitted separately
in this stage; this is not a single model containing all four periods.

| Model | Block added to the previous model |
| --- | --- |
| M2 | Mean normalized activity |
| M3 | Active-cell fraction |
| M4 | History of mean normalized activity |
| M5 | History of active-cell fraction |
| RM2 (starts from M1) | History of active-cell fraction |
| RM3 | History of mean normalized activity |
| RM4 | Active-cell fraction |
| RM5 | Mean normalized activity |

Each step is compared with its named parent using likelihood-ratio and
predictive-performance summaries. M5 and RM5 contain the same terms in a
different addition order. The two sequences show how incremental contributions
depend on which correlated predictors already enter the model.

**Outputs:** `models/outcomes/<outcome>/`, including full-data results and
`cross_validation/`.

## 8. Nested cell-count comparisons {#nested-count}

**Stage:** `nested-count` · **Implementation:** `nested_model_comparison_cell_counts.py`

**Example choices.** Fit both outcomes with all five count models and the
shared 50-holdout CV settings. The significance threshold is 0.05. No activity
weighting or history parameter enters these count-only models.

Fit five random-intercept models: M0, M1 with all three cell counts, and three
reduced models that each omit one count from M1. Compare M1 with M0 for the
joint count contribution and with each reduced model for the conditional
contribution of the omitted cell group. Use the same full-data and paired CV
procedures described above. This stage uses the unweighted prepared table.

Counts are constant within a session, so these predictors describe between-
session differences in sampled populations. Within-session holdouts do not
provide an independent new-session test of count effects. Results include
nested contrasts, coefficients, fit statistics, and paired CV changes.

**Outputs:** `nested-count/outcomes/<outcome>/`.

## 9. Nested period-activity comparisons {#nested-activity}

**Stage:** `nested-activity` · **Implementation:** `nested_model_comparison_mean_norm_activity.py`

**Example choices.** Use equal-weight preferred-cell mean activity, both
duration outcomes, and the shared 50-holdout CV settings. The significance
threshold is 0.05; the full M0–M5 sequence is fitted.

Start with M0 and add **preferred and selective-nonpreferred counts only** to
form M1. Then cumulatively add preferred-cell mean normalized activity from
baseline (M2), encoding (M3), pre-delay (M4), and delay (M5). This sequence uses
two count covariates, unlike the three-count M1 in the general model families.
It does not add active fractions or EMA predictors.

Each model is compared with its immediate predecessor using the shared
likelihood-ratio and CV procedures. PEV weighting would change the
preferred-cell mean features but is not enabled in this preset. Later-period
effects are conditional on earlier periods, so results depend on this addition
order. There are **six models per outcome**.

**Outputs:** `nested-activity/outcomes/<outcome>/`.

## 10. Active-cell threshold scan {#criticality}

**Stage:** `criticality` · **Implementation:** `find_active_cell_criticality.py`

**Example choices.** Scan percentiles **10, 20, 30, 40, 50, 60, 70, 80, and 90**
for both duration outcomes. Use equal-weight mean activity, EMA alpha 0.2,
and the shared 50-holdout CV settings, with significance threshold 0.05.

Map each configured percentile `p` to a standard-normal cutoff
`z_p = Phi_inverse(p / 100)`. These are theoretical normal quantiles, not
empirical percentiles of recorded activity. The example's cutoffs range from
approximately −1.282 to +1.282; its 50th-percentile cutoff is zero, matching
the main preparation stage's active threshold.
For each cutoff, regenerate active fractions and their histories from
normalized activity, using the same preparation and provenance checks.

Reuse the general model-family definitions. Fit models without active-fraction
terms once, and fit models containing current or historical active fractions
at each cutoff: **six threshold-independent models plus 28 models at each of
nine cutoffs, or 258 models per outcome**. During CV, rebuild cutoff-dependent
features using training-only normalization and common holdout splits. Output tables rank cutoffs within each
base model and summarize fit criteria, R², errors, and changes relative to the
50th percentile when present.

This stage measures sensitivity to the operational definition of an active
cell. Its name does not imply a test of a dynamical critical point. Choosing
the best cutoff from these results is exploratory; the pipeline does not add
an outer validation loop for that threshold choice.

**Outputs:** `criticality/outcomes/<outcome>/`; threshold-specific trial tables
and manifests in `criticality/prepared/active_thresholds/percentile_<NN>/`.
The shared raw holdout cache remains in `prepare/`.

## 11. Interactions across activity periods {#interactions}

**Stage:** `interactions` · **Implementation:** `test_interactions_across_periods.py`

**Example choices.** Fit both outcomes with equal-weight group mean activity
and the shared 50-holdout CV settings, using significance threshold 0.05.
The preset also supplies `history_alpha=0.2` for shared feature reconstruction;
the interaction-model formulas below contain no EMA predictors.

For each of the three cell groups separately, start from the shared M0 and
add the following cumulative blocks. Activity terms are that group's mean
normalized activity in the indicated period:

| Model | Added block |
| --- | --- |
| IM1 | Cell count |
| IM2 | Baseline activity |
| IM3 | Encoding activity |
| IM4 | Baseline × encoding |
| IM5 | Pre-delay activity |
| IM6 | Pre-delay × baseline and pre-delay × encoding |
| IM7 | Delay activity |
| IM8 | Delay × baseline, delay × encoding, and delay × pre-delay |
| IM9 | Cell count × each of the four period activities |

Interaction terms are products of existing predictors, with their constituent
main effects already included. Fit 28 models per outcome: one M0 and nine
steps for each group. Shared likelihood-ratio comparisons assess each added
block, while coefficient tables describe individual interaction terms and
common CV holdouts assess predictive changes. PEV weighting is an alternative
for selective-group activity means, not enabled here. No cross-cell-group
interaction terms are added.

**Outputs:** `interactions/outcomes/<outcome>/`, including interaction estimates,
model progression, final IM9 summaries, and CV results.

## Reporting an example-preset analysis

Record the code revision, input session IDs, preset and resolved manifest,
retained cell/trial counts, decoder time grid and null count, state rules and
duration convention, and the outcome/model families analyzed. Report actual
successful-fit counts alongside the requested 50 model holdouts. Distinguish
full-data model summaries from held-out performance, and describe the model
holdouts as within-session validation with full-session cell screening.

The runner preserves each invocation in `manifests/<run_id>.json`, including
partial reruns. `pipeline_manifest.json` shows the latest invocation only.
When an analysis spans multiple invocations, report the history records for
its upstream and downstream stages. For CLI runs, history also preserves the
Python invocation and working directory as a command reference. The saved
resolved settings remain the record of effective analysis choices. Manifest
history preserves settings and
execution records; it does not version the generated analysis files or copy
source code. Record the code revision separately. Standalone stage commands do
not add runner history; use `pipeline.py --stages <stage>` to record partial
work. See
[Validation](../validation/next.md) for the scope of integration tests and
[Outputs](outputs.md) for the recorded estimates, figures, and manifest history.
