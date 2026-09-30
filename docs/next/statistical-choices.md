# Statistical choices for reliable decoder state estimates

This document records three decisions that materially affect confidence and
ON/OFF-state estimates: how training classes are balanced, whether probabilities
are calibrated, and how regularization is selected. It separates the current
analysis policy from the evidence supporting it. **Shorter OFF states are not,
by themselves, evidence of a better estimator.**

**Follow-up, 2026-09-30:** [Decoder and state robustness](decoder-state-robustness.md)
adds two-class outer validation, session-level M1 prediction, and sensitivity
analyses. Weighting and calibration gain stronger support for probability
estimation. The C conclusion needs qualification: weighted calibrated fixed
C=0.01 outperforms the historical accuracy-based search in that validation.
The updated production comparison now tests this choice directly in `006`
versus `005`; stronger M1 performance is still not a criterion for choosing
a decoder.
The [smaller-C study](regularization-confidence.md) further checks C=0.01:
uniform probability shrinkage cannot explain its advantage, and C=0.001 does
not improve consistently across validation panels.

The evidence snapshot is updated **2026-09-30**. It covers the completed runs
`next_run_001` through `next_run_006`, plus the focal experiments described below.
Runs `001`–`004` test the four combinations of calibration and C selection
under downsampling; `005` tests the complete weighted procedure against `001`;
`006` tests weighted, calibrated fixed C=0.01 against weighted C search in `005`.
No analysis configuration or production cache was modified to prepare it.

**Current default (2026-09-30):** `configs/next/default_pipeline.json` implements
all-trial balanced class weights, **fixed C=0.01 with grid search disabled**,
and five-fold sigmoid calibration. It is the dashboard's default template.
This follows the two-class and smaller-C probability-score studies and is
now supported by completed fixed-C run `006`; weighted run `005` remains
the C-search reference. Choose
`decode.training_balance` to use balanced class weights, historical balanced
trial subsampling, or no balancing. Weighting applies within every classifier
fit and to pooled probability calibration, including null fits. The completed
weighted runs now provide evidence across 25 sessions, alongside the earlier
single-session experiment. See
[training-class balance](configuration.md#training-class-balance) for exact
settings, SVM handling, and migration.

| Decision | Current policy | Current evidence |
| --- | --- | --- |
| Training-class balance | Use all eligible training trials with balanced class weights; balance calibration for the same cue prior | `005` vs `001`: modest pooled probability-score gains, lower Brier in 16/25 sessions; improved seed stability in the earlier focal experiment |
| Confidence calibration | Retain training-only sigmoid calibration for observed and null fits | Downsampled production ablations improve both losses in all 25 sessions with search and with C=1; weighted two-class validation also supports calibration |
| Regularization | Fixed C=0.01, with C search disabled; retain weights and sigmoid calibration | `006` improves both preferred-cue losses over `005` in all 25 sessions; two-class validation agrees. Decoder wall time falls from 19.79 to 7.26 hours. This is not a universal optimum |

The completed-run comparisons align **25 sessions, 1,590 preferred-cue test
trials, and 161 time bins per trial**. The delay analysis uses the 91 bin starts
from 500 through 1400 ms, inclusive: 144,690 observed trial-bin predictions and
14,469,000 null predictions per run. Each window covers 50 ms and advances by
10 ms. Durations count bin starts ×10 ms, so the full delay grid represents
910 ms under the implemented convention.

Session cues, neuron IDs, test-trial IDs/order, and time grids match across all
six runs. Cached scientific decoder settings differ only as shown below after
translating the historical balancing boolean to its named mode. All six use
seed 42, stationary cells, and 100
independently permuted training-label null fits per trial/bin.
State thresholds and cluster rules match as well. This checks recorded settings
and cached populations. Older manifests alone do not establish software-version
parity. For the new `005`/`006` contrast, all 50 native decoder fingerprints
verify against the same current scientific source, recording/selection inputs,
and numerical package versions, without a legacy-trust override.

| Run | Training-class balance | Sigmoid calibration | C selection |
| --- | --- | --- | --- |
| `next_run_001` | Random downsampling | Enabled, five folds | Search over 1, 0.1, 0.01 |
| `next_run_002` | Random downsampling | Disabled | Search over 1, 0.1, 0.01 |
| `next_run_003` | Random downsampling | Enabled, five folds | Fixed C=1 |
| `next_run_004` | Random downsampling | Disabled | Fixed C=1 |
| `next_run_005` | All trials + balanced class weights | Enabled, five folds, balanced calibration weights | Search over 1, 0.1, 0.01 |
| `next_run_006` | All trials + balanced class weights | Enabled, five folds, balanced calibration weights | Fixed C=0.01 |

Reported run-comparison scores weight each delay trial-bin equally. They are
**preferred-cue-only scores**, not balanced two-class accuracy or a complete
calibration assessment: every cached test label is 1. Overlapping bins, shared
training sets, and repeated null fits are dependent; their large counts must
not be treated as independent statistical replicates. Cell screening remains
fixed at the full-session level. See the [analysis methods](methods.md) for
the underlying estimator and state definitions.

## 1. Training-class balance: retain trials instead of randomly discarding them

### The failure mode observed with downsampling

The historical procedure first excludes the test trial, then randomly samples
the larger cue class down to the smaller class's size. Each observed fit and
its shuffled-label null fits use the same selected training trials. This
equalizes class counts, but turns the identity of discarded trials into a
source of estimator variability.

In session **221024**, target trial **136** is cue 1. Its available training
pool contains 57 correct cue-1 trials and 86 correct opposite-cue trials.
Downsampling keeps 57+57=114 trials and discards 29 opposite-cue trials. The
decoder has 208 neuronal features. The legacy and next pipelines generate
different balancing sequences even when both receive seed 42, so equal seeds
do not imply equal training membership. Trial IDs here are zero-based indices
in the original recording, not rows in a decoder cache.

The discrepancy first appeared as a maximum OFF duration of **140 ms in
`run_037_full_session` versus 240 ms in `next_run_001`**. A decisive connecting
bin starts at 830 ms and uses activity in [830, 880) ms. A small confidence
change there can split one long contiguous OFF interval into two shorter ones.
The held-out trial's activity need not change at all.

A controlled substitution established that membership matters. Starting with
the original next training set, replace opposite-cue trial 233 with trial 199
in the same row position, retaining the other 113 trials and the 57/57 counts.
Rerun C selection, sigmoid calibration, and 100 null fits at all 161 bins:

| Quantity | Original next training set | Replace 233 with 199 |
| --- | ---: | ---: |
| Target confidence at 830 ms | 0.5158 | 0.5964 |
| Own-null OFF probability cutoff at 830 ms | 0.5774 | 0.5737 |
| Selected C at 830 ms | 0.01 | 0.01 |
| Maximum OFF duration | 240 ms | 130 ms |

The changed observed predictions still produce 130 ms when evaluated against
the original null; changing only the null leaves 240 ms. This substitution is
sufficient to break the long interval, but does not explain every old/next
difference elsewhere in the trace.

Activity inspection explains the influence. Trials 233 and 199 have similar
total counts in that window—61 versus 58 spikes across the 208 neurons—but
different population patterns. After common per-neuron standardization using
the non-target training pool, trial 233 ranks 7th most similar to the target
among 86 opposite-cue trials; trial 199 ranks 80th. Several influential neurons
fire in both trial 233 and the target, but are silent in 199. Replacing the ten
neuronal values with the largest positive individual substitution effects
reproduces approximately 88% of the full probability change. These partial
patterns are synthetic diagnostic probes, not biological trials.

Later seed experiments identified another influential pair: trials 181 and 5.
Seeds 1012 and 1018 included 181 and excluded 5 at the critical bin. A balanced
181→5 substitution raised confidence from 0.472→0.682 and 0.361→0.586 with C
and remaining calibration-fold assignments fixed. Trial 5 was already present
in both original seed-42 runs, so this pair explains later seed sensitivity,
not the original membership difference. Effects depend on the remaining
training data; no trial is a universal switch or an established bad recording.

### Implemented fix and how it is fitted

Retain every eligible non-target training trial and assign each trial in class
`k` weight `n / (2 * n_k)`. Both cue classes then have equal total weight. In
the focal 57/86 pool, individual weights are approximately 1.2544 and 0.8314,
respectively, with total weight 71.5 per class. This is the convention used by
[`LogisticRegression(class_weight="balanced")`](https://scikit-learn.org/1.8/modules/generated/sklearn.linear_model.LogisticRegression.html).

The implemented procedure is:

1. Exclude the complete test trial before training, scaling, C selection, or
   calibration. Keep all remaining eligible trials.
2. Recompute balanced class weights within every inner training fold. Fit its
   scaler using only that fold's training activity. Both the focal experiment
   and production implementation keep the usual unweighted `StandardScaler`.
3. Retune C rather than assuming that the optimum survives the change in
   sample count and weighted loss. In the experiment the grid and balanced-
   accuracy scoring rule remained unchanged.
4. Calibrate out-of-fold margins with equal total weight per cue as well.
   Otherwise ordinary calibration on the 57/86 pool targets its unequal cue
   prevalence even though the classifier uses balanced weights.
5. Use the same full training membership for observed and null fits. For every
   null, permute training labels and repeat weighting, C selection, and
   calibration under those labels. Never reuse a downsampled null as the
   production null for the weighted estimator.

The target is an **equal-cue-prior decoding probability**, consistent with the
previous balanced sampling design. It is not an estimate of the recording's
natural cue prevalence. The change retains difficult as well as easy examples;
it does not select trials based on their effect on the test confidence.

### Evidence from the all-trial weighting experiment

The focal comparison used five fitting seeds, 42–46, and 100 fresh null fits
per bin and seed. Both methods used common source-trial-based CV conventions.
A separate evaluation used three repeats of five outer folds on the 143
non-target trials, with C selection and calibration fitted entirely inside
each outer training fold. The target trial did not enter this evaluation.
Scores average individual fits, not a five-model ensemble.

| Outer-validation metric | Downsampling | All trials + balanced weights |
| --- | ---: | ---: |
| Class-balanced Brier score, lower is better | 0.1370 | 0.1269 |
| Class-balanced log loss, lower is better | 0.4267 | 0.3994 |
| Balanced accuracy | 80.59% | 82.08% |
| Mean prediction SD across fitting seeds | 0.0665 | 0.0243 |

Brier loss fell 7.4%, and prediction variability fell 63.5%. Brier improved
in all three outer-fold repetitions and on 132 of 143 trial-averaged
comparisons. These are exploratory, correlated comparisons within one session,
not 143 independent confirmations of a new default.

| Seed | Downsampled maximum OFF | Weighted maximum OFF |
| --- | ---: | ---: |
| 42 | 240 ms | 130 ms |
| 43 | 200 ms | 120 ms |
| 44 | 200 ms | 120 ms |
| 45 | 130 ms | 130 ms |
| 46 | 120 ms | 120 ms |

Every weighted fit classified 830 ms as not OFF. These runs regenerate only
the focal trial's nulls; using either historical session background, a
focal-only background, or skipping cluster correction did not change these
maxima. They are not a complete weighted session-level cluster analysis.

### Completed production evidence: `next_run_001` versus `next_run_005`

Both runs use C search and sigmoid calibration. `005` retains all eligible
training trials and uses balanced classifier and calibration weights for
observed and null fits. Unlike the earlier focal experiment, it regenerates
the nulls for **all 25 sessions**, including the full session backgrounds used
by state correction. The comparison changes the entire balancing procedure:
membership, fold-specific class costs, calibration weights, selected C values,
and the resulting null fits. It does not isolate classifier weighting alone.

| Delay-period metric | `001`: downsampling | `005`: all trials + balanced weights |
| --- | ---: | ---: |
| Preferred-cue-only Brier score | 0.20918 | 0.20762 |
| Preferred-cue-only log loss | 0.59968 | 0.59567 |
| Preferred-cue accuracy at p ≥0.5 | 64.19% | 64.51% |
| Mean per-trial/bin null probability SD | 0.05992 | 0.05775 |
| Delay bins classified ON after correction | 34.18% | 36.13% |
| Delay bins classified OFF after correction | 46.93% | 45.70% |
| Mean total OFF duration per trial | 427.0 ms | 415.8 ms |
| Mean maximum OFF duration per trial | 140.6 ms | 135.7 ms |
| Trials with maximum OFF ≥240 ms | 12.77% | 12.33% |

Pooled Brier improves **0.74%** and log loss **0.67%**. Session-average Brier
improves in **16 of 25 sessions**, and log loss in **15 of 25**. These are modest,
mixed gains, substantially smaller than the earlier focal validation gains.
The OFF mask differs in 10.53% of delay bins; maximum OFF duration changes for
1,097 trials, becoming shorter for 629 and longer for 468. Its mean absolute
change is 30.1 ms, even though the pooled mean falls only 4.9 ms.

For session 221024, trial 136, `005` reproduces the weighted maximum of
**130 ms**, versus `001`'s 240 ms. At 830 ms, confidence rises from 0.5158 to
0.5854 while the own-null OFF cutoff falls from 0.5774 to 0.5426. The connecting
bin is therefore no longer OFF. Its z score is 1.611, below the ON threshold
1.645: breaking an OFF interval does not require becoming ON.

For context, the same focal quantities across all six completed runs are:

| Run | Maximum OFF | Total OFF | Confidence at 830 ms | Own-null OFF cutoff at 830 ms |
| --- | ---: | ---: | ---: | ---: |
| `001` | 240 ms | 450 ms | 0.5158 | 0.5774 |
| `002` | 280 ms | 600 ms | 0.5071 | 0.6768 |
| `003` | 240 ms | 440 ms | 0.5416 | 0.5760 |
| `004` | 250 ms | 530 ms | 0.5866 | 0.7461 |
| `005` | 130 ms | 420 ms | 0.5854 | 0.5426 |
| `006` | 130 ms | 420 ms | 0.5854 | 0.5471 |

Each production procedure has only **one fitting seed**. `005` therefore does
not measure seed-to-seed variability across sessions; that evidence still
comes from the five-seed focal experiment. Equal seed numbers also do not
produce identical null assignments when training membership changes.

**Decision:** retain all-trial balanced weighting as the default. Its primary
advantage is retaining eligible training information and removing random trial
omission; the completed production run adds modest aggregate probability-score
support. Weighting does not remove finite-sample
influence. For example, removing trial 5 from the weighted seed-42 fit still
lowers 830 ms confidence from 0.5854 to 0.5280 in an observed-only diagnostic.
Nor does this change establish the biological correctness of a 120–130 ms
OFF duration. Two-class outer validation and repeated production seeds across
sessions would provide stronger evidence of generalization and stability.

## 2. Confidence calibration: the probability scale affects state inference

### Why classification accuracy is insufficient

The decoder produces probabilities, and the state stage compares each observed
probability with the mean and SD of probabilities from shuffled-label fits:
`z = (p_observed - mean(p_null)) / sd(p_null)`. The current candidate thresholds
are `z <= 0.842` for OFF and `z >= 1.645` for ON, followed by cluster rules.
Consequently, the probability scale and the behavior of the null fits matter
even when predicted cue labels barely change.

Sigmoid calibration learns a mapping from classifier margins to probabilities
using out-of-fold training predictions. In this implementation,
`CalibratedClassifierCV(method="sigmoid", ensemble=False)` uses five grouped
folds and a final base classifier fitted on the full selected training set.
Calibration is repeated for every observed and null training problem. The
held-out target is excluded. Merely applying a logistic sigmoid to a margin
does not guarantee reliable probabilities in finite, regularized fits;
cross-fitted calibration estimates a separate mapping.

Probability scores and reliability are related but distinct. Brier score and
log loss assess overall probabilistic prediction quality, including
discrimination; a lower Brier score alone does not prove better calibration.
A full reliability assessment needs representative held-out labels from both
classes. See [scikit-learn's calibration guide](https://scikit-learn.org/1.8/modules/calibration.html).

### Evidence: `next_run_001` versus `next_run_002`

The cached selected C values are **identical for every observed and null fit**
across these runs, including all 161 bins. The comparison therefore does not
confound calibration with a changed C-search result.

| Delay-period metric | `001`: sigmoid calibrated | `002`: calibration disabled |
| --- | ---: | ---: |
| Preferred-cue-only Brier score | 0.20918 | 0.22655 |
| Preferred-cue-only log loss | 0.59968 | 0.69287 |
| Preferred-cue accuracy at p ≥0.5 | 64.19% | 64.08% |
| Mean of per-trial/bin null probability SDs | 0.05992 | 0.24185 |
| Null probabilities below 0.1 or above 0.9 | 0.055% | 17.845% |
| Delay bins classified ON after correction | 34.18% | 1.14% |
| Delay bins classified OFF after correction | 46.93% | 53.31% |
| Mean maximum OFF duration per trial | 140.6 ms | 195.0 ms |
| Median maximum OFF duration per trial | 120 ms | 170 ms |
| Trials with maximum OFF ≥240 ms | 12.77% | 27.74% |

Calibration improves the available Brier score by 7.7% and log loss by 13.4%,
with lower session-average Brier in **all 25 sessions**. Preferred-cue accuracy
changes by only 0.12 percentage points. Thus the evidence concerns probability
quality and state inference, not a large change in basic cue classification.

Both null means remain close to 0.5 when averaged over the delay samples.
However, disabling calibration increases the mean per-bin null SD about
fourfold and frequently assigns near-certain probabilities to models trained
on randomized labels. This is consistent with overconfident outputs under
the shuffled-label training condition. It does not establish a complete
two-class reliability curve for the observed models.

The widened null has a direct consequence for the implemented thresholds:
the ON probability cutoff `mean(p_null) + 1.645 * sd(p_null)` exceeds 1 in
**16.17%** of uncalibrated delay trial-bins, versus none in the calibrated run.
No valid probability can cross that cutoff. This helps explain the loss of
detected ON bins despite nearly unchanged preferred-cue accuracy.

The OFF mask differs in 20.33% of delay trial-bins; maximum OFF duration changes
for 1,481 of 1,590 trials. Disabling calibration lengthens the maximum for
1,236 trials and shortens it for 245. For session 221024, trial 136, the maximum
is **240 ms calibrated versus 280 ms uncalibrated**. At 830 ms, p changes only
from 0.5158 to 0.5071, while the own-null OFF probability cutoff changes from
0.5774 to 0.6768. This example illustrates why observed confidence must be
interpreted with its own fitted null distribution.

A matched null is necessary, but it does not make state inference invariant
to calibration. A single common positive affine transformation would cancel
in the z score. These runs instead apply fitted nonlinear mappings separately
to observed and shuffled models; their distribution shapes and relative scales
can change.

### Calibration at fixed C=1: `next_run_003` versus `next_run_004`

The fourth run tests calibration without C selection. Both caches use C=1
for every observed and null fit, with the same downsampling settings.

| Delay-period metric | `003`: sigmoid calibrated | `004`: calibration disabled |
| --- | ---: | ---: |
| Preferred-cue-only Brier score | 0.21634 | 0.26205 |
| Preferred-cue-only log loss | 0.61859 | 0.94429 |
| Preferred-cue accuracy at p ≥0.5 | 62.82% | 64.14% |
| Mean per-trial/bin null probability SD | 0.05755 | 0.33416 |
| Null probabilities below 0.1 or above 0.9 | 0.046% | 39.895% |
| Delay bins classified ON after correction | 28.62% | 3.48% |
| Delay bins classified OFF after correction | 50.11% | 48.07% |
| Mean maximum OFF duration per trial | 139.5 ms | 128.1 ms |

Calibration improves Brier **17.4%** and log loss **34.5%**, with both scores
better in all 25 sessions. These improvements are larger than with C search,
despite the uncalibrated run's slightly higher preferred-cue hit rate. The
uncalibrated null SD is almost six times larger, and its ON probability cutoff
exceeds 1 in **68.40%** of delay trial-bins. The raw probabilities' scale is
poorly suited to the implemented state thresholds.

Yet disabling calibration here **shortens** the mean maximum OFF interval by
11.4 ms, unlike the lengthening seen with C search. This is particularly clear
evidence that shorter OFF intervals are not an estimator-quality criterion.
The focal trial moves in the opposite direction to the aggregate: its maximum
increases from 240 to 250 ms. Use probability scores and the observed/null
behavior to assess the estimator, rather than optimizing state durations.

**Decision:** retain training-only sigmoid calibration, including in the
weighted procedure and in every null fit. These comparisons support that
choice for this dataset. They do not imply that all logistic regression models
always require calibration or that the calibrated state labels are ground
truth. Confirm reliability with two-class held-out predictions and keep the
calibration cue prior consistent with the weighting policy.

## 3. C selection: prediction gains depend on calibration

### What the grid search changes

The logistic classifier uses L2 regularization. `C` is the inverse
regularization-strength parameter: smaller values shrink coefficients more
strongly. The optional search, used in historical runs `001`, `002`, and `005`, evaluates **C=(1, 0.1, 0.01)** using mean balanced
accuracy over five source-trial-grouped inner folds, with scaling fitted only
inside each training fold. A tie selects the first candidate in that order.
See the [estimator definition](https://scikit-learn.org/1.8/modules/generated/sklearn.linear_model.LogisticRegression.html).

Search is performed separately for every observed and null trial/bin problem.
The chosen C is then used for sigmoid calibration and final fitting. The
search objective is classification accuracy, not probability loss, and the
winning inner-CV score is not an unbiased estimate of generalization quality.
Selecting among noisy CV estimates can itself overfit the selection criterion;
see [Cawley and Talbot (2010)](https://www.jmlr.org/papers/v11/cawley10a.html).

### Evidence: `next_run_001` versus `next_run_003`

Both runs use sigmoid calibration and the same balancing seed, populations,
time grid, and null policy. `001` searches C; `003` fixes C=1. Where the search
selects C=1, the observed probabilities in the two caches are **exactly equal**
at every matching bin. This is a useful internal control for the comparison.

| Delay-period metric | `001`: search C | `003`: fixed C=1 |
| --- | ---: | ---: |
| Preferred-cue-only Brier score | 0.20918 | 0.21634 |
| Preferred-cue-only log loss | 0.59968 | 0.61859 |
| Preferred-cue accuracy at p ≥0.5 | 64.19% | 62.82% |
| Mean per-trial/bin null probability SD | 0.05992 | 0.05755 |
| Delay bins classified ON after correction | 34.18% | 28.62% |
| Delay bins classified OFF after correction | 46.93% | 50.11% |
| Mean total OFF duration per trial | 427.0 ms | 456.0 ms |
| Mean maximum OFF duration per trial | 140.6 ms | 139.5 ms |
| Median maximum OFF duration per trial | 120 ms | 120 ms |

Search improves the available Brier score by **3.3%**, log loss by **3.1%**,
and preferred-cue accuracy by **1.37 percentage points**. Its session-average
Brier is lower in all 25 sessions, although the advantage is very small in
one session. These are descriptive paired results for the preferred-cue test
population; they do not establish two-class performance or statistical
independence across bins.

The search frequently chooses stronger regularization:

| Selected C | Observed delay fits | Null delay fits |
| --- | ---: | ---: |
| 1 | 18.22% | 37.24% |
| 0.1 | 25.09% | 24.42% |
| 0.01 | 56.69% | 38.34% |

Thus C=1 differs from the selected value in 81.78% of observed delay fits.
This frequency supports evaluating stronger regularization, but does not
prove that the winning C in every individual noisy fold comparison is optimal.
Calibration and regularization address different parts of the estimator:
calibration adjusts the margin-to-probability mapping; C changes the fitted
classifier. Calibration does not substitute for choosing an appropriate C.

The nearly identical mean maximum OFF duration hides substantial changes:
10.997% of delay OFF-mask entries differ, and 1,042 of 1,590 trials have a
different maximum. Fixed C=1 lengthens the maximum for 552 trials and shortens
it for 490; the mean absolute per-trial change is 27.2 ms. Total OFF time also
increases even though mean maximum duration barely changes. Total duration
and the longest contiguous interval must therefore be evaluated separately.

For the focal session-221024 trial, both runs give a **240 ms** maximum.
At 830 ms, search selects C=0.01 and p=0.5158, whereas fixed C=1 gives p=0.5416;
both remain below their own OFF cutoffs. At 910 ms, search already selects
C=1, and the observed probabilities are identical. **Fixing C=1 does not
resolve the original long-segment example.**

### C search without calibration: `next_run_002` versus `next_run_004`

This comparison repeats the C ablation with raw logistic probabilities. Where
`002` selects C=1, observed probabilities again match `004` exactly at every
matching bin. Searching C improves preferred-cue Brier from **0.26205 to
0.22655 (13.5%)** and log loss from **0.94429 to 0.69287 (26.6%)**, with both
scores better in all 25 sessions. The gains are substantially larger than the
3.3% and 3.1% obtained when calibration is enabled.

The null SD falls from 0.33416 with fixed C to 0.24185 with search, but remains
much larger than in either calibrated run. Mean maximum OFF duration increases
from 128.1 to 195.0 ms with search; the OFF mask differs in 20.11% of delay
bins. Again, the run with better probability scores need not have shorter OFF
states.

The four downsampled runs therefore show **non-additive effects**. The absolute
Brier improvement from calibration is 0.01738 with C search and 0.04571 with
C=1. Conversely, C search improves Brier by 0.00716 with calibration and 0.03550
without it. These are descriptive contrasts of pooled preferred-cue scores,
not independent-bin significance tests. Neither setting compensates fully for
disabling the other in these runs: `001`, with both enabled, has the best
probability scores among the four downsampled configurations.

### Weighted fixed C=0.01: `next_run_005` versus `next_run_006`

This is the direct production test of the current default. Both runs retain
all training trials, use balanced classifier/calibration weights and sigmoid
calibration, and regenerate all 100 null fits per trial/bin. The only recorded
scientific-setting changes are `classifier_c: 1 → 0.01` and
`grid_search_for_c: true → false`; the scalar C=1 was overridden by search in
`005`. All 25 session populations, cues, trial IDs, time grids, and state
settings match. All observed and null C values in `006` are 0.01.

The current scientific decoder digest and runtime/input provenance verify for
all 50 cached session results. Where `005` already selected C=0.01, `006`
reproduces **139,714 observed and 10,072,312 null probabilities bit for bit**
over the full 161-bin grids. This directly checks that matched fits share the
same fitting and randomization behavior; the changed C policy is the operative
procedural difference.

| Delay-period metric | `005`: C search | `006`: fixed C=0.01 |
| --- | ---: | ---: |
| Preferred-cue-only Brier score | 0.20762 | 0.20542 |
| Preferred-cue-only log loss | 0.59567 | 0.58926 |
| Preferred-cue accuracy at p ≥0.5 | 64.51% | 64.79% |
| Mean absolute distance of observed probability from 0.5 | 0.17778 | 0.18163 |
| Mean per-trial/bin null probability SD | 0.05775 | 0.05939 |
| Mean maximum OFF duration per trial | 135.7 ms | 144.5 ms |
| Mean total OFF duration per trial | 415.8 ms | 410.2 ms |

Both probability losses improve in **25/25 sessions**, and their equal-session
means improve within **each of monkeys A, H, and J**. Pooled Brier improves
**1.06%** and log loss **1.08%**. The gain is modest but consistent, and agrees
with the earlier two-class validation: primary Brier **0.20175 → 0.19880** and
fixed-cue/all-cell Brier **0.22041 → 0.21897**. Production accuracy remains a
preferred-cue hit rate; its 0.28-percentage-point increase is not a claim about
balanced two-class accuracy.

The equal-session Brier change (`006 − 005`) is **−0.002346**, with a
descriptive 95% session-bootstrap interval **[−0.002922, −0.001792]**; log-loss
change is **−0.006911 [−0.008705, −0.005272]**. These intervals resample 25
sessions, not individual bins, and are nominal because sessions within only
three monkeys can be dependent. They do not quantify uncertainty over a
population of monkeys. Mean preferred-cue probability rises in every session;
such a shift can improve these one-class scores without improving
balanced discrimination or calibration. The earlier two-class results are
therefore essential complementary evidence.

The confidence change is **not uniform shrinkage toward 0.5**: observed mean
absolute distance from 0.5 increases. Null SD also increases slightly rather
than collapsing. The earlier [regularization-path study](regularization-confidence.md)
provides the stronger two-class discrimination and shrinkage-control evidence;
this production comparison adds full-session confirmation of the direction.

Fixed C=0.01 is therefore preferable to the tested balanced-accuracy C search
for the current probability-estimation goal and cohort. This does not establish
superiority to a search designed and validated for probability loss, nor a
universal optimum. The choice was motivated by earlier analyses of these same
recordings, so `006` is a new fitting run, not an independent confirmatory data
set. Both production runs use one fitting seed. A weighted fixed-C calibration
ablation is also absent: `006` alone does not isolate the benefit of calibration
at C=0.01.

### Downstream OFF and M1 outcomes

Improved probability scores do **not** produce uniformly shorter OFF states.
With fixed C=0.01, mean maximum OFF grows by **8.8 ms** and is longer in
22/25 session means, while mean total OFF falls by **5.6 ms** and is shorter
in 20/25 session means. The focal session-221024 trial 136 retains a **130 ms
maximum and 420 ms total** in both runs. The observed probabilities, null
scale, and nonlinear cluster filtering jointly determine these outcomes;
this comparison does not isolate a single mechanism for the duration changes.

The [focused M1 audit](../validation/off-m1-run-comparison.json) aligns the
**1,565 prepared rows**, predictors, and 50 stored within-session holdouts
between `005` and `006`. Each repeat holds out 20% of trials within already-seen
sessions. All 50 fits of each M0/M1 model, per outcome and run, converge and
are valid.
M1 adds preferred-selective and all-other-selective cell counts to M0; these
predictors vary between sessions, not between trials in the same session.

| M1 predictive quantity | `005`: C search | `006`: fixed C=0.01 |
| --- | ---: | ---: |
| Fixed-effect trial-holdout R², maximum OFF | 0.14695 | 0.14301 |
| Fixed-effect trial-holdout R², total OFF | 0.25453 | 0.24091 |
| Fixed-effect SSE reduction versus own M0, maximum OFF | 14.95% | 14.63% |
| Fixed-effect SSE reduction versus own M0, total OFF | 25.72% | 24.39% |
| Held-out-session R², maximum OFF | 0.45945 | 0.40893 |
| Held-out-session R², total OFF | 0.61440 | 0.61678 |
| Held-out-animal R², maximum OFF | 0.52269 | 0.49082 |
| Held-out-animal R², total OFF | 0.62831 | 0.62813 |

Trial-holdout R² is the mean of 50 repeat R² values. The SSE reductions sum
`n_test × RMSE²` across repeats before taking the M1/M0 ratio; they are not an
average of percentage gains. The table uses fixed-effect predictions for trial
holdouts and gives sessions equal weight for animal-holdout scoring.
Conditional predictions also include fitted intercepts for training sessions; adding counts reduces their
SSE by only **0.045% → 0.064%** for maximum OFF and **0.075% → 0.082%** for
total OFF. Session-level validation instead fits an equal-session OLS model
with the two counts and compares each held-out prediction with a
training-session-mean baseline; animal holdouts exclude all sessions from the
test monkey. These are distinct prediction targets. The 50 trial holdouts
reuse recordings and are not 50 independent replications.

There is **no general M1 performance improvement**. Pooled held-out count
prediction remains useful relative to each run's own baseline, but maximum-OFF
generalization weakens, and total-OFF results are mixed. Outcome distributions themselves
change between runs, so cross-run RMSE or R² is not a comparison against a
common biological target. The preferred-selective maximum-OFF effect remains
uncertain: the `006` equal-session bootstrap interval for that coefficient is
**[−5.899, 1.465] ms/cell**. This is a pointwise 95% session-pairs interval
conditional on the sampled animals; it is not animal-population uncertainty.
Three monkeys limit biological generalization.
This is compatible with retaining C=0.01 for its better probability estimates
while declining to claim stronger biological evidence from its M1 result.

### Runtime and the practical decision

The original fitting manifests record these decoder-stage wall times, each
configured for ten workers:

| Procedure | Decoder wall time |
| --- | ---: |
| Downsampled, C search + sigmoid calibration (`001`) | 15.68 hours |
| Downsampled, C search without calibration (`002`) | 11.39 hours |
| Downsampled, fixed C=1 + sigmoid calibration (`003`) | 6.37 hours |
| Downsampled, fixed C=1 without calibration (`004`) | 0.95 hours |
| All-trial weights, C search + sigmoid calibration (`005`) | 19.79 hours |
| All-trial weights, fixed C=0.01 + sigmoid calibration (`006`) | 7.26 hours |

These are observed invocation times, not a controlled hardware/load benchmark.
In particular, `001`'s latest manifest records a 21.8-second **plot-only**
decoder invocation; it must not be used as its fitting runtime. The original
fitting manifest is identified in the evidence snapshot.
The weighted run took **26.3% longer** than the corresponding downsampled run,
consistent with fitting more training trials, but the timing comparison does
not isolate the cause or predict runtime on other machines. `004` is much
faster and has the poorest probability scores of these six runs.
Fixed-C `006` uses **63.3% less decoder wall time** than `005` (a **2.73×**
observed speedup), a reduction of 12.53 hours, while improving the available
probability scores. Both ran with ten workers; hardware/load was not controlled.

**Historical decision from the five-run snapshot:** retain C search as the
reference because it improves probability scores over fixed C=1. That comparison
did not test fixed C=0.1 or C=0.01, so it did not establish that the full grid
was necessary. Its C ablations used downsampling; weighted training changes
sample count and the balance between loss and regularization.

**Current template decision, 2026-09-30:** use balanced class weights, fixed
C=0.01 with search disabled, and sigmoid calibration. The subsequent
[two-class validation](decoder-state-robustness.md) and
[regularization-path study](regularization-confidence.md) motivated this
choice through held-out probability scores, independently of target OFF duration
or M1 significance. Completed production run `006` now supports that choice.
C=0.001 has no consistent advantage across the two validation panels. These
remain exploratory comparisons on the existing cohort, not evidence of a
universal optimum; run `005` retains its searched-C configuration.

A revised selection rule, such as probability-loss-based C search, should be
evaluated on fresh holdouts with calibration inside training folds. Every null
must repeat the chosen fitting procedure: fixed C=0.01 for the current default,
or a complete new search when search is enabled. Regenerate observed, null,
and downstream outputs together when changing C policy. Do not choose C, its
scoring rule, or a seed by inspecting the target's state duration.

## Consolidated recommendation

Use the current default template for the next planned analysis: **all-trial
balanced class weights, fixed C=0.01, C search disabled, and five-fold sigmoid
calibration**. This recommendation concerns probability estimation and
computation. It does not claim that the shortest OFF intervals, largest M1
coefficients, or smallest p values identify the best decoder.

- **Retain information and define the probability target.** Exclude the test
  trial before fitting, retain the other eligible training trials, and recompute
  classifier weights within each training fold. Balance pooled calibration
  weights as well. This targets equal total cue weight; it is not an estimate of
  a population's natural cue prevalence. Keep scaling and calibration inside
  the training data.
- **Keep observed and null procedures matched.** Use the same training-trial
  identities for the observed estimate and its null fits. Refit scaling,
  classifier, and calibration after each label permutation with the same fixed
  C. Regenerate observed, null, evaluation, and state outputs together whenever
  the fitting procedure changes.
- **Prespecify seeds instead of searching for a favorable state.** Weighting
  removes random trial omission, but calibration folds and the finite null bank
  still vary with the seed. The production comparisons use one seed and cannot
  establish session-wide seed stability. If uncertainty in a particular
  maximum matters, repeat complete matched fits under predefined seeds and
  report the duration distribution. Probability averaging across seeds remains
  an optional candidate: validate the complete ensemble and reproduce that
  same procedure for its nulls before using it for states. Do not tune a seed
  or average observed confidence against an old single-fit null bank.
- **Evaluate probability quality on both classes.** Use held-out Brier/log loss,
  discrimination, and reliability with a stated cue prior and session weighting.
  Production preferred-cue-only scores are useful paired diagnostics. The
  existing two-class validation gives stronger support for the recommendation,
  but all these comparisons reuse the same cohort. A new cohort or a properly
  nested assessment is needed to estimate performance of the model-selection
  decision itself. Full-session screening remains outside the production
  trial holdouts.
- **Treat state validity and M1 prediction as separate questions.** OFF means
  low confidence relative to this fitted null and rule; there is no biological
  ON/OFF ground truth here. The default independent-per-bin shuffles do not
  preserve temporal dependence for cluster inference. Shared-time shuffles
  address within-trial consistency but alone do not establish a joint
  session-wide permutation test. Report sensitivity to the null bank and state
  rule, and evaluate cell-count prediction on held-out sessions and animals.
  Twenty-five sessions from three monkeys do not provide 25 independent
  biological replicates.

The full preset and standalone equivalents are documented under
[decoder regularization](configuration.md#decoder-regularization). Historical
saved runs and templates keep their recorded choices. Probability-loss-based
C tuning, much smaller C, and seed ensembles are future alternatives requiring
validation; none is justified solely by a more favorable OFF/M1 result.

### Evidence, reproduction, and remaining limitations

![Six completed runs: null distributions, OFF durations, calibration and C contrasts, weighting, and fitting runtime](../assets/statistical-choices-comparison.png)

The [decoder evidence snapshot](../validation/statistical-choices-evidence.json) includes
per-session and aggregate metrics, exact alignment checks, focal-bin values,
selected-C counts, and seven paired contrasts: `001` against `002`–`005`, `003`
against `004`, `002` against `004`, and `005` against `006`. The new contrast
also records nominal paired session-bootstrap summaries, per-monkey
directions, probability/label transitions, and exact C=0.01 matching anchors. It checks the expected scientific-setting
differences and identical state settings for each pair, retaining both raw
settings and normalized balancing modes. It records the actual fitting and
state-manifest IDs and SHA-256 hashes of those manifests and the selection,
decoder, and state caches. `005`/`006` additionally verify current decoder
provenance, matching primary caches against all 50 native checkpoints and
current input/source/package fingerprints. It also preserves compact summaries
and hashes of
the earlier weighting, activity, and trial-swap experiments.

The source script `scripts/next/compare_statistical_choices.py` regenerates
the completed-run evidence and figure using the existing local environment.
It is a reproduction utility for this dated six-run comparison, not a
pipeline stage or a tool that automatically selects the latest run.
Run it from the repository root:

```bash
.venv/bin/python scripts/next/compare_statistical_choices.py
```

It reads `cache/next_run_001` through `cache/next_run_006`
sequentially, with one numerical thread and no model fitting. Optional earlier
experiment summaries are read from
`cache/comparisons/run_037_vs_next_001_221024/`. If those optional local reports
are absent, their existing evidence snapshots are retained with their original
source hashes; they are not presented as newly recomputed experiments.
The command writes `docs/validation/statistical-choices-evidence.json` and
`docs/assets/statistical-choices-comparison.png`. The script checks source-file hashes
before and after each read; it does not contact the dashboard, synchronize
dependencies, stop workers, or regenerate production stages.

The separate [OFF/M1 evidence](../validation/off-m1-run-comparison.json)
records matching prepared rows and holdout indices, cached model validity,
M0/M1 summaries, relative predictive gains, and session/animal holdout results:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/next/compare_off_m1_runs.py
```

It defaults to `next_run_005` and `next_run_006`, verifies source hashes, reads
cached mixed-model results, and recomputes small session-level OLS sensitivity
fits with 5,000 paired-session bootstrap draws. It writes only
`docs/validation/off-m1-run-comparison.json`; it does not refit production
decoders or mixed models. The interval summaries remain conditional on these
25 sessions from three monkeys.

For all three decisions, the remaining limitations are shared: fixed
full-session cell selection, independently permuted null labels across time,
preferred-cue-only production evaluation, and no biological ON/OFF ground
truth. The earlier focal weighted experiment lacks a fully regenerated
session-level null pool; completed `005` and `006` supply full pools, but both
use only production seed 42. The existing
[null-time-structure option](configuration.md#null-shuffle-time-structure)
addresses within-trial permutation consistency separately; calibration, C
search, and balanced weights do not repair that issue automatically.
