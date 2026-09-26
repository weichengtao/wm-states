# Selection and decoding

For the statistical definitions and step-by-step procedures, see the methods
for [screening](methods.md#select), [decoding](methods.md#decode),
[evaluation](methods.md#evaluate), [state detection](methods.md#states), and
[activity comparison](methods.md#activity).

- Screening always uses a full session. Each selection result contains
  `num_trials` and one set of selected, stationary, and presence-passing cells.
  There are no session partitions or leave-one-out cell-selection variants.
- Selection uses correct trials for PEV and presence filtering. The example
  explicitly enables presence, baseline drift, and PEV checks, and disables
  firing-rate, variance-ratio, and preferred-cue drift checks. Enabled checks
  reject unavailable statistics; disabled checks do not reject cells or run
  applicability tests. Each check has its own CLI switch and JSON boolean.
  Opposing or symmetric bin preferences can leave the circular-mean preferred
  cue undefined. This is an error for otherwise selected cells; already-rejected
  cells with unavailable cue metadata produce a warning.
- Extended diagnostic CSVs use the same correct-trial population and configured
  presence window (the example uses [−400, 1400) ms). Their activity traces and additional
  baseline Spearman correlations still cover all trials. Diagnostic figure
  targets and caps affect plots only, with no change to screening or CSV rows; see
  [Diagnostic tools](outputs.md#diagnostic-tools).
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
- Observed and null fits use the same selected training trials, balanced by
  default. With `decode.preserve_null_time_structure=false` (the default and
  example), each null estimate independently permutes training-trial labels for
  each bin. With `true`, one permutation is reused across all bins for each
  held-out trial and shuffle. Both policies operate **after** the outer split
  and training-trial selection/balancing. Every observed and null fit uses only
  the current time bin, with one sample per training trial. Pooled-delay decoding
  and cell-wise label-preserving shuffles are not supported.
- `--grid-search-for-c` selects among C=(1, 0.1, 0.01) using balanced accuracy
  and exactly five source-trial-grouped folds for every distinct observed/null
  training problem. Calibration uses the selected C and grouped training-only
  folds. Calibration may reduce its fold count when necessary; C search requires
  five source-trial groups containing each class. Fold scaling is reused across C
  candidates without using validation data. Without search, `--classifier-c` is used directly.
  Preflight checks require at least six correct preferred-cue trials and five
  correct opposite-cue trials with C search enabled, so five of each remain
  after the preferred trial is held out. Calibration fold reductions warn.
- Unknown and ineligible sessions produce warnings. `--max-sessions-to-run`
  on the runner caps selection as well as decoding. Selection caps the candidate
  file list; decoding caps eligible sessions after screening.
- Evaluation does not refit models. It reports Brier score, natural-log loss,
  accuracy, confidence, and valid counts for observed data and individual null
  shuffles. Both accuracies use `p >= 0.5`; native classifier predictions are
  retained in the decoding cache, with a warning if they disagree. These scores
  concern preferred-cue test trials only.
- State detection retains the existing cluster-correction methods and total/
  maximum contiguous delay-duration outcomes. Bins with zero null variance are
  unclassified and produce a warning. Activity plots, including PCA, support
  sessions with no detected off-states and warn that maximum-off-state points
  are unavailable.

## Choosing null permutations

Set `decode.preserve_null_time_structure` in pipeline JSON or the dashboard's
decoding controls. The standalone decoder exposes
`--preserve-null-time-structure` and `--no-preserve-null-time-structure`.
See the [comparison table and complete command examples](configuration.md#null-shuffle-time-structure).

When enabled, both the permuted training labels and their inner C-search and
calibration folds are reused across a held-out trial's time bins. Each bin still
has its own fitted scaler, selected C, classifier, and calibration. Different
held-out trials use independent permutations, so this is not a joint
session-wide shuffle or a correction for full-session cell selection.

## Interpreting the estimates

Each decoding row corresponds to one held-out preferred-cue trial; `trial_idx`
maps that row back to the source recording. Confidence is the predicted
probability of the preferred cue. Both preferred- and opposite-cue trials can
contribute to training, but reported test scores concern preferred-cue trials.

Observed confidence has axes `(trial, bin)`. Null confidence adds a final
shuffle axis `(trial, bin, N)`. Each fit uses training activity from the same
bin as the test activity. Time bins are not pooled into a delay-wide model.

With the seed and other settings fixed, increasing N extends the existing null
prefix under either policy without creating extra observed estimates. Changing
only the time-structure option preserves the observed calculation but changes
the null policy. Decoder caches record the option, resolved configuration, and
`null_policy`; toggling it invalidates checkpoint reuse and requires downstream
regeneration. Mixed-effects `cv_shuffles` is a
separate trial-holdout setting; see [Mixed-effects analyses](mixed-effects.md).

## State outcomes

State detection compares observed confidence to its per-bin null distribution,
then applies the configured cluster-correction rules. Total off-state duration
and maximum contiguous off-state duration are distinct outcomes. Bins with zero
null variance are unclassified, and a session can have no detected off-states.
Cluster correction warns if decoding used independent per-bin permutations;
the shared-across-time option addresses that within-trial assignment issue,
without establishing session-wide permutation inference.

Very small null counts produce warnings: the three-shuffle smoke preset is for
integration checks, not scientific interpretation of cluster cutoffs. Invalid
probabilities, overlapping on/off candidate thresholds, missing delay bins, and
the absence of usable null off-cluster masses when observed off candidates need
correction are errors. The state stage does not invent a zero off-cluster cutoff
when the required null distribution is unavailable.

Use [inspection plots](outputs.md#inspect-and-compare-results) to examine
confidence, null ranges, and the resulting state masks together.
