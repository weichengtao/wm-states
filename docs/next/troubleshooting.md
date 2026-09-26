# Troubleshooting

## The dashboard does not open

Keep the dashboard's server terminal running and open the address it prints.
The default is `http://127.0.0.1:8000/`, with the pipeline guide at `/docs/` on
that same server. A separate documentation development preview uses port 8001,
and Vite uses port 5173 only for frontend development.
Opening one service's address does not start another service.

See [dashboard troubleshooting](dashboard.md#troubleshooting) for missing
dependencies, missing frontend builds, and connection errors. If the terminal
reports an occupied port, follow [Use another port](dashboard.md#use-another-port).
For the normal launch command, see
[Open the dashboard again](dashboard.md#open-the-dashboard-again).

## Dashboard help or documentation is unavailable

The **Help** panel's short guidance is part of the dashboard build. Full guide
links need the MkDocs build in `site/`. Run
`uv run --group dashboard --group docs --locked mkdocs build --strict` from the
repository root, then refresh the guide; restarting the backend is unnecessary.
A missing build returns an explanatory page with status 503; a missing page in
a built guide returns 404. Neither falls back to the dashboard.

If a Methods link points to old content, rebuild both with the launcher's
`--build` option after stopping the server when no analysis is running. If
`VITE_DOCS_BASE_URL` was set during the frontend build, its links use that
external guide; remove the setting and rebuild to restore local `/docs/` links.
The developer API reference moved from `/docs` to `/api/docs`.

## An existing run does not appear in the dashboard

The viewer discovers supported runs in named directories directly under this
repository's `cache/`. It does not automatically import arbitrary directories
or historical flat caches. Follow
[Find existing runs](dashboard.md#find-existing-runs) to check the location and
output layout before rerunning an analysis.

## Sessions are missing or skipped

The scripts process the intersection of session-list IDs and files in the data
directory. IDs use file stems without `.mat`. Missing listed files produce
warnings; an empty intersection is an error. A session can also fail screening
or decoding eligibility thresholds. Check those messages before increasing a
session cap. See [Session selection](configuration.md#session-selection).

## Screening reports an undefined preferred cue

Opposing or symmetric preferences across test bins can have a numerically zero
circular resultant, so there is no defined circular-mean cue. Screening stops
if such a cell otherwise passes selection. Inspect the listed cells' cue
responses and the configured selectivity window; do not substitute an arbitrary
cue. If the affected cells were already rejected by enabled checks, screening
warns and continues with their cue metadata unavailable.

## A stage cannot find its input cache

The runner does not schedule missing prerequisites. Run stages in the
[documented order](pipeline.md) and use the same data and cache directories.
For mixed-effects analyses, run `prepare` before fitting models. Pass the run
root to `--cache-dir`, without appending `select`, `decode`, or another stage
name. Caches from the earlier flat or shared `mixedlm/` layout must be
regenerated; see [Migration](migration.md). If you customize prepared-data
subdirectories, match the producing and consuming settings.

## The runner rejects settings or a stage name

JSON fields use underscores and must be inside the appropriate stage object.
Set input and cache directories on the runner command. Use `decode.n_decode_shuffle`
for the null count. Old `baseline` and `cell-count` stages, partition settings,
pooled-delay decoding, and decoder repeats are unsupported.

Start from a current preset and use `--stages all --dry-run` to validate every
stage's settings before running. See [Migration](migration.md).

## Decoding cannot construct the requested CV folds

C search requires at least **six correct preferred-cue trials and five correct
opposite-cue trials** before fitting. Each preferred-trial holdout must leave
five training trials from each class, including after training-class balancing.
Session preflight reports the actual counts before launching workers.
Calibration can reduce its fold count with a warning, but still needs at least
two usable folds. Inspect class counts and trial eligibility first. If using a
fixed C is appropriate for the analysis, set
`decode.grid_search_for_c` to `false`; that changes the analysis design.

## Observed evaluation accuracy differs from native decoder accuracy

Evaluation uses `p >= 0.5` for both observed and null accuracy. A classifier's
native prediction can disagree with that threshold, particularly for SVM
probabilities. A warning reports the number of disagreements. The evaluator
does not overwrite native predictions or their native accuracy in the decoding
cache. Compare probability-threshold evaluation metrics with each other;
use the native cached values only when that decision rule is intended.

## States fail with an observed-only run

State detection requires at least two finite null estimates per trial/bin.
With `decode.n_decode_shuffle` set to zero, run only `select decode evaluate`.
To detect states, increase the null count and rerun decoding before states and
downstream analyses.

## States warn about independent null permutations

The default `decode.preserve_null_time_structure=false` independently permutes
training labels at each bin. Cluster correction warns because those
permutations do not preserve the assignment through time. To use one assignment
across bins for each held-out trial/shuffle, set the boolean to `true` in
pipeline JSON or the dashboard's decoding settings, regenerate decoding, and
rerun states and downstream stages. Standalone decoder flags are
`--preserve-null-time-structure` and `--no-preserve-null-time-structure`; the
runner itself uses JSON. See [the policy comparison](configuration.md#null-shuffle-time-structure).

Both modes shuffle after the split and training-trial balancing. The enabled
option still permutes independently between held-out trials and retains
full-session screening; it does not establish a joint session-wide or
selection-corrected permutation test.

## States warn about very few null estimates or zero variance

The smoke preset's three null estimates test integration only. Under the
current standardization, they cannot exceed the default on-state z threshold,
so the null on-cluster cutoff is zero. Other small counts may provide poor
precision for the selected cluster tails. Increase `decode.n_decode_shuffle`
and regenerate decoding and dependent outputs for analysis.

Bins with zero null variance cannot be standardized. The stage warns and leaves
them unclassified in both masks; it does not label them off by default.
Inspect the confidence/null plots and the number of affected bins.

## States reject invalid thresholds, probabilities, or missing null clusters

On/off candidate thresholds must not overlap (`z_threshold_off` must not
exceed `z_threshold_on`). Observed and null probabilities must be finite and
within [0, 1], time bins must align with the confidence arrays and have uniform
positive spacing, and at least one bin must start within [500, 1400] ms.
Repair the configuration or regenerate invalid upstream caches before retrying.

If observed off candidates require correction but no null off clusters survive
the size filter, there is no usable reference distribution. The stage raises an
error instead of fabricating a zero cutoff. Inspect the null count, off
threshold, and minimum cluster size; revise them only as an explicit analysis
choice. A session with no observed off candidates can still return an empty
off-state mask.

## Decoder checkpoints are not reused

Reuse requires matching analysis settings, source data, selection cache contents,
and implementation code. Changes to any of these can trigger refitting; a change
to worker count alone does not. Ensure `decode.resume` is `true`. The runner
itself reruns every requested stage; checkpoint reuse occurs inside decoding.
Changing `preserve_null_time_structure` also invalidates reuse. Its value,
resolved settings, and `null_policy` are retained in decoder caches; use a
separate run directory when comparing the two policies.

## A cache is incompatible

Use a fresh cache directory and regenerate outputs with `scripts/next`.
Historical cache formats are not supported. Avoid mixing caches from different
screening or decoding settings in one analysis directory.

## Mixed-model inference is withheld or models fail

Read the model stage's `tables/` and `logs/` for each outcome. `fit_error`
identifies hard failures, such as rank-deficient predictors or nonconvergence.
When point estimates remain usable but the final Hessian, covariance, or standard
errors cannot support inference, `inference_valid=false` and `inference_error`
explain why p-values and confidence intervals are missing. These are not
nonsignificant results. Check session counts, constant/collinear predictors,
and optimizer warnings before changing the model or fitting budget.

Invalid nested comparisons similarly record `likelihood_ratio_error`; a
materially negative improvement is not silently converted to p=1. Individual
failed models do not discard independent valid fits. If every model or every CV
fit fails, diagnostics are saved and the stage raises an error.

Models missing successful CV folds warn and are excluded from rankings, while
remaining in the raw and summary tables. Inspect `rank_eligible`,
`rank_exclusion_reason`, and requested/successful/failed fit counts. Usable point
fits with withheld coefficient inference may still be assessed for prediction.

## Activity or preparation reports stale provenance

Keep `decode/decoding_confidence.pkl`, `select/cell_screening.pkl`, and
`states/on_off_states.pkl` under the same run root. Rerun `decode evaluate states` with the current selection and data, then
rerun activity and mixed-effects preparation before their dependent analyses.
If the session data or screening settings changed, rerun `select` first.
Code changes can also invalidate decoding fingerprints. Do not copy fingerprints
between caches to bypass the check.

## Activity warns that no off-state points are available

A valid session can have no cached off-state bins during the delay. Activity
plots continue without maximum-off-state points, and PCA still fits the
balanced cue groups with an empty projection for the absent points. Inspect
the states output and thresholds if the absence is unexpected. Invalid mask
values, nonfinite time bins, and non-increasing time bins are errors, not empty
state populations.

## Preparation warns about empty cell populations

The warning names the session and genuinely absent groups. Preparation retains
the established zero means, active fractions, and histories for those groups,
but those values denote absence, not measured activity. Inspect screening and
group counts before interpreting a model that uses them. Warnings occur at
session preparation, rather than for every period, cell, model row, or CV fold.
Populated groups whose measured activity is zero do not trigger this warning.

## The latest manifest lists only a partial run

`pipeline_manifest.json` describes the latest runner invocation. Find earlier
full and partial records under `manifests/`; see [Run manifest history](outputs.md#run-manifest-history).
An existing pre-history manifest is preserved as `prior-<content hash>.json` when
the new runner first replaces it. Already-overwritten records cannot be restored.
Standalone stage commands do not add history; run them through
`pipeline.py --stages <stage>` to record future invocations.

## A manifest still says running after a process stopped

A forced termination can prevent the final status write. Check whether the
process is active before rerunning. The history record preserves the last saved
stage state, and the next invocation gets a new record. History does not prove
that stage output files are complete; rerun the affected stage and its dependents
as needed. Decoder checkpoint reuse follows the usual provenance rules.

## Mixed-effects fits fail or do not converge

Inspect the result tables, fit warnings, and CV fit errors. A successful command
can contain failed or nonconverged statistical fits. Small smoke datasets can
be rank-deficient, and the smoke preset deliberately limits optimizer iterations.
Match preparation and analysis settings for weighting, history, and CV splits
before interpreting model comparisons. See [Validation](../validation/next.md)
for what was tested.

## Runtime or memory use is too high

Reduce `--n-jobs` to lower concurrent work. Use the smoke preset to check
integration on a small set of complete sessions. Coarser time bins, fewer null
estimates, or fewer mixed-effects holdouts reduce work but also change the
analysis, so use a separate cache and record those settings.
