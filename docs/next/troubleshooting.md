# Troubleshooting

## The dashboard does not open

Keep the dashboard's server terminal running and open the address it prints.
The default is `http://127.0.0.1:8000/`. The documentation preview uses port
8001 in this guide, and Vite uses port 5173 only for frontend development.
Opening one service's address does not start another service.

See [dashboard troubleshooting](dashboard.md#troubleshooting) for missing
dependencies, missing frontend builds, and connection errors. If the terminal
reports an occupied port, follow [Use another port](dashboard.md#use-another-port).
For the normal launch command, see
[Open the dashboard again](dashboard.md#open-the-dashboard-again).

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

C search requires five training trials from each class after holding out the
test trial and balancing training data. Calibration can reduce its fold count,
but still needs at least two usable folds. Inspect class counts and trial
eligibility first. If using a fixed C is appropriate for the analysis, set
`decode.grid_search_for_c` to `false`; that changes the analysis design.

## States fail with an observed-only run

State detection requires at least two finite null estimates per trial/bin.
With `decode.n_decode_shuffle` set to zero, run only `select decode evaluate`.
To detect states, increase the null count and rerun decoding before states and
downstream analyses.

## Decoder checkpoints are not reused

Reuse requires matching analysis settings, source data, selection cache contents,
and implementation code. Changes to any of these can trigger refitting; a change
to worker count alone does not. Ensure `decode.resume` is `true`. The runner
itself reruns every requested stage; checkpoint reuse occurs inside decoding.

## A cache is incompatible

Use a fresh cache directory and regenerate outputs with `scripts/next`.
Historical cache formats are not supported. Avoid mixing caches from different
screening or decoding settings in one analysis directory.

## Activity or preparation reports stale provenance

Keep `decode/decoding_confidence.pkl`, `select/cell_screening.pkl`, and
`states/on_off_states.pkl` under the same run root. Rerun `decode evaluate states` with the current selection and data, then
rerun activity and mixed-effects preparation before their dependent analyses.
If the session data or screening settings changed, rerun `select` first.
Code changes can also invalidate decoding fingerprints. Do not copy fingerprints
between caches to bypass the check.

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
