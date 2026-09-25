# Configuration

Copy a preset within `configs/next/`, edit its stage settings, and pass the file
with `--settings`. A settings file is a JSON object keyed by the
[pipeline stage names](pipeline.md#stage-order).
Each stage accepts its script's dataclass field names with underscores;
standalone script CLI flags use hyphens. Unknown stage names and unknown settings
in selected stages are errors. Omitted settings use the script's defaults.
A custom settings file replaces the preset; it is not merged with it.

For example, edit this field inside the preset's existing `decode` object:

```json
{
  "decode": {
    "n_decode_shuffle": 100
  }
}
```

`decode.n_decode_shuffle` controls the number of null estimates. The example
preset uses 100, the smoke preset uses 3, and the decoder default is 100.
The pipeline runner has no `--n-decode-shuffle` flag. When running
`decoding_confidence.py` directly, use `--n-decode-shuffle 100`.

Set `--data-dir` and `--cache-dir` on the pipeline command; they are not allowed
inside stage settings. Other shared options, such as worker count and session
filters, supply defaults to applicable stages; stage-specific JSON values take
precedence. File paths in JSON are relative to the working directory; cache
subdirectory fields follow the stage-relative rules below. The presets use
shared diagnostic figure settings at `configs/diagnostic_figure_config.json`.

Preview every stage's resolved settings without running analyses:

```bash
uv run python scripts/next/pipeline.py \
  --settings configs/next/example_pipeline.json --stages all --dry-run
```

`--n-jobs` controls selection session workers, decoding trial workers, and
mixed-effects CV model workers. Start with a value suited to the available CPUs
and memory. Numerical library threads inside workers are limited to one.
Standalone mixed-effects commands expose `--cv-n-jobs`. Decoding bins activity
once per session using chunked cumulative sums and sends binned activity to fit
workers. The decoder seed controls balancing and null permutations without
creating repeated observed estimates.

PNG is the default figure format. The runner's `--figure-formats png tif eps`
enables all three formats for analyses using the shared exporter; plots that
only support PNG keep that format.

## Cache directory layout

Pass the run root to `--cache-dir` on every command, such as `cache/my_run`,
without appending a stage name. Scripts resolve their own outputs and upstream
inputs under that root. See [Outputs](outputs.md) for the complete layout.

Subdirectory overrides stay inside their owning stage. Empty strings (the
default except for criticality's `prepared_subdir`) mean the stage directory
itself. Nested relative paths are allowed; absolute paths and `..` are rejected.

| Setting | Relative to | Default |
| --- | --- | --- |
| `prepare.output_subdir`, `prepare.cv_output_subdir` | `<cache>/prepare/` | `""` |
| Model-stage `input_subdir`, `cv_input_subdir` | `<cache>/prepare/` | `""` |
| Model-stage `output_subdir` | That model stage's directory | `""` |
| `criticality.prepared_subdir` | `<cache>/criticality/` | `"prepared"` |
| `activity.output_subdir` | `<cache>/activity/`, before `figures/` | `""` |

For example, `prepare.output_subdir: "custom/features"` writes
`<cache>/prepare/custom/features/trial_table.pkl`. Set each consuming model
stage's `input_subdir` to `"custom/features"` as well. Match `cv_output_subdir`
and consuming `cv_input_subdir` separately when relocating the holdout cache.
For weighted analyses, scripts append `pev_weighted/` automatically; omit that
suffix from subdirectory settings. Output filename overrides are filenames,
not paths.

Criticality reads shared holdout features from `prepare/`, while its own
threshold-specific preparations stay under `criticality/`. Changing a model's
`output_subdir` relocates its result subtree without changing its input location.
Both supplied presets use the default directory layout.

## Screening checks

Each screening check has an independent boolean in the JSON `select` object.
The standalone CLI enables it with `--check-<name>` and disables it with
`--no-check-<name>`. Thresholds always have valid numeric ranges, even when a
check is disabled. Negative rates/ratios, correlations outside [0, 1], and
nonfinite values are errors; they never mean "off".

| CLI enable flag / JSON key | Example and script default | Parameters (CLI names) |
| --- | --- | --- |
| `--check-min-trials` / `check_min_trials` | On | `--min-trial-per-session` (320 total trials) |
| `--check-firing-rate` / `check_firing_rate` | Off | `--min-fr-test` (default 0 Hz; nonnegative), test period |
| `--check-presence-ratio` / `check_presence_ratio` | On | `--min-presence-ratio` (0.9; range [0, 1]), `--presence-start`, `--presence-end` (−400, 1400 ms) |
| `--check-delay-variance` / `check_delay_variance` | Off | `--var-ratio-threshold-delay-over-baseline` (default 1; nonnegative) |
| `--check-baseline-variance` / `check_baseline_variance` | Off | `--var-ratio-threshold-sliding-over-all` (default 0.5; nonnegative) |
| `--check-baseline-drift` / `check_baseline_drift` | On | `--temp-dep-r-threshold-baseline` (0.3; range [0, 1]), `--baseline-drift-start`, `--baseline-drift-end` (−400, 0 ms) |
| `--check-selectivity` / `check_selectivity` | On | `--sig-pev-threshold` (2.5%; range [0, 100]), `--sig-pev-duration` (100 ms), `--pev-clip-at` (0%; range [0, 100]) |
| `--check-preferred-drift` / `check_preferred_drift` | Off | `--temp-dep-r-threshold` (default 0.3; range [0, 1]), test period |

The test period uses `--t-test-start` / `--t-test-end` (500/1400 ms), and PEV
uses `--t-test-window` / `--t-test-step` (50/10 ms). Variance checks use
`--temp-check-baseline-start` / `--temp-check-baseline-end` (−500/0 ms),
`--temp-check-delay-start` / `--temp-check-delay-end` (500/1000 ms), and
`--min-trial-for-temp-check` (50 correct trials, also the sliding window length).
Window lengths, strides, and counts must be positive; variance windows require
at least two trials. End times must follow start times.

For example, explicitly enable firing-rate screening at 1 Hz and disable
baseline drift:

```bash
uv run python scripts/next/cell_screening.py \
  --data-dir data/nature --cache-dir cache/next_screening_example \
  --check-firing-rate --min-fr-test 1 \
  --no-check-baseline-drift
```

An enabled check rejects cells whose required statistic is unavailable. A
disabled check skips both rejection and applicability checks; its diagnostic
status is `disabled`. Other statuses are `pass`, `fail`, and `not_applicable`.
Extended diagnostics may still calculate descriptive measures for disabled
checks. Selection caches record the check switches and resolved selection config.

Cue/PEV metadata is still needed for decoding when selectivity rejection is
disabled: PEV and preferred cue are then summarized across all test bins, with
no threshold-run test. Selected cells in this mode are not necessarily
cue-selective, despite historical downstream group names. Zero-PEV populations
cannot use PEV-weighted means; choose equal weighting for such an analysis.
Disabling presence admits all cells to the `PASSED_PRESENCE_RATIO` decoder mode.
The stationary pool bypasses selectivity rejection but respects every other
enabled cell check, including preferred-cue drift when enabled.

Input validity, at least two correct-trial cue conditions, and positive residual
degrees of freedom for cue metadata remain required. Disabling the selection
trial-count gate does not disable decoding's separate session eligibility gate.
`min_cell_per_group` belongs to decoding; it is no longer accepted by selection,
where it previously had no effect. The unused selection `seed` and combined
`temp_dep_detection` controls were also removed.

## Resume and rerun

Every `pipeline.py` invocation that begins execution gets its own
`<cache>/manifests/<run_id>.json` record. Later full or partial runs preserve it.
`pipeline_manifest.json` is the latest-run view. See [Run manifest history](outputs.md#run-manifest-history)
for the schema, a full-to-partial example, and commands to inspect prior records.
CLI records also retain the exact Python argument vector, a shell-quoted
command, working directory, and interpreter path; see
[Reuse a recorded command](outputs.md#reuse-a-recorded-command). Standalone stage
scripts do not create runner history.

| Status | Meaning |
| --- | --- |
| `running` | Execution started; progress is saved before and after stage attempts |
| `complete` | Every requested stage returned successfully |
| `failed` | Execution stopped with an ordinary exception; the failing stage records its error |
| `interrupted` | Execution stopped with a caught keyboard interrupt or explicit Python exit |

Failures and interruptions are re-raised after recording the outcome. An
uncatchable termination, such as a forced kill, can leave the last saved status
as `running` with no finish timestamp. `running` alone therefore does not prove
that a process is still active. Requested but unattempted stages appear in
`settings` without an execution entry in `stages`. A `complete` invocation does
not establish that every statistical fit succeeded; inspect stage result tables
and fit logs as described in [Methods](methods.md#performance-and-uncertainty).

`--dry-run` and configuration errors raised before execution create no new
history record and leave the previous latest view unchanged. Existing old-format
root manifests are preserved automatically before replacement.

History has no automatic pruning and retains execution metadata only. Reruns
still replace stage outputs, including preparation manifests. Use separate run
roots to retain different analysis variants, and avoid concurrent pipelines
writing to the same caches. A partial rerun neither merges old settings into its
record nor automatically reruns downstream stages.

The runner runs every requested stage on each invocation; only decoding
automatically reuses matching per-session checkpoints. A checkpoint
is reused when its analysis settings, source data, selection cache, and code
fingerprint match. Changing the worker count does not invalidate it.

To force decoding to refit, set `"resume": false` in the JSON's `decode` object,
or pass `--no-resume` to the standalone decoder. After changing decoding, rerun
evaluation, states, and downstream analyses. After changing screening, rerun
decoding and its downstream stages. Use a fresh cache directory when comparing
analysis settings or migrating from the historical scripts.

Activity comparison and mixed-effects preparation validate cache provenance
before analysis. State fingerprints, preferred cues, trial IDs, and time bins
must match the decoding cache. Decoding fingerprints must also match the current
selection cache, session files, and implementation code. Both stages therefore
require `decode/decoding_confidence.pkl`, `select/cell_screening.pkl`, and
`states/on_off_states.pkl` under the same run root.
Stale or missing provenance stops the stage with instructions to rerun decoding
and its dependents. This also applies to caches generated before a code change.

## Useful controls

| Setting | Where to set it | Effect |
| --- | --- | --- |
| `data_dir`, `cache_dir` | Runner CLI only | Keep stage inputs and outputs aligned |
| `session_list_file` | Runner CLI or selection/decoding JSON | Filter complete sessions by file stem |
| `max_sessions_to_run` | Runner CLI or selection/decoding JSON | Cap sessions, without partitioning their trials |
| `n_jobs` | Runner CLI | Supply worker counts to applicable stages |
| `n_decode_shuffle` | `decode` JSON | Number of null estimates per trial/bin |
| `seed` | `decode` JSON | Reproduce trial balancing, null permutations, and model randomness |
| `resume` | `decode` JSON | Reuse matching per-session checkpoints |
| `cv_shuffles` | Mixed-effects stage JSON | Number of trial-holdout repetitions |

To see all available fields, run the relevant script with `--help`. The runner
accepts enum names such as `SIGMOID` and enum values such as `sigmoid` in JSON.

## Session selection

The session list is an allowlist of file stems, one per line. It is intersected
with the `.mat` files actually present in `data_dir`. For example, a list with
A, B, and C and a directory containing A and B processes A and B and warns about C.
Available files not listed are excluded. No matching files is an error.

Selection applies its session cap to the sorted candidate file list. Decoding
applies its cap after checking screening eligibility. A shared cap therefore
need not produce that many decoded sessions: some candidates may fail screening.
Every selected session is still processed as a whole session.

## Changing the null count

Edit `n_decode_shuffle` within the existing preset's `decode` object. Setting it
to zero enables observed-only decoding and evaluation. State detection requires
at least two null estimates, so run only `select decode evaluate` for an
observed-only analysis. Preserve the other preset settings when editing this
field; unspecified values revert to script defaults.
