# Next pipeline validation

## Latest validation — 2026-09-26

Command capture passed **231 tests** in **6.589 seconds**: **114 next tests**
and **117 historical tests**. Four additional tests cover exact argument values
and quoting (including spaces, quotes, Unicode, and shell metacharacters),
programmatic calls without an invented CLI, successful direct/module subprocess
invocations with interpreter flags and working directories, and failed CLI
runs that retain their command reference. Existing history tests still pass.

A targeted integration run in `cache/test_run_046_next` reused a copy of the
`test_run_044_next` decoding fixture. Evaluation was invoked once as a direct
script and once through `python -X utf8 -m scripts.next.pipeline`. Both saved
the exact Python argument vector, quoted command, working directory, and
interpreter path. The first record remained byte-identical after the second
run; both latest/history pairs matched. The saved command, split back into
arguments and extended with `--dry-run`, successfully resolved the evaluation
configuration without creating another history record.

The strict MkDocs build, documented command-reference example, and whitespace
checks pass. Command capture preserves Python process arguments, not outer
launchers such as `uv run`, original shell quoting, environment assignments,
or redirections. Earlier records are not backfilled. These runner changes do
not refit or revalidate the full scientific pipeline.

## Manifest-history documentation review — 2026-09-26

The manifest-history documentation and pre-commit review passed **227 tests**
in **4.165 seconds** (110 next and 117 historical tests), a strict MkDocs build,
and `git diff --check`. The documented Python inspection example was executed
against `cache/test_run_045_next` and correctly listed the earlier `evaluate states`
invocation and the later `evaluate`-only invocation. Generated HTML contains the
new history, inspection, retention, status, and migration section anchors.

The documentation now distinguishes runner history from standalone stage
execution and replaceable preparation manifests. It specifies manifest fields,
status meanings, full-to-partial preservation, old-record archival, and the
limits of metadata retention. This review changed documentation only; the
runner integration and failure-path checks are recorded below.

## Manifest-history validation — 2026-09-26

Persistent run-manifest history passed **227 tests** in **4.168 seconds**:
**110 next tests** and **117 historical tests**. Seven new regression tests
cover complete-to-partial reruns, unique IDs at identical timestamps, recorded
failures and keyboard interruptions, byte-preserved legacy/orphaned records,
unchanged history after dry runs or invalid settings, progress before stage
execution, and atomic-write failure cleanup. Strict MkDocs and whitespace
checks also pass.

A targeted integration run in `cache/test_run_045_next` used a copy of
`test_run_044_next`'s decoding cache. It ran `evaluate states`, followed by an
`evaluate`-only invocation, using the smoke preset. Both runs completed, produced
distinct records in `manifests/`, and preserved the first record byte for byte.
The root `pipeline_manifest.json` exactly matches the second invocation's
history record. No refitting or full-pipeline rerun was needed for this runner
change; this check does not revalidate the full scientific analysis.

Each record includes runner configuration, resolved stage settings, UTC times,
overall status, and attempted-stage statuses/timings. Existing root manifests
are archived before replacement. History retains execution metadata, while
analysis outputs remain replaceable. Forced termination can leave the last
record marked `running`; records lost before this feature cannot be recovered.

## Documentation review — 2026-09-26

The documentation and pre-commit review passed **220 tests** in **4.088 seconds**
(103 next and 117 historical tests), a strict MkDocs build, and `git diff --check`.
The example preset resolves all eleven stages in a dry run. The methods were
checked against the resolved preset and implementation, including full-session
screening, training-only decoder scaling, calibration order, on/off cluster
rules, inclusive duration bins, model normalization, and holdout scoring.

Model-definition inspection confirmed **34 model-family, 5 nested-count,
6 nested-activity, 258 nine-threshold criticality, and 28 interaction models per
outcome** for the example. Methods now distinguish each stage's populations and
normalization, full-data versus held-out R², reused versus regenerated holdouts,
and descriptive activity plots versus predictive evaluation. All method output
paths follow the stage-directory layout.

This review changed documentation only. The complete `test_run_044_next`
integration results below still describe the committed analysis code; the
full-resolution example analysis was not rerun.

## Stage-directory validation — 2026-09-26

The stage-directory cache refactor passed **220 tests** in **4.108 seconds**:
**103 next tests** and **117 historical tests**. New layout tests cover stage
ownership, nested relative paths and rejected escapes, both model outcomes,
old flat-cache rejection, weighted/unweighted preparations, separate criticality
threshold caches, preparation manifest paths, and rejection-diagnostic outputs.
Existing provenance, decoding-resume, and cross-run tests use the new paths.

```bash
MPLCONFIGDIR=/tmp/wm-next-mpl .venv/bin/python -m unittest discover -s tests
MPLCONFIGDIR=/tmp/wm-next-mpl .venv/bin/python scripts/next/pipeline.py \
  --settings configs/next/smoke_pipeline.json \
  --data-dir data/example --cache-dir cache/test_run_044_next \
  --stages all --n-jobs 2
```

All eleven stages completed in **122.386 seconds** summed stage time. The run
root contains exactly eleven stage directories and `pipeline_manifest.json`;
no flat primary caches or shared `mixedlm/` directory were created. Criticality's
threshold-specific trial tables are under `criticality/prepared/active_thresholds/`,
while shared features and holdout caches are under `prepare/`.

The four primary caches each contain four sessions. Their scientific data match
`test_run_043_next` exactly, excluding configuration/provenance fields that
necessarily reflect the new paths and code. The prepared table also matches
exactly at **236 rows × 56 columns**. All **214 CV fits** succeeded and converged
with no fit errors.

A subsequent decoder invocation reused all four checkpoints without refitting.
Standalone trial inspection saved its figure under `decode/figures/inspection/`.
Cross-run comparison with a temporary copy of the evaluation cache wrote eight
plots under the two runs' `evaluate/figures/across_runs/` directories. The example
preset resolves all eleven stages in a dry run, the strict MkDocs build passes,
and the whitespace check is clean.

The smoke preset uses three null estimates, one holdout, and one criticality
threshold; this is an integration check, not a full example-preset analysis or
a controlled performance benchmark. Earlier flat caches must be regenerated
in a fresh run directory; no automatic migration or legacy-path fallback is
supported. Earlier validation entries below describe the layout at that time.

## Screening rename validation — 2026-09-26

The `cell_screening.py` rename passed **215 tests** in **4.226 seconds**.
Both direct and module-style `--help` entry points work, and the example
preset resolves all eleven stages in a dry run. The `select` stage now maps to
`cell_screening`; methods and CLI examples reference that implementation.
All next-pipeline readers and provenance checks use `cell_screening.pkl`, and
the CSV summary is `cell_screening.csv`. Historical scripts retain their names.

```bash
MPLCONFIGDIR=/tmp/wm-next-mpl .venv/bin/python scripts/next/pipeline.py \
  --settings configs/next/smoke_pipeline.json \
  --data-dir data/example --cache-dir cache/test_run_043_next \
  --stages all --n-jobs 2
```

All eleven stages completed using the renamed caches (125.349 seconds summed
stage time), with four sessions, 236 prepared rows, and 214 successful,
converged CV fits. No old-named screening outputs were created. Decoding
probabilities, predictions, C arrays, trial/time axes, state masks, duration
outcomes, and the prepared table match `test_run_042_next` exactly. Preparation
manifests reference `cell_screening.pkl`. The strict documentation build passes.
This smoke run uses three null estimates and one model holdout; its timing is
not a controlled performance benchmark.

## Explicit screening validation — 2026-09-26

The explicit-screening refactor passed **215 tests** in **4.072 seconds**:
**98 next tests** and **117 historical tests**. Its 15 focused screening tests
also passed after the final measurement-helper review.

```bash
MPLCONFIGDIR=/tmp/wm-next-mpl .venv/bin/python -m unittest discover -s tests -v
MPLCONFIGDIR=/tmp/wm-next-mpl .venv/bin/python -m unittest tests.next.test_cell_screening -v
```

The new tests cover every check's enable/disable CLI flags, independent rejection
and applicability decisions, valid cue metadata with selectivity disabled,
stationary-pool filtering, configured presence windows and correct trials,
known variance/correlation values, constant or missing measurements, parameter
validation, and removal of obsolete selection settings. Both presets resolve
with explicit booleans and no sentinel thresholds. The documented standalone
selection command resolves identically to the example preset. The strict MkDocs
build and whitespace checks pass.

The complete updated smoke pipeline passed in `cache/test_run_042_next`:

```bash
MPLCONFIGDIR=/tmp/wm-next-mpl .venv/bin/python scripts/next/pipeline.py \
  --settings configs/next/smoke_pipeline.json \
  --data-dir data/example --cache-dir cache/test_run_042_next \
  --stages all --n-jobs 2
```

All eleven stages completed (121.456 seconds summed stage time). The four
sessions retained 32, 9, 13, and 32 selected cells respectively. Selected,
stationary, and presence-passing cell IDs, selected trial IDs, PEV summaries,
and preferred cues exactly match `test_run_041_next`. Observed/null probabilities,
observed predictions, selected C arrays, trial/time axes, state masks, and
duration outcomes also match exactly. The prepared table is unchanged at 236
rows, and all 214 CV fits succeeded and converged without fit errors. The smoke
preset uses three null estimates and one model holdout; timing is not a
controlled performance benchmark.

Disabled checks now skip applicability exclusions. This can change results on
other data compared with the earlier sentinel-threshold approach; it is not a
promise of scientific equivalence for every dataset. Selection caches record
the enabled checks and complete selection configuration.

## Cache-consistency validation — 2026-09-26

The cache-consistency and diagnostics follow-up passed **200 tests** in
**4.599 seconds**: **83 next tests** and **117 historical tests**.

```bash
MPLCONFIGDIR=/tmp/wm-next-mpl .venv/bin/python -m unittest discover -s tests -v
```

Fifteen new regression tests cover:

- Correct-trial presence ratios with the same half-open screening window,
  including cells that fire only on incorrect trials.
- Valid cache provenance with serialized decoder settings and changed worker
  count; rejection of changed selection, source data, decoder settings, stale
  or missing fingerprints, missing settings, missing or duplicate decoder
  sessions, and mismatched state cues or trial/time axes.
- Both downstream entry points rejecting stale states before analysis.
- Cross-run warnings for different cues or trial sets and missing metadata,
  while allowing comparison to proceed. Reordered identical trial sets do not
  warn.

The full eleven-stage smoke pipeline completed in `cache/test_run_041_next`:

```bash
MPLCONFIGDIR=/tmp/wm-next-mpl .venv/bin/python scripts/next/pipeline.py \
  --settings configs/next/smoke_pipeline.json \
  --data-dir data/example --cache-dir cache/test_run_041_next \
  --stages all --n-jobs 2
```

All eleven manifest entries are `complete` (118.399 seconds summed stage time).
Sessions 210921, 211015, 221020, and 221024 produced 236 mixed-effects rows;
all 214 CV fits converged with no fit errors. Valid provenance passed in
activity comparison, preparation, and criticality's threshold preparation.
Observed/null probabilities, observed predictions, trial IDs, time bins,
on/off-state masks, and total/maximum off-state durations match
`test_run_040_next` exactly for all four sessions. The smoke preset uses three
null estimates and one mixed-effects holdout; this is an integration check,
not a production-scale validation or controlled performance benchmark.

The MkDocs strict build also passed. See the
[cache provenance requirements](../next/configuration.md#resume-and-rerun)
before using caches from an earlier code revision.

## Earlier eleven-stage validation — 2026-09-26

Validated with the repository's Python **3.12.12** environment. This section
records the current eleven-stage pipeline; the historical records below refer
to earlier versions, including the two removed fixed-effects analyses.

### Scope and configuration

- Five default stages: `select`, `decode`, `evaluate`, `states`, `activity`.
- Six mixed-effects stages: `prepare`, `models`, `nested-count`,
  `nested-activity`, `criticality`, `interactions`.
- JSON presets live in `configs/next`; new tests live in `tests/next`.
- Null-shuffle count belongs to `decode.n_decode_shuffle`: 100 in the example
  preset, 3 in the smoke preset, and 100 when omitted. The standalone decoder
  retains its CLI flag; the runner rejects the removed shared flag.
- Pooled-delay decoding and the standalone baseline-activity and cell-count
  duration regressions are removed. Mixed-effects baseline and cell-count
  predictors remain supported.

### Automated checks

```bash
MPLCONFIGDIR=/tmp/wm-states-mpl .venv/bin/python -m unittest discover -s tests -v
```

All **185 tests passed** in 4.138 seconds: **68 new tests** and **117 historical
tests**. Four tests specific to the removed regressions were deleted and three
runner regression tests were added, accounting for the change from 186 tests.

- Both presets dispatch the expected stages in default, mixed, and all modes.
  Tests verify execution order, manifest status, and decode null-shuffle counts.
- Removed `baseline` and `cell-count` stages are rejected by the CLI and JSON
  settings before writing outputs, even when a different stage is requested.
- All **15 remaining CLI help entry points** passed. README analysis commands
  parse, local links resolve, and the five standalone stage configurations match
  the documented one-command run.
- Both presets resolve in all/mixed modes. The runner uses explicit stage groups
  rather than positional slicing of the stage registry.
- No remaining new implementation imports either removed module. Shared activity
  weighting and mixed-effects tests remain in place. `git diff --check` passed.

### Full-session integration: `test_run_040_next`

The following command completed successfully using the updated smoke preset:

```bash
MPLCONFIGDIR=/tmp/wm-states-mpl .venv/bin/python scripts/next/pipeline.py \
  --settings configs/next/smoke_pipeline.json \
  --data-dir data/example --cache-dir cache/test_run_040_next \
  --stages all --n-jobs 2
```

The saved `cache/test_run_040_next/pipeline_manifest.json` contains exactly the
eleven supported stages, all marked `complete`. No `fixedlm` directory was
created.

| Result | Verified value |
| --- | --- |
| Full sessions | 210921, 211015, 221020, 221024 |
| Preferred-cue decoding trials | 240 |
| Time bins per trial | 5, with a 400 ms stride |
| Estimates per trial/bin | 1 observed and 3 null |
| Decoder | Logistic regression, C search, sigmoid calibration |
| Prepared mixed-effects rows | 236 |
| Mixed-effects outcomes | Total and maximum contiguous off-state duration |
| CV fits | 214 successful, 214 converged, 0 fit errors |

Observed/null probabilities, predictions, selected C values, on/off-state masks,
and off-state durations match `test_run_038_next` exactly for all four sessions.
These comparisons confirm that removing the regressions did not change upstream
results.

Stage timings from the saved manifest:

| Stage | Seconds |
| --- | ---: |
| select | 13.663 |
| decode | 45.133 |
| evaluate | 0.003 |
| states | 2.104 |
| activity | 10.632 |
| prepare | 0.713 |
| models | 28.832 |
| nested-count | 3.130 |
| nested-activity | 4.118 |
| criticality | 6.706 |
| interactions | 2.805 |
| **Sum of stage timings** | **117.839** |

The smoke preset uses one mixed-effects holdout, one active-cell threshold, and
at most 30 fitting iterations. These timings describe one local run, not a
controlled performance benchmark. The full 100-null-shuffle/50-holdout production
analysis was not run. CV convergence counts do not assert convergence of every
non-CV fit.

## Documentation site — 2026-09-26

The validation record moved to `docs/validation/next.md`, and pipeline guides
were organized under `docs/next`. The README links to these guides.

- MkDocs 1.6.1 and Material for MkDocs 9.7.7 are locked in the optional
  `docs` dependency group. Existing analysis-package versions are unchanged.
- `uv run --group docs --locked mkdocs build --strict` builds all eleven
  documentation pages successfully, with internal link and anchor validation.
- `uv lock --check --offline` passes.
- All local Markdown links resolve. Twenty analysis command examples parse,
  runner examples resolve through dry runs, and the five standalone stage
  configurations match the documented default pipeline.
- The local preview renders the Material theme. Browser verification confirms
  search results for checkpoints and navigation to the configuration guide.

This documentation update did not rerun the scientific analysis or the unit
suite; the analysis results above remain the last completed validation run.

## Population ISI removal — 2026-09-26

Population ISI was removed from `scripts/next` entirely. It is no longer an
optional analysis. The outputs guide no longer lists it, and the migration guide
records its removal. Historical validation notes are retained as historical only.

- The removed `scripts.next.population_isi` module cannot be imported.
- No implementation, preset, or test under the next directories references it.
- All 185 tests pass, including 68 new-pipeline tests and 117 historical tests.
- All 15 remaining CLI help entry points pass. Both presets resolve all eleven
  runner stages; population ISI was never part of that stage registry.
- The documentation site builds successfully with `mkdocs build --strict`.

The eleven-stage scientific integration was not rerun for this removal; the
`test_run_040_next` results above remain the last completed integration run.

## Historical validation records

The commands below document earlier runs; current presets and scripts reflect
subsequent removals. Historical counts are not the current test or stage counts.

### Original automated checks — 2026-09-25

- 69 new tests passed (now located under `tests/next`).
- All 117 existing tests directly under `tests` passed.
- All 18 CLI help entry points and the module-style pipeline dry run passed.
- `uv lock --check --offline` passed. `uv run --offline --locked python
  scripts/next/pipeline.py --help` successfully built the local project and ran.
- New implementation modules do not import the original scripts.
- Screening matched the original full-session implementation on all four example
  sessions: selected/stationary/presence-passing cells, trial IDs, cue groups,
  preferred cues, and PEV values (relative tolerance 1e-10).
- Optimized C search is tested against scikit-learn GridSearchCV for logistic
  regression and linear/RBF SVM, with observed and shuffled labels.
- Tests cover held-out-trial isolation, grouped calibration/C search,
  observed/null shapes, shuffle-prefix invariance, serial/parallel
  agreement, incompatible caches, interrupted writes, checkpoint invalidation,
  equivalent JSON/CLI numeric settings, zero-null evaluation, zero-variance state
  handling, and activity analysis without detected off-states.

### Original full-session integration — 2026-09-25

```bash
uv run python scripts/next/pipeline.py \
  --settings configs/next/smoke_pipeline.json \
  --data-dir data/example --cache-dir cache/test_run_038_next \
  --stages all --n-jobs 2
```

All 13 stages completed for sessions 210921, 211015, 221020, and 221024.
No session was partitioned or shortened. Screening used the example thresholds;
decoding used five bins with a 400 ms stride, three independent null estimates
per bin, per-fit C search, and sigmoid calibration. There were 240 preferred-cue
decoding trials. Mixed-effects preparation produced 236 model rows and analyzed
both outcomes.
All 214 CV fits succeeded and reported convergence.

| Stage | Seconds |
| --- | ---: |
| select | 13.861 |
| decode | 43.748 |
| evaluate | 0.003 |
| states | 2.119 |
| activity | 10.659 |
| baseline | 52.326 |
| cell-count | 1.037 |
| prepare | 0.731 |
| models | 29.072 |
| nested-count | 3.103 |
| nested-activity | 4.132 |
| criticality | 6.702 |
| interactions | 2.833 |

Timings are from one local smoke run, not a controlled performance benchmark.
The mixed-effects smoke preset uses one holdout, one active-cell threshold,
and a 30-iteration fitting budget. The full 100-null-shuffle/50-holdout production
analysis was not run.

The final configuration normalization fix was checked by refitting and verifying
bitwise-identical observed/null probabilities and C arrays. State detection and
evaluation were refreshed; all state masks and trial outcomes were unchanged.
A subsequent module-style decoding run with one worker reused all four session
checkpoints produced with two workers. Cached settings use JSON-compatible
values, and no decoding repeat fields remain.

Additional successful checks:

- Inspection with observed confidence, null ranges, and state masks.
- Full-session rejection diagnostics and histograms in `test_run_037_next`.
- Population ISI analysis with two trial shuffles (historical check; this analysis
  has since been removed from `scripts/next`).
- A fixed-C, uncalibrated, zero-null run in `test_run_039_next`, followed by cross-run
  comparison against `test_run_038_next`, including missing-null handling.

Generated cache directories are ignored by Git. Intermediate scratch runs
034–036 were removed; historical `cache/run_*` directories were not changed.

### Follow-up validation — 2026-09-26

Tests were moved to `tests/next`, and pooled-delay decoding was removed.
All 186 tests passed through discovery from `tests`, including the 69 new tests.
The decoder regression checks one sample per training trial and activity from
only the current bin for every observed and null fit, including delay bins.
Direct and module-style decoder CLIs omit and reject the removed pooling flag;
JSON settings reject it too.

A deterministic three-bin fixture with two null estimates, C search, and sigmoid
calibration produced bitwise-identical probabilities, predictions, and selected
C arrays before and after removal of pooling. The 13-stage integration run above
used per-bin decoding; it was not repeated for this follow-up.
