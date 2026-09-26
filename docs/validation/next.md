# Next pipeline validation

## Dashboard visual and workflow refinement — 2026-09-26

The dashboard passed **40 frontend tests**, TypeScript checking, the production
Vite build, and Prettier verification. The guide passed a strict MkDocs build.
The changes are confined to the frontend and documentation: analysis settings,
Python code, cache formats, and dependency versions are unchanged. No scientific
integration rerun was needed for this change.

```bash
npm --prefix dashboard test
npm --prefix dashboard run build
npm --prefix dashboard run format:check
MPLCONFIGDIR=/tmp/wm-states-matplotlib .venv/bin/python -m mkdocs build --strict
git diff --check
```

Browser review used existing local results at **1440 × 960** and **390 × 844**:

- Reviewed the new button, input, navigation, card, comparison, and help styling.
  The mobile library, configuration, comparison, and help views had no horizontal
  page overflow. Mobile navigation locks background scrolling, keeps keyboard
  focus inside the menu, closes with Escape, and restores focus to its opener.
- Changed a screening threshold, filtered to changed parameters, restored its
  example value, switched presets and undid the switch. Parameter search accepts
  underscores. A draft run name survived navigation to another workspace.
- Checked run search, status filtering, reset, and keyboard navigation between
  result tabs. Refresh now reloads selected session metrics as well as run details.
- Compared recordings across runs and verified **Match session A**. Regression
  tests cover comparison defaults when requests resolve in either order and a
  selected status whose last matching run disappears after refresh.
- Inspected a saved job log and verified an empty search disables copy/download
  and pauses following. Tests cover terminal-state controls, stage progress,
  and empty-log exports. No analysis jobs were started or stopped for UI review.
- No browser console warnings or errors appeared during the checked workflows.

The production assets were rebuilt for the existing local dashboard server.
The guide documents change indicators, reset/undo, run filters, comparisons,
buffered-log exports, and the stop-run confirmation.

## Methods references and cross-surface alignment — 2026-09-26

The documentation/help refinement passed **409 Python tests** in **11.022
seconds**: **292 next tests** and **117 historical tests**. The dashboard passed
**30 frontend tests**, TypeScript checking, a production Vite build, and
Prettier verification. The guide builds with MkDocs strict validation; the
Python suite also checks rendered guide links, anchors, search assets, and
canonical URLs under both `/docs/` and a GitHub Pages-style project prefix.

```bash
MPLCONFIGDIR=/tmp/wm-states-matplotlib .venv/bin/python -m unittest discover -s tests -v
npm --prefix dashboard test
npm --prefix dashboard run build
npm --prefix dashboard run format:check
MPLCONFIGDIR=/tmp/wm-states-matplotlib .venv/bin/python -m mkdocs build --strict
git diff --check
```

### Alignment and reference review

- Reviewed all eleven stage methods against their implementations and resolved
  example settings. References explain effect sizes, regularization,
  calibration, scoring, permutations, PCA, mixed models, R², threshold selection,
  interactions, and figure export. Project-specific thresholds and state rules
  are distinguished from the guarantees of the cited methods.
- Parsed the README pipeline command through the actual CLI configuration and
  confirmed its five-stage default. Parsed all eleven documented standalone
  stage commands and compared their resolved analysis settings with the example
  pipeline; they match, allowing the explicitly different worker counts.
- Resolved the dashboard's example request with the same shared inputs and
  confirmed it matches the eleven-stage CLI configuration. README and help now
  explicitly distinguish the dashboard's initial eleven stages from the CLI's
  default five and use the actual **Core stages** button label.
- Verified the table-specific inference/likelihood-ratio field names, clarified
  the session-wide longest-OFF activity highlight, and documented Gaussian
  assumptions and conditional validation limits.
- Reviewed primary papers and official API pages for the new external links.
  scikit-learn URLs target the supported 1.8 series. Build/link tests validate
  local destinations; they do not monitor third-party sites for future changes.
- Added a frontend regression check that external statistical references keep
  their original HTTPS URLs when the guide uses a GitHub Pages base, with
  accessible new-tab links and `noopener noreferrer`.

### Visual review and scope

Expanded help topics and reference links were reviewed at **1440 × 960** and
**390 × 844**. Text and controls remained readable without clipping; help searches
for calibration and mixed models found the expected topics. The temporary
preview server was stopped after review. No analysis jobs were launched.

An executable-syntax comparison confirmed that the three edited analysis
scripts differ only in comments/docstrings. No analysis defaults, numerical
algorithms, cache schemas, or dependency versions changed, so no new scientific
integration run was performed. The earlier four-session validation below
remains the integration record, not evidence that these references validate
the custom statistical choices.

**Checkpoint consequence:** the existing provenance policy hashes complete
analysis-script files, including comments. Improved decoder/state CLI help and
the mixed-model docstring therefore invalidate older decoding fingerprints.
Before continuing an older run, regenerate `decode evaluate states` and any
dependent activity/model outputs as described in
[checkpoint reuse](../next/configuration.md#resume-and-rerun). Selection need
not be repeated when its inputs and settings are unchanged. Existing saved
figures remain viewable. The fingerprint policy was not weakened for this update.

## Integrated dashboard guide and help — 2026-09-26

The combined changes passed **409 Python tests** in **10.346 seconds**:
**292 next tests** and **117 historical tests**. The frontend passed **29 tests**,
TypeScript checking, a production Vite build, and Prettier verification. The
integrated launcher's `--build` completed both the frontend and strict MkDocs
builds before starting the local server.

```bash
MPLCONFIGDIR=/tmp/wm-states-matplotlib .venv/bin/python -m unittest discover -s tests -v
npm --prefix dashboard test
npm --prefix dashboard run format:check
MPLCONFIGDIR=/tmp/wm-states-matplotlib .venv/bin/python -m scripts.next.dashboard --build --port 8020
git diff --check
```

### Routing and documentation portability

- Backend tests cover `/docs` redirects, nested pages/assets, HEAD requests,
  builds completed after startup, true documentation 404s, and symlink escapes.
  Missing builds return a useful HTML 503 for browsers and JSON for API clients.
- Swagger, ReDoc, and OpenAPI remain accessible at `/api/docs`, `/api/redoc`,
  and `/api/openapi.json`; the guide does not shadow the API or fall through to
  the React interface.
- Build-helper tests cover missing tools, installation of absent frontend
  dependencies, the active Python interpreter, and build failure before listen.
- Four integration tests build the actual MkDocs Material site under both
  `/docs/` and a GitHub Pages-style `/wm-states/` URL prefix. They check every
  generated internal navigation/asset target, configured canonical URLs,
  search worker/index/result locations, all dashboard help destinations, and
  methods coverage for all eleven stages.
- Frontend tests cover local/external documentation bases, repository subpaths,
  invalid-base fallback, stage/parameter/troubleshooting mapping, help search,
  and accessible links and help controls.

### Browser review

An isolated local server on port 8020 served the production dashboard and guide.
The review checked contextual help, searching by `preserve_null_time_structure`,
expanded explanations, an empty search result, and the full-guide link. Guide
links opened a separate tab at the intended anchor. Escape closed the panel,
returned focus to Help, and retained an edited run name. The integrated MkDocs
guide displayed its local dashboard shortcut and returned search results.

The help panel was visually checked at **390 × 844** and **1440 × 960**. The
narrow panel filled the viewport without horizontal overflow; the desktop panel
remained beside the visible workspace. Both retained a reachable close control,
scrolling content, and the full-guide action.

No new scientific analysis was needed for these navigation changes. The
four-session, all-stage null-policy integration and its scientific limitations
are recorded below. GitHub Pages itself was not deployed; portability is
validated against generated HTML and URL prefixes. Generated sites, dependencies,
and test outputs remain outside Git.

## Null time structure and statistical edge cases — 2026-09-26

The final code passed **394 Python tests** in **9.653 seconds**: **277 next
tests** and **117 historical tests**. The example preset resolves all eleven
stages in a dry run; the MkDocs documentation passes a strict build.

```bash
MPLCONFIGDIR=/tmp/wm-states-matplotlib .venv/bin/python -m unittest discover -s tests -v
MPLCONFIGDIR=/tmp/wm-states-matplotlib .venv/bin/python -m mkdocs build --strict
```

### New coverage

- Default-independent and optional shared-across-time null labels, including
  identical-bin data, training-trial exclusion, balancing before permutation,
  C search/calibration, shuffle-prefix stability, and parallel reproducibility.
- Standalone enable/disable flags, default-false preset/dashboard settings,
  checkpoint invalidation, cached policy metadata, and cross-run policy warnings.
- Decoder class-count preflight and explicit calibration-fold reduction;
  consistent probability-threshold accuracy when native SVM predictions disagree.
- Undefined circular preferences: antipodal/symmetric directions, valid weak
  resultants and wraparound, selected-cell errors, and rejected-cell warnings.
- Empty OFF-state PCA projections, empty-population preparation warnings, and
  malformed state/time inputs.
- Constant float64 nulls such as `0.1` remain unclassified despite mean-rounding
  artifacts; small null counts warn, including two-tailed tail probabilities.
  Missing OFF-cluster reference distributions and overlapping candidate
  thresholds raise errors.
- Invalid final model inference withholds uncertainty estimates while preserving
  usable point fits; rank-deficient models fail explicitly. Materially negative
  nested likelihood improvements are rejected; numerical roundoff warns.
- Partial failures retain diagnostic rows. Incomplete CV fits are excluded from
  rankings, no-valid-pair contrast reports export successfully, and entirely
  failed analyses save diagnostics before raising errors.

### Four-session integration

`cache/test_run_055_next_null_time_structure` runs all eleven stages on the four
`data/example` sessions. It uses a copy of the smoke preset with these decode
overrides:

```json
{
  "preserve_null_time_structure": true,
  "n_decode_shuffle": 4
}
```

The full local settings file is
`configs/next/.dashboard/validation_null_time_structure.json`. To reproduce it,
copy `configs/next/smoke_pipeline.json` there and change the two fields above
inside `decode`. All other smoke settings remain unchanged, including the
400 ms decoding stride, one model holdout, and reduced fitting budget.

```bash
MPLCONFIGDIR=/tmp/wm-states-matplotlib .venv/bin/python scripts/next/pipeline.py \
  --settings configs/next/.dashboard/validation_null_time_structure.json \
  --data-dir data/example \
  --cache-dir cache/test_run_055_next_null_time_structure \
  --stages all --n-jobs 2
```

All eleven stages completed. The final invocation reran the complete pipeline
after the last report-only edge-case fix; the preceding invocation remains in
manifest history. Results include **240 preferred-cue decoding trials**,
**236 × 56 prepared trial features**, **214 successful full-data model fits**,
**214 successful CV fits**, and **270 PNG figures**. Each decoding cache has
five time bins and four null estimates and records the enabled time-structure
policy.

Across all four sessions, observed probabilities, native predictions, selected
C values, trial IDs, cell IDs, and time grids match run 054 exactly. The first
bin's first three null estimates also match: the option changes how a shuffle
is shared across time, while preserving that RNG prefix. Both modes receive
focused regression coverage; the supplied example and smoke presets remain
default-independent.

Invalid-inference model rows retain point estimates but contain no reported
Wald p-values or significance claims. In each general-model outcome, 33 of 34
fits are flagged for unavailable inference, demonstrating why convergence
alone is insufficient. The coefficient-forest display was visually checked:
unavailable inference uses gray markers and an explicit legend instead of
appearing as a nonsignificant result.

This is an integration check, not statistical validation of null error rates,
state durations, or model hypotheses. Four null estimates provide very little
tail precision, and a four-session model fit does not establish population
inference. The shared-across-time option preserves permutations within each
held-out trial; it does not establish a joint session-wide permutation test.

## Shared downstream contracts and activity modules — 2026-09-26

The downstream refactor passed **345 Python tests** in **8.978 seconds**:
**228 next tests** and **117 historical tests**. This includes the existing
dashboard coverage. The full example preset also passes the runner's dry run.

```bash
MPLCONFIGDIR=/tmp/wm-next-mpl .venv/bin/python -m unittest discover -s tests -v

MPLCONFIGDIR=/tmp/wm-next-mpl .venv/bin/python scripts/next/pipeline.py \
  --settings configs/next/example_pipeline.json \
  --data-dir data/example \
  --cache-dir cache/test_run_054_next_downstream_refactor \
  --stages all --dry-run
```

Coverage includes validated cell IDs and cue/PEV alignment, optional metadata
for decoder populations, stable PEV ties and finite-value filtering, distinct
activity/model-preparation ordering, disabled-check labels, malformed cached
trial IDs, state-row alignment, and separate cell-selectivity/PCA metadata.
Preparation tests also check persisted population definitions without changing
predictor columns. Existing tests continue to cover training-only CV
normalization and joint sorting of trial IDs and both duration outcomes.

### Full pipeline and numerical comparison

The full run reused the previous validation settings: the smoke preset with
extended screening diagnostics enabled. All eleven stages completed:

```bash
MPLCONFIGDIR=/tmp/wm-next-mpl .venv/bin/python scripts/next/pipeline.py \
  --settings configs/next/.dashboard/validation_screening_names.json \
  --data-dir data/example \
  --cache-dir cache/test_run_054_next_downstream_refactor \
  --stages all --n-jobs 1 --figure-formats png pdf
```

Recorded stage durations sum to **216.290 seconds**. The run produced **318
PNG/PDF pairs**, including 48 cell diagnostic pairs. Every figure stem has both
formats. Real-data marginal and PCA figures were visually checked: population
labels wrap cleanly, cell axes identify selectivity PEV, and PC axes identify
PCA explained variance.

Comparison with `test_run_053_next_screening_names` found:

- Screening still selects **32, 9, 13, and 32 cells** in sessions `210921`,
  `211015`, `221020`, and `221024`.
- Screening values, observed/null decoding, predictions, classifier choices,
  evaluation, state masks/durations, and the raw CV feature cache match exactly,
  including array dtypes and holdout assignments. The recursive comparison
  performed **1,075 checks**, including **416 arrays**. It excludes changed
  implementation fingerprints and normalizes run-root paths.
- The prepared **236 × 56** table has identical columns, values, row order,
  and dtypes. Its new per-session screening flags and population labels agree
  with the screening cache, CV cache, and preparation manifest.
- All **55 model CSV tables (2,847 rows)** match exactly after normalizing run
  paths in the two `prepared_data_path` columns.

A separate comparison of the preserved pre-refactor activity implementation
against the new preparation module covered all four example sessions with
both equal and PEV weighting. **808 recursive checks** matched exactly,
including balanced trial IDs, cell IDs/PEV, normalized activity, population
means, state masks, longest-off-state extraction, PCA components/scores/variance,
and sampled display points. Scientific methods and normalization populations
are unchanged; population metadata and figure labels are intentionally clearer.

The strict MkDocs build and whitespace check pass:

```bash
.venv/bin/mkdocs build --strict
git diff --check
```

This is a four-session smoke integration check with three null estimates and
a coarse decoding grid, not a production-size validation or controlled
performance benchmark. Generated settings and caches remain ignored by Git.
Model identifiers and predictor names stay stable; see
[downstream migration](../next/migration.md#refresh-downstream-provenance) for
typed activity records and checkpoint regeneration after code changes.

## Descriptive screening names validation — 2026-09-26

The screening naming refactor passed **300 Python tests** in **8.739 seconds**:
**183 next tests** (including **33 dashboard tests**) and **117 historical
tests**. The strict MkDocs build and whitespace check also pass.

Coverage includes the new configuration names, rejection of former JSON/CLI
names, all seven canonical failure codes and measurement columns, unavailable
statistics, selected/stationary cell populations, downstream cue/PEV consumers,
and readable figure and histogram labels. Screening caches now use schema
version **2**; tests confirm that older screening caches fail with regeneration
instructions and that the other primary cache schemas remain at version 1.
Legacy diagnostic reason codes are rejected before the histogram tool writes
new output.

### Full pipeline and diagnostic reports

The current smoke preset was copied to
`configs/next/.dashboard/validation_screening_names.json`, with only
`select.save_extended_diagnostics` changed to `true`:

```bash
MPLCONFIGDIR=/tmp/wm-next-mpl .venv/bin/python scripts/next/pipeline.py \
  --settings configs/next/.dashboard/validation_screening_names.json \
  --data-dir data/example \
  --cache-dir cache/test_run_053_next_screening_names \
  --stages all --n-jobs 1 --figure-formats png pdf

MPLCONFIGDIR=/tmp/wm-next-mpl WM_STATES_FIGURE_FORMATS=png,pdf \
  .venv/bin/python scripts/next/reject_reason_histograms.py \
  --cache-dir cache/test_run_053_next_screening_names
```

All eleven stages completed; their recorded durations sum to **216.348
seconds**. This is an integration check, not a controlled benchmark. The
pipeline produced **318 PNG/PDF pairs**, including **48 cell diagnostic
pairs**. The separate histogram command added four pairs and a 16-row summary
containing canonical `reason` codes and readable `reason_label` text. Every
figure stem has both formats. Representative cell and histogram PDFs were
rendered with Poppler and visually inspected.

Screening still selects **32, 9, 13, and 32 cells** in sessions `210921`,
`211015`, `221020`, and `221024`. All **1,120 diagnostic rows × 15 columns**
match `test_run_052_next_diagnostics` after mapping the renamed columns and
reason codes. All scientific screening fields and selected-cell properties
also match; only their names and configuration metadata changed.

Against `test_run_051_next_pdf`, observed/null decoding, predictions, selected
classifier C values, trial/time coordinates, evaluation, state masks and
durations match exactly. The **236 × 56** prepared table and complete CV
feature cache match, including holdout splits. All **55 model CSV tables**
(**2,847 rows**) match after normalizing run-root prefixes in two
`prepared_data_path` columns. No scientific differences were found.

The refreshed dashboard exposes the new screening fields, omits the old names,
and successfully validates the current preset with diagnostics enabled.
Generated validation settings and caches remain ignored by Git. See
[screening migration](../next/migration.md#screening-names-and-cache-version)
for mappings and the required regeneration of older screening caches.

## Diagnostic configuration validation — 2026-09-26

The diagnostic configuration refactor passed **294 Python tests** in
**7.824 seconds**: **177 next tests** (including **33 dashboard tests**) and
**117 historical tests**. The strict MkDocs build and whitespace check pass.

New coverage verifies versioned parsing, unknown and duplicate JSON fields,
finite numeric values, explicit plot switches, zero-based cell lists, half-open
ranges, session overrides, and resolved configuration snapshots. Integration
tests cover missing/skipped sessions, cap warnings, CSV-only diagnostics,
unchanged CSV scope and cell ordering, full-session traces, PDF-only export,
cleanup after export failure, and early rejection of malformed configs in the
screening CLI, pipeline dry run, and dashboard preflight. The old figure-list
schema and the obsolete stage-level title-detail flag are rejected explicitly.

### Real-data screening and diagnostic exports

`configs/next/.dashboard/validation_diagnostics.json` was copied from the smoke
preset with only `select.save_extended_diagnostics` changed to `true`. It uses
the supplied `configs/next/diagnostic_figures.json`: all available sessions,
all cells requested, at most 12 plots per session, and 150 DPI.

```bash
MPLCONFIGDIR=/tmp/wm-next-mpl .venv/bin/python scripts/next/pipeline.py \
  --settings configs/next/.dashboard/validation_diagnostics.json \
  --data-dir data/example \
  --cache-dir cache/test_run_052_next_diagnostics \
  --stages select --n-jobs 1 --figure-formats png pdf
```

Screening and diagnostics completed in **37.291 seconds**. The four sessions
contained **135, 86, 183, and 716 cells**, respectively. Diagnostics retained
all **1,120 rows and 15 columns**, while producing **48 matching PNG/PDF pairs**
(cell indices 0–11 from each session). Four cap warnings were expected. The
resolved configuration and actual cell targets were saved in
`select/diagnostics/figure_config.json`. A diagnostic PDF was rendered with
Poppler and visually inspected, including its time-window labels and complete
trial axis.

The selected populations remain **32, 9, 13, and 32 cells** for sessions
`210921`, `211015`, `221020`, and `221024`. Every scientific screening output
matches `test_run_051_next_pdf` exactly, including cell properties and cell/trial
indices; configuration metadata differs as intended. Later analysis stages were
not rerun for this diagnostic-only change.

The refreshed local dashboard returned successful validation for enabled
diagnostics using the new preset path and no longer exposed the removed
`skip_not_applicable_reasons_in_diagnostics_figure` field. Generated validation
settings and run artifacts remain ignored by Git.

## Shared figure export and PDF validation — 2026-09-26

All `next` figure writers now use one exporter, including screening diagnostics,
decoder inspection, cross-run comparisons, state/activity plots, and model plots.
The exporter supports PNG, TIFF, EPS, and PDF. The test suite passed **270 tests**
in **8.698 seconds**: **153 next tests** (including **30 dashboard tests**) and
**117 historical tests**. The **19 frontend tests**, production build, and
formatting check also pass.

Export checks cover PDF-only and mixed-format output, actual returned paths,
unchanged DPI/layout options, invalid formats failing before any output is
written, and restoration of environment and Matplotlib settings after a run.
A source-level regression check prevents plot writers from bypassing the shared
helper. PDF tests verify embedded TrueType fonts, vector-only line plots,
transparency, and lossless Flate-compressed image streams without JPEG/JPEG 2000
encoding. Dashboard tests cover PDF requests, artifact classification, MIME
type, and byte-identical downloads.

### Full pipeline with PNG and PDF

```bash
MPLCONFIGDIR=/tmp/wm-next-mpl .venv/bin/python scripts/next/pipeline.py \
  --settings configs/next/smoke_pipeline.json \
  --data-dir data/example \
  --cache-dir cache/test_run_051_next_pdf \
  --stages all --n-jobs 1 --figure-formats png pdf
```

All eleven stages completed for the four example sessions. Recorded stage
durations sum to **205.229 seconds**; this is an integration check, not a
controlled performance comparison. The run produced **270 PNG/PDF pairs**
with identical relative stems and no missing counterparts:

| Stage | Figure pairs |
| --- | ---: |
| Decode | 16 |
| States | 36 |
| Activity | 48 |
| Models | 72 |
| Nested count | 30 |
| Nested activity | 34 |
| Criticality | 20 |
| Interactions | 14 |

Screening diagnostics are disabled in this smoke preset; focused tests exercise
their PDF-only export. Stages that do not produce figures have no figure pairs.
All 270 PDFs passed header/end-marker, embedded-font, and lossless-stream
checks. Representative decoding, state-confidence, marginal-effect, and
interaction PDFs were rendered with Poppler and visually inspected. The run
revealed missing Unicode subscript glyphs in an existing interaction-axis label;
that label now uses math text. The two affected outcome plots were regenerated
from their saved comparison tables with missing-glyph warnings treated as
errors, and the corrected PDF was visually checked.

Scientific results match the previous PNG run, `test_run_048_next_dashboard`,
exactly: all four sessions' screening, observed/null decoding, evaluation,
state masks and durations, the **236 × 56** prepared table, and **97,348 numeric
values across 55 model CSV tables**. All **214 CV fits** succeeded and converged
with no fit errors.

Browser checks confirmed that selecting PNG and PDF produces the corresponding
validated command and that the result viewer lists both formats with download
links. PDF preserves vector geometry and text; heatmaps and other image artists
remain raster content, compressed losslessly. This check does not claim that
every artist is vector or that the full-resolution example preset was run.

## Dashboard validation — 2026-09-26

The React/TypeScript dashboard and FastAPI backend passed **259 Python tests**
(**142 next tests**, including **28 dashboard tests**, and **117 historical
tests**) in **7.258 seconds**, plus **19 frontend unit tests**. The production
TypeScript/Vite build, frontend formatting check, strict MkDocs build, lockfile
check, and whitespace check pass. A clean `npm ci` installed the locked packages.

```bash
MPLCONFIGDIR=/tmp/wm-next-mpl .venv/bin/python -m unittest discover -s tests
cd dashboard
npm ci
npm test
npm run build
npm run format:check
cd ..
uv lock --check --offline
uv run --group dashboard --group docs --locked mkdocs build --strict
```

Backend coverage includes typed settings and path validation, real subprocess
execution, WebSocket snapshots, worker cancellation, restart history,
start/cancel races, process cleanup after metadata or log failures, safe artifact
paths, table pagination, corrupt caches, nonfinite values, stale stage warnings,
cache invalidation, and dashboard run names. Frontend coverage includes copied
manifest settings, fresh run directories, explicit null values, enum display,
JSON drafts, shared defaults, trial/time/cue comparison warnings, partial-history
configuration comparisons, and duration formatting.

### Full pipeline through the browser

The service was started with:

```bash
MPLCONFIGDIR=/tmp/wm-next-mpl .venv/bin/python -m scripts.next.dashboard
```

The browser selected the smoke preset, `data/example`, one worker, PNG figures,
and all eleven stages, validated the configuration, then launched
`cache/test_run_048_next_dashboard`. Live stage changes and streamed logs were
observed while other views remained usable. All stages completed; their recorded
durations sum to **165.621 seconds**. This is a local integration check, not a
controlled timing benchmark.

- Four sessions: `210921`, `211015`, `221020`, and `221024`.
- 240 decoded trials, five time bins, and three null estimates per session.
- 86 cells passed selective-cell screening; the stationary-cell decoder used
  46, 30, 73, and 208 cells respectively. These populations are distinct.
- Prepared trial table: 236 rows and 56 columns.
- 214 held-out model fits across both outcomes, all converged, with no fit errors:
  68 model-family, 10 nested-count, 12 nested-activity, 68 criticality, and
  56 interaction fits.
- 367 browsable artifacts: 270 figures, 57 tables, 18 JSON files, and 22 logs.
- Run and all four session APIs returned no errors or stale-data warnings.

Observed/null decoding values, predictions, selected C values, trial/time
coordinates, state masks and durations, the prepared table, and all numeric CV
repeat metrics match `test_run_044_next` exactly. Run 044 also contains five
extra figures from separate inspection/comparison calls; these are not missing
pipeline outputs in run 048.

Job `72769d0911564fa09b1d2586bdb1fc58` saved the exact settings at
`configs/next/.dashboard/72769d0911564fa09b1d2586bdb1fc58.json`. Its command and
argument vector match the pipeline manifest; shell-splitting the command recovers
that same vector. The history manifest is byte-identical to the latest manifest.
Generated settings, jobs, logs, and run caches remain ignored by Git.

### Browser and failure-path checks

Browser review verified editable example defaults, the smoke preset, validation,
settings reuse into a fresh cache, live progress, tables, stage/session figure
filters, figure zoom, split-screen session views, confidence overlays, and
cross-run configuration comparisons. The interface was inspected at desktop and
390-pixel widths. No page-wide horizontal overflow was found in the tested
configuration and table views; wide CSV tables scroll within their viewer.

Run `test_run_047_next_dashboard_backend` first checked a real evaluation-only
subprocess against copied decoding caches. Run `test_run_049_next_dashboard_cancel`
completed a one-session five-stage smoke run before the cancellation check was
issued. A separate run, `test_run_050_next_dashboard_cancel`, was cancelled from
the browser during screening. The job became `cancelled`, its screening stage
was recorded as `interrupted`, later stages remained pending, and the process
exited. Cancellation does not remove already-written outputs.

The full-resolution example preset (100 null estimates and 50 model holdouts)
was not run. The dashboard remains a local single-user service with one active
job; trusted stage-layout caches are required. Manifest history preserves
invocation metadata, not immutable copies of scientific outputs.

## Command-capture validation — 2026-09-26

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
