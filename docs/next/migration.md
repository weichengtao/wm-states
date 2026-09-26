# Migrate to the next pipeline

The new implementation lives in `scripts/next/`, presets in `configs/next/`, and
tests in `tests/next/`. Historical scripts remain available in their original
locations. Start each migrated analysis with a fresh cache directory.

## Removed behavior

| Historical behavior | Next pipeline |
| --- | --- |
| Session partitions and partition selection | Complete sessions only |
| Label-preserving shuffle before train/test split | Unsupported |
| Label-preserving shuffle inside training data | Unsupported |
| Decoder repeats across random seeds | One observed estimate per tested trial/bin |
| Pooled-delay activity decoding | Fit each time bin independently |
| Standalone baseline-activity duration regression | Removed |
| Standalone cell-count duration regression | Removed |
| Population ISI analysis | Removed; no optional next implementation |
| Pipeline-level `--n-decode-shuffle` | Set `decode.n_decode_shuffle` in JSON |

Null estimates still permute training-trial labels after the
outer train/test split. Mixed-effects trial-holdout repetitions and baseline/
cell-count predictors remain supported.

## Update an existing run

1. Copy `configs/next/example_pipeline.json` and edit supported stage settings.
2. Remove `baseline` and `cell-count` objects from older pipeline JSON files.
   The runner rejects these names even when another stage is selected.
3. Remove partition, repeat-selection, pooled-delay, and label-preserving shuffle
   options. The new scripts reject unknown configuration fields.
4. Set `decode.n_decode_shuffle` and inspect all resolved settings with `--dry-run`.
5. Run selection and decoding into a new cache directory, followed by downstream
   stages. Historical cache files are not a supported input format.

The resulting observed arrays have axes `(trial, bin)` and null arrays have
axes `(trial, bin, N)`. There is no observed repeat axis to average or select.
Update external analysis code accordingly; see [Outputs](outputs.md).

The runner now has five default stages and six mixed-effects stages. See
[Pipeline stages](pipeline.md) for their order and prerequisites.

## Upgrade an earlier next run

### Preserve runner history

New `pipeline.py` invocations preserve their records in `manifests/` and keep
`pipeline_manifest.json` as the latest view. New CLI records include command arguments, working directory, and a quoted
command reference. Earlier records are not backfilled with guessed commands.
An existing root manifest is archived automatically before replacement, including old-format records.
This cannot recover manifests overwritten by earlier versions. The history
upgrade itself needs no manual manifest migration; the cache-layout migration
below still requires regenerated caches. As with other code updates, decoding
fingerprints may change and trigger refitting when decoding next runs. See
[Run manifest history](outputs.md#run-manifest-history).

### Update diagnostic figure settings

Next screening uses `configs/next/diagnostic_figures.json`, with a versioned
`targets` and `plots` structure. Replace paths to the historical
`configs/diagnostic_figure_config.json` in custom next presets. Copy the new
default or `configs/next/diagnostic_figures.template.json`, then transfer the
sessions and cells you want to plot. The old `figures` array is not accepted.

Use `targets.sessions` for session stems, `targets.cells` for the default cell
selector, and `targets.cells_by_session` for per-session replacements. An old
inclusive range `cell_start: 0, cell_end: 4` becomes
`{"start": 0, "stop": 5}`. Use `"all"` explicitly for all cells; an empty cell
list now requests no figures. Remove `trial_start`, `trial_end`, and
`trial_holdout`: diagnostic traces always use complete sessions.

Remove `skip_not_applicable_reasons_in_diagnostics_figure` from stage JSON and
CLI commands. Its replacement, `plots.show_not_applicable_reasons`, lives in
the diagnostic file and has the inverse meaning: set it to `false` to hide
those title details. The complete rejection reasons remain in the CSV.

Diagnostics remain disabled by default in both pipeline presets. When enabled,
the new supplied figure config caps output at the first 12 sorted cells per
available session and warns about truncation. Set
`plots.max_cells_per_session` to `null` to plot all requested cells. Setting
the figure-config path to `null` or `plots.enabled` to `false` gives CSV-only
diagnostics. Plot targets never filter the screening or diagnostic table.
See [screening diagnostics](configuration.md#screening-diagnostics) for the
complete schema and commands. Historical configurations outside `configs/next/`
remain available for the historical scripts.

### Move to stage directories

Caches now live under stage directories: `select/`, `decode/`, `evaluate/`,
`states/`, `activity/`, `prepare/`, and each individual model stage. The root
also contains the latest runner manifest and the `manifests/` history directory.
Decoding checkpoints moved to
`decode/checkpoints/`; the shared `mixedlm/` directory is no longer used.
See [Outputs](outputs.md) for all paths, including figures and diagnostics.

Keep `--cache-dir` pointing to the run root. Remove old `"mixedlm/prepared"`
and `"mixedlm"` subdirectory settings to use the new defaults. Custom
subdirectories are now relative to their owning stage, not the run root;
update producer and consumer overrides together. Criticality's
`prepared_subdir` defaults to `"prepared"`, relative to `criticality/`.

Run **all required stages starting with `select` in a fresh cache directory**
when migrating the layout. Moving existing files is insufficient: decoding
provenance includes input paths and implementation code. No legacy-path
fallback or automatic cache migration is provided. The rerun guidance below
applies after establishing the new layout.

### Update screening commands and checks

The next screening entry point is now `scripts/next/cell_screening.py`
(module form: `python -m scripts.next.cell_screening`). Its outputs are
`select/cell_screening.pkl` and `select/tables/cell_screening.csv`. Update direct commands and any
external cache readers that used `cell_trial_selection.py` or its cache names.
The runner stage and JSON section remain `select`. Rerun screening and its
dependents rather than renaming an old cache, since decoding provenance also
depends on the source path and implementation code. Historical scripts outside
`scripts/next/` retain their original names.

Replace sentinel thresholds with explicit screening switches:

| Old selection setting used to avoid rejection | Replacement in the example |
| --- | --- |
| `min_fr_test: -1` | `check_firing_rate: false` |
| `var_ratio_threshold_delay_over_baseline: -1` | `check_delay_variance: false` |
| `var_ratio_threshold_sliding_over_all: -1` | `check_baseline_variance: false` |
| `temp_dep_r_threshold: 2` | `check_preferred_cue_drift: false` |

Remove the obsolete threshold entries or replace them with valid values. Remove
`temp_dep_detection` and set each temporal check independently. Remove selection's
unused `min_cell_per_group` and `seed`; keep `decode.min_cell_per_group` and
`decode.seed` where needed. Both current presets list all screening switches.
See [Screening checks](configuration.md#screening-checks) for CLI equivalents.

This is a deliberate behavior change: disabled checks no longer reject cells
with unavailable statistics. Old sentinel thresholds still performed those
applicability exclusions, so selected cells and downstream estimates can change.
Rerun selection and all dependent stages in a fresh cache directory.

### Screening names and cache version

Screening settings now identify the check, measured quantity, and units.
Update custom `select` objects using this mapping. CLI options use the same
names with underscores replaced by hyphens: for example, `--min-fr-test`
becomes `--min-test-firing-rate-hz`. Both old JSON keys and old CLI flags are
rejected; there are no aliases. The supplied presets already use the new names.

| Previous JSON key | Current JSON key |
| --- | --- |
| `min_trial_per_session` | `min_trials_per_session` |
| `min_fr_test` | `min_test_firing_rate_hz` |
| `presence_start` / `presence_end` | `presence_start_ms` / `presence_end_ms` |
| `var_ratio_threshold_delay_over_baseline` | `min_delay_to_baseline_variance_ratio` |
| `var_ratio_threshold_sliding_over_all` | `min_baseline_window_variance_ratio` |
| `min_trial_for_temp_check` | `variance_window_trials` |
| `temp_check_baseline_start` / `temp_check_baseline_end` | `variance_baseline_start_ms` / `variance_baseline_end_ms` |
| `temp_check_delay_start` / `temp_check_delay_end` | `variance_delay_start_ms` / `variance_delay_end_ms` |
| `temp_dep_r_threshold_baseline` | `max_abs_baseline_drift_r` |
| `baseline_drift_start` / `baseline_drift_end` | `baseline_drift_start_ms` / `baseline_drift_end_ms` |
| `sig_pev_threshold` | `selectivity_pev_threshold_pct` |
| `sig_pev_duration` | `selectivity_min_duration_ms` |
| `pev_clip_at` | `selectivity_pev_floor_pct` |
| `check_preferred_drift` | `check_preferred_cue_drift` |
| `temp_dep_r_threshold` | `max_abs_preferred_cue_drift_r` |
| `t_test_start` / `t_test_end` | `test_start_ms` / `test_end_ms` |
| `t_test_window` / `t_test_step` | `selectivity_bin_width_ms` / `selectivity_bin_step_ms` |

`variance_window_trials` retains both roles of its predecessor: the minimum
number of correct trials for either variance check and the baseline
sliding-window length. The rename does not alter the methods, thresholds,
or selected populations. “Selectivity” replaces “significant PEV” because
the PEV threshold is an effect-size criterion, not a statistical significance
test.

Update external readers of diagnostic CSVs and cached cell properties:

| Previous measurement or status field | Current field |
| --- | --- |
| `mean_fr_test` | `mean_test_firing_rate_hz` |
| `temp_dep_var_ratio_stage1` | `delay_to_baseline_variance_ratio` |
| `temp_dep_sliding_ratio_stage2` | `baseline_window_variance_ratio` |
| `temp_dep_r_baseline` | `baseline_drift_r` |
| `temp_dep_r` | `preferred_cue_drift_r` |
| `mean_pev_test` | `mean_selectivity_pev_pct` |
| `mean_pref_test` | `preferred_cue` |
| `num_sig_pev_bins` | `qualifying_selectivity_bin_count` |
| `r_s_baseline` | `baseline_all_trials_spearman_r` |
| `check_preferred_drift` | `check_preferred_cue_drift` |

`preferred_cue` and `qualifying_selectivity_bin_count` belong to the selected-cell
cache properties, rather than the per-cell diagnostic CSV. The latter is
present when selectivity screening is enabled.

Failure codes now use `fail_<check identifier>`. The `rejection_reason` column
and rejection-summary table use these names consistently. The summary also
adds `reason_label` for a readable label; figure titles and axes use those
labels.

| Previous failure code | Current failure code |
| --- | --- |
| `fail_min_fr_test` | `fail_firing_rate` |
| `fail_min_presence_ratio` | `fail_presence_ratio` |
| `fail_temp_dep_stage1` | `fail_delay_variance` |
| `fail_temp_dep_stage2` | `fail_baseline_variance` |
| `fail_temp_dep_stage3_baseline` | `fail_baseline_drift` |
| `fail_sig_pev` | `fail_selectivity` |
| `fail_temp_dep_stage3` | `fail_preferred_cue_drift` |

The `_not_applicable` suffix is unchanged and follows the new failure code.
`pass` still means no enabled cell check rejected the cell. Other check-status
values remain `disabled`, `pass`, `fail`, and `not_applicable`.

Screening caches now require envelope version 2. Earlier screening caches are
rejected with an instruction to rerun `select`; there is no automatic
conversion. Regenerate screening and diagnostics, then rerun dependent stages
to establish matching provenance. Use a fresh run directory to retain the old
results for comparison. Decoding, evaluation, and state cache envelope versions
remain 1. Existing manifests and historical validation records retain the
settings and field names used at the time of their runs.

### Refresh downstream provenance

Downstream consumers now share screening metadata and session-input helpers.
Activity preparation, typed records, and plotting live in separate modules;
`compare_activity_across_states.py` remains the CLI and re-exports its analysis
helpers. Python callers constructing `SessionActivity` should supply
`dimensions=CellActivityDimensions(cell_ids, selectivity_pev_pct)` for cells or
`PrincipalComponentDimensions(component_numbers, explained_variance_ratio,
source_cell_count)` for PCs. These records live in `activity_types.py` and
replace the overloaded `cell_ids`/`cell_pev` constructor fields. The PCA ratio
is stored as a fraction and converted to percent only for display.

Existing model group keys, predictor columns, and CLI options are unchanged.
Activity labels and preparation metadata describe the enabled screening checks;
they do not infer selectivity from a historical group-key name. Shared validation
rejects malformed cell and trial IDs instead of silently coercing them. These
changes preserve the numerical methods and normalization populations, but the
implementation fingerprint changes, so earlier decoding checkpoints require
regeneration as described below.

Activity comparison and mixed-effects preparation now require the decoding
cache to validate the selection/data provenance and the state's decoding
fingerprint, preferred cue, trial IDs, and time bins. Matching array shapes
alone are insufficient. Missing or stale provenance stops these stages.

After updating the code, rerun `decode evaluate states` using the original
preset, data directory, and cache directory, then rerun downstream analyses.
Rerun `select` first if data or screening settings changed. Decoder fingerprints
include implementation code, so checkpoints from an earlier revision may refit.
See [Resume and rerun](configuration.md#resume-and-rerun).

To refresh presence ratios in an existing diagnostic CSV, rerun selection with
extended diagnostics enabled. The CSV now uses correct trials in [−400, 1400) ms,
matching screening. Cross-run evaluation plots continue to run when cues or
trial sets differ, but now emit a warning identifying the session and runs.
