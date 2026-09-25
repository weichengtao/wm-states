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

Null estimates still independently permute training-trial labels after the
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

Caches now live under stage directories: `select/`, `decode/`, `evaluate/`,
`states/`, `activity/`, `prepare/`, and each individual model stage. The root
contains only the runner manifest. Decoding checkpoints moved to
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
| `temp_dep_r_threshold: 2` | `check_preferred_drift: false` |

Remove the obsolete threshold entries or replace them with valid values. Remove
`temp_dep_detection` and set each temporal check independently. Remove selection's
unused `min_cell_per_group` and `seed`; keep `decode.min_cell_per_group` and
`decode.seed` where needed. Both current presets list all screening switches.
See [Screening checks](configuration.md#screening-checks) for CLI equivalents.

This is a deliberate behavior change: disabled checks no longer reject cells
with unavailable statistics. Old sentinel thresholds still performed those
applicability exclusions, so selected cells and downstream estimates can change.
Rerun selection and all dependent stages in a fresh cache directory.

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
