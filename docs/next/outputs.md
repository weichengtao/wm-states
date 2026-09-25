# Outputs and inspection

`--cache-dir` always names the **run root**, for both the runner and standalone
scripts. Each stage owns a directory named after its pipeline stage ID.
When using `pipeline.py`, run-level metadata lives in `pipeline_manifest.json`
and `manifests/`. Standalone stage scripts use the same stage directories but
do not create runner-history records.

| Path relative to the run root | Contents |
| --- | --- |
| `pipeline_manifest.json` | Latest invocation’s resolved settings, progress, and status |
| `manifests/<run_id>.json` | Persistent record of each runner invocation, including partial and failed runs |
| `manifests/prior-<content hash>.json` | Preserved pre-history or orphaned latest manifests, when present |
| `select/cell_screening.pkl` | Full-session screening results |
| `select/tables/cell_screening.csv` | Screening summary, enabled checks, and settings |
| `select/diagnostics/` | Optional per-cell diagnostic CSV and rejection summary; plots in `figures/cells/` and `figures/reasons/` |
| `decode/decoding_confidence.pkl` | Observed and null decoding estimates |
| `decode/checkpoints/` | Per-session decoding checkpoints |
| `decode/figures/` | Confidence and classifier-C plots; trial inspection in `inspection/` |
| `evaluate/eval_confidence.pkl` | Decoding evaluation results |
| `evaluate/tables/eval_confidence.csv` | Evaluation summary |
| `evaluate/figures/across_runs/<comparison>/` | Cross-run evaluation plots, saved in each compared run |
| `states/on_off_states.pkl` | State masks and duration summaries |
| `states/figures/` | Plots grouped into `confidence/`, `masks/`, `durations/`, and `cluster_masses/` |
| `activity/figures/` | Activity plots grouped by `activity/` or `principal_components/`, then `states/` or `cues/` |
| `prepare/` | Shared `trial_table.pkl`, `cv_feature_cache.pkl`, and preparation `manifest.json` |
| `models/outcomes/<outcome>/` | Model-family comparison |
| `nested-count/outcomes/<outcome>/` | Nested cell-count comparisons |
| `nested-activity/outcomes/<outcome>/` | Nested activity comparisons |
| `criticality/outcomes/<outcome>/` | Active-cell threshold comparisons |
| `criticality/prepared/active_thresholds/` | Threshold-specific trial tables and manifests in `percentile_<NN>/`, plus `thresholds.csv` |
| `interactions/outcomes/<outcome>/` | Period interaction comparisons |

Model outcome directories contain `tables/`, `figures/`, `logs/`, and, when
enabled, `cross_validation/`. `<outcome>` is `total_off_state_duration` or
`maximum_off_state_duration`. Optional plots and diagnostics are created only
when requested and when the corresponding data are available.

PEV-weighted variants add `pev_weighted/` to the relevant output directory:
`prepare/pev_weighted/`, `activity/figures/pev_weighted/`, and
`<model-stage>/outcomes/<outcome>/pev_weighted/`. Criticality's weighted trial
tables live in `criticality/prepared/active_thresholds/percentile_<NN>/pev_weighted/`;
its threshold summary lives in `criticality/prepared/active_thresholds/pev_weighted/`.
The example preset weights activity plots but leaves mixed-effects inputs
unweighted. See [custom subdirectory settings](configuration.md#cache-directory-layout).

The four primary `.pkl` caches (screening, decoding, evaluation, and states) use
a versioned envelope. Load their result lists with `scripts.next.cache_io.read(path)`.
Read pickle caches only from trusted sources. Earlier flat next caches and the
shared `mixedlm/` layout must be regenerated in a fresh run directory; the new
scripts do not fall back to old locations.

Activity comparison includes preferred/opposite cue views, per-cell
and population plots, PCA, deterministic point sampling, and maximum off-state
highlights. PEV weighting applies to selective-cell population means; stationary
nonselective cells retain equal weights.

## Run manifest history

`pipeline.py` creates one history file for each invocation that begins execution.
Its `run_id` combines the UTC start timestamp and a random unique suffix, for
example `20260926T120000.123456Z_<32 hexadecimal characters>`. Runs starting at
the same time still receive distinct IDs.

The runner updates that invocation's file as stages run. Later invocations
create new files and leave earlier records unchanged. `pipeline_manifest.json`
is a replaceable copy of the latest invocation, including its progress while
running. It contains only that invocation's requested stages.

For example, a full run followed by an evaluation-only rerun produces:

| Invocation | Preserved history record | Latest view after completion |
| --- | --- | --- |
| `--stages all` | `manifests/<first_run_id>.json`, containing all eleven stages | Full run |
| `--stages evaluate` | `manifests/<second_run_id>.json`, containing evaluation only | Evaluation-only run; the first record remains unchanged |

Each new record has these fields:

| Field | Meaning |
| --- | --- |
| `run_id` | Unique identifier; matches the history filename without `.json` |
| `started_at`, `finished_at` | UTC timestamps; `finished_at` is `null` until a terminal status is recorded |
| `status` | Overall invocation status: `running`, `complete`, `failed`, or `interrupted` |
| `runner_config` | Runner options, including the original stage request, data/cache paths, preset path, and figure formats |
| `invocation` | Command origin, exact Python argument vector, quoted command, working directory, and interpreter path |
| `settings` | Resolved configuration for every requested stage, including inherited defaults and overrides |
| `stages` | Ordered execution entries for attempted stages, with status, elapsed `seconds` when the attempt ends, and `error` when applicable |

The preset's path is recorded, but the preset file and source code are not
copied into history. Keep the code revision and relevant input data with any
reported analysis. `settings` preserves the effective settings for requested
stages even if the preset is later edited. See [Resume and rerun](configuration.md#resume-and-rerun)
for status transitions and interrupted executions.

### Inspect saved invocations

Run this from the repository root, replacing the cache path with your run:

```python
import json
from pathlib import Path

cache = Path("cache/next_run_034_full_session")
for path in sorted((cache / "manifests").glob("[0-9]*.json")):
    record = json.loads(path.read_text())
    requested = ", ".join(record["settings"])
    attempted = ", ".join(f"{s['stage']}={s['status']}" for s in record["stages"])
    print(record["run_id"], record["status"], record["started_at"])
    print("  requested:", requested)
    print("  attempted:", attempted)
    invocation = record.get("invocation") or {}
    print("  command:", invocation.get("command") or "No captured CLI command")
```

This lists timestamped records; preserved `prior-*.json` files retain their old
format and are inspected separately. Save the relevant history filenames when
reporting an analysis assembled across multiple invocations.

### Reuse a recorded command

CLI invocations record the following under `invocation`:

| Field | Meaning |
| --- | --- |
| `source` | `"cli"` for a command-line run, or `"programmatic"` for a Python API call |
| `argv` | Exact argument values received by Python, including interpreter options and either the script path or `-m scripts.next.pipeline` |
| `command` | A POSIX-shell command built from `argv`, with spaces, quotes, and shell metacharacters escaped |
| `cwd` | Absolute working directory when the runner was called; relative paths are interpreted here |
| `python_executable` | Interpreter path reported by Python for that invocation |

`argv` preserves argument values and ordering rather than reconstructing them
from the resolved configuration. `command` represents those same arguments with
normalized shell quoting; it is not a verbatim copy of the text typed into the
shell. Python does not retain outer launchers such as `uv run`, shell aliases,
original quoting, environment assignments, or input/output redirections. Those
are not inferred or recorded. The interpreter path helps identify the original
environment, but that environment is not copied into the manifest.

Print a command reference for the latest run, or replace the manifest path with
any timestamped history record. This example only prints commands:

```python
import json
from pathlib import Path
import shlex

record = json.loads(Path("cache/next_run_034_full_session/pipeline_manifest.json").read_text())
invocation = record.get("invocation") or {}
if invocation.get("command"):
    print("cd -- " + shlex.quote(invocation["cwd"]))
    print(invocation["command"])
else:
    print("No captured CLI command; inspect runner_config and settings.")
```

The working directory matters when replaying relative paths. The same command
uses the files, code, environment, and defaults present when it is run again;
compare those with the saved resolved `settings` before interpreting it as a
reproduction of an earlier analysis.

A direct Python call such as `pipeline.main(Config(...))` has no pipeline CLI
command. It records `source: "programmatic"`, with `argv` and `command` set to
`null`, rather than recording the notebook or test runner's command. Its working
directory, interpreter path, runner configuration, and resolved settings are
still retained. Earlier history records may lack `invocation` entirely and
are left unchanged.

### Existing records and retention

Before replacing an existing latest manifest that has no identical history
record, the runner preserves its original bytes as
`manifests/prior-<SHA-256 content hash>.json`. This covers earlier manifest
formats and missing or differing history copies. Identical preserved content
is not duplicated. Existing fields are kept as-is, without inventing timestamps
or converting the old schema; even an unreadable previous record is preserved.
Records overwritten before this feature cannot be recovered.

There is no automatic history pruning. History preserves execution metadata;
it does not snapshot caches or figures, and does not restore old analysis
outputs. Stage-level preparation files such as `prepare/manifest.json` still
describe the current prepared data and are replaced when preparation reruns.
Use separate run roots to retain different analysis variants. Standalone
scripts also replace their stage outputs without adding a runner record; use
`pipeline.py --stages <stage>` when you want that execution recorded.

## Load a primary cache

Run this from the repository root in the analysis environment:

```python
from pathlib import Path
from scripts.next.cache_io import read

results = read(Path("cache/next_run_034_full_session/decode/decoding_confidence.pkl"))
for session in results:
    print(session["session"], session["decoding_confidence"].shape)
```

The loader checks the primary cache schema. Historical caches must be regenerated
with the new scripts. Do not use a raw `pickle.load` result as if it were the
session list: the four primary cache files store a versioned envelope.

## Inspect and compare results

After state detection, inspect a session present in the decoding cache. `--trial`
uses zero-based cached trial rows; figures also show the original trial IDs.

```bash
uv run python scripts/next/inspect_decoding_results.py \
  --cache-dir cache/next_run_034_full_session \
  --session 221024 --trial 0 1 \
  --time-bin-start -200 1400 --with-null --with-state
```

To compare two completed runs, evaluate each run first with
`eval_confidence.py`. Replace the second cache path and aliases below with your
runs. Comparison aligns common sessions and supports null shading with
`percentiles`, `confidence_intervals`, or `none`.

For each shared session, comparison warns when another run's preferred cue or
trial-ID set differs from the first run, or that metadata is missing. Figures
are still generated, but the scores may describe different trial populations.
Reordering the same trials does not produce a warning.

```bash
uv run python scripts/next/eval_confidence_across_runs.py \
  --cache-dirs cache/next_run_034_full_session cache/next_run_035_full_session \
  --run-aliases "Run 034" "Run 035" \
  --line-colors "tab:blue" "tab:orange" \
  --null-shading percentiles
```

## Diagnostic tools

These optional scripts are separate from the eleven-stage runner:

| Script in `scripts/next/` | Purpose |
| --- | --- |
| `inspect_decoding_results.py` | Plot observed/null confidence and state assignments |
| `eval_confidence_across_runs.py` | Compare evaluated runs on common sessions |
| `reject_reason_histograms.py` | Summarize saved screening rejection diagnostics |

Selection diagnostics are opt-in through `select.save_extended_diagnostics` in
JSON or `--save-extended-diagnostics` on the selection script. Use each script's
`--help` for its required inputs and plotting options.
The diagnostic CSV's `presence_ratio` uses correct trials in the configured
screening window ([−400, 1400) ms in the example), matching the screening criterion.
Per-check columns distinguish `disabled`, `pass`, `fail`, and `not_applicable`.
Activity traces and
the additional baseline Spearman correlation still describe all session trials.
