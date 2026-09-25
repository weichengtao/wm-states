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
precedence. Paths in JSON are relative to the working directory. The presets use
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

## Resume and rerun

The runner writes resolved settings, stage status, and timings to
`<cache>/pipeline_manifest.json`. It runs every requested stage on each invocation;
only decoding automatically reuses matching per-session checkpoints. A checkpoint
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
require `decoding_confidence.pkl` alongside the selection and state caches.
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
