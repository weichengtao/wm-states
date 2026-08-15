# WM States Analyses

This repo contains standalone analysis scripts under `scripts/`. These scripts
are not meant to be installed as a package; the `pyproject.toml` exists to
capture dependencies for reproducible setup.

## Setup with conda

```bash
conda create -n wm_states python=3.10.12
conda activate wm_states
pip install -e .
```

## Setup with uv

```bash
uv sync --python 3.10.12
```

## Data

Download the dataset from `https://datadryad.org/dataset/doi:10.5061/dryad.kkwh70sct` and place the `.mat` files in
`data/nature` before running the scripts.

## Run scripts

Run the scripts with `uv run` so they use the environment defined by
`pyproject.toml`. For a complete analysis, run cell selection, decoding,
optional repeat inspection, and on/off-state analysis in that order. Use the
same `--cache-dir` for all stages.

## New pipeline

The commands below run the full-session pipeline. Cell selection writes the
selection cache, decoding reuses it and writes repeat-level confidence and
accuracy results, and the final analysis uses the decoding cache to identify
on/off states. The inspection script is optional and saves one PNG per
selected trial and time bin; two range endpoints are inclusive, and time-bin
endpoints are matched to the nearest cached bin.

```bash
# 1. Select decoder cells and cache the selection results.
uv run python scripts/cell_trial_selection.py \
--n-jobs-session 10 \
--n-jobs-partition 1 \
--data-dir data/nature \
--cache-dir cache/run_029_full_session \
--t-test-window 50 \
--min-cell-per-group 1 \
--min-fr-test -1 \
--min-presence-ratio 0.9 \
--var-ratio-threshold-delay-over-baseline -1 \
--var-ratio-threshold-sliding-over-all -1 \
--temp-dep-r-threshold 2 \
--temp-dep-r-threshold-baseline 0.3 \
--sig-pev-threshold 2.5 \
--no-save-extended-diagnostics \
--diagnostics-figure-config configs/diagnostic_figure_config.json

# 2. Fit repeated decoders and estimate shuffled null confidence.
uv run python scripts/decoding_confidence.py \
--data-dir data/nature \
--cache-dir cache/run_029_full_session \
--t-decode-window 50 \
--min-cell-per-group 1 \
--n-repeats-for-model-fit 500 \
--n-decode-shuffle 500 \
--n-jobs 10 \
--cells-used-for-decoder STATIONARY \
--svm-kernel LINEAR \
--decoder-model LOGISTIC_REGRESSION \
--classifier-c 0.01 \
--seed 42 \
--max-sessions-to-run 25

# 3. Inspect repeat-level accuracy and confidence distributions (optional).
uv run python scripts/inspect_decoding_results.py \
--cache-dir cache/run_029_full_session \
--session 210921 \
--trial 0 1 \
--time-bin-start 500 1400

# 4. Identify and summarize on/off states.
uv run python scripts/on_off_states.py \
--cache-dir cache/run_029_full_session \
--cc-method-off one_tailed
```
