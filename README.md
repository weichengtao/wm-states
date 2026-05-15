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

```bash
## Leave-one-out (decoding)
# You may want to keep only 210921.mat in --data-dir for testing.
python scripts/cell_trial_selection.py --data-dir data/nature --cache-dir cache/run_001 --enable-trial-selection --trial-selection-window-size 320 --trial-selection-step-size 10 --n-jobs-partition 8
# python scripts/cell_trial_selection.py --data-dir data/nature --cache-dir cache/run_001_full_session --n-jobs-partition 8

# You may want to compute 5 shuffles for testing.
python scripts/decoding_confidence.py --data-dir data/nature --cache-dir cache/run_001 --n-decode-shuffle 500 --n-jobs 8

python scripts/on_off_states.py --cache-dir cache/run_001 --off-duration-xmax 400

## Leave-one-out (cell selection & decoding)
# You may want to keep only 210921.mat in --data-dir for testing.
python scripts/cell_trial_selection.py --data-dir data/nature --cache-dir cache/run_001_loo --enable-trial-selection --trial-selection-window-size 320 --trial-selection-step-size 10 --n-jobs-partition 8 --loo-cell-selection
# python scripts/cell_trial_selection.py --data-dir data/nature --cache-dir cache/run_001_loo_full_session --n-jobs-partition 8 --loo-cell-selection

# You may want to compute 5 shuffles for testing.
python scripts/decoding_confidence.py --data-dir data/nature --cache-dir cache/run_001_loo --n-decode-shuffle 500 --n-jobs 8 --loo-cell-selection

python scripts/on_off_states.py --cache-dir cache/run_001_loo --off-duration-xmax 400
```

Notes for cell_trial_selection.py:
- `--enable-trial-selection` runs across-trial sliding windows and uses `--trial-selection-window-size` / `--trial-selection-step-size`.
- Without `--enable-trial-selection` (or with `--no-enable-trial-selection`), one full-session trial window (`[0, num_trials)`) is used.
- Session-level minimum trial count is always enforced via `--min-trial-per-session` (default: `320`, counted on total trials including incorrect).

## New pipeline
```bash
# Cell triaging pipeline
# 1. Presence ratio
# 2. Correlation
# 3. PEV
uv run python scripts/cell_trial_selection.py \
--n-jobs-session 10 \
--n-jobs-partition 1 \
--data-dir data/nature \
--cache-dir cache/run_010_full_session \
--min-cell-per-group 15 \
--min-fr-test -1 \
--min-presence-ratio 0.9 \
--var-ratio-threshold-delay-over-baseline -1 \
--var-ratio-threshold-sliding-over-all -1 \
--temp-dep-r-threshold 2 \
--temp-dep-r-threshold-baseline 0.3 \
--sig-pev-threshold 2.5 \
--save-extended-diagnostics \
--diagnostics-figure-config configs/diagnostic_figure_config.json
```