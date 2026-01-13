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
# You may want to keep only 210921.mat in --data-dir for testing.
python scripts/cell_trial_selection.py --data-dir data/nature --cache-dir cache/run_001 --trial-selection-step-size 10 --n-jobs 8

# You may want to compute 5 shuffles for testing.
python scripts/decoding_confidence.py --data-dir data/nature --cache-dir cache/run_001 --n-decode-shuffle 500 --n-jobs 8

python scripts/on_off_states.py --cache-dir cache/run_001 --off-duration-xmax 400
```
