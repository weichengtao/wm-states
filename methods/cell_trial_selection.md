# Cell and trial selection

## Data and trial inclusion
We analyzed single-unit spike trains from each session's `.mat` file containing
trial-by-time-by-cell spike counts, trial correctness, cue labels, and time
stamps. Trials were indexed relative to cue onset. Only correct trials were
eligible for downstream selection. Sessions with fewer trials than the sliding
window size were excluded from analysis.

## Sliding trial windows
To identify stable subsets of trials within each session, we scanned the
sequence of trials using a fixed-length sliding window (window size 320 trials,
step size 10 trials). For each window, we intersected the windowed trials with
correct trials and recomputed cue label distributions and spike data for the
selected trials.

## Cell activity threshold
Within each trial window, cells were first filtered by mean firing rate during
the delay/test period (500–1400 ms after cue onset, 50 ms bins stepped every 10
ms). A cell was retained if its mean firing rate in this period was at least
1 Hz.

## Temporal dependence screening (stages 1–2)
To remove cells with strong temporal dependence unrelated to cue information, we
performed variance-ratio screening when at least 50 trials were available in
the window. For each active cell, we computed trial-wise firing rates in a
baseline window (−500 to 0 ms) and a delay window (500 to 1000 ms). Cells were
kept if the delay-to-baseline variance ratio exceeded 1.2 (stage 1). For cells
passing stage 1, we computed a sliding-window baseline variance (using a window
length of 50 trials) and required the ratio of sliding-window variance to
full-baseline variance to exceed 0.8 (stage 2). Only cells passing both stages
remained eligible.

## Cue selectivity and preferred cue
Cue selectivity was quantified using percent explained variance (PEV; omega
squared) across cue conditions. For each eligible cell and each test-period
bin, we computed PEV from trial-averaged firing rates and required a contiguous
selectivity period of at least 100 ms where PEV exceeded 2.5%. Cells without
any such period were excluded. For the remaining cells, we computed the
preferred cue using the circular mean of preferred cues across significant PEV
bins.

## Temporal stability on preferred-cue trials (stage 3)
For each selected cell, we assessed temporal stability on trials of its
preferred cue within the current sliding window. We computed test-period firing
rates and regressed firing rate against trial index; cells were retained if the
absolute correlation coefficient did not exceed 0.5. Cells failing this
criterion were removed.

## Cell grouping and window retention
Selected cells were grouped by preferred cue. For each cue group we recorded
cell counts and total PEV. A trial window was considered acceptable for
downstream analyses if at least one cue group contained 12 or more cells.
For each accepted window we stored the selected trial indices, selected cell
indices, and per-cell summary metrics (mean test firing rate, mean PEV within
significant bins, preferred cue, number of significant bins, and temporal
stability statistics when applicable).
