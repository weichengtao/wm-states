# Decoding confidence analysis

## Cached inputs and session filtering
Decoding confidence was computed using cached outputs from the cell and trial
selection procedure (`cell_trial_selection.pkl`). Each cached entry corresponds
to a sliding trial window (partition) and stores selected cells, preferred cue
estimates, and trial indices. We first retained only partitions with at least
12 cells in a single cue group. Partitions were then grouped by session, and we
computed the total number of unique trials covered across that session’s good
partitions. Sessions with fewer than 320 unique trials were excluded.

For each remaining session, we estimated a session-level preferred cue by
computing, for every partition, the most frequent preferred cue among its
cells and then taking the most frequent cue across partitions. This preferred
cue defined the decoding target for the session.

## Cell selection for decoding
For each session, we pooled all cells across good partitions whose preferred
cue matched the session-level preferred cue. These cells formed the decoding
population. If no cells met this criterion, the session was skipped.

## Trial selection
Decoding was restricted to correct trials where the cue matched either the
preferred cue or its opposite (180° away). Trials with the preferred cue were
used as test trials, and all preferred/opposite trials were used for training.

## Time-resolved firing rates
Spike trains from the selected trials and cells were converted to firing rates
using sliding time windows spanning −200 to 1400 ms relative to cue onset, with
50 ms windows stepped every 10 ms. For each bin, spike counts were normalized
by the bin duration to obtain rates in Hz.

## Decoder and confidence metric
For each preferred-cue test trial, we performed leave-one-out decoding at each
time bin. Training data were balanced by randomly subsampling the larger class
to match the smaller class. We trained a support vector machine with an RBF
kernel, preceded by z-score standardization, and used the model’s posterior
probability for the preferred cue as the decoding confidence. This yielded a
trial-by-time matrix of confidence values for each session.

## Null distribution (optional)
When requested, we generated a null distribution by shuffling class labels
within the training set and recomputing confidence for each time bin. Repeating
this procedure across shuffles produced a trial-by-time-by-shuffle null
confidence tensor for comparison with the observed decoding confidence.

## Outputs and visualization
For each session we cached decoding confidence, trial indices, time bins, and
population size in `decoding_confidence.pkl`. We also generated heatmaps of
confidence over time and preferred-cue trials, with optional labeling by actual
trial index.
