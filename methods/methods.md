# Methods

We performed a three-stage analysis: (1) identifying stable, cue-selective
single-unit populations within each recording session; (2) quantifying
time-resolved decoding confidence for the cue associated with the largest
selective population; and (3) extracting discrete on- and off-states from the
decoding-confidence time courses. Analyses were performed separately for each
session. Time is reported in milliseconds relative to cue onset. The primary
epoch of interest was the delay period (500–1400 ms).

## Data inputs and alignment

Each session’s data file provided trial-by-time-by-cell spike counts, trial
correctness labels, cue identity labels (eight equally spaced cue locations),
and event timestamps (Panichello et al., 2024).

## Cell and trial selection

In this stage, we identified session partitions that contained sufficient
correct trials and stable cells for decoding. Cells were required to exhibit
reliable cue selectivity during the delay period, and activity was not
dominated by slow drifts across trials. Selection was performed on sliding
trial windows (session partitions) to accommodate nonstationarity over long
sessions.

### Session inclusion and sliding trial windows

Sessions were required to contain at least 320 total trials so that a fixed-size
trial window could be scanned across the session. Within each session, trials
were scanned using a sliding window of 320 trials with a step size of 10 trials.
For each window, only correct trials within that window were retained for downstream
selection. The 320-trial window size provides sufficient data for reliable estimation,
while the 10-trial step yields dense sampling of stability across the session.

### Minimum firing rate

Within each sliding window, cells were first screened for sufficient activity.
For each cell, the mean firing rate (Hz) was computed over the delay period
(500–1400 ms). Cells with mean firing rate < 1 Hz were excluded. This
criterion avoids unstable estimates of cue selectivity and decoding driven by
near-silent units.

### Temporal stability screening (steps 1–2; variance-based)

To reduce contributions from cells whose trial-to-trial variability is dominated
by slow temporal structure unrelated to cue information (e.g., drift), we applied
two variance-based criteria and one correlation-based criterion when ≥50 selected
trials were available in the current window.

**Step 1 (delay-to-baseline variance ratio).** For each cell, we
computed a trial-wise firing rate in a baseline window (−500 to 0 ms) and a
delay window (500–1000 ms). We then calculated the ratio of trial-to-trial
variance in the delay window to that in the baseline window. Cells were
retained only if this ratio exceeded 1.2, favoring cells whose task-related
fluctuations were larger than baseline fluctuations.

**Step 2 (baseline stability across time).** For cells passing step 1, we
quantified the stability of baseline variance by (i) computing baseline firing-rate
variance within sliding 50-trial subwindows and averaging across subwindows, and (ii)
comparing this average with the baseline variance estimated using all selected trials
in the current session partition. Cells were retained only if the ratio of the
subwindow-averaged baseline variance to the overall baseline variance exceeded 0.8.
This step downweights cells with strongly nonstationary baseline variance within the
window.

### Cue selectivity during the delay period

Cue selectivity was quantified with percent explained variance (PEV), computed
as ω² (a bias-corrected one-way ANOVA effect size) expressed as a percentage.
For each cell, time-resolved PEV was computed at each time bin (sliding
50-ms time windows stepped every 10 ms) within the delay period. Negative ω² values
were clipped at 0. A time bin was considered selective if PEV exceeded 2.5%.

To ensure sustained (rather than transient) selectivity, cells were required to
exhibit at least one contiguous selective period lasting ≥100 ms (i.e., ≥10
consecutive 10-ms steps) during the delay. Cells failing this duration criterion
were excluded. This duration constraint reduces false positives driven by brief
noise fluctuations across bins.

### Preferred cue assignment

Within each selective time bin, a cell’s preferred cue was defined as the cue
label with the highest mean firing rate across trials. Because cue labels are
angular (circular) variables, each cell’s overall preferred cue was estimated
as the circular mean of its per-bin preferred cues, computed over selective bins
only, and then mapped back to the nearest of the eight discrete cue locations.

### Temporal stability on preferred-cue trials (step 3, correlation-based)

To further reduce the influence of slow drifts that can masquerade as stable
selectivity, we assessed whether each cell’s delay-period firing rate changed
systematically over trials of its preferred cue within the current 320-trial
window. For each cell, we computed its mean firing rate over the delay period on
preferred-cue trials, then fit a linear regression of firing rate against trial
index within the window. Cells were retained only if the absolute correlation
coefficient satisfied |r| ≤ 0.5.

### Partition acceptance for downstream analyses

Selected cells were grouped by their preferred cue. A partition was considered
eligible for downstream decoding analyses if at least one cue group contained
≥12 cells. For each partition, we recorded the selected trial indices, selected cell
indices, cue grouping, and per-cell summary measures (mean delay firing rate,
mean PEV across selective bins, preferred cue, and temporal-stability
statistics).

## Decoding confidence analysis

In this stage, we produced for each session a trial-by-time matrix of
decoding confidence reflecting how strongly population activity supported
representation of the session’s preferred cue versus its opposite (180° away).
Restricting decoding to a preferred-versus-opposite comparison yields a
well-defined, maximally separated axis of representation while avoiding
additional complexity of multi-class decoding.

### Session and population definition from selected partitions

Decoding was performed using results of the selection stage. Only
partitions (trial windows) containing ≥12 cells in at least one preferred-cue
group were retained. Partitions were grouped by session, and sessions were
required to have ≥320 unique trials covered across their retained partitions to
ensure sufficient coverage by stable partitions.

To define a single decoding target per session, we first computed a
partition-level preferred cue as the cue associated with the largest
selective-cell population, then took the mode across partitions to obtain a
session-level preferred cue. The decoding population for a session was defined as the union
of all cells (across retained partitions) whose preferred cue matched the
session-level preferred cue.

### Trial selection and labels

Decoding was restricted to correct trials whose cue matched either (i) the
session-level preferred cue or (ii) its opposite cue (180° away). Trials were
labeled as preferred (=1) or opposite (=0). Preferred-cue trials served as the
test set for leave-one-out decoding; both preferred and opposite trials were
used for training.

### Time-resolved firing rates

For each selected trial and cell, firing rates were computed using sliding
50-ms time windows stepped every 10 ms spanning −200 to 1400 ms. Spike counts
within each window were converted to firing rate before decoding.

### Single-trial decoding and confidence metric

For each preferred-cue test trial, we used a leave-one-out procedure to prepare
a training set. The held-out preferred-cue trial was decoded using a model
trained on remaining preferred/opposite trials. To avoid biases due to
unequal class counts, training sets were class-balanced by randomly subsampling
the larger class without replacement to match the smaller class.

At each time bin, a separate decoder was fit using a support vector machine
with an RBF kernel, preceded by z-score standardization across training trials.
Decoding confidence was defined as the model’s probability estimate assigned
to the preferred cue for the held-out trial at that time bin, yielding a
trial-by-time confidence map per session.

### Shuffle-based null distribution

A null distribution was generated by repeating the decoding procedure 500 times
with shuffled class labels within the balanced training set. For each test trial
and time bin, the shuffle procedure produced a distribution of decoding confidence
under the null hypothesis of no relationship between neural activity and cue label.

## On- and off-state extraction

### Z-score normalization relative to the null distribution

For each session, decoding confidence was converted to z-scores at each trial
and time bin using the mean and standard deviation of its null distribution.
This normalization yields a trial-by-time z-score map reflecting evidence for
the preferred cue relative to chance expectations.

### Cluster definition and cluster mass

Clusters were defined as contiguous runs of suprathreshold bins along the time
axis within a trial. For each cluster, cluster mass was defined as the sum of
z-scores across all bins in that cluster.

### On-state detection

On-state candidates were defined as clusters with z > 1.96. To control
for multiple comparisons across the full trial-by-time map, we used a
shuffle-based maximum cluster-mass procedure. For each shuffle, we computed a
z-map and extracted the maximum cluster mass across all suprathreshold clusters.
The 95th percentile of this null distribution of maximum cluster masses served
as the cluster-level significance cutoff (α=0.05). Observed clusters whose mass
exceeded this cutoff were labeled as on-states.

### Off-state detection (null-like clusters)

Off-state candidates were defined as clusters of non-significant z-scores,
using a two-tailed criterion (|z| ≤ 1.96). Candidate clusters were required
to span at least 5 time bins to reduce spurious short segments.

To identify clusters consistent with the null, we computed a null distribution
of cluster masses by applying the same candidate definition to each shuffle’s
z-map and collecting the masses of all clusters meeting the duration criterion.
Observed candidate clusters were retained as off-states if their cluster masses
fell within the central 95% of this null mass distribution (2.5th–97.5th
percentiles), operationalizing “off” periods as segments statistically
indistinguishable from the shuffle-derived null.

### Delay-period state durations

On- and off-state durations were summarized within the delay period. For each
retained cluster, its time span was intersected with the delay period, and the
resulting duration contributed to histogram summaries.
