# Methods

We performed three analysis stages to (i) identify stable, cue-selective
single-unit populations within each recording session, (ii) quantify
time-resolved *decoding confidence* for the represented cue, and (iii) extract
discrete on- and off-states from the confidence time courses. Analyses were
performed separately for each session. Time is reported in milliseconds
relative to cue onset; the primary epoch of interest was the delay period
(500–1400 ms).

## Data inputs and alignment

Each session’s MATLAB `.mat` file provided trial-by-time-by-cell spike counts,
trial correctness labels, cue identity labels (8 equally spaced cue locations),
and the time axis. Trials were aligned to cue onset using the provided
timestamps (fields: `spks`, `isCorr`, `cueAngIdx`, `tc`).

## 1) Cell and trial selection (stable, cue-selective populations)

The goal of this stage is to identify (within each session) subsets of trials
and cells that (i) contain sufficient correct trials for inference, (ii) show
reliable cue selectivity during the delay period, and (iii) are not dominated by
slow drifts across trials. This selection is performed on multiple overlapping
trial windows to accommodate nonstationarity over long sessions.
Using a fixed-size window maintains comparable sample size across estimates
while allowing selection to focus on locally stable segments of the session.

### Session inclusion and sliding trial windows

Sessions were required to contain at least 320 total trials so that a fixed-size
trial window could be scanned across the session. Within each session, trials
were scanned using a fixed-length sliding window of 320 trials with a step size
of 10 trials. For each window, only correct trials within that window were
retained for downstream selection to minimize contamination by error-related
activity.
The 320-trial window size provides sufficient data for reliable estimation,
while the 10-trial step yields dense sampling of stability across the session.

### Minimum firing rate

Within each sliding window, cells were first screened for sufficient activity.
For each cell, the mean firing rate (Hz) was computed over the delay period
(500–1400 ms) by dividing total spike counts by the total elapsed time across
all selected trials. Cells with mean firing rate < 1 Hz were excluded. This
criterion avoids unstable estimates of cue selectivity and decoding driven by
near-silent units.

### Temporal-dependence screening (variance-ratio; stages 1–2)

To reduce contributions from cells whose trial-to-trial variability is dominated
by slow temporal structure unrelated to cue information (e.g., drift), an
additional screening step was applied when ≥50 selected trials were available
in the current window.

**Stage 1 (delay vs baseline variance ratio).** For each active cell, we
computed a trial-wise firing rate in a baseline window (−500 to 0 ms) and a
delay window (500 to 1000 ms). We then calculated the ratio of trial-to-trial
variance in the delay window to that in the baseline window. Cells were
retained only if this ratio exceeded 1.2, favoring cells whose task-epoch
fluctuations are larger than baseline fluctuations.

**Stage 2 (baseline stability across time).** For cells passing stage 1, we
quantified the stability of baseline variance by computing baseline firing-rate
variance within sliding 50-trial sub-windows and averaging these sub-window
variances. Cells were retained only if the ratio of this sliding-window baseline
variance to the overall baseline variance exceeded 0.8. This step down-weights
cells with strongly nonstationary baseline variance within the window.

### Cue selectivity during the delay period (PEV / ω²)

Cue selectivity was quantified with percent explained variance (PEV), computed
as ω² (a bias-corrected one-way ANOVA effect size) expressed as a percentage.
For each cell, PEV was computed separately at each time bin using overlapping
50-ms time windows stepped every 10 ms from 500 to 1400 ms. Negative ω² values
were clipped at 0. A time bin was considered selective if PEV exceeded 2.5%.
We used ω² to reduce positive bias in variance-explained estimates at finite
sample sizes.

To ensure sustained (rather than transient) selectivity, cells were required to
exhibit at least one contiguous selective period lasting ≥100 ms (i.e., ≥10
consecutive 10-ms steps) during 500–1400 ms. Cells failing this duration
criterion were excluded.
This duration constraint reduces false positives driven by brief noise
fluctuations across bins.

### Preferred cue assignment (circular mean across selective bins)

Within each selective time bin, a cell’s *preferred cue* was defined as the cue
label with the highest mean firing rate across trials. Because cue labels are
angular (circular) variables, each cell’s overall preferred cue was estimated
as the circular mean of its per-bin preferred cues, computed over selective bins
only, and then mapped back to the nearest of the eight discrete cue locations.

### Temporal stability on preferred-cue trials (stage 3)

To further reduce the influence of slow drifts that can masquerade as stable
selectivity, we assessed whether each cell’s delay-period firing rate changed
systematically over trials of its preferred cue within the current 320-trial
window. For each cell, we computed its mean firing rate over 500–1400 ms on
preferred-cue trials, then fit a linear regression of firing rate against trial
index within the window. Cells were retained only if the absolute correlation
coefficient satisfied |r| ≤ 0.5.

### Window acceptance for downstream analyses

Selected cells were grouped by their preferred cue. A window was considered
eligible for downstream decoding analyses if at least one cue group contained
≥12 cells. For each window, we cached the selected trial indices, selected cell
indices, cue grouping, and per-cell summary measures (mean delay firing rate,
mean PEV across selective bins, preferred cue, and temporal-stability
statistics).

## 2) Decoding confidence analysis (time-resolved classification)

The goal of this stage is to produce, for each session, a trial-by-time matrix
of *decoding confidence* reflecting how strongly the population activity
supports the session’s preferred cue versus its opposite (180° away).
Restricting decoding to a preferred-versus-opposite comparison yields a
well-defined, maximally separated axis of representation while avoiding
additional complexity of multi-class decoding.

### Session and population definition from cached partitions

Decoding was performed using cached outputs from the selection stage. Only
partitions (trial windows) containing ≥12 cells in at least one preferred-cue
group were retained. Partitions were grouped by session, and sessions were
required to have ≥320 unique trials covered across their retained partitions to
ensure sufficient coverage by stable windows.

To define a single decoding target per session, we first computed a
partition-level preferred cue as the modal preferred cue across cells in that
partition, then took the mode across partitions to obtain a session-level
preferred cue. The decoding population for a session was defined as the union
of all cells (across retained partitions) whose preferred cue matched the
session-level preferred cue.

### Trial selection and labels

Decoding was restricted to correct trials whose cue matched either (i) the
session-level preferred cue or (ii) its opposite cue (180° away). Trials were
labeled as preferred (=1) or opposite (=0). Preferred-cue trials served as the
test set for leave-one-out decoding; both preferred and opposite trials were
used for training.

### Time-resolved firing rates

For each selected trial and cell, firing rates were computed using overlapping
50-ms time windows stepped every 10 ms spanning −200 to 1400 ms. Spike counts
within each window were converted to Hz by dividing by the window duration.

### Decoder, cross-validation, and confidence metric

For each preferred-cue test trial, we used leave-one-out cross-validation at
each time bin: the held-out preferred-cue trial was decoded using a model
trained on all remaining preferred/opposite trials. To avoid biases due to
unequal class counts, training sets were class-balanced at each leave-one-out
split by randomly subsampling the larger class without replacement to match the
smaller class.

At each time bin, a separate decoder was fit using a support vector machine
with an RBF kernel, preceded by z-score standardization across training trials.
Decoding confidence was defined as the model’s posterior probability assigned
to the preferred cue for the held-out trial at that time bin, yielding a
trial-by-time confidence map per session.
Leave-one-out decoding provides an unbiased, trial-specific confidence estimate
while using nearly all available trials for training.

### Shuffle-based null distribution (optional)

When requested, a null distribution was generated by repeating the decoding
procedure with shuffled class labels within the balanced training set. For each
test trial and time bin, the shuffle procedure produced a distribution of
confidence values under the null hypothesis of no relationship between neural
activity and cue label.
This shuffle procedure preserves the marginal structure of neural activity and
class balance while breaking the mapping between activity and cue identity.

## 3) On- and off-state extraction (cluster-based inference on confidence maps)

The goal of this stage is to convert the decoding confidence maps into discrete
periods of significantly elevated evidence (*on-states*) and null-like evidence
(*off-states*) during the delay period.
On/off state inference was only performed for sessions with a shuffle-derived
null distribution available from the decoding stage.

### Z-score normalization relative to the shuffle null

For sessions with a shuffle-derived null distribution, decoding confidence was
converted to z-scores at each trial and time bin by subtracting the shuffle
mean and dividing by the shuffle standard deviation. This normalization yields
a trial-by-time z-score map reflecting evidence for the preferred cue relative
to chance expectations.
Using a bin-wise null accounts for time-varying calibration of classifier
confidence.

### Cluster definition and cluster mass

Clusters were defined as contiguous runs of suprathreshold bins along the time
axis *within a trial*. For each cluster, cluster mass was defined as the sum of
z-scores across all bins in that cluster.

### On-state detection (maximum cluster-mass correction)

On-state candidates were defined as clusters where z exceeded 1.96. To control
for multiple comparisons across the full trial-by-time map, we used a
shuffle-based maximum cluster-mass procedure: for each shuffle, we computed a
z-map and extracted the maximum cluster mass across all suprathreshold clusters.
The 95th percentile of this null distribution of maximum cluster masses served
as the cluster-level significance cutoff (α=0.05). Observed clusters whose mass
exceeded this cutoff were labeled as on-states.

### Off-state detection (null-like clusters)

Off-state candidates were defined as clusters of *non-significant* z-scores,
using either a two-tailed criterion (|z| ≤ 1.96) or a one-tailed criterion (z ≤
1.96). Candidate clusters were required to span at least 5 time bins to reduce
spurious short segments (i.e., ≥5 consecutive 10-ms steps).

To identify clusters consistent with the null, we computed a null distribution
of cluster masses by applying the same candidate definition to each shuffle’s
z-map and collecting the masses of all clusters meeting the size criterion.
Observed candidate clusters were retained as off-states if their cluster masses
fell within the central 95% of this null mass distribution (2.5th–97.5th
percentiles), operationalizing “off” periods as segments statistically
indistinguishable from the shuffle-derived null.

### Delay-period state durations

On- and off-state durations were summarized within the delay window (500–1400
ms). For each retained cluster, its time span was intersected with the delay
window, and the resulting duration contributed to histogram summaries.
