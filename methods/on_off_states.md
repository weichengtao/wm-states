# On- and off-state extraction

## Inputs
On/off states were derived from the decoding confidence outputs
(`decoding_confidence.pkl`). For each session and preferred cue, we used the
trial-by-time decoding confidence matrix and, when available, the corresponding
null distribution from label shuffles.

## Z-score normalization
We converted decoding confidence to a z-score map by subtracting the mean and
dividing by the standard deviation of the shuffle-based null distribution at
each trial and time bin. Z-scores were computed only when null shuffles were
available; otherwise on/off states were not inferred. The binning used the same
time grid as decoding (50 ms windows stepped every 10 ms).

## Cluster definition
Clusters were identified on the z-score map using 2D connected components with
4-connectivity along the time axis (adjacent bins within a trial). Cluster mass
was defined as the sum of z-scores within a cluster.

## On-state detection
On-state candidates were defined as clusters where z-scores exceeded 1.96. We
computed the cluster mass for each candidate and compared it to a null
distribution of maximum cluster masses obtained from each shuffle. Clusters
with masses above the 95th percentile of the null maximum-mass distribution
were retained as on-states.

## Off-state detection
Off-state candidates were defined as clusters whose z-scores fell within a
non-significant range, using either a two-tailed criterion (|z| <= 1.96) or a
one-tailed criterion (z <= 1.96). Candidate clusters were required to contain
at least 5 bins. For each candidate we computed cluster mass and compared it to
a null distribution of cluster masses from shuffle-derived z-maps, retaining
clusters whose masses fell within the 2.5th to 97.5th percentile of the null
mass distribution. These retained clusters were labeled as off-states.

## Delay-period state durations
On- and off-state durations were summarized within the delay window (500 to
1400 ms). For each retained cluster, we extracted its time span, clipped it to
the delay window, and computed the resulting duration for histogram summaries.
