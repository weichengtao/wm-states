"""Verify sensitivity reclassification independently of the production cache."""
import unittest
import numpy as np

from scripts.next.validate_state_confidence import off_mask, duration_arrays, paired_session_bootstrap


def manual_clusters(z, threshold, minimum):
    output = []
    for trial, row in enumerate(z):
        start = None
        for time in range(len(row) + 1):
            accepted = time < len(row) and row[time] <= threshold
            if accepted and start is None:
                start = time
            elif not accepted and start is not None:
                if time - start >= minimum:
                    output.append((trial, start, time, float(row[start:time].sum())))
                start = None
    return output


class StateConfidenceValidationTest(unittest.TestCase):
    def test_vectorized_null_clusters_match_independent_trial_loop(self):
        rng = np.random.default_rng(132)
        null = rng.uniform(.25, .75, (4, 35, 17))
        observed = rng.uniform(.1, .9, (4, 35))
        mean, sd = null.mean(-1), null.std(-1)
        z, zn = (observed - mean) / sd, (null - mean[..., None]) / sd[..., None]
        for threshold in (.5, .842, 1.282):
            for minimum in (1, 5):
                pooled = [cluster[-1] for draw in range(null.shape[-1])
                          for cluster in manual_clusters(zn[..., draw], threshold, minimum)]
                cutoff = np.percentile(pooled, 95)
                for mass_filter in (True, False):
                    expected = np.zeros(observed.shape, dtype=bool)
                    for trial, start, end, mass in manual_clusters(z, threshold, minimum):
                        if not mass_filter or mass <= cutoff:
                            expected[trial, start:end] = True
                    actual, _ = off_mask(observed, null, threshold=threshold,
                                         minimum_bins=minimum, mass_filter=mass_filter)
                    with self.subTest(threshold=threshold, minimum=minimum, mass_filter=mass_filter):
                        np.testing.assert_array_equal(actual, expected)

    def test_zero_variance_is_unclassified_even_for_low_observed_confidence(self):
        actual, details = off_mask(np.zeros((2, 3)), np.full((2, 3, 100), .1))
        self.assertFalse(actual.any())
        self.assertEqual(details['zero_variance_bins'], 6)

    def test_rejects_nonfinite_out_of_range_and_empty_probabilities(self):
        null = np.full((1, 3, 10), .5)
        for observed in (np.array([[.2, np.nan, .4]]), np.array([[.2, 1.1, .4]]),
                         np.empty((0, 3)), np.array([.2, .3, .4])):
            with self.subTest(shape=observed.shape), self.assertRaises(ValueError):
                off_mask(observed, null)
        with self.assertRaises(ValueError):
            off_mask(np.ones((1, 3)), null - 1)

    def test_one_tailed_OFF_includes_strongly_negative_z(self):
        null = np.broadcast_to([.4, .6, .4, .6], (1, 3, 4)).copy()
        mask, _ = off_mask(np.zeros((1, 3)), null)
        self.assertTrue(mask.all())

    def test_missing_null_clusters_raise_instead_of_fabricating_cutoff(self):
        null = np.array([[[.4, .6], [.6, .4], [.4, .6]]])
        with self.assertRaisesRegex(ValueError, 'No usable null OFF clusters'):
            off_mask(np.zeros((1, 3)), null, minimum_bins=3)

    def test_window_containment_changes_duration_grid_not_mask(self):
        times = np.arange(500, 1401, 10)
        mask = np.ones((1, len(times)), dtype=bool)
        baseline = duration_arrays(mask, times)
        contained = duration_arrays(mask, times, delay_end=1350)
        self.assertEqual(baseline['maximum_off_state_duration_ms'][0], 910)
        self.assertEqual(contained['maximum_off_state_duration_ms'][0], 860)
        self.assertEqual(contained['total_off_state_duration_ms'][0], 860)

    def test_session_bootstrap_uses_session_equal_paired_differences(self):
        runs = {}
        for name, scores in [('next_run_001', [1., 2., 3.]), ('next_run_005', [0., 1., 2.])]:
            runs[name] = {'per_session': {str(i): {'observed_delay': {'brier': x, 'log_loss': x}}
                                         for i, x in enumerate(scores)}}
        evidence = {'runs': runs, 'comparisons': {'next_run_005_vs_next_run_001': {}}}
        result = paired_session_bootstrap(evidence, repeats=10)
        metric = result['next_run_005_vs_next_run_001']['brier']
        self.assertEqual(metric['session_equal_mean_other_minus_baseline'], -1.)
        self.assertEqual(metric['session_bootstrap_percentile_95'], [-1., -1.])
        self.assertEqual(metric['other_better_session_count'], 3)


if __name__ == '__main__':
    unittest.main()
