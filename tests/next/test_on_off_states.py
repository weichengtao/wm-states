import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch
import warnings

import numpy as np

from scripts.next.on_off_states import max_off_state_duration_per_trial, state_mask_for_cache
from scripts.next import cache_io, on_off_states as states


class StateEdgeCasesTest(unittest.TestCase):
    def run_states(self, source, **settings):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_io.save([source], root / 'decode/decoding_confidence.pkl')
            with patch.object(states, 'save_figure'):
                states.main(states.Config(cache_dir=root, **settings))
            return cache_io.read(root / 'states/on_off_states.pkl')[0]

    def source(self):
        return dict(session='edge', cue=1, trial_idx=[0], time_bins=np.array([500, 510, 520]),
                    decoding_confidence=np.full((1, 3), .1),
                    decoding_confidence_null=np.array([[[.4, .4, .6, .6], [.6, .6, .4, .4], [.4, .4, .6, .6]]]),
                    preserve_null_time_structure=True, null_policy='test policy')

    def test_empty_null_off_cluster_distribution_is_error(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            with self.assertRaisesRegex(ValueError, 'observed OFF candidates exist but no null OFF clusters'):
                self.run_states(self.source(), cc_method_on='skipped',
                                z_threshold_off=0, cluster_size_threshold_off=3)

    def test_zero_variance_is_warned_and_unclassified_with_policy_saved(self):
        source = self.source()
        source['decoding_confidence_null'] = np.full((1, 3, 3), .1, dtype=np.float64)
        source['decoding_confidence'][:] = .9
        with self.assertWarnsRegex(RuntimeWarning, 'zero null variance'):
            result = self.run_states(source, cc_method_on='skipped', cc_method_off='skipped')
        self.assertFalse(result['on_state_mask'].any())
        self.assertFalse(result['off_state_mask'].any())
        self.assertTrue(result['preserve_null_time_structure'])
        self.assertEqual(result['decoding_null_policy'], 'test policy')

    def test_independent_nulls_and_degenerate_small_null_count_warn(self):
        source = self.source()
        source['preserve_null_time_structure'] = False
        source['decoding_confidence_null'] = source['decoding_confidence_null'][:, :, :3]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            self.run_states(source, cc_method_off='skipped')
        messages = [str(item.message) for item in caught]
        self.assertTrue(any('do not preserve temporal dependence' in message for message in messages))
        self.assertTrue(any('ON cluster cutoff is necessarily zero' in message for message in messages))

    def test_invalid_probabilities_empty_arrays_and_absent_delay_raise(self):
        for update, message in [
            ({'decoding_confidence': np.full((1, 3), 1.1)}, r'must be in \[0, 1\]'),
            ({'decoding_confidence': np.empty((0, 3))}, 'finite observed confidence'),
            ({'decoding_confidence_null': np.full((1, 3, 4), np.nan)}, 'finite null estimates'),
            ({'time_bins': np.array([0, 10, 20])}, 'no delay bins'),
        ]:
            with self.subTest(update=update), self.assertRaisesRegex(ValueError, message):
                self.run_states({**self.source(), **update}, cc_method_on='skipped', cc_method_off='skipped')

    def test_two_tailed_cutoff_warns_for_insufficient_per_tail_samples(self):
        source = self.source()
        source['decoding_confidence_null'] = np.tile(source['decoding_confidence_null'], (1, 1, 5))
        with self.assertWarnsRegex(RuntimeWarning, 'very low Monte Carlo precision'):
            self.run_states(source, cc_method_on='two_tailed', cc_method_off='skipped')

    def test_overlapping_candidate_thresholds_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'candidates would overlap'):
            states.Config(z_threshold_on=1, z_threshold_off=2)


class MaxOffStateDurationPerTrialTest(unittest.TestCase):
    def test_uses_longest_delay_run(self):
        bin_starts = np.arange(0, 80, 10)
        off_state_mask = np.array(
            [
                [True, True, False, True, True, True, False, True],
                [False, False, True, False, True, False, True, False],
                [False, False, False, False, False, False, False, False],
            ]
        )

        durations = max_off_state_duration_per_trial(
            off_state_mask,
            bin_starts,
            t_decode_step=10,
            delay_start=20,
            delay_end=60,
        )

        np.testing.assert_array_equal(durations, [30, 10, 0])

    def test_clips_runs_to_delay_period(self):
        bin_starts = np.arange(0, 60, 10)
        off_state_mask = np.array([[True, True, True, True, True, False]])

        durations = max_off_state_duration_per_trial(
            off_state_mask,
            bin_starts,
            t_decode_step=10,
            delay_start=20,
            delay_end=40,
        )

        np.testing.assert_array_equal(durations, [30])


class StateMaskForCacheTest(unittest.TestCase):
    def test_missing_state_is_cached_as_all_false(self):
        mask = state_mask_for_cache(None, (2, 3))

        self.assertEqual(mask.dtype, np.bool_)
        np.testing.assert_array_equal(mask, np.zeros((2, 3), dtype=bool))

    def test_rejects_misaligned_state_mask(self):
        with self.assertRaisesRegex(ValueError, "does not match expected"):
            state_mask_for_cache(np.zeros((3, 2), dtype=bool), (2, 3))


if __name__ == '__main__':
    unittest.main()
