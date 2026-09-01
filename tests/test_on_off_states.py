import unittest

import numpy as np

from scripts.on_off_states import max_off_state_duration_per_trial, state_mask_for_cache


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
