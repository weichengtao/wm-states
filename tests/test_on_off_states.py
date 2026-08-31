import unittest

import numpy as np

from scripts.on_off_states import max_off_state_duration_per_trial


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


if __name__ == '__main__':
    unittest.main()
