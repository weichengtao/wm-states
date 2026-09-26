import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat

from scripts.next.session_inputs import (
    SessionInputs,
    load_session_inputs,
    validate_state_trial_ids,
    validate_trial_ids,
)


class SessionInputLoadingTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "recorded.mat"
        self.arrays = {
            "spks": np.arange(24, dtype=np.int16).reshape(4, 3, 2),
            "tc": np.asarray([0.0, 1.0, 2.0]),
            "cueAngIdx": np.asarray([1, 5, 1, 1]),
            "isCorr": np.asarray([1, 1, 0, 1]),
        }

    def test_loads_matching_arrays_without_changing_spike_values(self):
        savemat(self.path, self.arrays)
        loaded = load_session_inputs(self.path)

        self.assertEqual(loaded.session, "recorded")
        np.testing.assert_array_equal(loaded.spikes, self.arrays["spks"])
        self.assertEqual(loaded.spikes.dtype, np.int16)
        np.testing.assert_array_equal(loaded.times_ms, [0, 1, 2])
        np.testing.assert_array_equal(loaded.cue_labels, [1, 5, 1, 1])
        np.testing.assert_array_equal(loaded.correct_trials, [True, True, False, True])

    def test_reports_missing_input_with_session(self):
        with self.assertRaisesRegex(FileNotFoundError, "Session chosen: missing session data"):
            load_session_inputs(self.path, session="chosen")

    def test_reports_missing_mat_variable_with_session(self):
        del self.arrays["isCorr"]
        savemat(self.path, self.arrays)
        with self.assertRaisesRegex(ValueError, "Session recorded: missing required MAT variable 'isCorr'"):
            load_session_inputs(self.path)

    def test_rejects_misaligned_trial_metadata(self):
        self.arrays["isCorr"] = np.asarray([1, 1])
        savemat(self.path, self.arrays)
        with self.assertRaisesRegex(ValueError, "Session recorded:.*inconsistent trial/time/cell"):
            load_session_inputs(self.path)

    def test_rejects_irregular_timestamps(self):
        self.arrays["tc"] = np.asarray([0.0, 1.0, 3.0])
        savemat(self.path, self.arrays)
        with self.assertRaisesRegex(ValueError, "Session recorded:.*uniformly sampled"):
            load_session_inputs(self.path)


class CachedTrialAlignmentTest(unittest.TestCase):
    def setUp(self):
        self.inputs = SessionInputs(
            session="recorded",
            spikes=np.zeros((5, 3, 2)),
            times_ms=np.asarray([0, 1, 2]),
            cue_labels=np.asarray([1, 5, 1, 1, 1]),
            correct_trials=np.asarray([True, True, False, True, True]),
        )

    def test_preserves_cached_row_order_and_accepts_integral_float_vector(self):
        trial_ids = validate_state_trial_ids(self.inputs, np.asarray([[4.0, 0.0, 3.0]]), 1)
        np.testing.assert_array_equal(trial_ids, [4, 0, 3])
        self.assertEqual(trial_ids.dtype, np.int64)

    def test_accepts_empty_trial_vector(self):
        trial_ids = validate_state_trial_ids(self.inputs, np.asarray([]), 1)
        self.assertEqual(trial_ids.shape, (0,))

    def test_rejects_fractional_nonfinite_and_nonnumeric_ids_before_casting(self):
        for values in ([0.5], [np.nan], [np.inf], [True], ["0"], [1j]):
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, "Session recorded:.*finite integers"):
                    validate_state_trial_ids(self.inputs, np.asarray(values), 1)

    def test_rejects_out_of_bounds_and_unrepresentable_ids(self):
        for values in ([-1], [5], [float(2**63)], np.asarray([2**64 - 1], dtype=np.uint64)):
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, "Session recorded:.*out of range"):
                    validate_state_trial_ids(self.inputs, np.asarray(values), 1)

    def test_rejects_duplicate_ids(self):
        with self.assertRaisesRegex(ValueError, "Session recorded:.*not unique"):
            validate_state_trial_ids(self.inputs, np.asarray([3, 0, 3]), 1)

    def test_rejects_matrix_instead_of_flattening_state_rows(self):
        with self.assertRaisesRegex(ValueError, "Session recorded:.*must be a vector"):
            validate_state_trial_ids(self.inputs, np.asarray([[0, 1], [3, 4]]), 1)

    def test_rejects_incorrect_trial(self):
        with self.assertRaisesRegex(ValueError, "Session recorded:.*not all correct"):
            validate_state_trial_ids(self.inputs, np.asarray([0, 2]), 1)

    def test_rejects_wrong_cue(self):
        with self.assertRaisesRegex(ValueError, "Session recorded:.*do not all use the preferred cue"):
            validate_state_trial_ids(self.inputs, np.asarray([0, 1]), 1)

    def test_rejects_fractional_preferred_cue_before_matching_trials(self):
        with self.assertRaisesRegex(ValueError, "Session recorded: preferred cue.*integer cue"):
            validate_state_trial_ids(self.inputs, np.asarray([0, 3]), 1.5)

    def test_standalone_id_validation_preserves_unsorted_ids(self):
        np.testing.assert_array_equal(
            validate_trial_ids(np.asarray([8, 2, 5]), session="recorded"),
            [8, 2, 5],
        )


if __name__ == "__main__":
    unittest.main()
