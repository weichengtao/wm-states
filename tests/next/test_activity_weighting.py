import unittest
from pathlib import Path

import numpy as np

from scripts.next.activity_weighting import (
    cell_group_activity_weights,
    mean_cell_activity,
    weighting_policy,
    weighting_subdir,
)
from scripts.next.prepare_data_for_mixedlm import _group_features


class CellGroupActivityWeightsTest(unittest.TestCase):
    def setUp(self):
        self.selection = {
            "cell_properties": {
                "cell_idx": np.asarray([10, 11, 12, 13]),
                "mean_pev_test": np.asarray([2.0, 8.0, 6.0, 4.0]),
            }
        }
        self.groups = {
            "preferred": np.asarray([12, 10]),
            "selective_nonpreferred": np.asarray([11, 13]),
            "stationary_nonselective": np.asarray([20, 21]),
        }

    def test_aligns_pev_to_selective_cell_order_and_keeps_stationary_equal(self):
        weights = cell_group_activity_weights(
            self.selection,
            self.groups,
            pev_weighted_average=True,
        )

        np.testing.assert_array_equal(weights["preferred"], [6.0, 2.0])
        np.testing.assert_array_equal(
            weights["selective_nonpreferred"], [8.0, 4.0]
        )
        self.assertIsNone(weights["stationary_nonselective"])
        self.assertEqual(
            weighting_policy(True)["stationary_nonselective"], "equal"
        )

    def test_equal_mode_does_not_require_pev(self):
        weights = cell_group_activity_weights({}, self.groups, False)

        self.assertTrue(all(value is None for value in weights.values()))

    def test_rejects_invalid_selective_pev(self):
        self.selection["cell_properties"]["mean_pev_test"][2] = np.nan

        with self.assertRaisesRegex(ValueError, "Non-finite"):
            cell_group_activity_weights(self.selection, self.groups, True)


class MeanCellActivityTest(unittest.TestCase):
    def test_computes_weighted_and_equal_means(self):
        activity = np.asarray([[1.0, 3.0], [4.0, 8.0]])

        np.testing.assert_allclose(mean_cell_activity(activity, axis=1), [2.0, 6.0])
        np.testing.assert_allclose(
            mean_cell_activity(activity, np.asarray([1.0, 3.0]), axis=1),
            [2.5, 7.0],
        )

    def test_group_features_weights_only_the_mean(self):
        normalized = np.asarray([[1.0, -1.0], [2.0, 4.0]])

        mean_activity, active_fraction = _group_features(
            normalized,
            active_threshold=0.0,
            activity_weights=np.asarray([1.0, 3.0]),
        )

        np.testing.assert_allclose(mean_activity, [-0.5, 3.5])
        np.testing.assert_allclose(active_fraction, [0.5, 1.0])



class WeightingOutputPathTest(unittest.TestCase):
    def test_only_weighted_mode_gets_a_subfolder(self):
        base = Path("mixedlm/prepared")

        self.assertEqual(weighting_subdir(base, False), base)
        self.assertEqual(
            weighting_subdir(base, True),
            Path("mixedlm/prepared/pev_weighted"),
        )


if __name__ == "__main__":
    unittest.main()
