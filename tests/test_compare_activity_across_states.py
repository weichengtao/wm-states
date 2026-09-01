import unittest

import numpy as np

from scripts.compare_activity_across_states import (
    SessionActivity,
    activity_point_categories,
    balance_trial_groups,
    compute_binned_firing_rates,
    fixed_width_bin_edges,
    normalize_balanced_activity,
    plot_session_activity,
    plot_session_activity_marginal_histograms,
    plot_session_activity_pairwise,
    preferred_pev_cells,
    session_cell_groups,
    top_preferred_pev_cells,
)


class TopPreferredPevCellsTest(unittest.TestCase):
    def test_ranks_only_cells_preferred_for_session_cue(self):
        selection = {
            "session": "example",
            "cell_properties": {
                "cell_idx": np.asarray([10, 11, 12, 13, 14, 15]),
                "mean_pref_test": np.asarray([7, 3, 7, 7, 7, 3]),
                "mean_pev_test": np.asarray([4.0, 99.0, 8.0, 6.0, 2.0, 98.0]),
            },
        }

        cell_ids, pev = top_preferred_pev_cells(selection, preferred_cue=7)

        np.testing.assert_array_equal(cell_ids, [12, 13, 10])
        np.testing.assert_array_equal(pev, [8.0, 6.0, 4.0])
        all_cell_ids, all_pev = preferred_pev_cells(selection, preferred_cue=7)
        np.testing.assert_array_equal(all_cell_ids, [12, 13, 10, 14])
        np.testing.assert_array_equal(all_pev, [8.0, 6.0, 4.0, 2.0])

    def test_returns_all_available_cells_when_fewer_than_three(self):
        selection = {
            "session": "example",
            "cell_properties": {
                "cell_idx": np.asarray([1, 2, 3]),
                "mean_pref_test": np.asarray([7, 7, 3]),
                "mean_pev_test": np.asarray([3.0, 2.0, 100.0]),
            },
        }

        cell_ids, pev = top_preferred_pev_cells(selection, preferred_cue=7)

        np.testing.assert_array_equal(cell_ids, [1, 2])
        np.testing.assert_array_equal(pev, [3.0, 2.0])

    def test_returns_empty_arrays_when_no_preferred_cells_are_available(self):
        selection = {
            "session": "example",
            "cell_properties": {
                "cell_idx": np.asarray([1, 2]),
                "mean_pref_test": np.asarray([3, 3]),
                "mean_pev_test": np.asarray([5.0, 4.0]),
            },
        }

        cell_ids, pev = top_preferred_pev_cells(selection, preferred_cue=7)

        self.assertEqual(cell_ids.size, 0)
        self.assertEqual(pev.size, 0)

    def test_builds_disjoint_population_groups(self):
        selection = {
            "session": "example",
            "cell_idx_stationary": np.asarray([10, 11, 12, 13, 20, 21]),
            "cell_properties": {
                "cell_idx": np.asarray([10, 11, 12, 13]),
                "mean_pref_test": np.asarray([7, 3, 7, 5]),
                "mean_pev_test": np.asarray([2.0, 9.0, 4.0, 8.0]),
            },
        }

        groups = session_cell_groups(selection, preferred_cue=7)

        np.testing.assert_array_equal(groups["preferred"], [12, 10])
        np.testing.assert_array_equal(groups["selective_nonpreferred"], [11, 13])
        np.testing.assert_array_equal(groups["stationary_nonselective"], [20, 21])


class BalanceTrialGroupsTest(unittest.TestCase):
    def test_balances_reproducibly_and_preserves_preferred_positions(self):
        preferred = np.asarray([10, 20, 30, 40, 50])
        opposite = np.asarray([60, 70, 80])

        first = balance_trial_groups(preferred, opposite, seed=9)
        second = balance_trial_groups(preferred, opposite, seed=9)

        for first_value, second_value in zip(first, second):
            np.testing.assert_array_equal(first_value, second_value)
        positions, preferred_ids, opposite_ids = first
        self.assertEqual(preferred_ids.size, 3)
        self.assertEqual(opposite_ids.size, 3)
        np.testing.assert_array_equal(preferred_ids, preferred[positions])


class BinnedFiringRatesTest(unittest.TestCase):
    def test_computes_trial_bin_cell_rates(self):
        spikes = np.zeros((2, 4, 2), dtype=float)
        spikes[0, :, 0] = [1, 1, 2, 2]
        spikes[1, :, 1] = [0, 2, 0, 4]

        rates = compute_binned_firing_rates(
            spikes,
            times_ms=np.asarray([0, 10, 20, 30]),
            trial_ids=np.asarray([0, 1]),
            cell_ids=np.asarray([0, 1]),
            bin_starts=np.asarray([0, 20]),
            bin_width_ms=20,
        )

        expected = np.asarray(
            [
                [[100, 0], [200, 0]],
                [[0, 100], [0, 200]],
            ],
            dtype=float,
        )
        np.testing.assert_array_equal(rates, expected)


class NormalizeBalancedActivityTest(unittest.TestCase):
    def test_normalizes_each_bin_and_cell_across_both_cues(self):
        preferred = np.asarray(
            [
                [[1.0, 5.0], [10.0, 4.0]],
                [[3.0, 5.0], [14.0, 4.0]],
            ]
        )
        opposite = np.asarray(
            [
                [[5.0, 5.0], [18.0, 4.0]],
                [[7.0, 5.0], [22.0, 4.0]],
            ]
        )

        preferred_z, opposite_z = normalize_balanced_activity(preferred, opposite)
        combined = np.concatenate([preferred_z, opposite_z], axis=0)

        np.testing.assert_allclose(combined[:, :, 0].mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(combined[:, :, 0].std(axis=0), 1.0, atol=1e-12)
        np.testing.assert_array_equal(combined[:, :, 1], 0.0)

    def test_accepts_zero_cell_activity_for_placeholder_sessions(self):
        empty = np.empty((2, 3, 0), dtype=float)

        preferred_z, opposite_z = normalize_balanced_activity(empty, empty.copy())

        self.assertEqual(preferred_z.shape, (2, 3, 0))
        self.assertEqual(opposite_z.shape, (2, 3, 0))


class FixedWidthBinEdgesTest(unittest.TestCase):
    def test_uses_aligned_point_two_five_width_bins(self):
        edges = fixed_width_bin_edges(np.asarray([-0.31, 0.46]))

        np.testing.assert_allclose(edges, [-0.5, -0.25, 0.0, 0.25, 0.5])
        np.testing.assert_allclose(np.diff(edges), 0.25)


class PairwisePlotTest(unittest.TestCase):
    def test_plots_each_cell_pair_with_all_four_categories(self):
        activity = SessionActivity(
            session="example",
            preferred_cue=7,
            opposite_cue=3,
            cell_ids=np.asarray([10, 20, 30]),
            cell_pev=np.asarray([8.0, 7.0, 6.0]),
            delay_bin_starts=np.asarray([500, 510]),
            preferred_activity=np.arange(12, dtype=float).reshape(2, 2, 3),
            opposite_activity=np.arange(12, 24, dtype=float).reshape(2, 2, 3),
            on_state_mask=np.asarray([[True, False], [False, True]]),
            off_state_mask=np.asarray([[False, True], [True, False]]),
            preferred_trial_ids=np.asarray([1, 2]),
            opposite_trial_ids=np.asarray([3, 4]),
        )

        fig = plot_session_activity_pairwise(activity)

        self.assertEqual(len(fig.axes), 3)
        for ax in fig.axes:
            self.assertEqual(len(ax.collections), 4)
            self.assertEqual(
                [collection.get_zorder() for collection in ax.collections],
                [4, 3, 2, 1],
            )
        self.assertIn("Cell 10", fig.axes[0].get_xlabel())
        self.assertIn("Cell 20", fig.axes[0].get_ylabel())
        self.assertIn("Cell 30", fig.axes[2].get_ylabel())
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_adapts_all_plot_layouts_to_fewer_than_three_cells(self):
        expected_marginal_axes = {2: 10, 1: 8, 0: 6}
        for num_cells in (2, 1, 0):
            activity = SessionActivity(
                session=f"example-{num_cells}",
                preferred_cue=7,
                opposite_cue=3,
                cell_ids=np.arange(10, 10 + num_cells),
                cell_pev=np.arange(num_cells, 0, -1, dtype=float),
                delay_bin_starts=np.asarray([500, 510]),
                preferred_activity=np.arange(
                    2 * 2 * num_cells,
                    dtype=float,
                ).reshape(2, 2, num_cells),
                opposite_activity=np.arange(
                    2 * 2 * num_cells,
                    4 * 2 * num_cells,
                    dtype=float,
                ).reshape(2, 2, num_cells),
                on_state_mask=np.asarray([[True, False], [False, True]]),
                off_state_mask=np.asarray([[False, True], [True, False]]),
                preferred_trial_ids=np.asarray([1, 2]),
                opposite_trial_ids=np.asarray([3, 4]),
            )

            activity_fig = plot_session_activity(activity)
            pairwise_fig = plot_session_activity_pairwise(activity)
            marginal_fig = plot_session_activity_marginal_histograms(activity)

            self.assertEqual(len(activity_fig.axes), 1)
            self.assertEqual(len(pairwise_fig.axes), 1)
            self.assertEqual(
                len(marginal_fig.axes),
                expected_marginal_axes[num_cells],
            )
            if num_cells > 0:
                self.assertEqual(len(activity_fig.axes[0].collections), 4)
                self.assertEqual(len(pairwise_fig.axes[0].collections), 4)
            else:
                self.assertFalse(activity_fig.axes[0].axison)
                self.assertFalse(pairwise_fig.axes[0].axison)
                self.assertFalse(marginal_fig.axes[0].axison)

            import matplotlib.pyplot as plt

            plt.close(activity_fig)
            plt.close(pairwise_fig)
            plt.close(marginal_fig)

    def test_subsamples_each_color_group_deterministically(self):
        activity = SessionActivity(
            session="example",
            preferred_cue=7,
            opposite_cue=3,
            cell_ids=np.asarray([10, 20, 30]),
            cell_pev=np.asarray([8.0, 7.0, 6.0]),
            delay_bin_starts=np.asarray([500, 510]),
            preferred_activity=np.arange(12, dtype=float).reshape(2, 2, 3),
            opposite_activity=np.arange(12, 24, dtype=float).reshape(2, 2, 3),
            on_state_mask=np.asarray([[True, False], [False, True]]),
            off_state_mask=np.asarray([[False, True], [True, False]]),
            preferred_trial_ids=np.asarray([1, 2]),
            opposite_trial_ids=np.asarray([3, 4]),
            preferred_population_mean_activity=np.asarray(
                [[0.5, 1.0], [1.5, 2.0]]
            ),
            opposite_population_mean_activity=np.asarray(
                [[-0.5, -1.0], [-1.5, -2.0]]
            ),
            preferred_population_cell_count=5,
        )

        first = activity_point_categories(
            activity,
            max_points_per_color_group=1,
            seed=17,
        )
        second = activity_point_categories(
            activity,
            max_points_per_color_group=1,
            seed=17,
        )

        self.assertEqual([category[3] for category in first], [2, 2, 4, 4])
        for first_category, second_category in zip(first, second):
            self.assertEqual(first_category[0].shape, (1, 3))
            np.testing.assert_array_equal(first_category[0], second_category[0])

        fig = plot_session_activity_pairwise(
            activity,
            max_points_per_color_group=1,
            point_seed=17,
        )
        for ax in fig.axes:
            self.assertTrue(
                all(collection.get_offsets().shape[0] == 1 for collection in ax.collections)
            )
        legend_labels = [text.get_text() for text in fig.axes[0].get_legend().texts]
        self.assertTrue(all("shown=1" in label for label in legend_labels))
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_can_hide_opposite_cue_points(self):
        activity = SessionActivity(
            session="example",
            preferred_cue=7,
            opposite_cue=3,
            cell_ids=np.asarray([10, 20, 30]),
            cell_pev=np.asarray([8.0, 7.0, 6.0]),
            delay_bin_starts=np.asarray([500]),
            preferred_activity=np.arange(6, dtype=float).reshape(2, 1, 3),
            opposite_activity=np.arange(6, 12, dtype=float).reshape(2, 1, 3),
            on_state_mask=np.asarray([[True], [False]]),
            off_state_mask=np.asarray([[False], [True]]),
            preferred_trial_ids=np.asarray([1, 2]),
            opposite_trial_ids=np.asarray([3, 4]),
        )

        fig = plot_session_activity_pairwise(
            activity,
            hide_opposite_cue_points=True,
        )

        for ax in fig.axes:
            self.assertEqual(len(ax.collections), 3)
        legend_labels = [text.get_text() for text in fig.axes[0].get_legend().texts]
        self.assertFalse(any("Opposite cue" in label for label in legend_labels))
        three_dimensional_fig = plot_session_activity(
            activity,
            hide_opposite_cue_points=True,
        )
        self.assertEqual(len(three_dimensional_fig.axes[0].collections), 3)
        three_dimensional_legend_labels = [
            text.get_text()
            for text in three_dimensional_fig.axes[0].get_legend().texts
        ]
        self.assertFalse(
            any("Opposite cue" in label for label in three_dimensional_legend_labels)
        )
        hidden_all_preferred_fig = plot_session_activity_pairwise(
            activity,
            hide_all_preferred_cue_points=True,
        )
        self.assertTrue(
            all(len(ax.collections) == 3 for ax in hidden_all_preferred_fig.axes)
        )
        hidden_all_preferred_labels = [
            text.get_text()
            for text in hidden_all_preferred_fig.axes[0].get_legend().texts
        ]
        self.assertFalse(
            any("all delay bins" in label for label in hidden_all_preferred_labels)
        )
        import matplotlib.pyplot as plt

        plt.close(fig)
        plt.close(three_dimensional_fig)
        plt.close(hidden_all_preferred_fig)

    def test_adds_three_population_means_to_marginal_histograms(self):
        activity = SessionActivity(
            session="example",
            preferred_cue=7,
            opposite_cue=3,
            cell_ids=np.asarray([10, 20, 30]),
            cell_pev=np.asarray([8.0, 7.0, 6.0]),
            delay_bin_starts=np.asarray([500, 510]),
            preferred_activity=np.arange(12, dtype=float).reshape(2, 2, 3),
            opposite_activity=np.arange(12, 24, dtype=float).reshape(2, 2, 3),
            on_state_mask=np.asarray([[True, False], [False, True]]),
            off_state_mask=np.asarray([[False, True], [True, False]]),
            preferred_trial_ids=np.asarray([1, 2]),
            opposite_trial_ids=np.asarray([3, 4]),
            preferred_population_mean_activity=np.asarray(
                [[0.5, 1.0], [1.5, 2.0]]
            ),
            opposite_population_mean_activity=np.asarray(
                [[-0.5, -1.0], [-1.5, -2.0]]
            ),
            preferred_population_cell_count=5,
            population_mean_activities={
                "preferred": (
                    np.asarray([[0.5, 1.0], [1.5, 2.0]]),
                    np.asarray([[-0.5, -1.0], [-1.5, -2.0]]),
                    5,
                ),
                "selective_nonpreferred": (
                    np.asarray([[0.2, 0.4], [0.6, 0.8]]),
                    np.asarray([[-0.2, -0.4], [-0.6, -0.8]]),
                    4,
                ),
                "stationary_nonselective": (
                    np.asarray([[0.1, 0.2], [0.3, 0.4]]),
                    np.asarray([[-0.1, -0.2], [-0.3, -0.4]]),
                    6,
                ),
            },
        )

        fig = plot_session_activity_marginal_histograms(activity)
        hidden_opposite_fig = plot_session_activity_marginal_histograms(
            activity,
            hide_opposite_cue_points=True,
        )
        hidden_all_preferred_fig = plot_session_activity_marginal_histograms(
            activity,
            hide_all_preferred_cue_points=True,
        )

        self.assertEqual(len(fig.axes), 12)
        histogram_axes = fig.axes[:6]
        ecdf_axes = fig.axes[6:]
        self.assertTrue(all(len(ax.patches) == 4 for ax in histogram_axes))
        self.assertTrue(
            all(
                [patch.get_zorder() for patch in ax.patches] == [4, 3, 2, 1]
                for ax in histogram_axes
            )
        )
        for ax in histogram_axes:
            self.assertEqual(len(ax.lines), 1)
            zero_line = ax.lines[0]
            np.testing.assert_array_equal(zero_line.get_xdata(), [0, 0])
            self.assertEqual(zero_line.get_color(), "black")
            self.assertEqual(zero_line.get_linestyle(), "--")
            self.assertEqual(zero_line.get_zorder(), 0)
        self.assertTrue(all(len(ax.lines) == 5 for ax in ecdf_axes))
        self.assertTrue(
            all(
                [line.get_zorder() for line in ax.lines[1:]] == [4, 3, 2, 1]
                for ax in ecdf_axes
            )
        )
        for ax in ecdf_axes:
            zero_line = ax.lines[0]
            np.testing.assert_array_equal(zero_line.get_xdata(), [0, 0])
            self.assertEqual(zero_line.get_color(), "black")
            self.assertEqual(zero_line.get_linestyle(), "--")
            self.assertEqual(zero_line.get_zorder(), 0)
        self.assertTrue(
            all(
                np.isclose(line.get_ydata()[-1], 1.0)
                for ax in ecdf_axes
                for line in ax.lines[1:]
            )
        )
        self.assertTrue(
            all(len(ax.patches) == 3 for ax in hidden_opposite_fig.axes[:6])
        )
        self.assertTrue(
            all(len(ax.lines) == 4 for ax in hidden_opposite_fig.axes[6:])
        )
        self.assertTrue(
            all(len(ax.patches) == 3 for ax in hidden_all_preferred_fig.axes[:6])
        )
        self.assertTrue(
            all(len(ax.lines) == 4 for ax in hidden_all_preferred_fig.axes[6:])
        )
        hidden_all_preferred_labels = [
            text.get_text()
            for text in hidden_all_preferred_fig.axes[0].get_legend().texts
        ]
        self.assertFalse(
            any("all delay bins" in label for label in hidden_all_preferred_labels)
        )
        self.assertIn("Cell 10", fig.axes[6].get_xlabel())
        self.assertIn("Cell 30", fig.axes[8].get_xlabel())
        self.assertIn("all 5 preferred cells", fig.axes[9].get_xlabel())
        self.assertIn("4 selective non-preferred cells", fig.axes[10].get_xlabel())
        self.assertIn("6 stationary non-selective cells", fig.axes[11].get_xlabel())
        import matplotlib.pyplot as plt

        plt.close(fig)
        plt.close(hidden_opposite_fig)
        plt.close(hidden_all_preferred_fig)


if __name__ == "__main__":
    unittest.main()
