import unittest

import numpy as np

from scripts.next.compare_activity_across_states import (
    Config,
    CellActivityDimensions,
    PrincipalComponentDimensions,
    SessionActivity,
    activity_point_categories,
    apply_activity_normalization,
    balanced_activity_normalization_parameters,
    balance_trial_groups,
    compute_binned_firing_rates,
    compute_preferred_cell_principal_components,
    centered_histogram_bin_offsets,
    fixed_width_bin_edges,
    maximum_delay_off_state_mask,
    normalize_balanced_activity,
    plot_session_activity,
    plot_session_activity_marginal_histograms,
    plot_session_activity_pairwise,
    population_mean_point_categories,
    preferred_pev_cells,
    principal_component_session_activity,
    session_cell_groups,
    top_preferred_pev_cells,
)


class TopPreferredPevCellsTest(unittest.TestCase):
    def test_ranks_only_cells_preferred_for_session_cue(self):
        selection = {
            "session": "example",
            "cell_properties": {
                "cell_idx": np.asarray([10, 11, 12, 13, 14, 15]),
                "preferred_cue": np.asarray([7, 3, 7, 7, 7, 3]),
                "mean_selectivity_pev_pct": np.asarray([4.0, 99.0, 8.0, 6.0, 2.0, 98.0]),
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
                "preferred_cue": np.asarray([7, 7, 3]),
                "mean_selectivity_pev_pct": np.asarray([3.0, 2.0, 100.0]),
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
                "preferred_cue": np.asarray([3, 3]),
                "mean_selectivity_pev_pct": np.asarray([5.0, 4.0]),
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
                "preferred_cue": np.asarray([7, 3, 7, 5]),
                "mean_selectivity_pev_pct": np.asarray([2.0, 9.0, 4.0, 8.0]),
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


class MaximumDelayOffStateMaskTest(unittest.TestCase):
    def test_selects_longest_delay_overlap_and_keeps_one_contiguous_state(self):
        off_state_mask = np.asarray(
            [
                [False, True, True, False, True, True, False],
                [True, True, True, True, False, False, False],
            ]
        )
        delay_bins = np.asarray([False, True, True, True, True, False, False])

        maximum_mask = maximum_delay_off_state_mask(off_state_mask, delay_bins)

        expected = np.zeros_like(off_state_mask)
        expected[1, 1:4] = True
        np.testing.assert_array_equal(maximum_mask, expected)

    def test_resolves_equal_length_states_to_earliest_state(self):
        off_state_mask = np.asarray(
            [
                [False, True, True, False],
                [True, True, False, False],
            ]
        )

        maximum_mask = maximum_delay_off_state_mask(
            off_state_mask,
            np.ones(4, dtype=bool),
        )

        expected = np.zeros_like(off_state_mask)
        expected[0, 1:3] = True
        np.testing.assert_array_equal(maximum_mask, expected)


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

    def test_applies_balanced_reference_parameters_to_an_extra_trial(self):
        preferred = np.asarray([[[1.0], [10.0]], [[3.0], [14.0]]])
        opposite = np.asarray([[[5.0], [18.0]], [[7.0], [22.0]]])
        extra_trial = np.asarray([[[9.0], [26.0]]])

        means, stds = balanced_activity_normalization_parameters(
            preferred,
            opposite,
        )
        normalized = apply_activity_normalization(extra_trial, means, stds)

        np.testing.assert_allclose(
            normalized,
            (extra_trial - means) / stds,
        )


class PreferredCellPrincipalComponentsTest(unittest.TestCase):
    def test_fits_three_components_to_pooled_cues_and_projects_maximum_state(self):
        preferred = np.asarray(
            [
                [[4.0, 1.0, 0.0, 2.0], [3.0, 2.0, 1.0, 0.0]],
                [[2.0, 0.0, 3.0, 1.0], [1.0, 3.0, 2.0, 4.0]],
            ]
        )
        opposite = np.asarray(
            [
                [[-4.0, -1.0, 0.0, -2.0], [-3.0, -2.0, -1.0, 0.0]],
                [[-2.0, 0.0, -3.0, -1.0], [-1.0, -3.0, -2.0, -4.0]],
            ]
        )
        maximum_state = np.asarray([[5.0, 1.0, 2.0, 3.0]])

        projection = compute_preferred_cell_principal_components(
            preferred,
            opposite,
            maximum_state,
        )

        self.assertEqual(projection.source_cell_count, 4)
        self.assertEqual(projection.preferred_activity.shape, (2, 2, 3))
        self.assertEqual(projection.opposite_activity.shape, (2, 2, 3))
        self.assertEqual(projection.max_off_state_activity.shape, (1, 3))
        self.assertTrue(
            np.all(np.diff(projection.explained_variance_ratio) <= 0)
        )
        pooled_scores = np.concatenate(
            [
                projection.preferred_activity.reshape(-1, 3),
                projection.opposite_activity.reshape(-1, 3),
            ],
            axis=0,
        )
        np.testing.assert_allclose(pooled_scores.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(
            projection.max_off_state_activity,
            (maximum_state - projection.center) @ projection.components.T,
        )

    def test_adapts_to_fewer_preferred_cells_and_builds_a_plotting_view(self):
        preferred = np.arange(12, dtype=float).reshape(2, 3, 2)
        opposite = np.arange(12, 24, dtype=float).reshape(2, 3, 2)
        maximum_state = np.asarray([[24.0, 25.0]])
        projection = compute_preferred_cell_principal_components(
            preferred,
            opposite,
            maximum_state,
        )
        activity = SessionActivity(
            session="example",
            preferred_cue=7,
            opposite_cue=3,
            dimensions=CellActivityDimensions(
                cell_ids=np.asarray([10, 20]),
                selectivity_pev_pct=np.asarray([8.0, 7.0]),
            ),
            delay_bin_starts=np.asarray([500, 550, 600]),
            preferred_activity=preferred,
            opposite_activity=opposite,
            on_state_mask=np.asarray(
                [[True, False, True], [False, True, False]]
            ),
            off_state_mask=np.asarray(
                [[False, True, False], [True, False, True]]
            ),
            preferred_trial_ids=np.asarray([1, 2]),
            opposite_trial_ids=np.asarray([3, 4]),
            max_off_state_activity=maximum_state,
            principal_component_activity=projection,
        )

        pca_activity = principal_component_session_activity(activity)
        fig = plot_session_activity(pca_activity)

        self.assertFalse(Config().show_principal_components)
        self.assertEqual(pca_activity.activity_space, "principal_components")
        self.assertIsInstance(pca_activity.dimensions, PrincipalComponentDimensions)
        self.assertIsInstance(activity.dimensions, CellActivityDimensions)
        self.assertFalse(hasattr(pca_activity.dimensions, "selectivity_pev_pct"))
        self.assertFalse(hasattr(pca_activity.dimensions, "cell_ids"))
        self.assertEqual(pca_activity.activity_source_cell_count, 2)
        np.testing.assert_array_equal(pca_activity.dimensions.component_numbers, [1, 2])
        np.testing.assert_allclose(
            pca_activity.dimensions.explained_variance_ratio,
            projection.explained_variance_ratio,
        )
        self.assertIn("PC1 score", fig.axes[0].get_xlabel())
        self.assertIn("explained variance", fig.axes[0].get_xlabel())
        self.assertIn("PCA of 2 preferred cells", fig.axes[0].get_title())
        np.testing.assert_array_equal(activity.dimensions.cell_ids, [10, 20])
        np.testing.assert_array_equal(activity.dimensions.selectivity_pev_pct, [8.0, 7.0])
        self.assertEqual(population_mean_point_categories(pca_activity), (None, 0))
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_returns_empty_projection_when_no_preferred_cells_are_available(self):
        empty = np.empty((2, 3, 0), dtype=float)
        maximum_state = np.empty((1, 0), dtype=float)

        projection = compute_preferred_cell_principal_components(
            empty,
            empty.copy(),
            maximum_state,
        )

        self.assertEqual(projection.preferred_activity.shape, (2, 3, 0))
        self.assertEqual(projection.opposite_activity.shape, (2, 3, 0))
        self.assertEqual(projection.max_off_state_activity.shape, (1, 0))
        self.assertEqual(projection.explained_variance_ratio.size, 0)


class FixedWidthBinEdgesTest(unittest.TestCase):
    def test_uses_aligned_point_two_five_width_bins(self):
        edges = fixed_width_bin_edges(np.asarray([-0.31, 0.46]))

        np.testing.assert_allclose(edges, [-0.5, -0.25, 0.0, 0.25, 0.5])
        np.testing.assert_allclose(np.diff(edges), 0.25)

    def test_centers_histogram_offsets_around_shared_bins(self):
        offsets = centered_histogram_bin_offsets(
            category_count=3,
            bin_width=0.25,
            offset_fraction=0.2,
        )

        np.testing.assert_allclose(offsets, [-0.05, 0.0, 0.05])


class PairwisePlotTest(unittest.TestCase):
    def test_separates_state_and_cue_categories_in_each_cell_pair(self):
        activity = SessionActivity(
            session="example",
            preferred_cue=7,
            opposite_cue=3,
            dimensions=CellActivityDimensions(
                cell_ids=np.asarray([10, 20, 30]),
                selectivity_pev_pct=np.asarray([8.0, 7.0, 6.0]),
            ),
            delay_bin_starts=np.asarray([500, 510]),
            preferred_activity=np.arange(12, dtype=float).reshape(2, 2, 3),
            opposite_activity=np.arange(12, 24, dtype=float).reshape(2, 2, 3),
            on_state_mask=np.asarray([[True, False], [False, True]]),
            off_state_mask=np.asarray([[False, True], [True, False]]),
            preferred_trial_ids=np.asarray([1, 2]),
            opposite_trial_ids=np.asarray([3, 4]),
            max_off_state_activity=np.asarray(
                [[100.0, 101.0, 102.0], [103.0, 104.0, 105.0]]
            ),
        )

        default_state_fig = plot_session_activity_pairwise(
            activity,
            comparison="state",
        )
        state_fig = plot_session_activity_pairwise(
            activity,
            comparison="state",
            compare_with_max_off_state=True,
        )
        cue_fig = plot_session_activity_pairwise(activity, comparison="cue")

        self.assertFalse(Config().compare_with_max_off_state)
        self.assertTrue(
            all(len(ax.collections) == 2 for ax in default_state_fig.axes)
        )
        self.assertEqual(len(state_fig.axes), 3)
        for ax in state_fig.axes:
            self.assertEqual(len(ax.collections), 3)
            self.assertEqual(
                [collection.get_zorder() for collection in ax.collections],
                [4, 3, 5],
            )
        for ax in cue_fig.axes:
            self.assertEqual(len(ax.collections), 2)
            self.assertEqual(
                [collection.get_zorder() for collection in ax.collections],
                [2, 1],
            )
        self.assertIn("Cell 10", state_fig.axes[0].get_xlabel())
        self.assertIn("Cell 20", state_fig.axes[0].get_ylabel())
        self.assertIn("Cell 30", state_fig.axes[2].get_ylabel())
        import matplotlib.pyplot as plt

        plt.close(default_state_fig)
        plt.close(state_fig)
        plt.close(cue_fig)

    def test_adapts_all_plot_layouts_to_fewer_than_three_cells(self):
        expected_marginal_axes = {2: 10, 1: 8, 0: 6}
        for num_cells in (2, 1, 0):
            activity = SessionActivity(
                session=f"example-{num_cells}",
                preferred_cue=7,
                opposite_cue=3,
                dimensions=CellActivityDimensions(
                    cell_ids=np.arange(10, 10 + num_cells),
                    selectivity_pev_pct=np.arange(num_cells, 0, -1, dtype=float),
                ),
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
                max_off_state_activity=np.arange(
                    2 * num_cells,
                    dtype=float,
                ).reshape(2, num_cells),
            )

            activity_fig = plot_session_activity(
                activity,
                compare_with_max_off_state=True,
            )
            pairwise_fig = plot_session_activity_pairwise(
                activity,
                compare_with_max_off_state=True,
            )
            marginal_fig = plot_session_activity_marginal_histograms(
                activity,
                compare_with_max_off_state=True,
            )
            cue_activity_fig = plot_session_activity(activity, comparison="cue")
            cue_pairwise_fig = plot_session_activity_pairwise(
                activity,
                comparison="cue",
            )

            self.assertEqual(len(activity_fig.axes), 1)
            self.assertEqual(len(pairwise_fig.axes), 1)
            self.assertEqual(
                len(marginal_fig.axes),
                expected_marginal_axes[num_cells],
            )
            if num_cells > 0:
                self.assertEqual(len(activity_fig.axes[0].collections), 3)
                self.assertEqual(len(pairwise_fig.axes[0].collections), 3)
                self.assertEqual(len(cue_activity_fig.axes[0].collections), 2)
                self.assertEqual(len(cue_pairwise_fig.axes[0].collections), 2)
            else:
                self.assertFalse(activity_fig.axes[0].axison)
                self.assertFalse(pairwise_fig.axes[0].axison)
                self.assertFalse(marginal_fig.axes[0].axison)

            import matplotlib.pyplot as plt

            plt.close(activity_fig)
            plt.close(pairwise_fig)
            plt.close(marginal_fig)
            plt.close(cue_activity_fig)
            plt.close(cue_pairwise_fig)

    def test_subsamples_each_color_group_deterministically(self):
        activity = SessionActivity(
            session="example",
            preferred_cue=7,
            opposite_cue=3,
            dimensions=CellActivityDimensions(
                cell_ids=np.asarray([10, 20, 30]),
                selectivity_pev_pct=np.asarray([8.0, 7.0, 6.0]),
            ),
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
            max_off_state_activity=np.arange(30, 39, dtype=float).reshape(3, 3),
        )

        first = activity_point_categories(
            activity,
            compare_with_max_off_state=True,
            max_points_per_color_group=1,
            seed=17,
        )
        second = activity_point_categories(
            activity,
            compare_with_max_off_state=True,
            max_points_per_color_group=1,
            seed=17,
        )

        self.assertEqual([category[3] for category in first], [2, 2, 3])
        self.assertEqual(
            [category[0].shape for category in first],
            [(1, 3), (1, 3), (3, 3)],
        )
        for first_category, second_category in zip(first, second):
            np.testing.assert_array_equal(first_category[0], second_category[0])

        limited_first = activity_point_categories(
            activity,
            compare_with_max_off_state=True,
            max_points_per_color_group=1,
            max_points_per_max_off_state=1,
            seed=17,
        )
        limited_second = activity_point_categories(
            activity,
            compare_with_max_off_state=True,
            max_points_per_color_group=1,
            max_points_per_max_off_state=1,
            seed=17,
        )
        self.assertTrue(
            all(category[0].shape == (1, 3) for category in limited_first)
        )
        for first_category, second_category in zip(limited_first, limited_second):
            np.testing.assert_array_equal(first_category[0], second_category[0])

        cue_categories = activity_point_categories(
            activity,
            comparison="cue",
            max_points_per_color_group=1,
            seed=17,
        )
        self.assertEqual([category[3] for category in cue_categories], [4, 4])

        fig = plot_session_activity_pairwise(
            activity,
            compare_with_max_off_state=True,
            max_points_per_color_group=1,
            max_points_per_max_off_state=1,
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
            dimensions=CellActivityDimensions(
                cell_ids=np.asarray([10, 20, 30]),
                selectivity_pev_pct=np.asarray([8.0, 7.0, 6.0]),
            ),
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
            comparison="cue",
            hide_opposite_cue_points=True,
        )

        for ax in fig.axes:
            self.assertEqual(len(ax.collections), 1)
        legend_labels = [text.get_text() for text in fig.axes[0].get_legend().texts]
        self.assertFalse(any("Opposite cue" in label for label in legend_labels))
        three_dimensional_fig = plot_session_activity(
            activity,
            comparison="cue",
            hide_opposite_cue_points=True,
        )
        self.assertEqual(len(three_dimensional_fig.axes[0].collections), 1)
        three_dimensional_legend_labels = [
            text.get_text()
            for text in three_dimensional_fig.axes[0].get_legend().texts
        ]
        self.assertFalse(
            any("Opposite cue" in label for label in three_dimensional_legend_labels)
        )
        hidden_all_preferred_fig = plot_session_activity_pairwise(
            activity,
            comparison="cue",
            hide_all_preferred_cue_points=True,
        )
        self.assertTrue(
            all(len(ax.collections) == 1 for ax in hidden_all_preferred_fig.axes)
        )
        hidden_all_preferred_labels = [
            text.get_text()
            for text in hidden_all_preferred_fig.axes[0].get_legend().texts
        ]
        self.assertFalse(
            any(
                "Preferred cue: all delay bins" in label
                for label in hidden_all_preferred_labels
            )
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
            dimensions=CellActivityDimensions(
                cell_ids=np.asarray([10, 20, 30]),
                selectivity_pev_pct=np.asarray([8.0, 7.0, 6.0]),
            ),
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
            max_off_state_activity=np.asarray(
                [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]
            ),
            max_off_state_population_mean_activities={
                "preferred": np.asarray([0.8, 1.1]),
                "selective_nonpreferred": np.asarray([0.3, 0.7]),
                "stationary_nonselective": np.asarray([0.15, 0.35]),
            },
        )

        fig = plot_session_activity_marginal_histograms(
            activity,
            compare_with_max_off_state=True,
        )
        cue_fig = plot_session_activity_marginal_histograms(
            activity,
            comparison="cue",
        )
        hidden_opposite_fig = plot_session_activity_marginal_histograms(
            activity,
            comparison="cue",
            hide_opposite_cue_points=True,
        )
        hidden_all_preferred_fig = plot_session_activity_marginal_histograms(
            activity,
            comparison="cue",
            hide_all_preferred_cue_points=True,
        )

        self.assertEqual(len(fig.axes), 12)
        histogram_axes = fig.axes[:6]
        ecdf_axes = fig.axes[6:]
        self.assertTrue(all(len(ax.patches) == 3 for ax in histogram_axes))
        self.assertTrue(
            all(
                [patch.get_zorder() for patch in ax.patches] == [4, 3, 5]
                for ax in histogram_axes
            )
        )
        for ax in histogram_axes:
            first_bin_edges = [
                patch.get_path().vertices[0, 0] for patch in ax.patches
            ]
            np.testing.assert_allclose(np.diff(first_bin_edges), [0.05, 0.05])
        for ax in histogram_axes:
            self.assertEqual(len(ax.lines), 1)
            zero_line = ax.lines[0]
            np.testing.assert_array_equal(zero_line.get_xdata(), [0, 0])
            self.assertEqual(zero_line.get_color(), "black")
            self.assertEqual(zero_line.get_linestyle(), "--")
            self.assertEqual(zero_line.get_zorder(), 0)
        self.assertTrue(all(len(ax.lines) == 4 for ax in ecdf_axes))
        self.assertTrue(
            all(
                [line.get_zorder() for line in ax.lines[1:]] == [4, 3, 5]
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
            all(len(ax.patches) == 2 for ax in cue_fig.axes[:6])
        )
        self.assertTrue(
            all(len(ax.lines) == 3 for ax in cue_fig.axes[6:])
        )
        self.assertTrue(
            all(len(ax.patches) == 1 for ax in hidden_opposite_fig.axes[:6])
        )
        self.assertTrue(
            all(len(ax.lines) == 2 for ax in hidden_opposite_fig.axes[6:])
        )
        self.assertTrue(
            all(len(ax.patches) == 1 for ax in hidden_all_preferred_fig.axes[:6])
        )
        self.assertTrue(
            all(len(ax.lines) == 2 for ax in hidden_all_preferred_fig.axes[6:])
        )
        hidden_all_preferred_labels = [
            text.get_text()
            for text in hidden_all_preferred_fig.axes[0].get_legend().texts
        ]
        self.assertFalse(
            any(
                "Preferred cue: all delay bins" in label
                for label in hidden_all_preferred_labels
            )
        )
        self.assertIn("Cell 10", fig.axes[6].get_xlabel())
        self.assertIn("Cell 30", fig.axes[8].get_xlabel())
        self.assertIn("5 selected preferred cells", " ".join(fig.axes[9].get_xlabel().split()))
        self.assertIn("4 selected non-preferred cells", " ".join(fig.axes[10].get_xlabel().split()))
        self.assertIn("6 other cells passing enabled checks", " ".join(fig.axes[11].get_xlabel().split()))
        import matplotlib.pyplot as plt

        plt.close(fig)
        plt.close(cue_fig)
        plt.close(hidden_opposite_fig)
        plt.close(hidden_all_preferred_fig)


class ActivityPopulationLabelsTest(unittest.TestCase):
    def test_population_titles_follow_recorded_checks_and_cell_pev_has_correct_units(self):
        import matplotlib.pyplot as plt

        for checks, expected_selected, expected_remainder in [
            ({"selectivity": True}, "Selective", "Cells failing selectivity, passing other checks"),
            ({"selectivity": False}, "Selected", "Other cells passing enabled checks"),
            (None, "Selected", "Other cells passing enabled checks"),
        ]:
            with self.subTest(checks=checks):
                activity = SessionActivity(
                    session="example", preferred_cue=7, opposite_cue=3,
                    dimensions=CellActivityDimensions(
                        cell_ids=np.array([9]), selectivity_pev_pct=np.array([8.25]),
                    ),
                    delay_bin_starts=np.array([500, 550]),
                    preferred_activity=np.array([[[0.0], [1.0]]]),
                    opposite_activity=np.array([[[-1.0], [0.0]]]),
                    on_state_mask=np.array([[True, False]]),
                    off_state_mask=np.array([[False, True]]),
                    preferred_trial_ids=np.array([0]), opposite_trial_ids=np.array([1]),
                    population_mean_activities={
                        group: (np.array([[0.0, 1.0]]), np.array([[-1.0, 0.0]]), 1)
                        for group in ("preferred", "selective_nonpreferred", "stationary_nonselective")
                    },
                    screening_checks=checks,
                )
                figure = plot_session_activity_marginal_histograms(activity)
                try:
                    self.assertEqual(figure.axes[1].get_title(), f"{expected_selected} preferred cells")
                    self.assertEqual(figure.axes[2].get_title(), f"{expected_selected} non-preferred cells")
                    self.assertEqual(" ".join(figure.axes[3].get_title().split()), expected_remainder)
                    self.assertIn("selectivity PEV=8.25%", figure.axes[4].get_xlabel())
                    self.assertNotIn("delay PEV", figure.axes[4].get_xlabel())
                finally:
                    plt.close(figure)


if __name__ == "__main__":
    unittest.main()
