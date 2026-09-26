"""Shared metadata preserves population membership, ordering, and validation."""

import copy
import unittest

import numpy as np

from scripts.next.activity_weighting import cell_group_activity_weights
from scripts.next.decoder_models import CellsUsedForDecoder, decoder_cells_for_session
from scripts.next.screening_metadata import (
    ScreeningMetadata,
    cell_groups,
    population_labels,
    preferred_pev_cells,
    validate_cell_ids,
)


class ScreeningPopulationsTest(unittest.TestCase):
    def setUp(self):
        self.selection = {
            "cell_idx_stationary": np.array([8, 4, 2, 6, 0, 10, 12]),
            "cell_properties": {
                "cell_idx": np.array([4, 2, 6, 8, 0]),
                "preferred_cue": np.array([1, 1, 1, 1, 5]),
                "mean_selectivity_pev_pct": np.array([5.0, np.nan, 8.0, 8.0, np.inf]),
            },
        }

    def test_ranked_preferred_cells_filter_only_nonfinite_pev_and_keep_tie_order(self):
        cell_ids, pev_pct = preferred_pev_cells(self.selection, 1)
        np.testing.assert_array_equal(cell_ids, [6, 8, 4])
        np.testing.assert_array_equal(pev_pct, [8, 8, 5])

    def test_group_order_and_eligibility_remain_stage_specific(self):
        original = copy.deepcopy(self.selection)
        prepared = cell_groups(self.selection, 1)
        activity = cell_groups(self.selection, 1, rank_preferred_by_pev=True)
        np.testing.assert_array_equal(prepared["preferred"], [4, 2, 6, 8])
        np.testing.assert_array_equal(activity["preferred"], [6, 8, 4])
        for groups in (prepared, activity):
            np.testing.assert_array_equal(groups["selective_nonpreferred"], [0])
            np.testing.assert_array_equal(groups["stationary_nonselective"], [10, 12])
        for name, values in original["cell_properties"].items():
            np.testing.assert_array_equal(self.selection["cell_properties"][name], values)

    def test_unranked_groups_do_not_require_pev(self):
        del self.selection["cell_properties"]["mean_selectivity_pev_pct"]
        np.testing.assert_array_equal(cell_groups(self.selection, 1)["preferred"], [4, 2, 6, 8])
        with self.assertRaisesRegex(ValueError, "mean_selectivity_pev_pct"):
            cell_groups(self.selection, 1, rank_preferred_by_pev=True)

    def test_selected_ids_must_belong_to_stationary_pool(self):
        self.selection["cell_idx_stationary"] = np.array([2, 6, 8, 0])
        with self.assertRaisesRegex(ValueError, "subset"):
            cell_groups(self.selection, 1)

    def test_optional_session_size_rejects_out_of_range_metadata(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            cell_groups(self.selection, 1, num_cells_total=12)
        with self.assertRaisesRegex(ValueError, "outside"):
            preferred_pev_cells(self.selection, 1, num_cells_total=8)

    def test_empty_selected_pool_remains_valid(self):
        self.selection["cell_properties"] = {
            "cell_idx": [], "preferred_cue": [], "mean_selectivity_pev_pct": []
        }
        groups = cell_groups(self.selection, 1, rank_preferred_by_pev=True)
        self.assertEqual(groups["preferred"].size, 0)
        self.assertEqual(groups["selective_nonpreferred"].size, 0)
        np.testing.assert_array_equal(
            groups["stationary_nonselective"], self.selection["cell_idx_stationary"]
        )


class ScreeningMetadataValidationTest(unittest.TestCase):
    def test_invalid_cell_ids_are_rejected_before_indexing_or_integer_conversion(self):
        malformed_ids = ([1, 1], [-1, 2], [0.5, 1], [np.nan], [np.inf],
                         [True], ["1"], [[1], [2]], [2**63])
        for cell_ids in malformed_ids:
            with self.subTest(cell_ids=cell_ids), self.assertRaises(ValueError):
                validate_cell_ids(cell_ids, "cell_idx")

    def test_integral_ids_keep_their_recorded_order(self):
        np.testing.assert_array_equal(validate_cell_ids([4.0, 0.0, 2.0], "cell_idx"), [4, 0, 2])

    def test_aligned_metadata_and_cues_are_required_only_when_accessed(self):
        metadata = ScreeningMetadata({"cell_properties": {"cell_idx": [3, 1]}})
        np.testing.assert_array_equal(metadata.selected_cell_ids, [3, 1])
        with self.assertRaisesRegex(ValueError, "preferred_cue"):
            _ = metadata.preferred_cues
        with self.assertRaisesRegex(ValueError, "mean_selectivity_pev_pct"):
            _ = metadata.selectivity_pev_pct

    def test_cue_and_pev_alignment_are_validated(self):
        for property_name, accessor in (
            ("preferred_cue", "preferred_cues"),
            ("mean_selectivity_pev_pct", "selectivity_pev_pct"),
        ):
            metadata = ScreeningMetadata({"cell_properties": {"cell_idx": [3, 1], property_name: [1]}})
            with self.subTest(property_name=property_name), self.assertRaisesRegex(ValueError, "matching shapes"):
                getattr(metadata, accessor)

    def test_invalid_recorded_and_requested_cues_are_rejected(self):
        for cue in (0, 9, 1.5, np.nan, np.inf, True, "1"):
            metadata = ScreeningMetadata({"cell_properties": {"cell_idx": [0], "preferred_cue": [cue]}})
            with self.subTest(recorded_cue=cue), self.assertRaises(ValueError):
                _ = metadata.preferred_cues
            with self.subTest(requested_cue=cue), self.assertRaises(ValueError):
                preferred_pev_cells({}, cue)

    def test_redundant_selected_ids_must_match_property_order(self):
        metadata = ScreeningMetadata({"cell_idx_selected": [1, 3], "cell_properties": {"cell_idx": [3, 1]}})
        with self.assertRaisesRegex(ValueError, "same ordered cell IDs"):
            _ = metadata.selected_cell_ids

    def test_invalid_session_size_is_rejected(self):
        for size in (-1, 3.5, True):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "num_cells_total"):
                ScreeningMetadata({}, num_cells_total=size)

    def test_nonfinite_pev_is_left_for_the_consumers_policy(self):
        metadata = ScreeningMetadata({"cell_properties": {
            "cell_idx": [0, 1], "mean_selectivity_pev_pct": [np.nan, np.inf]
        }})
        np.testing.assert_array_equal(metadata.selectivity_pev_pct, [np.nan, np.inf])


class PopulationLabelsTest(unittest.TestCase):
    def test_selectivity_is_claimed_only_when_explicitly_enabled(self):
        for checks in (None, {}, {"baseline_drift": True}, {"selectivity": False}):
            with self.subTest(checks=checks):
                labels = population_labels(checks)
                self.assertEqual(labels["preferred"], "Selected preferred cells")
                self.assertEqual(labels["selective_nonpreferred"], "Selected non-preferred cells")
                self.assertNotIn("stationary", labels["stationary_nonselective"].lower())
                self.assertNotIn("selectiv", " ".join(labels.values()).lower())

    def test_enabled_selectivity_labels_do_not_assert_disabled_stationarity_checks(self):
        labels = population_labels({"selectivity": True, "baseline_drift": False})
        self.assertEqual(labels["preferred"], "Selective preferred cells")
        self.assertEqual(labels["stationary_nonselective"], "Cells failing selectivity, passing other checks")

    def test_labels_reject_truthy_strings_instead_of_claiming_a_check_was_enabled(self):
        with self.assertRaisesRegex(ValueError, "boolean"):
            population_labels({"selectivity": "False"})


class PartialMetadataConsumersTest(unittest.TestCase):
    def test_decoder_requires_only_the_selected_pool_for_its_mode(self):
        cases = (
            (CellsUsedForDecoder.ALL, {}, {0, 1, 2, 3}),
            (CellsUsedForDecoder.STATIONARY, {"cell_idx_stationary": [3, 1]}, {1, 3}),
            (CellsUsedForDecoder.PASSED_PRESENCE_RATIO, {"cell_idx_passed_presence_ratio": [2]}, {2}),
            (CellsUsedForDecoder.SELECTIVE, {"cell_properties": {"cell_idx": [3, 1]}}, {1, 3}),
            (CellsUsedForDecoder.PREFERRED, {"cell_properties": {"cell_idx": [3, 1], "preferred_cue": [1, 5]}}, {3}),
            (CellsUsedForDecoder.PREFERRED_AND_OPPOSITE, {"cell_properties": {"cell_idx": [3, 1], "preferred_cue": [1, 5]}}, {1, 3}),
        )
        for mode, selection, expected in cases:
            with self.subTest(mode=mode):
                self.assertEqual(decoder_cells_for_session(selection, mode, 1, 5, 4), expected)

    def test_decoder_rejects_out_of_range_cell_ids(self):
        for mode, selection in (
            (CellsUsedForDecoder.STATIONARY, {"cell_idx_stationary": [4]}),
            (CellsUsedForDecoder.PASSED_PRESENCE_RATIO, {"cell_idx_passed_presence_ratio": [4]}),
            (CellsUsedForDecoder.SELECTIVE, {"cell_properties": {"cell_idx": [4]}}),
        ):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "outside"):
                decoder_cells_for_session(selection, mode, 1, 5, 4)

    def test_weighting_aligns_ids_without_requiring_cue_metadata(self):
        selection = {"cell_properties": {"cell_idx": [3, 1], "mean_selectivity_pev_pct": [2, 8]}}
        groups = {"preferred": np.array([1, 3]), "stationary_nonselective": np.array([2])}
        weights = cell_group_activity_weights(selection, groups, True)
        np.testing.assert_array_equal(weights["preferred"], [8, 2])
        self.assertIsNone(weights["stationary_nonselective"])

    def test_weighting_rejects_malformed_group_ids_instead_of_silently_truncating(self):
        selection = {"cell_properties": {"cell_idx": [1], "mean_selectivity_pev_pct": [8]}}
        for cell_ids in ([1.5], [1, 1]):
            with self.subTest(cell_ids=cell_ids), self.assertRaises(ValueError):
                cell_group_activity_weights(selection, {"preferred": np.array(cell_ids)}, True)

    def test_weighting_does_not_reject_unused_nonfinite_pev(self):
        selection = {"cell_properties": {"cell_idx": [1, 3], "mean_selectivity_pev_pct": [8, np.nan]}}
        weights = cell_group_activity_weights(selection, {"preferred": np.array([1])}, True)
        np.testing.assert_array_equal(weights["preferred"], [8])


if __name__ == "__main__":
    unittest.main()
