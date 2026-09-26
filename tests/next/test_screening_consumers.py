"""Screening metadata must select the same populations across analysis stages."""

import unittest

import numpy as np

from scripts.next.activity_weighting import cell_group_activity_weights
from scripts.next.compare_activity_across_states import session_cell_groups
from scripts.next.decoder_models import CellsUsedForDecoder, decoder_cells_for_session
from scripts.next.prepare_data_for_mixedlm import _cell_groups


class ScreeningMetadataConsumerTest(unittest.TestCase):
    def test_named_metadata_aligns_decoder_and_activity_populations(self):
        selection = {
            "cell_idx_stationary": np.asarray([2, 4, 6, 8, 10]),
            "cell_properties": {
                "cell_idx": np.asarray([2, 4, 6, 8]),
                "preferred_cue": np.asarray([1, 5, 1, 3]),
                "mean_selectivity_pev_pct": np.asarray([4.0, 7.0, 9.0, 2.0]),
                "qualifying_selectivity_bin_count": np.asarray([3, 4, 5, 2]),
            },
        }
        expected_groups = {
            "preferred": {2, 6},
            "selective_nonpreferred": {4, 8},
            "stationary_nonselective": {10},
        }
        for make_groups in (session_cell_groups, _cell_groups):
            with self.subTest(stage=make_groups.__module__):
                groups = make_groups(selection, preferred_cue=1)
                self.assertEqual(
                    {name: set(cells) for name, cells in groups.items()},
                    expected_groups,
                )
                weights = cell_group_activity_weights(selection, groups, True)
                np.testing.assert_array_equal(
                    np.sort(weights["preferred"]), [4.0, 9.0]
                )
                np.testing.assert_array_equal(
                    weights["selective_nonpreferred"], [7.0, 2.0]
                )
                self.assertIsNone(weights["stationary_nonselective"])

        for mode, expected in (
            (CellsUsedForDecoder.PREFERRED, {2, 6}),
            (CellsUsedForDecoder.PREFERRED_AND_OPPOSITE, {2, 4, 6}),
            (CellsUsedForDecoder.SELECTIVE, {2, 4, 6, 8}),
            (CellsUsedForDecoder.STATIONARY, {2, 4, 6, 8, 10}),
        ):
            with self.subTest(mode=mode):
                self.assertEqual(
                    decoder_cells_for_session(selection, mode, 1, 5, 12),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
