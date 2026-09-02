import unittest
from pathlib import Path

from scripts.predict_off_state_duration_using_baseline_activity import (
    Config as ActivityConfig,
    _results_root,
)
from scripts.predict_off_state_duration_using_cell_count import (
    Config as CellCountConfig,
)


class FixedLmOutputPathTest(unittest.TestCase):
    def test_both_scripts_default_to_fixedlm(self):
        self.assertEqual(ActivityConfig().output_subdir, 'fixedlm')
        self.assertEqual(CellCountConfig().output_subdir, 'fixedlm')

    def test_results_root_stays_within_cache(self):
        self.assertEqual(
            _results_root(Path('cache/run'), 'fixedlm'),
            Path('cache/run/fixedlm'),
        )

    def test_rejects_parent_traversal(self):
        with self.assertRaisesRegex(ValueError, 'must stay within cache_dir'):
            _results_root(Path('cache/run'), '../outside')


if __name__ == '__main__':
    unittest.main()
