"""Check names stay intelligible across caches, diagnostic tables, and figures."""
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from scripts.next import cache_io, reject_reason_histograms
from scripts.next.screening_names import CHECK_NAMES, MEASUREMENT_COLUMNS, REASONS, reason_label, rejection_label


class ScreeningNamesTest(unittest.TestCase):
    def test_reason_codes_and_measurements_describe_all_seven_checks(self):
        expected = {
            'firing_rate': ('fail_firing_rate', 'mean_test_firing_rate_hz'),
            'presence_ratio': ('fail_presence_ratio', 'presence_ratio'),
            'delay_variance': ('fail_delay_variance', 'delay_to_baseline_variance_ratio'),
            'baseline_variance': ('fail_baseline_variance', 'baseline_window_variance_ratio'),
            'baseline_drift': ('fail_baseline_drift', 'baseline_drift_r'),
            'selectivity': ('fail_selectivity', 'mean_selectivity_pev_pct'),
            'preferred_cue_drift': ('fail_preferred_cue_drift', 'preferred_cue_drift_r'),
        }
        self.assertEqual(set(CHECK_NAMES), set(expected))
        for check, (failure, measurement) in expected.items():
            self.assertEqual(REASONS[check], failure)
            self.assertEqual(MEASUREMENT_COLUMNS[check], measurement)
            self.assertIn('failed', reason_label(failure))
            self.assertIn('unavailable', reason_label(failure + '_not_applicable'))
        self.assertEqual(rejection_label('fail_baseline_drift|fail_selectivity'),
                         'Baseline firing-rate drift: failed; Cue selectivity (PEV): failed')
        self.assertEqual(rejection_label('fail_baseline_drift_not_applicable', show_not_applicable=False),
                         'Checks not applicable')
        for obsolete in ('fail_temp_dep_stage3', 'fail_sig_pev', 'fail_min_fr_test', '', None):
            with self.subTest(reason=obsolete), self.assertRaisesRegex(ValueError, 'Regenerate diagnostics'):
                rejection_label(obsolete)

    def test_screening_cache_has_its_own_version_and_actionable_migration_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename in cache_io.PRIMARY:
                path = root / filename
                cache_io.save([{'session': 'example'}], path)
                expected_version = 2 if filename == 'cell_screening.pkl' else 1
                self.assertEqual(pickle.loads(path.read_bytes())['version'], expected_version)
                self.assertEqual(cache_io.read(path), [{'session': 'example'}])
            old_selection = root / 'cell_screening.pkl'
            old_selection.write_bytes(pickle.dumps({'schema': 'wm-states-next', 'version': 1,
                                                    'results': [{'cell_properties': {'mean_pref_test': [1]}}]}))
            with self.assertRaisesRegex(ValueError, 'Rerun select and its downstream stages'):
                cache_io.read(old_selection)

    def test_histogram_counts_keep_canonical_codes_and_readable_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            diagnostics = root / 'select/diagnostics'
            diagnostics.mkdir(parents=True)
            pd.DataFrame({'session': ['example'] * 4, 'rejection_reason': [
                'fail_baseline_drift|fail_selectivity', 'fail_selectivity',
                'fail_baseline_drift_not_applicable', 'pass',
            ]}).to_csv(diagnostics / 'cell_rejection_diagnostics.csv', index=False)
            labels = []
            def capture(figure, *_args):
                labels.extend(label.get_text() for label in figure.axes[0].get_yticklabels())
            with patch.object(reject_reason_histograms, 'save_figure', side_effect=capture):
                reject_reason_histograms.main(reject_reason_histograms.Config(cache_dir=root))
            table = pd.read_csv(diagnostics / 'reject_reason_histograms_summary.csv').set_index('reason')
            self.assertEqual(table.loc['fail_selectivity', 'n_cells'], 2)
            self.assertEqual(table.loc['fail_selectivity', 'percent'], 50)
            self.assertNotIn('fail_baseline_drift_not_applicable', table.index)
            self.assertEqual(table.loc['fail_baseline_drift', 'reason_label'], 'Baseline firing-rate drift: failed')
            self.assertIn('Cue selectivity (PEV): failed', labels)
            self.assertIn('Passed all enabled checks', labels)
            with patch.object(reject_reason_histograms, 'save_figure'):
                reject_reason_histograms.main(reject_reason_histograms.Config(cache_dir=root, skip_not_applicable=False))
            table = pd.read_csv(diagnostics / 'reject_reason_histograms_summary.csv').set_index('reason')
            self.assertEqual(table.loc['fail_baseline_drift_not_applicable', 'n_cells'], 1)
            self.assertEqual(table.loc['fail_baseline_drift_not_applicable', 'reason_label'],
                             'Baseline firing-rate drift: unavailable')

    def test_legacy_diagnostics_are_rejected_before_writing_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            diagnostics = root / 'select/diagnostics'
            diagnostics.mkdir(parents=True)
            pd.DataFrame({'session': ['a', 'b'], 'rejection_reason': ['pass', 'fail_temp_dep_stage3']}).to_csv(
                diagnostics / 'cell_rejection_diagnostics.csv', index=False)
            with patch.object(reject_reason_histograms, 'save_figure') as save:
                with self.assertRaisesRegex(ValueError, 'Regenerate diagnostics'):
                    reject_reason_histograms.main(reject_reason_histograms.Config(cache_dir=root))
            save.assert_not_called()
            self.assertFalse((diagnostics / 'reject_reason_histograms_summary.csv').exists())


if __name__ == '__main__':
    unittest.main()
