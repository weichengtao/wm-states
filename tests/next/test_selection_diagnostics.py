"""Diagnostic target choices never change screening or the per-cell table."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.next import cell_screening, pipeline, selection_diagnostics
from scripts.next.diagnostic_config import DiagnosticConfig, DiagnosticPlots, DiagnosticTargets
from scripts.next.figure_exports import figure_format_context


class SelectionDiagnosticsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = cell_screening.Config(cache_dir=self.root / 'run', check_min_trials=False)
        self.spikes = np.zeros((6, 3, 4))
        self.spikes[:2, 0, 0] = 1
        self.spikes[2:, 0, 1] = 1
        self.spikes[:2, 2, 2] = 1  # outside the half-open presence window
        self.spikes[:, 1, 3] = 1
        self.recording = (self.spikes, np.array([-400, 500, 1400]), np.ones(6), np.arange(6) < 2)
        self.rows = [dict(session=session, cell_idx=cell, rejection_reason='fail_baseline_drift_not_applicable',
                          check_presence_ratio='disabled')
                     for session in ('a', 'b') for cell in (3, 1, 0, 2)]
        self.files = [Path('a.mat'), Path('b.mat')]
        self.loader = patch.object(selection_diagnostics, 'load_session', return_value=self.recording).start()
        self.addCleanup(patch.stopall)
        self.addCleanup(plt.close, 'all')

    def table(self):
        return pd.read_csv(self.config.cache_dir / 'select/diagnostics/cell_rejection_diagnostics.csv')

    def snapshot(self):
        return json.loads((self.config.cache_dir / 'select/diagnostics/figure_config.json').read_text())

    def test_target_overrides_and_caps_do_not_filter_or_reorder_csv_measurements(self):
        settings = DiagnosticConfig(targets=DiagnosticTargets(cells_by_session={'b': [3, 1]}),
                                    plots=DiagnosticPlots(max_cells_per_session=2, dpi=72, size_inches=(4, 3)))
        captured = []
        def capture(figure, path, dpi):
            captured.append((path.name, dpi, list(figure.get_size_inches()),
                             figure.axes[1].get_xlabel(), figure._suptitle.get_text()))
            self.assertEqual(len(figure.axes[0].lines[0].get_xdata()), 6)
        with patch.object(selection_diagnostics, 'save_figure', side_effect=capture), self.assertWarnsRegex(UserWarning, 'first 2 of 4'):
            selection_diagnostics.save_diagnostics(self.rows, self.files, self.config, settings)
        self.assertEqual([item[0] for item in captured], ['a_0.png', 'a_1.png', 'b_1.png', 'b_3.png'])
        self.assertTrue(all(item[1:3] == (72, [4., 3.]) for item in captured))
        self.assertIn('correct + incorrect', captured[0][3])
        self.assertIn('Checks not applicable', captured[0][4])
        self.assertNotIn('Baseline firing-rate drift', captured[0][4])
        frame = self.table()
        self.assertEqual(len(frame), 8)
        np.testing.assert_array_equal(frame.presence_ratio, [1, 0, 1, 0] * 2)
        self.assertTrue((frame.check_presence_ratio == 'disabled').all())
        self.assertTrue((frame.rejection_reason == 'fail_baseline_drift_not_applicable').all())
        self.assertEqual(self.snapshot()['resolved_targets'], {'a': [0, 1], 'b': [1, 3]})
        self.assertEqual(plt.get_fignums(), [])

    def test_missing_or_skipped_target_sessions_warn_without_restricting_csv(self):
        settings = DiagnosticConfig(targets=DiagnosticTargets(sessions=['a', 'missing', 'skipped'], cells=[]))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            selection_diagnostics.save_diagnostics(self.rows, self.files + [Path('skipped.mat')], self.config, settings)
        self.assertTrue(any('missing' in str(item.message) and 'unavailable' in str(item.message) for item in caught))
        self.assertTrue(any('skipped' in str(item.message) and 'no per-cell' in str(item.message) for item in caught))
        self.assertEqual(self.snapshot()['resolved_targets'], {'a': []})
        self.assertEqual(len(self.table()), 8)

    def test_all_targets_validated_before_any_diagnostic_output(self):
        settings = DiagnosticConfig(targets=DiagnosticTargets(cells_by_session={'b': [4]}))
        with patch.object(selection_diagnostics, 'save_figure') as save, self.assertRaisesRegex(ValueError, 'b: diagnostic cell 4'):
            selection_diagnostics.save_diagnostics(self.rows, self.files, self.config, settings)
        save.assert_not_called()
        self.loader.assert_not_called()
        self.assertFalse(self.config.cache_dir.exists())

    def test_csv_only_settings_and_no_path_write_snapshot_without_plots(self):
        for settings in (None, DiagnosticConfig(targets=DiagnosticTargets(cells=[100]), plots=DiagnosticPlots(enabled=False))):
            with self.subTest(settings=settings), patch.object(selection_diagnostics, 'save_figure') as save:
                selection_diagnostics.save_diagnostics(self.rows, self.files, self.config, settings)
                save.assert_not_called()
                self.assertEqual(len(self.table()), 8)
                self.assertEqual(self.snapshot()['resolved_targets'], {})
                self.assertFalse(self.snapshot()['configuration']['plots']['enabled'])

    def test_pdf_only_and_visible_reason_details(self):
        settings = DiagnosticConfig(targets=DiagnosticTargets(sessions=['a'], cells=[1]),
                                    plots=DiagnosticPlots(show_not_applicable_reasons=True, dpi=60))
        with figure_format_context(('pdf',)):
            selection_diagnostics.save_diagnostics(self.rows, self.files, self.config, settings)
        output = self.config.cache_dir / 'select/diagnostics/figures/cells/a_1.pdf'
        self.assertTrue(output.read_bytes().startswith(b'%PDF-'))
        self.assertEqual(list(output.parent.iterdir()), [output])
        self.assertEqual(plt.get_fignums(), [])

    def test_export_failure_closes_figure(self):
        settings = DiagnosticConfig(targets=DiagnosticTargets(sessions=['a'], cells=[1]))
        with patch.object(selection_diagnostics, 'save_figure', side_effect=RuntimeError('export failed')):
            with self.assertRaisesRegex(RuntimeError, 'export failed'):
                selection_diagnostics.save_diagnostics(self.rows, self.files, self.config, settings)
        self.assertEqual(plt.get_fignums(), [])

    def test_empty_diagnostics_remain_readable(self):
        selection_diagnostics.save_diagnostics([], self.files, self.config)
        self.assertTrue(self.table().empty)
        self.loader.assert_not_called()

    def test_invalid_file_fails_screening_and_pipeline_dry_run_before_processing(self):
        invalid = self.root / 'diagnostics.json'
        invalid.write_text('{"figures": []}')
        config = replace(self.config, save_extended_diagnostics=True, diagnostics_figure_config=invalid)
        with patch.object(cell_screening, 'process_session') as process, self.assertRaises(ValueError):
            cell_screening.main(config)
        process.assert_not_called()
        preset = self.root / 'pipeline.json'
        preset.write_text(json.dumps({'select': {'save_extended_diagnostics': True,
                                                'diagnostics_figure_config': str(invalid)}}))
        with redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            pipeline.main(pipeline.Config(settings=preset, stages=('select',), cache_dir=self.config.cache_dir, dry_run=True))
        self.assertFalse(self.config.cache_dir.exists())


if __name__ == '__main__':
    unittest.main()
