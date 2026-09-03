import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

from scripts import inspect_decoding_results


class InspectionFigureTest(unittest.TestCase):
    def _capture_figure(self, **kwargs):
        captured = {}

        def capture(fig, _path, dpi):
            captured['fig'] = fig
            captured['dpi'] = dpi

        with TemporaryDirectory() as temporary_directory:
            with patch.object(
                inspect_decoding_results,
                'save_figure_png_only',
                side_effect=capture,
            ):
                inspect_decoding_results.save_inspection_figure(
                    Path(temporary_directory),
                    session='example',
                    trial=0,
                    time_bin_start=500.0,
                    accuracy_values=np.asarray([0.0, 1.0]),
                    confidence_values=np.asarray([0.4, 0.8]),
                    **kwargs,
                )
        return captured['fig']

    def tearDown(self):
        plt.close('all')

    def test_null_confidence_adds_default_state_thresholds(self):
        null_values = np.asarray([0.2, 0.4, 0.6, 0.8])
        fig = self._capture_figure(
            null_accuracy_values=np.asarray([0.0, 0.0, 1.0, 1.0]),
            null_confidence_values=null_values,
        )

        confidence_lines = {
            line.get_label(): line for line in fig.axes[1].lines
        }
        expected_off = np.mean(null_values) + 0.842 * np.std(null_values)
        expected_on = np.mean(null_values) + 1.645 * np.std(null_values)

        off_line = confidence_lines['Off-state threshold (+0.842 SD)']
        on_line = confidence_lines['On-state threshold (+1.645 SD)']
        self.assertAlmostEqual(off_line.get_xdata()[0], expected_off)
        self.assertAlmostEqual(on_line.get_xdata()[0], expected_on)
        self.assertEqual(off_line.get_color(), 'lightgreen')
        self.assertEqual(on_line.get_color(), 'darkgreen')
        self.assertEqual(off_line.get_linestyle(), '--')
        self.assertEqual(on_line.get_linestyle(), '--')

    def test_thresholds_are_absent_without_null_confidence(self):
        fig = self._capture_figure()

        labels = {line.get_label() for line in fig.axes[1].lines}
        self.assertNotIn('Off-state threshold (+0.842 SD)', labels)
        self.assertNotIn('On-state threshold (+1.645 SD)', labels)

    def test_repeat_zero_is_labeled_observed(self):
        fig = self._capture_figure(
            compare_repeat_idx=0,
            compare_accuracy_value=1.0,
            compare_confidence_value=0.8,
        )

        for axis in fig.axes:
            labels = {line.get_label() for line in axis.lines}
            self.assertIn('Observed', labels)
            self.assertNotIn('Repeat 0', labels)

    def test_nonzero_repeat_keeps_numbered_label(self):
        fig = self._capture_figure(
            compare_repeat_idx=3,
            compare_accuracy_value=1.0,
            compare_confidence_value=0.8,
        )

        for axis in fig.axes:
            labels = {line.get_label() for line in axis.lines}
            self.assertIn('Repeat 3', labels)


if __name__ == '__main__':
    unittest.main()
