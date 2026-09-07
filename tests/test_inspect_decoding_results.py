import pickle
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
                    accuracy_values=kwargs.pop(
                        'accuracy_values', np.asarray([0.0, 1.0])
                    ),
                    confidence_values=kwargs.pop(
                        'confidence_values', np.asarray([0.4, 0.8])
                    ),
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

    def test_single_repeat_count_is_omitted_from_subplot_titles(self):
        fig = self._capture_figure(
            accuracy_values=np.asarray([1.0]),
            confidence_values=np.asarray([0.8]),
        )

        self.assertEqual(fig.axes[0].get_title(), 'Accuracy')
        self.assertEqual(fig.axes[1].get_title(), 'Confidence')

    def test_only_confidence_legend_is_shown_at_top_left(self):
        fig = self._capture_figure(
            compare_repeat_idx=0,
            compare_accuracy_value=1.0,
            compare_confidence_value=0.8,
        )

        self.assertIsNone(fig.axes[0].get_legend())
        self.assertEqual(fig.axes[1].get_legend()._loc, 2)

    def test_state_label_is_in_figure_title(self):
        fig = self._capture_figure(state_label='not in any state')

        self.assertIn(', not in any state', fig._suptitle.get_text())
        self.assertNotIn('state:', fig._suptitle.get_text())


class StateCacheTest(unittest.TestCase):
    def test_state_labels_cover_on_off_and_unassigned_bins(self):
        self.assertEqual(
            inspect_decoding_results._state_label(True, False, 120.0),
            'on-state (120 ms)',
        )
        self.assertEqual(
            inspect_decoding_results._state_label(False, True, 80.0),
            'off-state (80 ms)',
        )
        self.assertEqual(
            inspect_decoding_results._state_label(False, False),
            'not in any state',
        )

    def test_state_duration_uses_the_contiguous_run_containing_the_bin(self):
        state_mask = np.asarray(
            [[False, True, True, True, False, True]],
            dtype=bool,
        )
        duration = inspect_decoding_results._state_duration_ms(
            state_mask,
            trial_row=0,
            bin_index=2,
            time_bins=np.arange(6, dtype=float) * 10,
        )

        self.assertEqual(duration, 30.0)

    def test_state_masks_are_aligned_by_trial_id_and_time_bin(self):
        state_results = [
            {
                'session': 'example',
                'cue': 3,
                'trial_idx': np.asarray([20, 10]),
                'time_bins': np.asarray([100.0, 0.0]),
                'on_state_mask': np.asarray([[False, True], [True, False]]),
                'off_state_mask': np.asarray([[True, False], [False, True]]),
            }
        ]
        with TemporaryDirectory() as temporary_directory:
            cache_path = Path(temporary_directory) / 'on_off_states.pkl'
            with cache_path.open('wb') as f:
                pickle.dump(state_results, f)
            on_mask, off_mask = inspect_decoding_results._load_state_masks(
                Path(temporary_directory),
                {'session': 'example', 'cue': 3},
                np.asarray([10, 20]),
                np.asarray([0.0, 100.0]),
            )

        np.testing.assert_array_equal(
            on_mask,
            np.asarray([[False, True], [True, False]]),
        )
        np.testing.assert_array_equal(
            off_mask,
            np.asarray([[True, False], [False, True]]),
        )


if __name__ == '__main__':
    unittest.main()
