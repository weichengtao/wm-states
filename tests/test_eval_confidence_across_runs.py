import contextlib
import io
import pickle
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np

from scripts import eval_confidence_across_runs as comparison


def session_result(session, times=(0, 50), offset=0):
    return {
        'session': session,
        'observed_repeat_idx': 0,
        'time_bins': np.asarray(times),
        'observed': {
            'brier_score_by_time_bin': np.array([0.1, 0.2]) + offset,
            'log_loss_by_time_bin': np.array([0.3, 0.4]) + offset,
            'accuracy_by_time_bin': np.array([0.8, 0.9]),
            'decoding_confidence_by_time_bin': np.array([0.7, 0.8]),
        },
        'null': {
            'brier_score_by_time_bin': np.array([0.25, 0.3]) + offset,
            'log_loss_by_time_bin': np.array([0.7, 0.8]) + offset,
            'accuracy_by_time_bin': np.array([0.4, 0.5]),
            'decoding_confidence_by_time_bin': np.array([0.5, 0.6]),
        },
    }


class AcrossRunsTest(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def write_run(self, root, name, results):
        path = root / name
        path.mkdir()
        (path / 'eval_confidence.pkl').write_bytes(pickle.dumps(results))
        return path

    def test_panels_use_cached_time_scores_and_each_runs_time_axis(self):
        runs = [{'a': session_result('a')},
                {'a': session_result('a', times=(10, 60), offset=0.1)}]
        fig = comparison.plot_session(['first', 'second'], runs, 'a')
        self.assertEqual(len(fig.axes), 8)
        for ax, (metric, kind) in zip(fig.axes, (
            ('brier_score', 'observed'), ('brier_score', 'null'),
            ('log_loss', 'observed'), ('log_loss', 'null'),
            ('accuracy', 'observed'), ('accuracy', 'null'),
            ('decoding_confidence', 'observed'), ('decoding_confidence', 'null'),
        )):
            self.assertEqual(len(ax.lines), 2)
            for line, run in zip(ax.lines, runs):
                np.testing.assert_array_equal(line.get_xdata(), run['a']['time_bins'])
                np.testing.assert_array_equal(line.get_ydata(), run['a'][kind][f'{metric}_by_time_bin'])

    def test_missing_null_remains_missing(self):
        row = session_result('a')
        row['null'] = None
        with contextlib.redirect_stdout(io.StringIO()):
            fig = comparison.plot_session(['first'], [{'a': row}], 'a')
        self.assertTrue(np.all(np.isnan(fig.axes[1].lines[0].get_ydata())))

    def test_three_runs_with_aliases_and_one_legend(self):
        aliases = ['Baseline', 'Calibrated', 'Uncalibrated']
        runs = [{'a': session_result('a', offset=i * 0.1)} for i in range(3)]
        fig = comparison.plot_session(['run1', 'run2', 'run3'], runs, 'a', aliases)
        self.assertEqual(
            [text.get_text() for text in fig.axes[0].get_legend().get_texts()], aliases
        )
        for index, ax in enumerate(fig.axes):
            self.assertEqual(len(ax.lines), 3)
            self.assertEqual([line.get_label() for line in ax.lines], aliases)
            if index:
                self.assertIsNone(ax.get_legend())

    def test_alias_validation(self):
        for aliases in ([], ['only one'], ['valid', '  ']):
            with self.subTest(aliases=aliases), self.assertRaisesRegex(ValueError, 'run-aliases'):
                comparison.main(comparison.Config(
                    cache_dirs=[Path('first'), Path('second')], run_aliases=aliases,
                ))

    def test_custom_colors_apply_to_lines_and_null_bands(self):
        colors = ['navy', '#e66101', 'forestgreen']
        runs = []
        for index in range(3):
            row = session_result('a')
            for metric, _ in comparison.METRICS:
                row['null'][f'{metric}_by_time_bin_and_sample'] = np.array([[0.2, 0.4], [0.3, 0.5]])
            runs.append({'a': row})
        fig = comparison.plot_session(['a', 'b', 'c'], runs, 'a', line_colors=colors)
        for ax in fig.axes:
            self.assertEqual([line.get_color() for line in ax.lines], colors)
            for band, color in zip(ax.collections, colors):
                np.testing.assert_allclose(band.get_facecolor()[0], comparison.matplotlib.colors.to_rgba(color, alpha=0.2))

    def test_invalid_colors_fail_before_loading_caches(self):
        for colors in ([], ['red'], ['red', 'not-a-color']):
            with self.subTest(colors=colors), self.assertRaisesRegex(ValueError, 'line-colors'):
                comparison.main(comparison.Config(
                    cache_dirs=[Path('first'), Path('second')], line_colors=colors,
                ))

    def test_percentiles_and_no_shading_keep_same_mean(self):
        row = session_result('a')
        samples = np.array([[0.1, 0.2, 0.9, np.nan], [0.3, np.nan, np.nan, np.nan]])
        for metric, _ in comparison.METRICS:
            row['null'][f'{metric}_by_time_bin_and_sample'] = samples
        means, (lower, upper) = comparison.score_curve(row, 'null', 'brier_score', 2, 'percentiles')
        np.testing.assert_allclose([lower[0], upper[0]], np.percentile([0.1, 0.2, 0.9], [2.5, 97.5]))
        self.assertTrue(np.isnan(lower[1]) and np.isnan(upper[1]))
        for mode in ('confidence_intervals', 'none'):
            other_means, interval = comparison.score_curve(row, 'null', 'brier_score', 2, mode)
            np.testing.assert_allclose(other_means, means)
            if mode == 'none':
                self.assertIsNone(interval)
        for mode in ('percentiles', 'none'):
            fig = comparison.plot_session(['run'], [{'a': row}], 'a', null_shading=mode)
            self.assertEqual([len(ax.collections) for ax in fig.axes],
                             [0, 1] * 4 if mode == 'percentiles' else [0] * 8)

    def test_sample_mean_and_t_interval_with_missing_values(self):
        row = {'null': {'brier_score_by_time_bin_and_sample':
               np.array([[0.1, 0.2, 0.3], [0.2, np.nan, np.nan], [np.nan] * 3])}}
        means, (lower, upper) = comparison.score_curve(row, 'null', 'brier_score', 3, 'confidence_intervals')
        np.testing.assert_allclose(means, [0.2, 0.2, np.nan], equal_nan=True)
        # Three samples: df=2, sample SD=0.1, t(.975)=4.3026527297.
        self.assertAlmostEqual(upper[0] - means[0], 4.3026527297 * 0.1 / np.sqrt(3))
        self.assertAlmostEqual(means[0] - lower[0], upper[0] - means[0])
        self.assertTrue(np.all(np.isnan(lower[1:])))
        self.assertTrue(np.all(np.isnan(upper[1:])))

    def test_bands_only_for_multiple_null_shuffles(self):
        row = session_result('a')
        row['observed_repeats'] = {}
        for metric, _ in comparison.METRICS:
            for kind in ('observed_repeats', 'null'):
                row[kind][f'{metric}_by_time_bin_and_sample'] = np.array([[0.1, 0.3], [0.2, 0.4]])
        fig = comparison.plot_session(['run'], [{'a': row}], 'a')
        for index, ax in enumerate(fig.axes):
            self.assertEqual(len(ax.collections), index % 2)
            if index % 2:
                np.testing.assert_allclose(ax.lines[0].get_ydata(), [0.2, 0.3])
            else:
                metric = comparison.METRICS[index // 2][0]
                np.testing.assert_allclose(ax.lines[0].get_ydata(), row['observed'][f'{metric}_by_time_bin'])
        for kind in ('observed_repeats', 'null'):
            for metric, _ in comparison.METRICS:
                row[kind][f'{metric}_by_time_bin_and_sample'] = np.array([[0.1], [0.2]])
        fig = comparison.plot_session(['run'], [{'a': row}], 'a')
        self.assertTrue(all(len(ax.collections) == 0 for ax in fig.axes))

    def test_legacy_cache_uses_first_repeat_and_rejects_ambiguous_average(self):
        row = session_result('a')
        row.pop('observed_repeat_idx')
        row['observed_repeats'] = {
            'brier_score_by_time_bin_and_sample': np.array([[0.1, 0.9], [0.2, 0.8]])
        }
        values, interval = comparison.score_curve(row, 'observed', 'brier_score', 2)
        np.testing.assert_allclose(values, [0.1, 0.2])
        self.assertIsNone(interval)
        row.pop('observed_repeats')
        with self.assertRaisesRegex(ValueError, 'Rerun'):
            comparison.score_curve(row, 'observed', 'brier_score', 2)

    def test_matching_by_session_and_output_paths(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = self.write_run(root, 'run_037_full_session',
                                   [session_result('b'), session_result('a'), session_result('extra')])
            second = self.write_run(root, 'run_038_full_session',
                                    [session_result('a'), session_result('b')])
            third = self.write_run(root, 'run_039_full_session',
                                   [session_result('b'), session_result('a')])
            cache_dirs = [first, second, third]
            with contextlib.redirect_stdout(io.StringIO()):
                _, _, sessions = comparison.load_runs(cache_dirs)
                paths = comparison.main(comparison.Config(
                    cache_dirs=cache_dirs, run_aliases=['First', 'Second', 'Third'],
                ))
            self.assertEqual(sessions, ['b', 'a'])
            self.assertEqual(len(paths), 6)
            expected_dirs = {
                run / 'eval_confidence_across_runs' / 'run_037_full_session_vs_run_038_full_session_vs_run_039_full_session'
                for run in cache_dirs
            }
            self.assertEqual({path.parent for path in paths}, expected_dirs)
            self.assertEqual({path.name for path in paths},
                             {'a_confidence_scores.png', 'b_confidence_scores.png'})
            for path in paths:
                self.assertTrue(path.read_bytes().startswith(b'\x89PNG'))
            for session in sessions:
                copies = [folder / f'{session}_confidence_scores.png' for folder in expected_dirs]
                self.assertEqual(copies[0].read_bytes(), copies[1].read_bytes())
                self.assertEqual(copies[0].read_bytes(), copies[2].read_bytes())

    def test_invalid_run_inputs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = self.write_run(root, 'first', [session_result('a')])
            second = self.write_run(root, 'second', [session_result('b')])
            for paths in ([first], [first, first], [first, second]):
                with self.subTest(paths=paths), self.assertRaises(ValueError):
                    comparison.load_runs(paths)
            with self.assertRaises(FileNotFoundError):
                comparison.load_runs([first, root / 'missing'])
            (second / 'eval_confidence.pkl').write_bytes(pickle.dumps([session_result('a')] * 2))
            with self.assertRaisesRegex(ValueError, 'Duplicate session'):
                comparison.load_runs([first, second])


if __name__ == '__main__':
    unittest.main()
