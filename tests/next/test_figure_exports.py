"""Export contracts shared by every next-pipeline figure writer."""
import ast
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
import warnings
from unittest.mock import Mock, patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.next import eval_confidence, pipeline
from scripts.next.figure_exports import (
    DEFAULT_FIGURE_FONT, FONT_ENV, FORMAT_ENV, figure_font_context,
    figure_format_context, resolve_figure_font, save_figure, validate_figure_font,
)


def worker_font_snapshot():
    """Run in real loky workers to catch stale import/environment defaults."""
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots()
    try:
        return os.environ[FONT_ENV], axes.set_title('Worker title').get_fontfamily()
    finally:
        plt.close(figure)


class FigureExportsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.addCleanup(plt.close, 'all')

    def test_default_and_selected_formats_return_real_paths_and_preserve_layout_options(self):
        figure = Mock()
        with patch.dict(os.environ, {}, clear=True):
            paths = save_figure(figure, self.root / 'nested/name.pdf', dpi=120, bbox_inches='tight')
        self.assertEqual(paths, (self.root / 'nested/name.png',))
        figure.savefig.assert_called_once_with(paths[0], dpi=120, format='png', bbox_inches='tight')
        figure.reset_mock()
        with patch.dict(os.environ, {FORMAT_ENV: 'png, tif,eps,pdf'}):
            paths = save_figure(figure, self.root / 'chart.png', dpi=150, bbox_inches='tight')
        self.assertEqual([path.suffix for path in paths], ['.png', '.tif', '.eps', '.pdf'])
        self.assertEqual([call.kwargs['format'] for call in figure.savefig.call_args_list], ['png', 'tiff', 'eps', 'pdf'])
        self.assertTrue(all(call.kwargs['dpi'] == 150 and call.kwargs['bbox_inches'] == 'tight'
                            for call in figure.savefig.call_args_list))

    def test_invalid_formats_fail_before_any_output_is_written(self):
        figure = Mock()
        for formats in ('', 'png,jpg', 'pdf,pdf', 'png,'):
            with self.subTest(formats=formats), patch.dict(os.environ, {FORMAT_ENV: formats}):
                with self.assertRaises(ValueError):
                    save_figure(figure, self.root / 'unwritten/chart.png')
        figure.savefig.assert_not_called()
        self.assertFalse((self.root / 'unwritten').exists())
        with patch.dict(os.environ, {FORMAT_ENV: 'png'}), self.assertRaises(ValueError):
            save_figure(figure, self.root / 'unwritten/chart.png', format='jpg')
        self.assertFalse((self.root / 'unwritten').exists())

    def test_pdf_keeps_line_plot_vector_and_embeds_fonts_without_changing_global_style(self):
        with matplotlib.rc_context({'font.family': 'DejaVu Sans', 'pdf.fonttype': 3,
                                    'pdf.compression': 0, 'pdf.use14corefonts': True}):
            fig, ax = plt.subplots(figsize=(3, 2))
            ax.plot([0, 1, 2], [0, 1, .5], label='Observed')
            ax.fill_between([0, 1, 2], [.1, .2, .1], [.4, .6, .4], alpha=.25)
            ax.set(xlabel='Time (ms)', ylabel='Confidence')
            with figure_format_context(('pdf',)):
                paths = save_figure(fig, self.root / 'vector.png', bbox_inches='tight')
            self.assertEqual(paths, (self.root / 'vector.pdf',))
            self.assertEqual(matplotlib.rcParams['pdf.fonttype'], 3)
            self.assertEqual(matplotlib.rcParams['pdf.compression'], 0)
            self.assertTrue(matplotlib.rcParams['pdf.use14corefonts'])
        content = paths[0].read_bytes()
        self.assertTrue(content.startswith(b'%PDF-'))
        self.assertIn(b'/FontFile2', content)
        self.assertIn(b'/FlateDecode', content)
        self.assertIn(b'/ca 0.25', content)
        self.assertNotIn(b'/Subtype /Image', content)
        self.assertFalse((self.root / 'vector.png').exists())

    def test_image_pdf_uses_lossless_compression_and_mixed_exports_are_valid(self):
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.imshow(np.array([[[255, 0, 0], [0, 255, 0]],
                            [[0, 0, 255], [255, 255, 255]]], dtype=np.uint8), interpolation='none')
        with figure_format_context(('png', 'tif', 'eps', 'pdf')):
            paths = save_figure(fig, self.root / 'image.png', dpi=60)
        self.assertTrue(all(path.stat().st_size > 100 for path in paths))
        self.assertTrue(paths[0].read_bytes().startswith(b'\x89PNG\r\n\x1a\n'))
        self.assertIn(paths[1].read_bytes()[:4], (b'II*\x00', b'MM\x00*'))
        self.assertTrue(paths[2].read_bytes().startswith(b'%!PS-Adobe'))
        content = paths[3].read_bytes()
        self.assertIn(b'/Subtype /Image', content)
        self.assertIn(b'/FlateDecode', content)
        self.assertNotIn(b'/DCTDecode', content)
        self.assertNotIn(b'/JPXDecode', content)

    def test_context_restores_unset_and_existing_environment_even_after_failure(self):
        with patch.dict(os.environ, {}, clear=True):
            with figure_format_context(('pdf',)):
                self.assertEqual(os.environ[FORMAT_ENV], 'pdf')
            self.assertNotIn(FORMAT_ENV, os.environ)
        with patch.dict(os.environ, {FORMAT_ENV: 'tif'}):
            with self.assertRaises(RuntimeError):
                with figure_format_context(('png', 'pdf')):
                    raise RuntimeError('failed export')
            self.assertEqual(os.environ[FORMAT_ENV], 'tif')

    def test_pipeline_scopes_formats_and_records_selection_for_success_and_failure(self):
        for fail in (False, True):
            cache = self.root / str(fail)
            config = pipeline.Config(cache_dir=cache, stages=('evaluate',), figure_formats=('pdf',))
            def stage(_config):
                self.assertEqual(os.environ[FORMAT_ENV], 'pdf')
                if fail:
                    raise RuntimeError('stage failed')
            with patch.dict(os.environ, {FORMAT_ENV: 'png'}), patch.object(eval_confidence, 'main', side_effect=stage), redirect_stdout(io.StringIO()):
                if fail:
                    with self.assertRaises(RuntimeError):
                        pipeline.main(config)
                else:
                    pipeline.main(config)
                self.assertEqual(os.environ[FORMAT_ENV], 'png')
            record = json.loads((cache / 'pipeline_manifest.json').read_text())
            self.assertEqual(record['runner_config']['figure_formats'], ['pdf'])
            self.assertEqual(record['status'], 'failed' if fail else 'complete')

    def test_invalid_programmatic_runner_formats_do_not_create_a_run(self):
        for formats in ((), ('jpg',), ('pdf', 'pdf')):
            with self.subTest(formats=formats), self.assertRaises(ValueError):
                pipeline.main(pipeline.Config(cache_dir=self.root / 'invalid', figure_formats=formats))
        self.assertFalse((self.root / 'invalid').exists())

    def test_font_defaults_override_and_restore_at_creation_and_export(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_figure_font(DEFAULT_FIGURE_FONT), 'DejaVu Sans')
            with matplotlib.rc_context({'font.family': 'DejaVu Serif'}):
                figure, axes = plt.subplots()
                title = axes.set_title('Font policy')
                original = title.get_fontfamily()
                with figure_format_context(('pdf',)):
                    path, = save_figure(figure, self.root / 'default.png')
                self.assertIn(b'DejaVuSans', path.read_bytes())
                self.assertEqual(title.get_fontfamily(), original)
                with figure_font_context('DejaVu Sans Mono'):
                    next_figure, next_axes = plt.subplots()
                    self.assertEqual(next_axes.set_title('New figure').get_fontfamily(), ['DejaVu Sans Mono'])
                    with figure_format_context(('pdf',)):
                        path, = save_figure(figure, self.root / 'override.png')
                    self.assertIn(b'DejaVuSansMono', path.read_bytes())
                self.assertEqual(matplotlib.rcParams['font.family'], ['DejaVu Serif'])
                self.assertEqual(title.get_fontfamily(), original)
            self.assertNotIn(FONT_ENV, os.environ)

    def test_font_context_restores_environment_and_style_after_failure(self):
        with patch.dict(os.environ, {FONT_ENV: 'DejaVu Serif'}), matplotlib.rc_context({'font.family': 'DejaVu Serif'}):
            with self.assertRaises(RuntimeError):
                with figure_font_context('DejaVu Sans Mono'):
                    self.assertEqual(os.environ[FONT_ENV], 'DejaVu Sans Mono')
                    raise RuntimeError('failed run')
            self.assertEqual(os.environ[FONT_ENV], 'DejaVu Serif')
            self.assertEqual(matplotlib.rcParams['font.family'], ['DejaVu Serif'])

    def test_missing_font_warns_once_and_falls_back_without_global_style_changes(self):
        resolve_figure_font.cache_clear()
        before = matplotlib.rcParams['font.family'][:]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            for _ in range(2):
                with figure_font_context('Nonexistent wm-states journal font 981273') as resolved:
                    self.assertEqual(resolved, DEFAULT_FIGURE_FONT)
                    self.assertEqual(matplotlib.rcParams['font.family'], [DEFAULT_FIGURE_FONT])
        self.assertEqual(len(caught), 1)
        self.assertIn('unavailable on this computer', str(caught[0].message))
        self.assertEqual(matplotlib.rcParams['font.family'], before)

    def test_invalid_font_settings_fail_before_creating_run_or_figure_output(self):
        for font in ('', '  ', 'Arial\n', 'font\x00name', 'x' * 121, None, ['Arial']):
            with self.subTest(font=font), self.assertRaises(ValueError):
                validate_figure_font(font)
        with self.assertRaises(ValueError):
            pipeline.main(pipeline.Config(cache_dir=self.root / 'invalid-font', figure_font=' '))
        self.assertFalse((self.root / 'invalid-font').exists())
        with patch.dict(os.environ, {FONT_ENV: ''}), self.assertRaises(ValueError):
            save_figure(Mock(), self.root / 'invalid-export/chart.png')
        self.assertFalse((self.root / 'invalid-export').exists())

    def test_pipeline_records_requested_and_resolved_font_and_applies_it_to_stage(self):
        cache = self.root / 'font-run'
        def stage(_config):
            self.assertEqual(os.environ[FONT_ENV], 'DejaVu Sans Mono')
            figure, axes = plt.subplots()
            self.assertEqual(axes.set_title('Stage').get_fontfamily(), ['DejaVu Sans Mono'])
        with patch.object(eval_confidence, 'main', side_effect=stage), redirect_stdout(io.StringIO()):
            pipeline.main(pipeline.Config(cache_dir=cache, stages=('evaluate',), figure_font='DejaVu Sans Mono'))
        record = json.loads((cache / 'pipeline_manifest.json').read_text())
        self.assertEqual(record['runner_config']['figure_font'], 'DejaVu Sans Mono')
        self.assertEqual(record['runner_config']['resolved_figure_font'], 'DejaVu Sans Mono')

    def test_worker_pools_follow_consecutive_font_changes(self):
        from joblib import Parallel, delayed
        from scripts.next.common import worker_context
        for font in ('DejaVu Sans', 'DejaVu Sans Mono'):
            with figure_font_context(font), worker_context(2):
                snapshots = Parallel()(delayed(worker_font_snapshot)() for _ in range(2))
            self.assertEqual(snapshots, [(font, [font])] * 2)

    def test_all_next_savefig_calls_are_owned_by_the_shared_export_helper(self):
        sources = Path(__file__).resolve().parents[2] / 'scripts/next'
        direct_writers = []
        for path in sources.rglob('*.py'):
            if path.name == 'figure_exports.py':
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ('savefig', 'print_figure'):
                    direct_writers.append(f'{path.relative_to(sources)}:{node.lineno}')
        self.assertEqual(direct_writers, [], 'Route all figure output through save_figure.')


if __name__ == '__main__':
    unittest.main()
