"""Scientific source compatibility is independent of plotting and dispatch."""
from pathlib import Path
import tempfile
import unittest

from scripts.next.decoding_signature import ScientificSourceError, scientific_source_digest


SOURCE_ROOT = Path(__file__).resolve().parents[2] / 'scripts/next'


class DecodingSignatureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = {path.relative_to(SOURCE_ROOT).as_posix(): path.read_bytes()
                       for path in SOURCE_ROOT.rglob('*.py')}
        cls.baseline = scientific_source_digest(cls.sources)

    def changed(self, filename, old, new, *, sources=None):
        updated = dict(self.sources if sources is None else sources)
        content = updated[filename].decode()
        self.assertIn(old, content)
        updated[filename] = content.replace(old, new, 1).encode()
        return updated

    def test_current_sources_and_an_explicit_source_snapshot_agree(self):
        self.assertEqual(scientific_source_digest(), self.baseline)
        self.assertEqual(len(self.baseline), 64)

    def test_unrelated_font_plot_pipeline_and_dashboard_sources_do_not_affect_science(self):
        for filename in ('figure_exports.py', 'pipeline.py', 'decoding_plots.py',
                         'dashboard/runner.py', 'activity_plots.py', 'on_off_states.py'):
            with self.subTest(filename=filename):
                sources = {**self.sources, filename: b'This unrelated source is deliberately not Python.'}
                self.assertEqual(scientific_source_digest(sources), self.baseline)

    def test_stage_dispatch_worker_initialization_and_display_defaults_are_excluded(self):
        edits = [
            ('decoding_confidence.py', '    if config.plot_only:', '    if config.plot_only is True:'),
            ('common.py', 'initargs=(formats, current_figure_font())', 'initargs=((\'pdf\',), \'Arial\')'),
            ('decoding_confidence.py', 'save_figures: bool = True', 'save_figures: bool = False'),
            ('decoding_confidence.py', 'par_verbose: int = 0', 'par_verbose: int = 4'),
        ]
        for filename, old, new in edits:
            with self.subTest(filename=filename, old=old):
                self.assertEqual(scientific_source_digest(self.changed(filename, old, new)), self.baseline)

    def test_downstream_screening_population_labels_and_pev_group_helpers_are_excluded(self):
        edits = [
            ('"Selective preferred cells"', '"New display label"'),
            ('order = np.argsort(-pev_pct, kind="stable")', 'order = np.argsort(pev_pct, kind="stable")'),
            ('is_preferred = metadata.preferred_cues == preferred_cue',
             'is_preferred = metadata.preferred_cues != preferred_cue'),
        ]
        for old, new in edits:
            with self.subTest(old=old):
                self.assertEqual(scientific_source_digest(self.changed('screening_metadata.py', old, new)),
                                 self.baseline)

    def test_comments_docstrings_and_unreferenced_helpers_are_ignored(self):
        sources = self.changed('common.py', '"""Overlapping rectangular bins in O(trials * (samples + bins) * cells).',
                               '"""Reworded documentation for cumulative binning.')
        sources['common.py'] += b'\n# New documentation.\ndef unrelated_display_helper():\n    return "new color"\n'
        sources['decoder_models.py'] += b'\n# An explanatory comment.\n'
        self.assertEqual(scientific_source_digest(sources), self.baseline)

    def test_computation_validation_and_function_default_changes_are_detected(self):
        edits = [
            ('common.py', "variable_names=['spks', 'tc', 'cueAngIdx', 'isCorr']",
             "variable_names=['other_spikes', 'tc', 'cueAngIdx', 'isCorr']"),
            ('common.py', 'dtype=np.float32):', 'dtype=np.float64):'),
            ('decoding_confidence.py', 'observed = np.empty(count, dtype=np.float32)',
             'observed = np.empty(count, dtype=np.float64)'),
            ('decoding_confidence.py', 'required = 1', 'required = 2'),
            ('decoding_confidence.py', 'self.n_decode_shuffle < 0', 'self.n_decode_shuffle < 1'),
            ('decoder_models.py', "solver='liblinear'", "solver='lbfgs'"),
            ('decoder_models.py', 'CLASSIFIER_C_GRID = (1.0, 0.1, 0.01)', 'CLASSIFIER_C_GRID = (0.01, 0.1, 1.0)'),
            ('screening_metadata.py', 'cue_indices > 8', 'cue_indices > 9'),
            ('cache_io.py', 'SCREENING_SCHEMA_VERSION = 2', 'SCREENING_SCHEMA_VERSION = 3'),
        ]
        for filename, old, new in edits:
            with self.subTest(filename=filename, old=old):
                self.assertNotEqual(scientific_source_digest(self.changed(filename, old, new)), self.baseline)

    def test_dataclass_defaults_are_resolved_configuration_not_source_identity(self):
        sources = self.changed('decoding_confidence.py', 'n_decode_shuffle: int = 100', 'n_decode_shuffle: int = 200')
        self.assertEqual(scientific_source_digest(sources), self.baseline)

    def test_referenced_import_bindings_are_part_of_identity(self):
        edits = [
            ('common.py', 'from scipy.io import loadmat', 'from scipy.io import whosmat as loadmat'),
            ('common.py', 'import numpy as np', 'import numpy.linalg as np'),
            ('decoding_confidence.py', 'from sklearn.calibration import CalibratedClassifierCV',
             'from sklearn.svm import SVC as CalibratedClassifierCV'),
            ('decoding_confidence.py', 'from threadpoolctl import threadpool_limits',
             'from another_threads import threadpool_limits'),
        ]
        for filename, old, new in edits:
            with self.subTest(filename=filename, old=old):
                self.assertNotEqual(scientific_source_digest(self.changed(filename, old, new)), self.baseline)

    def test_new_referenced_constants_and_helpers_are_followed_automatically(self):
        sources = self.changed('common.py', 'dtype=np.float32):', 'dtype=BIN_DTYPE):')
        sources['common.py'] += b'\nBIN_DTYPE = np.float32\n'
        first = scientific_source_digest(sources)
        updated = self.changed('common.py', 'BIN_DTYPE = np.float32', 'BIN_DTYPE = np.float64', sources=sources)
        self.assertNotEqual(scientific_source_digest(updated), first)

        sources = self.changed('common.py', 'if not np.all(np.isin(cues, np.arange(1, 9))):',
                               'if not valid_cues(cues):')
        sources['common.py'] += b'\ndef valid_cues(cues):\n    return np.all(np.isin(cues, np.arange(1, 9)))\n'
        first = scientific_source_digest(sources)
        updated = self.changed('common.py', 'return np.all(np.isin(cues, np.arange(1, 9)))',
                               'return np.all(np.isin(cues, np.arange(1, 10)))', sources=sources)
        self.assertNotEqual(scientific_source_digest(updated), first)

    def test_new_local_imports_and_their_transitive_helpers_are_followed(self):
        sources = self.changed('common.py', "spikes = np.asarray(data['spks'])", "spikes = convert_spikes(data['spks'])")
        sources['common.py'] += b'\nfrom .new_input_helper import convert_spikes\n'
        sources['new_input_helper.py'] = b'''
import numpy as np
def convert_spikes(spikes):
    return _convert(spikes)
def _convert(spikes):
    return np.asarray(spikes, dtype=np.float32)
'''
        first = scientific_source_digest(sources)
        changed = self.changed('new_input_helper.py', 'dtype=np.float32', 'dtype=np.float64', sources=sources)
        self.assertNotEqual(scientific_source_digest(changed), first)
        del changed['new_input_helper.py']
        with self.assertRaisesRegex(ScientificSourceError, 'Missing scientific source: new_input_helper.py'):
            scientific_source_digest(changed)

    def test_config_normalization_helpers_are_followed_without_hashing_plot_fields(self):
        sources = self.changed('decoding_confidence.py', 'self.classifier_c = float(self.classifier_c)',
                               'self.classifier_c = self._normalise_c(self.classifier_c)')
        sources = self.changed('decoding_confidence.py', '    def __post_init__(self):',
                               '    def _normalise_c(self, value):\n        return float(value)\n\n    def __post_init__(self):', sources=sources)
        first = scientific_source_digest(sources)
        changed = self.changed('decoding_confidence.py', 'return float(value)', 'return abs(float(value))', sources=sources)
        self.assertNotEqual(scientific_source_digest(changed), first)

    def test_source_is_not_executed_and_dynamic_dependency_lookup_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / 'must-not-exist'
            sources = dict(self.sources)
            sources['decoder_models.py'] += f'\nopen({str(marker)!r}, "w").write("executed")\n'.encode()
            self.assertNotEqual(scientific_source_digest(sources), self.baseline)
            self.assertFalse(marker.exists())
        sources = self.changed('common.py', "data = loadmat(path,", "data = globals()['loadmat'](path,")
        with self.assertRaisesRegex(ScientificSourceError, 'Dynamic source lookup'):
            scientific_source_digest(sources)
        sources = {**self.sources, 'decoder_models.py': b'from somewhere import *\n'}
        with self.assertRaisesRegex(ScientificSourceError, 'Wildcard imports'):
            scientific_source_digest(sources)
        sources = dict(self.sources)
        sources['__init__.py'] += b'\nraise RuntimeError("scientific import initialization changed")\n'
        self.assertNotEqual(scientific_source_digest(sources), self.baseline)


if __name__ == '__main__':
    unittest.main()
