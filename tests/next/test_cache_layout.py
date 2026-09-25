"""Stage ownership and producer/consumer contracts for the next cache layout."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.next import cache_io, pipeline
from scripts.next import prepare_data_for_mixedlm as prepare
from scripts.next import find_active_cell_criticality as criticality
from scripts.next import eval_confidence as evaluate
from scripts.next import reject_reason_histograms as reasons
from scripts.next.cache_paths import STAGES, primary_cache, stage_path
from scripts.next.mixedlm_outcomes import ALL_OUTCOMES, analysis_output_dir


class CacheLayoutTest(unittest.TestCase):
    def test_stage_directories_match_runner_and_reject_escaping_subpaths(self):
        self.assertEqual(STAGES, set(pipeline.STAGES))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for stage in STAGES:
                self.assertEqual(stage_path(root, stage, 'custom/nested'),
                                 root / stage / 'custom/nested')
                for invalid in ('../elsewhere', '/tmp/elsewhere', 'nested/../../elsewhere'):
                    with self.subTest(stage=stage, path=invalid), self.assertRaises(ValueError):
                        stage_path(root, stage, invalid)
            with self.assertRaises(ValueError):
                stage_path(root, 'unknown')
            (root / 'prepare').mkdir()
            (root / 'prepare' / 'escape').symlink_to(root, target_is_directory=True)
            with self.assertRaises(ValueError):
                stage_path(root, 'prepare', 'escape/select')

    def test_each_model_stage_owns_both_outcomes(self):
        root = Path('cache/run')
        for stage in ('models', 'nested-count', 'nested-activity', 'criticality', 'interactions'):
            for outcome in ALL_OUTCOMES:
                self.assertEqual(analysis_output_dir(root, 'custom/nested', outcome, stage),
                                 root / stage / 'custom/nested/outcomes' / outcome.slug)

    def test_evaluation_reads_decode_and_writes_only_evaluate(self):
        source = dict(session='example', cue=1, trial_idx=np.array([1, 3]),
                      time_bins=np.array([0, 50]), decoding_test_labels=np.ones(2),
                      decoding_confidence=np.full((2, 2), .8),
                      decoding_predicted_labels=np.ones((2, 2)),
                      decoding_confidence_null=np.full((2, 2, 3), .5))
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            # A valid flat cache alone must not silently supply an old run.
            cache_io.save([source], root / 'decoding_confidence.pkl')
            with self.assertRaises(FileNotFoundError):
                evaluate.main(evaluate.Config(cache_dir=root))
            (root / 'decoding_confidence.pkl').unlink()
            cache_io.save([source], root / 'decode/decoding_confidence.pkl')
            evaluate.main(evaluate.Config(cache_dir=root))
            self.assertEqual(cache_io.read(root / 'evaluate/eval_confidence.pkl')[0]['session'], 'example')
            self.assertTrue((root / 'evaluate/tables/eval_confidence.csv').is_file())
            self.assertEqual({path.name for path in root.iterdir()}, {'decode', 'evaluate'})

    def test_preparation_and_criticality_have_separate_weighted_caches(self):
        rows = [dict(session='example', trial_id=i, value=float(i)) for i in (1, 2)]
        cv_session = dict(session='example', trial_ids=np.array([0, 1, 2]))
        for weighted in (False, True):
            with self.subTest(weighted=weighted), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                cache_io.save([], primary_cache(root, 'cell_screening.pkl'))
                cache_io.save([dict(session='example')], primary_cache(root, 'on_off_states.pkl'))
                suffix = Path('pev_weighted') if weighted else Path('.')
                config = prepare.Config(cache_dir=root, output_subdir='custom/tables',
                                        cv_output_subdir='custom/cv', cv_shuffles=1,
                                        pev_weighted_average=weighted)
                with patch.object(prepare, 'validate_state_provenance'), redirect_stdout(io.StringIO()), \
                     patch.object(prepare, '_prepare_session_rows', return_value=(rows, cv_session)):
                    prepare.prepare_data(config)
                    _, _, paths = criticality._prepare_threshold_data(
                        criticality.Config(cache_dir=root, pev_weighted_average=weighted), [50])
                prepared = root / 'prepare/custom/tables' / suffix
                cv_path = root / 'prepare/custom/cv' / suffix / 'cv_feature_cache.pkl'
                self.assertTrue((prepared / 'trial_table.pkl').is_file())
                self.assertTrue(cv_path.is_file())
                manifest = json.loads((prepared / 'manifest.json').read_text())
                self.assertEqual(manifest['outputs']['cv_feature_cache'], str(cv_path))
                self.assertEqual(manifest['source_caches']['cell_selection'],
                                 str(root / 'select/cell_screening.pkl'))
                self.assertEqual(paths[50], root / 'criticality/prepared/active_thresholds/percentile_50'
                                 / suffix / 'trial_table.pkl')
                self.assertTrue(paths[50].is_file())
                self.assertFalse((root / 'prepare/active_thresholds').exists())
                self.assertEqual({p.name for p in root.iterdir()},
                                 {'select', 'states', 'prepare', 'criticality'})

    def test_rejection_tool_uses_selection_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            diagnostics = root / 'select/diagnostics'
            diagnostics.mkdir(parents=True)
            pd.DataFrame([dict(session='example', rejection_reason='presence_ratio')]).to_csv(
                diagnostics / 'cell_rejection_diagnostics.csv', index=False)
            with patch.object(reasons, 'save_figure_png_only') as save:
                reasons.main(reasons.Config(cache_dir=root))
            self.assertEqual(save.call_args.args[1], diagnostics / 'figures/reasons/example.png')
            self.assertTrue((diagnostics / 'reject_reason_histograms_summary.csv').is_file())


if __name__ == '__main__':
    unittest.main()
