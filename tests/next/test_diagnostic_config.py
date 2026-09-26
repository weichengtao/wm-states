"""Validation and target-resolution contracts for screening diagnostic figures."""
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import tempfile
import unittest

from scripts.next.diagnostic_config import (
    CellRange,
    DiagnosticConfig,
    DiagnosticPlots,
    DiagnosticTargets,
    load_diagnostic_config,
    select_cells,
)


class DiagnosticConfigTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / 'diagnostic_figures.json'

    def load(self, data):
        self.path.write_text(json.dumps(data), encoding='utf-8')
        return load_diagnostic_config(self.path)

    def test_omitted_file_means_table_only_while_supplied_minimal_file_enables_figures(self):
        table_only = load_diagnostic_config(None)
        with_figures = self.load({'schema_version': 1})
        self.assertFalse(table_only.plots.enabled)
        self.assertTrue(with_figures.plots.enabled)
        self.assertEqual(table_only.targets, with_figures.targets)
        self.assertEqual(with_figures.targets.sessions, 'available')
        self.assertEqual(with_figures.targets.cells, 'all')
        self.assertEqual(with_figures.plots.size_inches, (8.0, 5.0))
        self.assertEqual(with_figures.plots.dpi, 300)
        self.assertIsNone(with_figures.plots.max_cells_per_session)
        self.assertFalse(with_figures.plots.show_not_applicable_reasons)

    def test_explicit_selectors_and_resolved_snapshot_round_trip(self):
        config = self.load({
            'schema_version': 1,
            'targets': {
                'sessions': ['session-b', 'session-a'],
                'cells': [3, 1, 3, 0],
                'cells_by_session': {'session-a': {'start': 2, 'stop': 5}, 'session-b': []},
            },
            'plots': {'enabled': True, 'max_cells_per_session': 7, 'size_inches': [9, 6.5],
                      'dpi': 150, 'show_not_applicable_reasons': True},
        })
        self.assertEqual(config.targets.sessions, ('session-b', 'session-a'))
        self.assertEqual(config.targets.cells, (0, 1, 3))
        self.assertEqual(config.targets.cells_by_session['session-a'], CellRange(2, 5))
        self.assertEqual(config.targets.cells_by_session['session-b'], ())
        snapshot = config.to_dict()
        self.assertEqual(snapshot['targets']['cells'], [0, 1, 3])
        self.assertEqual(snapshot['targets']['cells_by_session']['session-a'], {'start': 2, 'stop': 5})
        self.assertEqual(list(snapshot['targets']['cells_by_session']), ['session-a', 'session-b'])
        self.assertEqual(self.load(snapshot), config)
        snapshot['targets']['cells'].append(99)
        self.assertEqual(config.targets.cells, (0, 1, 3))

    def test_config_is_deeply_immutable_and_copies_supplied_values(self):
        selectors = {'session-a': [2, 1]}
        sessions = ['session-a']
        targets = DiagnosticTargets(sessions=sessions, cells_by_session=selectors)
        selectors['session-a'].append(9)
        sessions.append('session-b')
        self.assertEqual(targets.sessions, ('session-a',))
        self.assertEqual(targets.cells_by_session['session-a'], (1, 2))
        with self.assertRaises(TypeError):
            targets.cells_by_session['session-b'] = 'all'
        with self.assertRaises(FrozenInstanceError):
            targets.cells = (3,)

    def test_schema_version_is_required_and_strict(self):
        for data in ({}, {'schema_version': 0}, {'schema_version': 2}, {'schema_version': True},
                     {'schema_version': 1.0}, {'schema_version': '1'}, [], None):
            with self.subTest(data=data), self.assertRaises(ValueError):
                self.load(data)

    def test_unknown_fields_and_legacy_trial_or_figure_fields_are_rejected_even_when_disabled(self):
        variants = [
            {'typo': 1},
            {'figures': []},
            {'targets': {'trial_start': 0}},
            {'targets': {'trial_end': None}},
            {'targets': {'trial_holdout': None}},
            {'targets': {'cell_start': 0}},
            {'targets': {'cells': {'start': 0, 'stop': 3, 'end': 3}}},
            {'plots': {'enabled': False, 'figsize': [8, 5]}},
        ]
        for extra in variants:
            data = {'schema_version': 1, 'plots': {'enabled': False}, **extra}
            with self.subTest(data=data), self.assertRaisesRegex(ValueError, 'unknown fields'):
                self.load(data)

    def test_duplicate_json_keys_are_rejected_at_every_depth(self):
        for content in (
            '{"schema_version":1,"schema_version":1}',
            '{"schema_version":1,"plots":{"enabled":true,"enabled":false}}',
            '{"schema_version":1,"targets":{"cells_by_session":{"s":[0],"s":[1]}}}',
        ):
            self.path.write_text(content, encoding='utf-8')
            with self.subTest(content=content), self.assertRaisesRegex(ValueError, 'Duplicate'):
                load_diagnostic_config(self.path)

    def test_session_targets_and_overrides_are_checked(self):
        variants = [
            {'sessions': []}, {'sessions': 'all'}, {'sessions': ['']},
            {'sessions': ['s', 's']}, {'sessions': [12]}, {'sessions': [' s']},
            {'sessions': ['s'], 'cells_by_session': {'other': 'all'}},
            {'cells_by_session': []}, {'cells_by_session': {'': 'all'}},
        ]
        for targets in variants:
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                self.load({'schema_version': 1, 'targets': targets})
        config = self.load({'schema_version': 1, 'targets': {'cells_by_session': {'unavailable': [1]}}})
        self.assertEqual(config.targets.cells_by_session['unavailable'], (1,))

    def test_cell_selectors_validate_indices_and_half_open_ranges(self):
        invalid = [None, 2, True, 'ALL', [True], [1.0], [-1], ['1'], {},
                   {'start': 0}, {'stop': 3}, {'start': True, 'stop': 3},
                   {'start': 0, 'stop': 3.0}, {'start': -1, 'stop': 3},
                   {'start': 3, 'stop': 3}, {'start': 4, 'stop': 3}]
        for selector in invalid:
            with self.subTest(selector=selector), self.assertRaises(ValueError):
                self.load({'schema_version': 1, 'targets': {'cells': selector}})

    def test_plot_controls_are_strict_and_checked_when_disabled(self):
        invalid = [
            {'enabled': 0}, {'show_not_applicable_reasons': 1},
            {'max_cells_per_session': 0}, {'max_cells_per_session': True}, {'max_cells_per_session': 1.5},
            {'dpi': False}, {'dpi': 0}, {'dpi': -100}, {'dpi': 150.0},
            {'size_inches': [8]}, {'size_inches': [8, 5, 2]}, {'size_inches': '8,5'},
            {'size_inches': [True, 5]}, {'size_inches': ['8', 5]}, {'size_inches': [0, 5]},
            {'size_inches': [float('inf'), 5]}, {'size_inches': [float('nan'), 5]},
            {'size_inches': [10 ** 400, 5]},
        ]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.load({'schema_version': 1, 'plots': {'enabled': False, **value}})
        for field in ('targets', 'plots'):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.load({'schema_version': 1, field: None})

    def test_json_failures_include_the_config_path(self):
        with self.assertRaisesRegex(ValueError, 'diagnostic_figures.json'):
            load_diagnostic_config(self.path)
        self.path.write_text('{broken', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'diagnostic_figures.json'):
            load_diagnostic_config(self.path)
        self.path.write_text('{"schema_version":1,"plots":{"dpi":NaN}}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'finite JSON numbers'):
            load_diagnostic_config(self.path)

    def test_select_cells_resolves_all_empty_lists_duplicates_and_ranges(self):
        self.assertEqual(select_cells('all', 4, 'session-a'), [0, 1, 2, 3])
        self.assertEqual(select_cells('all', 0, 'session-a'), [])
        self.assertEqual(select_cells((), 4, 'session-a'), [])
        self.assertEqual(select_cells((3, 1, 3), 4, 'session-a'), [1, 3])
        self.assertEqual(select_cells(CellRange(1, 4), 4, 'session-a'), [1, 2, 3])

    def test_select_cells_rejects_out_of_range_instead_of_silently_truncating(self):
        for selector in ((4,), CellRange(1, 5), (0,)):
            num_cells = 0 if selector == (0,) else 4
            with self.subTest(selector=selector), self.assertRaisesRegex(ValueError, 'session-a'):
                select_cells(selector, num_cells, 'session-a')
        for size in (True, -1, 1.5):
            with self.subTest(size=size), self.assertRaises(ValueError):
                select_cells('all', size, 'session-a')

    def test_direct_constructors_validate_the_same_contract(self):
        with self.assertRaises(ValueError):
            DiagnosticConfig(schema_version=True)
        with self.assertRaises(ValueError):
            DiagnosticPlots(dpi=True)
        with self.assertRaises(ValueError):
            DiagnosticTargets(cells=(-1,))
        with self.assertRaises(ValueError):
            CellRange(2, 2)


if __name__ == '__main__':
    unittest.main()
