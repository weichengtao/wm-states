"""Strict, versioned configuration for full-session screening diagnostic figures.

The selection applies only to figures. Screening decisions and diagnostic tables
always retain their full session/cell scope.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping, TypeAlias


def _integer(value: object, path: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f'{path} must be an integer >= {minimum}.')
    return value


def _boolean(value: object, path: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f'{path} must be a boolean.')
    return value


def _object(value: object, path: str, allowed: set[str]) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f'{path} must be an object.')
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f'{path} has unknown fields: {", ".join(sorted(unknown))}.')
    return value


def _session(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{path} must be a nonempty session name.')
    if value != value.strip():
        raise ValueError(f'{path} must not have leading or trailing whitespace.')
    return value


@dataclass(frozen=True)
class CellRange:
    """Zero-based half-open cell range: start is included, stop is excluded."""

    start: int
    stop: int

    def __post_init__(self):
        _integer(self.start, 'cells.start')
        _integer(self.stop, 'cells.stop')
        if self.stop <= self.start:
            raise ValueError('cells.stop must be greater than cells.start (half-open range).')


CellSelector: TypeAlias = Literal['all'] | tuple[int, ...] | CellRange
SessionSelector: TypeAlias = Literal['available'] | tuple[str, ...]


def _cells(value: object, path: str) -> CellSelector:
    if isinstance(value, str):
        if value == 'all':
            return 'all'
        raise ValueError(f'{path} must be "all", a list of cell indices, or a start/stop object.')
    if isinstance(value, CellRange):
        return value
    if isinstance(value, (tuple, list)):
        return tuple(sorted({_integer(cell, f'{path}[{index}]') for index, cell in enumerate(value)}))
    if isinstance(value, dict):
        entry = _object(value, path, {'start', 'stop'})
        if set(entry) != {'start', 'stop'}:
            raise ValueError(f'{path} requires both start and stop (half-open range).')
        start = _integer(entry['start'], f'{path}.start')
        stop = _integer(entry['stop'], f'{path}.stop')
        if stop <= start:
            raise ValueError(f'{path}.stop must be greater than {path}.start (half-open range).')
        return CellRange(start, stop)
    raise ValueError(f'{path} must be "all", a list of cell indices, or a start/stop object.')


def _selector_dict(selector: CellSelector) -> str | list[int] | dict[str, int]:
    if isinstance(selector, CellRange):
        return {'start': selector.start, 'stop': selector.stop}
    return selector if isinstance(selector, str) else list(selector)


@dataclass(frozen=True)
class DiagnosticTargets:
    sessions: SessionSelector = 'available'
    cells: CellSelector = 'all'
    cells_by_session: Mapping[str, CellSelector] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self):
        sessions = self.sessions
        if isinstance(sessions, str):
            if sessions != 'available':
                raise ValueError('targets.sessions must be "available" or a nonempty list of session names.')
        elif isinstance(sessions, (list, tuple)) and sessions:
            sessions = tuple(_session(value, f'targets.sessions[{index}]')
                             for index, value in enumerate(sessions))
            if len(set(sessions)) != len(sessions):
                raise ValueError('targets.sessions must not contain duplicate session names.')
        else:
            raise ValueError('targets.sessions must be "available" or a nonempty list of session names.')
        object.__setattr__(self, 'sessions', sessions)
        object.__setattr__(self, 'cells', _cells(self.cells, 'targets.cells'))
        if not isinstance(self.cells_by_session, Mapping):
            raise ValueError('targets.cells_by_session must be an object.')
        overrides = {_session(key, 'targets.cells_by_session key'): _cells(value, f'targets.cells_by_session.{key}')
                     for key, value in self.cells_by_session.items()}
        if not isinstance(sessions, str):
            unselected = set(overrides) - set(sessions)
            if unselected:
                raise ValueError('targets.cells_by_session contains sessions outside targets.sessions: '
                                 f'{", ".join(sorted(unselected))}.')
        object.__setattr__(self, 'cells_by_session', MappingProxyType(overrides))


@dataclass(frozen=True)
class DiagnosticPlots:
    enabled: bool = True
    max_cells_per_session: int | None = None
    size_inches: tuple[float, float] = (8.0, 5.0)
    dpi: int = 300
    show_not_applicable_reasons: bool = False

    def __post_init__(self):
        _boolean(self.enabled, 'plots.enabled')
        _boolean(self.show_not_applicable_reasons, 'plots.show_not_applicable_reasons')
        if self.max_cells_per_session is not None:
            _integer(self.max_cells_per_session, 'plots.max_cells_per_session', minimum=1)
        _integer(self.dpi, 'plots.dpi', minimum=1)
        if not isinstance(self.size_inches, (list, tuple)) or len(self.size_inches) != 2:
            raise ValueError('plots.size_inches must be a list of two positive finite numbers.')
        sizes = []
        for value in self.size_inches:
            try:
                number = float(value) if type(value) in (int, float) else float('nan')
            except OverflowError:
                number = float('inf')
            if not math.isfinite(number) or number <= 0:
                raise ValueError('plots.size_inches must be a list of two positive finite numbers.')
            sizes.append(number)
        object.__setattr__(self, 'size_inches', tuple(sizes))


@dataclass(frozen=True)
class DiagnosticConfig:
    schema_version: int = 1
    targets: DiagnosticTargets = field(default_factory=DiagnosticTargets)
    plots: DiagnosticPlots = field(default_factory=DiagnosticPlots)

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError('schema_version must be integer 1.')
        if not isinstance(self.targets, DiagnosticTargets):
            raise ValueError('targets must be DiagnosticTargets.')
        if not isinstance(self.plots, DiagnosticPlots):
            raise ValueError('plots must be DiagnosticPlots.')

    def to_dict(self) -> dict:
        """Return JSON-ready, resolved settings suitable for a run snapshot."""
        return {
            'schema_version': self.schema_version,
            'targets': {
                'sessions': (self.targets.sessions if isinstance(self.targets.sessions, str)
                             else list(self.targets.sessions)),
                'cells': _selector_dict(self.targets.cells),
                'cells_by_session': {session: _selector_dict(selector)
                                     for session, selector in sorted(self.targets.cells_by_session.items())},
            },
            'plots': {
                'enabled': self.plots.enabled,
                'max_cells_per_session': self.plots.max_cells_per_session,
                'size_inches': list(self.plots.size_inches),
                'dpi': self.plots.dpi,
                'show_not_applicable_reasons': self.plots.show_not_applicable_reasons,
            },
        }


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate diagnostic configuration key: {key}.')
        result[key] = value
    return result


def _invalid_constant(value: str):
    raise ValueError(f'Diagnostic configuration must use finite JSON numbers; found {value}.')


def load_diagnostic_config(path: Path | None) -> DiagnosticConfig:
    """Load and validate a config, or select table-only diagnostics for ``None``.

    A supplied file must declare schema_version 1. Omitted targets and plot
    options receive defaults; notably a supplied config enables figures unless
    plots.enabled is explicitly false. All fields are checked even when disabled.
    """
    if path is None:
        return DiagnosticConfig(plots=DiagnosticPlots(enabled=False))
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=_unique_object,
                          parse_constant=_invalid_constant)
        data = _object(data, 'diagnostic configuration', {'schema_version', 'targets', 'plots'})
        if 'schema_version' not in data:
            raise ValueError('Diagnostic configuration requires schema_version: 1; legacy figure configs are unsupported.')
        targets = _object(data.get('targets', {}), 'targets', {'sessions', 'cells', 'cells_by_session'})
        plots = _object(data.get('plots', {}), 'plots', {
            'enabled', 'max_cells_per_session', 'size_inches', 'dpi', 'show_not_applicable_reasons',
        })
        return DiagnosticConfig(schema_version=data['schema_version'], targets=DiagnosticTargets(**targets),
                                plots=DiagnosticPlots(**plots))
    except (OSError, ValueError) as exc:
        raise ValueError(f'{path}: {exc}') from exc


def select_cells(selector: CellSelector, num_cells: int, session: str) -> list[int]:
    """Resolve a selector to sorted unique indices, rejecting unavailable cells."""
    _integer(num_cells, f'{session}: num_cells')
    selector = _cells(selector, f'{session}: cells')
    if selector == 'all':
        return list(range(num_cells))
    if isinstance(selector, CellRange):
        if selector.stop > num_cells:
            raise ValueError(f'{session}: diagnostic cell range [{selector.start}, {selector.stop}) '
                             f'exceeds {num_cells} available cells.')
        return list(range(selector.start, selector.stop))
    if selector and selector[-1] >= num_cells:
        raise ValueError(f'{session}: diagnostic cell {selector[-1]} is out of range '
                         f'for {num_cells} available cells.')
    return list(selector)
