"""Source-derived decoder identity without importing or executing saved source.

The per-session estimate depends on the analytical decoder functions, model
construction, screening metadata interpretation, input loading, and binning.
Plotting, stage dispatch, and worker initialization are deliberately outside
this boundary. Dependency closure follows newly referenced helpers, constants,
and local imports, rather than relying on a manually maintained version number.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path


class ScientificSourceError(ValueError):
    """The scientific source boundary cannot safely be determined."""


_WHOLE_MODULES = {'decoder_models.py'}
_ROOTS = {
    'decoding_confidence.py': (
        'training_trials', 'validate_training_class_counts', 'decode_one_trial', 'decode_session', 'Config',
    ),
    'common.py': ('load_session', 'compute_binned_rates'),
    'cache_io.py': ('read', 'load'),
}
# This dependency controls execution and figure initialization, not estimates.
# Keep the decoder's import binding in its signature, but not the initializer.
_EXECUTION_ONLY = {('common.py', 'worker_context')}
_DYNAMIC_NAMES = {'eval', 'exec', 'compile', 'globals', 'locals', 'vars', 'getattr',
                  '__import__', '__builtins__'}


def _entrypoint_guard(node):
    return (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id in {'__name__', '__package__'})


def _target_names(node):
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, (ast.Tuple, ast.List)):
        return set().union(*(_target_names(value) for value in node.elts))
    if isinstance(node, ast.Starred):
        return _target_names(node.value)
    if isinstance(node, (ast.Attribute, ast.Subscript)):
        return _target_names(node.value)
    return set()


def _bindings(node):
    """Module-level writes, without treating function locals as globals."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, ast.Import):
        return {alias.asname or alias.name.split('.')[0] for alias in node.names}
    if isinstance(node, ast.ImportFrom):
        return {alias.asname or alias.name for alias in node.names}
    if isinstance(node, ast.Assign):
        return set().union(*(_target_names(target) for target in node.targets))
    if isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
        return _target_names(node.target)
    # A conditional definition is represented by its entire containing node:
    # both branches and the condition must affect the signature.
    result = set()
    for child in ast.iter_child_nodes(node):
        result.update(_bindings(child))
    return result


def _without_docstrings(node):
    class StripDocstrings(ast.NodeTransformer):
        def visit(self, item):
            item = super().visit(item)
            if isinstance(item, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if (item.body and isinstance(item.body[0], ast.Expr)
                        and isinstance(item.body[0].value, ast.Constant)
                        and isinstance(item.body[0].value.value, str)):
                    item.body = item.body[1:]
            return item
    return StripDocstrings().visit(copy.deepcopy(node))


class _SourceModule:
    def __init__(self, filename, source):
        try:
            self.tree = ast.parse(source, filename=filename)
        except (SyntaxError, UnicodeError, TypeError) as exc:
            raise ScientificSourceError(f'Cannot parse scientific source {filename}: {exc}') from exc
        self.nodes = []
        self.bindings = {}
        for node in self.tree.body:
            # Only referenced import bindings enter a partial-module signature.
            parts = []
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    if alias.name == '*':
                        raise ScientificSourceError(f'Wildcard imports are unsupported in scientific source {filename}.')
                    part = copy.copy(node)
                    part.names = [alias]
                    parts.append(part)
            else:
                parts.append(node)
            for part in parts:
                self.nodes.append(part)
                for name in _bindings(part):
                    self.bindings.setdefault(name, []).append(part)


class _ScientificClosure:
    def __init__(self, sources):
        self.sources = sources
        self.modules = {}
        self.selected = {}
        self.visited = set()
        self.config_projections = {}

    def module(self, filename):
        if filename not in self.modules:
            if filename not in self.sources:
                raise ScientificSourceError(f'Missing scientific source: {filename}.')
            module = self.modules[filename] = _SourceModule(filename, self.sources[filename])
            self.selected[filename] = {}
            # Future flags affect all definitions. Unassigned top-level calls
            # could mutate scientific globals, so retain them conservatively.
            for node in module.nodes:
                if ((isinstance(node, ast.ImportFrom) and node.module == '__future__')
                        or (isinstance(node, ast.Expr) and not (
                            isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)))
                        or (isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While,
                                              ast.Try, ast.TryStar, ast.With, ast.AsyncWith,
                                              ast.Assert, ast.Raise, ast.Delete)) and not _entrypoint_guard(node))
                        or (isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign))
                            and any(isinstance(part, ast.Call) for part in ast.walk(node)))
                        or (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                            and node.decorator_list)):
                    self.include(filename, node)
        return self.modules[filename]

    def whole_module(self, filename):
        module = self.module(filename)
        for node in module.nodes:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                continue
            self.include(filename, node)

    def symbol(self, filename, name):
        if (filename, name) in _EXECUTION_ONLY:
            return
        if filename in _WHOLE_MODULES:
            self.whole_module(filename)
            return
        module = self.module(filename)
        nodes = module.bindings.get(name)
        if not nodes:
            raise ScientificSourceError(f'Missing scientific definition {filename}:{name}.')
        for node in nodes:
            if filename == 'decoding_confidence.py' and name == 'Config' and isinstance(node, ast.ClassDef):
                self.include_config(filename, node)
            else:
                self.include(filename, node)

    def include_config(self, filename, node):
        if id(node) in self.config_projections:
            return
        if not any(isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == '__post_init__'
                   for item in node.body):
            raise ScientificSourceError('Decoder Config.__post_init__ is missing.')
        projection = copy.copy(node)
        # Resolved dataclass defaults enter the settings fingerprint. Preserve
        # every method (including magic methods), class constant, decorator,
        # and base: each can alter normalization or later attribute access.
        projection.body = [item for item in node.body if not isinstance(item, ast.AnnAssign)
                           or self.is_class_constant(filename, item.annotation)]
        self.config_projections[id(node)] = projection
        self.include(filename, projection, original=node)

    def is_class_constant(self, filename, annotation):
        for part in ast.walk(annotation):
            if isinstance(part, ast.Attribute) and part.attr == 'ClassVar':
                return True
            if isinstance(part, ast.Name):
                if part.id == 'ClassVar':
                    return True
                for binding in self.module(filename).bindings.get(part.id, []):
                    if (isinstance(binding, ast.ImportFrom) and binding.module == 'typing'
                            and any(alias.name == 'ClassVar' for alias in binding.names)):
                        return True
        return False

    def include(self, filename, node, *, original=None):
        module = self.module(filename)
        key = (filename, id(original if original is not None else node))
        if key in self.visited:
            return
        self.visited.add(key)
        self.selected[filename][key[1]] = node
        for part in ast.walk(node):
            if isinstance(part, ast.Name) and isinstance(part.ctx, ast.Load):
                if part.id in _DYNAMIC_NAMES:
                    raise ScientificSourceError(f'Dynamic source lookup {part.id!r} is unsupported in {filename}.')
                # This intentionally over-approximates local shadowing: extra
                # dependencies cause conservative invalidation, never omission.
                if part.id in module.bindings:
                    self.symbol(filename, part.id)
            elif isinstance(part, (ast.Import, ast.ImportFrom)):
                self.imports(filename, part)
            elif isinstance(part, ast.Call) and isinstance(part.func, ast.Attribute):
                if part.func.attr in {'import_module', '__import__', 'eval', 'exec', 'compile', 'FunctionType'}:
                    raise ScientificSourceError(f'Dynamic imports/evaluation are unsupported in {filename}.')
            elif isinstance(part, ast.Attribute) and part.attr in {'__dict__', 'modules'}:
                raise ScientificSourceError(f'Dynamic namespace access is unsupported in {filename}.')
        # Creating a new Config inside an analytical function would make its
        # defaults an additional input, separate from the caller's resolved one.
        if filename == 'decoding_confidence.py' and any(
            isinstance(part, ast.Call) and isinstance(part.func, ast.Name) and part.func.id == 'Config'
            for part in ast.walk(node)
        ):
            raise ScientificSourceError('Analytical functions must use the resolved decoder Config, not create another one.')

    def imports(self, filename, node):
        for alias in node.names:
            if alias.name == '*':
                raise ScientificSourceError(f'Wildcard imports are unsupported in {filename}.')
            if isinstance(node, ast.Import):
                qualified = alias.name
                symbol = None
            else:
                qualified = node.module or ''
                if node.level:
                    package = ['scripts', 'next', *Path(filename).parts[:-1]]
                    if node.level > len(package):
                        raise ScientificSourceError(f'Unsupported relative import in {filename}.')
                    qualified = '.'.join([*package[:len(package) - node.level + 1], *qualified.split('.')])
                    qualified = qualified.rstrip('.')
                symbol = alias.name
            if qualified == 'builtins' and alias.name in _DYNAMIC_NAMES:
                raise ScientificSourceError(f'Dynamic source lookup imports are unsupported in {filename}.')
            if qualified == 'importlib' and alias.name == 'import_module':
                raise ScientificSourceError(f'Dynamic imports are unsupported in {filename}.')
            if qualified == 'scripts.next' and symbol:
                target, symbol = symbol.replace('.', '/') + '.py', None
            elif qualified.startswith('scripts.next.'):
                target = qualified.removeprefix('scripts.next.').replace('.', '/') + '.py'
            else:
                continue  # External implementation versions belong to the run fingerprint.
            if symbol is None:
                self.whole_module(target)
            else:
                self.symbol(target, symbol)

    def payload(self):
        # Package initialization can change imported scientific definitions.
        # A documentation-only initializer contributes no AST and no identity.
        if '__init__.py' in self.sources:
            self.whole_module('__init__.py')
        for filename in sorted(_WHOLE_MODULES):
            self.whole_module(filename)
        for filename, symbols in _ROOTS.items():
            for name in symbols:
                self.symbol(filename, name)
        result = {}
        for filename in sorted(self.selected):
            selected = self.selected[filename]
            if not selected:
                continue
            result[filename] = [
                ast.dump(_without_docstrings(selected[id(node)]), include_attributes=False)
                for node in self.modules[filename].nodes if id(node) in selected
            ]
        return result


def scientific_source_digest(sources: Mapping[str, bytes] | None = None) -> str:
    """Hash scientific AST/dependencies; unrelated files may be present in sources.

    Historical source maps use filenames relative to scripts/next. No source is
    imported, compiled to executable code, or evaluated. Missing dependencies or
    unsupported dynamic lookup fail closed instead of silently allowing reuse.
    Installed package versions are deliberately separate from this source-only
    signature, allowing the caller to compare historical source independently.
    """
    if sources is None:
        root = Path(__file__).parent
        sources = {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob('*.py')}
    payload = _ScientificClosure(sources).payload()
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
