"""Build the dashboard and its documentation with the active local toolchain."""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys


def build_assets(repo_root: Path | None = None):
    """Build both sites before the HTTP server starts; fail with actionable errors."""
    root = (repo_root or Path(__file__).resolve().parents[3]).resolve()
    frontend = root / 'dashboard'
    npm = shutil.which('npm')
    if npm is None:
        raise RuntimeError('npm is not on PATH. Load Node.js (for example, nvm use) and retry --build.')
    if importlib.util.find_spec('mkdocs') is None:
        raise RuntimeError('MkDocs is not installed in this Python environment. '
                           'Run uv sync --locked --group dashboard --group docs, then retry --build.')
    if not (frontend / 'package.json').is_file() or not (root / 'mkdocs.yml').is_file():
        raise RuntimeError('The repository must contain dashboard/package.json and mkdocs.yml to build the sites.')
    commands = []
    if not (frontend / 'node_modules').is_dir():
        commands.append(('Installing frontend dependencies', [npm, 'ci'], frontend))
    commands.extend([
        ('Building the dashboard', [npm, 'run', 'build'], frontend),
        ('Building the documentation', [sys.executable, '-m', 'mkdocs', 'build', '--strict',
                                       '--config-file', str(root / 'mkdocs.yml'),
                                       '--site-dir', str(root / 'site')], root),
    ])
    for description, command, directory in commands:
        print(f'{description}…', flush=True)
        try:
            subprocess.run(command, cwd=directory, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f'{description} failed (exit code {exc.returncode}). '
                               'Fix the build error above before starting the dashboard.') from exc
        except OSError as exc:
            raise RuntimeError(f'{description} could not start: {exc}') from exc
