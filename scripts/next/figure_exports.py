"""One export policy for every next-pipeline figure, including diagnostics."""
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal, get_args

import matplotlib

FigureFormat = Literal['png', 'tif', 'eps', 'pdf']
FIGURE_FORMATS = get_args(FigureFormat)
FORMAT_ENV = 'WM_STATES_FIGURE_FORMATS'


class NoPostScriptTransparency(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith(
            "The PostScript backend does not support transparency"
        )


logging.getLogger("matplotlib.backends.backend_ps").addFilter(
    NoPostScriptTransparency()
)


def configure_figure_style(matplotlib_module: Any):
    """Set shared matplotlib font options for figure exports."""
    matplotlib_module.rcParams['font.family'] = 'Times New Roman'
    matplotlib_module.rcParams['font.serif'] = ['Times New Roman']
    matplotlib_module.rcParams['ps.fonttype'] = 42


def validate_figure_formats(formats) -> tuple[FigureFormat, ...]:
    """Validate the whole request before writing any files."""
    if isinstance(formats, str):
        raise ValueError('Figure formats must be a sequence of format names.')
    values = tuple(formats)
    if not values or any(value not in FIGURE_FORMATS for value in values):
        raise ValueError(f'Choose one or more figure formats from {FIGURE_FORMATS}; got {values}.')
    if len(set(values)) != len(values):
        raise ValueError('Figure formats must not contain duplicates.')
    return values


@contextmanager
def figure_format_context(formats):
    """Apply a runner's formats without leaking them into subsequent calls."""
    formats = validate_figure_formats(formats)
    previous = os.environ.get(FORMAT_ENV)
    os.environ[FORMAT_ENV] = ','.join(formats)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(FORMAT_ENV, None)
        else:
            os.environ[FORMAT_ENV] = previous


def save_figure(fig: Any, figure_file: Path, dpi: int = 300,
                **savefig_kwargs) -> tuple[Path, ...]:
    """Export requested formats and return their actual paths in request order.

    The filename supplies the common stem, not the export format. Formats come
    from WM_STATES_FIGURE_FORMATS (comma-separated; PNG when unset), configured
    by the runner's --figure-formats option. Layout options such as bbox_inches
    are forwarded unchanged. The caller owns the figure and closes it.

    PDF is written directly by Matplotlib's vector backend: paths and text stay
    vector, TrueType fonts are embedded, and embedded image streams use lossless
    Flate compression. Existing image/rasterized artists remain raster; DPI
    controls those artists, not the resolution of vector paths or text.
    """
    formats = validate_figure_formats(
        tuple(value.strip() for value in os.environ.get(FORMAT_ENV, 'png').split(','))
    )
    if 'format' in savefig_kwargs or 'backend' in savefig_kwargs:
        raise ValueError('Choose export formats through WM_STATES_FIGURE_FORMATS or --figure-formats.')
    figure_file = Path(figure_file)
    paths = tuple(figure_file.with_suffix('.' + suffix) for suffix in formats)
    figure_file.parent.mkdir(parents=True, exist_ok=True)
    for suffix, path in zip(formats, paths):
        options = dict(dpi=dpi, format='tiff' if suffix == 'tif' else suffix, **savefig_kwargs)
        if suffix == 'pdf':
            # Scoped settings also cover stages that do not call configure_figure_style.
            with matplotlib.rc_context({'pdf.fonttype': 42, 'pdf.use14corefonts': False,
                                        'pdf.compression': 6}):
                fig.savefig(path, backend='pdf', **options)
        else:
            fig.savefig(path, **options)
    return paths
