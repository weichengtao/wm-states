"""One export policy for every next-pipeline figure, including diagnostics."""
import logging
import os
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, get_args
import warnings

import matplotlib
from matplotlib import font_manager
from matplotlib.figure import Figure
from matplotlib.text import Text

FigureFormat = Literal['png', 'tif', 'eps', 'pdf']
FIGURE_FORMATS = get_args(FigureFormat)
FORMAT_ENV = 'WM_STATES_FIGURE_FORMATS'
FONT_ENV = 'WM_STATES_FIGURE_FONT'
DEFAULT_FIGURE_FONT = 'DejaVu Sans'


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
    matplotlib_module.rcParams['font.family'] = current_figure_font()
    matplotlib_module.rcParams['ps.fonttype'] = 42


def validate_figure_font(font: str) -> str:
    """Accept one named family, rejecting malformed settings before a run starts."""
    if not isinstance(font, str) or not font.strip() or len(font.strip()) > 120 or any(
        ord(character) < 32 or ord(character) == 127 for character in font
    ):
        raise ValueError('Figure font must be a nonblank font family name of at most 120 characters, without control characters.')
    return font.strip()


@lru_cache(maxsize=64)
def resolve_figure_font(font: str) -> str:
    """Resolve installed families once per process, with an explicit safe fallback."""
    font = validate_figure_font(font)
    try:
        path = font_manager.findfont(font_manager.FontProperties(family=[font]),
                                     fallback_to_default=False)
    except ValueError:
        warnings.warn(f'Figure font {font!r} is unavailable on this computer; using '
                      f'{DEFAULT_FIGURE_FONT!r}. Install the requested font and restart '
                      'the dashboard or command before exporting journal figures.',
                      UserWarning, stacklevel=2)
        return DEFAULT_FIGURE_FONT
    return font_manager.FontProperties(fname=path).get_name()


def current_figure_font() -> str:
    return resolve_figure_font(os.environ.get(FONT_ENV, DEFAULT_FIGURE_FONT))


@contextmanager
def figure_font_context(font: str):
    """Apply a run font at figure creation and inherit it in child processes."""
    font = resolve_figure_font(font)
    previous = os.environ.get(FONT_ENV)
    os.environ[FONT_ENV] = font
    try:
        with matplotlib.rc_context({'font.family': font, 'ps.fonttype': 42}):
            yield font
    finally:
        if previous is None:
            os.environ.pop(FONT_ENV, None)
        else:
            os.environ[FONT_ENV] = previous


def configure_worker_figure_exports(formats: tuple[FigureFormat, ...], font: str):
    """Initialize a loky worker explicitly, including pools from repeated API runs."""
    os.environ[FORMAT_ENV] = ','.join(validate_figure_formats(formats))
    os.environ[FONT_ENV] = resolve_figure_font(font)
    configure_figure_style(matplotlib)


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
    font = current_figure_font()
    figure_file = Path(figure_file)
    paths = tuple(figure_file.with_suffix('.' + suffix) for suffix in formats)
    figure_file.parent.mkdir(parents=True, exist_ok=True)
    # Figures created outside the runner (or before a font change) still obey
    # the same export policy. Restore their original artist styles afterwards.
    text_fonts = [(text, text.get_fontfamily()) for text in fig.findobj(Text)] if isinstance(fig, Figure) else []
    try:
        for text, _ in text_fonts:
            text.set_fontfamily(font)
        with matplotlib.rc_context({'font.family': font, 'ps.fonttype': 42}):
            for suffix, path in zip(formats, paths):
                options = dict(dpi=dpi, format='tiff' if suffix == 'tif' else suffix, **savefig_kwargs)
                if suffix == 'pdf':
                    with matplotlib.rc_context({'pdf.fonttype': 42, 'pdf.use14corefonts': False,
                                                'pdf.compression': 6}):
                        fig.savefig(path, backend='pdf', **options)
                else:
                    fig.savefig(path, **options)
    finally:
        for text, family in text_fonts:
            text.set_fontfamily(family)
    return paths


# Every figure writer imports this module before creating its figures. This
# covers standalone commands and fresh workers as well as the pipeline runner.
configure_figure_style(matplotlib)
