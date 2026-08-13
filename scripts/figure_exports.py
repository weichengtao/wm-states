import logging
from pathlib import Path
from typing import Any


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


def save_figure_all_formats(fig: Any, figure_file: Path, dpi: int = 300):
    """Save a matplotlib figure as PNG, TIFF, and EPS using the same base filename."""
    figure_file = Path(figure_file)
    figure_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_file.with_suffix('.png'), dpi=dpi)
    fig.savefig(figure_file.with_suffix('.tif'), dpi=dpi, format='tiff')
    fig.savefig(figure_file.with_suffix('.eps'), dpi=dpi, format='eps')


def save_figure_png_only(fig: Any, figure_file: Path, dpi: int = 300):
    """Save a matplotlib figure as PNG using the given base filename."""
    figure_file = Path(figure_file)
    figure_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_file.with_suffix('.png'), dpi=dpi)
