"""Summarize full-session screening rejection reasons."""

# Use one module namespace for direct CLI and package execution.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "scripts.next"

from scripts.next.cache_paths import stage_path
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import tyro
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.next.figure_exports import save_figure
from scripts.next.screening_names import reason_label, rejection_label, UNAVAILABLE_SUFFIX


@dataclass
class Config:
    cache_dir: Path = Path('cache/next_run')
    skip_not_applicable: bool = True
    dpi: int = 200


def main(config: Config):
    directory = stage_path(config.cache_dir, 'select', 'diagnostics')
    frame = pd.read_csv(directory / 'cell_rejection_diagnostics.csv', dtype={'session': str})
    # Reject old stage-number codes before writing any new summaries or figures.
    for reasons in frame.rejection_reason:
        rejection_label(reasons)
    rows = []
    for session, cells in frame.groupby('session', sort=True):
        counts = Counter(reason for value in cells.rejection_reason for reason in value.split('|')
                         if not (config.skip_not_applicable and reason.endswith(UNAVAILABLE_SUFFIX)))
        for reason, count in counts.items():
            rows.append({'session': session, 'reason': reason, 'reason_label': reason_label(reason), 'n_cells': count,
                         'percent': count / len(cells) * 100})
        fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
        ax.barh([reason_label(reason) for reason in counts], list(counts.values()))
        ax.set(title=f'{session}: screening reasons ({len(cells)} cells)', xlabel='Cell count')
        try:
            save_figure(fig, directory / 'figures' / 'reasons' / f'{session}.png', config.dpi)
        finally:
            plt.close(fig)
    pd.DataFrame(rows, columns=['session', 'reason', 'reason_label', 'n_cells', 'percent']).to_csv(
        directory / 'reject_reason_histograms_summary.csv', index=False)


if __name__ == '__main__':
    main(tyro.cli(Config))
