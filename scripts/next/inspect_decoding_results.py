"""Inspect observed confidence, null distributions and state labels."""

# Use one module namespace for direct CLI and package execution.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "scripts.next"

from scripts.next.cache_paths import primary_cache, stage_path
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import tyro
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.next import cache_io
from scripts.next.figure_exports import save_figure


@dataclass
class Config:
    session: str
    trial: list[int]  # One cache row, or inclusive first/last rows.
    time_bin_start: list[float]  # One time, or inclusive start/end times.
    cache_dir: Path = Path('cache/next_run')
    with_null: bool = False
    with_state: bool = False


def select_indices(values, requested, name):
    if len(requested) not in (1, 2) or requested[0] > requested[-1]:
        raise ValueError(f'{name}: supply one value or increasing inclusive endpoints.')
    selected = np.flatnonzero((values >= requested[0]) & (values <= requested[-1]))
    if not selected.size or not np.isclose(values[selected[0]], requested[0]) or not np.isclose(values[selected[-1]], requested[-1]):
        raise ValueError(f'{name}: endpoints must exist in the cache.')
    return selected


def main(config: Config):
    matches = [r for r in cache_io.read(primary_cache(config.cache_dir, 'decoding_confidence.pkl')) if str(r['session']) == config.session]
    if len(matches) != 1:
        raise ValueError(f'Expected one decoded session {config.session}.')
    result = matches[0]
    observed = result['decoding_confidence']
    rows = select_indices(np.arange(observed.shape[0]), config.trial, 'trial')
    bins = select_indices(result['time_bins'], config.time_bin_start, 'time-bin-start')
    states = None
    if config.with_state:
        matches = [r for r in cache_io.read(primary_cache(config.cache_dir, 'on_off_states.pkl')) if str(r['session']) == config.session]
        if len(matches) != 1:
            raise ValueError('No unique matching state result.')
        states = matches[0]
        if not np.array_equal(states['trial_idx'], result['trial_idx']) or not np.array_equal(states['time_bins'], result['time_bins']) or states.get('decoding_fingerprint') != result.get('fingerprint'):
            raise ValueError('States and decoding caches differ; rerun state detection.')
    for row in rows:
        fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
        times = result['time_bins'][bins]
        ax.plot(times, observed[row, bins], marker='.', label='Observed confidence')
        null = result['decoding_confidence_null'][row, bins]
        if config.with_null:
            if not null.shape[1]:
                raise ValueError('No null estimates were cached.')
            ax.plot(times, null.mean(axis=1), label='Null mean', color='gray')
            low, high = np.percentile(null, [2.5, 97.5], axis=1)
            ax.fill_between(times, low, high, color='gray', alpha=0.2, label='Null 95% range')
        if states is not None:
            for key, color in [('on_state_mask', 'tab:green'), ('off_state_mask', 'tab:orange')]:
                mask = states[key][row, bins]
                ax.scatter(times[mask], observed[row, bins][mask], color=color, label=key.replace('_mask', ''))
        ax.set(xlabel='Time (ms)', ylabel='P(preferred cue)', ylim=(0, 1),
               title=f'{config.session}: trial {result["trial_idx"][row]} (row {row})')
        ax.legend()
        output = stage_path(config.cache_dir, 'decode', 'figures', 'inspection', f'{config.session}_trial_{row}.png')
        outputs = save_figure(fig, output)
        plt.close(fig)
        for path in outputs:
            print(path)


if __name__ == '__main__':
    main(tyro.cli(Config))
