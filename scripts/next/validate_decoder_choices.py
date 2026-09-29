"""Two-class, outer-trial-held-out validation of decoder choices.

This supplementary experiment never reads OFF states or selects a setting by
downstream outcomes. Production caches define only a common session/population
panel. The fixed-cue/all-cell sensitivity removes activity-derived cue and cell
selection while remaining conditional on that session cohort.
"""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = 'scripts.next'

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed, parallel_config
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from scripts.next import cache_io
from scripts.next.cache_paths import primary_cache
from scripts.next.common import compute_binned_rates, load_session
from scripts.next.decoder_models import (
    DecoderModel, SVMKernel, create_base_decoder, fit_calibrated_decoder,
    make_logistic_calibration_cv_splits, select_classifier_c,
)

METHODS = {
    'weighted_search_calibrated': {'weighted': True, 'search': True, 'calibrate': True},
    'downsampled_search_calibrated': {'weighted': False, 'search': True, 'calibrate': True},
    'weighted_fixed1_calibrated': {'weighted': True, 'search': False, 'calibrate': True},
    'weighted_fixed001_calibrated': {'weighted': True, 'search': False, 'calibrate': True, 'c': .01},
    'weighted_search_raw': {'weighted': True, 'search': True, 'calibrate': False},
}
OUTER_SEEDS = (20260930, 20261001, 20261002)
BINS = (500, 950, 1350)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def equal_prior_weights(labels):
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.all(np.isin(labels, [0, 1])):
        raise ValueError('Expected one-dimensional binary labels.')
    counts = np.bincount(labels.astype(int), minlength=2)
    if counts.min() == 0:
        raise ValueError('Both classes must be present.')
    return 0.5 / counts[labels.astype(int)]


def probability_metrics(labels, probabilities):
    """Each cue receives total weight one half, regardless of observed counts."""
    labels, probabilities = np.asarray(labels), np.asarray(probabilities, dtype=float)
    if probabilities.shape != labels.shape or not np.all(np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError('Expected one probability in [0, 1] per label.')
    weights = equal_prior_weights(labels)
    safe = np.clip(probabilities, np.finfo(float).eps, 1 - np.finfo(float).eps)
    bins = np.minimum((probabilities * 10).astype(int), 9)
    reliability = []
    for i in range(10):
        keep = bins == i
        mass = float(weights[keep].sum())
        reliability.append({'lower': i / 10, 'upper': (i + 1) / 10,
                            'count': int(keep.sum()), 'weight': mass,
                            'mean_probability': float(np.average(probabilities[keep], weights=weights[keep])) if mass else None,
                            'observed_positive_fraction': float(np.average(labels[keep], weights=weights[keep])) if mass else None})
    return {'brier': float(np.sum(weights * (probabilities - labels)**2)),
            'log_loss': float(-np.sum(weights * (labels * np.log(safe) + (1 - labels) * np.log1p(-safe)))),
            'balanced_accuracy': float(np.sum(weights * ((probabilities >= 0.5) == labels))),
            'auc': float(roc_auc_score(labels, probabilities, sample_weight=weights)),
            'mean_probability_minus_prevalence': float(np.sum(weights * probabilities) - 0.5),
            'extreme_probability_fraction': float(np.sum(weights * ((probabilities < .1) | (probabilities > .9)))),
            'ece_10_equal_width': float(sum(r['weight'] * abs(r['mean_probability'] - r['observed_positive_fraction']) for r in reliability if r['weight'])),
            'reliability': reliability}


def choose_training_indices(labels, outer_train, *, weighted, seed):
    """Downsample once inside the outer training fold; no held-out activity."""
    outer_train = np.asarray(outer_train)
    if weighted:
        return outer_train.copy()
    by_class = [outer_train[labels[outer_train] == c] for c in (1, 0)]
    count = min(map(len, by_class))
    if count < 5:
        raise ValueError('At least five training examples of each class are required.')
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.choice(group, count, replace=False) for group in by_class])


def fit_predict(X, labels, training, test, *, method, seed):
    options = METHODS[method]
    chosen = choose_training_indices(labels, training, weighted=options['weighted'], seed=seed)
    train_x, train_y = X[chosen], labels[chosen]
    weight = 'balanced' if options['weighted'] else None
    groups = np.arange(chosen.size)
    c = select_classifier_c(train_x, train_y, groups, DecoderModel.LOGISTIC_REGRESSION,
                            SVMKernel.LINEAR, seed, class_weight=weight) if options['search'] else options.get('c', 1.0)
    model = create_base_decoder(c, DecoderModel.LOGISTIC_REGRESSION, SVMKernel.LINEAR, seed, class_weight=weight)
    if options['calibrate']:
        splits, _ = make_logistic_calibration_cv_splits(train_y, groups, 5, seed)
        model = fit_calibrated_decoder(model, train_x, train_y, 'sigmoid', splits,
                                        balanced_class_weights=options['weighted'])
    else:
        model.fit(train_x, train_y)
    probability = model.predict_proba(X[test])[:, np.flatnonzero(model.classes_ == 1)[0]]
    return probability, float(c), chosen


def evaluate_task(task, seeds=OUTER_SEEDS):
    """Fit one session/pair/bin with common outer folds for every procedure."""
    started = time.monotonic()
    X, labels = task['X'], task['labels']
    methods = {}
    with threadpool_limits(limits=1):
        for method in METHODS:
            predictions, selected, sizes = [], [], []
            for seed in seeds:
                probabilities = np.full(labels.size, np.nan)
                folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X, labels)
                for fold, (training, test) in enumerate(folds):
                    fit_seed = int(np.random.SeedSequence([seed, fold]).generate_state(1)[0])
                    probability, c, chosen = fit_predict(X, labels, training, test, method=method, seed=fit_seed)
                    probabilities[test] = probability
                    selected.append(c)
                    sizes.append(int(chosen.size))
                predictions.append(probabilities)
            predictions = np.stack(predictions)
            methods[method] = {
                'seed_metrics': [probability_metrics(labels, p) for p in predictions],
                'ensemble_metrics': probability_metrics(labels, predictions.mean(axis=0)),
                'probability_sd_across_outer_seeds': float(np.average(predictions.std(axis=0), weights=equal_prior_weights(labels))),
                'selected_c_counts': {str(c): selected.count(c) for c in (1.0, .1, .01)},
                'training_size_range': [min(sizes), max(sizes)],
                'predictions': predictions,
            }
    methods['constant_half'] = {'seed_metrics': [probability_metrics(labels, np.full(labels.size, .5)) for _ in seeds],
                                'ensemble_metrics': probability_metrics(labels, np.full(labels.size, .5)),
                                'probability_sd_across_outer_seeds': 0.0, 'selected_c_counts': {}, 'training_size_range': []}
    return {**{k: v for k, v in task.items() if k not in ('X', 'labels')}, 'labels': labels,
            'methods': methods, 'elapsed_seconds': time.monotonic() - started}


def build_panel(root, *, benchmark=False):
    """Define sessions from cache; prepare raw features without using outcomes."""
    sources = {}
    reference = None
    for run in range(1, 6):
        path = primary_cache(root / f'cache/next_run_{run:03d}', 'decoding_confidence.pkl')
        sources[str(path.relative_to(root))] = file_hash(path)
        rows = cache_io.read(path)
        panel = {r['session']: {'cue': int(r['cue']), 'cells': np.asarray(r['cell_idx']).tolist(),
                                'preferred_trials': np.asarray(r['trial_idx']).tolist()} for r in rows}
        if reference is None:
            reference = panel
        elif panel != reference:
            raise ValueError('Runs do not share the same session, cue, cell and preferred-trial panel.')
    # Largest cached feature count is chosen using metadata only, before scores.
    sessions = [max(reference, key=lambda s: len(reference[s]['cells']))] if benchmark else sorted(reference)
    tasks, session_metadata, exclusions = [], [], []
    for session in sessions:
        path = root / f'data/nature/{session}.mat'
        sources[str(path.relative_to(root))] = file_hash(path)
        spikes, times, cues, correct = load_session(path)
        rates = compute_binned_rates(spikes, times, BINS, 50).astype(np.float64)
        del spikes
        definition = reference[session]
        session_metadata.append({'session': session, 'cached_cue': definition['cue'],
                                 'cached_stationary_cells': len(definition['cells']), 'all_cells': rates.shape[2]})
        pairs = [('cached_population', definition['cue'], definition['cells'], tuple(range(len(BINS))))]
        if not benchmark:
            pairs.extend(('fixed_cues_all_cells', cue, list(range(rates.shape[2])), (1,)) for cue in range(1, 5))
        for scope, cue, cells, bins in pairs:
            opposite = (cue + 3) % 8 + 1
            trials = np.flatnonzero(correct & np.isin(cues, [cue, opposite]))
            labels = (cues[trials] == cue).astype(np.int8)
            counts = np.bincount(labels, minlength=2)
            if counts.min() < 7:
                exclusions.append({'session': session, 'scope': scope, 'cue': cue, 'counts': counts.tolist(), 'reason': 'fewer than 7 correct trials in one class'})
                continue
            if scope == 'cached_population' and not np.array_equal(trials[labels == 1], definition['preferred_trials']):
                raise ValueError('Raw preferred trials disagree with cached identities.')
            for b in bins:
                tasks.append({'session': session, 'scope': scope, 'cue': cue, 'opposite_cue': opposite,
                              'bin_start_ms': BINS[b], 'trial_ids': trials.tolist(), 'class_counts': counts.tolist(),
                              'n_cells': len(cells), 'X': rates[np.ix_(trials, [b], cells)][:, 0], 'labels': labels})
    return tasks, session_metadata, exclusions, sources


METRICS = ('brier', 'log_loss', 'balanced_accuracy', 'auc', 'mean_probability_minus_prevalence',
           'extreme_probability_fraction', 'ece_10_equal_width')


def aggregate_results(results, bootstrap=20000):
    """Average folds through OOF predictions, then seeds/bins/pairs within session."""
    scopes = {}
    rng = np.random.default_rng(20260930)
    for scope in sorted({r['scope'] for r in results}):
        rows = [r for r in results if r['scope'] == scope]
        sessions = sorted({r['session'] for r in rows})
        per_session = {}
        for session in sessions:
            local = [r for r in rows if r['session'] == session]
            per_session[session] = {}
            for method in (*METHODS, 'constant_half'):
                values = [m for r in local for m in r['methods'][method]['seed_metrics']]
                per_session[session][method] = {key: float(np.mean([v[key] for v in values])) for key in METRICS}
                per_session[session][method]['probability_sd_across_outer_seeds'] = float(np.mean([r['methods'][method]['probability_sd_across_outer_seeds'] for r in local]))
        aggregate = {method: {key: float(np.mean([per_session[s][method][key] for s in sessions]))
                               for key in (*METRICS, 'probability_sd_across_outer_seeds')}
                     for method in (*METHODS, 'constant_half')}
        indices = rng.integers(len(sessions), size=(bootstrap, len(sessions)))
        comparisons = {}
        for method in ('downsampled_search_calibrated', 'weighted_fixed1_calibrated', 'weighted_fixed001_calibrated', 'weighted_search_raw', 'constant_half'):
            comparisons[method] = {}
            for metric in ('brier', 'log_loss', 'balanced_accuracy', 'auc'):
                differences = np.asarray([per_session[s]['weighted_search_calibrated'][metric] - per_session[s][method][metric] for s in sessions])
                interval = np.quantile(differences[indices].mean(axis=1), [.025, .975])
                favorable = differences < 0 if metric in ('brier', 'log_loss') else differences > 0
                comparisons[method][metric] = {'default_minus_comparator': float(differences.mean()),
                                               'session_bootstrap_95_percentile_interval': interval.tolist(),
                                               'sessions_favoring_default': int(favorable.sum()), 'n_sessions': len(sessions)}
        ensemble = {}
        if all('ensemble_metrics' in r['methods'][m] for r in rows for m in METHODS):
            for method in (*METHODS, 'constant_half'):
                values = {s: {metric: float(np.mean([r['methods'][method]['ensemble_metrics'][metric] for r in rows if r['session'] == s]))
                              for metric in METRICS} for s in sessions}
                changes = {}
                for metric in ('brier', 'log_loss', 'balanced_accuracy', 'auc'):
                    delta = np.asarray([values[s][metric] - per_session[s][method][metric] for s in sessions])
                    changes[metric] = {'ensemble_minus_mean_individual': float(delta.mean()),
                                       'session_bootstrap_95_percentile_interval': np.quantile(delta[indices].mean(axis=1), [.025, .975]).tolist()}
                ensemble[method] = {'aggregate_equal_session': {metric: float(np.mean([values[s][metric] for s in sessions])) for metric in METRICS},
                                    'per_session': values, 'paired_change_from_individual': changes}
        reliability = {}
        for method in (*METHODS, 'constant_half'):
            weighted_bins = np.zeros((10, 3))
            for session in sessions:
                local_metrics = [m for r in rows if r['session'] == session for m in r['methods'][method]['seed_metrics']]
                for metric in local_metrics:
                    for i, b in enumerate(metric.get('reliability', [])):
                        if b['weight']:
                            factor = b['weight'] / len(local_metrics) / len(sessions)
                            weighted_bins[i] += factor * np.asarray([1, b['mean_probability'], b['observed_positive_fraction']])
            reliability[method] = [{'bin': i, 'weight': float(b[0]),
                                    'mean_probability': float(b[1] / b[0]) if b[0] else None,
                                    'observed_positive_fraction': float(b[2] / b[0]) if b[0] else None} for i, b in enumerate(weighted_bins)]
        scopes[scope] = {'n_tasks': len(rows), 'n_sessions': len(sessions), 'aggregate_equal_session': aggregate,
                         'per_session': per_session, 'paired_comparisons': comparisons,
                         'three_seed_ensemble': ensemble, 'reliability_equal_session': reliability}
    return scopes


def plot_summary(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scripts.next.figure_exports import save_figure
    labels = ['Weights; C search; calibrated', 'Downsample; C search; calibrated',
              'Weights; C=1; calibrated', 'Weights; C=0.01; calibrated', 'Weights; C search; raw']
    colors = ['#176b87', '#b17a1c', '#87919e', '#319c66', '#c14b4b']
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for row, (scope, title) in enumerate([('cached_population', 'Cached population / three delay bins'), ('fixed_cues_all_cells', 'All cells / four fixed cue pairs / 950 ms')]):
        result = summary[scope]
        ax = axes[row, 0]
        for i, (method, color) in enumerate(zip(METHODS, colors)):
            values = [v[method]['brier'] for v in result['per_session'].values()]
            ax.scatter(values, np.full(len(values), i), color=color, alpha=.35, s=18)
            ax.scatter(result['aggregate_equal_session'][method]['brier'], i, color=color, edgecolor='black', marker='D', s=65, zorder=5)
        ax.axvline(.25, linestyle='--', color='black', linewidth=1, label='Constant 0.5 baseline')
        ax.set_yticks(range(len(labels)), labels)
        ax.invert_yaxis()
        ax.set_xlabel('Equal-prior Brier score (lower is better)')
        ax.set_title(title)
        ax.legend(fontsize=8, loc='lower left')
        ax = axes[row, 1]
        ax.plot([0, 1], [0, 1], '--', color='black', linewidth=1)
        for method, label, color in zip(METHODS, labels, colors):
            bins = [b for b in result['reliability_equal_session'][method] if b['weight'] > .005]
            ax.plot([b['mean_probability'] for b in bins], [b['observed_positive_fraction'] for b in bins], 'o-', color=color, label=label, markersize=4)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Mean predicted probability', ylabel='Equal-prior positive fraction', title='Descriptive reliability (10 fixed bins)')
    axes[0, 1].legend(fontsize=7, loc='upper left')
    fig.suptitle('Held-out two-class decoder validation: 25 sessions', fontsize=15)
    fig.supxlabel('Dots: session means; diamonds: equal-session means. Three outer CV seeds, five folds.\nReliability combines equal cue, seed, bin/pair and session weights; bins with ≤0.5% mass omitted.', fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('docs/validation/decoder-choice-validation.json'))
    parser.add_argument('--work-dir', type=Path, default=Path('cache/decoder_choice_validation_20260930'))
    parser.add_argument('--figure', type=Path, default=Path('docs/assets/decoder-choice-validation.png'))
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        parser.error('workers must be between 1 and 6')
    started = time.monotonic()
    root = args.root.resolve()
    tasks, sessions, exclusions, sources = build_panel(root, benchmark=args.benchmark)
    for name in ('scripts/next/validate_decoder_choices.py', 'scripts/next/decoder_models.py', 'scripts/next/common.py'):
        sources[name] = file_hash(root / name)
    seeds = OUTER_SEEDS[:1] if args.benchmark else OUTER_SEEDS
    args.work_dir.mkdir(parents=True, exist_ok=True)
    design = {'date': '2026-09-30', 'benchmark': args.benchmark, 'outer_seeds': list(seeds),
              'outer_folds': 5, 'calibration_folds': 5, 'inner_search_folds': 5,
              'c_grid': [1.0, .1, .01], 'c_search_scoring': 'balanced_accuracy',
              'methods': METHODS, 'bin_starts_ms': list(BINS), 'window_ms': 50,
              'fixed_cue_sensitivity_bin_start_ms': 950, 'fixed_cue_pairs': [[1,5],[2,6],[3,7],[4,8]],
              'minimum_trials_per_class': 7, 'workers': args.workers,
              'sessions': sessions, 'exclusions': exclusions, 'task_count': len(tasks),
              'sources_sha256': sources,
              'inference': 'Equal-session paired 20,000-draw bootstrap; percentile 95% intervals are descriptive, not multiplicity-adjusted. Sessions are the resampling units; independence across animals is not established.',
              'limitations': [
                  'Cached-population validation conditions on full-session cell screening and activity-derived cue selection; it is not a fully nested estimate of selection plus decoding.',
                  'Fixed-cue sensitivity uses all cells and all four prespecified cue pairs without activity-derived screening, but conditions on the original 25-session cohort.',
                  'Five-fold outer CV trains on 80% of the eligible trials, whereas production uses leave-one-preferred-trial-out fitting.',
                  'Three outer seeds measure combined split and training-set sensitivity; they do not isolate balancing-seed variance.',
                  'C selection and sigmoid calibration mirror production: C is chosen using outer-training data before calibration OOF folds; hyperparameter selection is not nested again inside each calibration fold.',
                  'No permutation nulls or state outcomes are fitted; this experiment validates two-class probabilities, not OFF-state error control.',
                  'ECE and reliability curves are descriptive finite-bin diagnostics, not calibrated hypothesis tests.',
                  'Three-seed ensembles contain only held-out predictions. Brier and log loss cannot worsen relative to mean component scores by convexity. This does not validate a production ensemble or its matched nulls.',
              ]}
    # Persist the design and input hashes before obtaining fitted results.
    design_path = args.work_dir / ('benchmark-design.json' if args.benchmark else 'design.json')
    design_path.write_text(json.dumps(design, indent=2) + '\n')
    print(f'Saved prespecified design: {design_path}; {len(tasks)} tasks', flush=True)
    with parallel_config(backend='loky', n_jobs=args.workers, inner_max_num_threads=1):
        results = Parallel(verbose=10)(delayed(evaluate_task)(task, seeds) for task in tasks)
    predictions = {}
    for i, row in enumerate(results):
        predictions[f'task_{i}_labels'] = row.pop('labels')
        for method, value in row['methods'].items():
            if 'predictions' in value:
                predictions[f'task_{i}_{method}'] = value.pop('predictions')
    prediction_path = args.work_dir / ('benchmark-predictions.npz' if args.benchmark else 'predictions.npz')
    np.savez_compressed(prediction_path, **predictions)
    for name, before in sources.items():
        if file_hash(root / name) != before:
            raise ValueError(f'Source changed while validating: {name}')
    summary = aggregate_results(results)
    # Keep the tracked evidence compact; full probabilities are in the ignored
    # archive and equal-session reliability curves are retained in the summary.
    for row in results:
        for method in row['methods'].values():
            for metric in (*method['seed_metrics'], method['ensemble_metrics']):
                metric.pop('reliability', None)
    result = {'design': design, 'elapsed_seconds': time.monotonic() - started,
              'prediction_archive': str(prediction_path), 'prediction_archive_sha256': file_hash(prediction_path),
              'summary': summary, 'tasks': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f'Wrote {args.output}; elapsed {result["elapsed_seconds"]:.1f} s', flush=True)
    if not args.benchmark:
        plot_summary(summary, args.figure)
    print(json.dumps({scope: rows['aggregate_equal_session'] for scope, rows in result['summary'].items()}, indent=2), flush=True)


if __name__ == '__main__':
    main()
