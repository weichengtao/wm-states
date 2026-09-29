"""Held-out regularization and probability-shrinkage sensitivity.

Frozen exploratory curves, not an outer-test-selected production procedure.
Reuse the original two-class panel and folds without modifying its evidence.
"""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = 'scripts.next'

import argparse
import json
from pathlib import Path
import platform
import signal
import time
import warnings

import numpy as np
import sklearn
from joblib import Parallel, delayed, parallel_config
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from scripts.next.decoder_models import (
    DecoderModel, SVMKernel, create_base_decoder, fit_calibrated_decoder,
    make_logistic_calibration_cv_splits,
)
from scripts.next.validate_decoder_choices import (
    OUTER_SEEDS, build_panel, equal_prior_weights, file_hash, probability_metrics,
)

CS = (1., .1, .01, .001, .0001, .00001, .000001, .00000001, .0000000001)
ALPHAS = (0., .1, .25, .5, .75, 1.)
REFERENCE = 'weighted_search_calibrated'
SCORE_KEYS = ('brier', 'log_loss', 'balanced_accuracy', 'auc', 'mean_abs_probability_minus_half',
              'squared_probability_spread', 'label_alignment_term', 'brier_skill_vs_half',
              'information_gain_nats_vs_half', 'fraction_within_001_of_half', 'exact_half_fraction',
              'mean_probability_minus_prevalence', 'base_margin_auc')


def method_name(c, calibrated=True):
    return f"{'calibrated' if calibrated else 'raw'}_C{c:g}"


def shrink(probabilities, alpha):
    p = np.asarray(probabilities, dtype=float)
    if not np.isfinite(alpha) or not 0 <= alpha <= 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Shrinkage requires probabilities and alpha in [0, 1].')
    if alpha == 1:
        return p.copy()
    if alpha == 0:
        return np.full_like(p, .5)
    return .5 + alpha * (p - .5)


def score(labels, probabilities, margins=None):
    value = probability_metrics(labels, probabilities)
    p, y = np.asarray(probabilities), np.asarray(labels)
    w = equal_prior_weights(y)
    spread = float(np.sum(w * (p - .5)**2))
    alignment = float(2 * np.sum(w * (p - .5) * (y - .5)))
    if not np.isclose(value['brier'], .25 + spread - alignment, atol=1e-14):
        raise ValueError('Brier algebraic decomposition failed.')
    value.update(mean_abs_probability_minus_half=float(np.sum(w * abs(p - .5))),
                 squared_probability_spread=spread, label_alignment_term=alignment,
                 brier_skill_vs_half=1 - value['brier'] / .25,
                 information_gain_nats_vs_half=float(np.log(2) - value['log_loss']),
                 fraction_within_001_of_half=float(np.sum(w * (abs(p - .5) <= .01))),
                 exact_half_fraction=float(np.sum(w * (p == .5))),
                 base_margin_auc=float(roc_auc_score(y, margins, sample_weight=w)) if margins is not None else None)
    return {key: value[key] for key in SCORE_KEYS}


def fit_fold(X, labels, training, test, *, c, seed):
    """Fit production sigmoid calibration; its final refit also supplies raw p.

    ensemble=False refits the base on all outer training rows, independently of
    the calibrator. The raw branch uses that same final base without its sigmoid.
    """
    X_train, y_train = X[training], labels[training]
    base = create_base_decoder(c, DecoderModel.LOGISTIC_REGRESSION, SVMKernel.LINEAR,
                               seed, class_weight='balanced')
    splits, _ = make_logistic_calibration_cv_splits(y_train, np.arange(len(training)), 5, seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model = fit_calibrated_decoder(base, X_train, y_train, 'sigmoid', splits,
                                       balanced_class_weights=True)
    if len(model.calibrated_classifiers_) != 1:
        raise ValueError('Expected production pooled calibration with a single final base.')
    calibrated = model.calibrated_classifiers_[0]
    estimator = calibrated.estimator
    calibrator = calibrated.calibrators[0]
    margin = estimator.decision_function(X[test])
    raw = estimator.predict_proba(X[test])[:, np.flatnonzero(estimator.classes_ == 1)[0]]
    probability = model.predict_proba(X[test])[:, np.flatnonzero(model.classes_ == 1)[0]]
    return {'raw': raw, 'calibrated': probability, 'margin': margin,
            'diagnostics': {'coefficient_l2': float(np.linalg.norm(estimator.named_steps['classifier'].coef_)),
                            'test_margin_sd': float(np.std(margin)),
                            'test_margin_rms': float(np.sqrt(np.mean(margin**2))),
                            'calibration_a': float(calibrator.a_), 'calibration_b': float(calibrator.b_),
                            'training_class_counts': np.bincount(y_train, minlength=2).tolist(),
                            'warnings': [str(w.message) for w in caught]}}


def evaluate_task(task, reference, *, cs=CS, seeds=OUTER_SEEDS):
    start = time.monotonic()
    X, labels = task['X'], task['labels']
    if reference.shape != (len(seeds), len(labels)):
        raise ValueError('Reference prediction shape does not match held-out trials/seeds.')
    methods, diagnostics = {}, {}
    with threadpool_limits(limits=1):
        for c in cs:
            raw, calibrated, margins = (np.full(reference.shape, np.nan) for _ in range(3))
            folds = []
            for s, seed in enumerate(seeds):
                splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                for fold, (training, test) in enumerate(splitter.split(X, labels)):
                    fit_seed = int(np.random.SeedSequence([seed, fold]).generate_state(1)[0])
                    fitted = fit_fold(X, labels, training, test, c=c, seed=fit_seed)
                    raw[s, test], calibrated[s, test], margins[s, test] = fitted['raw'], fitted['calibrated'], fitted['margin']
                    folds.append({'outer_seed': seed, 'fold': fold, **fitted['diagnostics']})
            diagnostics[str(c)] = folds
            for is_calibrated, predictions in ((False, raw), (True, calibrated)):
                if not np.isfinite(predictions).all():
                    raise ValueError('Incomplete outer predictions.')
                methods[method_name(c, is_calibrated)] = {
                    'seed_metrics': [score(labels, p, m) for p, m in zip(predictions, margins)],
                    'predictions': predictions}
    methods[REFERENCE] = {'seed_metrics': [score(labels, p) for p in reference], 'predictions': reference}
    for alpha in ALPHAS:
        predictions = shrink(reference, alpha)
        methods[f'shrink_alpha_{alpha:g}'] = {'seed_metrics': [score(labels, p) for p in predictions], 'predictions': predictions}
    return {**{k: v for k, v in task.items() if k not in ('X', 'labels')},
            'methods': methods, 'fold_diagnostics': diagnostics, 'elapsed_seconds': time.monotonic() - start}


def validate_alignment(tasks, old, archive):
    if list(old['design']['outer_seeds']) != list(OUTER_SEEDS) or len(tasks) != len(old['tasks']):
        raise ValueError('The original validation panel or seeds changed.')
    keys = ('session', 'scope', 'cue', 'opposite_cue', 'bin_start_ms', 'trial_ids', 'class_counts', 'n_cells')
    for index, (task, previous) in enumerate(zip(tasks, old['tasks'])):
        if any(task[k] != previous[k] for k in keys):
            raise ValueError(f'Task {index} differs from the original validation.')
        if not np.array_equal(task['labels'], archive[f'task_{index}_labels']):
            raise ValueError('Reference labels disagree with raw trials.')


def aggregate(results, mapping, *, bootstrap=20000):
    def mean(values):
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise ValueError('Partially missing metric across tasks or sessions.')
        return float(np.mean(values))
    methods = tuple(results[0]['methods'])
    output = {}
    for scope in sorted({r['scope'] for r in results}):
        rows = [r for r in results if r['scope'] == scope]
        sessions = sorted({r['session'] for r in rows})
        if set(sessions) != set(mapping):
            raise ValueError('Animal mapping must cover the panel exactly.')
        per_session = {s: {m: {k: mean([v[k] for r in rows if r['session'] == s
                                                for v in r['methods'][m]['seed_metrics']])
                              for k in SCORE_KEYS} for m in methods} for s in sessions}
        means = {m: {k: mean([per_session[s][m][k] for s in sessions]) for k in SCORE_KEYS} for m in methods}
        indices = np.random.default_rng(20260930).integers(len(sessions), size=(bootstrap, len(sessions)))
        comparisons = {}
        for reference in (REFERENCE, method_name(.01), 'shrink_alpha_0'):
            comparisons[reference] = {}
            for m in methods:
                comparisons[reference][m] = {}
                for k in ('brier', 'log_loss', 'balanced_accuracy', 'auc'):
                    difference = np.array([per_session[s][m][k] - per_session[s][reference][k] for s in sessions])
                    comparisons[reference][m][k] = {
                        'method_minus_reference': float(difference.mean()),
                        'session_bootstrap_95_percentile': np.quantile(difference[indices].mean(axis=1), [.025, .975]).tolist(),
                        'sessions_favoring_method': int(np.sum(difference < 0 if k in ('brier', 'log_loss') else difference > 0))}
        by_animal = {animal: {m: {k: mean([per_session[s][m][k] for s in sessions if mapping[s] == animal])
                                  for k in SCORE_KEYS} for m in methods} for animal in sorted(set(mapping.values()))}
        output[scope] = {'n_sessions': len(sessions), 'n_tasks': len(rows), 'aggregate_equal_session': means,
                         'per_session': per_session, 'per_animal_equal_session': by_animal, 'paired_comparisons': comparisons}
    return output


def plot_summary(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scripts.next.figure_exports import save_figure
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    columns = [('brier', 'Brier score ↓', .25), ('log_loss', 'Log loss ↓', np.log(2)),
               ('auc', 'ROC AUC ↑', .5), ('mean_abs_probability_minus_half', 'Mean |p − 0.5|', 0)]
    for row, (scope, title) in enumerate([('cached_population', 'Cached population / three bins'), ('fixed_cues_all_cells', 'All cells / four fixed cue pairs')]):
        values = summary[scope]['aggregate_equal_session']
        for col, (metric, label, baseline) in enumerate(columns):
            ax = axes[row, col]
            for calibrated, color in ((True, '#176b87'), (False, '#c16c39')):
                ax.plot(CS, [values[method_name(c, calibrated)][metric] for c in CS], 'o-', color=color,
                        label='Sigmoid calibrated' if calibrated else 'Raw base model')
            ax.axhline(values[REFERENCE][metric], color='#319c66', linestyle='--', label='Calibrated C search')
            ax.axhline(baseline, color='gray', linestyle=':', label='Constant p=0.5')
            ax.set(xscale='log', xlabel='C (smaller = stronger regularization)', ylabel=label)
            ax.set_title(title if col == 0 else label)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Does smaller C only pull confidence toward 50%? Held-out two-class scores', fontsize=15)
    fig.supxlabel('Equal cue weights, then equal seeds/tasks within sessions and equal sessions. 25 sessions; three seeds; common five-fold outer holdouts.\nTiny-C calibrated collapse can reflect optimizer scaling; see the separate training-only numerical diagnostic.', fontsize=10)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output, dpi=160)
    plt.close(fig)


def summarize_diagnostics(results):
    output = {}
    fields = ('coefficient_l2', 'test_margin_sd', 'test_margin_rms', 'calibration_a', 'calibration_b')
    for scope in sorted({row['scope'] for row in results}):
        output[scope] = {}
        for c in CS:
            folds = [fold for row in results if row['scope'] == scope for fold in row['fold_diagnostics'][str(c)]]
            output[scope][str(c)] = {
                'n_outer_fits': len(folds),
                'zero_calibration_slope_count': sum(f['calibration_a'] == 0 for f in folds),
                'warnings': sorted({w for f in folds for w in f['warnings']}),
                'descriptive_min_median_max': {key: np.quantile([f[key] for f in folds], [0, .5, 1]).tolist() for key in fields}}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--max-wall-seconds', type=int, default=3300)
    parser.add_argument('--work-dir', type=Path, default=Path('cache/regularization_path_20260930'))
    parser.add_argument('--output', type=Path, default=Path('docs/validation/regularization-path-validation.json'))
    parser.add_argument('--figure', type=Path, default=Path('docs/assets/regularization-path-validation.png'))
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or not 1 <= args.max_wall_seconds <= 3500:
        parser.error('Require 1–8 workers and a wall limit of 1–3500 seconds.')
    def deadline(signum, frame):
        raise TimeoutError('Regularization-path validation exceeded its wall-time budget.')
    signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, args.max_wall_seconds)
    start, root = time.monotonic(), Path(__file__).resolve().parents[2]
    tasks, sessions, exclusions, sources = build_panel(root)
    old_path = root / 'docs/validation/decoder-choice-validation.json'
    old = json.loads(old_path.read_text())
    archive_path = root / old['prediction_archive']
    if file_hash(archive_path) != old['prediction_archive_sha256']:
        raise ValueError('Original held-out prediction archive changed.')
    for path, digest in old['design']['sources_sha256'].items():
        if file_hash(root / path) != digest:
            raise ValueError(f'Original validation source changed: {path}')
    mapping_path = root / 'docs/validation/session-animal-mapping.json'
    mapping = json.loads(mapping_path.read_text())['sessions']
    for path in (old_path, archive_path, mapping_path, Path(__file__), root / 'scripts/next/validate_decoder_choices.py',
                 root / 'scripts/next/decoder_models.py', root / 'scripts/next/common.py'):
        sources[str(path.relative_to(root))] = file_hash(path)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    design = {'date': '2026-09-30', 'C_values': list(CS), 'shrinkage_alphas': list(ALPHAS),
              'workers': args.workers, 'max_wall_seconds': args.max_wall_seconds, 'outer_seeds': list(OUTER_SEEDS),
              'outer_folds': 5, 'sessions': sessions, 'exclusions': exclusions, 'sources_sha256': sources,
              'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'sklearn': sklearn.__version__},
              'raw_fit': 'Use the uncalibrated final all-outer-training refit inside ensemble=False production sigmoid calibration; calibration does not alter its fitted coefficients.',
              'shrinkage_control': 'Fixed p_alpha=.5+alpha*(p_search-.5), no label-dependent tuning; positive alpha preserves discrimination and decisions except numerical ties.',
              'brier_identity': 'Brier=.25+E[(p-.5)^2]-2E[(p-.5)(y-.5)]. These are algebraic spread/alignment terms, not a reliability/resolution decomposition.',
              'limitations': ['Exploratory follow-up on an existing cohort. No C or alpha selected using these outer scores has unbiased selected-model performance.',
                              'Cached screening/cue selection uses full-session data; all-cell/fixed-cue sensitivity removes it but retains the original cohort.',
                              'Session-bootstrap intervals are nominal and do not account for recordings clustered within only three animals; comparisons are not multiplicity-adjusted.',
                              'Tiny input margins can cause sigmoid optimizer stopping artifacts; exact-arithmetic calibration may undo pure margin shrinkage.',
                              'No state durations, permutation nulls, or M1 outcomes are used to score or choose C.'],
              'task_count': len(tasks)}
    (args.work_dir / 'design.json').write_text(json.dumps(design, indent=2) + '\n')
    with np.load(archive_path, allow_pickle=False) as archive:
        validate_alignment(tasks, old, archive)
        references = [archive[f'task_{i}_{REFERENCE}'].copy() for i in range(len(tasks))]
        checks = {(i, c): archive[f'task_{i}_weighted_fixed{suffix}_calibrated'].copy()
                  for i in range(len(tasks)) for c, suffix in ((1., '1'), (.01, '001'))}
    print(f'Frozen design: {len(tasks)} tasks, {len(CS)} C values, raw/calibrated, {args.workers} workers.', flush=True)
    with parallel_config(backend='loky', n_jobs=args.workers, inner_max_num_threads=1):
        results = Parallel(verbose=10)(delayed(evaluate_task)(task, ref) for task, ref in zip(tasks, references))
    predictions = {}
    for index, row in enumerate(results):
        for c in (1., .01):
            if not np.array_equal(row['methods'][method_name(c)]['predictions'], checks[index, c]):
                raise ValueError('Anchor fits do not exactly reproduce the original outer predictions.')
        predictions[f'task_{index}_labels'] = tasks[index]['labels']
        for method, values in row['methods'].items():
            predictions[f'task_{index}_{method}'] = values.pop('predictions')
    path = args.work_dir / 'predictions.npz'
    np.savez_compressed(path, **predictions)
    for name, digest in sources.items():
        if file_hash(root / name) != digest:
            raise ValueError(f'Source changed during fitting: {name}')
    summary = aggregate(results, mapping)
    task_path = args.work_dir / 'task-results.json'
    task_path.write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')
    output = {'design': design, 'elapsed_seconds': time.monotonic() - start,
              'anchor_predictions_exact_match': True, 'prediction_archive': str(path),
              'prediction_archive_sha256': file_hash(path),
              'task_results_archive': str(task_path), 'task_results_archive_sha256': file_hash(task_path),
              'summary': summary, 'fold_diagnostic_summary': summarize_diagnostics(results)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    plot_summary(summary, args.figure)
    signal.setitimer(signal.ITIMER_REAL, 0)
    print(f'Wrote {args.output}; {output["elapsed_seconds"]:.1f}s, sources unchanged.', flush=True)


if __name__ == '__main__':
    main()
