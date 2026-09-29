"""Diagnose sigmoid calibration conditioning at very small classifier C.

One metadata-selected session/bin and one prespecified outer holdout are used.
Training-only normalization of OOF margins is a numerical diagnostic of the
same weighted Platt objective, not a proposed or selected production model.
"""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = 'scripts.next'

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import platform
import time
from unittest.mock import patch
import warnings

import numpy as np
import scipy
from scipy.special import expit
import sklearn
import sklearn.calibration as calibration
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

from scripts.next import cache_io
from scripts.next.cache_paths import primary_cache
from scripts.next.common import compute_binned_rates, load_session
from scripts.next.decoder_models import (
    DecoderModel, SVMKernel, create_base_decoder, fit_calibrated_decoder,
    make_logistic_calibration_cv_splits,
)


CS = (1., .1, .01, .001, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12)
OUTER_SEED = 20260930


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def normalize_training_margins(margins, weights):
    margins, weights = np.asarray(margins, dtype=float), np.asarray(weights, dtype=float)
    if margins.ndim != 1 or weights.shape != margins.shape or not np.all(np.isfinite(margins)) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError('Expected finite training margins and positive sample weights.')
    center = float(np.average(margins, weights=weights))
    scale = float(np.sqrt(np.average((margins - center)**2, weights=weights)))
    if not np.isfinite(scale) or scale <= 0 or np.ptp(margins) == 0:
        raise ValueError('A constant OOF margin has no scale to normalize.')
    return (margins - center) / scale, center, scale


def sigmoid_objective(margins, labels, weights, a, b):
    """The same balanced-weight Platt soft targets used by sklearn 1.8."""
    labels, weights = np.asarray(labels), np.asarray(weights, dtype=float)
    n0, n1 = weights[labels == 0].sum(), weights[labels == 1].sum()
    targets = np.where(labels == 1, (n1 + 1) / (n1 + 2), 1 / (n0 + 2))
    logits = -(a * np.asarray(margins, dtype=float) + b)
    residual = weights * (expit(logits) - targets)
    return {'loss': float(np.sum(weights * (np.logaddexp(0, logits) - targets * logits))),
            'gradient_a': float(-residual @ margins), 'gradient_b': float(-residual.sum())}


def capture_optimization(callback):
    """Record the actual optimizer termination used inside sklearn's sigmoid."""
    original = calibration.minimize
    reports = []

    def tracked(*args, **kwargs):
        result = original(*args, **kwargs)
        reports.append({'success': bool(result.success), 'status': int(result.status),
                        'message': str(result.message), 'iterations': int(result.nit),
                        'function_evaluations': int(result.nfev), 'objective': float(result.fun),
                        'gradient': np.asarray(result.jac, dtype=float).tolist(),
                        'options': kwargs.get('options', {})})
        return result

    with patch.object(calibration, 'minimize', tracked):
        value = callback()
    if len(reports) != 1:
        raise ValueError('Expected exactly one pooled sigmoid optimizer call.')
    return value, reports[0]


def proper_scores(labels, probabilities):
    weights = compute_sample_weight('balanced', labels)
    weights /= weights.sum()
    probabilities = np.asarray(probabilities, dtype=float)
    safe = np.clip(probabilities, np.finfo(float).eps, 1 - np.finfo(float).eps)
    return {'brier': float(np.sum(weights * (probabilities - labels)**2)),
            'log_loss': float(-np.sum(weights * (labels * np.log(safe) + (1 - labels) * np.log1p(-safe)))),
            'balanced_accuracy': float(np.sum(weights * ((probabilities >= .5) == labels))),
            'probability_sd': float(np.std(probabilities)),
            'probability_min': float(probabilities.min()), 'probability_max': float(probabilities.max())}


def diagnose_one_c(X, y, train, test, splits, seed, c):
    weights = compute_sample_weight('balanced', y[train])
    def base():
        return create_base_decoder(c, DecoderModel.LOGISTIC_REGRESSION, SVMKernel.LINEAR,
                                   seed, class_weight='balanced')
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        model, production_optimization = capture_optimization(
            lambda: fit_calibrated_decoder(base(), X[train], y[train], 'sigmoid', splits, balanced_class_weights=True))
        # Reconstruct each calibration margin using only its own inner training
        # fold, independently of CalibratedClassifierCV's OOF implementation.
        oof = np.full(len(train), np.nan)
        for inner_train, inner_test in splits:
            estimator = base().fit(X[train[inner_train]], y[train[inner_train]])
            oof[inner_test] = estimator.decision_function(X[train[inner_test]])
        direct, direct_optimization = capture_optimization(
            lambda: calibration._SigmoidCalibration().fit(oof, y[train], sample_weight=weights))
        normalized, center, scale = normalize_training_margins(oof, weights)
        normalized_calibrator, normalized_optimization = capture_optimization(
            lambda: calibration._SigmoidCalibration().fit(normalized, y[train], sample_weight=weights))
    fitted = model.calibrated_classifiers_[0]
    calibrator = fitted.calibrators[0]
    if not np.allclose([calibrator.a_, calibrator.b_], [direct.a_, direct.b_], rtol=1e-8, atol=1e-10):
        raise ValueError('Independent OOF reconstruction does not reproduce production calibration.')
    estimator = fitted.estimator
    classifier = estimator.named_steps['classifier']
    test_margin = estimator.decision_function(X[test])
    production_p = model.predict_proba(X[test])[:, 1]
    normalized_p = normalized_calibrator.predict((test_margin - center) / scale)
    raw_p = estimator.predict_proba(X[test])[:, 1]
    coef_norm = float(np.linalg.norm(classifier.coef_))
    original_objective = sigmoid_objective(oof, y[train], weights, calibrator.a_, calibrator.b_)
    normalized_objective = sigmoid_objective(normalized, y[train], weights, normalized_calibrator.a_, normalized_calibrator.b_)
    return ({'c': c, 'base_coefficient_l2_norm': coef_norm,
            'base_intercept': classifier.intercept_.tolist(), 'base_iterations': classifier.n_iter_.tolist(),
            'training_oof_margin_sd': float(oof.std()), 'training_oof_margin_max_abs': float(np.abs(oof).max()),
            'heldout_margin_sd': float(test_margin.std()),
            'production_sigmoid_a': float(calibrator.a_), 'production_sigmoid_b': float(calibrator.b_),
            'production_effective_coefficient_norm': float(abs(calibrator.a_) * coef_norm),
            'training_normalization_center': center, 'training_normalization_scale': scale,
            'normalized_sigmoid_a': float(normalized_calibrator.a_), 'normalized_sigmoid_b': float(normalized_calibrator.b_),
            'normalized_effective_coefficient_norm': float(abs(normalized_calibrator.a_) / scale * coef_norm),
            'production_optimizer': production_optimization, 'independent_replay_optimizer': direct_optimization,
            'normalized_optimizer': normalized_optimization,
            'production_training_platt_objective': original_objective,
            'normalized_training_platt_objective': normalized_objective,
            'normalized_minus_production_training_objective': normalized_objective['loss'] - original_objective['loss'],
            'heldout_raw': proper_scores(y[test], raw_p), 'heldout_production_calibrated': proper_scores(y[test], production_p),
            'heldout_normalized_calibrated': proper_scores(y[test], normalized_p),
            'heldout_production_vs_normalized_max_abs_probability_difference': float(np.max(np.abs(production_p - normalized_p))),
            'warnings': [str(item.message) for item in captured]},
            {'training_oof_margin': oof, 'heldout_margin': test_margin,
             'raw_probability': raw_p, 'production_probability': production_p, 'normalized_probability': normalized_p})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('docs/validation/calibration-scale-validation.json'))
    parser.add_argument('--work-dir', type=Path, default=Path('cache/calibration_scale_validation_20260930'))
    args = parser.parse_args()
    started = time.monotonic()
    source = primary_cache(Path('cache/next_run_005'), 'decoding_confidence.pkl')
    rows = cache_io.read(source)
    row = max(sorted(rows, key=lambda r: r['session']), key=lambda r: len(r['cell_idx']))
    path = Path(f'data/nature/{row["session"]}.mat')
    sources = {str(p): digest(p) for p in (source, path, Path(__file__), Path('scripts/next/decoder_models.py'), Path('scripts/next/common.py'))}
    sklearn_source = Path(inspect.getsourcefile(calibration))
    spikes, times, cues, correct = load_session(path)
    cue = int(row['cue'])
    opposite = (cue + 3) % 8 + 1
    trials = np.flatnonzero(correct & np.isin(cues, [cue, opposite]))
    y = (cues[trials] == cue).astype(np.int8)
    X = compute_binned_rates(spikes[np.ix_(trials, np.arange(times.size), row['cell_idx'])], times, [950], 50)[:, 0].astype(float)
    del spikes
    train, test = next(StratifiedKFold(5, shuffle=True, random_state=OUTER_SEED).split(X, y))
    seed = int(np.random.SeedSequence([OUTER_SEED, 0]).generate_state(1)[0])
    splits, _ = make_logistic_calibration_cv_splits(y[train], np.arange(len(train)), 5, seed)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    design = {'date': '2026-09-30', 'session': str(row['session']), 'selection_rule': 'Largest cached stationary decoder population; ties broken by session ID.',
              'bin_start_ms': 950, 'window_ms': 50, 'n_cells': int(X.shape[1]), 'preferred_cue': cue,
              'outer_seed': OUTER_SEED, 'outer_fold': 0, 'outer_folds': 5, 'inner_seed': seed, 'calibration_folds': 5,
              'training_trial_ids': trials[train].tolist(), 'heldout_trial_ids': trials[test].tolist(),
              'training_class_counts': np.bincount(y[train], minlength=2).tolist(), 'heldout_class_counts': np.bincount(y[test], minlength=2).tolist(),
              'c_values': list(CS), 'numerical_only_extended_c_values': [1e-10, 1e-12],
              'normalization': 'Weighted mean/SD of outer-training OOF margins only; apply the same transform to final-base held-out margins.',
              'calibration': 'Same unpenalized sigmoid family, equal-class sample weights, sklearn Platt soft targets; only margin coordinates change.',
              'workers': 1, 'sources_sha256': sources,
              'sklearn_calibration_source': str(sklearn_source), 'sklearn_calibration_source_sha256': digest(sklearn_source),
              'software': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__},
              'limitations': ['One session, one bin, one outer fold: numerical diagnosis, not a performance-ranking study.',
                              'Cached full-session screening/cue selection remains outside the outer holdout.',
                              'Normalization is a diagnostic of the same calibration objective, not a production change or selected estimator.',
                              'At exactly zero or constant margins the normalization is undefined and no nonconstant calibration can recover discrimination.']}
    (args.work_dir / 'design.json').write_text(json.dumps(design, indent=2) + '\n')
    records, arrays = [], {}
    with threadpool_limits(limits=1):
        for c in CS:
            record, values = diagnose_one_c(X, y, train, test, splits, seed, c)
            records.append(record)
            arrays.update({f'C_{c:g}_{name}': value for name, value in values.items()})
    archive = args.work_dir / 'margins-and-probabilities.npz'
    np.savez_compressed(archive, **arrays)
    for name, before in sources.items():
        if digest(name) != before:
            raise ValueError(f'Source changed: {name}')
    output = {'design': design, 'elapsed_seconds': time.monotonic() - started, 'results': records,
              'array_archive': str(archive), 'array_archive_sha256': digest(archive)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(f'Wrote {args.output} in {output["elapsed_seconds"]:.1f}s.', flush=True)


if __name__ == '__main__':
    main()
