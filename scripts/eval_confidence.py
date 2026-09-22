"""Evaluate cached preferred-cue probabilities without refitting decoders."""

import csv
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

try:
    from scripts.decoding_confidence import save_pickle_atomic
except ModuleNotFoundError:
    from decoding_confidence import save_pickle_atomic


@dataclass
class Config:
    """Report and cache binary probability scores for every decoded session."""

    cache_dir: Path = Path('cache/run_001')


def _mean_and_count(values, axis=None):
    valid = np.isfinite(values)
    count = valid.sum(axis=axis)
    total = np.where(valid, values, 0.0).sum(axis=axis)
    mean = np.divide(
        total, count, out=np.full(np.shape(total), np.nan), where=count > 0
    )
    return mean, count


def score_probabilities(probabilities, labels, predicted_labels=None, *, context='Scores'):
    """Score (trial, bin, sample) probabilities of label 1.

    Samples are individual repeats or null shuffles. NaN denotes an unavailable
    prediction and is excluded with explicit counts. Log loss uses natural logs
    and clips probabilities to float64 epsilon; Brier uses unmodified values.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels)
    if probabilities.ndim != 3 or any(size == 0 for size in probabilities.shape):
        raise ValueError('Probabilities must have nonempty (trial, bin, sample) axes.')
    if labels.shape != (probabilities.shape[0],) or not np.all(np.isin(labels, [0, 1])):
        raise ValueError('Test labels must contain one binary label per trial.')
    if np.any(np.isinf(probabilities)) or np.any(
        (probabilities < 0) | (probabilities > 1)
    ):
        raise ValueError('Probabilities must be in [0, 1] or NaN.')

    targets = labels[:, None, None]
    if predicted_labels is None:
        predicted_labels = probabilities >= 0.5
    else:
        predicted_labels = np.asarray(predicted_labels)
        if predicted_labels.shape != probabilities.shape or not np.all(
            np.isin(predicted_labels, [-1, 0, 1]) | np.isnan(predicted_labels)
        ):
            raise ValueError('Predicted labels must match probability shape and be binary, -1, or NaN.')
        predicted_labels = np.where(predicted_labels == -1, np.nan, predicted_labels)
    missing_probabilities = int(np.isnan(probabilities).sum())
    missing_predictions = int(np.isnan(predicted_labels).sum())
    accuracy_only_missing = int((np.isfinite(probabilities) & np.isnan(predicted_labels)).sum())
    if missing_probabilities or missing_predictions:
        warnings.warn(
            f'{context}: {missing_probabilities} missing probabilities and '
            f'{missing_predictions} missing predicted labels (NaN or -1); '
            f'excluding missing probabilities from all metrics and '
            f'{accuracy_only_missing} additional predictions from accuracy. '
            'Aggregations with no valid entries are NaN.',
            RuntimeWarning, stacklevel=2,
        )
    epsilon = np.finfo(np.float64).eps
    clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
    losses = {
        'brier_score': (probabilities - targets) ** 2,
        'log_loss': -(targets * np.log(clipped) + (1 - targets) * np.log1p(-clipped)),
        'accuracy': np.where(
            np.isfinite(probabilities) & np.isfinite(predicted_labels),
            predicted_labels == targets, np.nan,
        ),
        'decoding_confidence': probabilities,
    }
    result = {}
    for name, values in losses.items():
        overall, count = _mean_and_count(values)
        result[name] = float(overall)
        result[f'{name}_n_valid'] = int(count)
        for suffix, axis in (('by_time_bin', (0, 2)), ('by_sample', (0, 1)),
                             ('by_time_bin_and_sample', 0)):
            result[f'{name}_{suffix}'], result[f'{name}_n_valid_{suffix}'] = _mean_and_count(values, axis=axis)
    valid = np.isfinite(probabilities)
    result['n_valid'] = int(valid.sum())
    result['n_valid_by_time_bin'] = valid.sum(axis=(0, 2))
    result['n_valid_by_sample'] = valid.sum(axis=(0, 1))
    result['n_valid_by_time_bin_and_sample'] = valid.sum(axis=0)
    return result


def evaluate_session(source):
    """Score observed repeat 0 and individual null shuffles only."""
    time_bins = np.asarray(source['time_bins'])
    labels = np.asarray(source['decoding_test_labels'])
    repeats = source.get('decoding_confidence_repeats')
    if repeats is None:
        raise ValueError('Missing decoding_confidence_repeats; repeat 0 cannot be '
                         'recovered safely from averaged confidence. Rerun decoding.')
    repeats = np.asarray(repeats)
    if repeats.ndim != 3 or repeats.shape[1] < 1:
        raise ValueError('Repeated confidence must have nonempty (trial, repeat, bin) axes.')
    observed = repeats[:, 0, :]
    if time_bins.shape != (observed.shape[1],):
        raise ValueError('Observed confidence must have shape (trial, time bin).')
    predictions = source.get('decoding_predicted_labels')
    if predictions is not None:
        predictions = np.asarray(predictions)
        if predictions.shape != repeats.shape:
            raise ValueError('Predicted labels must have the same shape as repeated confidence.')
        predictions = predictions[:, 0, :, None]

    result = {
        'session': source['session'],
        'cue': source.get('cue'),
        'trial_idx': source['trial_idx'],
        'time_bins': time_bins,
        'decoding_test_labels': labels,
        'num_trials': int(observed.shape[0]),
        'num_time_bins': int(observed.shape[1]),
        'log_loss_epsilon': np.finfo(np.float64).eps,
        'observed': score_probabilities(observed[:, :, None], labels, predictions,
                                       context=f'Session {source["session"]}, observed repeat 0'),
        'observed_accuracy_source': 'cached_predictions' if predictions is not None else 'probability_threshold_0.5',
        'null_accuracy_source': 'probability_threshold_0.5',
        'observed_repeat_idx': 0,
        'null': None,
        'num_repeats': 1,
        'num_null_shuffles': 0,
    }
    null = source.get('decoding_confidence_null')
    if null is not None:
        null = np.asarray(null)
        if null.ndim != 3 or null.shape[:2] != observed.shape:
            raise ValueError('Null confidence must have shape (trial, bin, shuffle).')
        if null.shape[2] > 0:
            result['null'] = score_probabilities(null, labels, context=f'Session {source["session"]}, null')
            result['num_null_shuffles'] = int(null.shape[2])
    return result


def summary_row(result):
    row = {key: result[key] for key in (
        'session', 'cue', 'num_trials', 'num_time_bins', 'num_repeats', 'num_null_shuffles'
    )}
    for kind in ('observed', 'null'):
        scores = result[kind]
        for metric in ('brier_score', 'log_loss', 'accuracy', 'decoding_confidence', 'n_valid', 'accuracy_n_valid'):
            row[f'{kind}_{metric}'] = scores[metric] if scores is not None else None
    return row


def main(config: Config):
    source_path = config.cache_dir / 'decoding_confidence.pkl'
    with source_path.open('rb') as stream:
        sources = pickle.load(stream)
    if not sources:
        raise ValueError(f'No decoded sessions in {source_path}.')
    results = []
    seen = set()
    for source in sources:
        session = source['session']
        if str(session) in seen:
            raise ValueError(f'Duplicate session in decoding cache: {session}')
        seen.add(str(session))
        try:
            result = evaluate_session(source)
        except (ValueError, KeyError) as exc:
            raise ValueError(f'Session {session}: {exc}') from exc
        results.append(result)
        descriptions = []
        for kind in ('observed', 'null'):
            scores = result[kind]
            descriptions.append(
                f'{kind}: unavailable' if scores is None else
                f'{kind}: Brier={scores["brier_score"]:.6f}, '
                f'logloss={scores["log_loss"]:.6f}, n={scores["n_valid"]}'
            )
        print(f'{session} | ' + ' | '.join(descriptions))

    output_path = config.cache_dir / 'eval_confidence.pkl'
    save_pickle_atomic(results, output_path)
    csv_path = config.cache_dir / 'eval_confidence.csv'
    rows = [summary_row(result) for result in results]
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Cached {len(results)} sessions to {output_path} and {csv_path}')
    return results


if __name__ == '__main__':
    main(tyro.cli(Config))
