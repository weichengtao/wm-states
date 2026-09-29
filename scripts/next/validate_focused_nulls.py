"""Prespecified three-trial sensitivity to temporal nulls and regularization.

Calls the production decoder directly and writes only supplementary artifacts.
OFF summaries are uncorrected candidates, not production whole-session states.
"""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = 'scripts.next'

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import signal
import time

import numpy as np
from joblib import Parallel, delayed, parallel_config
from threadpoolctl import threadpool_limits

from scripts.next import cache_io
from scripts.next.cache_paths import primary_cache
from scripts.next.common import compute_binned_rates, json_value, load_session
from scripts.next.decoding_confidence import Config, decode_one_trial
from scripts.next.decoder_models import TrainingBalance


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def choose_targets(rows):
    """Return the original focal trial plus two targets selected by metadata."""
    by_session = {str(r['session']): r for r in rows}
    first = min(by_session)
    largest = max(sorted(by_session), key=lambda s: len(by_session[s]['cell_idx']))
    return [('221024', 136), (first, int(by_session[first]['trial_idx'][len(by_session[first]['trial_idx']) // 2])),
            (largest, int(by_session[largest]['trial_idx'][len(by_session[largest]['trial_idx']) // 2]))]


def candidate_durations(mask, step=10):
    mask = np.atleast_2d(np.asarray(mask, dtype=bool))
    maxima = []
    lengths = []
    for row in mask:
        edges = np.diff(np.pad(row.astype(np.int8), (1, 1)))
        runs = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
        maxima.append(int(runs.max(initial=0)) * step)
        lengths.extend((runs * step).tolist())
    return np.asarray(maxima), mask.sum(axis=1) * step, np.asarray(lengths)


def null_summary(observed, null, starts, *, z_off=.842):
    """Production ddof=0 standardization, but no session cluster-mass rule."""
    observed, null, starts = np.asarray(observed, dtype=np.float64), np.asarray(null, dtype=np.float64), np.asarray(starts)
    if null.ndim != 2 or null.shape[0] != observed.size or starts.shape != observed.shape or null.shape[1] < 2:
        raise ValueError('Expected one observed vector and at least two null vectors.')
    if not np.all(np.isfinite(null)) or not np.all(np.isfinite(observed)):
        raise ValueError('Nonfinite probabilities are not supported.')
    mean, std = null.mean(axis=1), null.std(axis=1, ddof=0)
    # Identical non-binary probabilities can have tiny positive SD because of
    # mean-rounding; production states also reject exact zero-range nulls.
    valid = (std > 0) & (np.ptp(null, axis=1) > 0)
    z = np.divide(observed - mean, std, out=np.zeros_like(mean), where=valid)
    z_null = np.divide(null - mean[:, None], std[:, None], out=np.zeros_like(null), where=valid[:, None])
    delay = (starts >= 500) & (starts <= 1400)
    if not delay.any():
        raise ValueError('No delay bins present.')
    mask = valid & (z <= z_off)
    maximum, total, _ = candidate_durations(mask[delay])
    null_mask = valid[:, None] & (z_null <= z_off)
    null_max, null_total, null_lengths = candidate_durations(null_mask[delay].T)
    centered = null - mean[:, None]
    numerators = np.mean(centered[:-1] * centered[1:], axis=1)
    denominator = std[:-1] * std[1:]
    adjacent = np.divide(numerators, denominator, out=np.full_like(numerators, np.nan),
                         where=(denominator > 0) & valid[:-1] & valid[1:])
    adjacency_delay = delay[:-1] & delay[1:]
    values = adjacent[adjacency_delay & np.isfinite(adjacent)]
    def distribution(values):
        return {'mean': float(np.mean(values)), 'median': float(np.median(values)),
                'p05': float(np.percentile(values, 5)), 'p95': float(np.percentile(values, 95)), 'maximum': float(np.max(values))}
    return {'observed_candidate_max_off_ms': int(maximum[0]), 'observed_candidate_total_off_ms': int(total[0]),
            'null_mean_probability_delay': float(mean[delay].mean()), 'null_mean_sd_delay': float(std[delay].mean()),
            'valid_null_delay_bins': int(valid[delay].sum()), 'delay_bin_count': int(delay.sum()),
            'constant_null_bin_count_full_grid': int(np.count_nonzero(np.ptp(null, axis=1) == 0)),
            'null_adjacent_bin_correlation_delay': distribution(values) if values.size else None,
            'null_candidate_fraction_delay': float(null_mask[delay].mean()),
            'null_candidate_max_off_ms': distribution(null_max),
            'null_candidate_total_off_ms': distribution(null_total),
            'null_candidate_cluster_length_ms': distribution(null_lengths) if null_lengths.size else None,
            'null_cluster_count_delay': int(null_lengths.size),
            'observed_off_probability_cutoff_delay': distribution((mean + z_off * std)[delay]),
            'observed_probability_delay': distribution(observed[delay]),
            'observed_probability_at_830ms': float(observed[starts == 830][0]) if np.any(starts == 830) else None,
            'off_probability_cutoff_at_830ms': float((mean + z_off * std)[starts == 830][0]) if np.any(starts == 830) else None}


def fit_task(task, *, n_null, benchmark, output_dir):
    started = time.monotonic()
    config = Config(seed=42, training_balance=TrainingBalance.BALANCED_CLASS_WEIGHTS,
                    grid_search_for_c=task['c_policy'] == 'search', classifier_c=.01 if task['c_policy'] == 'fixed_001' else 1.0,
                    preserve_null_time_structure=task['shared_time'], n_decode_shuffle=n_null,
                    n_jobs=1, save_figures=False)
    if benchmark:
        bins = np.array([70, 115, 155])
        rates, starts = task['rates'][:, bins], task['starts'][bins]
    else:
        rates, starts = task['rates'], task['starts']
    with threadpool_limits(limits=1):
        result = decode_one_trial(task['test_index'], rates, task['labels'], starts, config)
    observed, labels, c, null, null_c, folds = result
    artifact = output_dir / f"{task['session']}_trial_{task['trial']}_{task['c_policy']}_{'shared' if task['shared_time'] else 'independent'}.npz"
    np.savez_compressed(artifact, observed=observed, predicted_labels=labels, selected_c=c, null=null, null_c=null_c,
                        bin_starts=starts, training_trial_ids=task['trial_ids'][np.arange(len(task['trial_ids'])) != task['test_index']],
                        configuration=json.dumps(asdict(config), default=json_value))
    summary = null_summary(observed, null, starts) if not benchmark else None
    cached = task['cached_observed'][[70, 115, 155]] if benchmark else task['cached_observed']
    elapsed = time.monotonic() - started
    print(f"{artifact.stem}: completed in {elapsed:.1f}s", flush=True)
    return {'session': task['session'], 'trial': task['trial'], 'c_policy': task['c_policy'],
            'shared_time': task['shared_time'], 'n_cells': rates.shape[2], 'training_class_counts': np.bincount(np.delete(task['labels'], task['test_index']), minlength=2).tolist(),
            'n_null': n_null, 'time_bins': starts.tolist(), 'artifact': str(artifact), 'artifact_sha256': file_hash(artifact),
            'elapsed_seconds': elapsed, 'calibration_folds': list(folds), 'summary': summary,
            'observed_matches_run005': bool(np.array_equal(observed, cached)) if task['c_policy'] == 'search' else None,
            'observed_max_abs_difference_from_run005': float(np.max(np.abs(observed - cached))) if task['c_policy'] == 'search' else None,
            'observed_c_counts': {str(value): int(np.sum(c == value)) for value in [1.0, .1, .01]},
            'null_c_counts': {str(value): int(np.sum(null_c == value)) for value in [1.0, .1, .01]}}


def compare_policies(results):
    output = []
    for session, trial, c_policy in sorted({(r['session'], r['trial'], r['c_policy']) for r in results}):
        independent = next(r for r in results if (r['session'], r['trial'], r['c_policy'], r['shared_time']) == (session, trial, c_policy, False))
        shared = next(r for r in results if (r['session'], r['trial'], r['c_policy'], r['shared_time']) == (session, trial, c_policy, True))
        with np.load(independent['artifact']) as a, np.load(shared['artifact']) as b:
            if not np.array_equal(a['observed'], b['observed']) or not np.array_equal(a['selected_c'], b['selected_c']):
                raise ValueError('Observed fits changed across null policies.')
            null_a, null_b = a['null'].astype(np.float64), b['null'].astype(np.float64)
            mean_a, mean_b = null_a.mean(axis=1), null_b.mean(axis=1)
            sd_a, sd_b = null_a.std(axis=1), null_b.std(axis=1)
            output.append({'session': session, 'trial': trial, 'c_policy': c_policy, 'observed_exact_match_between_null_policies': True,
                           'null_first_bin_exact_match': bool(np.array_equal(null_a[0], null_b[0])),
                           'pointwise_null_mean_mean_abs_difference': float(np.mean(np.abs(mean_a - mean_b))),
                           'pointwise_null_sd_mean_abs_difference': float(np.mean(np.abs(sd_a - sd_b))),
                           'candidate_max_off_shared_minus_independent_ms': shared['summary']['observed_candidate_max_off_ms'] - independent['summary']['observed_candidate_max_off_ms'],
                           'candidate_total_off_shared_minus_independent_ms': shared['summary']['observed_candidate_total_off_ms'] - independent['summary']['observed_candidate_total_off_ms']})
    return output


def load_verified_artifact(row, *, n_null, n_training):
    """Check the recorded archive digest, numerical axes and fitting settings."""
    path = Path(row['artifact'])
    if file_hash(path) != row['artifact_sha256']:
        raise ValueError(f'Archive digest mismatch: {path}')
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    expected_shapes = {'observed': (161,), 'predicted_labels': (161,), 'selected_c': (161,),
                       'null': (161, n_null), 'null_c': (161, n_null), 'bin_starts': (161,),
                       'training_trial_ids': (n_training,)}
    for name, shape in expected_shapes.items():
        if name not in arrays or arrays[name].shape != shape:
            raise ValueError(f'Unexpected {name} shape in {path}.')
    if not np.array_equal(arrays['bin_starts'], np.arange(-200, 1401, 10)):
        raise ValueError('Archive time grid differs from the prespecified full grid.')
    for name in ('observed', 'null'):
        values = arrays[name]
        if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
            raise ValueError(f'Invalid {name} probability in {path}.')
    if not np.all(np.isin(arrays['predicted_labels'], [0, 1])):
        raise ValueError('Predicted labels must be binary.')
    grid = [1.0, .1, .01] if row['c_policy'] == 'search' else [.01]
    if any(not np.all(np.isin(arrays[name], grid)) for name in ('selected_c', 'null_c')):
        raise ValueError('Archive classifier C values violate the fitting policy.')
    trials = arrays['training_trial_ids']
    if len(np.unique(trials)) != len(trials) or row['trial'] in trials:
        raise ValueError('Training-trial IDs contain duplicates or the held-out trial.')
    config = json.loads(arrays['configuration'].item())
    expected = {'seed': 42, 'training_balance': 'balanced_class_weights',
                'decoder_model': 'logistic_regression', 'logistic_calibration_method': 'sigmoid',
                'logistic_calibration_cv': 5, 'grid_search_for_c': row['c_policy'] == 'search',
                'classifier_c': 1.0 if row['c_policy'] == 'search' else .01,
                'preserve_null_time_structure': row['shared_time'], 'n_decode_shuffle': n_null}
    if any(config.get(name) != value for name, value in expected.items()):
        raise ValueError('Archive fitting configuration differs from the prespecified procedure.')
    return arrays


def compare_cached_null_prefix(null, cached_null):
    """Audit the common draws when a focused bank uses a different null count."""
    null, cached_null = np.asarray(null), np.asarray(cached_null)
    if null.ndim != 2 or cached_null.ndim != 2 or null.shape[0] != cached_null.shape[0] or min(null.shape[1], cached_null.shape[1]) < 1:
        raise ValueError('Null banks must have matching bin axes and at least one draw.')
    count = min(null.shape[1], cached_null.shape[1])
    return {'null_matches_run005': bool(np.array_equal(null[:, :count], cached_null[:, :count])),
            'null_reproduction_compared_count': count, 'null_reproduction_cached_count': cached_null.shape[1]}


def summarize_existing(args):
    """Recompute diagnostics from verified archives without fitting a model."""
    started = time.monotonic()
    output = json.loads(args.output.read_text())
    design = output['design']
    output_dir = args.work_dir / 'full'
    if design.get('benchmark') or design != json.loads((output_dir / 'design.json').read_text()):
        raise ValueError('Saved result and full-run pre-fit design disagree.')
    script_name = 'scripts/next/validate_focused_nulls.py'
    snapshot = output_dir / 'fitting-script.py'
    if file_hash(snapshot) != design['sources_sha256'][script_name]:
        raise ValueError('Archived fitting-script snapshot does not match its pre-fit hash.')
    for name, digest in design['sources_sha256'].items():
        if name != script_name and file_hash(name) != digest:
            raise ValueError(f'Fitting input or model source changed: {name}')
    identities = {(t['session'], t['trial']): t for t in design['targets']}
    required = {(session, trial, c, shared) for session, trial in identities
                for c in ('search', 'fixed_001') for shared in (False, True)}
    found = [(r['session'], r['trial'], r['c_policy'], r['shared_time']) for r in output['results']]
    if len(found) != len(set(found)) or set(found) != required:
        raise ValueError('Archives do not cover exactly the prespecified target/procedure panel.')
    production = cache_io.read(primary_cache(Path('cache/next_run_005'), 'decoding_confidence.pkl'))
    production = {r['session']: r for r in production}
    duration_changes = []
    for row in output['results']:
        target = identities[row['session'], row['trial']]
        if Path(row['artifact']).parent.resolve() != output_dir.resolve():
            raise ValueError('Archive is outside the specified full-run work directory.')
        arrays = load_verified_artifact(row, n_null=design['n_null'], n_training=target['eligible_trial_count'] - 1)
        original = row['summary']
        row['summary'] = null_summary(arrays['observed'], arrays['null'], arrays['bin_starts'])
        for name in ('observed_candidate_max_off_ms', 'observed_candidate_total_off_ms'):
            if row['summary'][name] != original[name]:
                duration_changes.append({'session': row['session'], 'trial': row['trial'], 'c_policy': row['c_policy'],
                                         'shared_time': row['shared_time'], 'metric': name,
                                         'previous': original[name], 'updated': row['summary'][name]})
        if row['c_policy'] == 'search':
            cached = production[row['session']]
            index = int(np.flatnonzero(cached['trial_idx'] == row['trial'])[0])
            if not np.array_equal(arrays['observed'], cached['decoding_confidence'][index]):
                raise ValueError('Default observed estimates fail exact production-cache reproduction.')
            row['observed_matches_run005'] = True
            if not row['shared_time']:
                row.update(compare_cached_null_prefix(arrays['null'], cached['decoding_confidence_null'][index]))
                if not row['null_matches_run005']:
                    raise ValueError('Default independent nulls fail exact production-cache reproduction.')
    output['paired_null_policy_comparisons'] = compare_policies(output['results'])
    output['postprocessing'] = {
        'date': '2026-09-30', 'recomputed_from_verified_archives_without_fitting': True,
        'standardization': 'float64, ddof=0; require positive SD and positive null range, matching production states.',
        'fitting_script_snapshot': str(snapshot), 'fitting_script_sha256': file_hash(snapshot),
        'postprocessing_sources_sha256': {script_name: file_hash(script_name),
                                         'scripts/next/on_off_states.py': file_hash('scripts/next/on_off_states.py')},
        'duration_changes_from_previous_summary': duration_changes,
        'elapsed_seconds': time.monotonic() - started,
    }
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(f'Re-summarized {len(output["results"])} verified archives; {len(duration_changes)} candidate duration changes.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--summarize-only', action='store_true', help='Validate saved design/archives and recompute summaries without any fitting.')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--n-null', type=int, default=100)
    parser.add_argument('--max-wall-seconds', type=int, default=3300)
    parser.add_argument('--output', type=Path, default=Path('docs/validation/focused-null-validation.json'))
    parser.add_argument('--work-dir', type=Path, default=Path('cache/focused_null_validation_20260930'))
    args = parser.parse_args()
    if args.summarize_only:
        if args.benchmark:
            parser.error('--summarize-only cannot be combined with --benchmark')
        summarize_existing(args)
        return
    if not 1 <= args.workers <= 8 or args.n_null < 2 or not 1 <= args.max_wall_seconds <= 3500:
        parser.error('Require 1–8 workers, >=2 nulls, and a wall limit of 1–3500 seconds.')
    def deadline(signum, frame):
        raise TimeoutError('Focused null validation exceeded its explicit wall-time budget.')
    signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, args.max_wall_seconds)
    started = time.monotonic()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    output_dir = args.work_dir / ('benchmark' if args.benchmark else 'full')
    output_dir.mkdir(parents=True, exist_ok=True)
    path = primary_cache(Path('cache/next_run_005'), 'decoding_confidence.pkl')
    sources = {str(path): file_hash(path)}
    rows = cache_io.read(path)
    identities = choose_targets(rows)
    if args.benchmark:
        identities = [max(identities, key=lambda identity: len(next(r for r in rows if r['session'] == identity[0])['cell_idx']))]
    tasks = []
    target_metadata = []
    for session, trial in identities:
        row = next(r for r in rows if r['session'] == session)
        data_path = Path(f'data/nature/{session}.mat')
        sources[str(data_path)] = file_hash(data_path)
        spikes, times, cues, correct = load_session(data_path)
        cue = int(row['cue'])
        opposite = (cue + 3) % 8 + 1
        trial_ids = np.flatnonzero(correct & np.isin(cues, [cue, opposite]))
        labels = (cues[trial_ids] == cue).astype(np.int8)
        test = int(np.flatnonzero(trial_ids == trial)[0])
        if labels[test] != 1:
            raise ValueError('The target must be a cached preferred-cue trial.')
        starts = np.asarray(row['time_bins'])
        if not np.array_equal(starts, np.arange(-200, 1401, 10)):
            raise ValueError('Expected the full 161-bin production time grid.')
        rates = compute_binned_rates(spikes[np.ix_(trial_ids, np.arange(times.size), row['cell_idx'])], times, starts, 50)
        del spikes
        cached = row['decoding_confidence'][np.flatnonzero(row['trial_idx'] == trial)[0]]
        target_metadata.append({'session': session, 'trial': trial, 'preferred_cue': cue, 'opposite_cue': opposite,
                                'n_cells': rates.shape[2], 'eligible_trial_count': len(trial_ids), 'heldout_binary_index': test})
        for c_policy in ('search', 'fixed_001'):
            for shared in (False, True):
                tasks.append({'session': session, 'trial': trial, 'c_policy': c_policy, 'shared_time': shared,
                              'rates': rates, 'starts': starts, 'labels': labels, 'trial_ids': trial_ids,
                              'test_index': test, 'cached_observed': cached})
    # Start all expensive searched-C jobs first so the focused panel does not
    # need a second long wave after the cheaper fixed-C jobs finish.
    tasks.sort(key=lambda task: (task['c_policy'] != 'search', -task['rates'].shape[2], task['session'], task['shared_time']))
    for name in ('scripts/next/validate_focused_nulls.py', 'scripts/next/decoding_confidence.py',
                 'scripts/next/decoder_models.py', 'scripts/next/common.py'):
        sources[name] = file_hash(name)
    (output_dir / 'fitting-script.py').write_bytes(Path(__file__).read_bytes())
    design = {'date': '2026-09-30', 'benchmark': args.benchmark, 'seed': 42, 'n_null': args.n_null,
              'workers': args.workers, 'max_wall_seconds': args.max_wall_seconds, 'targets': target_metadata,
              'target_rule': 'Original 221024 trial136; median cached preferred-trial ID by index in lexicographically first session; same median in session with most cached decoder cells.',
              'training': 'All eligible correct preferred/opposite trials except the held-out target; balanced class weights and sigmoid five-fold calibration.',
              'contrasts': 'C search [1,.1,.01] by balanced accuracy versus fixed C=.01; independently permuted training labels per bin versus one permutation reused through time.',
              'delay_rule': 'Bin starts500 through1400 inclusive, 91 bins at10ms stride, matching production convention; 50ms activity windows.',
              'off_rule': 'One-tailed z<=.842 using own target/bin null mean and ddof=0 SD; invalid zero-SD bins excluded. Candidate masks only, no whole-session pooled cluster-mass filtering.',
              'limitations': ['Three prespecified preferred-cue targets cannot assess two-class decoder accuracy or population-level performance.',
                              'Null candidate masks are not biological OFF states; no inactivity, equivalence, or family-wise error claim.',
                              'Per-bin permutation destroys the original temporal dependence of the label assignment; shared permutations retain that assignment while fitting each bin separately.',
                              'The two policies have the same pointwise permutation target. Finite-null mean/SD and candidate-mask differences are Monte Carlo variation; temporal dependence directly affects joint run-length/cluster reference distributions.',
                              'Both modes assume exchangeability of training labels across trials; neither addresses drift/block exchangeability.',
                              'Full-session observed and matched-null refits are required before comparing production state masks or downstream M1.'],
              'sources_sha256': sources}
    (output_dir / 'design.json').write_text(json.dumps(design, indent=2) + '\n')
    print(f'Wrote pre-fit design; {len(tasks)} tasks, {args.n_null} nulls, {args.workers} workers.', flush=True)
    with parallel_config(backend='loky', n_jobs=args.workers, inner_max_num_threads=1):
        results = Parallel(verbose=10)(delayed(fit_task)(task, n_null=args.n_null, benchmark=args.benchmark, output_dir=output_dir) for task in tasks)
    for name, digest in sources.items():
        if file_hash(name) != digest:
            raise ValueError(f'Source changed during analysis: {name}')
    comparisons = compare_policies(results) if not args.benchmark else []
    output = {'design': design, 'elapsed_seconds': time.monotonic() - started, 'results': results, 'paired_null_policy_comparisons': comparisons}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    signal.setitimer(signal.ITIMER_REAL, 0)
    print(f'Wrote {args.output}; {output["elapsed_seconds"]:.1f} seconds.', flush=True)


if __name__ == '__main__':
    main()
