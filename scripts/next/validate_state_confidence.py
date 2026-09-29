"""Audit cached confidence and OFF-state robustness without refitting decoders.

Run with the existing environment from the repository root:
    .venv/bin/python scripts/next/validate_state_confidence.py

Production caches are read-only. Null resampling measures Monte Carlo sensitivity
conditional on the cached fitting procedure, not biological uncertainty or the
validity of the null time structure. Settings are declared before reading results.
"""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = 'scripts.next'

import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/wm-state-sensitivity-mpl')

import hashlib
import json
import pickle
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.ndimage import label

from scripts.next.on_off_states import max_off_state_duration_per_trial, off_state_duration_per_trial

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'docs/validation/state-confidence-sensitivity.json'
DESIGN = {
    'date': '2026-09-30',
    'runs': ['next_run_001', 'next_run_005'],
    'numerical_workers': 1,
    'seed': 20300930,
    'null_bootstrap_replicates': 40,
    'null_split_pairs': 5,
    'negative_control_null_indices': list(range(20)),
    'session_bootstrap_replicates': 20000,
    'off_thresholds': [0.5, 0.842, 1.282],
    'baseline': {'threshold': .842, 'minimum_bins': 1, 'mass_percentile': 95., 'delay_end': 1400},
    'additional_variants': ['skip_OFF_mass_filter', 'minimum_5_bins', 'windows_contained_in_delay'],
    'interpretation': [
        'No settings chosen by M1 performance or state duration.',
        'Session bootstrap is descriptive conditional on sampled animals; no animal-level independence claim.',
        'Null bootstrap and split halves condition on observed fits and the existing independent-per-bin null policy.',
        'Negative controls hold one shuffled fit out from the other 99; independent-time null traces are not biological trials.',
        'M1 comparisons align the prepared trial set, which excludes the first preferred-cue trial per session.',
    ],
}
COUNTS = ['preferred_cell_count', 'selective_nonpreferred_cell_count', 'stationary_nonselective_cell_count']
CONNECTIVITY = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def clusters(z, valid, threshold, minimum_bins):
    """Label along time only; leading rows are separate trials/shuffles."""
    mask = valid & (z <= threshold)
    labeled, _ = label(mask, structure=CONNECTIVITY)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    ids = np.flatnonzero(sizes >= minimum_bins)
    masses = np.bincount(labeled.ravel(), weights=np.where(valid, z, 0).ravel())
    return labeled, ids, masses


def off_mask(observed, null, *, threshold=.842, minimum_bins=1, mass_filter=True):
    """Reproduce the production one-tailed OFF rule, vectorizing null rows.

    This intentionally retains the existing pooled signed-mass rule. It is not
    a new permutation test or a claim that OFF means an absent representation.
    """
    observed = np.asarray(observed, dtype=float)
    null = np.asarray(null, dtype=float)
    if observed.ndim != 2 or 0 in observed.shape or null.ndim != 3 or null.shape[:2] != observed.shape or null.shape[-1] < 2:
        raise ValueError('Expected observed (trial, bin) and >=2 nulls per trial/bin.')
    if not np.isfinite(observed).all() or not np.isfinite(null).all():
        raise ValueError('Confidence must be finite.')
    if np.any((observed < 0) | (observed > 1)) or np.any((null < 0) | (null > 1)):
        raise ValueError('Confidence must be in [0, 1].')
    if minimum_bins < 1 or not np.isfinite(threshold):
        raise ValueError('Invalid OFF candidate settings.')
    mean = null.mean(-1)
    sd = null.std(-1)
    valid = (sd > 0) & (np.ptp(null, axis=-1) > 0)
    safe_sd = np.where(valid, sd, 1.)
    z = (observed - mean) / safe_sd
    labeled, ids, masses = clusters(z, valid, threshold, minimum_bins)
    cutoff = None
    if ids.size and mass_filter:
        z_null = (null - mean[..., None]) / safe_sd[..., None]
        # Flatten (shuffle, trial) into disconnected rows. A cluster can never
        # cross a trial or a shuffle, exactly as in the production loop.
        rows = np.moveaxis(z_null, -1, 0).reshape(-1, observed.shape[1])
        valid_rows = np.broadcast_to(valid, (null.shape[-1], *valid.shape)).reshape(rows.shape)
        _, null_ids, null_masses = clusters(rows, valid_rows, threshold, minimum_bins)
        if not null_ids.size:
            raise ValueError('No usable null OFF clusters for the pooled mass cutoff.')
        cutoff = float(np.percentile(null_masses[null_ids], 95))
        ids = ids[masses[ids] <= cutoff]
    return np.isin(labeled, ids), {'mass_cutoff': cutoff, 'zero_variance_bins': int((~valid).sum())}


def duration_arrays(mask, times, delay_end=1400):
    step = float(times[1] - times[0])
    if not np.allclose(np.diff(times), step):
        raise ValueError('Non-uniform time grid.')
    return {
        'maximum_off_state_duration_ms': max_off_state_duration_per_trial(mask, times, step, 500, delay_end),
        'total_off_state_duration_ms': off_state_duration_per_trial(mask, times, step, 500, delay_end),
    }


def describe(values):
    values = np.asarray(values, dtype=float)
    return {'mean': float(values.mean()), 'median': float(np.median(values)),
            'p90': float(np.quantile(values, .9)), 'min': float(values.min()), 'max': float(values.max())}


def summarize_variant(mask, reference, times, prepared_rows, counts, *, delay_end=1400):
    values = duration_arrays(mask, times, delay_end)
    baseline = duration_arrays(reference, times)
    delay = (times >= 500) & (times <= 1400)
    return {
        'all_trial_count': len(mask), 'prepared_trial_count': len(prepared_rows),
        'off_mask_disagreement_delay': float((mask[:, delay] != reference[:, delay]).mean()),
        'prepared_session_means': {key: float(value[prepared_rows].mean()) for key, value in values.items()},
        'counts': counts,
        'all_trial_metrics': {key: {
            **describe(value),
            'mean_abs_change_ms': float(np.abs(value - baseline[key]).mean()),
            'changed_fraction': float((value != baseline[key]).mean()),
            'change_at_least_50ms_fraction': float((np.abs(value - baseline[key]) >= 50).mean()),
        } for key, value in values.items()},
    }


def paired_session_bootstrap(evidence, repeats=20000, seed=20300930):
    """Resample sessions jointly, keeping each session's correlated bins together."""
    rng = np.random.default_rng(seed)
    runs = evidence['runs']
    sessions = sorted(runs['next_run_001']['per_session'])
    indices = rng.integers(len(sessions), size=(repeats, len(sessions)))
    output = {}
    for pair, comparison in evidence['comparisons'].items():
        other, baseline = pair.split('_vs_')
        rows = {}
        for metric in ('brier', 'log_loss'):
            a = np.array([runs[baseline]['per_session'][s]['observed_delay'][metric] for s in sessions])
            b = np.array([runs[other]['per_session'][s]['observed_delay'][metric] for s in sessions])
            delta = b - a
            sampled = delta[indices].mean(1)
            deleted_means = (delta.sum() - delta) / (len(delta) - 1)
            rows[metric] = {
                'session_equal_mean_other_minus_baseline': float(delta.mean()),
                'session_bootstrap_percentile_95': np.quantile(sampled, [.025, .975]).tolist(),
                'other_better_session_count': int((delta < 0).sum()),
                'leave_one_session_out_mean_difference_range': [float(deleted_means.min()), float(deleted_means.max())],
            }
        output[pair] = rows
    return output


def attach_session_models(result):
    """Reuse the M1 audit's equal-session model for each declared state variant.

    These models are descriptive sensitivities with fixed counts and outcomes
    regenerated from cached estimates. No bootstrap is run over the Monte Carlo
    realizations and no new hypothesis test is inferred from their repetition.
    """
    from scripts.next.validate_m1_robustness import analyze_session_model
    for run in result['runs'].values():
        run['session_M1_by_variant'] = {}
        for name, sessions in run['variants'].items():
            frame = pd.DataFrame([
                {'session': session, **run['variants']['baseline'][session]['counts'],
                 **row['prepared_session_means']} for session, row in sessions.items()
            ])
            models = {}
            for outcome in ('maximum_off_state_duration_ms', 'total_off_state_duration_ms'):
                model = analyze_session_model(frame, outcome, n_bootstrap=0)
                models[outcome] = {
                    'coefficients_ms_per_cell': {key: model['terms'][key]['coefficient_ms_per_cell']
                                                for key in COUNTS[:2]},
                    'LOSO_r2_vs_training_session_mean': model['loso']['r2_vs_heldout_training_mean'],
                    'LOSO_rmse_ms': model['loso']['rmse_ms'],
                    'M0_LOSO_rmse_ms': model['loso']['m0_rmse_ms'],
                }
            run['session_M1_by_variant'][name] = models


def aggregate_variants(result):
    """Small trial-weighted summaries, with Monte Carlo ranges labelled as such."""
    for run in result['runs'].values():
        baseline = run['variants']['baseline']
        counts = {session: row['all_trial_count'] for session, row in baseline.items()}
        total = sum(counts.values())
        def average(rows, getter):
            return float(sum(counts[session] * getter(row) for session, row in rows.items()) / total)
        summaries = {}
        for name, sessions in run['variants'].items():
            summaries[name] = {
                'off_mask_disagreement_delay': average(sessions, lambda row: row['off_mask_disagreement_delay']),
                'outcomes': {outcome: {metric: average(sessions, lambda row: row['all_trial_metrics'][outcome][metric])
                                      for metric in ('mean', 'mean_abs_change_ms', 'changed_fraction', 'change_at_least_50ms_fraction')}
                             for outcome in ('maximum_off_state_duration_ms', 'total_off_state_duration_ms')},
            }
        run['aggregate_by_variant'] = summaries
        run['Monte_Carlo_ranges_not_biological_intervals'] = {}
        for prefix in ('null50', 'null100', 'heldout_null'):
            names = [name for name in summaries if name.startswith(prefix)]
            if not names:
                continue
            run['Monte_Carlo_ranges_not_biological_intervals'][prefix] = {
                'realizations': len(names),
                'outcomes': {outcome: {
                    'mean_abs_trial_change_ms_averaged_over_realizations': float(np.mean([summaries[name]['outcomes'][outcome]['mean_abs_change_ms'] for name in names])),
                    'changed_trial_fraction_averaged_over_realizations': float(np.mean([summaries[name]['outcomes'][outcome]['changed_fraction'] for name in names])),
                    'change_at_least_50ms_fraction_averaged_over_realizations': float(np.mean([summaries[name]['outcomes'][outcome]['change_at_least_50ms_fraction'] for name in names])),
                    'session_M1_LOSO_R2_range': [float(func([run['session_M1_by_variant'][name][outcome]['LOSO_r2_vs_training_session_mean'] for name in names])) for func in (min, max)],
                    'coefficient_ms_per_cell_range': {term: [float(func([run['session_M1_by_variant'][name][outcome]['coefficients_ms_per_cell'][term] for name in names])) for func in (min, max)] for term in COUNTS[:2]},
                } for outcome in ('maximum_off_state_duration_ms', 'total_off_state_duration_ms')},
            }


def main():
    started = time.monotonic()
    scratch = ROOT / 'cache/comparisons/statistical_robustness_20260930/state'
    scratch.mkdir(parents=True, exist_ok=True)
    # Persist the design before loading outcomes or generating sensitivity results.
    (scratch / 'design.json').write_text(json.dumps(DESIGN, indent=2) + '\n')
    comparison_path = ROOT / 'docs/validation/statistical-choices-evidence.json'
    source_hashes = {str(comparison_path.relative_to(ROOT)): sha256(comparison_path)}
    evidence = json.loads(comparison_path.read_text())
    result = {'design': DESIGN, 'source_sha256': source_hashes,
              'analysis_source_sha256': {str(path.relative_to(ROOT)): sha256(path) for path in (
                  Path(__file__).resolve(), ROOT / 'scripts/next/on_off_states.py',
                  ROOT / 'scripts/next/validate_m1_robustness.py')},
              'paired_session_probability_scores': paired_session_bootstrap(
                  evidence, DESIGN['session_bootstrap_replicates'], DESIGN['seed']),
              'runs': {}}
    for run in DESIGN['runs']:
        root = ROOT / 'cache' / run
        paths = [root / 'decode/decoding_confidence.pkl', root / 'states/on_off_states.pkl', root / 'prepare/trial_table.pkl']
        for path in paths:
            source_hashes[str(path.relative_to(ROOT))] = sha256(path)
        with paths[0].open('rb') as stream:
            decoded = pickle.load(stream)['results']
        with paths[1].open('rb') as stream:
            cached = {str(row['session']): row for row in pickle.load(stream)['results']}
        prepared = pd.read_pickle(paths[2])
        prepared['session'] = prepared['session'].astype(str)
        variants = {}
        for session_number, d in enumerate(decoded):
            session = str(d['session'])
            times = d['time_bins']
            p = d['decoding_confidence']
            null = d['decoding_confidence_null']
            reference = cached[session]['off_state_mask']
            if d['fingerprint'] != cached[session]['decoding_fingerprint']:
                raise ValueError('Decode/state provenance mismatch.')
            np.testing.assert_array_equal(d['trial_idx'], cached[session]['trial_idx'])
            np.testing.assert_array_equal(times, cached[session]['time_bins'])
            frame = prepared[prepared['session'] == session]
            mapping = {int(trial): i for i, trial in enumerate(d['trial_idx'])}
            rows = np.array([mapping[int(trial)] for trial in frame['trial_id']], dtype=int)
            if frame[COUNTS].nunique().max() != 1:
                raise ValueError('Expected session-constant M1 counts.')
            counts = {key: int(frame[key].iloc[0]) for key in COUNTS}

            def save(name, observed=p, null_values=null, delay_end=1400, **kwargs):
                mask, details = off_mask(observed, null_values, **kwargs)
                row = summarize_variant(mask, reference, times, rows, counts, delay_end=delay_end)
                row.update(details)
                if session == '221024':
                    focal = mapping[136]
                    row['trial136'] = {k: float(v[focal]) for k, v in duration_arrays(mask, times, delay_end).items()}
                variants.setdefault(name, {})[session] = row
                return mask

            baseline = save('baseline')
            np.testing.assert_array_equal(baseline, reference)
            mu = np.mean(null, axis=-1, dtype=float)
            sd = np.std(null, axis=-1, dtype=float)
            delay = (times >= 500) & (times <= 1400)
            negative = (p - mu) < -1.645 * sd
            variants['baseline'][session]['operational_negative_z'] = {
                'fraction_delay_bins_below_minus_1_645': float(negative[:, delay].mean()),
                'fraction_OFF_bins_below_minus_1_645': float((negative & reference)[:, delay].sum() / max(reference[:, delay].sum(), 1)),
            }
            durations = duration_arrays(baseline, times)
            for key, cached_key in [('maximum_off_state_duration_ms', 'max_off_state_duration_per_trial'),
                                    ('total_off_state_duration_ms', 'off_state_duration_per_trial')]:
                np.testing.assert_array_equal(durations[key], cached[session][cached_key])
                np.testing.assert_array_equal(durations[key][rows], frame[key].to_numpy())
            for threshold in DESIGN['off_thresholds']:
                if threshold != .842:
                    save(f'z_{threshold}', threshold=threshold)
            save('skip_OFF_mass_filter', mass_filter=False)
            save('minimum_5_bins', minimum_bins=5)
            save('windows_contained_in_delay', delay_end=1350)
            rng = np.random.default_rng(np.random.SeedSequence([DESIGN['seed'], session_number]))
            for repeat in range(DESIGN['null_split_pairs']):
                permutation = rng.permutation(null.shape[-1])
                for half, indices in enumerate(np.array_split(permutation, 2)):
                    save(f'null50_pair{repeat}_half{half}', null_values=null[..., indices])
            for repeat in range(DESIGN['null_bootstrap_replicates']):
                indices = rng.integers(null.shape[-1], size=null.shape[-1])
                save(f'null100_bootstrap{repeat:02d}', null_values=null[..., indices])
            if run == 'next_run_005':
                for heldout in DESIGN['negative_control_null_indices']:
                    indices = np.delete(np.arange(null.shape[-1]), heldout)
                    save(f'heldout_null{heldout:02d}', observed=null[..., heldout], null_values=null[..., indices])
            print(f'{run} {session}: production mask reproduced; {len(variants)} variants', flush=True)
        result['runs'][run] = {'production_masks_exactly_reproduced': True, 'variants': variants}
        # Keep resampling draws auditable without repeating static count columns
        # or entire trial distributions for every Monte Carlo realization.
        for name, sessions in variants.items():
            if name.startswith(('null', 'heldout_null')):
                for row in sessions.values():
                    row.pop('counts')
                    row.pop('all_trial_count')
                    row.pop('prepared_trial_count')
                    row.pop('zero_variance_bins')
                    for metrics in row['all_trial_metrics'].values():
                        for key in ('median', 'p90', 'min', 'max'):
                            metrics.pop(key)
        del decoded
    attach_session_models(result)
    aggregate_variants(result)
    for path, fingerprint in result['analysis_source_sha256'].items():
        if sha256(ROOT / path) != fingerprint:
            raise RuntimeError(f'Analysis source changed during computation: {path}')
    for path, fingerprint in source_hashes.items():
        if sha256(ROOT / path) != fingerprint:
            raise RuntimeError(f'Source changed during analysis: {path}')
    result['elapsed_seconds'] = time.monotonic() - started
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f'Wrote {OUTPUT.relative_to(ROOT)} in {result["elapsed_seconds"]:.1f}s; source files unchanged.', flush=True)


if __name__ == '__main__':
    main()
