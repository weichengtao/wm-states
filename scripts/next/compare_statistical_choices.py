"""Read completed caches; reproduce evidence for the statistical-choices guide.

Run from the repository root with its existing Python environment:
    .venv/bin/python scripts/next/compare_statistical_choices.py

No fitting, worker pool, environment synchronization, or dashboard access.
Only the evidence JSON and documentation figure are written.
This is a reproduction utility for the dated next_run_001 through 006 comparison,
not a pipeline stage or a general-purpose run selector.
"""
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "scripts.next"

import os
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[variable] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/wm_statistical_choices_mpl')
import gc
from collections import Counter
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_PATH = ROOT / 'docs/validation/statistical-choices-evidence.json'
FIGURE_PATH = ROOT / 'docs/assets/statistical-choices-comparison.png'
RUNS = [f'next_run_{number:03d}' for number in range(1, 7)]
ANIMAL_MAP_PATH = ROOT / 'docs/validation/session-animal-mapping.json'
BOOTSTRAP_SEED = 20260930
BOOTSTRAP_REPEATS = 10000
PROBABILITY_CHANGE_TOLERANCE = 1e-6
HIST_EDGES = np.linspace(0, 1, 51)
NON_SCIENTIFIC_SETTINGS = {
    'cache_dir', 'plot_only', 'resume', 'save_figures', 'plot_actual_trial_id',
    'n_jobs', 'par_verbose', 'max_sessions_to_run', 'session_list_file',
}
CONTRASTS = [
    ('next_run_001', 'next_run_002', {'logistic_calibration_method'}, 'calibration with C search'),
    ('next_run_001', 'next_run_003', {'grid_search_for_c'}, 'C search with calibration'),
    ('next_run_001', 'next_run_004', {'grid_search_for_c', 'logistic_calibration_method'}, 'both choices disabled'),
    ('next_run_001', 'next_run_005', {'training_balance'}, 'all-trial class weighting'),
    ('next_run_003', 'next_run_004', {'logistic_calibration_method'}, 'calibration at fixed C=1'),
    ('next_run_002', 'next_run_004', {'grid_search_for_c'}, 'C search without calibration'),
    ('next_run_005', 'next_run_006', {'grid_search_for_c', 'classifier_c'}, 'weighted calibrated fixed C=0.01 versus C search'),
]


def scientific_settings(config):
    """Compare recorded choices, translating the old boolean without new defaults."""
    settings = {key: value for key, value in config.items() if key not in NON_SCIENTIFIC_SETTINGS}
    if 'balance_decoder_training_trials' in settings:
        previous = settings.pop('balance_decoder_training_trials')
        assert type(previous) is bool, 'Invalid historical balancing boolean'
        mode = 'balanced_training_trials' if previous else 'none'
        assert settings.get('training_balance', mode) == mode, 'Conflicting balancing settings'
        settings['training_balance'] = mode
    assert 'training_balance' in settings, 'Training balance was not recorded'
    return settings


def config_differences(first, second):
    return {key: [first.get(key), second.get(key)] for key in sorted(set(first) | set(second))
            if first.get(key) != second.get(key)}


def completed_stage(manifest, stage):
    return next((item for item in manifest['stages']
                 if item['stage'] == stage and item['status'] == 'complete'), None)


def run_manifests(folder, cached_config):
    """Identify fitting time and the latest completed state invocation separately."""
    manifests = [(path, json.loads(path.read_text())) for path in sorted((folder / 'manifests').glob('*.json'))]
    fitting = [(path, item) for path, item in manifests
               if completed_stage(item, 'decode') is not None
               and not item['settings']['decode']['plot_only']]
    assert len(fitting) == 1, f'{folder.name}: choose the actual fitting invocation explicitly'
    fit_path, fit = fitting[0]
    assert scientific_settings(fit['settings']['decode']) == scientific_settings(cached_config)
    state_runs = [(path, item) for path, item in manifests if completed_stage(item, 'states') is not None]
    assert state_runs, f'{folder.name}: no completed state invocation'
    state_path, state = state_runs[-1]
    state_config = {key: value for key, value in state['settings']['states'].items()
                    if key not in {'cache_dir', 'on_duration_xmax', 'off_duration_xmax',
                                   'compare_with_cc_skipped_on', 'compare_with_cc_skipped_off'}}
    return fit_path, fit, state_path, state, state_config

def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            result.update(chunk)
    return result.hexdigest()

def read(path):
    with path.open('rb') as stream:
        value = pickle.load(stream)
    return value['results']

def describe(values):
    a = np.asarray(values, dtype=float)
    return {'n': a.size, 'mean': float(a.mean()), 'median': float(np.median(a)),
            'p90': float(np.percentile(a, 90)), 'max': float(a.max()),
            'fraction_ge240': float((a >= 240).mean())}

def metrics(p):
    # All cached test labels are preferred cue = 1. Convert to float64 before
    # clipping so float32(1-eps) cannot round back to one.
    a = np.asarray(p, dtype=float)
    return {'n': a.size, 'mean_probability': float(a.mean()),
            'brier': float(np.square(1-a).mean()),
            'log_loss': float(-np.log(np.clip(a, np.finfo(float).eps, 1)).mean()),
            'preferred_cue_accuracy': float((a >= .5).mean()),
            'fraction_p_lt_0_1_or_gt_0_9': float(((a < .1) | (a > .9)).mean()),
            'mean_abs_probability_minus_half': float(np.abs(a - .5).mean())}

def encode_c(a):
    a = np.asarray(a)
    encoded = np.full(a.shape, 255, dtype=np.uint8)
    for i, c in enumerate((1, .1, .01)):
        encoded[a == c] = i
    assert np.all(encoded != 255)
    return encoded

def counts_c(encoded):
    counts = np.bincount(encoded.ravel(), minlength=3)
    return {str(c): {'count': int(n), 'fraction': float(n / counts.sum())}
            for c, n in zip((1., .1, .01), counts)}

def animal_mapping(sessions, path=ANIMAL_MAP_PATH):
    """Use the user's explicit mapping only, with exact cohort coverage."""
    mapping = json.loads(path.read_text())
    identities = mapping['sessions']
    assert set(sessions) == set(identities), 'Animal mapping must cover exactly the analyzed sessions'
    assert dict(Counter(identities.values())) == mapping['expected_session_counts'], 'Animal counts differ'
    assert set(identities.values()) == {'A', 'H', 'J'}, 'Expected three user-confirmed monkeys'
    return identities


def paired_session_summary(values, *, repeats=BOOTSTRAP_REPEATS, seed=BOOTSTRAP_SEED):
    """Descriptive session resampling; this is not a population-of-monkeys CI."""
    values = np.asarray(values, dtype=float)
    assert values.ndim == 1 and len(values) and np.isfinite(values).all()
    assert repeats > 0
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(repeats, len(values)))].mean(axis=1)
    return {'n_sessions': len(values), 'mean': float(values.mean()), 'median': float(np.median(values)),
            'session_bootstrap_95_interval': np.quantile(means, [.025, .975]).tolist(),
            'n_negative': int((values < 0).sum()), 'n_zero': int((values == 0).sum()),
            'n_positive': int((values > 0).sum())}


def session_contrast_inference(per_session, identities):
    """Aggregate each session once; show each monkey separately without a 3-cluster CI."""
    assert set(per_session) == set(identities)
    result = {'unit': 'one mean paired contrast per session, equally weighted',
              'bootstrap_seed': BOOTSTRAP_SEED, 'bootstrap_repeats': BOOTSTRAP_REPEATS,
              'interval_caveat': 'Descriptive paired session-resampling percentile intervals. Sessions within a monkey may be dependent; 25 sessions from only three monkeys do not establish population-of-monkeys uncertainty.',
              'metrics': {}}
    for metric in next(iter(per_session.values())):
        if metric == 'changed_max_off_trials':
            continue
        entries = {session: row[metric] for session, row in per_session.items()}
        groups = {}
        for animal in sorted(set(identities.values())):
            values = np.array([value for session, value in entries.items() if identities[session] == animal])
            groups[animal] = {'n_sessions': len(values), 'mean': float(values.mean()),
                              'median': float(np.median(values)), 'n_negative': int((values < 0).sum()),
                              'n_zero': int((values == 0).sum()), 'n_positive': int((values > 0).sum())}
        result['metrics'][metric] = {
            **paired_session_summary(list(entries.values())), 'per_monkey': groups,
            'monkey_equal_mean': float(np.mean([group['mean'] for group in groups.values()])),
        }
    return result


def verify_current_primary_row(run, row):
    """Read-only v2 verification, including primary/checkpoint payload equality."""
    from scripts.next.decoding_provenance import verify_decoding_fingerprint
    checkpoint_path = ROOT / 'cache' / run / 'decode/checkpoints' / f"{row['session']}.pkl"
    selection_path = ROOT / 'cache' / run / 'select/cell_screening.pkl'
    data_path = Path(row['config']['data_dir']) / f"{row['session']}.mat"
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in (checkpoint_path, data_path)}
    with checkpoint_path.open('rb') as stream:
        cached = pickle.load(stream)
    assert cached['fingerprint'] == row['fingerprint'] == cached['result']['fingerprint']
    assert cached['result'].keys() == row.keys()
    for key, value in row.items():
        other = cached['result'][key]
        assert (np.array_equal(value, other, equal_nan=True) if isinstance(value, np.ndarray) else value == other), (run, row['session'], key, 'primary/checkpoint mismatch')
    verification = verify_decoding_fingerprint(row['fingerprint'], row['config'], selection_path, data_path)
    assert verification.scheme == 'current', (run, row['session'], 'not current verified provenance')
    assert all(digest(ROOT / path) == value for path, value in hashes.items()), 'Checkpoint/data changed while reading'
    return {'scheme': verification.scheme, 'fingerprint': row['fingerprint'],
            'primary_checkpoint_payload_equal': True, 'source_hashes': hashes}


def matching_c_anchor(base, other, c=.01):
    """Require exact probabilities when selected C and all remaining procedures match."""
    encoded_c = (1., .1, .01).index(c)
    totals = {'observed_entries': 0, 'null_entries': 0}
    per_session = {}
    for session, a in base.items():
        b = other[session]
        row = {}
        for label, c_key, p_key in [('observed', 'C', 'p'), ('null', 'C_null', 'null')]:
            assert np.all(b[c_key] == encoded_c), 'Comparison run must use the stated fixed C everywhere'
            mask = a[c_key] == encoded_c
            first, second = a[p_key][mask], b[p_key][mask]
            count = int(mask.sum())
            assert np.array_equal(first, second), (session, label, 'matched-C probabilities differ')
            row[f'{label}_entries'] = count
            totals[f'{label}_entries'] += count
        per_session[session] = row
    return {'fixed_c': c, 'scope': 'all 161 time bins, observed and null fits separately',
            'exact_equality_verified': True, 'mismatches': 0, **totals, 'per_session': per_session}


def verify_cached_off_state(observed, null, state, state_config, times):
    """Recompute the recorded OFF policy before comparing cached state durations."""
    from scripts.next.validate_state_confidence import off_mask, duration_arrays
    assert state_config['cp_method_off'] == state_config['cc_method_off'] == 'one_tailed'
    assert state_config['cc_alpha_off'] == .05, 'Audit helper uses the recorded 95th percentile cutoff'
    mask, details = off_mask(observed, null, threshold=state_config['z_threshold_off'],
                             minimum_bins=state_config['cluster_size_threshold_off'], mass_filter=True)
    assert np.array_equal(mask, state['off_state_mask']), 'Cached OFF mask differs from reconstructed policy'
    durations = duration_arrays(mask, times)
    for computed, cached in [('maximum_off_state_duration_ms', 'max_off_state_duration_per_trial'),
                             ('total_off_state_duration_ms', 'off_state_duration_per_trial')]:
        assert np.array_equal(durations[computed], state[cached]), ('Cached OFF durations differ', computed)
    return {'exact_mask_and_durations_verified': True, 'scope': 'full 161-bin mask and cached inclusive-delay max/total durations',
            'null_mass_cutoff': details['mass_cutoff'], 'zero_variance_bins': details['zero_variance_bins']}


def run_summary(run):
    folder = ROOT / 'cache' / run
    decode_path = folder / 'decode/decoding_confidence.pkl'
    state_path = folder / 'states/on_off_states.pkl'
    source_paths = [decode_path, state_path, folder / 'select/cell_screening.pkl']
    before = {str(p.relative_to(ROOT)): digest(p) for p in source_paths}
    state_rows = read(state_path)
    states = {d['session']: d for d in state_rows}
    assert len(states) == len(state_rows), f'{run}: duplicate state sessions'
    decoded = read(decode_path)
    summaries = {}; compact = {}; config = decoded[0]['config']; provenance_rows = {}
    assert len({d['session'] for d in decoded}) == len(decoded), f'{run}: duplicate decoded sessions'
    assert {d['session'] for d in decoded} == set(states), f'{run}: decode/state sessions differ'
    fit_path, manifest, states_manifest_path, states_manifest, state_config = run_manifests(folder, config)
    manifest_hashes = {str(path.relative_to(ROOT)): digest(path) for path in sorted({fit_path, states_manifest_path})}
    for d in decoded:
        session = d['session']; s = states[session]
        if run in {'next_run_005', 'next_run_006'}:
            provenance_rows[session] = verify_current_primary_row(run, d)
        assert scientific_settings(d['config']) == scientific_settings(config), (run, session, 'config')
        assert s['decoding_fingerprint'] == d['fingerprint']
        assert np.array_equal(s['trial_idx'], d['trial_idx'])
        assert np.array_equal(s['time_bins'], d['time_bins'])
        assert np.all(d['decoding_test_labels'] == 1)
        assert s['z_threshold_on'] == state_config['z_threshold_on']
        assert s['z_threshold_off'] == state_config['z_threshold_off']
        times = d['time_bins']; delay = (times >= 500) & (times <= 1400)
        assert delay.sum() == 91 and len(times) == 161
        p = d['decoding_confidence']; null = d['decoding_confidence_null']
        assert null.shape == (*p.shape, 100)
        assert np.isfinite(p).all() and np.isfinite(null).all()
        assert ((p >= 0) & (p <= 1)).all() and ((null >= 0) & (null <= 1)).all()
        if run in {'next_run_005', 'next_run_006'}:
            provenance_rows[session]['state_reconstruction'] = verify_cached_off_state(p, null, s, state_config, times)
        # Null summaries are per-bin fitted distributions, not evaluation over
        # opposite-cue held-out labels (which these caches do not contain).
        mu = null.mean(-1, dtype=float); sd = null.std(-1, dtype=float)
        c_obs = encode_c(d['decoding_classifier_c'])
        c_null = encode_c(d['decoding_classifier_c_null'])
        assert d['decoding_predicted_labels'].shape == p.shape
        assert np.isin(d['decoding_predicted_labels'], [0, 1]).all()
        entry = {'native_preferred_cue_accuracy_delay': float((d['decoding_predicted_labels'][:, delay] == 1).mean()),
                 'trials': len(d['trial_idx']), 'cells': d['num_cells'], 'cue': int(d['cue']),
                 'observed_all_bins': metrics(p), 'observed_delay': metrics(p[:, delay]),
                 'null_delay': metrics(null[:, delay]),
                 'null_mean_delay': float(mu[:, delay].mean()),
                 'mean_null_sd_delay': float(sd[:, delay].mean()),
                 'fraction_on_probability_cutoff_above_one_delay': float((mu[:,delay]+state_config['z_threshold_on']*sd[:,delay]>1).mean()),
                 'null_sd_delay_quantiles': np.quantile(sd[:, delay], [.1, .5, .9]).tolist(),
                 'C_observed_delay': counts_c(c_obs[:, delay]),
                 'C_null_delay': counts_c(c_null[:, delay]),
                 'max_off': describe(s['max_off_state_duration_per_trial']),
                 'total_off': describe(s['off_state_duration_per_trial']),
                 'off_fraction_delay': float(s['off_state_mask'][:, delay].mean()),
                 'on_fraction_delay': float(s['on_state_mask'][:, delay].mean()),
                 'null_hist_delay': np.histogram(null[:, delay], bins=HIST_EDGES)[0].tolist()}
        compact[session] = {'p': p.copy(), 'mu': mu, 'sd': sd,
                            'C': c_obs, 'C_null': c_null,
                            'native_prediction': d['decoding_predicted_labels'].copy(),
                            'trial_idx': d['trial_idx'].copy(), 'cell_idx': d['cell_idx'].copy(),
                            'times': times.copy(), 'cue': int(d['cue']),
                            'max_off': s['max_off_state_duration_per_trial'].copy(),
                            'total_off': s['off_state_duration_per_trial'].copy(),
                            'off_mask': s['off_state_mask'].copy(), 'on_mask': s['on_state_mask'].copy()}
        if run in {'next_run_005', 'next_run_006'}:
            compact[session]['null'] = null.copy()
        summaries[session] = entry
    del decoded
    gc.collect()
    aggregate = {'sessions': len(summaries), 'trials': sum(s['trials'] for s in summaries.values())}
    for period in ['observed_all_bins', 'observed_delay', 'null_delay']:
        total = sum(s[period]['n'] for s in summaries.values())
        aggregate[period] = {'n': total}
        for metric in next(iter(summaries.values()))[period]:
            if metric != 'n':
                aggregate[period][metric] = sum(s[period][metric]*s[period]['n'] for s in summaries.values()) / total
        aggregate[period]['session_mean_brier'] = float(np.mean([s[period]['brier'] for s in summaries.values()]))
    for key in ['mean_null_sd_delay', 'null_mean_delay', 'off_fraction_delay', 'on_fraction_delay', 'fraction_on_probability_cutoff_above_one_delay', 'native_preferred_cue_accuracy_delay']:
        aggregate[key] = sum(s[key]*s['trials'] for s in summaries.values())/aggregate['trials']
    for key in ['max_off', 'total_off']:
        aggregate[key] = describe(np.concatenate([v[key] for v in compact.values()]))
    for key in ['C_observed_delay', 'C_null_delay']:
        counts = {c: sum(s[key][c]['count'] for s in summaries.values()) for c in ['1.0','0.1','0.01']}
        aggregate[key] = {c: {'count': n, 'fraction': n/sum(counts.values())} for c, n in counts.items()}
    aggregate['null_hist_delay'] = np.sum([s['null_hist_delay'] for s in summaries.values()], axis=0).tolist()
    timing = completed_stage(manifest, 'decode')
    fingerprints = {str(p.relative_to(ROOT)): digest(p) for p in source_paths}
    assert fingerprints == before, f'{run}: files changed while reading'
    assert {path: digest(ROOT / path) for path in manifest_hashes} == manifest_hashes
    return {'config': config, 'fitting_manifest_id': manifest['run_id'],
            'scientific_config': scientific_settings(config),
            'selection_scientific_config': {key: value for key, value in manifest['settings']['select'].items()
                                            if key not in NON_SCIENTIFIC_SETTINGS | {'n_jobs_session'}},
            'state_config': state_config, 'state_manifest_id': states_manifest['run_id'],
            'manifest_fingerprints': manifest_hashes,
            'current_primary_checkpoint_verification': provenance_rows,
            'decode_workers': config['n_jobs'],
            'decode_wall_seconds': timing['seconds'], 'fingerprints': fingerprints,
            'aggregate': aggregate, 'per_session': summaries}, compact

def compare(base, other, a_summary, b_summary):
    assert set(base) == set(other)
    per_session = {}; structural = True; c_equal = True; cn_equal = True
    max_deltas = []; p_deltas = []; all_off_diff = []; all_native_diff = []
    c1_prob_deltas = []
    probability_changed_native_same = []; off_changed_native_same = []
    native_transitions = np.zeros((2, 2), dtype=int)
    probability_transitions = np.zeros((2, 2), dtype=int)
    for session, a in base.items():
        b = other[session]
        for field in ['trial_idx','cell_idx','times']:
            assert np.array_equal(a[field], b[field]), (session, field)
        assert a['cue'] == b['cue']
        delay = (a['times'] >= 500) & (a['times'] <= 1400)
        c_equal &= np.array_equal(a['C'],b['C'])
        cn_equal &= np.array_equal(a['C_null'],b['C_null'])
        delta = b['max_off']-a['max_off']; max_deltas.extend(delta.tolist())
        pd = b['p'][:,delay].astype(float)-a['p'][:,delay].astype(float);p_deltas.extend(pd.ravel().tolist())
        off_diff = a['off_mask'][:,delay] != b['off_mask'][:,delay];all_off_diff.extend(off_diff.ravel().tolist())
        native_diff = a['native_prediction'][:,delay] != b['native_prediction'][:,delay];all_native_diff.extend(native_diff.ravel().tolist())
        for first, second, destination in [(a['native_prediction'][:,delay], b['native_prediction'][:,delay], native_transitions),
                                            (a['p'][:,delay] >= .5, b['p'][:,delay] >= .5, probability_transitions)]:
            destination += np.bincount((2 * first.astype(int) + second.astype(int)).ravel(), minlength=4).reshape(2, 2)
        same_native_probability_change = (~native_diff) & (np.abs(pd) > PROBABILITY_CHANGE_TOLERANCE)
        same_native_off_change = (~native_diff) & off_diff
        probability_changed_native_same.extend(same_native_probability_change.ravel().tolist())
        off_changed_native_same.extend(same_native_off_change.ravel().tolist())
        selected_c1 = a['C'] == 0
        if selected_c1.any():c1_prob_deltas.extend(np.abs(a['p'][selected_c1]-b['p'][selected_c1]).tolist())
        per_session[session] = {'other_minus_baseline_brier_delay': b_summary['per_session'][session]['observed_delay']['brier']-a_summary['per_session'][session]['observed_delay']['brier'],
                                'other_minus_baseline_log_loss_delay': b_summary['per_session'][session]['observed_delay']['log_loss']-a_summary['per_session'][session]['observed_delay']['log_loss'],
                                'other_minus_baseline_native_preferred_cue_accuracy_delay': float((b['native_prediction'][:,delay] == 1).mean() - (a['native_prediction'][:,delay] == 1).mean()),
                                'other_minus_baseline_probability_threshold_preferred_cue_accuracy_delay': float((b['p'][:,delay] >= .5).mean() - (a['p'][:,delay] >= .5).mean()),
                                'native_prediction_disagreement_delay': float(native_diff.mean()),
                                'probability_changed_native_prediction_unchanged_delay': float(same_native_probability_change.mean()),
                                'off_changed_native_prediction_unchanged_delay': float(same_native_off_change.mean()),
                                'other_minus_baseline_mean_abs_probability_minus_half_delay': float(np.abs(b['p'][:,delay].astype(float) - .5).mean() - np.abs(a['p'][:,delay].astype(float) - .5).mean()),
                                'mean_abs_probability_difference_delay': float(np.abs(pd).mean()),
                                'off_mask_disagreement_delay': float(off_diff.mean()),
                                'mean_max_off_difference_ms': float(delta.mean()),
                                'mean_total_off_difference_ms': float((b['total_off'] - a['total_off']).mean()),
                                'changed_max_off_trials': int(np.count_nonzero(delta))}
    return {'structural_alignment_verified': structural,
            'observed_C_identical_all_bins': bool(c_equal), 'null_C_identical_all_bins': bool(cn_equal),
            'native_prediction_disagreement_delay': float(np.mean(all_native_diff)),
            'probability_changed_native_prediction_unchanged_delay': float(np.mean(probability_changed_native_same)),
            'off_changed_native_prediction_unchanged_delay': float(np.mean(off_changed_native_same)),
            'probability_change_tolerance': PROBABILITY_CHANGE_TOLERANCE,
            'native_prediction_transition_counts_delay': native_transitions.tolist(),
            'probability_threshold_transition_counts_delay': probability_transitions.tolist(),
            'transition_matrix_definition': 'Rows baseline label, columns other label; order [0,1]. Probability label1 includes p=0.5.',
            'probability_mean_abs_difference_delay': float(np.abs(p_deltas).mean()),
            'off_mask_disagreement_delay': float(np.mean(all_off_diff)),
            'max_off_changed_trial_count': int(np.count_nonzero(max_deltas)),
            'max_off_other_longer_count': int(np.sum(np.array(max_deltas)>0)),
            'max_off_other_shorter_count': int(np.sum(np.array(max_deltas)<0)),
            'max_off_mean_abs_change_ms': float(np.abs(max_deltas).mean()),
            'sessions_other_has_lower_preferred_only_brier': sum(s['other_minus_baseline_brier_delay']<0 for s in per_session.values()),
            'sessions_other_has_lower_preferred_only_log_loss': sum(s['other_minus_baseline_log_loss_delay']<0 for s in per_session.values()),
            'other_minus_baseline_brier_delay': b_summary['aggregate']['observed_delay']['brier']-a_summary['aggregate']['observed_delay']['brier'],
            'other_minus_baseline_log_loss_delay': b_summary['aggregate']['observed_delay']['log_loss']-a_summary['aggregate']['observed_delay']['log_loss'],
            'max_abs_probability_difference_where_baseline_C_is_1': float(max(c1_prob_deltas,default=0)),
            'per_session': per_session}


def compare_contrast(baseline, other, expected, label, runs, datasets, identities=None):
    a, b = runs[baseline], runs[other]
    difference = config_differences(a['scientific_config'], b['scientific_config'])
    assert set(difference) == expected, (baseline, other, difference)
    assert a['state_config'] == b['state_config'], (baseline, other, 'state methods differ')
    assert a['selection_scientific_config'] == b['selection_scientific_config'], (baseline, other, 'selection settings differ')
    result = compare(datasets[baseline], datasets[other], a, b)
    result.update({
        'contrast': label,
        'scientific_config_differences': difference,
        'cached_config_differences': config_differences(
            {k: v for k, v in a['config'].items() if k not in NON_SCIENTIFIC_SETTINGS},
            {k: v for k, v in b['config'].items() if k not in NON_SCIENTIFIC_SETTINGS}),
        'state_config_identical': True,
        'selection_config_identical': True,
    })
    if identities is not None:
        result['paired_session_equal'] = session_contrast_inference(result['per_session'], identities)
    return result


def calibration_c_effects(runs):
    """Descriptive 2x2 contrasts; no independent-bin inferential test."""
    result = {}
    for metric in ('brier', 'log_loss'):
        a, b, c, d = [runs[run]['aggregate']['observed_delay'][metric] for run in RUNS[:4]]
        result[metric] = {
            'calibration_minus_none_with_C_search': a - b,
            'calibration_minus_none_with_fixed_C': c - d,
            'C_search_minus_fixed_with_calibration': a - c,
            'C_search_minus_fixed_without_calibration': b - d,
            'difference_of_calibration_effects_search_minus_fixed': (a - b) - (c - d),
        }
    return result


def prior_experiments(previous, recorded):
    """Refresh local reports, retaining recorded evidence when caches are absent."""
    result = {}
    for name, relative in [
        ('balanced_weights', 'balanced_weights/results.json'),
        ('activity_mechanism', 'activity_mechanism/results.json'),
        ('trial_swap', 'trial_influence/cross_null_checks.json'),
    ]:
        path = previous / relative
        if not path.exists():
            result[name] = recorded.get(name, {'available': False})
            if result[name].get('available'):
                print(f'{name}: local report missing; retained recorded evidence.', flush=True)
            else:
                print(f'{name}: no local report or recorded evidence available.', flush=True)
            continue
        data = json.loads(path.read_text())
        saved = {'available': True, 'path': str(path.relative_to(ROOT)), 'sha256': digest(path)}
        if name == 'balanced_weights':
            saved.update({
                'design': data['design'],
                'stability': data['stability'],
                'nested_validation': {
                    key: value for key, value in data['nested_validation'].items()
                    if key != 'paired_trial_brier'
                },
                'paired_trial_brier': {
                    key: value for key, value in data['nested_validation']['paired_trial_brier'].items()
                    if key not in ['trial_ids', 'per_trial_difference']
                },
            })
        elif name == 'activity_mechanism':
            saved.update({'design': data['design'], 'activity_geometry': data['activity_geometry']})
        else:
            saved['results'] = data
        result[name] = saved
    return result


def main():
    from scripts.next.decoding_provenance import _runtime_versions
    from scripts.next.decoding_signature import scientific_source_digest
    generator_hash = digest(Path(__file__))
    scientific_hash = scientific_source_digest()
    state_source_hashes = {str(path.relative_to(ROOT)): digest(path) for path in [
        ROOT / 'scripts/next/validate_state_confidence.py', ROOT / 'scripts/next/on_off_states.py']}
    evidence = {'analysis_date':'2026-09-30','population':'preferred-cue test trials only; all labels = 1',
                'delay_bin_starts_ms':[500,1400], 'delay_bin_count':91,
                'aggregation':'Historical pooled metrics retain equal trial-bin weighting. Additional paired contrasts first average within session, then weight each session equally; descriptive session bootstrap, not independent-bin inference.',
                'limitations': ['Cached observed test trials contain only preferred-cue labels (=1): neither two-class discrimination nor calibration validity can be established from these scores alone.',
                                'All runs reuse full-session cell screening and preferred-cue selection; reported cache scores are conditional, not independent validation of the entire decoder procedure.',
                                'Only three user-confirmed monkeys; session-resampling intervals are descriptive and do not establish animal-population uncertainty.',
                                'All six runs use 100 independently shuffled labels per time bin. Longer/shorter OFF durations do not validate the temporal null or biological interpretation.',
                                'Historical delay summaries retain bin starts 500 through 1400 inclusive (91 bins); the final 50 ms window extends beyond the 1400 ms delay boundary.',
                                'C=.01 was motivated by earlier analyses of this cohort. Run006 is a controlled procedure comparison, not an untouched confirmatory data set.'],
                'prediction_definitions': {'native': 'cached estimator model.predict label; this includes calibration when enabled, not the uncalibrated base decision', 'probability_threshold': 'preferred-cue probability >=0.5, including ties', 'unchanged_native_probability_changed': f'fraction of all delay trial-bins with equal native labels and absolute probability change >{PROBABILITY_CHANGE_TOLERANCE}'},
                'generator': {'path': str(Path(__file__).resolve().relative_to(ROOT)), 'sha256': generator_hash},
                'histogram_edges':HIST_EDGES.tolist(),'runs':{},'comparisons':{}}
    datasets = {}
    for run in RUNS:
        summary, compact = run_summary(run);evidence['runs'][run]=summary
        datasets[run] = compact
        focal=compact['221024'];row=int(np.flatnonzero(focal['trial_idx']==136)[0])
        summary['session221024_trial136']={'max_off_ms':float(focal['max_off'][row]),'total_off_ms':float(focal['total_off'][row]),
                                         'critical_bins':{}}
        for target_time in [830,910]:
            i=int(np.flatnonzero(focal['times']==target_time)[0]);p=float(focal['p'][row,i]);mu=float(focal['mu'][row,i]);sd=float(focal['sd'][row,i])
            summary['session221024_trial136']['critical_bins'][str(target_time)]={'probability':p,'null_mean':mu,'null_sd':sd,'z':(p-mu)/sd,'off_probability_cutoff':mu+summary['state_config']['z_threshold_off']*sd,'C':[1,.1,.01][int(focal['C'][row,i])]}
        gc.collect()
        print(run, json.dumps({'brier':summary['aggregate']['observed_delay']['brier'],
                              'mean_max_off_ms':summary['aggregate']['max_off']['mean']}), flush=True)
    identities = animal_mapping(datasets[RUNS[0]])
    mapping_hash = digest(ANIMAL_MAP_PATH)
    evidence['animal_mapping'] = {'path': str(ANIMAL_MAP_PATH.relative_to(ROOT)), 'sha256': mapping_hash,
                                  'n_monkeys': len(set(identities.values())), 'session_counts': dict(Counter(identities.values()))}
    for baseline, other, expected, label in CONTRASTS:
        evidence['comparisons'][other+'_vs_'+baseline] = compare_contrast(
            baseline, other, expected, label, evidence['runs'], datasets, identities)
    evidence['comparisons']['next_run_006_vs_next_run_005']['matching_C_anchor'] = matching_c_anchor(
        datasets['next_run_005'], datasets['next_run_006'])
    evidence['comparisons']['next_run_006_vs_next_run_005']['verified_current_provenance'] = {
        'scientific_source_digest': scientific_hash, 'runtime_packages': _runtime_versions(),
        'current_primary_checkpoint_pairs_verified': sum(len(evidence['runs'][run]['current_primary_checkpoint_verification'])
                                                          for run in ['next_run_005', 'next_run_006']),
        'detail_location': 'runs.next_run_005/006.current_primary_checkpoint_verification',
        'current_off_masks_and_durations_reconstructed': True, 'state_reconstruction_source_hashes': state_source_hashes,
        'same_recorded_raw_inputs_verified': all(
            evidence['runs']['next_run_005']['current_primary_checkpoint_verification'][session]['source_hashes'][f'data/nature/{session}.mat'] ==
            evidence['runs']['next_run_006']['current_primary_checkpoint_verification'][session]['source_hashes'][f'data/nature/{session}.mat']
            for session in identities)}
    assert evidence['comparisons']['next_run_006_vs_next_run_005']['verified_current_provenance']['same_recorded_raw_inputs_verified']
    evidence['calibration_C_factorial_effects'] = calibration_c_effects(evidence['runs'])
    previous=ROOT/'cache/comparisons/run_037_vs_next_001_221024'
    recorded = json.loads(EVIDENCE_PATH.read_text()).get('prior_experiments', {}) if EVIDENCE_PATH.exists() else {}
    evidence['prior_experiments'] = prior_experiments(previous, recorded)
    # Preserve only aggregated evidence in tracked documentation; raw caches stay local.
    assert digest(ANIMAL_MAP_PATH) == mapping_hash, 'Animal mapping changed while reading'
    assert digest(Path(__file__)) == generator_hash, 'Comparison script changed while reading'
    assert scientific_source_digest() == scientific_hash, 'Scientific decoder source changed while reading'
    assert all(digest(ROOT / path) == value for path, value in state_source_hashes.items()), 'State reconstruction source changed while reading'
    EVIDENCE_PATH.write_text(json.dumps(evidence,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scripts.next.figure_exports import save_figure
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(4,2,figsize=(13,16),layout='constrained')
    colors=['#267b9a','#d67b43','#735aa8','#b49529','#248656','#b94c73']
    labels=['001: downsampled, calibrated, C search', '002: downsampled, uncalibrated, C search',
            '003: downsampled, calibrated, C = 1', '004: downsampled, uncalibrated, C = 1',
            '005: weighted, calibrated, C search', '006: weighted, calibrated, C = 0.01']
    for run,color,label in zip(RUNS,colors,labels):
        agg=evidence['runs'][run]['aggregate'];hist=np.asarray(agg['null_hist_delay'],dtype=float)
        axes[0,0].stairs(hist/hist.sum()/.02,HIST_EDGES,color=color,label=label,lw=1.8)
        v=np.sort(np.concatenate([d['max_off'] for d in datasets[run].values()]));axes[0,1].step(v,np.arange(1,len(v)+1)/len(v),where='post',color=color,label=label,lw=1.8)
    axes[0,0].set(xlabel='Shuffled-null preferred-cue probability',ylabel='Density',title='A  Calibration changes the null probability scale')
    axes[0,0].legend(fontsize=7.5)
    axes[0,1].set(xlabel='Maximum OFF duration per trial (ms)',ylabel='Fraction of trials at or below duration',title='B  State durations depend on statistical choices')
    sessions=list(evidence['runs'][RUNS[0]]['per_session'])
    def scores(run):
        return [evidence['runs'][run]['per_session'][s]['observed_delay']['brier'] for s in sessions]
    def scatter(ax, pairs, xlabel, ylabel, title, note):
        values = []
        for a, b, color, marker, label in pairs:
            x, y = scores(a), scores(b)
            values.extend(x+y)
            ax.scatter(x,y,c=color,marker=marker,s=28,alpha=.8,label=label)
        lim=[min(values)*.9,max(values)*1.06]
        ax.plot(lim,lim,'--',c='gray',lw=1)
        ax.set(xlim=lim,ylim=lim,xlabel=xlabel,ylabel=ylabel,title=title)
        ax.text(.04,.95,note,transform=ax.transAxes,ha='left',va='top',fontsize=8)
        ax.legend(loc='lower right',fontsize=8)
    scatter(axes[1,0], [(RUNS[0],RUNS[1],colors[1],'o','C search: 001 vs 002'),
                       (RUNS[2],RUNS[3],colors[3],'s','C = 1: 003 vs 004')],
            'Calibrated Brier','Uncalibrated Brier','C  Calibration at both C settings',
            'Above line favors calibration')
    scatter(axes[1,1], [(RUNS[0],RUNS[2],colors[2],'o','Calibrated: 001 vs 003'),
                       (RUNS[1],RUNS[3],colors[3],'s','Uncalibrated: 002 vs 004')],
            'C-search Brier','Fixed C = 1 Brier','D  C search with / without calibration',
            'Above line favors C search')
    scatter(axes[2,0], [(RUNS[0],RUNS[4],colors[4],'o','001 vs 005')],
            'Downsampled Brier','All-trial weighted Brier','E  Training balance across 25 sessions',
            'Below line favors class weighting')
    scatter(axes[2,1], [(RUNS[4],RUNS[5],colors[5],'o','005 vs 006')],
            'Weighted C-search Brier','Weighted fixed C = 0.01 Brier','F  Fixed C=.01 versus C search',
            'Below line favors fixed C=.01')
    hours=[evidence['runs'][run]['decode_wall_seconds']/3600 for run in RUNS]
    axes[3,1].barh([run[-3:] for run in RUNS],hours,color=colors)
    axes[3,1].invert_yaxis()
    for i,hours_i in enumerate(hours): axes[3,1].text(hours_i+.2,i,f'{hours_i:.2f} h',va='center',fontsize=9)
    axes[3,1].set(xlabel='Decoder-stage wall time (hours; 10 workers)',xlim=(0,max(hours)*1.2),
                  title='H  Actual fitting invocations')
    contrast=evidence['comparisons']['next_run_006_vs_next_run_005']['per_session']
    monkey_colors={'A':'#267b9a','H':'#248656','J':'#b94c73'}
    for i,animal in enumerate(sorted(monkey_colors)):
        values=[row['other_minus_baseline_brier_delay'] for session,row in contrast.items() if identities[session]==animal]
        offsets=np.linspace(-.16,.16,len(values))
        axes[3,0].scatter(i+offsets,values,color=monkey_colors[animal],s=27,alpha=.8)
        axes[3,0].plot([i-.23,i+.23],[np.mean(values)]*2,color=monkey_colors[animal],lw=3)
    axes[3,0].axhline(0,color='gray',ls='--',lw=1)
    axes[3,0].set(xticks=range(3),xticklabels=['A (10 sessions)','H (8)','J (7)'],
                  ylabel='Brier difference: fixed C=.01 minus search',title='G  Paired directions within each monkey')
    fig.suptitle('Completed-run evidence · 25 aligned sessions · delay-bin starts 500–1400 ms',fontsize=13)
    fig.supxlabel('Panels C–G: one point per session; bars in G are monkey means. Observed scores use preferred-cue test trials only.\nAll six runs use 100 independent-per-bin label shuffles. Runtime is not a controlled hardware/load benchmark.',fontsize=9)
    FIGURE_PATH.parent.mkdir(exist_ok=True)
    save_figure(fig,FIGURE_PATH,dpi=160);plt.close(fig)
    print('Wrote documentation evidence and figure; source caches unchanged.',flush=True)

if __name__=='__main__':main()
