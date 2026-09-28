"""Read completed caches; reproduce evidence for the statistical-choices guide.

Run from the repository root with its existing Python environment:
    .venv/bin/python scripts/next/compare_statistical_choices.py

No fitting, worker pool, environment synchronization, or dashboard access.
Only the evidence JSON and documentation figure are written.
This is a reproduction utility for the dated next_run_001/002/003 comparison,
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
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_PATH = ROOT / 'docs/validation/statistical-choices-evidence.json'
FIGURE_PATH = ROOT / 'docs/assets/statistical-choices-comparison.png'
RUNS = ['next_run_001', 'next_run_002', 'next_run_003']
HIST_EDGES = np.linspace(0, 1, 51)

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
            'fraction_p_lt_0_1_or_gt_0_9': float(((a < .1) | (a > .9)).mean())}

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

def run_summary(run):
    folder = ROOT / 'cache' / run
    decode_path = folder / 'decode/decoding_confidence.pkl'
    state_path = folder / 'states/on_off_states.pkl'
    before = {str(p.relative_to(ROOT)): digest(p) for p in [decode_path, state_path]}
    states = {d['session']: d for d in read(state_path)}
    decoded = read(decode_path)
    summaries = {}; compact = {}; config = decoded[0]['config']
    for d in decoded:
        session = d['session']; s = states[session]
        assert s['decoding_fingerprint'] == d['fingerprint']
        assert np.array_equal(s['trial_idx'], d['trial_idx'])
        assert np.array_equal(s['time_bins'], d['time_bins'])
        assert np.all(d['decoding_test_labels'] == 1)
        times = d['time_bins']; delay = (times >= 500) & (times <= 1400)
        assert delay.sum() == 91 and len(times) == 161
        p = d['decoding_confidence']; null = d['decoding_confidence_null']
        assert null.shape == (*p.shape, 100)
        assert np.isfinite(p).all() and np.isfinite(null).all()
        assert ((p >= 0) & (p <= 1)).all() and ((null >= 0) & (null <= 1)).all()
        # Null summaries are per-bin fitted distributions, not evaluation over
        # opposite-cue held-out labels (which these caches do not contain).
        mu = null.mean(-1, dtype=float); sd = null.std(-1, dtype=float)
        c_obs = encode_c(d['decoding_classifier_c'])
        c_null = encode_c(d['decoding_classifier_c_null'])
        entry = {'trials': len(d['trial_idx']), 'cells': d['num_cells'], 'cue': int(d['cue']),
                 'observed_all_bins': metrics(p), 'observed_delay': metrics(p[:, delay]),
                 'null_delay': metrics(null[:, delay]),
                 'null_mean_delay': float(mu[:, delay].mean()),
                 'mean_null_sd_delay': float(sd[:, delay].mean()),
                 'fraction_on_probability_cutoff_above_one_delay': float((mu[:,delay]+1.645*sd[:,delay]>1).mean()),
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
        summaries[session] = entry
    del decoded
    gc.collect()
    paths = sorted((folder / 'manifests').glob('*.json'))
    manifests = [json.loads(p.read_text()) for p in paths]
    fitting = [m for m in manifests if not m['settings']['decode']['plot_only']]
    assert len(fitting) == 1, f'{run}: choose the actual fitting invocation explicitly'
    manifest = fitting[0]
    aggregate = {'sessions': len(summaries), 'trials': sum(s['trials'] for s in summaries.values())}
    for period in ['observed_all_bins', 'observed_delay', 'null_delay']:
        total = sum(s[period]['n'] for s in summaries.values())
        aggregate[period] = {'n': total}
        for metric in next(iter(summaries.values()))[period]:
            if metric != 'n':
                aggregate[period][metric] = sum(s[period][metric]*s[period]['n'] for s in summaries.values()) / total
        aggregate[period]['session_mean_brier'] = float(np.mean([s[period]['brier'] for s in summaries.values()]))
    for key in ['mean_null_sd_delay', 'null_mean_delay', 'off_fraction_delay', 'on_fraction_delay', 'fraction_on_probability_cutoff_above_one_delay']:
        aggregate[key] = sum(s[key]*s['trials'] for s in summaries.values())/aggregate['trials']
    for key in ['max_off', 'total_off']:
        aggregate[key] = describe(np.concatenate([v[key] for v in compact.values()]))
    for key in ['C_observed_delay', 'C_null_delay']:
        counts = {c: sum(s[key][c]['count'] for s in summaries.values()) for c in ['1.0','0.1','0.01']}
        aggregate[key] = {c: {'count': n, 'fraction': n/sum(counts.values())} for c, n in counts.items()}
    aggregate['null_hist_delay'] = np.sum([s['null_hist_delay'] for s in summaries.values()], axis=0).tolist()
    timing = next(s for s in manifest['stages'] if s['stage'] == 'decode')
    fingerprints = {str(p.relative_to(ROOT)): digest(p) for p in [decode_path, state_path]}
    assert fingerprints == before, f'{run}: files changed while reading'
    return {'config': config, 'fitting_manifest_id': manifest['run_id'],
            'decode_wall_seconds': timing['seconds'], 'fingerprints': fingerprints,
            'aggregate': aggregate, 'per_session': summaries}, compact

def compare(base, other, a_summary, b_summary):
    assert set(base) == set(other)
    per_session = {}; structural = True; c_equal = True; cn_equal = True
    max_deltas = []; p_deltas = []; all_off_diff = []; all_native_diff = []
    c1_prob_deltas = []
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
        selected_c1 = a['C'] == 0
        if selected_c1.any():c1_prob_deltas.extend(np.abs(a['p'][selected_c1]-b['p'][selected_c1]).tolist())
        per_session[session] = {'other_minus_baseline_brier_delay': b_summary['per_session'][session]['observed_delay']['brier']-a_summary['per_session'][session]['observed_delay']['brier'],
                                'mean_abs_probability_difference_delay': float(np.abs(pd).mean()),
                                'off_mask_disagreement_delay': float(off_diff.mean()),
                                'mean_max_off_difference_ms': float(delta.mean()),
                                'changed_max_off_trials': int(np.count_nonzero(delta))}
    return {'structural_alignment_verified': structural,
            'observed_C_identical_all_bins': bool(c_equal), 'null_C_identical_all_bins': bool(cn_equal),
            'native_prediction_disagreement_delay': float(np.mean(all_native_diff)),
            'probability_mean_abs_difference_delay': float(np.abs(p_deltas).mean()),
            'off_mask_disagreement_delay': float(np.mean(all_off_diff)),
            'max_off_changed_trial_count': int(np.count_nonzero(max_deltas)),
            'max_off_other_longer_count': int(np.sum(np.array(max_deltas)>0)),
            'max_off_other_shorter_count': int(np.sum(np.array(max_deltas)<0)),
            'max_off_mean_abs_change_ms': float(np.abs(max_deltas).mean()),
            'sessions_other_has_lower_preferred_only_brier': sum(s['other_minus_baseline_brier_delay']<0 for s in per_session.values()),
            'max_abs_probability_difference_where_baseline_C_is_1': float(max(c1_prob_deltas,default=0)),
            'per_session': per_session}


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
    evidence = {'analysis_date':'2026-09-28','population':'preferred-cue test trials only; all labels = 1',
                'delay_bin_starts_ms':[500,1400], 'delay_bin_count':91,
                'aggregation':'each trial-bin equally weighted; descriptive paired comparisons, not independent-bin tests',
                'histogram_edges':HIST_EDGES.tolist(),'runs':{},'comparisons':{}}
    base = None; compact_runs = {}
    for run in RUNS:
        summary, compact = run_summary(run);evidence['runs'][run]=summary
        if base is None:base=compact
        else:
            evidence['comparisons'][run+'_vs_next_run_001'] = compare(base,compact,evidence['runs']['next_run_001'],summary)
            ignore={'cache_dir','plot_only'}
            expected={'logistic_calibration_method'} if run=='next_run_002' else {'grid_search_for_c'}
            bc=evidence['runs']['next_run_001']['config'];oc=summary['config']
            diff={k:[bc.get(k),oc.get(k)] for k in set(bc)|set(oc) if bc.get(k)!=oc.get(k) and k not in ignore}
            assert set(diff)==expected,diff
            evidence['comparisons'][run+'_vs_next_run_001']['cached_config_differences']=diff
        focal=compact['221024'];row=int(np.flatnonzero(focal['trial_idx']==136)[0])
        summary['session221024_trial136']={'max_off_ms':float(focal['max_off'][row]),'total_off_ms':float(focal['total_off'][row]),
                                         'critical_bins':{}}
        for target_time in [830,910]:
            i=int(np.flatnonzero(focal['times']==target_time)[0]);p=float(focal['p'][row,i]);mu=float(focal['mu'][row,i]);sd=float(focal['sd'][row,i])
            summary['session221024_trial136']['critical_bins'][str(target_time)]={'probability':p,'null_mean':mu,'null_sd':sd,'z':(p-mu)/sd,'off_probability_cutoff':mu+.842*sd,'C':[1,.1,.01][int(focal['C'][row,i])]}
        compact_runs[run]={'max_off':np.concatenate([d['max_off'] for d in compact.values()]),'focal':{k:v.copy() if isinstance(v,np.ndarray) else v for k,v in focal.items() if k not in ['C_null']}}
        if compact is not base:del compact
        gc.collect()
        print(run, json.dumps(summary['aggregate']), flush=True)
    previous=ROOT/'cache/comparisons/run_037_vs_next_001_221024'
    recorded = json.loads(EVIDENCE_PATH.read_text()).get('prior_experiments', {}) if EVIDENCE_PATH.exists() else {}
    evidence['prior_experiments'] = prior_experiments(previous, recorded)
    # Preserve only aggregated evidence in tracked documentation; raw caches stay local.
    EVIDENCE_PATH.write_text(json.dumps(evidence,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scripts.next.figure_exports import save_figure
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    colors=['#267b9a','#d67b43','#735aa8'];labels=['001: calibrated + C search','002: uncalibrated + C search','003: calibrated + C = 1']
    for run,color,label in zip(RUNS,colors,labels):
        agg=evidence['runs'][run]['aggregate'];hist=np.asarray(agg['null_hist_delay'],dtype=float)
        axes[0,0].stairs(hist/hist.sum()/.02,HIST_EDGES,color=color,label=label,lw=1.8)
        v=np.sort(compact_runs[run]['max_off']);axes[0,1].step(v,np.arange(1,len(v)+1)/len(v),where='post',color=color,label=label,lw=1.8)
    axes[0,0].set(xlabel='Shuffled-null preferred-cue probability',ylabel='Density',title='A  Calibration changes the null probability scale')
    axes[0,0].legend(fontsize=8)
    axes[0,1].set(xlabel='Maximum OFF duration per trial (ms)',ylabel='Fraction of trials at or below duration',title='B  State durations depend on statistical choices')
    sessions=list(evidence['runs'][RUNS[0]]['per_session'])
    for j,other in enumerate(RUNS[1:]):
        a=[evidence['runs'][RUNS[0]]['per_session'][s]['observed_delay']['brier'] for s in sessions]
        b=[evidence['runs'][other]['per_session'][s]['observed_delay']['brier'] for s in sessions]
        ax=axes[1,j];ax.scatter(a,b,c=colors[j+1],s=28,alpha=.8)
        lim=[min(a+b)*.9,max(a+b)*1.06];ax.plot(lim,lim,'--',c='gray',lw=1)
        ax.set(xlim=lim,ylim=lim,xlabel='001 calibrated + C search: Brier',ylabel=f'{other}: Brier',title=f'{"C" if j==0 else "D"}  Preferred-cue-only scores: 001 vs {other[-3:]}')
        ax.text(.04,.95,'Each point = one session\nBelow line favors the alternative',transform=ax.transAxes,ha='left',va='top',fontsize=9)
    fig.suptitle('Completed-run evidence · 25 aligned sessions · delay-bin starts 500–1400 ms',fontsize=13)
    fig.supxlabel('Observed scores contain preferred-cue test trials only; they do not establish two-class calibration.\nAll three runs downsample training classes and use 100 independent-per-bin label shuffles.',fontsize=9)
    FIGURE_PATH.parent.mkdir(exist_ok=True)
    save_figure(fig,FIGURE_PATH,dpi=160);plt.close(fig)
    print('Wrote documentation evidence and figure; source caches unchanged.',flush=True)

if __name__=='__main__':main()
