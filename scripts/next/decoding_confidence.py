"""One observed leave-one-trial-out estimate and N training-label null estimates."""

# Use one module namespace for direct CLI and package execution.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "scripts.next"

from dataclasses import dataclass, asdict
import json
from pathlib import Path
import warnings

import numpy as np
import tyro
from joblib import Parallel, delayed
from sklearn.calibration import CalibratedClassifierCV
from threadpoolctl import threadpool_limits

from scripts.next import cache_io
from scripts.next.common import compute_binned_rates, fingerprint, json_value, load_session, session_files, worker_context
from scripts.next.decoder_models import (
    CellsUsedForDecoder, DecoderModel, SVMKernel, LogisticCalibrationMethod,
    create_base_decoder, decoder_cells_for_session, preferred_cue_from_cells,
    make_logistic_calibration_cv_splits, make_grouped_stratified_cv_splits, select_classifier_c,
    CLASSIFIER_C_GRID, CLASSIFIER_C_GRID_SEARCH_CV,
)
from scripts.next.decoding_plots import plot_session

save_pickle_atomic = cache_io.save


@dataclass
class Config:
    data_dir: Path = Path('data/nature')
    cache_dir: Path = Path('cache/next_run')
    session_list_file: Path | None = None
    max_sessions_to_run: int | None = None
    n_jobs: int = 1
    par_verbose: int = 0
    seed: int = 42
    cells_used_for_decoder: CellsUsedForDecoder = CellsUsedForDecoder.STATIONARY
    decoder_model: DecoderModel = DecoderModel.LOGISTIC_REGRESSION
    svm_kernel: SVMKernel = SVMKernel.LINEAR
    balance_decoder_training_trials: bool = True
    classifier_c: float = 1.0
    grid_search_for_c: bool = False
    logistic_calibration_method: LogisticCalibrationMethod = LogisticCalibrationMethod.SIGMOID
    logistic_calibration_cv: int = 5
    min_cell_per_group: int = 1
    min_trials_good_session: int = 320
    t_decode_start: int = -200
    t_decode_end: int = 1400
    t_decode_window: int = 50
    t_decode_step: int = 10
    n_decode_shuffle: int = 100
    resume: bool = True
    plot_only: bool = False
    save_figures: bool = True
    plot_actual_trial_id: bool = False

    def __post_init__(self):
        self.classifier_c = float(self.classifier_c)
        self.cells_used_for_decoder = CellsUsedForDecoder(self.cells_used_for_decoder)
        self.decoder_model = DecoderModel(self.decoder_model)
        self.svm_kernel = SVMKernel(self.svm_kernel)
        self.logistic_calibration_method = LogisticCalibrationMethod(self.logistic_calibration_method)
        if self.n_decode_shuffle < 0 or self.seed < 0:
            raise ValueError('Shuffle count and seed must be nonnegative.')
        if not np.isfinite(self.classifier_c) or self.classifier_c <= 0:
            raise ValueError('classifier_c must be finite and positive.')
        if self.t_decode_window <= 0 or self.t_decode_step <= 0 or self.t_decode_end < self.t_decode_start:
            raise ValueError('Decoding window/step must be positive with start <= end.')
        if self.logistic_calibration_method is not LogisticCalibrationMethod.NONE and self.logistic_calibration_cv < 2:
            raise ValueError('Calibration requires at least two folds.')
        if self.n_jobs == 0 or self.min_cell_per_group < 1 or self.min_trials_good_session < 1:
            raise ValueError('Invalid worker count or session screening threshold.')


def training_trials(labels, test_idx, seed, balance):
    """Select the outer training set once, before any null permutation."""
    indices = np.delete(np.arange(labels.size), test_idx)
    by_class = [indices[labels[indices] == value] for value in (1, 0)]
    if min(map(len, by_class)) < 1:
        raise ValueError(f'Trial {test_idx}: both classes must remain after holding out the test trial.')
    if balance:
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(test_idx), 0]))
        size = min(map(len, by_class))
        indices = np.concatenate([rng.choice(group, size, replace=False) for group in by_class])
    return indices


def decode_one_trial(test_idx, binned_rates, labels, bin_starts, config):
    """Hold out all activity from a trial in fitting, C search and calibration.

    Each fit uses one sample per training trial from the test bin only.
    Null labels are independently permuted for each bin after selecting the
    outer training trials.
    """
    labels = np.asarray(labels)
    rates = np.asarray(binned_rates)
    if rates.ndim != 3 or labels.shape != (rates.shape[0],) or not np.all(np.isin(labels, [0, 1])):
        raise ValueError('Expected rates (trial, bin, cell) and binary labels (trial,).')
    if not 0 <= test_idx < labels.size or not np.all(np.isfinite(rates)):
        raise ValueError('Invalid test trial or nonfinite activity.')
    bins = np.asarray(bin_starts)
    if bins.shape != (rates.shape[1],):
        raise ValueError('time_bins must match the activity bin axis.')
    train = training_trials(labels, test_idx, config.seed, config.balance_decoder_training_trials)
    train_rates = rates[train].astype(np.float64)
    test_rates = rates[test_idx].astype(np.float64)
    train_labels = labels[train]
    count = rates.shape[1]
    observed = np.empty(count, dtype=np.float32)
    predictions = np.empty(count, dtype=np.int8)
    observed_c = np.empty(count, dtype=np.float64)
    null = np.empty((count, config.n_decode_shuffle), dtype=np.float32)
    null_c = np.empty(null.shape, dtype=np.float64)
    groups = np.arange(train.size)
    effective_folds = set()
    for estimate in range(config.n_decode_shuffle + 1):
        calibration_splits = {}
        search_splits = {}
        for b in range(count):
            X, y = train_rates[:, b, :], train_labels
            if estimate > 0:
                rng = np.random.default_rng(np.random.SeedSequence([config.seed, int(test_idx), 1, estimate, b]))
                y = rng.permutation(y)
            split_key = (X.shape[0], b if estimate > 0 else None)
            selected_c = config.classifier_c
            if config.grid_search_for_c:
                if split_key not in search_splits:
                    try:
                        search_splits[split_key], _ = make_grouped_stratified_cv_splits(
                            y, groups, CLASSIFIER_C_GRID_SEARCH_CV, config.seed,
                            purpose='Classifier C grid search', allow_fold_reduction=False)
                    except ValueError as exc:
                        raise ValueError(f'Trial {test_idx}, bin {b}, estimate {estimate}: {exc}') from exc
                selected_c = select_classifier_c(X, y, groups, config.decoder_model,
                    config.svm_kernel, config.seed,
                    fit_context=f'Trial {test_idx}, bin {b}, estimate {estimate}',
                    cv_splits=search_splits[split_key])
            model = create_base_decoder(selected_c, config.decoder_model, config.svm_kernel, config.seed)
            if config.decoder_model is DecoderModel.LOGISTIC_REGRESSION and config.logistic_calibration_method is not LogisticCalibrationMethod.NONE:
                if split_key not in calibration_splits:
                    calibration_splits[split_key] = make_logistic_calibration_cv_splits(
                        y, groups, config.logistic_calibration_cv, config.seed)
                splits, folds = calibration_splits[split_key]
                effective_folds.add(folds)
                model = CalibratedClassifierCV(estimator=model,
                    method=config.logistic_calibration_method.value, cv=splits, ensemble=False, n_jobs=1)
            model.fit(X, y)
            test_sample = test_rates[b:b + 1]
            probability = model.predict_proba(test_sample)[0, np.flatnonzero(model.classes_ == 1)[0]]
            if estimate == 0:
                observed[b] = probability
                predictions[b] = model.predict(test_sample)[0]
                observed_c[b] = selected_c
            else:
                null[b, estimate - 1] = probability
                null_c[b, estimate - 1] = selected_c
    return observed, predictions, observed_c, null, null_c, tuple(sorted(effective_folds))


def decode_session(path, selection, config):
    spikes, times, cues, correct = load_session(path)
    if selection['num_trials'] != spikes.shape[0]:
        raise ValueError(f'{path.stem}: selection and dataset trial counts differ.')
    cue = preferred_cue_from_cells(np.asarray(selection['cell_properties']['mean_pref_test']))
    if cue is None:
        raise ValueError(f'{path.stem}: no preferred cue.')
    opposite = (int(cue) + 3) % 8 + 1
    cells = np.array(sorted(decoder_cells_for_session(selection, config.cells_used_for_decoder, cue, opposite, spikes.shape[2])), dtype=int)
    if not cells.size:
        raise ValueError(f'{path.stem}: no cells for {config.cells_used_for_decoder.value}.')
    trials = np.flatnonzero(correct & np.isin(cues, [cue, opposite]))
    labels = (cues[trials] == cue).astype(np.int8)
    tests = np.flatnonzero(labels == 1)
    if not tests.size:
        raise ValueError(f'{path.stem}: no preferred-cue test trials.')
    starts = np.arange(config.t_decode_start, config.t_decode_end + 1, config.t_decode_step)
    # Bin the selected cells once; never send raw spike tensors to fit workers.
    rates = compute_binned_rates(spikes[np.ix_(trials, np.arange(times.size), cells)], times, starts, config.t_decode_window)
    del spikes
    with threadpool_limits(limits=1), worker_context(config.n_jobs):
        decoded = Parallel(verbose=config.par_verbose, max_nbytes='1M')(
            delayed(decode_one_trial)(int(index), rates, labels, starts, config) for index in tests)
    observed, predicted, c_values, null, null_c = [np.stack([d[i] for d in decoded]) for i in range(5)]
    return {
        'session': path.stem, 'cue': int(cue), 'cue_deg': int((cue - 1) * 45 - 135),
        'trial_idx': trials[tests], 'time_bins': starts, 'cell_idx': cells,
        'num_cells': cells.size, 'num_trials': tests.size,
        'decoding_test_labels': labels[tests], 'decoding_confidence': observed,
        'decoding_predicted_labels': predicted, 'decoding_classifier_c': c_values,
        'decoding_accuracy': (predicted == labels[tests, None]).mean(axis=0),
        'decoding_confidence_null': null, 'decoding_classifier_c_null': null_c,
        'n_decode_shuffle': config.n_decode_shuffle,
        'logistic_calibration_effective_cv_folds': sorted({n for d in decoded for n in d[5]}),
        'classifier_c_grid': CLASSIFIER_C_GRID,
        'classifier_c_grid_search_cv_folds': CLASSIFIER_C_GRID_SEARCH_CV,
        'null_policy': 'training-trial labels independently permuted per bin and shuffle',
        'config': json.loads(json.dumps(asdict(config), default=json_value)),
    }


def main(config: Config):
    path = config.cache_dir / 'decoding_confidence.pkl'
    if config.plot_only:
        results = cache_io.read(path)
        for result in results:
            plot_session(result, config.cache_dir, config.plot_actual_trial_id)
        return results
    selection_path = config.cache_dir / 'cell_trial_selection.pkl'
    selections = cache_io.read(selection_path)
    selection_by_session = {r['session']: r for r in selections}
    if len(selection_by_session) != len(selections):
        raise ValueError('Selection cache must contain exactly one entry per session.')
    files = session_files(config.data_dir, config.session_list_file)
    eligible = []
    for file in files:
        selection = selection_by_session.get(file.stem)
        if selection is None or selection['max_num_cells_per_group'] < config.min_cell_per_group or selection['num_trials'] < config.min_trials_good_session:
            warnings.warn(f'{file.stem}: does not pass decoding selection thresholds.', stacklevel=2)
        else:
            eligible.append(file)
    if config.max_sessions_to_run is not None:
        if config.max_sessions_to_run < 1:
            raise ValueError('max_sessions_to_run must be positive.')
        eligible = eligible[:config.max_sessions_to_run]
    if not eligible:
        raise ValueError('No sessions passed decoding selection thresholds.')
    results = []
    checkpoint_dir = config.cache_dir / 'checkpoints' / 'decoding'
    excluded = ('n_jobs', 'par_verbose', 'resume', 'plot_only', 'save_figures',
                'plot_actual_trial_id', 'session_list_file', 'max_sessions_to_run')
    for file in eligible:
        key = fingerprint(config, [selection_path, file], exclude=excluded)
        checkpoint = checkpoint_dir / f'{file.stem}.pkl'
        cached = cache_io.read(checkpoint) if config.resume and checkpoint.exists() else None
        if cached is not None and cached['fingerprint'] == key:
            result = cached['result']
            print(f'{file.stem}: reused completed decoding checkpoint', flush=True)
        else:
            print(f'{file.stem}: decoding one observed + {config.n_decode_shuffle} null estimates', flush=True)
            result = decode_session(file, selection_by_session[file.stem], config)
            result['fingerprint'] = key
            cache_io.save({'fingerprint': key, 'result': result}, checkpoint)
        results.append(result)
        cache_io.save(results, path)
        if config.save_figures:
            plot_session(result, config.cache_dir, config.plot_actual_trial_id)
    return results


if __name__ == '__main__':
    main(tyro.cli(Config))
