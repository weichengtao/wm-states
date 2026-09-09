import os
import pickle
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tyro
from joblib import Parallel, delayed
from scipy.io import loadmat
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from scripts.figure_exports import configure_figure_style, save_figure_all_formats
except ModuleNotFoundError:
    from figure_exports import configure_figure_style, save_figure_all_formats

matplotlib.use('Agg')
configure_figure_style(matplotlib)


CLASSIFIER_C_GRID = (1.0, 0.1, 0.01)
CLASSIFIER_C_GRID_SEARCH_CV = 5
CLASSIFIER_C_GRID_SEARCH_SCORING = 'balanced_accuracy'


class CellsUsedForDecoder(str, Enum):
    """Cell pool to use as decoder features."""

    PREFERRED = 'preferred'
    PREFERRED_AND_OPPOSITE = 'preferred_and_opposite'
    SELECTIVE = 'selective'
    STATIONARY = 'stationary'
    PASSED_PRESENCE_RATIO = 'passed_presence_ratio'
    ALL = 'all'


class SVMKernel(str, Enum):
    """Kernel used by the decoder's SVM classifier."""

    RBF = 'rbf'
    LINEAR = 'linear'


class DecoderModel(str, Enum):
    """Classifier used by the decoder."""

    SVM = 'svm'
    LOGISTIC_REGRESSION = 'logistic_regression'


class LogisticCalibrationMethod(str, Enum):
    """Probability calibration applied to logistic regression."""

    NONE = 'none'
    SIGMOID = 'sigmoid'
    ISOTONIC = 'isotonic'


def save_pickle_atomic(value, output_path):
    """Replace a pickle only after its complete contents reach disk."""
    output_path = Path(output_path)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='wb',
            dir=output_path.parent,
            prefix=f'.{output_path.name}.',
            suffix='.tmp',
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            pickle.dump(value, temporary_file)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, output_path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def cue_to_deg(cue):
    """Convert cue indices (1-8) to degrees on a -135..180 scale."""
    cue = np.asarray(cue)
    cue = (cue - 1) % 8 + 1
    return (cue - 1) * 45 - 135

def get_opposite_cue(cue):
    """Return the cue index opposite to the given cue (1-8)."""
    return (cue + 3) % 8 + 1

def preferred_cue_from_cells(pref_cues):
    """Pick the most frequent preferred cue across cells."""
    cues, counts = np.unique(pref_cues, return_counts=True)
    if cues.size == 0:
        return None
    best = np.argmax(counts)
    return int(cues[best])

def preferred_cue_from_partitions(partition_cues):
    """Pick the most frequent preferred cue across partitions."""
    cues, counts = np.unique(partition_cues, return_counts=True)
    if cues.size == 0:
        return None
    best = np.argmax(counts)
    return int(cues[best])


def decoder_cells_for_partition(
    partition,
    mode: CellsUsedForDecoder,
    preferred_cue: int,
    opposite_cue: int,
    num_cells_total: int,
) -> set[int]:
    """Return the cell indices available to the decoder in one partition."""
    if mode is CellsUsedForDecoder.ALL:
        return set(range(num_cells_total))

    if mode is CellsUsedForDecoder.STATIONARY:
        if 'cell_idx_stationary' not in partition:
            raise ValueError(
                'The selection cache does not contain cell_idx_stationary. '
                'Rerun scripts/cell_trial_selection.py before using '
                'cells_used_for_decoder=stationary.'
            )
        return set(np.asarray(partition['cell_idx_stationary'], dtype=np.int64).tolist())

    if mode is CellsUsedForDecoder.PASSED_PRESENCE_RATIO:
        if 'cell_idx_passed_presence_ratio' not in partition:
            raise ValueError(
                'The selection cache does not contain cell_idx_passed_presence_ratio. '
                'Rerun scripts/cell_trial_selection.py before using '
                'cells_used_for_decoder=passed_presence_ratio.'
            )
        return set(
            np.asarray(partition['cell_idx_passed_presence_ratio'], dtype=np.int64).tolist()
        )

    cell_properties = partition['cell_properties']
    cells = np.asarray(cell_properties['cell_idx'], dtype=np.int64)
    if mode is CellsUsedForDecoder.SELECTIVE:
        return set(cells.tolist())

    preferred_cues = np.asarray(cell_properties['mean_pref_test'])
    if mode is CellsUsedForDecoder.PREFERRED:
        keep = preferred_cues == preferred_cue
    elif mode is CellsUsedForDecoder.PREFERRED_AND_OPPOSITE:
        keep = (preferred_cues == preferred_cue) | (preferred_cues == opposite_cue)
    else:
        raise ValueError(f'Unsupported decoder cell mode: {mode}')
    return set(cells[keep].tolist())

def compute_binned_rates(spikes, t, bin_starts, window_ms):
    """Compute firing rates per trial, time bin, and cell."""
    dt = float(t[1] - t[0])
    num_trials, _, num_cells = spikes.shape
    num_bins = len(bin_starts)
    rates = np.empty((num_trials, num_bins, num_cells), dtype=np.float32)
    max_float32 = np.finfo(np.float32).max
    for i, start in enumerate(bin_starts):
        # Build a mask for this time window and convert spike counts to Hz.
        mask = (t >= start) & (t < start + window_ms)
        if not np.any(mask):
            rates[:, i, :] = 0.0
            continue
        duration_s = mask.sum() * dt / 1000.0
        if not np.isfinite(duration_s) or duration_s <= 0:
            rates[:, i, :] = 0.0
            continue
        counts = np.asarray(spikes[:, mask, :], dtype=np.float64).sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            rate_bin = np.divide(
                counts,
                duration_s,
                out=np.zeros_like(counts, dtype=np.float64),
                where=duration_s != 0,
            )
        rate_bin = np.nan_to_num(rate_bin, nan=0.0, posinf=0.0, neginf=0.0)
        rate_bin = np.clip(rate_bin, -max_float32, max_float32)
        rates[:, i, :] = rate_bin.astype(np.float32, copy=False)
    return rates


def shuffle_trial_idx_within_labels(spikes, labels, rng):
    """Shuffle trial indices independently for each cell within each label.

    The same cell-specific trial permutation is applied to every time sample,
    so each cell keeps its trial waveform while its trial assignment is
    shuffled. Labels and the trial axis itself are unchanged.
    """
    spikes = np.asarray(spikes)
    labels = np.asarray(labels)
    if spikes.ndim != 3:
        raise ValueError('spikes must have shape (trial, time, cell).')
    if labels.shape != (spikes.shape[0],):
        raise ValueError('labels must contain one label per trial.')

    shuffled_spikes = np.array(spikes, copy=True)
    for label in np.unique(labels):
        label_trial_idx = np.flatnonzero(labels == label)
        for cell_idx in range(spikes.shape[2]):
            source_idx = label_trial_idx[rng.permutation(label_trial_idx.size)]
            shuffled_spikes[label_trial_idx, :, cell_idx] = spikes[
                source_idx, :, cell_idx
            ]
    return shuffled_spikes


def prepare_decoder_training_samples(
    binned_rates,
    labels,
    test_bin_idx,
    delay_bin_mask,
    train_delay_decoder_using_all_delay_time_bins: bool,
):
    """Return the feature matrix and labels used to fit one decoder.

    Delay-bin pooling treats every delay bin from every selected training trial
    as a separate sample. The caller is responsible for selecting training
    trials first, which ensures that a held-out test trial cannot contribute
    activity from any time bin.
    """
    binned_rates = np.asarray(binned_rates)
    labels = np.asarray(labels)
    delay_bin_mask = np.asarray(delay_bin_mask, dtype=np.bool_)
    if binned_rates.ndim != 3:
        raise ValueError('binned_rates must have shape (trial, bin, cell).')
    if labels.shape != (binned_rates.shape[0],):
        raise ValueError('labels must contain one label per training trial.')
    if delay_bin_mask.shape != (binned_rates.shape[1],):
        raise ValueError('delay_bin_mask must contain one value per time bin.')

    if (
        train_delay_decoder_using_all_delay_time_bins
        and delay_bin_mask[test_bin_idx]
    ):
        num_delay_bins = int(delay_bin_mask.sum())
        X_train = binned_rates[:, delay_bin_mask, :].reshape(
            binned_rates.shape[0] * num_delay_bins,
            binned_rates.shape[2],
        )
        y_train = np.repeat(labels, num_delay_bins)
    else:
        X_train = binned_rates[:, test_bin_idx, :]
        y_train = labels
    return X_train, y_train


def decoder_training_sample_groups(num_training_trials, num_training_samples):
    """Map decoder samples to their source trials.

    Delay-bin pooling stores samples in trial-major order, so repeating each
    trial index keeps every time bin from one trial in the same calibration
    fold. Without pooling, there is one sample per trial.
    """
    if num_training_trials < 1:
        raise ValueError('num_training_trials must be at least 1.')
    if num_training_samples % num_training_trials != 0:
        raise ValueError(
            'The number of decoder training samples must be divisible by the '
            'number of source trials.'
        )
    samples_per_trial = num_training_samples // num_training_trials
    if samples_per_trial < 1:
        raise ValueError('Each training trial must contribute at least one sample.')
    return np.repeat(
        np.arange(num_training_trials, dtype=np.int64),
        samples_per_trial,
    )


def make_grouped_stratified_cv_splits(
    labels,
    sample_groups,
    requested_cv_folds: int,
    seed: int,
    *,
    purpose: str,
    allow_fold_reduction: bool,
):
    """Build deterministic, class-stratified folds grouped by source trial.

    Every returned train and validation fold contains both classes, and no
    source trial can appear on both sides of a split. Fold reduction is useful
    for calibration, while classifier-C selection deliberately requires the
    configured five-fold design.
    """
    labels = np.asarray(labels)
    sample_groups = np.asarray(sample_groups)
    if labels.ndim != 1:
        raise ValueError(f'{purpose} labels must be one-dimensional.')
    if sample_groups.shape != labels.shape:
        raise ValueError(f'{purpose} groups must contain one value per sample.')
    if requested_cv_folds < 2:
        raise ValueError(f'{purpose} CV must use at least two folds.')

    classes = np.unique(labels)
    if classes.size != 2:
        raise ValueError(
            f'{purpose} requires exactly two classes in the training set.'
        )
    unique_groups = np.unique(sample_groups)
    labels_by_group = [
        np.unique(labels[sample_groups == group]) for group in unique_groups
    ]
    groups_are_class_homogeneous = all(
        group_labels.size == 1 for group_labels in labels_by_group
    )
    if groups_are_class_homogeneous:
        group_labels = np.asarray(
            [labels_for_group[0] for labels_for_group in labels_by_group]
        )
        groups_per_class = [
            np.count_nonzero(group_labels == class_label)
            for class_label in classes
        ]
    else:
        groups_per_class = [
            np.unique(sample_groups[labels == class_label]).size
            for class_label in classes
        ]
    requested_cv_folds = int(requested_cv_folds)
    max_folds = min(requested_cv_folds, min(groups_per_class))
    if max_folds < 2:
        raise ValueError(
            f'{purpose} requires at least two source-trial groups containing '
            'each class.'
        )
    if not allow_fold_reduction and max_folds < requested_cv_folds:
        raise ValueError(
            f'{purpose} requires {requested_cv_folds} source-trial groups '
            f'containing each class, but at most {max_folds} folds are feasible.'
        )

    fold_counts = (
        range(max_folds, 1, -1)
        if allow_fold_reduction
        else (requested_cv_folds,)
    )

    dummy_features = np.zeros((labels.size, 1), dtype=np.float32)
    if groups_are_class_homogeneous:
        n_splits = max_folds if allow_fold_reduction else requested_cv_folds
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=int(seed),
        )
        splits = []
        for train_group_indices, validation_group_indices in splitter.split(
            unique_groups,
            group_labels,
        ):
            train_groups = unique_groups[train_group_indices]
            validation_groups = unique_groups[validation_group_indices]
            splits.append((
                np.flatnonzero(np.isin(sample_groups, train_groups)),
                np.flatnonzero(np.isin(sample_groups, validation_groups)),
            ))
        return splits, n_splits

    for n_splits in fold_counts:
        # A few deterministic alternatives make grouped stratification robust
        # when shuffled null labels produce mixed-label trial groups.
        for split_attempt in range(10):
            splitter = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=int(seed) + split_attempt,
            )
            splits = list(
                splitter.split(dummy_features, labels, groups=sample_groups)
            )
            valid = True
            for train_indices, validation_indices in splits:
                if (
                    np.unique(labels[train_indices]).size != 2
                    or np.unique(labels[validation_indices]).size != 2
                    or np.intersect1d(
                        sample_groups[train_indices],
                        sample_groups[validation_indices],
                    ).size
                    != 0
                ):
                    valid = False
                    break
            if valid:
                return splits, n_splits

    raise ValueError(
        f'Could not construct {purpose.lower()} folds with both classes in '
        'every train and validation split while keeping source trials grouped.'
    )


def make_logistic_calibration_cv_splits(
    labels,
    sample_groups,
    requested_cv_folds: int,
    seed: int,
):
    """Build deterministic grouped folds for logistic calibration."""
    return make_grouped_stratified_cv_splits(
        labels,
        sample_groups,
        requested_cv_folds,
        seed,
        purpose='Logistic calibration',
        allow_fold_reduction=True,
    )


def create_base_decoder(
    classifier_c: float,
    decoder_model: DecoderModel,
    svm_kernel: SVMKernel,
    seed: int,
    *,
    svm_probability: bool = True,
):
    """Create the scaled, uncalibrated classifier used by one decoder fit."""
    decoder_model = DecoderModel(decoder_model)
    svm_kernel = SVMKernel(svm_kernel)
    if decoder_model is DecoderModel.SVM:
        classifier = SVC(
            kernel=svm_kernel.value,
            C=float(classifier_c),
            probability=svm_probability,
            random_state=seed,
        )
    elif decoder_model is DecoderModel.LOGISTIC_REGRESSION:
        classifier = LogisticRegression(
            solver='liblinear',
            C=float(classifier_c),
            max_iter=1000,
            random_state=seed,
        )
    else:
        raise ValueError(f'Unsupported decoder model: {decoder_model}')
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', classifier),
    ])


def select_classifier_c(
    X_train,
    y_train,
    sample_groups,
    decoder_model: DecoderModel,
    svm_kernel: SVMKernel,
    seed: int,
    *,
    fit_context: str = 'Decoder',
) -> float:
    """Select C on one decoder's fully prepared outer-training set."""
    try:
        search_cv, _ = make_grouped_stratified_cv_splits(
            y_train,
            sample_groups,
            CLASSIFIER_C_GRID_SEARCH_CV,
            seed,
            purpose='Classifier C grid search',
            allow_fold_reduction=False,
        )
    except ValueError as exc:
        raise ValueError(f'{fit_context}: {exc}') from exc

    # Balanced accuracy depends only on class predictions. Avoid SVC's costly
    # internal probability fitting during candidate evaluation; the final
    # selected-C model still enables predict_proba below.
    search_estimator = create_base_decoder(
        CLASSIFIER_C_GRID[0],
        decoder_model,
        svm_kernel,
        seed,
        svm_probability=False,
    )
    search = GridSearchCV(
        estimator=search_estimator,
        param_grid={'classifier__C': list(CLASSIFIER_C_GRID)},
        scoring=CLASSIFIER_C_GRID_SEARCH_SCORING,
        n_jobs=1,
        refit=False,
        cv=search_cv,
        error_score='raise',
    )
    search.fit(X_train, y_train)
    return float(search.best_params_['classifier__C'])


def decode_one_trial(
    test_idx,
    binned_rates,
    labels,
    seed,
    balance_decoder_training_trials: bool,
    classifier_c: float,
    decoder_model: DecoderModel,
    svm_kernel: SVMKernel,
    n_repeats_for_model_fit: int,
    n_shuffle: int,
    cue_preserved_train_set_shuffle: bool = False,
    bin_starts=None,
    train_delay_decoder_using_all_delay_time_bins: bool = False,
    logistic_calibration_method: LogisticCalibrationMethod = (
        LogisticCalibrationMethod.NONE
    ),
    logistic_calibration_cv: int = 5,
    grid_search_for_c: bool = False,
):
    """Decode a single test trial across all bins with optional shuffles.

    The model-fit repeats use one RNG stream to draw a fresh balanced training
    set for every repeat. When cue_preserved_train_set_shuffle is enabled,
    repeats after the first independently shuffle each cell's trial indices
    within cue labels after training-set selection. The first repeat remains
    unchanged. Null shuffles use a separate stream and one unshuffled training
    set, so changing repeat behavior does not change the null estimates. When
    delay pooling is enabled, a decoder tested at a bin starting in [500, 1400]
    is trained on every delay bin from each selected training trial.
    When grid_search_for_c is enabled, each empirical or null decoder searches
    C independently after its balancing, feature shuffling, pooling, and label
    shuffling steps. The returned C arrays align with the empirical-repeat and
    null confidence arrays.
    """
    if n_repeats_for_model_fit < 1:
        raise ValueError('n_repeats_for_model_fit must be at least 1.')
    logistic_calibration_method = LogisticCalibrationMethod(
        logistic_calibration_method
    )
    if (
        decoder_model is DecoderModel.LOGISTIC_REGRESSION
        and logistic_calibration_method is not LogisticCalibrationMethod.NONE
        and logistic_calibration_cv < 2
    ):
        raise ValueError('logistic_calibration_cv must be at least 2.')

    repeat_rng = np.random.default_rng(seed + int(test_idx))
    num_trials = binned_rates.shape[0]
    num_bins = binned_rates.shape[1]
    if bin_starts is None:
        if train_delay_decoder_using_all_delay_time_bins:
            raise ValueError(
                'bin_starts is required when '
                'train_delay_decoder_using_all_delay_time_bins is enabled.'
            )
        delay_bin_mask = np.zeros(num_bins, dtype=np.bool_)
    else:
        bin_starts = np.asarray(bin_starts)
        if bin_starts.shape != (num_bins,):
            raise ValueError('bin_starts must contain one start time per time bin.')
        delay_bin_mask = (bin_starts >= 500) & (bin_starts <= 1400)
    # Leave-one-out split for the current test trial.
    train_mask = np.ones(num_trials, dtype=np.bool_)
    train_mask[test_idx] = False
    train_idx = np.nonzero(train_mask)[0]
    y_train_full = labels[train_idx]
    pref_idx = train_idx[y_train_full == 1]
    opp_idx = train_idx[y_train_full == 0]
    if pref_idx.size == 0 or opp_idx.size == 0:
        repeat_conf = np.full(
            (n_repeats_for_model_fit, num_bins), np.nan, dtype=np.float32
        )
        repeat_predicted_labels = np.full(
            (n_repeats_for_model_fit, num_bins), -1, dtype=np.int8
        )
        repeat_accuracy = np.full(
            (n_repeats_for_model_fit, num_bins), np.nan, dtype=np.float32
        )
        null_conf = (
            None
            if n_shuffle <= 0
            else np.full((num_bins, n_shuffle), np.nan, dtype=np.float32)
        )
        repeat_classifier_c = np.full(
            (n_repeats_for_model_fit, num_bins), np.nan, dtype=np.float64
        )
        null_classifier_c = (
            None
            if n_shuffle <= 0
            else np.full((num_bins, n_shuffle), np.nan, dtype=np.float64)
        )
        return (
            repeat_conf,
            repeat_predicted_labels,
            repeat_accuracy,
            null_conf,
            (),
            repeat_classifier_c,
            null_classifier_c,
        )

    train_balanced_repeats = []
    for _ in range(n_repeats_for_model_fit):
        if balance_decoder_training_trials:
            # Balance classes independently for every model-fit repeat.
            n_train = min(pref_idx.size, opp_idx.size)
            pref_sel = repeat_rng.choice(pref_idx, size=n_train, replace=False)
            opp_sel = repeat_rng.choice(opp_idx, size=n_train, replace=False)
            train_balanced = np.concatenate([pref_sel, opp_sel])
        else:
            # Use every available leave-one-out training trial.
            train_balanced = train_idx
        train_balanced_repeats.append(train_balanced)

    train_shuffle_rng = np.random.default_rng(seed + int(test_idx))

    repeat_conf = np.empty(
        (n_repeats_for_model_fit, num_bins), dtype=np.float32
    )
    repeat_predicted_labels = np.empty(
        (n_repeats_for_model_fit, num_bins), dtype=np.int8
    )
    repeat_accuracy = np.empty(
        (n_repeats_for_model_fit, num_bins), dtype=np.float32
    )
    repeat_classifier_c = np.empty(
        (n_repeats_for_model_fit, num_bins), dtype=np.float64
    )

    effective_calibration_cv_folds: set[int] = set()

    def create_model(selected_classifier_c, y_train, sample_groups):
        base_estimator = create_base_decoder(
            selected_classifier_c,
            decoder_model,
            svm_kernel,
            seed,
        )
        if decoder_model is DecoderModel.SVM:
            return base_estimator
        elif decoder_model is DecoderModel.LOGISTIC_REGRESSION:
            if logistic_calibration_method is LogisticCalibrationMethod.NONE:
                return base_estimator
            calibration_cv, effective_cv_folds = (
                make_logistic_calibration_cv_splits(
                    y_train,
                    sample_groups,
                    logistic_calibration_cv,
                    seed,
                )
            )
            effective_calibration_cv_folds.add(effective_cv_folds)
            return CalibratedClassifierCV(
                estimator=base_estimator,
                method=logistic_calibration_method.value,
                cv=calibration_cv,
                ensemble=False,
            )
        else:
            raise ValueError(f'Unsupported decoder model: {decoder_model}')

    for repeat_idx, train_balanced in enumerate(train_balanced_repeats):
        y_bal = labels[train_balanced]
        repeat_binned_rates = binned_rates[train_balanced]
        if cue_preserved_train_set_shuffle and repeat_idx > 0:
            repeat_binned_rates = shuffle_trial_idx_within_labels(
                repeat_binned_rates,
                y_bal,
                train_shuffle_rng,
            )
        pooled_delay_training = None
        if (
            train_delay_decoder_using_all_delay_time_bins
            and np.any(delay_bin_mask)
        ):
            first_delay_bin = int(np.flatnonzero(delay_bin_mask)[0])
            pooled_delay_training = prepare_decoder_training_samples(
                repeat_binned_rates,
                y_bal,
                first_delay_bin,
                delay_bin_mask,
                True,
            )
        for b in range(num_bins):
            if pooled_delay_training is not None and delay_bin_mask[b]:
                X_train_raw, y_train = pooled_delay_training
            else:
                X_train_raw, y_train = prepare_decoder_training_samples(
                    repeat_binned_rates,
                    y_bal,
                    b,
                    delay_bin_mask,
                    False,
                )
            X_train = np.nan_to_num(
                X_train_raw.astype(np.float64, copy=False),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            X_test = np.nan_to_num(
                binned_rates[test_idx, b, :][None, :].astype(np.float64, copy=False),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            sample_groups = decoder_training_sample_groups(
                repeat_binned_rates.shape[0],
                X_train.shape[0],
            )
            selected_classifier_c = classifier_c
            if grid_search_for_c:
                selected_classifier_c = select_classifier_c(
                    X_train,
                    y_train,
                    sample_groups,
                    decoder_model,
                    svm_kernel,
                    seed,
                    fit_context=(
                        f'Empirical decoder test_idx={test_idx}, '
                        f'repeat={repeat_idx}, bin={b}'
                    ),
                )
            repeat_classifier_c[repeat_idx, b] = selected_classifier_c
            model = create_model(selected_classifier_c, y_train, sample_groups)
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)[0]
            class_index = int(np.flatnonzero(model.classes_ == 1)[0])
            repeat_conf[repeat_idx, b] = proba[class_index]
            predicted_label = model.predict(X_test)[0]
            repeat_predicted_labels[repeat_idx, b] = predicted_label
            if predicted_label in (0, 1):
                repeat_accuracy[repeat_idx, b] = float(
                    predicted_label == labels[test_idx]
                )

    null_conf = None
    null_classifier_c = None
    if n_shuffle > 0:
        null_conf = np.empty((num_bins, n_shuffle), dtype=np.float32)
        null_classifier_c = np.empty((num_bins, n_shuffle), dtype=np.float64)
        # Keep the shuffle path independent from model-fit repeats. Its first
        # balanced training set matches the pre-repeat implementation.
        shuffle_rng = np.random.default_rng(seed + int(test_idx))
        if balance_decoder_training_trials:
            n_train = min(pref_idx.size, opp_idx.size)
            pref_sel = shuffle_rng.choice(pref_idx, size=n_train, replace=False)
            opp_sel = shuffle_rng.choice(opp_idx, size=n_train, replace=False)
            shuffle_train_balanced = np.concatenate([pref_sel, opp_sel])
        else:
            shuffle_train_balanced = train_idx
        y_bal = labels[shuffle_train_balanced]
        shuffle_binned_rates = binned_rates[shuffle_train_balanced]
        pooled_delay_training = None
        if (
            train_delay_decoder_using_all_delay_time_bins
            and np.any(delay_bin_mask)
        ):
            first_delay_bin = int(np.flatnonzero(delay_bin_mask)[0])
            pooled_delay_training = prepare_decoder_training_samples(
                shuffle_binned_rates,
                y_bal,
                first_delay_bin,
                delay_bin_mask,
                True,
            )

        for b in range(num_bins):
            if pooled_delay_training is not None and delay_bin_mask[b]:
                X_train_raw, y_train = pooled_delay_training
            else:
                X_train_raw, y_train = prepare_decoder_training_samples(
                    shuffle_binned_rates,
                    y_bal,
                    b,
                    delay_bin_mask,
                    False,
                )
            X_train = np.nan_to_num(
                X_train_raw.astype(np.float64, copy=False),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            X_test = np.nan_to_num(
                binned_rates[test_idx, b, :][None, :].astype(
                    np.float64, copy=False
                ),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            sample_groups = decoder_training_sample_groups(
                shuffle_binned_rates.shape[0],
                X_train.shape[0],
            )
            # Shuffle labels to estimate a null confidence distribution.
            for s in range(n_shuffle):
                y_shuf = shuffle_rng.permutation(y_train)
                selected_classifier_c = classifier_c
                if grid_search_for_c:
                    selected_classifier_c = select_classifier_c(
                        X_train,
                        y_shuf,
                        sample_groups,
                        decoder_model,
                        svm_kernel,
                        seed,
                        fit_context=(
                            f'Null decoder test_idx={test_idx}, bin={b}, '
                            f'shuffle={s}'
                        ),
                    )
                null_classifier_c[b, s] = selected_classifier_c
                model = create_model(
                    selected_classifier_c,
                    y_shuf,
                    sample_groups,
                )
                model.fit(X_train, y_shuf)
                proba = model.predict_proba(X_test)[0]
                class_index = int(np.flatnonzero(model.classes_ == 1)[0])
                null_conf[b, s] = proba[class_index]

    return (
        repeat_conf,
        repeat_predicted_labels,
        repeat_accuracy,
        null_conf,
        tuple(sorted(effective_calibration_cv_folds)),
        repeat_classifier_c,
        null_classifier_c,
    )

def plot_decoding_heatmap(
    fig_dir,
    session,
    pref_cue,
    cue_angle,
    trial_idx_pref,
    bin_starts,
    decode_confidence,
    plot_actual_trial_id,
    num_cells,
):
    """Save a heatmap of decoding confidence over time and trials."""
    fig, ax = plt.subplots(1, 1, figsize=(5, 4), layout='constrained')
    sns.heatmap(
        decode_confidence,
        ax=ax,
        vmin=0.5,
        vmax=1.0,
        cmap=None,
        cbar_kws={'label': 'Decoding confidence'},
    )
    xticks = [i for i, t_val in enumerate(bin_starts) if t_val % 200 == 0]
    ax.set_xticks([x + 0.5 for x in xticks])
    ax.set_xticklabels([str(bin_starts[x]) for x in xticks], rotation=0)
    ytick_positions = np.arange(9, trial_idx_pref.size, 10)
    ax.set_yticks(ytick_positions + 0.5)
    if plot_actual_trial_id:
        ytick_labels = [str(trial_idx_pref[i]) for i in ytick_positions]
    else:
        ytick_labels = [str(i + 1) for i in ytick_positions]
    ax.set_yticklabels(ytick_labels, rotation=0)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Trial')
    ax.set_title(
        f'{session} ({cue_angle}$\\degree$), {num_cells} cells, {trial_idx_pref.size} trials'
    )
    save_figure_all_formats(fig, fig_dir / f'{session}_{pref_cue}.png', dpi=300)
    plt.close(fig)

def plot_decoding_confidence_lineplot(
    fig_dir,
    session,
    pref_cue,
    cue_angle,
    trial_idx_pref,
    bin_starts,
    decode_confidence,
    decode_predicted_labels,
    decode_accuracy_repeats,
    test_labels,
    num_cells,
):
    """Save a line plot of decoding confidence and accuracy over time."""
    fig, ax = plt.subplots(1, 1, figsize=(5, 4), layout='constrained')
    for trial_confidence in decode_confidence:
        ax.plot(bin_starts, trial_confidence, color='darkgray', alpha=0.1)

    decode_predicted_labels = np.asarray(decode_predicted_labels)
    decode_accuracy_repeats = np.asarray(decode_accuracy_repeats)
    test_labels = np.asarray(test_labels)
    if decode_predicted_labels.ndim != 3:
        raise ValueError(
            'decode_predicted_labels must have shape (trial, repeat, bin).'
        )
    if (
        decode_predicted_labels.shape[0] != decode_confidence.shape[0]
        or decode_predicted_labels.shape[2] != decode_confidence.shape[1]
    ):
        raise ValueError(
            'decode_predicted_labels must have trial and bin dimensions matching '
            'decode_confidence.'
        )
    if decode_accuracy_repeats.shape != (
        decode_predicted_labels.shape[1],
        decode_confidence.shape[1],
    ):
        raise ValueError(
            'decode_accuracy_repeats must have shape (repeat, bin).'
        )
    if test_labels.shape != (decode_confidence.shape[0],):
        raise ValueError('test_labels must contain one label per test trial.')

    decoding_accuracy = np.nanmean(decode_accuracy_repeats, axis=0)
    mean_confidence = np.nanmean(decode_confidence, axis=0)
    ax.plot(
        bin_starts,
        decoding_accuracy,
        color='tab:blue',
        linestyle='-',
        alpha=0.5,
        label='Mean decoding accuracy',
    )
    ax.plot(
        bin_starts,
        mean_confidence,
        color='tab:orange',
        linestyle='-',
        alpha=0.5,
        label='Mean decoding confidence',
    )
    ax.axvline(0, color='black', linewidth=1)
    ax.axhline(0.5, color='black', linestyle='--', linewidth=1)
    ax.set_xlim(-200, 1400)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 500, 1000])
    ax.set_yticks(
        [0.0, 0.25, 0.5, 0.75, 1.0],
        labels=['0', '', '0.5', '', '1'],
    )
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Decoding Confidence / Accuracy')
    ax.legend(loc='lower left', frameon=False)
    ax.set_title(
        f'{session} ({cue_angle}$\\degree$), {num_cells} cells, {trial_idx_pref.size} trials'
    )
    save_figure_all_formats(fig, fig_dir / f'{session}_{pref_cue}_lineplot.png', dpi=300)
    plt.close(fig)


def log10_classifier_c(classifier_c):
    values = np.asarray(classifier_c, dtype=np.float64)
    invalid = (~np.isnan(values)) & ((~np.isfinite(values)) | (values <= 0))
    if np.any(invalid):
        raise ValueError('Classifier C values must be positive and finite or NaN.')
    transformed = np.full(values.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(values)
    transformed[valid] = np.log10(values[valid])
    return transformed


def mean_finite(values, axis: int):
    """Average finite values without emitting warnings for empty slices."""
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    counts = np.sum(valid, axis=axis)
    totals = np.sum(np.where(valid, values, 0.0), axis=axis)
    return np.divide(
        totals,
        counts,
        out=np.full(totals.shape, np.nan, dtype=np.float64),
        where=counts > 0,
    )


def mean_log10_classifier_c(classifier_c, axis: int):
    """Average selected C values in log10 space without all-NaN warnings."""
    return mean_finite(log10_classifier_c(classifier_c), axis=axis)


def classifier_c_plot_limits(log10_c, grid_search_for_c: bool):
    """Return stable log10(C) plot limits and candidate ticks."""
    if grid_search_for_c:
        candidate_ticks = np.sort(log10_classifier_c(CLASSIFIER_C_GRID))
        return float(candidate_ticks[0]), float(candidate_ticks[-1]), candidate_ticks

    finite_values = np.asarray(log10_c, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return -0.5, 0.5, None
    lower = float(np.min(finite_values))
    upper = float(np.max(finite_values))
    if np.isclose(lower, upper):
        lower -= 0.5
        upper += 0.5
    return lower, upper, None


def plot_classifier_c_heatmap(
    fig_dir,
    session,
    pref_cue,
    cue_angle,
    trial_idx_pref,
    bin_starts,
    mean_log10_c,
    plot_actual_trial_id,
    num_cells,
    grid_search_for_c,
    *,
    label,
    filename_suffix,
):
    """Save a trial-by-time heatmap of mean selected log10(C)."""
    mean_log10_c = np.asarray(mean_log10_c, dtype=np.float64)
    vmin, vmax, candidate_ticks = classifier_c_plot_limits(
        mean_log10_c,
        grid_search_for_c,
    )
    colorbar_options = {'label': r'$\log_{10}(C)$'}
    if candidate_ticks is not None:
        colorbar_options['ticks'] = candidate_ticks

    fig, ax = plt.subplots(1, 1, figsize=(5, 4), layout='constrained')
    sns.heatmap(
        mean_log10_c,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        cmap='viridis',
        cbar_kws=colorbar_options,
    )
    xticks = [i for i, t_val in enumerate(bin_starts) if t_val % 200 == 0]
    ax.set_xticks([x + 0.5 for x in xticks])
    ax.set_xticklabels([str(bin_starts[x]) for x in xticks], rotation=0)
    ytick_positions = np.arange(9, trial_idx_pref.size, 10)
    ax.set_yticks(ytick_positions + 0.5)
    if plot_actual_trial_id:
        ytick_labels = [str(trial_idx_pref[i]) for i in ytick_positions]
    else:
        ytick_labels = [str(i + 1) for i in ytick_positions]
    ax.set_yticklabels(ytick_labels, rotation=0)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Trial')
    ax.set_title(
        f'{session} ({cue_angle}$\\degree$), {num_cells} cells, '
        f'{trial_idx_pref.size} trials\n{label}'
    )
    save_figure_all_formats(
        fig,
        fig_dir / f'{session}_{pref_cue}_{filename_suffix}.png',
        dpi=300,
    )
    plt.close(fig)


def plot_classifier_c_lineplot(
    fig_dir,
    session,
    pref_cue,
    cue_angle,
    trial_idx_pref,
    bin_starts,
    mean_log10_c,
    num_cells,
    grid_search_for_c,
    *,
    label,
    filename_suffix,
):
    """Save trial and session summaries of mean selected log10(C)."""
    mean_log10_c = np.asarray(mean_log10_c, dtype=np.float64)
    vmin, vmax, candidate_ticks = classifier_c_plot_limits(
        mean_log10_c,
        grid_search_for_c,
    )
    fig, ax = plt.subplots(1, 1, figsize=(5, 4), layout='constrained')
    for trial_log10_c in mean_log10_c:
        ax.plot(bin_starts, trial_log10_c, color='darkgray', alpha=0.1)
    session_mean_log10_c = mean_finite(mean_log10_c, axis=0)
    ax.plot(
        bin_starts,
        session_mean_log10_c,
        color='tab:orange',
        alpha=0.8,
        label='Session mean',
    )
    ax.axvline(0, color='black', linewidth=1)
    if bin_starts.size:
        ax.set_xlim(float(bin_starts[0]), float(bin_starts[-1]))
    ax.set_ylim(vmin, vmax)
    if candidate_ticks is not None:
        ax.set_yticks(candidate_ticks)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel(r'$\log_{10}(C)$')
    ax.legend(loc='best', frameon=False)
    ax.set_title(
        f'{session} ({cue_angle}$\\degree$), {num_cells} cells, '
        f'{trial_idx_pref.size} trials\n{label}'
    )
    save_figure_all_formats(
        fig,
        fig_dir / f'{session}_{pref_cue}_{filename_suffix}_lineplot.png',
        dpi=300,
    )
    plt.close(fig)


def plot_decoding_classifier_c(
    fig_dir,
    session,
    pref_cue,
    cue_angle,
    trial_idx_pref,
    bin_starts,
    classifier_c_repeats,
    classifier_c_null,
    plot_actual_trial_id,
    num_cells,
    grid_search_for_c,
):
    """Save empirical and optional null selected-C heatmaps and line plots."""
    trial_idx_pref = np.asarray(trial_idx_pref, dtype=np.int64)
    bin_starts = np.asarray(bin_starts)
    classifier_c_repeats = np.asarray(classifier_c_repeats, dtype=np.float64)
    if classifier_c_repeats.ndim != 3:
        raise ValueError(
            'classifier_c_repeats must have shape (trial, repeat, bin).'
        )
    if (
        classifier_c_repeats.shape[0] != trial_idx_pref.size
        or classifier_c_repeats.shape[2] != bin_starts.size
    ):
        raise ValueError(
            'classifier_c_repeats must match the trial and bin dimensions.'
        )
    empirical_mean_log10_c = mean_log10_classifier_c(
        classifier_c_repeats,
        axis=1,
    )
    plot_classifier_c_heatmap(
        fig_dir,
        session,
        pref_cue,
        cue_angle,
        trial_idx_pref,
        bin_starts,
        empirical_mean_log10_c,
        plot_actual_trial_id,
        num_cells,
        grid_search_for_c,
        label='Empirical selected $\\log_{10}(C)$ (mean over repeats)',
        filename_suffix='classifier_c',
    )
    plot_classifier_c_lineplot(
        fig_dir,
        session,
        pref_cue,
        cue_angle,
        trial_idx_pref,
        bin_starts,
        empirical_mean_log10_c,
        num_cells,
        grid_search_for_c,
        label='Empirical selected $\\log_{10}(C)$ (mean over repeats)',
        filename_suffix='classifier_c',
    )

    if classifier_c_null is None:
        return
    classifier_c_null = np.asarray(classifier_c_null, dtype=np.float64)
    if classifier_c_null.ndim != 3:
        raise ValueError('classifier_c_null must have shape (trial, bin, shuffle).')
    if (
        classifier_c_null.shape[0] != trial_idx_pref.size
        or classifier_c_null.shape[1] != bin_starts.size
    ):
        raise ValueError('classifier_c_null must match the trial and bin dimensions.')
    if classifier_c_null.shape[2] == 0:
        return
    null_mean_log10_c = mean_log10_classifier_c(classifier_c_null, axis=2)
    plot_classifier_c_heatmap(
        fig_dir,
        session,
        pref_cue,
        cue_angle,
        trial_idx_pref,
        bin_starts,
        null_mean_log10_c,
        plot_actual_trial_id,
        num_cells,
        grid_search_for_c,
        label='Null selected $\\log_{10}(C)$ (mean over shuffles)',
        filename_suffix='classifier_c_null',
    )
    plot_classifier_c_lineplot(
        fig_dir,
        session,
        pref_cue,
        cue_angle,
        trial_idx_pref,
        bin_starts,
        null_mean_log10_c,
        num_cells,
        grid_search_for_c,
        label='Null selected $\\log_{10}(C)$ (mean over shuffles)',
        filename_suffix='classifier_c_null',
    )

def total_unique_trials(partitions):
    """Count unique trials covered by a list of partitions."""
    if not partitions:
        return 0
    max_end = max(p['trial_end'] for p in partitions)
    covered = np.zeros(max_end, dtype=np.bool_)
    for p in partitions:
        covered[p['trial_start']:p['trial_end']] = True
    return int(covered.sum())


def load_session_list(path: Path) -> list[str]:
    """Load unique session IDs from non-empty, uncommented lines."""
    if not path.is_file():
        raise FileNotFoundError(f'Missing session list file: {path}')

    sessions = []
    seen = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            session = line.strip()
            if not session or session.startswith('#') or session in seen:
                continue
            sessions.append(session)
            seen.add(session)
    return sessions


def filter_sessions_by_list(good_sessions, known_sessions, requested_sessions):
    """Keep requested eligible sessions and report unknown/ineligible IDs."""
    known_session_ids = {str(session) for session in known_sessions}
    eligible_session_ids = {str(session) for session in good_sessions}
    requested_session_ids = set(requested_sessions)
    warnings = []

    for session in requested_sessions:
        if session not in known_session_ids:
            warnings.append(
                f'Session {session} is not present in cell_trial_selection.pkl; '
                'skipping.'
            )
        elif session not in eligible_session_ids:
            warnings.append(
                f'Session {session} is not eligible for decoding under the '
                'current thresholds; skipping.'
            )

    filtered_sessions = {
        session: info
        for session, info in good_sessions.items()
        if str(session) in requested_session_ids
    }
    return filtered_sessions, warnings


@dataclass
class Config:
    """CLI configuration for decoding confidence analysis."""
    n_jobs: int = 1 # number of parallel jobs for single-trial decoding
    par_verbose: int = 10 # joblib Parallel verbosity level
    seed: int = 42 # random seed for cue label balancing and shuffling
    data_dir: Path = Path('data/nature') # directory with {session}.mat files
    cache_dir: Path = Path('cache/run_001') # directory for cached results and figures
    loo_cell_selection: bool = False # unsupported; raises if enabled
    cells_used_for_decoder: CellsUsedForDecoder = CellsUsedForDecoder.PREFERRED # decoder feature-cell pool
    decoder_model: DecoderModel = DecoderModel.SVM # classifier used by the decoder
    svm_kernel: SVMKernel = SVMKernel.RBF # SVM kernel used by the decoder
    balance_decoder_training_trials: bool = True # balance preferred/opposite training trials
    classifier_c: float = 0.1 # fixed C used by either classifier when grid search is disabled
    grid_search_for_c: bool = False # select C independently for every decoder fit using grouped CV
    logistic_calibration_method: LogisticCalibrationMethod = LogisticCalibrationMethod.NONE # optional CV-based logistic probability calibration
    logistic_calibration_cv: int = 5 # requested inner CV folds for logistic calibration
    min_cell_per_group: int = 12 # a good partition has at least one group with this many cells
    min_trials_good_session: int = 320 # a good session has at least this many trials in good partitions
    t_decode_start: int = -200
    t_decode_end: int = 1400
    t_decode_window: int = 50
    t_decode_step: int = 10
    n_cue_preserved_trial_idx_shuffle: int = 0 # number of cue-label-preserved, cell-independent trial-index shuffles
    n_repeats_for_model_fit: int = 1 # number of model fits per trial/bin
    cue_preserved_train_set_shuffle: bool = False # shuffle each cell's within-cue training-trial indices after repeat 0
    train_delay_decoder_using_all_delay_time_bins: bool = False # pool training activity from bins starting in [500, 1400] for delay-bin tests
    n_decode_shuffle: int = 0 # number of label shuffles for null distribution of decoding confidence (0 to skip)
    plot_only: bool = False # if True, only generate plots from cached decoding results
    plot_actual_trial_id: bool = False # if True, y-axis shows actual trial ids instead of 1 to N
    session_list_file: Path | None = None # optional file with one session ID per uncommented line
    max_sessions_to_run: int | None = None # max number of good sessions to process (None to run all)

def main(config: Config):
    """Run decoding confidence analysis or generate plots from cached results.

    This analysis filters cached partition analyses to find sessions with enough
    cells in at least one preferred-cue group and sufficient stable trials. The
    decoder feature-cell pool is controlled by cells_used_for_decoder, and the
    classifier is controlled by decoder_model. SVM uses svm_kernel and
    probability=True; logistic regression can optionally use nested, CV-based
    sigmoid or isotonic probability calibration. Calibration folds contain only
    outer training trials and keep pooled delay-bin samples grouped by trial.
    Both classifiers use classifier_c as their regularization parameter unless
    grid_search_for_c selects C independently for every prepared decoder fit.
    The search uses five source-trial-grouped folds and balanced accuracy over
    C=(1, 0.1, 0.01), before optional logistic probability calibration.
    For each eligible session, it keeps correct trials from the preferred cue
    and its opposite, bins spike rates over sliding time windows, and decodes
    each preferred-cue trial in parallel to produce a time-by-trial confidence map.
    Each trial/bin model can be fit repeatedly with fresh balanced training
    trials; the repeat count is controlled by n_repeats_for_model_fit. The
    optional cue_preserved_train_set_shuffle leaves repeat zero unchanged and
    independently shuffles each cell's trial indices within cue labels for
    every later repeat after its training set has been selected. This does not
    affect label-shuffled null estimates. When
    train_delay_decoder_using_all_delay_time_bins is enabled, models tested at
    bins starting in [500, 1400] are trained using all such delay bins from the
    selected training trials; the held-out test trial remains entirely excluded.
    The cached confidence map is averaged over repeats, while per-repeat
    confidence, accuracy, and predictions are retained. Optional label shuffles
    generate a null distribution. When
    n_cue_preserved_trial_idx_shuffle is positive,
    trial indices are shuffled independently for each cell within each cue
    label before decoding; this mode forces one model-fit repeat and one label
    shuffle per cue-preserved shuffle. Training trials are balanced by default,
    controlled by balance_decoder_training_trials. Results are atomically
    checkpointed after each completed session, and per-session heatmaps and
    confidence line plots are saved; when plot-only mode is enabled, the cached
    results are used to regenerate both figures without recomputing decoding.
    """
    decoder_mode = CellsUsedForDecoder(config.cells_used_for_decoder)
    decoder_model = DecoderModel(config.decoder_model)
    svm_kernel = SVMKernel(config.svm_kernel)
    configured_logistic_calibration_method = LogisticCalibrationMethod(
        config.logistic_calibration_method
    )
    logistic_calibration_method = (
        configured_logistic_calibration_method
        if decoder_model is DecoderModel.LOGISTIC_REGRESSION
        else LogisticCalibrationMethod.NONE
    )
    logistic_calibration_cv = int(config.logistic_calibration_cv)
    if (
        logistic_calibration_method is not LogisticCalibrationMethod.NONE
        and logistic_calibration_cv < 2
    ):
        raise ValueError('logistic_calibration_cv must be at least 2.')
    classifier_c = float(config.classifier_c)
    if not np.isfinite(classifier_c) or classifier_c <= 0:
        raise ValueError('classifier_c must be finite and greater than zero.')
    if config.n_repeats_for_model_fit < 1:
        raise ValueError('n_repeats_for_model_fit must be at least 1.')
    if config.n_decode_shuffle < 0:
        raise ValueError('n_decode_shuffle must be non-negative.')
    if config.n_cue_preserved_trial_idx_shuffle < 0:
        raise ValueError(
            'n_cue_preserved_trial_idx_shuffle must be non-negative.'
        )
    if config.n_cue_preserved_trial_idx_shuffle > 0:
        if config.n_repeats_for_model_fit != 1 or config.n_decode_shuffle != 0:
            print(
                'Warning: n_cue_preserved_trial_idx_shuffle > 0 forces '
                'n_repeats_for_model_fit=1 and n_decode_shuffle=1; '
                'the supplied repeat/shuffle values will be ignored.'
            )
        n_repeats_for_model_fit = 1
        n_decode_shuffle = 1
    else:
        n_repeats_for_model_fit = int(config.n_repeats_for_model_fit)
        n_decode_shuffle = int(config.n_decode_shuffle)
    if config.loo_cell_selection:
        raise ValueError(
            'loo_cell_selection is not supported by decoding_confidence.py; '
            'set loo_cell_selection=False.'
        )

    cache_dir = config.cache_dir
    plot_actual_trial_id = config.plot_actual_trial_id

    cache_dir.mkdir(parents=True, exist_ok=True)
    selection_pkl = cache_dir / 'cell_trial_selection.pkl'
    decode_pkl = cache_dir / 'decoding_confidence.pkl'
    fig_dir = cache_dir / 'decoding_confidence'
    fig_dir.mkdir(parents=True, exist_ok=True)

    if config.plot_only:
        if not decode_pkl.exists():
            raise FileNotFoundError(f'Missing decoding file: {decode_pkl}')
        with open(decode_pkl, 'rb') as f:
            results = pickle.load(f)
        for res in results:
            trial_idx_pref = np.asarray(res.get('trial_idx', []), dtype=np.int64)
            bin_starts = np.asarray(res.get('time_bins', []))
            decode_confidence = np.asarray(res.get('decoding_confidence', []))
            decode_predicted_labels = np.asarray(
                res.get('decoding_predicted_labels', [])
            )
            decode_accuracy_repeats = np.asarray(
                res.get('decoding_accuracy_repeats', [])
            )
            decode_test_labels = np.asarray(res.get('decoding_test_labels', []))
            if decode_confidence.size == 0:
                continue
            if (
                decode_predicted_labels.size == 0
                or decode_accuracy_repeats.size == 0
                or decode_test_labels.size == 0
            ):
                raise ValueError(
                    f'Cached result for {res.get("session", "unknown_session")} is missing '
                    'decoding_predicted_labels, decoding_accuracy_repeats, or '
                    'decoding_test_labels; rerun decoding '
                    'without --plot-only to regenerate the cache.'
                )
            plot_decoding_heatmap(
                fig_dir,
                res.get('session', 'unknown_session'),
                res.get('cue', 0),
                res.get('cue_deg', 0),
                trial_idx_pref,
                bin_starts,
                decode_confidence,
                plot_actual_trial_id,
                int(res.get('num_cells', 0)),
            )
            plot_decoding_confidence_lineplot(
                fig_dir,
                res.get('session', 'unknown_session'),
                res.get('cue', 0),
                res.get('cue_deg', 0),
                trial_idx_pref,
                bin_starts,
                decode_confidence,
                decode_predicted_labels,
                decode_accuracy_repeats,
                decode_test_labels,
                int(res.get('num_cells', 0)),
            )
            cached_classifier_c_repeats = res.get(
                'decoding_classifier_c_repeats'
            )
            if cached_classifier_c_repeats is None:
                print(
                    f'Warning: cached result for '
                    f'{res.get("session", "unknown_session")} does not contain '
                    'decoder C values; skipping C plots. Rerun decoding '
                    'without --plot-only to create them.'
                )
            else:
                plot_decoding_classifier_c(
                    fig_dir,
                    res.get('session', 'unknown_session'),
                    res.get('cue', 0),
                    res.get('cue_deg', 0),
                    trial_idx_pref,
                    bin_starts,
                    cached_classifier_c_repeats,
                    res.get('decoding_classifier_c_null'),
                    plot_actual_trial_id,
                    int(res.get('num_cells', 0)),
                    bool(res.get('grid_search_for_c', False)),
                )
        return
    
    if not selection_pkl.exists():
        raise FileNotFoundError(f'Missing selection file: {selection_pkl}')
    with open(selection_pkl, 'rb') as f:
        selection_outs = pickle.load(f)

    requested_sessions = None
    if config.session_list_file is not None:
        requested_sessions = load_session_list(config.session_list_file)
        print(
            f'Loaded {len(requested_sessions)} requested session(s) from '
            f'{config.session_list_file}'
        )

    # Filter partitions to those with enough cells per group.
    good_partitions = []
    for out in selection_outs:
        if out.get('max_num_cells_per_group', 0) < config.min_cell_per_group:
            continue
        # When LOO decoding is off, ignore LOO selection entries.
        trial_holdout = out.get('trial_holdout')
        if not config.loo_cell_selection and trial_holdout is not None:
            continue
        good_partitions.append(out)
    partitions_by_session = {}
    for out in good_partitions:
        partitions_by_session.setdefault(out['session'], []).append(out)

    # Identify good sessions with enough trials in good partitions.
    # Prefer no-holdout partitions (w/o LOO) for trial coverage and preferred cue estimation.
    good_sessions = {}
    for session, parts in partitions_by_session.items():
        no_holdout_parts = [p for p in parts if p.get('trial_holdout') is None]
        parts_for_trials = no_holdout_parts if no_holdout_parts else parts
        # Count unique trials covered by good partitions
        covered_trial_count = total_unique_trials(parts_for_trials)
        if covered_trial_count < config.min_trials_good_session:
            continue
        partition_pref = []
        # Session-level preferred cue estimation
        pref_parts = no_holdout_parts if no_holdout_parts else parts
        for p in pref_parts:
            pref_cues = np.asarray(p['cell_properties']['mean_pref_test'])
            pref_cue = preferred_cue_from_cells(pref_cues)
            if pref_cue is not None:
                partition_pref.append(pref_cue)
        session_pref = preferred_cue_from_partitions(np.asarray(partition_pref))
        if session_pref is not None:
            good_sessions[session] = {
                'preferred_cue': session_pref,
                'partitions': parts,
                'no_holdout_partitions': no_holdout_parts,
            }

    if requested_sessions is not None:
        known_sessions = {
            out['session'] for out in selection_outs if 'session' in out
        }
        good_sessions, session_filter_warnings = filter_sessions_by_list(
            good_sessions,
            known_sessions,
            requested_sessions,
        )
        for warning in session_filter_warnings:
            print(f'Warning: {warning}')

    if not good_sessions:
        return

    results = []
    sessions = list(good_sessions.keys())
    if config.max_sessions_to_run is not None:
        sessions = sessions[:config.max_sessions_to_run]
    for idx_session, session in enumerate(sessions, start=1):
        session_info = good_sessions[session]
        print(f'Processing session {session} ({idx_session}/{len(sessions)})')
        data_file = config.data_dir / f'{session}.mat'
        if not data_file.exists():
            print(f'  Skipping: missing data file {data_file}')
            continue
        data = loadmat(data_file)
        spikes = np.asarray(data['spks'])
        cue_labels = np.asarray(data['cueAngIdx']).flatten().astype(np.int64)
        trial_boo_correct = np.asarray(data['isCorr']).flatten().astype(np.bool_)
        t = np.asarray(data['tc']).flatten()

        pref_cue = session_info['preferred_cue']
        opposite_cue = get_opposite_cue(pref_cue)

        no_holdout_parts = session_info.get('no_holdout_partitions', [])
        if not no_holdout_parts:
            no_holdout_parts = [
                p for p in session_info['partitions'] if p.get('trial_holdout') is None
            ]

        if decoder_mode is CellsUsedForDecoder.ALL:
            decoder_cell_set = set(range(spikes.shape[2]))
        else:
            decoder_cell_set: set[int] | None = None
            for partition in no_holdout_parts:
                partition_cell_set = decoder_cells_for_partition(
                    partition,
                    decoder_mode,
                    pref_cue,
                    opposite_cue,
                    spikes.shape[2],
                )
                if decoder_cell_set is None:
                    decoder_cell_set = partition_cell_set
                else:
                    decoder_cell_set &= partition_cell_set
            if decoder_cell_set is None:
                decoder_cell_set = set()

        if not decoder_cell_set:
            print(f'  Skipping: no cells available for decoder mode {decoder_mode.value}')
            continue

        # Restrict to correct trials for preferred vs. opposite cue decoding.
        trial_mask = trial_boo_correct & ((cue_labels == pref_cue) | (cue_labels == opposite_cue))
        selected_trial_idx = np.nonzero(trial_mask)[0]
        if selected_trial_idx.size == 0:
            print('  Skipping: no trials for preferred/opposite cue')
            continue
        labels_sel = (cue_labels[selected_trial_idx] == pref_cue).astype(np.int64)
        test_sel_indices = np.nonzero(labels_sel == 1)[0]
        if test_sel_indices.size == 0:
            print('  Skipping: no preferred-cue trials to test')
            continue
        print(
            f'  Cue {pref_cue} vs {opposite_cue}, '
            f'~{len(decoder_cell_set)} cells ({decoder_mode.value}), '
            f'decoder={decoder_model.value}, '
            f'kernel={svm_kernel.value if decoder_model is DecoderModel.SVM else "ignored"}, '
            f'logistic_calibration={logistic_calibration_method.value}, '
            f'logistic_calibration_cv='
            f'{logistic_calibration_cv if logistic_calibration_method is not LogisticCalibrationMethod.NONE else "ignored"}, '
            f'balance_trials={config.balance_decoder_training_trials}, '
            f'model_fit_repeats={n_repeats_for_model_fit}, '
            f'cue_preserved_train_set_shuffle='
            f'{config.cue_preserved_train_set_shuffle}, '
            f'train_delay_decoder_using_all_delay_time_bins='
            f'{config.train_delay_decoder_using_all_delay_time_bins}, '
            f'decode_shuffles={n_decode_shuffle}, '
            f'cue_preserved_trial_idx_shuffles='
            f'{config.n_cue_preserved_trial_idx_shuffle}, '
            f'grid_search_for_c={config.grid_search_for_c}, '
            f'C='
            f'{CLASSIFIER_C_GRID if config.grid_search_for_c else classifier_c}, '
            f'{selected_trial_idx.size} trials '
            f'({test_sel_indices.size} test trials)'
        )

        bin_starts = np.arange(config.t_decode_start, config.t_decode_end + 1, config.t_decode_step)
        cell_idx = np.asarray(sorted(decoder_cell_set), dtype=np.int64)
        spikes_sel = spikes[selected_trial_idx][:, :, cell_idx]

        cue_shuffle_seed_sequences = None
        if config.n_cue_preserved_trial_idx_shuffle > 0:
            cue_shuffle_seed_sequences = np.random.SeedSequence(config.seed).spawn(
                config.n_cue_preserved_trial_idx_shuffle
            )

        def decode_trial(idx_test: int, binned_rates=None):
            trial_abs = int(selected_trial_idx[idx_test])
            if binned_rates is None:
                binned_rates = compute_binned_rates(
                    spikes_sel,
                    t,
                    bin_starts,
                    config.t_decode_window,
                )

            (
                repeat_conf,
                repeat_predicted_labels,
                repeat_accuracy,
                null_conf,
                effective_calibration_cv_folds,
                repeat_classifier_c,
                null_classifier_c,
            ) = decode_one_trial(
                idx_test,
                binned_rates,
                labels_sel,
                config.seed,
                config.balance_decoder_training_trials,
                classifier_c,
                decoder_model,
                svm_kernel,
                n_repeats_for_model_fit,
                n_decode_shuffle,
                grid_search_for_c=config.grid_search_for_c,
                cue_preserved_train_set_shuffle=(
                    config.cue_preserved_train_set_shuffle
                ),
                bin_starts=bin_starts,
                train_delay_decoder_using_all_delay_time_bins=(
                    config.train_delay_decoder_using_all_delay_time_bins
                ),
                logistic_calibration_method=logistic_calibration_method,
                logistic_calibration_cv=logistic_calibration_cv,
            )
            return (
                'ok',
                trial_abs,
                labels_sel[idx_test],
                repeat_conf,
                repeat_predicted_labels,
                repeat_accuracy,
                null_conf,
                effective_calibration_cv_folds,
                repeat_classifier_c,
                null_classifier_c,
                int(cell_idx.size),
            )

        if cue_shuffle_seed_sequences is not None:
            decoded_by_cue_shuffle = []
            for cue_shuffle_seed_sequence in cue_shuffle_seed_sequences:
                cue_shuffle_rng = np.random.default_rng(cue_shuffle_seed_sequence)
                shuffled_spikes_sel = shuffle_trial_idx_within_labels(
                    spikes_sel,
                    labels_sel,
                    cue_shuffle_rng,
                )
                binned_rates = compute_binned_rates(
                    shuffled_spikes_sel,
                    t,
                    bin_starts,
                    config.t_decode_window,
                )
                decoded_by_cue_shuffle.append(
                    Parallel(n_jobs=config.n_jobs, verbose=config.par_verbose)(
                        delayed(decode_trial)(idx, binned_rates)
                        for idx in test_sel_indices
                    )
                )
        else:
            decoded_by_cue_shuffle = [
                Parallel(n_jobs=config.n_jobs, verbose=config.par_verbose)(
                    delayed(decode_trial)(idx) for idx in test_sel_indices
                )
            ]

        def collect_decoded_batch(decoded):
            conf_list = []
            null_list = []
            trial_idx_pref = []
            predicted_labels_list = []
            accuracy_list = []
            classifier_c_repeats_list = []
            classifier_c_null_list = []
            test_labels_pref = []
            num_cells_per_trial: list[int] = []
            effective_calibration_cv_folds: set[int] = set()
            for (
                status,
                trial_abs,
                test_label,
                repeat_conf,
                repeat_predicted_labels,
                repeat_accuracy,
                null_conf,
                trial_effective_calibration_cv_folds,
                repeat_classifier_c,
                null_classifier_c,
                n_cell,
            ) in decoded:
                if status == 'warn_no_partitions':
                    print(f'  Skipping test trial {trial_abs}: no matching LOO partition and all no-holdout partitions overlap this trial')
                    continue
                if status == 'warn_no_cells':
                    print(f'  Skipping test trial {trial_abs}: no preferred-cue cells from applicable partitions')
                    continue
                conf_list.append(repeat_conf)
                predicted_labels_list.append(repeat_predicted_labels)
                accuracy_list.append(repeat_accuracy)
                classifier_c_repeats_list.append(repeat_classifier_c)
                test_labels_pref.append(test_label)
                if null_conf is not None:
                    null_list.append(null_conf)
                if null_classifier_c is not None:
                    classifier_c_null_list.append(null_classifier_c)
                trial_idx_pref.append(trial_abs)
                num_cells_per_trial.append(n_cell)
                effective_calibration_cv_folds.update(
                    trial_effective_calibration_cv_folds
                )

            if not conf_list:
                return None
            return {
                'confidence': np.stack(conf_list, axis=0),
                'predicted_labels': np.stack(predicted_labels_list, axis=0),
                'accuracy_per_trial': np.stack(accuracy_list, axis=0),
                'classifier_c_repeats': np.stack(
                    classifier_c_repeats_list,
                    axis=0,
                ),
                'null': np.stack(null_list, axis=0) if null_list else None,
                'classifier_c_null': (
                    np.stack(classifier_c_null_list, axis=0)
                    if classifier_c_null_list
                    else None
                ),
                'trial_idx': np.asarray(trial_idx_pref, dtype=np.int64),
                'test_labels': np.asarray(test_labels_pref, dtype=np.int8),
                'num_cells_per_trial': num_cells_per_trial,
                'effective_calibration_cv_folds': tuple(
                    sorted(effective_calibration_cv_folds)
                ),
            }

        decoded_batches = [
            collect_decoded_batch(decoded) for decoded in decoded_by_cue_shuffle
        ]
        if any(batch is None for batch in decoded_batches):
            print('  Skipping: no decodable preferred-cue trials with available cells')
            continue

        first_batch = decoded_batches[0]
        trial_idx_pref = first_batch['trial_idx']
        decode_test_labels = first_batch['test_labels']
        num_cells_per_trial = first_batch['num_cells_per_trial']
        effective_calibration_cv_folds = sorted({
            fold_count
            for batch in decoded_batches
            for fold_count in batch['effective_calibration_cv_folds']
        })
        for batch in decoded_batches[1:]:
            if not np.array_equal(batch['trial_idx'], trial_idx_pref):
                raise RuntimeError(
                    'Cue-preserved shuffles produced different test-trial rows.'
                )
            if not np.array_equal(batch['test_labels'], decode_test_labels):
                raise RuntimeError(
                    'Cue-preserved shuffles produced different test labels.'
                )

        # Treat each cue-preserved shuffle as one repeat so existing consumers
        # can continue to use the repeat-shaped arrays unchanged.
        decode_confidence_repeats = np.concatenate(
            [batch['confidence'] for batch in decoded_batches],
            axis=1,
        )
        decode_classifier_c_repeats = np.concatenate(
            [batch['classifier_c_repeats'] for batch in decoded_batches],
            axis=1,
        )
        decode_predicted_labels = np.concatenate(
            [batch['predicted_labels'] for batch in decoded_batches],
            axis=1,
        )
        decode_accuracy_per_trial_repeats = np.concatenate(
            [batch['accuracy_per_trial'] for batch in decoded_batches],
            axis=1,
        )
        decode_confidence = np.nanmean(decode_confidence_repeats, axis=1)
        decode_accuracy_repeats = np.nanmean(
            decode_accuracy_per_trial_repeats, axis=0
        )
        decode_accuracy = np.nanmean(decode_accuracy_repeats, axis=0)
        decode_confidence_null = None
        decode_classifier_c_null = None
        if n_decode_shuffle > 0:
            null_batches = [batch['null'] for batch in decoded_batches]
            classifier_c_null_batches = [
                batch['classifier_c_null'] for batch in decoded_batches
            ]
            if all(null_batch is not None for null_batch in null_batches):
                # Each cue-preserved shuffle contributes its null samples to
                # the existing shuffle axis.
                decode_confidence_null = np.concatenate(null_batches, axis=2)
                if not all(
                    null_classifier_c is not None
                    for null_classifier_c in classifier_c_null_batches
                ):
                    raise RuntimeError(
                        'Missing classifier C values for decoded null confidence.'
                    )
                decode_classifier_c_null = np.concatenate(
                    classifier_c_null_batches,
                    axis=2,
                )
            else:
                decode_confidence_null = np.empty(
                    (0, len(bin_starts), n_decode_shuffle),
                    dtype=np.float32,
                )
                decode_classifier_c_null = np.empty(
                    (0, len(bin_starts), n_decode_shuffle),
                    dtype=np.float64,
                )

        cue_angle = int(cue_to_deg(pref_cue))
        session_result = {
            'session': session,
            'cue': int(pref_cue),
            'cue_deg': cue_angle,
            'trial_idx': np.asarray(trial_idx_pref, dtype=np.int64),
            'time_bins': bin_starts,
            'decoding_confidence': decode_confidence,
            'decoding_confidence_repeats': decode_confidence_repeats,
            'decoding_classifier_c_repeats': decode_classifier_c_repeats,
            'decoding_predicted_labels': decode_predicted_labels,
            'decoding_accuracy': decode_accuracy,
            'decoding_accuracy_repeats': decode_accuracy_repeats,
            'decoding_accuracy_per_trial_repeats': decode_accuracy_per_trial_repeats,
            'n_repeats_for_model_fit': int(n_repeats_for_model_fit),
            'cue_preserved_train_set_shuffle': bool(
                config.cue_preserved_train_set_shuffle
            ),
            'train_delay_decoder_using_all_delay_time_bins': bool(
                config.train_delay_decoder_using_all_delay_time_bins
            ),
            'n_decode_shuffle': int(n_decode_shuffle),
            'grid_search_for_c': bool(config.grid_search_for_c),
            'configured_classifier_c': classifier_c,
            'classifier_c_grid': CLASSIFIER_C_GRID,
            'classifier_c_grid_search_cv_folds': CLASSIFIER_C_GRID_SEARCH_CV,
            'classifier_c_grid_search_scoring': (
                CLASSIFIER_C_GRID_SEARCH_SCORING
            ),
            'n_cue_preserved_trial_idx_shuffle': int(
                config.n_cue_preserved_trial_idx_shuffle
            ),
            'logistic_calibration_method': logistic_calibration_method.value,
            'logistic_calibration_requested_cv_folds': logistic_calibration_cv,
            'logistic_calibration_effective_cv_folds': (
                effective_calibration_cv_folds
            ),
            'logistic_calibration_grouped_by_trial': bool(
                logistic_calibration_method is not LogisticCalibrationMethod.NONE
            ),
            'decoding_test_labels': decode_test_labels,
            'decoding_confidence_null': decode_confidence_null,
            'decoding_classifier_c_null': decode_classifier_c_null,
            'num_cells': int(max(num_cells_per_trial)),
            'num_cells_per_trial': num_cells_per_trial,
            'num_trials': int(len(trial_idx_pref)),
        }
        results.append(session_result)
        save_pickle_atomic(results, decode_pkl)
        print(
            f'  Cached {len(results)} completed session(s) to {decode_pkl}'
        )

        plot_decoding_heatmap(
            fig_dir,
            session,
            pref_cue,
            cue_angle,
            np.asarray(trial_idx_pref, dtype=np.int64),
            bin_starts,
            decode_confidence,
            plot_actual_trial_id,
            int(max(num_cells_per_trial)),
        )
        plot_decoding_confidence_lineplot(
            fig_dir,
            session,
            pref_cue,
            cue_angle,
            np.asarray(trial_idx_pref, dtype=np.int64),
            bin_starts,
            decode_confidence,
            decode_predicted_labels,
            decode_accuracy_repeats,
            decode_test_labels,
            int(max(num_cells_per_trial)),
        )
        plot_decoding_classifier_c(
            fig_dir,
            session,
            pref_cue,
            cue_angle,
            np.asarray(trial_idx_pref, dtype=np.int64),
            bin_starts,
            decode_classifier_c_repeats,
            decode_classifier_c_null,
            plot_actual_trial_id,
            int(max(num_cells_per_trial)),
            config.grid_search_for_c,
        )

if __name__ == "__main__":
    config = tyro.cli(Config)
    main(config)
