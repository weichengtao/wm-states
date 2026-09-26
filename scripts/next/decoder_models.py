"""Decoder estimators and source-trial-grouped inner cross-validation."""
from enum import Enum
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from scripts.next.screening_metadata import ScreeningMetadata, validate_cue

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

def preferred_cue_from_cells(pref_cues):
    """Pick the most frequent preferred cue across cells."""
    cues, counts = np.unique(pref_cues, return_counts=True)
    if cues.size == 0:
        return None
    best = np.argmax(counts)
    return int(cues[best])

def decoder_cells_for_session(
    selection,
    mode: CellsUsedForDecoder,
    preferred_cue: int,
    opposite_cue: int,
    num_cells_total: int,
) -> set[int]:
    """Return the cell indices available to the decoder in one selection."""
    mode = CellsUsedForDecoder(mode)
    metadata = ScreeningMetadata(selection, num_cells_total=num_cells_total)
    if mode is CellsUsedForDecoder.ALL:
        return set(range(num_cells_total))

    if mode is CellsUsedForDecoder.STATIONARY:
        return set(metadata.stationary_cell_ids.tolist())

    if mode is CellsUsedForDecoder.PASSED_PRESENCE_RATIO:
        return set(metadata.presence_passed_cell_ids.tolist())

    selected_cell_ids = metadata.selected_cell_ids
    if mode is CellsUsedForDecoder.SELECTIVE:
        return set(selected_cell_ids.tolist())

    preferred_cue = validate_cue(preferred_cue)
    preferred_cues = metadata.preferred_cues
    if mode is CellsUsedForDecoder.PREFERRED:
        keep = preferred_cues == preferred_cue
    elif mode is CellsUsedForDecoder.PREFERRED_AND_OPPOSITE:
        opposite_cue = validate_cue(opposite_cue, 'opposite_cue')
        keep = (preferred_cues == preferred_cue) | (preferred_cues == opposite_cue)
    else:
        raise ValueError(f'Unsupported decoder cell mode: {mode}')
    return set(selected_cell_ids[keep].tolist())

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
    cv_splits=None,
) -> float:
    """Grouped five-fold C search, scaling each fold once for all candidates.

    The scaler is fitted only on the inner training fold. Candidate order and
    first-best tie breaking match GridSearchCV's mean balanced accuracy policy.
    """
    if cv_splits is None:
        try:
            cv_splits, _ = make_grouped_stratified_cv_splits(
                y_train, sample_groups, CLASSIFIER_C_GRID_SEARCH_CV, seed,
                purpose='Classifier C grid search', allow_fold_reduction=False)
        except ValueError as exc:
            raise ValueError(f'{fit_context}: {exc}') from exc
    scores = np.empty((len(CLASSIFIER_C_GRID), len(cv_splits)), dtype=float)
    for fold, (training, validation) in enumerate(cv_splits):
        scaler = StandardScaler()
        x_fit = scaler.fit_transform(X_train[training])
        x_score = scaler.transform(X_train[validation])
        y_fit = y_train[training]
        y_score = y_train[validation]
        for candidate, c in enumerate(CLASSIFIER_C_GRID):
            classifier = create_base_decoder(c, decoder_model, svm_kernel, seed,
                                             svm_probability=False).named_steps['classifier']
            classifier.fit(x_fit, y_fit)
            prediction = classifier.predict(x_score)
            scores[candidate, fold] = balanced_accuracy_score(y_score, prediction)
    return float(CLASSIFIER_C_GRID[int(np.argmax(np.average(scores, axis=1)))])
