"""Shared leakage-safe trial-holdout cross-validation for MixedLM analyses."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scripts.compare_mixed_effect_models import (
        OUTCOME,
        SESSION,
        ModelSpec,
        _fit_model,
    )
    from scripts.prepare_data_for_mixedlm import (
        CV_CACHE_SCHEMA_VERSION,
        GROUP_NAMES,
        PERIODS,
        _causal_history_ema,
        _group_features,
    )
    from scripts.mixedlm_outcomes import ALL_OUTCOMES
except ModuleNotFoundError:
    from compare_mixed_effect_models import OUTCOME, SESSION, ModelSpec, _fit_model
    from prepare_data_for_mixedlm import (
        CV_CACHE_SCHEMA_VERSION,
        GROUP_NAMES,
        PERIODS,
        _causal_history_ema,
        _group_features,
    )
    from mixedlm_outcomes import ALL_OUTCOMES


@dataclass(frozen=True)
class CVModelRequest:
    """One model fit, with the active threshold and output annotations it uses."""

    spec: ModelSpec
    active_threshold: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrialHoldoutConfig:
    n_shuffles: int = 50
    holdout_fraction: float = 0.2
    seed: int = 42
    history_alpha: float = 0.2
    max_iterations: int = 1000
    figure_dpi: int = 200
    prediction_sample_per_model: int = 1000


def load_cv_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing cross-validation feature cache: {path}. "
            "Run prepare_data_for_mixedlm.py first."
        )
    with path.open("rb") as handle:
        cache = pickle.load(handle)
    if (
        not isinstance(cache, dict)
        or cache.get("schema_version") != CV_CACHE_SCHEMA_VERSION
    ):
        raise ValueError(
            f"Unsupported cross-validation cache format in {path}; regenerate it "
            "with prepare_data_for_mixedlm.py."
        )
    if not isinstance(cache.get("sessions"), list) or not cache["sessions"]:
        raise ValueError(f"Cross-validation cache has no sessions: {path}")
    return cache


def _validate_config(config: TrialHoldoutConfig) -> None:
    if config.n_shuffles < 1:
        raise ValueError("cv_shuffles must be positive.")
    if not np.isfinite(config.holdout_fraction) or not (
        0 < config.holdout_fraction < 1
    ):
        raise ValueError("cv_holdout_fraction must be in (0, 1).")
    if not np.isfinite(config.history_alpha) or not 0 < config.history_alpha <= 1:
        raise ValueError("history_alpha must be in (0, 1].")
    if config.max_iterations < 1:
        raise ValueError("max_iterations must be positive.")
    if config.figure_dpi < 1:
        raise ValueError("figure_dpi must be positive.")
    if config.prediction_sample_per_model < 0:
        raise ValueError("prediction_sample_per_model must be nonnegative.")


def _make_splits(
    sessions: list[dict[str, Any]],
    n_shuffles: int,
    holdout_fraction: float,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    splits: list[dict[str, Any]] = []
    for repeat in range(n_shuffles):
        test_by_session: dict[str, np.ndarray] = {}
        for session_data in sessions:
            eligible = np.asarray(session_data["trial_ids"], dtype=np.int64)[1:]
            if eligible.size < 2:
                raise ValueError(
                    f"Session {session_data['session']} has fewer than two "
                    "model-eligible trials."
                )
            n_test = min(
                max(int(np.ceil(holdout_fraction * eligible.size)), 1),
                eligible.size - 1,
            )
            test_by_session[str(session_data["session"])] = np.sort(
                rng.choice(eligible, size=n_test, replace=False)
            )
        splits.append(
            {"repeat": repeat, "test_trial_ids_by_session": test_by_session}
        )
    return splits


def _select_splits(
    cache: dict[str, Any], config: TrialHoldoutConfig
) -> tuple[list[dict[str, Any]], str]:
    cached_splits = cache.get("splits")
    cache_matches = (
        isinstance(cached_splits, list)
        and len(cached_splits) >= config.n_shuffles
        and int(cache.get("cv_seed", -1)) == config.seed
        and np.isclose(
            float(cache.get("cv_holdout_fraction", np.nan)),
            config.holdout_fraction,
        )
    )
    if cache_matches:
        return cached_splits[: config.n_shuffles], "prepared cache"
    return (
        _make_splits(
            cache["sessions"],
            config.n_shuffles,
            config.holdout_fraction,
            config.seed,
        ),
        "generated from raw cache",
    )


def _normalize_from_training(
    raw_rates: np.ndarray, train_positions: np.ndarray
) -> np.ndarray:
    """Apply cell-wise z scores whose moments use training model rows only."""
    raw_rates = np.asarray(raw_rates, dtype=float)
    normalized = np.zeros_like(raw_rates, dtype=float)
    if raw_rates.shape[1] == 0:
        return normalized
    training = raw_rates[train_positions]
    means = np.mean(training, axis=0)
    stds = np.std(training, axis=0, ddof=0)
    usable = (
        np.isfinite(means)
        & np.isfinite(stds)
        & (means != 0)
        & (stds != 0)
    )
    normalized[:, usable] = (raw_rates[:, usable] - means[usable]) / stds[usable]
    return normalized


def build_fold_frame(
    cache: dict[str, Any],
    split: dict[str, Any],
    active_threshold: float,
    history_alpha: float,
) -> pd.DataFrame:
    """Build train-normalized features for one split and one active cutoff."""
    test_by_session = split["test_trial_ids_by_session"]
    rows: list[dict[str, Any]] = []
    for session_data in cache["sessions"]:
        session = str(session_data["session"])
        trial_ids = np.asarray(session_data["trial_ids"], dtype=np.int64)
        outcomes = {
            outcome.column: np.asarray(
                session_data[outcome.column], dtype=float
            ).ravel()
            for outcome in ALL_OUTCOMES
        }
        if any(values.shape != trial_ids.shape for values in outcomes.values()):
            raise ValueError(f"Outcome arrays do not align for session {session}.")
        test_ids = np.asarray(test_by_session[session], dtype=np.int64)
        is_test = np.isin(trial_ids, test_ids)
        is_model_row = np.arange(trial_ids.size) > 0
        train_positions = np.flatnonzero(is_model_row & ~is_test)
        if train_positions.size == 0 or not np.any(is_model_row & is_test):
            raise ValueError(f"Split leaves no train or test rows in {session}.")

        features: dict[str, np.ndarray] = {}
        raw_by_period = session_data["raw_firing_rates_hz"]
        for period_name in PERIODS:
            for group_name in GROUP_NAMES:
                normalized = _normalize_from_training(
                    raw_by_period[period_name][group_name], train_positions
                )
                mean_activity, active_fraction = _group_features(
                    normalized, active_threshold
                )
                mean_column = (
                    f"{period_name}_mean_normalized_activity_{group_name}"
                )
                fraction_column = f"{period_name}_active_fraction_{group_name}"
                features[mean_column] = mean_activity
                features[fraction_column] = active_fraction
                features[f"history_ema_{mean_column}"] = _causal_history_ema(
                    mean_activity, history_alpha
                )
                features[f"history_ema_{fraction_column}"] = _causal_history_ema(
                    active_fraction, history_alpha
                )

        for position in range(1, trial_ids.size):
            row: dict[str, Any] = {
                SESSION: session,
                "trial_id": int(trial_ids[position]),
                "preferred_cue": int(session_data["preferred_cue"]),
                "cv_is_test": bool(is_test[position]),
                **session_data["cell_counts"],
                **{
                    column: float(values[position])
                    for column, values in outcomes.items()
                },
            }
            row.update(
                {name: float(values[position]) for name, values in features.items()}
            )
            rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty or not np.all(
        np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy(dtype=float))
    ):
        raise ValueError("Fold feature frame is empty or contains non-finite values.")
    return frame


def _test_predictions(result: Any, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    # Let statsmodels rebuild the fixed-effect design matrix from the fitted
    # formula metadata.  Accessing model.data.design_info directly broke in
    # statsmodels 0.15, where that metadata moved to model_spec.
    fixed = np.asarray(result.predict(test), dtype=float)
    random_effects = result.random_effects
    offsets = np.empty(len(test), dtype=float)
    for index, session in enumerate(test[SESSION].astype(str)):
        effect = random_effects[session]
        offsets[index] = float(np.asarray(effect, dtype=float).ravel()[0])
    return fixed, fixed + offsets


def _pearson(observed: np.ndarray, predicted: np.ndarray) -> float:
    if np.std(observed) == 0 or np.std(predicted) == 0:
        return np.nan
    return float(np.corrcoef(observed, predicted)[0, 1])


def _metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    sessions: np.ndarray,
) -> dict[str, float]:
    residual = observed - predicted
    sse = float(np.sum(residual**2))
    sst = float(np.sum((observed - np.mean(observed)) ** 2))
    centered_observed = np.empty_like(observed)
    centered_predicted = np.empty_like(predicted)
    for session in np.unique(sessions):
        mask = sessions == session
        centered_observed[mask] = observed[mask] - np.mean(observed[mask])
        centered_predicted[mask] = predicted[mask] - np.mean(predicted[mask])
    centered_sst = float(np.sum(centered_observed**2))
    centered_sse = float(np.sum((centered_observed - centered_predicted) ** 2))
    return {
        "rmse_ms": float(np.sqrt(np.mean(residual**2))),
        "mae_ms": float(np.mean(np.abs(residual))),
        "r2": 1.0 - sse / sst if sst > 0 else np.nan,
        "session_centered_r2": (
            1.0 - centered_sse / centered_sst if centered_sst > 0 else np.nan
        ),
        "pearson_r": _pearson(observed, predicted),
    }


def _aggregate_metrics(repeat_metrics: pd.DataFrame) -> pd.DataFrame:
    successful = repeat_metrics[repeat_metrics["fit_success"]].copy()
    metric_columns = [
        column
        for column in successful.columns
        if column.startswith(("fixed_", "conditional_"))
        and pd.api.types.is_numeric_dtype(successful[column])
    ]
    static_columns = [
        "model",
        "outcome",
        "description",
        "formula",
        "active_threshold",
        *sorted(
            column
            for column in repeat_metrics.columns
            if column.startswith("meta__")
        ),
    ]
    rows: list[dict[str, Any]] = []
    for model, all_model_rows in repeat_metrics.groupby("model", sort=False):
        model_rows = successful[successful["model"] == model]
        first = all_model_rows.iloc[0]
        row = {column: first.get(column, np.nan) for column in static_columns}
        row.update(
            {
                "n_shuffles_requested": int(len(all_model_rows)),
                "n_successful_fits": int(len(model_rows)),
                "n_failed_fits": int(len(all_model_rows) - len(model_rows)),
            }
        )
        for column in metric_columns:
            values = model_rows[column].dropna().to_numpy(dtype=float)
            if values.size == 0:
                for suffix in ("mean", "std", "median", "q025", "q975"):
                    row[f"{column}_{suffix}"] = np.nan
                continue
            row[f"{column}_mean"] = float(np.mean(values))
            row[f"{column}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            row[f"{column}_median"] = float(np.median(values))
            row[f"{column}_q025"] = float(np.quantile(values, 0.025))
            row[f"{column}_q975"] = float(np.quantile(values, 0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_overview(
    summary: pd.DataFrame,
    output_dir: Path,
    figure_dpi: int,
    outcome: str,
) -> Path:
    successful = summary[summary["n_successful_fits"] > 0].sort_values(
        "conditional_rmse_ms_mean"
    )
    shown = successful.head(30).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(15, max(6, 0.3 * len(shown))), layout="constrained")
    y = np.arange(len(shown))
    for ax, metric, label in (
        (axes[0], "conditional_rmse_ms", "Held-out conditional RMSE (ms)"),
        (axes[1], "conditional_r2", "Held-out conditional R²"),
    ):
        means = shown[f"{metric}_mean"].to_numpy(dtype=float)
        stds = shown[f"{metric}_std"].to_numpy(dtype=float)
        ax.errorbar(means, y, xerr=stds, fmt="o", capsize=2)
        ax.set_yticks(y, shown["model"] if ax is axes[0] else [])
        ax.set_xlabel(label)
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle(
        f"{outcome.replace('_', ' ')}\n"
        "Top models by mean held-out conditional RMSE; bars show ±1 SD"
    )
    path = output_dir / "cv_model_performance.png"
    fig.savefig(path, dpi=figure_dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_prediction_sample(
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    figure_dpi: int,
    outcome: str,
) -> Path | None:
    if predictions.empty:
        return None
    best_models = summary[summary["n_successful_fits"] > 0].nsmallest(
        4, "conditional_rmse_ms_mean"
    )["model"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), layout="constrained")
    for ax, model in zip(axes.ravel(), best_models):
        rows = predictions[predictions["model"] == model]
        x = rows["conditional_prediction_ms"].to_numpy(dtype=float)
        y = rows[outcome].to_numpy(dtype=float)
        ax.scatter(x, y, s=9, alpha=0.2, edgecolors="none")
        limits = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(limits, limits, "k--", linewidth=1)
        ax.set_title(model)
        ax.set_xlabel("Held-out conditional prediction (ms)")
        ax.set_ylabel(f"Observed {outcome.replace('_', ' ')}")
    path = output_dir / "cv_observed_vs_predicted_sample.png"
    fig.savefig(path, dpi=figure_dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def run_trial_holdout_cv(
    cache_path: Path,
    requests: list[CVModelRequest],
    output_dir: Path,
    config: TrialHoldoutConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit all requested models across repeated within-session trial holdouts."""
    _validate_config(config)
    if not requests:
        raise ValueError("At least one CV model request is required.")
    names = [request.spec.name for request in requests]
    if len(names) != len(set(names)):
        raise ValueError("CV model names must be unique.")
    outcomes = {request.spec.outcome for request in requests}
    if len(outcomes) != 1:
        raise ValueError("All CV model requests must use the same outcome.")
    outcome = outcomes.pop()
    cache = load_cv_cache(cache_path)
    splits, split_source = _select_splits(cache, config)
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    log_dir = output_dir / "logs"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    requests_by_threshold: dict[float, list[CVModelRequest]] = {}
    for request in requests:
        if not np.isfinite(request.active_threshold):
            raise ValueError(f"Non-finite active threshold for {request.spec.name}.")
        requests_by_threshold.setdefault(request.active_threshold, []).append(request)

    metric_rows: list[dict[str, Any]] = []
    prediction_samples: dict[str, list[dict[str, Any]]] = {
        name: [] for name in names
    }
    prediction_counts = {name: 0 for name in names}
    prediction_rngs = {
        name: np.random.default_rng(
            np.random.SeedSequence([config.seed, model_index, 0xC0DE])
        )
        for model_index, name in enumerate(names)
    }
    log_path = log_dir / "cross_validation.log"
    with log_path.open("w") as log:
        log.write("Repeated within-session trial-holdout MixedLM comparison\n")
        log.write("=" * 88 + "\n")
        log.write(f"Raw feature cache: {cache_path}\n")
        log.write(f"Outcome: {outcome}\n")
        log.write(f"Shuffles: {config.n_shuffles}\n")
        log.write(f"Holdout fraction: {config.holdout_fraction}\n")
        log.write(f"Seed: {config.seed}\n")
        log.write(f"Split source: {split_source}\n")
        log.write("REML: False; random effects: session intercept\n")
        log.write(
            "Cell-wise means/SDs are estimated from training model rows in each "
            "fold. Fixed predictions omit random effects; conditional predictions "
            "add each session BLUP estimated from its training rows.\n\n"
        )
        log.write(
            "Plotting predictions are uniformly reservoir-sampled per model "
            "across every held-out row from all repeats.\n\n"
        )

        for repeat, split in enumerate(splits):
            print(
                f"CV shuffle {repeat + 1}/{config.n_shuffles} "
                f"({len(requests)} models)"
            )
            for threshold, threshold_requests in requests_by_threshold.items():
                fold = build_fold_frame(
                    cache, split, threshold, config.history_alpha
                )
                train = fold[~fold["cv_is_test"]].copy()
                test = fold[fold["cv_is_test"]].copy()
                for request in threshold_requests:
                    spec = request.spec
                    base_row: dict[str, Any] = {
                        "repeat": repeat,
                        "model": spec.name,
                        "outcome": spec.outcome,
                        "description": spec.description,
                        "formula": spec.formula,
                        "parent_model": spec.parent or "",
                        "active_threshold": threshold,
                        "n_train": len(train),
                        "n_test": len(test),
                        "n_train_sessions": train[SESSION].nunique(),
                        "n_test_sessions": test[SESSION].nunique(),
                        **{
                            f"meta__{key}": value
                            for key, value in request.metadata.items()
                        },
                    }
                    try:
                        result, warning_messages = _fit_model(
                            train, spec, config.max_iterations
                        )
                        fixed, conditional = _test_predictions(result, test)
                        observed = test[spec.outcome].to_numpy(dtype=float)
                        sessions = test[SESSION].astype(str).to_numpy()
                        fixed_metrics = _metrics(observed, fixed, sessions)
                        conditional_metrics = _metrics(
                            observed, conditional, sessions
                        )
                        row = {
                            **base_row,
                            "fit_success": True,
                            "fit_error": "",
                            "converged": bool(result.converged),
                            "train_log_likelihood": float(result.llf),
                            "train_aic": float(result.aic),
                            "train_bic": float(result.bic),
                            "train_random_intercept_variance": float(
                                np.asarray(result.cov_re, dtype=float)[0, 0]
                            ),
                            "train_residual_variance": float(result.scale),
                            "fixed_effect_coefficients": json.dumps(
                                {
                                    str(term): float(value)
                                    for term, value in result.fe_params.items()
                                },
                                sort_keys=True,
                            ),
                            "n_fit_warnings": len(warning_messages),
                            "fit_warnings": " | ".join(
                                dict.fromkeys(warning_messages)
                            ),
                            **{
                                f"fixed_{key}": value
                                for key, value in fixed_metrics.items()
                            },
                            **{
                                f"conditional_{key}": value
                                for key, value in conditional_metrics.items()
                            },
                        }
                        sample = prediction_samples[spec.name]
                        sample_size = config.prediction_sample_per_model
                        sample_rng = prediction_rngs[spec.name]
                        for index in range(len(test)):
                            prediction_counts[spec.name] += 1
                            seen = prediction_counts[spec.name]
                            if len(sample) < sample_size:
                                replacement_index = len(sample)
                            else:
                                replacement_index = int(
                                    sample_rng.integers(0, seen)
                                )
                                if replacement_index >= sample_size:
                                    continue
                            prediction_row = {
                                "repeat": repeat,
                                "model": spec.name,
                                SESSION: sessions[index],
                                "trial_id": int(test.iloc[index]["trial_id"]),
                                spec.outcome: observed[index],
                                "fixed_prediction_ms": fixed[index],
                                "conditional_prediction_ms": conditional[index],
                            }
                            if replacement_index == len(sample):
                                sample.append(prediction_row)
                            else:
                                sample[replacement_index] = prediction_row
                    except Exception as error:  # Preserve the remaining CV run.
                        row = {
                            **base_row,
                            "fit_success": False,
                            "fit_error": f"{type(error).__name__}: {error}",
                            "converged": False,
                        }
                        print(f"  CV fit failed: {spec.name}: {error}")
                    metric_rows.append(row)

        metrics = pd.DataFrame(metric_rows)
        for prediction_type in ("fixed", "conditional"):
            for metric in ("rmse_ms", "mae_ms", "r2", "session_centered_r2"):
                value_column = f"{prediction_type}_{metric}"
                delta_column = f"{value_column}_delta_vs_parent"
                lookup = metrics.set_index(["repeat", "model"])[value_column]
                deltas: list[float] = []
                for row in metrics.itertuples(index=False):
                    parent = str(row.parent_model)
                    if not parent or not bool(row.fit_success):
                        deltas.append(np.nan)
                        continue
                    parent_value = lookup.get((int(row.repeat), parent), np.nan)
                    child_value = getattr(row, value_column, np.nan)
                    deltas.append(float(child_value - parent_value))
                metrics[delta_column] = deltas
        summary = _aggregate_metrics(metrics)
        predictions = pd.DataFrame(
            prediction
            for model_name in names
            for prediction in prediction_samples[model_name]
        )
        log.write(f"Models: {len(requests)}\n")
        log.write(f"Attempted fits: {len(metrics)}\n")
        log.write(f"Successful fits: {int(metrics['fit_success'].sum())}\n")
        log.write(f"Failed fits: {int((~metrics['fit_success']).sum())}\n\n")
        log.write("Mean held-out metrics by model\n")
        log.write("-" * 88 + "\n")
        display_columns = [
            "model",
            "n_successful_fits",
            "fixed_rmse_ms_mean",
            "conditional_rmse_ms_mean",
            "fixed_r2_mean",
            "conditional_r2_mean",
            "conditional_session_centered_r2_mean",
        ]
        log.write(summary[display_columns].to_string(index=False))
        log.write("\n")

    metrics.to_csv(table_dir / "cv_repeat_metrics.csv", index=False)
    metrics.to_pickle(table_dir / "cv_repeat_metrics.pkl")
    summary.to_csv(table_dir / "cv_model_summary.csv", index=False)
    summary.to_pickle(table_dir / "cv_model_summary.pkl")
    predictions.to_pickle(table_dir / "cv_prediction_sample.pkl")
    with (table_dir / "cv_config.json").open("w") as handle:
        json.dump(
            {
                "n_shuffles": config.n_shuffles,
                "holdout_fraction": config.holdout_fraction,
                "seed": config.seed,
                "history_alpha": config.history_alpha,
                "max_iterations": config.max_iterations,
                "split_source": split_source,
                "prediction_sample_per_model": config.prediction_sample_per_model,
                "outcome": outcome,
            },
            handle,
            indent=2,
        )
    _plot_overview(summary, figure_dir, config.figure_dpi, outcome)
    _plot_prediction_sample(
        predictions, summary, figure_dir, config.figure_dpi, outcome
    )
    print(f"Saved cross-validation metrics and plots to {output_dir}")
    return metrics, summary, predictions
