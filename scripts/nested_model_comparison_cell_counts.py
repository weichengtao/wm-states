"""Compare full and drop-one cell-count random-intercept models.

M0 contains a fixed intercept and a session random intercept. M1 adds the
preferred, selective-nonpreferred, and stationary-nonselective cell counts.
Three reduced models drop one count from M1 so that each count's contribution
can be tested with a full-versus-reduced likelihood-ratio test. Repeated
within-session trial holdouts provide paired out-of-sample comparisons.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tyro
from scipy.stats import chi2

try:
    from scripts.compare_mixed_effect_models import (
        COUNT_PREDICTORS,
        OUTCOME,
        SESSION,
        ModelSpec,
        _comparison_row,
        _fit_model,
        _fixed_effect_rows,
        _load_and_validate_data,
        _predictions_and_r2,
        _save_coefficient_forest,
        _save_model_plot,
        _validate_relative_component,
    )
    from scripts.mixedlm_outcomes import (
        OutcomeSelection,
        OutcomeSpec,
        analysis_output_dir,
        select_outcomes,
    )
except ModuleNotFoundError:
    from compare_mixed_effect_models import (
        COUNT_PREDICTORS,
        OUTCOME,
        SESSION,
        ModelSpec,
        _comparison_row,
        _fit_model,
        _fixed_effect_rows,
        _load_and_validate_data,
        _predictions_and_r2,
        _save_coefficient_forest,
        _save_model_plot,
        _validate_relative_component,
    )
    from mixedlm_outcomes import (
        OutcomeSelection,
        OutcomeSpec,
        analysis_output_dir,
        select_outcomes,
    )


UNUSED_CV_HISTORY_ALPHA = 0.2


@dataclass(frozen=True)
class ContrastSpec:
    name: str
    full_model: str
    reduced_model: str
    tested_predictors: tuple[str, ...]
    description: str


@dataclass
class Config:
    """Input, fitting, cross-validation, and output settings."""

    cache_dir: Path = Path("cache/run_029_full_session")
    input_subdir: str = "mixedlm/prepared"
    input_filename: str = "trial_table.pkl"
    output_subdir: str = "mixedlm"
    cv_input_subdir: str = "mixedlm/prepared"
    cv_input_filename: str = "cv_feature_cache.pkl"
    outcome: OutcomeSelection = "both"
    run_cv: bool = True
    cv_shuffles: int = 50
    cv_holdout_fraction: float = 0.2
    cv_seed: int = 42
    cv_prediction_sample_per_model: int = 1000
    significance_alpha: float = 0.05
    max_iterations: int = 1000
    figure_dpi: int = 200


def _model_specs(outcome: str = OUTCOME) -> list[ModelSpec]:
    preferred, non_preferred, non_selective = COUNT_PREDICTORS
    specs = [
        ModelSpec(
            name="M0",
            description="Fixed intercept plus session random intercept",
            predictors=(),
            parent=None,
        ),
        ModelSpec(
            name="M1",
            description="M0 plus all three raw cell counts",
            predictors=COUNT_PREDICTORS,
            parent="M0",
        ),
        ModelSpec(
            name="M1-drop-preferred",
            description="M1 without preferred cell count",
            predictors=(non_preferred, non_selective),
            parent=None,
        ),
        ModelSpec(
            name="M1-drop-non-preferred",
            description="M1 without selective non-preferred cell count",
            predictors=(preferred, non_selective),
            parent=None,
        ),
        ModelSpec(
            name="M1-drop-non-selective",
            description="M1 without stationary non-selective cell count",
            predictors=(preferred, non_preferred),
            parent=None,
        ),
    ]
    return [replace(spec, outcome=outcome) for spec in specs]


def _contrast_specs() -> tuple[ContrastSpec, ...]:
    return (
        ContrastSpec(
            name="M1-vs-M0",
            full_model="M1",
            reduced_model="M0",
            tested_predictors=COUNT_PREDICTORS,
            description="Joint contribution of all three cell counts",
        ),
        ContrastSpec(
            name="preferred-count",
            full_model="M1",
            reduced_model="M1-drop-preferred",
            tested_predictors=("preferred_cell_count",),
            description="Unique contribution of preferred cell count",
        ),
        ContrastSpec(
            name="selective-non-preferred-count",
            full_model="M1",
            reduced_model="M1-drop-non-preferred",
            tested_predictors=("selective_nonpreferred_cell_count",),
            description=(
                "Unique contribution of selective non-preferred cell count"
            ),
        ),
        ContrastSpec(
            name="stationary-non-selective-count",
            full_model="M1",
            reduced_model="M1-drop-non-selective",
            tested_predictors=("stationary_nonselective_cell_count",),
            description=(
                "Unique contribution of stationary non-selective cell count"
            ),
        ),
    )


def _validate_config(config: Config) -> None:
    if not np.isfinite(config.significance_alpha) or not (
        0 < config.significance_alpha < 1
    ):
        raise ValueError("significance_alpha must be in (0, 1).")
    if config.max_iterations < 1:
        raise ValueError("max_iterations must be positive.")
    if config.figure_dpi < 1:
        raise ValueError("figure_dpi must be positive.")
    _validate_relative_component(config.cv_input_subdir, "cv_input_subdir")
    if Path(config.cv_input_filename).name != config.cv_input_filename:
        raise ValueError("cv_input_filename must be a filename, not a path.")


def _nested_contrast_rows(
    comparison: pd.DataFrame,
    results: dict[str, Any],
    contrasts: tuple[ContrastSpec, ...],
    significance_alpha: float,
) -> pd.DataFrame:
    rows_by_model = comparison.set_index("model")
    rows: list[dict[str, Any]] = []
    for contrast in contrasts:
        full = rows_by_model.loc[contrast.full_model]
        reduced = rows_by_model.loc[contrast.reduced_model]
        full_result = results[contrast.full_model]
        reduced_result = results[contrast.reduced_model]
        full_predictors = set(full_result.model.exog_names) - {"Intercept"}
        reduced_predictors = set(reduced_result.model.exog_names) - {"Intercept"}
        tested = full_predictors - reduced_predictors
        if not reduced_predictors.issubset(full_predictors):
            raise RuntimeError(f"Models in {contrast.name} are not nested.")
        if tested != set(contrast.tested_predictors):
            raise RuntimeError(
                f"Unexpected tested predictors for {contrast.name}: {sorted(tested)}"
            )
        degrees_of_freedom = int(
            full_result.df_modelwc - reduced_result.df_modelwc
        )
        if degrees_of_freedom <= 0:
            raise RuntimeError(f"Invalid LRT degrees of freedom for {contrast.name}.")
        likelihood_ratio = 2.0 * (
            float(full_result.llf) - float(reduced_result.llf)
        )
        p_value = float(
            chi2.sf(max(likelihood_ratio, 0.0), degrees_of_freedom)
        )
        rows.append(
            {
                "contrast": contrast.name,
                "description": contrast.description,
                "full_model": contrast.full_model,
                "reduced_model": contrast.reduced_model,
                "tested_predictors": "; ".join(contrast.tested_predictors),
                "n_tested_predictors": len(contrast.tested_predictors),
                "likelihood_ratio": likelihood_ratio,
                "likelihood_ratio_df": degrees_of_freedom,
                "likelihood_ratio_p_value": p_value,
                "significant": bool(p_value < significance_alpha),
                "delta_marginal_r2": float(
                    full["marginal_r2"] - reduced["marginal_r2"]
                ),
                "delta_conditional_r2": float(
                    full["conditional_r2"] - reduced["conditional_r2"]
                ),
                "delta_log_likelihood": float(
                    full["log_likelihood"] - reduced["log_likelihood"]
                ),
                "delta_aic_full_minus_reduced": float(full["aic"] - reduced["aic"]),
                "aic_improvement_reduced_minus_full": float(
                    reduced["aic"] - full["aic"]
                ),
                "delta_bic_full_minus_reduced": float(full["bic"] - reduced["bic"]),
                "bic_improvement_reduced_minus_full": float(
                    reduced["bic"] - full["bic"]
                ),
                "delta_marginal_rmse_ms_full_minus_reduced": float(
                    full["marginal_rmse_ms"] - reduced["marginal_rmse_ms"]
                ),
                "marginal_rmse_improvement_ms": float(
                    reduced["marginal_rmse_ms"] - full["marginal_rmse_ms"]
                ),
                "delta_conditional_rmse_ms_full_minus_reduced": float(
                    full["conditional_rmse_ms"]
                    - reduced["conditional_rmse_ms"]
                ),
                "conditional_rmse_improvement_ms": float(
                    reduced["conditional_rmse_ms"]
                    - full["conditional_rmse_ms"]
                ),
                "delta_marginal_mae_ms_full_minus_reduced": float(
                    full["marginal_mae_ms"] - reduced["marginal_mae_ms"]
                ),
                "marginal_mae_improvement_ms": float(
                    reduced["marginal_mae_ms"] - full["marginal_mae_ms"]
                ),
                "delta_conditional_mae_ms_full_minus_reduced": float(
                    full["conditional_mae_ms"] - reduced["conditional_mae_ms"]
                ),
                "conditional_mae_improvement_ms": float(
                    reduced["conditional_mae_ms"] - full["conditional_mae_ms"]
                ),
                "delta_session_variance": float(
                    full["random_intercept_variance"]
                    - reduced["random_intercept_variance"]
                ),
                "delta_residual_variance": float(
                    full["residual_variance"] - reduced["residual_variance"]
                ),
            }
        )
    return pd.DataFrame(rows)


def _plot_model_metrics(
    comparison: pd.DataFrame,
    output_dir: Path,
    figure_dpi: int,
    outcome_label: str,
) -> Path:
    labels = comparison["model"].tolist()
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), layout="constrained")
    panels = (
        ("marginal_r2", "conditional_r2", "R²", "Marginal", "Conditional"),
        ("aic", "bic", "Information criterion", "AIC", "BIC"),
        ("log_likelihood", None, "Log-likelihood", "Log-likelihood", ""),
        (
            "marginal_rmse_ms",
            "conditional_rmse_ms",
            "RMSE (ms)",
            "Marginal",
            "Conditional",
        ),
        (
            "marginal_mae_ms",
            "conditional_mae_ms",
            "MAE (ms)",
            "Marginal",
            "Conditional",
        ),
        (
            "random_intercept_variance",
            "residual_variance",
            "Variance",
            "Session",
            "Residual",
        ),
    )
    width = 0.36
    for ax, (first, second, ylabel, first_label, second_label) in zip(
        axes.ravel(), panels
    ):
        if second is None:
            ax.bar(x, comparison[first], color="C0", width=0.65)
        else:
            ax.bar(x - width / 2, comparison[first], width, label=first_label)
            ax.bar(x + width / 2, comparison[second], width, label=second_label)
            ax.legend(fontsize=8)
        ax.set_xticks(x, labels, rotation=28, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(f"{outcome_label}: nested cell-count model metrics")
    path = output_dir / "model_metric_overview.png"
    fig.savefig(path, dpi=figure_dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_nested_contrasts(
    contrasts: pd.DataFrame,
    output_dir: Path,
    figure_dpi: int,
    outcome_label: str,
    significance_alpha: float,
) -> Path:
    labels = contrasts["contrast"].tolist()
    x = np.arange(len(labels))
    width = 0.36
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")

    axes[0, 0].bar(
        x - width / 2, contrasts["delta_marginal_r2"], width, label="ΔR²m"
    )
    axes[0, 0].bar(
        x + width / 2,
        contrasts["delta_conditional_r2"],
        width,
        label="ΔR²c",
    )
    axes[0, 0].axhline(0, color="black", linewidth=0.8)
    axes[0, 0].set_ylabel("Full minus reduced")
    axes[0, 0].legend()

    axes[0, 1].bar(
        x - width / 2,
        contrasts["conditional_rmse_improvement_ms"],
        width,
        label="RMSE",
    )
    axes[0, 1].bar(
        x + width / 2,
        contrasts["conditional_mae_improvement_ms"],
        width,
        label="MAE",
    )
    axes[0, 1].axhline(0, color="black", linewidth=0.8)
    axes[0, 1].set_ylabel("Conditional error improvement (ms)")
    axes[0, 1].legend()

    tiny = np.finfo(float).tiny
    minus_log_p = -np.log10(
        np.clip(contrasts["likelihood_ratio_p_value"], tiny, 1.0)
    )
    axes[1, 0].bar(x, minus_log_p, color="C3")
    axes[1, 0].axhline(
        -np.log10(significance_alpha), color="black", linestyle="--", linewidth=1
    )
    axes[1, 0].set_ylabel("LRT −log10(p)")

    axes[1, 1].bar(
        x - width / 2,
        contrasts["aic_improvement_reduced_minus_full"],
        width,
        label="AIC",
    )
    axes[1, 1].bar(
        x + width / 2,
        contrasts["bic_improvement_reduced_minus_full"],
        width,
        label="BIC",
    )
    axes[1, 1].axhline(0, color="black", linewidth=0.8)
    axes[1, 1].set_ylabel("Criterion improvement (reduced − full)")
    axes[1, 1].legend()

    for ax in axes.ravel():
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(f"{outcome_label}: full-versus-reduced contrasts")
    path = output_dir / "nested_contrasts.png"
    fig.savefig(path, dpi=figure_dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _cv_contrast_repeat_metrics(
    metrics: pd.DataFrame,
    contrasts: tuple[ContrastSpec, ...],
    significance_alpha: float,
) -> pd.DataFrame:
    if metrics.duplicated(["repeat", "model"]).any():
        raise ValueError("CV metrics contain duplicate repeat/model rows.")
    indexed = metrics.set_index(["repeat", "model"])
    rows: list[dict[str, Any]] = []
    heldout_metrics = (
        "fixed_r2",
        "conditional_r2",
        "fixed_session_centered_r2",
        "conditional_session_centered_r2",
        "fixed_rmse_ms",
        "conditional_rmse_ms",
        "fixed_mae_ms",
        "conditional_mae_ms",
        "fixed_pearson_r",
        "conditional_pearson_r",
    )
    train_metrics = (
        "train_marginal_r2",
        "train_conditional_r2",
        "train_log_likelihood",
        "train_aic",
        "train_bic",
        "train_random_intercept_variance",
        "train_residual_variance",
        "train_marginal_rmse_ms",
        "train_conditional_rmse_ms",
        "train_marginal_mae_ms",
        "train_conditional_mae_ms",
    )
    for repeat in sorted(metrics["repeat"].unique()):
        for contrast in contrasts:
            full = indexed.loc[(repeat, contrast.full_model)]
            reduced = indexed.loc[(repeat, contrast.reduced_model)]
            success = bool(full["fit_success"]) and bool(reduced["fit_success"])
            row: dict[str, Any] = {
                "repeat": int(repeat),
                "contrast": contrast.name,
                "description": contrast.description,
                "full_model": contrast.full_model,
                "reduced_model": contrast.reduced_model,
                "tested_predictors": "; ".join(contrast.tested_predictors),
                "paired_fit_success": success,
                "full_fit_error": str(full.get("fit_error", "")),
                "reduced_fit_error": str(reduced.get("fit_error", "")),
            }
            if success:
                degrees_of_freedom = int(
                    full["train_df_modelwc"] - reduced["train_df_modelwc"]
                )
                if degrees_of_freedom <= 0:
                    raise RuntimeError(
                        f"Invalid CV LRT degrees of freedom for {contrast.name}."
                    )
                likelihood_ratio = 2.0 * (
                    float(full["train_log_likelihood"])
                    - float(reduced["train_log_likelihood"])
                )
                p_value = float(
                    chi2.sf(max(likelihood_ratio, 0.0), degrees_of_freedom)
                )
                row.update(
                    {
                        "train_likelihood_ratio": likelihood_ratio,
                        "train_likelihood_ratio_df": degrees_of_freedom,
                        "train_likelihood_ratio_p_value": p_value,
                        "train_lrt_significant": bool(
                            p_value < significance_alpha
                        ),
                    }
                )
                for metric in (*heldout_metrics, *train_metrics):
                    full_value = float(full[metric])
                    reduced_value = float(reduced[metric])
                    row[f"{metric}_full"] = full_value
                    row[f"{metric}_reduced"] = reduced_value
                    row[f"{metric}_delta_full_minus_reduced"] = (
                        full_value - reduced_value
                    )
                for prediction_type in ("fixed", "conditional"):
                    for error in ("rmse_ms", "mae_ms"):
                        metric = f"{prediction_type}_{error}"
                        row[f"{metric}_improvement_reduced_minus_full"] = float(
                            reduced[metric] - full[metric]
                        )
                row["heldout_delta_marginal_r2"] = row[
                    "fixed_r2_delta_full_minus_reduced"
                ]
                row["heldout_delta_conditional_r2"] = row[
                    "conditional_r2_delta_full_minus_reduced"
                ]
                row["train_aic_improvement_reduced_minus_full"] = float(
                    reduced["train_aic"] - full["train_aic"]
                )
                row["train_bic_improvement_reduced_minus_full"] = float(
                    reduced["train_bic"] - full["train_bic"]
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _summarize_cv_contrasts(repeat_metrics: pd.DataFrame) -> pd.DataFrame:
    successful = repeat_metrics[repeat_metrics["paired_fit_success"]].copy()
    excluded = {
        "repeat",
        "paired_fit_success",
        "train_lrt_significant",
    }
    metric_columns = [
        column
        for column in successful.select_dtypes(include=[np.number]).columns
        if column not in excluded
    ]
    rows: list[dict[str, Any]] = []
    for contrast, all_rows in repeat_metrics.groupby("contrast", sort=False):
        model_rows = successful[successful["contrast"] == contrast]
        first = all_rows.iloc[0]
        row: dict[str, Any] = {
            "contrast": contrast,
            "description": first["description"],
            "full_model": first["full_model"],
            "reduced_model": first["reduced_model"],
            "tested_predictors": first["tested_predictors"],
            "n_shuffles_requested": int(len(all_rows)),
            "n_successful_pairs": int(len(model_rows)),
            "n_failed_pairs": int(len(all_rows) - len(model_rows)),
        }
        if len(model_rows):
            row["train_lrt_significant_fraction"] = float(
                model_rows["train_lrt_significant"].mean()
            )
        else:
            row["train_lrt_significant_fraction"] = np.nan
        for column in metric_columns:
            values = model_rows[column].dropna().to_numpy(dtype=float)
            for suffix in ("mean", "std", "median", "q025", "q975"):
                row[f"{column}_{suffix}"] = np.nan
            if values.size:
                row[f"{column}_mean"] = float(np.mean(values))
                row[f"{column}_std"] = (
                    float(np.std(values, ddof=1)) if values.size > 1 else 0.0
                )
                row[f"{column}_median"] = float(np.median(values))
                row[f"{column}_q025"] = float(np.quantile(values, 0.025))
                row[f"{column}_q975"] = float(np.quantile(values, 0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_cv_contrasts(
    summary: pd.DataFrame,
    output_dir: Path,
    figure_dpi: int,
    outcome_label: str,
) -> Path | None:
    rows = summary[summary["n_successful_pairs"] > 0]
    if rows.empty:
        return None
    labels = rows["contrast"].tolist()
    x = np.arange(len(labels))
    width = 0.36
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), layout="constrained")

    for offset, metric, label in (
        (-width / 2, "heldout_delta_marginal_r2", "Marginal/fixed"),
        (width / 2, "heldout_delta_conditional_r2", "Conditional"),
    ):
        means = rows[f"{metric}_mean"].to_numpy(dtype=float)
        lower = rows[f"{metric}_q025"].to_numpy(dtype=float)
        upper = rows[f"{metric}_q975"].to_numpy(dtype=float)
        axes[0].bar(x + offset, means, width, label=label)
        axes[0].errorbar(
            x + offset,
            means,
            yerr=np.vstack([means - lower, upper - means]),
            fmt="none",
            color="black",
            capsize=2,
        )
    axes[0].set_ylabel("Held-out ΔR² (full − reduced)")
    axes[0].legend(fontsize=8)

    for offset, metric, label in (
        (
            -width / 2,
            "conditional_rmse_ms_improvement_reduced_minus_full",
            "RMSE",
        ),
        (
            width / 2,
            "conditional_mae_ms_improvement_reduced_minus_full",
            "MAE",
        ),
    ):
        means = rows[f"{metric}_mean"].to_numpy(dtype=float)
        lower = rows[f"{metric}_q025"].to_numpy(dtype=float)
        upper = rows[f"{metric}_q975"].to_numpy(dtype=float)
        axes[1].bar(x + offset, means, width, label=label)
        axes[1].errorbar(
            x + offset,
            means,
            yerr=np.vstack([means - lower, upper - means]),
            fmt="none",
            color="black",
            capsize=2,
        )
    axes[1].set_ylabel("Held-out conditional improvement (ms)")
    axes[1].legend(fontsize=8)

    axes[2].bar(x, rows["train_lrt_significant_fraction"], color="C3")
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Fraction of training-fold LRTs significant")

    for ax in axes:
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(f"{outcome_label}: paired held-out-trial comparisons")
    path = output_dir / "cv_nested_contrasts.png"
    fig.savefig(path, dpi=figure_dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_fit_log_header(
    handle: Any,
    config: Config,
    input_path: Path,
    frame: pd.DataFrame,
    outcome: OutcomeSpec,
) -> None:
    handle.write("Nested cell-count mixed-effects model comparison\n")
    handle.write("=" * 88 + "\n")
    handle.write(f"Input: {input_path}\n")
    handle.write(f"Outcome: {outcome.label} ({outcome.column})\n")
    handle.write(f"Rows: {len(frame)}\n")
    handle.write(f"Sessions: {frame[SESSION].nunique()}\n")
    handle.write("Estimator: statsmodels MixedLM; REML: False\n")
    handle.write("Random effects: session random intercept only\n")
    handle.write(f"Significance alpha: {config.significance_alpha}\n")
    handle.write(
        "Marginal R2 is fixed-effect variance divided by total model variance; "
        "conditional R2 additionally includes session variance.\n"
    )
    handle.write(
        "All deltas are full minus reduced. Error and information-criterion "
        "improvements are additionally reported as reduced minus full, so "
        "positive values favor the full model.\n\n"
    )


def _append_model_log(
    handle: Any,
    spec: ModelSpec,
    result: Any,
    row: dict[str, Any],
    warning_messages: list[str],
    plot_path: Path,
    coefficient_plot_path: Path,
) -> None:
    handle.write("=" * 88 + "\n")
    handle.write(f"{spec.name}: {spec.description}\n")
    handle.write(f"Formula: {spec.formula}\n")
    handle.write(f"Diagnostic figure: {plot_path}\n")
    handle.write(f"Coefficient forest: {coefficient_plot_path}\n")
    handle.write(
        "Metrics:\n"
        + "\n".join(
            f"  {key}: {row[key]}"
            for key in (
                "log_likelihood",
                "aic",
                "bic",
                "marginal_r2",
                "conditional_r2",
                "marginal_rmse_ms",
                "conditional_rmse_ms",
                "marginal_mae_ms",
                "conditional_mae_ms",
                "random_intercept_variance",
                "residual_variance",
                "icc",
            )
        )
        + "\n"
    )
    if warning_messages:
        handle.write("Fit warnings:\n")
        for message in dict.fromkeys(warning_messages):
            handle.write(f"  - {message}\n")
    else:
        handle.write("Fit warnings: none\n")
    handle.write("\nStatsmodels fit summary:\n")
    handle.write(result.summary().as_text())
    handle.write("\n\n")


def _write_cv_contrast_log(
    path: Path,
    repeat_metrics: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    display_columns = [
        "contrast",
        "n_successful_pairs",
        "heldout_delta_marginal_r2_mean",
        "heldout_delta_conditional_r2_mean",
        "conditional_rmse_ms_improvement_reduced_minus_full_mean",
        "conditional_mae_ms_improvement_reduced_minus_full_mean",
        "train_likelihood_ratio_mean",
        "train_likelihood_ratio_p_value_median",
        "train_lrt_significant_fraction",
    ]
    with path.open("w") as handle:
        handle.write("Paired nested cell-count cross-validation contrasts\n")
        handle.write("=" * 88 + "\n")
        handle.write(
            "Held-out marginal metrics use fixed-effect predictions. Held-out "
            "conditional metrics add session BLUPs learned only from training "
            "trials. AIC, BIC, likelihood, variance components, variance-based "
            "R2, and LRT are training-fold diagnostics.\n"
        )
        handle.write(
            "R2 deltas are full minus reduced; positive error improvements are "
            "reduced minus full. Intervals are empirical 2.5% and 97.5% "
            "quantiles across paired shuffles.\n\n"
        )
        handle.write(
            f"Paired attempts: {len(repeat_metrics)}; successful: "
            f"{int(repeat_metrics['paired_fit_success'].sum())}\n\n"
        )
        handle.write(summary[display_columns].to_string(index=False))
        handle.write("\n")


def _run_cross_validation(
    config: Config,
    outcome: OutcomeSpec,
    specs: list[ModelSpec],
    contrasts: tuple[ContrastSpec, ...],
    output_dir: Path,
) -> None:
    try:
        from scripts.mixedlm_trial_holdout_cv import (
            CVModelRequest,
            TrialHoldoutConfig,
            run_trial_holdout_cv,
        )
    except ModuleNotFoundError:
        from mixedlm_trial_holdout_cv import (
            CVModelRequest,
            TrialHoldoutConfig,
            run_trial_holdout_cv,
        )

    cv_dir = output_dir / "cross_validation"
    requests = [
        CVModelRequest(
            spec=spec,
            active_threshold=0.0,
            metadata={"model_family": "nested_cell_counts"},
        )
        for spec in specs
    ]
    metrics, _, _ = run_trial_holdout_cv(
        config.cache_dir / config.cv_input_subdir / config.cv_input_filename,
        requests,
        cv_dir,
        TrialHoldoutConfig(
            n_shuffles=config.cv_shuffles,
            holdout_fraction=config.cv_holdout_fraction,
            seed=config.cv_seed,
            # The shared fold builder requires this setting while constructing
            # all feature families. No history feature enters these formulas.
            history_alpha=UNUSED_CV_HISTORY_ALPHA,
            max_iterations=config.max_iterations,
            figure_dpi=config.figure_dpi,
            prediction_sample_per_model=config.cv_prediction_sample_per_model,
        ),
    )
    repeat_contrasts = _cv_contrast_repeat_metrics(
        metrics, contrasts, config.significance_alpha
    )
    contrast_summary = _summarize_cv_contrasts(repeat_contrasts)
    table_dir = cv_dir / "tables"
    log_dir = cv_dir / "logs"
    figure_dir = cv_dir / "figures"
    repeat_contrasts.to_csv(
        table_dir / "cv_nested_contrast_repeat_metrics.csv", index=False
    )
    repeat_contrasts.to_pickle(
        table_dir / "cv_nested_contrast_repeat_metrics.pkl"
    )
    contrast_summary.to_csv(
        table_dir / "cv_nested_contrast_summary.csv", index=False
    )
    contrast_summary.to_pickle(table_dir / "cv_nested_contrast_summary.pkl")
    _write_cv_contrast_log(
        log_dir / "nested_contrast_cross_validation.log",
        repeat_contrasts,
        contrast_summary,
    )
    _plot_cv_contrasts(
        contrast_summary,
        figure_dir,
        config.figure_dpi,
        outcome.label,
    )


def _run_outcome(config: Config, outcome: OutcomeSpec) -> None:
    _validate_config(config)
    specs = _model_specs(outcome.column)
    contrasts = _contrast_specs()
    frame = _load_and_validate_data(config, specs)
    input_path = config.cache_dir / config.input_subdir / config.input_filename
    output_dir = analysis_output_dir(
        config.cache_dir,
        config.output_subdir,
        outcome,
        "nested_cell_count_comparison",
    )
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    marginal_figure_dir = figure_dir / "marginal_effects"
    coefficient_figure_dir = figure_dir / "coefficient_forests"
    log_dir = output_dir / "logs"
    table_dir.mkdir(parents=True, exist_ok=True)
    marginal_figure_dir.mkdir(parents=True, exist_ok=True)
    coefficient_figure_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}
    comparison_rows: list[dict[str, Any]] = []
    fixed_effect_rows: list[dict[str, Any]] = []
    log_path = log_dir / "nested_cell_count_model_fits.log"
    with log_path.open("w") as log:
        _write_fit_log_header(log, config, input_path, frame, outcome)
        for index, spec in enumerate(specs, start=1):
            print(f"[{outcome.name}] Fitting {index}/{len(specs)}: {spec.name}")
            result, warning_messages = _fit_model(
                frame, spec, config.max_iterations
            )
            variance_metrics = _predictions_and_r2(result, frame, spec.outcome)
            model_fixed_rows = _fixed_effect_rows(
                result, spec, config.significance_alpha
            )
            parent_result = results.get(spec.parent) if spec.parent else None
            row = _comparison_row(
                result,
                spec,
                variance_metrics,
                model_fixed_rows,
                parent_result,
                warning_messages,
            )
            plot_path = _save_model_plot(
                frame,
                result,
                spec,
                variance_metrics,
                marginal_figure_dir,
                config.figure_dpi,
            )
            coefficient_plot_path = _save_coefficient_forest(
                model_fixed_rows,
                spec,
                coefficient_figure_dir,
                config.figure_dpi,
                outcome.label,
            )
            _append_model_log(
                log,
                spec,
                result,
                row,
                warning_messages,
                plot_path,
                coefficient_plot_path,
            )
            results[spec.name] = result
            comparison_rows.append(row)
            fixed_effect_rows.extend(model_fixed_rows)

        comparison = pd.DataFrame(comparison_rows)
        nested_contrasts = _nested_contrast_rows(
            comparison, results, contrasts, config.significance_alpha
        )
        log.write("=" * 88 + "\n")
        log.write("Nested full-versus-reduced contrasts\n")
        log.write("=" * 88 + "\n")
        log.write(nested_contrasts.to_string(index=False))
        log.write("\n")

    fixed_effects = pd.DataFrame(fixed_effect_rows)
    comparison.to_csv(table_dir / "model_comparison.csv", index=False)
    comparison.to_pickle(table_dir / "model_comparison.pkl")
    nested_contrasts.to_csv(table_dir / "nested_contrasts.csv", index=False)
    nested_contrasts.to_pickle(table_dir / "nested_contrasts.pkl")
    fixed_effects.to_csv(table_dir / "fixed_effect_estimates.csv", index=False)
    with (table_dir / "analysis_config.json").open("w") as handle:
        json.dump(
            {
                "outcome": outcome.name,
                "significance_alpha": config.significance_alpha,
                "max_iterations": config.max_iterations,
                "run_cv": config.run_cv,
                "cv_shuffles": config.cv_shuffles,
                "cv_holdout_fraction": config.cv_holdout_fraction,
                "cv_seed": config.cv_seed,
            },
            handle,
            indent=2,
        )

    _plot_model_metrics(
        comparison, figure_dir, config.figure_dpi, outcome.label
    )
    _plot_nested_contrasts(
        nested_contrasts,
        figure_dir,
        config.figure_dpi,
        outcome.label,
        config.significance_alpha,
    )
    print(f"Saved nested cell-count comparison to {output_dir}")
    if config.run_cv:
        _run_cross_validation(config, outcome, specs, contrasts, output_dir)


def main(config: Config) -> None:
    for outcome in select_outcomes(config.outcome):
        _run_outcome(config, outcome)


if __name__ == "__main__":
    main(tyro.cli(Config))
