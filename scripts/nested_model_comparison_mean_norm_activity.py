"""Compare nested preferred-cell mean-normalized-activity MixedLM models.

The sequence starts with a fixed intercept and session random intercept, adds
preferred and selective-nonpreferred cell counts, and then cumulatively adds
preferred-cell mean normalized activity from baseline, encoding, pre-delay,
and full-delay periods. Maximum-likelihood fits support nested LRTs; repeated
within-session trial holdouts provide a separate predictive comparison.
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

try:
    from scripts.activity_weighting import (
        weighting_mode,
        weighting_policy,
        weighting_subdir,
    )
    from scripts.compare_mixed_effect_models import (
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
    from activity_weighting import weighting_mode, weighting_policy, weighting_subdir
    from compare_mixed_effect_models import (
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


COUNT_PREDICTORS = (
    "preferred_cell_count",
    "selective_nonpreferred_cell_count",
)
ACTIVITY_PREDICTORS = (
    "baseline_mean_normalized_activity_preferred",
    "encoding_mean_normalized_activity_preferred",
    "pre_delay_mean_normalized_activity_preferred",
    "delay_mean_normalized_activity_preferred",
)
UNUSED_CV_HISTORY_ALPHA = 0.2


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
    # Use the separately prepared hybrid PEV-weighted activity features.
    pev_weighted_average: bool = False


def _model_specs(outcome: str = OUTCOME) -> list[ModelSpec]:
    baseline, encoding, pre_delay, full_delay = ACTIVITY_PREDICTORS
    specs = [
        ModelSpec(
            name="M0",
            description="Fixed intercept plus session random intercept",
            predictors=(),
            parent=None,
        ),
        ModelSpec(
            name="M1",
            description=(
                "M0 plus preferred and selective non-preferred cell counts"
            ),
            predictors=COUNT_PREDICTORS,
            parent="M0",
        ),
        ModelSpec(
            name="M2",
            description="M1 plus preferred-cell baseline mean normalized activity",
            predictors=(*COUNT_PREDICTORS, baseline),
            parent="M1",
        ),
        ModelSpec(
            name="M3",
            description="M2 plus preferred-cell encoding mean normalized activity",
            predictors=(*COUNT_PREDICTORS, baseline, encoding),
            parent="M2",
        ),
        ModelSpec(
            name="M4",
            description="M3 plus preferred-cell pre-delay mean normalized activity",
            predictors=(*COUNT_PREDICTORS, baseline, encoding, pre_delay),
            parent="M3",
        ),
        ModelSpec(
            name="M5",
            description="M4 plus preferred-cell full-delay mean normalized activity",
            predictors=(
                *COUNT_PREDICTORS,
                baseline,
                encoding,
                pre_delay,
                full_delay,
            ),
            parent="M4",
        ),
    ]
    return [replace(spec, outcome=outcome) for spec in specs]


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


def _nested_contrast_table(
    comparison: pd.DataFrame,
    specs: list[ModelSpec],
) -> pd.DataFrame:
    rows_by_model = comparison.set_index("model")
    specs_by_model = {spec.name: spec for spec in specs}
    rows: list[dict[str, Any]] = []
    for spec in specs:
        if spec.parent is None:
            continue
        full = rows_by_model.loc[spec.name]
        reduced = rows_by_model.loc[spec.parent]
        parent_predictors = set(specs_by_model[spec.parent].predictors)
        added_predictors = tuple(
            predictor
            for predictor in spec.predictors
            if predictor not in parent_predictors
        )
        if not parent_predictors.issubset(spec.predictors):
            raise RuntimeError(f"{spec.name} is not nested within {spec.parent}.")
        rows.append(
            {
                "contrast": f"{spec.name}-vs-{spec.parent}",
                "full_model": spec.name,
                "reduced_model": spec.parent,
                "added_predictors": "; ".join(added_predictors),
                "n_added_predictors": len(added_predictors),
                "likelihood_ratio": full["likelihood_ratio_vs_parent"],
                "likelihood_ratio_df": full["likelihood_ratio_df"],
                "likelihood_ratio_p_value": full[
                    "likelihood_ratio_p_value"
                ],
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
                "marginal_rmse_improvement_ms": float(
                    reduced["marginal_rmse_ms"] - full["marginal_rmse_ms"]
                ),
                "conditional_rmse_improvement_ms": float(
                    reduced["conditional_rmse_ms"]
                    - full["conditional_rmse_ms"]
                ),
                "marginal_mae_improvement_ms": float(
                    reduced["marginal_mae_ms"] - full["marginal_mae_ms"]
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


def _plot_model_progression(
    comparison: pd.DataFrame,
    output_dir: Path,
    figure_dpi: int,
    outcome_label: str,
) -> Path:
    x = np.arange(len(comparison))
    labels = comparison["model"].tolist()
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout="constrained")
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
    for ax, (first, second, ylabel, first_label, second_label) in zip(
        axes.ravel(), panels
    ):
        ax.plot(x, comparison[first], marker="o", label=first_label)
        if second is not None:
            ax.plot(x, comparison[second], marker="o", label=second_label)
            ax.legend(fontsize=8)
        ax.set_xticks(x, labels)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
    fig.suptitle(
        f"{outcome_label}: nested preferred-cell mean-activity progression"
    )
    path = output_dir / "model_progression.png"
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
    axes[0, 0].set_ylabel("Full minus parent")
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
    axes[0, 1].set_ylabel("Conditional error improvement (ms)")
    axes[0, 1].legend()

    tiny = np.finfo(float).tiny
    axes[1, 0].bar(
        x,
        -np.log10(
            np.clip(contrasts["likelihood_ratio_p_value"], tiny, 1.0)
        ),
        color="C3",
    )
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
    axes[1, 1].set_ylabel("Criterion improvement (parent − full)")
    axes[1, 1].legend()
    for ax in axes.ravel():
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(f"{outcome_label}: sequential nested contrasts")
    path = output_dir / "nested_contrasts.png"
    fig.savefig(path, dpi=figure_dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_cv_progression(
    summary: pd.DataFrame,
    output_dir: Path,
    figure_dpi: int,
    outcome_label: str,
) -> Path:
    successful = summary[summary["n_successful_fits"] > 0]
    x = np.arange(len(successful))
    labels = successful["model"].tolist()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    panels = (
        ("fixed_r2", "conditional_r2", "Held-out R²"),
        ("fixed_rmse_ms", "conditional_rmse_ms", "Held-out RMSE (ms)"),
        ("fixed_mae_ms", "conditional_mae_ms", "Held-out MAE (ms)"),
        ("train_marginal_r2", "train_conditional_r2", "Training-fold R²"),
    )
    for ax, (fixed_metric, conditional_metric, ylabel) in zip(
        axes.ravel(), panels
    ):
        for metric, label in (
            (fixed_metric, "Marginal/fixed"),
            (conditional_metric, "Conditional"),
        ):
            means = successful[f"{metric}_mean"].to_numpy(dtype=float)
            lower = successful[f"{metric}_q025"].to_numpy(dtype=float)
            upper = successful[f"{metric}_q975"].to_numpy(dtype=float)
            ax.errorbar(
                x,
                means,
                yerr=np.vstack([means - lower, upper - means]),
                marker="o",
                capsize=2,
                label=label,
            )
        ax.set_xticks(x, labels)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(
        f"{outcome_label}: held-out-trial nested-model progression\n"
        "Error bars show empirical 95% intervals across shuffles"
    )
    path = output_dir / "cv_model_progression.png"
    fig.savefig(path, dpi=figure_dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_log_header(
    handle: Any,
    config: Config,
    input_path: Path,
    frame: pd.DataFrame,
    outcome: OutcomeSpec,
) -> None:
    handle.write("Nested preferred-cell mean-activity MixedLM comparison\n")
    handle.write("=" * 88 + "\n")
    handle.write(f"Input: {input_path}\n")
    handle.write(f"Outcome: {outcome.label} ({outcome.column})\n")
    handle.write(f"Rows: {len(frame)}\n")
    handle.write(f"Sessions: {frame[SESSION].nunique()}\n")
    handle.write("Estimator: statsmodels MixedLM; REML: False\n")
    handle.write("Random effects: session random intercept only\n")
    handle.write(f"Significance alpha: {config.significance_alpha}\n")
    handle.write(
        f"Activity weighting: {weighting_mode(config.pev_weighted_average)}; "
        f"policy={weighting_policy(config.pev_weighted_average)}\n"
    )
    handle.write(
        "Marginal R2 uses fixed-effect variance; conditional R2 additionally "
        "includes session random-intercept variance. Positive R2 deltas and "
        "positive reduced-minus-full error improvements favor the added block.\n\n"
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
    handle.write(f"Parent: {spec.parent or 'none'}\n")
    handle.write(f"Formula: {spec.formula}\n")
    handle.write(f"Diagnostic figure: {plot_path}\n")
    handle.write(f"Coefficient forest: {coefficient_plot_path}\n")
    handle.write("Computed metrics:\n")
    for key in (
        "log_likelihood",
        "aic",
        "bic",
        "likelihood_ratio_vs_parent",
        "likelihood_ratio_df",
        "likelihood_ratio_p_value",
        "marginal_r2",
        "conditional_r2",
        "marginal_rmse_ms",
        "conditional_rmse_ms",
        "marginal_mae_ms",
        "conditional_mae_ms",
        "random_intercept_variance",
        "residual_variance",
        "icc",
    ):
        handle.write(f"  {key}: {row[key]}\n")
    if warning_messages:
        handle.write("Fit warnings:\n")
        for message in dict.fromkeys(warning_messages):
            handle.write(f"  - {message}\n")
    else:
        handle.write("Fit warnings: none\n")
    handle.write("\nStatsmodels fit summary:\n")
    handle.write(result.summary().as_text())
    handle.write("\n\n")


def _run_cross_validation(
    config: Config,
    outcome: OutcomeSpec,
    specs: list[ModelSpec],
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
            metadata={"stage": index, "added_block": spec.description},
        )
        for index, spec in enumerate(specs)
    ]
    _, summary, _ = run_trial_holdout_cv(
        config.cache_dir
        / weighting_subdir(config.cv_input_subdir, config.pev_weighted_average)
        / config.cv_input_filename,
        requests,
        cv_dir,
        TrialHoldoutConfig(
            n_shuffles=config.cv_shuffles,
            holdout_fraction=config.cv_holdout_fraction,
            seed=config.cv_seed,
            # Required by the shared all-feature fold builder, but no history
            # feature enters any model in this analysis.
            history_alpha=UNUSED_CV_HISTORY_ALPHA,
            max_iterations=config.max_iterations,
            figure_dpi=config.figure_dpi,
            prediction_sample_per_model=config.cv_prediction_sample_per_model,
            pev_weighted_average=config.pev_weighted_average,
        ),
    )
    _plot_cv_progression(
        summary,
        cv_dir / "figures",
        config.figure_dpi,
        outcome.label,
    )


def _run_outcome(config: Config, outcome: OutcomeSpec) -> None:
    _validate_config(config)
    specs = _model_specs(outcome.column)
    frame = _load_and_validate_data(config, specs)
    input_path = (
        config.cache_dir
        / weighting_subdir(config.input_subdir, config.pev_weighted_average)
        / config.input_filename
    )
    output_dir = weighting_subdir(
        analysis_output_dir(
            config.cache_dir,
            config.output_subdir,
            outcome,
            "nested_mean_norm_activity_comparison",
        ),
        config.pev_weighted_average,
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
    all_fixed_effect_rows: list[dict[str, Any]] = []
    log_path = log_dir / "nested_mean_norm_activity_model_fits.log"
    with log_path.open("w") as log:
        _write_log_header(log, config, input_path, frame, outcome)
        for index, spec in enumerate(specs, start=1):
            print(f"[{outcome.name}] Fitting {index}/{len(specs)}: {spec.name}")
            result, warning_messages = _fit_model(
                frame, spec, config.max_iterations
            )
            variance_metrics = _predictions_and_r2(result, frame, spec.outcome)
            fixed_effect_rows = _fixed_effect_rows(
                result, spec, config.significance_alpha
            )
            parent_result = results.get(spec.parent) if spec.parent else None
            row = _comparison_row(
                result,
                spec,
                variance_metrics,
                fixed_effect_rows,
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
                fixed_effect_rows,
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
            all_fixed_effect_rows.extend(fixed_effect_rows)

        comparison = pd.DataFrame(comparison_rows)
        nested_contrasts = _nested_contrast_table(comparison, specs)
        log.write("=" * 88 + "\n")
        log.write("Sequential nested contrasts\n")
        log.write("=" * 88 + "\n")
        log.write(nested_contrasts.to_string(index=False))
        log.write("\n")

    fixed_effects = pd.DataFrame(all_fixed_effect_rows)
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
                "pev_weighted_average": config.pev_weighted_average,
                "activity_weighting_mode": weighting_mode(
                    config.pev_weighted_average
                ),
                "activity_weighting_policy": weighting_policy(
                    config.pev_weighted_average
                ),
            },
            handle,
            indent=2,
        )

    _plot_model_progression(
        comparison, figure_dir, config.figure_dpi, outcome.label
    )
    _plot_nested_contrasts(
        nested_contrasts,
        figure_dir,
        config.figure_dpi,
        outcome.label,
        config.significance_alpha,
    )
    print(f"Saved nested mean-normalized-activity comparison to {output_dir}")
    if config.run_cv:
        _run_cross_validation(config, outcome, specs, output_dir)


def main(config: Config) -> None:
    for outcome in select_outcomes(config.outcome):
        _run_outcome(config, outcome)


if __name__ == "__main__":
    main(tyro.cli(Config))
