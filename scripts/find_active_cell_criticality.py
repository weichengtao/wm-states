"""Scan active-cell thresholds for trial-level off-state MixedLM models.

Percentile cutoffs are converted to standard-normal z scores.  Prepared data
and all threshold-dependent fits are kept under a dedicated output directory;
the default 50th-percentile cutoff maps exactly to z=0.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tyro
from scipy.stats import norm

try:
    from scripts.compare_mixed_effect_models import (
        ModelSpec,
        _comparison_row,
        _fit_model,
        _fixed_effect_rows,
        _model_specs,
        _predictions_and_r2,
    )
    from scripts.prepare_data_for_mixedlm import (
        Config as PrepareConfig,
        prepare_data,
    )
    from scripts.mixedlm_outcomes import (
        OutcomeSelection,
        OutcomeSpec,
        analysis_output_dir,
        select_outcomes,
    )
except ModuleNotFoundError:
    from compare_mixed_effect_models import (
        ModelSpec,
        _comparison_row,
        _fit_model,
        _fixed_effect_rows,
        _model_specs,
        _predictions_and_r2,
    )
    from prepare_data_for_mixedlm import Config as PrepareConfig, prepare_data
    from mixedlm_outcomes import (
        OutcomeSelection,
        OutcomeSpec,
        analysis_output_dir,
        select_outcomes,
    )


PERIOD_LABELS = ("baseline", "encoding", "pre-delay", "full-delay")


@dataclass
class Config:
    """Data locations, percentile scan, and model-fitting settings."""

    data_dir: Path = Path("data/nature")
    cache_dir: Path = Path("cache/run_029_full_session")
    output_subdir: str = "mixedlm"
    prepared_subdir: str = "mixedlm/prepared"
    cv_input_subdir: str = "mixedlm/prepared"
    cv_input_filename: str = "cv_feature_cache.pkl"
    outcome: OutcomeSelection = "both"
    run_cv: bool = True
    cv_shuffles: int = 50
    cv_holdout_fraction: float = 0.2
    cv_seed: int = 42
    cv_prediction_sample_per_model: int = 1000
    active_percentiles: list[int] = field(
        default_factory=lambda: list(range(10, 100, 10))
    )
    history_alpha: float = 0.2
    significance_alpha: float = 0.05
    max_iterations: int = 1000
    figure_dpi: int = 200


def _validate_config(config: Config) -> list[int]:
    output_subdir = Path(config.output_subdir)
    if output_subdir.is_absolute() or ".." in output_subdir.parts:
        raise ValueError("output_subdir must stay within cache_dir.")
    prepared_subdir = Path(config.prepared_subdir)
    if prepared_subdir.is_absolute() or ".." in prepared_subdir.parts:
        raise ValueError("prepared_subdir must stay within cache_dir.")
    if not config.active_percentiles:
        raise ValueError("active_percentiles must not be empty.")
    if any(
        isinstance(percentile, bool)
        or not isinstance(percentile, (int, np.integer))
        for percentile in config.active_percentiles
    ):
        raise ValueError("active_percentiles must contain integer percentiles.")
    percentiles = sorted(set(int(value) for value in config.active_percentiles))
    if len(percentiles) != len(config.active_percentiles):
        raise ValueError("active_percentiles must not contain duplicates.")
    if any(percentile <= 0 or percentile >= 100 for percentile in percentiles):
        raise ValueError("active_percentiles must be strictly between 0 and 100.")
    if not np.isfinite(config.history_alpha) or not 0 < config.history_alpha <= 1:
        raise ValueError("history_alpha must be finite and in (0, 1].")
    if not np.isfinite(config.significance_alpha) or not (
        0 < config.significance_alpha < 1
    ):
        raise ValueError("significance_alpha must be in (0, 1).")
    if config.max_iterations < 1:
        raise ValueError("max_iterations must be positive.")
    if config.figure_dpi < 1:
        raise ValueError("figure_dpi must be positive.")
    cv_input_subdir = Path(config.cv_input_subdir)
    if cv_input_subdir.is_absolute() or ".." in cv_input_subdir.parts:
        raise ValueError("cv_input_subdir must stay within cache_dir.")
    if Path(config.cv_input_filename).name != config.cv_input_filename:
        raise ValueError("cv_input_filename must be a filename, not a path.")
    return percentiles


def _ordinal(percentile: int) -> str:
    if 10 <= percentile % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(percentile % 10, "th")
    return f"{percentile}{suffix}"


def _active_z_threshold(percentile: int) -> float:
    return float(norm.ppf(percentile / 100.0))


def _uses_active_fraction(spec: ModelSpec) -> bool:
    return any("active_fraction" in predictor for predictor in spec.predictors)


def _threshold_spec(
    spec: ModelSpec,
    percentile: int,
    z_threshold: float,
    dependent_model_names: set[str],
) -> ModelSpec:
    suffix = f"-active-z-{_ordinal(percentile)}"
    parent = spec.parent
    if parent in dependent_model_names:
        parent = f"{parent}{suffix}"
    return ModelSpec(
        name=f"{spec.name}{suffix}",
        description=(
            f"{spec.description}; active cutoff={_ordinal(percentile)} percentile "
            f"(z={z_threshold:.6f})"
        ),
        predictors=spec.predictors,
        parent=parent,
        outcome=spec.outcome,
    )


def _prepare_threshold_data(
    config: Config,
    percentiles: list[int],
) -> tuple[dict[int, pd.DataFrame], dict[int, float], dict[int, Path]]:
    frames: dict[int, pd.DataFrame] = {}
    thresholds: dict[int, float] = {}
    paths: dict[int, Path] = {}
    for index, percentile in enumerate(percentiles, start=1):
        z_threshold = _active_z_threshold(percentile)
        threshold_subdir = (
            Path(config.prepared_subdir)
            / "active_thresholds"
            / f"percentile_{percentile:02d}"
        )
        print(
            f"Preparing threshold {index}/{len(percentiles)}: "
            f"{_ordinal(percentile)} percentile (z={z_threshold:.6f})"
        )
        prepare_config = PrepareConfig(
            data_dir=config.data_dir,
            cache_dir=config.cache_dir,
            output_subdir=str(threshold_subdir),
            output_filename="trial_table.pkl",
            active_threshold=z_threshold,
            history_alpha=config.history_alpha,
            save_cv_cache=False,
        )
        frames[percentile] = prepare_data(prepare_config)
        thresholds[percentile] = z_threshold
        paths[percentile] = (
            config.cache_dir / threshold_subdir / prepare_config.output_filename
        )
    return frames, thresholds, paths


def _fit_one(
    frame: pd.DataFrame,
    spec: ModelSpec,
    results: dict[str, Any],
    config: Config,
) -> tuple[Any, dict[str, Any], list[dict[str, Any]], list[str]]:
    result, warning_messages = _fit_model(frame, spec, config.max_iterations)
    variance_metrics = _predictions_and_r2(result, frame, spec.outcome)
    fixed_effect_rows = _fixed_effect_rows(
        result, spec, config.significance_alpha
    )
    parent_result = results.get(spec.parent) if spec.parent else None
    if spec.parent and parent_result is None:
        raise RuntimeError(
            f"Parent {spec.parent} was not fit before child model {spec.name}."
        )
    comparison_row = _comparison_row(
        result,
        spec,
        variance_metrics,
        fixed_effect_rows,
        parent_result,
        warning_messages,
    )
    return result, comparison_row, fixed_effect_rows, warning_messages


def _append_fit_log(
    handle: Any,
    spec: ModelSpec,
    comparison_row: dict[str, Any],
    result: Any,
    warning_messages: list[str],
    percentile: int | None,
    z_threshold: float | None,
) -> None:
    handle.write("=" * 88 + "\n")
    handle.write(f"Model: {spec.name}\n")
    handle.write(f"Description: {spec.description}\n")
    handle.write(f"Parent: {spec.parent or 'none'}\n")
    handle.write(f"Formula: {spec.formula}\n")
    handle.write("REML: False; random effects: session intercept; CV: none\n")
    if percentile is None:
        handle.write("Active threshold: not used by this model\n")
    else:
        handle.write(
            f"Active threshold: {_ordinal(percentile)} percentile; "
            f"z={z_threshold:.12g}\n"
        )
    handle.write(
        f"AIC={comparison_row['aic']}; BIC={comparison_row['bic']}; "
        f"marginal_R2={comparison_row['marginal_r2']}; "
        f"conditional_R2={comparison_row['conditional_r2']}; "
        f"conditional_RMSE_ms={comparison_row['conditional_rmse_ms']}\n"
    )
    handle.write(
        f"LRT vs parent: statistic="
        f"{comparison_row['likelihood_ratio_vs_parent']}; "
        f"df={comparison_row['likelihood_ratio_df']}; "
        f"p={comparison_row['likelihood_ratio_p_value']}\n"
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


def _add_default_differences(comparison: pd.DataFrame) -> pd.DataFrame:
    comparison = comparison.copy()
    for column in (
        "delta_aic_vs_50th",
        "delta_bic_vs_50th",
        "delta_marginal_r2_vs_50th",
        "delta_conditional_r2_vs_50th",
        "delta_conditional_rmse_ms_vs_50th",
        "aic_rank_within_base_model",
    ):
        comparison[column] = np.nan

    threshold_rows = comparison[comparison["active_percentile"].notna()]
    for base_model, model_rows in threshold_rows.groupby("base_model", sort=False):
        indices = model_rows.index
        default_rows = model_rows[model_rows["active_percentile"] == 50]
        if not default_rows.empty:
            default = default_rows.iloc[0]
            comparison.loc[indices, "delta_aic_vs_50th"] = (
                model_rows["aic"] - default["aic"]
            )
            comparison.loc[indices, "delta_bic_vs_50th"] = (
                model_rows["bic"] - default["bic"]
            )
            comparison.loc[indices, "delta_marginal_r2_vs_50th"] = (
                model_rows["marginal_r2"] - default["marginal_r2"]
            )
            comparison.loc[indices, "delta_conditional_r2_vs_50th"] = (
                model_rows["conditional_r2"] - default["conditional_r2"]
            )
            comparison.loc[indices, "delta_conditional_rmse_ms_vs_50th"] = (
                model_rows["conditional_rmse_ms"]
                - default["conditional_rmse_ms"]
            )
        comparison.loc[indices, "aic_rank_within_base_model"] = model_rows[
            "aic"
        ].rank(method="min").to_numpy()
    return comparison


def _best_row(
    rows: pd.DataFrame,
    metric: str,
    minimize: bool,
) -> pd.Series:
    index = rows[metric].idxmin() if minimize else rows[metric].idxmax()
    return rows.loc[index]


def _criticality_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    threshold_rows = comparison[comparison["active_percentile"].notna()]
    summaries: list[dict[str, Any]] = []
    for base_model, rows in threshold_rows.groupby("base_model", sort=False):
        best_aic = _best_row(rows, "aic", minimize=True)
        best_bic = _best_row(rows, "bic", minimize=True)
        best_marginal = _best_row(rows, "marginal_r2", minimize=False)
        best_conditional = _best_row(rows, "conditional_r2", minimize=False)
        best_rmse = _best_row(rows, "conditional_rmse_ms", minimize=True)
        default_rows = rows[rows["active_percentile"] == 50]
        default = default_rows.iloc[0] if not default_rows.empty else None
        summaries.append(
            {
                "base_model": base_model,
                "n_thresholds": len(rows),
                "best_aic_percentile": int(best_aic["active_percentile"]),
                "best_aic_z_threshold": best_aic["active_z_threshold"],
                "best_aic": best_aic["aic"],
                "aic_improvement_vs_50th": (
                    default["aic"] - best_aic["aic"]
                    if default is not None
                    else np.nan
                ),
                "best_bic_percentile": int(best_bic["active_percentile"]),
                "best_bic_z_threshold": best_bic["active_z_threshold"],
                "best_bic": best_bic["bic"],
                "best_marginal_r2_percentile": int(
                    best_marginal["active_percentile"]
                ),
                "best_marginal_r2": best_marginal["marginal_r2"],
                "best_conditional_r2_percentile": int(
                    best_conditional["active_percentile"]
                ),
                "best_conditional_r2": best_conditional["conditional_r2"],
                "best_conditional_rmse_percentile": int(
                    best_rmse["active_percentile"]
                ),
                "best_conditional_rmse_ms": best_rmse["conditional_rmse_ms"],
                "default_50th_aic": (
                    default["aic"] if default is not None else np.nan
                ),
                "default_50th_marginal_r2": (
                    default["marginal_r2"] if default is not None else np.nan
                ),
                "default_50th_conditional_r2": (
                    default["conditional_r2"] if default is not None else np.nan
                ),
                "default_50th_conditional_rmse_ms": (
                    default["conditional_rmse_ms"]
                    if default is not None
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(summaries)


def _plot_period_criticality(
    comparison: pd.DataFrame,
    period: str,
    output_dir: Path,
    figure_dpi: int,
) -> Path:
    rows = comparison[
        comparison["base_model"].str.endswith(f"-{period}")
        & comparison["active_percentile"].notna()
    ]
    if rows.empty:
        raise ValueError(f"No threshold-dependent rows found for {period}.")

    metrics = (
        ("aic", "AIC (lower is better)"),
        ("marginal_r2", "Marginal R² (higher is better)"),
        ("conditional_r2", "Conditional R² (higher is better)"),
        ("conditional_rmse_ms", "Conditional RMSE, ms (lower is better)"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    for ax, (metric, ylabel) in zip(axes.ravel(), metrics):
        for base_model, model_rows in rows.groupby("base_model", sort=False):
            ordered = model_rows.sort_values("active_percentile")
            ax.plot(
                ordered["active_percentile"],
                ordered[metric],
                marker="o",
                linewidth=1.5,
                markersize=4,
                label=base_model,
            )
        ax.axvline(50, color="black", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("Active-cell cutoff percentile")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted(rows["active_percentile"].astype(int).unique()))
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle(
        f"{comparison['outcome'].iloc[0].replace('_', ' ')}\n"
        f"Active-cell threshold criticality: {period}\n"
        "Dashed line marks the default 50th percentile (z=0)"
    )
    output_path = output_dir / f"criticality_{period.replace('-', '_')}.png"
    fig.savefig(output_path, dpi=figure_dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_cv_period_criticality(
    summary: pd.DataFrame,
    period: str,
    output_dir: Path,
    figure_dpi: int,
) -> Path:
    rows = summary[
        summary["meta__base_model"].astype(str).str.endswith(f"-{period}")
        & summary["meta__active_percentile"].notna()
    ]
    metrics = (
        ("fixed_rmse_ms_mean", "Held-out fixed RMSE (ms)"),
        ("conditional_rmse_ms_mean", "Held-out conditional RMSE (ms)"),
        ("fixed_r2_mean", "Held-out fixed R²"),
        ("conditional_r2_mean", "Held-out conditional R²"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    for ax, (metric, ylabel) in zip(axes.ravel(), metrics):
        for base_model, model_rows in rows.groupby(
            "meta__base_model", sort=False
        ):
            ordered = model_rows.sort_values("meta__active_percentile")
            ax.plot(
                ordered["meta__active_percentile"],
                ordered[metric],
                marker="o",
                linewidth=1.5,
                markersize=4,
                label=base_model,
            )
        ax.axvline(50, color="black", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("Active-cell cutoff percentile")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle(
        f"{summary['outcome'].iloc[0].replace('_', ' ')}\n"
        f"Cross-validated active-cell threshold comparison: {period}\n"
        "Each point is the mean across trial-holdout shuffles"
    )
    path = output_dir / f"cv_criticality_{period.replace('-', '_')}.png"
    fig.savefig(path, dpi=figure_dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _run_outcome(
    config: Config,
    outcome: OutcomeSpec,
    percentiles: list[int],
    frames: dict[int, pd.DataFrame],
    z_thresholds: dict[int, float],
    prepared_paths: dict[int, Path],
) -> None:
    output_dir = analysis_output_dir(
        config.cache_dir,
        config.output_subdir,
        outcome,
        "active_cell_criticality",
    )
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    log_dir = output_dir / "logs"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    base_specs = _model_specs(outcome.column)
    independent_specs = [spec for spec in base_specs if not _uses_active_fraction(spec)]
    dependent_specs = [spec for spec in base_specs if _uses_active_fraction(spec)]
    dependent_model_names = {spec.name for spec in dependent_specs}

    results: dict[str, Any] = {}
    comparison_rows: list[dict[str, Any]] = []
    all_fixed_effect_rows: list[dict[str, Any]] = []
    total_fits = len(independent_specs) + len(percentiles) * len(dependent_specs)
    fit_position = 0
    reference_percentile = 50 if 50 in frames else percentiles[0]
    log_path = log_dir / "active_cell_criticality.log"

    with log_path.open("w") as log_handle:
        log_handle.write("Active-cell threshold criticality scan\n")
        log_handle.write("=" * 88 + "\n")
        log_handle.write(f"Outcome: {outcome.label} ({outcome.column})\n")
        log_handle.write(f"Percentiles: {percentiles}\n")
        log_handle.write(
            "Z thresholds: "
            + ", ".join(
                f"{_ordinal(p)}={z_thresholds[p]:.12g}" for p in percentiles
            )
            + "\n"
        )
        log_handle.write("REML: False; random intercept: session; CV: none\n")
        log_handle.write(
            "Models without active-fraction predictors are fit once. Models "
            "with current or historical active fraction are refit at every "
            "threshold.\n\n"
        )

        reference_frame = frames[reference_percentile]
        for spec in independent_specs:
            fit_position += 1
            print(
                f"[{outcome.name}] Fitting {fit_position}/{total_fits}: "
                f"{spec.name}"
            )
            result, row, fixed_rows, warning_messages = _fit_one(
                reference_frame, spec, results, config
            )
            row.update(
                {
                    "base_model": spec.name,
                    "active_percentile": np.nan,
                    "active_z_threshold": np.nan,
                    "prepared_data_path": str(prepared_paths[reference_percentile]),
                }
            )
            for fixed_row in fixed_rows:
                fixed_row.update(
                    {
                        "base_model": spec.name,
                        "active_percentile": np.nan,
                        "active_z_threshold": np.nan,
                    }
                )
            _append_fit_log(
                log_handle,
                spec,
                row,
                result,
                warning_messages,
                percentile=None,
                z_threshold=None,
            )
            results[spec.name] = result
            comparison_rows.append(row)
            all_fixed_effect_rows.extend(fixed_rows)

        for percentile in percentiles:
            frame = frames[percentile]
            z_threshold = z_thresholds[percentile]
            for base_spec in dependent_specs:
                fit_position += 1
                spec = _threshold_spec(
                    base_spec,
                    percentile,
                    z_threshold,
                    dependent_model_names,
                )
                print(
                    f"[{outcome.name}] Fitting {fit_position}/{total_fits}: "
                    f"{spec.name}"
                )
                result, row, fixed_rows, warning_messages = _fit_one(
                    frame, spec, results, config
                )
                row.update(
                    {
                        "base_model": base_spec.name,
                        "active_percentile": percentile,
                        "active_z_threshold": z_threshold,
                        "prepared_data_path": str(prepared_paths[percentile]),
                    }
                )
                for fixed_row in fixed_rows:
                    fixed_row.update(
                        {
                            "base_model": base_spec.name,
                            "active_percentile": percentile,
                            "active_z_threshold": z_threshold,
                        }
                    )
                _append_fit_log(
                    log_handle,
                    spec,
                    row,
                    result,
                    warning_messages,
                    percentile=percentile,
                    z_threshold=z_threshold,
                )
                results[spec.name] = result
                comparison_rows.append(row)
                all_fixed_effect_rows.extend(fixed_rows)

    comparison = _add_default_differences(pd.DataFrame(comparison_rows))
    fixed_effects = pd.DataFrame(all_fixed_effect_rows)
    summary = _criticality_summary(comparison)

    summary.insert(0, "outcome", outcome.column)
    comparison.to_csv(
        table_dir / "active_cell_model_comparison.csv", index=False
    )
    comparison.to_pickle(table_dir / "active_cell_model_comparison.pkl")
    fixed_effects.to_csv(table_dir / "fixed_effect_estimates.csv", index=False)
    summary.to_csv(
        table_dir / "active_cell_criticality_summary.csv", index=False
    )
    for period in PERIOD_LABELS:
        _plot_period_criticality(
            comparison, period, figure_dir, config.figure_dpi
        )

    print(f"Saved {len(comparison)} model rows to {output_dir}")
    print(f"Saved {len(summary)} per-model criticality summaries")
    print(f"Saved {len(PERIOD_LABELS)} criticality figures to {figure_dir}")

    if config.run_cv:
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

        cv_requests = [
            CVModelRequest(
                spec=spec,
                active_threshold=0.0,
                metadata={
                    "base_model": spec.name,
                    "active_percentile": np.nan,
                    "active_z_threshold": np.nan,
                },
            )
            for spec in independent_specs
        ]
        for percentile in percentiles:
            z_threshold = z_thresholds[percentile]
            for base_spec in dependent_specs:
                threshold_spec = _threshold_spec(
                    base_spec,
                    percentile,
                    z_threshold,
                    dependent_model_names,
                )
                cv_requests.append(
                    CVModelRequest(
                        spec=threshold_spec,
                        active_threshold=z_threshold,
                        metadata={
                            "base_model": base_spec.name,
                            "active_percentile": percentile,
                            "active_z_threshold": z_threshold,
                        },
                    )
                )
        _, cv_summary, _ = run_trial_holdout_cv(
            config.cache_dir / config.cv_input_subdir / config.cv_input_filename,
            cv_requests,
            output_dir / "cross_validation",
            TrialHoldoutConfig(
                n_shuffles=config.cv_shuffles,
                holdout_fraction=config.cv_holdout_fraction,
                seed=config.cv_seed,
                history_alpha=config.history_alpha,
                max_iterations=config.max_iterations,
                figure_dpi=config.figure_dpi,
                prediction_sample_per_model=(
                    config.cv_prediction_sample_per_model
                ),
            ),
        )
        cv_figure_dir = output_dir / "cross_validation" / "figures"
        cv_figure_dir.mkdir(parents=True, exist_ok=True)
        for period in PERIOD_LABELS:
            _plot_cv_period_criticality(
                cv_summary, period, cv_figure_dir, config.figure_dpi
            )


def main(config: Config) -> None:
    percentiles = _validate_config(config)
    frames, z_thresholds, prepared_paths = _prepare_threshold_data(
        config, percentiles
    )
    threshold_table = pd.DataFrame(
        {
            "active_percentile": percentiles,
            "active_z_threshold": [z_thresholds[p] for p in percentiles],
            "active_rule": [
                f"normalized activity > {z_thresholds[p]:.12g}"
                for p in percentiles
            ],
        }
    )
    threshold_path = (
        config.cache_dir
        / config.prepared_subdir
        / "active_thresholds"
        / "thresholds.csv"
    )
    threshold_path.parent.mkdir(parents=True, exist_ok=True)
    threshold_table.to_csv(threshold_path, index=False)
    for outcome in select_outcomes(config.outcome):
        _run_outcome(
            config,
            outcome,
            percentiles,
            frames,
            z_thresholds,
            prepared_paths,
        )


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        main(tyro.cli(Config))
