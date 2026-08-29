"""Fit and compare nested random-intercept models of off-state duration.

All models are fit by maximum likelihood (``reml=False``) without cross
validation.  M0 contains a fixed intercept and a session random intercept.  M1
adds the three session-level raw cell counts.  Separate baseline, encoding,
pre-delay, and full-delay branches then add four sets of trial-level predictors
cumulatively through M5.
"""

from __future__ import annotations

import json
import math
import re
import textwrap
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import tyro
from scipy.stats import chi2


OUTCOME = "off_state_duration_ms"
SESSION = "session"
GROUP_NAMES = (
    "preferred",
    "selective_nonpreferred",
    "stationary_nonselective",
)
COUNT_PREDICTORS = tuple(f"{group}_cell_count" for group in GROUP_NAMES)
PERIODS = (
    ("baseline", "baseline"),
    ("encoding", "encoding"),
    ("pre-delay", "pre_delay"),
    ("full-delay", "delay"),
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    description: str
    predictors: tuple[str, ...]
    parent: str | None

    @property
    def formula(self) -> str:
        right_hand_side = " + ".join(self.predictors) if self.predictors else "1"
        return f"{OUTCOME} ~ {right_hand_side}"


@dataclass
class Config:
    """Locations, fitting settings, and output settings."""

    cache_dir: Path = Path("cache/run_029_full_session")
    input_subdir: str = "prepare_data_for_mixedlm"
    input_filename: str = "mixedlm_data.pkl"
    output_subdir: str = "compare_mixed_effect_models"
    significance_alpha: float = 0.05
    max_iterations: int = 1000
    figure_dpi: int = 200


def _period_predictors(period_column_name: str) -> tuple[tuple[str, ...], ...]:
    mean_activity = tuple(
        f"{period_column_name}_mean_normalized_activity_{group}"
        for group in GROUP_NAMES
    )
    active_fraction = tuple(
        f"{period_column_name}_active_fraction_{group}" for group in GROUP_NAMES
    )
    mean_history = tuple(f"history_ema_{column}" for column in mean_activity)
    fraction_history = tuple(f"history_ema_{column}" for column in active_fraction)
    return mean_activity, active_fraction, mean_history, fraction_history


def _model_specs() -> list[ModelSpec]:
    specs = [
        ModelSpec(
            name="M0",
            description="Fixed intercept plus session random intercept",
            predictors=(),
            parent=None,
        ),
        ModelSpec(
            name="M1",
            description="M0 plus three raw cell counts",
            predictors=COUNT_PREDICTORS,
            parent="M0",
        ),
    ]
    for period_label, period_column_name in PERIODS:
        mean_activity, active_fraction, mean_history, fraction_history = (
            _period_predictors(period_column_name)
        )
        m2_name = f"M2-{period_label}"
        m3_name = f"M3-{period_label}"
        m4_name = f"M4-{period_label}"
        m5_name = f"M5-{period_label}"
        specs.extend(
            [
                ModelSpec(
                    name=m2_name,
                    description=(
                        f"M1 plus {period_label} mean normalized activity"
                    ),
                    predictors=(*COUNT_PREDICTORS, *mean_activity),
                    parent="M1",
                ),
                ModelSpec(
                    name=m3_name,
                    description=f"{m2_name} plus {period_label} active fraction",
                    predictors=(
                        *COUNT_PREDICTORS,
                        *mean_activity,
                        *active_fraction,
                    ),
                    parent=m2_name,
                ),
                ModelSpec(
                    name=m4_name,
                    description=(
                        f"{m3_name} plus history EMA of {period_label} mean "
                        "normalized activity"
                    ),
                    predictors=(
                        *COUNT_PREDICTORS,
                        *mean_activity,
                        *active_fraction,
                        *mean_history,
                    ),
                    parent=m3_name,
                ),
                ModelSpec(
                    name=m5_name,
                    description=(
                        f"{m4_name} plus history EMA of {period_label} active "
                        "fraction"
                    ),
                    predictors=(
                        *COUNT_PREDICTORS,
                        *mean_activity,
                        *active_fraction,
                        *mean_history,
                        *fraction_history,
                    ),
                    parent=m4_name,
                ),
            ]
        )
    return specs


def _validate_relative_component(value: str, field_name: str) -> None:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field_name} must stay within cache_dir.")


def _load_and_validate_data(config: Config, specs: list[ModelSpec]) -> pd.DataFrame:
    _validate_relative_component(config.input_subdir, "input_subdir")
    _validate_relative_component(config.output_subdir, "output_subdir")
    if Path(config.input_filename).name != config.input_filename:
        raise ValueError("input_filename must be a filename, not a path.")

    input_path = config.cache_dir / config.input_subdir / config.input_filename
    if not input_path.exists():
        raise FileNotFoundError(f"Missing prepared data: {input_path}")
    frame = pd.read_pickle(input_path)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame in {input_path}.")

    required = {
        OUTCOME,
        SESSION,
        "trial_id",
        *(predictor for spec in specs for predictor in spec.predictors),
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Prepared data is missing columns: {missing}")
    if frame.empty or frame[SESSION].nunique() < 2:
        raise ValueError("At least two sessions with trial rows are required.")
    if frame.duplicated([SESSION, "trial_id"]).any():
        raise ValueError("Prepared data contains duplicate session/trial IDs.")

    numeric_columns = sorted(required.difference({SESSION}))
    numeric_values = frame[numeric_columns].to_numpy(dtype=float)
    if not np.all(np.isfinite(numeric_values)):
        raise ValueError("Prepared data contains non-finite required values.")
    return frame.sort_values([SESSION, "trial_id"], kind="stable").reset_index(
        drop=True
    )


def _fit_model(
    frame: pd.DataFrame,
    spec: ModelSpec,
    max_iterations: int,
) -> tuple[Any, list[str]]:
    model = smf.mixedlm(
        spec.formula,
        data=frame,
        groups=frame[SESSION],
        re_formula="1",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.fit(
            reml=False,
            method=["lbfgs", "bfgs", "powell", "nm"],
            maxiter=max_iterations,
            full_output=True,
            disp=False,
        )
    warning_messages = [str(item.message) for item in caught]
    if not result.converged:
        raise RuntimeError(f"{spec.name} did not converge.")
    if not np.isfinite(result.llf) or not np.isfinite(result.scale):
        raise RuntimeError(f"{spec.name} produced non-finite fit statistics.")
    if not np.all(np.isfinite(np.asarray(result.fe_params, dtype=float))):
        raise RuntimeError(f"{spec.name} produced non-finite fixed effects.")
    return result, warning_messages


def _predictions_and_r2(
    result: Any,
    frame: pd.DataFrame,
) -> dict[str, float | np.ndarray]:
    outcome = frame[OUTCOME].to_numpy(dtype=float)
    fixed_design = np.asarray(result.model.exog, dtype=float)
    fixed_coefficients = np.asarray(result.fe_params, dtype=float)
    # Elementwise multiplication avoids a macOS Accelerate/NumPy matmul warning
    # observed after MixedLM optimization while producing the same row products.
    fixed_prediction = np.sum(
        fixed_design * fixed_coefficients[None, :], axis=1
    )
    conditional_prediction = np.asarray(result.fittedvalues, dtype=float)

    fixed_variance = (
        0.0
        if result.fe_params.size == 1
        else float(np.var(fixed_prediction, ddof=0))
    )
    random_covariance = np.asarray(result.cov_re, dtype=float)
    random_variance = float(random_covariance[0, 0])
    residual_variance = float(result.scale)
    total_variance = fixed_variance + random_variance + residual_variance
    if not np.isfinite(total_variance) or total_variance <= 0:
        raise RuntimeError("Model variance components do not have a positive sum.")

    row = {
        "fixed_prediction": fixed_prediction,
        "conditional_prediction": conditional_prediction,
        "fixed_effect_variance": fixed_variance,
        "random_intercept_variance": random_variance,
        "residual_variance": residual_variance,
        "marginal_r2": fixed_variance / total_variance,
        "conditional_r2": (fixed_variance + random_variance) / total_variance,
        "icc": random_variance / (random_variance + residual_variance),
        "marginal_rmse_ms": float(
            np.sqrt(np.mean((outcome - fixed_prediction) ** 2))
        ),
        "conditional_rmse_ms": float(
            np.sqrt(np.mean((outcome - conditional_prediction) ** 2))
        ),
    }
    return row


def _fixed_effect_rows(
    result: Any,
    spec: ModelSpec,
    significance_alpha: float,
) -> list[dict[str, Any]]:
    confidence_intervals = result.conf_int().loc[result.fe_params.index]
    rows = []
    for term in result.fe_params.index:
        p_value = float(result.pvalues[term])
        rows.append(
            {
                "model": spec.name,
                "term": str(term),
                "coefficient": float(result.fe_params[term]),
                "std_error": float(result.bse_fe[term]),
                "z_value": float(result.tvalues[term]),
                "p_value": p_value,
                "ci_95_lower": float(confidence_intervals.loc[term, 0]),
                "ci_95_upper": float(confidence_intervals.loc[term, 1]),
                "significant": bool(p_value < significance_alpha),
            }
        )
    return rows


def _comparison_row(
    result: Any,
    spec: ModelSpec,
    variance_metrics: dict[str, float | np.ndarray],
    fixed_effect_rows: list[dict[str, Any]],
    parent_result: Any | None,
    warning_messages: list[str],
) -> dict[str, Any]:
    likelihood_ratio = np.nan
    likelihood_ratio_df = np.nan
    likelihood_ratio_p_value = np.nan
    if parent_result is not None:
        likelihood_ratio = 2.0 * (float(result.llf) - float(parent_result.llf))
        likelihood_ratio_df = int(result.df_modelwc - parent_result.df_modelwc)
        if likelihood_ratio_df <= 0:
            raise RuntimeError(f"Invalid likelihood-ratio df for {spec.name}.")
        likelihood_ratio_p_value = float(
            chi2.sf(max(likelihood_ratio, 0.0), likelihood_ratio_df)
        )

    fixed_p_values = {
        row["term"]: row["p_value"] for row in fixed_effect_rows
    }
    significant_predictors = [
        row["term"]
        for row in fixed_effect_rows
        if row["term"] != "Intercept" and row["significant"]
    ]
    intercept_row = next(
        row for row in fixed_effect_rows if row["term"] == "Intercept"
    )
    row = {
        "model": spec.name,
        "description": spec.description,
        "parent_model": spec.parent or "",
        "formula": spec.formula,
        "n_observations": int(result.nobs),
        "n_sessions": int(np.unique(result.model.groups).size),
        "n_fixed_effects": int(result.fe_params.size),
        "converged": bool(result.converged),
        "log_likelihood": float(result.llf),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "likelihood_ratio_vs_parent": likelihood_ratio,
        "likelihood_ratio_df": likelihood_ratio_df,
        "likelihood_ratio_p_value": likelihood_ratio_p_value,
        "marginal_r2": float(variance_metrics["marginal_r2"]),
        "conditional_r2": float(variance_metrics["conditional_r2"]),
        "icc": float(variance_metrics["icc"]),
        "fixed_effect_variance": float(
            variance_metrics["fixed_effect_variance"]
        ),
        "random_intercept_variance": float(
            variance_metrics["random_intercept_variance"]
        ),
        "residual_variance": float(variance_metrics["residual_variance"]),
        "marginal_rmse_ms": float(variance_metrics["marginal_rmse_ms"]),
        "conditional_rmse_ms": float(variance_metrics["conditional_rmse_ms"]),
        "intercept": intercept_row["coefficient"],
        "intercept_p_value": intercept_row["p_value"],
        "n_significant_predictors": len(significant_predictors),
        "significant_predictors": "; ".join(significant_predictors),
        "fixed_effect_p_values": json.dumps(fixed_p_values, sort_keys=True),
        "n_fit_warnings": len(warning_messages),
        "fit_warnings": " | ".join(dict.fromkeys(warning_messages)),
    }
    for fixed_effect_row in fixed_effect_rows:
        term = fixed_effect_row["term"]
        row[f"coefficient__{term}"] = fixed_effect_row["coefficient"]
        row[f"p_value__{term}"] = fixed_effect_row["p_value"]
    return row


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).replace("-", "_").lower()


def _plot_observed_vs_fitted(
    ax: Any,
    outcome: np.ndarray,
    fitted: np.ndarray,
    spec: ModelSpec,
    variance_metrics: dict[str, float | np.ndarray],
) -> None:
    ax.scatter(fitted, outcome, s=11, alpha=0.28, edgecolors="none")
    limits = np.asarray(
        [min(float(np.min(fitted)), float(np.min(outcome))),
         max(float(np.max(fitted)), float(np.max(outcome)))],
        dtype=float,
    )
    if np.isclose(limits[0], limits[1]):
        limits += np.asarray([-0.5, 0.5])
    ax.plot(limits, limits, color="black", linestyle="--", linewidth=1)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("Conditional fitted duration (ms)")
    ax.set_ylabel("Observed duration (ms)")
    ax.set_title(
        f"{spec.name}: observed vs fitted\n"
        f"conditional R²={float(variance_metrics['conditional_r2']):.3f}"
    )


def _plot_predictor_effect(
    ax: Any,
    frame: pd.DataFrame,
    result: Any,
    predictor: str,
) -> None:
    outcome = frame[OUTCOME].to_numpy(dtype=float)
    x = frame[predictor].to_numpy(dtype=float)
    coefficients = result.fe_params

    other_contribution = np.zeros(frame.shape[0], dtype=float)
    fixed_reference = float(coefficients["Intercept"])
    for other_predictor in result.model.exog_names:
        if other_predictor in ("Intercept", predictor):
            continue
        other_values = frame[other_predictor].to_numpy(dtype=float)
        coefficient = float(coefficients[other_predictor])
        other_mean = float(np.mean(other_values))
        other_contribution += coefficient * (other_values - other_mean)
        fixed_reference += coefficient * other_mean

    adjusted_outcome = outcome - other_contribution
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if np.isclose(x_min, x_max):
        x_line = np.asarray([x_min - 0.5, x_max + 0.5])
    else:
        x_line = np.linspace(x_min, x_max, 200)
    coefficient = float(coefficients[predictor])
    fitted_line = fixed_reference + coefficient * x_line

    ax.scatter(x, adjusted_outcome, s=10, alpha=0.22, edgecolors="none")
    ax.plot(x_line, fitted_line, color="C3", linewidth=2)
    ax.set_xlabel(textwrap.fill(predictor.replace("_", " "), width=32))
    ax.set_ylabel("Adjusted off-state duration (ms)")
    ax.set_title(
        f"β={coefficient:.3g}, p={float(result.pvalues[predictor]):.3g}"
    )


def _save_model_plot(
    frame: pd.DataFrame,
    result: Any,
    spec: ModelSpec,
    variance_metrics: dict[str, float | np.ndarray],
    output_dir: Path,
    figure_dpi: int,
) -> Path:
    num_panels = 1 + len(spec.predictors)
    num_columns = min(4, num_panels)
    num_rows = math.ceil(num_panels / num_columns)
    fig, axes = plt.subplots(
        num_rows,
        num_columns,
        figsize=(4.2 * num_columns, 3.4 * num_rows),
        squeeze=False,
        layout="constrained",
    )
    flat_axes = axes.ravel()
    _plot_observed_vs_fitted(
        flat_axes[0],
        frame[OUTCOME].to_numpy(dtype=float),
        np.asarray(variance_metrics["conditional_prediction"], dtype=float),
        spec,
        variance_metrics,
    )
    for ax, predictor in zip(flat_axes[1:], spec.predictors):
        _plot_predictor_effect(ax, frame, result, predictor)
    for ax in flat_axes[num_panels:]:
        ax.set_visible(False)

    fig.suptitle(
        f"{spec.name}: {spec.description}\n"
        "Marginal lines use fixed effects with other predictors at their means",
        fontsize=12,
    )
    output_path = output_dir / f"marginal_effects_{_safe_filename(spec.name)}.png"
    fig.savefig(output_path, dpi=figure_dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _write_log_header(
    handle: Any,
    config: Config,
    input_path: Path,
    frame: pd.DataFrame,
) -> None:
    handle.write("Mixed-effects model comparison\n")
    handle.write("=" * 80 + "\n")
    handle.write(f"Input: {input_path}\n")
    handle.write(f"Rows: {len(frame)}\n")
    handle.write(f"Sessions: {frame[SESSION].nunique()}\n")
    handle.write("Estimator: statsmodels MixedLM\n")
    handle.write("REML: False (maximum likelihood)\n")
    handle.write("Random effects: session random intercept only\n")
    handle.write("Cross-validation: none\n")
    handle.write(f"Significance alpha: {config.significance_alpha}\n")
    handle.write(
        "Marginal R2 uses fixed-effect variance; conditional R2 adds the "
        "session random-intercept variance.\n"
    )
    handle.write(
        "Marginal-effect plots adjust observed outcomes for all other fixed "
        "predictors and hold those predictors at their sample means.\n\n"
    )


def _append_model_log(
    handle: Any,
    spec: ModelSpec,
    result: Any,
    comparison_row: dict[str, Any],
    warning_messages: list[str],
    plot_path: Path,
) -> None:
    handle.write("=" * 80 + "\n")
    handle.write(f"{spec.name}: {spec.description}\n")
    handle.write(f"Parent model: {spec.parent or 'none'}\n")
    handle.write(f"Formula: {spec.formula}\n")
    handle.write("Groups: session; random formula: 1\n")
    handle.write("REML: False\n")
    handle.write(f"Marginal-effects figure: {plot_path}\n")
    if warning_messages:
        handle.write("Fit warnings:\n")
        for message in dict.fromkeys(warning_messages):
            handle.write(f"  - {message}\n")
    else:
        handle.write("Fit warnings: none\n")
    handle.write("\nComputed comparison metrics:\n")
    for key in (
        "log_likelihood",
        "aic",
        "bic",
        "likelihood_ratio_vs_parent",
        "likelihood_ratio_df",
        "likelihood_ratio_p_value",
        "marginal_r2",
        "conditional_r2",
        "icc",
        "fixed_effect_variance",
        "random_intercept_variance",
        "residual_variance",
        "marginal_rmse_ms",
        "conditional_rmse_ms",
        "significant_predictors",
    ):
        handle.write(f"  {key}: {comparison_row[key]}\n")
    handle.write("\nStatsmodels fit summary:\n")
    handle.write(result.summary().as_text())
    handle.write("\n\n")


def main(config: Config) -> None:
    if not np.isfinite(config.significance_alpha) or not (
        0 < config.significance_alpha < 1
    ):
        raise ValueError("significance_alpha must be in (0, 1).")
    if config.max_iterations < 1:
        raise ValueError("max_iterations must be positive.")
    if config.figure_dpi < 1:
        raise ValueError("figure_dpi must be positive.")

    specs = _model_specs()
    frame = _load_and_validate_data(config, specs)
    input_path = config.cache_dir / config.input_subdir / config.input_filename
    output_dir = config.cache_dir / config.output_subdir
    figure_dir = output_dir / "marginal_effects"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}
    comparison_rows: list[dict[str, Any]] = []
    all_fixed_effect_rows: list[dict[str, Any]] = []
    log_path = output_dir / "mixed_effect_models.log"
    with log_path.open("w") as log_handle:
        _write_log_header(log_handle, config, input_path, frame)
        for index, spec in enumerate(specs, start=1):
            print(f"Fitting {index}/{len(specs)}: {spec.name}")
            result, warning_messages = _fit_model(
                frame, spec, config.max_iterations
            )
            variance_metrics = _predictions_and_r2(result, frame)
            fixed_effect_rows = _fixed_effect_rows(
                result, spec, config.significance_alpha
            )
            parent_result = results.get(spec.parent) if spec.parent else None
            comparison_row = _comparison_row(
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
                figure_dir,
                config.figure_dpi,
            )
            _append_model_log(
                log_handle,
                spec,
                result,
                comparison_row,
                warning_messages,
                plot_path,
            )
            results[spec.name] = result
            comparison_rows.append(comparison_row)
            all_fixed_effect_rows.extend(fixed_effect_rows)

    comparison = pd.DataFrame(comparison_rows)
    fixed_effects = pd.DataFrame(all_fixed_effect_rows)
    comparison_csv_path = output_dir / "model_comparison.csv"
    comparison_pickle_path = output_dir / "model_comparison.pkl"
    fixed_effects_csv_path = output_dir / "fixed_effects.csv"
    comparison.to_csv(comparison_csv_path, index=False)
    comparison.to_pickle(comparison_pickle_path)
    fixed_effects.to_csv(fixed_effects_csv_path, index=False)

    print(f"Saved detailed log to {log_path}")
    print(f"Saved model comparison table to {comparison_csv_path}")
    print(f"Saved fixed-effect table to {fixed_effects_csv_path}")
    print(f"Saved {len(specs)} marginal-effect figures to {figure_dir}")


if __name__ == "__main__":
    main(tyro.cli(Config))
