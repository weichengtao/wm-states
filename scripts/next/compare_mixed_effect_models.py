"""Fit and compare nested random-intercept models of off-state duration.

All models are fit by maximum likelihood (``reml=False``) without cross
validation.  M0 contains a fixed intercept and a session random intercept.  M1
adds the three session-level raw cell counts.  Separate baseline, encoding,
pre-delay, and full-delay branches then add four sets of trial-level predictors
cumulatively through M5.  Parallel RM2--RM5 branches add the same four blocks
in reverse order so each block can be evaluated under the opposite sequence.
Repeated within-session trial holdouts are additionally fit from a separate
raw-feature cache so preprocessing is learned independently in every fold.
"""

from __future__ import annotations

# Use one module namespace for direct CLI and package execution.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "scripts.next"


from scripts.next.cache_paths import stage_path
from scripts.next.figure_exports import save_figure
import json
import math
import re
import textwrap
import warnings
from dataclasses import dataclass, replace
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

from scripts.next.activity_weighting import (
    weighting_mode,
    weighting_policy,
    weighting_subdir,
)
from scripts.next.mixedlm_outcomes import (
    OutcomeSelection,
    OutcomeSpec,
    TOTAL_OUTCOME,
    analysis_output_dir,
    select_outcomes,
)


OUTCOME = TOTAL_OUTCOME.column
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
    outcome: str = OUTCOME

    @property
    def formula(self) -> str:
        right_hand_side = " + ".join(self.predictors) if self.predictors else "1"
        return f"{self.outcome} ~ {right_hand_side}"


@dataclass
class Config:
    """Locations, fitting settings, and output settings."""

    cache_dir: Path = Path('cache/next_run')
    input_subdir: str = ""  # relative to prepare/ under cache_dir
    input_filename: str = "trial_table.pkl"
    output_subdir: str = ""  # relative to this stage under cache_dir
    cv_input_subdir: str = ""  # relative to prepare/ under cache_dir
    cv_input_filename: str = "cv_feature_cache.pkl"
    outcome: OutcomeSelection = "both"
    run_cv: bool = True
    cv_n_jobs: int = 1  # Parallel model fits within each trial holdout.
    cv_shuffles: int = 50
    cv_holdout_fraction: float = 0.2
    cv_seed: int = 42
    history_alpha: float = 0.2
    cv_prediction_sample_per_model: int = 1000
    significance_alpha: float = 0.05
    max_iterations: int = 1000
    figure_dpi: int = 200
    # Use the separately prepared hybrid PEV-weighted activity features.
    pev_weighted_average: bool = False


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


def _model_specs(outcome: str = OUTCOME) -> list[ModelSpec]:
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
        rm2_name = f"RM2-{period_label}"
        rm3_name = f"RM3-{period_label}"
        rm4_name = f"RM4-{period_label}"
        rm5_name = f"RM5-{period_label}"
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
                ModelSpec(
                    name=rm2_name,
                    description=(
                        f"M1 plus history EMA of {period_label} active fraction"
                    ),
                    predictors=(*COUNT_PREDICTORS, *fraction_history),
                    parent="M1",
                ),
                ModelSpec(
                    name=rm3_name,
                    description=(
                        f"{rm2_name} plus history EMA of {period_label} mean "
                        "normalized activity"
                    ),
                    predictors=(
                        *COUNT_PREDICTORS,
                        *fraction_history,
                        *mean_history,
                    ),
                    parent=rm2_name,
                ),
                ModelSpec(
                    name=rm4_name,
                    description=f"{rm3_name} plus {period_label} active fraction",
                    predictors=(
                        *COUNT_PREDICTORS,
                        *fraction_history,
                        *mean_history,
                        *active_fraction,
                    ),
                    parent=rm3_name,
                ),
                ModelSpec(
                    name=rm5_name,
                    description=(
                        f"{rm4_name} plus {period_label} mean normalized activity"
                    ),
                    predictors=(
                        *COUNT_PREDICTORS,
                        *fraction_history,
                        *mean_history,
                        *active_fraction,
                        *mean_activity,
                    ),
                    parent=rm4_name,
                ),
            ]
        )
    return [replace(spec, outcome=outcome) for spec in specs]


def _validate_relative_component(value: str, field_name: str) -> None:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field_name} must stay within its owning stage directory.")


def _load_and_validate_data(config: Config, specs: list[ModelSpec]) -> pd.DataFrame:
    _validate_relative_component(config.input_subdir, "input_subdir")
    _validate_relative_component(config.output_subdir, "output_subdir")
    if Path(config.input_filename).name != config.input_filename:
        raise ValueError("input_filename must be a filename, not a path.")

    input_path = (
        stage_path(config.cache_dir, "prepare", weighting_subdir(
            config.input_subdir,
            getattr(config, "pev_weighted_average", False),
        ))
        / config.input_filename
    )
    if not input_path.exists():
        raise FileNotFoundError(f"Missing prepared data: {input_path}")
    frame = pd.read_pickle(input_path)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame in {input_path}.")

    required = {
        *(spec.outcome for spec in specs),
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
    if np.linalg.matrix_rank(model.exog) < model.exog.shape[1]:
        raise RuntimeError(f"{spec.name} has a rank-deficient fixed-effect design.")
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
    inference_errors = _inference_errors(result, warning_messages)
    result.inference_valid = not inference_errors
    result.inference_error = "; ".join(inference_errors)
    if inference_errors:
        message = f"{spec.name}: inference withheld: {result.inference_error}"
        warning_messages.append(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    return result, warning_messages


def _inference_errors(result: Any, warning_messages: list[str]) -> list[str]:
    """Check final inference without rejecting harmless optimizer retries."""
    errors = []
    if any("hessian" in message.lower() and "not positive definite" in message.lower()
           for message in warning_messages):
        errors.append("final Hessian is not positive definite")
    try:
        covariance = np.asarray(result.cov_params(), dtype=float)
        if (covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]
                or not np.all(np.isfinite(covariance))
                or not np.allclose(covariance, covariance.T)):
            raise ValueError("nonfinite or asymmetric covariance")
        # Scale to a correlation matrix before factorization so parameter units
        # do not decide the numerical positive-definiteness check.
        if np.any(np.diag(covariance) <= 0):
            raise ValueError("nonpositive parameter variance")
        scales = np.sqrt(np.diag(covariance))
        np.linalg.cholesky(covariance / np.outer(scales, scales))
    except (ValueError, np.linalg.LinAlgError, AttributeError) as error:
        errors.append(f"final parameter covariance is unavailable or not positive definite ({error})")
    try:
        standard_errors = np.asarray(result.bse_fe, dtype=float)
        if (standard_errors.shape != np.asarray(result.fe_params).shape
                or not np.all(np.isfinite(standard_errors)) or np.any(standard_errors <= 0)):
            raise ValueError("nonfinite or nonpositive standard errors")
    except (ValueError, AttributeError) as error:
        errors.append(f"fixed-effect standard errors are unavailable ({error})")
    return errors


def _nested_likelihood_ratio(
    full_log_likelihood: float,
    reduced_log_likelihood: float,
    degrees_of_freedom: float,
    *,
    context: str,
) -> tuple[float, float]:
    """Reject impossible nested improvements; tolerate only floating roundoff."""
    values = np.asarray([full_log_likelihood, reduced_log_likelihood, degrees_of_freedom])
    if not np.all(np.isfinite(values)) or degrees_of_freedom <= 0:
        raise RuntimeError(f"{context}: invalid nested likelihood statistics or degrees of freedom.")
    likelihood_ratio = 2.0 * (full_log_likelihood - reduced_log_likelihood)
    tolerance = 1e-8 * max(1.0, abs(full_log_likelihood), abs(reduced_log_likelihood))
    if likelihood_ratio < -tolerance:
        raise RuntimeError(
            f"{context}: full model has a materially lower log likelihood than its nested "
            f"reduced model (LR={likelihood_ratio:.8g}); refit before comparing models."
        )
    if likelihood_ratio < 0:
        warnings.warn(
            f"{context}: tiny negative likelihood ratio ({likelihood_ratio:.8g}) "
            "was clamped to zero as floating-point roundoff.", RuntimeWarning, stacklevel=2,
        )
        likelihood_ratio = 0.0
    return likelihood_ratio, float(chi2.sf(likelihood_ratio, degrees_of_freedom))


def _predictions_and_r2(
    result: Any,
    frame: pd.DataFrame,
    outcome: str = OUTCOME,
) -> dict[str, float | np.ndarray]:
    observed = frame[outcome].to_numpy(dtype=float)
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
            np.sqrt(np.mean((observed - fixed_prediction) ** 2))
        ),
        "conditional_rmse_ms": float(
            np.sqrt(np.mean((observed - conditional_prediction) ** 2))
        ),
        "marginal_mae_ms": float(np.mean(np.abs(observed - fixed_prediction))),
        "conditional_mae_ms": float(
            np.mean(np.abs(observed - conditional_prediction))
        ),
    }
    return row


def _fixed_effect_rows(
    result: Any,
    spec: ModelSpec,
    significance_alpha: float,
) -> list[dict[str, Any]]:
    inference_valid = bool(getattr(result, "inference_valid", True))
    confidence_intervals = result.conf_int().loc[result.fe_params.index] if inference_valid else None
    rows = []
    for term in result.fe_params.index:
        p_value = float(result.pvalues[term]) if inference_valid else np.nan
        rows.append(
            {
                "model": spec.name,
                "outcome": spec.outcome,
                "term": str(term),
                "coefficient": float(result.fe_params[term]),
                "std_error": float(result.bse_fe[term]) if inference_valid else np.nan,
                "z_value": float(result.tvalues[term]) if inference_valid else np.nan,
                "p_value": p_value,
                "ci_95_lower": float(confidence_intervals.loc[term, 0]) if inference_valid else np.nan,
                "ci_95_upper": float(confidence_intervals.loc[term, 1]) if inference_valid else np.nan,
                "significant": bool(inference_valid and p_value < significance_alpha),
                "inference_valid": inference_valid,
                "inference_error": getattr(result, "inference_error", ""),
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
    likelihood_ratio_valid = False
    likelihood_ratio_error = "parent model unavailable" if spec.parent and parent_result is None else ""
    if parent_result is not None:
        likelihood_ratio_df = int(result.df_modelwc - parent_result.df_modelwc)
        likelihood_ratio, candidate_p_value = _nested_likelihood_ratio(
            float(result.llf), float(parent_result.llf), likelihood_ratio_df, context=spec.name,
        )
        likelihood_ratio_valid = all(bool(getattr(fit, "inference_valid", True))
                                    for fit in (result, parent_result))
        if likelihood_ratio_valid:
            likelihood_ratio_p_value = candidate_p_value
        else:
            likelihood_ratio_error = "inference unavailable for full or reduced model"

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
        "outcome": spec.outcome,
        "description": spec.description,
        "parent_model": spec.parent or "",
        "formula": spec.formula,
        "n_observations": int(result.nobs),
        "n_sessions": int(np.unique(result.model.groups).size),
        "n_fixed_effects": int(result.fe_params.size),
        "converged": bool(result.converged),
        "fit_success": True,
        "fit_error": "",
        "inference_valid": bool(getattr(result, "inference_valid", True)),
        "inference_error": getattr(result, "inference_error", ""),
        "log_likelihood": float(result.llf),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "likelihood_ratio_vs_parent": likelihood_ratio,
        "likelihood_ratio_df": likelihood_ratio_df,
        "likelihood_ratio_p_value": likelihood_ratio_p_value,
        "likelihood_ratio_valid": likelihood_ratio_valid,
        "likelihood_ratio_error": likelihood_ratio_error,
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
        "marginal_mae_ms": float(variance_metrics["marginal_mae_ms"]),
        "conditional_mae_ms": float(variance_metrics["conditional_mae_ms"]),
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


def _fit_model_summary(
    frame: pd.DataFrame, spec: ModelSpec, results: dict[str, Any], config: Any,
) -> tuple[Any | None, dict[str, Any], list[dict[str, Any]], list[str]]:
    """Retain each failed model explicitly without discarding independent fits."""
    try:
        result, messages = _fit_model(frame, spec, config.max_iterations)
        metrics = _predictions_and_r2(result, frame, spec.outcome)
        fixed_rows = _fixed_effect_rows(result, spec, config.significance_alpha)
        row = _comparison_row(
            result, spec, metrics, fixed_rows, results.get(spec.parent), messages,
        )
        return result, row, fixed_rows, messages
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        warnings.warn(f"{spec.name}: model failed: {message}", RuntimeWarning, stacklevel=2)
        numeric_fields = (
            "log_likelihood aic bic likelihood_ratio_vs_parent likelihood_ratio_df "
            "likelihood_ratio_p_value marginal_r2 conditional_r2 icc fixed_effect_variance "
            "random_intercept_variance residual_variance marginal_rmse_ms conditional_rmse_ms "
            "marginal_mae_ms conditional_mae_ms intercept intercept_p_value"
        ).split()
        row = {
            **dict.fromkeys(numeric_fields, np.nan),
            "model": spec.name, "outcome": spec.outcome, "description": spec.description,
            "parent_model": spec.parent or "", "formula": spec.formula,
            "n_observations": len(frame), "n_sessions": frame[SESSION].nunique(),
            "n_fixed_effects": len(spec.predictors) + 1,
            "converged": False, "fit_success": False, "fit_error": message,
            "inference_valid": False, "inference_error": message,
            "likelihood_ratio_valid": False, "likelihood_ratio_error": message,
            "n_significant_predictors": 0, "significant_predictors": "",
            "fixed_effect_p_values": "{}", "n_fit_warnings": 1, "fit_warnings": message,
        }
        fixed_rows = [
            {"model": spec.name, "outcome": spec.outcome, "term": term,
             **dict.fromkeys(("coefficient", "std_error", "z_value", "p_value",
                              "ci_95_lower", "ci_95_upper"), np.nan),
             "significant": False, "inference_valid": False, "inference_error": message}
            for term in ("Intercept", *spec.predictors)
        ]
        return None, row, fixed_rows, [message]


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).replace("-", "_").lower()


def _require_usable_models(comparison: pd.DataFrame, output_dir: Path) -> None:
    """Fail an unusable analysis only after its diagnostic tables are saved."""
    if comparison.empty or not comparison["fit_success"].any():
        raise RuntimeError(
            f"All model fits failed; no usable analysis was produced. "
            f"Inspect fit_error in the saved tables and logs under {output_dir}."
        )


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
    ax.set_ylabel(f"Observed {spec.outcome.replace('_', ' ')}")
    ax.set_title(
        f"{spec.name}: observed vs fitted\n"
        f"conditional R²={float(variance_metrics['conditional_r2']):.3f}"
    )


def _plot_predictor_effect(
    ax: Any,
    frame: pd.DataFrame,
    result: Any,
    predictor: str,
    outcome: str = OUTCOME,
) -> None:
    observed = frame[outcome].to_numpy(dtype=float)
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

    adjusted_outcome = observed - other_contribution
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
    ax.set_ylabel(f"Adjusted {outcome.replace('_', ' ')}")
    inference_label = (
        f"p={float(result.pvalues[predictor]):.3g}"
        if getattr(result, "inference_valid", True) else "inference unavailable"
    )
    ax.set_title(f"β={coefficient:.3g}, {inference_label}")


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
        frame[spec.outcome].to_numpy(dtype=float),
        np.asarray(variance_metrics["conditional_prediction"], dtype=float),
        spec,
        variance_metrics,
    )
    for ax, predictor in zip(flat_axes[1:], spec.predictors):
        _plot_predictor_effect(ax, frame, result, predictor, spec.outcome)
    for ax in flat_axes[num_panels:]:
        ax.set_visible(False)

    fig.suptitle(
        f"{spec.outcome.replace('_', ' ')}\n"
        f"{spec.name}: {spec.description}\n"
        "Marginal lines use fixed effects with other predictors at their means",
        fontsize=12,
    )
    output_path = output_dir / f"marginal_effects_{_safe_filename(spec.name)}.png"
    outputs = save_figure(fig, output_path, dpi=figure_dpi, bbox_inches="tight")
    plt.close(fig)
    return outputs[0]


def _save_coefficient_forest(
    fixed_effect_rows: list[dict[str, Any]],
    spec: ModelSpec,
    output_dir: Path,
    figure_dpi: int,
    outcome_label: str,
) -> Path:
    """Save one 95% Wald-CI forest plot for a model's predictors."""
    rows = pd.DataFrame(fixed_effect_rows)
    rows = rows[rows["term"] != "Intercept"].copy()
    figure_height = max(3.2, 1.8 + 0.55 * len(rows))
    fig, ax = plt.subplots(
        figsize=(10, figure_height),
        layout="constrained",
    )
    if rows.empty:
        ax.text(
            0.5,
            0.5,
            "No non-intercept fixed effects",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
    else:
        y = np.arange(len(rows))
        coefficients = rows["coefficient"].to_numpy(dtype=float)
        lower = rows["ci_95_lower"].to_numpy(dtype=float)
        upper = rows["ci_95_upper"].to_numpy(dtype=float)
        errors = np.maximum(
            np.vstack([coefficients - lower, upper - coefficients]),
            0.0,
        )
        ax.errorbar(
            coefficients,
            y,
            xerr=errors,
            fmt="none",
            ecolor="0.35",
            capsize=3,
            linewidth=1.3,
        )
        valid_inference = rows.get("inference_valid", pd.Series(True, index=rows.index)).to_numpy(dtype=bool)
        colors = np.where(~valid_inference, "0.5", np.where(rows["significant"].to_numpy(dtype=bool), "C3", "C0"))
        ax.scatter(coefficients, y, c=colors, zorder=3)
        ax.scatter([], [], color="C3", label="p < significance alpha")
        ax.scatter([], [], color="C0", label="not significant")
        if not np.all(valid_inference):
            ax.scatter([], [], color="0.5", label="inference unavailable")
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_yticks(y, rows["term"].str.replace("_", " "))
        ax.set_xlabel("Fixed-effect coefficient (95% Wald CI)")
        ax.grid(axis="x", alpha=0.2)
        ax.legend(fontsize=8, loc="best")
    ax.set_title(f"{outcome_label}\n{spec.name}: {spec.description}")
    output_path = output_dir / f"coefficient_forest_{_safe_filename(spec.name)}.png"
    outputs = save_figure(fig, output_path, dpi=figure_dpi, bbox_inches="tight")
    plt.close(fig)
    return outputs[0]


def _write_log_header(
    handle: Any,
    config: Config,
    input_path: Path,
    frame: pd.DataFrame,
    outcome: OutcomeSpec,
) -> None:
    handle.write("Mixed-effects model comparison\n")
    handle.write("=" * 80 + "\n")
    handle.write(f"Input: {input_path}\n")
    handle.write(f"Outcome: {outcome.label} ({outcome.column})\n")
    handle.write(f"Rows: {len(frame)}\n")
    handle.write(f"Sessions: {frame[SESSION].nunique()}\n")
    handle.write("Estimator: statsmodels MixedLM\n")
    handle.write("REML: False (maximum likelihood)\n")
    handle.write("Random effects: session random intercept only\n")
    handle.write("Cross-validation: none\n")
    handle.write(f"Significance alpha: {config.significance_alpha}\n")
    handle.write(
        f"Activity weighting: {weighting_mode(config.pev_weighted_average)}; "
        f"policy={weighting_policy(config.pev_weighted_average)}\n"
    )
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
        "marginal_mae_ms",
        "conditional_mae_ms",
        "significant_predictors",
    ):
        handle.write(f"  {key}: {comparison_row[key]}\n")
    handle.write("\nStatsmodels fit summary:\n")
    if getattr(result, "inference_valid", True):
        handle.write(result.summary().as_text())
    else:
        handle.write(f"Inferential summary withheld: {result.inference_error}")
    handle.write("\n\n")


def _run_outcome(config: Config, outcome: OutcomeSpec) -> None:
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

    specs = _model_specs(outcome.column)
    frame = _load_and_validate_data(config, specs)
    input_path = (
        stage_path(config.cache_dir, "prepare", weighting_subdir(config.input_subdir, config.pev_weighted_average))
        / config.input_filename
    )
    output_dir = weighting_subdir(
        analysis_output_dir(
            config.cache_dir, config.output_subdir, outcome, "models"
        ),
        config.pev_weighted_average,
    )
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures" / "marginal_effects"
    log_dir = output_dir / "logs"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}
    comparison_rows: list[dict[str, Any]] = []
    all_fixed_effect_rows: list[dict[str, Any]] = []
    log_path = log_dir / "model_family_fits.log"
    with log_path.open("w") as log_handle:
        _write_log_header(log_handle, config, input_path, frame, outcome)
        for index, spec in enumerate(specs, start=1):
            print(
                f"[{outcome.name}] Fitting {index}/{len(specs)}: {spec.name}"
            )
            result, comparison_row, fixed_effect_rows, warning_messages = _fit_model_summary(
                frame, spec, results, config,
            )
            if result is None:
                log_handle.write(f"{spec.name}: FAILED: {comparison_row['fit_error']}\n")
                comparison_rows.append(comparison_row)
                all_fixed_effect_rows.extend(fixed_effect_rows)
                continue
            variance_metrics = _predictions_and_r2(result, frame, spec.outcome)
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
    comparison_csv_path = table_dir / "model_family_comparison.csv"
    comparison_pickle_path = table_dir / "model_family_comparison.pkl"
    fixed_effects_csv_path = table_dir / "fixed_effect_estimates.csv"
    comparison.to_csv(comparison_csv_path, index=False)
    comparison.to_pickle(comparison_pickle_path)
    fixed_effects.to_csv(fixed_effects_csv_path, index=False)

    print(f"Saved detailed log to {log_path}")
    print(f"Saved model comparison table to {comparison_csv_path}")
    print(f"Saved fixed-effect table to {fixed_effects_csv_path}")
    print(f"Saved {int(comparison['fit_success'].sum())} marginal-effect figures to {figure_dir}")

    _require_usable_models(comparison, output_dir)
    if config.run_cv:
        try:
            from scripts.next.mixedlm_trial_holdout_cv import (
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

        requests = [
            CVModelRequest(
                spec=spec,
                active_threshold=0.0,
                metadata={"model_family": "standard"},
            )
            for spec in specs
        ]
        run_trial_holdout_cv(
            stage_path(config.cache_dir, "prepare", weighting_subdir(
                config.cv_input_subdir,
                config.pev_weighted_average,
            ))
            / config.cv_input_filename,
            requests,
            output_dir / "cross_validation",
            TrialHoldoutConfig(
                n_shuffles=config.cv_shuffles,
                holdout_fraction=config.cv_holdout_fraction,
                seed=config.cv_seed,
                history_alpha=config.history_alpha,
                max_iterations=config.max_iterations,
                n_jobs=config.cv_n_jobs,
                figure_dpi=config.figure_dpi,
                prediction_sample_per_model=(
                    config.cv_prediction_sample_per_model
                ),
                pev_weighted_average=config.pev_weighted_average,
            ),
        )


def main(config: Config) -> None:
    for outcome in select_outcomes(config.outcome):
        _run_outcome(config, outcome)


if __name__ == "__main__":
    main(tyro.cli(Config))
