"""Test nested across-period activity interactions for each cell group.

The three IM1--IM9 branches use preferred, selective-nonpreferred, or
stationary-nonselective predictors.  Models are fit by maximum likelihood with
a session random intercept. Patsy's ``:`` operator is used so interaction
stages add only the requested two-way product terms. Repeated within-session
trial holdouts provide a separate out-of-sample comparison.
"""

from __future__ import annotations

import textwrap
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
    from scripts.compare_mixed_effect_models import (
        OUTCOME,
        SESSION,
        ModelSpec,
        _comparison_row,
        _fit_model,
        _fixed_effect_rows,
        _predictions_and_r2,
    )
    from scripts.mixedlm_outcomes import (
        OutcomeSelection,
        OutcomeSpec,
        analysis_output_dir,
        select_outcomes,
    )
except ModuleNotFoundError:
    from compare_mixed_effect_models import (
        OUTCOME,
        SESSION,
        ModelSpec,
        _comparison_row,
        _fit_model,
        _fixed_effect_rows,
        _predictions_and_r2,
    )
    from mixedlm_outcomes import (
        OutcomeSelection,
        OutcomeSpec,
        analysis_output_dir,
        select_outcomes,
    )


GROUPS = (
    ("preferred", "preferred"),
    ("selective-nonpreferred", "selective_nonpreferred"),
    ("stationary-nonselective", "stationary_nonselective"),
)
INTERACTION_ONLY_STAGES = frozenset({4, 6, 8, 9})


@dataclass(frozen=True)
class SpecMetadata:
    cell_group: str
    stage: int
    added_terms: tuple[str, ...]
    interaction_terms: tuple[str, ...]


@dataclass
class Config:
    """Input, model-fitting, and output settings."""

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
    history_alpha: float = 0.2
    cv_prediction_sample_per_model: int = 1000
    significance_alpha: float = 0.05
    max_iterations: int = 1000
    figure_dpi: int = 200


def _period_activity_columns(group_column_name: str) -> dict[str, str]:
    return {
        "baseline": f"baseline_mean_normalized_activity_{group_column_name}",
        "encoding": f"encoding_mean_normalized_activity_{group_column_name}",
        "pre_delay": f"pre_delay_mean_normalized_activity_{group_column_name}",
        "full_delay": f"delay_mean_normalized_activity_{group_column_name}",
    }


def _interaction(left: str, right: str) -> str:
    return f"{left}:{right}"


def _model_specs(
    outcome: str = OUTCOME,
) -> tuple[list[ModelSpec], dict[str, SpecMetadata]]:
    specs = [
        ModelSpec(
            name="M0",
            description="Fixed intercept plus session random intercept",
            predictors=(),
            parent=None,
        )
    ]
    metadata = {
        "M0": SpecMetadata(
            cell_group="all",
            stage=0,
            added_terms=(),
            interaction_terms=(),
        )
    }

    for group_label, group_column_name in GROUPS:
        count = f"{group_column_name}_cell_count"
        activity = _period_activity_columns(group_column_name)
        baseline = activity["baseline"]
        encoding = activity["encoding"]
        pre_delay = activity["pre_delay"]
        full_delay = activity["full_delay"]

        baseline_encoding = _interaction(baseline, encoding)
        pre_delay_interactions = (
            _interaction(pre_delay, baseline),
            _interaction(pre_delay, encoding),
        )
        full_delay_interactions = (
            _interaction(full_delay, baseline),
            _interaction(full_delay, encoding),
            _interaction(full_delay, pre_delay),
        )
        count_interactions = (
            _interaction(count, baseline),
            _interaction(count, encoding),
            _interaction(count, pre_delay),
            _interaction(count, full_delay),
        )

        stage_blocks = (
            (count,),
            (baseline,),
            (encoding,),
            (baseline_encoding,),
            (pre_delay,),
            pre_delay_interactions,
            (full_delay,),
            full_delay_interactions,
            count_interactions,
        )
        stage_descriptions = (
            f"M0 plus {group_label} cell count",
            f"IM1-{group_label} plus baseline mean normalized activity",
            f"IM2-{group_label} plus encoding mean normalized activity",
            f"IM3-{group_label} plus baseline-by-encoding interaction",
            f"IM4-{group_label} plus pre-delay mean normalized activity",
            (
                f"IM5-{group_label} plus pre-delay interactions with baseline "
                "and encoding"
            ),
            f"IM6-{group_label} plus full-delay mean normalized activity",
            (
                f"IM7-{group_label} plus full-delay interactions with baseline, "
                "encoding, and pre-delay"
            ),
            (
                f"IM8-{group_label} plus cell-count interactions with all four "
                "period activities"
            ),
        )

        cumulative_predictors: list[str] = []
        cumulative_interactions: list[str] = []
        parent = "M0"
        for stage, (block, description) in enumerate(
            zip(stage_blocks, stage_descriptions), start=1
        ):
            name = f"IM{stage}-{group_label}"
            cumulative_predictors.extend(block)
            cumulative_interactions.extend(term for term in block if ":" in term)
            specs.append(
                ModelSpec(
                    name=name,
                    description=description,
                    predictors=tuple(cumulative_predictors),
                    parent=parent,
                )
            )
            metadata[name] = SpecMetadata(
                cell_group=group_label,
                stage=stage,
                added_terms=tuple(block),
                interaction_terms=tuple(cumulative_interactions),
            )
            parent = name
    return [replace(spec, outcome=outcome) for spec in specs], metadata


def _validate_relative_path(value: str, field_name: str) -> None:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field_name} must stay within cache_dir.")


def _base_columns_from_terms(terms: tuple[str, ...]) -> set[str]:
    columns: set[str] = set()
    for term in terms:
        columns.update(term.split(":"))
    return columns


def _load_data(config: Config, specs: list[ModelSpec]) -> tuple[pd.DataFrame, Path]:
    _validate_relative_path(config.input_subdir, "input_subdir")
    _validate_relative_path(config.output_subdir, "output_subdir")
    if Path(config.input_filename).name != config.input_filename:
        raise ValueError("input_filename must be a filename, not a path.")
    input_path = config.cache_dir / config.input_subdir / config.input_filename
    if not input_path.exists():
        raise FileNotFoundError(f"Missing prepared data: {input_path}")
    frame = pd.read_pickle(input_path)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame in {input_path}.")

    required = {*(spec.outcome for spec in specs), SESSION, "trial_id"}
    for spec in specs:
        required.update(_base_columns_from_terms(spec.predictors))
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Prepared data is missing columns: {missing}")
    if frame.empty or frame[SESSION].nunique() < 2:
        raise ValueError("At least two sessions with trial rows are required.")
    if frame.duplicated([SESSION, "trial_id"]).any():
        raise ValueError("Prepared data contains duplicate session/trial IDs.")
    numeric_columns = sorted(required.difference({SESSION}))
    if not np.all(np.isfinite(frame[numeric_columns].to_numpy(dtype=float))):
        raise ValueError("Prepared data contains non-finite required values.")
    frame = frame.sort_values([SESSION, "trial_id"], kind="stable").reset_index(
        drop=True
    )
    return frame, input_path


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
            f"Parent {spec.parent} was not fit before child {spec.name}."
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


def _append_log(
    handle: Any,
    spec: ModelSpec,
    metadata: SpecMetadata,
    result: Any,
    row: dict[str, Any],
    warning_messages: list[str],
) -> None:
    handle.write("=" * 88 + "\n")
    handle.write(f"Model: {spec.name}\n")
    handle.write(f"Description: {spec.description}\n")
    handle.write(f"Cell group: {metadata.cell_group}\n")
    handle.write(f"Parent: {spec.parent or 'none'}\n")
    handle.write(f"Added terms: {'; '.join(metadata.added_terms) or 'none'}\n")
    handle.write(f"Formula: {spec.formula}\n")
    handle.write("REML: False; random effects: session intercept; CV: none\n")
    handle.write(
        f"AIC={row['aic']}; BIC={row['bic']}; "
        f"marginal_R2={row['marginal_r2']}; "
        f"conditional_R2={row['conditional_r2']}; "
        f"conditional_RMSE_ms={row['conditional_rmse_ms']}\n"
    )
    handle.write(
        f"LRT vs parent: statistic={row['likelihood_ratio_vs_parent']}; "
        f"df={row['likelihood_ratio_df']}; "
        f"p={row['likelihood_ratio_p_value']}\n"
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


def _plot_model_progression(
    comparison: pd.DataFrame,
    output_dir: Path,
    figure_dpi: int,
    outcome_label: str,
) -> Path:
    metrics = (
        ("aic", "AIC (lower is better)"),
        ("marginal_r2", "Marginal R² (higher is better)"),
        ("conditional_r2", "Conditional R² (higher is better)"),
        ("conditional_rmse_ms", "Conditional RMSE, ms (lower is better)"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    m0 = comparison.loc[comparison["model"] == "M0"].iloc[0]
    for ax, (metric, ylabel) in zip(axes.ravel(), metrics):
        for group_label, _ in GROUPS:
            rows = comparison[comparison["cell_group"] == group_label].sort_values(
                "stage"
            )
            x = np.concatenate([[0], rows["stage"].to_numpy(dtype=int)])
            y = np.concatenate([[m0[metric]], rows[metric].to_numpy(dtype=float)])
            ax.plot(x, y, marker="o", linewidth=1.8, label=group_label)
        ax.set_xlabel("Interaction model stage")
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(10), ["M0", *(f"IM{i}" for i in range(1, 10))])
        for stage, label in enumerate(ax.get_xticklabels()):
            if stage in INTERACTION_ONLY_STAGES:
                label.set_color("lightgray")
        ax.grid(alpha=0.2)
    axes[0, 0].legend()
    fig.suptitle(f"{outcome_label}: across-period interaction model progression")
    output_path = output_dir / "model_progression.png"
    fig.savefig(output_path, dpi=figure_dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_likelihood_ratio_tests(
    comparison: pd.DataFrame,
    output_dir: Path,
    figure_dpi: int,
    outcome_label: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
    tiny = np.finfo(float).tiny
    for group_label, _ in GROUPS:
        rows = comparison[comparison["cell_group"] == group_label].sort_values(
            "stage"
        )
        values = -np.log10(
            np.clip(rows["likelihood_ratio_p_value"].to_numpy(dtype=float), tiny, 1)
        )
        ax.plot(
            rows["stage"], values, marker="o", linewidth=1.8, label=group_label
        )
    ax.axhline(-np.log10(0.05), color="black", linestyle="--", linewidth=1)
    ax.set_xticks(range(1, 10), [f"IM{i}" for i in range(1, 10)])
    for stage, label in zip(range(1, 10), ax.get_xticklabels()):
        if stage in INTERACTION_ONLY_STAGES:
            label.set_color("lightgray")
    ax.set_xlabel("Newly completed model stage")
    ax.set_ylabel("−log₁₀ LRT p-value vs parent")
    ax.set_title(
        f"{outcome_label}: joint significance of each added predictor block"
    )
    ax.legend()
    ax.grid(alpha=0.2)
    output_path = output_dir / "likelihood_ratio_progression.png"
    fig.savefig(output_path, dpi=figure_dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _short_interaction_label(term: str, group_column_name: str) -> str:
    replacements = {
        f"{group_column_name}_cell_count": "cell count",
        f"baseline_mean_normalized_activity_{group_column_name}": "baseline",
        f"encoding_mean_normalized_activity_{group_column_name}": "encoding",
        f"pre_delay_mean_normalized_activity_{group_column_name}": "pre-delay",
        f"delay_mean_normalized_activity_{group_column_name}": "full-delay",
    }
    return " × ".join(replacements.get(part, part) for part in term.split(":"))


def _plot_final_interactions(
    fixed_effects: pd.DataFrame,
    output_dir: Path,
    figure_dpi: int,
    outcome_label: str,
) -> list[Path]:
    def draw_panel(
        ax: plt.Axes,
        panel_rows: pd.DataFrame,
        title: str,
    ) -> None:
        panel_rows = panel_rows.iloc[::-1].reset_index(drop=True)
        y = np.arange(len(panel_rows))
        coefficients = panel_rows["coefficient"].to_numpy(dtype=float)
        lower = panel_rows["ci_95_lower"].to_numpy(dtype=float)
        upper = panel_rows["ci_95_upper"].to_numpy(dtype=float)
        significant = panel_rows["significant"].to_numpy(dtype=bool)

        for mask, color in ((~significant, "C0"), (significant, "C3")):
            if not np.any(mask):
                continue
            ax.errorbar(
                coefficients[mask],
                y[mask],
                xerr=np.vstack(
                    [
                        coefficients[mask] - lower[mask],
                        upper[mask] - coefficients[mask],
                    ]
                ),
                fmt="o",
                color=color,
                ecolor=color,
                markersize=5,
                elinewidth=1.5,
                capsize=3,
            )
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_yticks(
            y,
            [textwrap.fill(label, width=30) for label in panel_rows["label"]],
        )
        ax.set_xlabel("Interaction coefficient (ms)")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.2)
        ax.margins(x=0.1)

    output_paths: list[Path] = []
    for group_label, group_column_name in GROUPS:
        model = f"IM9-{group_label}"
        rows = fixed_effects[
            fixed_effects["model"].eq(model)
            & fixed_effects["term"].str.contains(":", regex=False)
        ].copy()
        if rows.empty:
            raise ValueError(f"No final interaction terms found for {model}.")
        rows["label"] = [
            _short_interaction_label(term, group_column_name) for term in rows["term"]
        ]
        count_term = f"{group_column_name}_cell_count"
        is_count_by_activity = rows["term"].str.contains(count_term, regex=False)
        activity_by_activity = rows.loc[~is_count_by_activity]
        count_by_activity = rows.loc[is_count_by_activity]
        if len(activity_by_activity) != 6 or len(count_by_activity) != 4:
            raise ValueError(
                f"Expected 6 activity-by-activity and 4 count-by-activity "
                f"terms for {model}; found {len(activity_by_activity)} and "
                f"{len(count_by_activity)}."
            )

        fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), layout="constrained")
        draw_panel(axes[0], activity_by_activity, "Activity × activity")
        draw_panel(axes[1], count_by_activity, "Cell count × activity")
        fig.suptitle(
            f"{outcome_label}: IM9-{group_label} interaction estimates\n"
            "Panels use independent x-axes; red indicates nominal p < 0.05"
        )
        output_path = output_dir / f"im9_interactions_{group_label}.png"
        fig.savefig(output_path, dpi=figure_dpi, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(output_path)
    return output_paths


def _run_outcome(config: Config, outcome: OutcomeSpec) -> None:
    if not np.isfinite(config.significance_alpha) or not (
        0 < config.significance_alpha < 1
    ):
        raise ValueError("significance_alpha must be in (0, 1).")
    if config.max_iterations < 1:
        raise ValueError("max_iterations must be positive.")
    if config.figure_dpi < 1:
        raise ValueError("figure_dpi must be positive.")
    _validate_relative_path(config.cv_input_subdir, "cv_input_subdir")
    if Path(config.cv_input_filename).name != config.cv_input_filename:
        raise ValueError("cv_input_filename must be a filename, not a path.")

    specs, metadata_by_model = _model_specs(outcome.column)
    frame, input_path = _load_data(config, specs)
    output_dir = analysis_output_dir(
        config.cache_dir,
        config.output_subdir,
        outcome,
        "period_interactions",
    )
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    log_dir = output_dir / "logs"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}
    comparison_rows: list[dict[str, Any]] = []
    all_fixed_effect_rows: list[dict[str, Any]] = []
    log_path = log_dir / "period_interaction_fits.log"
    with log_path.open("w") as log_handle:
        log_handle.write("Across-period interaction MixedLM comparison\n")
        log_handle.write("=" * 88 + "\n")
        log_handle.write(f"Input: {input_path}\n")
        log_handle.write(f"Outcome: {outcome.label} ({outcome.column})\n")
        log_handle.write(f"Rows: {len(frame)}; sessions: {frame[SESSION].nunique()}\n")
        log_handle.write(
            "REML: False; random effects: session intercept; CV: none\n"
        )
        log_handle.write(
            "Interaction syntax uses ':' and adds product terms only.\n\n"
        )

        for position, spec in enumerate(specs, start=1):
            print(
                f"[{outcome.name}] Fitting {position}/{len(specs)}: "
                f"{spec.name}"
            )
            metadata = metadata_by_model[spec.name]
            result, row, fixed_rows, warning_messages = _fit_one(
                frame, spec, results, config
            )
            significant_interactions = [
                fixed_row["term"]
                for fixed_row in fixed_rows
                if ":" in fixed_row["term"] and fixed_row["significant"]
            ]
            row.update(
                {
                    "cell_group": metadata.cell_group,
                    "stage": metadata.stage,
                    "added_terms": "; ".join(metadata.added_terms),
                    "n_added_terms": len(metadata.added_terms),
                    "interaction_terms": "; ".join(metadata.interaction_terms),
                    "n_interaction_terms": len(metadata.interaction_terms),
                    "significant_interactions": "; ".join(
                        significant_interactions
                    ),
                    "n_significant_interactions": len(significant_interactions),
                }
            )
            for fixed_row in fixed_rows:
                fixed_row.update(
                    {
                        "cell_group": metadata.cell_group,
                        "stage": metadata.stage,
                        "is_interaction": ":" in fixed_row["term"],
                    }
                )
            _append_log(
                log_handle,
                spec,
                metadata,
                result,
                row,
                warning_messages,
            )
            results[spec.name] = result
            comparison_rows.append(row)
            all_fixed_effect_rows.extend(fixed_rows)

    comparison = pd.DataFrame(comparison_rows)
    fixed_effects = pd.DataFrame(all_fixed_effect_rows)
    interaction_effects = fixed_effects[fixed_effects["is_interaction"]].copy()
    final_models = comparison[comparison["stage"] == 9].copy()

    comparison.to_csv(table_dir / "period_model_comparison.csv", index=False)
    comparison.to_pickle(table_dir / "period_model_comparison.pkl")
    fixed_effects.to_csv(table_dir / "fixed_effect_estimates.csv", index=False)
    interaction_effects.to_csv(
        table_dir / "period_interaction_effects.csv", index=False
    )
    final_models.to_csv(table_dir / "final_im9_models.csv", index=False)

    _plot_model_progression(
        comparison, figure_dir, config.figure_dpi, outcome.label
    )
    _plot_likelihood_ratio_tests(
        comparison, figure_dir, config.figure_dpi, outcome.label
    )
    _plot_final_interactions(
        fixed_effects, figure_dir, config.figure_dpi, outcome.label
    )

    print(f"Saved {len(comparison)} model rows to {output_dir}")
    print(f"Saved {len(interaction_effects)} interaction-effect rows")
    print(f"Saved 5 comparison figures to {figure_dir}")

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

        requests = []
        for spec in specs:
            model_metadata = metadata_by_model[spec.name]
            requests.append(
                CVModelRequest(
                    spec=spec,
                    active_threshold=0.0,
                    metadata={
                        "cell_group": model_metadata.cell_group,
                        "stage": model_metadata.stage,
                        "added_terms": "; ".join(model_metadata.added_terms),
                    },
                )
            )
        run_trial_holdout_cv(
            config.cache_dir / config.cv_input_subdir / config.cv_input_filename,
            requests,
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


def main(config: Config) -> None:
    for outcome in select_outcomes(config.outcome):
        _run_outcome(config, outcome)


if __name__ == "__main__":
    main(tyro.cli(Config))
