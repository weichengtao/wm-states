"""Audit cell-count associations using one observation per recording session.

Read-only validation of next_run_001--005. This deliberately uses an equal-session
OLS estimand, distinct from the production trial-level random-intercept MixedLM.
It tests generalization to held-out sessions, not a causal effect of adding cells.
The cached cell selection and decoded outcomes are held fixed throughout.
"""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "scripts.next"

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat, whosmat
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

COUNTS = ("preferred_cell_count", "selective_nonpreferred_cell_count")
NONSELECTIVE = "stationary_nonselective_cell_count"
OUTCOMES = ("maximum_off_state_duration_ms", "total_off_state_duration_ms")
RUNS = tuple(f"next_run_{i:03d}" for i in range(1, 6))
QUALITY_METRICS = ("brier", "balanced_accuracy", "auc")
QUALITY_METHOD = "weighted_search_calibrated"
DEFAULT_DECODER_EVIDENCE = Path("docs/validation/decoder-choice-validation.json")
DEFAULT_ANIMAL_MAP = Path("docs/validation/session-animal-mapping.json")
ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_sessions(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate constant count exposures and retain both session-mean outcomes."""
    required = ["session", "trial_id", "preferred_cue", *COUNTS, NONSELECTIVE, *OUTCOMES]
    if set(required).difference(frame):
        raise ValueError("Missing required prepared trial-table columns.")
    frame = frame.copy()
    frame["session"] = frame["session"].astype(str)
    if frame[["session", "trial_id"]].duplicated().any():
        raise ValueError("Duplicate session/trial IDs.")
    values = frame[[*COUNTS, NONSELECTIVE, *OUTCOMES]].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Counts and durations must be finite and nonnegative.")
    if (frame[list(COUNTS) + [NONSELECTIVE]] % 1 != 0).any().any():
        raise ValueError("Cell counts must be integers.")
    if not np.isin(frame.preferred_cue.to_numpy(), np.arange(1, 9)).all():
        raise ValueError("Preferred cue must be an integer in 1--8.")
    groups = frame.groupby("session", sort=True)
    if (groups.preferred_cue.nunique() != 1).any():
        raise ValueError("Preferred cue must be constant within session.")
    if (groups[[*COUNTS, NONSELECTIVE]].nunique() != 1).any().any():
        raise ValueError("M1 cell counts must be constant within each session.")
    result = groups[["preferred_cue", *COUNTS, NONSELECTIVE]].first()
    result[list(OUTCOMES)] = groups[list(OUTCOMES)].mean()
    result["n_prepared_trials"] = groups.size()
    result["selective_total_cell_count"] = result[list(COUNTS)].sum(axis=1)
    result["decoder_cell_count"] = result[[*COUNTS, NONSELECTIVE]].sum(axis=1)
    result["recording_year"] = "20" + result.index.str[:2]
    return result.reset_index()


def binary_training_pool_size(cues: np.ndarray, correct: np.ndarray, preferred: int) -> int:
    """Eligible correct preferred/opposite trials after one preferred holdout.

    This is the available pool before any downsampling, not all-cue correct
    trials and not an inner CV fold size. It equals the outer fit size in the
    weighted run, which retains all eligible non-test trials.
    """
    cues = np.asarray(cues).ravel()
    correct = np.asarray(correct).ravel()
    if cues.shape != correct.shape or preferred not in range(1, 9):
        raise ValueError("Invalid cue/accuracy metadata.")
    if not np.isin(cues, np.arange(1, 9)).all() or not np.isin(correct, [0, 1]).all():
        raise ValueError("Invalid cue/accuracy metadata.")
    opposite = (preferred + 3) % 8 + 1
    usable = correct.astype(bool) & np.isin(cues, [preferred, opposite])
    if not np.any(usable & (cues == preferred)) or not np.any(usable & (cues == opposite)):
        raise ValueError("Missing preferred or opposite trials.")
    return int(np.count_nonzero(usable)) - 1


def align_decoder_quality(sessions: pd.DataFrame, evidence: dict) -> pd.DataFrame:
    """Attach fixed, separately computed CV quality summaries by session ID.

    These estimates use neural data from the same sessions as the OFF outcomes;
    they are measured quality proxies, not externally randomized covariates.
    """
    frame = sessions.copy()
    frame["session"] = frame.session.astype(str)
    if frame.session.duplicated().any():
        raise ValueError("Duplicate session IDs in M1 input.")
    expected = set(frame.session)
    panel = evidence["summary"]["cached_population"]
    scores = panel["per_session"]
    design_rows = evidence["design"]["sessions"]
    design = {str(row["session"]): row for row in design_rows}
    if len(design) != len(design_rows) or set(design) != expected or set(scores) != expected or panel["n_sessions"] != len(expected):
        raise ValueError("Decoder quality and M1 session IDs do not align exactly.")
    if evidence["design"]["methods"].get(QUALITY_METHOD) != {"weighted": True, "search": True, "calibrate": True}:
        raise ValueError("Decoder quality requires weighted C-search sigmoid-calibrated estimates.")
    for row in frame.itertuples():
        if row.preferred_cue != design[row.session]["cached_cue"]:
            raise ValueError("Decoder quality and M1 preferred cues do not match.")
        if row.decoder_cell_count != design[row.session]["cached_stationary_cells"] or row.recorded_cell_count != design[row.session]["all_cells"]:
            raise ValueError("Decoder quality and M1 population sizes do not match.")
    for metric in QUALITY_METRICS:
        values = np.array([scores[session][QUALITY_METHOD][metric] for session in frame.session], dtype=float)
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError("Decoder quality must be finite probabilities/scores in [0, 1].")
        frame[f"decoder_cv_{metric}"] = values
    return frame


def require_shared_decoder_hashes(evidence: dict, shared_sources: dict[str, str]) -> None:
    """Reject quality estimates produced from missing or different shared data.

    The required sources are the 25 raw MAT inputs and the run005 decoder cache.
    Source paths recorded relative to this repository and absolute paths resolve
    to the same identity; matching population sizes alone is insufficient.
    """
    def canonical(path):
        value = Path(path)
        return (value if value.is_absolute() else ROOT / value).resolve()
    recorded = {}
    for path, digest in evidence["design"]["sources_sha256"].items():
        key = canonical(path)
        if key in recorded and recorded[key] != digest:
            raise ValueError("Conflicting decoder evidence source hashes.")
        recorded[key] = digest
    for path, digest in shared_sources.items():
        key = canonical(path)
        if key not in recorded or recorded[key] != digest:
            raise ValueError(f"Decoder quality shared source missing or changed: {path}")


def attach_animal_mapping(sessions: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Apply only the explicitly confirmed identities; never infer from dates."""
    frame = sessions.copy()
    frame["session"] = frame.session.astype(str)
    identities = mapping["sessions"]
    if frame.session.duplicated().any() or set(frame.session) != set(identities):
        raise ValueError("Animal mapping must cover exactly the analyzed sessions.")
    counts = pd.Series(identities).value_counts().to_dict()
    if counts != mapping["expected_session_counts"]:
        raise ValueError("Animal mapping counts disagree with declared coverage.")
    if set(counts) != {"A", "H", "J"}:
        raise ValueError("Expected the three user-confirmed animals A, H, J.")
    frame["animal"] = frame.session.map(identities)
    return frame


def leave_one_animal_out(frame: pd.DataFrame, outcome: str, predictors: tuple[str, ...] = COUNTS) -> dict:
    """Descriptive transport to an unseen animal; no held-out animal intercept.

    Fit equal-session OLS on the other animals. Both scoring systems compare
    against that same training-session mean; animal-equal scoring changes only
    evaluation weights. Three folds do not support calibrated population CIs.
    """
    if frame.session.duplicated().any() or frame.animal.isna().any():
        raise ValueError("Expected unique sessions with known animal identity.")
    animals = sorted(frame.animal.unique())
    if len(animals) < 3:
        raise ValueError("Require at least three animals for this transport audit.")
    x = _design(frame, predictors)
    y = frame[outcome].to_numpy(float)
    if not np.isfinite(y).all():
        raise ValueError("Outcome must be finite.")
    predicted, baseline = np.empty(len(y)), np.empty(len(y))
    folds = []
    for animal in animals:
        test = (frame.animal == animal).to_numpy()
        train = ~test
        if np.linalg.matrix_rank(x[train]) != x.shape[1]:
            raise ValueError("Animal-holdout training design is rank deficient.")
        coefficients = np.linalg.lstsq(x[train], y[train], rcond=None)[0]
        predicted[test] = x[test] @ coefficients
        baseline[test] = np.mean(y[train])
        errors = y[test] - predicted[test]
        reference_errors = y[test] - baseline[test]
        sse, baseline_sse = float(np.sum(errors**2)), float(np.sum(reference_errors**2))
        outside = np.zeros(int(test.sum()), dtype=bool)
        ranges = {}
        for predictor in predictors:
            training_values = frame.loc[train, predictor].to_numpy(float)
            test_values = frame.loc[test, predictor].to_numpy(float)
            local_outside = (test_values < training_values.min()) | (test_values > training_values.max())
            outside |= local_outside
            ranges[predictor] = {"training_range": [float(training_values.min()), float(training_values.max())],
                                 "heldout_range": [float(test_values.min()), float(test_values.max())],
                                 "n_heldout_sessions_outside_training_range": int(local_outside.sum())}
        folds.append({"heldout_animal": animal, "training_animals": sorted(frame.loc[train, "animal"].unique()),
                      "training_sessions": frame.loc[train, "session"].tolist(), "heldout_sessions": frame.loc[test, "session"].tolist(),
                      "n_training_sessions": int(train.sum()), "n_heldout_sessions": int(test.sum()),
                      "coefficients": dict(zip(("Intercept", *predictors), coefficients.tolist())),
                      "sse_ms2": sse, "baseline_sse_ms2": baseline_sse,
                      "rmse_ms": float(np.sqrt(np.mean(errors**2))), "baseline_rmse_ms": float(np.sqrt(np.mean(reference_errors**2))),
                      "mae_ms": float(np.mean(np.abs(errors))), "baseline_mae_ms": float(np.mean(np.abs(reference_errors))),
                      "r2_vs_training_session_mean": 1 - sse / baseline_sse if baseline_sse > 0 else None,
                      "negative_prediction_count": int(np.count_nonzero(predicted[test] < 0)),
                      "count_ranges": ranges, "n_sessions_with_any_count_outside_training_range": int(outside.sum()),
                      "sessions_with_any_count_outside_training_range": frame.loc[test, "session"].to_numpy()[outside].tolist()})
    session_mse = float(np.mean((y - predicted)**2))
    session_baseline_mse = float(np.mean((y - baseline)**2))
    animal_mse = float(np.mean([f["sse_ms2"] / f["n_heldout_sessions"] for f in folds]))
    animal_baseline_mse = float(np.mean([f["baseline_sse_ms2"] / f["n_heldout_sessions"] for f in folds]))
    return {"n_animals": len(animals), "n_sessions": len(frame),
            "training_weighting": "Each training session receives equal OLS weight.",
            "baseline": "Mean outcome of training sessions; no held-out animal outcomes or intercepts. Same predictions and baseline used for both evaluation weightings.",
            "interpretation": "Descriptive three-animal transport sensitivity; no p-value or population confidence interval.",
            "session_equal": {"rmse_ms": float(np.sqrt(session_mse)), "baseline_rmse_ms": float(np.sqrt(session_baseline_mse)),
                              "r2_vs_training_session_mean": 1 - session_mse / session_baseline_mse if session_baseline_mse > 0 else None,
                              "mae_ms": float(np.mean(np.abs(y - predicted))), "baseline_mae_ms": float(np.mean(np.abs(y - baseline)))},
            "animal_equal": {"rmse_ms": float(np.sqrt(animal_mse)), "baseline_rmse_ms": float(np.sqrt(animal_baseline_mse)),
                             "r2_vs_training_session_mean": 1 - animal_mse / animal_baseline_mse if animal_baseline_mse > 0 else None,
                             "mae_ms": float(np.mean([f["mae_ms"] for f in folds])), "baseline_mae_ms": float(np.mean([f["baseline_mae_ms"] for f in folds]))},
            "folds": folds,
            "session_predictions": [{"session": str(session), "animal": animal, "observed_mean_ms": float(obs), "prediction_ms": float(pred), "baseline_prediction_ms": float(base)}
                                    for session, animal, obs, pred, base in zip(frame.session, frame.animal, y, predicted, baseline)]}


def decoder_by_animal(evidence: dict, mapping: dict) -> dict:
    """Descriptive within-animal means of fixed per-session decoder summaries."""
    output = {"weighting": "Equal sessions within each animal; no population inference from three animals.", "panels": {}}
    identities = mapping["sessions"]
    methods = tuple(evidence["design"]["methods"])
    for scope in ("cached_population", "fixed_cues_all_cells"):
        panel = evidence["summary"][scope]["per_session"]
        if set(panel) != set(identities):
            raise ValueError("Decoder panel and animal-map sessions do not align.")
        animals = {}
        for animal in sorted(set(identities.values())):
            sessions = [session for session in sorted(identities) if identities[session] == animal]
            animals[animal] = {"n_sessions": len(sessions), "methods": {
                method: {metric: float(np.mean([panel[session][method][metric] for session in sessions]))
                         for metric in ("brier", "log_loss", "balanced_accuracy", "auc")}
                for method in methods}}
        output["panels"][scope] = animals
    return output


def _design(frame: pd.DataFrame, predictors: tuple[str, ...]) -> np.ndarray:
    x = np.column_stack([np.ones(len(frame)), frame[list(predictors)].to_numpy(float)])
    if not np.isfinite(x).all() or len(frame) <= x.shape[1] + 1:
        raise ValueError("Insufficient sessions or nonfinite design.")
    if np.linalg.matrix_rank(x) != x.shape[1]:
        raise ValueError("Rank-deficient session design.")
    return x


def permutation_omnibus(
    x: np.ndarray, y: np.ndarray, tested_columns: tuple[int, ...], *,
    n_permutations: int, seed: int, blocks: np.ndarray | None = None,
) -> dict:
    """Permutation partial-F test; blocks preserve recorded-year means.

    Unrestricted outcome permutation assumes exchangeable independent sessions.
    Blocked permutation additionally requires matching block fixed effects in X.
    Neither test supplies animal-level replication or allows unequal variances
    to be ignored. The statistic is a two-sided omnibus test of the count block.
    """
    tested = np.asarray(tested_columns, dtype=int)
    if tested.size == 0 or np.any(tested == 0) or np.unique(tested).size != tested.size:
        raise ValueError("Test at least one distinct non-intercept column.")
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive.")
    reduced = np.delete(x, tested, axis=1)
    if blocks is not None:
        blocks = np.asarray(blocks)
        for block in np.unique(blocks):
            indicator = (blocks == block).astype(float)
            if not np.allclose(reduced @ np.linalg.lstsq(reduced, indicator, rcond=None)[0], indicator):
                raise ValueError("Blocked permutation requires block fixed effects in reduced model.")
    full_residualizer = np.eye(len(y)) - x @ np.linalg.pinv(x)
    reduced_residualizer = np.eye(len(y)) - reduced @ np.linalg.pinv(reduced)

    def statistic(values):
        full = np.sum((values @ full_residualizer) ** 2, axis=-1)
        reduced_sse = np.sum((values @ reduced_residualizer) ** 2, axis=-1)
        return np.maximum(reduced_sse - full, 0) / tested.size / (full / (len(y) - x.shape[1]))

    observed = float(statistic(y))
    rng = np.random.default_rng(seed)
    exceedances = 0
    groups = [np.arange(len(y))] if blocks is None else [np.flatnonzero(blocks == b) for b in np.unique(blocks)]
    for start in range(0, n_permutations, 500):
        count = min(500, n_permutations - start)
        samples = np.tile(y, (count, 1))
        for group in groups:
            samples[:, group] = np.take(y[group], np.argsort(rng.random((count, len(group))), axis=1))
        exceedances += int(np.count_nonzero(statistic(samples) >= observed - 1e-12))
    p = (exceedances + 1) / (n_permutations + 1)
    return {"statistic_partial_f": observed, "n_permutations": n_permutations,
            "exceedances": exceedances, "p_value_plus_one": p,
            "monte_carlo_standard_error_approx": float(np.sqrt(p * (1 - p) / (n_permutations + 1))),
            "exchangeability": "within recording year" if blocks is not None else "among sessions",
            "seed": seed}


def analyze_session_model(
    frame: pd.DataFrame, outcome: str, predictors: tuple[str, ...] = COUNTS, *,
    n_bootstrap: int = 5000, seed: int = 1729, reduced_predictors: tuple[str, ...] = (),
) -> dict:
    """Session-equal OLS, HC3/t intervals, pairs bootstrap, and exact LOSO.

    Public helper for alternative state definitions. Each input row MUST be one
    session; session IDs are required. Predictions never use the held-out outcome.
    Bootstrap intervals are pointwise, conditional on session independence and
    the sampled animals. No inference is attached to overlapping LOSO errors.
    """
    if n_bootstrap < 0:
        raise ValueError("n_bootstrap must be nonnegative.")
    if "session" not in frame or frame.session.duplicated().any():
        raise ValueError("Exactly one row per unique session is required.")
    if not set(reduced_predictors).issubset(predictors):
        raise ValueError("Reduced-model predictors must be a subset of full-model predictors.")
    x = _design(frame, predictors)
    reduced_x = _design(frame, reduced_predictors)
    y = frame[outcome].to_numpy(float)
    if not np.isfinite(y).all() or np.var(y) == 0:
        raise ValueError("Outcome must be finite and nonconstant.")
    fit = sm.OLS(y, x).fit(cov_type="HC3", use_t=True)
    ci = np.asarray(fit.conf_int())
    n = len(y)
    predictions = np.empty(n)
    baseline = np.empty(n)
    reduced_predictions = np.empty(n)
    deletion_coefficients = np.empty((n, x.shape[1]))
    for held_out in range(n):
        keep = np.arange(n) != held_out
        if np.linalg.matrix_rank(x[keep]) != x.shape[1]:
            raise ValueError("A leave-one-session-out training design is rank deficient.")
        coefficient = np.linalg.lstsq(x[keep], y[keep], rcond=None)[0]
        deletion_coefficients[held_out] = coefficient
        predictions[held_out] = x[held_out] @ coefficient
        baseline[held_out] = np.mean(y[keep])
        reduced_predictions[held_out] = reduced_x[held_out] @ np.linalg.lstsq(reduced_x[keep], y[keep], rcond=None)[0]
    bootstrap = []
    invalid_bootstrap = 0
    rng = np.random.default_rng(seed)
    for _ in range(n_bootstrap):
        sample = rng.integers(n, size=n)
        if np.linalg.matrix_rank(x[sample]) != x.shape[1]:
            invalid_bootstrap += 1
            continue
        bootstrap.append(np.linalg.lstsq(x[sample], y[sample], rcond=None)[0])
    bootstrap = np.asarray(bootstrap)
    terms = {}
    for column, name in enumerate(("Intercept", *predictors)):
        terms[name] = {"coefficient_ms_per_cell" if name.endswith("cell_count") else "coefficient": float(fit.params[column]),
                       "hc3_standard_error": float(fit.bse[column]),
                       "hc3_t_p_value": float(fit.pvalues[column]), "hc3_t_ci95": ci[column].tolist(),
                       "leave_one_session_out_coefficient_range": [float(deletion_coefficients[:, column].min()), float(deletion_coefficients[:, column].max())],
                       "leave_one_session_out_negative_count": int(np.count_nonzero(deletion_coefficients[:, column] < 0))}
        if len(bootstrap):
            terms[name]["session_pairs_bootstrap_percentile_ci95"] = np.quantile(bootstrap[:, column], [0.025, 0.975]).tolist()
    residual = y - predictions
    null_residual = y - baseline
    sse = float(residual @ residual)
    null_sse = float(null_residual @ null_residual)
    reduced_sse = float(np.sum((y - reduced_predictions)**2))
    centered = x[:, 1:] - x[:, 1:].mean(axis=0)
    scaled = centered / centered.std(axis=0) if centered.shape[1] else centered
    leverage = np.diag(x @ np.linalg.pinv(x))
    # Cook's distance is diagnostic only, computed from ordinary OLS residuals.
    ols_residual = np.asarray(fit.resid)
    cooks = ols_residual**2 / (x.shape[1] * fit.mse_resid) * leverage / (1 - leverage)**2
    return {"n_sessions": n, "outcome": outcome, "predictors": list(predictors),
            "residual_df": int(fit.df_resid), "ols_r2": float(fit.rsquared), "terms": terms,
            "bootstrap": {"requested": n_bootstrap, "valid": len(bootstrap), "rank_deficient": invalid_bootstrap, "seed": seed},
            "diagnostics": {"standardized_design_condition_number": float(np.linalg.cond(np.column_stack([np.ones(n), scaled]))),
                            "max_leverage": float(leverage.max()), "max_leverage_session": str(frame.iloc[int(np.argmax(leverage))].session), "max_cooks_distance": float(cooks.max()),
                            "max_cooks_session": str(frame.iloc[int(np.argmax(cooks))].session),
                            "predictor_correlation": frame[list(predictors)].corr().fillna(0).to_dict()},
            "loso": {"rmse_ms": float(np.sqrt(sse / n)), "m0_rmse_ms": float(np.sqrt(null_sse / n)),
                     "reduced_model_predictors": list(reduced_predictors),
                     "reduced_model_rmse_ms": float(np.sqrt(reduced_sse / n)),
                     "rmse_improvement_vs_reduced_model_ms": float(np.sqrt(reduced_sse / n) - np.sqrt(sse / n)),
                     "r2_vs_reduced_model": 1 - sse / reduced_sse,
                     "mae_ms": float(np.mean(np.abs(residual))), "m0_mae_ms": float(np.mean(np.abs(null_residual))),
                     "rmse_improvement_ms": float(np.sqrt(null_sse / n) - np.sqrt(sse / n)),
                     "r2_vs_heldout_training_mean": 1 - sse / null_sse,
                     "n_sessions_lower_squared_error_than_m0": int(np.count_nonzero(residual**2 < null_residual**2)),
                     "negative_prediction_count": int(np.count_nonzero(predictions < 0)),
                     "rows": [{"session": str(session), "observed_mean_ms": float(obs), "prediction_ms": float(pred), "m0_prediction_ms": float(base), "reduced_model_prediction_ms": float(red),
                               "coefficients_after_deletion": dict(zip(("Intercept", *predictors), coeff.tolist()))}
                              for session, obs, pred, base, red, coeff in zip(frame.session, y, predictions, baseline, reduced_predictions, deletion_coefficients)]}}


def _cached_models(run: Path, outcome: str, sources: dict) -> dict:
    folder = run / "nested-activity" / "outcomes" / outcome.removesuffix("_ms")
    paths = [folder / "tables/model_comparison.csv", folder / "tables/fixed_effect_estimates.csv",
             folder / "cross_validation/tables/cv_repeat_metrics.csv"]
    for path in paths:
        sources[str(path)] = sha256(path)
    comparison, coefficients, cv = (pd.read_csv(p) for p in paths)
    comparison = comparison[comparison.model.isin(["M0", "M1"])]
    coefficients = coefficients[coefficients.model == "M1"]
    metrics = ("fixed_rmse_ms", "conditional_rmse_ms", "fixed_r2", "conditional_r2",
               "fixed_session_centered_r2", "conditional_session_centered_r2")
    summary = {}
    for model in ("M0", "M1"):
        rows = cv[cv.model == model]
        usable = rows.fit_success.astype(bool) & rows.inference_valid.astype(bool)
        summary[model] = {"n_repeats": len(rows), "n_usable": int(usable.sum()),
                          "metrics": {name: {"mean": float(rows.loc[usable, name].mean()),
                                              "repeat_q025_q975_not_confidence_interval": rows.loc[usable, name].quantile([0.025, 0.975]).tolist()} for name in metrics}}
    joined = cv[cv.model == "M1"].merge(cv[cv.model == "M0"], on="repeat", suffixes=("_m1", "_m0"), validate="one_to_one")
    joined = joined[joined.fit_success_m1.astype(bool) & joined.fit_success_m0.astype(bool)
                    & joined.inference_valid_m1.astype(bool) & joined.inference_valid_m0.astype(bool)]
    summary["paired_usable_repeats"] = len(joined)
    summary["paired_m1_minus_m0"] = {name: float((joined[f"{name}_m1"] - joined[f"{name}_m0"]).mean()) for name in metrics}
    return {"model_comparison": comparison[["model", "converged", "inference_valid", "likelihood_ratio_vs_parent", "likelihood_ratio_p_value", "marginal_r2", "conditional_r2"]].to_dict("records"),
            "m1_fixed_effects": coefficients[["term", "coefficient", "std_error", "p_value", "ci_95_lower", "ci_95_upper", "inference_valid"]].to_dict("records"),
            "within_session_trial_holdouts": summary}


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_evidence(cache_root: Path, data_dir: Path, n_bootstrap: int, n_permutations: int, seed: int,
                   decoder_evidence: Path = DEFAULT_DECODER_EVIDENCE, animal_map_path: Path = DEFAULT_ANIMAL_MAP) -> dict:
    sources = {str(decoder_evidence): sha256(decoder_evidence)}
    quality_evidence = json.loads(decoder_evidence.read_text())
    sources[str(animal_map_path)] = sha256(animal_map_path)
    animal_map = json.loads(animal_map_path.read_text())
    result = {"schema_version": 2, "analysis_date": "2026-09-30", "runs": {}, "animal_mapping": animal_map, "decoder_quality_by_animal": decoder_by_animal(quality_evidence, animal_map), "methods": {
        "analysis_unit": "25 session means, equally weighted; production MixedLM and its trial holdouts retained separately",
        "count_definitions": "Preferred counts cells selective for the decoded dominant cue; selective-nonpreferred counts cells selective for any of the other seven cues, not just the opposite cue. Their sum is the total selected population; adding stationary-nonselective gives the stationary decoder population.",
        "training_pool_covariate": "Correct preferred/opposite trials minus one held-out preferred trial, before downsampling. It equals the weighted outer fit size and excludes inner-CV fold reductions.",
        "primary_model": "M1: intercept + preferred cell count + selective nonpreferred cell count",
        "sensitivity_models": "combined selective count; adjust for stationary decoder count, recorded count, recording-year or equivalent confirmed-animal indicators, or decoder count and eligible binary training-pool size",
        "decoder_quality_sensitivity": {
            "run": "next_run_005", "source": str(decoder_evidence), "panel": "cached_population",
            "shared_input_identity": "Exact preferred cue/population-size/session alignment and matching SHA256 for all 25 raw MAT inputs and run005 decoder cache.",
            "method": QUALITY_METHOD, "metrics_in_separate_models": list(QUALITY_METRICS),
            "design": "All three quality proxies specified before these adjusted fits; report both outcomes and matching quality-only LOSO baselines regardless of sign or significance.",
            "interpretation": "Exploratory conditional prediction after the held-out session's OOF quality estimate is available; not prospective count-only prediction, an independent cohort, or causal adjustment.",
            "limitations": "Quality may mediate a biological effect, reflect measurement precision, or induce collider bias. It shares neural observations with OFF outcomes and is measured with error. Session bootstrap holds quality estimates fixed and omits their estimation uncertainty. Neither attenuation nor persistence distinguishes causal explanations."
        },
        "animal_transport": "Counts-only OLS leave-one-animal-out; fit with equal session weights, score all 25 heldout predictions with session-equal and animal-equal weights against the same training-session mean. Three folds are descriptive only; no animal-level random intercept or population uncertainty estimate.",
        "inference": "HC3/t, session-pairs bootstrap, and session permutations are nominal session-level summaries. They do not account for dependence among sessions from the same three animals; intervals are exploratory and not simultaneous.",
        "permutation": "plus-one omnibus partial-F, unrestricted for M1 and within-year for year-adjusted M1; exchangeability assumptions are required",
        "multiplicity": "Holm adjustment of primary M1 HC3 coefficient tests across 5 runs x 2 outcomes x 2 predictors; other intervals/tests exploratory",
        "prediction": "LOSO refits OLS on 24 session means and compares both to their mean and, for adjusted models, a covariates-only reduced fit; no held-out session random intercept, clipping, hyperparameter tuning, or independent-test claim",
        "limitations": ["Five runs reuse the same sessions and are not independent replications.",
                        "The user confirmed A/H/J identities for 10/8/7 sessions. Twenty-five sessions are not 25 independent biological subjects. Session bootstrap/HC3/permutation summaries do not supply animal-population uncertainty; three animal holdouts are descriptive.",
                        "Confirmed animal identities correspond exactly to recording years in this cohort. Animal and year fixed-effect designs are identical, so animal, time, and associated batch differences cannot be separated.",
                        "Counts and decoder outcomes reuse selected neurons and session-wide screening; association can reflect measurement precision or selection and is not causal proof that biological OFF states shorten.",
                        "Prepared tables omit the first preferred-cue trial of each session for history; all comparisons use the same 1,565 rows.",
                        "Unrestricted and blocked permutations assume outcome exchangeability, whereas HC3 allows heteroskedastic errors; neither solves animal-level dependence.",
                        "M0 and M1 predictions are session-constant; their session-centered trial prediction R2 is algebraically zero. M1 does not identify which trials within one session will have longer OFF durations.",
                        "New OLS session-mean target differs from production trial-level MixedLM; its LOSO R2 cannot be compared numerically to within-session trial CV R2."]}}
    reference = None
    primary_tests = []
    for run_name in RUNS:
        run = cache_root / run_name
        prepared = run / "prepare/trial_table.pkl"
        selection_path = run / "select/cell_screening.pkl"
        sources[str(prepared)] = sha256(prepared)
        sources[str(selection_path)] = sha256(selection_path)
        trials = pd.read_pickle(prepared)
        sessions = attach_animal_mapping(aggregate_sessions(trials), animal_map)
        alignment = trials.assign(session=trials.session.astype(str)).sort_values(["session", "trial_id"])[["session", "trial_id", *COUNTS, NONSELECTIVE]]
        if reference is None:
            reference = alignment.reset_index(drop=True)
        elif not reference.equals(alignment.reset_index(drop=True)):
            raise ValueError("Runs do not share prepared trial IDs and count exposures.")
        selected = {str(row["session"]): row for row in pd.read_pickle(selection_path)["results"]}
        recorded = []
        selected_trials = []
        for row in sessions.itertuples():
            path = data_dir / f"{row.session}.mat"
            if str(path) not in sources:
                sources[str(path)] = sha256(path)
            shape = {name: dims for name, dims, _ in whosmat(path)}["spks"]
            recorded.append(shape[2])
            screen = selected[row.session]
            if row.decoder_cell_count != len(screen["cell_idx_stationary"]):
                raise ValueError("Prepared population sum differs from stationary decoder population.")
            cue = trials.loc[trials.session.astype(str) == row.session, "preferred_cue"].unique()
            if len(cue) != 1:
                raise ValueError("Expected one preferred cue per session.")
            metadata = loadmat(path, variable_names=["cueAngIdx", "isCorr"])
            selected_trials.append(binary_training_pool_size(metadata["cueAngIdx"], metadata["isCorr"], int(cue[0])))
        sessions["recorded_cell_count"] = recorded
        sessions["available_binary_training_trial_count"] = selected_trials
        if run_name == "next_run_005":
            decode_source = run / "decode/decoding_confidence.pkl"
            sources[str(decode_source)] = sha256(decode_source)
            shared = {str(data_dir / f"{session}.mat"): sources[str(data_dir / f"{session}.mat")] for session in sessions.session}
            shared[str(decode_source)] = sources[str(decode_source)]
            require_shared_decoder_hashes(quality_evidence, shared)
            sessions = align_decoder_quality(sessions, quality_evidence)
        year_columns = []
        for year in sorted(sessions.recording_year.unique())[1:]:
            column = f"recording_year_{year}"
            sessions[column] = (sessions.recording_year == year).astype(int)
            year_columns.append(column)
        animal_columns = []
        for animal in sorted(sessions.animal.unique())[1:]:
            column = f"animal_{animal}"
            sessions[column] = (sessions.animal == animal).astype(int)
            animal_columns.append(column)
        if not np.array_equal(sessions[year_columns].to_numpy(), sessions[animal_columns].to_numpy()):
            raise ValueError("Animal/year correspondence changed; do not treat the effects as interchangeable.")
        models = {"M1": COUNTS, "selective_total": ("selective_total_cell_count",),
                  "M1_adjust_decoder_count": (*COUNTS, "decoder_cell_count"),
                  "M1_adjust_recorded_count": (*COUNTS, "recorded_cell_count"),
                  "M1_adjust_recording_year": (*COUNTS, *year_columns),
                  "M1_adjust_animal": (*COUNTS, *animal_columns),
                  "M1_adjust_decoder_and_training_pool": (*COUNTS, "decoder_cell_count", "available_binary_training_trial_count")}
        if run_name == "next_run_005":
            for metric in QUALITY_METRICS:
                models[f"M1_adjust_decoder_cv_{metric}"] = (*COUNTS, f"decoder_cv_{metric}")
        analysis = {"n_trials": len(trials), "n_sessions": len(sessions), "session_rows": sessions.to_dict("records"), "animal_year_fixed_effect_designs_identical": True,
                    "animal_year_correspondence": sessions.groupby("animal").recording_year.first().to_dict(), "outcomes": {}}
        for outcome in OUTCOMES:
            report = {"cached_nested_activity": _cached_models(run, outcome, sources), "session_models": {}, "leave_one_animal_out_M1": leave_one_animal_out(sessions, outcome)}
            for name, predictors in models.items():
                reduced = tuple(p for p in predictors if p not in (*COUNTS, "selective_total_cell_count"))
                model = analyze_session_model(sessions, outcome, predictors, n_bootstrap=n_bootstrap, seed=seed, reduced_predictors=reduced)
                if name in ("M1", "M1_adjust_recording_year"):
                    model["omnibus_permutation"] = permutation_omnibus(_design(sessions, predictors), sessions[outcome].to_numpy(float), (1, 2), n_permutations=n_permutations, seed=seed + 1,
                        blocks=sessions.recording_year.to_numpy() if name.endswith("recording_year") else None)
                if name == "M1":
                    for term in COUNTS:
                        primary_tests.append(model["terms"][term])
                report["session_models"][name] = model
            animal_model = report["session_models"]["M1_adjust_animal"]
            year_model = report["session_models"]["M1_adjust_recording_year"]
            for term in COUNTS:
                if animal_model["terms"][term] != year_model["terms"][term]:
                    raise ValueError("Equivalent animal/year designs produced different count estimates.")
            if animal_model["loso"]["r2_vs_reduced_model"] != year_model["loso"]["r2_vs_reduced_model"]:
                raise ValueError("Equivalent animal/year designs produced different prediction results.")
            analysis["outcomes"][outcome] = report
            primary = report["session_models"]["M1"]
            print(f"{run_name} {outcome}: LOSO R2={primary['loso']['r2_vs_heldout_training_mean']:.4f}, permutation p={primary['omnibus_permutation']['p_value_plus_one']:.5f}", flush=True)
        result["runs"][run_name] = analysis
    adjusted = multipletests([t["hc3_t_p_value"] for t in primary_tests], method="holm")[1]
    for term, p in zip(primary_tests, adjusted):
        term["hc3_t_p_holm_20_primary_tests"] = float(p)
    changed = [path for path, fingerprint in sources.items() if sha256(Path(path)) != fingerprint]
    if changed:
        raise RuntimeError(f"Sources changed while reading: {changed}")
    result["source_sha256"] = sources
    result["source_files_verified_unchanged"] = True
    return _json_safe(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=Path("cache"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/nature"))
    parser.add_argument("--output", type=Path, default=Path("docs/validation/m1-robustness-evidence.json"))
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--n-permutations", type=int, default=19999)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--decoder-evidence", type=Path, default=DEFAULT_DECODER_EVIDENCE,
                        help="Required completed two-class validation JSON; used for three run005 quality-proxy adjustments.")
    parser.add_argument("--session-animal-map", type=Path, default=DEFAULT_ANIMAL_MAP,
                        help="Explicit user-confirmed animal identities; exact session coverage required.")
    args = parser.parse_args()
    evidence = build_evidence(args.cache_root, args.data_dir, args.n_bootstrap, args.n_permutations, args.seed, args.decoder_evidence, args.session_animal_map)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {args.output}; all sources unchanged.", flush=True)


if __name__ == "__main__":
    main()
