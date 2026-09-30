"""Read-only, matched downstream comparison of two completed pipeline runs.

The estimated OFF outcomes change across runs. Compare M1 against each run's
own M0, and distinguish existing-session trial holdouts from new-session/animal
prediction. No decoder setting is selected by these downstream results.
"""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "scripts.next"

import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from .validate_m1_robustness import (
    COUNTS, DEFAULT_ANIMAL_MAP, OUTCOMES, _json_safe, aggregate_sessions,
    analyze_session_model, attach_animal_mapping, leave_one_animal_out, sha256,
)

ROOT = Path(__file__).resolve().parents[2]
MODELS = ("M0", "M1")
METRICS = tuple(f"{kind}_{metric}" for kind in ("fixed", "conditional")
                for metric in ("rmse_ms", "mae_ms", "r2", "session_centered_r2"))


def require_equal(left, right, context="input"):
    """Require exact recursive equality, including array order and NaN locations."""
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            raise ValueError(f"Mismatched {context} keys.")
        for key in left:
            require_equal(left[key], right[key], f"{context}.{key}")
    elif isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            raise ValueError(f"Mismatched {context} length.")
        for index, (a, b) in enumerate(zip(left, right)):
            require_equal(a, b, f"{context}[{index}]")
    elif isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        a, b = np.asarray(left), np.asarray(right)
        equal = np.array_equal(a, b, equal_nan=True) if a.dtype.kind in "fci" and b.dtype.kind in "fci" else np.array_equal(a, b)
        if not equal:
            raise ValueError(f"Mismatched {context} array.")
    elif left != right:
        raise ValueError(f"Mismatched {context}.")


def align_trials(left: pd.DataFrame, right: pd.DataFrame):
    frames = []
    for frame in (left, right):
        frame = frame.copy()
        frame["session"] = frame.session.astype(str)
        if frame[["session", "trial_id"]].duplicated().any():
            raise ValueError("Duplicate prepared session/trial IDs.")
        frames.append(frame.sort_values(["session", "trial_id"]).reset_index(drop=True))
    predictors = [column for column in frames[0] if column not in OUTCOMES]
    if set(frames[0]) != set(frames[1]) or not frames[0][predictors].equals(frames[1][predictors]):
        raise ValueError("Prepared trial identities or predictors differ between runs.")
    return frames


def align_cv_features(left: dict, right: dict):
    # Outcomes legitimately differ; every cached raw feature and holdout must agree.
    def without_outcomes(cache):
        return {**cache, "sessions": [{k: v for k, v in row.items() if k not in OUTCOMES}
                                       for row in cache["sessions"]]}
    require_equal(without_outcomes(left), without_outcomes(right), "CV features/splits")
    splits = left["splits"]
    repeats = [split["repeat"] for split in splits]
    if len(set(repeats)) != len(repeats):
        raise ValueError("Duplicate cached CV repeats.")
    return {int(split["repeat"]): sum(len(ids) for ids in split["test_trial_ids_by_session"].values())
            for split in splits}


def _strict_flags(rows: pd.DataFrame, names):
    result = np.ones(len(rows), dtype=bool)
    for name in names:
        if not rows[name].isin([True, False]).all():
            raise ValueError(f"Invalid boolean {name} flags.")
        result &= rows[name].to_numpy(bool)
    return result


def summarize_cv(rows: pd.DataFrame, expected_n_test: dict[int, int]) -> dict:
    """Pair M0/M1 within each repeat; accumulate SSE before taking relative gains.

    Repeated test observations are not independent. These are descriptive scores,
    not inferential standard errors or confidence intervals across the repeats.
    """
    rows = rows[rows.model.isin(MODELS)].copy()
    if not expected_n_test or any(n <= 0 for n in expected_n_test.values()):
        raise ValueError("Expected positive test counts for each CV repeat.")
    if rows[["model", "repeat"]].duplicated().any():
        raise ValueError("Duplicate M0/M1 repeat rows.")
    valid = _strict_flags(rows, ("fit_success", "converged", "inference_valid"))
    rows["usable"] = valid
    for model in MODELS:
        model_rows = rows[rows.model == model]
        if set(model_rows.repeat) != set(expected_n_test):
            raise ValueError("CV repeats do not match cached splits.")
        if not np.array_equal(model_rows.n_test.to_numpy(), model_rows.repeat.map(expected_n_test).to_numpy()):
            raise ValueError("CV test counts do not match cached split identities.")
    matched = rows[rows.model == "M0"].merge(rows[rows.model == "M1"], on="repeat", suffixes=("_m0", "_m1"), validate="one_to_one")
    matched = matched[matched.usable_m0 & matched.usable_m1].sort_values("repeat")
    if matched.empty:
        raise ValueError("No matched valid M0/M1 CV fits.")
    result = {"n_cached_repeats": len(expected_n_test), "n_paired_usable_repeats": len(matched),
              "usable_repeats": matched.repeat.astype(int).tolist(),
              "eligibility": "fit_success AND converged AND inference_valid for both M0 and M1; these flags jointly require production fixed-design rank, convergence, and inference/identifiability checks",
              "flags": {model: {name: int(rows.loc[rows.model == model, name].sum()) for name in ("fit_success", "converged", "inference_valid", "usable")} for model in MODELS},
              "models": {}, "m1_vs_m0": {}}
    counts = matched.n_test_m0.to_numpy(float)
    for model, suffix in (("M0", "m0"), ("M1", "m1")):
        if not np.isfinite(matched[[f"{metric}_{suffix}" for metric in METRICS]].to_numpy(float)).all():
            raise ValueError("Nonfinite usable CV metrics.")
        result["models"][model] = {metric: float(matched[f"{metric}_{suffix}"].mean()) for metric in METRICS}
    for kind in ("fixed", "conditional"):
        mse0 = matched[f"{kind}_rmse_ms_m0"].to_numpy(float)**2
        mse1 = matched[f"{kind}_rmse_ms_m1"].to_numpy(float)**2
        sse0, sse1 = float(counts @ mse0), float(counts @ mse1)
        if sse0 <= 0:
            raise ValueError("M0 SSE must be positive for relative comparisons.")
        result["m1_vs_m0"][kind] = {
            "rmse_improvement_mean_ms": float((np.sqrt(mse0) - np.sqrt(mse1)).mean()),
            "r2_improvement_mean": float((matched[f"{kind}_r2_m1"] - matched[f"{kind}_r2_m0"]).mean()),
            "pooled_test_rmse_m0_ms": float(np.sqrt(sse0 / counts.sum())),
            "pooled_test_rmse_m1_ms": float(np.sqrt(sse1 / counts.sum())),
            "relative_sse_reduction": 1 - sse1 / sse0 if sse0 > 0 else None,
            "n_repeats_m1_lower_squared_error": int(np.count_nonzero(mse1 < mse0)),
            "m0_sse_ms2": sse0, "m1_sse_ms2": sse1,
            "n_scored_trial_appearances": int(counts.sum()),
        }
    return result


def state_trials(cache: dict) -> pd.DataFrame:
    rows = []
    if cache.get("schema") != "wm-states-next" or cache.get("version") != 1:
        raise ValueError("Unsupported states cache schema.")
    for session in cache["results"]:
        if not (session["off_state_duration_delay_start"] == 500 and session["off_state_duration_delay_end"] == 1400):
            raise ValueError("Expected matched 500--1400ms inclusive delay convention.")
        ids = np.asarray(session["trial_idx"])
        maximum = np.asarray(session["max_off_state_duration_per_trial"], float)
        total = np.asarray(session["off_state_duration_per_trial"], float)
        if (ids.ndim != 1 or ids.shape != maximum.shape or ids.shape != total.shape
                or not np.isfinite([maximum, total]).all() or (maximum < 0).any()
                or (total < maximum).any()):
            raise ValueError("Invalid state outcome alignment.")
        rows.extend({"session": str(session["session"]), "trial_id": int(trial), "preferred_cue": int(session["cue"]),
                     OUTCOMES[0]: float(a), OUTCOMES[1]: float(b)} for trial, a, b in zip(ids, maximum, total))
    frame = pd.DataFrame(rows).sort_values(["session", "trial_id"]).reset_index(drop=True)
    if frame[["session", "trial_id"]].duplicated().any():
        raise ValueError("Duplicate state session/trial IDs.")
    return frame


def duration_summary(frame, outcome):
    values = frame[outcome].to_numpy(float)
    means = frame.groupby("session")[outcome].mean()
    return {"n_trials": len(values), "n_sessions": len(means), "trial_weighted_mean_ms": float(values.mean()),
            "equal_session_mean_ms": float(means.mean()), "median_ms": float(np.median(values)),
            "trial_sd_ms": float(values.std(ddof=1)), "quantile_05_95_ms": np.quantile(values, [.05, .95]).tolist(),
            "maximum_ms": float(values.max()), "n_zero": int(np.count_nonzero(values == 0)),
            "per_session_mean_ms": means.to_dict()}


def build_evidence(run_a: Path, run_b: Path, animal_map_path: Path = DEFAULT_ANIMAL_MAP,
                   *, n_bootstrap: int = 5000) -> dict:
    runs = [Path(run_a), Path(run_b)]
    if runs[0].resolve() == runs[1].resolve() or runs[0].name == runs[1].name:
        raise ValueError("Two distinct run directories with distinct names are required.")
    sources = {}
    def load(path, kind="pickle"):
        path = Path(path)
        sources[str(path)] = sha256(path)
        if kind == "json":
            return json.loads(path.read_text())
        return pd.read_csv(path) if kind == "csv" else pd.read_pickle(path)
    for source in (Path(__file__), Path(__file__).with_name("validate_m1_robustness.py")):
        sources[str(source.relative_to(ROOT))] = sha256(source)
    mapping = load(animal_map_path, "json")
    manifests = [load(run / "pipeline_manifest.json", "json") for run in runs]
    if any(m["status"] != "complete" for m in manifests):
        raise ValueError("Both runs must be complete.")
    settings_diff = []
    require_equal(sorted(manifests[0]["settings"]), sorted(manifests[1]["settings"]), "configured stages")
    for stage in manifests[0]["settings"]:
        a, b = [m["settings"][stage] for m in manifests]
        for key in sorted(set(a) | set(b)):
            if key == "cache_dir":
                continue
            if a.get(key) != b.get(key):
                settings_diff.append({"stage": stage, "setting": key, runs[0].name: a.get(key), runs[1].name: b.get(key)})
    if any((row["stage"], row["setting"]) not in {("decode", "classifier_c"), ("decode", "grid_search_for_c")} for row in settings_diff):
        raise ValueError("This focused comparison requires all settings except decoder C policy to agree.")
    tables = align_trials(*(load(run / "prepare/trial_table.pkl") for run in runs))
    features = [load(run / "prepare/cv_feature_cache.pkl") for run in runs]
    expected_n_test = align_cv_features(*features)
    states = [state_trials(load(run / "states/on_off_states.pkl")) for run in runs]
    require_equal(states[0][["session", "trial_id", "preferred_cue"]].to_numpy(), states[1][["session", "trial_id", "preferred_cue"]].to_numpy(), "state trial IDs/cues")
    result = {"design": {
        "comparison": [run.name for run in runs], "settings_differences_excluding_cache_path": settings_diff,
        "analysis_software": {"python": platform.python_version(), **{name: version(name) for name in ("numpy", "pandas", "scipy", "statsmodels")}},
        "alignment": {"prepared_trials": len(tables[0]), "state_trials": len(states[0]), "sessions": tables[0].session.nunique(),
                      "all_nonoutcome_prepared_columns_identical": True, "all_raw_cv_features_and_split_maps_identical": True, "n_matched_split_maps": len(expected_n_test)},
        "estimands": {"cached_cv": "50 repeated 20% within-session trial holdouts; fixed prediction excludes random intercepts, conditional prediction adds session intercept estimated from other trials in that same session.",
                      "relative_sse": "1 - sum(n_test * RMSE_M1^2) / sum(n_test * RMSE_M0^2), separately for each run, outcome and prediction type. Repeated trial appearances are descriptive, not independent replicates.",
                      "session_ols": "Equal-session mean outcomes; counts-only OLS versus training-session-mean baseline in LOSO and three descriptive leave-one-animal-out folds."},
        "limitations": ["Outcomes differ between runs; lower raw RMSE can reflect a narrower/easier outcome distribution, so also report each run's own M0 and relative SSE reduction.",
                        "All runs share one cohort and only three animals. The overlapping 50 holdouts are not 50 independent replications; no p-value is computed from their differences.",
                        "M0/M1 predict a constant within each session, so session-centered trial R2 is zero. Their count terms explain between-session differences only.",
                        "OLS HC3 and session-bootstrap intervals are nominal conditional summaries, not animal-population uncertainty. Three animal folds are descriptive.",
                        "Cell counts, full-session screening, neural decoder quality, and derived OFF outcomes share data. Association is not proof of causal shortening of biological OFF states.",
                        "No decoder is chosen by favorable OFF durations or M1 significance. These are downstream checks after defining both C procedures."],
        "sources_sha256": sources}, "runs": {}, "paired_duration_change_b_minus_a": {}}
    for index, run in enumerate(runs):
        table, state = tables[index], states[index]
        state_indexed = state.set_index(["session", "trial_id"])
        matched = state_indexed.loc[pd.MultiIndex.from_frame(table[["session", "trial_id"]])]
        require_equal(table[list(OUTCOMES)].to_numpy(), matched[list(OUTCOMES)].to_numpy(), "prepared vs states outcomes")
        for cached_session in features[index]["sessions"]:
            session_states = state[state.session == str(cached_session["session"])].set_index("trial_id")
            ids = np.asarray(cached_session["trial_ids"])
            if set(ids) != set(session_states.index):
                raise ValueError("CV feature trial IDs and decoded states differ.")
            for outcome in OUTCOMES:
                require_equal(cached_session[outcome], session_states.loc[ids, outcome].to_numpy(), "CV vs states outcomes")
        sessions = attach_animal_mapping(aggregate_sessions(table), mapping)
        entry = {"manifest_run_id": manifests[index]["run_id"], "resolved_decode": manifests[index]["settings"]["decode"],
                 "decode_seconds": next(stage["seconds"] for stage in manifests[index]["stages"] if stage["stage"] == "decode"),
                 "original_focal_trial": state[(state.session == "221024") & (state.trial_id == 136)].to_dict("records"),
                 "session_rows": sessions.to_dict("records"), "outcomes": {}}
        for outcome in OUTCOMES:
            folder = run / "nested-activity/outcomes" / outcome.removesuffix("_ms")
            analysis_config = load(folder / "tables/analysis_config.json", "json")
            cv_config = load(folder / "cross_validation/tables/cv_config.json", "json")
            cv = load(folder / "cross_validation/tables/cv_repeat_metrics.csv", "csv")
            expected_formulas = {"M0": f"{outcome} ~ 1", "M1": f"{outcome} ~ preferred_cell_count + selective_nonpreferred_cell_count"}
            for model, formula in expected_formulas.items():
                if set(cv.loc[cv.model == model, "formula"]) != {formula}:
                    raise ValueError("Unexpected cached M0/M1 formula.")
            focus = cv[cv.model.isin(MODELS)]
            if not ((focus.n_train + focus.n_test == len(table)).all()
                    and (focus.n_train_sessions == sessions.session.nunique()).all()
                    and (focus.n_test_sessions == sessions.session.nunique()).all()):
                raise ValueError("Cached CV sample/session counts do not match prepared rows.")
            summary = summarize_cv(cv, expected_n_test)
            model_table = load(folder / "tables/model_comparison.csv", "csv")
            coeff = load(folder / "tables/fixed_effect_estimates.csv", "csv")
            focused_models = model_table[model_table.model.isin(MODELS)]
            if (focused_models.model.tolist() != list(MODELS)
                    or not (focused_models.n_observations == len(table)).all()
                    or not (focused_models.n_sessions == len(sessions)).all()):
                raise ValueError("Cached full M0/M1 fits do not match prepared rows.")
            cols = ["model", "fit_success", "converged", "inference_valid", "inference_error", "likelihood_ratio_vs_parent", "likelihood_ratio_p_value", "marginal_r2", "conditional_r2"]
            report = {"all_decoded_trial_durations": duration_summary(state, outcome),
                      "prepared_trial_durations": duration_summary(table, outcome),
                      "analysis_config": analysis_config, "cv_config": cv_config,
                      "cached_full_fit": model_table.loc[model_table.model.isin(MODELS), cols].to_dict("records"),
                      "cached_m1_coefficients": coeff.loc[coeff.model == "M1"].to_dict("records"),
                      "cached_within_session_cv": summary,
                      "session_equal_m1": analyze_session_model(sessions, outcome, n_bootstrap=n_bootstrap),
                      "leave_one_animal_out_m1": leave_one_animal_out(sessions, outcome)}
            if index:
                prior = result["runs"][runs[0].name]["outcomes"][outcome]
                require_equal(prior["analysis_config"], analysis_config, "M1 analysis configuration")
                require_equal(prior["cv_config"], cv_config, "M1 CV configuration")
                require_equal(prior["cached_within_session_cv"]["usable_repeats"], summary["usable_repeats"], "usable CV repeats across runs")
            entry["outcomes"][outcome] = report
        result["runs"][run.name] = entry
    for outcome in OUTCOMES:
        delta = states[1][outcome] - states[0][outcome]
        session_delta = states[1].groupby("session")[outcome].mean() - states[0].groupby("session")[outcome].mean()
        result["paired_duration_change_b_minus_a"][outcome] = {
            "mean_ms": float(delta.mean()), "median_ms": float(delta.median()),
            "n_shorter": int((delta < 0).sum()), "n_same": int((delta == 0).sum()), "n_longer": int((delta > 0).sum()),
            "n_abs_change_ge50ms": int((delta.abs() >= 50).sum()), "equal_session_mean_ms": float(session_delta.mean()),
            "n_session_mean_shorter": int((session_delta < 0).sum()), "n_session_mean_longer": int((session_delta > 0).sum()),
            "session_mean_changes_ms": session_delta.to_dict()}
    result["paired_cv_change_b_minus_a"] = {}
    for outcome in OUTCOMES:
        prior, current = [result["runs"][run.name]["outcomes"][outcome] for run in runs]
        result["paired_cv_change_b_minus_a"][outcome] = {
            "m1_mean_metrics": {metric: current["cached_within_session_cv"]["models"]["M1"][metric] - prior["cached_within_session_cv"]["models"]["M1"][metric] for metric in METRICS},
            "relative_sse_reduction_vs_own_m0": {kind: current["cached_within_session_cv"]["m1_vs_m0"][kind]["relative_sse_reduction"] - prior["cached_within_session_cv"]["m1_vs_m0"][kind]["relative_sse_reduction"] for kind in ("fixed", "conditional")},
            "loso_r2_vs_own_training_mean": current["session_equal_m1"]["loso"]["r2_vs_heldout_training_mean"] - prior["session_equal_m1"]["loso"]["r2_vs_heldout_training_mean"],
            "loao_session_equal_r2_vs_own_training_mean": current["leave_one_animal_out_m1"]["session_equal"]["r2_vs_training_session_mean"] - prior["leave_one_animal_out_m1"]["session_equal"]["r2_vs_training_session_mean"],
        }
    return _json_safe(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", type=Path, default=Path("cache/next_run_005"))
    parser.add_argument("--run-b", type=Path, default=Path("cache/next_run_006"))
    parser.add_argument("--animal-map", type=Path, default=DEFAULT_ANIMAL_MAP)
    parser.add_argument("--output", type=Path, default=Path("docs/validation/off-m1-run-comparison.json"))
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    args = parser.parse_args()
    evidence = build_evidence(args.run_a, args.run_b, args.animal_map, n_bootstrap=args.n_bootstrap)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
