"""Read-only views of trusted local next-pipeline results.

Run lists use manifests and small summary tables. Session arrays are loaded only
on demand, with a bounded cache invalidated whenever a primary file changes.
Pickles never leave the server; this viewer must only index trusted local runs.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import pickle
import re
from typing import Any
from urllib.parse import quote
import warnings

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
import numpy as np

from scripts.next import cache_io
from scripts.next.cache_paths import PRIMARY_STAGES, STAGES

STAGE_ORDER = (
    "select", "decode", "evaluate", "states", "activity", "prepare", "models",
    "nested-count", "nested-activity", "criticality", "interactions",
)
KINDS = {
    ".png": "figure", ".tif": "figure", ".tiff": "figure", ".eps": "figure",
    ".pdf": "figure", ".csv": "table", ".json": "json", ".log": "log",
}
METRICS = ("brier_score", "log_loss", "accuracy", "decoding_confidence")


def clean_json(value: Any) -> Any:
    """Keep missing/nonfinite measurements explicit in strict JSON responses."""
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _iso_timestamp(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat()


def _contained_file(root: Path, relative: str | Path) -> Path:
    part = Path(relative)
    if part.is_absolute() or ".." in part.parts or any(p.startswith(".") for p in part.parts):
        raise HTTPException(404, "Result file not found.")
    candidate = root / part
    if not candidate.resolve().is_relative_to(root.resolve()) or not candidate.is_file():
        raise HTTPException(404, "Result file not found.")
    return candidate


def _manifest_status(record: dict) -> str:
    if record.get("status") in {"running", "complete", "failed", "interrupted", "cancelled"}:
        return record["status"]
    statuses = [stage.get("status") for stage in record.get("stages", []) if isinstance(stage, dict)]
    if "failed" in statuses:
        return "failed"
    if "running" in statuses:
        return "running"
    if statuses and all(status == "complete" for status in statuses):
        return "complete"
    return "unknown"


def _read_manifests(root: Path) -> tuple[list[dict], list[str]]:
    errors: list[str] = []
    files = list((root / "manifests").glob("*.json"))
    files.sort(key=lambda path: path.stat().st_mtime)
    latest = root / "pipeline_manifest.json"
    if latest.exists():
        files.append(latest)
    records: dict[str, dict] = {}
    for candidate in files:
        try:
            path = _contained_file(root, candidate.relative_to(root))
            payload = json.loads(path.read_text())
            if not isinstance(payload, dict):
                raise ValueError("Expected a JSON object")
            identity = str(payload.get("run_id") or hashlib.sha256(
                json.dumps(payload, sort_keys=True).encode()
            ).hexdigest())
            record = {
                "id": identity,
                "path": candidate.relative_to(root).as_posix(),
                "started_at": payload.get("started_at"),
                "finished_at": payload.get("finished_at"),
                "updated_at": _iso_timestamp(path.stat().st_mtime),
                "status": _manifest_status(payload),
                "stages": payload.get("stages", []),
                "invocation": payload.get("invocation"),
                "settings": payload.get("settings", {}),
                "runner_config": payload.get("runner_config"),
            }
            # The latest view wins over its history copy and must sort last.
            records.pop(identity, None)
            records[identity] = record
        except (OSError, ValueError, HTTPException) as exc:
            errors.append(f"Could not read {candidate.relative_to(root)}: {exc}")
    return list(records.values()), errors


def _read_summary_csv(root: Path, relative: str, errors: list[str]) -> list[dict]:
    if not (root / relative).exists():
        return []
    try:
        path = _contained_file(root, relative)
        with path.open(newline="") as stream:
            return list(csv.DictReader(stream))
    except (OSError, ValueError, csv.Error, HTTPException) as exc:
        errors.append(f"Could not read {relative}: {exc}")
        return []


def _number(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _sum_available(rows: list[dict], column: str) -> int | None:
    values = [_number(row.get(column)) for row in rows]
    available = [value for value in values if value is not None]
    return int(sum(available)) if available else None


@lru_cache(maxsize=8)
def _load_primary_cached(path: str, modified_ns: int, size: int) -> list[dict]:
    del modified_ns, size  # They deliberately form part of the cache key.
    payload = cache_io.read(Path(path))
    if not isinstance(payload, list) or any(not isinstance(item, dict) for item in payload):
        raise ValueError("Expected a list of session records")
    return payload


class ResultStore:
    def __init__(self, repo_root: Path, cache_root: Path | None = None):
        self.repo_root = Path(repo_root).resolve()
        self.cache_root = Path(cache_root or self.repo_root / "cache").resolve()

    def run_path(self, run_id: str) -> Path:
        if not run_id or run_id.startswith(".") or Path(run_id).name != run_id or "\\" in run_id:
            raise HTTPException(404, "Run not found.")
        candidate = self.cache_root / run_id
        if (candidate.is_symlink() or not candidate.is_dir()
                or candidate.resolve().parent != self.cache_root or not self.is_next_run(candidate)):
            raise HTTPException(404, "Run not found.")
        return candidate

    @staticmethod
    def is_next_run(root: Path) -> bool:
        if any((root / stage).is_dir() and not (root / stage).is_symlink() for stage in STAGES):
            return True
        # A newly started pipeline has its manifest before creating stage output.
        latest = root / "pipeline_manifest.json"
        if latest.is_file() and not latest.is_symlink():
            try:
                record = json.loads(latest.read_text())
                return bool(isinstance(record, dict) and record.get("run_id") and record.get("runner_config"))
            except (OSError, ValueError):
                return False
        return False

    def list_runs(self) -> list[dict]:
        if not self.cache_root.is_dir():
            return []
        run_names = self.dashboard_names()
        runs = []
        for root in self.cache_root.iterdir():
            if root.name.startswith(".") or root.is_symlink() or not root.is_dir() or not self.is_next_run(root):
                continue
            runs.append(self.summary(root, run_names=run_names)[0])
        return sorted(runs, key=lambda run: run["updated_at"], reverse=True)

    def dashboard_names(self) -> dict[str, str]:
        """Index small job records once, keeping the latest name for each run."""
        storage = self.cache_root / ".dashboard"
        if storage.is_symlink() or not storage.is_dir():
            return {}
        names: dict[str, tuple[str, str]] = {}
        for path in storage.glob("*.json"):
            if path.is_symlink() or not path.is_file():
                continue
            try:
                job = json.loads(path.read_text())
                name, directory = job.get("name"), job.get("cache_dir")
                if not isinstance(name, str) or not name.strip() or not isinstance(directory, str):
                    continue
                run_path = Path(directory)
                if not run_path.is_absolute():
                    run_path = self.repo_root / run_path
                run_path = run_path.resolve()
                if run_path.parent != self.cache_root:
                    continue
                created = job.get("created_at") or _iso_timestamp(path.stat().st_mtime)
                if not isinstance(created, str):
                    continue
                identity = str(run_path)
                if identity not in names or created > names[identity][0]:
                    names[identity] = (created, name.strip())
            except (OSError, ValueError, AttributeError):
                continue
        return {directory: name for directory, (_, name) in names.items()}

    def summary(self, root: Path, manifests: list[dict] | None = None,
                run_names: dict[str, str] | None = None) -> tuple[dict, list[str]]:
        if run_names is None:
            run_names = self.dashboard_names()
        manifests, errors = _read_manifests(root) if manifests is None else (manifests, [])
        latest = manifests[-1] if manifests else {}
        by_stage: dict[str, dict] = {}
        for manifest in manifests:
            for stage in manifest.get("stages", []):
                if isinstance(stage, dict) and stage.get("stage") in STAGES:
                    by_stage[stage["stage"]] = {**stage, "manifest_id": manifest["id"]}
        selection = _read_summary_csv(root, "select/tables/cell_screening.csv", errors)
        evaluation = _read_summary_csv(root, "evaluate/tables/eval_confidence.csv", errors)
        sessions = {str(row["session"]) for row in selection + evaluation if row.get("session")}
        null_counts = {_number(row.get("num_null_shuffles")) for row in evaluation}
        null_counts.discard(None)
        null_shuffles = int(next(iter(null_counts))) if len(null_counts) == 1 else None
        if not evaluation:
            null_shuffles = latest.get("settings", {}).get("decode", {}).get("n_decode_shuffle")
        timestamps = [root.stat().st_mtime]
        if (root / "pipeline_manifest.json").is_file():
            timestamps.append((root / "pipeline_manifest.json").stat().st_mtime)
        for stage in STAGES:
            if (root / stage).is_dir():
                timestamps.append((root / stage).stat().st_mtime)
        result = {
            "id": root.name, "name": run_names.get(str(root.resolve()), root.name), "path": str(root),
            "updated_at": _iso_timestamp(max(timestamps)),
            "status": latest.get("status", "unknown"),
            "stages": [by_stage[stage] for stage in STAGE_ORDER if stage in by_stage],
            "session_count": len(sessions),
            "summary": {
                "selected_cells": _sum_available(selection, "num_cells_selected"),
                "decoded_trials": _sum_available(evaluation, "num_trials"),
                "null_shuffles": null_shuffles,
            },
            "errors": errors,
        }
        return clean_json(result), errors

    def load(self, root: Path, filename: str, errors: list[str]) -> list[dict]:
        relative = Path(PRIMARY_STAGES[filename]) / filename
        if not (root / relative).exists():
            return []
        try:
            path = _contained_file(root, relative)
            stat = path.stat()
            return _load_primary_cached(str(path.resolve()), stat.st_mtime_ns, stat.st_size)
        except (OSError, ValueError, EOFError, ImportError, AttributeError, TypeError, pickle.UnpicklingError, HTTPException) as exc:
            errors.append(f"Could not load {relative}: {exc}")
            return []

    def records(self, root: Path, errors: list[str]) -> dict[str, dict[str, dict]]:
        result: dict[str, dict[str, dict]] = {}
        for filename, stage in PRIMARY_STAGES.items():
            for row in self.load(root, filename, errors):
                if "session" not in row:
                    errors.append(f"{stage} cache contains a record without a session ID.")
                    continue
                session = str(row["session"])
                if stage in result.setdefault(session, {}):
                    errors.append(f"Duplicate session {session} in {stage} cache.")
                result[session][stage] = row
        return result

    @staticmethod
    def session_summary(session: str, records: dict[str, dict]) -> dict:
        decode = records.get("decode", {})
        select = records.get("select", {})
        evaluate = records.get("evaluate", {})
        states = records.get("states", {})
        evaluation_aligned = _aligned(decode, evaluate)
        scores = (evaluate.get("observed") or {}) if evaluation_aligned else {}
        return clean_json({
            "id": session, "session": session,
            "cue": decode.get("cue", evaluate.get("cue", states.get("cue"))),
            "trial_count": decode.get("num_trials", evaluate.get("num_trials", 0)),
            "cell_count": decode.get("num_cells"),
            "selected_cells": select.get("num_cells_selected"),
            "null_shuffles": decode.get("n_decode_shuffle", evaluate.get("num_null_shuffles")),
            "metrics": {key: scores.get(key) for key in METRICS},
            "stages": [stage for stage in STAGE_ORDER if stage in records],
            "warnings": [] if evaluation_aligned else [
                "Evaluation and decoding use different cues, trial rows, or time bins. Rerun evaluate."
            ],
        })

    def artifacts(self, root: Path, sessions: list[str]) -> list[dict]:
        artifacts = []
        for candidate in root.rglob("*"):
            if candidate.suffix.lower() not in KINDS or not candidate.is_file():
                continue
            relative = candidate.relative_to(root)
            try:
                path = _contained_file(root, relative)
            except HTTPException:
                continue
            relative_string = relative.as_posix()
            stage = relative.parts[0] if relative.parts[0] in STAGES else "run"
            session = next((value for value in sessions if re.search(
                rf"(?<![a-zA-Z0-9]){re.escape(value)}(?![a-zA-Z0-9])", relative_string
            )), None)
            artifacts.append({
                "id": relative_string, "path": relative_string, "name": path.name,
                "stage": stage, "kind": KINDS[path.suffix.lower()], "session": session,
                "url": f"/api/runs/{quote(root.name, safe='')}/artifacts/{quote(relative_string, safe='/')}",
                "bytes": path.stat().st_size,
            })
        return sorted(artifacts, key=lambda artifact: artifact["path"])

    def detail(self, run_id: str) -> dict:
        root = self.run_path(run_id)
        manifests, errors = _read_manifests(root)
        summary, summary_errors = self.summary(root, manifests)
        errors.extend(summary_errors)
        records = self.records(root, errors)
        sessions = [self.session_summary(session, records[session]) for session in sorted(records)]
        summary["session_count"] = len(sessions)
        if summary["summary"]["decoded_trials"] is None:
            decoded = [row["trial_count"] for row in sessions if "decode" in row["stages"]]
            summary["summary"]["decoded_trials"] = sum(decoded) if decoded else None
        return clean_json({
            "run": summary, "sessions": sessions, "manifests": list(reversed(manifests)),
            "artifacts": self.artifacts(root, list(records)), "errors": errors,
        })

    def session(self, run_id: str, session: str) -> dict:
        root = self.run_path(run_id)
        errors: list[str] = []
        records = self.records(root, errors)
        if session not in records:
            raise HTTPException(404, "Session not found in this run.")
        source = records[session]
        decode = source.get("decode", {})
        evaluate = source.get("evaluate", {})
        states = source.get("states", {})
        result = {
            **self.session_summary(session, source),
            "trial_ids": decode.get("trial_idx", evaluate.get("trial_idx", states.get("trial_idx", []))),
            "time_bins": decode.get("time_bins", evaluate.get("time_bins", states.get("time_bins", []))),
            "observed": [], "null_mean": [], "null_low": [], "null_high": [],
            "accuracy": [], "total_off_durations": [], "max_off_durations": [],
            "errors": errors, "warnings": [],
        }
        if decode:
            observed = np.asarray(decode.get("decoding_confidence", []), dtype=float)
            if observed.ndim == 2 and observed.shape[1] == len(result["time_bins"]):
                result["observed"] = _mean(observed, axis=0)
            elif observed.size:
                errors.append("Observed confidence dimensions do not match the saved time bins.")
            null = np.asarray(decode.get("decoding_confidence_null", []), dtype=float)
            if null.ndim == 3 and null.shape[:2] == observed.shape and null.shape[2]:
                shuffle_curves = _mean(null, axis=0)
                result["null_mean"] = _mean(shuffle_curves, axis=1)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    result["null_low"], result["null_high"] = np.nanpercentile(shuffle_curves, [2.5, 97.5], axis=1)
            elif null.size:
                errors.append("Null confidence dimensions do not match observed estimates.")
            result["accuracy"] = decode.get("decoding_accuracy", [])
        if evaluate and _aligned(decode, evaluate):
            result["accuracy"] = (evaluate.get("observed") or {}).get("accuracy_by_time_bin", result["accuracy"])
        elif decode and evaluate:
            result["metrics"] = {key: None for key in METRICS}
            result["warnings"].append("Evaluation and decoding use different cues, trial rows, or time bins. Evaluation metrics are hidden; rerun evaluate.")
        state_aligned = _aligned(decode, states) and (
            not decode.get("fingerprint") or not states.get("decoding_fingerprint")
            or decode["fingerprint"] == states["decoding_fingerprint"]
        )
        if states and state_aligned:
            result["total_off_durations"] = states.get("off_state_duration_per_trial", [])
            result["max_off_durations"] = states.get("max_off_state_duration_per_trial", [])
        elif decode and states:
            result["warnings"].append("State results do not match this decoding cache. State durations are hidden; rerun states.")
        return clean_json(result)


def _mean(values: np.ndarray, axis: int) -> np.ndarray:
    finite = np.where(np.isfinite(values), values, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(finite, axis=axis)


def _aligned(decode: dict, other: dict) -> bool:
    if not decode or not other:
        return True
    return decode.get("cue") == other.get("cue") and all(
        np.array_equal(decode.get(key, []), other.get(key, [])) for key in ("trial_idx", "time_bins")
    )


def _csv_value(value: str | None) -> Any:
    if value is None or value == "":
        return None
    if value.lower() in {"nan", "inf", "+inf", "-inf", "infinity", "-infinity"}:
        return None
    # Keep IDs and formatted strings (for example 00123) intact.
    if re.fullmatch(r"-?(?:0|[1-9][0-9]*)", value):
        return int(value)
    if re.fullmatch(r"-?0[0-9]+", value):
        return value
    try:
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    except ValueError:
        return value


def create_results_router(repo_root: Path, cache_root: Path | None = None) -> APIRouter:
    store = ResultStore(repo_root, cache_root)
    router = APIRouter(prefix="/api", tags=["results"])

    @router.get("/runs")
    def list_runs():
        return {"runs": store.list_runs()}

    @router.get("/runs/{run_id}")
    def run_detail(run_id: str):
        return store.detail(run_id)

    @router.get("/runs/{run_id}/sessions/{session}")
    def session_detail(run_id: str, session: str):
        return store.session(run_id, session)

    @router.get("/runs/{run_id}/artifacts/{artifact_path:path}")
    def artifact(run_id: str, artifact_path: str):
        root = store.run_path(run_id)
        path = _contained_file(root, artifact_path)
        if path.suffix.lower() not in KINDS:
            raise HTTPException(404, "Result file not found.")
        if path.suffix.lower() == ".png":
            return FileResponse(path, media_type="image/png", headers={"X-Content-Type-Options": "nosniff"})
        return FileResponse(path, filename=path.name,
                            media_type="application/pdf" if path.suffix.lower() == ".pdf" else None,
                            headers={"X-Content-Type-Options": "nosniff"})

    @router.get("/runs/{run_id}/tables/{table_path:path}")
    def table(run_id: str, table_path: str, limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0)):
        root = store.run_path(run_id)
        path = _contained_file(root, table_path)
        if path.suffix.lower() != ".csv":
            raise HTTPException(404, "Table not found.")
        try:
            with path.open(newline="") as stream:
                reader = csv.DictReader(stream)
                columns = reader.fieldnames or []
                rows = []
                total = 0
                for index, row in enumerate(reader):
                    if offset <= index < offset + limit:
                        rows.append({key: _csv_value(row.get(key)) for key in columns})
                    total += 1
        except (OSError, UnicodeDecodeError, csv.Error) as exc:
            raise HTTPException(422, f"Could not read table: {exc}") from exc
        return {"columns": columns, "rows": rows, "total": total, "limit": limit, "offset": offset}

    return router
