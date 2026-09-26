"""Contracts for browsing real next outputs without exposing arbitrary files."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np

from scripts.next import cache_io
from scripts.next.dashboard.results import ResultStore, _load_primary_cached, create_results_router


class DashboardResultsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / "cache" / "example"
        self.run.mkdir(parents=True)
        self.selection = [{"session": "0102", "num_cells_selected": 7}]
        self.decoded = [{
            "session": "0102", "cue": 7, "trial_idx": np.array([10, 20]),
            "time_bins": np.array([500, 550]), "num_trials": 2, "num_cells": 9,
            "n_decode_shuffle": 2, "fingerprint": "current",
            "decoding_confidence": np.array([[0.8, 0.4], [0.6, 0.8]]),
            "decoding_confidence_null": np.array([[[0.4, 0.6], [0.3, 0.7]], [[0.2, 0.8], [0.5, 0.5]]]),
            "decoding_accuracy": np.array([1.0, 0.5]),
        }]
        self.evaluation = [{
            "session": "0102", "cue": 7, "trial_idx": np.array([10, 20]),
            "time_bins": np.array([500, 550]), "num_trials": 2, "num_null_shuffles": 2,
            "observed": {"brier_score": 0.12, "log_loss": 0.3, "accuracy": 0.75,
                         "decoding_confidence": 0.65, "accuracy_by_time_bin": np.array([1.0, 0.5])},
        }]
        self.states = [{
            "session": "0102", "cue": 7, "trial_idx": np.array([10, 20]),
            "time_bins": np.array([500, 550]), "decoding_fingerprint": "current",
            "off_state_duration_per_trial": np.array([50, 100]),
            "max_off_state_duration_per_trial": np.array([50, 50]),
        }]
        self.save_primary()
        self.write("select/tables/cell_screening.csv", "session,num_cells_selected\n0102,7\n")
        self.write("evaluate/tables/eval_confidence.csv", "session,num_trials,num_null_shuffles\n0102,2,2\n")
        self.write("pipeline_manifest.json", json.dumps({
            "run_id": "latest", "status": "complete", "started_at": "2026-01-02T00:00:00Z",
            "stages": [{"stage": "evaluate", "status": "complete", "seconds": 0.1}],
            "invocation": {"command": "python scripts/next/pipeline.py --stages evaluate"},
        }))
        app = FastAPI()
        app.include_router(create_results_router(self.root))
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        self.addCleanup(_load_primary_cached.cache_clear)

    def write(self, relative, text):
        path = self.run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def save_primary(self):
        for relative, payload in (
            ("select/cell_screening.pkl", self.selection),
            ("decode/decoding_confidence.pkl", self.decoded),
            ("evaluate/eval_confidence.pkl", self.evaluation),
            ("states/on_off_states.pkl", self.states),
        ):
            cache_io.save(payload, self.run / relative)

    def test_run_listing_uses_lightweight_tables_and_excludes_legacy_and_hidden(self):
        legacy = self.root / "cache" / "legacy"
        legacy.mkdir()
        (legacy / "cell_screening.pkl").write_bytes(b"legacy")
        hidden = self.root / "cache" / ".dashboard" / "select"
        hidden.mkdir(parents=True)
        with patch("scripts.next.dashboard.results.cache_io.read", side_effect=AssertionError("list loaded pickle")):
            response = self.client.get("/api/runs")
        self.assertEqual(response.status_code, 200)
        runs = response.json()["runs"]
        self.assertEqual([run["id"] for run in runs], ["example"])
        self.assertEqual(runs[0]["session_count"], 1)
        self.assertEqual(runs[0]["summary"], {"selected_cells": 7, "decoded_trials": 2, "null_shuffles": 2})

    def test_detail_keeps_manifest_history_and_merges_stage_statuses(self):
        self.write("manifests/prior.json", json.dumps({
            "run_id": "prior", "status": "complete",
            "stages": [{"stage": "select", "status": "complete"}, {"stage": "decode", "status": "complete"}],
        }))
        self.write("manifests/latest.json", (self.run / "pipeline_manifest.json").read_text())
        response = self.client.get("/api/runs/example")
        self.assertEqual(response.status_code, 200)
        detail = response.json()
        self.assertEqual(len(detail["manifests"]), 2)
        self.assertEqual(detail["manifests"][0]["id"], "latest")
        self.assertIn("--stages evaluate", detail["manifests"][0]["invocation"]["command"])
        self.assertEqual([stage["stage"] for stage in detail["run"]["stages"]], ["select", "decode", "evaluate"])
        self.assertEqual(detail["sessions"][0]["id"], "0102")
        self.assertEqual(detail["sessions"][0]["cell_count"], 9)
        self.assertEqual(detail["errors"], [])

    def test_dashboard_names_use_latest_job_and_cli_names_fall_back(self):
        storage = self.root / "cache/.dashboard"
        storage.mkdir()
        for job_id, name, created_at in (
            ("older", "First analysis", "2026-01-01T00:00:00+00:00"),
            ("newer", "Preferred cue comparison", "2026-01-02T00:00:00+00:00"),
        ):
            (storage / f"{job_id}.json").write_text(json.dumps({
                "id": job_id, "name": name, "cache_dir": str(self.run), "created_at": created_at,
            }))
        (storage / "broken.json").write_text("broken")
        (storage / "unrelated.json").write_text(json.dumps({
            "name": "Outside cache", "cache_dir": str(self.root / "outside"),
        }))
        (self.root / "cache/cli_run/select").mkdir(parents=True)
        store = ResultStore(self.root)
        with patch.object(store, "dashboard_names", wraps=store.dashboard_names) as index:
            runs = {run["id"]: run for run in store.list_runs()}
        self.assertEqual(index.call_count, 1)
        self.assertEqual(runs["example"]["name"], "Preferred cue comparison")
        self.assertEqual(runs["cli_run"]["name"], "cli_run")
        detail = self.client.get("/api/runs/example").json()
        self.assertEqual(detail["run"]["name"], "Preferred cue comparison")
        # A metadata symlink must not import a name from another directory.
        (storage / "newer.json").unlink()
        outside = self.root / "outside.json"
        outside.write_text(json.dumps({"name": "Wrong name", "cache_dir": str(self.run)}))
        (storage / "newer.json").symlink_to(outside)
        renamed = next(run for run in store.list_runs() if run["id"] == "example")
        self.assertEqual(renamed["name"], "First analysis")

    def test_session_curves_null_envelope_and_outcomes(self):
        response = self.client.get("/api/runs/example/sessions/0102")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        np.testing.assert_allclose(data["observed"], [0.7, 0.6])
        np.testing.assert_allclose(data["null_mean"], [0.5, 0.5])
        np.testing.assert_allclose(data["null_low"], [0.31, 0.405])
        np.testing.assert_allclose(data["null_high"], [0.69, 0.595])
        self.assertEqual(data["trial_ids"], [10, 20])
        self.assertEqual(data["total_off_durations"], [50, 100])
        self.assertEqual(data["max_off_durations"], [50, 50])
        self.assertEqual(data["metrics"]["accuracy"], 0.75)

    def test_nonfinite_measurements_are_json_null(self):
        self.decoded[0]["decoding_confidence"][:, 0] = np.nan
        self.decoded[0]["decoding_confidence_null"][:, 0, :] = np.inf
        self.evaluation[0]["observed"]["brier_score"] = np.float64(np.nan)
        self.save_primary()
        response = self.client.get("/api/runs/example/sessions/0102")
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIsNone(result["observed"][0])
        self.assertIsNone(result["null_mean"][0])
        self.assertIsNone(result["null_low"][0])
        self.assertIsNone(result["metrics"]["brier_score"])
        self.assertNotIn("NaN", response.text)

    def test_stale_downstream_cache_is_visible_without_misleading_metrics(self):
        self.states[0]["decoding_fingerprint"] = "older"
        self.evaluation[0]["trial_idx"] = [20, 10]
        self.save_primary()
        response = self.client.get("/api/runs/example/sessions/0102")
        result = response.json()
        self.assertEqual(len(result["warnings"]), 2)
        self.assertEqual(result["total_off_durations"], [])
        self.assertIsNone(result["metrics"]["accuracy"])
        summary = self.client.get("/api/runs/example").json()["sessions"][0]
        self.assertIsNone(summary["metrics"]["accuracy"])
        self.assertTrue(summary["warnings"])
        np.testing.assert_allclose(result["observed"], [0.7, 0.6])

    def test_missing_optional_caches_yield_empty_data(self):
        (self.run / "states/on_off_states.pkl").unlink()
        (self.run / "evaluate/eval_confidence.pkl").unlink()
        self.decoded[0].pop("decoding_confidence_null")
        self.save_primary()
        (self.run / "states/on_off_states.pkl").unlink()
        (self.run / "evaluate/eval_confidence.pkl").unlink()
        response = self.client.get("/api/runs/example/sessions/0102")
        result = response.json()
        self.assertEqual(result["null_mean"], [])
        self.assertEqual(result["total_off_durations"], [])
        self.assertIsNone(result["metrics"]["accuracy"])

    def test_cache_invalidates_when_file_changes(self):
        with patch("scripts.next.dashboard.results.cache_io.read", wraps=cache_io.read) as reader:
            self.client.get("/api/runs/example/sessions/0102")
            self.client.get("/api/runs/example/sessions/0102")
            self.assertEqual(reader.call_count, 4)
            self.decoded[0]["num_cells"] = 11
            cache_io.save(self.decoded, self.run / "decode/decoding_confidence.pkl")
            response = self.client.get("/api/runs/example/sessions/0102")
            self.assertEqual(reader.call_count, 5)
            self.assertEqual(response.json()["cell_count"], 11)

    def test_artifact_inventory_serving_and_session_attribution(self):
        image = self.run / "decode/figures/0102_observed.png"
        image.parent.mkdir(parents=True)
        image.write_bytes(b"png data")
        self.write("states/figures/all_sessions.eps", "eps data")
        self.write("models/logs/model.log", "fit complete")
        artifacts = self.client.get("/api/runs/example").json()["artifacts"]
        self.assertNotIn("decode/decoding_confidence.pkl", [artifact["path"] for artifact in artifacts])
        figure = next(artifact for artifact in artifacts if artifact["path"] == "decode/figures/0102_observed.png")
        self.assertEqual(figure["session"], "0102")
        self.assertEqual(figure["stage"], "decode")
        response = self.client.get(figure["url"])
        self.assertEqual(response.content, b"png data")
        self.assertEqual(response.headers["content-type"], "image/png")
        self.assertEqual(self.client.get("/api/runs/example/artifacts/decode/decoding_confidence.pkl").status_code, 404)

    def test_pdf_figure_is_listed_and_downloaded_without_conversion(self):
        pdf_bytes = b"%PDF-1.4\nOriginal vector figure\n%%EOF\n"
        pdf = self.run / "decode/figures/0102_observed.pdf"
        pdf.parent.mkdir(parents=True)
        pdf.write_bytes(pdf_bytes)
        artifacts = self.client.get("/api/runs/example").json()["artifacts"]
        figure = next(artifact for artifact in artifacts if artifact["path"] == "decode/figures/0102_observed.pdf")
        self.assertEqual(figure["kind"], "figure")
        self.assertEqual(figure["session"], "0102")
        response = self.client.get(figure["url"])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, pdf_bytes)
        self.assertEqual(response.headers["content-type"], "application/pdf")
        self.assertIn('attachment;', response.headers["content-disposition"])
        self.assertIn('0102_observed.pdf', response.headers["content-disposition"])
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")

    def test_table_pagination_numeric_conversion_and_query_bounds(self):
        self.write("models/tables/scores.csv", "model,score,missing\n00123,1.5,NaN\nm1,2,inf\nm2,-3,\n")
        response = self.client.get("/api/runs/example/tables/models/tables/scores.csv?limit=1&offset=1")
        self.assertEqual(response.status_code, 200)
        result = response.json()
        first = self.client.get("/api/runs/example/tables/models/tables/scores.csv?limit=1").json()
        self.assertEqual(first["rows"][0]["model"], "00123")
        self.assertEqual(result["total"], 3)
        self.assertEqual(result["columns"], ["model", "score", "missing"])
        self.assertEqual(result["rows"], [{"model": "m1", "score": 2, "missing": None}])
        self.assertEqual(self.client.get("/api/runs/example/tables/models/tables/scores.csv?limit=501").status_code, 422)
        self.assertEqual(self.client.get("/api/runs/example/tables/models/tables/scores.csv?offset=-1").status_code, 422)
        self.assertEqual(self.client.get("/api/runs/example/tables/pipeline_manifest.json").status_code, 404)

    def test_traversal_and_outside_symlinks_are_not_served_or_loaded(self):
        secret = self.root / "secret.csv"
        secret.write_text("secret\nprivate\n")
        (self.run / "select/tables/escape.csv").symlink_to(secret)
        link = self.root / "cache" / "alias"
        link.symlink_to(self.run, target_is_directory=True)
        self.assertEqual(self.client.get("/api/runs/alias").status_code, 404)
        self.assertEqual(self.client.get("/api/runs/example/artifacts/select/tables/escape.csv").status_code, 404)
        self.assertEqual(self.client.get("/api/runs/example/tables/select/tables/escape.csv").status_code, 404)
        self.assertEqual(self.client.get("/api/runs/example/artifacts/%2E%2E/%2E%2E/secret.csv").status_code, 404)
        outside_pickle = self.root / "on_off_states.pkl"
        cache_io.save(self.states, outside_pickle)
        (self.run / "states/on_off_states.pkl").unlink()
        (self.run / "states/on_off_states.pkl").symlink_to(outside_pickle)
        result = self.client.get("/api/runs/example").json()
        self.assertTrue(any("on_off_states.pkl" in error for error in result["errors"]))
        self.assertNotIn("select/tables/escape.csv", [artifact["path"] for artifact in result["artifacts"]])

    def test_corrupt_manifest_and_incompatible_primary_cache_are_reported(self):
        self.write("pipeline_manifest.json", "broken")
        cache_io.save([], self.run / "states/on_off_states.pkl")
        (self.run / "select/cell_screening.pkl").write_bytes(b"not a pickle")
        # Test corrupt pickle errors are contained in the response, not a server crash.
        response = self.client.get("/api/runs/example")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["run"]["status"], "unknown")
        self.assertGreaterEqual(len(response.json()["errors"]), 2)

    def test_unknown_run_and_session_are_explicit(self):
        self.assertEqual(self.client.get("/api/runs/missing").status_code, 404)
        self.assertEqual(self.client.get("/api/runs/example/sessions/missing").status_code, 404)
        with self.assertRaises(Exception):
            ResultStore(self.root).run_path("..")


if __name__ == "__main__":
    unittest.main()
