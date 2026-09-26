import { describe, expect, it } from "vitest";
import { filterRuns, runStatuses } from "./run-library";
import type { Run } from "./types";

function run(id: string, overrides: Partial<Run> = {}): Run {
  return {
    id,
    name: id,
    path: `/cache/${id}`,
    updated_at: "2026-09-26T12:00:00Z",
    status: "complete",
    stages: [],
    session_count: 4,
    summary: {},
    ...overrides,
  };
}

describe("run library", () => {
  it("retains the selected status when a refresh removes its last matching run", () => {
    const before = [run("analysis", { status: "running" })];
    const after = [run("analysis", { status: "complete" })];
    expect(runStatuses(before, "running")).toEqual(["running"]);
    expect(runStatuses(after, "running")).toEqual(["complete", "running"]);
    expect(filterRuns(after, "", "running", "recent")).toEqual([]);
    expect(runStatuses(after, "all")).toEqual(["complete"]);
  });
  it("combines trimmed, case-insensitive name/cache search with invocation status", () => {
    const runs = [
      run("test_run_001", { name: "Baseline", status: "failed" }),
      run("test_run_002", { name: "Baseline refined" }),
      run("test_run_003", { path: "/archive/Example/results" }),
    ];
    expect(
      filterRuns(runs, " BASELINE ", "complete", "recent").map((run) => run.id),
    ).toEqual(["test_run_002"]);
    expect(
      filterRuns(runs, "EXAMPLE", "all", "recent").map((run) => run.id),
    ).toEqual(["test_run_003"]);
    expect(filterRuns(runs, "test_run_001", "all", "recent")).toHaveLength(1);
    expect(filterRuns(runs, "missing", "all", "recent")).toEqual([]);
  });

  it("sorts recent runs without mutating discovery order and keeps unknown dates last", () => {
    const runs = [
      run("run_10", { updated_at: "2026-09-24T12:00:00Z" }),
      run("run_2"),
      run("unknown", { updated_at: "" }),
    ];
    expect(filterRuns(runs, "", "all", "recent").map((run) => run.id)).toEqual([
      "run_2",
      "run_10",
      "unknown",
    ]);
    expect(filterRuns(runs, "", "all", "name").map((run) => run.id)).toEqual([
      "run_2",
      "run_10",
      "unknown",
    ]);
    expect(runs.map((run) => run.id)).toEqual(["run_10", "run_2", "unknown"]);
  });
});
