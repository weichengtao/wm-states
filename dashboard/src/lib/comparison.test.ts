import { describe, expect, it } from "vitest";
import { comparisonWarnings } from "./comparison";
import type { SessionData } from "./types";

function session(overrides: Partial<SessionData> = {}): SessionData {
  return {
    id: "210921",
    session: "210921",
    cue: 3,
    trial_count: 3,
    cell_count: 8,
    selected_cells: 8,
    null_shuffles: 100,
    metrics: {},
    stages: ["decode"],
    trial_ids: [1, 4, 9],
    time_bins: [0, 50, 100],
    observed: [0.4, 0.7, 0.8],
    null_mean: [0.5, 0.5, 0.5],
    null_low: [0.4, 0.4, 0.4],
    null_high: [0.6, 0.6, 0.6],
    accuracy: [],
    total_off_durations: [],
    max_off_durations: [],
    errors: [],
    warnings: [],
    ...overrides,
  };
}

describe("comparisonWarnings", () => {
  it("allows directly comparable results and trial reordering without modifying inputs", () => {
    const left = session({ trial_ids: [9, 1, 4] });
    const right = session({ trial_ids: [4, 9, 1] });
    expect(comparisonWarnings(left, right)).toEqual([]);
    expect(left.trial_ids).toEqual([9, 1, 4]);
    expect(right.trial_ids).toEqual([4, 9, 1]);
  });

  it("warns when a different run decodes a different trial set for the same session", () => {
    const warnings = comparisonWarnings(
      session(),
      session({ trial_ids: [1, 4, 10] }),
    );
    expect(warnings).toHaveLength(1);
    expect(warnings[0]).toContain("different trial sets");
  });

  it("does not confuse session-local trial identifiers across separate sessions", () => {
    expect(
      comparisonWarnings(
        session(),
        session({
          id: "211015",
          session: "211015",
          trial_ids: [30, 40],
          trial_count: 2,
        }),
      ),
    ).toEqual([]);
  });

  it("reports cue populations and time grids independently of trial sets", () => {
    const warnings = comparisonWarnings(
      session(),
      session({
        cue: 7,
        trial_ids: [1, 4, 10],
        time_bins: [0, 100],
      }),
    );
    expect(warnings).toHaveLength(3);
    expect(warnings.join(" ")).toContain("Preferred cues differ");
    expect(warnings.join(" ")).toContain("different trial sets");
    expect(warnings.join(" ")).toContain("Time grids differ");
  });

  it("identifies missing metadata on just one side without treating it as matching", () => {
    const warnings = comparisonWarnings(
      session(),
      session({
        cue: null,
        trial_ids: [],
        time_bins: [],
      }),
    );
    expect(warnings).toHaveLength(3);
  });
});

import { settingsDifferences } from "./comparison";
import type { Manifest } from "./types";
it("compares latest recorded settings per stage across partial invocation history", () => {
  const record = (settings: Manifest["settings"]): Manifest => ({
    id: "fixture",
    status: "complete",
    stages: [],
    settings,
  });
  const rows = settingsDifferences(
    [
      record({ evaluate: { bins: [500, 1400] } }),
      record({
        decode: { n_decode_shuffle: 100, cache_dir: "cache/a" },
        evaluate: { bins: [0, 100] },
      }),
    ],
    [
      record({
        decode: { n_decode_shuffle: 3, cache_dir: "cache/b" },
        evaluate: { bins: [500, 1400] },
      }),
    ],
  );
  expect(rows).toEqual([
    { stage: "decode", parameter: "n_decode_shuffle", left: "100", right: "3" },
  ]);
});
