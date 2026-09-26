import { describe, expect, it } from "vitest";
import { cn, duration, formatDate, formatNumber, humanize } from "./utils";

describe("result formatting", () => {
  it("keeps missing and invalid measurements distinct from zero", () => {
    for (const value of [
      null,
      undefined,
      Number.NaN,
      Infinity,
      -Infinity,
      "0",
    ]) {
      expect(formatNumber(value)).toBe("—");
    }
    expect(formatNumber(0)).toBe("0");
    expect(formatNumber(0.1234, 2)).toBe(
      (0.12).toLocaleString(undefined, {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      }),
    );
  });

  it("handles legacy manifest timestamps without an invalid date label", () => {
    expect(formatDate(null)).toBe("Not recorded");
    expect(formatDate("not-a-timestamp")).toBe("not-a-timestamp");
    expect(formatDate("2026-09-26T00:00:00Z")).not.toContain("Invalid");
  });

  it("formats completed stage durations and leaves missing timings empty", () => {
    expect(duration()).toBe("—");
    expect(duration(0)).toBe("0.0s");
    expect(duration(23.45)).toBe("23.4s");
    expect(duration(125)).toBe("2m 5s");
    expect(duration(119.9)).toBe("2m 0s");
    expect(duration(60)).toBe("1m 0s");
    for (const value of [Number.NaN, Infinity, -Infinity, -1]) {
      expect(duration(value)).toBe("—");
    }
  });

  it("merges conditional Tailwind overrides", () => {
    expect(cn("p-2 text-sm", false && "hidden", ["p-4"])).toBe("text-sm p-4");
  });

  it("makes stage and setting identifiers readable", () => {
    expect(humanize("nested-cell_count")).toBe("Nested Cell Count");
  });
});
