import { describe, expect, it } from "vitest";
import {
  pathCompletionQuery,
  pathPopupPlacement,
  tabCompletion,
  type PathEntry,
} from "./path-completion";

const entries: PathEntry[] = [
  { name: "session-01/", path: "data/session-01/", kind: "directory" },
  { name: "session-02/", path: "data/session-02/", kind: "directory" },
];

describe("local path completion", () => {
  it("does not infer uniqueness or a shared prefix from truncated results", () => {
    expect(tabCompletion("data/se", entries, -1, true)).toBeNull();
    expect(tabCompletion("data/se", [entries[0]], -1, true)).toBeNull();
    expect(tabCompletion("data/se", entries, 1, true)).toBe("data/session-02/");
  });
  it("places bottom-edge mobile suggestions above the field and caps their height", () => {
    expect(
      pathPopupPlacement({ top: 690, bottom: 730 }, { top: 0, height: 844 }),
    ).toEqual({ side: "above", maxHeight: 360 });
    expect(
      pathPopupPlacement({ top: 210, bottom: 250 }, { top: 100, height: 300 }),
    ).toEqual({ side: "below", maxHeight: 138 });
    expect(
      pathPopupPlacement({ top: 170, bottom: 210 }, { top: 100, height: 150 }),
    ).toEqual({ side: "above", maxHeight: 58 });
  });
  it("prefers below when there is ample room and never returns a negative height", () => {
    expect(
      pathPopupPlacement({ top: 80, bottom: 120 }, { top: 0, height: 844 }),
    ).toEqual({ side: "below", maxHeight: 360 });
    expect(
      pathPopupPlacement({ top: 0, bottom: 40 }, { top: 0, height: 40 }),
    ).toEqual({ side: "below", maxHeight: 0 });
  });
  it("completes a common prefix without arbitrarily choosing a folder", () => {
    expect(tabCompletion("data/se", entries, -1)).toBe("data/session-0");
    expect(tabCompletion("data/session-0", entries, -1)).toBeNull();
  });
  it("uses the explicitly selected match and completes a unique match", () => {
    expect(tabCompletion("data/se", entries, 1)).toBe("data/session-02/");
    expect(tabCompletion("data/se", [entries[0]], -1)).toBe("data/session-01/");
  });
  it("allows normal focus navigation when there are no useful completions", () => {
    expect(tabCompletion("missing", [], -1)).toBeNull();
    expect(tabCompletion("data/session-0", entries, 100)).toBeNull();
  });
  it("escapes local paths and passes mode and extension filters as query parameters", () => {
    const query = pathCompletionQuery("~/my data/a#b&c", "any", [
      ".json",
      ".txt",
    ]);
    const params = new URLSearchParams(query.split("?")[1]);
    expect(params.get("path")).toBe("~/my data/a#b&c");
    expect(params.get("mode")).toBe("any");
    expect(params.get("extensions")).toBe(".json,.txt");
    expect(pathCompletionQuery("cache/", "directory", [])).not.toContain(
      "extensions=",
    );
  });
});
