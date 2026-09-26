import { describe, expect, it, vi, afterEach } from "vitest";
import {
  docsHref,
  fieldHelpPath,
  helpTopics,
  matchesHelp,
  pageHelp,
  stageHelp,
  stageMethodsPath,
  troubleshootingPath,
} from "./help";

afterEach(() => vi.unstubAllEnvs());

describe("documentation URLs", () => {
  it("uses the local docs mount and preserves stage fragments", () => {
    vi.stubEnv("VITE_DOCS_BASE_URL", "");
    expect(docsHref()).toBe("/docs/");
    expect(docsHref("next/methods/#decode")).toBe("/docs/next/methods/#decode");
  });
  it("retains GitHub Pages subpaths with either trailing slash form", () => {
    for (const base of [
      "https://example.github.io/wm-states",
      "https://example.github.io/wm-states/",
    ]) {
      expect(
        docsHref("/next/configuration/#null-shuffle-time-structure", base),
      ).toBe(
        "https://example.github.io/wm-states/next/configuration/#null-shuffle-time-structure",
      );
    }
  });
  it("reads the configured external base and clears base queries/fragments", () => {
    vi.stubEnv(
      "VITE_DOCS_BASE_URL",
      "https://example.org/project/docs/?old=yes#home",
    );
    expect(docsHref("next/dashboard/")).toBe(
      "https://example.org/project/docs/next/dashboard/",
    );
    expect(docsHref("next/dashboard/", "/project/docs")).toBe(
      "/project/docs/next/dashboard/",
    );
  });
  it("falls back to local docs for unsupported or malformed bases", () => {
    for (const base of [
      "javascript:alert(1)",
      "data:text/html,hello",
      "//elsewhere.test",
      "https://",
      "docs",
    ])
      expect(docsHref("next/methods/", base)).toBe("/docs/next/methods/");
  });
});

describe("contextual help", () => {
  it("covers all eleven stages with stable documentation fragments", () => {
    expect(stageHelp.map((stage) => stage.id)).toEqual([
      "select",
      "decode",
      "evaluate",
      "states",
      "activity",
      "prepare",
      "models",
      "nested-count",
      "nested-activity",
      "criticality",
      "interactions",
    ]);
    for (const stage of stageHelp)
      expect(stageMethodsPath(stage.id)).toBe(`next/methods/#${stage.id}`);
    expect(stageMethodsPath("unknown-stage")).toBe("next/methods/");
    expect(Object.keys(pageHelp).sort()).toEqual([
      "compare",
      "configure",
      "monitor",
      "results",
    ]);
  });
  it("links key fields to their relevant method or configuration section", () => {
    expect(fieldHelpPath("decode", "preserve_null_time_structure")).toBe(
      "next/configuration/#null-shuffle-time-structure",
    );
    expect(fieldHelpPath("select", "check_baseline_drift")).toBe(
      "next/configuration/#screening-checks",
    );
    expect(fieldHelpPath("models", "cv_shuffles")).toBe(
      "next/methods/#mixed-effects-estimation",
    );
    expect(fieldHelpPath("activity", "show_principal_components")).toBe(
      "next/methods/#activity",
    );
    expect(fieldHelpPath("decode", "t_decode_step")).toBe(
      "next/methods/#decode",
    );
    expect(fieldHelpPath("criticality", "active_percentiles")).toBe(
      "next/methods/#criticality",
    );
    expect(fieldHelpPath("activity", "figure_dpi")).toBeNull();
  });
  it("finds parameter names and multiword help searches without case sensitivity", () => {
    const topic = helpTopics.find(
      (item) => item.title === "Null shuffles through time",
    )!;
    expect(
      matchesHelp(
        " PRESERVE_NULL_TIME_STRUCTURE ",
        topic.title,
        topic.answer,
        topic.keywords,
      ),
    ).toBe(true);
    expect(
      matchesHelp("null TIME", topic.title, topic.answer, topic.keywords),
    ).toBe(true);
    expect(matchesHelp("unknown pineapple", topic.title, topic.answer)).toBe(
      false,
    );
    expect(matchesHelp("  ", topic.title)).toBe(true);
  });
  it("matches actionable troubleshooting guidance with a safe general fallback", () => {
    expect(troubleshootingPath("Fingerprint mismatch. Rerun decode.")).toBe(
      "next/troubleshooting/#activity-or-preparation-reports-stale-provenance",
    );
    expect(
      troubleshootingPath(
        "example: decoding requires at least 6 correct preferred-cue trials",
      ),
    ).toBe(
      "next/troubleshooting/#decoding-cannot-construct-the-requested-cv-folds",
    );
    expect(troubleshootingPath("Connection refused")).toBe(
      "next/troubleshooting/#the-dashboard-does-not-open",
    );
    expect(troubleshootingPath("Another problem")).toBe(
      "next/troubleshooting/",
    );
  });
});
