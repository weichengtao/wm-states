import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "./api";
import { docsHref } from "./help";
import { apiHref, dashboardBase, docsBase, websocketHref } from "./urls";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

function deployment(dashboard: string, docs = "/wm-states/docs/") {
  vi.stubGlobal("document", {
    querySelector: (selector: string) => ({
      content: selector.includes("dashboard-base") ? dashboard : docs,
    }),
  });
}

describe("deployment URLs", () => {
  it("uses development API routes and the canonical docs mount without metadata", () => {
    expect(dashboardBase()).toBe("/");
    expect(apiHref("/schema")).toBe("/api/schema");
    expect(docsBase()).toBe("/wm-states/docs/");
  });

  it("keeps API paths, queries and encoded identifiers under the runtime prefix", () => {
    deployment("/wm-states/dashboard/");
    expect(apiHref("/runs/a%2Fb/tables/report.csv?offset=50&limit=50")).toBe(
      "/wm-states/dashboard/api/runs/a%2Fb/tables/report.csv?offset=50&limit=50",
    );
    expect(apiHref("docs")).toBe("/wm-states/dashboard/api/docs");
  });

  it("supports nested custom prefixes and the empty common prefix", () => {
    deployment("/lab/analysis/dashboard", "/lab/analysis/docs");
    expect(apiHref("/jobs")).toBe("/lab/analysis/dashboard/api/jobs");
    expect(docsHref("next/methods/#decode")).toBe(
      "/lab/analysis/docs/next/methods/#decode",
    );
    deployment("/dashboard/", "/docs/");
    expect(apiHref("/jobs")).toBe("/dashboard/api/jobs");
    expect(docsHref()).toBe("/docs/");
  });

  it("uses injected metadata even on deep SPA paths", () => {
    deployment("/research/dashboard/", "/research/docs/");
    vi.stubGlobal("window", {
      location: new URL("http://127.0.0.1:8000/research/dashboard/deep/view"),
    });
    expect(apiHref("/runs")).toBe("/research/dashboard/api/runs");
    expect(websocketHref("/jobs/job%20one/events")).toBe(
      "ws://127.0.0.1:8000/research/dashboard/api/jobs/job%20one/events",
    );
  });

  it.each([
    ["https://host.tail.ts.net/wm-states/dashboard/", "wss://host.tail.ts.net"],
    ["https://host.tail.ts.net:8443/", "wss://host.tail.ts.net:8443"],
    ["http://100.101.102.103:8000/", "ws://100.101.102.103:8000"],
    ["http://[fd7a:115c:a1e0::1]:8000/", "ws://[fd7a:115c:a1e0::1]:8000"],
  ])("keeps WebSockets on the page origin for %s", (href, origin) => {
    deployment("/wm-states/dashboard/");
    vi.stubGlobal("window", { location: new URL(href) });
    expect(websocketHref("/jobs/run/events")).toBe(
      `${origin}/wm-states/dashboard/api/jobs/run/events`,
    );
  });

  it.each([
    "",
    "dashboard",
    "https://other.test/dashboard/",
    "//other.test/dashboard/",
    "/\\other.test/",
    "/lab//dashboard/",
    "/lab/../dashboard/",
    "/lab/./dashboard/",
    "/lab/%2e%2e/dashboard/",
    "/lab/dashboard/?query=yes",
    "/lab/dashboard/#fragment",
    "/lab/\ndashboard/",
  ])("falls back safely for malformed runtime bases: %s", (base) => {
    deployment(base, base);
    expect(dashboardBase()).toBe("/");
    expect(docsBase()).toBe("/wm-states/docs/");
  });

  it("honors external documentation overrides and uses runtime docs on invalid overrides", () => {
    deployment("/analysis/dashboard/", "/analysis/docs/");
    vi.stubEnv("VITE_DOCS_BASE_URL", "https://example.github.io/wm-states/");
    expect(docsHref("next/dashboard/")).toBe(
      "https://example.github.io/wm-states/next/dashboard/",
    );
    expect(docsHref("next/dashboard/", "javascript:alert(1)")).toBe(
      "/analysis/docs/next/dashboard/",
    );
  });

  it("sends API requests through the runtime base without changing request options", async () => {
    deployment("/analysis/dashboard/");
    const fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ status: "ready" }),
    });
    vi.stubGlobal("fetch", fetch);
    const result = await api("/data-status", { method: "POST", body: "{}" });
    expect(result).toEqual({ status: "ready" });
    expect(fetch).toHaveBeenCalledWith("/analysis/dashboard/api/data-status", {
      method: "POST",
      body: "{}",
      headers: { "Content-Type": "application/json" },
    });
  });
});
