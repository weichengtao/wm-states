import { describe, expect, it, vi } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import DataAvailability from "./DataAvailability";
import type { DataAvailabilityState } from "@/lib/data-availability";

const missing: DataAvailabilityState = {
  result: {
    status: "missing",
    data_dir: "/repo/data/nature",
    file_count: 0,
    blocking: true,
    message: "The recording folder does not exist.",
  },
  error: "",
  checking: false,
};
const render = (state: DataAvailabilityState, compact = false) =>
  renderToStaticMarkup(
    <DataAvailability state={state} onRecheck={vi.fn()} compact={compact} />,
  );

describe("recording availability guidance", () => {
  it("explains the server check without claiming missing recordings while loading", () => {
    const html = render({ result: null, error: "", checking: true });
    expect(html).toContain("Checking recordings on the server");
    expect(html).toContain("continue editing and save a template");
    expect(html).not.toContain("data-availability-warning");
  });
  it("offers preparation help, the server path, and a retry for an unavailable folder", () => {
    const html = render(missing);
    expect(html).toContain("data-availability-warning");
    expect(html).toContain("/repo/data/nature");
    expect(html).toContain("0 recording files found");
    expect(html).toContain(
      "/docs/next/getting-started/#prepare-the-recordings",
    );
    expect(html).toContain("Check again");
    expect(html).toContain("still save this configuration as a template");
    expect(html).not.toContain('role="alert"');
  });
  it("does not describe missing raw files as a launch blocker for cache-only stages", () => {
    const html = render({
      ...missing,
      result: {
        ...missing.result!,
        blocking: false,
        message:
          "Selected stages use cached outputs; this will not prevent launch.",
      },
    });
    expect(html).toContain("will not prevent launch");
    expect(html).not.toContain("Choose a prepared recording folder");
  });
  it("shows missing or empty server paths once and omits them from the compact reminder", () => {
    const path = `/server/${"long-directory-name-".repeat(15)}/recordings`;
    for (const status of ["missing", "empty"] as const) {
      const state = {
        ...missing,
        result: {
          ...missing.result!,
          status,
          data_dir: path,
          message: `Recording directory ${path} has no available recordings.`,
        },
      };
      expect(render(state).split(path)).toHaveLength(2);
      expect(render(state, true)).not.toContain(path);
      expect(render(state)).toContain(
        status === "missing"
          ? "This folder does not exist on the dashboard server."
          : "This server folder has no .mat recording files.",
      );
    }
  });
  it("preserves actionable server detail for unreadable folders and invalid or unmatched settings", () => {
    for (const status of ["unreadable", "invalid", "no_matches"] as const) {
      const message =
        "Check permissions or the allowlist in /server/config/session-list.json.";
      expect(
        render(
          {
            ...missing,
            result: { ...missing.result!, status, message },
          },
          true,
        ),
      ).toContain(message);
    }
  });
  it("distinguishes a failed check from a missing folder", () => {
    const html = render({
      result: null,
      checking: false,
      error: "Server unavailable",
    });
    expect(html).toContain("Recording check unavailable");
    expect(html).toContain("Server unavailable");
    expect(html).not.toContain("Recordings are not available yet");
  });
  it("keeps the launch reminder compact and omits healthy duplicate notices", () => {
    expect(render(missing, true)).not.toContain("/repo/data/nature");
    expect(
      render(
        {
          result: {
            ...missing.result!,
            status: "ready",
            blocking: false,
            file_count: 5,
          },
          checking: false,
          error: "",
        },
        true,
      ),
    ).toBe("");
  });
});
