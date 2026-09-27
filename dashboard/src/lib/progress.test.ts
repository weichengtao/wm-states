import { describe, expect, it } from "vitest";
import { stageElapsedSeconds } from "./progress";

const start = "2026-09-27T01:00:00Z";
const current = Date.parse(start) + 125_000;
const running = { stage: "decode", status: "running", started_at: start };

describe("stage elapsed time", () => {
  it("advances from the persisted stage start independently of socket messages", () => {
    expect(stageElapsedSeconds(running, { status: "running" }, current)).toBe(
      125,
    );
    expect(
      stageElapsedSeconds(running, { status: "running" }, current + 1_000),
    ).toBe(126);
    expect(
      stageElapsedSeconds(running, { status: "cancelling" }, current),
    ).toBe(125);
  });

  it("prefers measured runtime and freezes terminal jobs and interrupted stages", () => {
    expect(
      stageElapsedSeconds(
        { ...running, status: "complete", seconds: 3.125 },
        { status: "complete" },
        current,
      ),
    ).toBe(3.125);
    expect(
      stageElapsedSeconds(
        running,
        { status: "failed", finished_at: "2026-09-27T01:00:15Z" },
        current,
      ),
    ).toBe(15);
    expect(
      stageElapsedSeconds(
        {
          ...running,
          status: "interrupted",
          finished_at: "2026-09-27T01:00:10Z",
        },
        { status: "cancelled" },
        current,
      ),
    ).toBe(10);
  });

  it("does not fabricate timing for old or invalid records and handles clock skew", () => {
    expect(
      stageElapsedSeconds(
        { stage: "decode", status: "running" },
        { status: "running" },
        current,
      ),
    ).toBeUndefined();
    expect(
      stageElapsedSeconds(
        { ...running, started_at: "invalid" },
        { status: "running" },
        current,
      ),
    ).toBeUndefined();
    expect(
      stageElapsedSeconds(running, { status: "failed" }, current),
    ).toBeUndefined();
    expect(
      stageElapsedSeconds(
        { ...running, status: "pending" },
        { status: "running" },
        current,
      ),
    ).toBeUndefined();
    expect(
      stageElapsedSeconds(
        running,
        { status: "running" },
        Date.parse(start) - 1000,
      ),
    ).toBe(0);
  });

  it("excludes dashboard downtime when an unfinished job is restored", () => {
    const restored = { status: "failed", finished_at: "2026-09-29T01:00:00Z" };
    expect(
      stageElapsedSeconds(
        { ...running, elapsed_unavailable: true },
        restored,
        Date.parse("2026-09-30T01:00:00Z"),
      ),
    ).toBeUndefined();
    expect(
      stageElapsedSeconds(
        { stage: "select", status: "complete", seconds: 3.125 },
        restored,
        current,
      ),
    ).toBe(3.125);
  });
});
