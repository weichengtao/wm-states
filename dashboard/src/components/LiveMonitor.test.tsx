import { describe, expect, it, vi } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import type { Job } from "@/lib/types";
import LiveMonitor from "./LiveMonitor";

const job: Job = {
  id: "example-job",
  name: "Example analysis",
  cache_dir: "cache/example",
  status: "running",
  created_at: "2026-09-26T10:00:00Z",
  started_at: "2026-09-26T10:01:00Z",
  command: "python -m scripts.next.run_pipeline",
  stages: [
    { stage: "select", status: "complete", seconds: 3 },
    { stage: "decode", status: "running" },
  ],
  requested_stages: ["select", "decode"],
  logs: ["Cell screening complete.", "Decoding session 210921."],
};

function renderJob(overrides: Partial<Job> = {}) {
  return renderToStaticMarkup(
    <LiveMonitor
      jobs={[{ ...job, ...overrides }]}
      selectedId={job.id}
      onJobsChanged={vi.fn()}
      onViewRun={vi.fn()}
      onNewRun={vi.fn()}
    />,
  );
}

describe("live monitor controls", () => {
  it("labels stage progress and offers a separate stop confirmation", () => {
    const html = renderJob();
    expect(html).toContain('aria-valuenow="1"');
    expect(html).toContain('aria-valuemax="2"');
    expect(html).toContain('aria-current="step"');
    expect(html).toContain('aria-expanded="false"');
    expect(html).toContain("Stop run");
    expect(html).not.toContain("Yes, stop run");
    expect(html).toContain('aria-label="Filter log lines"');
    expect(html).toContain("Download full log");
    expect(html).toContain('href="/api/jobs/example-job/log"');
    expect(html).toContain("Copy shown lines");
  });

  it("shows a finished log without offering to stop a completed run", () => {
    const html = renderJob({ status: "complete" });
    expect(html).toContain("Saved log");
    expect(html).toContain("All requested stages finished");
    expect(html).not.toContain("Stop run");
    expect(html).not.toContain("Connecting");
  });

  it("explains an empty preview while keeping the full saved log downloadable", () => {
    const html = renderJob({ status: "failed", logs: [] });
    expect(html).toContain("No processing output was recorded.");
    expect(html).toContain("Run stopped with an error");
    const logActions = html.match(
      /<div class="monitor-log-actions">([\s\S]*?)<\/div>/,
    )?.[1];
    expect(logActions?.match(/disabled=""/g)).toHaveLength(1);
    expect(logActions).toContain('href="/api/jobs/example-job/log"');
  });

  it("shows running elapsed time and freezes a stopped job", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-09-26T10:03:00Z"));
    try {
      const stages = [
        {
          stage: "decode",
          status: "running",
          started_at: "2026-09-26T10:01:30Z",
        },
      ];
      expect(renderJob({ stages })).toContain("Now: Decode · 1m 30s elapsed");
      expect(renderJob({ stages })).toContain(
        "Elapsed time · updates every second",
      );
      const stopped = renderJob({
        stages,
        status: "failed",
        finished_at: "2026-09-26T10:02:00Z",
      });
      expect(stopped).toContain("30.0s");
      expect(stopped).not.toContain("updates every second");
    } finally {
      vi.useRealTimers();
    }
  });
});
