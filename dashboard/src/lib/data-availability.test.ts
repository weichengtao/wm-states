import { afterEach, describe, expect, it, vi } from "vitest";
import { dataAvailabilityQuery, scheduleDataCheck } from "./data-availability";
import { api } from "./api";
import type { DataStatus, RunRequest } from "./types";

vi.mock("./api", () => ({
  api: vi.fn(),
  errorMessage: (error: unknown) =>
    error instanceof Error ? error.message : String(error),
}));

const form: RunRequest = {
  name: "A run",
  cache_dir: "cache/new",
  data_dir: "data/nature",
  stages: ["decode", "select"],
  settings: { decode: { plot_only: false, n_decode_shuffle: 100 } },
  n_jobs: 2,
  session_list_file: null,
  max_sessions_to_run: null,
  figure_formats: ["png"],
  figure_font: "DejaVu Sans",
  allow_existing: false,
  trust_unverified_legacy_results: false,
};
const ready: DataStatus = {
  status: "ready",
  data_dir: "/repo/data/nature",
  file_count: 5,
  blocking: false,
  message: "5 files found.",
};

afterEach(() => {
  vi.useRealTimers();
  vi.resetAllMocks();
});

describe("recording availability checks", () => {
  it("only refreshes for choices that affect recording access", () => {
    const query = dataAvailabilityQuery(form);
    expect(
      dataAvailabilityQuery({
        ...form,
        name: "Other",
        n_jobs: 8,
        cache_dir: "cache/other",
        stages: ["select", "decode"],
        settings: { decode: { plot_only: false, n_decode_shuffle: 4 } },
      }),
    ).toBe(query);
    const patches: Partial<RunRequest>[] = [
      { data_dir: "data/other" },
      { stages: ["states"] },
      { session_list_file: "sessions.json" },
      { settings: { decode: { plot_only: true } } },
      { settings: { decode: { session_list_file: "decode-sessions.json" } } },
      { settings: { select: { session_list_file: null } } },
      { trust_unverified_legacy_results: true },
    ];
    for (const patch of patches)
      expect(dataAvailabilityQuery({ ...form, ...patch })).not.toBe(query);
    expect(JSON.parse(query)).not.toHaveProperty("source_template");
  });

  it("debounces a new query before asking the server", async () => {
    vi.useFakeTimers();
    vi.mocked(api).mockResolvedValue(ready);
    const result = vi.fn();
    const cancel = scheduleDataCheck(
      dataAvailabilityQuery(form),
      result,
      vi.fn(),
    );
    expect(api).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(299);
    expect(api).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(1);
    expect(api).toHaveBeenCalledWith(
      "/data-status",
      expect.objectContaining({
        method: "POST",
        body: dataAvailabilityQuery(form),
      }),
    );
    expect(result).toHaveBeenCalledWith(ready);
    cancel();
  });

  it("does not request an old path when typing replaces a pending timer", async () => {
    vi.useFakeTimers();
    const cancel = scheduleDataCheck("old", vi.fn(), vi.fn());
    cancel();
    await vi.advanceTimersByTimeAsync(500);
    expect(api).not.toHaveBeenCalled();
  });

  it("aborts an in-flight request and ignores a late result even if fetch ignores cancellation", async () => {
    vi.useFakeTimers();
    let resolve!: (data: DataStatus) => void;
    vi.mocked(api).mockReturnValue(
      new Promise((done) => {
        resolve = done;
      }),
    );
    const result = vi.fn();
    const error = vi.fn();
    const cancel = scheduleDataCheck("old", result, error);
    await vi.advanceTimersByTimeAsync(300);
    const signal = vi.mocked(api).mock.calls[0][1]?.signal;
    cancel();
    expect(signal?.aborted).toBe(true);
    resolve(ready);
    await Promise.resolve();
    expect(result).not.toHaveBeenCalled();
    expect(error).not.toHaveBeenCalled();
  });

  it("reports connection failure as unknown rather than claiming recordings are missing", async () => {
    vi.useFakeTimers();
    vi.mocked(api).mockRejectedValue(new Error("Server unavailable"));
    const result = vi.fn();
    const error = vi.fn();
    const cancel = scheduleDataCheck("current", result, error);
    await vi.advanceTimersByTimeAsync(300);
    expect(result).not.toHaveBeenCalled();
    expect(error).toHaveBeenCalledWith("Server unavailable");
    cancel();
  });
});
