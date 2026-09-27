import type { Job, StageProgress } from "./types";

const finished = (status: string) =>
  ["complete", "failed", "cancelled"].includes(status);

function timestamp(value?: string | null): number | undefined {
  if (!value) return undefined;
  const time = Date.parse(value);
  return Number.isFinite(time) ? time : undefined;
}

/** Completed stages use the measured monotonic runtime; live stages use UTC. */
export function stageElapsedSeconds(
  stage: StageProgress,
  job: Pick<Job, "status" | "finished_at">,
  currentTime: number,
): number | undefined {
  if (stage.elapsed_unavailable) return undefined;
  if (
    stage.seconds != null &&
    Number.isFinite(stage.seconds) &&
    stage.seconds >= 0
  )
    return stage.seconds;
  const started = timestamp(stage.started_at);
  if (started == null) return undefined;
  const ended = timestamp(stage.finished_at);
  // Freeze interrupted jobs at their recorded end; older records must never
  // acquire a fabricated start time or keep accumulating elapsed time forever.
  const end =
    ended ??
    (finished(job.status)
      ? timestamp(job.finished_at)
      : stage.status === "running"
        ? currentTime
        : undefined);
  if (end == null || !Number.isFinite(end)) return undefined;
  return Math.max(0, (end - started) / 1000);
}
