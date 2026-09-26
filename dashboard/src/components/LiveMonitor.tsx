import { useEffect, useRef, useState } from "react";
import {
  ArrowRight,
  Check,
  Circle,
  Clock3,
  LoaderCircle,
  Plus,
  Radio,
  Square,
  Terminal,
  WifiOff,
} from "lucide-react";
import { api, errorMessage } from "@/lib/api";
import type { Job } from "@/lib/types";
import { cn, duration, formatDate, humanize } from "@/lib/utils";
import { Button } from "./ui/button";
import { CopyButton, Empty, Loading, Notice, Status } from "./shared";

const terminal = (status: string) =>
  ["complete", "failed", "cancelled"].includes(status);

type Props = {
  jobs: Job[];
  selectedId: string | null;
  onJobsChanged: () => void;
  onViewRun: (cacheDir: string) => void;
  onNewRun: () => void;
};

export default function LiveMonitor({
  jobs,
  selectedId,
  onJobsChanged,
  onViewRun,
  onNewRun,
}: Props) {
  const [activeId, setActiveId] = useState<string | null>(
    selectedId ?? jobs[0]?.id ?? null,
  );
  const [snapshot, setSnapshot] = useState<Job | null>(null);
  const [error, setError] = useState("");
  const [connection, setConnection] = useState<
    "connecting" | "live" | "polling" | "finished"
  >("connecting");
  const [cancelling, setCancelling] = useState(false);
  const [followLog, setFollowLog] = useState(true);
  const logRef = useRef<HTMLPreElement>(null);
  const jobsChangedRef = useRef(onJobsChanged);
  jobsChangedRef.current = onJobsChanged;

  useEffect(() => {
    if (selectedId) setActiveId(selectedId);
  }, [selectedId]);
  useEffect(() => {
    if (!activeId && jobs.length) setActiveId(jobs[0].id);
  }, [activeId, jobs]);
  useEffect(() => {
    if (!activeId) return;
    let disposed = false;
    let socket: WebSocket | null = null;
    let retry: ReturnType<typeof setTimeout> | undefined;
    let done = false;
    let retryDelay = 1500;
    let previousStatus = "";
    setSnapshot(null);
    setError("");
    setConnection("connecting");
    const accept = (job: Job) => {
      if (disposed) return;
      setSnapshot(job);
      setError("");
      if (terminal(job.status)) {
        done = true;
        setConnection("finished");
        if (previousStatus !== job.status) jobsChangedRef.current();
      }
      previousStatus = job.status;
    };
    const refresh = async () => {
      try {
        accept(await api<Job>(`/jobs/${encodeURIComponent(activeId)}`));
      } catch (cause) {
        if (!disposed) setError(errorMessage(cause));
      }
    };
    const connect = () => {
      if (disposed || done) return;
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      socket = new WebSocket(
        `${protocol}//${window.location.host}/api/jobs/${encodeURIComponent(activeId)}/events`,
      );
      socket.onopen = () => {
        if (!disposed) {
          retryDelay = 1500;
          setConnection("live");
        }
      };
      socket.onmessage = (event) => {
        try {
          accept(JSON.parse(event.data) as Job);
        } catch {
          if (!disposed)
            setError(
              "Could not read the live update. Refreshing through the API.",
            );
        }
      };
      socket.onerror = () => {
        if (!disposed && !done) setConnection("polling");
      };
      socket.onclose = () => {
        if (disposed || done) return;
        setConnection("polling");
        void refresh();
        retry = setTimeout(connect, retryDelay);
        retryDelay = Math.min(retryDelay * 2, 10000);
      };
    };
    void refresh();
    connect();
    const polling = setInterval(() => {
      if (!done && socket?.readyState !== WebSocket.OPEN) void refresh();
    }, 5000);
    return () => {
      disposed = true;
      clearInterval(polling);
      if (retry) clearTimeout(retry);
      socket?.close();
    };
  }, [activeId]);

  const job =
    snapshot?.id === activeId
      ? snapshot
      : jobs.find((item) => item.id === activeId);
  useEffect(() => {
    if (followLog && logRef.current)
      logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [job?.logs, followLog]);

  const cancel = async () => {
    if (!job) return;
    setCancelling(true);
    setError("");
    try {
      setSnapshot(
        await api<Job>(`/jobs/${encodeURIComponent(job.id)}/cancel`, {
          method: "POST",
        }),
      );
      jobsChangedRef.current();
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setCancelling(false);
    }
  };

  if (!activeId)
    return (
      <section className="panel">
        <Empty
          title="Your next analysis starts here"
          icon={<Terminal size={26} />}
        >
          <p>
            Launch a pipeline to follow its stages, timing, and processing log
            here.
          </p>
          <Button onClick={onNewRun}>
            <Plus />
            Configure a run
          </Button>
        </Empty>
      </section>
    );

  const complete =
    job?.stages.filter((stage) => stage.status === "complete").length ?? 0;
  const count = job?.requested_stages.length ?? 0;
  const activeStage = job?.stages.find((stage) => stage.status === "running");
  const cancelPending = cancelling || job?.status === "cancelling";
  return (
    <div className="monitor-layout">
      <div className="section-heading">
        <div>
          <p className="eyebrow">PROCESSING</p>
          <h2>Watch the work unfold</h2>
          <p>Live stage status and the full story in the log.</p>
        </div>
        <div className="heading-actions">
          <label className="sr-only" htmlFor="monitor-job">
            Pipeline job
          </label>
          <select
            id="monitor-job"
            className="monitor-job-select"
            value={activeId}
            onChange={(event) => setActiveId(event.target.value)}
          >
            {jobs.map((item) => (
              <option key={item.id} value={item.id}>
                {item.name} · {formatDate(item.created_at)}
              </option>
            ))}
            {!jobs.some((item) => item.id === activeId) && (
              <option value={activeId}>{job?.name ?? "Current run"}</option>
            )}
          </select>
          <Button variant="outline" onClick={onNewRun}>
            <Plus />
            New run
          </Button>
        </div>
      </div>
      {error && <Notice>{error}</Notice>}
      {!job ? (
        <Loading label="Connecting to the pipeline…" />
      ) : (
        <>
          <section className="panel monitor-summary">
            <div className="monitor-summary-top">
              <div>
                <div className="monitor-title">
                  <h3>{job.name}</h3>
                  <Status value={job.status} />
                </div>
                <p className="monitor-cache">{job.cache_dir}</p>
              </div>
              <div className="heading-actions">
                {!terminal(job.status) && (
                  <Button
                    variant="outline"
                    disabled={cancelPending}
                    onClick={() => void cancel()}
                  >
                    {cancelPending ? (
                      <LoaderCircle className="animate-spin" />
                    ) : (
                      <Square />
                    )}
                    {cancelPending ? "Stopping workers…" : "Cancel run"}
                  </Button>
                )}
                <Button onClick={() => onViewRun(job.cache_dir)}>
                  View results
                  <ArrowRight />
                </Button>
              </div>
            </div>
            <div className="monitor-progress-caption">
              <span>
                <strong>{complete}</strong> / {count} stages complete
              </span>
              <span>
                {activeStage
                  ? humanize(activeStage.stage)
                  : terminal(job.status)
                    ? "Processing finished"
                    : "Waiting for the first stage"}
              </span>
            </div>
            <div
              className="monitor-progress"
              role="progressbar"
              aria-label="Completed pipeline stages"
              aria-valuenow={complete}
              aria-valuemin={0}
              aria-valuemax={count || 1}
            >
              <span
                style={{ width: `${count ? (complete / count) * 100 : 0}%` }}
              />
            </div>
            <div className="monitor-metadata">
              <span>
                <Clock3 size={13} />
                Started {formatDate(job.started_at)}
              </span>
              <span>
                Stages have different runtimes; this is not a time estimate.
              </span>
            </div>
            {job.error && (
              <Notice tone={job.status === "cancelled" ? "info" : "error"}>
                {job.error}
              </Notice>
            )}
          </section>
          <div className="monitor-content">
            <section className="panel monitor-stages">
              <div className="panel-heading">
                <h3>Pipeline stages</h3>
                <span>{count} requested</span>
              </div>
              <ol>
                {job.stages.map((stage, index) => (
                  <li
                    key={stage.stage}
                    className={cn(
                      "monitor-stage",
                      `monitor-stage-${stage.status}`,
                    )}
                  >
                    <span className="monitor-stage-icon">
                      {stage.status === "complete" ? (
                        <Check size={15} />
                      ) : stage.status === "running" ? (
                        <LoaderCircle className="animate-spin" size={15} />
                      ) : (
                        <Circle size={13} />
                      )}
                    </span>
                    <div>
                      <span className="monitor-stage-name">
                        {String(index + 1).padStart(2, "0")} ·{" "}
                        {humanize(stage.stage)}
                      </span>
                      <span className="monitor-stage-state">
                        {stage.status}
                        {stage.error ? ` · ${stage.error}` : ""}
                      </span>
                    </div>
                    <span className="monitor-stage-duration">
                      {duration(stage.seconds)}
                    </span>
                  </li>
                ))}
              </ol>
            </section>
            <section className="panel monitor-log-panel">
              <div className="panel-heading">
                <h3>
                  <Terminal size={16} />
                  Processing log
                </h3>
                <span
                  className={cn(
                    "monitor-connection",
                    connection === "live" && "monitor-connection-live",
                  )}
                >
                  {connection === "live" ? (
                    <Radio size={13} />
                  ) : connection === "polling" ? (
                    <WifiOff size={13} />
                  ) : (
                    <Circle size={10} />
                  )}
                  {connection === "live"
                    ? "Live"
                    : connection === "polling"
                      ? "Reconnecting · API updates"
                      : connection === "finished"
                        ? "Saved log"
                        : "Connecting"}
                </span>
              </div>
              <pre
                ref={logRef}
                className="monitor-log"
                aria-label="Processing log"
                tabIndex={0}
              >
                {job.logs.length
                  ? job.logs.join("\n")
                  : "Waiting for pipeline output…"}
              </pre>
              <div className="monitor-log-footer">
                <label>
                  <input
                    type="checkbox"
                    checked={followLog}
                    onChange={(event) => setFollowLog(event.target.checked)}
                  />
                  Follow latest output
                </label>
                <span>
                  Latest {job.logs.length} lines · full log saved locally
                </span>
              </div>
            </section>
          </div>
          <details className="panel monitor-command">
            <summary>Exact pipeline command</summary>
            <p>
              Run from the repository root. The generated settings file stays
              available locally.
            </p>
            <pre>{job.command}</pre>
            <CopyButton text={job.command} label="Copy command" />
          </details>
        </>
      )}
    </div>
  );
}
