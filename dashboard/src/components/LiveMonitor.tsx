import { useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  ArrowRight,
  Check,
  Circle,
  Clock3,
  Download,
  LoaderCircle,
  Plus,
  Radio,
  Search,
  Square,
  Terminal,
  WifiOff,
  X,
  XCircle,
} from "lucide-react";
import { api, errorMessage } from "@/lib/api";
import type { Job } from "@/lib/types";
import { cn, duration, formatDate, humanize } from "@/lib/utils";
import { Button } from "./ui/button";
import { Select } from "./ui/select";
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
  const [confirmCancel, setConfirmCancel] = useState(false);
  const [followLog, setFollowLog] = useState(true);
  const [logQuery, setLogQuery] = useState("");
  const [wrapLog, setWrapLog] = useState(false);
  const logRef = useRef<HTMLPreElement>(null);
  const stopButtonRef = useRef<HTMLButtonElement>(null);
  const keepRunningRef = useRef<HTMLButtonElement>(null);
  const jobsChangedRef = useRef(onJobsChanged);
  jobsChangedRef.current = onJobsChanged;

  useEffect(() => {
    if (selectedId) setActiveId(selectedId);
  }, [selectedId]);
  useEffect(() => {
    if (confirmCancel && !cancelling) keepRunningRef.current?.focus();
  }, [confirmCancel, cancelling]);
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
    setConfirmCancel(false);
    setLogQuery("");
    setFollowLog(true);
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
        if (!disposed && !done) {
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
    if (followLog && !logQuery.trim() && logRef.current)
      logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [job?.logs, followLog, logQuery, wrapLog]);

  const cancel = async () => {
    if (!job || terminal(job.status) || job.status === "cancelling") return;
    setCancelling(true);
    setError("");
    try {
      setSnapshot(
        await api<Job>(`/jobs/${encodeURIComponent(job.id)}/cancel`, {
          method: "POST",
        }),
      );
      setConfirmCancel(false);
      jobsChangedRef.current();
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setCancelling(false);
    }
  };
  const dismissCancel = () => {
    setConfirmCancel(false);
    stopButtonRef.current?.focus();
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
  const connectionStatus =
    job && terminal(job.status) ? "finished" : connection;
  const query = logQuery.trim().toLocaleLowerCase();
  const visibleLogs =
    job?.logs.filter(
      (line) => !query || line.toLocaleLowerCase().includes(query),
    ) ?? [];
  const logText = visibleLogs.join("\n");
  const downloadLog = () => {
    if (!job || !visibleLogs.length) return;
    const url = URL.createObjectURL(
      new Blob([`${logText}\n`], { type: "text/plain;charset=utf-8" }),
    );
    const link = document.createElement("a");
    link.href = url;
    link.download = `${job.name.replace(/[^a-zA-Z0-9_-]+/g, "-") || "pipeline"}-${job.id}${query ? "-filtered" : ""}-log.txt`;
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  return (
    <div className="monitor-layout">
      <div className="section-heading">
        <div>
          <p className="eyebrow">PROCESSING</p>
          <h2>Follow your analysis</h2>
          <p>
            Track each stage and inspect the processing log as results arrive.
          </p>
        </div>
        <div className="heading-actions">
          <label className="sr-only" htmlFor="monitor-job">
            Pipeline job
          </label>
          <Select
            id="monitor-job"
            className="monitor-job-select"
            value={activeId}
            onValueChange={setActiveId}
            options={[
              ...jobs.map((item) => ({
                value: item.id,
                label: `${item.name} · ${humanize(item.status)}`,
                description: formatDate(item.created_at),
              })),
              ...(!jobs.some((item) => item.id === activeId)
                ? [
                    {
                      value: activeId,
                      label: job?.name ?? "Current run",
                      description: job ? formatDate(job.created_at) : undefined,
                    },
                  ]
                : []),
            ]}
          />
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
                    ref={stopButtonRef}
                    variant="outline"
                    disabled={cancelPending}
                    aria-expanded={confirmCancel}
                    aria-controls="cancel-run-confirmation"
                    onClick={() => setConfirmCancel((value) => !value)}
                  >
                    {cancelPending ? (
                      <LoaderCircle className="animate-spin" />
                    ) : (
                      <Square />
                    )}
                    {cancelPending ? "Stopping workers…" : "Stop run"}
                  </Button>
                )}
                <Button onClick={() => onViewRun(job.cache_dir)}>
                  View results
                  <ArrowRight />
                </Button>
              </div>
            </div>
            {confirmCancel && !terminal(job.status) && !cancelPending && (
              <div
                id="cancel-run-confirmation"
                className="monitor-cancel-confirmation"
                role="region"
                aria-label="Confirm stopping this run"
                onKeyDown={(event) => {
                  if (event.key === "Escape") dismissCancel();
                }}
              >
                <AlertTriangle size={19} aria-hidden="true" />
                <div>
                  <strong>Stop this run?</strong>
                  <p>
                    Running workers will stop. Files already saved remain in the
                    run folder.
                  </p>
                </div>
                <div className="monitor-cancel-actions">
                  <Button
                    ref={keepRunningRef}
                    variant="outline"
                    size="sm"
                    onClick={dismissCancel}
                  >
                    Keep running
                  </Button>
                  <Button
                    variant="destructive"
                    size="sm"
                    onClick={() => void cancel()}
                  >
                    <Square />
                    Yes, stop run
                  </Button>
                </div>
              </div>
            )}
            <div className="monitor-progress-caption">
              <span>
                <strong>{complete}</strong> / {count} stages complete
              </span>
              <span className="monitor-progress-detail">
                {cancelPending
                  ? "Waiting for workers to stop"
                  : job.status === "complete"
                    ? "All requested stages finished"
                    : job.status === "failed"
                      ? "Run stopped with an error"
                      : job.status === "cancelled"
                        ? "Run stopped by request"
                        : activeStage
                          ? `Now: ${humanize(activeStage.stage)}`
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
              aria-valuetext={`${complete} of ${count} stages complete${activeStage ? `, ${humanize(activeStage.stage)} running` : ""}`}
            >
              <span
                style={{ width: `${count ? (complete / count) * 100 : 0}%` }}
              />
            </div>
            <div className="monitor-metadata">
              <span>
                <Clock3 size={13} />
                {job.started_at
                  ? `Started ${formatDate(job.started_at)}`
                  : `Queued ${formatDate(job.created_at)}`}
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
                    aria-current={
                      stage.status === "running" ? "step" : undefined
                    }
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
                      ) : ["failed", "cancelled"].includes(stage.status) ? (
                        <XCircle size={15} />
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
                        {humanize(stage.status)}
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
                  role="status"
                  className={cn(
                    "monitor-connection",
                    connectionStatus === "live" && "monitor-connection-live",
                  )}
                >
                  {connectionStatus === "live" ? (
                    <Radio size={13} />
                  ) : connectionStatus === "polling" ? (
                    <WifiOff size={13} />
                  ) : (
                    <Circle size={10} />
                  )}
                  {connectionStatus === "live"
                    ? "Live"
                    : connectionStatus === "polling"
                      ? "Reconnecting · API updates"
                      : connectionStatus === "finished"
                        ? "Saved log"
                        : "Connecting"}
                </span>
              </div>
              {connectionStatus === "polling" && (
                <p className="monitor-connection-note">
                  Live connection interrupted. Checking every 5 seconds while
                  reconnecting; your analysis can continue.
                </p>
              )}
              <div className="monitor-log-toolbar">
                <div className="monitor-log-search">
                  <Search size={15} aria-hidden="true" />
                  <input
                    type="search"
                    aria-label="Filter log lines"
                    placeholder="Filter log lines…"
                    value={logQuery}
                    onChange={(event) => setLogQuery(event.target.value)}
                  />
                  {logQuery && (
                    <Button
                      variant="ghost"
                      size="icon"
                      aria-label="Clear log filter"
                      onClick={() => setLogQuery("")}
                    >
                      <X />
                    </Button>
                  )}
                </div>
                <div className="monitor-log-actions">
                  <CopyButton
                    text={logText}
                    label="Copy log"
                    disabled={!visibleLogs.length}
                  />
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={!visibleLogs.length}
                    onClick={downloadLog}
                    title="Download the lines currently shown"
                  >
                    <Download />
                    Save log
                  </Button>
                </div>
              </div>
              <pre
                ref={logRef}
                className="monitor-log"
                aria-label="Processing log"
                tabIndex={0}
                style={
                  wrapLog
                    ? { whiteSpace: "pre-wrap", overflowWrap: "anywhere" }
                    : undefined
                }
                onScroll={(event) => {
                  if (query) return;
                  const element = event.currentTarget;
                  setFollowLog(
                    element.scrollHeight -
                      element.scrollTop -
                      element.clientHeight <
                      36,
                  );
                }}
              >
                {job.logs.length
                  ? logText || "No log lines match this filter."
                  : terminal(job.status)
                    ? "No processing output was recorded."
                    : "Waiting for pipeline output…"}
              </pre>
              <div className="monitor-log-footer">
                <div className="monitor-log-options">
                  <label>
                    <input
                      type="checkbox"
                      checked={followLog && !query}
                      disabled={Boolean(query)}
                      onChange={(event) => setFollowLog(event.target.checked)}
                    />
                    Follow output
                  </label>
                  <label>
                    <input
                      type="checkbox"
                      checked={wrapLog}
                      onChange={(event) => setWrapLog(event.target.checked)}
                    />
                    Wrap lines
                  </label>
                </div>
                <span className="monitor-log-count">
                  {query
                    ? `${visibleLogs.length} of ${job.logs.length} lines`
                    : `Latest ${job.logs.length} lines`}{" "}
                  · full log saved locally
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
