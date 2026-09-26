import { useEffect, useState } from "react";
import {
  Activity,
  ArrowDownToLine,
  ArrowRight,
  BarChart3,
  CalendarDays,
  Check,
  ChevronDown,
  CircleDot,
  Files,
  GitCompareArrows,
  Layers3,
  Plus,
  RefreshCw,
  Search,
  Terminal,
  Users,
} from "lucide-react";
import type {
  Run,
  RunDetail,
  RunRequest,
  Session,
  SessionData,
} from "@/lib/types";
import { api, errorMessage, runPath } from "@/lib/api";
import { duration, formatDate, formatNumber, humanize } from "@/lib/utils";
import { manifestSeed } from "@/lib/configuration";
import { Button } from "./ui/button";
import { CopyButton, Empty, Loading, Notice, Stat, Status } from "./shared";
import ConfidenceChart from "./ConfidenceChart";
import { FigureGallery, SupportingFiles, TableBrowser } from "./Artifacts";
export function useRunDetail(id: string | null) {
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [error, setError] = useState("");
  const [version, setVersion] = useState(0);
  useEffect(() => {
    setDetail(null);
    setError("");
    if (!id) return;
    let stopped = false;
    api<RunDetail>(runPath(id))
      .then((d) => {
        if (!stopped) setDetail(d);
      })
      .catch((e) => {
        if (!stopped) setError(errorMessage(e));
      });
    return () => {
      stopped = true;
    };
  }, [id, version]);
  return { detail, error, reload: () => setVersion((v) => v + 1) };
}
export function useSession(runId: string | null, session: string | null) {
  const [data, setData] = useState<SessionData | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    setData(null);
    setError("");
    if (!runId || !session) return;
    let stopped = false;
    api<SessionData>(
      `${runPath(runId)}/sessions/${encodeURIComponent(session)}`,
    )
      .then((d) => {
        if (!stopped) setData(d);
      })
      .catch((e) => {
        if (!stopped) setError(errorMessage(e));
      });
    return () => {
      stopped = true;
    };
  }, [runId, session]);
  return { data, error };
}
export function SessionSelect({
  sessions,
  value,
  onChange,
  label = "Session",
}: {
  sessions: Session[];
  value: string;
  onChange: (id: string) => void;
  label?: string;
}) {
  return (
    <label className="session-selector">
      <span>{label}</span>
      <select
        aria-label={label}
        className="select-control"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        {!sessions.length && <option value="">No sessions yet</option>}
        {sessions.map((s) => (
          <option key={s.id} value={s.id}>
            {s.session} · cue {s.cue ?? "—"}
          </option>
        ))}
      </select>
    </label>
  );
}
function mean(values: number[]) {
  return values.length
    ? values.reduce((a, b) => a + b, 0) / values.length
    : null;
}
export function SessionStats({ data }: { data: SessionData }) {
  return (
    <div className="session-metrics">
      <div>
        <span>Trials</span>
        <strong>{formatNumber(data.trial_count)}</strong>
      </div>
      <div>
        <span>Decoded cells</span>
        <strong>{formatNumber(data.cell_count)}</strong>
      </div>
      <div>
        <span>Accuracy</span>
        <strong>
          {data.metrics.accuracy == null
            ? "—"
            : `${formatNumber(data.metrics.accuracy * 100, 1)}%`}
        </strong>
      </div>
      <div>
        <span>Mean total OFF</span>
        <strong>
          {formatNumber(mean(data.total_off_durations), 1)}
          <small> ms</small>
        </strong>
      </div>
    </div>
  );
}
export default function Results({
  runs,
  loading,
  error,
  selectedId,
  onSelect,
  onRefresh,
  onConfigure,
  onCompare,
}: {
  runs: Run[];
  loading: boolean;
  error: string;
  selectedId: string | null;
  onSelect: (id: string) => void;
  onRefresh: () => void;
  onConfigure: (seed?: Partial<RunRequest>) => void;
  onCompare: (id: string) => void;
}) {
  const [query, setQuery] = useState("");
  const [tab, setTab] = useState("overview");
  const [session, setSession] = useState("");
  const { detail, error: detailError, reload } = useRunDetail(selectedId);
  const { data, error: sessionError } = useSession(selectedId, session);
  useEffect(() => {
    setSession("");
    setTab("overview");
  }, [selectedId]);
  useEffect(() => {
    if (detail)
      setSession((prev) =>
        detail.sessions.some((s) => s.id === prev)
          ? prev
          : (detail.sessions[0]?.id ?? ""),
      );
  }, [detail]);
  const filtered = runs.filter((r) =>
    `${r.id} ${r.name}`.toLowerCase().includes(query.toLowerCase()),
  );
  const last = detail?.manifests[0];
  function reuse() {
    if (last) onConfigure(manifestSeed(last, detail?.run.name ?? "Analysis"));
  }
  return (
    <>
      <div className="page-heading">
        <div>
          <div className="eyebrow">
            <span className="tiny-dot" /> YOUR ANALYSIS WORKSPACE
          </div>
          <h1>Your analysis, in focus.</h1>
          <p>
            From recordings to results. Explore every stage of your next
            pipeline.
          </p>
        </div>
        <Button onClick={() => onConfigure()}>
          <Plus />
          New analysis
        </Button>
      </div>
      <div className="stats-grid">
        <Stat
          label="Analysis runs"
          value={formatNumber(runs.length)}
          sub="Available in your local cache"
          icon={<Layers3 />}
        />
        <Stat
          label="Sessions across runs"
          value={formatNumber(runs.reduce((n, r) => n + r.session_count, 0))}
          sub="Includes sessions in multiple runs"
          icon={<Users />}
        />
        <Stat
          label="Latest invocations complete"
          value={formatNumber(
            runs.filter((r) => r.status === "complete").length,
          )}
          sub="Latest invocation in each cache"
          icon={<Check />}
        />
        <Stat
          label="Pipeline stages"
          value="11"
          sub="Screening through statistical models"
          icon={<Activity />}
        />
      </div>
      <section className="panel run-library">
        <div className="section-heading">
          <div>
            <h2>
              Run library <span className="count-badge">{runs.length}</span>
            </h2>
            <p>Choose a run to explore its sessions and outputs.</p>
          </div>
          <div className="heading-actions">
            <div className="search-field">
              <Search size={15} />
              <input
                aria-label="Search runs"
                placeholder="Find a run…"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
              />
            </div>
            <Button
              variant="ghost"
              size="icon"
              aria-label="Refresh runs"
              onClick={() => {
                onRefresh();
                reload();
              }}
            >
              <RefreshCw />
            </Button>
          </div>
        </div>
        {error ? (
          <Notice>{error}</Notice>
        ) : loading && !runs.length ? (
          <Loading label="Discovering analysis runs…" />
        ) : !filtered.length ? (
          <Empty
            title={query ? "No matching runs" : "Your first result starts here"}
          >
            <p>
              {query
                ? "Try another name."
                : "Launch an analysis using the example configuration, or browse an existing next pipeline cache."}
            </p>
            {!query && (
              <Button onClick={() => onConfigure()}>
                <Plus />
                Set up analysis
              </Button>
            )}
          </Empty>
        ) : (
          <div className="run-list">
            {filtered.map((run) => (
              <button
                className={`run-row ${selectedId === run.id ? "selected" : ""}`}
                key={run.id}
                onClick={() => onSelect(run.id)}
              >
                <span className="run-icon">
                  <Layers3 size={18} />
                </span>
                <span className="run-identity">
                  <strong>{run.name}</strong>
                  <span>{run.id}</span>
                </span>
                <span className="run-sessions">
                  <Users size={14} />
                  {run.session_count} sessions
                </span>
                <span className="run-date">{formatDate(run.updated_at)}</span>
                <Status value={run.status} />
                <ArrowRight size={16} className="run-arrow" />
              </button>
            ))}
          </div>
        )}
      </section>
      {selectedId && (
        <section className="result-workspace">
          {detailError ? (
            <Notice>{detailError}</Notice>
          ) : !detail ? (
            <Loading label="Reading run results…" />
          ) : (
            <>
              <div className="result-heading">
                <div>
                  <div className="eyebrow">EXPLORING RUN</div>
                  <h2>{detail.run.name}</h2>
                  <p>
                    <CalendarDays size={13} />
                    {formatDate(detail.run.updated_at)}
                    <span>·</span>
                    {detail.run.path}
                  </p>
                </div>
                <div className="heading-actions">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={reuse}
                    disabled={!last}
                  >
                    <ArrowDownToLine />
                    Reuse settings
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => onCompare(selectedId)}
                  >
                    <GitCompareArrows />
                    Compare
                  </Button>
                </div>
              </div>
              {detail.errors?.map((e, i) => (
                <Notice key={i}>{e}</Notice>
              ))}
              <div className="result-toolbar">
                <div className="tabs" role="tablist" aria-label="Run results">
                  {[
                    { id: "overview", label: "Overview", icon: BarChart3 },
                    { id: "figures", label: "Figures", icon: Files },
                    { id: "tables", label: "Tables", icon: Layers3 },
                    { id: "history", label: "Run history", icon: Terminal },
                  ].map((t) => (
                    <button
                      role="tab"
                      aria-selected={tab === t.id}
                      key={t.id}
                      className={tab === t.id ? "active" : ""}
                      onClick={() => setTab(t.id)}
                    >
                      <t.icon size={15} />
                      {t.label}
                      {t.id === "figures" && (
                        <span>
                          {
                            detail.artifacts.filter((a) => a.kind === "figure")
                              .length
                          }
                        </span>
                      )}
                    </button>
                  ))}
                </div>
                {tab !== "history" && (
                  <SessionSelect
                    sessions={detail.sessions}
                    value={session}
                    onChange={setSession}
                  />
                )}
              </div>
              <div className="result-content" role="tabpanel">
                {tab === "overview" && (
                  <>
                    {sessionError ? (
                      <Notice>{sessionError}</Notice>
                    ) : !session ? (
                      <Empty title="No session results yet">
                        Run cell screening and decoding to populate this
                        overview.
                      </Empty>
                    ) : !data ? (
                      <Loading label="Reading session…" />
                    ) : (
                      <>
                        {data.errors.map((e, i) => (
                          <Notice key={i}>{e}</Notice>
                        ))}
                        {data.warnings.map((e, i) => (
                          <Notice key={i} tone="info">
                            {e}
                          </Notice>
                        ))}
                        <SessionStats data={data} />
                        <div className="overview-grid">
                          <section className="panel chart-panel">
                            <div className="section-heading">
                              <div>
                                <h3>Decoding confidence</h3>
                                <p>
                                  Observed estimate and the null shuffle
                                  distribution
                                </p>
                              </div>
                              <span className="chart-tag">
                                Session {data.session}
                              </span>
                            </div>
                            <ConfidenceChart
                              series={[
                                { label: "Observed", color: "#6559de", data },
                              ]}
                            />
                          </section>
                          <section className="panel summary-panel">
                            <div className="section-heading">
                              <h3>Session at a glance</h3>
                              <CircleDot size={18} />
                            </div>
                            <dl>
                              <div>
                                <dt>Selected by screening</dt>
                                <dd>{formatNumber(data.selected_cells)}</dd>
                              </div>
                              <div>
                                <dt>Preferred cue</dt>
                                <dd>{data.cue ?? "—"}</dd>
                              </div>
                              <div>
                                <dt>Null shuffles</dt>
                                <dd>{formatNumber(data.null_shuffles)}</dd>
                              </div>
                              <div>
                                <dt>Brier score</dt>
                                <dd>
                                  {formatNumber(data.metrics.brier_score, 3)}
                                </dd>
                              </div>
                              <div>
                                <dt>Log loss</dt>
                                <dd>
                                  {formatNumber(data.metrics.log_loss, 3)}
                                </dd>
                              </div>
                              <div>
                                <dt>Mean longest OFF</dt>
                                <dd>
                                  {formatNumber(
                                    mean(data.max_off_durations),
                                    1,
                                  )}{" "}
                                  ms
                                </dd>
                              </div>
                            </dl>
                            <p className="fine-print">
                              Metrics reflect the stored evaluation window. Null
                              bands show the middle 95% of shuffled trial-mean
                              curves.
                            </p>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => setTab("figures")}
                            >
                              Explore session figures
                              <ArrowRight />
                            </Button>
                          </section>
                        </div>
                      </>
                    )}
                    <div className="panel stages-summary">
                      <div className="section-heading">
                        <h3>Recorded stage outcomes</h3>
                        <span className="muted">
                          Latest record for each stage
                        </span>
                      </div>
                      <div className="stage-chips">
                        {detail.run.stages.map((s) => (
                          <div key={s.stage}>
                            <span>{humanize(s.stage)}</span>
                            <Status value={s.status} />
                          </div>
                        ))}
                      </div>
                      <p className="fine-print">
                        Stage history can include partial invocations. Outputs
                        are shared within this cache directory.
                      </p>
                    </div>
                  </>
                )}
                {tab === "figures" && (
                  <FigureGallery
                    artifacts={detail.artifacts}
                    session={session}
                  />
                )}
                {tab === "tables" && (
                  <>
                    <TableBrowser
                      runId={selectedId}
                      artifacts={detail.artifacts}
                    />
                    <SupportingFiles artifacts={detail.artifacts} />
                  </>
                )}
                {tab === "history" && (
                  <>
                    <div className="history-intro">
                      <h3>Every invocation, preserved.</h3>
                      <p>
                        Exact commands and resolved settings for complete and
                        partial pipeline runs. Re-running a stage can replace
                        its outputs; the manifest history remains.
                      </p>
                    </div>
                    {detail.manifests.length ? (
                      detail.manifests.map((m, i) => (
                        <details
                          className="panel manifest-card"
                          key={m.id ?? i}
                          open={i === 0}
                        >
                          <summary>
                            <span className="manifest-icon">
                              <Terminal size={17} />
                            </span>
                            <span>
                              <strong>{formatDate(m.started_at)}</strong>
                              <small>
                                {m.stages?.map((s) => s.stage).join(" → ") ||
                                  "No recorded stages"}
                              </small>
                            </span>
                            <Status value={m.status ?? "unknown"} />
                            <ChevronDown size={16} />
                          </summary>
                          <div className="manifest-body">
                            <div className="section-heading">
                              <span className="eyebrow">EXACT INVOCATION</span>
                              {m.invocation?.command && (
                                <CopyButton text={m.invocation.command} />
                              )}
                            </div>
                            <pre>
                              {m.invocation?.command ??
                                "This older manifest did not record an invocation command."}
                            </pre>
                            <div className="stage-timings">
                              {m.stages?.map((s) => (
                                <div key={s.stage}>
                                  <span>{humanize(s.stage)}</span>
                                  <Status value={s.status} />
                                  <span>{duration(s.seconds)}</span>
                                </div>
                              ))}
                            </div>
                            <details>
                              <summary>Resolved settings & manifest</summary>
                              <pre>{JSON.stringify(m, null, 2)}</pre>
                            </details>
                          </div>
                        </details>
                      ))
                    ) : (
                      <Empty title="No manifests found">
                        Stage outputs may still be available in the other views.
                      </Empty>
                    )}
                  </>
                )}
              </div>
            </>
          )}
        </section>
      )}
    </>
  );
}
