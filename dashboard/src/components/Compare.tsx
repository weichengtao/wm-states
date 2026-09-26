import { useEffect, useState } from "react";
import {
  ArrowLeftRight,
  GitCompareArrows,
  Layers3,
  PanelLeft,
  Plus,
} from "lucide-react";
import type { Artifact, Run, RunDetail, SessionData } from "@/lib/types";
import { comparisonWarnings, settingsDifferences } from "@/lib/comparison";
import { humanize } from "@/lib/utils";
import { Button } from "./ui/button";
import { Empty, Loading, Notice } from "./shared";
import {
  SessionSelect,
  SessionStats,
  useRunDetail,
  useSession,
} from "./Results";
import ConfidenceChart from "./ConfidenceChart";
import { FigurePreview } from "./Artifacts";
function FigureChoice({
  detail,
  session,
  side,
}: {
  detail: RunDetail;
  session: string;
  side: string;
}) {
  const allFigures = detail.artifacts.filter(
    (a) => a.kind === "figure" && (!a.session || a.session === session),
  );
  const [stage, setStage] = useState("states");
  const stages = [...new Set(allFigures.map((a) => a.stage))];
  const activeStage = stages.includes(stage) ? stage : "all";
  const figures = allFigures.filter(
    (a) => activeStage === "all" || a.stage === activeStage,
  );
  const [selected, setSelected] = useState("");
  useEffect(() => {
    const preferred = figures.find(
      (a) =>
        a.stage === "states" &&
        a.path.includes("masks") &&
        a.path.endsWith(".png"),
    );
    setSelected(
      preferred?.path ??
        figures.find((a) => a.path.endsWith(".png"))?.path ??
        figures[0]?.path ??
        "",
    );
  }, [detail, session, activeStage]);
  const artifact: Artifact | undefined = figures.find(
    (a) => a.path === selected,
  );
  return (
    <div className="comparison-figure">
      <label htmlFor={`figure-${side}`}>Compare an output figure</label>
      <select
        className="select-control"
        aria-label={`Figure stage ${side}`}
        value={activeStage}
        onChange={(e) => setStage(e.target.value)}
      >
        <option value="all">All stages</option>
        {stages.map((value) => (
          <option key={value} value={value}>
            {humanize(value)}
          </option>
        ))}
      </select>
      <select
        id={`figure-${side}`}
        className="select-control"
        value={selected}
        onChange={(e) => setSelected(e.target.value)}
      >
        {!figures.length && <option value="">No figures yet</option>}
        {figures.map((a) => (
          <option key={a.path} value={a.path}>
            {a.path}
          </option>
        ))}
      </select>
      {artifact && <FigurePreview compact artifact={artifact} />}
    </div>
  );
}
function Pane({
  detail,
  data,
  error,
  label,
  color,
  split,
}: {
  detail: RunDetail | null;
  data: SessionData | null;
  error: string;
  label: string;
  color: string;
  split: boolean;
}) {
  return (
    <section
      className="panel comparison-pane"
      style={{ "--pane-color": color } as React.CSSProperties}
    >
      <div className="comparison-pane-heading">
        <span className="comparison-letter">{label}</span>
        <div>
          <h3>{data ? `Session ${data.session}` : "Session results"}</h3>
          <p>
            {detail?.run.name ?? "Choose a run"}
            {data && ` · preferred cue ${data.cue ?? "—"}`}
          </p>
        </div>
      </div>
      {error ? (
        <Notice>{error}</Notice>
      ) : !detail ? (
        <Loading />
      ) : !detail.sessions.length ? (
        <Empty title="No sessions to compare">
          Choose a run that has screening or decoding results.
        </Empty>
      ) : !data ? (
        <Loading label="Reading session…" />
      ) : (
        <>
          <SessionStats data={data} />
          {data.errors.map((e, i) => (
            <Notice key={i}>{e}</Notice>
          ))}
          {data.warnings.map((w, i) => (
            <Notice key={i} tone="info">
              {w}
            </Notice>
          ))}
          {split && (
            <ConfidenceChart
              series={[{ label: `${label} · ${data.session}`, color, data }]}
              height={230}
            />
          )}
          <FigureChoice detail={detail} session={data.session} side={label} />
        </>
      )}
    </section>
  );
}
export default function Compare({
  runs,
  initialRun,
  onNewRun,
}: {
  runs: Run[];
  initialRun: string | null;
  onNewRun: () => void;
}) {
  const [mode, setMode] = useState("sessions");
  const [view, setView] = useState("split");
  const [leftRun, setLeftRun] = useState(initialRun ?? runs[0]?.id ?? "");
  const [rightRun, setRightRun] = useState(initialRun ?? runs[0]?.id ?? "");
  const [leftSession, setLeftSession] = useState("");
  const [rightSession, setRightSession] = useState("");
  const left = useRunDetail(leftRun || null);
  const right = useRunDetail(rightRun || null);
  const a = useSession(leftRun, leftSession);
  const b = useSession(rightRun, rightSession);
  useEffect(() => {
    if (left.detail)
      setLeftSession((previous) =>
        left.detail!.sessions.some((s) => s.id === previous)
          ? previous
          : (left.detail!.sessions[0]?.id ?? ""),
      );
  }, [left.detail]);
  useEffect(() => {
    if (right.detail)
      setRightSession((previous) =>
        right.detail!.sessions.some((s) => s.id === previous)
          ? previous
          : (right.detail!.sessions[mode === "sessions" ? 1 : 0]?.id ??
            right.detail!.sessions[0]?.id ??
            ""),
      );
  }, [right.detail, mode]);
  function changeMode(value: string) {
    setMode(value);
    if (value === "sessions") setRightRun(leftRun);
    else
      setRightRun(
        runs.find((r) => r.id !== leftRun && r.session_count > 0)?.id ??
          leftRun,
      );
    setRightSession("");
  }
  function swap() {
    setLeftRun(rightRun);
    setRightRun(leftRun);
    setLeftSession(rightSession);
    setRightSession(leftSession);
  }
  const warnings = a.data && b.data ? comparisonWarnings(a.data, b.data) : [];
  const differences =
    left.detail && right.detail
      ? settingsDifferences(left.detail.manifests, right.detail.manifests)
      : [];
  return (
    <>
      <div className="page-heading">
        <div>
          <div className="eyebrow">SIDE BY SIDE, WITH CONTEXT</div>
          <h1>Compare & discover</h1>
          <p>
            Explore differences across sessions, or see how a new run changes
            the result.
          </p>
        </div>
        <div className="segmented">
          <button
            onClick={() => changeMode("sessions")}
            className={mode === "sessions" ? "active" : ""}
          >
            Across sessions
          </button>
          <button
            onClick={() => changeMode("runs")}
            className={mode === "runs" ? "active" : ""}
          >
            Across runs
          </button>
        </div>
      </div>
      {!runs.length ? (
        <Empty title="Start with an analysis run" icon={<GitCompareArrows />}>
          <p>
            Results from your next pipeline will be available here for
            comparison.
          </p>
          <Button onClick={onNewRun}>
            <Plus />
            New analysis
          </Button>
        </Empty>
      ) : (
        <>
          <section className="panel comparison-controls">
            <div className="comparison-picker">
              <span className="comparison-letter">A</span>
              <label className="field">
                <span>Analysis run</span>
                <select
                  className="select-control"
                  aria-label="Run A"
                  value={leftRun}
                  onChange={(e) => {
                    setLeftRun(e.target.value);
                    if (mode === "sessions") setRightRun(e.target.value);
                  }}
                >
                  {runs.map((r) => (
                    <option key={r.id} value={r.id}>
                      {r.name}
                    </option>
                  ))}
                </select>
              </label>
              <SessionSelect
                label="Session A"
                value={leftSession}
                sessions={left.detail?.sessions ?? []}
                onChange={setLeftSession}
              />
            </div>
            <Button
              size="icon"
              variant="outline"
              onClick={swap}
              aria-label="Swap comparison sides"
            >
              <ArrowLeftRight />
            </Button>
            <div className="comparison-picker pane-b">
              <span className="comparison-letter">B</span>
              <label className="field">
                <span>Analysis run</span>
                <select
                  className="select-control"
                  aria-label="Run B"
                  disabled={mode === "sessions"}
                  value={rightRun}
                  onChange={(e) => setRightRun(e.target.value)}
                >
                  {runs.map((r) => (
                    <option key={r.id} value={r.id}>
                      {r.name}
                    </option>
                  ))}
                </select>
              </label>
              <SessionSelect
                label="Session B"
                value={rightSession}
                sessions={right.detail?.sessions ?? []}
                onChange={setRightSession}
              />
            </div>
          </section>
          <div className="compare-subtoolbar">
            <p>
              <span className="tiny-dot" />
              Shared confidence scale · independent figure viewers
            </p>
            <div className="segmented">
              <button
                onClick={() => setView("split")}
                className={view === "split" ? "active" : ""}
              >
                <PanelLeft size={14} />
                Split screen
              </button>
              <button
                onClick={() => setView("overlay")}
                className={view === "overlay" ? "active" : ""}
              >
                <Layers3 size={14} />
                Overlay curves
              </button>
            </div>
          </div>
          {warnings.map((w, i) => (
            <Notice tone="info" key={i}>
              {w}
            </Notice>
          ))}
          {leftRun === rightRun &&
            leftSession === rightSession &&
            leftSession && (
              <Notice tone="info">
                Both sides show the same session and run. Choose another session
                or switch to across runs.
              </Notice>
            )}
          {view === "overlay" && a.data && b.data && (
            <section className="panel overlay-panel">
              <div className="section-heading">
                <div>
                  <h3>Decoding confidence overlay</h3>
                  <p>
                    Each curve retains its original time grid and null
                    distribution.
                  </p>
                </div>
              </div>
              <ConfidenceChart
                series={[
                  {
                    label: `A · ${a.data.session}`,
                    color: "#6559de",
                    data: a.data,
                  },
                  {
                    label: `B · ${b.data.session}`,
                    color: "#2caaa6",
                    data: b.data,
                  },
                ]}
                height={300}
              />
            </section>
          )}
          {mode === "runs" && left.detail && right.detail && (
            <details className="panel config-comparison">
              <summary>
                Configuration differences{" "}
                <span className="count-badge">{differences.length}</span>
              </summary>
              <p className="fine-print">
                Latest recorded settings per stage across invocation history.
                Cache and data paths are omitted; missing history cannot verify
                equivalent configurations.
              </p>
              {differences.length ? (
                <div className="table-scroll">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Stage / parameter</th>
                        <th>Run A</th>
                        <th>Run B</th>
                      </tr>
                    </thead>
                    <tbody>
                      {differences.map((d) => (
                        <tr key={`${d.stage}.${d.parameter}`}>
                          <td>
                            {d.stage} / {d.parameter}
                          </td>
                          <td>{d.left}</td>
                          <td>{d.right}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p className="muted">
                  No differences in recorded stage parameters.
                </p>
              )}
            </details>
          )}
          <div className="comparison-grid">
            <Pane
              detail={left.detail}
              data={a.data}
              error={left.error || a.error}
              label="A"
              color="#6559de"
              split={view === "split"}
            />
            <Pane
              detail={right.detail}
              data={b.data}
              error={right.error || b.error}
              label="B"
              color="#2caaa6"
              split={view === "split"}
            />
          </div>
          <p className="fine-print compare-footnote">
            Compare the configuration and trial sets before interpreting
            differences. Figures retain the axes and scales used when they were
            generated.
          </p>
        </>
      )}
    </>
  );
}
