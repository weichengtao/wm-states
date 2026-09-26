import { useEffect, useState } from "react";
import {
  ArrowLeftRight,
  GitCompareArrows,
  Layers3,
  PanelLeft,
  Plus,
} from "lucide-react";
import type { Artifact, Run, RunDetail, SessionData } from "@/lib/types";
import {
  chooseComparisonSession,
  comparisonWarnings,
  matchingSession,
  settingsDifferences,
} from "@/lib/comparison";
import { humanize } from "@/lib/utils";
import { Button } from "./ui/button";
import { Select } from "./ui/select";
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
      <Select
        className="select-control"
        aria-label={`Figure stage ${side}`}
        value={activeStage}
        onValueChange={setStage}
        options={[
          { value: "all", label: "All stages" },
          ...stages.map((value) => ({ value, label: humanize(value) })),
        ]}
      />
      <Select
        id={`figure-${side}`}
        aria-label={`Output figure ${side}`}
        className="select-control"
        disabled={!figures.length}
        value={selected}
        onValueChange={setSelected}
        placeholder="Choose an output figure"
        options={
          figures.length
            ? figures.map((figure) => ({
                value: figure.path,
                label: figure.name.replaceAll("_", " "),
                description: figure.path,
              }))
            : [{ value: "", label: "No figures yet" }]
        }
      />
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
  const [mode, setMode] = useState<"sessions" | "runs">("sessions");
  const [view, setView] = useState("split");
  const [leftRun, setLeftRun] = useState(initialRun ?? runs[0]?.id ?? "");
  const [rightRun, setRightRun] = useState(initialRun ?? runs[0]?.id ?? "");
  const [leftSession, setLeftSession] = useState("");
  const [rightSession, setRightSession] = useState("");
  const left = useRunDetail(leftRun || null);
  const right = useRunDetail(rightRun || null);
  const a = useSession(leftRun, leftSession);
  const b = useSession(rightRun, rightSession);
  const selectedLeft = left.detail?.sessions.find(
    (session) => session.id === leftSession,
  );
  const matchedRight = matchingSession(
    selectedLeft,
    right.detail?.sessions ?? [],
  );
  useEffect(() => {
    if (!runs.length) return;
    const primary = runs.some((run) => run.id === leftRun)
      ? leftRun
      : (runs.find((run) => run.id === initialRun)?.id ?? runs[0].id);
    if (primary !== leftRun) setLeftRun(primary);
    const secondary =
      mode === "sessions"
        ? primary
        : runs.some((run) => run.id === rightRun)
          ? rightRun
          : (runs.find((run) => run.id !== primary)?.id ?? primary);
    if (secondary !== rightRun) setRightRun(secondary);
  }, [runs, initialRun, leftRun, rightRun, mode]);
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
        chooseComparisonSession(
          right.detail!.sessions,
          previous,
          selectedLeft ?? left.detail?.sessions[0],
          mode,
        ),
      );
  }, [right.detail, mode]);
  function changeMode(value: "sessions" | "runs") {
    if (value === mode) return;
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
        <div className="segmented" role="group" aria-label="Comparison mode">
          <button
            type="button"
            aria-pressed={mode === "sessions"}
            onClick={() => changeMode("sessions")}
            className={mode === "sessions" ? "active" : ""}
          >
            Across sessions
          </button>
          <button
            type="button"
            aria-pressed={mode === "runs"}
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
          <div className="comparison-context">
            <div>
              <strong>
                {mode === "sessions"
                  ? "Two sessions, one analysis run"
                  : "Compare results across analysis runs"}
              </strong>
              <p>
                {mode === "sessions"
                  ? "Choose a run and two sessions below. Both sides use the same run."
                  : "Choose a run for each side. Match the session to compare the same recording, or choose sessions independently."}
              </p>
            </div>
            {mode === "runs" && left.detail && right.detail && (
              <div className="comparison-match">
                {matchedRight && matchedRight.id === rightSession ? (
                  <span className="chart-tag">
                    Same session · {matchedRight.session}
                  </span>
                ) : matchedRight ? (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setRightSession(matchedRight.id)}
                  >
                    <GitCompareArrows /> Match session A
                  </Button>
                ) : (
                  <span className="muted">
                    Session A is not available in run B
                  </span>
                )}
              </div>
            )}
          </div>
          <section className="panel comparison-controls">
            <div className="comparison-picker">
              <span className="comparison-letter">A</span>
              <label className="field" htmlFor="comparison-run-a">
                <span>Analysis run</span>
                <Select
                  id="comparison-run-a"
                  className="select-control"
                  aria-label="Run A"
                  value={leftRun}
                  onValueChange={(value) => {
                    setLeftRun(value);
                    if (mode === "sessions") setRightRun(value);
                  }}
                  options={runs.map((run) => ({
                    value: run.id,
                    label: `${run.name}${run.name !== run.id ? ` · ${run.id}` : ""}`,
                    description: run.path,
                  }))}
                />
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
              title="Swap comparison sides"
              disabled={!leftSession || !rightSession}
            >
              <ArrowLeftRight />
            </Button>
            <div className="comparison-picker pane-b">
              <span className="comparison-letter">B</span>
              <label className="field" htmlFor="comparison-run-b">
                <span>
                  {mode === "sessions" ? "Same analysis run" : "Analysis run"}
                </span>
                <Select
                  id="comparison-run-b"
                  className="select-control"
                  aria-label="Run B"
                  disabled={mode === "sessions"}
                  value={rightRun}
                  onValueChange={setRightRun}
                  options={runs.map((run) => ({
                    value: run.id,
                    label: `${run.name}${run.name !== run.id ? ` · ${run.id}` : ""}`,
                    description: run.path,
                  }))}
                />
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
            <div
              className="segmented"
              role="group"
              aria-label="Comparison layout"
            >
              <button
                type="button"
                aria-pressed={view === "split"}
                onClick={() => setView("split")}
                className={view === "split" ? "active" : ""}
              >
                <PanelLeft size={14} />
                Split screen
              </button>
              <button
                type="button"
                aria-pressed={view === "overlay"}
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
