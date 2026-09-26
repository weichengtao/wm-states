import { useCallback, useEffect, useRef, useState } from "react";
import {
  Activity,
  ArrowUpRight,
  BookOpen,
  Code2,
  ChevronRight,
  FlaskConical,
  GitCompareArrows,
  Layers3,
  Menu,
  Radio,
  SlidersHorizontal,
  X,
} from "lucide-react";
import { api, errorMessage } from "./lib/api";
import type { Job, Run, RunRequest, Schema } from "./lib/types";
import { Button } from "./components/ui/button";
import { Loading, Notice } from "./components/shared";
import Results from "./components/Results";
import Configure from "./components/Configure";
import Compare from "./components/Compare";
import LiveMonitor from "./components/LiveMonitor";
import HelpPanel from "./components/HelpPanel";
import GuideLink from "./components/GuideLink";
type Page = "results" | "configure" | "compare" | "monitor";
const navigation = [
  { id: "results", label: "Run library", icon: Layers3 },
  { id: "configure", label: "Configure pipeline", icon: SlidersHorizontal },
  { id: "compare", label: "Compare results", icon: GitCompareArrows },
  { id: "monitor", label: "Live progress", icon: Radio },
] as const;
export default function App() {
  const [page, setPage] = useState<Page>("results");
  const [runs, setRuns] = useState<Run[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [schema, setSchema] = useState<Schema | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [selectedJob, setSelectedJob] = useState<string | null>(null);
  const [seed, setSeed] = useState<Partial<RunRequest> | null>(null);
  const [configKey, setConfigKey] = useState(0);
  const [error, setError] = useState("");
  const [schemaError, setSchemaError] = useState("");
  const [loading, setLoading] = useState(true);
  const [online, setOnline] = useState<boolean | null>(null);
  const [menu, setMenu] = useState(false);
  const sidebarRef = useRef<HTMLElement>(null);
  const menuRef = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (!menu) return;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const sidebar = sidebarRef.current;
    const focusable = () =>
      Array.from(
        sidebar?.querySelectorAll<HTMLElement>(
          "a[href], button:not(:disabled)",
        ) ?? [],
      ).filter((element) => element.getClientRects().length > 0);
    focusable()[0]?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        setMenu(false);
      }
      if (event.key === "Tab") {
        const elements = focusable();
        const first = elements[0],
          last = elements.at(-1);
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault();
          last?.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault();
          first?.focus();
        }
      }
    };
    const breakpoint = window.matchMedia("(min-width: 761px)");
    const onResize = () => {
      if (breakpoint.matches) setMenu(false);
    };
    document.addEventListener("keydown", onKey);
    breakpoint.addEventListener("change", onResize);
    return () => {
      document.body.style.overflow = previousOverflow;
      document.removeEventListener("keydown", onKey);
      breakpoint.removeEventListener("change", onResize);
      menuRef.current?.focus();
    };
  }, [menu]);
  const refreshRuns = useCallback(() => {
    setLoading(true);
    api<{ runs: Run[] }>("/runs")
      .then((d) => {
        setRuns(d.runs);
        setSelected((previous) =>
          d.runs.some((r) => r.id === previous)
            ? previous
            : ([...d.runs].sort(
                (a, b) =>
                  b.session_count - a.session_count ||
                  b.stages.length - a.stages.length,
              )[0]?.id ?? null),
        );
        setError("");
        setOnline(true);
      })
      .catch((e) => {
        setError(errorMessage(e));
        setOnline(false);
      })
      .finally(() => setLoading(false));
  }, []);
  const refreshJobs = useCallback(() => {
    api<{ jobs: Job[] }>("/jobs")
      .then((d) => setJobs(d.jobs))
      .catch(() => {});
  }, []);
  const loadSchema = useCallback(() => {
    setSchemaError("");
    api<Schema>("/schema")
      .then(setSchema)
      .catch((e) => setSchemaError(errorMessage(e)));
  }, []);
  useEffect(() => {
    refreshRuns();
    refreshJobs();
    loadSchema();
    const id = setInterval(() => {
      refreshJobs();
      api("/health")
        .then(() => setOnline(true))
        .catch(() => setOnline(false));
    }, 10000);
    return () => clearInterval(id);
  }, [refreshRuns, refreshJobs, loadSchema]);
  const activeJobs = jobs.filter((j) =>
    ["queued", "running", "cancelling"].includes(j.status),
  );
  function navigate(next: Page) {
    setPage(next);
    setMenu(false);
    window.scrollTo({ top: 0, behavior: "smooth" });
    if (next === "results") refreshRuns();
  }
  function configure(next?: Partial<RunRequest>) {
    setSeed(next ?? null);
    setConfigKey((k) => k + 1);
    navigate("configure");
  }
  function started(job: Job) {
    setJobs((js) => [job, ...js.filter((j) => j.id !== job.id)]);
    setSelectedJob(job.id);
    navigate("monitor");
  }
  function viewRun(cacheDir: string) {
    const id = cacheDir.split("/").filter(Boolean).at(-1) ?? "";
    setSelected(id);
    navigate("results");
  }
  return (
    <div className="app-shell">
      <a className="skip-link" href="#workspace-content">
        Skip to workspace
      </a>
      <button
        className={`mobile-shade ${menu ? "visible" : ""}`}
        aria-label="Close navigation"
        tabIndex={-1}
        onClick={() => setMenu(false)}
      />
      <aside
        ref={sidebarRef}
        id="workspace-navigation"
        aria-label="Workspace navigation"
        className={`sidebar ${menu ? "open" : ""}`}
      >
        <Button
          className="mobile-nav-close"
          variant="ghost"
          size="icon"
          aria-label="Close navigation"
          onClick={() => setMenu(false)}
        >
          <X />
        </Button>
        <a
          href="#"
          className="brand"
          onClick={(e) => {
            e.preventDefault();
            navigate("results");
          }}
        >
          <span className="brand-mark">
            <Activity size={25} strokeWidth={2} />
          </span>
          <span>
            wm<span className="brand-slash">/</span>states
            <small>ANALYSIS WORKSPACE</small>
          </span>
        </a>
        <div className="workspace-label">
          <span className="workspace-icon">
            <FlaskConical size={16} />
          </span>
          <div>
            Next pipeline<small>Local workspace</small>
          </div>
          <span className="version-tag">v1</span>
        </div>
        <div className="nav-section-label">WORKSPACE</div>
        <nav>
          {navigation.map((item) => (
            <button
              key={item.id}
              className={page === item.id ? "active" : ""}
              aria-current={page === item.id ? "page" : undefined}
              onClick={() => navigate(item.id)}
            >
              <item.icon size={18} />
              <span>{item.label}</span>
              {item.id === "monitor" && activeJobs.length > 0 && (
                <span className="nav-count">{activeJobs.length}</span>
              )}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="workspace-note">
            <span className="icon-tile small">
              <Layers3 size={17} />
            </span>
            <strong>From recording to result</strong>
            <p>
              Configure, inspect, compare.
              <br />
              All in one local workspace.
            </p>
          </div>
          <GuideLink className="help-link guide-sidebar-link">
            <BookOpen size={16} />
            Pipeline guide
          </GuideLink>
          <a
            href="/api/docs"
            target="_blank"
            rel="noopener noreferrer"
            className="help-link api-sidebar-link"
          >
            <Code2 size={15} />
            API reference
            <ArrowUpRight size={14} />
            <span className="sr-only"> (opens in a new tab)</span>
          </a>
          <div className="connection-status" role="status">
            <span className={`connection-dot ${online ? "online" : ""}`} />
            <div>
              {online === null
                ? "Connecting to backend…"
                : online
                  ? "Backend connected"
                  : "Backend unavailable"}
              <small>Local Python engine</small>
            </div>
          </div>
        </div>
      </aside>
      <main className="main-shell" inert={menu}>
        <header className="topbar">
          <Button
            className="mobile-menu"
            variant="ghost"
            size="icon"
            aria-label="Toggle navigation"
            ref={menuRef}
            aria-expanded={menu}
            aria-controls="workspace-navigation"
            onClick={() => setMenu(!menu)}
          >
            {menu ? <X /> : <Menu />}
          </Button>
          <div className="breadcrumb">
            <span>Workspace</span>
            <ChevronRight size={13} />
            <span>Next pipeline</span>
            <ChevronRight size={13} />
            <strong>{navigation.find((n) => n.id === page)?.label}</strong>
          </div>
          <div className="topbar-actions">
            <HelpPanel page={page} />
            {activeJobs.length > 0 ? (
              <button
                className="running-pill"
                onClick={() => navigate("monitor")}
              >
                <span className="pulse-dot" />
                Pipeline running
              </button>
            ) : (
              <span className="local-pill">
                <span className="tiny-dot" />
                Local workspace
              </span>
            )}
            <span className="avatar">WS</span>
          </div>
        </header>
        <div className="main-content" id="workspace-content" tabIndex={-1}>
          {page === "results" && (
            <Results
              runs={runs}
              loading={loading}
              error={error}
              selectedId={selected}
              onSelect={setSelected}
              onRefresh={refreshRuns}
              onConfigure={configure}
              onCompare={(id) => {
                setSelected(id);
                navigate("compare");
              }}
            />
          )}
          {/* Keep an in-progress configuration mounted while browsing other views. */}
          <div hidden={page !== "configure"}>
            {schema ? (
              <Configure
                key={configKey}
                schema={schema}
                seed={seed}
                onStarted={started}
                onBack={() => navigate("results")}
              />
            ) : (
              page === "configure" &&
              (schemaError ? (
                <Notice>
                  {schemaError}
                  <Button variant="outline" size="sm" onClick={loadSchema}>
                    Retry
                  </Button>
                </Notice>
              ) : (
                <Loading label="Reading pipeline parameters…" />
              ))
            )}
          </div>
          {page === "compare" && (
            <Compare
              runs={runs}
              initialRun={selected}
              onNewRun={() => configure()}
            />
          )}
          {page === "monitor" && (
            <LiveMonitor
              jobs={jobs}
              selectedId={selectedJob}
              onJobsChanged={refreshJobs}
              onViewRun={viewRun}
              onNewRun={() => configure()}
            />
          )}
        </div>
        <footer className="app-footer">
          <span>
            WM / STATES <i /> Next analysis pipeline
          </span>
          <span>Processed locally. Kept in your workspace.</span>
        </footer>
      </main>
    </div>
  );
}
