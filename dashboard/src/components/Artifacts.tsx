import { useEffect, useId, useMemo, useState, type CSSProperties } from "react";
import {
  Download,
  Image,
  RotateCcw,
  Search,
  Table2,
  ZoomIn,
} from "lucide-react";
import type { Artifact, TableData } from "@/lib/types";
import { api, errorMessage, runPath } from "@/lib/api";
import { humanize } from "@/lib/utils";
import { Button } from "./ui/button";
import { Select } from "./ui/select";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
  DialogTrigger,
} from "./ui/dialog";
import { Empty, Loading, Notice } from "./shared";
export function FigurePreview({
  artifact,
  compact = false,
}: {
  artifact: Artifact;
  compact?: boolean;
}) {
  const [zoom, setZoom] = useState(100);
  const zoomId = useId();
  const preview = /\.(png|jpe?g|webp|svg)$/i.test(artifact.path);
  return (
    <Dialog>
      <div className={`figure-card ${compact ? "compact" : ""}`}>
        <DialogTrigger asChild>
          <button
            className="figure-preview"
            onClick={() => setZoom(100)}
            aria-label={`Open ${artifact.name}`}
          >
            {preview ? (
              <img src={artifact.url} alt={artifact.name} loading="lazy" />
            ) : (
              <div className="figure-placeholder">
                <Image size={30} />
                <span>
                  {artifact.path.split(".").pop()?.toUpperCase()} figure
                </span>
              </div>
            )}
            <span className="zoom-hint">
              <ZoomIn size={16} />
              Inspect
            </span>
          </button>
        </DialogTrigger>
        <div className="figure-caption">
          <div>
            <span className="eyebrow">
              {humanize(artifact.stage)}
              {artifact.session && ` · ${artifact.session}`}
            </span>
            <h4 title={artifact.path}>{artifact.name.replaceAll("_", " ")}</h4>
          </div>
          <a
            href={artifact.url}
            download
            className="icon-link"
            aria-label={`Download ${artifact.name}`}
          >
            <Download size={16} />
          </a>
        </div>
      </div>
      <DialogContent>
        <DialogTitle>{artifact.name}</DialogTitle>
        <DialogDescription>{artifact.path}</DialogDescription>
        <div className="figure-tools">
          {preview && (
            <div className="figure-zoom-controls">
              <label htmlFor={zoomId}>
                Zoom
                <input
                  id={zoomId}
                  aria-label="Figure zoom"
                  aria-valuetext={`${zoom}%`}
                  type="range"
                  min={50}
                  max={250}
                  step={10}
                  value={zoom}
                  style={
                    {
                      "--range-progress": `${(zoom - 50) / 2}%`,
                    } as CSSProperties
                  }
                  onChange={(e) => setZoom(Number(e.target.value))}
                />
              </label>
              <output htmlFor={zoomId} className="figure-zoom-value">
                {zoom}%
              </output>
              <Button
                variant="ghost"
                size="sm"
                disabled={zoom === 100}
                onClick={() => setZoom(100)}
              >
                <RotateCcw />
                Reset zoom
              </Button>
            </div>
          )}
          <Button asChild variant="outline" size="sm">
            <a href={artifact.url} download>
              <Download />
              Download original
            </a>
          </Button>
        </div>
        {preview ? (
          <div className="figure-inspector">
            <img
              src={artifact.url}
              alt={artifact.name}
              style={{ width: `${zoom}%`, maxWidth: "none" }}
            />
          </div>
        ) : (
          <Empty title="Download to inspect this format">
            Choose PNG in the run settings for an in-browser preview.
          </Empty>
        )}
      </DialogContent>
    </Dialog>
  );
}
export function FigureGallery({
  artifacts,
  session,
}: {
  artifacts: Artifact[];
  session?: string;
}) {
  const [stage, setStage] = useState("all");
  const [scope, setScope] = useState("session");
  const [query, setQuery] = useState("");
  const [limit, setLimit] = useState(12);
  const figures = artifacts.filter((a) => a.kind === "figure");
  const stages = [...new Set(figures.map((a) => a.stage))];
  const filtered = figures.filter(
    (a) =>
      (stage === "all" || a.stage === stage) &&
      (scope === "all" || !session || !a.session || a.session === session) &&
      `${a.path}`.toLowerCase().includes(query.toLowerCase()),
  );
  useEffect(() => setLimit(12), [stage, scope, query, session]);
  return (
    <>
      <div className="filter-bar">
        <div className="search-field">
          <Search size={16} />
          <input
            aria-label="Search figures"
            placeholder="Search figures…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>
        <Select
          className="select-control"
          aria-label="Figure stage"
          value={stage}
          onValueChange={setStage}
          options={[
            { value: "all", label: "All stages" },
            ...stages.map((value) => ({ value, label: humanize(value) })),
          ]}
        />
        {session && (
          <Select
            className="select-control"
            aria-label="Figure session scope"
            value={scope}
            onValueChange={setScope}
            options={[
              { value: "session", label: `Session ${session} & shared` },
              { value: "all", label: "All sessions" },
            ]}
          />
        )}
        <span className="muted">{filtered.length} figures</span>
      </div>
      {filtered.length ? (
        <div className="figure-grid">
          {filtered.slice(0, limit).map((a) => (
            <FigurePreview key={a.path} artifact={a} />
          ))}
        </div>
      ) : (
        <Empty title="No matching figures" icon={<Image />}>
          Try a different stage or session, or include PNG output in your next
          run.
        </Empty>
      )}
      {filtered.length > limit && (
        <div className="load-more">
          <Button variant="outline" onClick={() => setLimit((l) => l + 12)}>
            Show more figures ({filtered.length - limit} remaining)
          </Button>
        </div>
      )}
    </>
  );
}
export function TableBrowser({
  runId,
  artifacts,
}: {
  runId: string;
  artifacts: Artifact[];
}) {
  const tables = useMemo(
    () =>
      artifacts.filter((a) => a.kind === "table" && a.path.endsWith(".csv")),
    [artifacts],
  );
  const initialTable =
    tables.find((a) => a.path === "evaluate/tables/eval_confidence.csv")
      ?.path ??
    tables[0]?.path ??
    "";
  const [selected, setSelected] = useState(initialTable);
  const [page, setPage] = useState(0);
  const [data, setData] = useState<TableData | null>(null);
  const [error, setError] = useState("");
  const artifact = tables.find((a) => a.path === selected);
  useEffect(() => {
    setSelected(initialTable);
    setPage(0);
  }, [tables, initialTable]);
  useEffect(() => {
    if (!selected) return;
    let cancelled = false;
    setData(null);
    setError("");
    api<TableData>(
      `${runPath(runId)}/tables/${selected.split("/").map(encodeURIComponent).join("/")}?offset=${page * 50}&limit=50`,
    )
      .then((d) => {
        if (!cancelled) setData(d);
      })
      .catch((e) => {
        if (!cancelled) setError(errorMessage(e));
      });
    return () => {
      cancelled = true;
    };
  }, [runId, selected, page]);
  if (!tables.length)
    return (
      <Empty title="No result tables yet" icon={<Table2 />}>
        Tables will appear when their analysis stage finishes.
      </Empty>
    );
  return (
    <>
      <div className="filter-bar">
        <Select
          className="select-control table-select"
          aria-label="Result table"
          value={selected}
          onValueChange={(value) => {
            setSelected(value);
            setPage(0);
          }}
          options={tables.map((a) => ({
            value: a.path,
            label: a.name,
            description: a.path,
          }))}
        />
        {artifact && (
          <Button variant="outline" size="sm" asChild>
            <a download href={artifact.url}>
              <Download />
              CSV
            </a>
          </Button>
        )}
      </div>
      {error ? (
        <Notice>{error}</Notice>
      ) : !data ? (
        <Loading label="Reading table…" />
      ) : (
        <>
          <div className="table-scroll">
            <table className="data-table">
              <thead>
                <tr>
                  {data.columns.map((c) => (
                    <th key={c}>{c.replaceAll("_", " ")}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.rows.map((row, index) => (
                  <tr key={index}>
                    {data.columns.map((c) => (
                      <td key={c}>
                        {row[c] === null
                          ? "—"
                          : typeof row[c] === "object"
                            ? JSON.stringify(row[c])
                            : String(row[c] ?? "—")}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="table-footer">
            <span>
              {data.total
                ? `${page * 50 + 1}–${Math.min((page + 1) * 50, data.total)}`
                : "0"}{" "}
              of {data.total} rows
            </span>
            <div className="heading-actions">
              <Button
                size="sm"
                variant="outline"
                disabled={!page}
                onClick={() => setPage((p) => p - 1)}
              >
                Previous
              </Button>
              <Button
                size="sm"
                variant="outline"
                disabled={(page + 1) * 50 >= data.total}
                onClick={() => setPage((p) => p + 1)}
              >
                Next
              </Button>
            </div>
          </div>
        </>
      )}
    </>
  );
}

export function SupportingFiles({ artifacts }: { artifacts: Artifact[] }) {
  const files = artifacts.filter((a) => ["json", "log"].includes(a.kind));
  if (!files.length) return null;
  return (
    <details className="panel supporting-files">
      <summary>
        Supporting files <span className="count-badge">{files.length}</span>
      </summary>
      <div className="supporting-file-list">
        {files.map((a) => (
          <a key={a.path} href={a.url} download>
            <span>{a.path}</span>
            <Download size={14} />
          </a>
        ))}
      </div>
    </details>
  );
}
