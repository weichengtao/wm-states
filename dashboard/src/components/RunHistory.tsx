import { useEffect, useMemo, useState } from "react";
import { ChevronDown, RefreshCw, Terminal } from "lucide-react";
import type { Manifest, PipelineTemplate, Schema } from "@/lib/types";
import { api, errorMessage } from "@/lib/api";
import { builtInTemplates } from "@/lib/templates";
import { duration, formatDate, humanize } from "@/lib/utils";
import { manifestLegacyTrust } from "@/lib/legacy-trust";
import { CopyButton, Empty, Notice, Status } from "./shared";
import { Button } from "./ui/button";
import { LegacyTrustNotice } from "./LegacyTrust";
import { HistoryComparison } from "./HistoryComparison";

export default function RunHistory({
  manifests,
  schema,
}: {
  manifests: Manifest[];
  schema?: Schema | null;
}) {
  const builtins = useMemo(
    () => (schema ? builtInTemplates(schema) : []),
    [schema],
  );
  const [templates, setTemplates] = useState<PipelineTemplate[] | null>(null);
  const [error, setError] = useState("");
  const [warnings, setWarnings] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [revision, setRevision] = useState(0);
  useEffect(() => {
    let stopped = false;
    setLoading(true);
    setError("");
    api<{ templates: PipelineTemplate[]; warnings: string[] }>("/templates")
      .then((response) => {
        if (stopped) return;
        setTemplates(response.templates);
        setWarnings(response.warnings ?? []);
      })
      .catch((reason) => {
        if (!stopped) setError(errorMessage(reason));
      })
      .finally(() => {
        if (!stopped) setLoading(false);
      });
    return () => {
      stopped = true;
    };
  }, [revision]);
  return (
    <>
      <div className="history-intro">
        <h3>Every invocation, preserved.</h3>
        <p>
          Compare recorded settings with their original template or another
          reference. Exact commands and resolved settings remain available for
          complete and partial runs. Re-running a stage can replace its outputs;
          the manifest history remains.
        </p>
      </div>
      {manifests.length ? (
        <>
          <div className="history-library-actions">
            <Button
              variant="outline"
              size="sm"
              disabled={loading}
              onClick={() => setRevision((value) => value + 1)}
            >
              <RefreshCw size={13} />
              {loading ? "Reading templates…" : "Refresh templates"}
            </Button>
            <span>Original snapshots stay with each invocation.</span>
          </div>
          {error && (
            <Notice>
              Could not refresh current templates: {error}. Original snapshots
              remain available.
            </Notice>
          )}
          {warnings.map((warning, index) => (
            <Notice key={index} tone="info">
              {warning}
            </Notice>
          ))}
          {manifests.map((manifest, index) => (
            <details
              className="panel manifest-card"
              key={manifest.id ?? index}
              open={index === 0}
            >
              <summary>
                <span className="manifest-icon">
                  <Terminal size={17} />
                </span>
                <span>
                  <strong>{formatDate(manifest.started_at)}</strong>
                  <small>
                    {manifest.stages?.map((stage) => stage.stage).join(" → ") ||
                      "No recorded stages"}
                  </small>
                </span>
                <Status value={manifest.status ?? "unknown"} />
                <ChevronDown size={16} />
              </summary>
              <div className="manifest-body">
                <LegacyTrustNotice status={manifestLegacyTrust(manifest)} />
                <HistoryComparison
                  manifest={manifest}
                  templates={templates ?? builtins}
                  schema={schema}
                />
                <div className="section-heading">
                  <span className="eyebrow">EXACT INVOCATION</span>
                  {manifest.invocation?.command && (
                    <CopyButton text={manifest.invocation.command} />
                  )}
                </div>
                <pre>
                  {manifest.invocation?.command ??
                    "This older manifest did not record an invocation command."}
                </pre>
                <div className="stage-timings">
                  {manifest.stages?.map((stage) => (
                    <div key={stage.stage}>
                      <span>{humanize(stage.stage)}</span>
                      <Status value={stage.status} />
                      <span>{duration(stage.seconds)}</span>
                    </div>
                  ))}
                </div>
                <details>
                  <summary>Resolved settings &amp; manifest</summary>
                  <pre>{JSON.stringify(manifest, null, 2)}</pre>
                </details>
              </div>
            </details>
          ))}
        </>
      ) : (
        <Empty title="No manifests found">
          Stage outputs may still be available in the other views.
        </Empty>
      )}
    </>
  );
}
