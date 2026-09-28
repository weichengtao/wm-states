import { useId, useMemo, useState } from "react";
import {
  Check,
  GitCompareArrows,
  Search,
  SlidersHorizontal,
} from "lucide-react";
import type { Manifest, PipelineTemplate, Schema } from "@/lib/types";
import {
  filterHistorySettings,
  historicalStages,
  historySettings,
  historyValue,
} from "@/lib/history-comparison";
import { humanize } from "@/lib/utils";
import { freezeTemplate } from "@/lib/templates";
import { Select } from "./ui/select";
import { Button } from "./ui/button";
import "./HistoryComparison.css";

export function HistoryComparison({
  manifest,
  templates,
  schema,
}: {
  manifest: Manifest;
  templates: PipelineTemplate[];
  schema?: Schema | null;
}) {
  const id = useId();
  const original = manifest.source_template;
  const [choice, setChoice] = useState(
    original
      ? "original"
      : templates.some((item) => item.id === "default")
        ? "current:default"
        : "current:example",
  );
  const [changedOnly, setChangedOnly] = useState(true);
  const [query, setQuery] = useState("");
  const currentTemplates = useMemo(
    () =>
      schema
        ? templates.map((item) => freezeTemplate(item, schema))
        : templates,
    [templates, schema],
  );
  const template =
    choice === "original"
      ? original
      : currentTemplates.find((item) => `current:${item.id}` === choice);
  const rows = useMemo(
    () => (template ? historySettings(manifest, template, schema) : []),
    [manifest, template, schema],
  );
  const changed = rows.filter((row) => row.status === "changed").length;
  const unavailable = rows.filter((row) => row.status === "unavailable").length;
  const filtered = filterHistorySettings(rows, changedOnly, query);
  const groups = [...new Set(filtered.map((row) => row.stage))];
  const selectedOriginal = choice === "original" && Boolean(original);
  const stageNames = historicalStages(manifest).map(
    (stage) =>
      schema?.stages.find((item) => item.id === stage)?.label ??
      humanize(stage),
  );
  return (
    <section className="history-comparison" aria-labelledby={`${id}-heading`}>
      <div className="history-comparison-heading">
        <div className="history-comparison-title">
          <span className="history-comparison-icon">
            <GitCompareArrows size={18} />
          </span>
          <div>
            <h4 id={`${id}-heading`}>Settings compared with a template</h4>
            <p>
              Review what this invocation recorded. Changing the comparison
              leaves the run untouched.
            </p>
          </div>
        </div>
        <span
          className={`history-change-count ${changed ? "has-changes" : ""}`}
        >
          {changed ? <SlidersHorizontal size={14} /> : <Check size={14} />}
          {changed} {changed === 1 ? "change" : "changes"}
        </span>
      </div>
      <div className="history-baseline">
        <label htmlFor={`${id}-template`}>Compare with</label>
        <Select
          id={`${id}-template`}
          aria-label="Run history comparison template"
          value={choice}
          onValueChange={setChoice}
          placeholder="Choose a comparison template"
          options={[
            ...(original
              ? [
                  {
                    value: "original",
                    label: `${original.name} · Original snapshot`,
                    description:
                      "Captured when this invocation was launched. Kept even if the template changes or is deleted.",
                  },
                ]
              : []),
            ...currentTemplates.map((item) => ({
              value: `current:${item.id}`,
              label: `${item.name} · Current`,
              description: item.builtin
                ? "Built-in template as it is configured now."
                : "Saved template as it is configured now.",
            })),
          ]}
        />
        <p
          className={`history-baseline-note ${!original ? "history-baseline-unknown" : ""}`}
        >
          {selectedOriginal
            ? "Original template snapshot · The reference captured for this invocation."
            : !original
              ? "Original template not recorded. This is a comparison reference, not a claim about how the run was configured."
              : "Current template · This comparison can differ from the original snapshot above."}
        </p>
      </div>
      {template ? (
        <>
          <p className="history-comparison-scope">
            <strong>Recorded stages:</strong>{" "}
            {stageNames.join(" · ") || "None available"}. Only requested stages'
            parameters are compared. Run identity is excluded.
          </p>
          <div className="history-comparison-toolbar">
            <div
              className="history-view-options"
              aria-label="Settings comparison filter"
            >
              <Button
                size="sm"
                variant={changedOnly ? "secondary" : "ghost"}
                aria-pressed={changedOnly}
                onClick={() => setChangedOnly(true)}
              >
                Changed ({changed})
              </Button>
              <Button
                size="sm"
                variant={!changedOnly ? "secondary" : "ghost"}
                aria-pressed={!changedOnly}
                onClick={() => setChangedOnly(false)}
              >
                All settings ({rows.length})
              </Button>
            </div>
            <label className="search-field history-setting-search">
              <Search size={14} aria-hidden="true" />
              <input
                type="search"
                aria-label="Search recorded settings"
                placeholder="Find a setting…"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
              />
            </label>
          </div>
          {unavailable > 0 && (
            <div className="history-unavailable-note">
              {unavailable}{" "}
              {unavailable === 1 ? "setting cannot" : "settings cannot"} be
              compared because a value is missing from the record or template.{" "}
              <button
                type="button"
                onClick={() => {
                  setChangedOnly(false);
                  setQuery("");
                }}
              >
                Show all settings
              </button>
              <span>
                Missing values are never filled with today's defaults.
              </span>
            </div>
          )}
          {filtered.length ? (
            <div className="history-comparison-groups">
              {groups.map((stage) => (
                <section className="history-setting-group" key={stage}>
                  <h5>
                    {stage === "shared"
                      ? "Run options"
                      : (schema?.stages.find((item) => item.id === stage)
                          ?.label ?? humanize(stage))}
                    <span>
                      {filtered.filter((row) => row.stage === stage).length}
                    </span>
                  </h5>
                  <div className="history-setting-columns" aria-hidden="true">
                    <span>Setting</span>
                    <span>Template</span>
                    <span>Recorded run</span>
                  </div>
                  {filtered
                    .filter((row) => row.stage === stage)
                    .map((row) => (
                      <div
                        className={`history-setting-row history-setting-${row.status}`}
                        key={row.field}
                      >
                        <div className="history-setting-name">
                          <span>{row.label}</span>
                          <code>{row.field}</code>
                          {row.status === "unavailable" && (
                            <span className="history-setting-unknown">
                              Unavailable
                            </span>
                          )}
                        </div>
                        <div className="history-setting-value">
                          <span className="history-mobile-label">Template</span>
                          {historyValue(row.before, "template")}
                        </div>
                        <div className="history-setting-value history-recorded-value">
                          <span className="history-mobile-label">
                            Recorded run
                          </span>
                          {historyValue(row.after, "recorded")}
                        </div>
                      </div>
                    ))}
                </section>
              ))}
            </div>
          ) : (
            <div className="history-comparison-empty">
              <Check size={19} />
              <strong>
                {query
                  ? "No settings match your search"
                  : changedOnly
                    ? "No differences among comparable settings"
                    : "No recorded settings available"}
              </strong>
              <p>
                {query
                  ? "Try a shorter search or show all settings."
                  : unavailable
                    ? "Some values are unavailable. Open All settings to inspect them."
                    : "The recorded values match this comparison template."}
              </p>
            </div>
          )}
          <p className="history-comparison-footer">
            Recording and session-list paths are compared only when the template
            captures them. See the original manifest below for output paths,
            execution details, and all recorded values.
          </p>
        </>
      ) : (
        <div className="history-comparison-empty">
          <strong>Comparison template unavailable</strong>
          <p>
            Select another template, or refresh the template list above. The
            original manifest is still available below.
          </p>
        </div>
      )}
    </section>
  );
}
