import { useMemo, useRef, useState } from "react";
import {
  ArrowLeft,
  ArrowRight,
  Check,
  ChevronRight,
  Code2,
  FlaskConical,
  FolderOpen,
  LoaderCircle,
  Play,
  RotateCcw,
  Search,
  SlidersHorizontal,
  Undo2,
  X,
} from "lucide-react";
import type {
  Field,
  Json,
  Job,
  RunRequest,
  Schema,
  Settings,
} from "@/lib/types";
import { api, errorMessage } from "@/lib/api";
import { humanize } from "@/lib/utils";
import {
  choiceValue,
  fieldValue,
  initialRun,
  nonNullValue,
  parseSettings,
  parameterMatchesQuery,
  presetName,
  sameFieldValue,
} from "@/lib/configuration";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { CopyButton, Notice } from "./shared";
import GuideLink from "./GuideLink";
import { fieldHelpPath, stageMethodsPath } from "@/lib/help";
function FieldEditor({
  stageId,
  field,
  value,
  exampleValue,
  onChange,
}: {
  stageId: string;
  field: Field;
  value: Json;
  exampleValue: Json;
  onChange: (value: Json) => void;
}) {
  const id = `field-${stageId}-${field.name}`;
  const selectedChoice = choiceValue(field, value);
  const helpPath = fieldHelpPath(stageId, field.name);
  const changed = !sameFieldValue(field, value, exampleValue);
  return (
    <div className={`parameter ${changed ? "parameter-changed" : ""}`}>
      <div className="parameter-label">
        <label htmlFor={id}>{humanize(field.name)}</label>
        <div className="parameter-actions">
          {changed && (
            <button
              type="button"
              className="text-button parameter-reset"
              title={`Example: ${JSON.stringify(exampleValue)}`}
              aria-label={`Reset ${humanize(field.name)} to example value`}
              onClick={() => onChange(structuredClone(exampleValue))}
            >
              <RotateCcw size={12} /> Use example
            </button>
          )}
          {field.nullable && (
            <button
              type="button"
              className="text-button"
              onClick={() =>
                onChange(value === null ? nonNullValue(field) : null)
              }
            >
              {value === null ? "Set value" : "Use none"}
            </button>
          )}
        </div>
      </div>
      {field.type === "boolean" ? (
        <label className="switch-row" htmlFor={id}>
          <input
            id={id}
            type="checkbox"
            role="switch"
            aria-describedby={`${id}-description`}
            checked={value === true}
            onChange={(e) => onChange(e.target.checked)}
          />
          <span>{value ? "Enabled" : "Disabled"}</span>
        </label>
      ) : field.choices?.length ? (
        <select
          id={id}
          aria-describedby={`${id}-description`}
          className="select-control"
          value={selectedChoice}
          onChange={(e) =>
            onChange(
              e.target.value === "" && field.nullable ? null : e.target.value,
            )
          }
        >
          {field.nullable && <option value="">None</option>}
          {selectedChoice && !field.choices.includes(selectedChoice) && (
            <option value={selectedChoice}>Invalid: {selectedChoice}</option>
          )}
          {field.choices.map((v) => (
            <option value={v} key={v}>
              {v}
            </option>
          ))}
        </select>
      ) : ["integer", "number"].includes(field.type) ? (
        <Input
          id={id}
          aria-describedby={`${id}-description`}
          type="number"
          step={field.type === "integer" ? 1 : "any"}
          value={value === null ? "" : String(value)}
          placeholder={field.nullable ? "None (optional)" : ""}
          onChange={(e) =>
            onChange(
              e.target.value === ""
                ? field.nullable
                  ? null
                  : ""
                : Number(e.target.value),
            )
          }
        />
      ) : (
        <Input
          id={id}
          aria-describedby={`${id}-description`}
          value={
            value === null
              ? ""
              : typeof value === "object"
                ? JSON.stringify(value)
                : String(value)
          }
          placeholder={
            field.nullable
              ? "None (optional)"
              : field.type === "array"
                ? "[values]"
                : ""
          }
          onChange={(e) => {
            if (["array", "object"].includes(field.type)) {
              try {
                onChange(JSON.parse(e.target.value));
              } catch {
                onChange(e.target.value);
              }
            } else onChange(e.target.value);
          }}
        />
      )}
      <p id={`${id}-description`}>
        {field.description || field.name.replaceAll("_", " ")}
      </p>
      {changed && (
        <span className="parameter-difference">Changed from example</span>
      )}
      {helpPath && (
        <GuideLink
          path={helpPath}
          className="parameter-help-link"
          label={`Learn more about ${humanize(field.name)}`}
        >
          Learn more
        </GuideLink>
      )}
    </div>
  );
}
export default function Configure({
  schema,
  onStarted,
  onBack,
  seed,
}: {
  schema: Schema;
  onStarted: (job: Job) => void;
  onBack: () => void;
  seed?: Partial<RunRequest> | null;
}) {
  const initial = useMemo(() => initialRun(schema, seed), [schema, seed]);
  const [form, setForm] = useState(initial);
  const [active, setActive] = useState(schema.stages[0].id);
  const [query, setQuery] = useState("");
  const [changedOnly, setChangedOnly] = useState(false);
  const [jsonMode, setJsonMode] = useState(false);
  const [jsonText, setJsonText] = useState(
    JSON.stringify(initial.settings, null, 2),
  );
  const [jsonDirty, setJsonDirty] = useState(false);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState("");
  const revision = useRef(0);
  const [undo, setUndo] = useState<{
    label: string;
    form: RunRequest;
    jsonText: string;
    jsonDirty: boolean;
    jsonMode: boolean;
  } | null>(null);
  const [validation, setValidation] = useState<{
    command: string;
    resolved: Settings;
  } | null>(null);
  const update = (patch: Partial<RunRequest>) => {
    revision.current += 1;
    setForm((v) => ({ ...v, ...patch }));
    setUndo(null);
    setValidation(null);
    setError("");
  };
  const stage = schema.stages.find((s) => s.id === active)!;
  const preset = presetName(form.settings, schema);
  const exampleForm = { ...form, settings: schema.presets.example };
  const changedFields = (stageId: string, fields: Field[]) =>
    fields.filter(
      (field) =>
        !sameFieldValue(
          field,
          fieldValue(form, stageId, field),
          fieldValue(exampleForm, stageId, field),
        ),
    );
  const changes = changedFields(stage.id, stage.fields);
  const filtered = stage.fields.filter(
    (field) =>
      parameterMatchesQuery(field, query) &&
      (!changedOnly || changes.includes(field)),
  );
  const replaceDraft = (next: RunRequest, label: string) => {
    const previous = structuredClone({
      label,
      form,
      jsonText,
      jsonDirty,
      jsonMode,
    });
    update(next);
    setJsonText(JSON.stringify(next.settings, null, 2));
    setJsonDirty(false);
    setUndo(previous);
  };
  const presetChange = (value: "example" | "smoke") => {
    const settings = structuredClone(schema.presets[value]);
    replaceDraft(
      { ...form, settings },
      `${value === "example" ? "Example pipeline" : "Smoke test"} settings loaded. Your run details and stage selection are unchanged.`,
    );
  };
  const applyJson = () => {
    try {
      update({ settings: parseSettings(jsonText) });
      setJsonDirty(false);
      return true;
    } catch (e) {
      setError(errorMessage(e));
      return false;
    }
  };
  const toggleJson = () => {
    if (jsonMode) {
      if (jsonDirty && !applyJson()) return;
      setJsonMode(false);
    } else {
      setJsonText(JSON.stringify(form.settings, null, 2));
      setJsonDirty(false);
      setJsonMode(true);
    }
  };
  async function submit(launch: boolean) {
    const requestedRevision = revision.current;
    setBusy(launch ? "launch" : "validate");
    setError("");
    try {
      if (jsonDirty)
        throw new Error(
          "Apply your JSON changes before validating or starting.",
        );
      if (!form.stages.length) throw new Error("Choose at least one stage.");
      if (launch) {
        onStarted(
          await api<Job>("/jobs", {
            method: "POST",
            body: JSON.stringify(form),
          }),
        );
      } else {
        const result = await api<{ command: string; resolved: Settings }>(
          "/validate",
          {
            method: "POST",
            body: JSON.stringify(form),
          },
        );
        if (revision.current === requestedRevision) setValidation(result);
      }
    } catch (e) {
      if (launch || revision.current === requestedRevision)
        setError(errorMessage(e));
    } finally {
      setBusy("");
    }
  }
  return (
    <>
      <div className="page-heading">
        <div>
          <button className="back-link" onClick={onBack}>
            <ArrowLeft size={14} /> Run library
          </button>
          <h1>Set up your analysis</h1>
          <p>
            Choose your data, select stages, and fine-tune the example pipeline.
          </p>
        </div>
        <div className="heading-actions">
          <Button
            variant="outline"
            disabled={!!busy}
            onClick={() =>
              replaceDraft(
                structuredClone(initial),
                "Setup restored to its initial values.",
              )
            }
          >
            <RotateCcw />
            Reset setup
          </Button>
          <Button
            onClick={() => submit(true)}
            disabled={!!busy || jsonDirty || !form.stages.length}
          >
            {busy === "launch" ? (
              <LoaderCircle className="animate-spin" />
            ) : (
              <Play />
            )}
            {busy === "launch" ? "Starting…" : "Start pipeline"}
          </Button>
        </div>
      </div>
      <div className="setup-top">
        <div className="preset-tile">
          <div className="icon-tile">
            <FlaskConical size={20} />
          </div>
          <div>
            <h3>Analysis preset</h3>
            <p>
              {preset === "example"
                ? "Full example · 100 null shuffles · 50 model holdouts"
                : preset === "smoke"
                  ? "Integration check · 3 null shuffles · 1 model holdout"
                  : "Custom settings · review your stage arguments"}
            </p>
          </div>
          <select
            className="select-control"
            aria-label="Analysis preset"
            disabled={jsonDirty || !!busy}
            value={preset}
            onChange={(e) =>
              presetChange(e.target.value as "example" | "smoke")
            }
          >
            <option value="custom" disabled>
              Custom settings
            </option>
            <option value="example">Example pipeline</option>
            <option value="smoke">Smoke test</option>
          </select>
        </div>
      </div>
      {undo && (
        <div className="configuration-feedback" role="status">
          <span>
            <Check size={16} /> {undo.label}
          </span>
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              update(undo.form);
              setJsonText(undo.jsonText);
              setJsonDirty(undo.jsonDirty);
              setJsonMode(undo.jsonMode);
            }}
          >
            <Undo2 /> Undo
          </Button>
        </div>
      )}
      <section className="panel run-setup">
        <div className="section-heading">
          <div>
            <h2>Run details</h2>
            <p>Keep each analysis in its own cache directory.</p>
          </div>
          <span className="eyebrow">01 / WORKSPACE</span>
        </div>
        <div className="form-grid">
          <div className="field">
            <label htmlFor="run-name">Run name</label>
            <Input
              id="run-name"
              value={form.name}
              onChange={(e) => update({ name: e.target.value })}
            />
          </div>
          <div className="field">
            <label htmlFor="cache-dir">Cache directory</label>
            <Input
              id="cache-dir"
              aria-describedby="cache-dir-help"
              value={form.cache_dir}
              onChange={(e) => update({ cache_dir: e.target.value })}
            />
            <p id="cache-dir-help" className="field-hint">
              A unique folder keeps this run easy to compare later.
            </p>
          </div>
          <div className="field">
            <label htmlFor="data-dir">
              Recording directory{" "}
              <button
                className="text-button"
                onClick={() => update({ data_dir: "data/example" })}
              >
                Use example data
              </button>
            </label>
            <Input
              id="data-dir"
              value={form.data_dir}
              onChange={(e) => update({ data_dir: e.target.value })}
            />
          </div>
          <div className="field">
            <label htmlFor="session-list">
              Session allowlist <span>optional</span>
            </label>
            <Input
              id="session-list"
              aria-describedby="session-list-help"
              placeholder="All available sessions"
              value={form.session_list_file ?? ""}
              onChange={(e) =>
                update({ session_list_file: e.target.value || null })
              }
            />
            <p id="session-list-help" className="field-hint">
              Only listed sessions found in the recording directory are used.
            </p>
          </div>
          <div className="field">
            <label htmlFor="workers">Parallel workers</label>
            <Input
              id="workers"
              aria-describedby="workers-help"
              type="number"
              step={1}
              value={form.n_jobs}
              onChange={(e) => update({ n_jobs: Number(e.target.value) })}
            />
            <p id="workers-help" className="field-hint">
              Use 1 for a single worker or −1 for all available CPUs.
            </p>
          </div>
          <div className="field">
            <label htmlFor="max-sessions">
              Maximum sessions <span>optional</span>
            </label>
            <Input
              id="max-sessions"
              type="number"
              min={1}
              placeholder="All eligible sessions"
              value={form.max_sessions_to_run ?? ""}
              onChange={(e) =>
                update({
                  max_sessions_to_run: e.target.value
                    ? Number(e.target.value)
                    : null,
                })
              }
            />
          </div>
        </div>
        <div className="setup-options">
          <div className="inline-checks">
            <span>Figure formats</span>
            {["png", "tif", "eps", "pdf"].map((format) => (
              <label key={format}>
                <input
                  type="checkbox"
                  checked={form.figure_formats.includes(format)}
                  onChange={(e) =>
                    update({
                      figure_formats: e.target.checked
                        ? [...form.figure_formats, format]
                        : form.figure_formats.filter((f) => f !== format),
                    })
                  }
                />
                {format.toUpperCase()}
              </label>
            ))}
          </div>
          <label className="inline-check">
            <input
              type="checkbox"
              checked={form.allow_existing}
              onChange={(e) => update({ allow_existing: e.target.checked })}
            />{" "}
            Reuse an existing cache directory
          </label>
        </div>
        {form.allow_existing && (
          <Notice tone="info">
            Selected stages can replace existing outputs. Earlier invocation
            records will be preserved.
          </Notice>
        )}
      </section>
      <div className="section-heading standalone">
        <div>
          <h2>Build your pipeline</h2>
          <p>
            {form.stages.length} of {schema.stages.length} stages selected.
            Execution follows the pipeline order.
          </p>
        </div>
        <div className="heading-actions">
          <Button
            variant="outline"
            size="sm"
            aria-pressed={form.stages.length === schema.stages.length}
            onClick={() => update({ stages: schema.stages.map((s) => s.id) })}
          >
            Select all
          </Button>
          <Button
            variant="outline"
            size="sm"
            aria-pressed={
              form.stages.length === 5 &&
              schema.stages.slice(0, 5).every((s) => form.stages.includes(s.id))
            }
            onClick={() =>
              update({ stages: schema.stages.slice(0, 5).map((s) => s.id) })
            }
          >
            Core stages
          </Button>
        </div>
      </div>
      <div className="configuration-layout">
        <aside className="stage-picker">
          {schema.stages.map((item, index) => (
            <div
              className={`stage-option ${active === item.id ? "active" : ""}`}
              key={item.id}
            >
              <input
                aria-label={`Include ${item.label}`}
                type="checkbox"
                checked={form.stages.includes(item.id)}
                onChange={(e) =>
                  update({
                    stages: schema.stages
                      .map((s) => s.id)
                      .filter((id) =>
                        id === item.id
                          ? e.target.checked
                          : form.stages.includes(id),
                      ),
                  })
                }
              />
              <button
                aria-current={active === item.id ? "step" : undefined}
                aria-controls="stage-parameters"
                onClick={() => setActive(item.id)}
              >
                <span className="stage-index">
                  {String(index + 1).padStart(2, "0")}
                </span>
                <span>{item.label || humanize(item.id)}</span>
                {changedFields(item.id, item.fields).length > 0 && (
                  <span
                    className="stage-change-count"
                    title="Parameters changed from the example"
                    aria-label={`${changedFields(item.id, item.fields).length} parameters changed from example`}
                  >
                    {changedFields(item.id, item.fields).length}
                  </span>
                )}
                <ChevronRight size={14} />
              </button>
            </div>
          ))}
        </aside>
        <section id="stage-parameters" className="panel parameters-panel">
          <div className="section-heading">
            <div>
              <h2>
                {jsonMode
                  ? "Pipeline settings JSON"
                  : stage.label || humanize(stage.id)}
              </h2>
              <p>
                {jsonMode
                  ? "Edit stage overrides directly; unknown parameters are rejected."
                  : stage.description}
              </p>
              {!jsonMode && (
                <span className="stage-selection-state">
                  {form.stages.includes(stage.id)
                    ? "Included in this run"
                    : "Not included · select this stage to run it"}
                </span>
              )}
            </div>
            <div className="parameter-heading-actions">
              <GuideLink
                path={
                  jsonMode ? "next/configuration/" : stageMethodsPath(stage.id)
                }
              >
                {jsonMode ? "Configuration guide" : "Stage methods"}
              </GuideLink>
              <Button variant="outline" size="sm" onClick={toggleJson}>
                {jsonMode ? <SlidersHorizontal /> : <Code2 />}
                {jsonMode ? "Form" : "JSON"}
              </Button>
            </div>
          </div>
          {jsonMode ? (
            <>
              <textarea
                className="json-editor"
                aria-label="Pipeline settings JSON"
                spellCheck={false}
                value={jsonText}
                onChange={(e) => {
                  revision.current += 1;
                  setJsonText(e.target.value);
                  setJsonDirty(true);
                  setValidation(null);
                  setUndo(null);
                }}
              />
              <div className="json-actions">
                <span>
                  {jsonDirty
                    ? "Unapplied changes · switching to Form applies valid JSON"
                    : "Settings are up to date"}
                </span>
                <Button size="sm" disabled={!jsonDirty} onClick={applyJson}>
                  <Check />
                  Apply JSON
                </Button>
              </div>
            </>
          ) : (
            <>
              <div className="parameter-toolbar">
                <div className="search-field parameter-search">
                  <Search size={16} />
                  <input
                    aria-label="Search parameters"
                    aria-controls="parameter-fields"
                    placeholder="Find a parameter…"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                  />
                  {query && (
                    <button
                      type="button"
                      className="search-clear"
                      aria-label="Clear parameter search"
                      onClick={() => setQuery("")}
                    >
                      <X size={14} />
                    </button>
                  )}
                </div>
                <Button
                  className="parameter-filter"
                  variant={changedOnly ? "secondary" : "outline"}
                  size="sm"
                  aria-pressed={changedOnly}
                  onClick={() => setChangedOnly((value) => !value)}
                >
                  <SlidersHorizontal /> Changed ({changes.length})
                </Button>
              </div>
              <p className="parameter-results-count" aria-live="polite">
                {filtered.length} of {stage.fields.length} parameters
                {changedOnly ? " · differences from the example" : ""}
              </p>
              <div id="parameter-fields" className="parameters-grid">
                {filtered.map((field) => (
                  <FieldEditor
                    key={`${stage.id}-${field.name}`}
                    stageId={stage.id}
                    field={field}
                    value={fieldValue(form, stage.id, field)}
                    exampleValue={fieldValue(exampleForm, stage.id, field)}
                    onChange={(value) =>
                      update({
                        settings: {
                          ...form.settings,
                          [stage.id]: {
                            ...form.settings[stage.id],
                            [field.name]: value,
                          },
                        },
                      })
                    }
                  />
                ))}
              </div>
              {!filtered.length && (
                <div className="parameter-empty">
                  <Search size={24} />
                  <h3>
                    {query
                      ? "No matching parameters"
                      : "No changes from the example"}
                  </h3>
                  <p>
                    {query
                      ? `Try a shorter search in ${stage.label || humanize(stage.id)}${changedOnly ? ", or show all parameters" : ""}.`
                      : "The parameters in this stage match the example pipeline."}
                  </p>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      setQuery("");
                      setChangedOnly(false);
                    }}
                  >
                    Show all parameters
                  </Button>
                </div>
              )}
            </>
          )}
        </section>
      </div>
      <section className="panel launch-panel">
        <div>
          <div className="eyebrow">READY WHEN YOU ARE</div>
          <h2>Check the configuration, then let it run.</h2>
          <p>
            {jsonDirty
              ? "Apply your JSON edits to validate or start this run."
              : !form.stages.length
                ? "Select at least one stage to continue."
                : `${form.stages.length} stages · ${form.figure_formats.map((format) => format.toUpperCase()).join(" + ") || "No figure format selected"} · ${form.max_sessions_to_run ? `up to ${form.max_sessions_to_run} sessions` : "all eligible sessions"}`}
          </p>
        </div>
        <div className="heading-actions">
          <Button
            variant="outline"
            onClick={() => submit(false)}
            disabled={!!busy || jsonDirty || !form.stages.length}
          >
            {busy === "validate" ? (
              <LoaderCircle className="animate-spin" />
            ) : (
              <FolderOpen />
            )}
            Validate & preview
          </Button>
          <Button
            onClick={() => submit(true)}
            disabled={!!busy || jsonDirty || !form.stages.length}
          >
            {busy === "launch" ? "Starting…" : "Start pipeline"}
            {busy === "launch" ? (
              <LoaderCircle className="animate-spin" />
            ) : (
              <ArrowRight />
            )}
          </Button>
        </div>
      </section>
      {error && <Notice>{error}</Notice>}
      {validation && (
        <section className="panel validation-panel">
          <div className="section-heading">
            <h3>
              <CheckCircle /> Configuration validated
            </h3>
            <CopyButton text={validation.command} />
          </div>
          <p className="fine-print">
            Command preview. Starting the run creates its own settings file and
            saves the exact command in run history.
          </p>
          <pre>{validation.command}</pre>
          <details>
            <summary>
              View resolved settings for{" "}
              {Object.keys(validation.resolved).length} stages
            </summary>
            <pre>{JSON.stringify(validation.resolved, null, 2)}</pre>
          </details>
        </section>
      )}
    </>
  );
}
function CheckCircle() {
  return <Check size={17} color="#259973" />;
}
