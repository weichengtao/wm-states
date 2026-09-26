import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowLeft,
  ArrowRight,
  Check,
  Bookmark,
  ChevronRight,
  Code2,
  FolderOpen,
  LoaderCircle,
  Play,
  RotateCcw,
  RefreshCw,
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
  PipelineTemplate,
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
import { Select } from "./ui/select";
import SaveTemplateDialog from "./SaveTemplateDialog";
import {
  builtInTemplates,
  formatTemplateValue,
  templateChanges,
  templateRun,
} from "@/lib/templates";
import { CopyButton, Notice } from "./shared";
import GuideLink from "./GuideLink";
import { fieldHelpPath, stageMethodsPath } from "@/lib/help";
function FieldEditor({
  stageId,
  field,
  value,
  templateValue,
  templateName,
  onChange,
}: {
  stageId: string;
  field: Field;
  value: Json;
  templateValue: Json;
  templateName: string;
  onChange: (value: Json) => void;
}) {
  const id = `field-${stageId}-${field.name}`;
  const selectedChoice = choiceValue(field, value);
  const helpPath = fieldHelpPath(stageId, field.name);
  const changed = !sameFieldValue(field, value, templateValue);
  return (
    <div className={`parameter ${changed ? "parameter-changed" : ""}`}>
      <div className="parameter-label">
        <label htmlFor={id}>{humanize(field.name)}</label>
        <div className="parameter-actions">
          {changed && (
            <button
              type="button"
              className="text-button parameter-reset"
              title={`${templateName}: ${formatTemplateValue(templateValue)}`}
              aria-label={`Reset ${humanize(field.name)} to template value`}
              onClick={() => onChange(structuredClone(templateValue))}
            >
              <RotateCcw size={12} /> Use template
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
        <Select
          id={id}
          aria-describedby={`${id}-description`}
          className="select-control"
          value={selectedChoice}
          onValueChange={(selected) =>
            onChange(selected === "" && field.nullable ? null : selected)
          }
          options={[
            ...(field.nullable ? [{ value: "", label: "None" }] : []),
            ...(selectedChoice && !field.choices.includes(selectedChoice)
              ? [{ value: selectedChoice, label: `Invalid: ${selectedChoice}` }]
              : []),
            ...field.choices.map((choice) => ({
              value: choice,
              label: choice,
            })),
          ]}
        />
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
        <span className="parameter-difference" title={templateName}>
          <span>Template</span>{" "}
          <code>{formatTemplateValue(templateValue)}</code>
          <ArrowRight size={12} aria-hidden="true" />{" "}
          <code>{formatTemplateValue(value)}</code>
        </span>
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
  running = false,
}: {
  schema: Schema;
  onStarted: (job: Job) => void;
  onBack: () => void;
  seed?: Partial<RunRequest> | null;
  running?: boolean;
}) {
  const initial = useMemo(() => initialRun(schema, seed), [schema, seed]);
  const builtins = useMemo(() => builtInTemplates(schema), [schema]);
  const initialTemplate = useMemo(
    () =>
      builtins.find(
        (item) => item.id === presetName(initial.settings, schema),
      ) ?? builtins[0],
    [builtins, initial, schema],
  );
  const [templates, setTemplates] = useState(builtins);
  const [selectedTemplate, setSelectedTemplate] = useState(initialTemplate);
  const [templateError, setTemplateError] = useState("");
  const [templateWarnings, setTemplateWarnings] = useState<string[]>([]);
  const [templatesLoading, setTemplatesLoading] = useState(false);
  const templateRevision = useRef(0);
  const [templateFeedback, setTemplateFeedback] = useState("");
  const [reviewChanges, setReviewChanges] = useState(false);
  const [form, setForm] = useState(initial);
  const [active, setActive] = useState(schema.stages[0].id);
  const [query, setQuery] = useState("");
  const [changedOnly, setChangedOnly] = useState(false);
  const [jsonMode, setJsonMode] = useState(false);
  const [jsonText, setJsonText] = useState(
    JSON.stringify(initial.settings, null, 2),
  );
  const [jsonDirty, setJsonDirty] = useState(false);
  const draftRef = useRef({ form, jsonDirty });
  draftRef.current = { form, jsonDirty };
  const [error, setError] = useState("");
  const [busy, setBusy] = useState("");
  const revision = useRef(0);
  const [undo, setUndo] = useState<{
    label: string;
    form: RunRequest;
    jsonText: string;
    jsonDirty: boolean;
    jsonMode: boolean;
    template: PipelineTemplate;
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
    setTemplateFeedback("");
  };
  const stage = schema.stages.find((s) => s.id === active)!;
  const templateForm = templateRun(form, selectedTemplate);
  const differences = templateChanges(form, selectedTemplate, schema);
  const sharedDifferences = differences.filter(
    (item) => item.stage === "shared",
  );
  const fieldClass = (name: string) =>
    `field${sharedDifferences.some((item) => item.field === name) ? " field-changed" : ""}`;
  const sharedHint = (name: string) => {
    const difference = sharedDifferences.find((item) => item.field === name);
    return difference ? (
      <span className="shared-template-value">
        Template: {formatTemplateValue(difference.before)}
      </span>
    ) : null;
  };
  const changedFields = (stageId: string, fields: Field[]) =>
    fields.filter((field) =>
      differences.some(
        (difference) =>
          difference.stage === stageId && difference.field === field.name,
      ),
    );
  const changes = changedFields(stage.id, stage.fields);
  const filtered = stage.fields.filter(
    (field) =>
      parameterMatchesQuery(field, query) &&
      (!changedOnly || changes.includes(field)),
  );
  const replaceDraft = (
    next: RunRequest,
    label: string,
    nextTemplate = selectedTemplate,
  ) => {
    const previous = structuredClone({
      label,
      form,
      jsonText,
      jsonDirty,
      jsonMode,
      template: selectedTemplate,
    });
    update(next);
    setJsonText(JSON.stringify(next.settings, null, 2));
    setJsonDirty(false);
    setUndo(previous);
    setSelectedTemplate(nextTemplate);
  };
  const templateChange = (value: string) => {
    const next = templates.find((item) => item.id === value);
    if (!next) return;
    replaceDraft(
      templateRun(form, next),
      `${next.name} applied. Run name and cache directory are unchanged.`,
      next,
    );
  };
  const refreshTemplates = useCallback(async () => {
    const requestedRevision = ++templateRevision.current;
    setTemplatesLoading(true);
    setTemplateError("");
    try {
      const result = await api<{
        templates: PipelineTemplate[];
        warnings: string[];
      }>("/templates");
      if (templateRevision.current === requestedRevision) {
        setTemplates(result.templates);
        setSelectedTemplate(
          (current) =>
            result.templates.find((item) => item.id === current.id) ?? current,
        );
        setTemplateWarnings(result.warnings);
      }
    } catch (reason) {
      if (templateRevision.current === requestedRevision)
        setTemplateError(
          `Saved templates could not be loaded. ${errorMessage(reason)} Use Refresh templates to try again.`,
        );
    } finally {
      if (templateRevision.current === requestedRevision)
        setTemplatesLoading(false);
    }
  }, []);
  useEffect(() => {
    void refreshTemplates();
  }, [refreshTemplates]);
  const savedTemplate = (saved: PipelineTemplate, snapshot: RunRequest) => {
    templateRevision.current += 1;
    setTemplatesLoading(false);
    setTemplates((items) => [
      ...items.filter((item) => item.id !== saved.id),
      saved,
    ]);
    const unchanged =
      !draftRef.current.jsonDirty &&
      JSON.stringify(draftRef.current.form) === JSON.stringify(snapshot);
    if (unchanged) {
      setSelectedTemplate(saved);
      setUndo(null);
    }
    setTemplateFeedback(
      unchanged
        ? `Saved “${saved.name}”. It is now your comparison template.`
        : `Saved “${saved.name}” from the captured setup. Your later edits are still here.`,
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
      if (launch && running)
        throw new Error(
          "Another pipeline is running. You can still save this setup as a template.",
        );
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
            Choose a template, make it yours, and keep useful setups for next
            time.
          </p>
        </div>
        <div className="heading-actions configuration-actions">
          <Button
            variant="outline"
            disabled={!!busy}
            onClick={() =>
              replaceDraft(
                structuredClone(initial),
                "Setup restored to its initial values.",
                initialTemplate,
              )
            }
          >
            <RotateCcw />
            Reset setup
          </Button>
          <SaveTemplateDialog
            schema={schema}
            form={form}
            jsonDirty={jsonDirty}
            running={running}
            onSaved={savedTemplate}
            onError={setTemplateError}
          />
          <Button
            onClick={() => submit(true)}
            disabled={!!busy || jsonDirty || !form.stages.length || running}
            title={
              running
                ? "Another pipeline is running. You can still edit and save templates."
                : undefined
            }
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
      {running && (
        <div className="configuration-running" role="status">
          <LoaderCircle size={17} className="animate-spin" />
          <p>
            <strong>An analysis is running.</strong> You can keep editing and
            save templates here. These changes apply to future runs.
          </p>
        </div>
      )}
      <section
        className={`panel template-panel ${differences.length ? "template-modified" : ""}`}
        aria-labelledby="template-heading"
      >
        <div className="template-picker-row">
          <span className="template-panel-icon">
            <Bookmark size={22} />
          </span>
          <div className="template-picker-copy">
            <h2 id="template-heading">Analysis template</h2>
            <p>
              {selectedTemplate.description || "Your saved analysis setup."}
            </p>
          </div>
          <div className="template-picker-controls">
            <Select
              aria-label="Analysis template"
              disabled={jsonDirty || !!busy}
              value={selectedTemplate.id}
              onValueChange={templateChange}
              options={(templates.some(
                (item) => item.id === selectedTemplate.id,
              )
                ? templates
                : [...templates, selectedTemplate]
              ).map((item) => ({
                value: item.id,
                label: item.name,
                description: `${item.builtin ? "Built-in" : "Saved template"}${item.description ? ` · ${item.description}` : ""}`,
              }))}
            />
            <Button
              variant="ghost"
              size="icon"
              aria-label="Refresh templates"
              disabled={templatesLoading}
              onClick={() => void refreshTemplates()}
            >
              <RefreshCw className={templatesLoading ? "animate-spin" : ""} />
            </Button>
          </div>
        </div>
        <div className="template-comparison-bar">
          <div className="template-comparison-status" aria-live="polite">
            {jsonDirty ? (
              <>
                <Code2 size={16} />
                <span>
                  Apply JSON to update comparisons and save a template.
                </span>
              </>
            ) : differences.length ? (
              <>
                <span className="template-change-dot" />
                <span>
                  <strong>
                    {differences.length}{" "}
                    {differences.length === 1 ? "change" : "changes"}
                  </strong>{" "}
                  from {selectedTemplate.name}
                </span>
              </>
            ) : (
              <>
                <Check size={16} />
                <span>
                  Matches <strong>{selectedTemplate.name}</strong>
                </span>
              </>
            )}
          </div>
          <div className="heading-actions">
            <Button
              variant="ghost"
              size="sm"
              disabled={jsonDirty || !differences.length}
              aria-expanded={reviewChanges}
              aria-controls="template-change-review"
              onClick={() => setReviewChanges((value) => !value)}
            >
              {reviewChanges ? "Hide changes" : "Review changes"}
            </Button>
            {!!differences.length && (
              <Button
                variant="outline"
                size="sm"
                disabled={jsonDirty || !!busy}
                onClick={() =>
                  replaceDraft(
                    templateRun(form, selectedTemplate),
                    `Restored ${selectedTemplate.name}.`,
                  )
                }
              >
                <RotateCcw /> Restore template
              </Button>
            )}
          </div>
        </div>
        {reviewChanges && !jsonDirty && !!differences.length && (
          <div id="template-change-review" className="template-change-review">
            <div className="template-change-review-heading">
              <span>Setting</span>
              <span>Template</span>
              <span>Current</span>
            </div>
            {differences.map((difference) => (
              <div
                className="template-change-row"
                key={`${difference.stage}.${difference.field}`}
              >
                <div>
                  <strong>{difference.label}</strong>
                  <small>
                    {difference.stage === "shared"
                      ? "Run options"
                      : (schema.stages.find(
                          (entry) => entry.id === difference.stage,
                        )?.label ?? difference.stage)}
                  </small>
                </div>
                <code>{formatTemplateValue(difference.before)}</code>
                <code>{formatTemplateValue(difference.after)}</code>
              </div>
            ))}
          </div>
        )}
        <p className="template-scope-note">
          Templates set analysis choices and selected stages. Run names and
          cache directories stay with this run.
        </p>
      </section>
      {templateFeedback && (
        <div className="configuration-feedback" role="status">
          <span>
            <Check size={16} /> {templateFeedback}
          </span>
        </div>
      )}
      {templateError && <Notice>{templateError}</Notice>}
      {templateWarnings.length > 0 && (
        <Notice>{templateWarnings.join(" ")}</Notice>
      )}
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
              setSelectedTemplate(undo.template);
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
          <div className={fieldClass("data_dir")}>
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
            {sharedHint("data_dir")}
          </div>
          <div className={fieldClass("session_list_file")}>
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
            {sharedHint("session_list_file")}
            <p id="session-list-help" className="field-hint">
              Only listed sessions found in the recording directory are used.
            </p>
          </div>
          <div className={fieldClass("n_jobs")}>
            <label htmlFor="workers">Parallel workers</label>
            <Input
              id="workers"
              aria-describedby="workers-help"
              type="number"
              step={1}
              value={form.n_jobs}
              onChange={(e) => update({ n_jobs: Number(e.target.value) })}
            />
            {sharedHint("n_jobs")}
            <p id="workers-help" className="field-hint">
              Use 1 for a single worker or −1 for all available CPUs.
            </p>
          </div>
          <div className={fieldClass("max_sessions_to_run")}>
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
            {sharedHint("max_sessions_to_run")}
          </div>
        </div>
        {sharedHint("figure_formats")}
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
          {sharedHint("stages")}
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
              className={`stage-option ${active === item.id ? "active" : ""} ${form.stages.includes(item.id) !== selectedTemplate.config.stages.includes(item.id) ? "stage-inclusion-changed" : ""}`}
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
                {differences.some(
                  (difference) => difference.stage === item.id,
                ) && (
                  <span
                    className="stage-change-count"
                    title={`Parameters changed from ${selectedTemplate.name}`}
                    aria-label={`${differences.filter((difference) => difference.stage === item.id).length} parameters changed from template`}
                  >
                    {
                      differences.filter(
                        (difference) => difference.stage === item.id,
                      ).length
                    }
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
                {changedOnly
                  ? ` · differences from ${selectedTemplate.name}`
                  : ""}
              </p>
              <div id="parameter-fields" className="parameters-grid">
                {filtered.map((field) => (
                  <FieldEditor
                    key={`${stage.id}-${field.name}`}
                    stageId={stage.id}
                    field={field}
                    value={fieldValue(form, stage.id, field)}
                    templateValue={fieldValue(templateForm, stage.id, field)}
                    templateName={selectedTemplate.name}
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
                      : "No changes from this template"}
                  </h3>
                  <p>
                    {query
                      ? `Try a shorter search in ${stage.label || humanize(stage.id)}${changedOnly ? ", or show all parameters" : ""}.`
                      : `The parameters in this stage match ${selectedTemplate.name}.`}
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
            disabled={!!busy || jsonDirty || !form.stages.length || running}
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
