import { useMemo, useState } from "react";
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
  presetName,
} from "@/lib/configuration";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { CopyButton, Notice } from "./shared";
function FieldEditor({
  field,
  value,
  onChange,
}: {
  field: Field;
  value: Json;
  onChange: (value: Json) => void;
}) {
  const id = `field-${field.name}`;
  const selectedChoice = choiceValue(field, value);
  return (
    <div className="parameter">
      <div className="parameter-label">
        <label htmlFor={id}>{humanize(field.name)}</label>
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
      {field.type === "boolean" ? (
        <label className="switch-row" htmlFor={id}>
          <input
            id={id}
            type="checkbox"
            role="switch"
            checked={value === true}
            onChange={(e) => onChange(e.target.checked)}
          />
          <span>{value ? "Enabled" : "Disabled"}</span>
        </label>
      ) : field.choices?.length ? (
        <select
          id={id}
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
      <p>{field.description || field.name.replaceAll("_", " ")}</p>
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
  const [jsonMode, setJsonMode] = useState(false);
  const [jsonText, setJsonText] = useState(
    JSON.stringify(initial.settings, null, 2),
  );
  const [jsonDirty, setJsonDirty] = useState(false);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState("");
  const [validation, setValidation] = useState<{
    command: string;
    resolved: Settings;
  } | null>(null);
  const update = (patch: Partial<RunRequest>) => {
    setForm((v) => ({ ...v, ...patch }));
    setValidation(null);
    setError("");
  };
  const stage = schema.stages.find((s) => s.id === active)!;
  const preset = presetName(form.settings, schema);
  const filtered = stage.fields.filter((f) =>
    `${f.name} ${f.description}`.toLowerCase().includes(query.toLowerCase()),
  );
  const presetChange = (value: "example" | "smoke") => {
    const settings = structuredClone(schema.presets[value]);
    update({ settings });
    setJsonText(JSON.stringify(settings, null, 2));
    setJsonDirty(false);
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
      } else
        setValidation(
          await api("/validate", {
            method: "POST",
            body: JSON.stringify(form),
          }),
        );
    } catch (e) {
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
          <p>Start with a considered preset. Make every parameter your own.</p>
        </div>
        <div className="heading-actions">
          <Button
            variant="outline"
            onClick={() => {
              setForm(structuredClone(initial));
              setJsonText(JSON.stringify(initial.settings, null, 2));
              setJsonDirty(false);
              setValidation(null);
              setError("");
            }}
          >
            <RotateCcw />
            Reset
          </Button>
          <Button onClick={() => submit(true)} disabled={!!busy || jsonDirty}>
            <Play />
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
            disabled={jsonDirty}
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
              value={form.cache_dir}
              onChange={(e) => update({ cache_dir: e.target.value })}
            />
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
              placeholder="All available sessions"
              value={form.session_list_file ?? ""}
              onChange={(e) =>
                update({ session_list_file: e.target.value || null })
              }
            />
          </div>
          <div className="field">
            <label htmlFor="workers">Parallel workers</label>
            <Input
              id="workers"
              type="number"
              step={1}
              value={form.n_jobs}
              onChange={(e) => update({ n_jobs: Number(e.target.value) })}
            />
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
            onClick={() => update({ stages: schema.stages.map((s) => s.id) })}
          >
            Select all
          </Button>
          <Button
            variant="outline"
            size="sm"
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
              <button onClick={() => setActive(item.id)}>
                <span className="stage-index">
                  {String(index + 1).padStart(2, "0")}
                </span>
                <span>{item.label || humanize(item.id)}</span>
                <ChevronRight size={14} />
              </button>
            </div>
          ))}
        </aside>
        <section className="panel parameters-panel">
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
            </div>
            <Button variant="outline" size="sm" onClick={toggleJson}>
              {jsonMode ? <SlidersHorizontal /> : <Code2 />}
              {jsonMode ? "Form" : "JSON"}
            </Button>
          </div>
          {jsonMode ? (
            <>
              <textarea
                className="json-editor"
                aria-label="Pipeline settings JSON"
                spellCheck={false}
                value={jsonText}
                onChange={(e) => {
                  setJsonText(e.target.value);
                  setJsonDirty(true);
                  setValidation(null);
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
              <div className="search-field parameter-search">
                <Search size={16} />
                <input
                  aria-label="Search parameters"
                  placeholder="Find a parameter…"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                />
                <span>{filtered.length} fields</span>
              </div>
              <div className="parameters-grid">
                {filtered.map((field) => (
                  <FieldEditor
                    key={`${stage.id}-${field.name}`}
                    field={field}
                    value={fieldValue(form, stage.id, field)}
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
            </>
          )}
        </section>
      </div>
      <section className="panel launch-panel">
        <div>
          <div className="eyebrow">READY WHEN YOU ARE</div>
          <h2>Check the configuration, then let it run.</h2>
          <p>
            Progress and logs stay available while you explore previous results.
          </p>
        </div>
        <div className="heading-actions">
          <Button
            variant="outline"
            onClick={() => submit(false)}
            disabled={!!busy || jsonDirty}
          >
            {busy === "validate" ? (
              <LoaderCircle className="animate-spin" />
            ) : (
              <FolderOpen />
            )}
            Validate & preview
          </Button>
          <Button onClick={() => submit(true)} disabled={!!busy || jsonDirty}>
            Start pipeline
            <ArrowRight />
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
