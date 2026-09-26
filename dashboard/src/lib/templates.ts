import { fieldValue, initialRun, sameFieldValue } from "./configuration";
import { humanize } from "./utils";
import type {
  Field,
  Json,
  PipelineTemplate,
  RunRequest,
  Schema,
  TemplateChange,
  TemplateConfig,
} from "./types";

const sharedStageFields = {
  n_jobs: "n_jobs",
  n_jobs_session: "n_jobs",
  cv_n_jobs: "n_jobs",
  session_list_file: "session_list_file",
  max_sessions_to_run: "max_sessions_to_run",
} as const;

function capturesAllowlists(template: PipelineTemplate) {
  return (
    Object.hasOwn(template.config, "session_list_file") ||
    Object.values(template.config.settings).some((settings) =>
      Object.hasOwn(settings, "session_list_file"),
    )
  );
}

/** Capture stage defaults now, rather than inheriting changed defaults later. */
export function templateConfig(
  form: RunRequest,
  schema: Schema,
  includePaths: boolean,
): TemplateConfig {
  // Keep unknown overrides for server validation; saving must not silently fix a
  // misspelled field or discard a stage from a user's JSON draft.
  const settings = structuredClone(form.settings);
  for (const stage of schema.stages) {
    settings[stage.id] ??= {};
    for (const field of stage.fields) {
      // These values are runner settings, not independent stage defaults. The
      // real schema excludes them; retain an explicit JSON override if present.
      if (
        Object.hasOwn(sharedStageFields, field.name) ||
        field.name === "cache_dir" ||
        field.name === "data_dir" ||
        field.name === "figure_formats"
      )
        continue;
      if (!Object.hasOwn(settings[stage.id], field.name)) {
        settings[stage.id][field.name] = structuredClone(
          fieldValue(form, stage.id, field),
        );
      }
    }
  }
  if (!includePaths) {
    for (const overrides of Object.values(settings))
      delete overrides.session_list_file;
  }
  return structuredClone({
    settings,
    stages: form.stages,
    n_jobs: form.n_jobs,
    max_sessions_to_run: form.max_sessions_to_run,
    // The pipeline applies these through its figure-export context. Formats
    // must not be injected into individual stage dataclasses.
    figure_formats: form.figure_formats,
    ...(includePaths
      ? { data_dir: form.data_dir, session_list_file: form.session_list_file }
      : {}),
  });
}

/** Apply analysis choices while keeping this run's identity and output target. */
export function templateRun(
  form: RunRequest,
  template: PipelineTemplate,
): RunRequest {
  const { config } = template;
  const next = structuredClone({
    ...form,
    settings: config.settings,
    stages: config.stages,
    n_jobs: config.n_jobs,
    max_sessions_to_run: config.max_sessions_to_run,
    figure_formats: config.figure_formats,
    ...(Object.hasOwn(config, "data_dir")
      ? { data_dir: config.data_dir! }
      : {}),
    ...(Object.hasOwn(config, "session_list_file")
      ? { session_list_file: config.session_list_file! }
      : {}),
  });
  if (!capturesAllowlists(template)) {
    for (const [stage, overrides] of Object.entries(form.settings)) {
      if (Object.hasOwn(overrides, "session_list_file")) {
        next.settings[stage] ??= {};
        next.settings[stage].session_list_file = structuredClone(
          overrides.session_list_file,
        );
      }
    }
  }
  return next;
}

export function builtInTemplates(schema: Schema): PipelineTemplate[] {
  return (["example", "smoke"] as const).map((id) => ({
    id,
    name: id === "example" ? "Example pipeline" : "Smoke test",
    description:
      id === "example"
        ? "The example pipeline's analysis settings."
        : "Reduced decoding and model work for an integration check.",
    builtin: true,
    config: templateConfig(
      initialRun(
        schema,
        { settings: schema.presets[id] },
        "cache/template-baseline",
      ),
      schema,
      false,
    ),
  }));
}

function effectiveStageValue(
  form: RunRequest,
  stage: string,
  name: string,
  field?: Field,
): Json | undefined {
  if (Object.hasOwn(form.settings[stage] ?? {}, name))
    return form.settings[stage][name];
  if (field) return fieldValue(form, stage, field);
  const shared = sharedStageFields[name as keyof typeof sharedStageFields];
  return shared ? form[shared] : undefined;
}

/** Compare effective values against the chosen template, including JSON overrides. */
export function templateChanges(
  form: RunRequest,
  template: PipelineTemplate,
  schema: Schema,
): TemplateChange[] {
  const baseline = templateRun(form, template);
  const changes: TemplateChange[] = [];
  const compare = (
    stage: string,
    name: string,
    before: Json | undefined,
    after: Json | undefined,
    label = humanize(name),
    field?: Field,
  ) => {
    // Reordering checked export formats changes no analysis or output choice.
    // Retain duplicates here so invalid duplicate values still show a change.
    const orderIndependent = stage === "shared" && name === "figure_formats";
    const comparisonBefore =
      orderIndependent && Array.isArray(before) ? [...before].sort() : before;
    const comparisonAfter =
      orderIndependent && Array.isArray(after) ? [...after].sort() : after;
    const same =
      before === undefined || after === undefined
        ? before === after
        : sameFieldValue(
            field ?? { name, type: "unknown", default: null },
            comparisonBefore!,
            comparisonAfter!,
          );
    if (!same) changes.push({ stage, field: name, before, after, label });
  };
  for (const [name, label] of [
    ["stages", "Selected stages"],
    ["n_jobs", "Parallel workers"],
    ["max_sessions_to_run", "Session limit"],
    ["figure_formats", "Figure formats"],
  ] as const)
    compare("shared", name, baseline[name], form[name], label);
  for (const [name, label] of [
    ["data_dir", "Recording directory"],
    ["session_list_file", "Session list file"],
  ] as const) {
    if (Object.hasOwn(template.config, name))
      compare("shared", name, baseline[name], form[name], label);
  }
  const stages = [
    ...new Set([
      ...schema.stages.map((stage) => stage.id),
      ...Object.keys(baseline.settings),
      ...Object.keys(form.settings),
    ]),
  ];
  for (const stage of stages) {
    const stageSchema = schema.stages.find((entry) => entry.id === stage);
    if (!stageSchema) {
      // Even an empty unknown stage is invalid; no per-field loop can detect it.
      compare(
        stage,
        "__stage__",
        template.config.settings[stage],
        form.settings[stage],
        "Unknown stage settings",
      );
      continue;
    }
    const fields = stageSchema.fields;
    const names = new Set([
      ...fields.map((field) => field.name),
      ...Object.keys(baseline.settings[stage] ?? {}),
      ...Object.keys(form.settings[stage] ?? {}),
    ]);
    for (const name of names) {
      if (!capturesAllowlists(template) && name === "session_list_file")
        continue;
      const field = fields.find((entry) => entry.name === name);
      compare(
        stage,
        name,
        effectiveStageValue(baseline, stage, name, field),
        effectiveStageValue(form, stage, name, field),
        humanize(name),
        field,
      );
    }
  }
  return changes;
}

export function formatTemplateValue(value: Json | undefined): string {
  if (value === undefined) return "Not set";
  if (value === null) return "None";
  if (typeof value === "boolean") return value ? "Enabled" : "Disabled";
  if (typeof value === "string") return value || "Empty string";
  if (Array.isArray(value))
    return value.length
      ? value.map(formatTemplateValue).join(", ")
      : "Empty list";
  return typeof value === "object" ? JSON.stringify(value) : String(value);
}
