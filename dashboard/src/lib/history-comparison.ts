import { normalizeSettings, sameFieldValue } from "./configuration";
import { formatTemplateValue } from "./templates";
import { humanize } from "./utils";
import type { Field, Json, Manifest, PipelineTemplate, Schema } from "./types";

export type HistorySetting = {
  stage: string;
  field: string;
  label: string;
  before: Json | undefined;
  after: Json | undefined;
  status: "changed" | "same" | "unavailable";
};

const sharedFields = {
  n_jobs: "n_jobs",
  n_jobs_session: "n_jobs",
  cv_n_jobs: "n_jobs",
  max_sessions_to_run: "max_sessions_to_run",
} as const;
const omittedStageFields = new Set([
  "cache_dir",
  "data_dir",
  "figure_formats",
  "figure_font",
  "trust_unverified_legacy_results",
  "dry_run",
]);
const sharedLabels = {
  n_jobs: "Parallel workers",
  max_sessions_to_run: "Session limit",
  figure_formats: "Figure formats",
  figure_font: "Figure font",
  data_dir: "Recording directory",
  session_list_file: "Session list file",
};

function record(value: unknown): Record<string, Json> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, Json>)
    : {};
}

/** Selected stages belong to this invocation, not today's pipeline defaults. */
export function historicalStages(manifest: Manifest): string[] {
  const runner = record(manifest.runner_config);
  const requested = manifest.original_request?.stages ?? runner.stages;
  if (
    Array.isArray(requested) &&
    !requested.some((value) => value === "all" || value === "mixed")
  )
    return [
      ...new Set(
        requested.filter((value): value is string => typeof value === "string"),
      ),
    ];
  const settings = Object.keys(manifest.settings ?? {});
  return settings.length
    ? settings
    : [...new Set(manifest.stages.map((stage) => stage.stage))];
}

// Resolve only paths whose base is known. Do not guess HOME, symlink targets,
// Windows drive state, or a missing invocation directory for older records.
function comparablePath(value: Json, cwd: string | undefined): Json {
  if (
    typeof value !== "string" ||
    !cwd?.startsWith("/") ||
    value.startsWith("~") ||
    value.includes("\\")
  )
    return value;
  const joined = value.startsWith("/") ? value : `${cwd}/${value}`;
  const pieces: string[] = [];
  for (const piece of joined.split("/")) {
    if (!piece || piece === ".") continue;
    if (piece === "..") pieces.pop();
    else pieces.push(piece);
  }
  return `/${pieces.join("/")}`;
}

/** Compare recorded values only; missing historical/template values stay unknown. */
export function historySettings(
  manifest: Manifest,
  template: PipelineTemplate,
  schema?: Schema | null,
): HistorySetting[] {
  const rows: HistorySetting[] = [];
  const runner = record(manifest.runner_config);
  const original = manifest.original_request;
  const baseline = template.config;
  const recordedSettings = normalizeSettings(manifest.settings ?? {});
  const baselineSettings = normalizeSettings(baseline.settings);
  const cwd = manifest.invocation?.cwd;
  function compare(
    stage: string,
    name: string,
    before: Json | undefined,
    after: Json | undefined,
    label = humanize(name),
    field?: Field,
    path = false,
  ) {
    const unknown = before === undefined || after === undefined;
    const normalize = (value: Json) =>
      name === "figure_formats" && Array.isArray(value)
        ? [...value].sort()
        : path
          ? comparablePath(value, cwd)
          : value;
    rows.push({
      stage,
      field: name,
      label,
      before,
      after,
      status: unknown
        ? "unavailable"
        : sameFieldValue(
              field ?? { name, type: "unknown", default: null },
              normalize(before!),
              normalize(after!),
            )
          ? "same"
          : "changed",
    });
  }
  const selectedStages = historicalStages(manifest);
  compare(
    "shared",
    "stages",
    baseline.stages,
    selectedStages.length || Array.isArray(runner.stages) || original?.stages
      ? selectedStages
      : undefined,
    "Selected stages",
  );
  for (const [name, label] of Object.entries(sharedLabels)) {
    const key = name as keyof typeof sharedLabels;
    const path = key === "data_dir" || key === "session_list_file";
    if (path && !Object.hasOwn(baseline, key)) continue;
    const before = baseline[key];
    const after =
      original && Object.hasOwn(original, key) ? original[key] : runner[key];
    if (before === undefined && after === undefined) continue;
    compare("shared", name, before, after, label, undefined, path);
  }
  for (const stage of historicalStages(manifest)) {
    const recorded = recordedSettings[stage] ?? {};
    const templateSettings = baselineSettings[stage] ?? {};
    const fields =
      schema?.stages.find((item) => item.id === stage)?.fields ?? [];
    for (const name of new Set([
      ...Object.keys(templateSettings),
      ...Object.keys(recorded),
    ])) {
      if (omittedStageFields.has(name)) continue;
      const field = fields.find((item) => item.name === name);
      const shared = sharedFields[name as keyof typeof sharedFields];
      const before = Object.hasOwn(templateSettings, name)
        ? templateSettings[name]
        : shared
          ? baseline[shared]
          : name === "session_list_file" && Object.hasOwn(baseline, name)
            ? baseline[name]
            : undefined;
      if (name === "session_list_file" && before === undefined) continue;
      // Shared runner values already have one row. Keep actual per-stage
      // overrides visible, including fields absent from historical runners.
      if (
        shared &&
        !Object.hasOwn(templateSettings, name) &&
        Object.hasOwn(runner, shared) &&
        recorded[name] !== undefined &&
        sameFieldValue(
          { name, type: "unknown", default: null },
          recorded[name],
          runner[shared],
        )
      )
        continue;
      const raw = original?.settings?.[stage];
      const after =
        name === "session_list_file" && raw && Object.hasOwn(raw, name)
          ? raw[name]
          : recorded[name];
      compare(
        stage,
        name,
        before,
        after,
        humanize(name),
        field,
        field?.path_kind != null || name === "session_list_file",
      );
    }
  }
  return rows;
}

export function historyValue(
  value: Json | undefined,
  side: "template" | "recorded",
) {
  return value === undefined
    ? side === "recorded"
      ? "Not recorded"
      : "Not in template"
    : formatTemplateValue(value);
}

export function filterHistorySettings(
  rows: HistorySetting[],
  changedOnly: boolean,
  query: string,
) {
  const words = query
    .toLowerCase()
    .replaceAll(/[_-]/g, " ")
    .trim()
    .split(/\s+/);
  return rows.filter(
    (row) =>
      (!changedOnly || row.status === "changed") &&
      words.every((word) =>
        `${row.stage} ${row.field} ${row.label} ${historyValue(row.before, "template")} ${historyValue(row.after, "recorded")}`
          .toLowerCase()
          .replaceAll(/[_-]/g, " ")
          .includes(word),
      ),
  );
}
