import type {
  Field,
  Json,
  Manifest,
  RunRequest,
  Schema,
  Settings,
} from "./types";

export function newCacheDirectory() {
  const timestamp = new Date().toISOString().replace(/[-:.]/g, "");
  return `cache/dashboard_${timestamp}_${crypto.randomUUID().slice(0, 8)}`;
}

export function initialRun(
  schema: Schema,
  seed?: Partial<RunRequest> | null,
  cache = newCacheDirectory(),
): RunRequest {
  return structuredClone({
    name: "Working memory analysis",
    data_dir: "data/nature",
    stages: schema.stages.map((stage) => stage.id),
    n_jobs: 1,
    session_list_file: null,
    max_sessions_to_run: null,
    figure_formats: ["png"],
    allow_existing: false,
    ...schema.defaults,
    cache_dir: cache,
    settings: schema.presets.example,
    ...seed,
  });
}

const sharedFields: Record<string, keyof RunRequest> = {
  data_dir: "data_dir",
  cache_dir: "cache_dir",
  n_jobs: "n_jobs",
  n_jobs_session: "n_jobs",
  cv_n_jobs: "n_jobs",
  session_list_file: "session_list_file",
  max_sessions_to_run: "max_sessions_to_run",
  figure_formats: "figure_formats",
};

export function fieldValue(
  form: RunRequest,
  stage: string,
  field: Field,
): Json {
  const overrides = form.settings[stage];
  // Explicit null is an override, not a request to restore the default.
  if (overrides && Object.hasOwn(overrides, field.name))
    return overrides[field.name];
  const shared = sharedFields[field.name];
  return shared ? (form[shared] as Json) : field.default;
}

export function choiceValue(field: Field, value: Json): string {
  if (value === null) return "";
  const text = String(value);
  // Python accepts enum member names (SIGMOID) and their values (sigmoid).
  return (
    field.choices?.find(
      (choice) => choice.toLowerCase() === text.toLowerCase(),
    ) ?? text
  );
}

export function nonNullValue(field: Field): Json {
  if (field.default !== null) return field.default;
  if (field.choices?.length) return field.choices[0];
  if (field.type === "boolean") return false;
  if (field.type === "number" || field.type === "integer") return 1;
  if (field.type === "array") return [];
  return "";
}

export function parseSettings(text: string): Settings {
  const value: unknown = JSON.parse(text);
  if (!value || Array.isArray(value) || typeof value !== "object") {
    throw new Error("Settings must be an object keyed by stage.");
  }
  for (const [stage, settings] of Object.entries(value)) {
    if (!settings || Array.isArray(settings) || typeof settings !== "object") {
      throw new Error(
        `Settings for ${stage} must be an object of parameter names and values.`,
      );
    }
  }
  return value as Settings;
}

function canonical(value: Json): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value !== null && typeof value === "object") {
    return `{${Object.keys(value)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${canonical(value[key])}`)
      .join(",")}}`;
  }
  return JSON.stringify(value);
}

export function presetName(
  settings: Settings,
  schema: Schema,
): "example" | "smoke" | "custom" {
  for (const name of ["example", "smoke"] as const) {
    if (canonical(settings) === canonical(schema.presets[name])) return name;
  }
  return "custom";
}

export function manifestSeed(
  manifest: Manifest,
  name: string,
): Partial<RunRequest> {
  const runner = (
    manifest.runner_config && typeof manifest.runner_config === "object"
      ? manifest.runner_config
      : {}
  ) as Record<string, Json>;
  const seed: Partial<RunRequest> = {
    name: `${name} · copy`,
    allow_existing: false,
  };
  // Do not forward runner-only flags, old output paths, or a settings filename to RunRequest.
  for (const key of [
    "data_dir",
    "n_jobs",
    "session_list_file",
    "max_sessions_to_run",
    "figure_formats",
  ] as const) {
    if (Object.hasOwn(runner, key)) Object.assign(seed, { [key]: runner[key] });
  }
  const settings = structuredClone(manifest.settings ?? {});
  for (const overrides of Object.values(settings)) {
    delete overrides.cache_dir;
    delete overrides.data_dir;
    for (const [field, shared] of Object.entries(sharedFields)) {
      if (
        Object.hasOwn(overrides, field) &&
        Object.hasOwn(runner, shared) &&
        canonical(overrides[field]) === canonical(runner[shared])
      )
        delete overrides[field];
    }
  }
  seed.settings = settings;
  // Resolved settings include every requested stage, even when a run failed early.
  if (Object.keys(settings).length) seed.stages = Object.keys(settings);
  else if (
    Array.isArray(runner.stages) &&
    runner.stages.every((stage) => typeof stage === "string") &&
    !runner.stages.some((stage) => stage === "all" || stage === "mixed")
  )
    seed.stages = runner.stages as string[];
  return seed;
}
