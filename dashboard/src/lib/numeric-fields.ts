import type { Json, RunRequest, Schema } from "./types";
import { fieldValue } from "./configuration";
import { humanize } from "./utils";

export type NumericDrafts = Record<string, string>;
export type NumericSpec = {
  key: string;
  label: string;
  type: "integer" | "number";
  nullable?: boolean;
  stage?: string;
  name: string;
};
export type NumericResult =
  { value: number | null; error?: never } | { error: string; value?: never };

export const numericKey = (stage: string, name: string) => `${stage}.${name}`;

export function numericSpecs(schema: Schema): NumericSpec[] {
  return [
    {
      key: "shared.n_jobs",
      label: "Parallel workers",
      type: "integer",
      name: "n_jobs",
    },
    {
      key: "shared.max_sessions_to_run",
      label: "Maximum sessions",
      type: "integer",
      name: "max_sessions_to_run",
      nullable: true,
    },
    ...schema.stages.flatMap((stage) =>
      stage.fields.flatMap((field): NumericSpec[] =>
        field.type === "integer" || field.type === "number"
          ? [
              {
                key: numericKey(stage.id, field.name),
                label: `${stage.label || humanize(stage.id)}: ${humanize(field.name)}`,
                type: field.type,
                nullable: field.nullable,
                stage: stage.id,
                name: field.name,
              },
            ]
          : [],
      ),
    ),
  ];
}

function checkNumber(value: number | null, spec: NumericSpec): NumericResult {
  if (value === null)
    return spec.nullable ? { value } : { error: "Enter a number." };
  if (!Number.isFinite(value)) return { error: "Enter a finite number." };
  if (spec.type === "integer" && !Number.isSafeInteger(value))
    return { error: "Enter a whole number within the safe integer range." };
  if (
    ["n_jobs", "n_jobs_session", "cv_n_jobs"].includes(spec.name) &&
    value === 0
  )
    return { error: "Worker count cannot be zero." };
  if (["max_sessions_to_run", "classifier_c"].includes(spec.name) && value <= 0)
    return { error: "Enter a number greater than zero." };
  return { value };
}

/** Check a draft without mutating it; the caller decides when to commit a valid value. */
export function parseNumericDraft(
  text: string,
  spec: NumericSpec,
): NumericResult {
  const trimmed = text.trim();
  if (!trimmed) return checkNumber(null, spec);
  // Number() alone also accepts hex and blanks; neither is a decimal field entry.
  if (!/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(trimmed))
    return {
      error: "Enter a complete decimal number (for example, 0.01 or 1e-2).",
    };
  return checkNumber(Number(trimmed), spec);
}

export function checkNumericValue(
  value: Json,
  spec: NumericSpec,
): NumericResult {
  if (value !== null && typeof value !== "number")
    return {
      error: "Use a numeric JSON value, not text or another value type.",
    };
  return checkNumber(value, spec);
}

export function setNumericValue(
  form: RunRequest,
  spec: NumericSpec,
  value: number | null,
): RunRequest {
  if (spec.stage)
    return {
      ...form,
      settings: {
        ...form.settings,
        [spec.stage]: { ...form.settings[spec.stage], [spec.name]: value },
      },
    };
  return { ...form, [spec.name]: value };
}

/** Resolve against the latest form, including edits committed in the same event turn. */
export function resolveNumericDrafts(
  form: RunRequest,
  drafts: NumericDrafts,
  schema: Schema,
) {
  let resolved = form;
  const issues: Record<string, string> = {};
  const specs = numericSpecs(schema);
  // Apply all explicit drafts before checking fields that inherit shared values.
  for (const spec of specs) {
    if (!Object.hasOwn(drafts, spec.key)) continue;
    const result = parseNumericDraft(drafts[spec.key], spec);
    if (result.error !== undefined) issues[spec.key] = result.error;
    else resolved = setNumericValue(resolved, spec, result.value);
  }
  for (const spec of specs) {
    if (issues[spec.key]) continue;
    const field = spec.stage
      ? schema.stages
          .find((stage) => stage.id === spec.stage)!
          .fields.find((field) => field.name === spec.name)!
      : null;
    const value = spec.stage
      ? fieldValue(resolved, spec.stage, field!)
      : resolved[spec.name as "n_jobs" | "max_sessions_to_run"];
    const result = checkNumericValue(value, spec);
    if (result.error !== undefined) issues[spec.key] = result.error;
  }
  return { form: resolved, issues, valid: Object.keys(issues).length === 0 };
}
