import { describe, expect, it } from "vitest";
import { initialRun, fieldValue } from "./configuration";
import type { Schema } from "./types";
import {
  numericSpecs,
  parseNumericDraft,
  resolveNumericDrafts,
  setNumericValue,
} from "./numeric-fields";

const schema: Schema = {
  stages: [
    {
      id: "decode",
      label: "Decode",
      description: "",
      fields: [
        { name: "classifier_c", type: "number", default: 1 },
        { name: "n_decode_shuffle", type: "integer", default: 100 },
        { name: "cv_n_jobs", type: "integer", default: 1 },
        {
          name: "optional_seed",
          type: "integer",
          default: null,
          nullable: true,
        },
      ],
    },
  ],
  defaults: { n_jobs: 4, max_sessions_to_run: null },
  presets: { example: { decode: { classifier_c: 1 } }, smoke: {} },
};
const specs = numericSpecs(schema);
const spec = (name: string) => specs.find((item) => item.key === name)!;
const form = () => initialRun(schema, null, "cache/numeric-test");

describe("numeric drafts and submission snapshots", () => {
  it("retains each raw decimal keystroke without changing the committed form", () => {
    const current = form();
    for (const text of ["", "0", "0.", "0.0", "0.01"]) {
      const drafts = { "decode.classifier_c": text };
      const result = resolveNumericDrafts(current, drafts, schema);
      expect(drafts["decode.classifier_c"]).toBe(text);
      expect(current.settings.decode.classifier_c).toBe(1);
      expect(result.valid).toBe(text === "0.01");
    }
    const snapshot = resolveNumericDrafts(
      current,
      { "decode.classifier_c": "0.01" },
      schema,
    );
    expect(snapshot.form.settings.decode.classifier_c).toBe(0.01);
  });

  it("resolves every pending field against the latest form without losing a preceding blur commit", () => {
    const original = form();
    const latest = setNumericValue(original, spec("shared.n_jobs"), 8);
    const result = resolveNumericDrafts(
      latest,
      {
        "decode.classifier_c": "1e-2",
        "decode.n_decode_shuffle": "200",
        "shared.max_sessions_to_run": "12",
      },
      schema,
    );
    expect(result.valid).toBe(true);
    expect(result.form.n_jobs).toBe(8);
    expect(result.form.max_sessions_to_run).toBe(12);
    expect(result.form.settings.decode).toEqual({
      classifier_c: 0.01,
      n_decode_shuffle: 200,
    });
    expect(original.n_jobs).toBe(4);
    expect(original.settings.decode.classifier_c).toBe(1);
    // Resolve is idempotent; subsequent action snapshots retain all committed edits.
    expect(resolveNumericDrafts(result.form, {}, schema).form).toEqual(
      result.form,
    );
  });

  it("does not accept an old numeric value when a hidden stage contains an incomplete draft", () => {
    const current = form();
    const result = resolveNumericDrafts(
      current,
      { "decode.classifier_c": "1e-" },
      schema,
    );
    expect(result.valid).toBe(false);
    expect(result.issues["decode.classifier_c"]).toContain("complete decimal");
    expect(current.settings.decode.classifier_c).toBe(1);
  });

  it("accepts signed and exponent forms only when complete, finite, and integral if required", () => {
    for (const text of [".01", "0.010", "+1e-2", " 0.01 "])
      expect(parseNumericDraft(text, spec("decode.classifier_c"))).toEqual({
        value: 0.01,
      });
    for (const text of [
      "-",
      "+",
      ".",
      "1e",
      "1e-",
      "1e309",
      "NaN",
      "Infinity",
      "0x10",
      "1,2",
    ])
      expect(
        parseNumericDraft(text, spec("decode.classifier_c")).error,
      ).toBeTruthy();
    expect(parseNumericDraft("-1", spec("shared.n_jobs"))).toEqual({
      value: -1,
    });
    expect(parseNumericDraft("1e1", spec("shared.n_jobs"))).toEqual({
      value: 10,
    });
    for (const text of ["", "0", "1.5", "9007199254740992"])
      expect(parseNumericDraft(text, spec("shared.n_jobs")).error).toBeTruthy();
    for (const text of ["0", "-1", "1.5"])
      expect(
        parseNumericDraft(text, spec("shared.max_sessions_to_run")).error,
      ).toBeTruthy();
  });

  it("preserves nullable blank values and detects malformed numeric JSON instead of coercing it", () => {
    const result = resolveNumericDrafts(
      form(),
      {
        "decode.optional_seed": "",
        "shared.max_sessions_to_run": " ",
      },
      schema,
    );
    expect(result.valid).toBe(true);
    expect(result.form.settings.decode.optional_seed).toBeNull();
    expect(result.form.max_sessions_to_run).toBeNull();
    for (const value of ["0.01", null, false, [], 0]) {
      const current = form();
      current.settings.decode.classifier_c = value;
      expect(resolveNumericDrafts(current, {}, schema).valid).toBe(false);
    }
    const current = form();
    current.settings.decode.n_decode_shuffle = 1.5;
    expect(
      resolveNumericDrafts(current, {}, schema).issues[
        "decode.n_decode_shuffle"
      ],
    ).toContain("whole number");
  });

  it("applies shared numeric drafts before validating inherited stage values", () => {
    const current = form();
    current.n_jobs = 0;
    const result = resolveNumericDrafts(
      current,
      { "shared.n_jobs": "6" },
      schema,
    );
    expect(result.valid).toBe(true);
    const field = schema.stages[0].fields.find(
      (field) => field.name === "cv_n_jobs",
    )!;
    expect(fieldValue(result.form, "decode", field)).toBe(6);
  });
});
