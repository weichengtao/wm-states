import { describe, expect, it } from "vitest";
import {
  choiceValue,
  fieldValue,
  initialRun,
  manifestSeed,
  nonNullValue,
  parseSettings,
  parameterMatchesQuery,
  presetName,
  sameFieldValue,
} from "./configuration";
import type { Field, Manifest, Schema } from "./types";

const field: Field = {
  name: "max_points",
  type: "integer",
  nullable: true,
  default: 50,
};
const schema: Schema = {
  stages: [{ id: "decode", label: "Decode", description: "", fields: [] }],
  defaults: { cache_dir: "cache/next_run", n_jobs: 2 },
  presets: {
    example: { decode: { grid_search_for_c: true } },
    smoke: { decode: { grid_search_for_c: false } },
  },
};

describe("configuration defaults and manual overrides", () => {
  it("starts with example settings and a unique cache rather than the backend placeholder", () => {
    const form = initialRun(schema, null, "cache/unique");
    expect(form.cache_dir).toBe("cache/unique");
    expect(form.settings).toEqual(schema.presets.example);
    expect(form.n_jobs).toBe(2);
    form.settings.decode.grid_search_for_c = false;
    expect(schema.presets.example.decode.grid_search_for_c).toBe(true);
  });

  it("preserves the requested seed, including empty settings and shared nulls", () => {
    const form = initialRun(
      schema,
      {
        cache_dir: "cache/specific",
        settings: {},
        n_jobs: -1,
        session_list_file: null,
        max_sessions_to_run: null,
        figure_formats: ["tif"],
      },
      "cache/unused",
    );
    expect(form.cache_dir).toBe("cache/specific");
    expect(form.settings).toEqual({});
    expect(form.n_jobs).toBe(-1);
    expect(form.figure_formats).toEqual(["tif"]);
  });

  it("retains explicit null, false, and zero instead of replacing them with defaults", () => {
    const form = initialRun(schema, {
      settings: { decode: { max_points: null, enabled: false, threshold: 0 } },
    });
    expect(fieldValue(form, "decode", field)).toBeNull();
    expect(
      fieldValue(form, "decode", {
        name: "enabled",
        type: "boolean",
        default: true,
      }),
    ).toBe(false);
    expect(
      fieldValue(form, "decode", {
        name: "threshold",
        type: "number",
        default: 10,
      }),
    ).toBe(0);
    expect(fieldValue(form, "activity", field)).toBe(50);
  });

  it("uses the real script fallback for missing JSON fields and shared form values where applicable", () => {
    const form = initialRun(schema, {
      settings: {},
      n_jobs: 7,
      max_sessions_to_run: 4,
    });
    expect(
      fieldValue(form, "decode", {
        name: "grid_search_for_c",
        type: "boolean",
        default: false,
      }),
    ).toBe(false);
    expect(
      fieldValue(form, "decode", {
        name: "n_jobs_session",
        type: "integer",
        default: 1,
      }),
    ).toBe(7);
    expect(
      fieldValue(form, "decode", {
        name: "max_sessions_to_run",
        type: "integer",
        default: null,
      }),
    ).toBe(4);
  });

  it("displays Python enum names as the corresponding option without silently selecting a different value", () => {
    const option: Field = {
      name: "model",
      type: "string",
      default: "svm",
      choices: ["svm", "logistic_regression"],
    };
    expect(choiceValue(option, "LOGISTIC_REGRESSION")).toBe(
      "logistic_regression",
    );
    expect(choiceValue(option, "unknown")).toBe("unknown");
    expect(choiceValue(option, null)).toBe("");
    expect(nonNullValue({ ...field, default: null })).toBe(1);
  });

  it("rejects malformed stage JSON and preserves unknown names for server validation", () => {
    for (const text of [
      "{",
      "null",
      "[]",
      '{"decode":null}',
      '{"decode":[]}',
    ]) {
      expect(() => parseSettings(text)).toThrow();
    }
    expect(parseSettings('{"decode":{"misspelled":false}}')).toEqual({
      decode: { misspelled: false },
    });
  });

  it("labels custom settings and reset seeds accurately rather than calling every form the example", () => {
    expect(presetName(schema.presets.example, schema)).toBe("example");
    expect(presetName(schema.presets.smoke, schema)).toBe("smoke");
    expect(
      presetName({ decode: { grid_search_for_c: true, seed: 7 } }, schema),
    ).toBe("custom");
  });

  it("finds parameters by natural words, CLI names, and descriptions", () => {
    const searchable: Field = {
      name: "preserve_null_time_structure",
      type: "boolean",
      default: false,
      description: "Reuse label permutations across time bins.",
    };
    for (const query of [
      "null time",
      "preserve_null",
      " time-structure ",
      "PERMUTATIONS bins",
      "",
    ]) {
      expect(parameterMatchesQuery(searchable, query)).toBe(true);
    }
    expect(parameterMatchesQuery(searchable, "null accuracy")).toBe(false);
  });

  it("compares example values without mislabeling enum spelling or object key order as changes", () => {
    const option: Field = {
      name: "model",
      type: "string",
      default: "svm",
      choices: ["svm", "logistic_regression"],
    };
    expect(
      sameFieldValue(option, "LOGISTIC_REGRESSION", "logistic_regression"),
    ).toBe(true);
    expect(sameFieldValue(option, "svm", "logistic_regression")).toBe(false);
    expect(sameFieldValue(option, null, "")).toBe(false);
    expect(sameFieldValue(field, null, 0)).toBe(false);
    expect(sameFieldValue(field, false, 0)).toBe(false);
    expect(sameFieldValue(field, { a: 1, b: false }, { b: false, a: 1 })).toBe(
      true,
    );
    expect(sameFieldValue(field, [1, 2], [2, 1])).toBe(false);
  });
});

describe("reuse manifest settings", () => {
  it("copies only run request fields and allows a new output directory", () => {
    const manifest: Manifest = {
      id: "invocation",
      status: "failed",
      stages: [{ stage: "select", status: "failed" }],
      runner_config: {
        cache_dir: "cache/old",
        data_dir: "data/example",
        settings: "configs/old.json",
        stages: ["all"],
        n_jobs: 2,
        session_list_file: null,
        max_sessions_to_run: null,
        figure_formats: ["png"],
        dry_run: false,
      },
      settings: {
        select: {
          cache_dir: "cache/old",
          data_dir: "data/example",
          n_jobs: 2,
          session_list_file: null,
          max_sessions_to_run: null,
          check_presence_ratio: true,
        },
        decode: {
          cache_dir: "cache/old",
          data_dir: "data/example",
          n_jobs: 3,
          seed: 42,
        },
      },
    };
    const seed = manifestSeed(manifest, "Prior analysis");
    expect(seed).not.toHaveProperty("cache_dir");
    expect(seed).not.toHaveProperty("dry_run");
    expect(seed.settings).toEqual({
      select: { check_presence_ratio: true },
      decode: { n_jobs: 3, seed: 42 },
    });
    expect(seed.stages).toEqual(["select", "decode"]);
    expect(seed.data_dir).toBe("data/example");
    expect(seed.allow_existing).toBe(false);
    expect(manifest.settings?.select.cache_dir).toBe("cache/old");
    expect(initialRun(schema, seed, "cache/copied").cache_dir).toBe(
      "cache/copied",
    );
  });
});
