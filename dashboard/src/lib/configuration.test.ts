import { afterEach, describe, expect, it, vi } from "vitest";
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
  updateRunDraft,
  launchRun,
  newCacheDirectory,
} from "./configuration";
import type { Field, Job, Manifest, Schema } from "./types";
import { api } from "./api";

vi.mock("./api", () => ({ api: vi.fn() }));

afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

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
  it("carries an independent recorded template baseline when copying a run", () => {
    const manifest: Manifest = {
      id: "previous",
      status: "complete",
      stages: [],
      settings: { decode: { grid_search_for_c: false } },
      source_template: {
        id: "saved-baseline",
        name: "Original template",
        description: "",
        builtin: false,
        config: {
          settings: { decode: { grid_search_for_c: true } },
          stages: ["decode"],
          n_jobs: 1,
          max_sessions_to_run: null,
          figure_formats: ["png"],
        },
      },
    };
    const copied = manifestSeed(manifest, "A previous run");
    expect(copied.source_template).toEqual(manifest.source_template);
    copied.source_template!.config.settings.decode.grid_search_for_c = false;
    expect(
      manifest.source_template!.config.settings.decode.grid_search_for_c,
    ).toBe(true);
    expect(
      manifestSeed({ ...manifest, source_template: undefined }, "Old")
        .source_template,
    ).toBeNull();
  });
  it("creates cache directories on HTTP without crypto.randomUUID", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-09-27T12:34:56.007Z"));
    const getRandomValues = vi
      .fn()
      .mockImplementationOnce((bytes: Uint8Array) => {
        bytes.set([0, 1, 16, 255]);
        return bytes;
      })
      .mockImplementationOnce((bytes: Uint8Array) => {
        bytes.set([16, 32, 48, 64]);
        return bytes;
      });
    vi.stubGlobal("crypto", { getRandomValues });
    expect(newCacheDirectory()).toBe(
      "cache/dashboard_20260927T123456007Z_000110ff",
    );
    expect(initialRun(schema).cache_dir).toBe(
      "cache/dashboard_20260927T123456007Z_10203040",
    );
    expect(getRandomValues).toHaveBeenCalledTimes(2);
    expect(getRandomValues.mock.calls[0][0]).toHaveLength(4);
  });

  it("submits enabled trust before consuming consent after an accepted launch", async () => {
    const form = {
      ...initialRun(schema, { allow_existing: true }),
      trust_unverified_legacy_results: true,
    };
    let draft = form;
    let resolve!: (job: Job) => void;
    const pending = new Promise<Job>((done) => {
      resolve = done;
    });
    vi.mocked(api).mockReturnValueOnce(pending);
    const accepted = vi.fn(() => {
      draft = { ...draft, trust_unverified_legacy_results: false };
    });
    const launch = launchRun(form, accepted);
    expect(
      JSON.parse(vi.mocked(api).mock.calls.at(-1)![1]!.body as string)
        .trust_unverified_legacy_results,
    ).toBe(true);
    expect(accepted).not.toHaveBeenCalled();
    draft = { ...draft, name: "Edited while starting" };
    const job = {
      id: "accepted-job",
      trust_unverified_legacy_results: true,
    } as Job;
    resolve(job);
    expect(await launch).toBe(job);
    expect(accepted).toHaveBeenCalledOnce();
    expect(draft.trust_unverified_legacy_results).toBe(false);
    expect(draft.name).toBe("Edited while starting");
    expect(form.trust_unverified_legacy_results).toBe(true);
  });

  it("retains explicit trust when a launch fails so the user can correct the request", async () => {
    const form = {
      ...initialRun(schema, { allow_existing: true }),
      trust_unverified_legacy_results: true,
    };
    vi.mocked(api).mockRejectedValueOnce(new Error("Missing required input"));
    const accepted = vi.fn();
    await expect(launchRun(form, accepted)).rejects.toThrow(
      "Missing required input",
    );
    expect(accepted).not.toHaveBeenCalled();
    expect(form.trust_unverified_legacy_results).toBe(true);
  });
  it("never inherits legacy trust from backend defaults or a copied seed", () => {
    const trusted = {
      ...schema,
      defaults: { ...schema.defaults, trust_unverified_legacy_results: true },
    };
    expect(
      initialRun(trusted, {
        allow_existing: true,
        trust_unverified_legacy_results: true,
      }).trust_unverified_legacy_results,
    ).toBe(false);
    const seed = manifestSeed(
      {
        id: "old",
        status: "complete",
        stages: [],
        runner_config: { trust_unverified_legacy_results: true },
        settings: { decode: { trust_unverified_legacy_results: true } },
      },
      "Old run",
    );
    expect(seed.trust_unverified_legacy_results).toBe(false);
    expect(seed.settings?.decode).not.toHaveProperty(
      "trust_unverified_legacy_results",
    );
  });

  it("scopes explicit legacy trust to the current input and reusable output directory", () => {
    const initial = initialRun(schema, { allow_existing: true });
    const trusted = updateRunDraft(initial, {
      trust_unverified_legacy_results: true,
    });
    expect(trusted.trust_unverified_legacy_results).toBe(true);
    expect(
      updateRunDraft(trusted, { n_jobs: 4 }).trust_unverified_legacy_results,
    ).toBe(true);
    for (const change of [
      { cache_dir: "cache/another" },
      { data_dir: "data/other" },
      { allow_existing: false },
    ])
      expect(
        updateRunDraft(trusted, change).trust_unverified_legacy_results,
      ).toBe(false);
    expect(
      updateRunDraft(initialRun(schema), {
        trust_unverified_legacy_results: true,
      }).trust_unverified_legacy_results,
    ).toBe(false);
  });
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
        figure_font: "DejaVu Serif",
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
    expect(seed.figure_font).toBe("DejaVu Serif");
    expect(seed.allow_existing).toBe(false);
    expect(manifest.settings?.select.cache_dir).toBe("cache/old");
    expect(initialRun(schema, seed, "cache/copied").cache_dir).toBe(
      "cache/copied",
    );
  });
});
