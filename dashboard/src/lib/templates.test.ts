import { describe, expect, it } from "vitest";
import { initialRun } from "./configuration";
import {
  builtInTemplates,
  formatTemplateValue,
  templateChanges,
  templateConfig,
  templateRun,
} from "./templates";
import type { PipelineTemplate, RunRequest, Schema } from "./types";

const schema: Schema = {
  stages: [
    {
      id: "select",
      label: "Cell screening",
      description: "",
      fields: [
        { name: "check_presence_ratio", type: "boolean", default: true },
        { name: "min_presence_ratio", type: "number", default: 0.9 },
      ],
    },
    {
      id: "decode",
      label: "Decode",
      description: "",
      fields: [
        { name: "n_decode_shuffle", type: "integer", default: 50 },
        {
          name: "model",
          type: "string",
          choices: ["logistic_regression", "svm"],
          default: "logistic_regression",
        },
        { name: "max_points", type: "integer", nullable: true, default: null },
        { name: "grid_search_for_c", type: "boolean", default: true },
        { name: "time_bins", type: "array", default: [0, 50, 100] },
      ],
    },
    {
      id: "models",
      label: "Models",
      description: "",
      fields: [{ name: "history_alpha", type: "number", default: 0.2 }],
    },
  ],
  defaults: {
    stages: ["select", "decode"],
    n_jobs: 2,
    max_sessions_to_run: null,
    figure_formats: ["png"],
    data_dir: "data/default",
    session_list_file: "configs/default-sessions.json",
  },
  presets: {
    example: { decode: { n_decode_shuffle: 100 } },
    smoke: { decode: { n_decode_shuffle: 3, grid_search_for_c: false } },
  },
};

function draft(overrides: Partial<RunRequest> = {}) {
  return initialRun(schema, overrides, "cache/my-new-run");
}

function saved(form: RunRequest, includePaths = false): PipelineTemplate {
  return {
    id: "saved-analysis",
    name: "My analysis",
    description: "",
    builtin: false,
    config: templateConfig(form, schema, includePaths),
  };
}

describe("pipeline templates", () => {
  it("offers Example and Smoke baselines with resolved settings and no recording paths", () => {
    const templates = builtInTemplates(schema);
    expect(templates.map((template) => template.id)).toEqual([
      "example",
      "smoke",
    ]);
    const smoke = templates[1];
    expect(smoke.config.settings.decode.n_decode_shuffle).toBe(3);
    expect(smoke.config.settings.decode.max_points).toBeNull();
    expect(smoke.config.settings.models.history_alpha).toBe(0.2);
    expect(smoke.config.stages).toEqual(["select", "decode"]);
    expect(smoke.config).not.toHaveProperty("data_dir");
    expect(smoke.config).not.toHaveProperty("session_list_file");
    const form = templateRun(draft(), smoke);
    expect(templateChanges(form, smoke, schema)).toEqual([]);
    expect(
      templateChanges(form, templates[0], schema).map((change) => change.field),
    ).toEqual(["n_decode_shuffle", "grid_search_for_c"]);
  });

  it("freezes omitted stage defaults, including stages not currently selected", () => {
    const form = draft();
    const template = saved(form);
    expect(template.config.settings.select.min_presence_ratio).toBe(0.9);
    expect(template.config.settings.models.history_alpha).toBe(0.2);
    const futureSchema = structuredClone(schema);
    futureSchema.stages[2].fields[0].default = 0.7;
    expect(
      templateConfig(templateRun(form, template), futureSchema, false).settings
        .models.history_alpha,
    ).toBe(0.2);
    expect(template.config).not.toHaveProperty("name");
    expect(template.config).not.toHaveProperty("cache_dir");
    expect(template.config).not.toHaveProperty("allow_existing");
  });

  it("retains explicit hidden worker overrides and unknown JSON for server validation", () => {
    const form = draft({
      settings: {
        select: { n_jobs_session: 4, max_sessions_to_run: 6 },
        decode: {
          n_jobs: -1,
          typo_parameter: false,
          cache_dir: "cache/invalid",
          data_dir: "data/invalid",
        },
        models: { cv_n_jobs: 3 },
        misspelled_stage: { wrong: 1 },
      },
    });
    const config = templateConfig(form, schema, false);
    expect(config.settings.select.n_jobs_session).toBe(4);
    expect(config.settings.select.max_sessions_to_run).toBe(6);
    expect(config.settings.decode.n_jobs).toBe(-1);
    expect(config.settings.models.cv_n_jobs).toBe(3);
    expect(config.settings.decode.typo_parameter).toBe(false);
    expect(config.settings.decode.cache_dir).toBe("cache/invalid");
    expect(config.settings.decode.data_dir).toBe("data/invalid");
    expect(config.settings.misspelled_stage).toEqual({ wrong: 1 });
    const inherited = templateConfig(draft(), schema, false);
    expect(inherited.settings.decode).not.toHaveProperty("n_jobs");
    expect(inherited.settings.models).not.toHaveProperty("cv_n_jobs");
  });

  it("keeps figure formats in the runner config instead of adding unsupported stage fields", () => {
    const config = templateConfig(
      draft({ figure_formats: ["png", "pdf"] }),
      schema,
      false,
    );
    expect(config.figure_formats).toEqual(["png", "pdf"]);
    for (const settings of Object.values(config.settings))
      expect(settings).not.toHaveProperty("figure_formats");
    // An invalid explicit override remains visible to backend validation.
    const invalid = templateConfig(
      draft({ settings: { decode: { figure_formats: ["pdf"] } } }),
      schema,
      false,
    );
    expect(invalid.settings.decode.figure_formats).toEqual(["pdf"]);
  });

  it("preserves run identity and every recording selection when a pathless template is applied", () => {
    const template = builtInTemplates(schema)[1];
    const form = draft({
      name: "My run",
      cache_dir: "cache/keep",
      allow_existing: true,
      data_dir: "/recordings/local",
      session_list_file: "/lists/all.json",
      settings: {
        decode: {
          session_list_file: "/lists/decode.json",
          n_decode_shuffle: 88,
        },
      },
    });
    const applied = templateRun(form, template);
    expect(applied).toMatchObject({
      name: "My run",
      cache_dir: "cache/keep",
      allow_existing: true,
      data_dir: "/recordings/local",
      session_list_file: "/lists/all.json",
      settings: {
        decode: {
          session_list_file: "/lists/decode.json",
          n_decode_shuffle: 3,
        },
      },
    });
    expect(templateChanges(applied, template, schema)).toEqual([]);
  });

  it("excludes shared and per-stage allowlist paths when saving without paths", () => {
    const form = draft({
      settings: { decode: { session_list_file: "/lists/decode.json" } },
    });
    const template = saved(form);
    expect(template.config).not.toHaveProperty("data_dir");
    expect(template.config).not.toHaveProperty("session_list_file");
    expect(template.config.settings.decode).not.toHaveProperty(
      "session_list_file",
    );
    expect(form.settings.decode.session_list_file).toBe("/lists/decode.json");
    form.settings.decode.session_list_file = null;
    expect(templateChanges(form, template, schema)).toEqual([]);
  });

  it("preserves allowlists when a template captures only its recording directory", () => {
    const form = draft({
      data_dir: "/local/data",
      session_list_file: "/local/all.json",
      settings: {
        decode: {
          n_decode_shuffle: 100,
          session_list_file: "/local/decode.json",
        },
      },
    });
    const template = saved(draft());
    template.config.data_dir = "/shared/data";
    const applied = templateRun(form, template);
    expect(applied.data_dir).toBe("/shared/data");
    expect(applied.session_list_file).toBe("/local/all.json");
    expect(applied.settings.decode.session_list_file).toBe(
      "/local/decode.json",
    );
    expect(templateChanges(form, template, schema)).toEqual([
      {
        stage: "shared",
        field: "data_dir",
        label: "Recording directory",
        before: "/shared/data",
        after: "/local/data",
      },
    ]);
    expect(templateChanges(applied, template, schema)).toEqual([]);
  });

  it("applies and compares captured recording paths, including explicit null allowlists", () => {
    const template = saved(
      draft({
        data_dir: "/shared/data",
        session_list_file: null,
        settings: { decode: { session_list_file: "/shared/subset.json" } },
      }),
      true,
    );
    const form = draft({
      data_dir: "/local/data",
      session_list_file: "/local/list.json",
    });
    const applied = templateRun(form, template);
    expect(applied.data_dir).toBe("/shared/data");
    expect(applied.session_list_file).toBeNull();
    expect(applied.settings.decode.session_list_file).toBe(
      "/shared/subset.json",
    );
    const changes = templateChanges(form, template, schema);
    expect(changes).toContainEqual({
      stage: "shared",
      field: "session_list_file",
      label: "Session list file",
      before: null,
      after: "/local/list.json",
    });
    expect(changes).toContainEqual({
      stage: "decode",
      field: "session_list_file",
      label: "Session List File",
      before: "/shared/subset.json",
      after: "/local/list.json",
    });
  });

  it("treats omitted defaults and enum spelling as equivalent without conflating null, false, and zero", () => {
    const template = saved(draft());
    const form = draft({
      settings: {
        decode: { n_decode_shuffle: 100, model: "LOGISTIC_REGRESSION" },
      },
    });
    expect(templateChanges(form, template, schema)).toEqual([]);
    form.settings.decode.max_points = 0;
    form.settings.decode.grid_search_for_c = false;
    expect(templateChanges(form, template, schema)).toEqual([
      {
        stage: "decode",
        field: "max_points",
        label: "Max Points",
        before: null,
        after: 0,
      },
      {
        stage: "decode",
        field: "grid_search_for_c",
        label: "Grid Search For C",
        before: true,
        after: false,
      },
    ]);
  });

  it("reports selected-stage and shared choices without reporting output identity", () => {
    const template = saved(draft());
    const form = draft({
      name: "Different",
      cache_dir: "cache/elsewhere",
      allow_existing: true,
      stages: ["select"],
      n_jobs: -1,
      max_sessions_to_run: 4,
      figure_formats: ["pdf"],
    });
    expect(
      templateChanges(form, template, schema).map((change) => [
        change.stage,
        change.field,
      ]),
    ).toEqual([
      ["shared", "stages"],
      ["shared", "n_jobs"],
      ["shared", "max_sessions_to_run"],
      ["shared", "figure_formats"],
    ]);
  });

  it("reports hidden worker overrides and draft-only unknown fields rather than losing them", () => {
    const template = saved(draft());
    const form = draft({
      settings: { decode: { n_decode_shuffle: 100, n_jobs: 7, typo: null } },
    });
    expect(templateChanges(form, template, schema)).toEqual([
      {
        stage: "decode",
        field: "n_jobs",
        label: "N Jobs",
        before: 2,
        after: 7,
      },
      {
        stage: "decode",
        field: "typo",
        label: "Typo",
        before: undefined,
        after: null,
      },
    ]);
  });

  it("reports an empty unknown settings stage instead of claiming a template match", () => {
    const template = saved(draft());
    const form = draft({
      settings: { decode: { n_decode_shuffle: 100 }, typo: {} },
    });
    expect(templateChanges(form, template, schema)).toEqual([
      {
        stage: "typo",
        field: "__stage__",
        label: "Unknown stage settings",
        before: undefined,
        after: {},
      },
    ]);
    expect(templateConfig(form, schema, false).settings.typo).toEqual({});
  });

  it("reports illegal stage output and recording overrides even for pathless templates", () => {
    const template = saved(draft());
    const form = draft({
      settings: {
        decode: {
          n_decode_shuffle: 100,
          cache_dir: "cache/illegal",
          data_dir: "data/illegal",
        },
      },
    });
    expect(templateChanges(form, template, schema)).toEqual([
      {
        stage: "decode",
        field: "cache_dir",
        label: "Cache Dir",
        before: undefined,
        after: "cache/illegal",
      },
      {
        stage: "decode",
        field: "data_dir",
        label: "Data Dir",
        before: undefined,
        after: "data/illegal",
      },
    ]);
  });

  it("compares figure formats without ordering differences while retaining invalid duplicates", () => {
    const template = saved(draft({ figure_formats: ["png", "pdf"] }));
    const reordered = draft({ figure_formats: ["pdf", "png"] });
    expect(templateChanges(reordered, template, schema)).toEqual([]);
    expect(reordered.figure_formats).toEqual(["pdf", "png"]);
    expect(template.config.figure_formats).toEqual(["png", "pdf"]);
    const duplicate = draft({ figure_formats: ["pdf", "png", "png"] });
    expect(templateChanges(duplicate, template, schema)).toEqual([
      {
        stage: "shared",
        field: "figure_formats",
        label: "Figure formats",
        before: ["png", "pdf"],
        after: ["pdf", "png", "png"],
      },
    ]);
  });

  it("never mutates the draft, template, or schema through returned values", () => {
    const form = draft();
    const original = structuredClone(form);
    const originalSchema = structuredClone(schema);
    const template = saved(form);
    const baseline = structuredClone(template);
    const applied = templateRun(form, template);
    applied.settings.decode.time_bins = [999];
    applied.stages.push("models");
    templateChanges(form, template, schema);
    expect(form).toEqual(original);
    expect(template).toEqual(baseline);
    expect(schema).toEqual(originalSchema);
  });

  it("formats missing, nullable, and boolean values without hiding their distinctions", () => {
    expect(formatTemplateValue(undefined)).toBe("Not set");
    expect(formatTemplateValue(null)).toBe("None");
    expect(formatTemplateValue(false)).toBe("Disabled");
    expect(formatTemplateValue(0)).toBe("0");
    expect(formatTemplateValue(["png", "pdf"])).toBe("png, pdf");
    expect(formatTemplateValue("")).toBe("Empty string");
  });
});
