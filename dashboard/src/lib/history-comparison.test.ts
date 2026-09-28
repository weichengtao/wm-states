import { describe, expect, it } from "vitest";
import type { Manifest, PipelineTemplate, Schema, Settings } from "./types";
import {
  filterHistorySettings,
  historicalStages,
  historySettings,
  historyValue,
} from "./history-comparison";

const schema: Schema = {
  defaults: {},
  presets: { example: {}, smoke: {} },
  stages: [
    {
      id: "decode",
      label: "Decode confidence",
      description: "",
      fields: [
        {
          name: "method",
          type: "string",
          default: "logistic",
          choices: ["logistic", "svm"],
        },
        { name: "new_default", type: "number", default: 999 },
      ],
    },
  ],
};
function template(
  settings: Settings = { decode: { n_decode_shuffle: 100 } },
): PipelineTemplate {
  return {
    id: "example",
    name: "Example pipeline",
    builtin: true,
    description: "",
    config: {
      settings,
      stages: ["decode"],
      n_jobs: 2,
      max_sessions_to_run: null,
      figure_formats: ["png", "pdf"],
      figure_font: "DejaVu Sans",
    },
  };
}
function manifest(patch: Partial<Manifest> = {}): Manifest {
  return {
    id: "run",
    status: "complete",
    stages: [{ stage: "decode", status: "complete" }],
    runner_config: {
      stages: ["decode"],
      n_jobs: 2,
      max_sessions_to_run: null,
      figure_formats: ["png", "pdf"],
      figure_font: "DejaVu Sans",
    },
    settings: { decode: { n_decode_shuffle: 100 } },
    ...patch,
  };
}

describe("historical template comparison", () => {
  it("compares old balancing booleans with the equivalent mode without filling missing history", () => {
    const actual = manifest({
      settings: { decode: { balance_decoder_training_trials: false } },
    });
    const baseline = template({ decode: { training_balance: "none" } });
    expect(
      historySettings(actual, baseline, schema).find(
        (row) => row.field === "training_balance",
      )?.status,
    ).toBe("same");
    expect(actual.settings?.decode.balance_decoder_training_trials).toBe(false);
    const missing = manifest({ settings: { decode: {} } });
    expect(
      historySettings(missing, baseline, schema).find(
        (row) => row.field === "training_balance",
      )?.status,
    ).toBe("unavailable");
    expect(
      historySettings(
        actual,
        template({ decode: { training_balance: "balanced_class_weights" } }),
        schema,
      ).find((row) => row.field === "training_balance")?.status,
    ).toBe("changed");
  });
  it("compares only stages selected by this invocation and excludes run identity", () => {
    const source = template();
    source.config.stages = ["select", "decode"];
    source.config.settings.select = { check_presence_ratio: true };
    const actual = manifest({
      settings: {
        decode: { n_decode_shuffle: 3, cache_dir: "/run", data_dir: "/data" },
        select: { check_presence_ratio: false },
      },
    });
    const rows = historySettings(actual, source, schema);
    expect(
      rows.filter((row) => row.status === "changed").map((row) => row.field),
    ).toEqual(["stages", "n_decode_shuffle"]);
    expect(
      rows.some(
        (row) =>
          row.stage === "select" ||
          row.field === "cache_dir" ||
          row.field === "data_dir",
      ),
    ).toBe(false);
  });
  it("never supplies current defaults for missing historical values", () => {
    const source = template();
    source.config.settings.decode.new_default = 5;
    const rows = historySettings(manifest(), source, schema);
    expect(rows.find((row) => row.field === "new_default")).toMatchObject({
      before: 5,
      after: undefined,
      status: "unavailable",
    });
    expect(
      historySettings(manifest(), template(), schema).some(
        (row) => row.field === "new_default",
      ),
    ).toBe(false);
  });
  it("keeps absent template values unavailable rather than borrowing current defaults", () => {
    const rows = historySettings(
      manifest({
        settings: { decode: { n_decode_shuffle: 100, old_field: 7 } },
      }),
      template(),
      schema,
    );
    expect(rows.find((row) => row.field === "old_field")).toMatchObject({
      before: undefined,
      after: 7,
      status: "unavailable",
    });
  });
  it("compares explicit nulls and false values as values", () => {
    const source = template();
    source.config.settings.decode.optional = null;
    const actual = manifest({
      settings: { decode: { optional: false, n_decode_shuffle: 100 } },
    });
    expect(
      historySettings(actual, source, schema).find(
        (row) => row.field === "optional",
      )?.status,
    ).toBe("changed");
    expect(historyValue(null, "recorded")).toBe("None");
    expect(historyValue(undefined, "recorded")).toBe("Not recorded");
    expect(historyValue(undefined, "template")).toBe("Not in template");
  });
  it("uses semantic enums/object equality and ignores export-format ordering", () => {
    const source = template();
    Object.assign(source.config.settings.decode, {
      method: "logistic",
      diagnostics: { a: true, b: 2 },
    });
    const actual = manifest({
      original_request: { figure_formats: ["pdf", "png"] },
      settings: {
        decode: {
          n_decode_shuffle: 100,
          method: "LOGISTIC",
          diagnostics: { b: 2, a: true },
        },
      },
    });
    expect(
      historySettings(actual, source, schema).every(
        (row) => row.status === "same",
      ),
    ).toBe(true);
  });
  it("uses resolved stage keys when CLI requested all or mixed aliases", () => {
    for (const alias of ["all", "mixed"])
      expect(
        historicalStages(
          manifest({
            runner_config: { stages: [alias] },
            settings: { decode: {}, states: {} },
          }),
        ),
      ).toEqual(["decode", "states"]);
    expect(historicalStages(manifest({ runner_config: undefined }))).toEqual([
      "decode",
    ]);
    expect(
      historicalStages(
        manifest({ runner_config: undefined, settings: undefined }),
      ),
    ).toEqual(["decode"]);
  });
  it("prefers captured request paths and compares only paths captured by the template", () => {
    const source = template();
    source.config.data_dir = "data/original";
    source.config.session_list_file = "configs/list.json";
    const actual = manifest({
      invocation: { cwd: "/repo" },
      original_request: {
        data_dir: "data/other",
        session_list_file: "configs/list.json",
      },
      runner_config: {
        data_dir: "/repo/data/other",
        session_list_file: "/repo/configs/list.json",
      },
    });
    const rows = historySettings(actual, source, schema);
    expect(rows.find((row) => row.field === "data_dir")).toMatchObject({
      after: "data/other",
      status: "changed",
    });
    expect(rows.find((row) => row.field === "session_list_file")?.status).toBe(
      "same",
    );
    expect(
      historySettings(actual, template(), schema).some(
        (row) => row.field === "data_dir" || row.field === "session_list_file",
      ),
    ).toBe(false);
  });
  it("resolves recorded absolute paths against known invocation cwd without guessing unknown bases", () => {
    const source = template();
    source.config.settings.decode.session_list_file = "configs/list.json";
    const actual = manifest({
      invocation: { cwd: "/repo" },
      settings: { decode: { session_list_file: "/repo/configs/list.json" } },
    });
    expect(
      historySettings(actual, source, schema).find(
        (row) => row.field === "session_list_file",
      )?.status,
    ).toBe("same");
    delete actual.invocation;
    expect(
      historySettings(actual, source, schema).find(
        (row) => row.field === "session_list_file",
      )?.status,
    ).toBe("changed");
  });
  it("deduplicates runner worker settings but preserves actual stage overrides", () => {
    const actual = manifest({
      settings: { decode: { n_decode_shuffle: 100, n_jobs: 2, cv_n_jobs: 4 } },
    });
    const rows = historySettings(actual, template(), schema);
    expect(rows.filter((row) => row.field === "n_jobs")).toHaveLength(1);
    expect(rows.find((row) => row.field === "cv_n_jobs")).toMatchObject({
      before: 2,
      after: 4,
      status: "changed",
    });
  });
  it("filters changes/search without mutating records or templates", () => {
    const source = template();
    const actual = manifest({ settings: { decode: { n_decode_shuffle: 3 } } });
    const before = JSON.stringify({ source, actual });
    const rows = historySettings(actual, source, schema);
    expect(
      filterHistorySettings(rows, true, "decode shuffle").map(
        (row) => row.field,
      ),
    ).toEqual(["n_decode_shuffle"]);
    expect(
      filterHistorySettings(rows, false, "DejaVu").map((row) => row.field),
    ).toEqual(["figure_font"]);
    expect(JSON.stringify({ source, actual })).toBe(before);
  });
});
