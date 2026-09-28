import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { HistoryComparison } from "./HistoryComparison";
import type { Manifest, PipelineTemplate, Schema } from "@/lib/types";

const original: PipelineTemplate = {
  id: "saved",
  name: "Careful analysis",
  description: "",
  builtin: false,
  config: {
    settings: { decode: { n_decode_shuffle: 100 } },
    stages: ["decode"],
    n_jobs: 1,
    max_sessions_to_run: null,
    figure_formats: ["png"],
    figure_font: "DejaVu Sans",
  },
};
const manifest: Manifest = {
  id: "run",
  status: "complete",
  stages: [{ stage: "decode", status: "complete" }],
  source_template: original,
  settings: { decode: { n_decode_shuffle: 3 } },
  runner_config: {
    stages: ["decode"],
    n_jobs: 1,
    max_sessions_to_run: null,
    figure_formats: ["png"],
    figure_font: "DejaVu Sans",
  },
};

describe("Run history template comparison", () => {
  it("defaults to the original snapshot even if it is no longer in the template list", () => {
    const html = renderToStaticMarkup(
      <HistoryComparison manifest={manifest} templates={[]} />,
    );
    expect(html).toContain("Careful analysis · Original snapshot");
    expect(html).toContain("Original template snapshot");
    expect(html).toContain("1 change");
    expect(html).toContain("100");
    expect(html).toContain("Recorded run");
    expect(html).not.toContain("Original template not recorded");
  });
  it("labels the Example fallback honestly for older records", () => {
    const example = {
      ...original,
      id: "example",
      name: "Example pipeline",
      builtin: true,
    };
    const html = renderToStaticMarkup(
      <HistoryComparison
        manifest={{ ...manifest, source_template: null }}
        templates={[example]}
      />,
    );
    expect(html).toContain("Example pipeline · Current");
    expect(html).toContain("Original template not recorded");
    expect(html).toContain("not a claim about how the run was configured");
    expect(html).toContain("Changed (1)");
    expect(html).toContain("Search recorded settings");
  });
  it("explains unavailable records without inventing a comparison", () => {
    const html = renderToStaticMarkup(
      <HistoryComparison
        manifest={{ ...manifest, settings: {} }}
        templates={[]}
      />,
    );
    expect(html).toContain("1 setting cannot be compared");
    expect(html).toContain("Missing values are never filled with today");
    expect(html).toContain("No differences among comparable settings");
    expect(html).not.toContain("recorded values match");
  });
  it("resolves sparse current templates with current schema defaults only", () => {
    const schema: Schema = {
      defaults: {},
      presets: { example: {}, smoke: {} },
      stages: [
        {
          id: "decode",
          label: "Decode confidence",
          description: "",
          fields: [{ name: "n_decode_shuffle", type: "integer", default: 50 }],
        },
      ],
    };
    const current = {
      ...original,
      id: "example",
      config: { ...original.config, settings: { decode: {} } },
    };
    const html = renderToStaticMarkup(
      <HistoryComparison
        manifest={{ ...manifest, source_template: null }}
        templates={[current]}
        schema={schema}
      />,
    );
    expect(html).toContain("1 change");
    expect(html).toContain(">50</div>");
    const historic = renderToStaticMarkup(
      <HistoryComparison
        manifest={{ ...manifest, source_template: current }}
        templates={[]}
        schema={schema}
      />,
    );
    expect(historic).toContain("1 setting cannot be compared");
    expect(historic).not.toContain(">50</div>");
  });
});
