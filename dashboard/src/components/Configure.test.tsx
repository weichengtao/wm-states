import { describe, expect, it, vi } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import type { RunRequest, Schema } from "@/lib/types";
import Configure from "./Configure";

const schema: Schema = {
  stages: [
    {
      id: "select",
      label: "Cell screening",
      description: "Choose screening checks.",
      fields: [
        { name: "check_presence_ratio", type: "boolean", default: true },
        { name: "min_presence_ratio", type: "number", default: 0.9 },
      ],
    },
    {
      id: "decode",
      label: "Decode confidence",
      description: "Fit the observed and shuffled estimates.",
      fields: [{ name: "n_decode_shuffle", type: "integer", default: 100 }],
    },
  ],
  defaults: {
    stages: ["select", "decode"],
    n_jobs: 2,
    max_sessions_to_run: null,
    figure_formats: ["png"],
  },
  presets: {
    example: {
      select: { check_presence_ratio: true, min_presence_ratio: 0.9 },
      decode: { n_decode_shuffle: 100 },
    },
    smoke: {
      select: { check_presence_ratio: false, min_presence_ratio: 0.7 },
      decode: { n_decode_shuffle: 3 },
    },
  },
};

function renderConfigure({
  running = false,
  seed,
}: {
  running?: boolean;
  seed?: Partial<RunRequest>;
} = {}) {
  return renderToStaticMarkup(
    <Configure
      schema={schema}
      running={running}
      seed={seed}
      onStarted={vi.fn()}
      onBack={vi.fn()}
    />,
  );
}

const visibleText = (html: string) =>
  html
    .replace(/<[^>]*>/g, " ")
    .replace(/\s+/g, " ")
    .trim();

function buttonsNamed(html: string, label: string) {
  return [...html.matchAll(/<button\b[^>]*>[\s\S]*?<\/button>/g)]
    .map((match) => match[0])
    .filter((button) => visibleText(button) === label);
}

const disabledAttribute = /\sdisabled(?:=|\s|>)/;

describe("configuration templates", () => {
  it("keeps template saving and editing available while another job blocks both launch buttons", () => {
    const html = renderConfigure({ running: true });
    expect(visibleText(html)).toContain("An analysis is running.");
    const launchButtons = buttonsNamed(html, "Start pipeline");
    expect(launchButtons).toHaveLength(2);
    for (const button of launchButtons)
      expect(button).toMatch(disabledAttribute);

    const saveButtons = buttonsNamed(html, "Save as template");
    expect(saveButtons).toHaveLength(1);
    expect(saveButtons[0]).not.toMatch(disabledAttribute);
    const templateSelector = html.match(
      /<button\b[^>]*aria-label="Analysis template"[^>]*>/,
    )?.[0];
    expect(templateSelector).toBeDefined();
    expect(templateSelector).not.toMatch(disabledAttribute);
    for (const id of ["workers", "field-select-min_presence_ratio"]) {
      const input = html.match(
        new RegExp(`<input\\b[^>]*id="${id}"[^>]*>`),
      )?.[0];
      expect(input).toBeDefined();
      expect(input).not.toMatch(disabledAttribute);
    }
    expect(buttonsNamed(html, "JSON")[0]).not.toMatch(disabledAttribute);
  });

  it("uses a Smoke seed as the comparison baseline even when it differs from Example", () => {
    const html = renderConfigure({ seed: { settings: schema.presets.smoke } });
    expect(visibleText(html)).toContain("Matches Smoke test");
    expect(visibleText(html)).toContain("Changed (0)");
    expect(html).not.toContain("parameter-changed");
    expect(html).not.toContain("template-modified");
    expect(buttonsNamed(html, "Restore template")).toHaveLength(0);
    const threshold = html.match(
      /<input\b[^>]*id="field-select-min_presence_ratio"[^>]*>/,
    )?.[0];
    expect(threshold).toContain('value="0.7"');
    expect(buttonsNamed(html, "Review changes")[0]).toMatch(disabledAttribute);
  });

  it("shows modified stage and shared values against the selected template", () => {
    const html = renderConfigure({
      seed: {
        n_jobs: 4,
        settings: {
          ...schema.presets.example,
          select: {
            ...schema.presets.example.select,
            min_presence_ratio: 0.65,
          },
        },
      },
    });
    expect(visibleText(html)).toContain("2 changes from Example pipeline");
    expect(visibleText(html)).toContain("Changed (1)");
    expect(html).toContain('title="Example pipeline: 0.9"');
    expect(html).toContain(
      'aria-label="Reset Min Presence Ratio to template value"',
    );
    const difference = html.match(
      /<span class="parameter-difference"[^>]*>[\s\S]*?<\/code>\s*<\/span>/,
    )?.[0];
    expect(difference).toContain("<code>0.9</code>");
    expect(difference).toContain("<code>0.65</code>");
    expect(difference!.indexOf("<code>0.9</code>")).toBeLessThan(
      difference!.indexOf("<code>0.65</code>"),
    );
    expect(visibleText(html)).toContain("Template: 2");
    expect(buttonsNamed(html, "Restore template")).toHaveLength(1);
    expect(buttonsNamed(html, "Review changes")[0]).not.toMatch(
      disabledAttribute,
    );
  });
});
