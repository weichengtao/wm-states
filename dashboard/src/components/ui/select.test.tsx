import type { ComponentProps } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { Select } from "./select";

function renderTrigger(props: Partial<ComponentProps<typeof Select>> = {}) {
  const html = renderToStaticMarkup(
    <Select
      value="example"
      onValueChange={vi.fn()}
      options={[{ value: "example", label: "Example pipeline" }]}
      placeholder="Choose a preset"
      {...props}
    />,
  );
  const trigger = html.match(/<button\b[\s\S]*?<\/button>/)?.[0];
  expect(trigger).toBeDefined();
  return trigger!;
}

describe("shared select", () => {
  it("displays an explicitly selected empty-string option instead of its placeholder", () => {
    const trigger = renderTrigger({
      value: "",
      options: [
        { value: "", label: "None / automatic" },
        { value: "sigmoid", label: "Sigmoid" },
      ],
      placeholder: "Choose calibration",
    });
    expect(trigger).toContain(">None / automatic<");
    expect(trigger).not.toContain("Choose calibration");
    expect(trigger).not.toContain("data-placeholder");
    expect(trigger).not.toContain('disabled=""');
  });

  it("keeps empty, prefix-like, and punctuation-containing values distinct", () => {
    const options = [
      { value: "", label: "No override" },
      { value: "option:", label: "Literal prefix" },
      { value: "option:option:", label: "Repeated prefix" },
      {
        value: "figures/a:b/with spaces?cue=3&format=pdf#result",
        label: "Figure with punctuation",
      },
    ];
    for (const selected of options) {
      const trigger = renderTrigger({ value: selected.value, options });
      expect(trigger).toContain(`>${selected.label}<`);
      expect(trigger).not.toContain("data-placeholder");
      for (const other of options.filter((option) => option !== selected)) {
        expect(trigger).not.toContain(`>${other.label}<`);
      }
    }
  });

  it("forwards the trigger's accessible name, description, and label association", () => {
    const trigger = renderTrigger({
      id: "calibration-method",
      "aria-label": "Probability calibration",
      "aria-describedby": "calibration-help",
      options: [
        {
          value: "example",
          label: "Example pipeline",
          description: "Uses sigmoid calibration",
        },
      ],
    });
    expect(trigger).toContain('role="combobox"');
    expect(trigger).toContain('id="calibration-method"');
    expect(trigger).toContain('aria-label="Probability calibration"');
    expect(trigger).toContain('aria-describedby="calibration-help"');
    expect(trigger).toContain(
      'title="Example pipeline · Uses sigmoid calibration"',
    );
  });

  it("disables a selector with no options while explaining its empty state", () => {
    const trigger = renderTrigger({
      value: "",
      options: [],
      placeholder: "No sessions yet",
    });
    expect(trigger).toContain('disabled=""');
    expect(trigger).toContain(">No sessions yet<");
  });

  it("disables a selector when every option is unavailable", () => {
    const trigger = renderTrigger({
      options: [{ value: "example", label: "Unavailable run", disabled: true }],
    });
    expect(trigger).toContain('disabled=""');
    expect(trigger).toContain(">Unavailable run<");
  });

  it("preserves an explicit disabled state even when options are available", () => {
    expect(renderTrigger({ disabled: true })).toContain('disabled=""');
  });

  it("shows the placeholder for a stale value without silently selecting an option", () => {
    const onValueChange = vi.fn();
    const trigger = renderTrigger({ value: "removed-preset", onValueChange });
    expect(trigger).toContain(">Choose a preset<");
    expect(trigger).toContain("data-placeholder");
    expect(trigger).not.toContain("Example pipeline");
    expect(trigger).not.toContain('disabled=""');
    expect(onValueChange).not.toHaveBeenCalled();
  });
});
