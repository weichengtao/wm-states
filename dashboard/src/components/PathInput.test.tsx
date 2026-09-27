import { describe, expect, it, vi } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import PathInput from "./PathInput";

describe("local path input", () => {
  it("provides a labelled combobox with discoverable keyboard help without fetching before focus", () => {
    const html = renderToStaticMarkup(
      <PathInput
        id="recordings"
        aria-label="Recording directory"
        value="data/"
        mode="directory"
        onValueChange={vi.fn()}
      />,
    );
    expect(html).toContain('role="combobox"');
    expect(html).toContain('aria-label="Recording directory"');
    expect(html).toContain('aria-expanded="false"');
    expect(html).toContain('value="data/"');
    expect(html).toContain("Tab completes a match");
    expect(html).not.toContain('role="listbox"');
  });
  it("keeps caller descriptions and custom path guidance", () => {
    const html = renderToStaticMarkup(
      <PathInput
        aria-describedby="cache-warning"
        completionHint="Use a folder beneath cache/."
        value=""
        onValueChange={vi.fn()}
      />,
    );
    expect(html).toContain('aria-describedby="cache-warning ');
    expect(html).toContain("Use a folder beneath cache/.");
  });
});
