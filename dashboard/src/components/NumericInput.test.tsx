import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import NumericInput from "./NumericInput";

describe("numeric text editing", () => {
  it("renders raw partial decimal, sign, exponent and blank drafts without browser number coercion", () => {
    for (const value of ["", "0", "0.", "0.0", "0.01", "-", "1e-"]) {
      const html = renderToStaticMarkup(
        <NumericInput
          id="c"
          value={value}
          onDraft={vi.fn()}
          onCommit={vi.fn()}
        />,
      );
      expect(html).toContain('type="text"');
      expect(html).toContain('inputMode="decimal"');
      expect(html).toContain(`value="${value}"`);
      expect(html).not.toContain('role="alert"');
    }
  });
  it("associates visible invalid entries with their error and retains the invalid text", () => {
    const html = renderToStaticMarkup(
      <NumericInput
        id="workers"
        value="1.5"
        integer
        error="Enter a whole number."
        aria-describedby="workers-help"
        onDraft={vi.fn()}
        onCommit={vi.fn()}
      />,
    );
    expect(html).toContain('aria-invalid="true"');
    expect(html).toContain('aria-describedby="workers-help workers-error"');
    expect(html).toContain('value="1.5"');
    expect(html).toContain('inputMode="numeric"');
    expect(html).toContain('id="workers-error"');
  });
});
