import { describe, expect, it, vi, afterEach } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import GuideLink, { HelpLink } from "./GuideLink";
import HelpPanel from "./HelpPanel";

afterEach(() => vi.unstubAllEnvs());

describe("help controls", () => {
  it("opens external-base field guidance separately with an accessible notice", () => {
    vi.stubEnv("VITE_DOCS_BASE_URL", "https://example.github.io/wm-states/");
    const html = renderToStaticMarkup(
      <GuideLink path="next/methods/#decode" label="Decoding methods">
        Stage methods
      </GuideLink>,
    );
    expect(html).toContain(
      'href="https://example.github.io/wm-states/next/methods/#decode"',
    );
    expect(html).toContain('target="_blank"');
    expect(html).toContain('rel="noopener noreferrer"');
    expect(html).toContain(
      'aria-label="Decoding methods (opens in a new tab)"',
    );
  });
  it("offers a labelled dialog trigger rather than navigating away from the workspace", () => {
    const html = renderToStaticMarkup(<HelpPanel page="configure" />);
    expect(html).toContain('aria-label="Open pipeline help"');
    expect(html).toContain('aria-haspopup="dialog"');
    expect(html).toContain('aria-expanded="false"');
    expect(html).not.toContain("href=");
  });
  it("opens a primary reference without rewriting it to the guide's deployment base", () => {
    vi.stubEnv("VITE_DOCS_BASE_URL", "https://example.github.io/wm-states/");
    const html = renderToStaticMarkup(
      <HelpLink href="https://scikit-learn.org/1.8/modules/calibration.html">
        scikit-learn: probability calibration
      </HelpLink>,
    );
    expect(html).toContain(
      'href="https://scikit-learn.org/1.8/modules/calibration.html"',
    );
    expect(html).toContain('target="_blank"');
    expect(html).toContain('rel="noopener noreferrer"');
    expect(html).toContain("opens in a new tab");
    expect(html).not.toContain("example.github.io");
  });
});
