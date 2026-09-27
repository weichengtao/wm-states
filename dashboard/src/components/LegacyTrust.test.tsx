import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { LegacyTrustNotice, LegacyTrustPanel } from "./LegacyTrust";

describe("manual legacy trust controls", () => {
  it("offers an explicit unchecked choice and guide without a confirmation modal", () => {
    const html = renderToStaticMarkup(
      <LegacyTrustPanel enabled={false} onChange={vi.fn()} />,
    );
    expect(html).toContain("Trust unverified legacy results");
    expect(html).toContain("Off by default; never saved in templates");
    expect(html).toContain("#trust-unverified-legacy-results");
    expect(html).not.toContain('checked=""');
    expect(html).not.toContain('role="dialog"');
  });
  it("explains the scope and recorded acceptance when enabled", () => {
    const html = renderToStaticMarkup(
      <LegacyTrustPanel enabled onChange={vi.fn()} />,
    );
    expect(html).toContain('checked=""');
    expect(html).toContain(
      "does not verify their scientific provenance or fit new estimates",
    );
    expect(html).toContain(
      "Required files and cache structure are still checked",
    );
    expect(html).toContain("Current-format fingerprints are not bypassed");
    expect(html).toContain(
      "Original inputs, source code, and runtime environment remain unverified",
    );
    expect(html).toContain("recorded in the manifest");
  });
  it("labels actual acceptance separately from permission and ordinary runs", () => {
    expect(renderToStaticMarkup(<LegacyTrustNotice status={null} />)).toBe("");
    const enabled = renderToStaticMarkup(
      <LegacyTrustNotice status="enabled" />,
    );
    expect(enabled).toContain("setting alone does not mean a cache was reused");
    expect(enabled).not.toContain("Unverified legacy results accepted");
    expect(renderToStaticMarkup(<LegacyTrustNotice status="used" />)).toContain(
      "Unverified legacy results accepted",
    );
    expect(
      renderToStaticMarkup(<LegacyTrustNotice status="used" scope="history" />),
    ).toContain("outputs are shared across invocations");
  });
});
