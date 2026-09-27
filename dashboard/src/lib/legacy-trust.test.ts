import { describe, expect, it } from "vitest";
import type { Manifest } from "./types";
import {
  historyLegacyTrust,
  legacyTrustStatus,
  manifestLegacyTrust,
} from "./legacy-trust";

const manifest: Manifest = { id: "invocation", status: "complete", stages: [] };

describe("legacy trust provenance labels", () => {
  it("distinguishes enabled permission from evidence of manual acceptance", () => {
    expect(legacyTrustStatus(undefined)).toBeNull();
    expect(legacyTrustStatus(true)).toBe("enabled");
    expect(legacyTrustStatus(false, { enabled: true })).toBe("enabled");
    expect(legacyTrustStatus(false, { manual_trust_used: true })).toBe("used");
  });
  it("reads run-level manifest evidence without accepting truthy strings", () => {
    expect(manifestLegacyTrust(manifest)).toBeNull();
    expect(
      manifestLegacyTrust({
        ...manifest,
        runner_config: { trust_unverified_legacy_results: "false" },
      }),
    ).toBeNull();
    expect(
      manifestLegacyTrust({
        ...manifest,
        runner_config: { trust_unverified_legacy_results: true },
      }),
    ).toBe("enabled");
    expect(
      manifestLegacyTrust({
        ...manifest,
        legacy_trust: { manual_trust_used: true },
      }),
    ).toBe("used");
  });
  it("retains an earlier acceptance warning after a later partial invocation", () => {
    expect(
      historyLegacyTrust([
        manifest,
        { ...manifest, legacy_trust: { manual_trust_used: true } },
      ]),
    ).toBe("used");
    expect(historyLegacyTrust([manifest])).toBeNull();
  });
});
