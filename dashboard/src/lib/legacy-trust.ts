import type { LegacyTrust, Manifest } from "./types";

export type LegacyTrustStatus = "used" | "enabled" | null;

export function legacyTrustStatus(
  enabled: boolean | undefined,
  evidence?: LegacyTrust,
): LegacyTrustStatus {
  if (evidence?.manual_trust_used === true) return "used";
  return enabled === true || evidence?.enabled === true ? "enabled" : null;
}

export function manifestLegacyTrust(manifest: Manifest): LegacyTrustStatus {
  const runner = manifest.runner_config;
  const enabled =
    !!runner &&
    typeof runner === "object" &&
    "trust_unverified_legacy_results" in runner &&
    runner.trust_unverified_legacy_results === true;
  return legacyTrustStatus(enabled, manifest.legacy_trust);
}

export function historyLegacyTrust(manifests: Manifest[]): LegacyTrustStatus {
  const statuses = manifests.map(manifestLegacyTrust);
  return statuses.includes("used")
    ? "used"
    : statuses.includes("enabled")
      ? "enabled"
      : null;
}
