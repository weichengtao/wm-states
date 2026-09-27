import { AlertTriangle, ShieldQuestion } from "lucide-react";
import type { LegacyTrustStatus } from "@/lib/legacy-trust";
import GuideLink from "./GuideLink";
import "./LegacyTrust.css";

const guide = "next/configuration/#trust-unverified-legacy-results";

export function LegacyTrustPanel({
  enabled,
  onChange,
}: {
  enabled: boolean;
  onChange: (enabled: boolean) => void;
}) {
  return (
    <div
      className={`legacy-trust-panel ${enabled ? "legacy-trust-panel-enabled" : ""}`}
      role="group"
      aria-labelledby="legacy-trust-heading"
    >
      <div className="legacy-trust-heading">
        <ShieldQuestion size={18} aria-hidden="true" />
        <div>
          <h3 id="legacy-trust-heading">
            Advanced · Legacy cache compatibility
          </h3>
          <p>
            Keep automatic provenance checks unless you need to accept an older,
            unverifiable decoding cache.
          </p>
        </div>
        <GuideLink path={guide}>When to use this</GuideLink>
      </div>
      <label className="legacy-trust-control">
        <input
          type="checkbox"
          checked={enabled}
          onChange={(event) => onChange(event.target.checked)}
          aria-describedby="legacy-trust-description"
        />
        <span>
          <strong>Trust unverified legacy results</strong>
          <small id="legacy-trust-description">
            Applies only to existing results with legacy fingerprints. Off by
            default; never saved in templates.
          </small>
        </span>
      </label>
      {enabled && (
        <div className="legacy-trust-warning" role="status">
          <AlertTriangle size={18} aria-hidden="true" />
          <p>
            You are manually accepting unverified legacy results for this
            invocation. This does not verify their scientific provenance or fit
            new estimates. Original inputs, source code, and runtime environment
            remain unverified. Required files and cache structure are still
            checked. Current-format fingerprints are not bypassed. The choice
            and any acceptance are recorded in the manifest.
          </p>
        </div>
      )}
    </div>
  );
}

export function LegacyTrustNotice({
  status,
  scope = "invocation",
}: {
  status: LegacyTrustStatus;
  scope?: "invocation" | "history";
}) {
  if (!status) return null;
  const used = status === "used";
  return (
    <div className="legacy-trust-notice" role="note">
      <AlertTriangle size={18} aria-hidden="true" />
      <div>
        <strong>
          {used ? "Unverified legacy results accepted" : "Legacy trust enabled"}
        </strong>
        <p>
          {scope === "history"
            ? used
              ? "An earlier invocation manually accepted legacy results without verified provenance. Review run history for the affected sessions and reasons; outputs are shared across invocations."
              : "Legacy trust was enabled in this run's history. This records permission, not proof that an unverified result was reused. Review each invocation's acceptance records."
            : used
              ? "This invocation manually accepted legacy results without verified provenance. Acceptance does not refit those estimates. The manifest records the affected sessions and reasons."
              : "This invocation permits manual acceptance of unverified legacy results. This setting alone does not mean a cache was reused; check the manifest's acceptance records."}
        </p>
        <GuideLink path={guide}>Understand legacy trust</GuideLink>
      </div>
    </div>
  );
}
