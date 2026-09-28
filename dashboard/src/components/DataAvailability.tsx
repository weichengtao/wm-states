import {
  AlertTriangle,
  CheckCircle2,
  FolderOpen,
  LoaderCircle,
  RefreshCw,
} from "lucide-react";
import type { DataAvailabilityState } from "@/lib/data-availability";
import GuideLink from "./GuideLink";
import { Button } from "./ui/button";
import "./DataAvailability.css";

export default function DataAvailability({
  state,
  onRecheck,
  compact = false,
}: {
  state: DataAvailabilityState;
  onRecheck: () => void;
  compact?: boolean;
}) {
  const { result, checking, error } = state;
  const warning = result !== null && result.status !== "ready";
  if (compact && !warning && !error) return null;
  const folderSummary =
    result?.status === "missing"
      ? "This folder does not exist on the dashboard server."
      : result?.status === "empty"
        ? "This server folder has no .mat recording files."
        : null;
  const message = folderSummary
    ? `${folderSummary}${result?.blocking ? "" : " Selected stages use cached outputs; this will not prevent launch."}`
    : result?.message;
  const title = error
    ? "Recording check unavailable"
    : !result
      ? "Checking recordings on the server…"
      : result.status === "ready"
        ? "Recording folder checked"
        : result.status === "no_matches"
          ? "No sessions match these settings"
          : result.status === "empty"
            ? "No recording files found"
            : result.status === "missing"
              ? "Recordings are not available yet"
              : result.status === "unreadable"
                ? "Recording folder cannot be read"
                : "Check the recording settings";
  return (
    <div
      className={`data-availability ${warning ? "data-availability-warning" : error ? "data-availability-unknown" : ""}${compact ? " data-availability-compact" : ""}`}
      role="status"
      aria-live="polite"
      aria-atomic="true"
    >
      {checking ? (
        <LoaderCircle size={17} className="animate-spin" />
      ) : warning ? (
        <AlertTriangle size={17} />
      ) : error ? (
        <FolderOpen size={17} />
      ) : (
        <CheckCircle2 size={17} />
      )}
      <div>
        <strong>{title}</strong>
        <p>
          {error
            ? "The dashboard couldn’t check this folder. Try again; validation and launch will check the server’s files again."
            : (message ??
              "Folder paths refer to the computer running the dashboard. You can continue editing and save a template while this is checked.")}
        </p>
        {!compact && result && (
          <div className="data-availability-path">
            <span>Server folder</span>
            <code>{result.data_dir}</code>
            <small>
              {result.file_count} recording{" "}
              {result.file_count === 1 ? "file" : "files"} found
            </small>
          </div>
        )}
        {!compact && error && (
          <small className="data-availability-error">{error}</small>
        )}
        {result?.blocking && (
          <p>
            {compact
              ? "Prepare recordings or choose another folder before starting."
              : "Choose a prepared recording folder, then check again. You can still save this configuration as a template."}
          </p>
        )}
        {(result || error) && (
          <div className="data-availability-actions">
            {(warning || error) && (
              <GuideLink path="next/getting-started/#prepare-the-recordings">
                Download & prepare recordings
              </GuideLink>
            )}
            <Button
              variant="ghost"
              size="sm"
              disabled={checking}
              onClick={onRecheck}
            >
              <RefreshCw className={checking ? "animate-spin" : ""} />
              {checking ? "Checking…" : "Check again"}
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}
