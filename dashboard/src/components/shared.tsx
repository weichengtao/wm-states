import { useEffect, useState, type ReactNode } from "react";
import {
  AlertCircle,
  Check,
  CheckCircle2,
  Circle,
  Copy,
  LoaderCircle,
  XCircle,
} from "lucide-react";
import { Button } from "./ui/button";
import { cn } from "@/lib/utils";
import GuideLink from "./GuideLink";
import { troubleshootingPath } from "@/lib/help";
export function Status({ value }: { value: string }) {
  const running = ["running", "queued", "cancelling"].includes(value);
  const success = ["complete", "completed"].includes(value);
  return (
    <span
      className={cn(
        "status",
        running
          ? "status-running"
          : success
            ? "status-complete"
            : ["failed", "interrupted", "cancelled"].includes(value)
              ? "status-failed"
              : "status-unknown",
      )}
    >
      {running ? (
        <LoaderCircle size={12} className="animate-spin" />
      ) : success ? (
        <CheckCircle2 size={12} />
      ) : ["failed", "interrupted", "cancelled"].includes(value) ? (
        <XCircle size={12} />
      ) : (
        <Circle size={10} />
      )}{" "}
      {value.replaceAll("_", " ")}
    </span>
  );
}
export function Notice({
  children,
  tone = "error",
}: {
  children: ReactNode;
  tone?: "error" | "info";
}) {
  return (
    <div
      role={tone === "error" ? "alert" : "status"}
      className={cn("notice", tone === "info" && "notice-info")}
    >
      <AlertCircle size={17} />
      <div>
        {children}
        {tone === "error" && (
          <GuideLink
            path={troubleshootingPath(
              typeof children === "string" ? children : "",
            )}
            className="notice-guide-link"
          >
            Troubleshooting guide
          </GuideLink>
        )}
      </div>
    </div>
  );
}
export function Empty({
  title,
  children,
  icon,
}: {
  title: string;
  children?: ReactNode;
  icon?: ReactNode;
}) {
  return (
    <div className="empty-state">
      {icon && <div className="empty-icon">{icon}</div>}
      <h3>{title}</h3>
      <div>{children}</div>
    </div>
  );
}
export function Loading({ label = "Loading results…" }: { label?: string }) {
  return (
    <div className="loading-state">
      <LoaderCircle className="animate-spin" size={22} />
      {label}
    </div>
  );
}
export function CopyButton({
  text,
  label = "Copy",
}: {
  text: string;
  label?: string;
}) {
  const [state, setState] = useState("");
  useEffect(() => {
    if (!state) return;
    const timer = setTimeout(() => setState(""), 1800);
    return () => clearTimeout(timer);
  }, [state]);
  return (
    <Button
      size="sm"
      variant="outline"
      onClick={() =>
        navigator.clipboard
          .writeText(text)
          .then(() => setState("Copied"))
          .catch(() => setState("Select text to copy"))
      }
    >
      {state === "Copied" ? <Check /> : <Copy />}
      {state || label}
    </Button>
  );
}
export function Stat({
  label,
  value,
  sub,
  icon,
}: {
  label: string;
  value: ReactNode;
  sub?: string;
  icon?: ReactNode;
}) {
  return (
    <div className="stat-card">
      <div className="stat-top">
        {label}
        {icon}
      </div>
      <strong>{value}</strong>
      {sub && <span>{sub}</span>}
    </div>
  );
}
