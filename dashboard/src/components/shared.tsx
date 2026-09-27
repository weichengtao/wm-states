import { useEffect, useRef, useState, type ReactNode } from "react";
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
import { copyText } from "@/lib/clipboard";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "./ui/dialog";
import "./CopyButton.css";
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
    <div className="loading-state" role="status" aria-live="polite">
      <LoaderCircle className="animate-spin" size={22} />
      {label}
    </div>
  );
}
export function CopyButton({
  text,
  label = "Copy",
  disabled = false,
}: {
  text: string;
  label?: string;
  disabled?: boolean;
}) {
  const [state, setState] = useState("");
  const [manualText, setManualText] = useState<string | null>(null);
  const button = useRef<HTMLButtonElement>(null);
  const textarea = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    if (state !== "Copied") return;
    const timer = setTimeout(() => setState(""), 1800);
    return () => clearTimeout(timer);
  }, [state]);
  async function copy() {
    if (state === "Copying…") return;
    const captured = text;
    setState("Copying…");
    if (await copyText(captured)) setState("Copied");
    else {
      setState("");
      setManualText(captured);
    }
  }
  return (
    <Dialog
      open={manualText !== null}
      onOpenChange={(open) => {
        if (!open) setManualText(null);
      }}
    >
      <Button
        ref={button}
        size="sm"
        variant="outline"
        disabled={disabled}
        aria-disabled={disabled || state === "Copying…"}
        aria-busy={state === "Copying…"}
        aria-live="polite"
        onClick={() => void copy()}
      >
        {state === "Copied" ? (
          <Check />
        ) : state === "Copying…" ? (
          <LoaderCircle className="animate-spin" />
        ) : (
          <Copy />
        )}
        {state || label}
      </Button>
      <DialogContent
        className="manual-copy-dialog"
        onOpenAutoFocus={(event) => {
          event.preventDefault();
          textarea.current?.focus({ preventScroll: true });
        }}
        onCloseAutoFocus={(event) => {
          event.preventDefault();
          button.current?.focus({ preventScroll: true });
        }}
      >
        <DialogTitle>Copy this text</DialogTitle>
        <DialogDescription>
          Your browser didn’t allow automatic copying. Select the text below,
          then press ⌘C or Ctrl+C, or use your device’s Copy menu.
        </DialogDescription>
        <textarea
          ref={textarea}
          aria-label="Text to copy"
          value={manualText ?? ""}
          readOnly
          spellCheck={false}
          rows={7}
        />
        <div className="manual-copy-actions">
          <Button variant="outline" onClick={() => setManualText(null)}>
            Done
          </Button>
          <Button
            onClick={() => {
              textarea.current?.focus({ preventScroll: true });
              textarea.current?.select();
            }}
          >
            Select text
          </Button>
        </div>
      </DialogContent>
    </Dialog>
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
