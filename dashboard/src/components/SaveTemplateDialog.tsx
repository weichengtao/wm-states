import { useRef, useState } from "react";
import { BookmarkPlus, Check, LoaderCircle, Save } from "lucide-react";
import type { PipelineTemplate, RunRequest, Schema } from "@/lib/types";
import { api, errorMessage } from "@/lib/api";
import { templateConfig } from "@/lib/templates";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
  DialogTrigger,
} from "./ui/dialog";

export default function SaveTemplateDialog({
  schema,
  form,
  jsonDirty,
  running = false,
  onSaved,
  onError,
}: {
  schema: Schema;
  form: RunRequest;
  jsonDirty: boolean;
  running?: boolean;
  onSaved: (template: PipelineTemplate, snapshot: RunRequest) => void;
  onError: (message: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const [snapshot, setSnapshot] = useState(form);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [includePaths, setIncludePaths] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const attempt = useRef(0);
  const openRef = useRef(false);
  const errorRef = useRef<HTMLParagraphElement>(null);
  function changeOpen(next: boolean) {
    openRef.current = next;
    setOpen(next);
    if (next) {
      setSnapshot(structuredClone(form));
      setName("");
      setDescription("");
      setIncludePaths(false);
      setError("");
    }
  }
  async function save() {
    if (saving) return;
    const requestedAttempt = ++attempt.current;
    const captured = structuredClone(snapshot);
    setSaving(true);
    setError("");
    try {
      const saved = await api<PipelineTemplate>("/templates", {
        method: "POST",
        body: JSON.stringify({
          name: name.trim(),
          description: description.trim(),
          config: templateConfig(captured, schema, includePaths),
        }),
      });
      onSaved(saved, captured);
      if (attempt.current === requestedAttempt) changeOpen(false);
    } catch (reason) {
      if (attempt.current === requestedAttempt && openRef.current) {
        setError(errorMessage(reason));
        requestAnimationFrame(() => errorRef.current?.focus());
      } else {
        onError(`Template could not be saved. ${errorMessage(reason)}`);
      }
    } finally {
      if (attempt.current === requestedAttempt) setSaving(false);
    }
  }
  return (
    <Dialog open={open} onOpenChange={changeOpen}>
      <DialogTrigger asChild>
        <Button
          variant="outline"
          disabled={jsonDirty || saving}
          title={
            jsonDirty
              ? "Apply your JSON changes before saving a template."
              : undefined
          }
        >
          {saving ? (
            <LoaderCircle className="animate-spin" />
          ) : (
            <BookmarkPlus />
          )}
          {saving ? "Saving template…" : "Save as template"}
        </Button>
      </DialogTrigger>
      <DialogContent className="template-save-dialog">
        <span className="template-dialog-icon">
          <BookmarkPlus size={24} />
        </span>
        <DialogTitle>Keep this setup for next time</DialogTitle>
        <DialogDescription>
          Save a new reusable template from your current configuration.
        </DialogDescription>
        <form
          className="template-form"
          onSubmit={(event) => {
            event.preventDefault();
            void save();
          }}
        >
          <div className="field">
            <label htmlFor="template-name">Template name</label>
            <Input
              id="template-name"
              value={name}
              onChange={(event) => setName(event.target.value)}
              maxLength={120}
              placeholder="e.g. Main analysis · 200 null shuffles"
              required
              disabled={saving}
              autoComplete="off"
            />
          </div>
          <div className="field">
            <label htmlFor="template-description">
              Description <span>optional</span>
            </label>
            <textarea
              id="template-description"
              value={description}
              onChange={(event) => setDescription(event.target.value)}
              maxLength={500}
              rows={3}
              placeholder="What is this setup useful for?"
              disabled={saving}
            />
          </div>
          <div className="template-save-summary">
            <strong>
              <Check size={16} /> Saved with this template
            </strong>
            <p>
              {snapshot.stages.length} selected stages, all stage settings,
              worker count, session limit, and figure formats.
            </p>
            <p>
              Run name, cache directory, and permission to reuse outputs stay
              with each run.
            </p>
          </div>
          <label className="template-path-option">
            <input
              type="checkbox"
              checked={includePaths}
              onChange={(event) => setIncludePaths(event.target.checked)}
              disabled={saving}
              aria-describedby="template-path-help"
            />
            <span>
              <strong>Include recording paths</strong>
              <small id="template-path-help">
                Save the recording directory and session allowlist paths too.
                Leave off to use the template with other recordings.
              </small>
            </span>
          </label>
          {running && (
            <p className="template-running-note">
              Saving a template does not change the running analysis.
            </p>
          )}
          {!snapshot.stages.length && (
            <p className="template-save-error" role="alert">
              Select at least one pipeline stage before saving a template.
            </p>
          )}
          {error && (
            <p
              className="template-save-error"
              role="alert"
              ref={errorRef}
              tabIndex={-1}
            >
              {error}
            </p>
          )}
          <div className="template-dialog-actions">
            <Button variant="ghost" onClick={() => changeOpen(false)}>
              {saving ? "Close" : "Cancel"}
            </Button>
            <Button
              type="submit"
              disabled={saving || !name.trim() || !snapshot.stages.length}
            >
              {saving ? <LoaderCircle className="animate-spin" /> : <Save />}
              {saving ? "Saving…" : "Save new template"}
            </Button>
          </div>
          <p className="template-storage-note">
            {saving
              ? "Saving continues if you close this dialog."
              : "Saved locally in configs/next/templates. Existing templates are kept unchanged."}
          </p>
        </form>
      </DialogContent>
    </Dialog>
  );
}
