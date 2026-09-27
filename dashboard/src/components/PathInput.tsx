import {
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
  type ComponentProps,
} from "react";
import { FileText, Folder, LoaderCircle } from "lucide-react";
import { api, errorMessage } from "@/lib/api";
import {
  pathCompletionQuery,
  pathPopupPlacement,
  tabCompletion,
  type PathCompletion,
  type PathEntry,
} from "@/lib/path-completion";
import { Input } from "./ui/input";
import "./PathInput.css";

type Props = Omit<ComponentProps<"input">, "value" | "onChange" | "type"> & {
  value: string;
  onValueChange: (value: string) => void;
  mode?: "directory" | "any";
  extensions?: string[];
  completionHint?: string;
};

export default function PathInput({
  value,
  onValueChange,
  mode = "any",
  extensions = [],
  completionHint,
  onFocus,
  onBlur,
  onKeyDown,
  ...props
}: Props) {
  const id = useId();
  const input = useRef<HTMLInputElement>(null);
  const options = useRef<HTMLUListElement>(null);
  const [focused, setFocused] = useState(false);
  const [dismissed, setDismissed] = useState(false);
  const [active, setActive] = useState(-1);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PathCompletion | null>(null);
  const [resolvedQuery, setResolvedQuery] = useState("");
  const [notice, setNotice] = useState("");
  const [placement, setPlacement] = useState<{
    side: "above" | "below";
    maxHeight: number;
    inputHeight: number;
  }>({ side: "below", maxHeight: 360, inputHeight: 40 });
  const query = pathCompletionQuery(value, mode, extensions);
  const open = focused && !dismissed && !props.disabled;
  const current = resolvedQuery === query ? result : null;
  const entries = current?.entries ?? [];
  const activeEntry = entries[active];

  useLayoutEffect(() => {
    if (!open) return;
    function reposition() {
      const element = input.current;
      if (!element) return;
      const rect = element.getBoundingClientRect();
      const viewport = window.visualViewport;
      const next = {
        ...pathPopupPlacement(rect, {
          top: viewport?.offsetTop ?? 0,
          height: viewport?.height ?? window.innerHeight,
        }),
        inputHeight: rect.height,
      };
      setPlacement((previous) =>
        previous.side === next.side &&
        previous.maxHeight === next.maxHeight &&
        previous.inputHeight === next.inputHeight
          ? previous
          : next,
      );
    }
    reposition();
    window.addEventListener("resize", reposition);
    window.addEventListener("scroll", reposition, true);
    window.visualViewport?.addEventListener("resize", reposition);
    window.visualViewport?.addEventListener("scroll", reposition);
    return () => {
      window.removeEventListener("resize", reposition);
      window.removeEventListener("scroll", reposition, true);
      window.visualViewport?.removeEventListener("resize", reposition);
      window.visualViewport?.removeEventListener("scroll", reposition);
    };
  }, [open]);

  useEffect(() => {
    if (open && activeEntry)
      options.current?.children[active]?.scrollIntoView({ block: "nearest" });
  }, [active, activeEntry, open]);

  useEffect(() => {
    if (!open) return;
    const controller = new AbortController();
    let currentRequest = true;
    setActive(-1);
    setLoading(true);
    const timer = window.setTimeout(async () => {
      try {
        const completion = await api<PathCompletion>(query, {
          signal: controller.signal,
        });
        if (!currentRequest) return;
        setResult(completion);
        setResolvedQuery(query);
        setNotice("");
      } catch (error) {
        if (!currentRequest || controller.signal.aborted) return;
        setResult({
          entries: [],
          truncated: false,
          warning: `Suggestions unavailable: ${errorMessage(error)}. You can still type a path manually.`,
        });
        setResolvedQuery(query);
      } finally {
        if (currentRequest) setLoading(false);
      }
    }, 180);
    return () => {
      currentRequest = false;
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [query, open]);

  function choose(path: string, entry?: PathEntry) {
    const directory = entry?.kind === "directory" || path.endsWith("/");
    onValueChange(path);
    setActive(-1);
    setNotice(
      !entry
        ? "Shared prefix completed. Keep typing, or use ↓ to choose a match."
        : directory
          ? "Folder completed. Keep typing to browse inside, or press Tab to move on."
          : "Path completed. Press Tab to move to the next field.",
    );
    // Completing a path must not trap Tab inside nested directories. ArrowDown
    // or typing reopens suggestions when the user wants to keep browsing.
    setDismissed(true);
    input.current?.focus();
  }

  return (
    <div className="path-input">
      <Input
        {...props}
        ref={input}
        value={value}
        type="text"
        autoComplete="off"
        spellCheck={false}
        role="combobox"
        aria-autocomplete="list"
        aria-expanded={open}
        aria-controls={open ? `${id}-options` : undefined}
        aria-activedescendant={
          open && activeEntry ? `${id}-option-${active}` : undefined
        }
        aria-describedby={[props["aria-describedby"], `${id}-hint`]
          .filter(Boolean)
          .join(" ")}
        onFocus={(event) => {
          setFocused(true);
          setDismissed(false);
          onFocus?.(event);
        }}
        onBlur={(event) => {
          setFocused(false);
          setNotice("");
          onBlur?.(event);
        }}
        onChange={(event) => {
          setDismissed(false);
          setNotice("");
          onValueChange(event.target.value);
        }}
        onKeyDown={(event) => {
          onKeyDown?.(event);
          if (event.defaultPrevented || event.nativeEvent.isComposing) return;
          if (event.key === "Escape") {
            if (open) {
              event.preventDefault();
              event.stopPropagation();
              setDismissed(true);
            }
          } else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
            event.preventDefault();
            if (!open) {
              setDismissed(false);
              return;
            }
            if (entries.length)
              setActive((previous) =>
                event.key === "ArrowDown"
                  ? (previous + 1) % entries.length
                  : previous <= 0
                    ? entries.length - 1
                    : previous - 1,
              );
          } else if (event.key === "Enter" && open && activeEntry) {
            event.preventDefault();
            choose(activeEntry.path, activeEntry);
          } else if (event.key === "Tab" && !event.shiftKey && open) {
            const element = event.currentTarget;
            if (
              element.selectionStart !== value.length ||
              element.selectionEnd !== value.length
            )
              return;
            const completion = tabCompletion(
              value,
              entries,
              active,
              current?.truncated,
            );
            if (completion !== null && completion !== value) {
              event.preventDefault();
              choose(
                completion,
                entries.find((entry) => entry.path === completion),
              );
            }
          }
        }}
      />
      <div id={`${id}-hint`} className="path-input-hint" aria-live="polite">
        {notice ||
          completionHint ||
          "Local path · Tab completes a match · ↑↓ chooses a path"}
      </div>
      {open && (
        <div
          className={`path-suggestions path-suggestions-${placement.side}`}
          style={{
            maxHeight: placement.maxHeight,
            top:
              placement.side === "below"
                ? placement.inputHeight + 4
                : undefined,
            bottom: placement.side === "above" ? "calc(100% + 4px)" : undefined,
          }}
        >
          <div className="path-suggestions-heading">
            <Folder size={13} />{" "}
            {mode === "directory" ? "Local folders" : "Local files and folders"}
            {loading && (
              <LoaderCircle
                size={13}
                className="path-loading"
                aria-label="Loading suggestions"
              />
            )}
          </div>
          <ul
            ref={options}
            role="listbox"
            id={`${id}-options`}
            aria-label="Path suggestions"
          >
            {entries.map((entry, index) => (
              <li
                key={entry.path}
                id={`${id}-option-${index}`}
                role="option"
                aria-selected={index === active}
                className={
                  index === active
                    ? "path-option path-option-active"
                    : "path-option"
                }
                onMouseDown={(event) => event.preventDefault()}
                onMouseMove={() => setActive(index)}
                onClick={() => choose(entry.path, entry)}
              >
                {entry.kind === "directory" ? (
                  <Folder size={15} />
                ) : (
                  <FileText size={15} />
                )}
                <span title={entry.path}>{entry.name}</span>
                <small>{entry.kind === "directory" ? "Folder" : "File"}</small>
              </li>
            ))}
          </ul>
          {current?.warning && (
            <p className="path-suggestions-warning">{current.warning}</p>
          )}
          {!loading && current && !entries.length && !current.warning && (
            <p className="path-suggestions-empty">
              No matching paths. You can keep typing a new path.
            </p>
          )}
          <div className="path-suggestions-footer">
            <span>
              <kbd>↑↓</kbd> choose · <kbd>Enter</kbd> use
            </span>
            <span>
              <kbd>Esc</kbd> close
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
