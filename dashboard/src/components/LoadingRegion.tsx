import { useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { cn } from "@/lib/utils";
import { Loading } from "./shared";
import "./LoadingRegion.css";

/** Track settled content, including charts and images that resize after mount. */
export function observeContentHeight(
  element: HTMLElement,
  onHeight: (height: number) => void,
) {
  let active = true;
  const measure = () => {
    if (!active) return;
    const height = element.getBoundingClientRect().height;
    if (Number.isFinite(height) && height >= 0) onHeight(Math.ceil(height));
  };
  measure();
  const observer =
    typeof ResizeObserver === "undefined" ? null : new ResizeObserver(measure);
  observer?.observe(element);
  const view = element.ownerDocument.defaultView;
  // ResizeObserver handles content changes; the fallback still tracks layout
  // changes when the browser does not provide it.
  if (!observer) view?.addEventListener("resize", measure);
  return () => {
    active = false;
    observer?.disconnect();
    if (!observer) view?.removeEventListener("resize", measure);
  };
}

/** Keep a loading request from briefly collapsing a previously loaded region. */
export function LoadingRegion({
  loading,
  label = "Loading results…",
  retainChildren = false,
  className,
  children,
}: {
  loading: boolean;
  label?: string;
  /** Use only when nested regions already replace their own pending content. */
  retainChildren?: boolean;
  className?: string;
  children?: ReactNode;
}) {
  const contentRef = useRef<HTMLDivElement>(null);
  const [contentHeight, setContentHeight] = useState(0);
  useLayoutEffect(() => {
    if (loading || !contentRef.current) return;
    return observeContentHeight(contentRef.current, setContentHeight);
  }, [loading, children]);

  return (
    <div
      className={cn("loading-region", className)}
      aria-busy={loading}
      style={
        loading && contentHeight > 0 ? { minHeight: contentHeight } : undefined
      }
    >
      {loading && !retainChildren ? (
        <Loading label={label} />
      ) : (
        <div ref={contentRef} className="loading-region-content">
          {children}
        </div>
      )}
    </div>
  );
}
