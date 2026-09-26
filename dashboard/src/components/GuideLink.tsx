import type { ReactNode } from "react";
import { ArrowUpRight } from "lucide-react";
import { docsHref } from "@/lib/help";

export default function GuideLink({
  path = "",
  children,
  className = "guide-link",
  label,
}: {
  path?: string;
  children: ReactNode;
  className?: string;
  label?: string;
}) {
  return (
    <a
      href={docsHref(path)}
      target="_blank"
      rel="noopener noreferrer"
      className={className}
      aria-label={label ? `${label} (opens in a new tab)` : undefined}
      title="Opens the guide in a new tab"
    >
      {children}
      <ArrowUpRight size={13} aria-hidden="true" />
      <span className="sr-only"> (opens in a new tab)</span>
    </a>
  );
}
