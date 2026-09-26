import type { ReactNode } from "react";
import { ArrowUpRight } from "lucide-react";
import { docsHref } from "@/lib/help";

export function HelpLink({
  href,
  children,
  className = "guide-link",
  label,
  title = "Opens a reference in a new tab",
}: {
  href: string;
  children: ReactNode;
  className?: string;
  label?: string;
  title?: string;
}) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className={className}
      aria-label={label ? `${label} (opens in a new tab)` : undefined}
      title={title}
    >
      {children}
      <ArrowUpRight size={13} aria-hidden="true" />
      <span className="sr-only"> (opens in a new tab)</span>
    </a>
  );
}

export default function GuideLink({
  path = "",
  ...props
}: {
  path?: string;
  children: ReactNode;
  className?: string;
  label?: string;
}) {
  return (
    <HelpLink
      {...props}
      href={docsHref(path)}
      title="Opens the guide in a new tab"
    />
  );
}
