const DEFAULT_DOCS_BASE = "/wm-states/docs/";

/** Read deployment paths from the server, independent of the current page depth. */
function runtimeBase(name: string, fallback: string): string {
  if (typeof document === "undefined") return fallback;
  const value = document
    .querySelector<HTMLMetaElement>(`meta[name="${name}"]`)
    ?.content?.trim();
  if (!value || !value.startsWith("/") || value.startsWith("//"))
    return fallback;
  if (/[\\\s?#%]/.test(value) || value.includes("//")) return fallback;
  const parts = value.split("/");
  if (parts.some((part) => part === "." || part === "..")) return fallback;
  return `${value.replace(/\/+$/, "")}/`;
}

// Vite's development server does not inject metadata and proxies /api itself.
export function dashboardBase(): string {
  return runtimeBase("wm-states-dashboard-base", "/");
}

export function docsBase(): string {
  return runtimeBase("wm-states-docs-base", DEFAULT_DOCS_BASE);
}

export function apiHref(path: string): string {
  return `${dashboardBase()}api/${path.replace(/^\/+/, "")}`;
}

export function websocketHref(path: string): string {
  const url = new URL(apiHref(path), window.location.origin);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  return url.href;
}
