export type PathEntry = {
  name: string;
  path: string;
  kind: "directory" | "file";
};
export type PathCompletion = {
  entries: PathEntry[];
  warning: string | null;
  truncated: boolean;
};

/** Complete a shared prefix first; arrows explicitly choose a specific match. */
export function tabCompletion(
  value: string,
  entries: PathEntry[],
  active: number,
  truncated = false,
): string | null {
  if (entries.length === 0) return null;
  if (active >= 0 && active < entries.length) return entries[active].path;
  // A partial result cannot establish uniqueness or a shared prefix across all
  // matches. The user can still explicitly select any visible row.
  if (truncated) return null;
  if (entries.length === 1) return entries[0].path;
  let common = entries[0].path;
  for (const entry of entries.slice(1)) {
    let length = 0;
    while (length < common.length && common[length] === entry.path[length])
      length++;
    common = common.slice(0, length);
  }
  return common.length > value.length ? common : null;
}

export function pathPopupPlacement(
  input: { top: number; bottom: number },
  viewport: { top: number; height: number },
): { side: "above" | "below"; maxHeight: number } {
  const margin = 8;
  const gap = 4;
  const above = Math.max(0, input.top - viewport.top - margin - gap);
  const below = Math.max(
    0,
    viewport.top + viewport.height - input.bottom - margin - gap,
  );
  const side = below >= 280 || below >= above ? "below" : "above";
  return { side, maxHeight: Math.min(360, side === "above" ? above : below) };
}

export function pathCompletionQuery(
  value: string,
  mode: "directory" | "any",
  extensions: string[],
): string {
  const query = new URLSearchParams({ path: value, mode });
  if (extensions.length) query.set("extensions", extensions.join(","));
  return `/paths/complete?${query.toString()}`;
}
