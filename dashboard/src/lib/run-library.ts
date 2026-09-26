import type { Run } from "./types";

export type RunSort = "recent" | "name";

export function runStatuses(runs: Run[], selectedStatus: string) {
  const statuses = new Set(runs.map((run) => run.status));
  // A refresh can remove the last run with this status. Keep the filter visible
  // so its zero-result state and the select's displayed value still agree.
  if (selectedStatus !== "all") statuses.add(selectedStatus);
  return [...statuses].sort();
}

/** Search the same names and cache identifiers displayed in the run library. */
export function filterRuns(
  runs: Run[],
  query: string,
  status: string,
  sort: RunSort,
) {
  const search = query.trim().toLowerCase();
  return runs
    .filter(
      (run) =>
        (status === "all" || run.status === status) &&
        `${run.name} ${run.id} ${run.path}`.toLowerCase().includes(search),
    )
    .sort((a, b) => {
      if (sort === "recent") {
        const difference =
          (Date.parse(b.updated_at) || 0) - (Date.parse(a.updated_at) || 0);
        if (difference) return difference;
      }
      return a.name.localeCompare(b.name, undefined, {
        numeric: true,
        sensitivity: "base",
      });
    });
}
