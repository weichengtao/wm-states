import type { Session, SessionData } from "./types";

/** Preserve the recording identity across runs, preferring an identical cue. */
export function matchingSession(
  source: Session | undefined,
  sessions: Session[],
) {
  if (!source) return undefined;
  return (
    sessions.find(
      (candidate) =>
        candidate.session === source.session && candidate.cue === source.cue,
    ) ?? sessions.find((candidate) => candidate.session === source.session)
  );
}

export function chooseComparisonSession(
  sessions: Session[],
  previous: string,
  source: Session | undefined,
  mode: "sessions" | "runs",
) {
  if (sessions.some((session) => session.id === previous)) return previous;
  // In one-run mode both requests return the same session list. Its first
  // session is also A's default even if B's response arrives first.
  const preferred =
    mode === "sessions"
      ? sessions.find(
          (session) => session.id !== (source?.id ?? sessions[0]?.id),
        )
      : matchingSession(source, sessions);
  return preferred?.id ?? sessions[0]?.id ?? "";
}

export function comparisonWarnings(a: SessionData, b: SessionData): string[] {
  const warnings: string[] = [];
  if (a.cue !== b.cue)
    warnings.push(
      "Preferred cues differ. The two curves describe different cue populations.",
    );
  if (
    a.session === b.session &&
    [...a.trial_ids].sort((x, y) => x - y).join(",") !==
      [...b.trial_ids].sort((x, y) => x - y).join(",")
  )
    warnings.push("These runs contain different trial sets for this session.");
  if (a.time_bins.join(",") !== b.time_bins.join(","))
    warnings.push(
      "Time grids differ; each curve uses its recorded bin starts.",
    );
  return warnings;
}

import type { Manifest, Settings } from "./types";
export function settingsDifferences(left: Manifest[], right: Manifest[]) {
  function latest(manifests: Manifest[]): Settings {
    const settings: Settings = {};
    // API history is newest first. A partial run only replaces its own stages.
    for (const manifest of manifests)
      for (const [stage, values] of Object.entries(manifest.settings ?? {})) {
        if (!Object.hasOwn(settings, stage)) settings[stage] = values;
      }
    return settings;
  }
  const a = latest(left),
    b = latest(right);
  const rows: {
    stage: string;
    parameter: string;
    left: string;
    right: string;
  }[] = [];
  for (const stage of [
    ...new Set([...Object.keys(a), ...Object.keys(b)]),
  ].sort()) {
    for (const parameter of [
      ...new Set([
        ...Object.keys(a[stage] ?? {}),
        ...Object.keys(b[stage] ?? {}),
      ]),
    ].sort()) {
      if (["cache_dir", "data_dir"].includes(parameter)) continue;
      const av = JSON.stringify(a[stage]?.[parameter]) ?? "Not recorded",
        bv = JSON.stringify(b[stage]?.[parameter]) ?? "Not recorded";
      if (av !== bv) rows.push({ stage, parameter, left: av, right: bv });
    }
  }
  return rows;
}
