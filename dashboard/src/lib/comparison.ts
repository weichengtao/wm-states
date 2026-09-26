import type { SessionData } from "./types";
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
