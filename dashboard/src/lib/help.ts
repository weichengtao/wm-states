import links from "./help-links.json";

export type WorkspacePage = "results" | "configure" | "compare" | "monitor";

/** Keep the deployment's base path, including a GitHub Pages repository prefix. */
export function docsHref(
  path = "",
  base = import.meta.env.VITE_DOCS_BASE_URL || "/docs/",
): string {
  let normalized = base.trim();
  if (/^https?:\/\//i.test(normalized)) {
    try {
      const url = new URL(normalized);
      url.hash = "";
      url.search = "";
      normalized = url.href;
    } catch {
      normalized = "/docs/";
    }
  } else if (!normalized.startsWith("/") || normalized.startsWith("//")) {
    normalized = "/docs/";
  } else {
    normalized = normalized.split(/[?#]/, 1)[0];
  }
  return `${normalized.replace(/\/+$/, "")}/${path.replace(/^\/+/, "")}`;
}

export const stageHelp = links.stageHelp;
export const pageHelp: Record<
  WorkspacePage,
  { label: string; summary: string; path: string }
> = links.pageHelp;
export const helpTopics = links.helpTopics;

export function stageMethodsPath(stage: string): string {
  return stageHelp.find((item) => item.id === stage)?.path ?? "next/methods/";
}

export function fieldHelpPath(stage: string, field: string): string | null {
  if (field === "preserve_null_time_structure")
    return links.fieldLinks.preserve_null_time_structure;
  if (field === "n_decode_shuffle") return links.fieldLinks.n_decode_shuffle;
  if (field === "resume") return links.fieldLinks.resume;
  if (["session_list_file", "max_sessions_to_run"].includes(field))
    return links.fieldLinks.session_selection;
  if (stage === "select")
    return field.includes("diagnostic")
      ? links.fieldLinks.screening_diagnostics
      : links.fieldLinks.screening_checks;
  if (
    field.startsWith("cv_") ||
    ["max_iterations", "significance_alpha"].includes(field)
  )
    return links.fieldLinks.mixed_models;
  if (
    [
      "pev_weighted_average",
      "active_threshold",
      "active_percentiles",
      "history_alpha",
      "show_principal_components",
      "compare_with_max_off_state",
    ].includes(field)
  )
    return stageMethodsPath(stage);
  if (
    stage === "decode" &&
    [
      "grid_search_for_c",
      "classifier_c",
      "decoder_model",
      "cells_used_for_decoder",
      "balance_decoder_training_trials",
      "svm_kernel",
      "min_cell_per_group",
      "min_trials_good_session",
      "t_decode_start",
      "t_decode_end",
      "t_decode_window",
      "t_decode_step",
    ].includes(field)
  )
    return stageMethodsPath(stage);
  if (stage === "decode" && field.startsWith("logistic_calibration"))
    return stageMethodsPath(stage);
  if (stage === "states" && /^(cc_|cp_|z_threshold|cluster_size)/.test(field))
    return stageMethodsPath(stage);
  return null;
}

export function troubleshootingPath(message = ""): string {
  const match = links.troubleshooting.find(({ pattern }) =>
    new RegExp(pattern, "i").test(message),
  );
  return `next/troubleshooting/${match ? `#${match.anchor}` : ""}`;
}

export function matchesHelp(query: string, ...text: string[]): boolean {
  const haystack = text.join(" ").toLocaleLowerCase().replaceAll("_", " ");
  return query
    .toLocaleLowerCase()
    .replaceAll("_", " ")
    .trim()
    .split(/\s+/)
    .every((word) => haystack.includes(word));
}
