import { useEffect, useState } from "react";
import { api, errorMessage } from "./api";
import type { DataStatus, RunRequest, Settings } from "./types";

/** Only choices that affect raw recording discovery belong in this request. */
export function dataAvailabilityQuery(form: RunRequest): string {
  const settings: Settings = {};
  for (const stage of ["select", "decode"]) {
    const source = form.settings[stage] ?? {};
    const fields =
      stage === "decode"
        ? ["session_list_file", "plot_only"]
        : ["session_list_file"];
    const relevant = Object.fromEntries(
      fields
        .filter((field) => Object.hasOwn(source, field))
        .map((field) => [field, source[field]]),
    );
    if (Object.keys(relevant).length) settings[stage] = relevant;
  }
  return JSON.stringify({
    data_dir: form.data_dir,
    stages: [...form.stages].sort(),
    session_list_file: form.session_list_file,
    trust_unverified_legacy_results: form.trust_unverified_legacy_results,
    settings,
  });
}

/** Debounce typing and ignore responses from requests that have been replaced. */
export function scheduleDataCheck(
  query: string,
  onResult: (result: DataStatus) => void,
  onError: (message: string) => void,
  delay = 300,
): () => void {
  const controller = new AbortController();
  let active = true;
  const timer = setTimeout(async () => {
    try {
      const result = await api<DataStatus>("/data-status", {
        method: "POST",
        body: query,
        signal: controller.signal,
      });
      if (active) onResult(result);
    } catch (error) {
      if (active && !controller.signal.aborted) onError(errorMessage(error));
    }
  }, delay);
  return () => {
    active = false;
    clearTimeout(timer);
    controller.abort();
  };
}

export type DataAvailabilityState = {
  result: DataStatus | null;
  error: string;
  checking: boolean;
};

export function useDataAvailability(form: RunRequest) {
  const query = dataAvailabilityQuery(form);
  const [attempt, setAttempt] = useState(0);
  const [checked, setChecked] = useState<{
    query: string;
    attempt: number;
    result: DataStatus | null;
    error: string;
  } | null>(null);
  useEffect(
    () =>
      scheduleDataCheck(
        query,
        (result) => setChecked({ query, attempt, result, error: "" }),
        (error) => setChecked({ query, attempt, result: null, error }),
      ),
    [query, attempt],
  );
  // Invalidate a changed path synchronously, before the replacement effect runs.
  const sameQuery = checked?.query === query;
  const checking = !sameQuery || checked.attempt !== attempt;
  return {
    result: sameQuery ? checked.result : null,
    error: sameQuery && !checking ? checked.error : "",
    checking,
    recheck: () => setAttempt((value) => value + 1),
  };
}
