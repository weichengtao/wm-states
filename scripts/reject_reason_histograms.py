from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import tyro

try:
    from scripts.figure_exports import configure_figure_style, save_figure_all_formats
except ModuleNotFoundError:
    from figure_exports import configure_figure_style, save_figure_all_formats

matplotlib.use("Agg")
configure_figure_style(matplotlib)

REASON_LABELS = {
    "pass": "Pass",
    "fail_sig_pev": "PEV",
    "fail_min_presence_ratio": "Presence\nratio",
    "fail_min_fr_test": "Firing\nrate",
    "fail_temp_dep_stage1": "Var ratio\n(delay / baseline)",
    "fail_temp_dep_stage2": "Var ratio\n(sliding / full)",
    "fail_temp_dep_stage3": "|r| of\ndelay",
    "fail_temp_dep_stage3_baseline": "|r| of\nbaseline",
}
NOT_APPLICABLE_SUFFIX = "_not_applicable"


def split_rejection_reasons(value: object) -> list[str]:
    """Split a rejection reason string into individual reasons.

    - "a|b|c" -> ["a", "b", "c"]
    - "pass" (or empty/NaN) -> ["pass"]
    """
    if not isinstance(value, str):
        return ["pass"]
    text = value.strip()
    if text == "" or text.lower() == "nan":
        return ["pass"]
    parts = [p.strip() for p in text.split("|") if p.strip()]
    return parts if parts else ["pass"]


def format_reason_label(reason: str) -> str:
    if reason.endswith(NOT_APPLICABLE_SUFFIX):
        base = reason[: -len(NOT_APPLICABLE_SUFFIX)]
        base_label = REASON_LABELS.get(base, base)
        return f"{base_label} N/A"
    return REASON_LABELS.get(reason, reason)


def compute_percent_grid(max_count: float, n_cells: int, target_ticks: int = 6) -> tuple[float, float]:
    if n_cells <= 0:
        return 1.0, 1.0
    max_percent = max(0.0, (float(max_count) / float(n_cells)) * 100.0)
    if max_percent <= 0.0:
        return 1.0, 1.0
    steps = (1.0, 2.0, 5.0, 10.0, 20.0, 25.0, 50.0)
    step = 100.0
    for cand in steps:
        if max_percent / cand <= target_ticks:
            step = cand
            break
    max_percent_rounded = float(np.ceil(max_percent / step) * step)
    return max_percent_rounded, step


def format_holdout_value(trial_holdout: object) -> str:
    if pd.isna(trial_holdout):
        return "none"
    try:
        return str(int(trial_holdout))
    except (TypeError, ValueError):
        return str(trial_holdout)


@dataclass
class Config:
    """Summarize cell rejection diagnostics into per-partition histograms."""

    cache_dir: Path = Path("cache/run_001")
    diagnostics_csv: Path | None = None
    output_dir: Path | None = None
    summary_csv: Path | None = None
    skip_not_applicable: bool = True
    dpi: int = 300


def main(config: Config) -> None:
    cache_dir = config.cache_dir

    diagnostics_csv = config.diagnostics_csv
    if diagnostics_csv is None:
        diagnostics_csv = cache_dir / "diagnostics" / "cell_rejection_diagnostics.csv"
    if not diagnostics_csv.exists():
        raise FileNotFoundError(f"Missing diagnostics CSV: {diagnostics_csv}")

    output_dir = config.output_dir
    if output_dir is None:
        output_dir = cache_dir / "diagnostics" / "reject_reason_histograms"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = config.summary_csv
    if summary_csv is None:
        summary_csv = output_dir / "reject_reason_histograms_summary.csv"

    df = pd.read_csv(diagnostics_csv)
    required_cols = [
        "session",
        "trial_start",
        "trial_end",
        "trial_holdout",
        "cell_idx",
        "rejection_reason",
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Diagnostics CSV missing required columns: {missing_cols}. "
            f"Got columns: {sorted(df.columns.tolist())}"
        )

    # Ensure stable grouping keys.
    for col in ("trial_start", "trial_end", "cell_idx"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df["trial_holdout"] = pd.to_numeric(df["trial_holdout"], errors="coerce").astype("Int64")

    group_cols = ["session", "trial_start", "trial_end", "trial_holdout"]
    summary_rows: list[dict[str, object]] = []

    grouped = df.groupby(group_cols, dropna=False, sort=True)
    for (session, trial_start, trial_end, trial_holdout), g in grouped:
        n_cells = int(len(g))
        if n_cells == 0:
            continue

        counter: Counter[str] = Counter()
        unique_counter: Counter[str] = Counter()
        for raw in g["rejection_reason"].tolist():
            # Each cell should contribute at most once per reason, even if the string is malformed.
            reasons = split_rejection_reasons(raw)
            if config.skip_not_applicable:
                reasons = [r for r in reasons if not r.endswith("_not_applicable")]
            if not reasons:
                continue
            uniq = set(reasons)
            for reason in uniq:
                counter[reason] += 1
            if len(uniq) == 1:
                unique_counter[next(iter(uniq))] += 1

        items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        reasons = [r for r, _ in items]
        reason_labels = [format_reason_label(r) for r in reasons]
        counts = np.asarray([c for _, c in items], dtype=np.int64)
        unique_counts = np.asarray([unique_counter.get(r, 0) for r in reasons], dtype=np.int64)

        # Summary rows (tidy / long format).
        for reason, count in items:
            unique_count = unique_counter.get(reason, 0)
            summary_rows.append(
                {
                    "session": session,
                    "trial_start": int(trial_start) if not pd.isna(trial_start) else None,
                    "trial_end": int(trial_end) if not pd.isna(trial_end) else None,
                    "trial_holdout": int(trial_holdout) if not pd.isna(trial_holdout) else None,
                    "reason": reason,
                    "count": int(count),
                    "percent": float(count) / float(n_cells) * 100.0,
                    "unique_count": int(unique_count),
                    "unique_percent": float(unique_count) / float(n_cells) * 100.0,
                    "n_cells": n_cells,
                }
            )

        holdout_label = format_holdout_value(trial_holdout)
        title = (
            f"session = {session} | trial window = [{trial_start},{trial_end}) | trial holdout = {holdout_label}"
        )

        # Scale figure height with number of reasons to keep y-axis labels readable.
        fig_h = max(4.5, 0.42 * max(1, len(reasons)))
        fig_w = 10.0
        y = np.arange(len(reasons), dtype=np.int64)

        fig, (ax_total, ax_unique) = plt.subplots(
            1,
            2,
            figsize=(fig_w, fig_h),
            sharey=True,
            layout="constrained",
        )
        fig.suptitle(title)

        ax_total.barh(y, counts, color="#4C78A8")
        ax_total.set_xlabel("Cell count")
        ax_total.set_ylabel("Rejection reason")
        ax_total.set_yticks(y)
        ax_total.set_yticklabels(reason_labels)
        total_max_count = float(counts.max()) if counts.size else 0.0
        total_max_percent, total_percent_step = compute_percent_grid(total_max_count, n_cells)
        total_xlim_max = max(1.0, (total_max_percent / 100.0) * float(n_cells))
        ax_total.set_xlim(0, total_xlim_max)
        ax_total.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax_total.invert_yaxis()

        ax_unique.barh(y, unique_counts, color="#F58518")
        ax_unique.set_xlabel("Unique cell count")
        unique_max_count = float(unique_counts.max()) if unique_counts.size else 0.0
        unique_max_percent, unique_percent_step = compute_percent_grid(unique_max_count, n_cells)
        unique_xlim_max = max(1.0, (unique_max_percent / 100.0) * float(n_cells))
        ax_unique.set_xlim(0, unique_xlim_max)
        ax_unique.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax_unique.tick_params(axis="y", labelleft=False)

        # Top x-axes show percent of cells in the partition.
        ax_total2 = ax_total.twiny()
        ax_total2.set_xlim(0, total_max_percent)
        total_percent_ticks = np.arange(0.0, total_max_percent + 0.5 * total_percent_step, total_percent_step)
        ax_total2.set_xticks(total_percent_ticks)
        ax_total2.set_xticklabels([f"{int(t)}" for t in total_percent_ticks])
        ax_total2.set_xlabel("Percent of cells (%)")

        ax_unique2 = ax_unique.twiny()
        ax_unique2.set_xlim(0, unique_max_percent)
        unique_percent_ticks = np.arange(
            0.0,
            unique_max_percent + 0.5 * unique_percent_step,
            unique_percent_step,
        )
        ax_unique2.set_xticks(unique_percent_ticks)
        ax_unique2.set_xticklabels([f"{int(t)}" for t in unique_percent_ticks])
        ax_unique2.set_xlabel("Percent of cells (%)")

        file_name = (
            f"session_{session}__trial_{trial_start}_{trial_end}__holdout_{holdout_label}.png"
        )
        save_figure_all_formats(fig, output_dir / file_name, dpi=config.dpi)
        plt.close(fig)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_cols = [
            "session",
            "trial_start",
            "trial_end",
            "trial_holdout",
            "reason",
            "count",
            "percent",
            "unique_count",
            "unique_percent",
            "n_cells",
        ]
        summary_df = summary_df[summary_cols]
    else:
        summary_df = pd.DataFrame(
            columns=[
                "session",
                "trial_start",
                "trial_end",
                "trial_holdout",
                "reason",
                "count",
                "percent",
                "unique_count",
                "unique_percent",
                "n_cells",
            ]
        )
    summary_df.to_csv(summary_csv, index=False)

    print(f"Wrote {len(grouped)} histogram(s) to {output_dir}")
    print(f"Wrote summary CSV to {summary_csv}")


if __name__ == "__main__":
    main(tyro.cli(Config))
