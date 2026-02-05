from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tyro

matplotlib.use("Agg")


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
    dpi: int = 200


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
            f"session={session} | window=[{trial_start},{trial_end}) | holdout={holdout_label}"
        )

        # Make the figure wide enough so labels remain readable for typical reason counts.
        fig_w = max(10.0, 0.55 * max(1, len(reasons)))
        x = np.arange(len(reasons), dtype=np.int64)

        fig, (ax_total, ax_unique) = plt.subplots(
            2,
            1,
            figsize=(fig_w, 7.5),
            sharex=True,
            layout="constrained",
        )

        ax_total.bar(x, counts, color="#4C78A8")
        ax_total.set_title(title)
        ax_total.set_ylabel("Cell count")
        ax_total.set_ylim(bottom=0)
        ax_total.tick_params(axis="x", labelbottom=False)

        ax_unique.bar(x, unique_counts, color="#F58518")
        ax_unique.set_ylabel("Unique cell count")
        ax_unique.set_xlabel("Reject reason")
        ax_unique.set_xticks(x)
        ax_unique.set_xticklabels(reasons, rotation=45, ha="right")
        ax_unique.set_ylim(bottom=0)

        # Right axis is percent of cells in the partition (denominator is always total cells).
        ax_total2 = ax_total.twinx()
        ax_total2.set_ylim(ax_total.get_ylim())
        ticks = ax_total.get_yticks()
        ax_total2.set_yticks(ticks)
        ax_total2.set_yticklabels([f"{(t / n_cells) * 100.0:.1f}" for t in ticks])
        ax_total2.set_ylabel("Percent of cells (%)")

        ax_unique2 = ax_unique.twinx()
        ax_unique2.set_ylim(ax_unique.get_ylim())
        ticks = ax_unique.get_yticks()
        ax_unique2.set_yticks(ticks)
        ax_unique2.set_yticklabels([f"{(t / n_cells) * 100.0:.1f}" for t in ticks])
        ax_unique2.set_ylabel("Percent of cells (%)")

        file_name = (
            f"session_{session}__trial_{trial_start}_{trial_end}__holdout_{holdout_label}.png"
        )
        fig.savefig(output_dir / file_name, dpi=config.dpi)
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
