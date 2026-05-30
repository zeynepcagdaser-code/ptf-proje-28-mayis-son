"""Feature build reports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def missing_pct(df: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    subset = df[columns]
    pct = (subset.isna().mean() * 100).round(4)
    return {col: float(pct[col]) for col in columns if pct[col] > 0}


def build_features_report(
    *,
    df: pd.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
    rows_master: int,
    rows_dropped_targets: int,
    rows_dropped_history: int,
    missing_before_ffill: dict[str, float],
    missing_after_ffill: dict[str, float],
    split_counts: dict[str, int],
    output_path: Path,
    leakage_checklist: list[dict[str, str]],
    training_format: dict[str, Any],
) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_path": str(output_path),
        "source_master": "data/master/master_hourly_v1.parquet",
        "training_format": training_format,
        "row_count": len(df),
        "rows_master": rows_master,
        "rows_dropped_missing_targets": rows_dropped_targets,
        "rows_dropped_insufficient_history": rows_dropped_history,
        "feature_count": len(feature_columns),
        "target_count": len(target_columns),
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "ts_hour_start": str(df["ts_hour"].min()) if len(df) else None,
        "ts_hour_end": str(df["ts_hour"].max()) if len(df) else None,
        "split_counts": split_counts,
        "missing_pct_before_ffill": missing_before_ffill,
        "missing_pct_after_ffill": missing_after_ffill,
        "ffill_applied": True,
        "ffill_limit": 2,
        "leakage_checklist": leakage_checklist,
    }


def write_features_report(
    report: dict[str, Any],
    reports_dir: Path,
    *,
    basename: str = "features_report",
) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    json_path = reports_dir / f"{basename}_{stamp}.json"
    md_path = reports_dir / f"{basename}_{stamp}.md"

    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    md_path.write_text(_to_markdown(report), encoding="utf-8")

    latest_json = reports_dir / f"{basename}_latest.json"
    latest_md = reports_dir / f"{basename}_latest.md"
    latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_md.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")

    return json_path, md_path


def _to_markdown(report: dict[str, Any]) -> str:
    tf = report.get("training_format", {})
    lines = [
        "# LSTM Feature Dataset Report",
        "",
        f"- **Generated (UTC):** {report['generated_at_utc']}",
        f"- **Output:** `{report['output_path']}`",
        f"- **Source:** `{report['source_master']}`",
        "",
        "## Training format (reference)",
        "",
        f"- **Input window:** {tf.get('input_window')} hours",
        f"- **Output horizon:** {tf.get('output_horizon')} hours",
        f"- **Index mapping:** `{tf.get('index_mapping')}`",
        "",
        "## Row counts",
        "",
        f"- Master rows: {report['rows_master']}",
        f"- Dropped (missing targets): {report['rows_dropped_missing_targets']}",
        f"- Dropped (insufficient history for lags/rolls): {report['rows_dropped_insufficient_history']}",
        f"- **Final rows:** {report['row_count']}",
        "",
        "## Features & targets",
        "",
        f"- Feature columns: {report['feature_count']}",
        f"- Target columns: {report['target_count']}",
        f"- ts_hour range: {report['ts_hour_start']} → {report['ts_hour_end']}",
        "",
        "## Split counts",
        "",
        "| Split | Rows |",
        "|-------|-----:|",
    ]

    for split, count in report.get("split_counts", {}).items():
        lines.append(f"| {split} | {count} |")

    lines.extend(["", "## Missing features (before ffill, % > 0)", ""])
    before = report.get("missing_pct_before_ffill", {})
    if before:
        for col, pct in sorted(before.items(), key=lambda x: -x[1]):
            lines.append(f"- `{col}`: {pct:.4f}%")
    else:
        lines.append("_None._")

    lines.extend(["", "## Missing features (after ffill limit=2, % > 0)", ""])
    after = report.get("missing_pct_after_ffill", {})
    if after:
        for col, pct in sorted(after.items(), key=lambda x: -x[1]):
            lines.append(f"- `{col}`: {pct:.4f}%")
    else:
        lines.append("_None._")

    lines.extend(["", "## Leakage checklist", ""])
    for item in report.get("leakage_checklist", []):
        lines.append(
            f"- [{item.get('status', 'n/a').upper()}] {item.get('check')}: {item.get('detail')}"
        )

    return "\n".join(lines)
