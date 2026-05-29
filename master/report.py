"""Master build reports (JSON + Markdown)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from master.schema import Availability


def compute_missing_stats(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {}
    pct = (df.isna().mean() * 100).round(4)
    return {col: float(pct[col]) for col in pct.index if pct[col] > 0}


def build_master_report(
    df: pd.DataFrame,
    *,
    spine_rows: int,
    column_registry: dict[str, Availability],
    columns_by_dataset: dict[str, list[str]],
    output_path: Path,
) -> dict[str, Any]:
    missing_pct = compute_missing_stats(df)
    availability_counts: dict[str, int] = {}
    for col in df.columns:
        label = column_registry.get(col, "metadata")
        availability_counts[label] = availability_counts.get(label, 0) + 1

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_path": str(output_path),
        "row_count": len(df),
        "spine_row_count": spine_rows,
        "row_count_matches_spine": len(df) == spine_rows,
        "column_count": len(df.columns),
        "ts_hour_unique": int(df["ts_hour"].nunique()) if "ts_hour" in df.columns else None,
        "ts_hour_start": str(df["ts_hour"].min()) if len(df) else None,
        "ts_hour_end": str(df["ts_hour"].max()) if len(df) else None,
        "columns_by_dataset": columns_by_dataset,
        "column_count_by_dataset": {k: len(v) for k, v in columns_by_dataset.items()},
        "availability_by_column": column_registry,
        "availability_column_counts": availability_counts,
        "missing_pct": missing_pct,
        "schema": {
            col: str(dtype)
            for col, dtype in df.dtypes.items()
        },
    }


def write_master_report(report: dict[str, Any], reports_dir: Path) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    json_path = reports_dir / f"master_report_{stamp}.json"
    md_path = reports_dir / f"master_report_{stamp}.md"

    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    md_path.write_text(_to_markdown(report), encoding="utf-8")

    latest_json = reports_dir / "master_report_latest.json"
    latest_md = reports_dir / "master_report_latest.md"
    latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_md.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")

    return json_path, md_path


def _to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Master Dataset Report",
        "",
        f"- **Generated (UTC):** {report['generated_at_utc']}",
        f"- **Output:** `{report['output_path']}`",
        f"- **Rows:** {report['row_count']} (spine: {report['spine_row_count']}, "
        f"match: {report['row_count_matches_spine']})",
        f"- **Columns:** {report['column_count']}",
        f"- **ts_hour unique:** {report['ts_hour_unique']}",
        f"- **ts_hour range:** {report['ts_hour_start']} → {report['ts_hour_end']}",
        "",
        "## Columns by dataset",
        "",
        "| Dataset | Column count |",
        "|---------|-------------:|",
    ]

    for name, count in report.get("column_count_by_dataset", {}).items():
        lines.append(f"| {name} | {count} |")

    lines.extend(["", "## Availability summary", ""])
    for label, count in sorted(report.get("availability_column_counts", {}).items()):
        lines.append(f"- **{label}:** {count} columns")

    lines.extend(["", "## Missing values (% > 0)", ""])
    missing = report.get("missing_pct", {})
    if not missing:
        lines.append("_No missing values._")
    else:
        lines.append("| Column | Missing % | Availability |")
        lines.append("|--------|----------:|----------------|")
        registry: dict[str, str] = report.get("availability_by_column", {})
        for col, pct in sorted(missing.items(), key=lambda x: -x[1]):
            lines.append(f"| `{col}` | {pct:.4f} | {registry.get(col, 'n/a')} |")

    lines.extend(["", "## Schema", "", "| Column | dtype | Availability |", "|--------|-------|----------------|"])
    registry = report.get("availability_by_column", {})
    for col, dtype in report.get("schema", {}).items():
        lines.append(f"| `{col}` | {dtype} | {registry.get(col, 'n/a')} |")

    return "\n".join(lines)
