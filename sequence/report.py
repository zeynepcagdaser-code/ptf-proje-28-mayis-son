"""Sequence build reports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def build_sequence_report(
    *,
    feature_columns: list[str],
    target_columns: list[str],
    window: int,
    horizon: int,
    shapes: dict[str, dict[str, list[int]]],
    sequence_counts: dict[str, int],
    dropped_nan: dict[str, int],
    dropped_insufficient: dict[str, int],
    scaler_fit_split: str,
    leakage_checklist: list[dict[str, str]],
    output_dir: Path,
    source_path: Path,
) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_features": str(source_path),
        "output_dir": str(output_dir),
        "window_size": window,
        "horizon": horizon,
        "index_mapping": f"X[t-{window - 1}:t] -> y[t+1:t+{horizon}]",
        "feature_count": len(feature_columns),
        "target_count": len(target_columns),
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "shapes": shapes,
        "sequence_counts": sequence_counts,
        "dropped_nan_sequences": dropped_nan,
        "dropped_insufficient_rows": dropped_insufficient,
        "scaler_fit_split": scaler_fit_split,
        "leakage_checklist": leakage_checklist,
    }


def write_sequence_report(
    report: dict[str, Any],
    reports_dir: Path,
    *,
    basename: str = "sequence_report",
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
    lines = [
        "# Sequence Dataset Report",
        "",
        f"- **Generated (UTC):** {report['generated_at_utc']}",
        f"- **Source:** `{report['source_features']}`",
        f"- **Output dir:** `{report['output_dir']}`",
        "",
        "## Configuration",
        "",
        f"- Window: **{report['window_size']}**",
        f"- Horizon: **{report['horizon']}**",
        f"- Mapping: `{report['index_mapping']}`",
        f"- Features: {report['feature_count']}",
        f"- Targets: {report['target_count']}",
        f"- Scaler fit split: **{report['scaler_fit_split']}**",
        "",
        "## Tensor shapes",
        "",
        "| Split | X shape | y shape |",
        "|-------|---------|---------|",
    ]

    for split, shape_info in report.get("shapes", {}).items():
        x_shape = shape_info.get("X", [])
        y_shape = shape_info.get("y", [])
        lines.append(f"| {split} | {tuple(x_shape)} | {tuple(y_shape)} |")

    lines.extend(["", "## Sequence counts", "", "| Split | Sequences |", "|-------|----------:|"])
    for split, count in report.get("sequence_counts", {}).items():
        lines.append(f"| {split} | {count} |")

    lines.extend(
        [
            "",
            "## Dropped sequences (NaN)",
            "",
            "| Split | Dropped |",
            "|-------|--------:|",
        ]
    )
    for split, count in report.get("dropped_nan_sequences", {}).items():
        lines.append(f"| {split} | {count} |")

    lines.extend(["", "## Leakage checklist", ""])
    for item in report.get("leakage_checklist", []):
        detail = item.get("detail", "")
        suffix = f" — {detail}" if detail else ""
        lines.append(f"- [{item.get('status', 'n/a').upper()}] {item.get('check')}{suffix}")

    lines.extend(["", "## Feature columns", ""])
    for col in report.get("feature_columns", []):
        lines.append(f"- `{col}`")

    lines.extend(["", "## Target columns", ""])
    for col in report.get("target_columns", []):
        lines.append(f"- `{col}`")

    return "\n".join(lines)
