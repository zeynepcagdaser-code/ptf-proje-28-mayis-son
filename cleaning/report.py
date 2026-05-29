"""Write cleaning run report (JSON + Markdown)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_report(
    datasets: dict[str, Any],
    *,
    output_dir: Path,
    rules: list[str],
) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "rules_applied": rules,
        "datasets": datasets,
    }


def write_report(report: dict[str, Any], reports_dir: Path) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = reports_dir / f"cleaning_report_{stamp}.json"
    md_path = reports_dir / f"cleaning_report_{stamp}.md"

    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    md_path.write_text(_report_to_markdown(report), encoding="utf-8")

    latest_json = reports_dir / "cleaning_report_latest.json"
    latest_md = reports_dir / "cleaning_report_latest.md"
    latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_md.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")

    return json_path, md_path


def _report_to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Cleaning Report",
        "",
        f"- **Generated (UTC):** {report['generated_at_utc']}",
        f"- **Output directory:** `{report['output_dir']}`",
        "",
        "## Rules applied",
        "",
    ]
    for rule in report.get("rules_applied", []):
        lines.append(f"- {rule}")

    lines.extend(["", "## Datasets", ""])

    for name, info in report.get("datasets", {}).items():
        lines.append(f"### {name}")
        lines.append("")
        if info.get("error"):
            lines.append(f"- **Error:** {info['error']}")
            lines.append("")
            continue

        lines.append(f"- **Output:** `{info.get('output_path', 'n/a')}`")
        lines.append(f"- **Rows in:** {info.get('rows_in', 'n/a')}")
        lines.append(f"- **Rows out:** {info.get('rows_out', 'n/a')}")
        if info.get("ts_min"):
            lines.append(f"- **ts_hour range:** {info['ts_min']} → {info['ts_max']}")
        for key, value in info.get("stats", {}).items():
            if key in ("rows_in", "rows_out", "ts_min", "ts_max"):
                continue
            lines.append(f"- **{key}:** {value}")
        lines.append("")

    return "\n".join(lines)
