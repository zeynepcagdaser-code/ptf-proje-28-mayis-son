#!/usr/bin/env python3
"""Audit historical interim MCP CSV continuity and quality."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
CSV_PATH = PROJECT_ROOT / "data" / "raw" / "interim_mcp.csv"
REPORT_JSON = PROJECT_ROOT / "reports" / "interim_dataset_audit.json"
REPORT_MD = PROJECT_ROOT / "reports" / "interim_dataset_audit.md"
EXPECTED_COLUMNS = ["date", "hour", "marketTradePrice"]


def load_dataset() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    missing = [column for column in EXPECTED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}; columns={list(df.columns)}")
    return df


def parse_ts(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_convert("Europe/Istanbul")


def build_report(df: pd.DataFrame) -> dict:
    ts = parse_ts(df)
    valid_ts = ts.dropna()

    duplicate_rows = int(df.duplicated(subset=["date", "hour"]).sum())
    null_counts = {column: int(df[column].isna().sum()) for column in EXPECTED_COLUMNS}
    null_rates = {column: float(df[column].isna().mean()) for column in EXPECTED_COLUMNS}

    if valid_ts.empty:
        return {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "csv_path": str(CSV_PATH),
            "rows": int(len(df)),
            "error": "No parseable timestamps",
            "duplicate_rows": duplicate_rows,
            "null_counts": null_counts,
            "null_rates": null_rates,
        }

    ts_hour = valid_ts.dt.floor("h")
    min_ts = ts_hour.min()
    max_ts = ts_hour.max()
    expected_hours = pd.date_range(min_ts, max_ts, freq="h", tz="Europe/Istanbul")
    actual_hours = pd.DatetimeIndex(ts_hour.drop_duplicates().sort_values())
    missing_hours = expected_hours.difference(actual_hours)

    daily_counts = pd.Series(1, index=actual_hours).groupby(actual_hours.normalize()).sum()
    expected_days = pd.date_range(min_ts.normalize(), max_ts.normalize(), freq="D", tz="Europe/Istanbul")
    missing_days = [day.date().isoformat() for day in expected_days if daily_counts.get(day, 0) == 0]
    incomplete_days = {
        day.date().isoformat(): int(count)
        for day, count in daily_counts.items()
        if int(count) != 24
    }

    timezone_offsets = sorted({str(value.utcoffset()) for value in actual_hours.to_pydatetime()})
    hour_mismatch = int((ts_hour.dt.hour.astype(str).str.zfill(2) + ":00" != df.loc[valid_ts.index, "hour"].astype(str)).sum())

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "csv_path": str(CSV_PATH),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "coverage": {
            "start": min_ts.isoformat(),
            "end": max_ts.isoformat(),
            "expected_days": int(len(expected_days)),
            "actual_days": int(len(daily_counts)),
            "expected_hours": int(len(expected_hours)),
            "actual_unique_hours": int(len(actual_hours)),
            "missing_day_count": int(len(missing_days)),
            "missing_hour_count": int(len(missing_hours)),
        },
        "missing_days": missing_days[:500],
        "missing_hours": [hour.isoformat() for hour in missing_hours[:1000]],
        "incomplete_days": dict(list(incomplete_days.items())[:500]),
        "duplicate_rows": duplicate_rows,
        "null_counts": null_counts,
        "null_rates": null_rates,
        "timezone": {
            "offsets": timezone_offsets,
            "hour_label_mismatch_count": hour_mismatch,
        },
        "quality_pass": bool(
            duplicate_rows == 0
            and len(missing_hours) == 0
            and all(count == 0 for count in null_counts.values())
            and hour_mismatch == 0
        ),
    }


def write_markdown(report: dict) -> None:
    coverage = report.get("coverage", {})
    timezone = report.get("timezone", {})
    lines = [
        "# Interim MCP Dataset Audit",
        "",
        f"Generated: {report['generated_at_utc']}",
        f"CSV: `{report['csv_path']}`",
        "",
        "## Summary",
        "",
        f"- Rows: {report['rows']}",
        f"- Coverage: {coverage.get('start')} → {coverage.get('end')}",
        f"- Expected days / actual days: {coverage.get('expected_days')} / {coverage.get('actual_days')}",
        f"- Expected hours / actual unique hours: {coverage.get('expected_hours')} / {coverage.get('actual_unique_hours')}",
        f"- Missing days: {coverage.get('missing_day_count')}",
        f"- Missing hours: {coverage.get('missing_hour_count')}",
        f"- Duplicate rows: {report.get('duplicate_rows')}",
        f"- Quality pass: {report.get('quality_pass')}",
        "",
        "## Null Counts",
        "",
        "| Column | Null count | Null rate |",
        "|---|---:|---:|",
    ]
    for column, count in report.get("null_counts", {}).items():
        rate = report.get("null_rates", {}).get(column, 0.0)
        lines.append(f"| `{column}` | {count} | {rate:.6f} |")

    lines.extend(
        [
            "",
            "## Timezone",
            "",
            f"- Offsets: {timezone.get('offsets')}",
            f"- Hour label mismatches: {timezone.get('hour_label_mismatch_count')}",
        ]
    )

    if report.get("missing_days"):
        lines.extend(["", "## Missing Days", ""])
        lines.extend(f"- {day}" for day in report["missing_days"][:50])

    if report.get("missing_hours"):
        lines.extend(["", "## Missing Hours", ""])
        lines.extend(f"- {hour}" for hour in report["missing_hours"][:50])

    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    df = load_dataset()
    report = build_report(df)
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report)
    print(f"Audit JSON: {REPORT_JSON}")
    print(f"Audit MD: {REPORT_MD}")
    print(f"Quality pass: {report.get('quality_pass')}")


if __name__ == "__main__":
    main()
