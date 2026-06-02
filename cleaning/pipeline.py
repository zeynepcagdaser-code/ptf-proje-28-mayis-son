"""Orchestrate cleaning of all raw CSV files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from cleaning.config import (
    CLEAN_DATA_DIR,
    HOURLY_CSV_SOURCES,
    OUTAGES_CSV,
    REPORTS_DIR,
    WIND_CSV,
)
from cleaning.hourly_csv import clean_hourly_csv
from cleaning.outages import clean_outages_csv
from cleaning.report import build_report, write_report
from cleaning.wind import clean_wind_csv

CLEANING_RULES = [
    "PTF zero prices are never removed, modified, or imputed.",
    "Price spikes are never removed.",
    "No interpolation, bfill, or centered rolling.",
    "All outputs use ts_hour in Europe/Istanbul.",
    "Wind: 10-minute → hourly aggregation (mean/min/max/std).",
    "Outages: event rows → hourly aggregates.",
]


def _save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    from src.utils.safe_io import atomic_parquet_write
    atomic_parquet_write(df, str(path), index=False)


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    return pd.read_csv(path, low_memory=False)


def run_pipeline(
    *,
    raw_dir: Path | None = None,
    clean_dir: Path | None = None,
    reports_dir: Path | None = None,
) -> dict[str, Any]:
    out_dir = clean_dir or CLEAN_DATA_DIR
    report_dir = reports_dir or REPORTS_DIR
    data_dir = raw_dir or (out_dir.parent)

    hourly_sources = {}
    for name, spec in HOURLY_CSV_SOURCES.items():
        hourly_sources[name] = {**spec, "path": data_dir / spec["path"].name}

    wind_csv = data_dir / WIND_CSV.name
    outages_csv = data_dir / OUTAGES_CSV.name

    out_dir.mkdir(parents=True, exist_ok=True)

    datasets: dict[str, Any] = {}

    def save_hourly(name: str, spec: dict[str, Any]) -> dict[str, Any]:
        path: Path = spec["path"]
        df = _load_csv(path)
        cleaned, stats = clean_hourly_csv(
            df,
            dedupe_keys=spec["dedupe_keys"],
            drop_columns=spec.get("drop_columns"),
            price_columns=spec.get("price_columns"),
            clip_non_negative=spec.get("clip_non_negative"),
        )
        out_path = out_dir / f"{name}_hourly.parquet"
        _save_parquet(cleaned, out_path)
        return {
            "output_path": str(out_path),
            "rows_in": stats["rows_in"],
            "rows_out": stats["rows_out"],
            "ts_min": stats["ts_min"],
            "ts_max": stats["ts_max"],
            "stats": stats,
        }

    for name, spec in hourly_sources.items():
        try:
            datasets[name] = save_hourly(name, spec)
        except Exception as exc:
            datasets[name] = {"error": str(exc)}

    try:
        df = _load_csv(wind_csv)
        cleaned, stats = clean_wind_csv(df)
        out_path = out_dir / "wind_hourly.parquet"
        _save_parquet(cleaned, out_path)
        datasets["wind"] = {
            "output_path": str(out_path),
            "rows_in": stats["rows_in"],
            "rows_out": stats["rows_out"],
            "ts_min": stats["ts_min"],
            "ts_max": stats["ts_max"],
            "stats": stats,
        }
    except Exception as exc:
        datasets["wind"] = {"error": str(exc)}

    try:
        df = _load_csv(outages_csv)
        cleaned, stats = clean_outages_csv(df)
        out_path = out_dir / "outages_hourly.parquet"
        _save_parquet(cleaned, out_path)
        datasets["outages"] = {
            "output_path": str(out_path),
            "rows_in": stats["rows_in"],
            "rows_out": stats["rows_out"],
            "ts_min": stats["ts_min"],
            "ts_max": stats["ts_max"],
            "stats": stats,
        }
    except Exception as exc:
        datasets["outages"] = {"error": str(exc)}

    report = build_report(
        datasets,
        output_dir=out_dir,
        rules=CLEANING_RULES,
    )
    json_path, md_path = write_report(report, report_dir)
    report["report_json"] = str(json_path)
    report["report_md"] = str(md_path)
    return report


if __name__ == "__main__":
    result = run_pipeline()
    print("Cleaning finished.")
    print("Report:", result.get("report_json"))
