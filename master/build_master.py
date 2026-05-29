"""Build master_hourly_v1 from cleaned parquet files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from master.report import build_master_report, write_master_report
from master.schema import (
    CLEAN_DATA_DIR,
    JOIN_DATASETS,
    JOIN_KEY,
    MASTER_DATA_DIR,
    MASTER_OUTPUT,
    REPORTS_DIR,
    SPINE_FILE,
    SPINE_SPEC,
    build_column_registry,
    prefixed_name,
)
from master.schema import DatasetSpec


def _load_and_prefix(df_path: Path, spec: DatasetSpec) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_parquet(df_path)

    if JOIN_KEY not in df.columns:
        raise ValueError(f"{df_path.name} missing {JOIN_KEY}")

    if df[JOIN_KEY].duplicated().any():
        dup_count = int(df[JOIN_KEY].duplicated().sum())
        raise ValueError(f"{df_path.name} has {dup_count} duplicate {JOIN_KEY} values")

    value_columns = [col for col in df.columns if col != JOIN_KEY]
    rename_map = {col: prefixed_name(spec, col) for col in value_columns}
    df = df.rename(columns=rename_map)
    return df, value_columns


def _dedupe_master_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate column names, keeping the first occurrence."""
    if df.columns.duplicated().any():
        dup_names = df.columns[df.columns.duplicated()].unique().tolist()
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
        return df, dup_names
    return df, []


def build_master_dataframe(
    *,
    clean_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    clean_dir = clean_dir or CLEAN_DATA_DIR
    spine_path = clean_dir / SPINE_FILE.name

    spine_df, spine_value_cols = _load_and_prefix(spine_path, SPINE_SPEC)
    spine_rows = len(spine_df)
    master = spine_df.sort_values(JOIN_KEY).reset_index(drop=True)

    columns_by_dataset: dict[str, list[str]] = {
        SPINE_SPEC.name: [JOIN_KEY]
        + [prefixed_name(SPINE_SPEC, c) for c in spine_value_cols],
    }
    joined_specs: list[tuple[DatasetSpec, list[str]]] = []

    join_stats: dict[str, Any] = {}

    for spec in JOIN_DATASETS:
        path = clean_dir / spec.filename
        if not path.exists():
            raise FileNotFoundError(f"Missing cleaned file: {path}")

        side, original_cols = _load_and_prefix(path, spec)
        side_cols = [prefixed_name(spec, c) for c in original_cols]
        rows_before = len(master)

        master = master.merge(side, on=JOIN_KEY, how="left", suffixes=("", "_dup"))
        master, removed_dups = _dedupe_master_columns(master)

        matched = int(master[side_cols[0]].notna().sum()) if side_cols else 0
        join_stats[spec.name] = {
            "side_rows": len(side),
            "matched_rows": matched,
            "match_pct": round(100 * matched / spine_rows, 4) if spine_rows else 0,
            "columns_added": side_cols,
            "duplicate_columns_removed": removed_dups,
        }

        columns_by_dataset[spec.name] = side_cols
        joined_specs.append((spec, original_cols))

    if len(master) != spine_rows:
        raise RuntimeError(
            f"Row count changed after joins: spine={spine_rows}, master={len(master)}"
        )

    if master[JOIN_KEY].duplicated().any():
        raise RuntimeError("ts_hour is not unique in master dataset")

    metadata = {
        "spine_rows": spine_rows,
        "spine_value_cols": spine_value_cols,
        "columns_by_dataset": columns_by_dataset,
        "join_stats": join_stats,
        "joined_specs": joined_specs,
    }
    return master, metadata


def run_build(
    *,
    clean_dir: Path | None = None,
    output_path: Path | None = None,
    reports_dir: Path | None = None,
) -> dict[str, Any]:
    output_path = output_path or MASTER_OUTPUT
    reports_dir = reports_dir or REPORTS_DIR
    output_path.parent.mkdir(parents=True, exist_ok=True)

    master, metadata = build_master_dataframe(clean_dir=clean_dir)

    column_registry = build_column_registry(
        metadata["spine_value_cols"],
        metadata["joined_specs"],
    )

    master.to_parquet(output_path, index=False)

    report = build_master_report(
        master,
        spine_rows=metadata["spine_rows"],
        column_registry=column_registry,
        columns_by_dataset=metadata["columns_by_dataset"],
        output_path=output_path,
    )
    report["join_stats"] = metadata["join_stats"]

    json_path, md_path = write_master_report(report, reports_dir)
    report["report_json"] = str(json_path)
    report["report_md"] = str(md_path)

    return report
