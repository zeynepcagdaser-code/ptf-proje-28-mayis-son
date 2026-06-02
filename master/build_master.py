"""Build master_hourly_v1 from cleaned parquet files."""

from __future__ import annotations

import sys
from pathlib import Path
from io import StringIO
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


def _load_open_meteo_temperature_csv(path: Path) -> pd.DataFrame:
    """
    Open-Meteo archive CSV includes a small metadata section before the actual time series table.
    We parse from the first line that starts with 'time,' and normalize to:
      ts_hour, temp_2m, apparent_temp
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith("time,"):
            start = i
            break
    if start is None:
        raise ValueError(f"Could not find table header 'time,' in {path}")
    csv_text = "\n".join(lines[start:])
    df = pd.read_csv(StringIO(csv_text))

    rename = {}
    for c in df.columns:
        if c == "time":
            rename[c] = JOIN_KEY
        elif c.startswith("temperature_2m"):
            rename[c] = "temp_2m"
        elif c.startswith("apparent_temperature"):
            rename[c] = "apparent_temp"
    df = df.rename(columns=rename)

    if JOIN_KEY not in df.columns:
        raise ValueError(f"Temperature CSV missing {JOIN_KEY} after rename. cols={df.columns.tolist()}")

    df[JOIN_KEY] = pd.to_datetime(df[JOIN_KEY], errors="coerce")
    if df[JOIN_KEY].dt.tz is None:
        df[JOIN_KEY] = df[JOIN_KEY].dt.tz_localize("Europe/Istanbul")
    else:
        df[JOIN_KEY] = df[JOIN_KEY].dt.tz_convert("Europe/Istanbul")

    df["temp_2m"] = pd.to_numeric(df.get("temp_2m"), errors="coerce")
    df["apparent_temp"] = pd.to_numeric(df.get("apparent_temp"), errors="coerce")
    df = df.dropna(subset=[JOIN_KEY]).sort_values(JOIN_KEY)
    df = df.drop_duplicates(subset=[JOIN_KEY], keep="last").reset_index(drop=True)
    return df[[JOIN_KEY, "temp_2m", "apparent_temp"]]


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

    # Optional: raw Open-Meteo temperature (Ankara) merge.
    # Kept outside the cleaned parquet join list since it's a raw external source.
    temp_path = (Path(__file__).resolve().parent.parent / "data" / "raw" / "temperature_hourly.csv").resolve()
    if temp_path.exists():
        temp_df = _load_open_meteo_temperature_csv(temp_path)
        master = master.merge(temp_df, on=JOIN_KEY, how="left")
        matched = int(master["temp_2m"].notna().sum())
        join_stats["open_meteo_temperature"] = {
            "side_rows": len(temp_df),
            "matched_rows": matched,
            "match_pct": round(100 * matched / spine_rows, 4) if spine_rows else 0,
            "columns_added": ["temp_2m", "apparent_temp"],
            "duplicate_columns_removed": [],
        }
        columns_by_dataset["open_meteo_temperature"] = ["temp_2m", "apparent_temp"]

    # Optional: external EPİAŞ market tables (processed parquet).
    processed_dir = (Path(__file__).resolve().parent.parent / "data" / "processed").resolve()

    # Hourly processed merges (ts_hour join).
    for name, filename, cols_added in [
        ("dam_price_independent_buy", "dam_price_independent_buy.parquet", ["dam_price_independent_buy_mwh"]),
        ("dam_price_independent_sell", "dam_price_independent_sell.parquet", ["dam_price_independent_sell_mwh"]),
    ]:
        path = processed_dir / filename
        if not path.exists():
            continue
        side = pd.read_parquet(path)
        if JOIN_KEY not in side.columns:
            raise ValueError(f"{path.name} missing {JOIN_KEY}")
        side[JOIN_KEY] = pd.to_datetime(side[JOIN_KEY], errors="coerce")
        if side[JOIN_KEY].dt.tz is None:
            side[JOIN_KEY] = side[JOIN_KEY].dt.tz_localize("Europe/Istanbul")
        else:
            side[JOIN_KEY] = side[JOIN_KEY].dt.tz_convert("Europe/Istanbul")
        side = side.dropna(subset=[JOIN_KEY]).drop_duplicates(subset=[JOIN_KEY], keep="last")
        master = master.merge(side[[JOIN_KEY] + cols_added], on=JOIN_KEY, how="left")
        matched = int(master[cols_added[0]].notna().sum()) if cols_added else 0
        join_stats[name] = {
            "side_rows": int(len(side)),
            "matched_rows": matched,
            "match_pct": round(100 * matched / spine_rows, 4) if spine_rows else 0,
            "columns_added": cols_added,
            "duplicate_columns_removed": [],
        }
        columns_by_dataset[name] = cols_added

    # GRF daily: join by day and forward-fill to hourly.
    grf_path = processed_dir / "grf_daily_reference_price.parquet"
    if grf_path.exists():
        grf = pd.read_parquet(grf_path)
        if "ts_day" not in grf.columns:
            raise ValueError(f"{grf_path.name} missing ts_day")
        grf["ts_day"] = pd.to_datetime(grf["ts_day"], errors="coerce")
        if grf["ts_day"].dt.tz is None:
            grf["ts_day"] = grf["ts_day"].dt.tz_localize("Europe/Istanbul")
        else:
            grf["ts_day"] = grf["ts_day"].dt.tz_convert("Europe/Istanbul")
        grf = grf.dropna(subset=["ts_day"]).drop_duplicates(subset=["ts_day"], keep="last").sort_values("ts_day")

        cols_added = [c for c in ["grf_tl_1000sm3", "grf_usd_1000sm3", "grf_eur_mwh", "grf_usd_mmbtu"] if c in grf.columns]
        if cols_added:
            # Map each hourly ts_hour to its day key.
            day_key = "ts_day"
            master_day = master[[JOIN_KEY]].copy()
            master_day[day_key] = master_day[JOIN_KEY].dt.floor("D")
            merged = master.merge(master_day, on=JOIN_KEY, how="left")
            merged = merged.merge(grf[[day_key] + cols_added], on=day_key, how="left")
            merged = merged.sort_values(JOIN_KEY).reset_index(drop=True)
            merged[cols_added] = merged[cols_added].ffill()
            merged = merged.drop(columns=[day_key])
            master = merged

            matched = int(master[cols_added[0]].notna().sum())
            join_stats["grf_daily_reference_price"] = {
                "side_rows": int(len(grf)),
                "matched_rows": matched,
                "match_pct": round(100 * matched / spine_rows, 4) if spine_rows else 0,
                "columns_added": cols_added,
                "duplicate_columns_removed": [],
                "note": "Joined by day (ts_day) then forward-filled on hourly spine.",
            }
            columns_by_dataset["grf_daily_reference_price"] = cols_added

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

    # Extend availability registry for external processed columns (so report shows labels).
    availability_extra = {
        "dam_price_independent_buy_mwh": "planned",
        "dam_price_independent_sell_mwh": "planned",
        "grf_tl_1000sm3": "realized",
        "grf_usd_1000sm3": "realized",
        "grf_eur_mwh": "realized",
        "grf_usd_mmbtu": "realized",
    }
    report_registry = report.get("availability_by_column", {})
    for col, label in availability_extra.items():
        if col in master.columns:
            report_registry[col] = label
    report["availability_by_column"] = report_registry

    json_path, md_path = write_master_report(report, reports_dir)
    report["report_json"] = str(json_path)
    report["report_md"] = str(md_path)

    return report


if __name__ == "__main__":
    rep = run_build()
    print("Master dataset built.")
    print("Output:", rep.get("output_path"))
    print("Rows:", rep.get("row_count"), "| Columns:", rep.get("column_count"))
    print("ts_hour:", rep.get("ts_hour_start"), "→", rep.get("ts_hour_end"))
    print("JSON report:", rep.get("report_json"))
    print("Markdown report:", rep.get("report_md"))
