"""Aggregate 10-minute wind data to hourly."""

from __future__ import annotations

from typing import Any

import pandas as pd

from cleaning.config import EXPECTED_INTERVALS_PER_HOUR, TIMEZONE
from cleaning.quality import clip_with_flag


def _make_ts_10min(df: pd.DataFrame) -> pd.Series:
    if "time" in df.columns:
        time_strings = df["time"].astype(str)
        if time_strings.str.contains("T", na=False).mean() > 0.9:
            return pd.to_datetime(df["time"], errors="coerce", utc=True).dt.tz_convert(
                TIMEZONE
            )

    if "date" in df.columns:
        date_strings = df["date"].astype(str)
        if date_strings.str.contains("T", na=False).mean() > 0.9:
            return pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_convert(
                TIMEZONE
            )

    raise ValueError("Wind dataset requires ISO timestamps in time or date column")


def clean_wind_csv(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    stats: dict[str, Any] = {"rows_in": len(df)}
    working = df.copy()

    working["ts_10min"] = _make_ts_10min(working)
    working["ts_hour"] = working["ts_10min"].dt.floor("h")
    working = working.dropna(subset=["ts_hour"])
    stats["rows_after_ts_drop"] = len(working)

    if {"date", "time"}.issubset(working.columns):
        stats["duplicate_key_rows"] = int(
            working.duplicated(subset=["date", "time"], keep=False).sum()
        )
        working = working.drop_duplicates(subset=["date", "time"], keep="last")
    stats["rows_after_dedupe"] = len(working)

    value_cols = ["quarter1", "quarter2", "quarter3", "quarter4", "generation", "forecast"]
    present = [c for c in value_cols if c in working.columns]

    agg_dict: dict[str, tuple[str, str]] = {
        "wind_interval_count": ("ts_10min", "count"),
    }
    for col in present:
        agg_dict[f"{col}_mean"] = (col, "mean")
        if col in ("forecast", "generation"):
            agg_dict[f"{col}_min"] = (col, "min")
            agg_dict[f"{col}_max"] = (col, "max")
        if col == "forecast":
            agg_dict[f"{col}_std"] = (col, "std")

    hourly = working.groupby("ts_hour", as_index=False).agg(**agg_dict)
    hourly["is_partial_hour"] = hourly["wind_interval_count"] != EXPECTED_INTERVALS_PER_HOUR

    if "generation_mean" in hourly.columns:
        hourly = clip_with_flag(hourly, ["generation_mean"], lower=0.0)
        hourly = hourly.rename(
            columns={"was_generation_mean_clipped": "was_generation_clipped"}
        )

    hourly = hourly.sort_values("ts_hour").reset_index(drop=True)

    stats["rows_out"] = len(hourly)
    stats["partial_hours"] = int(hourly["is_partial_hour"].sum())
    stats["ts_min"] = str(hourly["ts_hour"].min()) if len(hourly) else None
    stats["ts_max"] = str(hourly["ts_hour"].max()) if len(hourly) else None
    stats["numeric_na_pct"] = {
        c: float(hourly[c].isna().mean() * 100)
        for c in hourly.columns
        if c != "ts_hour" and hourly[c].isna().any()
    }

    return hourly, stats
