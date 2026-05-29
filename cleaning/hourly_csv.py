"""Clean standard hourly CSV datasets."""

from __future__ import annotations

from typing import Any

import pandas as pd

from cleaning.datetime_utils import make_ts_hour
from cleaning.quality import add_price_flags, clip_with_flag


def clean_hourly_csv(
    df: pd.DataFrame,
    *,
    dedupe_keys: list[str],
    drop_columns: list[str] | None = None,
    price_columns: list[str] | None = None,
    clip_non_negative: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    stats: dict[str, Any] = {"rows_in": len(df)}
    working = df.copy()

    working["ts_hour"] = make_ts_hour(working)
    stats["ts_parse_fail"] = int(working["ts_hour"].isna().sum())

    working = working.dropna(subset=["ts_hour"])
    stats["rows_after_ts_drop"] = len(working)

    if dedupe_keys:
        dup_mask = working.duplicated(subset=dedupe_keys, keep=False)
        stats["duplicate_key_rows"] = int(dup_mask.sum())
        working = working.drop_duplicates(subset=dedupe_keys, keep="last")

    working = working.drop_duplicates(subset=["ts_hour"], keep="last")
    stats["rows_after_dedupe"] = len(working)
    stats["duplicate_ts_hour_removed"] = stats["rows_after_ts_drop"] - stats["rows_after_dedupe"]

    if price_columns:
        working = add_price_flags(working, price_columns)

    if clip_non_negative:
        working = clip_with_flag(working, clip_non_negative, lower=0.0)

    drop_columns = drop_columns or []
    redundant = [c for c in drop_columns if c in working.columns]
    working = working.drop(columns=redundant, errors="ignore")
    stats["dropped_columns"] = redundant

    time_cols = [c for c in ("date", "hour", "time") if c in working.columns]
    working = working.drop(columns=time_cols, errors="ignore")

    cols = ["ts_hour"] + [c for c in working.columns if c != "ts_hour"]
    working = working[cols].sort_values("ts_hour").reset_index(drop=True)

    stats["rows_out"] = len(working)
    stats["ts_min"] = str(working["ts_hour"].min()) if len(working) else None
    stats["ts_max"] = str(working["ts_hour"].max()) if len(working) else None
    stats["numeric_na_pct"] = _numeric_na_pct(working)

    return working, stats


def _numeric_na_pct(df: pd.DataFrame) -> dict[str, float]:
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return {}
    pct = (numeric.isna().mean() * 100).round(4)
    return {k: float(v) for k, v in pct.items() if v > 0}
