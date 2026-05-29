"""Canonical ts_hour in Europe/Istanbul."""

from __future__ import annotations

import pandas as pd

from cleaning.config import TIMEZONE


def _time_column_name(df: pd.DataFrame) -> str | None:
    if "time" in df.columns:
        return "time"
    if "hour" in df.columns:
        return "hour"
    return None


def make_ts_hour(df: pd.DataFrame, date_col: str = "date") -> pd.Series:
    """
    Build timezone-aware hourly timestamps.

    Prefer full ISO strings in `date`. If only calendar dates exist, combine
  with HH:MM in `time` or `hour`. Never parse HH:MM alone without a date.
    """
    if date_col not in df.columns:
        raise ValueError(f"Missing {date_col} column")

    date_strings = df[date_col].astype(str)
    has_full_timestamp = date_strings.str.contains("T", na=False).mean() > 0.9

    if has_full_timestamp:
        ts = pd.to_datetime(df[date_col], errors="coerce", utc=True)
        return ts.dt.tz_convert(TIMEZONE).dt.floor("h")

    time_col = _time_column_name(df)
    if time_col is None:
        ts = pd.to_datetime(df[date_col], errors="coerce", utc=True)
        return ts.dt.tz_convert(TIMEZONE).dt.floor("h")

    time_strings = df[time_col].astype(str)
    if time_strings.str.contains("T", na=False).mean() > 0.9:
        ts = pd.to_datetime(df[time_col], errors="coerce", utc=True)
        return ts.dt.tz_convert(TIMEZONE).dt.floor("h")

    date_only = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    combined = pd.to_datetime(
        date_only.dt.strftime("%Y-%m-%d") + " " + df[time_col].astype(str),
        errors="coerce",
        utc=True,
    )
    return combined.dt.tz_convert(TIMEZONE).dt.floor("h")


def parse_event_timestamps(series: pd.Series) -> pd.Series:
    """Parse outage event timestamps to ts_hour timezone."""
    ts = pd.to_datetime(series, errors="coerce", utc=True)
    return ts.dt.tz_convert(TIMEZONE)
