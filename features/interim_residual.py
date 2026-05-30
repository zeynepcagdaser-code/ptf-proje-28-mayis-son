"""Interim MCP baseline + residual targets (leakage-safe interim features)."""

from __future__ import annotations

import pandas as pd

from features.config import OUTPUT_HORIZON, PTF_COL, TARGET_HORIZONS


def csv_to_ts_hour(df: pd.DataFrame, date_col: str = "date") -> pd.Series:
    """Normalize EPİAŞ date (+ optional hour) to ts_hour Europe/Istanbul."""
    ts = pd.to_datetime(df[date_col], utc=True).dt.tz_convert("Europe/Istanbul")
    if "hour" in df.columns and df["hour"].notna().any():
        hour_str = df["hour"].astype(str)
        hour_part = hour_str.str.split(":").str[0].astype(int)
        ts = ts.dt.normalize() + pd.to_timedelta(hour_part, unit="h")
    return ts


def load_interim_prices(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["ts_hour"] = csv_to_ts_hour(df)
    return df[["ts_hour", "marketTradePrice"]].rename(columns={"marketTradePrice": "interim_mcp"})


def load_finalized_prices(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["ts_hour"] = csv_to_ts_hour(df)
    return df[["ts_hour", "price"]].rename(columns={"price": "finalized_mcp"})


def add_interim_anchor_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features known at anchor ts_hour t (no same-hour finalized in X)."""
    out = df.copy()
    im = out["interim_mcp"]
    im_l1 = im.shift(1)
    final = out[PTF_COL]

    out["interim_ptf_change_1h"] = im_l1 - im.shift(2)
    out["interim_ptf_change_24h"] = im_l1 - im.shift(25)
    out["interim_final_spread_lag_24"] = (final - im).shift(24)
    out["interim_volatility_24h"] = im_l1.rolling(24, min_periods=12).std()
    return out


def add_interim_baselines_and_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """
    At anchor t:
      interim_baseline_h = interim_mcp(t+h)
      target_h = finalized_mcp(t+h)  (must already exist)
      target_residual_h = target_h - interim_baseline_h
    """
    out = df.copy()
    for h in TARGET_HORIZONS:
        tcol = f"target_{h}h"
        bcol = f"interim_baseline_{h}h"
        rcol = f"target_residual_{h}h"
        out[bcol] = out["interim_mcp"].shift(-h)
        out[rcol] = out[tcol] - out[bcol]
    return out


INTERIM_FEATURE_COLUMNS = [
    "interim_mcp",
    "interim_ptf_change_1h",
    "interim_ptf_change_24h",
    "interim_final_spread_lag_24",
    "interim_volatility_24h",
]


def list_interim_baseline_columns() -> list[str]:
    return [f"interim_baseline_{h}h" for h in TARGET_HORIZONS]


def list_interim_residual_columns() -> list[str]:
    return [f"target_residual_{h}h" for h in TARGET_HORIZONS]


def exclude_from_model_features(columns: list[str]) -> list[str]:
    drop = set(
        [f"target_{h}h" for h in TARGET_HORIZONS]
        + list_interim_baseline_columns()
        + list_interim_residual_columns()
        + ["finalized_mcp", "persistence_1h"]  # persistence may exist from tree build
    )
    drop.update(c for c in columns if c.startswith("persistence_"))
    return [c for c in columns if c not in drop and c not in {"ts_hour", "split", "anchor_hour"}]
