"""Feature and target builders (leakage-safe)."""

from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

from features.config import (
    KGUP_FEATURE_COLS,
    LAG_STEPS,
    LAGGED_SOURCE_COLS,
    LOAD_FEATURE_COLS,
    OUTAGE_FEATURE_COLS,
    OUTPUT_HORIZON,
    PTF_COL,
    SMF_COL,
    TARGET_HORIZONS,
    WIND_FORECAST_COLS,
)


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for h in TARGET_HORIZONS:
        out[f"target_{h}h"] = out[PTF_COL].shift(-h)
    return out


def add_ptf_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ptf = out[PTF_COL]

    out["ptf_lag_1"] = ptf.shift(1)
    out["ptf_lag_24"] = ptf.shift(24)
    out["ptf_lag_48"] = ptf.shift(48)
    out["ptf_lag_168"] = ptf.shift(168)

    ptf_past = ptf.shift(1)
    out["ptf_roll_mean_24"] = ptf_past.rolling(24, min_periods=24).mean()
    out["ptf_roll_std_24"] = ptf_past.rolling(24, min_periods=24).std()
    out["ptf_roll_mean_168"] = ptf_past.rolling(168, min_periods=168).mean()
    out["ptf_roll_std_168"] = ptf_past.rolling(168, min_periods=168).std()

    return out


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = out["ts_hour"]
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert("Europe/Istanbul")

    hour = ts.dt.hour
    dow = ts.dt.dayofweek
    month = ts.dt.month

    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)
    out["is_weekend"] = (dow >= 5).astype(int)

    return out


def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turkish public holidays from ts_hour (calendar-only, no leakage)."""
    out = df.copy()
    ts = out["ts_hour"]
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert("Europe/Istanbul")

    dates = ts.dt.date
    years = range(int(dates.min().year), int(dates.max().year) + 1)
    tr_holidays = holidays.Turkey(years=years)

    out["is_holiday_tr"] = dates.map(lambda d: 1 if d in tr_holidays else 0).astype(int)

    if "is_weekend" not in out.columns:
        raise ValueError("is_weekend must exist before is_holiday_or_weekend")

    out["is_holiday_or_weekend"] = (
        (out["is_holiday_tr"] == 1) | (out["is_weekend"] == 1)
    ).astype(int)

    return out


def add_spread_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    spread = out[SMF_COL] - out[PTF_COL]
    out["smf_ptf_spread_lag_24"] = spread.shift(24)
    out["smf_ptf_spread_lag_168"] = spread.shift(168)
    return out


def add_supply_demand_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    total = out["kgup_toplam"]
    renewable = (
        out["kgup_ruzgar"]
        + out["kgup_gunes"]
        + out["kgup_barajli"]
        + out["kgup_akarsu"]
        + out["kgup_biokutle"]
    )
    thermal = (
        out["kgup_dogalgaz"]
        + out["kgup_linyit"]
        + out["kgup_tasKomur"]
        + out["kgup_ithalKomur"]
        + out["kgup_fuelOil"]
        + out["kgup_nafta"]
        + out["kgup_diger"]
    )

    out["kgup_total_minus_load"] = total - out["load_lep"]
    out["kgup_renewable_share"] = np.where(total > 0, renewable / total, np.nan)
    out["kgup_thermal_share"] = np.where(total > 0, thermal / total, np.nan)
    out["wind_forecast_share"] = np.where(
        total > 0, out["wind_forecast_mean"] / total, np.nan
    )
    return out


def add_lagged_realized_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in LAGGED_SOURCE_COLS:
        if col not in out.columns:
            continue
        for lag in LAG_STEPS:
            safe_name = col.replace("smf_systemMarginalPrice", "smf")
            out[f"{safe_name}_lag_{lag}"] = out[col].shift(lag)
    return out


def list_engineered_feature_columns() -> list[str]:
    calendar = [
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "is_weekend",
        "is_holiday_tr",
        "is_holiday_or_weekend",
    ]
    ptf_lags = [
        "ptf_lag_1",
        "ptf_lag_24",
        "ptf_lag_48",
        "ptf_lag_168",
        "ptf_roll_mean_24",
        "ptf_roll_std_24",
        "ptf_roll_mean_168",
        "ptf_roll_std_168",
    ]
    spread = ["smf_ptf_spread_lag_24", "smf_ptf_spread_lag_168"]
    supply = [
        "kgup_total_minus_load",
        "kgup_renewable_share",
        "kgup_thermal_share",
        "wind_forecast_share",
    ]

    lagged = []
    for col in LAGGED_SOURCE_COLS:
        safe_name = col.replace("smf_systemMarginalPrice", "smf")
        for lag in LAG_STEPS:
            lagged.append(f"{safe_name}_lag_{lag}")

    return (
        calendar
        + ptf_lags
        + spread
        + supply
        + KGUP_FEATURE_COLS
        + LOAD_FEATURE_COLS
        + WIND_FORECAST_COLS
        + OUTAGE_FEATURE_COLS
        + lagged
    )


def list_target_columns() -> list[str]:
    return [f"target_{h}h" for h in TARGET_HORIZONS]


def assign_split(ts_hour: pd.Series) -> pd.Series:
    years = ts_hour.dt.tz_convert("Europe/Istanbul").dt.year
    split = pd.Series(index=ts_hour.index, dtype="object")
    split[(years >= 2020) & (years <= 2024)] = "train"
    split[years == 2025] = "validation"
    split[years == 2026] = "test"
    return split
