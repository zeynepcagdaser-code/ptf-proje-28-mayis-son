from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


def safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.replace(0, np.nan)
    return num / den


def winsorize_numeric(df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    numeric = df.select_dtypes(include=["number"]).columns
    for col in numeric:
        q = df[col].quantile([lower, upper])
        if pd.isna(q.iloc[0]) or pd.isna(q.iloc[1]):
            continue
        df[col] = df[col].clip(lower=q.iloc[0], upper=q.iloc[1])
    return df


def clean_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ts_hour"] = pd.to_datetime(df["ts_hour"], errors="coerce")
    if "hour" not in df.columns:
        df["hour"] = df["ts_hour"].dt.hour
    if "weekday" not in df.columns:
        df["weekday"] = df["ts_hour"].dt.dayofweek
    if "is_weekend2" not in df.columns:
        df["is_weekend2"] = (df["weekday"] >= 5).astype(int)
    if "delta_1h" not in df.columns and "target_1h" in df.columns and "ptf_lag_24" in df.columns:
        df["delta_1h"] = pd.to_numeric(df["target_1h"], errors="coerce") - pd.to_numeric(df["ptf_lag_24"], errors="coerce")
    if "ptf_lag_48" in df.columns and "ptf_lag_24" in df.columns:
        df["ptf_lag_24_diff_48"] = df["ptf_lag_24"] - df["ptf_lag_48"]
        df["ptf_lag_24_pct_48"] = safe_ratio(df["ptf_lag_24_diff_48"], df["ptf_lag_48"])
    if "ptf_lag_168" in df.columns and "ptf_lag_24" in df.columns:
        df["ptf_lag_24_diff_168"] = df["ptf_lag_24"] - df["ptf_lag_168"]
        df["ptf_lag_24_pct_168"] = safe_ratio(df["ptf_lag_24_diff_168"], df["ptf_lag_168"])
    if "kgup_renewable_share" in df.columns and "gas_share" in df.columns:
        df["renewable_vs_gas"] = df["kgup_renewable_share"] - df["gas_share"]
    if "renewable_pressure" in df.columns and "kgup_renewable_share" in df.columns:
        df["renewable_pressure_gap"] = df["kgup_renewable_share"] - df["renewable_pressure"]
    split_values = df["split"].copy() if "split" in df.columns else None
    df = df.apply(pd.to_numeric, errors="coerce")
    if split_values is not None:
        df["split"] = split_values
    df = winsorize_numeric(df)
    return df


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if not c.startswith("target_") and c not in {"ts_hour", "split"}]


def target_column_name(horizon: int) -> str:
    return "target_1h" if horizon == 1 else f"target_{horizon}h"


def make_regressor(objective: str = "regression", alpha: float | None = None) -> LGBMRegressor:
    params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "random_state": 42,
        "objective": objective,
    }
    if objective == "quantile" and alpha is not None:
        params["alpha"] = alpha
    return LGBMRegressor(**params)
