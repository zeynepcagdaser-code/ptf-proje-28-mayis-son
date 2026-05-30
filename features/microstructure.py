"""Market microstructure features (leakage-safe, anchor-time t only)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.config import PTF_COL, SMF_COL


def add_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ptf = out[PTF_COL]
    ptf_l1 = ptf.shift(1)
    ptf_l24 = ptf.shift(24)

    # --- Price dynamics ---
    out["ptf_return_1h"] = (ptf_l1 - ptf.shift(2)) / ptf.shift(2).clip(lower=1.0)
    out["ptf_return_24h"] = (ptf_l1 - ptf_l24) / ptf_l24.clip(lower=1.0)
    out["ptf_momentum_24h"] = ptf_l1 - out.get("ptf_roll_mean_24", ptf_l1.rolling(24).mean())
    std24 = out.get("ptf_roll_std_24", ptf_l1.rolling(24).std())
    out["ptf_zscore_24h"] = out["ptf_momentum_24h"] / std24.replace(0, np.nan)
    out["ptf_range_24h"] = ptf_l1.rolling(24, min_periods=24).max() - ptf_l1.rolling(
        24, min_periods=24
    ).min()
    out["ptf_vol_ratio_24_168"] = std24 / out.get("ptf_roll_std_168", std24).replace(0, np.nan)

    # --- SMF–PTF microstructure ---
    if SMF_COL in out.columns:
        spread = out[SMF_COL] - ptf
        out["spread_nowcast_lag1"] = spread.shift(1)
        out["spread_change_24h"] = spread.shift(1) - spread.shift(24)
        spread_l24 = spread.shift(24)
        out["spread_zscore_24h"] = (spread.shift(1) - spread_l24) / spread_l24.abs().clip(
            lower=50
        ).rolling(24, min_periods=12).std()

    # --- Supply / demand stress ---
    if "kgup_toplam" in out.columns and "load_lep" in out.columns:
        gap = out["kgup_toplam"] - out["load_lep"]
        out["supply_gap"] = gap
        out["supply_gap_change_1h"] = gap - gap.shift(1)
        out["supply_gap_change_24h"] = gap - gap.shift(24)
        out["supply_gap_zscore_24h"] = (gap - gap.rolling(24, min_periods=12).mean()) / gap.rolling(
            24, min_periods=12
        ).std().replace(0, np.nan)

    if "kgup_renewable_share" in out.columns:
        ren = out["kgup_renewable_share"]
        out["renewable_share_change_24h"] = ren - ren.shift(24)

    # --- Regulation / balancing stress (lagged realized) ---
    for col, alias in [
        ("yal_yat_net", "yal_yat_net"),
        ("yal_yat_upRegulationDelivered", "yal_up"),
        ("yal_yat_downRegulationDelivered", "yal_down"),
    ]:
        if col in out.columns:
            out[f"{alias}_stress_24h"] = out[col].shift(24).abs()
            out[f"{alias}_stress_168h"] = out[col].shift(168).abs()

    # --- Outage stress ---
    if "outage_fault_mw_loss_sum" in out.columns:
        out["outage_stress"] = (
            out["outage_fault_mw_loss_sum"].fillna(0)
            + out.get("outage_maint_capacity_sum", 0).fillna(0)
        )

    # --- Wind forecast ramp ---
    if "wind_forecast_mean" in out.columns:
        wf = out["wind_forecast_mean"]
        out["wind_forecast_ramp_1h"] = wf - wf.shift(1)
        out["wind_forecast_ramp_24h"] = wf - wf.shift(24)

    # --- Calendar interactions (microstructure timing) ---
    if "hour_sin" in out.columns and "ptf_lag_24" in out.columns:
        out["hour_x_ptf_lag24"] = out["hour_sin"] * out["ptf_lag_24"]
        out["hour_x_spread_lag24"] = out["hour_sin"] * out.get("smf_ptf_spread_lag_24", 0)

    if "is_holiday_or_weekend" in out.columns:
        out["weekend_x_ptf_vol"] = out["is_holiday_or_weekend"] * std24

    # --- Zero / spike proximity (past only) ---
    ptf_past = ptf.shift(1)
    out["ptf_zero_share_24h"] = (ptf_past == 0).rolling(24, min_periods=12).mean()
    out["ptf_zero_share_168h"] = (ptf_past == 0).rolling(168, min_periods=48).mean()
    out["ptf_spike_share_24h"] = (ptf_past >= 4800).rolling(24, min_periods=12).mean()
    out["ptf_spike_share_168h"] = (ptf_past >= 4800).rolling(168, min_periods=48).mean()
    out["ptf_hours_since_zero"] = _hours_since_event(ptf_past, threshold=0.5)
    out["ptf_hours_since_spike"] = _hours_since_event(ptf_past, threshold=4800.0, ge=True)

    return out


def _hours_since_event(series: pd.Series, *, threshold: float, ge: bool = False) -> pd.Series:
    flags = series >= threshold if ge else series <= threshold
    out = np.full(len(series), np.nan, dtype=float)
    last = np.nan
    for i, flag in enumerate(flags.to_numpy()):
        if flag:
            last = 0.0
        elif not np.isnan(last):
            last += 1.0
        out[i] = last
    return pd.Series(out, index=series.index)


def list_microstructure_columns() -> list[str]:
    return [
        "ptf_return_1h",
        "ptf_return_24h",
        "ptf_momentum_24h",
        "ptf_zscore_24h",
        "ptf_range_24h",
        "ptf_vol_ratio_24_168",
        "spread_nowcast_lag1",
        "spread_change_24h",
        "spread_zscore_24h",
        "supply_gap",
        "supply_gap_change_1h",
        "supply_gap_change_24h",
        "supply_gap_zscore_24h",
        "renewable_share_change_24h",
        "yal_yat_net_stress_24h",
        "yal_yat_net_stress_168h",
        "yal_up_stress_24h",
        "yal_up_stress_168h",
        "yal_down_stress_24h",
        "yal_down_stress_168h",
        "outage_stress",
        "wind_forecast_ramp_1h",
        "wind_forecast_ramp_24h",
        "hour_x_ptf_lag24",
        "hour_x_spread_lag24",
        "weekend_x_ptf_vol",
        "ptf_zero_share_24h",
        "ptf_zero_share_168h",
        "ptf_spike_share_24h",
        "ptf_spike_share_168h",
        "ptf_hours_since_zero",
        "ptf_hours_since_spike",
    ]
