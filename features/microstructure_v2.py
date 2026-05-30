"""Leakage-safe market microstructure features (v2 spec)."""

from __future__ import annotations

import numpy as np
import pandas as pd

PTF = "ptf_price"
SMF = "smf_systemMarginalPrice"
KGUP_TOTAL = "kgup_toplam"
LOAD = "load_lep"
WIND_FC = "wind_forecast_mean"
YAL_NET = "yal_yat_net"
YAL_UP = "yal_yat_upRegulationDelivered"
YAL_DOWN = "yal_yat_downRegulationDelivered"

RENEWABLE_KGUP = [
    "kgup_ruzgar",
    "kgup_gunes",
    "kgup_barajli",
    "kgup_akarsu",
    "kgup_biokutle",
]
THERMAL_KGUP = [
    "kgup_dogalgaz",
    "kgup_linyit",
    "kgup_tasKomur",
    "kgup_ithalKomur",
    "kgup_fuelOil",
    "kgup_nafta",
    "kgup_diger",
]

NEW_FEATURE_NAMES = [
    "smf_ptf_spread_lag_1",
    "smf_ptf_spread_lag_24",
    "smf_ptf_spread_change_3h",
    "smf_volatility_24h",
    "ptf_volatility_24h",
    "ptf_change_1h",
    "ptf_change_3h",
    "ptf_change_24h",
    "kgup_total_ramp_1h",
    "kgup_total_ramp_3h",
    "kgup_total_ramp_24h",
    "kgup_renewable_ramp_1h",
    "kgup_thermal_ramp_1h",
    "renewable_pressure",
    "thermal_margin",
    "wind_forecast_ramp_1h",
    "wind_forecast_ramp_24h",
    "load_forecast_ramp_1h",
    "load_forecast_ramp_24h",
    "yal_yat_net_pressure_lag_1",
    "yal_yat_net_pressure_lag_24",
    "yal_yat_transition_lag_1",
    "yal_yat_abs_pressure_lag_1",
    "gas_marginal_proxy",
]


def _renewable(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in RENEWABLE_KGUP if c in df.columns]
    return df[cols].sum(axis=1) if cols else pd.Series(np.nan, index=df.index)


def _thermal(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in THERMAL_KGUP if c in df.columns]
    return df[cols].sum(axis=1) if cols else pd.Series(np.nan, index=df.index)


def build_microstructure_columns(master: pd.DataFrame) -> pd.DataFrame:
    """
    Compute microstructure columns from master hourly (realized via shift(1)+).
    Plan/forecast same-hour (kgup, load_lep, wind_forecast) allowed at anchor t.
    """
    out = pd.DataFrame(index=master.index)
    ptf = master[PTF]
    smf = master[SMF]
    ptf_past = ptf.shift(1)
    smf_past = smf.shift(1)

    spread = smf - ptf
    spread_l1 = spread.shift(1)

    out["smf_ptf_spread_lag_1"] = spread_l1
    out["smf_ptf_spread_lag_24"] = spread.shift(24)
    out["smf_ptf_spread_change_3h"] = spread_l1 - spread.shift(4)
    out["smf_volatility_24h"] = smf_past.rolling(24, min_periods=12).std()
    out["ptf_volatility_24h"] = ptf_past.rolling(24, min_periods=12).std()

    out["ptf_change_1h"] = ptf_past - ptf.shift(2)
    out["ptf_change_3h"] = ptf_past - ptf.shift(4)
    out["ptf_change_24h"] = ptf_past - ptf.shift(25)

    if KGUP_TOTAL in master.columns:
        kg = master[KGUP_TOTAL]
        kg_p = kg.shift(1)
        out["kgup_total_ramp_1h"] = kg_p - kg.shift(2)
        out["kgup_total_ramp_3h"] = kg_p - kg.shift(4)
        out["kgup_total_ramp_24h"] = kg_p - kg.shift(25)

    renewable = _renewable(master)
    thermal = _thermal(master)
    ren_p = renewable.shift(1)
    th_p = thermal.shift(1)
    out["kgup_renewable_ramp_1h"] = ren_p - renewable.shift(2)
    out["kgup_thermal_ramp_1h"] = th_p - thermal.shift(2)

    total_p = master[KGUP_TOTAL].shift(1) if KGUP_TOTAL in master.columns else np.nan
    load_p = master[LOAD].shift(1) if LOAD in master.columns else np.nan
    out["renewable_pressure"] = np.where(
        load_p > 0, ren_p / load_p, np.nan
    )
    out["thermal_margin"] = np.where(
        total_p > 0, th_p / total_p, np.nan
    )

    if WIND_FC in master.columns:
        wf = master[WIND_FC]
        out["wind_forecast_ramp_1h"] = wf.shift(1) - wf.shift(2)
        out["wind_forecast_ramp_24h"] = wf.shift(1) - wf.shift(25)

    if LOAD in master.columns:
        ld = master[LOAD]
        out["load_forecast_ramp_1h"] = ld.shift(1) - ld.shift(2)
        out["load_forecast_ramp_24h"] = ld.shift(1) - ld.shift(25)

    if YAL_NET in master.columns:
        net = master[YAL_NET]
        out["yal_yat_net_pressure_lag_1"] = net.shift(1)
        out["yal_yat_net_pressure_lag_24"] = net.shift(24)
        out["yal_yat_abs_pressure_lag_1"] = net.shift(1).abs()
        if YAL_UP in master.columns and YAL_DOWN in master.columns:
            transition = (
                master[YAL_UP].shift(1).fillna(0) + master[YAL_DOWN].shift(1).fillna(0)
            )
            out["yal_yat_transition_lag_1"] = transition

    if "kgup_dogalgaz" in master.columns and KGUP_TOTAL in master.columns:
        gas = master["kgup_dogalgaz"].shift(1)
        tot = master[KGUP_TOTAL].shift(1)
        out["gas_marginal_proxy"] = np.where(tot > 0, gas / tot, np.nan)

    return out[NEW_FEATURE_NAMES]
