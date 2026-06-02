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

    # Point-in-time safe lags: anchor hour t may only use <= t-1 data.
    out["ptf_lag_1"] = ptf.shift(1)
    out["ptf_lag_2"] = ptf.shift(2)
    out["ptf_lag_3"] = ptf.shift(3)
    out["ptf_lag_24"] = ptf.shift(24)
    out["ptf_lag_48"] = ptf.shift(48)
    out["ptf_lag_168"] = ptf.shift(168)

    # Aliases with explicit hour suffix for clarity in downstream pipelines.
    out["ptf_lag_1h"] = out["ptf_lag_1"]
    out["ptf_lag_2h"] = out["ptf_lag_2"]
    out["ptf_lag_3h"] = out["ptf_lag_3"]
    out["ptf_lag_24h"] = out["ptf_lag_24"]
    out["ptf_lag_168h"] = out["ptf_lag_168"]

    ptf_past = ptf.shift(1)
    out["ptf_roll_mean_24"] = ptf_past.rolling(24, min_periods=24).mean()
    out["ptf_roll_std_24"] = ptf_past.rolling(24, min_periods=24).std()
    out["ptf_roll_mean_168"] = ptf_past.rolling(168, min_periods=168).mean()
    out["ptf_roll_std_168"] = ptf_past.rolling(168, min_periods=168).std()

    # Aliases requested by user (same values; still uses t-1 only).
    out["ptf_rolling_mean_24h"] = out["ptf_roll_mean_24"]
    out["ptf_rolling_std_24h"] = out["ptf_roll_std_24"]
    out["ptf_rolling_mean_168h"] = out["ptf_roll_mean_168"]

    return out


def add_ptf_low_regime_history_features(
    df: pd.DataFrame,
    *,
    low_threshold: float = 50.0,
) -> pd.DataFrame:
    """
    Low/zero regime history from past PTF only (through t-1).

    Uses ptf.shift(1) before rolling counts/ratios — leakage-safe for DAM anchor t.
    """
    out = df.copy()
    ptf_past = out[PTF_COL].shift(1)

    roll_24 = ptf_past.rolling(24, min_periods=24)
    roll_168 = ptf_past.rolling(168, min_periods=168)
    out["ptf_roll_min_24"] = roll_24.min()
    out["ptf_roll_max_24"] = roll_24.max()
    out["ptf_roll_min_168"] = roll_168.min()
    out["ptf_roll_max_168"] = roll_168.max()

    is_low = (ptf_past <= low_threshold).astype(float)
    is_zero = (ptf_past == 0.0).astype(float)

    out["ptf_low_count_24"] = is_low.rolling(24, min_periods=24).sum()
    out["ptf_zero_count_24"] = is_zero.rolling(24, min_periods=24).sum()
    out["ptf_low_count_168"] = is_low.rolling(168, min_periods=168).sum()
    out["ptf_zero_count_168"] = is_zero.rolling(168, min_periods=168).sum()

    out["ptf_low_ratio_24"] = out["ptf_low_count_24"] / 24.0
    out["ptf_zero_ratio_24"] = out["ptf_zero_count_24"] / 24.0
    out["ptf_low_ratio_168"] = out["ptf_low_count_168"] / 168.0
    out["ptf_zero_ratio_168"] = out["ptf_zero_count_168"] / 168.0

    return out


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = out["ts_hour"]
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert("Europe/Istanbul")

    hour = ts.dt.hour
    dow = ts.dt.dayofweek
    month = ts.dt.month

    # Integer calendar fields (often useful for tree models / dashboard).
    out["hour_of_day"] = hour.astype(int)
    out["month"] = month.astype(int)

    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)
    out["is_weekend"] = (dow >= 5).astype(int)

    # Seasonal flags (simple month-bucket proxy).
    out["is_summer"] = out["month"].map(lambda m: 1 if int(m) in (6, 7, 8) else 0).astype(int)
    out["is_winter"] = out["month"].map(lambda m: 1 if int(m) in (12, 1, 2) else 0).astype(int)

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

    # Keep both a short alias ("is_holiday") and a namespaced one ("is_holiday_tr") for compatibility.
    out["is_holiday"] = dates.map(lambda d: 1 if d in tr_holidays else 0).astype(int)
    out["is_holiday_tr"] = out["is_holiday"]

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


def add_fiba_fibs_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    DAM price-independent buy/sell order features (leakage-safe).

    Assumes raw hourly volumes are available at anchor hour t as market/orderbook totals.
    Safe division rules:
      - ratio: sell==0 -> NaN
      - pressure: total==0 -> NaN
    """
    out = df.copy()
    buy = pd.to_numeric(out.get("dam_price_independent_buy_mwh"), errors="coerce")
    sell = pd.to_numeric(out.get("dam_price_independent_sell_mwh"), errors="coerce")

    if buy is None or sell is None:
        return out

    balance = buy - sell
    total = buy + sell

    out["fiba_fibs_ratio"] = np.where(sell > 0, buy / sell, np.nan).astype(float)
    out["fiba_fibs_balance"] = balance.astype(float)
    out["fiba_fibs_total"] = total.astype(float)
    out["fiba_fibs_pressure"] = np.where(total > 0, balance / total, np.nan).astype(float)

    # Strict-forecast alternatives (lagged).
    out["dam_price_independent_buy_lag_24"] = buy.shift(24).astype(float)
    out["dam_price_independent_sell_lag_24"] = sell.shift(24).astype(float)
    out["fiba_fibs_ratio_lag_24"] = out["fiba_fibs_ratio"].shift(24).astype(float)
    out["fiba_fibs_balance_lag_24"] = out["fiba_fibs_balance"].shift(24).astype(float)
    out["fiba_fibs_pressure_lag_24"] = out["fiba_fibs_pressure"].shift(24).astype(float)
    out["fiba_fibs_ratio_lag_168"] = out["fiba_fibs_ratio"].shift(168).astype(float)
    out["fiba_fibs_pressure_lag_168"] = out["fiba_fibs_pressure"].shift(168).astype(float)
    return out


def add_grf_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Natural gas GRF (daily reference price) features.

    - grf_tl_lag_1d is past-only (t-24 hours) to reduce publication-time leakage.
    - rolling mean uses lagged series (past-only).
    """
    out = df.copy()
    grf = pd.to_numeric(out.get("grf_tl_1000sm3"), errors="coerce")
    if grf is None:
        return out

    grf_lag_1d = grf.shift(24)
    out["grf_tl_lag_1d"] = grf_lag_1d.astype(float)

    grf_lag_8d = grf_lag_1d.shift(24 * 7)
    out["grf_tl_change_7d"] = (grf_lag_1d - grf_lag_8d).astype(float)

    out["grf_tl_rolling_mean_7d"] = grf_lag_1d.rolling(24 * 7, min_periods=24 * 7).mean().astype(float)

    gas_share = pd.to_numeric(out.get("gas_share"), errors="coerce")
    thermal_share = pd.to_numeric(out.get("thermal_price_setting_share"), errors="coerce")
    if gas_share is not None:
        out["gas_cost_pressure"] = (gas_share * grf).astype(float)
    else:
        out["gas_cost_pressure"] = np.nan
    if thermal_share is not None:
        out["thermal_cost_pressure"] = (thermal_share * grf).astype(float)
    else:
        out["thermal_cost_pressure"] = np.nan
    if gas_share is not None and thermal_share is not None:
        out["gas_marginal_pressure"] = (gas_share * thermal_share * grf).astype(float)
    else:
        out["gas_marginal_pressure"] = np.nan

    # Timing-safer alternatives (use grf_tl_lag_1d).
    if gas_share is not None:
        out["gas_cost_pressure_lag_1d"] = (gas_share * grf_lag_1d).astype(float)
    else:
        out["gas_cost_pressure_lag_1d"] = np.nan
    if thermal_share is not None:
        out["thermal_cost_pressure_lag_1d"] = (thermal_share * grf_lag_1d).astype(float)
    else:
        out["thermal_cost_pressure_lag_1d"] = np.nan
    if gas_share is not None and thermal_share is not None:
        out["gas_marginal_pressure_lag_1d"] = (gas_share * thermal_share * grf_lag_1d).astype(float)
    else:
        out["gas_marginal_pressure_lag_1d"] = np.nan

    return out


def add_dam_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    DAM offer/match/block microstructure features.

    Safe division:
      - Any ratio uses denominator > 0 else NaN (never inf).

    Matched volume:
      - If matched buy/sell differ, use average (robust + symmetric).
    """
    out = df.copy()

    bid = pd.to_numeric(out.get("dam_bid_volume_mwh"), errors="coerce")
    sell = pd.to_numeric(out.get("dam_sell_offer_volume_mwh"), errors="coerce")
    m_buy = pd.to_numeric(out.get("dam_matched_buy_mwh"), errors="coerce")
    m_sell = pd.to_numeric(out.get("dam_matched_sell_mwh"), errors="coerce")
    b_match = pd.to_numeric(out.get("dam_block_matched_buy_mwh"), errors="coerce")
    b_unmatch = pd.to_numeric(out.get("dam_block_unmatched_buy_mwh"), errors="coerce")

    if bid is not None and sell is not None:
        gap = bid - sell
        total = bid + sell
        out["dam_buy_sell_ratio"] = np.where(sell > 0, bid / sell, np.nan).astype(float)
        out["dam_offer_supply_demand_gap"] = gap.astype(float)
        out["dam_offer_total_volume_mwh"] = total.astype(float)
        out["dam_offer_balance_pressure"] = np.where(total > 0, gap / total, np.nan).astype(float)
    else:
        out["dam_buy_sell_ratio"] = np.nan
        out["dam_offer_supply_demand_gap"] = np.nan
        out["dam_offer_total_volume_mwh"] = np.nan
        out["dam_offer_balance_pressure"] = np.nan

    # matched volume
    if m_buy is not None and m_sell is not None:
        out["dam_matched_volume_mwh"] = np.where(
            (m_buy.notna()) & (m_sell.notna()),
            0.5 * (m_buy + m_sell),
            np.where(m_buy.notna(), m_buy, m_sell),
        ).astype(float)
        out["dam_unmatched_buy_proxy"] = (bid - m_buy).astype(float) if bid is not None else np.nan
        out["dam_unmatched_sell_proxy"] = (sell - m_sell).astype(float) if sell is not None else np.nan
    else:
        out["dam_matched_volume_mwh"] = np.nan
        out["dam_unmatched_buy_proxy"] = np.nan
        out["dam_unmatched_sell_proxy"] = np.nan

    mv = pd.to_numeric(out.get("dam_matched_volume_mwh"), errors="coerce")
    if bid is not None and mv is not None:
        out["dam_bid_to_match_ratio"] = np.where(mv > 0, bid / mv, np.nan).astype(float)
    else:
        out["dam_bid_to_match_ratio"] = np.nan
    if sell is not None and mv is not None:
        out["dam_sell_to_match_ratio"] = np.where(mv > 0, sell / mv, np.nan).astype(float)
    else:
        out["dam_sell_to_match_ratio"] = np.nan
    if mv is not None and "dam_offer_total_volume_mwh" in out.columns:
        tv = pd.to_numeric(out["dam_offer_total_volume_mwh"], errors="coerce")
        out["dam_match_ratio"] = np.where(tv > 0, mv / tv, np.nan).astype(float)
    else:
        out["dam_match_ratio"] = np.nan

    # Block buy
    if b_match is not None and b_unmatch is not None:
        out["dam_block_total_buy_mwh"] = (b_match + b_unmatch).astype(float)
        bt = pd.to_numeric(out["dam_block_total_buy_mwh"], errors="coerce")
        out["dam_block_unmatched_ratio"] = np.where(bt > 0, b_unmatch / bt, np.nan).astype(float)
        out["dam_block_pressure"] = bt.astype(float)
    else:
        out["dam_block_total_buy_mwh"] = np.nan
        out["dam_block_unmatched_ratio"] = np.nan
        out["dam_block_pressure"] = np.nan

    # Lagged alternatives (strict forecast)
    for lag in (24, 168):
        if "dam_bid_volume_mwh" in out.columns:
            out[f"dam_bid_volume_lag_{lag}"] = pd.to_numeric(out["dam_bid_volume_mwh"], errors="coerce").shift(lag)
        if "dam_sell_offer_volume_mwh" in out.columns:
            out[f"dam_sell_offer_volume_lag_{lag}"] = pd.to_numeric(out["dam_sell_offer_volume_mwh"], errors="coerce").shift(lag)
        if "dam_buy_sell_ratio" in out.columns:
            out[f"dam_buy_sell_ratio_lag_{lag}"] = pd.to_numeric(out["dam_buy_sell_ratio"], errors="coerce").shift(lag)
        if "dam_offer_balance_pressure" in out.columns:
            out[f"dam_offer_balance_pressure_lag_{lag}"] = pd.to_numeric(out["dam_offer_balance_pressure"], errors="coerce").shift(lag)
    # only requested lag_24 for these
    out["dam_match_ratio_lag_24"] = pd.to_numeric(out.get("dam_match_ratio"), errors="coerce").shift(24)
    out["dam_block_unmatched_ratio_lag_24"] = pd.to_numeric(out.get("dam_block_unmatched_ratio"), errors="coerce").shift(24)
    return out


def add_cap_and_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cap / ratio proxies (leakage-safe).

    IMPORTANT:
    - We never use same-hour finalized PTF/SMF as features.
    - Ratios are computed from lagged values that are already safe in this dataset.

    Definitions:
      price_cap: constant cap level proxy (TL/MWh).
      ptf_to_cap_ratio: ptf_lag_24 / price_cap (yesterday same-hour proximity to cap).
      smf_to_cap_ratio: smf_lag_24 / price_cap (yesterday same-hour SMF proximity to cap).
    """
    out = df.copy()
    cap = 4800.0
    out["price_cap"] = cap

    if "ptf_lag_24" in out.columns:
        out["ptf_to_cap_ratio"] = (out["ptf_lag_24"] / cap).astype(float)
    else:
        out["ptf_to_cap_ratio"] = np.nan

    if "smf_lag_24" in out.columns:
        out["smf_to_cap_ratio"] = (out["smf_lag_24"] / cap).astype(float)
    else:
        out["smf_to_cap_ratio"] = np.nan

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


def add_ptf_downside_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature family to represent "PTF downside / drop risk" logic:
      - demand weakness
      - renewable pressure
      - gas marginality proxy (gas share / gas vs coal balance)

    All features are derived from same-hour KGUP (planned) and load forecast at ts_hour=t
    and simple calendar flags (no future target leakage).
    """
    out = df.copy()

    total = out.get("kgup_toplam")
    if total is None:
        return out

    # Components
    gas = out.get("kgup_dogalgaz", 0.0)
    wind = out.get("kgup_ruzgar", 0.0)
    solar = out.get("kgup_gunes", 0.0)
    hydro = out.get("kgup_barajli", 0.0) + out.get("kgup_akarsu", 0.0)
    biomass = out.get("kgup_biokutle", 0.0)
    geothermal = out.get("kgup_jeotermal", 0.0)

    # Coal proxy: imported coal + lignite (+ hard coal if present)
    coal = out.get("kgup_ithalKomur", 0.0) + out.get("kgup_linyit", 0.0) + out.get("kgup_tasKomur", 0.0)

    renewable = wind + solar + hydro + biomass + geothermal
    thermal = total - renewable

    # Shares (safe against divide-by-zero).
    out["res_share"] = np.where(total > 0, wind / total, np.nan)
    out["solar_share"] = np.where(total > 0, solar / total, np.nan)
    out["hydro_share"] = np.where(total > 0, hydro / total, np.nan)
    out["gas_share"] = np.where(total > 0, gas / total, np.nan)
    out["coal_share"] = np.where(total > 0, coal / total, np.nan)

    out["renewable_pressure"] = np.where(total > 0, renewable / total, np.nan)
    out["thermal_price_setting_share"] = np.where(total > 0, thermal / total, np.nan)

    # Gas vs coal balance: [-1..+1] where +1 => all gas, -1 => all coal.
    denom_gc = (gas + coal)
    out["gas_coal_balance"] = np.where(denom_gc > 0, (gas - coal) / denom_gc, 0.0)

    # Gas/coal competition: [0..1], 1 means gas and coal are similar scale, 0 means one dominates.
    out["gas_coal_competition_index"] = np.where(
        denom_gc > 0,
        (2.0 * np.minimum(gas, coal) / denom_gc).astype(float),
        0.0,
    )

    # Demand weakness proxy: load forecast below its recent past mean (past-only).
    load = out.get("load_lep")
    if load is not None:
        load_past = load.shift(1)
        load_mean_30d = load_past.rolling(24 * 30, min_periods=24 * 7).mean()
        load_std_30d = load_past.rolling(24 * 30, min_periods=24 * 7).std()
        out["low_load_flag"] = (load < (load_mean_30d - 0.5 * load_std_30d)).astype(int)
    else:
        out["low_load_flag"] = 0

    # Holiday interaction.
    if "is_holiday_or_weekend" in out.columns:
        out["holiday_low_load_flag"] = ((out["is_holiday_or_weekend"] == 1) & (out["low_load_flag"] == 1)).astype(int)
    elif "is_holiday" in out.columns and "is_weekend" in out.columns:
        out["holiday_low_load_flag"] = (((out["is_holiday"] == 1) | (out["is_weekend"] == 1)) & (out["low_load_flag"] == 1)).astype(int)
    else:
        out["holiday_low_load_flag"] = 0

    # Solar peak hours (simple operational proxy).
    if "hour_of_day" in out.columns:
        out["solar_peak_hour_flag"] = out["hour_of_day"].between(10, 15).astype(int)
    else:
        out["solar_peak_hour_flag"] = 0

    # Zero-price risk proxy: high renewable pressure + weak demand + low gas share.
    # Kept as a smooth score [0..1] rather than a hard label.
    ren = out["renewable_pressure"].astype(float)
    gas_s = out["gas_share"].astype(float)
    low_load = out["low_load_flag"].astype(float)
    # Score = 0.5*renewables + 0.3*low_load + 0.2*(1-gas_share)
    score = 0.5 * ren.fillna(0.0) + 0.3 * low_load.fillna(0.0) + 0.2 * (1.0 - gas_s.fillna(0.0))
    out["zero_price_risk_proxy"] = score.clip(0.0, 1.0)

    # Renewable suppression pressure: a monotone proxy for "renewables can suppress price".
    # This is not a label; it's a feature in [0..1] and is strongest when renewables are high and demand is weak.
    out["renewable_suppression_pressure"] = (
        (0.7 * ren.fillna(0.0) + 0.3 * low_load.fillna(0.0)).clip(0.0, 1.0)
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
        "ptf_lag_2",
        "ptf_lag_3",
        "ptf_lag_24",
        "ptf_lag_48",
        "ptf_lag_168",
        "ptf_lag_1h",
        "ptf_lag_2h",
        "ptf_lag_3h",
        "ptf_lag_24h",
        "ptf_lag_168h",
        "ptf_roll_mean_24",
        "ptf_roll_std_24",
        "ptf_roll_mean_168",
        "ptf_roll_std_168",
        "ptf_rolling_mean_24h",
        "ptf_rolling_std_24h",
        "ptf_rolling_mean_168h",
    ]
    ptf_low_regime = [
        "ptf_roll_min_24",
        "ptf_roll_max_24",
        "ptf_roll_min_168",
        "ptf_roll_max_168",
        "ptf_low_count_24",
        "ptf_zero_count_24",
        "ptf_low_count_168",
        "ptf_zero_count_168",
        "ptf_low_ratio_24",
        "ptf_zero_ratio_24",
        "ptf_low_ratio_168",
        "ptf_zero_ratio_168",
    ]
    spread = ["smf_ptf_spread_lag_24", "smf_ptf_spread_lag_168"]
    supply = [
        "kgup_total_minus_load",
        "kgup_renewable_share",
        "kgup_thermal_share",
        "wind_forecast_share",
    ]
    downside = [
        "res_share",
        "solar_share",
        "hydro_share",
        "renewable_pressure",
        "thermal_price_setting_share",
        "gas_coal_competition_index",
        "renewable_suppression_pressure",
        "gas_share",
        "coal_share",
        "gas_coal_balance",
        "low_load_flag",
        "holiday_low_load_flag",
        "solar_peak_hour_flag",
        "zero_price_risk_proxy",
    ]
    market_orders = [
        "dam_price_independent_buy_mwh",
        "dam_price_independent_sell_mwh",
        "dam_price_independent_buy_lag_24",
        "dam_price_independent_sell_lag_24",
        "fiba_fibs_ratio",
        "fiba_fibs_balance",
        "fiba_fibs_total",
        "fiba_fibs_pressure",
        "fiba_fibs_ratio_lag_24",
        "fiba_fibs_balance_lag_24",
        "fiba_fibs_pressure_lag_24",
        "fiba_fibs_ratio_lag_168",
        "fiba_fibs_pressure_lag_168",
    ]
    grf = [
        "grf_tl_1000sm3",
        "grf_usd_1000sm3",
        "grf_eur_mwh",
        "grf_usd_mmbtu",
        "grf_tl_lag_1d",
        "grf_tl_change_7d",
        "grf_tl_rolling_mean_7d",
        "gas_cost_pressure",
        "thermal_cost_pressure",
        "gas_marginal_pressure",
        "gas_cost_pressure_lag_1d",
        "thermal_cost_pressure_lag_1d",
        "gas_marginal_pressure_lag_1d",
    ]
    dam_micro = [
        "dam_bid_volume_mwh",
        "dam_sell_offer_volume_mwh",
        "dam_matched_buy_mwh",
        "dam_matched_sell_mwh",
        "dam_matched_volume_mwh",
        "dam_bid_to_match_ratio",
        "dam_sell_to_match_ratio",
        "dam_unmatched_buy_proxy",
        "dam_unmatched_sell_proxy",
        "dam_block_matched_buy_mwh",
        "dam_block_unmatched_buy_mwh",
        "dam_block_total_buy_mwh",
        "dam_block_unmatched_ratio",
        "dam_block_pressure",
        "dam_buy_sell_ratio",
        "dam_offer_supply_demand_gap",
        "dam_offer_total_volume_mwh",
        "dam_offer_balance_pressure",
        "dam_match_ratio",
        "dam_bid_volume_lag_24",
        "dam_sell_offer_volume_lag_24",
        "dam_buy_sell_ratio_lag_24",
        "dam_offer_balance_pressure_lag_24",
        "dam_match_ratio_lag_24",
        "dam_block_unmatched_ratio_lag_24",
        "dam_bid_volume_lag_168",
        "dam_sell_offer_volume_lag_168",
        "dam_buy_sell_ratio_lag_168",
        "dam_offer_balance_pressure_lag_168",
    ]
    cap_ratios = ["price_cap", "ptf_to_cap_ratio", "smf_to_cap_ratio"]

    lagged = []
    for col in LAGGED_SOURCE_COLS:
        safe_name = col.replace("smf_systemMarginalPrice", "smf")
        for lag in LAG_STEPS:
            lagged.append(f"{safe_name}_lag_{lag}")

    return (
        calendar
        + ptf_lags
        + ptf_low_regime
        + spread
        + market_orders
        + grf
        + dam_micro
        + supply
        + downside
        + cap_ratios
        + KGUP_FEATURE_COLS
        + LOAD_FEATURE_COLS
        + WIND_FORECAST_COLS
        + OUTAGE_FEATURE_COLS
        + lagged
    )


def list_target_columns() -> list[str]:
    return [f"target_{h}h" for h in TARGET_HORIZONS]


def list_persistence_columns() -> list[str]:
    return [f"persistence_{h}h" for h in TARGET_HORIZONS]


def list_residual_target_columns() -> list[str]:
    return [f"target_residual_{h}h" for h in TARGET_HORIZONS]


def add_persistence_and_residual_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    persistence(t+h) = PTF(t+h-24); residual = actual - persistence.
    Uses target_h.shift(24) == PTF(t+h-24) at anchor t (no ptf_price column needed).
    """
    out = df.copy()
    for h in TARGET_HORIZONS:
        tcol = f"target_{h}h"
        persistence = out[tcol].shift(24)
        out[f"persistence_{h}h"] = persistence
        out[f"target_residual_{h}h"] = out[tcol] - persistence
    return out


def assign_split(ts_hour: pd.Series) -> pd.Series:
    years = ts_hour.dt.tz_convert("Europe/Istanbul").dt.year
    split = pd.Series(index=ts_hour.index, dtype="object")
    split[(years >= 2020) & (years <= 2024)] = "train"
    split[years == 2025] = "validation"
    split[years == 2026] = "test"
    return split
