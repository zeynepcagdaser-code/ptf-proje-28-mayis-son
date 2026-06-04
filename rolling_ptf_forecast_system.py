#!/usr/bin/env python3
"""Rolling next-24h PTF forecast system from existing EPİAŞ data files.

This pipeline is intentionally separate from the older experimental scripts.
It trains on anchor-hour rows: for every known PTF hour t, predict PTF(t+h)
for h=1..24 using only data available at or before t plus planned/forecast
delivery-hour variables already present in the local EPİAŞ files.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import holidays
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PRED_DIR = DATA_DIR / "predictions"
MODEL_DIR = PROJECT_ROOT / "models" / "rolling_ptf_next24"
REPORT_DIR = PROJECT_ROOT / "reports"

TARGET_HORIZONS = list(range(1, 25))
TZ = "Europe/Istanbul"
USE_HOLIDAY_COOLDOWN_FEATURE = False
USE_GRF_GATED_INTERACTIONS = False


@dataclass(frozen=True)
class Profile:
    name: str
    include_market_forecasts: bool
    include_orderbook: bool
    include_lagged_realized: bool


PROFILES = [
    Profile("ptf_only", False, False, False),
    Profile("forecast_market", True, False, False),
    Profile("full_market", True, True, True),
]

# Columns that are realized (post-settlement) at delivery time.
# Must never appear as delivery_* features; only lagged anchor versions are safe.
REALIZED_DELIVERY_COLS: frozenset[str] = frozenset({
    "ptf",                    # target itself
    "smf",                    # balancing mechanism realized price
    "real_consumption",       # realized consumption (not forecast)
    "yal_yat_net",            # realized net regulation
    "upRegulationDelivered",  # actual up-regulation delivered
    "downRegulationDelivered",  # actual down-regulation delivered
    "gen_total",              # realized total generation
    "gen_gas",
    "gen_wind",
    "gen_solar",
    "gen_dammed_hydro",
    "gen_river",
    "gen_import_coal",
    "gen_lignite",
    "dam_matched_volume",     # post-auction clearing quantity (published with PTF)
    "idm_price",              # GİP (IDM) hourly WAP: post-auction, not available before PTF is set
})


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def parse_ts(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce")
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert(TZ).dt.tz_localize(None)
    return ts.dt.floor("h")


def assert_naive_ts(df: pd.DataFrame, col: str = "ts_hour", context: str = "") -> None:
    """Raise if a timestamp column is tz-aware; tz-aware + naive merge silently empties joins."""
    if col not in df.columns or df.empty:
        return
    tz = getattr(df[col].dt, "tz", None)
    if tz is not None:
        raise ValueError(
            f"tz-aware '{col}' ({tz}) detected{f' in {context}' if context else ''}. "
            "All join keys must be naive — call parse_ts() first."
        )


def read_csv_hourly(path: Path, *, date_col: str = "date") -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if date_col not in df.columns:
        return pd.DataFrame()
    df["ts_hour"] = parse_ts(df[date_col])
    return df.dropna(subset=["ts_hour"]).sort_values("ts_hour")


def safe_div(num: pd.Series | np.ndarray, den: pd.Series | np.ndarray) -> pd.Series:
    n = pd.Series(num, dtype="float64")
    d = pd.Series(den, dtype="float64").replace(0, np.nan)
    return n / d


def numeric_frame(df: pd.DataFrame, exclude: set[str] | None = None) -> pd.DataFrame:
    exclude = exclude or set()
    out = df.copy()
    for col in out.columns:
        if col not in exclude:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def load_ptf() -> pd.DataFrame:
    df = read_csv_hourly(DATA_DIR / "ptf_dataset.csv")
    if df.empty:
        raise FileNotFoundError(DATA_DIR / "ptf_dataset.csv")
    df["ptf"] = pd.to_numeric(df["price"], errors="coerce")
    return df[["ts_hour", "ptf"]].dropna().drop_duplicates("ts_hour", keep="last")


def load_load_forecast() -> pd.DataFrame:
    df = read_csv_hourly(DATA_DIR / "load_forecast.csv")
    if df.empty or "lep" not in df.columns:
        return pd.DataFrame(columns=["ts_hour", "load_forecast"])
    df["load_forecast"] = pd.to_numeric(df["lep"], errors="coerce")
    return df[["ts_hour", "load_forecast"]].drop_duplicates("ts_hour", keep="last")


def load_kgup() -> pd.DataFrame:
    df = read_csv_hourly(DATA_DIR / "kgup_combined.csv")
    if df.empty:
        return pd.DataFrame(columns=["ts_hour"])
    rename = {
        "toplam": "kgup_total",
        "dogalgaz": "kgup_gas",
        "ruzgar": "kgup_wind",
        "gunes": "kgup_solar",
        "barajli": "kgup_dammed_hydro",
        "akarsu": "kgup_river",
        "linyit": "kgup_lignite",
        "tasKomur": "kgup_black_coal",
        "ithalKomur": "kgup_import_coal",
        "fuelOil": "kgup_fuel_oil",
        "jeotermal": "kgup_geothermal",
        "biokutle": "kgup_biomass",
    }
    cols = ["ts_hour"] + [c for c in rename if c in df.columns]
    out = df[cols].rename(columns=rename)
    out = numeric_frame(out, {"ts_hour"})
    renew = [c for c in ["kgup_wind", "kgup_solar", "kgup_dammed_hydro", "kgup_river", "kgup_geothermal", "kgup_biomass"] if c in out.columns]
    thermal = [c for c in ["kgup_gas", "kgup_lignite", "kgup_black_coal", "kgup_import_coal", "kgup_fuel_oil"] if c in out.columns]
    if renew:
        out["kgup_renewable"] = out[renew].sum(axis=1)
    if thermal:
        out["kgup_thermal"] = out[thermal].sum(axis=1)
    return out.drop_duplicates("ts_hour", keep="last")


def load_wind_forecast() -> pd.DataFrame:
    df = read_csv_hourly(DATA_DIR / "wind_forecast.csv")
    if df.empty:
        return pd.DataFrame(columns=["ts_hour"])
    cols = ["ts_hour"] + [c for c in ["quarter1", "quarter2", "quarter3", "quarter4", "forecast", "generation"] if c in df.columns]
    out = df[cols].copy()
    for col in cols:
        if col != "ts_hour":
            out[col] = pd.to_numeric(out[col], errors="coerce")
    qcols = [c for c in ["quarter1", "quarter2", "quarter3", "quarter4"] if c in out.columns]
    if qcols:
        out["wind_forecast_mean"] = out[qcols].mean(axis=1)
        out["wind_forecast_std"] = out[qcols].std(axis=1)
    if "forecast" in out.columns:
        out["wind_forecast"] = out["forecast"]
    return out.drop_duplicates("ts_hour", keep="last")


def load_simple_series(path: Path, source_col: str, out_col: str) -> pd.DataFrame:
    df = read_csv_hourly(path)
    if df.empty or source_col not in df.columns:
        return pd.DataFrame(columns=["ts_hour", out_col])
    df[out_col] = pd.to_numeric(df[source_col], errors="coerce")
    return df[["ts_hour", out_col]].drop_duplicates("ts_hour", keep="last")


def load_yal_yat() -> pd.DataFrame:
    df = read_csv_hourly(DATA_DIR / "yal_yat.csv")
    if df.empty:
        return pd.DataFrame(columns=["ts_hour"])
    cols = ["ts_hour"] + [c for c in df.columns if c != "date" and c != "hour" and c != "ts_hour"]
    out = df[cols].copy()
    out = numeric_frame(out, {"ts_hour"})
    if "net" in out.columns:
        out = out.rename(columns={"net": "yal_yat_net"})
    return out.drop_duplicates("ts_hour", keep="last")


def load_generation() -> pd.DataFrame:
    parquet_path = DATA_DIR / "clean" / "realtime_generation_hourly.parquet"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        if "ts_hour" in df.columns:
            df["ts_hour"] = parse_ts(df["ts_hour"])
        elif df.index.name == "ts_hour":
            df = df.reset_index()
            df["ts_hour"] = parse_ts(df["ts_hour"])
    else:
        df = read_csv_hourly(DATA_DIR / "realtime_generation.csv")
    if df.empty:
        return pd.DataFrame(columns=["ts_hour"])
    rename = {
        "total": "gen_total",
        "naturalGas": "gen_gas",
        "dammedHydro": "gen_dammed_hydro",
        "river": "gen_river",
        "importCoal": "gen_import_coal",
        "lignite": "gen_lignite",
        "wind": "gen_wind",
        "sun": "gen_solar",
        "importExport": "gen_import_export",  # net export (+) / import (-) in MWh
    }
    cols = ["ts_hour"] + [c for c in rename if c in df.columns]
    out = df[cols].rename(columns=rename)
    return numeric_frame(out, {"ts_hour"}).drop_duplicates("ts_hour", keep="last")


def load_grf() -> pd.DataFrame:
    hourly_path = PROCESSED_DIR / "grf_hourly_reference_price.parquet"
    if hourly_path.exists():
        df = pd.read_parquet(hourly_path)
        if "ts_hour" in df.columns:
            df["ts_hour"] = parse_ts(df["ts_hour"])
            return numeric_frame(df, {"ts_hour"}).drop_duplicates("ts_hour", keep="last")

    path = PROCESSED_DIR / "grf_daily_reference_price.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["ts_hour"])
    df = pd.read_parquet(path)
    if "ts_day" not in df.columns:
        return pd.DataFrame(columns=["ts_hour"])
    df["ts_day"] = parse_ts(df["ts_day"])
    df = numeric_frame(df, {"ts_day"})
    start = df["ts_day"].min()
    end = df["ts_day"].max() + pd.Timedelta(hours=23)
    hourly = pd.DataFrame({"ts_hour": pd.date_range(start, end, freq="h")})
    hourly["ts_day"] = hourly["ts_hour"].dt.floor("D")
    out = hourly.merge(df, on="ts_day", how="left").drop(columns=["ts_day"])
    return out


def load_temperature() -> pd.DataFrame:
    # Primary: multi-city parquet produced by fetch_temperature.py
    parquet_path = PROCESSED_DIR / "temperature_hourly.parquet"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        if "ts_hour" in df.columns:
            df["ts_hour"] = parse_ts(df["ts_hour"])
            return numeric_frame(df, {"ts_hour"}).drop_duplicates("ts_hour", keep="last")

    # Fallback: legacy single-city CSV (Ankara only, old format)
    csv_path = DATA_DIR / "raw" / "temperature_hourly.csv"
    if not csv_path.exists():
        return pd.DataFrame(columns=["ts_hour"])
    raw = pd.read_csv(csv_path, header=None)
    header_idx = None
    for idx, row in raw.iterrows():
        vals = [str(v) for v in row.tolist()]
        if "time" in vals:
            header_idx = idx
            break
    if header_idx is None:
        return pd.DataFrame(columns=["ts_hour"])
    df = pd.read_csv(csv_path, skiprows=header_idx)
    if "time" not in df.columns:
        return pd.DataFrame(columns=["ts_hour"])
    df["ts_hour"] = parse_ts(df["time"])
    df = df.rename(columns={
        "temperature_2m (°C)":      "temperature_2m",
        "apparent_temperature (°C)": "apparent_temperature",
    })
    cols = ["ts_hour"] + [c for c in ["temperature_2m", "apparent_temperature"] if c in df.columns]
    return numeric_frame(df[cols].copy(), {"ts_hour"}).drop_duplicates("ts_hour", keep="last")


def load_exchange_rates() -> pd.DataFrame:
    path = PROCESSED_DIR / "tcmb_exchange_rates_hourly.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["ts_hour"])
    df = pd.read_parquet(path)
    if "ts_hour" not in df.columns:
        return pd.DataFrame(columns=["ts_hour"])
    df["ts_hour"] = parse_ts(df["ts_hour"])
    return numeric_frame(df, {"ts_hour"}).drop_duplicates("ts_hour", keep="last")


def load_processed_orderbook() -> pd.DataFrame:
    specs = {
        "dam_bid_volume.parquet": "dam_bid_volume",
        "dam_sell_offer_volume.parquet": "dam_sell_offer_volume",
        "dam_matched_volume.parquet": "dam_matched_volume",
        "dam_price_independent_buy.parquet": "dam_price_independent_buy",
        "dam_price_independent_sell.parquet": "dam_price_independent_sell",
        "dam_block_buy_volume.parquet": "dam_block_buy_volume",
    }
    frames = []
    for name, prefix in specs.items():
        path = PROCESSED_DIR / name
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if "ts_hour" not in df.columns:
            continue
        df["ts_hour"] = parse_ts(df["ts_hour"])
        value_cols = [c for c in df.columns if c != "ts_hour" and pd.api.types.is_numeric_dtype(df[c])]
        if not value_cols:
            continue
        col = value_cols[0]
        frames.append(df[["ts_hour", col]].rename(columns={col: prefix}))
    if not frames:
        return pd.DataFrame(columns=["ts_hour"])
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="ts_hour", how="outer")
    return out.sort_values("ts_hour").drop_duplicates("ts_hour", keep="last")


def load_idm_prices() -> pd.DataFrame:
    path = PROCESSED_DIR / "idm_prices.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["ts_hour"])
    df = pd.read_parquet(path)
    if "ts_hour" not in df.columns:
        return pd.DataFrame(columns=["ts_hour"])
    df["ts_hour"] = parse_ts(df["ts_hour"])
    return numeric_frame(df, {"ts_hour"}).drop_duplicates("ts_hour", keep="last")


def load_international_prices() -> pd.DataFrame:
    path = PROCESSED_DIR / "international_commodity_prices_hourly.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["ts_hour"])
    df = pd.read_parquet(path)
    if "ts_hour" not in df.columns:
        return pd.DataFrame(columns=["ts_hour"])
    df["ts_hour"] = parse_ts(df["ts_hour"])
    return numeric_frame(df, {"ts_hour"}).drop_duplicates("ts_hour", keep="last")


def load_dam_fullness() -> pd.DataFrame:
    path = PROCESSED_DIR / "dam_fullness_hourly.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["ts_hour"])
    df = pd.read_parquet(path)
    if "ts_hour" not in df.columns:
        return pd.DataFrame(columns=["ts_hour"])
    df["ts_hour"] = parse_ts(df["ts_hour"])
    return numeric_frame(df, {"ts_hour"}).drop_duplicates("ts_hour", keep="last")


def load_istanbul_dam_fullness() -> pd.DataFrame:
    """İBB CKAN verisinden İstanbul 10 barajı günlük doluluk % (2000-2024)."""
    path = PROJECT_ROOT / "data" / "external" / "ibb_istanbul" / "ibb_dam_hourly.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["ts_hour"])
    df = pd.read_parquet(path)
    if "ts_hour" not in df.columns:
        return pd.DataFrame(columns=["ts_hour"])
    df["ts_hour"] = parse_ts(df["ts_hour"])
    return numeric_frame(df, {"ts_hour"}).drop_duplicates("ts_hour", keep="last")


def build_hourly_market_table() -> pd.DataFrame:
    ptf = load_ptf()
    loaders = [
        load_load_forecast,
        load_kgup,
        load_wind_forecast,
        lambda: load_simple_series(DATA_DIR / "smf.csv", "systemMarginalPrice", "smf"),
        lambda: load_simple_series(DATA_DIR / "real_consumption.csv", "consumption", "real_consumption"),
        load_yal_yat,
        load_generation,
        load_grf,
        load_temperature,
        load_exchange_rates,
        load_processed_orderbook,
        load_idm_prices,
        load_international_prices,
        load_dam_fullness,
        load_istanbul_dam_fullness,
    ]
    source_frames = [loader() for loader in loaders]
    # Validate all ts_hour columns are naive before merging; tz-aware + naive silently empties joins
    assert_naive_ts(ptf, context="ptf")
    for i, frame in enumerate(source_frames):
        assert_naive_ts(frame, context=f"source_frame[{i}]")
    max_ts = ptf["ts_hour"].max()
    for frame in source_frames:
        if not frame.empty and "ts_hour" in frame.columns:
            frame_max = frame["ts_hour"].max()
            if pd.notna(frame_max):
                max_ts = max(max_ts, frame_max)
    spine = pd.DataFrame({"ts_hour": pd.date_range(ptf["ts_hour"].min(), max_ts, freq="h")})
    data = spine.merge(ptf, on="ts_hour", how="left")
    for frame in source_frames:
        if not frame.empty:
            data = data.merge(frame, on="ts_hour", how="left")

    data = data.sort_values("ts_hour").reset_index(drop=True)
    data["hour"] = data["ts_hour"].dt.hour
    data["dow"] = data["ts_hour"].dt.dayofweek
    data["month"] = data["ts_hour"].dt.month
    data = add_market_composites(data)
    return add_merit_order_features(data)


def add_market_composites(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    total = pd.to_numeric(out.get("kgup_total"), errors="coerce")
    load = pd.to_numeric(out.get("load_forecast"), errors="coerce")
    wind = pd.to_numeric(out.get("kgup_wind"), errors="coerce").fillna(pd.to_numeric(out.get("wind_forecast_mean"), errors="coerce"))
    solar = pd.to_numeric(out.get("kgup_solar"), errors="coerce")
    hydro = pd.to_numeric(out.get("kgup_dammed_hydro"), errors="coerce").fillna(0) + pd.to_numeric(out.get("kgup_river"), errors="coerce").fillna(0)
    gas = pd.to_numeric(out.get("kgup_gas"), errors="coerce")
    coal_cols = [c for c in ["kgup_lignite", "kgup_black_coal", "kgup_import_coal"] if c in out.columns]
    coal = out[coal_cols].sum(axis=1) if coal_cols else pd.Series(np.nan, index=out.index)
    renewable = pd.to_numeric(out.get("kgup_renewable"), errors="coerce").fillna(wind.fillna(0) + solar.fillna(0) + hydro.fillna(0))

    out["net_load_after_wind_solar"] = load - wind.fillna(0) - solar.fillna(0)
    out["net_load_after_renewable"] = load - renewable
    out["load_minus_kgup_total"] = load - total
    out["renewable_share"] = safe_div(renewable, total).to_numpy()
    out["gas_share"] = safe_div(gas, total).to_numpy()
    out["coal_share"] = safe_div(coal, total).to_numpy()
    out["hydro_share"] = safe_div(hydro, total).to_numpy()
    out["wind_load_share"] = safe_div(wind, load).to_numpy()
    out["solar_load_share"] = safe_div(solar, load).to_numpy()
    out["renewable_load_share"] = safe_div(renewable, load).to_numpy()
    out["thermal_share"] = safe_div(gas.fillna(0) + coal.fillna(0), total).to_numpy()
    out["gas_vs_renewable"] = out["gas_share"] - out["renewable_share"]
    out["thermal_tightness_pressure"] = (
        100 * (
            0.35 * out["gas_share"].fillna(0)
            + 0.25 * out["coal_share"].fillna(0)
            + 0.25 * (1 - out["renewable_share"].fillna(0).clip(0, 1))
            + 0.15 * safe_div(out["net_load_after_renewable"], load).fillna(0).clip(-1, 2)
        )
    )
    out["kgup_total_delta_1h"] = total.diff()
    out["kgup_gas_delta_1h"] = gas.diff()
    out["kgup_renewable_delta_1h"] = renewable.diff()
    out["load_forecast_delta_1h"] = load.diff()
    out["net_load_renewable_delta_1h"] = out["net_load_after_renewable"].diff()
    out["ramp_tightness"] = (
        out["net_load_renewable_delta_1h"].fillna(0)
        - out["kgup_gas_delta_1h"].fillna(0)
        - coal.diff().fillna(0)
    )
    out["morning_ramp_flag"] = out["hour"].between(5, 8).astype(int)
    out["evening_ramp_flag"] = out["hour"].between(17, 21).astype(int)
    out["night_block_flag"] = out["hour"].between(0, 6).astype(int)

    # Solar rampa feature'ları — duck curve evening spike encoding
    # Saat 19-21 PTF spike'ı, solar'ın ÖNCEKI 3-6 saatte düşmesinden kaynaklanıyor.
    # Örnek: saat 16'da solar=8,000 MW → saat 19'da solar=100 MW → 7,900 MW gaz rampa
    # shift(3): 3 saat ÖNCEKİ solar → bugünden çıkar = geçen 3h düşüş
    solar_clean = solar.fillna(0)
    out["solar_drop_past_3h"] = (solar_clean.shift(3) - solar_clean).clip(lower=0)  # son 3h düşüş
    out["solar_drop_past_6h"] = (solar_clean.shift(6) - solar_clean).clip(lower=0)  # son 6h düşüş
    # KGÜP ile ileriye bakan: 3 saat sonra solar ne kadar düşecek? (gün öncesi bilinir)
    out["solar_drop_next_3h"] = (solar_clean - solar_clean.shift(-3)).clip(lower=0)
    # Geçmiş düşüş × gaz maliyeti: gaz rampa zorunlu + pahalı → PTF spike
    if "grf_tl_lag_24" in out.columns:
        grf = pd.to_numeric(out["grf_tl_lag_24"], errors="coerce").ffill()
        out["grf_x_solar_drop_3h"] = grf * out["solar_drop_past_3h"] / 10000
        out["grf_x_solar_drop_6h"] = grf * out["solar_drop_past_6h"] / 10000  # 6h daha güçlü sinyal
    # Günlük solar max'a göre normalize (şiddeti ölçer)
    daily_solar_max = solar_clean.groupby(out["ts_hour"].dt.date).transform("max").replace(0, np.nan)
    out["solar_drop_pct_of_day"] = safe_div(out["solar_drop_past_3h"], daily_solar_max).to_numpy()

    # Duck curve temel sinyali: günlük solar zirvesinden ne kadar uzaklaştık?
    # Log-transform ve yük normalizasyonu ile ölçek bağımsız hale getiriyoruz.
    # Böylece 2020'deki 5,000 MW peak ile 2026'daki 17,000 MW peak aynı ölçekte kalıyor.
    out["solar_pct_of_daily_peak"] = safe_div(solar_clean, daily_solar_max).to_numpy()  # 0=gece, 1=zirve

    # Load-normalized solar drop from peak: yükün yüzdesi olarak solar kaybı
    # 2020: kayıp=5,000 MW / load=35,000 = 0.14 | 2026: 17,000/42,000 = 0.40 → anlamlı fark, drift değil
    solar_deficit_vs_load = safe_div(
        (daily_solar_max - solar_clean).clip(lower=0), load.replace(0, np.nan)
    ).to_numpy()
    out["solar_deficit_pct_load"] = solar_deficit_vs_load

    # Gaz rampa oranı — yüke göre normalize (ölçek bağımsız)
    # 2020: gaz 3h artışı 500 MW / load 30,000 = 0.017 | 2026: 4,000 / 40,000 = 0.10
    gas_clean = gas.fillna(0)
    gas_ramp_3h = (gas_clean - gas_clean.shift(3)).clip(lower=0)
    out["gas_ramp_pct_load"] = safe_div(gas_ramp_3h, load.replace(0, np.nan)).to_numpy()

    # Duck curve stress — bileşik ama ölçek bağımsız:
    # (solar kayıp / load) × (gaz / load) → her iki faktör load'a göre normalize
    out["duck_stress_normalized"] = solar_deficit_vs_load * safe_div(gas_clean, load.replace(0, np.nan)).to_numpy()

    # net_load_coverage: load/kgup_total — supply tightness signal independent of renewable mix
    out["net_load_coverage"] = safe_div(load, total).to_numpy()

    # Hydro displacement features — capture February-style hydro surge pushing gas off margin
    dammed_hydro = pd.to_numeric(out.get("kgup_dammed_hydro"), errors="coerce").fillna(0)
    out["hydro_demand_ratio"] = safe_div(dammed_hydro, load).to_numpy()
    out["hydro_gas_displacement"] = dammed_hydro - gas.fillna(0)

    # Gas marginality threshold — when gas_share is 3-25%, gas is the marginal price-setter
    # even with low volume; below 3% gas is fully off margin; above 25% gas dominates volume
    gas_s = out["gas_share"].fillna(0)
    out["gas_margin_proximity"] = (0.15 - gas_s).clip(lower=0)  # >0 when gas below marginal threshold

    if "grf_tl_1000sm3" in out.columns:
        grf = pd.to_numeric(out["grf_tl_1000sm3"], errors="coerce")
        peak_mask = out["hour"].between(17, 22).astype(int)
        # Guard: parquet may already have these; also fixes formula — was shift(24)-shift(8d), correct is current-shift(7d)
        if "grf_tl_lag_24" not in out.columns:
            out["grf_tl_lag_24"] = grf.shift(24)
        if "grf_tl_change_7d" not in out.columns:
            out["grf_tl_change_7d"] = grf - grf.shift(24 * 7)
        if "grf_tl_roll_mean_7d" not in out.columns:
            out["grf_tl_roll_mean_7d"] = grf.rolling(24 * 7, min_periods=24 * 3).mean()
        out["grf_peak_effect"] = grf * peak_mask
        out["demand_per_grf"] = safe_div(load, grf).to_numpy()
        out["net_load_x_grf"] = out["net_load_after_renewable"] * grf
        # GRF × gas_share: gas cost when gas is the marginal price-setter (merit-order signal)
        out["grf_x_gas_share"] = grf * out["gas_share"].fillna(0)

    if "tr_temp_mean" in out.columns:
        temp    = pd.to_numeric(out["tr_temp_mean"], errors="coerce")
        app     = pd.to_numeric(out.get("tr_apparent_temp_mean"), errors="coerce")
        cooling = pd.to_numeric(out.get("tr_cooling_degree"), errors="coerce")
        heating = pd.to_numeric(out.get("tr_heating_degree"), errors="coerce")
        heatwave = pd.to_numeric(out.get("tr_heatwave_flag"), errors="coerce")
        # Re-derive if parquet columns are somehow missing
        if cooling.isna().all():
            cooling = (temp - 22).clip(lower=0)
            out["tr_cooling_degree"] = cooling
        if heating.isna().all():
            heating = (18 - temp).clip(lower=0)
            out["tr_heating_degree"] = heating
        if heatwave.isna().all():
            heatwave = (temp >= 35).astype(float)
            out["tr_heatwave_flag"] = heatwave
        out["tr_temp_lag_24"]          = temp.shift(24)
        out["tr_temp_lag_168"]         = temp.shift(168)
        out["tr_temp_delta_24"]        = temp - temp.shift(24)
        out["tr_apparent_temp_delta_24"] = app - app.shift(24)
        out["load_x_cooling"]  = load * cooling
        out["load_x_heating"]  = load * heating
        out["load_x_heatwave"] = load * heatwave.fillna(0)
    elif "temperature_2m" in out.columns:
        # Backward compat: legacy single-city CSV
        temp = pd.to_numeric(out["temperature_2m"], errors="coerce")
        app  = pd.to_numeric(out.get("apparent_temperature"), errors="coerce")
        out["tr_temp_mean"]              = temp
        out["tr_temp_lag_24"]            = temp.shift(24)
        out["tr_temp_delta_24"]          = temp - temp.shift(24)
        out["tr_cooling_degree"]         = (temp - 22).clip(lower=0)
        out["tr_heating_degree"]         = (18 - temp).clip(lower=0)
        out["tr_apparent_temp_delta_24"] = app - app.shift(24)
        out["load_x_cooling"]  = load * out["tr_cooling_degree"]
        out["load_x_heating"]  = load * out["tr_heating_degree"]

    if "dam_price_independent_buy" in out.columns and "dam_price_independent_sell" in out.columns:
        buy = pd.to_numeric(out["dam_price_independent_buy"], errors="coerce")
        sell = pd.to_numeric(out["dam_price_independent_sell"], errors="coerce")
        out["price_independent_balance"] = buy - sell
        out["price_independent_pressure"] = safe_div(buy - sell, buy + sell).to_numpy()
    if "dam_bid_volume" in out.columns and "dam_sell_offer_volume" in out.columns:
        bid = pd.to_numeric(out["dam_bid_volume"], errors="coerce")
        sell_offer = pd.to_numeric(out["dam_sell_offer_volume"], errors="coerce")
        out["dam_bid_sell_ratio"] = safe_div(bid, sell_offer).to_numpy()
        out["dam_bid_sell_balance"] = bid - sell_offer
    if "dam_block_buy_volume" in out.columns:
        block = pd.to_numeric(out["dam_block_buy_volume"], errors="coerce")
        if "dam_matched_volume" in out.columns:
            matched = pd.to_numeric(out["dam_matched_volume"], errors="coerce")
            out["block_to_matched_ratio"] = safe_div(block, matched).to_numpy()
        out["block_buy_delta_1h"] = block.diff()
        out["night_block_pressure"] = out["night_block_flag"] * block

    # International commodity price composites
    if "ttf_try_mwh" in out.columns:
        ttf_try = pd.to_numeric(out["ttf_try_mwh"], errors="coerce")
        gas_s = pd.to_numeric(out.get("gas_share", pd.Series(np.nan, index=out.index)), errors="coerce")
        out["ttf_x_gas_share"] = ttf_try * gas_s
        if "grf_tl_1000sm3" in out.columns:
            grf = pd.to_numeric(out["grf_tl_1000sm3"], errors="coerce")
            # GRF is TL/1000Sm³; TTF-TRY is TL/MWh — ratio signals whether domestic gas is cheap vs Europe
            out["ttf_vs_grf_premium"] = ttf_try - grf * (1 / 10.55)  # approx TL/MWh from TL/1000Sm³ via calorific
    if "brent_try" in out.columns and "ttf_try_mwh" in out.columns:
        brent_try = pd.to_numeric(out["brent_try"], errors="coerce")
        ttf_try = pd.to_numeric(out["ttf_try_mwh"], errors="coerce")
        out["brent_ttf_try_ratio"] = safe_div(brent_try, ttf_try * 8.14).to_numpy()  # oil barrel vs gas MWh equiv

    # Dam fullness composites (mostly NaN until daily pipeline accumulates data; LightGBM handles NaN natively)
    if "dam_fullness_mean" in out.columns and "hydro_share" in out.columns:
        dam = pd.to_numeric(out["dam_fullness_mean"], errors="coerce")
        hydro = pd.to_numeric(out["hydro_share"], errors="coerce")
        out["hydro_dam_interaction"] = dam * hydro  # high dam + high hydro share → cheap supply signal

    # İstanbul baraj doluluk türev featureları (İBB, 2000-2024 günlük, ffill → saatlik)
    # Marmara bölgesi hidro koşulu proxy: düşük doluluk → kış kurakları → hidro kısıtı → PTF yüksek
    if "istanbul_genel" in out.columns:
        ist = pd.to_numeric(out["istanbul_genel"], errors="coerce")
        out["istanbul_dam_lag_7d"] = ist.shift(168)                        # 7 gün önceki seviye
        out["istanbul_dam_lag_30d"] = ist.shift(720)                       # 30 gün önceki seviye
        out["istanbul_dam_change_30d"] = ist - ist.shift(720)              # dolma/boşalma trendi
        roll90 = ist.rolling(2160, min_periods=240).mean()
        out["istanbul_dam_seasonal_dev"] = ist - roll90                    # mevsimsel sapma

    # --- Sıfır fiyat sinyal feature'ları ---
    # Şubat 2026 analizinden: PTF=0 saatlerinde gas_share=%3.7, ren_share=%52.7, gas≈980 MW
    _gas = pd.to_numeric(out.get("kgup_gas"), errors="coerce").fillna(0)
    _tot = pd.to_numeric(out.get("kgup_total"), errors="coerce").replace(0, np.nan)
    _lep = pd.to_numeric(out.get("load_forecast"), errors="coerce").replace(0, np.nan)
    _ren = pd.to_numeric(out.get("renewable_share"), errors="coerce").fillna(0)

    # Gaz payı — sıfır fiyat saatlerinde %3.7'ye düşüyor
    out["gas_share_raw"] = safe_div(_gas, _tot).to_numpy()

    # Gaz × yenilenebilir ters etkileşim: gaz düşükken yenilenebilir yüksek → PTF=0 bölgesi
    out["low_gas_high_ren"] = (1 - out["gas_share_raw"].fillna(1)) * _ren

    # Hidro surge sinyali: hidro/talep oranı yüksekse + gaz düşükse → Şubat 2026 rejimi
    _hyd = pd.to_numeric(out.get("kgup_dammed_hydro"), errors="coerce").fillna(0)
    out["hydro_gas_ratio"] = safe_div(_hyd, _gas.replace(0, np.nan)).to_numpy()

    # Talep normalize: saatlik yük / o ayin ortalama yükü (düşük talep + ucuz supply = PTF=0)
    if "month" in out.columns:
        load_monthly_avg = _lep.groupby(out["month"]).transform("mean")
        out["load_vs_monthly_avg"] = safe_div(_lep, load_monthly_avg).to_numpy()

    return out


# YEKDEM kapasiteleri — Excel listelerinden (MWe), 2020 için 2021 değeri kullanıldı
_YEKDEM_WIND_CAP = {
    2020: 8060, 2021: 8060, 2022: 9286, 2023: 8043,
    2024: 6751, 2025: 6805, 2026: 6268,
}
_YEKDEM_RIVER_CAP = {  # akarsu kanal tipi — su gelince üretiyor, gerçek must-run
    2020: 5681, 2021: 5681, 2022: 5113, 2023: 3637,
    2024: 2997, 2025: 2291, 2026: 1643,
}
_YEKDEM_BARAJLI_CAP = {  # rezervuarlı YEKDEM — garantili fiyat ama operatör zamanlar
    2020: 7577, 2021: 7577, 2022: 6911, 2023: 4111,
    2024: 3997, 2025: 4251, 2026: 2708,
}


def add_merit_order_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ucuzdan pahalıya merit order stack kurarak hangi santralin marjinal
    olduğunu her saat için sayısal olarak encode eder.

    Stack sırası:
      Tier 0 (~0 TL)   : Solar + YEKDEM Rüzgar + YEKDEM Akarsu + YEKDEM Barajlı
                         + Jeotermal + Biyokütle
      Tier 1 (~50 TL)  : Piyasa Rüzgar + Piyasa Akarsu (YEKDEM dışı)
      Tier 2 (~300 TL) : Linyit
      Tier 3 (~600 TL) : Taş Kömür + İthal Kömür
      Tier 4 (~800 TL) : Piyasa Barajlı HES (stratejik swing, YEKDEM dışı)
      Tier 5 (~2000 TL): Doğalgaz
    """
    out = df.copy()
    year = out["ts_hour"].dt.year

    # Kaynak bazlı KGÜP kolonları
    solar   = pd.to_numeric(out.get("kgup_solar"),       errors="coerce").fillna(0)
    wind    = pd.to_numeric(out.get("kgup_wind"),         errors="coerce").fillna(0)
    river   = pd.to_numeric(out.get("kgup_river"),        errors="coerce").fillna(0)
    geo     = pd.to_numeric(out.get("kgup_geothermal"),   errors="coerce").fillna(0)
    bio     = pd.to_numeric(out.get("kgup_biomass"),      errors="coerce").fillna(0)
    hydro   = pd.to_numeric(out.get("kgup_dammed_hydro"), errors="coerce").fillna(0)
    lignite = pd.to_numeric(out.get("kgup_lignite"),      errors="coerce").fillna(0)
    bcoal   = pd.to_numeric(out.get("kgup_black_coal"),   errors="coerce").fillna(0)
    icoal   = pd.to_numeric(out.get("kgup_import_coal"),  errors="coerce").fillna(0)
    load    = pd.to_numeric(out.get("load_forecast"),     errors="coerce")

    # YEKDEM kapasitesi tavan — min(KGÜP, YEKDEM_cap[yıl])
    yekdem_wind_cap   = year.map(_YEKDEM_WIND_CAP).fillna(6268).astype(float)
    yekdem_river_cap  = year.map(_YEKDEM_RIVER_CAP).fillna(1643).astype(float)
    yekdem_hydro_cap  = year.map(_YEKDEM_BARAJLI_CAP).fillna(2708).astype(float)

    yekdem_wind  = np.minimum(wind.values,  yekdem_wind_cap.values)
    yekdem_river = np.minimum(river.values, yekdem_river_cap.values)
    yekdem_hydro = np.minimum(hydro.values, yekdem_hydro_cap.values)  # YEKDEM barajlı → tier0

    # Piyasa kısımları (YEKDEM dışı) — fiyat duyarlı
    market_wind  = (wind  - yekdem_wind).clip(lower=0)
    market_river = (river - yekdem_river).clip(lower=0)
    market_hydro = (hydro - yekdem_hydro).clip(lower=0)  # piyasa barajlı → tier4

    # Merit order kümülatif arz katmanları
    tier0 = solar + yekdem_wind + yekdem_river + yekdem_hydro + geo + bio  # ~0 TL
    tier1 = tier0 + market_wind + market_river                              # ~50 TL
    tier2 = tier1 + lignite                                                 # ~300 TL
    tier3 = tier2 + bcoal + icoal                                           # ~600 TL
    tier4 = tier3 + market_hydro                                            # ~800 TL

    # Residual yük — her kattan sonra ne kadar kaldı?
    out["merit_residual_zero"]    = load - tier0  # <0 → PTF≈0 bölgesi
    out["merit_residual_mkt_ren"] = load - tier1  # <0 → piyasa yenilenebilir marjinal
    out["merit_residual_lignite"] = load - tier2  # <0 → linyit marjinal
    out["merit_residual_coal"]    = load - tier3  # <0 → kömür marjinal
    out["merit_residual_hydro"]   = load - tier4  # <0 → piyasa baraj marjinal

    # Gaz baskısı — tier4'ü geçen talep = gaz devreye giriyor
    out["gas_running_mw"] = (load - tier4).clip(lower=0)

    # Bileşen feature'lar (modele ek context)
    out["yekdem_total_mw"] = tier0
    out["market_ren_mw"]   = market_wind + market_river
    out["coal_total_mw"]   = bcoal + icoal + lignite

    return out


ZERO_PRICE_CLF_FEATURES = [
    # Ham KGÜP kaynakları
    "kgup_solar",
    "kgup_wind",
    "kgup_dammed_hydro",
    "kgup_river",
    "kgup_geothermal",
    "kgup_biomass",
    "kgup_total",
    "load_forecast",
    # Oran feature'ları
    "renewable_share",
    "solar_load_share",
    "hydro_share",
    "hydro_demand_ratio",
    "net_load_after_wind_solar",
    "net_load_coverage",
    # Merit order sinyalleri
    "merit_residual_zero",
    "merit_residual_hydro",
    "gas_running_mw",
    "yekdem_total_mw",
    # Şubat 2026 analizinden türetilen sıfır fiyat sinyalleri
    "gas_share_raw",           # PTF=0'da %3.7'ye düşüyor (normal: %26)
    "low_gas_high_ren",        # (1-gas_share) × ren_share etkileşimi
    "hydro_gas_ratio",         # hidro surge + gaz geri çekilmesi
    "load_vs_monthly_avg",     # talep / ay ortalaması — düşük talep PTF=0 riskini artırıyor
    # Zaman ve hava
    "hour",
    "month",
    "tr_radiation_mean",
]


def fit_zero_price_classifier(hourly: pd.DataFrame):
    """Fit binary LightGBM P(PTF<1) classifier using only 2020-2024 train-split rows.

    Returns (clf, feature_names) so the same feature list is used at apply time.
    Only feature columns that exist in the DataFrame are included.
    """
    from lightgbm import LGBMClassifier

    available = [f for f in ZERO_PRICE_CLF_FEATURES if f in hourly.columns]
    # 2025 dahil: yeni solar zero saatlerini (200+ örnek) classifier'a öğret
    train_mask = (hourly["ts_hour"] < pd.Timestamp("2026-01-01")) & hourly["ptf"].notna()
    train_df = hourly[train_mask].copy()
    y_train = (train_df["ptf"] < 1.0).astype(int)
    n_pos = int(y_train.sum())
    log(f"Zero-price classifier: {n_pos} positive / {len(y_train)} total (train split 2020-2025)")
    clf = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        n_jobs=-1,
        random_state=42,
        is_unbalance=True,
        verbose=-1,
    )
    clf.fit(train_df[available].fillna(-1), y_train)
    return clf, available


def apply_zero_price_classifier(hourly: pd.DataFrame, clf, clf_features: list[str]) -> pd.DataFrame:
    """Add 'zero_price_prob' column via the fitted zero-price classifier."""
    X = hourly[clf_features].fillna(-1)
    hourly = hourly.copy()
    hourly["zero_price_prob"] = clf.predict_proba(X)[:, 1]
    return hourly


def calendar_features(ts: pd.Series, prefix: str) -> pd.DataFrame:
    hour = ts.dt.hour
    dow = ts.dt.dayofweek
    month = ts.dt.month
    dates = ts.dt.date
    years = range(int(ts.dt.year.min()), int(ts.dt.year.max()) + 1)
    tr_holidays = holidays.Turkey(years=years)
    is_holiday = dates.map(lambda d: 1 if d in tr_holidays else 0).astype(int)
    is_pre_holiday = dates.map(lambda d: 1 if (pd.Timestamp(d) + pd.Timedelta(days=1)).date() in tr_holidays else 0).astype(int)
    is_post_holiday = dates.map(lambda d: 1 if (pd.Timestamp(d) - pd.Timedelta(days=1)).date() in tr_holidays else 0).astype(int)
    features = {
        f"{prefix}_hour": hour.astype(int),
        f"{prefix}_dow": dow.astype(int),
        f"{prefix}_month": month.astype(int),
        f"{prefix}_hour_sin": np.sin(2 * np.pi * hour / 24),
        f"{prefix}_hour_cos": np.cos(2 * np.pi * hour / 24),
        f"{prefix}_dow_sin": np.sin(2 * np.pi * dow / 7),
        f"{prefix}_dow_cos": np.cos(2 * np.pi * dow / 7),
        f"{prefix}_month_sin": np.sin(2 * np.pi * month / 12),
        f"{prefix}_month_cos": np.cos(2 * np.pi * month / 12),
        f"{prefix}_is_weekend": (dow >= 5).astype(int),
        f"{prefix}_is_holiday": is_holiday,
        f"{prefix}_is_holiday_or_weekend": ((dow >= 5).astype(int) | is_holiday).astype(int),
        f"{prefix}_is_pre_holiday": is_pre_holiday,
        f"{prefix}_is_post_holiday": is_post_holiday,
        f"{prefix}_ramadan_season_proxy": month.isin([3, 4]).astype(int),
    }
    return pd.DataFrame(features, index=ts.index)


def add_delivery_grf_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    required = {"delivery_grf_tl_1000sm3", "delivery_load_forecast"}
    if not required.issubset(out.columns):
        return out

    grf = pd.to_numeric(out["delivery_grf_tl_1000sm3"], errors="coerce")
    load = pd.to_numeric(out["delivery_load_forecast"], errors="coerce")
    default = pd.Series(np.nan, index=out.index)
    hour = pd.to_numeric(out.get("delivery_hour", default), errors="coerce")
    peak = hour.between(17, 22).astype(float)
    night = ((hour >= 22) | (hour < 6)).astype(float)
    day = 1.0 - night
    cooldown_default = pd.Series(1.0, index=out.index)
    cooldown = pd.to_numeric(out.get("delivery_holiday_cooldown", cooldown_default), errors="coerce").fillna(1.0)

    load_30d = pd.to_numeric(out.get("delivery_load_roll_mean_30d", default), errors="coerce")
    load_vs_30d = safe_div(load, load_30d).clip(lower=0, upper=1.5).fillna(1.0)
    demand_gate = np.log1p(load_vs_30d.clip(lower=0)).fillna(0.0)
    out["delivery_load_log1p"] = np.log1p(load.clip(lower=0))
    out["delivery_demand_ratio_log1p"] = demand_gate
    out["delivery_demand_gate_30d"] = demand_gate
    out["delivery_grf_peak_holiday_adjusted"] = grf * peak * cooldown * day
    out["delivery_grf_peak_demand_gated"] = grf * peak * cooldown * demand_gate * day
    out["delivery_grf_floor_proxy"] = grf * 0.4
    out["delivery_night_floor_proxy"] = out["delivery_grf_floor_proxy"] * night

    if "delivery_net_load_after_renewable" in out.columns:
        net_load = pd.to_numeric(out["delivery_net_load_after_renewable"], errors="coerce")
        out["delivery_net_load_grf_demand_gated"] = net_load * grf * cooldown * demand_gate * day
    return out


def add_history_features(hourly: pd.DataFrame) -> pd.DataFrame:
    out = hourly.copy()
    ptf = out["ptf"]
    for lag in [1, 2, 3, 4, 6, 12, 24, 25, 26, 48]:
        out[f"ptf_lag_{lag}"] = ptf.shift(lag)
    past = ptf.shift(1)
    for win in [6, 12, 24, 48]:
        roll = past.rolling(win, min_periods=max(3, min(win, 24)))
        out[f"ptf_roll_mean_{win}"] = roll.mean()
        out[f"ptf_roll_std_{win}"] = roll.std()
        out[f"ptf_roll_min_{win}"] = roll.min()
        out[f"ptf_roll_max_{win}"] = roll.max()
    out["ptf_low_ratio_24"] = (past <= 100).astype(float).rolling(24, min_periods=12).mean()
    out["ptf_zero_ratio_24"] = (past == 0).astype(float).rolling(24, min_periods=12).mean()
    out["ptf_spike_ratio_24"] = (past >= 3000).astype(float).rolling(24, min_periods=12).mean()
    out["ptf_d1_momentum"] = out["ptf_lag_24"] - out["ptf_lag_48"]

    # PTF rejim proxyleri — kısa vadeli volatilite ve spike prevalence
    out["ptf_spike_ratio_168"] = (past >= 3000).astype(float).rolling(168, min_periods=24).mean()
    # Akşam saatlerinde (18-22) son 7 günlük peak — duck curve şiddeti proxy
    evening_mask = out["hour"].between(18, 22)
    ptf_evening = ptf.where(evening_mask)
    out["ptf_evening_max_7d"] = ptf_evening.shift(1).rolling(168, min_periods=12).max()

    lagged_cols = [
        "smf",
        "real_consumption",
        "yal_yat_net",
        "gen_total",
        "gen_gas",
        "gen_wind",
        "gen_solar",
        "gen_dammed_hydro",    # realized dammed hydro — lag captures "yesterday was high hydro"
        "dam_matched_volume",  # post-auction: only safe as lagged anchor feature
    ]
    for col in lagged_cols:
        if col in out.columns:
            s = pd.to_numeric(out[col], errors="coerce")
            out[f"{col}_lag_24"] = s.shift(24)
            out[f"{col}_lag_168"] = s.shift(168)

    # Sustained hydro regime signal: 7-day rolling mean lagged by 24/168h
    # Distinguishes persistent hydro dominance (Feb 2026) from one-day spikes
    if "gen_dammed_hydro" in out.columns:
        hydro_s = pd.to_numeric(out["gen_dammed_hydro"], errors="coerce")
        roll7d = hydro_s.rolling(168, min_periods=72).mean()
        out["gen_hydro_roll_7d_lag_24"]  = roll7d.shift(24)
        out["gen_hydro_roll_7d_lag_168"] = roll7d.shift(168)

    if "smf" in out.columns:
        spread = pd.to_numeric(out["smf"], errors="coerce") - ptf
        out["smf_ptf_spread_lag_24"] = spread.shift(24)
        out["smf_ptf_spread_lag_168"] = spread.shift(168)
    return out


def assign_split(ts: pd.Series) -> pd.Series:
    split = pd.Series("ignore", index=ts.index, dtype="object")
    # 2025 Q2 (Nis-Haz) validation olarak ayrılıyor:
    # → Solar sezon alpha kalibrasyonu, 2026 ilkbaharının en iyi temsili
    q2_2025 = (ts >= "2025-04-01") & (ts < "2025-07-01")
    # Train: 2020-2024 + 2025'in Q2 dışındaki tüm dönemleri
    split[(ts >= "2020-01-01") & (ts < "2026-01-01") & ~q2_2025] = "train"
    # Val: 2025 Q2 — yoğun solar sezon, 2026 koşullarının proxy'i
    split[q2_2025] = "validation"
    # Test: 2026 full yıl
    split[ts >= "2026-01-01"] = "test"
    return split


def build_supervised_dataset(profile: Profile) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    log(f"Building hourly market table for profile={profile.name}")
    hourly = add_history_features(build_hourly_market_table())
    hourly = hourly.sort_values("ts_hour").reset_index(drop=True)
    _zero_clf, _zero_clf_features = fit_zero_price_classifier(hourly)
    hourly = apply_zero_price_classifier(hourly, _zero_clf, _zero_clf_features)
    by_ts = hourly.set_index("ts_hour", drop=False).sort_index()
    ptf_max = hourly.dropna(subset=["ptf"])["ts_hour"].max()
    anchors = hourly.loc[
        (hourly["ts_hour"] <= ptf_max - pd.Timedelta(hours=1))
        & (hourly["ts_hour"] >= hourly["ts_hour"].min() + pd.Timedelta(hours=168))
    ].copy()

    market_cols = [
        "load_forecast",
        "kgup_total",
        "kgup_gas",
        "kgup_wind",
        "kgup_solar",
        "kgup_dammed_hydro",
        "kgup_river",
        "kgup_import_coal",
        "net_load_after_wind_solar",
        "net_load_after_renewable",
        "load_minus_kgup_total",
        "renewable_share",
        "gas_share",
        "coal_share",
        "hydro_share",
        "wind_load_share",
        "solar_load_share",
        "renewable_load_share",
        "thermal_share",
        "gas_vs_renewable",
        "thermal_tightness_pressure",
        "kgup_total_delta_1h",
        "kgup_gas_delta_1h",
        "kgup_renewable_delta_1h",
        "load_forecast_delta_1h",
        "load_roll_mean_30d",
        "load_week_ratio",
        "load_week_momentum",
        "load_vs_30d_ratio",
        "load_log1p",
        "demand_below_30d_avg",
        "net_load_renewable_delta_1h",
        "ramp_tightness",
        "morning_ramp_flag",
        "evening_ramp_flag",
        "night_block_flag",
        "net_load_coverage",
        # GRF: teslimat günü sabah yayımlanıyor → gün öncesi bilinmiyor → sadece lag güvenli
        "grf_tl_lag_24",          # dünkü GRF TL (safe)
        "grf_usd_mmbtu_lag_24",   # dünkü GRF USD (safe)
        # grf_tl_change_7d vb. hesaplamak için lag_24 yeterli; aşağıdaki türevler lag_24'ten türetilmeli
        # National weighted temperature & HVAC features (from fetch_temperature.py)
        "tr_temp_mean",
        "tr_apparent_temp_mean",
        "tr_humidity_mean",
        "tr_cloud_cover_mean",
        "tr_radiation_mean",
        "tr_wind_speed_mean",
        "tr_temp_lag_24",
        "tr_temp_lag_168",
        "tr_temp_delta_24",
        "tr_cooling_degree",
        "tr_heating_degree",
        "tr_heatwave_flag",
        "tr_apparent_temp_delta_24",
        "load_x_cooling",
        "load_x_heating",
        "load_x_heatwave",
        # Per-city temperatures (regional load signal: Istanbul vs Adana summer dynamics differ)
        "temp_istanbul",
        "temp_ankara",
        "temp_izmir",
        "temp_bursa",
        "temp_antalya",
        "temp_adana",
        "temp_konya",
        "temp_diyarbakir",
        "temp_samsun",
        # International commodity prices (daily forward-filled — known at anchor time)
        # Exchange rates excluded: redundant with GRF (TL gas cost already encodes USD/TRY); ablation ΔVAL=−2.4
        "brent_usd",
        "ttf_eur_mwh",
        "ttf_try_mwh",
        "coal_api2_usd",
        # "brent_try" → ablasyon: drift=2.5, importance=211, 2026'da TL depresiasyonu+İran şoku nedeniyle
        #               eğitim dağılımı dışına çıkıyor → test MAE'yi +13.5 artırıyor. Kaldırıldı.
        "henry_hub_usd",
        "brent_usd_change_7d",
        "ttf_eur_mwh_change_7d",
        "brent_usd_roll_mean_30d",
        "ttf_eur_mwh_roll_mean_30d",
        # "ttf_x_gas_share" → ttf_try_mwh look-ahead (teslimat günü bilinmiyor) — kaldırıldı
        # "ttf_vs_grf_premium" → grf_tl_1000sm3 look-ahead içeriyor — kaldırıldı
        # "brent_ttf_try_ratio" → brent_try'a bağımlı, o kaldırılınca bu da anlamsız

        # Zero-price classifier output — P(PTF<1): solar overcapacity signal
        "zero_price_prob",
        # Hydro displacement features (Feb 2026 hydro-surge regime)
        "hydro_demand_ratio",
        "hydro_gas_displacement",
        # "gas_margin_proximity" → ablasyon: drift=2.3, gas 2026'da %10'a düştü,
        #                          2020-2024 rejimi için tasarlandı, test MAE'yi +7.0 artırıyor. Kaldırıldı.
        # "grf_x_gas_share" → grf_tl_1000sm3 look-ahead içeriyor — kaldırıldı
        # İstanbul baraj doluluk (İBB 2000-2024) — Marmara hidro stres proxy
        "istanbul_genel",
        "istanbul_dam_lag_7d",
        "istanbul_dam_lag_30d",
        "istanbul_dam_change_30d",
        "istanbul_dam_seasonal_dev",
        # Merit order stack features — YEKDEM tabanlı marjinal santral tespiti
        "merit_residual_zero",     # <0 → PTF≈0 bölgesi (must-run talebi aşıyor)
        "merit_residual_mkt_ren",  # <0 → piyasa yenilenebilir marjinal
        "merit_residual_lignite",  # <0 → linyit marjinal
        "merit_residual_coal",     # <0 → kömür marjinal
        "merit_residual_hydro",    # <0 → barajlı HES marjinal
        "gas_running_mw",          # >0 → kaç MW gaz çalışıyor
        "yekdem_total_mw",         # toplam YEKDEM must-run üretim
        "market_ren_mw",           # piyasa yenilenebilir (YEKDEM dışı)
        "coal_total_mw",           # toplam kömür üretimi
        # Sıfır fiyat sinyal feature'ları (Şubat 2026 analizi)
        "gas_share_raw",
        "low_gas_high_ren",
        # "hydro_gas_ratio" → ablasyon: drift=6.4 (en yüksek), test MAE'yi +7.0 artırıyor. Kaldırıldı.
        "load_vs_monthly_avg",
        # Solar rampa — duck curve evening spike (geçmiş düşüş → gaz rampa baskısı)
        "solar_drop_past_3h",      # son 3 saatte solar kaç MW düştü
        "solar_drop_past_6h",      # son 6 saatte solar kaç MW düştü
        "solar_drop_next_3h",      # önümüzdeki 3 saatte solar kaç MW düşecek (KGÜP'ten)
        # "grf_x_solar_drop_3h" → grf_tl_1000sm3 look-ahead — kaldırıldı
        "solar_drop_pct_of_day",   # günlük solar max'a göre normalize
        # "grf_x_solar_drop_6h"   → grf_tl_1000sm3 look-ahead — kaldırıldı
        # Duck curve sinyalleri — yük-normalize, ölçek bağımsız
        "solar_pct_of_daily_peak",  # kalan solar oranı (1=zirve, 0=güneş battı)
        "solar_deficit_pct_load",   # (zirve-şimdiki solar) / load — akşam spike sinyali
        "gas_ramp_pct_load",        # 3h gaz artışı / load — hızlı devreye giriş
        "duck_stress_normalized",   # (solar_kayıp/load) × (gaz/load) — bileşik
    ]
    orderbook_cols = [
        "dam_bid_volume",
        "dam_sell_offer_volume",
        # dam_matched_volume excluded: post-auction realized value, not available before PTF is set.
        # Use dam_matched_volume_lag_24 (anchor feature) instead.
        "dam_price_independent_buy",
        "dam_price_independent_sell",
        "dam_block_buy_volume",
        "price_independent_balance",
        "price_independent_pressure",
        "dam_bid_sell_ratio",
        "dam_bid_sell_balance",
        "block_to_matched_ratio",
        "block_buy_delta_1h",
        "night_block_pressure",
    ]
    history_cols = [c for c in hourly.columns if c.startswith("ptf_lag_") or c.startswith("ptf_roll_") or c.startswith("ptf_low_") or c.startswith("ptf_zero_") or c.startswith("ptf_spike_")]
    history_cols += ["ptf_d1_momentum", "ptf_evening_max_7d", "ptf_spike_ratio_168"]
    lagged_realized_cols = [c for c in hourly.columns if c.endswith("_lag_24") or c.endswith("_lag_168")]

    base_anchor_cols = ["ts_hour", "ptf"] + history_cols
    if profile.include_lagged_realized:
        base_anchor_cols += [c for c in lagged_realized_cols if c not in base_anchor_cols]
    base_anchor_cols = [c for c in base_anchor_cols if c in anchors.columns]

    delivery_cols: list[str] = ["ptf"]
    if profile.include_market_forecasts:
        delivery_cols += [c for c in market_cols if c in hourly.columns]
    if profile.include_orderbook:
        delivery_cols += [c for c in orderbook_cols if c in hourly.columns]
    delivery_cols = list(dict.fromkeys(delivery_cols))
    delivery_lookup = by_ts[delivery_cols].copy()

    frames: list[pd.DataFrame] = []
    for horizon in TARGET_HORIZONS:
        frame = anchors[base_anchor_cols].copy()
        frame = frame.rename(columns={"ts_hour": "anchor_ts_hour", "ptf": "anchor_ptf"})
        frame["delivery_ts_hour"] = frame["anchor_ts_hour"] + pd.Timedelta(hours=horizon)
        frame["horizon"] = horizon

        delivery = delivery_lookup.reindex(frame["delivery_ts_hour"].to_numpy()).reset_index(drop=True)
        delivery = delivery.rename(columns={c: f"delivery_{c}" for c in delivery.columns})
        frame = pd.concat([frame.reset_index(drop=True), delivery], axis=1)
        frame["target_ptf"] = frame["delivery_ptf"]

        baseline_ts = frame["delivery_ts_hour"] - pd.Timedelta(hours=24)
        frame["baseline_d1_ptf"] = by_ts["ptf"].reindex(baseline_ts.to_numpy()).to_numpy()
        frame["baseline_d2_ptf"] = by_ts["ptf"].reindex((frame["delivery_ts_hour"] - pd.Timedelta(hours=48)).to_numpy()).to_numpy()
        frame["baseline_week_ptf"] = by_ts["ptf"].reindex((frame["delivery_ts_hour"] - pd.Timedelta(hours=168)).to_numpy()).to_numpy()
        frame["baseline_recent_3d_mean"] = frame[["baseline_d1_ptf", "baseline_d2_ptf", "baseline_week_ptf"]].mean(axis=1)
        frame["baseline_d1_d2_momentum"] = frame["baseline_d1_ptf"] - frame["baseline_d2_ptf"]
        frame["baseline_d1_week_momentum"] = frame["baseline_d1_ptf"] - frame["baseline_week_ptf"]
        frame = frame.drop(columns=["delivery_ptf"])
        frame = frame.dropna(subset=["target_ptf", "baseline_d1_ptf"])
        if frame.empty:
            continue
        frame["residual_target"] = frame["target_ptf"] - frame["baseline_d1_ptf"]

        rename_anchor = {
            c: f"anchor_{c}"
            for c in frame.columns
            if c not in {
                "anchor_ts_hour",
                "delivery_ts_hour",
                "horizon",
                "anchor_ptf",
                "target_ptf",
                "baseline_d1_ptf",
                "residual_target",
            }
            and not c.startswith("delivery_")
        }
        frame = frame.rename(columns=rename_anchor)

        if profile.include_market_forecasts:
            frame["delivery_load_x_renewable"] = frame.get("delivery_load_forecast", np.nan) * frame.get("delivery_renewable_share", np.nan)
            frame["delivery_load_x_gas"] = frame.get("delivery_load_forecast", np.nan) * frame.get("delivery_gas_share", np.nan)


        frames.append(frame)

    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if data.empty:
        raise RuntimeError("No supervised rows could be built")
    cal_anchor = calendar_features(data["anchor_ts_hour"], "anchor")
    cal_delivery = calendar_features(data["delivery_ts_hour"], "delivery")
    data = pd.concat([data, cal_anchor, cal_delivery], axis=1)
    if USE_GRF_GATED_INTERACTIONS:
        data = add_delivery_grf_interactions(data)
    data["split"] = assign_split(data["delivery_ts_hour"])
    data = data[data["split"] != "ignore"].copy()

    forbidden = {"anchor_ts_hour", "delivery_ts_hour", "target_ptf", "residual_target", "split"}
    forbidden |= {
        "daytime_flag",
        "load_roll_mean_30d",
        "load_week_ratio",
        "load_week_momentum",
        "load_vs_30d_ratio",
        "load_log1p",
        "demand_below_30d_avg",
        "grf_peak_effect",
        "grf_floor_proxy",
        "night_floor_proxy",
        "demand_per_grf",
        "demand_per_grf_log1p",
        "net_load_x_grf",
        "delivery_load_log1p",
        "delivery_demand_ratio_log1p",
        "delivery_demand_gate_30d",
        "delivery_load_roll_mean_30d",
        "delivery_load_week_ratio",
        "delivery_load_week_momentum",
        "delivery_load_vs_30d_ratio",
        "delivery_demand_below_30d_avg",
        "delivery_grf_peak_holiday_adjusted",
        "delivery_grf_peak_demand_gated",
        "delivery_grf_floor_proxy",
        "delivery_night_floor_proxy",
        "delivery_net_load_grf_demand_gated",
    }
    # Also forbid delivery-hour versions of any realized column
    forbidden |= {f"delivery_{c}" for c in REALIZED_DELIVERY_COLS}
    feature_cols = [c for c in data.columns if c not in forbidden]
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(data[c])]
    # Safety assertion: verify no realized delivery columns slipped through
    leaky = [c for c in feature_cols if any(c == f"delivery_{r}" for r in REALIZED_DELIVERY_COLS)]
    if leaky:
        raise RuntimeError(f"Leakage detected — realized delivery columns in feature set: {leaky}")
    meta = {
        "profile": profile.name,
        "rows": int(len(data)),
        "feature_count": int(len(feature_cols)),
        "ptf_max_ts_hour": str(ptf_max),
        "split_counts": {str(k): int(v) for k, v in data["split"].value_counts().items()},
        "hourly_columns": list(hourly.columns),
    }
    return data, feature_cols, meta


def metric_blob(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    err = pred - y_true
    abs_err = np.abs(err)
    return {
        "rows": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, pred))),
        "bias": float(err.mean()),
        "median_ae": float(np.median(abs_err)),
        "p90_ae": float(np.quantile(abs_err, 0.9)),
        "pct_le_100": float((abs_err <= 100).mean()),
        "wape": float(abs_err.sum() / np.abs(y_true).sum() * 100),
    }


def compute_sample_weights(df: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    """Compute per-row training weights to emphasize hard-to-learn regimes."""
    w = np.ones(len(df), dtype=float)
    w[np.abs(y) >= 1000] *= 2.0
    w[df["target_ptf"].to_numpy(float) <= 100] *= 1.7
    return w


def walk_forward_cv(
    data: pd.DataFrame,
    feature_cols: list[str],
    params: dict[str, Any],
) -> list[dict[str, Any]]:
    """3-fold expanding-window time-series CV on yearly delivery boundaries.

    Gives an honest estimate of generalization before the final train/val/test fit.
    Each fold trains on everything up to the end of the train window and validates on
    the immediately following year — mimicking the way the model is deployed.
    """
    folds = [
        ("2020", "2022", "2023"),
        ("2020", "2023", "2024"),
        ("2020", "2024", "2025"),
        ("2020", "2025", "2026"),  # canlı tahmin performansını CV'de ölç
    ]
    cv_params = {**params, "n_estimators": min(params.get("n_estimators", 350), 300), "learning_rate": 0.05}
    results: list[dict[str, Any]] = []
    for train_start, train_end, val_year in folds:
        ts = data["delivery_ts_hour"].dt.year
        train = data[(ts >= int(train_start)) & (ts <= int(train_end))].copy()
        val = data[ts == int(val_year)].copy()
        if train.empty or val.empty:
            continue
        x_tr = train[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y_tr = train["residual_target"].to_numpy(float)
        cv_weights = compute_sample_weights(train, y_tr)
        model = LGBMRegressor(**cv_params)
        model.fit(x_tr, y_tr, sample_weight=cv_weights)
        x_val = val[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        raw_res = model.predict(x_val)
        pred = np.clip(val["baseline_d1_ptf"].to_numpy(float) + raw_res, 0, 5000)
        base = val["baseline_d1_ptf"].to_numpy(float)
        y_val = val["target_ptf"].to_numpy(float)
        model_mae = float(mean_absolute_error(y_val, pred))
        pers_mae = float(mean_absolute_error(y_val, base))
        results.append({
            "fold": f"train={train_start}-{train_end} val={val_year}",
            "train_rows": int(len(train)),
            "val_rows": int(len(val)),
            "model_mae": round(model_mae, 2),
            "persistence_mae": round(pers_mae, 2),
            "improvement_pct": round((1 - model_mae / pers_mae) * 100, 1) if pers_mae > 0 else 0.0,
        })
        log(f"  CV fold {train_start}-{train_end}→{val_year}: MAE={model_mae:.1f} vs persistence={pers_mae:.1f} ({results[-1]['improvement_pct']}% better)")
    return results


def fit_profile(profile: Profile, *, quick: bool = False) -> dict[str, Any]:
    data, feature_cols, meta = build_supervised_dataset(profile)
    train = data[data["split"] == "train"].copy()
    val = data[data["split"] == "validation"].copy()
    test = data[data["split"] == "test"].copy()

    params = {
        "n_estimators": 2000 if not quick else 500,
        "learning_rate": 0.02 if not quick else 0.04,
        "num_leaves": 63,
        "min_child_samples": 50,    # min_data_in_leaf — overfitting'i azaltır
        "subsample": 0.9,
        "colsample_bytree": 0.85,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    early_stopping_rounds = 50

    x_train = train[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_train = train["residual_target"].to_numpy(float)
    weights = compute_sample_weights(train, y_train)

    x_val_es = val[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_val_es = val["residual_target"].to_numpy(float)

    log(f"Training {profile.name}: rows={len(train)} features={len(feature_cols)}")
    log(f"Running walk-forward CV for {profile.name}…")
    cv_folds = walk_forward_cv(data, feature_cols, params)

    from lightgbm import early_stopping as lgb_early_stopping, log_evaluation as lgb_log_evaluation
    model = LGBMRegressor(**params)
    model.fit(
        x_train, y_train,
        sample_weight=weights,
        eval_set=[(x_val_es, y_val_es)],
        callbacks=[
            lgb_early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            lgb_log_evaluation(period=-1),
        ],
    )
    log(f"Best iteration: {model.best_iteration_} (early stopping at {early_stopping_rounds})")

    def predict(frame: pd.DataFrame) -> np.ndarray:
        x = frame[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        raw = model.predict(x)
        return np.clip(frame["baseline_d1_ptf"].to_numpy(float) + raw, 0, 5000)

    val_pred_raw = predict(val)
    # Validation shrinkage keeps the residual model from over-correcting when market features are stale.
    # Delivery hour is more important than horizon for DAM shape: solar-window corrections and
    # evening-ramp corrections need different trust levels.
    alphas: dict[str, float] = {}
    val_raw_res = val_pred_raw - val["baseline_d1_ptf"].to_numpy(float)
    delivery_hours = val["delivery_ts_hour"].dt.hour.to_numpy(int)
    for hour in range(24):
        mask = delivery_hours == hour
        if not mask.any():
            alphas[str(hour)] = 0.5
            continue
        best_alpha = 0.0
        best_mae = float("inf")
        for alpha in np.linspace(0, 1, 21):
            pred = np.clip(val.loc[mask, "baseline_d1_ptf"].to_numpy(float) + alpha * val_raw_res[mask], 0, 5000)
            mae = mean_absolute_error(val.loc[mask, "target_ptf"].to_numpy(float), pred)
            if mae < best_mae:
                best_alpha = float(alpha)
                best_mae = float(mae)
        alphas[str(hour)] = best_alpha

    def predict_with_alpha(frame: pd.DataFrame) -> np.ndarray:
        x = frame[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        raw_res = model.predict(x)
        hours = frame["delivery_ts_hour"].dt.hour.to_numpy(int)
        alpha = np.array([alphas.get(str(int(h)), 0.5) for h in hours])
        return np.clip(frame["baseline_d1_ptf"].to_numpy(float) + alpha * raw_res, 0, 5000)

    val_pred = predict_with_alpha(val)
    test_pred = predict_with_alpha(test)
    val_base = val["baseline_d1_ptf"].to_numpy(float)
    test_base = test["baseline_d1_ptf"].to_numpy(float)
    result = {
        "profile": profile.name,
        "meta": meta,
        "params": params,
        "hour_alphas": alphas,
        "cv_folds": cv_folds,
        "metrics": {
            "validation_model": metric_blob(val["target_ptf"].to_numpy(float), val_pred),
            "validation_persistence": metric_blob(val["target_ptf"].to_numpy(float), val_base),
            "test_model": metric_blob(test["target_ptf"].to_numpy(float), test_pred),
            "test_persistence": metric_blob(test["target_ptf"].to_numpy(float), test_base),
        },
        "feature_cols": feature_cols,
        "model": model,
    }
    return result


def save_profile_result(result: dict[str, Any]) -> None:
    profile_dir = MODEL_DIR / result["profile"]
    profile_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(result["model"], profile_dir / "model.joblib")
    (profile_dir / "feature_columns.json").write_text(json.dumps(result["feature_cols"], indent=2) + "\n")
    (profile_dir / "hour_alphas.json").write_text(json.dumps(result["hour_alphas"], indent=2) + "\n")
    payload = {k: v for k, v in result.items() if k not in {"model", "feature_cols"}}
    (profile_dir / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")


def train_all(*, quick: bool = False, profiles: list[str] | None = None) -> dict[str, Any]:
    selected = [p for p in PROFILES if profiles is None or p.name in profiles]
    results = []
    for profile in selected:
        result = fit_profile(profile, quick=quick)
        save_profile_result(result)
        slim = {k: v for k, v in result.items() if k not in {"model", "feature_cols"}}
        slim["feature_count"] = len(result["feature_cols"])
        results.append(slim)

    best = min(results, key=lambda r: r["metrics"]["validation_model"]["mae"])
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "selected_profile.json").write_text(json.dumps({"profile": best["profile"]}, indent=2) + "\n")
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "quick": quick,
        "selected_profile": best["profile"],
        "results": results,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "rolling_ptf_next24_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n")
    write_metrics_md(report)
    return report


def write_metrics_md(report: dict[str, Any]) -> None:
    lines = [
        "# Rolling Next-24 PTF Metrics",
        "",
        f"Generated: `{report['generated_at']}`",
        f"Selected profile: `{report['selected_profile']}`",
        "",
        "## Final Train/Val/Test Split",
        "",
        "| Profile | Val MAE | Val persistence | Test MAE | Test persistence | Features | Rows |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in report["results"]:
        m = r["metrics"]
        lines.append(
            "| {profile} | {vmae:.2f} | {vp:.2f} | {tmae:.2f} | {tp:.2f} | {fc} | {rows} |".format(
                profile=r["profile"],
                vmae=m["validation_model"]["mae"],
                vp=m["validation_persistence"]["mae"],
                tmae=m["test_model"]["mae"],
                tp=m["test_persistence"]["mae"],
                fc=r["feature_count"],
                rows=r["meta"]["rows"],
            )
        )
    lines += [
        "",
        "## Walk-Forward Cross-Validation (expanding window)",
        "",
        "| Profile | Fold | Train rows | Val rows | Model MAE | Persistence MAE | Improvement |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in report["results"]:
        for fold in r.get("cv_folds", []):
            lines.append(
                "| {profile} | {fold} | {tr} | {vr} | {mae:.2f} | {pers:.2f} | {imp:.1f}% |".format(
                    profile=r["profile"],
                    fold=fold["fold"],
                    tr=fold["train_rows"],
                    vr=fold["val_rows"],
                    mae=fold["model_mae"],
                    pers=fold["persistence_mae"],
                    imp=fold["improvement_pct"],
                )
            )
    (REPORT_DIR / "rolling_ptf_next24_metrics.md").write_text("\n".join(lines) + "\n")


def load_selected_model(profile_name: str | None = None) -> tuple[str, Any, list[str], dict[str, float]]:
    if profile_name is None:
        selected_path = MODEL_DIR / "selected_profile.json"
        if selected_path.exists():
            profile_name = json.loads(selected_path.read_text())["profile"]
        else:
            profile_name = "forecast_market"
    profile_dir = MODEL_DIR / profile_name
    model = joblib.load(profile_dir / "model.joblib")
    feature_cols = json.loads((profile_dir / "feature_columns.json").read_text())
    alphas = {str(k): float(v) for k, v in json.loads((profile_dir / "hour_alphas.json").read_text()).items()}
    return profile_name, model, feature_cols, alphas


def forecast_next24(*, profile_name: str | None = None) -> pd.DataFrame:
    profile_name, model, feature_cols, alphas = load_selected_model(profile_name)
    profile = next((p for p in PROFILES if p.name == profile_name), PROFILES[1])
    hourly = add_history_features(build_hourly_market_table()).sort_values("ts_hour").reset_index(drop=True)
    _zero_clf, _zero_clf_features = fit_zero_price_classifier(hourly)
    hourly = apply_zero_price_classifier(hourly, _zero_clf, _zero_clf_features)
    latest_ptf_ts = hourly.dropna(subset=["ptf"])["ts_hour"].max()
    by_ts = hourly.set_index("ts_hour", drop=False)
    rows = []
    for horizon in TARGET_HORIZONS:
        delivery_ts = latest_ptf_ts + pd.Timedelta(hours=horizon)
        baseline_ts = delivery_ts - pd.Timedelta(hours=24)
        if baseline_ts not in by_ts.index:
            raise RuntimeError(f"Missing baseline timestamp {baseline_ts}")
        delivery_row = by_ts.loc[delivery_ts] if delivery_ts in by_ts.index else by_ts.loc[baseline_ts].copy()
        anchor_row = by_ts.loc[latest_ptf_ts]
        baseline_d1 = float(by_ts.loc[baseline_ts, "ptf"])
        baseline_d2 = by_ts["ptf"].reindex([delivery_ts - pd.Timedelta(hours=48)]).iloc[0]
        baseline_week = by_ts["ptf"].reindex([delivery_ts - pd.Timedelta(hours=168)]).iloc[0]
        row = {
            "anchor_ts_hour": latest_ptf_ts,
            "delivery_ts_hour": delivery_ts,
            "horizon": horizon,
            "baseline_d1_ptf": baseline_d1,
            "baseline_d2_ptf": float(baseline_d2) if pd.notna(baseline_d2) else np.nan,
            "baseline_week_ptf": float(baseline_week) if pd.notna(baseline_week) else np.nan,
            "anchor_ptf": float(anchor_row["ptf"]),
        }
        row["baseline_recent_3d_mean"] = np.nanmean([row["baseline_d1_ptf"], row["baseline_d2_ptf"], row["baseline_week_ptf"]])
        row["baseline_d1_d2_momentum"] = row["baseline_d1_ptf"] - row["baseline_d2_ptf"] if pd.notna(row["baseline_d2_ptf"]) else np.nan
        row["baseline_d1_week_momentum"] = row["baseline_d1_ptf"] - row["baseline_week_ptf"] if pd.notna(row["baseline_week_ptf"]) else np.nan
        for col in [c for c in hourly.columns if c.startswith("ptf_lag_") or c.startswith("ptf_roll_") or c.startswith("ptf_low_") or c.startswith("ptf_zero_") or c.startswith("ptf_spike_")]:
            row[f"anchor_{col}"] = anchor_row.get(col, np.nan)
        row["anchor_ptf_d1_momentum"] = anchor_row.get("ptf_d1_momentum", np.nan)
        row["anchor_ptf_week_momentum"] = anchor_row.get("ptf_week_momentum", np.nan)
        if profile.include_lagged_realized:
            for col in [c for c in hourly.columns if c.endswith("_lag_24") or c.endswith("_lag_168")]:
                row[f"anchor_{col}"] = anchor_row.get(col, np.nan)
        if profile.include_market_forecasts:
            for col in hourly.columns:
                if col in {
                    "load_forecast",
                    "kgup_total",
                    "kgup_gas",
                    "kgup_wind",
                    "kgup_solar",
                    "kgup_dammed_hydro",
                    "kgup_river",
                    "kgup_import_coal",
                    "net_load_after_wind_solar",
                    "net_load_after_renewable",
                    "load_minus_kgup_total",
                    "renewable_share",
                    "gas_share",
                    "coal_share",
                    "hydro_share",
                    "wind_load_share",
                    "solar_load_share",
                    "renewable_load_share",
                    "thermal_share",
                    "gas_vs_renewable",
                    "cheap_supply_pressure",
                    "thermal_tightness_pressure",
                    "kgup_total_delta_1h",
                    "kgup_gas_delta_1h",
                    "kgup_renewable_delta_1h",
                    "load_forecast_delta_1h",
                    "load_roll_mean_30d",
                    "load_week_ratio",
                    "load_week_momentum",
                    "load_vs_30d_ratio",
                    "load_log1p",
                    "demand_below_30d_avg",
                    "net_load_renewable_delta_1h",
                    "ramp_tightness",
                    "morning_ramp_flag",
                    "evening_ramp_flag",
                    "night_block_flag",
                    "net_load_coverage",
                    "grf_tl_1000sm3",
                    "grf_usd_1000sm3",
                    "grf_eur_mwh",
                    "grf_usd_mmbtu",
                    "grf_tl_lag_24",
                    "grf_tl_change_7d",
                    "grf_tl_pct_change_7d",
                    "grf_tl_roll_mean_7d",
                    "grf_tl_roll_mean_30d",
                    "grf_usd_mmbtu_lag_24",
                    "grf_usd_mmbtu_change_7d",
                    "tr_temp_mean",
                    "tr_apparent_temp_mean",
                    "tr_humidity_mean",
                    "tr_cloud_cover_mean",
                    "tr_radiation_mean",
                    "tr_wind_speed_mean",
                    "tr_temp_lag_24",
                    "tr_temp_lag_168",
                    "tr_temp_delta_24",
                    "tr_cooling_degree",
                    "tr_heating_degree",
                    "tr_heatwave_flag",
                    "tr_apparent_temp_delta_24",
                    "load_x_cooling",
                    "load_x_heating",
                    "load_x_heatwave",
                    "temp_istanbul",
                    "temp_ankara",
                    "temp_izmir",
                    "temp_bursa",
                    "temp_antalya",
                    "temp_adana",
                    "temp_konya",
                    "temp_diyarbakir",
                    "temp_samsun",
                    "usd_try_buy",
                    "eur_try_buy",
                    "eur_usd_cross_buy",
                    "usd_try_buy_lag_1d",
                    "eur_try_buy_lag_1d",
                    "usd_try_buy_change_7d",
                    "eur_try_buy_change_7d",
                    "usd_try_buy_pct_change_7d",
                    "eur_try_buy_pct_change_7d",
                    "usd_try_buy_roll_mean_7d",
                    "eur_try_buy_roll_mean_7d",
                    "zero_price_prob",
                    "hydro_demand_ratio",
                    "hydro_gas_displacement",
                    "gas_margin_proximity",
                    "grf_x_gas_share",
                }:
                    row[f"delivery_{col}"] = delivery_row.get(col, np.nan)
            row["delivery_load_x_renewable"] = delivery_row.get("load_forecast", np.nan) * delivery_row.get("renewable_share", np.nan)
            row["delivery_load_x_gas"] = delivery_row.get("load_forecast", np.nan) * delivery_row.get("gas_share", np.nan)
            row["delivery_cheap_minus_thermal"] = delivery_row.get("cheap_supply_pressure", np.nan) - delivery_row.get("thermal_tightness_pressure", np.nan)
            row["delivery_netload_x_thermal"] = delivery_row.get("net_load_after_renewable", np.nan) * delivery_row.get("thermal_share", np.nan)
        if profile.include_orderbook:
            for col in [
                "dam_bid_volume",
                "dam_sell_offer_volume",
                # dam_matched_volume excluded: post-auction, not available before PTF
                "dam_price_independent_buy",
                "dam_price_independent_sell",
                "dam_block_buy_volume",
                "price_independent_balance",
                "price_independent_pressure",
                "dam_bid_sell_ratio",
                "dam_bid_sell_balance",
                "block_to_matched_ratio",
                "block_buy_delta_1h",
                "night_block_pressure",
            ]:
                if col in hourly.columns:
                    row[f"delivery_{col}"] = delivery_row.get(col, np.nan)
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame = pd.concat([frame, calendar_features(frame["anchor_ts_hour"], "anchor"), calendar_features(frame["delivery_ts_hour"], "delivery")], axis=1)
    if USE_GRF_GATED_INTERACTIONS:
        frame = add_delivery_grf_interactions(frame)
    x = pd.DataFrame(
        {col: frame[col] if col in frame.columns else 0.0 for col in feature_cols},
        index=frame.index,
    )
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    raw_residual = model.predict(x)
    delivery_hours = frame["delivery_ts_hour"].dt.hour.to_numpy(int)
    alpha = np.array([alphas.get(str(int(h)), 0.5) for h in delivery_hours])
    frame["predicted_residual"] = raw_residual
    frame["shrinkage_alpha"] = alpha
    frame["predicted_ptf"] = np.clip(frame["baseline_d1_ptf"].to_numpy(float) + alpha * raw_residual, 0, 5000)
    frame["profile"] = profile_name
    out_cols = [
        "anchor_ts_hour",
        "delivery_ts_hour",
        "horizon",
        "baseline_d1_ptf",
        "anchor_ptf",
        "predicted_residual",
        "shrinkage_alpha",
        "predicted_ptf",
        "profile",
    ]
    out = frame[out_cols].copy()
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PRED_DIR / "rolling_ptf_next24_forecast.csv"
    out.to_csv(out_path, index=False)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "profile": profile_name,
        "anchor_ts_hour": str(latest_ptf_ts),
        "rows": int(len(out)),
        "output_csv": str(out_path),
        "mean_predicted_ptf": float(out["predicted_ptf"].mean()),
        "min_predicted_ptf": float(out["predicted_ptf"].min()),
        "max_predicted_ptf": float(out["predicted_ptf"].max()),
    }
    (PRED_DIR / "rolling_ptf_next24_forecast.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "forecast", "all"])
    parser.add_argument("--quick", action="store_true", help="Use smaller LightGBM models for faster iteration.")
    parser.add_argument("--profile", action="append", help="Train/use only this profile. Can be repeated.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command in {"train", "all"}:
        report = train_all(quick=args.quick, profiles=args.profile)
        log(f"Selected profile: {report['selected_profile']}")
    if args.command in {"forecast", "all"}:
        out = forecast_next24(profile_name=args.profile[0] if args.profile else None)
        log(f"Wrote rolling forecast with {len(out)} rows")
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
