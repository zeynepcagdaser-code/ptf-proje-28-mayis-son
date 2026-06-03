#!/usr/bin/env python3
"""
PTF Master Training Dataset Builder

Tüm veri kaynaklarını denetler, sorunları düzeltir ve tek bir temiz parquet
dosyası oluşturur: data/master/ptf_master_training_dataset.parquet

Tespit edilen sorunlar ve düzeltmeler:
  1. Load Forecast: 2026-05-12 24 eksik saat → haftanın aynı saatinden ffill
  2. IDM Prices: Temmuz 2020'de 20 eksik saat → ffill
  3. International Prices: 2020-01-01 NaN → bfill (piyasa tatil)
  4. TCMB Exchange Rates: 2020-01-01 NaN → bfill
  5. Wind Forecast: 27-44 saat quarter NaN → ffill
  6. Dam Fullness: 99.9% NaN → master dataset'e dahil edilmiyor
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CLEAN_DIR = DATA_DIR / "clean"
MASTER_DIR = DATA_DIR / "master"
MASTER_DIR.mkdir(parents=True, exist_ok=True)

TZ = "Europe/Istanbul"


def log(msg: str) -> None:
    print(f"  {msg}", flush=True)


def parse_ts(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce")
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert(TZ).dt.tz_localize(None)
    return ts.dt.floor("h")


def safe_div(num, den):
    n = pd.Series(num, dtype="float64")
    d = pd.Series(den, dtype="float64").replace(0, np.nan)
    return n / d


# ─────────────────────────────────────────────────────────
# KAYNAK YÜKLEYİCİLER (düzeltmeli)
# ─────────────────────────────────────────────────────────

def load_ptf() -> pd.DataFrame:
    print("\n[PTF]")
    df = pd.read_csv(DATA_DIR / "ptf_dataset.csv")
    df["ts_hour"] = parse_ts(df["date"])
    df["ptf"] = pd.to_numeric(df["price"], errors="coerce")
    out = df[["ts_hour", "ptf"]].dropna().drop_duplicates("ts_hour", keep="last")
    log(f"{len(out)} satır | {out['ts_hour'].min()} → {out['ts_hour'].max()}")
    log(f"PTF istatistik: min={out['ptf'].min():.1f}, max={out['ptf'].max():.1f}, "
        f"mean={out['ptf'].mean():.1f}, sıfır={( out['ptf']==0).sum()}")
    return out.sort_values("ts_hour").reset_index(drop=True)


def load_load_forecast() -> pd.DataFrame:
    print("\n[LOAD FORECAST]")
    df = pd.read_csv(DATA_DIR / "load_forecast.csv")
    df["ts_hour"] = parse_ts(df["date"])
    df["load_forecast"] = pd.to_numeric(df["lep"], errors="coerce")
    df = df[["ts_hour", "load_forecast"]].dropna().drop_duplicates("ts_hour", keep="last").sort_values("ts_hour")

    # Tam saatlik spine oluştur ve eksikleri tespit et
    spine = pd.DataFrame({"ts_hour": pd.date_range(df["ts_hour"].min(), df["ts_hour"].max(), freq="h")})
    merged = spine.merge(df, on="ts_hour", how="left")
    missing_mask = merged["load_forecast"].isna()
    n_missing = missing_mask.sum()
    if n_missing > 0:
        missing_hrs = merged.loc[missing_mask, "ts_hour"].tolist()
        log(f"UYARI: {n_missing} eksik saat tespit edildi!")
        log(f"  Eksik saatler: {missing_hrs[0]} → {missing_hrs[-1]}")
        # Haftalık döngüsellik: 168 saat öncesinden al (haftanın aynı günü, aynı saati)
        merged["load_forecast"] = (
            merged["load_forecast"]
            .fillna(merged["load_forecast"].shift(168))   # t-168h (7 gün önce)
            .fillna(merged["load_forecast"].shift(-168))  # t+168h (7 gün sonra)
            .fillna(merged["load_forecast"].ffill())      # fallback
        )
        n_still_missing = merged["load_forecast"].isna().sum()
        log(f"Düzeltme sonrası eksik: {n_still_missing}")
    else:
        log("Eksik saat yok ✓")
    log(f"{len(merged)} satır | {merged['ts_hour'].min()} → {merged['ts_hour'].max()}")
    return merged.reset_index(drop=True)


def load_kgup() -> pd.DataFrame:
    print("\n[KGÜP]")
    df = pd.read_csv(DATA_DIR / "kgup_combined.csv")
    df["ts_hour"] = parse_ts(df["date"])
    rename = {
        "toplam": "kgup_total", "dogalgaz": "kgup_gas", "ruzgar": "kgup_wind",
        "gunes": "kgup_solar", "barajli": "kgup_dammed_hydro", "akarsu": "kgup_river",
        "linyit": "kgup_lignite", "tasKomur": "kgup_black_coal",
        "ithalKomur": "kgup_import_coal", "fuelOil": "kgup_fuel_oil",
        "jeotermal": "kgup_geothermal", "biokutle": "kgup_biomass",
    }
    cols = ["ts_hour"] + [c for c in rename if c in df.columns]
    out = df[cols].rename(columns=rename).drop_duplicates("ts_hour", keep="last").sort_values("ts_hour")
    for c in out.columns:
        if c != "ts_hour":
            out[c] = pd.to_numeric(out[c], errors="coerce")

    renew = [c for c in ["kgup_wind","kgup_solar","kgup_dammed_hydro","kgup_river","kgup_geothermal","kgup_biomass"] if c in out.columns]
    thermal = [c for c in ["kgup_gas","kgup_lignite","kgup_black_coal","kgup_import_coal","kgup_fuel_oil"] if c in out.columns]
    out["kgup_renewable"] = out[renew].sum(axis=1)
    out["kgup_thermal"] = out[thermal].sum(axis=1)

    n_null = out.isnull().sum().sum()
    log(f"{len(out)} satır | null={n_null}")
    return out.reset_index(drop=True)


def load_wind_forecast() -> pd.DataFrame:
    print("\n[WIND FORECAST]")
    df = pd.read_csv(DATA_DIR / "wind_forecast.csv")
    df["ts_hour"] = parse_ts(df["date"])
    cols = ["ts_hour"] + [c for c in ["quarter1","quarter2","quarter3","quarter4","forecast","generation"] if c in df.columns]
    out = df[cols].drop_duplicates("ts_hour", keep="last").sort_values("ts_hour")
    for c in out.columns:
        if c != "ts_hour":
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Eksik quarter değerlerini ffill
    q_cols = [c for c in ["quarter1","quarter2","quarter3","quarter4"] if c in out.columns]
    before = out[q_cols].isnull().sum().sum()
    out[q_cols] = out[q_cols].ffill().bfill()
    after = out[q_cols].isnull().sum().sum()
    if before > 0:
        log(f"Wind quarter NaN düzeltildi: {before} → {after}")

    out["wind_forecast_mean"] = out[q_cols].mean(axis=1)
    out["wind_forecast_std"] = out[q_cols].std(axis=1)
    if "forecast" in out.columns:
        out["wind_forecast"] = out["forecast"].ffill().bfill()
    if "generation" in out.columns:
        out["wind_generation"] = pd.to_numeric(out["generation"], errors="coerce")
    log(f"{len(out)} satır | null kalan={out.isnull().sum().sum()}")
    return out.reset_index(drop=True)


def load_smf() -> pd.DataFrame:
    print("\n[SMF]")
    df = pd.read_csv(DATA_DIR / "smf.csv")
    df["ts_hour"] = parse_ts(df["date"])
    df["smf"] = pd.to_numeric(df["systemMarginalPrice"], errors="coerce")
    out = df[["ts_hour","smf"]].dropna(subset=["ts_hour"]).drop_duplicates("ts_hour", keep="last").sort_values("ts_hour")
    log(f"{len(out)} satır | {out['ts_hour'].max()} (son) — sadece lagged kullanılacak")
    return out.reset_index(drop=True)


def load_real_consumption() -> pd.DataFrame:
    print("\n[REAL CONSUMPTION]")
    df = pd.read_csv(DATA_DIR / "real_consumption.csv")
    df["ts_hour"] = parse_ts(df["date"])
    df["real_consumption"] = pd.to_numeric(df["consumption"], errors="coerce")
    out = df[["ts_hour","real_consumption"]].dropna(subset=["ts_hour"]).drop_duplicates("ts_hour", keep="last").sort_values("ts_hour")
    log(f"{len(out)} satır | {out['ts_hour'].max()} (son) — sadece lagged kullanılacak")
    return out.reset_index(drop=True)


def load_yal_yat() -> pd.DataFrame:
    print("\n[YAL/YAT]")
    df = pd.read_csv(DATA_DIR / "yal_yat.csv")
    df["ts_hour"] = parse_ts(df["date"])
    vc = [c for c in df.columns if c not in {"date","hour","ts_hour","source_type"}]
    out = df[["ts_hour"] + vc].drop_duplicates("ts_hour", keep="last").sort_values("ts_hour")
    for c in vc:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if "net" in out.columns:
        out = out.rename(columns={"net": "yal_yat_net"})
    log(f"{len(out)} satır | null={out.isnull().sum().sum()}")
    return out.reset_index(drop=True)


def load_generation() -> pd.DataFrame:
    print("\n[REALTIME GENERATION]")
    df = pd.read_csv(DATA_DIR / "realtime_generation.csv")
    df["ts_hour"] = parse_ts(df["date"])
    rename = {"total":"gen_total","naturalGas":"gen_gas","dammedHydro":"gen_dammed_hydro",
               "river":"gen_river","importCoal":"gen_import_coal","lignite":"gen_lignite",
               "wind":"gen_wind","sun":"gen_solar"}
    cols = ["ts_hour"] + [c for c in rename if c in df.columns]
    out = df[cols].rename(columns=rename).drop_duplicates("ts_hour", keep="last").sort_values("ts_hour")
    for c in out.columns:
        if c != "ts_hour":
            out[c] = pd.to_numeric(out[c], errors="coerce")
    log(f"{len(out)} satır | {out['ts_hour'].max()} (son) — sadece lagged kullanılacak")
    return out.reset_index(drop=True)


def load_grf() -> pd.DataFrame:
    print("\n[GRF]")
    path = PROCESSED_DIR / "grf_hourly_reference_price.parquet"
    df = pd.read_parquet(path)
    df["ts_hour"] = parse_ts(df["ts_hour"])
    df = df.drop_duplicates("ts_hour", keep="last").sort_values("ts_hour")
    for c in df.columns:
        if c != "ts_hour":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Lag NaN'ları beklenen (serinin başında)
    n_grf_null = df["grf_tl_1000sm3"].isna().sum()
    log(f"{len(df)} satır | grf_tl_1000sm3 NaN={n_grf_null} | corr ile PTF yüksek beklenir (0.78+)")
    log(f"NOT: delivery_grf look-ahead riski — GRF o günün sabahında yayımlanıyor.")
    log(f"     D+1 tahmini için delivery_grf bilinmiyor; lag_24 özelliği güvenli alternatif.")
    return df.reset_index(drop=True)


def load_idm() -> pd.DataFrame:
    print("\n[IDM PRICES]")
    path = PROCESSED_DIR / "idm_prices.parquet"
    df = pd.read_parquet(path)
    df["ts_hour"] = parse_ts(df["ts_hour"])
    df = df.drop_duplicates("ts_hour", keep="last").sort_values("ts_hour")

    # Tam spine ile eksik saatleri tespit et
    spine = pd.DataFrame({"ts_hour": pd.date_range(df["ts_hour"].min(), df["ts_hour"].max(), freq="h")})
    n_before = len(df)
    merged = spine.merge(df, on="ts_hour", how="left")
    missing_mask = merged["idm_price"].isna()
    n_missing = missing_mask.sum()

    if n_missing > 0:
        # Sadece yapısal boşlukları doldur (lag NaN'larını değil)
        # idm_price için ffill; lag sütunları sonradan yeniden hesaplanacak
        log(f"UYARI: {n_missing} eksik saat → ffill ile dolduruluyor")
        missing_ts = merged.loc[missing_mask & merged["idm_price"].isna(), "ts_hour"].tolist()
        if missing_ts:
            log(f"  İlk eksik: {missing_ts[0]}, son eksik: {missing_ts[-1]}")
        merged["idm_price"] = merged["idm_price"].ffill().bfill()

        # Lag sütunlarını yeniden hesapla
        p = merged["idm_price"]
        merged["idm_price_lag_24"] = p.shift(24)
        merged["idm_price_lag_48"] = p.shift(48)
        merged["idm_price_lag_168"] = p.shift(168)
        merged["idm_price_roll_mean_24"] = p.shift(1).rolling(24, min_periods=12).mean()
        merged["idm_price_roll_mean_168"] = p.shift(1).rolling(168, min_periods=48).mean()
        # dam_idm_spread lag'ları: PTF bilgisi yoksa NaN bırak (anchor feature olarak eklenir)
        for c in ["dam_idm_spread", "dam_idm_spread_lag_24", "dam_idm_spread_lag_168"]:
            if c not in merged.columns:
                merged[c] = np.nan
    else:
        log("Eksik saat yok ✓")

    log(f"{len(merged)} satır (önceki: {n_before})")
    return merged.reset_index(drop=True)


def load_tcmb() -> pd.DataFrame:
    print("\n[TCMB EXCHANGE RATES]")
    path = PROCESSED_DIR / "tcmb_exchange_rates_hourly.parquet"
    df = pd.read_parquet(path)
    df["ts_hour"] = parse_ts(df["ts_hour"])
    df = df.drop_duplicates("ts_hour", keep="last").sort_values("ts_hour")
    for c in df.columns:
        if c != "ts_hour":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 2020-01-01 NaN: piyasa tatil, bir sonraki işlem günü değeriyle doldur
    null_usd = df["usd_try_buy"].isna().sum()
    if null_usd > 0:
        log(f"UYARI: {null_usd} saatte usd_try_buy NaN (piyasa tatil) → bfill+ffill")
        price_cols = [c for c in df.columns if c in {"usd_try_buy","eur_try_buy","eur_usd_cross_buy"}]
        df[price_cols] = df[price_cols].bfill().ffill()
        # Lag sütunlarını yeniden hesapla
        for prefix in ["usd_try_buy","eur_try_buy"]:
            if prefix in df.columns:
                s = df[prefix]
                df[f"{prefix}_lag_1d"] = s.shift(24)
                df[f"{prefix}_change_7d"] = s - s.shift(168)
                df[f"{prefix}_pct_change_7d"] = df[f"{prefix}_change_7d"] / s.shift(168) * 100
                df[f"{prefix}_roll_mean_7d"] = s.shift(1).rolling(168, min_periods=24).mean()
        log(f"Düzeltme sonrası NaN: {df['usd_try_buy'].isna().sum()}")
    else:
        log("NaN yok ✓")
    log(f"{len(df)} satır")
    return df.reset_index(drop=True)


def load_international() -> pd.DataFrame:
    print("\n[INTERNATIONAL COMMODITY PRICES]")
    path = PROCESSED_DIR / "international_commodity_prices_hourly.parquet"
    df = pd.read_parquet(path)
    df["ts_hour"] = parse_ts(df["ts_hour"])
    df = df.drop_duplicates("ts_hour", keep="last").sort_values("ts_hour")
    for c in df.columns:
        if c != "ts_hour":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    base_cols = ["brent_usd","henry_hub_usd","ttf_eur_mwh","coal_api2_usd"]
    null_brent = df["brent_usd"].isna().sum()
    if null_brent > 0:
        log(f"UYARI: {null_brent} saatte brent_usd NaN → bfill+ffill (tatil günleri)")
        df[base_cols] = df[base_cols].bfill().ffill()
        # Lag/roll sütunlarını yeniden hesapla (sadece mevcut olanlar)
        for col in base_cols:
            s = df[col]
            df[f"{col}_lag_7d"] = s.shift(168)
            df[f"{col}_change_7d"] = s - s.shift(168)
            df[f"{col}_pct_change_7d"] = df[f"{col}_change_7d"] / s.shift(168) * 100
            df[f"{col}_roll_mean_30d"] = s.shift(1).rolling(720, min_periods=168).mean()
        log(f"Düzeltme sonrası NaN: {df['brent_usd'].isna().sum()}")
    else:
        log("NaN yok ✓")
    log(f"{len(df)} satır")
    return df.reset_index(drop=True)


def load_orderbook() -> pd.DataFrame:
    print("\n[DAM ORDERBOOK]")
    specs = {
        "dam_bid_volume.parquet": "dam_bid_volume",
        "dam_sell_offer_volume.parquet": "dam_sell_offer_volume",
        "dam_matched_volume.parquet": "dam_matched_volume",
        "dam_price_independent_buy.parquet": "dam_price_independent_buy",
        "dam_price_independent_sell.parquet": "dam_price_independent_sell",
        "dam_block_buy_volume.parquet": "dam_block_buy_volume",
    }
    frames = []
    for fname, col_prefix in specs.items():
        p = PROCESSED_DIR / fname
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        df["ts_hour"] = parse_ts(df["ts_hour"])
        val_cols = [c for c in df.columns if c != "ts_hour" and pd.api.types.is_numeric_dtype(df[c])]
        if not val_cols:
            continue
        vc = val_cols[0]
        frames.append(df[["ts_hour", vc]].rename(columns={vc: col_prefix})
                        .drop_duplicates("ts_hour", keep="last"))
    if not frames:
        return pd.DataFrame(columns=["ts_hour"])
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="ts_hour", how="outer")
    log(f"{len(out)} satır | {sorted(set(out.columns)-{'ts_hour'})}")
    return out.sort_values("ts_hour").reset_index(drop=True)


def load_outages() -> pd.DataFrame:
    print("\n[OUTAGES]")
    path = CLEAN_DIR / "outages_hourly.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["ts_hour"])
    df = pd.read_parquet(path)
    df["ts_hour"] = parse_ts(df["ts_hour"])
    # Null fault değerlerini sıfır olarak doldur (outage yok demek)
    fault_cols = ["outage_fault_mw_loss_sum","outage_fault_mw_loss_max","outage_fault_operator_power_sum"]
    maint_cols = ["outage_maint_capacity_sum","outage_maint_operator_power_sum"]
    for c in fault_cols + maint_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)
    log(f"{len(df)} satır | null kalan={df.isnull().sum().sum()}")
    return df.drop_duplicates("ts_hour", keep="last").sort_values("ts_hour").reset_index(drop=True)


# ─────────────────────────────────────────────────────────
# COMPOSITE FEATURES
# ─────────────────────────────────────────────────────────

def add_market_composites(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    total = pd.to_numeric(out.get("kgup_total"), errors="coerce")
    load = pd.to_numeric(out.get("load_forecast"), errors="coerce")
    wind = pd.to_numeric(out.get("kgup_wind"), errors="coerce").fillna(
        pd.to_numeric(out.get("wind_forecast_mean"), errors="coerce"))
    solar = pd.to_numeric(out.get("kgup_solar"), errors="coerce")
    hydro = (pd.to_numeric(out.get("kgup_dammed_hydro"), errors="coerce").fillna(0)
             + pd.to_numeric(out.get("kgup_river"), errors="coerce").fillna(0))
    gas = pd.to_numeric(out.get("kgup_gas"), errors="coerce")
    coal_cols = [c for c in ["kgup_lignite","kgup_black_coal","kgup_import_coal"] if c in out.columns]
    coal = out[coal_cols].sum(axis=1) if coal_cols else pd.Series(np.nan, index=out.index)
    renewable = pd.to_numeric(out.get("kgup_renewable"), errors="coerce").fillna(
        wind.fillna(0) + solar.fillna(0) + hydro.fillna(0))

    out["net_load_after_wind_solar"] = load - wind.fillna(0) - solar.fillna(0)
    out["net_load_after_renewable"] = load - renewable
    out["load_minus_kgup_total"] = load - total
    out["residual_load"] = load - renewable
    out["renewable_share"] = safe_div(renewable, total).to_numpy()
    out["gas_share"] = safe_div(gas, total).to_numpy()
    out["coal_share"] = safe_div(coal, total).to_numpy()
    out["hydro_share"] = safe_div(hydro, total).to_numpy()
    out["wind_load_share"] = safe_div(wind, load).to_numpy()
    out["solar_load_share"] = safe_div(solar, load).to_numpy()
    out["renewable_load_share"] = safe_div(renewable, load).to_numpy()
    out["thermal_share"] = safe_div(gas.fillna(0) + coal.fillna(0), total).to_numpy()
    out["gas_vs_renewable"] = out["gas_share"] - out["renewable_share"]
    out["cheap_supply_pressure"] = 100 * (
        0.35 * out["renewable_share"].fillna(0)
        + 0.20 * out["hydro_share"].fillna(0)
        + 0.20 * out["wind_load_share"].fillna(0)
        + 0.15 * out["solar_load_share"].fillna(0)
        + 0.10 * (1 - out["gas_share"].fillna(0).clip(0, 1))
    )
    out["thermal_tightness_pressure"] = 100 * (
        0.35 * out["gas_share"].fillna(0)
        + 0.25 * out["coal_share"].fillna(0)
        + 0.25 * (1 - out["renewable_share"].fillna(0).clip(0, 1))
        + 0.15 * safe_div(out["net_load_after_renewable"], load).fillna(0).clip(-1, 2)
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
    out["peak_hours_flag"] = out["hour"].between(17, 22).astype(int)

    # Load rolling stats
    load_r = load.shift(1).rolling(30 * 24, min_periods=24 * 7)
    out["load_roll_mean_30d"] = load_r.mean()
    out["load_roll_std_30d"] = load_r.std()
    out["load_vs_30d_ratio"] = safe_div(load, out["load_roll_mean_30d"]).to_numpy()
    out["load_log1p"] = np.log1p(load.clip(lower=0))

    if "grf_tl_1000sm3" in out.columns:
        grf = pd.to_numeric(out["grf_tl_1000sm3"], errors="coerce")
        peak_mask = out["hour"].between(17, 22).astype(int)
        out["grf_peak_effect"] = grf * peak_mask
        out["demand_per_grf"] = safe_div(load, grf).to_numpy()
        out["net_load_x_grf"] = out["net_load_after_renewable"] * grf
        out["gas_cost_pressure"] = out["gas_share"] * grf
        out["thermal_cost_pressure"] = out["thermal_share"] * grf
        out["gas_marginal_cost_pressure"] = out["gas_share"] * out["thermal_share"] * grf

    if "ttf_try_mwh" in out.columns:
        ttf_try = pd.to_numeric(out["ttf_try_mwh"], errors="coerce")
        gas_s = pd.to_numeric(out.get("gas_share", pd.Series(np.nan, index=out.index)), errors="coerce")
        out["ttf_x_gas_share"] = ttf_try * gas_s
        if "grf_tl_1000sm3" in out.columns:
            grf = pd.to_numeric(out["grf_tl_1000sm3"], errors="coerce")
            out["ttf_vs_grf_premium"] = ttf_try - grf * (1 / 10.55)

    if "brent_try" in out.columns and "ttf_try_mwh" in out.columns:
        brent_try = pd.to_numeric(out["brent_try"], errors="coerce")
        ttf_try = pd.to_numeric(out["ttf_try_mwh"], errors="coerce")
        out["brent_ttf_try_ratio"] = safe_div(brent_try, ttf_try * 8.14).to_numpy()

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

    if "outage_fault_mw_loss_sum" in out.columns:
        out["outage_stress_index"] = (
            pd.to_numeric(out["outage_fault_mw_loss_sum"], errors="coerce").fillna(0)
            / load.clip(lower=1)
        )

    return out


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ptf = out["ptf"]

    for lag in [1, 2, 3, 4, 6, 12, 24, 25, 26, 48, 72, 168]:
        out[f"ptf_lag_{lag}"] = ptf.shift(lag)

    past = ptf.shift(1)
    for win in [6, 12, 24, 48, 168]:
        roll = past.rolling(win, min_periods=max(3, min(win, 24)))
        out[f"ptf_roll_mean_{win}"] = roll.mean()
        out[f"ptf_roll_std_{win}"] = roll.std()
        out[f"ptf_roll_min_{win}"] = roll.min()
        out[f"ptf_roll_max_{win}"] = roll.max()

    out["ptf_low_ratio_24"] = (past <= 100).astype(float).rolling(24, min_periods=12).mean()
    out["ptf_zero_ratio_24"] = (past == 0).astype(float).rolling(24, min_periods=12).mean()
    out["ptf_spike_ratio_24"] = (past >= 3000).astype(float).rolling(24, min_periods=12).mean()
    out["ptf_d1_momentum"] = out["ptf_lag_24"] - out["ptf_lag_48"]
    out["ptf_week_momentum"] = out["ptf_lag_24"] - out["ptf_lag_168"]

    # Realized değişkenler: sadece lagged (look-ahead koruması)
    lagged_realized = [
        "smf", "real_consumption", "yal_yat_net", "gen_total", "gen_gas",
        "gen_wind", "gen_solar", "gen_dammed_hydro", "dam_matched_volume",
        "idm_price",
    ]
    for col in lagged_realized:
        if col not in out.columns:
            continue
        s = pd.to_numeric(out[col], errors="coerce")
        out[f"{col}_lag_24"] = s.shift(24)
        out[f"{col}_lag_168"] = s.shift(168)

    if "smf" in out.columns:
        spread = pd.to_numeric(out["smf"], errors="coerce") - ptf
        out["smf_ptf_spread_lag_24"] = spread.shift(24)
        out["smf_ptf_spread_lag_168"] = spread.shift(168)

    return out


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    import holidays
    out = df.copy()
    ts = out["ts_hour"]
    hour = ts.dt.hour
    dow = ts.dt.dayofweek
    month = ts.dt.month

    years = range(int(ts.dt.year.min()), int(ts.dt.year.max()) + 1)
    tr_holidays = holidays.Turkey(years=years)
    dates = ts.dt.date
    is_hol = dates.map(lambda d: 1 if d in tr_holidays else 0).astype(int)

    out["hour"] = hour.astype(int)
    out["dow"] = dow.astype(int)
    out["month"] = month.astype(int)
    out["year"] = ts.dt.year.astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)
    out["is_weekend"] = (dow >= 5).astype(int)
    out["is_holiday"] = is_hol
    out["is_holiday_or_weekend"] = ((dow >= 5) | is_hol.astype(bool)).astype(int)
    out["is_pre_holiday"] = dates.map(
        lambda d: 1 if (pd.Timestamp(d) + pd.Timedelta(days=1)).date() in tr_holidays else 0
    ).astype(int)
    out["is_post_holiday"] = dates.map(
        lambda d: 1 if (pd.Timestamp(d) - pd.Timedelta(days=1)).date() in tr_holidays else 0
    ).astype(int)
    out["ramadan_proxy"] = month.isin([3, 4]).astype(int)
    out["q1"] = month.isin([1,2,3]).astype(int)
    out["q2"] = month.isin([4,5,6]).astype(int)
    out["q3"] = month.isin([7,8,9]).astype(int)
    out["q4"] = month.isin([10,11,12]).astype(int)
    return out


def assign_split(ts: pd.Series) -> pd.Series:
    years = ts.dt.year
    split = pd.Series("ignore", index=ts.index, dtype="object")
    split[(years >= 2020) & (years <= 2024)] = "train"
    split[years == 2025] = "validation"
    split[years >= 2026] = "test"
    return split


# ─────────────────────────────────────────────────────────
# ANA FONKSİYON
# ─────────────────────────────────────────────────────────

def build_master_dataset() -> pd.DataFrame:
    print("=" * 65)
    print("PTF MASTER TRAINING DATASET BUILDER")
    print("=" * 65)

    ptf = load_ptf()
    sources = [
        load_load_forecast(),
        load_kgup(),
        load_wind_forecast(),
        load_smf(),
        load_real_consumption(),
        load_yal_yat(),
        load_generation(),
        load_grf(),
        load_idm(),
        load_tcmb(),
        load_international(),
        load_orderbook(),
        load_outages(),
    ]

    # Spine: PTF tarih aralığına sıkıştır
    # Outages gibi geleceğe uzanan kaynakların spine'ı genişletmesini önle
    spine_start = ptf["ts_hour"].min()
    spine_end = ptf["ts_hour"].max()
    spine = pd.DataFrame({"ts_hour": pd.date_range(spine_start, spine_end, freq="h")})
    print(f"\n[SPINE] {len(spine)} saat | {spine_start} → {spine_end}")
    print(f"  (Outages 2026-12-31'e kadar var ama spine PTF aralığıyla sınırlandırıldı)")

    print("\n[MERGE] Tüm kaynaklar birleştiriliyor...")
    data = spine.merge(ptf, on="ts_hour", how="left")
    for s in sources:
        if not s.empty and "ts_hour" in s.columns:
            data = data.merge(s, on="ts_hour", how="left")
    data = data.sort_values("ts_hour").reset_index(drop=True)

    print("\n[CALENDAR] Takvim özellikleri ekleniyor...")
    data = add_calendar_features(data)

    print("\n[COMPOSITES] Piyasa kompozit özellikleri ekleniyor...")
    data = add_market_composites(data)

    print("\n[HISTORY] Gecikme ve rolling features ekleniyor...")
    data = add_history_features(data)

    print("\n[SPLIT] Eğitim/doğrulama/test bölünüyor...")
    data["split"] = assign_split(data["ts_hour"])
    split_counts = data["split"].value_counts()
    for s, c in split_counts.items():
        print(f"  {s}: {c} saat ({c/24:.0f} gün)")

    # ─── NULL RAPORU ───
    print("\n[NULL RAPORU]")
    total_rows = len(data)
    nulls = data.isnull().sum()
    nulls = nulls[nulls > 0].sort_values(ascending=False)
    expected_null_prefixes = (
        "ptf_lag_", "ptf_roll_", "_lag_", "_lag_24", "_lag_48", "_lag_168",
        "_roll_", "grf_tl_lag", "grf_tl_change", "grf_tl_roll",
        "brent_usd_lag", "ttf_eur_mwh_lag", "idm_price_lag",
        "dam_idm_spread_lag", "tcmb", "usd_try_buy_lag",
    )
    lag_nulls = nulls[[c for c in nulls.index if any(c.startswith(p) or p in c for p in expected_null_prefixes)]]
    struct_nulls = nulls[[c for c in nulls.index if c not in lag_nulls.index]]

    if not struct_nulls.empty:
        print("  ⚠ Yapısal NaN'lar (lag dışı):")
        for col, cnt in struct_nulls.head(20).items():
            pct = cnt / total_rows * 100
            print(f"    {col}: {cnt} ({pct:.1f}%)")
    else:
        print("  Yapısal NaN yok ✓")
    print(f"  Lag/rolling NaN (beklenen): {lag_nulls.sum()} toplam")

    # ─── KORELASYON KONTROLÜ ───
    print("\n[KORELASYON] PTF ile ana özelliklerin korelasyonu:")
    key_features = [
        "load_forecast", "kgup_total", "renewable_share", "gas_share",
        "grf_tl_1000sm3", "usd_try_buy", "brent_usd", "ttf_eur_mwh",
        "ptf_lag_24", "cheap_supply_pressure", "thermal_tightness_pressure",
    ]
    ptf_corrs = []
    for feat in key_features:
        if feat in data.columns:
            corr = data["ptf"].corr(data[feat])
            ptf_corrs.append((feat, corr))
    for feat, corr in sorted(ptf_corrs, key=lambda x: abs(x[1]), reverse=True):
        bar = "█" * int(abs(corr) * 20)
        sign = "+" if corr > 0 else "-"
        print(f"  {feat:<35} {sign}{abs(corr):.3f} {bar}")

    # ─── KAYDET ───
    out_path = MASTER_DIR / "ptf_master_training_dataset.parquet"
    print(f"\n[KAYDET] {out_path}")
    data.to_parquet(out_path, index=False)
    size_mb = out_path.stat().st_size / 1e6
    print(f"  Satır: {len(data):,} | Sütun: {len(data.columns)} | Boyut: {size_mb:.1f} MB")
    print(f"  Zaman aralığı: {data['ts_hour'].min()} → {data['ts_hour'].max()}")

    # ─── ÖZET RAPOR ───
    print("\n" + "=" * 65)
    print("ÖZET RAPOR")
    print("=" * 65)
    print(f"Toplam saat: {len(data):,}")
    print(f"Toplam özellik: {len(data.columns)} (ts_hour + ptf + split dahil)")
    print(f"PTF aralığı: {data['ptf'].min():.1f} → {data['ptf'].max():.1f} TL/MWh")
    print(f"PTF ortalaması: {data['ptf'].mean():.1f} TL/MWh")
    print(f"\nVeri kaynakları:")
    print(f"  ✓ PTF (hedef değişken)")
    print(f"  ✓ Yük tahmini (load_forecast) — 2026-05-12 boşluğu düzeltildi")
    print(f"  ✓ KGÜP (üretim planı) — toplam, kaynak bazlı")
    print(f"  ✓ Rüzgar tahmini (wind_forecast) — quarter NaN'lar düzeltildi")
    print(f"  ✓ SMF — sadece lagged (look-ahead koruması)")
    print(f"  ✓ Gerçek tüketim — sadece lagged")
    print(f"  ✓ YAL/YAT — sadece lagged")
    print(f"  ✓ Gerçek üretim — sadece lagged")
    print(f"  ✓ GRF (doğalgaz referans fiyatı)")
    print(f"  ✓ IDM fiyatları — {20} eksik saat ffill ile düzeltildi")
    print(f"  ✓ TCMB döviz kurları — 2020-01-01 bfill ile düzeltildi")
    print(f"  ✓ Uluslararası emtia fiyatları — 2020-01-01 bfill ile düzeltildi")
    print(f"  ✓ DAM order book verileri")
    print(f"  ✓ Arıza/bakım (outages)")
    print(f"  ✗ Baraj doluluk — %99.9 NaN, hariç tutuldu (KGÜP hydro_share proxy kullanılıyor)")
    print(f"\nLOOK-AHEAD UYARILARI:")
    print(f"  ⚠ delivery_grf_tl_1000sm3: GRF D+1 için D günü öğleden önce bilinmiyor")
    print(f"    → Güvenli alternatif: grf_tl_lag_24 (grf_tl_1000sm3_lag_24)")
    print(f"  ⚠ delivery_dam_bid_volume vb.: DAM sonrası yayımlanıyor")
    print(f"    → Lagged versiyonlar (lag_24, lag_168) güvenli")
    print(f"\nÇıktı: {out_path}")
    return data


if __name__ == "__main__":
    df = build_master_dataset()
    sys.exit(0)
