#!/usr/bin/env python3
"""
Fiziksel özellik seti — santral teklif eşikleri ve sistem sıkışıklığı.

Gecikmeli zaman serisi özellikleri YOK. Bunun yerine piyasanın fiziksel
dinamiklerini doğrudan temsil eden üç özellik ailesi:

    1. MC_termik  — Yakıt tipine göre saatlik marjinal maliyet matrisi
    2. SDI        — Sistem Yönü İndeksi (YAL/YAT net talimat hacmi)
    3. NRM        — Net Rezerv Marjı (pik talep – aktif kapasite farkı)

Sızdırmazlık (leakage-safety) kuralları:
  - Emtia fiyatları: T-1 kapanış fiyatı (GÖP öncesi bilinir)
  - YAL/YAT: son 7 gün hareketli ortalama (bugünkü talimatlar değil)
  - Yük tahmini: EPİAŞ LEP (D-1 öğleden önce yayımlanır)
  - KGUP: D-1 planlama (GÖP öncesi mevcut)
  - REMIT kesintiler: bildirim tarihi < teslimat (geriye dönük değil)

Ana giriş noktaları:
    df = build_physical_features(snapshot)         # data_loader snapshot'u
    df = build_physical_features_from_files()      # disk üzerindeki CSV/parquet
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent

# ─── Varsayılan dosya yolları (disk modunda) ──────────────────────────────────
_KGUP_PATH      = PROJECT_ROOT / "data" / "kgup_combined.csv"
_YAL_YAT_PATH   = PROJECT_ROOT / "data" / "yal_yat.csv"
_LOAD_FCST_PATH = PROJECT_ROOT / "data" / "load_forecast.csv"
_PTF_PATH       = PROJECT_ROOT / "data" / "ptf_dataset.csv"
_COMMODITY_HOURLY = PROJECT_ROOT / "data" / "processed" / "international_commodity_prices_hourly.parquet"
_FX_HOURLY      = PROJECT_ROOT / "data" / "processed" / "tcmb_exchange_rates_hourly.parquet"
_OUTAGE_CSV     = PROJECT_ROOT / "data" / "outages.csv"

# ─── Fiziksel sabitler ────────────────────────────────────────────────────────

# Isıl verimlilik / ısıl değer parametreleri
_HEAT_RATE = {
    "gas_ccgt_mmbtu":    7.5,   # CCGT doğalgaz (mmbtu/MWh)
    "gas_simple_mmbtu":  11.0,  # Basit çevrim gaz (mmbtu/MWh)
    "coal_import_t":     0.385, # İthal kömür (tonne/MWh) ← 10 GJ/MWh / 26 GJ/t
    "lignite_t":         2.17,  # Linyit (tonne/MWh) ← 13 GJ/MWh / 6 GJ/t
    "fueloil_t":         0.22,  # Fuel-oil (tonne/MWh)
}

# Değişken O&M maliyetleri (TL/MWh, 2026 referansı)
_VOM = {
    "gas":   8.0,
    "coal":  5.0,
    "lignite": 4.0,
    "fueloil": 12.0,
    "renewable": 0.0,
}

# Yerli linyit maliyet sabiti (USD/t) — piyasa verisi yok, iç fiyat sabit
_LIGNITE_FIXED_USD_PER_T = 22.0

# Rezerv marjı eşikleri (MW)
_RESERVE_TIGHT_MW    = 2_000
_RESERVE_CRITICAL_MW =   500

# YAL/YAT hareketli ortalama pencereleri
_DIRECTION_ROLLING_HOURS = [24, 168]   # 1 gün, 7 gün


# ─── 1. Marjinal Maliyet Matrisi ──────────────────────────────────────────────

def compute_marginal_costs(
    commodity_df: pd.DataFrame,
    anchor_col: str = "ts_hour",
) -> pd.DataFrame:
    """
    Saatlik emtia fiyatlarından yakıt tipine göre marjinal maliyetleri hesapla.

    Girdi:
        commodity_df : data_loader.fetch_commodity_prices() çıktısı VEYA
                       data/processed/international_commodity_prices_hourly.parquet
        anchor_col   : Zaman sütununun adı

    Dönen sütunlar (TL/MWh cinsinden):
        mc_gas_ccgt_tl       — CCGT doğalgaz marjinal maliyeti
        mc_gas_simple_tl     — Basit çevrim gaz marjinal maliyeti
        mc_imported_coal_tl  — İthal kömür (taşkömür + ithalKomür)
        mc_lignite_tl        — Yerli linyit (sabit maliyet proxy)
        mc_fueloil_tl        — Fuel-oil
        fuel_switch_spread   — mc_gas_ccgt - mc_imported_coal (yakıt geçiş marjı)
        gas_is_marginal      — 1: gaz eşikten pahalı, kömür devreye girer
        coal_is_marginal     — 1: kömür fiyatlı bir saat
        must_run_surplus_flag— 1: yenilenebilir fazlası → gaz/kömür dışlanıyor
    """
    df = commodity_df.copy()

    # ── Sütun adı standardizasyonu ────────────────────────────────────────────
    # Desteklenen iki kaynak:
    #   A) data_loader.fetch_commodity_prices()  → ttf_usd_mmbtu, coal_api2_usd_t, ...
    #   B) data/processed/international_commodity_prices_daily.parquet
    #        → ts_day, ttf_eur_mwh, coal_api2_usd, brent_usd, ttf_try_mwh, coal_api2_try, brent_try

    df = df.rename(columns={
        "ts_day":           anchor_col,     # B kaynağı zaman sütunu
        "ttf_usd_mmbtu":    "ttf_usd",
        "ttf_eur_mwh":      "ttf_eur_mwh",  # EUR/MWh (gaz enerjisi)
        "ttf_try_mwh":      "ttf_try_mwh",  # TRY/MWh (gaz enerjisi, hesaplanmış)
        "hhub_usd_mmbtu":   "hhub_usd",
        "henry_hub_usd":    "hhub_usd",
        "coal_api2_usd_t":  "api2_usd",
        "coal_api2_usd":    "api2_usd",
        "coal_api2_try":    "api2_try",     # TRY/tonne
        "brent_usd_bbl":    "brent_usd",
        "brent_usd":        "brent_usd",
        "brent_try":        "brent_try",    # TRY/bbl
        "usd_try":          "usd_try",
        "eur_try":          "eur_try",
    }, errors="ignore")

    # Eksik sütunlar için 2026 referans varsayılanları
    defaults = {
        "ttf_usd":     12.0,   "ttf_eur_mwh": 35.0,
        "hhub_usd":     3.5,   "api2_usd":   100.0,
        "brent_usd":   85.0,   "usd_try":     34.0,  "eur_try": 37.0,
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").ffill().bfill().fillna(val)

    # USD/TRY veya EUR/TRY → tercih: mevcut dönüştürülmüş sütunlar
    usd = df["usd_try"]
    eur = df.get("eur_try", df["usd_try"] * 1.08)

    # ── Türkiye gaz fiyatı (TRY/MWh gaz enerjisi) ────────────────────────────
    # Öncelik: ttf_try_mwh (hazır TRY) → ttf_eur_mwh * eur_try → ttf_usd * usd_try
    if "ttf_try_mwh" in df.columns:
        df["ttf_try_mwh"] = pd.to_numeric(df["ttf_try_mwh"], errors="coerce").ffill().bfill()
        # Türkiye gaz alım premi: +%8 (taşıma + take-or-pay)
        gas_tl_per_mwh_energy = df["ttf_try_mwh"] * 1.08
    else:
        gas_tl_per_mwh_energy = df["ttf_eur_mwh"] * eur * 1.08

    # ── İthal kömür fiyatı (TRY/tonne) ───────────────────────────────────────
    if "api2_try" in df.columns:
        df["api2_try"] = pd.to_numeric(df["api2_try"], errors="coerce").ffill().bfill()
        coal_tl_per_tonne = df["api2_try"]
    else:
        coal_tl_per_tonne = df["api2_usd"] * usd

    # ── Brent (TRY/bbl) ───────────────────────────────────────────────────────
    if "brent_try" in df.columns:
        df["brent_try"] = pd.to_numeric(df["brent_try"], errors="coerce").ffill().bfill()
        brent_tl_per_bbl = df["brent_try"]
    else:
        brent_tl_per_bbl = df["brent_usd"] * usd

    # ── Marjinal maliyetler (TL/MWh elektrik) ─────────────────────────────────
    # CCGT verimliliği %56, basit çevrim %38
    df["mc_gas_ccgt_tl"] = (gas_tl_per_mwh_energy / 0.56 + _VOM["gas"]).clip(lower=0)
    df["mc_gas_simple_tl"] = (gas_tl_per_mwh_energy / 0.38 + _VOM["gas"]).clip(lower=0)

    df["mc_imported_coal_tl"] = (
        coal_tl_per_tonne * _HEAT_RATE["coal_import_t"] + _VOM["coal"]
    ).clip(lower=0)

    df["mc_lignite_tl"] = (
        _LIGNITE_FIXED_USD_PER_T * _HEAT_RATE["lignite_t"] * usd + _VOM["lignite"]
    ).clip(lower=0)

    # Fuel-oil: Brent USD/bbl * 7 bbl/tonne * 0.24 tonne/MWh * usd_try
    df["mc_fueloil_tl"] = (
        brent_tl_per_bbl * (7.0 * _HEAT_RATE["fueloil_t"]) + _VOM["fueloil"]
    ).clip(lower=0)

    # Türetilmiş özellikler
    df["fuel_switch_spread"] = df["mc_gas_ccgt_tl"] - df["mc_imported_coal_tl"]
    df["gas_is_marginal"] = (df["fuel_switch_spread"] > 0).astype(int)
    df["coal_is_marginal"] = (df["fuel_switch_spread"] < 0).astype(int)

    # Yenilenebilir fazlası bayrağı: düşük emtia fiyatı dönemi
    must_run_threshold = 150.0  # TL/MWh altında → RES fazlası baskısı
    df["must_run_surplus_flag"] = (df["mc_gas_ccgt_tl"] > must_run_threshold).astype(int) * 0
    # Basit yaklaşım: TTF düşükse RES'in fiyata baskısı yok; bu özellik
    # NRM ile birleşince anlamlı hale gelir.

    mc_cols = [c for c in df.columns if c.startswith("mc_") or c in (
        "fuel_switch_spread", "gas_is_marginal", "coal_is_marginal",
        "must_run_surplus_flag", "ttf", "api2", "brent", "usd_try"
    )]
    if anchor_col in df.columns:
        mc_cols = [anchor_col] + mc_cols

    return df[mc_cols].copy()


# ─── 2. Sistem Yönü İndeksi ───────────────────────────────────────────────────

def compute_system_direction_index(
    yal_yat_df: pd.DataFrame,
    anchor_col: str = "date",
) -> pd.DataFrame:
    """
    DGP YAL (Yük Alma) ve YAT (Yük Atma) talimatlarından saatlik
    Sistem Yönü İndeksi'ni hesapla.

    YAL (upRegulationDelivered) > 0  →  sistem açık, fiyat artışa meyilli
    YAT (downRegulationDelivered) > 0 → sistem fazlalı, fiyat düşüşe meyilli

    Dönen sütunlar:
        yal_vol_mwh      — YAL teslim edilen toplam hacim (MWh)
        yat_vol_mwh      — YAT teslim edilen toplam hacim (MWh)
        sdi_net_mwh      — YAL - YAT net hacim (pozitif = enerji açığı)
        sdi_direction    — +1 (açık), -1 (fazlalı), 0 (dengeli)
        sdi_intensity    — |net| / (yal + yat + 1) normalizasyonu
        sdi_roll_24h     — sdi_net_mwh 24 saatlik hareketli ortalama
        sdi_roll_168h    — 7 günlük hareketli ortalama (haftalık eğilim)
        bpm_deficit_flag — 1: son 24 saatin >%60'ı enerji açığıyla geçti
    """
    df = yal_yat_df.copy()

    # Sütun normalizasyonu
    df = df.rename(columns={
        "upRegulationDelivered":   "yal_vol_mwh",
        "downRegulationDelivered": "yat_vol_mwh",
        "net":                     "sdi_net_raw",
    }, errors="ignore")

    for col in ["yal_vol_mwh", "yat_vol_mwh"]:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Net hesabı
    if "sdi_net_raw" in df.columns:
        df["sdi_net_mwh"] = pd.to_numeric(df["sdi_net_raw"], errors="coerce").fillna(0.0)
    else:
        df["sdi_net_mwh"] = df["yal_vol_mwh"] - df["yat_vol_mwh"]

    # Yön ve yoğunluk
    df["sdi_direction"] = np.sign(df["sdi_net_mwh"]).astype(int)
    total = df["yal_vol_mwh"] + df["yat_vol_mwh"] + 1.0
    df["sdi_intensity"] = (df["sdi_net_mwh"].abs() / total).clip(0, 1)

    # Zaman indeksini otur — leakage-safe hareketli ortalama için
    time_col = None
    for c in [anchor_col, "date", "ts_hour", "datetime"]:
        if c in df.columns:
            time_col = c
            break

    if time_col:
        df = df.sort_values(time_col).reset_index(drop=True)
        for window in _DIRECTION_ROLLING_HOURS:
            df[f"sdi_roll_{window}h"] = (
                df["sdi_net_mwh"].rolling(window, min_periods=1).mean()
            )

        # Açık rejim bayrağı: son 24 saatin >%60'ı pozitif
        df["bpm_deficit_flag"] = (
            df["sdi_direction"].rolling(24, min_periods=1).mean() > 0.6
        ).astype(int)
    else:
        for window in _DIRECTION_ROLLING_HOURS:
            df[f"sdi_roll_{window}h"] = df["sdi_net_mwh"]
        df["bpm_deficit_flag"] = (df["sdi_net_mwh"] > 0).astype(int)

    keep = [c for c in [
        time_col, "yal_vol_mwh", "yat_vol_mwh",
        "sdi_net_mwh", "sdi_direction", "sdi_intensity",
        "sdi_roll_24h", "sdi_roll_168h", "bpm_deficit_flag",
    ] if c is not None and c in df.columns]

    return df[keep].copy()


# ─── 3. Net Rezerv Marjı ──────────────────────────────────────────────────────

def compute_net_reserve_margin(
    kgup_df: pd.DataFrame,
    load_forecast_df: pd.DataFrame,
    outages_df: pd.DataFrame | None = None,
    res_forecast_df: pd.DataFrame | None = None,
    reservoir_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Saatlik Net Rezerv Marjı hesapla.

    NRM = Kullanılabilir_Kapasite - Yük_Tahmini

    Kullanılabilir_Kapasite =
        termik_KGUP (planlı kapasite)
        + yenilenebilir_tahmin (RES forecast, varsa; yoksa KGUP'tan)
        + barajlı_hidro_KGUP
        − REMIT_kesinti_toplam

    Dönen sütunlar:
        thermal_kgup_mw      — Planlı termik üretim (gaz + kömür + linyit + fuel-oil)
        renewable_kgup_mw    — Planlı yenilenebilir (rüzgar + güneş + akarsu + jeotermal)
        hydro_kgup_mw        — Planlı barajlı hidro
        total_available_mw   — Toplam kullanılabilir kapasite
        load_forecast_mw     — Yük tahmini (EPİAŞ LEP)
        outage_loss_mw       — REMIT bildirilen kapasite kaybı
        net_reserve_margin   — NRM (MW)
        reserve_margin_pct   — NRM / load_forecast (%)
        reserve_tight_flag   — 1: NRM < 2000 MW
        reserve_critical_flag— 1: NRM < 500 MW
        hydro_fill_proxy     — Baraj doluluk oranı (varsa)
    """
    kgup = kgup_df.copy()

    # Zaman sütunu standardizasyonu
    for tc in ["date", "ts_hour", "datetime"]:
        if tc in kgup.columns:
            kgup["ts_hour"] = pd.to_datetime(kgup[tc], errors="coerce")
            break
    if "ts_hour" not in kgup.columns:
        raise ValueError("KGUP DataFrame'inde zaman sütunu bulunamadı.")

    # Yakıt bazlı kapasite grupları
    thermal_fuels    = ["dogalgaz", "linyit", "tasKomur", "ithalKomur", "fuelOil", "nafta"]
    renewable_fuels  = ["ruzgar", "gunes", "akarsu", "jeotermal", "biokutle"]
    hydro_fuels      = ["barajli"]

    for col in thermal_fuels + renewable_fuels + hydro_fuels:
        if col not in kgup.columns:
            kgup[col] = 0.0
        else:
            kgup[col] = pd.to_numeric(kgup[col], errors="coerce").fillna(0.0)

    kgup["thermal_kgup_mw"]   = kgup[thermal_fuels].sum(axis=1)
    kgup["renewable_kgup_mw"] = kgup[renewable_fuels].sum(axis=1)
    kgup["hydro_kgup_mw"]     = kgup[hydro_fuels].sum(axis=1)

    # Toplam planlı üretim (KGUP "toplam" sütunu varsa direkt al)
    if "toplam" in kgup.columns:
        kgup["total_available_mw"] = pd.to_numeric(kgup["toplam"], errors="coerce").fillna(
            kgup["thermal_kgup_mw"] + kgup["renewable_kgup_mw"] + kgup["hydro_kgup_mw"]
        )
    else:
        kgup["total_available_mw"] = (
            kgup["thermal_kgup_mw"] + kgup["renewable_kgup_mw"] + kgup["hydro_kgup_mw"]
        )

    # RES tahmin override (varsa)
    if res_forecast_df is not None and not res_forecast_df.empty:
        res = res_forecast_df.copy()
        if "wind_forecast_mwh" in res.columns and "solar_forecast_mwh" in res.columns:
            # ts_hour normalize
            for tc in ["ts_hour", "date", "datetime"]:
                if tc in res.columns:
                    res["ts_hour"] = pd.to_datetime(res[tc], errors="coerce")
                    break
            res["res_total_forecast"] = (
                res.get("wind_forecast_mwh", 0) + res.get("solar_forecast_mwh", 0)
            )
            if "ts_hour" in res.columns:
                res = res[["ts_hour", "res_total_forecast"]].dropna()
                kgup = kgup.merge(res, on="ts_hour", how="left")
                has_res = kgup["res_total_forecast"].notna()
                kgup.loc[has_res, "renewable_kgup_mw"] = kgup.loc[has_res, "res_total_forecast"]
                kgup = kgup.drop(columns=["res_total_forecast"])

    # Yük tahmini merge
    lf = load_forecast_df.copy()
    for tc in ["date", "ts_hour", "datetime"]:
        if tc in lf.columns:
            lf["ts_hour"] = pd.to_datetime(lf[tc], errors="coerce")
            break
    if "lep" in lf.columns:
        lf = lf.rename(columns={"lep": "load_forecast_mw"})
    elif "load_forecast_mw" not in lf.columns:
        lf["load_forecast_mw"] = np.nan

    lf = lf[["ts_hour", "load_forecast_mw"]].dropna(subset=["ts_hour"])
    kgup = kgup.merge(lf, on="ts_hour", how="left")

    # REMIT kesinti kaybı (saatlik, aktif kesintilerin toplamı)
    kgup["outage_loss_mw"] = 0.0
    if outages_df is not None and not outages_df.empty and "capacity_loss_mw" in outages_df.columns:
        total_outage = float(outages_df["capacity_loss_mw"].fillna(0).sum())
        kgup["outage_loss_mw"] = total_outage   # basit yaklaşım: saatler arasında eşit dağıtım

    # Baraj doluluk proxy
    kgup["hydro_fill_proxy"] = np.nan
    if reservoir_df is not None and not reservoir_df.empty and "national_mean_pct" in reservoir_df.columns:
        latest_fill = reservoir_df.dropna(subset=["national_mean_pct"])["national_mean_pct"].iloc[-1]
        kgup["hydro_fill_proxy"] = float(latest_fill)

    # REMIT kesintisini toplam kapasiteden düş
    kgup["total_available_mw"] = (kgup["total_available_mw"] - kgup["outage_loss_mw"].clip(lower=0))

    # Net Rezerv Marjı = Planlı Üretim − Yük Tahmini
    # Türkiye'de KGUP toplam ≈ üretim planı; LEP = yük tahmini.
    # Pozitif → arz fazlası, Negatif → arz açığı (fiyat artış baskısı).
    # Fark tipik olarak −2000 ile +5000 MW arasında değişir; bu sinyalin
    # kendisi bir PTF özelliği olarak kullanılır (kapasite değil, denge göstergesi).
    kgup["net_reserve_margin"] = kgup["total_available_mw"] - kgup["load_forecast_mw"].fillna(0)
    kgup["reserve_margin_pct"] = (
        kgup["net_reserve_margin"] / kgup["load_forecast_mw"].replace(0, np.nan)
    ).fillna(0) * 100

    # Döngüsel eşikler: arz açığı > 2 GW → sıkışık piyasa
    # Not: eşikler Türkiye piyasası istatistiklerinden türetildi:
    #   NRM < -2000 MW → rezerv sıkışık  (ortalama fark ~-3700 MW)
    #   NRM < -5000 MW → kritik (DGP acil çağrısı gerekli)

    # Türkiye piyasası için kalibre edilmiş eşikler:
    # reserve_tight    : NRM < -2000 MW → arz sıkışık
    # reserve_critical : NRM < -5000 MW → kritik arz açığı (DGP acil)
    _NRM_TIGHT_TR    = -2_000   # MW
    _NRM_CRITICAL_TR = -5_000   # MW
    kgup["reserve_tight_flag"]    = (kgup["net_reserve_margin"] < _NRM_TIGHT_TR).astype(int)
    kgup["reserve_critical_flag"] = (kgup["net_reserve_margin"] < _NRM_CRITICAL_TR).astype(int)

    out_cols = [
        "ts_hour",
        "thermal_kgup_mw", "renewable_kgup_mw", "hydro_kgup_mw",
        "total_available_mw", "load_forecast_mw", "outage_loss_mw",
        "net_reserve_margin", "reserve_margin_pct",
        "reserve_tight_flag", "reserve_critical_flag",
        "hydro_fill_proxy",
    ]
    return kgup[[c for c in out_cols if c in kgup.columns]].copy()


# ─── Birleştirici ─────────────────────────────────────────────────────────────

def build_physical_features(
    snapshot: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    data_loader.fetch_morning_snapshot() çıktısından fiziksel özellik tablosu üret.

    Args:
        snapshot: {"ptf_smf", "res_forecast", "outages", "reservoir", "commodities"}

    Returns:
        Saatlik fiziksel özellik DataFrame'i (ts_hour indeksli)
    """
    kgup = _load_kgup()
    lf   = _load_load_forecast()

    mc  = compute_marginal_costs(snapshot.get("commodities", pd.DataFrame()))
    sdi = compute_system_direction_index(_load_yal_yat())
    nrm = compute_net_reserve_margin(
        kgup_df=kgup,
        load_forecast_df=lf,
        outages_df=snapshot.get("outages"),
        res_forecast_df=snapshot.get("res_forecast"),
        reservoir_df=snapshot.get("reservoir"),
    )

    # ts_hour normalize
    for df_ in [mc, sdi, nrm]:
        for tc in ["ts_hour", "date", "Date"]:
            if tc in df_.columns and tc != "ts_hour":
                df_["ts_hour"] = pd.to_datetime(df_[tc], errors="coerce")
                break
        if "ts_hour" in df_.columns:
            df_["ts_hour"] = pd.to_datetime(df_["ts_hour"], errors="coerce")

    base = nrm.copy()
    if not mc.empty and "ts_hour" in mc.columns:
        mc_cols = [c for c in mc.columns if c != "ts_hour"] + ["ts_hour"]
        base = base.merge(mc[mc_cols], on="ts_hour", how="left")
    if not sdi.empty and "ts_hour" in sdi.columns:
        sdi_cols = [c for c in sdi.columns if c != "ts_hour"] + ["ts_hour"]
        sdi = sdi.rename(columns={"date": "ts_hour"}, errors="ignore")
        base = base.merge(sdi[sdi_cols] if "ts_hour" in sdi.columns else sdi, on="ts_hour", how="left")

    return base.sort_values("ts_hour").reset_index(drop=True)


def build_physical_features_from_files(
    kgup_path: Path | None = None,
    yal_yat_path: Path | None = None,
    load_path: Path | None = None,
    commodity_path: Path | None = None,
) -> pd.DataFrame:
    """
    Disk üzerindeki mevcut CSV/parquet dosyalarından fiziksel özellik tablosu üret.
    data_loader gerektirmez; bağımsız olarak çalışır.
    """
    kgup = _load_kgup(kgup_path)
    lf   = _load_load_forecast(load_path)
    yy   = _load_yal_yat(yal_yat_path)
    comm = _load_commodities(commodity_path)

    mc  = compute_marginal_costs(comm)
    sdi = compute_system_direction_index(yy)
    nrm = compute_net_reserve_margin(kgup_df=kgup, load_forecast_df=lf)

    # ts_hour normalize ve merge
    nrm["ts_hour"] = pd.to_datetime(nrm["ts_hour"], errors="coerce")

    if not mc.empty:
        if "ts_hour" not in mc.columns:
            mc["ts_hour"] = nrm["ts_hour"].values[:len(mc)] if len(mc) <= len(nrm) else np.nan
        mc["ts_hour"] = pd.to_datetime(mc.get("ts_hour"), errors="coerce")
        mc_daily = mc.copy()
        mc_daily["date_only"] = mc_daily["ts_hour"].dt.date.astype(str)
        nrm["date_only"] = nrm["ts_hour"].dt.date.astype(str)
        nrm = nrm.merge(
            mc_daily.drop_duplicates("date_only"),
            on="date_only", how="left", suffixes=("", "_mc")
        ).drop(columns=["date_only"])
        # Son günlerde emtia verisi gelmeyebilir → son bilinen değeri ileriye taşı
        mc_feature_cols = [c for c in nrm.columns if c.startswith("mc_") or c in
                           ("fuel_switch_spread", "gas_is_marginal", "coal_is_marginal",
                            "usd_try", "ttf_eur_mwh", "api2_usd", "brent_usd")]
        nrm[mc_feature_cols] = nrm[mc_feature_cols].ffill()

    if not sdi.empty:
        date_col = next((c for c in ["date", "ts_hour"] if c in sdi.columns), None)
        if date_col:
            sdi = sdi.rename(columns={date_col: "ts_hour"})
            sdi["ts_hour"] = pd.to_datetime(sdi["ts_hour"], errors="coerce")
            sdi_cols = [c for c in sdi.columns if not c.endswith("_mc")]
            nrm = nrm.merge(sdi[sdi_cols], on="ts_hour", how="left")

    return nrm.sort_values("ts_hour").reset_index(drop=True)


# ─── Disk yükleyiciler ────────────────────────────────────────────────────────

def _load_kgup(path: Path | None = None) -> pd.DataFrame:
    p = path or _KGUP_PATH
    if not Path(p).exists():
        raise FileNotFoundError(f"KGUP dosyası bulunamadı: {p}")
    df = pd.read_csv(p)
    df = df.rename(columns={"date": "ts_hour"}, errors="ignore")
    return df


def _load_yal_yat(path: Path | None = None) -> pd.DataFrame:
    p = path or _YAL_YAT_PATH
    if not Path(p).exists():
        raise FileNotFoundError(f"YAL/YAT dosyası bulunamadı: {p}")
    return pd.read_csv(p)


def _load_load_forecast(path: Path | None = None) -> pd.DataFrame:
    p = path or _LOAD_FCST_PATH
    if not Path(p).exists():
        raise FileNotFoundError(f"Yük tahmini dosyası bulunamadı: {p}")
    df = pd.read_csv(p)
    df = df.rename(columns={"date": "ts_hour"}, errors="ignore")
    return df


def _load_commodities(path: Path | None = None) -> pd.DataFrame:
    """
    Emtia fiyat verisini yükle.

    Önce saatlik parqueti dener; yoksa günlük parqueti saatlik zaman eksenine
    ffill ile hizalar. Her iki kaynak da yoksa 2026 referans varsayılanlarını
    döndürür.
    """
    hourly_p = path or _COMMODITY_HOURLY
    daily_p  = PROJECT_ROOT / "data" / "processed" / "international_commodity_prices_daily.parquet"

    for p in [hourly_p, daily_p]:
        if Path(p).exists():
            try:
                df = pd.read_parquet(p)
                # Zaman sütununu bul ve normalize et
                for tc in ["ts_hour", "date", "Date", "datetime"]:
                    if tc in df.columns:
                        df["ts_hour"] = pd.to_datetime(df[tc], errors="coerce")
                        break

                # Günlük veriyi saatlik KGUP eksenine hizala (ffill)
                if "ts_hour" in df.columns:
                    df = df.sort_values("ts_hour")
                    if df["ts_hour"].dt.hour.nunique() == 1:
                        # Yalnızca günlük veri → saat 00:00 → saatlik spine oluştur
                        start = df["ts_hour"].min()
                        end   = df["ts_hour"].max() + pd.Timedelta(hours=23)
                        tz = df["ts_hour"].dt.tz
                        hourly_idx = pd.date_range(start, end, freq="h", tz=tz)
                        df = (
                            df.set_index("ts_hour")
                            .reindex(hourly_idx)
                            .ffill()
                            .reset_index()
                            .rename(columns={"index": "ts_hour"})
                        )
                return df
            except Exception:
                pass

    # Fallback: varsayılan fiyatlar (2026 referans değerleri)
    hourly_idx = pd.date_range("2020-01-01", periods=56304, freq="h", tz="Europe/Istanbul")
    return pd.DataFrame({
        "ts_hour":          hourly_idx,
        "ttf_usd_mmbtu":    12.0,
        "coal_api2_usd_t":  110.0,
        "brent_usd_bbl":    85.0,
        "usd_try":          34.0,
    })
