#!/usr/bin/env python3
"""Baraj doluluk oranı — zenginleştirilmiş dataset.

Veri kaynakları:
  1. İBB CKAN API  — İstanbul 10 baraj, günlük %, 2011-2021
  2. EPİAŞ active-fullness  — 83 HES barajı, bugünün %-i (birikimli ileri)

Çıktılar:
  data/processed/dam_capacity_table.csv            — statik kapasite referansı
  data/external/ibb_istanbul/ibb_dam_daily.parquet — İstanbul tarihi veri
  data/processed/dam_fullness_enriched_daily.parquet
  data/processed/dam_fullness_enriched_hourly.parquet

Kullanım:
  python3 fetch_dam_fullness_enriched.py            # hem EPİAŞ hem İBB
  python3 fetch_dam_fullness_enriched.py --ibb-only  # sadece İstanbul tarihi
  python3 fetch_dam_fullness_enriched.py --epias-only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

USERNAME = os.getenv("EPIAS_USERNAME")
PASSWORD = os.getenv("EPIAS_PASSWORD")

EPIAS_LOGIN_URL = "https://giris.epias.com.tr/cas/v1/tickets"
EPIAS_FULLNESS_URL = (
    "https://seffaflik.epias.com.tr/electricity-service/v1/dams/data/active-fullness"
)

IBB_RESOURCE_INDIVIDUAL = "af0b3902-cfd9-4096-85f7-e2c3017e4f21"  # 10 dam, 2011-2021
IBB_RESOURCE_GENERAL = "b68cbdb0-9bf5-474c-91c4-9256c07c4bdf"     # genel, 2005-2018
IBB_API = "https://data.ibb.gov.tr/api/3/action/datastore_search"

OUT_CAPACITY_CSV = PROJECT_ROOT / "data" / "processed" / "dam_capacity_table.csv"
OUT_IBB_DIR = PROJECT_ROOT / "data" / "external" / "ibb_istanbul"
OUT_IBB_PARQUET = OUT_IBB_DIR / "ibb_dam_daily.parquet"
OUT_DAILY = PROJECT_ROOT / "data" / "processed" / "dam_fullness_enriched_daily.parquet"
OUT_HOURLY = PROJECT_ROOT / "data" / "processed" / "dam_fullness_enriched_hourly.parquet"


# ---------------------------------------------------------------------------
# Statik kapasite tablosu  (havza, il, max_hm3, aktif_hm3, MW)
# Kaynaklar: enerjiatlasi.com, Wikipedia, DSİ teknik bültenleri
# ---------------------------------------------------------------------------
DAM_CAPACITY: dict[str, dict] = {
    # ── Kızılırmak ──────────────────────────────────────────────────────────
    "ALTINKAYA":         dict(basin="Kızılırmak", il="Samsun",         cap_max=5763.0, cap_active=2892.0, mw=703.0),
    "BOYABAT":           dict(basin="Kızılırmak", il="Sinop",          cap_max=3557.0, cap_active=1402.0, mw=513.0),
    "HIRFANLI":          dict(basin="Kızılırmak", il="Kırşehir",       cap_max=5740.0, cap_active=2035.0, mw=128.0),
    "KAPULUKAYA":        dict(basin="Kızılırmak", il="Kırıkkale",      cap_max=200.0,  cap_active=140.8,  mw=54.0),
    "KESİKKÖPRÜ":        dict(basin="Kızılırmak", il="Kırıkkale",      cap_max=1355.0, cap_active=525.0,  mw=72.0),
    "YAMULA":            dict(basin="Kızılırmak", il="Kayseri",        cap_max=2200.0, cap_active=580.0,  mw=60.0),
    "DERBENT":           dict(basin="Kızılırmak", il="Kırşehir",       cap_max=100.0,  cap_active=50.0,   mw=17.0),
    "GÜLDÜRCEK":         dict(basin="Kızılırmak", il="Çorum",          cap_max=60.0,   cap_active=30.0,   mw=5.0),
    "BAYRAMHACILI HES":  dict(basin="Kızılırmak", il="Kırıkkale",      cap_max=50.0,   cap_active=25.0,   mw=8.0),
    "OBRUK":             dict(basin="Kızılırmak", il="Konya",          cap_max=100.0,  cap_active=50.0,   mw=15.0),
    "ÇERMİKLER":         dict(basin="Kızılırmak", il="Kırıkkale",      cap_max=30.0,   cap_active=15.0,   mw=3.5),
    "KARGI KIZILIRMAK":  dict(basin="Kızılırmak", il="Çankırı",        cap_max=400.0,  cap_active=200.0,  mw=110.0),
    # ── Antalya ─────────────────────────────────────────────────────────────
    "OYMAPINAR":         dict(basin="Antalya",     il="Antalya",        cap_max=296.7,  cap_active=76.5,   mw=540.0),
    "KARACAÖREN I":      dict(basin="Antalya",     il="Burdur",         cap_max=435.0,  cap_active=400.0,  mw=27.0),
    "KARACAÖREN II":     dict(basin="Antalya",     il="Antalya",        cap_max=100.0,  cap_active=80.0,   mw=8.5),
    "MANAVGAT":          dict(basin="Antalya",     il="Antalya",        cap_max=210.0,  cap_active=175.0,  mw=30.5),
    "SORGUN":            dict(basin="Antalya",     il="Isparta",        cap_max=90.0,   cap_active=70.0,   mw=10.0),
    # ── Sakarya ─────────────────────────────────────────────────────────────
    "SARIYAR":           dict(basin="Sakarya",     il="Ankara",         cap_max=1500.0, cap_active=942.0,  mw=160.0),
    "GÖKÇEKAYA":         dict(basin="Sakarya",     il="Eskişehir",      cap_max=900.0,  cap_active=300.0,  mw=278.0),
    "YENİCE":            dict(basin="Sakarya",     il="Sakarya",        cap_max=600.0,  cap_active=400.0,  mw=80.0),
    "KARGI BARAJI VE HES": dict(basin="Sakarya",   il="Kastamonu",      cap_max=500.0,  cap_active=300.0,  mw=80.0),
    "BOĞAZKÖY":          dict(basin="Sakarya",     il="Bolu",           cap_max=50.0,   cap_active=25.0,   mw=7.0),
    "GÜRSÖĞÜT-1":        dict(basin="Sakarya",     il="Bilecik",        cap_max=30.0,   cap_active=15.0,   mw=17.0),
    "GÜRSÖĞÜT-2":        dict(basin="Sakarya",     il="Bilecik",        cap_max=30.0,   cap_active=15.0,   mw=24.0),
    # ── Ceyhan ──────────────────────────────────────────────────────────────
    "SIR":               dict(basin="Ceyhan",      il="Kahramanmaraş",  cap_max=1200.0, cap_active=747.9,  mw=283.5),
    "MENZELET":          dict(basin="Ceyhan",      il="Kahramanmaraş",  cap_max=2760.0, cap_active=1440.0, mw=124.0),
    "BERKE ":            dict(basin="Ceyhan",      il="Osmaniye",       cap_max=427.0,  cap_active=302.0,  mw=510.0),
    "KARTALKAYA":        dict(basin="Ceyhan",      il="Adana",          cap_max=300.0,  cap_active=100.0,  mw=35.0),
    "ASLANTAŞ":          dict(basin="Ceyhan",      il="Adana",          cap_max=1344.0, cap_active=900.0,  mw=95.0),
    "KOZAN":             dict(basin="Ceyhan",      il="Adana",          cap_max=200.0,  cap_active=120.0,  mw=18.0),
    "KILAVUZLU":         dict(basin="Ceyhan",      il="Adana",          cap_max=100.0,  cap_active=60.0,   mw=77.0),
    "KANDİL HES":        dict(basin="Ceyhan",      il="Kahramanmaraş",  cap_max=50.0,   cap_active=25.0,   mw=12.0),
    "SARIGÜZEL HES":     dict(basin="Ceyhan",      il="Kahramanmaraş",  cap_max=200.0,  cap_active=100.0,  mw=45.0),
    "ADATEPE":           dict(basin="Ceyhan",      il="Adana",          cap_max=100.0,  cap_active=50.0,   mw=12.0),
    # ── Yeşilırmak ──────────────────────────────────────────────────────────
    "H.UĞURLU":          dict(basin="Yeşilırmak",  il="Samsun",         cap_max=4500.0, cap_active=3200.0, mw=180.0),
    "S.UĞURLU":          dict(basin="Yeşilırmak",  il="Samsun",         cap_max=2000.0, cap_active=1500.0, mw=200.0),
    "ALMUS":             dict(basin="Yeşilırmak",  il="Tokat",          cap_max=620.0,  cap_active=380.0,  mw=27.0),
    "KILIÇKAYA":         dict(basin="Yeşilırmak",  il="Samsun",         cap_max=330.0,  cap_active=220.0,  mw=32.0),
    "GÖLOVA":            dict(basin="Yeşilırmak",  il="Sivas",          cap_max=100.0,  cap_active=60.0,   mw=11.5),
    "ÇAMLIGÖZE":         dict(basin="Yeşilırmak",  il="Sivas",          cap_max=50.0,   cap_active=25.0,   mw=15.0),
    "TEPEKIŞLA":         dict(basin="Yeşilırmak",  il="Tokat",          cap_max=60.0,   cap_active=30.0,   mw=14.0),
    "ÇEKEREK":           dict(basin="Yeşilırmak",  il="Yozgat",         cap_max=80.0,   cap_active=40.0,   mw=20.0),
    # ── Seyhan ──────────────────────────────────────────────────────────────
    "ÇATALAN":           dict(basin="Seyhan",      il="Adana",          cap_max=2086.0, cap_active=1200.0, mw=124.0),
    "YEDİGÖZE":          dict(basin="Seyhan",      il="Adana",          cap_max=200.0,  cap_active=100.0,  mw=33.0),
    "BAHÇELİK":          dict(basin="Seyhan",      il="Adana",          cap_max=50.0,   cap_active=25.0,   mw=6.5),
    "FEKE-2":            dict(basin="Seyhan",      il="Adana",          cap_max=50.0,   cap_active=25.0,   mw=8.0),
    "KÖPRÜ":             dict(basin="Seyhan",      il="Adana",          cap_max=30.0,   cap_active=15.0,   mw=5.0),
    "MENGE HES":         dict(basin="Seyhan",      il="Adana",          cap_max=30.0,   cap_active=15.0,   mw=7.0),
    "GÜMÜŞÖREN":         dict(basin="Seyhan",      il="Adana",          cap_max=30.0,   cap_active=15.0,   mw=8.0),
    "GÖKTAŞ-1":          dict(basin="Seyhan",      il="Adana",          cap_max=30.0,   cap_active=15.0,   mw=7.5),
    "KAVŞAKBENDİ HES":   dict(basin="Seyhan",      il="Adana",          cap_max=100.0,  cap_active=50.0,   mw=25.0),
    # ── Doğu Akdeniz ────────────────────────────────────────────────────────
    "ERMENEK":           dict(basin="Doğu Akdeniz", il="Karaman",        cap_max=4500.0, cap_active=1747.0, mw=302.4),
    "GEZENDE":           dict(basin="Doğu Akdeniz", il="Mersin",         cap_max=585.0,  cap_active=400.0,  mw=159.0),
    "ALAKÖPRÜ":          dict(basin="Doğu Akdeniz", il="Adana",          cap_max=100.0,  cap_active=50.0,   mw=60.0),
    "BERDAN":            dict(basin="Doğu Akdeniz", il="Mersin",         cap_max=150.0,  cap_active=75.0,   mw=22.0),
    "PAMUKLUK":          dict(basin="Doğu Akdeniz", il="Mersin",         cap_max=50.0,   cap_active=25.0,   mw=7.5),
    "YALNIZARDIÇ":       dict(basin="Doğu Akdeniz", il="Mersin",         cap_max=100.0,  cap_active=50.0,   mw=15.0),
    # ── Doğu Karadeniz ──────────────────────────────────────────────────────
    "TORUL":             dict(basin="Doğu Karadeniz", il="Gümüşhane",   cap_max=375.0,  cap_active=250.0,  mw=103.0),
    "KÜRTÜN":            dict(basin="Doğu Karadeniz", il="Gümüşhane",   cap_max=100.0,  cap_active=50.0,   mw=30.0),
    "ALADEREÇAM":        dict(basin="Doğu Karadeniz", il="Rize",         cap_max=30.0,   cap_active=15.0,   mw=5.0),
    "TOPÇAM-ORDU":       dict(basin="Doğu Karadeniz", il="Ordu",         cap_max=60.0,   cap_active=30.0,   mw=17.0),
    "YAŞMAKLI":          dict(basin="Doğu Karadeniz", il="Trabzon",      cap_max=30.0,   cap_active=15.0,   mw=10.0),
    # ── Van Gölü ────────────────────────────────────────────────────────────
    "MORGEDİK":          dict(basin="Van Gölü",    il="Van",            cap_max=80.0,   cap_active=40.0,   mw=17.0),
    "SARIMEHMET":        dict(basin="Van Gölü",    il="Van",            cap_max=100.0,  cap_active=50.0,   mw=24.0),
    "ZERNEK":            dict(basin="Van Gölü",    il="Van",            cap_max=150.0,  cap_active=80.0,   mw=36.0),
    # ── Büyük Menderes ──────────────────────────────────────────────────────
    "ADIGÜZEL":          dict(basin="Büyük Menderes", il="Denizli",      cap_max=1480.0, cap_active=620.0,  mw=176.0),
    "ADIGÜZEL-2 HES":    dict(basin="Büyük Menderes", il="Denizli",      cap_max=50.0,   cap_active=25.0,   mw=14.0),
    "CİNDERE":           dict(basin="Büyük Menderes", il="Denizli",      cap_max=285.0,  cap_active=150.0,  mw=54.0),
    "KEMER":             dict(basin="Büyük Menderes", il="Burdur",       cap_max=700.0,  cap_active=350.0,  mw=42.0),
    "ÇAPALI DİNAR KARAKUYU": dict(basin="Büyük Menderes", il="Afyon",   cap_max=30.0,   cap_active=15.0,   mw=4.0),
    "ÇİNE ADNAN MENDERES": dict(basin="Büyük Menderes", il="Aydın",     cap_max=200.0,  cap_active=100.0,  mw=44.65),
    # ── Gediz ───────────────────────────────────────────────────────────────
    "DEMİRKÖPRÜ":        dict(basin="Gediz",       il="Manisa",         cap_max=1325.0, cap_active=800.0,  mw=69.0),
    # ── Susurluk ────────────────────────────────────────────────────────────
    "MANYAS":            dict(basin="Susurluk",    il="Balıkesir",      cap_max=80.0,   cap_active=40.0,   mw=10.0),
    "ÇINARCIK":          dict(basin="Susurluk",    il="Bursa",          cap_max=50.0,   cap_active=25.0,   mw=8.0),
    # ── Marmara ─────────────────────────────────────────────────────────────
    "YENİCE-GÖNEN":      dict(basin="Marmara",     il="Balıkesir",      cap_max=200.0,  cap_active=100.0,  mw=18.0),
    # ── Kuzey Ege ───────────────────────────────────────────────────────────
    "BAYRAMİÇ":          dict(basin="Kuzey Ege",   il="Çanakkale",      cap_max=250.0,  cap_active=150.0,  mw=29.0),
    # ── Batı Akdeniz ────────────────────────────────────────────────────────
    "AKKÖPRÜ":           dict(basin="Batı Akdeniz", il="Antalya",        cap_max=100.0,  cap_active=50.0,   mw=22.0),
    "ALAKIR":            dict(basin="Batı Akdeniz", il="Antalya",        cap_max=30.0,   cap_active=15.0,   mw=12.0),
    "EŞEN 1":            dict(basin="Batı Akdeniz", il="Muğla",          cap_max=50.0,   cap_active=25.0,   mw=8.0),
    # ── Batı Karadeniz ──────────────────────────────────────────────────────
    "ARAÇ BARAJI":       dict(basin="Batı Karadeniz", il="Kastamonu",    cap_max=100.0,  cap_active=50.0,   mw=17.0),
    "KÖPRÜBAŞI":         dict(basin="Batı Karadeniz", il="Bolu",         cap_max=50.0,   cap_active=25.0,   mw=10.0),
    "KİRAZLIKÖPRÜ":      dict(basin="Batı Karadeniz", il="Bartın",       cap_max=60.0,   cap_active=30.0,   mw=15.0),
    # ── Asi ─────────────────────────────────────────────────────────────────
    "BÜYÜK KARAÇAY":     dict(basin="Asi",         il="Hatay",          cap_max=100.0,  cap_active=50.0,   mw=10.0),
}


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def save_capacity_table() -> pd.DataFrame:
    rows = []
    for dam, info in DAM_CAPACITY.items():
        rows.append({
            "dam_name": dam.strip(),
            "basin": info["basin"],
            "il": info["il"],
            "cap_max_hm3": info["cap_max"],
            "cap_active_hm3": info["cap_active"],
            "installed_mw": info["mw"],
        })
    df = pd.DataFrame(rows).sort_values(["basin", "dam_name"]).reset_index(drop=True)
    OUT_CAPACITY_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CAPACITY_CSV, index=False, encoding="utf-8-sig")
    _log(f"Kapasite tablosu → {OUT_CAPACITY_CSV}  ({len(df)} baraj)")
    return df


# ---------------------------------------------------------------------------
# İBB CKAN API — İstanbul 10 baraj, günlük %, 2011-2021
# ---------------------------------------------------------------------------
def fetch_ibb_individual_dams() -> pd.DataFrame:
    """İBB CKAN'dan 10 İstanbul barajı, tüm kayıtları çeker."""
    _log("İBB CKAN API: bireysel barajlar çekiliyor…")
    records: list[dict] = []
    offset = 0
    limit = 1000
    total = None
    while True:
        url = (
            f"{IBB_API}?resource_id={IBB_RESOURCE_INDIVIDUAL}"
            f"&limit={limit}&offset={offset}"
        )
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
        except Exception as exc:
            _log(f"  İBB API hata offset={offset}: {exc}")
            break
        data = resp.json()
        if not data.get("success"):
            _log(f"  İBB API başarısız: {data.get('error')}")
            break
        result = data["result"]
        if total is None:
            total = result.get("total", 0)
            _log(f"  Toplam kayıt: {total}")
        batch = result.get("records", [])
        if not batch:
            break
        records.extend(batch)
        offset += len(batch)
        if offset >= total:
            break
        time.sleep(0.3)
    if not records:
        _log("  İBB: kayıt gelmedi.")
        return pd.DataFrame()
    df = pd.DataFrame(records)
    _log(f"  İBB bireysel: {len(df)} kayıt, kolonlar={list(df.columns)}")
    return df


def fetch_ibb_general() -> pd.DataFrame:
    """İBB genel İstanbul doluluk verisi (2005-2018)."""
    url = (
        "https://data.ibb.gov.tr/dataset/19c14482-14f2-4803-b4df-4cf4c6c42016"
        "/resource/b68cbdb0-9bf5-474c-91c4-9256c07c4bdf/download/dam_occupancy.csv"
    )
    _log(f"İBB genel CSV çekiliyor: {url}")
    try:
        df = pd.read_csv(url, parse_dates=["DATE"])
        _log(f"  İBB genel: {len(df)} satır, {df['DATE'].min()} → {df['DATE'].max()}")
        return df
    except Exception as exc:
        _log(f"  İBB genel hata: {exc}")
        return pd.DataFrame()


def process_ibb_individual(raw: pd.DataFrame) -> pd.DataFrame:
    """Bireysel baraj tablosunu normalize eder."""
    df = raw.copy()
    # Sütun isimlerini küçük harfe çevir
    df.columns = [c.lower() for c in df.columns]

    # Tarih sütunu
    date_col = next((c for c in df.columns if "date" in c or "tarih" in c), None)
    if date_col is None:
        _log(f"  Tarih sütunu bulunamadı: {list(df.columns)}")
        return pd.DataFrame()
    df["ts_day"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    df = df.dropna(subset=["ts_day"])

    # Doluluk sütunları — baraj adları veya % içeren sütunlar
    skip = {date_col, "_id", "ts_day"}
    value_cols = [c for c in df.columns if c not in skip]
    _log(f"  İBB bireysel sütunlar: {value_cols}")

    # Geniş → uzun format
    long = df.melt(id_vars=["ts_day"], value_vars=value_cols, var_name="dam_raw", value_name="fullness_pct")
    long["fullness_pct"] = pd.to_numeric(long["fullness_pct"], errors="coerce")
    long = long.dropna(subset=["fullness_pct"])
    long["source"] = "ibb_individual"
    return long.sort_values("ts_day").reset_index(drop=True)


def process_ibb_general(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    df.columns = [c.upper() for c in df.columns]
    df["ts_day"] = pd.to_datetime(df.get("DATE", df.iloc[:, 0]), errors="coerce").dt.normalize()
    rate_col = next((c for c in df.columns if "RATE" in c or "ORAN" in c), df.columns[1])
    df["fullness_pct"] = pd.to_numeric(df[rate_col], errors="coerce")
    df["dam_raw"] = "istanbul_genel"
    df["source"] = "ibb_general"
    return df[["ts_day", "dam_raw", "fullness_pct", "source"]].dropna().sort_values("ts_day").reset_index(drop=True)


def build_istanbul_daily(df_ind: pd.DataFrame, df_gen: pd.DataFrame) -> pd.DataFrame:
    """Bireysel + genel İstanbul verisini birleştirir, pivot oluşturur."""
    frames = []
    if not df_ind.empty:
        frames.append(df_ind)
    if not df_gen.empty:
        frames.append(df_gen)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    # İstanbul genel doluluk % → pivot
    pivot = combined.pivot_table(index="ts_day", columns="dam_raw", values="fullness_pct", aggfunc="mean")
    pivot.columns = [f"istanbul_{c.lower().replace(' ', '_')}" for c in pivot.columns]
    pivot = pivot.reset_index()

    # Normalleştir: bazı kayıtlar 0-1 (fraction), bazıları 0-100 (%) formatında
    # fraction < 1.5 ise ×100 uygula → tüm değerleri 0-100 % ölçeğine getir
    pct_cols = [c for c in pivot.columns if c != "ts_day"]
    for col in pct_cols:
        mask_frac = pivot[col] < 1.5
        pivot.loc[mask_frac, col] = pivot.loc[mask_frac, col] * 100.0

    # Genel ortalama
    if "istanbul_genel" not in pct_cols:
        pivot["istanbul_genel"] = pivot[pct_cols].mean(axis=1)
    pivot = pivot.sort_values("ts_day").reset_index(drop=True)
    _log(f"İstanbul günlük: {len(pivot)} gün, {pivot['ts_day'].min()} → {pivot['ts_day'].max()}")
    return pivot


# ---------------------------------------------------------------------------
# EPİAŞ — bugünkü aktif doluluk
# ---------------------------------------------------------------------------
def epias_login() -> str | None:
    if not USERNAME or not PASSWORD:
        _log("EPİAŞ credentials eksik (.env: EPIAS_USERNAME / EPIAS_PASSWORD)")
        return None
    try:
        resp = requests.post(
            EPIAS_LOGIN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "text/plain"},
            data={"username": USERNAME, "password": PASSWORD},
            timeout=(10, 30),
        )
        text = resp.text.strip()
        if text.startswith("TGT-"):
            return text
        m = re.search(r"/cas/v1/tickets/([^\" ]+)", text)
        return m.group(1) if m else None
    except Exception as exc:
        _log(f"EPİAŞ login hata: {exc}")
        return None


def fetch_epias_today(tgt: str) -> pd.DataFrame:
    today = datetime.now().strftime("%Y-%m-%dT00:00:00+03:00")
    payload = {"startDate": today, "endDate": today}
    headers = {"Accept": "application/json", "Content-Type": "application/json", "TGT": tgt}
    try:
        resp = requests.post(EPIAS_FULLNESS_URL, json=payload, headers=headers, timeout=(10, 60))
        if resp.status_code != 200:
            _log(f"EPİAŞ doluluk HTTP {resp.status_code}")
            return pd.DataFrame()
        data = resp.json()
        items = data.get("items", data.get("body", {}).get("items", []))
        if not items:
            return pd.DataFrame()
        df = pd.DataFrame(items)
        _log(f"EPİAŞ bugün: {len(df)} satır, kolonlar={list(df.columns)}")
        return df
    except Exception as exc:
        _log(f"EPİAŞ fetch hata: {exc}")
        return pd.DataFrame()


def process_epias(raw: pd.DataFrame, cap_df: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    ts_col = next((c for c in df.columns if "date" in c.lower()), None)
    if ts_col:
        df["ts_day"] = pd.to_datetime(df[ts_col], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
    else:
        df["ts_day"] = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    val_col = next((c for c in df.columns if "fullness" in c.lower() or "doluluk" in c.lower()), None)
    if val_col is None:
        _log(f"EPİAŞ doluluk kolonu yok: {list(df.columns)}")
        return pd.DataFrame()
    df["fullness_pct"] = pd.to_numeric(df[val_col], errors="coerce")

    # Baraj adı
    dam_col = next((c for c in df.columns if c.lower() in ("dam", "baraj", "name")), None)
    if dam_col:
        df["dam_name"] = df[dam_col].str.strip()

    basin_col = next((c for c in df.columns if "basin" in c.lower() or "havza" in c.lower()), None)
    if basin_col:
        df["basin_epias"] = df[basin_col]

    # Kapasite eşleştir
    cap_map = cap_df.set_index("dam_name").to_dict("index")
    df["dam_name_clean"] = df["dam_name"].str.strip()

    for col in ["basin", "il", "cap_max_hm3", "cap_active_hm3", "installed_mw"]:
        df[col] = df["dam_name_clean"].map(lambda n, c=col: cap_map.get(n, {}).get(c))

    # Depolanan su (hm³)
    df["stored_water_hm3"] = df["fullness_pct"] * df["cap_active_hm3"] / 100.0

    keep = ["ts_day", "dam_name_clean", "basin", "il",
            "fullness_pct", "cap_max_hm3", "cap_active_hm3", "stored_water_hm3", "installed_mw"]
    keep = [c for c in keep if c in df.columns or c == "ts_day"]
    out = df[[c for c in keep if c in df.columns]].copy()
    out = out.rename(columns={"dam_name_clean": "dam_name"})
    return out.dropna(subset=["fullness_pct"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Havza bazlı günlük özet
# ---------------------------------------------------------------------------
def build_basin_daily(df_epias: pd.DataFrame) -> pd.DataFrame:
    """Her havza için kapasite-ağırlıklı doluluk % ve depolanan su."""
    if df_epias.empty or "basin" not in df_epias.columns:
        return pd.DataFrame()
    df = df_epias.dropna(subset=["basin", "fullness_pct"]).copy()
    # Kapasite-ağırlıklı doluluk
    df["weight_x_full"] = df["cap_active_hm3"].fillna(0) * df["fullness_pct"]
    agg = df.groupby(["ts_day", "basin"]).agg(
        dam_count=("fullness_pct", "count"),
        fullness_pct_mean=("fullness_pct", "mean"),
        fullness_pct_wtd=("weight_x_full", "sum"),
        cap_active_sum=("cap_active_hm3", "sum"),
        stored_water_sum=("stored_water_hm3", "sum"),
        installed_mw_sum=("installed_mw", "sum"),
    ).reset_index()
    agg["fullness_pct_wtd"] = agg["fullness_pct_wtd"] / agg["cap_active_sum"].replace(0, float("nan"))
    return agg


def build_national_daily(df_epias: pd.DataFrame) -> pd.DataFrame:
    if df_epias.empty:
        return pd.DataFrame()
    df = df_epias.dropna(subset=["fullness_pct"]).copy()
    df["weight_x_full"] = df["cap_active_hm3"].fillna(0) * df["fullness_pct"]
    agg = df.groupby("ts_day").agg(
        dam_count=("fullness_pct", "count"),
        national_fullness_mean=("fullness_pct", "mean"),
        national_cap_active_hm3=("cap_active_hm3", "sum"),
        national_stored_hm3=("stored_water_hm3", "sum"),
        national_mw=("installed_mw", "sum"),
        weight_sum=("weight_x_full", "sum"),
    ).reset_index()
    agg["national_fullness_wtd"] = agg["weight_sum"] / agg["national_cap_active_hm3"].replace(0, float("nan"))
    return agg.drop(columns=["weight_sum"])


# ---------------------------------------------------------------------------
# Saatlik hizalama
# ---------------------------------------------------------------------------
def align_to_hourly(daily: pd.DataFrame, date_col: str, start: datetime, end: datetime) -> pd.DataFrame:
    spine = pd.date_range(start, end, freq="h")
    h = pd.DataFrame({"ts_hour": spine})
    h["ts_day"] = h["ts_hour"].dt.normalize()
    d = daily.rename(columns={date_col: "ts_day"}).copy()
    d["ts_day"] = pd.to_datetime(d["ts_day"]).dt.normalize()
    merged = h.merge(d, on="ts_day", how="left")
    value_cols = [c for c in d.columns if c != "ts_day"]
    merged[value_cols] = merged[value_cols].ffill()
    return merged.drop(columns=["ts_day"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Ana akış
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ibb-only", action="store_true")
    ap.add_argument("--epias-only", action="store_true")
    args = ap.parse_args()

    # 1. Kapasite tablosu kaydet
    cap_df = save_capacity_table()

    # 2. EPİAŞ verisi
    df_epias_enriched = pd.DataFrame()
    df_basin = pd.DataFrame()
    df_national = pd.DataFrame()

    if not args.ibb_only:
        tgt = epias_login()
        if tgt:
            _log(f"EPİAŞ TGT ok: {tgt[:12]}…")
            raw_epias = fetch_epias_today(tgt)
            if not raw_epias.empty:
                df_epias_enriched = process_epias(raw_epias, cap_df)
                df_basin = build_basin_daily(df_epias_enriched)
                df_national = build_national_daily(df_epias_enriched)
                _log(f"EPİAŞ zenginleştirildi: {len(df_epias_enriched)} baraj")
                _log(f"Ulusal depolama: {df_national['national_stored_hm3'].sum():.0f} hm³")
        else:
            _log("EPİAŞ auth başarısız — sadece İBB verisi işlenecek.")

    # 3. İBB verisi
    df_istanbul = pd.DataFrame()
    if not args.epias_only:
        raw_ind = fetch_ibb_individual_dams()
        raw_gen = fetch_ibb_general()
        df_ind = process_ibb_individual(raw_ind) if not raw_ind.empty else pd.DataFrame()
        df_gen = process_ibb_general(raw_gen) if not raw_gen.empty else pd.DataFrame()
        df_istanbul = build_istanbul_daily(df_ind, df_gen)

    # 4. Çıktıları kaydet
    OUT_IBB_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DAILY.parent.mkdir(parents=True, exist_ok=True)

    if not df_istanbul.empty:
        df_istanbul.to_parquet(OUT_IBB_PARQUET, index=False)
        _log(f"İBB parquet → {OUT_IBB_PARQUET}  ({len(df_istanbul)} gün)")

    if not df_epias_enriched.empty:
        df_epias_enriched.to_parquet(OUT_DAILY, index=False)
        _log(f"Günlük enriched → {OUT_DAILY}")

    if not df_basin.empty:
        basin_path = OUT_DAILY.parent / "dam_basin_daily.parquet"
        df_basin.to_parquet(basin_path, index=False)
        _log(f"Havza günlük → {basin_path}")

        # Havza bazlı rapor
        print("\n── Havza Özeti (bugün) ───────────────────────────────────────")
        report = df_basin.sort_values("cap_active_sum", ascending=False)
        for _, row in report.iterrows():
            print(f"  {row['basin']:<22} {row['fullness_pct_wtd']:5.1f}%  "
                  f"{row['stored_water_sum']:7.0f} hm³  "
                  f"({row['installed_mw_sum']:.0f} MW)")
        print()

    if not df_national.empty:
        nat_path = OUT_DAILY.parent / "dam_national_daily.parquet"
        df_national.to_parquet(nat_path, index=False)
        row = df_national.iloc[0]
        _log(f"Ulusal: {row['national_fullness_wtd']:.1f}%  {row['national_stored_hm3']:.0f} hm³  {row['national_mw']:.0f} MW")

    # 5. Saatlik versiyon (EPİAŞ için sadece bugün, İBB için tüm geçmiş)
    if not df_istanbul.empty:
        start_h = datetime(2005, 1, 1)
        end_h = datetime.now().replace(minute=0, second=0, microsecond=0)
        hourly_ist = align_to_hourly(df_istanbul, "ts_day", start_h, end_h)
        hourly_ist_path = OUT_IBB_DIR / "ibb_dam_hourly.parquet"
        hourly_ist.to_parquet(hourly_ist_path, index=False)
        _log(f"İstanbul saatlik → {hourly_ist_path}  ({len(hourly_ist)} saat)")

    _log("Tamamlandı.")


if __name__ == "__main__":
    main()
