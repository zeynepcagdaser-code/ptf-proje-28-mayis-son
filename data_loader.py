#!/usr/bin/env python3
"""
GÖP sabah veri yükleyicisi — statik dosya okuma YOK.

Her gün saat 11:00'de (GÖP kapısı kapanmadan önce) çalıştırılır ve
model için gereken tüm ham veriyi canlı olarak çeker.

Ana giriş noktası:
    snapshot = fetch_morning_snapshot("2026-06-05")
    # snapshot["ptf_smf"]        → son 30 gün saatlik PTF + SMF
    # snapshot["res_forecast"]   → rüzgar + güneş üretim tahmini + kapasite faktörü
    # snapshot["outages"]        → aktif REMIT plansız kesintiler
    # snapshot["reservoir"]      → baraj doluluk oranları
    # snapshot["commodities"]    → TTF, API2 kömür, Brent, döviz kurları

Kimlik doğrulama:
    EPİAŞ : .env → EPIAS_USERNAME / EPIAS_PASSWORD
    TCMB  : .env → EVDS_API_KEY
    Yahoo Finance: kimlik gerekmez

Fiyat tavanı kuralı:
    EPDK Nisan 2026 kararı → GÖP/DGP tavan = 4500 TL/MWh.
    get_price_ceiling(date) her zaman kullanılmalı; sabit 3400 yazılmamalıdır.
"""

from __future__ import annotations

import os
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

# ─── Sabitler ─────────────────────────────────────────────────────────────────

_EPIAS_LOGIN = "https://giris.epias.com.tr/cas/v1/tickets"
_SEFFAF_BASE = "https://seffaflik.epias.com.tr/electricity-service"

_URL_MCP           = f"{_SEFFAF_BASE}/v1/markets/dam/data/mcp"
_URL_SMP           = f"{_SEFFAF_BASE}/v1/markets/bpm/data/smp"
_URL_LOAD_EST      = f"{_SEFFAF_BASE}/v1/consumption/data/load-estimation-plan"
_URL_SYS_DIR       = f"{_SEFFAF_BASE}/v1/markets/bpm/data/system-direction"
_URL_RES_FORECAST  = f"{_SEFFAF_BASE}/v1/renewables/data/res-generation-and-forecast"
_URL_OUTAGES       = f"{_SEFFAF_BASE}/v1/markets/data/market-message-system"
_URL_RESERVOIR     = f"{_SEFFAF_BASE}/v1/dams/data/active-fullness"

_URL_EVDS          = "https://evds3.tcmb.gov.tr/igmevdsms-dis/"
_EVDS_SERIES = {
    "TP_DK_USD_A_YTL": "usd_try",
    "TP_DK_EUR_A_YTL": "eur_try",
}

# Türkiye 2026 kurulu RES kapasitesi tahmini (MW) — kapasite faktörü hesabı için
_INSTALLED_WIND_MW  = 15_500.0
_INSTALLED_SOLAR_MW = 10_000.0

REQUEST_TIMEOUT = (10, 120)
MAX_RETRIES = 4

# ─── Fiyat tavanı ─────────────────────────────────────────────────────────────

_CEILING_CHANGE_DATE = date(2026, 4, 1)
PRICE_FLOOR: float = 0.0
PRICE_CEILING_OLD: float = 3400.0
PRICE_CEILING_NEW: float = 4500.0


def get_price_ceiling(for_date: date | datetime | str) -> float:
    """Teslimat tarihine göre GÖP tavan fiyatını döndür (TL/MWh)."""
    if isinstance(for_date, str):
        for_date = datetime.fromisoformat(for_date[:10]).date()
    elif isinstance(for_date, datetime):
        for_date = for_date.date()
    return PRICE_CEILING_NEW if for_date >= _CEILING_CHANGE_DATE else PRICE_CEILING_OLD


# ─── Ortak HTTP yardımcıları ──────────────────────────────────────────────────

def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _post(url: str, **kwargs) -> requests.Response:
    last: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return requests.post(url, timeout=REQUEST_TIMEOUT, **kwargs)
        except requests.exceptions.RequestException as exc:
            last = exc
            wait = min(60, 5 * 2 ** (attempt - 1))
            if attempt < MAX_RETRIES:
                time.sleep(wait)
    raise last  # type: ignore[misc]


def _get(url: str, **kwargs) -> requests.Response:
    last: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return requests.get(url, timeout=REQUEST_TIMEOUT, **kwargs)
        except requests.exceptions.RequestException as exc:
            last = exc
            wait = min(60, 5 * 2 ** (attempt - 1))
            if attempt < MAX_RETRIES:
                time.sleep(wait)
    raise last  # type: ignore[misc]


def _ds(d: date | datetime | str) -> str:
    if isinstance(d, str):
        return d[:10]
    return d.strftime("%Y-%m-%d")


# ─── EPİAŞ kimlik doğrulama ───────────────────────────────────────────────────

_tgt_cache: str | None = None
_tgt_ts: datetime | None = None
_TGT_TTL_HOURS = 6


def _epias_creds() -> tuple[str, str]:
    u = os.environ.get("EPIAS_USERNAME") or os.environ.get("EPTR_USERNAME", "")
    p = os.environ.get("EPIAS_PASSWORD") or os.environ.get("EPTR_PASSWORD", "")
    if not u or not p:
        raise EnvironmentError(".env içinde EPIAS_USERNAME / EPIAS_PASSWORD tanımlı olmalı.")
    return u, p


def _get_tgt() -> str:
    """TGT biletini döndür; süresi dolmuşsa yenile. Bellek içi cache'ler."""
    global _tgt_cache, _tgt_ts
    if (
        _tgt_cache
        and _tgt_ts
        and (datetime.now() - _tgt_ts).total_seconds() < _TGT_TTL_HOURS * 3600
    ):
        return _tgt_cache
    u, p = _epias_creds()
    resp = _post(
        _EPIAS_LOGIN,
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "text/plain"},
        data={"username": u, "password": p},
    )
    text = resp.text.strip()
    if text.startswith("TGT-"):
        _tgt_cache = text
    else:
        m = re.search(r"/cas/v1/tickets/([^\" ]+)", text)
        if not m:
            raise RuntimeError(f"TGT alınamadı: {text[:400]}")
        _tgt_cache = m.group(1)
    _tgt_ts = datetime.now()
    return _tgt_cache


def _headers() -> dict[str, str]:
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "TGT": _get_tgt(),
    }


def _items(resp: requests.Response, fallback_key: str = "items") -> list[dict]:
    if resp.status_code != 200:
        _log(f"UYARI HTTP {resp.status_code}: {resp.text[:300]}")
        return []
    try:
        payload = resp.json()
    except Exception:
        return []
    if isinstance(payload, list):
        return payload
    for key in [fallback_key, "items", "data", "result", "content"]:
        if isinstance(payload.get(key), list):
            return payload[key]
    return []


# ─── eptr2 istemcisi (PTF/SMF için) ──────────────────────────────────────────

_eptr2_client: Any = None


def _get_eptr2() -> Any:
    global _eptr2_client
    if _eptr2_client is None:
        try:
            from eptr2 import EPTR
        except ImportError as exc:
            raise ImportError("pip install 'eptr2[allextras]>=1.2.4'") from exc
        u, p = _epias_creds()
        _eptr2_client = EPTR(username=u, password=p)
    return _eptr2_client


def _eptr2_call(endpoint: str, start: str, end: str) -> pd.DataFrame:
    raw = _get_eptr2().call(endpoint, start_date=start, end_date=end)
    return raw if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw)


# ─── 1. PTF ve SMF (son N gün) ────────────────────────────────────────────────

def fetch_ptf_smf_rolling(
    end_date: str | date | None = None,
    lookback_days: int = 30,
) -> pd.DataFrame:
    """
    Son `lookback_days` günün saatlik PTF (GÖP takas fiyatı) ve
    SMF (DGP sistem marjinal fiyatı) verilerini çek.

    Dönen sütunlar: date, hour, ptf, smf, system_direction, price_ceiling
    """
    end = date.today() if end_date is None else (
        datetime.fromisoformat(_ds(end_date)).date() if isinstance(end_date, str) else end_date
    )
    start = end - timedelta(days=lookback_days)
    s, e = _ds(start), _ds(end)

    _log(f"PTF/SMF çekiliyor: {s} → {e}")
    try:
        ptf = _eptr2_call("mcp", s, e).rename(
            columns={"marketTradePrice": "ptf", "price": "ptf", "mcp": "ptf"}, errors="ignore"
        )
        smf = _eptr2_call("smp", s, e).rename(
            columns={
                "systemMarginalPrice": "smf", "smp": "smf",
                "systemDirection": "system_direction", "direction": "system_direction",
            }, errors="ignore"
        )
    except Exception as exc:
        _log(f"eptr2 başarısız, ham endpoint'e geçiliyor: {exc}")
        ptf = _fetch_mcp_raw(s, e)
        smf = _fetch_smf_raw(s, e)

    # Tavan kırpması
    if "ptf" in ptf.columns and "date" in ptf.columns:
        ptf["ptf"] = ptf.apply(
            lambda r: float(
                min(max(float(r["ptf"]), PRICE_FLOOR), get_price_ceiling(str(r["date"])[:10]))
            ),
            axis=1,
        )

    merge_cols = [c for c in ["date", "hour"] if c in ptf.columns and c in smf.columns]
    if merge_cols:
        df = ptf.merge(smf, on=merge_cols, how="outer")
    else:
        df = ptf

    if "date" in df.columns:
        df["price_ceiling"] = df["date"].apply(lambda d: get_price_ceiling(str(d)[:10]))

    return df.sort_values(merge_cols).reset_index(drop=True) if merge_cols else df


def _fetch_mcp_raw(start: str, end: str) -> pd.DataFrame:
    """Yedek: eptr2 olmadan doğrudan endpoint."""
    resp = _post(
        _URL_MCP,
        json={"startDate": f"{start}T00:00:00+03:00", "endDate": f"{end}T23:00:00+03:00"},
        headers=_headers(),
    )
    rows = _items(resp)
    if not rows:
        return pd.DataFrame(columns=["date", "hour", "ptf"])
    df = pd.DataFrame(rows)
    df = df.rename(columns={"marketTradePrice": "ptf"}, errors="ignore")
    return df


def _fetch_smf_raw(start: str, end: str) -> pd.DataFrame:
    resp = _post(
        _URL_SMP,
        json={"startDate": f"{start}T00:00:00+03:00", "endDate": f"{end}T23:00:00+03:00"},
        headers=_headers(),
    )
    rows = _items(resp)
    if not rows:
        return pd.DataFrame(columns=["date", "hour", "smf", "system_direction"])
    df = pd.DataFrame(rows)
    df = df.rename(
        columns={"systemMarginalPrice": "smf", "systemDirection": "system_direction"},
        errors="ignore",
    )
    return df


# ─── 2. RES üretim tahmini + kapasite faktörü ─────────────────────────────────

def fetch_res_forecast(
    start: str | date,
    end: str | date,
) -> pd.DataFrame:
    """
    EPİAŞ RES (yenilenebilir enerji) üretim ve tahmin verisini çek.
    Kapasite faktörlerini hesapla.

    Dönen sütunlar: date, hour, wind_actual_mwh, wind_forecast_mwh,
                    solar_actual_mwh, solar_forecast_mwh,
                    wind_cf, solar_cf  (0–1 arası kapasite faktörü)
    """
    s, e = _ds(start), _ds(end)
    _log(f"RES tahmini çekiliyor: {s} → {e}")
    resp = _post(
        _URL_RES_FORECAST,
        json={"startDate": f"{s}T00:00:00+03:00", "endDate": f"{e}T23:00:00+03:00"},
        headers=_headers(),
    )
    rows = _items(resp)
    if not rows:
        _log("UYARI: RES tahmini boş döndü")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Sütun adı normalizasyonu (EPİAŞ API farklı sürümlerde değişebilir)
    col_map = {
        "windPowerForecast": "wind_forecast_mwh",
        "rtgWind": "wind_actual_mwh",
        "solarPowerForecast": "solar_forecast_mwh",
        "rtgSolar": "solar_actual_mwh",
        "windGeneration": "wind_actual_mwh",
        "solarGeneration": "solar_actual_mwh",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    for col in ["wind_forecast_mwh", "solar_forecast_mwh",
                "wind_actual_mwh", "solar_actual_mwh"]:
        if col not in df.columns:
            df[col] = float("nan")
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["wind_cf"] = df["wind_forecast_mwh"] / _INSTALLED_WIND_MW
    df["solar_cf"] = df["solar_forecast_mwh"] / _INSTALLED_SOLAR_MW
    df["wind_cf"] = df["wind_cf"].clip(0.0, 1.0)
    df["solar_cf"] = df["solar_cf"].clip(0.0, 1.0)

    return df


# ─── 3. REMIT plansız kesintiler ──────────────────────────────────────────────

def fetch_active_outages(
    reference_date: str | date | None = None,
    lookback_days: int = 7,
) -> pd.DataFrame:
    """
    REMIT / UMM üzerinden aktif santral plansız kesintileri ve kapasite düşüşlerini çek.

    EPİAŞ 'market-message-system' endpoint'i (UMM = Urgent Market Messages),
    Türkiye'nin REMIT uyumlu kapasitede azalma bildirimlerini içerir.

    Dönen sütunlar: powerPlantName, uevcbName, caseStartDate, caseEndDate,
                    operatorPower, capacityAtCaseTime, capacity_loss_mw
    """
    end = date.today() if reference_date is None else (
        datetime.fromisoformat(_ds(reference_date)).date()
        if isinstance(reference_date, str) else reference_date
    )
    start = end - timedelta(days=lookback_days)
    s, e = _ds(start), _ds(end)
    _log(f"REMIT kesintileri çekiliyor: {s} → {e}")

    resp = _post(
        _URL_OUTAGES,
        json={
            "startDate": f"{s}T00:00:00+03:00",
            "endDate": f"{e}T23:59:59+03:00",
            "page": {"number": 1, "size": 1000},
        },
        headers=_headers(),
    )
    rows = _items(resp, fallback_key="items")
    if not rows:
        _log("UYARI: Kesinti verisi boş döndü")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Kapasite kaybı
    for col in ["operatorPower", "capacityAtCaseTime"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "operatorPower" in df.columns and "capacityAtCaseTime" in df.columns:
        df["capacity_loss_mw"] = (
            df["operatorPower"].fillna(0) - df["capacityAtCaseTime"].fillna(0)
        ).clip(lower=0)

    # Sadece aktif kesintiler (bugün hâlâ devam edenler)
    if "caseEndDate" in df.columns:
        df["caseEndDate"] = pd.to_datetime(df["caseEndDate"], errors="coerce")
        now = pd.Timestamp.now()
        df = df[df["caseEndDate"].isna() | (df["caseEndDate"] >= now)]

    return df.reset_index(drop=True)


# ─── 4. Baraj doluluk oranları ────────────────────────────────────────────────

def fetch_reservoir_fullness(
    end_date: str | date | None = None,
    lookback_days: int = 30,
) -> pd.DataFrame:
    """
    EPİAŞ baraj aktif doluluk oranlarını çek ve ulusal günlük ortalamaya
    topla (mean / min / max).

    Dönen sütunlar: date, basin, dam, active_fullness_pct,
                    + pivot: national_mean_pct, national_min_pct, national_max_pct
    """
    end = date.today() if end_date is None else (
        datetime.fromisoformat(_ds(end_date)).date()
        if isinstance(end_date, str) else end_date
    )
    start = end - timedelta(days=lookback_days)
    s, e = _ds(start), _ds(end)
    _log(f"Baraj dolulukları çekiliyor: {s} → {e}")

    resp = _post(
        _URL_RESERVOIR,
        json={"startDate": s, "endDate": e},
        headers=_headers(),
    )
    rows = _items(resp)
    if not rows:
        _log("UYARI: Baraj verisi boş döndü")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    col_map = {
        "activeFullnessAmount": "active_fullness_pct",
        "basin": "basin",
        "dam": "dam",
        "date": "date",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    if "active_fullness_pct" in df.columns:
        df["active_fullness_pct"] = pd.to_numeric(df["active_fullness_pct"], errors="coerce")

    # Ulusal agregasyon
    if "date" in df.columns and "active_fullness_pct" in df.columns:
        national = (
            df.groupby("date")["active_fullness_pct"]
            .agg(national_mean_pct="mean", national_min_pct="min", national_max_pct="max")
            .reset_index()
        )
        df = df.merge(national, on="date", how="left")

    return df


# ─── 5. Emtia fiyatları (TTF, API2, Brent) + döviz kurları ──────────────────

_YFINANCE_TICKERS = {
    "TTF=F":  "ttf_usd_mmbtu",   # TTF Avrupa doğalgaz
    "NG=F":   "hhub_usd_mmbtu",  # Henry Hub doğalgaz (TTF proxy)
    "BZ=F":   "brent_usd_bbl",   # Brent ham petrol
    "MTF=F":  "coal_api2_usd_t", # API2 Rotterdam kömür
}
_REQUIRED_TICKERS = {"NG=F", "BZ=F"}


def fetch_commodity_prices(
    end_date: str | date | None = None,
    lookback_days: int = 35,
) -> pd.DataFrame:
    """
    Yahoo Finance üzerinden TTF gaz, API2 kömür, Brent ham petrol fiyatlarını
    çek. TCMB EVDS'den USD/TRY ve EUR/TRY kurlarını ekle.

    Dönen sütunlar: date, ttf_usd_mmbtu, hhub_usd_mmbtu, brent_usd_bbl,
                    coal_api2_usd_t, usd_try, eur_try,
                    ttf_try_mmbtu (varsa)
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("pip install yfinance") from exc

    end = date.today() if end_date is None else (
        datetime.fromisoformat(_ds(end_date)).date()
        if isinstance(end_date, str) else end_date
    )
    start = end - timedelta(days=lookback_days)
    _log(f"Emtia fiyatları çekiliyor: {_ds(start)} → {_ds(end)}")

    frames: dict[str, pd.Series] = {}
    missing_required: list[str] = []

    for ticker, col in _YFINANCE_TICKERS.items():
        try:
            raw = yf.download(ticker, start=_ds(start), end=_ds(end), progress=False, auto_adjust=True)
            if raw.empty:
                raise ValueError("boş")
            close = raw["Close"]
            if hasattr(close, "iloc") and close.ndim > 1:
                close = close.iloc[:, 0]
            close.index = pd.to_datetime(close.index).normalize()
            frames[col] = close.rename(col)
            _log(f"  {ticker}: {len(close)} gün")
        except Exception as exc:
            _log(f"  {ticker}: {exc}")
            if ticker in _REQUIRED_TICKERS:
                missing_required.append(ticker)

    if missing_required:
        _log(f"UYARI: Zorunlu ticker(lar) alınamadı: {missing_required}")

    if not frames:
        return pd.DataFrame(columns=["date"])

    df = pd.concat(frames.values(), axis=1).reset_index()
    df = df.rename(columns={"index": "date", "Date": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)

    # TCMB döviz kurları
    try:
        fx = _fetch_tcmb_rates(_ds(start), _ds(end))
        if not fx.empty:
            df = df.merge(fx, on="date", how="left")
            # TTF → TRY
            if "ttf_usd_mmbtu" in df.columns and "usd_try" in df.columns:
                df["ttf_try_mmbtu"] = df["ttf_usd_mmbtu"] * df["usd_try"]
            if "brent_usd_bbl" in df.columns and "usd_try" in df.columns:
                df["brent_try_bbl"] = df["brent_usd_bbl"] * df["usd_try"]
    except Exception as exc:
        _log(f"UYARI: TCMB kuru alınamadı — {exc}")

    return df.sort_values("date").reset_index(drop=True)


def _fetch_tcmb_rates(start: str, end: str) -> pd.DataFrame:
    """TCMB EVDS'den USD/TRY ve EUR/TRY günlük kurlarını çek."""
    key = (
        os.environ.get("EVDS_API_KEY")
        or os.environ.get("TCMB_EVDS_API_KEY")
        or os.environ.get("EVDS_KEY", "")
    )
    if not key:
        raise EnvironmentError(".env içinde EVDS_API_KEY tanımlı olmalı.")

    series_str = "-".join(_EVDS_SERIES.keys())
    url = (
        f"{_URL_EVDS}series={series_str}"
        f"&startDate={start.replace('-', '-')}&endDate={end.replace('-', '-')}"
        f"&type=json&key={key}"
    )
    resp = _get(url)
    if resp.status_code != 200:
        raise RuntimeError(f"EVDS HTTP {resp.status_code}: {resp.text[:300]}")

    payload = resp.json()
    items = payload.get("items", payload) if isinstance(payload, dict) else payload
    if not isinstance(items, list) or not items:
        raise RuntimeError("EVDS boş döndü")

    df = pd.DataFrame(items)
    if "Tarih" not in df.columns:
        raise RuntimeError(f"EVDS yanıtında Tarih kolonu yok: {df.columns.tolist()}")

    df["date"] = pd.to_datetime(df["Tarih"], dayfirst=True, errors="coerce").dt.date.astype(str)
    for raw_name, col_name in _EVDS_SERIES.items():
        if raw_name in df.columns:
            df[col_name] = pd.to_numeric(df[raw_name], errors="coerce")

    keep = ["date"] + [v for v in _EVDS_SERIES.values() if v in df.columns]
    return df[keep].dropna(subset=["date"])


# ─── Ana giriş noktası ────────────────────────────────────────────────────────

def fetch_morning_snapshot(
    for_date: str | date | None = None,
    lookback_days: int = 30,
) -> dict[str, pd.DataFrame]:
    """
    GÖP kapısı kapanmadan önce (saat 11:00) çalıştırılacak toplu veri çekimi.

    Beş veri kategorisinin tamamını canlı olarak çeker; hiçbir statik
    dosyaya dokunmaz.

    Args:
        for_date     : Teslimat günü (varsayılan: bugün)
        lookback_days: PTF/SMF ve emtia serileri için geriye bakış (varsayılan: 30)

    Returns:
        {
          "ptf_smf"      : pd.DataFrame  — saatlik PTF + SMF + sistem yönü
          "res_forecast" : pd.DataFrame  — rüzgar + güneş üretim tahmini + cf
          "outages"      : pd.DataFrame  — REMIT aktif kesintiler
          "reservoir"    : pd.DataFrame  — baraj doluluk oranları
          "commodities"  : pd.DataFrame  — TTF, API2, Brent, döviz
        }
    """
    target = date.today() if for_date is None else (
        datetime.fromisoformat(_ds(for_date)).date()
        if isinstance(for_date, str) else for_date
    )
    _log(f"Sabah snapshot başlıyor → {_ds(target)}")

    results: dict[str, pd.DataFrame] = {}

    # 1. PTF + SMF
    try:
        results["ptf_smf"] = fetch_ptf_smf_rolling(target, lookback_days)
        _log(f"  ptf_smf: {len(results['ptf_smf'])} satır")
    except Exception as exc:
        _log(f"  ptf_smf HATA: {exc}")
        results["ptf_smf"] = pd.DataFrame()

    # 2. RES üretim tahmini (bugün + 7 gün ileri)
    try:
        res_end = target + timedelta(days=7)
        results["res_forecast"] = fetch_res_forecast(target, res_end)
        _log(f"  res_forecast: {len(results['res_forecast'])} satır")
    except Exception as exc:
        _log(f"  res_forecast HATA: {exc}")
        results["res_forecast"] = pd.DataFrame()

    # 3. REMIT kesintiler
    try:
        results["outages"] = fetch_active_outages(target, lookback_days=7)
        _log(f"  outages: {len(results['outages'])} aktif kesinti")
    except Exception as exc:
        _log(f"  outages HATA: {exc}")
        results["outages"] = pd.DataFrame()

    # 4. Baraj dolulukları
    try:
        results["reservoir"] = fetch_reservoir_fullness(target, lookback_days)
        _log(f"  reservoir: {len(results['reservoir'])} satır")
    except Exception as exc:
        _log(f"  reservoir HATA: {exc}")
        results["reservoir"] = pd.DataFrame()

    # 5. Emtia + döviz
    try:
        results["commodities"] = fetch_commodity_prices(target, lookback_days + 5)
        _log(f"  commodities: {len(results['commodities'])} gün")
    except Exception as exc:
        _log(f"  commodities HATA: {exc}")
        results["commodities"] = pd.DataFrame()

    _log("Sabah snapshot tamamlandı.")
    return results
