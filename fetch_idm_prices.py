#!/usr/bin/env python3
"""Fetch EPİAŞ Gün İçi Piyasa (GİP/IDM) hourly weighted-average prices.

Endpoint (electricity-service):
  POST /v1/markets/idm/data/idm-volume-weighted-average-price

Data frequency: hourly (settled after GÖP, corrected through the day).

The GÖP–GİP spread is a real-time imbalance signal: when GİP trades above
PTF, the market is short; when it trades below, there is surplus.
Lagged versions (lag_24, lag_168) are safe anchor-hour features.

Outputs:
  - Raw CSV:  data/external/epias_market/idm/idm_prices_raw.csv
  - Parquet:  data/processed/idm_prices.parquet
  - Report:   reports/idm_prices_report.{json,md}
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

LOGIN_URL = "https://giris.epias.com.tr/cas/v1/tickets"
ENDPOINT_URL = (
    "https://seffaflik.epias.com.tr/electricity-service"
    "/v1/markets/idm/data/weighted-average-price"
)

RAW_DIR = PROJECT_ROOT / "data" / "external" / "epias_market" / "idm"
RAW_CSV = RAW_DIR / "idm_prices_raw.csv"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "idm_prices.parquet"
REPORT_JSON = PROJECT_ROOT / "reports" / "idm_prices_report.json"
REPORT_MD = PROJECT_ROOT / "reports" / "idm_prices_report.md"

REQUEST_TIMEOUT = (10, 120)
MAX_RETRIES = 4
MAX_DAYS_PER_CHUNK = 90
SLEEP_SECONDS = 1.5
ROLLING_REFRESH_HOURS = 72


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def post_with_retries(url: str, **kwargs: Any) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(url, timeout=REQUEST_TIMEOUT, **kwargs)
            if resp.status_code == 429 and attempt < MAX_RETRIES:
                wait_s = min(120, 3 * 2 ** (attempt - 1))
                _log(f"rate-limit retry={attempt}/{MAX_RETRIES} wait={wait_s}s")
                time.sleep(wait_s)
                continue
            return resp
        except requests.exceptions.RequestException as exc:
            last_error = exc
            wait_s = min(60, 3 * 2 ** (attempt - 1))
            _log(f"network retry={attempt}/{MAX_RETRIES} wait={wait_s}s err={exc}")
            time.sleep(wait_s)
    raise last_error  # type: ignore[misc]


def obtain_tgt(username: str, password: str) -> str:
    resp = post_with_retries(
        LOGIN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "text/plain"},
        data={"username": username, "password": password},
    )
    text = resp.text.strip()
    if text.startswith("TGT-"):
        return text
    m = re.search(r"/cas/v1/tickets/([^\" ]+)", text)
    if not m:
        raise RuntimeError(f"TGT alınamadı: {text[:500]}")
    return m.group(1)


def _ptf_max_date() -> datetime:
    ptf_path = PROJECT_ROOT / "data" / "ptf_dataset.csv"
    if ptf_path.exists():
        ts = pd.to_datetime(pd.read_csv(ptf_path, usecols=["date"])["date"], errors="coerce")
        if getattr(ts.dt, "tz", None) is not None:
            ts = ts.dt.tz_convert("Europe/Istanbul").dt.tz_localize(None)
        return ts.max().to_pydatetime().replace(tzinfo=None)
    return datetime.now()


def _incremental_start() -> datetime:
    if RAW_CSV.exists():
        old = pd.read_csv(RAW_CSV)
        if not old.empty and "ts_hour" in old.columns:
            ts = pd.to_datetime(old["ts_hour"], errors="coerce")
            last = ts.max()
            if not pd.isna(last):
                return (last - timedelta(hours=ROLLING_REFRESH_HOURS)).to_pydatetime()
    return datetime(2020, 1, 1)


def _normalize_items(items: list[dict[str, Any]]) -> pd.DataFrame:
    if not items:
        return pd.DataFrame(columns=["ts_hour", "idm_price"])
    df = pd.DataFrame(items)
    _log(f"  API kolonları: {list(df.columns)}")

    # Identify timestamp column
    date_col = next(
        (c for c in ["date", "effectiveDate", "hour", "datetime", "startDate"] if c in df.columns),
        None,
    )
    if date_col is None:
        raise RuntimeError(f"Zaman kolonu bulunamadı. Kolonlar: {list(df.columns)}")

    # Identify price column
    price_col = next(
        (c for c in ["weightedAveragePrice", "price", "idmPrice", "marketPrice",
                     "weightedAvgPrice", "vwap", "wap", "value"]
         if c in df.columns),
        None,
    )
    if price_col is None:
        numeric_cols = [c for c in df.columns if c != date_col]
        if not numeric_cols:
            raise RuntimeError(f"Fiyat kolonu bulunamadı. Kolonlar: {list(df.columns)}")
        price_col = numeric_cols[0]
        _log(f"  Fiyat kolonu otomatik seçildi: {price_col}")

    out = pd.DataFrame()
    ts = pd.to_datetime(df[date_col], errors="coerce")
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert("Europe/Istanbul").dt.tz_localize(None)
    out["ts_hour"] = ts.dt.floor("h")
    out["idm_price"] = pd.to_numeric(df[price_col], errors="coerce")
    out = out.dropna(subset=["ts_hour", "idm_price"])
    out = out.drop_duplicates(subset=["ts_hour"], keep="last").sort_values("ts_hour").reset_index(drop=True)
    return out


def fetch_chunk(start: datetime, end: datetime, tgt: str) -> pd.DataFrame:
    headers = {"Accept": "application/json", "Content-Type": "application/json", "TGT": tgt}
    payload = {
        "startDate": start.strftime("%Y-%m-%dT00:00:00+03:00"),
        "endDate": end.strftime("%Y-%m-%dT23:00:00+03:00"),
    }
    _log(f"Fetch: {payload['startDate']} → {payload['endDate']}")
    resp = post_with_retries(ENDPOINT_URL, json=payload, headers=headers)
    _log(f"HTTP {resp.status_code}")
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:2000]}")
    data = resp.json()
    items = data.get("items", data.get("body", {}).get("items", []))
    return _normalize_items(items)


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add PTF join and lag features. PTF join is optional (file may not be current)."""
    out = df.copy().sort_values("ts_hour").reset_index(drop=True)

    # Lag features — safe as anchor-hour features (GİP is settled, not future)
    out["idm_price_lag_24"] = out["idm_price"].shift(24)
    out["idm_price_lag_48"] = out["idm_price"].shift(48)
    out["idm_price_lag_168"] = out["idm_price"].shift(168)
    out["idm_price_roll_mean_24"] = out["idm_price"].rolling(24, min_periods=6).mean()
    out["idm_price_roll_mean_168"] = out["idm_price"].rolling(168, min_periods=48).mean()

    # GÖP–GİP spread: requires PTF; compute if available
    ptf_path = PROJECT_ROOT / "data" / "ptf_dataset.csv"
    if ptf_path.exists():
        ptf = pd.read_csv(ptf_path, usecols=["date", "price"])
        ts = pd.to_datetime(ptf["date"], errors="coerce")
        if getattr(ts.dt, "tz", None) is not None:
            ts = ts.dt.tz_convert("Europe/Istanbul").dt.tz_localize(None)
        ptf["ts_hour"] = ts.dt.floor("h")
        ptf = ptf.rename(columns={"price": "ptf"})[["ts_hour", "ptf"]].drop_duplicates("ts_hour")
        out = out.merge(ptf, on="ts_hour", how="left")
        out["dam_idm_spread"] = out["idm_price"] - out["ptf"]
        out["dam_idm_spread_lag_24"] = out["dam_idm_spread"].shift(24)
        out["dam_idm_spread_lag_168"] = out["dam_idm_spread"].shift(168)
        out = out.drop(columns=["ptf"])

    return out


def write_report(df: pd.DataFrame) -> None:
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "source": "EPİAŞ Şeffaflık Platformu electricity-service IDM VWAP",
        "endpoint": ENDPOINT_URL,
        "rows": int(len(df)),
        "ts_start": str(df["ts_hour"].min()),
        "ts_end": str(df["ts_hour"].max()),
        "idm_price_mean": float(df["idm_price"].mean()),
        "idm_price_min": float(df["idm_price"].min()),
        "idm_price_max": float(df["idm_price"].max()),
        "missing_pct": {
            c: float(df[c].isna().mean() * 100)
            for c in df.columns
            if c != "ts_hour"
        },
    }
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    spread_note = ""
    if "dam_idm_spread" in df.columns:
        spread_note = f"\n- `dam_idm_spread` mean: `{df['dam_idm_spread'].mean():.1f}` TL/MWh"
    lines = [
        "# IDM Prices Report",
        "",
        f"Generated: `{report['generated_at']}`",
        f"Endpoint: `{ENDPOINT_URL}`",
        "",
        f"- Rows: `{report['rows']}`",
        f"- Range: `{report['ts_start']}` → `{report['ts_end']}`",
        f"- Mean GİP price: `{report['idm_price_mean']:.1f}` TL/MWh",
        f"- Min / Max: `{report['idm_price_min']:.1f}` / `{report['idm_price_max']:.1f}` TL/MWh",
        spread_note,
        "",
        "## Columns",
        "",
        "- `idm_price`: GİP saatlik ağırlıklı ortalama fiyat (TL/MWh)",
        "- `idm_price_lag_24/48/168`: gecikmeli GİP fiyatı",
        "- `idm_price_roll_mean_24/168`: kısa/haftalık ortalama",
        "- `dam_idm_spread`: GÖP PTF − GİP farkı (piyasa kısa/uzun sinyali)",
        "- `dam_idm_spread_lag_24/168`: gecikmeli spread",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--full", action="store_true", help="Re-fetch from 2020-01-01.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not USERNAME or not PASSWORD:
        raise SystemExit("EPİAŞ credentials missing (EPIAS_USERNAME/EPIAS_PASSWORD) in .env")

    start = (
        datetime(2020, 1, 1) if args.full
        else (_incremental_start() if args.start_date is None
              else pd.Timestamp(args.start_date).to_pydatetime())
    )
    end = (
        _ptf_max_date() if args.end_date is None
        else pd.Timestamp(args.end_date).to_pydatetime()
    )
    # GİP is realized data; don't request beyond today
    end = min(end, datetime.now().replace(minute=0, second=0, microsecond=0))
    _log(f"Fetch range: {start.date()} → {end.date()}")

    tgt = obtain_tgt(USERNAME, PASSWORD)
    _log(f"TGT ok: {tgt[:12]}…")

    rows: list[pd.DataFrame] = []
    cur = start
    while cur <= end:
        cur_end = min(end, cur + timedelta(days=MAX_DAYS_PER_CHUNK - 1))
        try:
            chunk = fetch_chunk(cur, cur_end, tgt)
            if not chunk.empty:
                rows.append(chunk)
                _log(f"  chunk rows={len(chunk)}")
        except RuntimeError as exc:
            _log(f"  chunk error (skipping): {exc}")
        cur = cur_end + timedelta(days=1)
        time.sleep(SLEEP_SECONDS)

    if not rows:
        raise SystemExit("API'den hiç veri gelmedi.")

    new_data = pd.concat(rows, ignore_index=True)

    # Merge with existing
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_CSV.exists() and not args.full:
        old = pd.read_csv(RAW_CSV)
        old["ts_hour"] = pd.to_datetime(old["ts_hour"], errors="coerce")
        combined = pd.concat([old[["ts_hour", "idm_price"]], new_data[["ts_hour", "idm_price"]]], ignore_index=True)
        combined = combined.drop_duplicates("ts_hour", keep="last").sort_values("ts_hour").reset_index(drop=True)
    else:
        combined = new_data

    combined.to_csv(RAW_CSV, index=False)
    _log(f"Raw CSV: {RAW_CSV} rows={len(combined)}")

    processed = add_lag_features(combined)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    processed.to_parquet(PROCESSED_PATH, index=False)
    _log(f"Processed parquet: {PROCESSED_PATH} rows={len(processed)}")

    write_report(processed)
    _log(f"Report: {REPORT_MD}")


if __name__ == "__main__":
    main()
