#!/usr/bin/env python3
"""
Fetch EPİAŞ natural gas SGP Daily Reference Price (GRF/DRP).

Endpoint (natural-gas-service docs):
  POST /v1/markets/sgp/data/daily-reference-price

Outputs:
  - Raw CSV: data/external/epias_market/grf/grf_daily_reference_price_raw.csv
  - Processed parquet (daily): data/processed/grf_daily_reference_price.parquet

No model training.
"""

from __future__ import annotations

import os
import re
import time
import argparse
import json
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
ENDPOINT_URL = "https://seffaflik.epias.com.tr/natural-gas-service/v1/markets/sgp/data/daily-reference-price"

RAW_DIR = PROJECT_ROOT / "data" / "external" / "epias_market" / "grf"
RAW_CSV = RAW_DIR / "grf_daily_reference_price_raw.csv"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "grf_daily_reference_price.parquet"
HOURLY_PATH = PROJECT_ROOT / "data" / "processed" / "grf_hourly_reference_price.parquet"
REPORT_JSON = PROJECT_ROOT / "reports" / "grf_reference_price_report.json"
REPORT_MD = PROJECT_ROOT / "reports" / "grf_reference_price_report.md"

REQUEST_TIMEOUT = (10, 120)
MAX_RETRIES = 4
CHUNK_DAYS = 365
SLEEP_SECONDS = 1.5


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def post_with_retries(url: str, **kwargs) -> requests.Response:
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
            wait_s = min(120, 3 * 2 ** (attempt - 1))
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


def _master_date_range() -> tuple[datetime, datetime]:
    master_path = PROJECT_ROOT / "data" / "master" / "master_hourly_v1.parquet"
    if not master_path.exists():
        return datetime(2020, 1, 1), datetime.now()
    from src.utils.io_utils import read_parquet_with_normalized_ts
    ts = read_parquet_with_normalized_ts(master_path, columns=["ts_hour"])["ts_hour"]
    ts = pd.to_datetime(ts, errors="coerce")
    start = ts.min()
    end = ts.max()
    if pd.isna(start) or pd.isna(end):
        return datetime(2020, 1, 1), datetime.now()
    return start.to_pydatetime().replace(tzinfo=None), end.to_pydatetime().replace(tzinfo=None)


def _ptf_date_range(start_date: str | None = None, end_date: str | None = None) -> tuple[datetime, datetime]:
    if start_date:
        start = pd.Timestamp(start_date).to_pydatetime().replace(tzinfo=None)
    else:
        start = datetime(2020, 1, 1)

    if end_date:
        end = pd.Timestamp(end_date).to_pydatetime().replace(tzinfo=None)
    else:
        ptf_path = PROJECT_ROOT / "data" / "ptf_dataset.csv"
        if ptf_path.exists():
            ptf = pd.read_csv(ptf_path, usecols=["date"])
            ts = pd.to_datetime(ptf["date"], errors="coerce")
            if getattr(ts.dt, "tz", None) is not None:
                ts = ts.dt.tz_convert("Europe/Istanbul").dt.tz_localize(None)
            end = ts.max().to_pydatetime().replace(tzinfo=None)
        else:
            _, end = _master_date_range()
    return start, end


def _normalize_items(items: list[dict[str, Any]]) -> pd.DataFrame:
    if not items:
        return pd.DataFrame(columns=["date", "grfTl", "grfUsd", "grfEur", "grfUsdMmBtu"])
    df = pd.DataFrame(items)
    keep = [c for c in ["date", "grfTl", "grfUsd", "grfEur", "grfUsdMmBtu"] if c in df.columns]
    out = df[keep].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce", utc=True).dt.tz_convert("Europe/Istanbul").dt.floor("D")
    for c in ["grfTl", "grfUsd", "grfEur", "grfUsdMmBtu"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    return out


def fetch_chunk(start_date: datetime, end_date: datetime, tgt: str) -> pd.DataFrame:
    headers = {"Accept": "application/json", "Content-Type": "application/json", "TGT": tgt}
    payload = {
        "startDate": start_date.strftime("%Y-%m-%dT00:00:00+03:00"),
        "endDate": end_date.strftime("%Y-%m-%dT23:00:00+03:00"),
    }
    _log(f"Fetch: {payload['startDate']} → {payload['endDate']}")
    resp = post_with_retries(ENDPOINT_URL, json=payload, headers=headers)
    _log(f"HTTP {resp.status_code}")
    if resp.status_code != 200:
        raise RuntimeError(resp.text[:2000])
    data = resp.json()
    items = data.get("items", [])
    return _normalize_items(items)


def align_daily_to_hourly(processed: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    spine = pd.DataFrame({"ts_hour": pd.date_range(pd.Timestamp(start).floor("h"), pd.Timestamp(end).floor("h"), freq="h")})
    daily = processed.copy()
    daily["ts_day"] = pd.to_datetime(daily["ts_day"], errors="coerce")
    if getattr(daily["ts_day"].dt, "tz", None) is not None:
        daily["ts_day"] = daily["ts_day"].dt.tz_convert("Europe/Istanbul").dt.tz_localize(None)
    daily["ts_day"] = daily["ts_day"].dt.floor("D")

    hourly = spine.copy()
    hourly["ts_day"] = hourly["ts_hour"].dt.floor("D")
    hourly = hourly.merge(daily, on="ts_day", how="left").sort_values("ts_hour")
    value_cols = [c for c in daily.columns if c != "ts_day"]
    hourly[value_cols] = hourly[value_cols].ffill()

    # Cost trend features aligned to project hourly axis.
    hourly["grf_tl_lag_24"] = hourly["grf_tl_1000sm3"].shift(24)
    hourly["grf_tl_change_7d"] = hourly["grf_tl_1000sm3"] - hourly["grf_tl_1000sm3"].shift(24 * 7)
    hourly["grf_tl_pct_change_7d"] = hourly["grf_tl_1000sm3"].pct_change(24 * 7)
    hourly["grf_tl_roll_mean_7d"] = hourly["grf_tl_1000sm3"].rolling(24 * 7, min_periods=24 * 3).mean()
    hourly["grf_tl_roll_mean_30d"] = hourly["grf_tl_1000sm3"].rolling(24 * 30, min_periods=24 * 10).mean()
    hourly["grf_usd_mmbtu_lag_24"] = hourly["grf_usd_mmbtu"].shift(24)
    hourly["grf_usd_mmbtu_change_7d"] = hourly["grf_usd_mmbtu"] - hourly["grf_usd_mmbtu"].shift(24 * 7)
    return hourly.drop(columns=["ts_day"])


def write_report(processed: pd.DataFrame, hourly: pd.DataFrame) -> None:
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "source": "EPİAŞ Şeffaflık Platformu natural-gas-service daily-reference-price",
        "endpoint": ENDPOINT_URL,
        "daily_output": str(PROCESSED_PATH),
        "hourly_output": str(HOURLY_PATH),
        "daily_rows": int(len(processed)),
        "hourly_rows": int(len(hourly)),
        "daily_start": str(processed["ts_day"].min()),
        "daily_end": str(processed["ts_day"].max()),
        "hourly_start": str(hourly["ts_hour"].min()),
        "hourly_end": str(hourly["ts_hour"].max()),
        "columns": list(hourly.columns),
        "missing_hourly_pct": {
            c: float(hourly[c].isna().mean() * 100)
            for c in hourly.columns
            if c != "ts_hour"
        },
    }
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# GRF Reference Price Report",
        "",
        f"Generated: `{report['generated_at']}`",
        f"Endpoint: `{ENDPOINT_URL}`",
        "",
        f"- Daily rows: `{report['daily_rows']}`",
        f"- Hourly rows: `{report['hourly_rows']}`",
        f"- Daily range: `{report['daily_start']}` → `{report['daily_end']}`",
        f"- Hourly range: `{report['hourly_start']}` → `{report['hourly_end']}`",
        "",
        "## Columns",
        "",
        "- `grf_tl_1000sm3`: TL/1000Sm3 spot natural gas daily reference price",
        "- `grf_usd_1000sm3`: USD/1000Sm3 equivalent",
        "- `grf_eur_mwh`: EUR/MWh equivalent",
        "- `grf_usd_mmbtu`: USD/MMBtu equivalent",
        "- lag/change/rolling columns: short-term natural gas cost trend features",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not USERNAME or not PASSWORD:
        raise SystemExit("Missing EPIAS credentials (EPIAS_USERNAME/EPIAS_PASSWORD) in .env")

    start, end = _ptf_date_range(args.start_date, args.end_date)
    _log(f"Fetch range: {start.date()} → {end.date()}")

    tgt = obtain_tgt(USERNAME, PASSWORD)
    _log(f"TGT ok: {tgt[:12]}…")

    rows: list[pd.DataFrame] = []
    cur = start
    while cur <= end:
        cur_end = min(end, cur + timedelta(days=CHUNK_DAYS - 1))
        chunk = fetch_chunk(cur, cur_end, tgt)
        rows.append(chunk)
        cur = cur_end + timedelta(days=1)
        time.sleep(SLEEP_SECONDS)

    raw = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(RAW_CSV, index=False)

    processed = raw.rename(
        columns={
            "date": "ts_day",
            "grfTl": "grf_tl_1000sm3",
            "grfUsd": "grf_usd_1000sm3",
            "grfEur": "grf_eur_mwh",
            "grfUsdMmBtu": "grf_usd_mmbtu",
        }
    )
    keep = ["ts_day", "grf_tl_1000sm3", "grf_usd_1000sm3", "grf_eur_mwh", "grf_usd_mmbtu"]
    processed = processed[[c for c in keep if c in processed.columns]].copy()
    processed = processed.drop_duplicates(subset=["ts_day"], keep="last").sort_values("ts_day").reset_index(drop=True)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    from src.utils.safe_io import atomic_parquet_write
    atomic_parquet_write(processed, str(PROCESSED_PATH), index=False)
    hourly = align_daily_to_hourly(processed, start, end)
    atomic_parquet_write(hourly, str(HOURLY_PATH), index=False)
    write_report(processed, hourly)

    _log(f"Wrote raw: {RAW_CSV} rows={len(raw)}")
    _log(f"Wrote processed: {PROCESSED_PATH} rows={len(processed)}")
    _log(f"Wrote hourly: {HOURLY_PATH} rows={len(hourly)}")
    _log(f"Wrote report: {REPORT_MD}")
    _log(f"Endpoint: {ENDPOINT_URL}")


if __name__ == "__main__":
    main()
