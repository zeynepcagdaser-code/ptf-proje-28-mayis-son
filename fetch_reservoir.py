#!/usr/bin/env python3
"""Fetch EPİAŞ dam active fullness percentages and aggregate to national daily index.

Endpoint (electricity-service):
  POST /v1/dams/data/active-fullness
  Web UI: https://seffaflik.epias.com.tr/electricity/dams/active-fullness

Response: one row per dam per day (84 dams × days).
  Fields: date, basin, dam, activeFullnessAmount (%)

Processing:
  - Aggregate all dams to daily national mean/min/max fullness %
  - Forward-fill to hourly project time axis
  - Derive: 7d/30d rolling mean, deviation from seasonal baseline, low/high regime flags

Outputs:
  - Raw CSV:  data/external/epias_market/reservoir/dam_fullness_raw.csv
  - Daily parquet: data/processed/dam_fullness_daily.parquet
  - Hourly parquet: data/processed/dam_fullness_hourly.parquet
  - Report: reports/dam_fullness_report.{json,md}
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
    "/v1/dams/data/active-fullness"
)

RAW_DIR = PROJECT_ROOT / "data" / "external" / "epias_market" / "reservoir"
RAW_CSV = RAW_DIR / "dam_fullness_raw.csv"
DAILY_PATH = PROJECT_ROOT / "data" / "processed" / "dam_fullness_daily.parquet"
HOURLY_PATH = PROJECT_ROOT / "data" / "processed" / "dam_fullness_hourly.parquet"
REPORT_JSON = PROJECT_ROOT / "reports" / "dam_fullness_report.json"
REPORT_MD = PROJECT_ROOT / "reports" / "dam_fullness_report.md"

REQUEST_TIMEOUT = (10, 120)
MAX_RETRIES = 4
CHUNK_DAYS = 90
SLEEP_SECONDS = 1.2
ROLLING_REFRESH_DAYS = 14


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
        if not old.empty and "date" in old.columns:
            ts = pd.to_datetime(old["date"], errors="coerce", utc=True)
            last = ts.dropna().max()
            if not pd.isna(last):
                return (last.tz_localize(None) - timedelta(days=ROLLING_REFRESH_DAYS)).to_pydatetime()
    return datetime(2020, 1, 1)


def fetch_chunk(start: datetime, end: datetime, tgt: str) -> pd.DataFrame:
    headers = {"Accept": "application/json", "Content-Type": "application/json", "TGT": tgt}
    payload = {
        "startDate": start.strftime("%Y-%m-%dT00:00:00+03:00"),
        "endDate": end.strftime("%Y-%m-%dT23:00:00+03:00"),
    }
    _log(f"Fetch: {payload['startDate']} → {payload['endDate']}")
    resp = post_with_retries(ENDPOINT_URL, json=payload, headers=headers)
    _log(f"HTTP {resp.status_code} ({len(resp.content)} bytes)")
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:2000]}")
    data = resp.json()
    items = data.get("items", data.get("body", {}).get("items", []))
    if not items:
        return pd.DataFrame()
    df = pd.DataFrame(items)
    _log(f"  kolonlar: {list(df.columns)}  rows={len(df)}")
    return df


def aggregate_daily(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-dam rows to one national row per day."""
    df = raw.copy()

    # Parse date
    ts = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None).dt.floor("D")
    df["ts_day"] = ts

    val_col = next(
        (c for c in ["activeFullnessAmount", "fullness", "dolulukOrani", "value"] if c in df.columns),
        None,
    )
    if val_col is None:
        raise RuntimeError(f"Doluluk kolonu bulunamadı. Kolonlar: {list(df.columns)}")

    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.dropna(subset=["ts_day", val_col])

    agg = df.groupby("ts_day")[val_col].agg(
        dam_fullness_mean="mean",
        dam_fullness_min="min",
        dam_fullness_max="max",
        dam_count="count",
    ).reset_index()

    # Basin-level breakdown if available
    if "basin" in df.columns:
        basins = df.groupby(["ts_day", "basin"])[val_col].mean().unstack("basin")
        basins.columns = [f"dam_fullness_{b.lower().replace(' ', '_')}" for b in basins.columns]
        agg = agg.merge(basins.reset_index(), on="ts_day", how="left")

    return agg.sort_values("ts_day").reset_index(drop=True)


def add_derived_features(daily: pd.DataFrame) -> pd.DataFrame:
    out = daily.copy().sort_values("ts_day").reset_index(drop=True)
    m = out["dam_fullness_mean"]
    out["dam_fullness_change_7d"] = m - m.shift(7)
    out["dam_fullness_change_30d"] = m - m.shift(30)
    out["dam_fullness_roll_7d"] = m.rolling(7, min_periods=3).mean()
    out["dam_fullness_roll_30d"] = m.rolling(30, min_periods=10).mean()
    out["dam_fullness_roll_90d"] = m.rolling(90, min_periods=21).mean()
    # Seasonal deviation: how full vs 90-day baseline
    out["dam_fullness_seasonal_dev"] = m - out["dam_fullness_roll_90d"]
    # Low / high hydro regime flags (below 40% or above 70%)
    out["dam_low_hydro_flag"] = (m < 40).astype(float)
    out["dam_high_hydro_flag"] = (m > 70).astype(float)
    return out


def align_to_hourly(daily: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    spine = pd.DataFrame({
        "ts_hour": pd.date_range(
            pd.Timestamp(start).floor("h"),
            pd.Timestamp(end).floor("h"),
            freq="h",
        )
    })
    d = daily.copy()
    d["ts_day"] = pd.to_datetime(d["ts_day"]).dt.floor("D")
    hourly = spine.copy()
    hourly["ts_day"] = hourly["ts_hour"].dt.floor("D")
    hourly = hourly.merge(d, on="ts_day", how="left").sort_values("ts_hour")
    value_cols = [c for c in d.columns if c != "ts_day"]
    hourly[value_cols] = hourly[value_cols].ffill()
    return hourly.drop(columns=["ts_day"]).reset_index(drop=True)


def write_report(daily: pd.DataFrame, hourly: pd.DataFrame) -> None:
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "source": "EPİAŞ electricity-service /v1/dams/data/active-fullness",
        "endpoint": ENDPOINT_URL,
        "daily_rows": int(len(daily)),
        "hourly_rows": int(len(hourly)),
        "daily_start": str(daily["ts_day"].min()),
        "daily_end": str(daily["ts_day"].max()),
        "dam_fullness_mean_overall": float(daily["dam_fullness_mean"].mean()),
        "dam_fullness_min_ever": float(daily["dam_fullness_mean"].min()),
        "dam_fullness_max_ever": float(daily["dam_fullness_mean"].max()),
        "missing_hourly_pct": {
            c: float(hourly[c].isna().mean() * 100)
            for c in ["dam_fullness_mean", "dam_fullness_roll_30d", "dam_fullness_seasonal_dev"]
            if c in hourly.columns
        },
    }
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# Dam Fullness Report",
        "",
        f"Generated: `{report['generated_at']}`",
        f"Endpoint: `{ENDPOINT_URL}`",
        "",
        f"- Daily rows: `{report['daily_rows']}`",
        f"- Hourly rows: `{report['hourly_rows']}`",
        f"- Daily range: `{report['daily_start']}` → `{report['daily_end']}`",
        "",
        "## Stats (national mean)",
        "",
        f"- Overall mean: `{report['dam_fullness_mean_overall']:.1f}%`",
        f"- Min / Max: `{report['dam_fullness_min_ever']:.1f}%` / `{report['dam_fullness_max_ever']:.1f}%`",
        "",
        "## Columns",
        "",
        "- `dam_fullness_mean`: national mean active fullness % (all dams, forward-filled hourly)",
        "- `dam_fullness_min/max`: cross-dam spread for that day",
        "- `dam_fullness_change_7d/30d`: weekly/monthly trend",
        "- `dam_fullness_roll_7d/30d/90d`: rolling averages",
        "- `dam_fullness_seasonal_dev`: deviation from 90d baseline (seasonal signal)",
        "- `dam_low_hydro_flag`: 1 when national mean < 40% (hydro-scarce regime)",
        "- `dam_high_hydro_flag`: 1 when national mean > 70% (hydro-abundant, low PTF pressure)",
        "- `dam_fullness_<basin>`: per-basin averages (if available)",
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
    end = _ptf_max_date() if args.end_date is None else pd.Timestamp(args.end_date).to_pydatetime()
    _log(f"Fetch range: {start.date()} → {end.date()}")

    tgt = obtain_tgt(USERNAME, PASSWORD)
    _log(f"TGT ok: {tgt[:12]}…")

    raw_chunks: list[pd.DataFrame] = []
    cur = start
    while cur <= end:
        cur_end = min(end, cur + timedelta(days=CHUNK_DAYS - 1))
        chunk = fetch_chunk(cur, cur_end, tgt)
        if not chunk.empty:
            raw_chunks.append(chunk)
            _log(f"  chunk rows={len(chunk)}")
        cur = cur_end + timedelta(days=1)
        time.sleep(SLEEP_SECONDS)

    if not raw_chunks:
        raise SystemExit("API'den hiç veri gelmedi.")

    new_raw = pd.concat(raw_chunks, ignore_index=True)

    # Merge with existing raw CSV
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_CSV.exists() and not args.full:
        old_raw = pd.read_csv(RAW_CSV)
        combined_raw = pd.concat([old_raw, new_raw], ignore_index=True)
        # Deduplicate on date + dam
        key_cols = [c for c in ["date", "dam", "damId"] if c in combined_raw.columns]
        combined_raw = combined_raw.drop_duplicates(subset=key_cols, keep="last")
    else:
        combined_raw = new_raw

    combined_raw.to_csv(RAW_CSV, index=False)
    _log(f"Raw CSV: {RAW_CSV} rows={len(combined_raw)}")

    daily = add_derived_features(aggregate_daily(combined_raw))
    DAILY_PATH.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(DAILY_PATH, index=False)
    _log(f"Daily parquet: {DAILY_PATH} rows={len(daily)}")

    hourly = align_to_hourly(daily, datetime(2020, 1, 1), end)
    hourly.to_parquet(HOURLY_PATH, index=False)
    _log(f"Hourly parquet: {HOURLY_PATH} rows={len(hourly)}")

    write_report(daily, hourly)
    _log(f"Report: {REPORT_MD}")


if __name__ == "__main__":
    main()
