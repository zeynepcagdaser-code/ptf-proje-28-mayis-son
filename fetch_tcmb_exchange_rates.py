#!/usr/bin/env python3
"""Fetch TCMB EVDS USD/TRY and EUR/TRY rates and align them to hourly project time.

Output files:
- data/processed/tcmb_exchange_rates_daily.parquet
- data/processed/tcmb_exchange_rates_hourly.parquet
- reports/tcmb_exchange_rates_report.json
- reports/tcmb_exchange_rates_report.md

Requires one of these environment variables in .env:
- EVDS_API_KEY
- TCMB_EVDS_API_KEY
- EVDS_KEY
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

PTF_PATH = DATA_DIR / "ptf_dataset.csv"
DAILY_OUT = PROCESSED_DIR / "tcmb_exchange_rates_daily.parquet"
HOURLY_OUT = PROCESSED_DIR / "tcmb_exchange_rates_hourly.parquet"
REPORT_JSON = REPORTS_DIR / "tcmb_exchange_rates_report.json"
REPORT_MD = REPORTS_DIR / "tcmb_exchange_rates_report.md"

EVDS_URL = "https://evds3.tcmb.gov.tr/igmevdsms-dis/"
SERIES = {
    "TP_DK_USD_A_YTL": "usd_try_buy",
    "TP_DK_EUR_A_YTL": "eur_try_buy",
}


def parse_ts(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce")
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert("Europe/Istanbul").dt.tz_localize(None)
    return ts.dt.floor("h")


def evds_key() -> str:
    load_dotenv(PROJECT_ROOT / ".env")
    key = os.getenv("EVDS_API_KEY") or os.getenv("TCMB_EVDS_API_KEY") or os.getenv("EVDS_KEY")
    if not key:
        raise RuntimeError(
            "EVDS API anahtarı bulunamadı. .env içine EVDS_API_KEY=... ekleyin. "
            "Anahtar TCMB EVDS profilinden alınır: https://evds2.tcmb.gov.tr/"
        )
    return key


def project_hourly_spine(start: str) -> pd.DataFrame:
    if PTF_PATH.exists():
        ptf = pd.read_csv(PTF_PATH, usecols=["date"])
        ts = parse_ts(ptf["date"])
        end = ts.max()
    else:
        end = pd.Timestamp.now(tz="Europe/Istanbul").tz_localize(None).floor("h")
    start_ts = pd.Timestamp(start)
    return pd.DataFrame({"ts_hour": pd.date_range(start_ts, end, freq="h")})


def fetch_evds_daily_chunk(start: str, end: str, key: str) -> pd.DataFrame:
    start_date = pd.Timestamp(start).strftime("%d-%m-%Y")
    end_date = pd.Timestamp(end).strftime("%d-%m-%Y")
    series = "-".join(SERIES.keys()).replace("_", ".")
    url = (
        f"{EVDS_URL}series={series}"
        f"&startDate={start_date}"
        f"&endDate={end_date}"
        "&type=json"
    )
    resp = requests.get(url, headers={"key": key}, timeout=(10, 120))
    if resp.status_code != 200:
        raise RuntimeError(f"EVDS status={resp.status_code}: {resp.text[:1000]}")
    try:
        payload = resp.json()
    except ValueError as exc:
        raise RuntimeError(f"EVDS JSON dönmedi: {resp.text[:1000]}") from exc
    items = payload.get("items", [])
    if not items:
        raise RuntimeError(f"EVDS boş döndü: {payload}")
    df = pd.DataFrame(items)
    if "Tarih" not in df.columns:
        raise RuntimeError(f"EVDS yanıtında Tarih kolonu yok: {df.columns.tolist()}")

    out = pd.DataFrame({"ts_day": pd.to_datetime(df["Tarih"], dayfirst=True, errors="coerce")})
    for evds_col, name in SERIES.items():
        raw_name = evds_col.replace("_", ".")
        candidates = [raw_name, evds_col]
        src = next((c for c in candidates if c in df.columns), None)
        if src is None:
            raise RuntimeError(f"EVDS yanıtında {raw_name} kolonu yok: {df.columns.tolist()}")
        out[name] = pd.to_numeric(df[src], errors="coerce")

    out = out.dropna(subset=["ts_day"]).sort_values("ts_day").drop_duplicates("ts_day", keep="last")
    return out


def fetch_evds_daily(start: str, end: str, key: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    frames: list[pd.DataFrame] = []
    chunk_start = start_ts
    while chunk_start <= end_ts:
        chunk_end = min(chunk_start + relativedelta(months=11, days=27), end_ts)
        frames.append(fetch_evds_daily_chunk(str(chunk_start.date()), str(chunk_end.date()), key))
        chunk_start = chunk_end + pd.Timedelta(days=1)
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["ts_day"]).sort_values("ts_day").drop_duplicates("ts_day", keep="last")
    for col in ["usd_try_buy", "eur_try_buy"]:
        out[col] = out[col].ffill()
    out["eur_usd_cross_buy"] = out["eur_try_buy"] / out["usd_try_buy"]
    out["usd_try_buy_lag_1d"] = out["usd_try_buy"].shift(1)
    out["eur_try_buy_lag_1d"] = out["eur_try_buy"].shift(1)
    out["usd_try_buy_change_7d"] = out["usd_try_buy"] - out["usd_try_buy"].shift(7)
    out["eur_try_buy_change_7d"] = out["eur_try_buy"] - out["eur_try_buy"].shift(7)
    out["usd_try_buy_pct_change_7d"] = out["usd_try_buy"].pct_change(7)
    out["eur_try_buy_pct_change_7d"] = out["eur_try_buy"].pct_change(7)
    out["usd_try_buy_roll_mean_7d"] = out["usd_try_buy"].rolling(7, min_periods=3).mean()
    out["eur_try_buy_roll_mean_7d"] = out["eur_try_buy"].rolling(7, min_periods=3).mean()
    return out


def align_daily_to_hourly(daily: pd.DataFrame, spine: pd.DataFrame) -> pd.DataFrame:
    out = spine.copy()
    out["ts_day"] = out["ts_hour"].dt.floor("D")
    daily_norm = daily.copy()
    daily_norm["ts_day"] = pd.to_datetime(daily_norm["ts_day"]).dt.floor("D")
    out = out.merge(daily_norm, on="ts_day", how="left").sort_values("ts_hour")

    value_cols = [c for c in daily_norm.columns if c != "ts_day"]
    out[value_cols] = out[value_cols].ffill()
    return out.drop(columns=["ts_day"])


def write_report(daily: pd.DataFrame, hourly: pd.DataFrame) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "TCMB EVDS",
        "series": SERIES,
        "daily_output": str(DAILY_OUT),
        "hourly_output": str(HOURLY_OUT),
        "daily_rows": int(len(daily)),
        "hourly_rows": int(len(hourly)),
        "daily_start": str(daily["ts_day"].min()),
        "daily_end": str(daily["ts_day"].max()),
        "hourly_start": str(hourly["ts_hour"].min()),
        "hourly_end": str(hourly["ts_hour"].max()),
        "missing_hourly_pct": {
            c: float(hourly[c].isna().mean() * 100)
            for c in hourly.columns
            if c != "ts_hour"
        },
    }
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# TCMB Exchange Rates Report",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        f"- Daily rows: `{report['daily_rows']}`",
        f"- Hourly rows: `{report['hourly_rows']}`",
        f"- Daily range: `{report['daily_start']}` → `{report['daily_end']}`",
        f"- Hourly range: `{report['hourly_start']}` → `{report['hourly_end']}`",
        "",
        "## Columns",
        "",
        "- `usd_try_buy`: TCMB USD/TRY döviz alış kuru",
        "- `eur_try_buy`: TCMB EUR/TRY döviz alış kuru",
        "- `eur_usd_cross_buy`: EUR/TRY divided by USD/TRY",
        "- lag/change/rolling columns: short-term FX trend features",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default=None, help="Default: max PTF date or today if PTF is missing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spine = project_hourly_spine(args.start_date)
    end = args.end_date or str(spine["ts_hour"].max().date())
    key = evds_key()
    daily = fetch_evds_daily(args.start_date, end, key)
    hourly = align_daily_to_hourly(daily, spine)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(DAILY_OUT, index=False)
    hourly.to_parquet(HOURLY_OUT, index=False)
    write_report(daily, hourly)

    print(f"Wrote {DAILY_OUT} rows={len(daily)}")
    print(f"Wrote {HOURLY_OUT} rows={len(hourly)}")
    print(f"Wrote {REPORT_MD}")


if __name__ == "__main__":
    main()
