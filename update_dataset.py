#!/usr/bin/env python3
"""
Fetch finalized DAM PTF/MCP data and update data/ptf_dataset.csv.

Features:
- CAS/TGT auth via .env
- retry + backoff
- rolling 48h refresh
- forward look window
- duplicate-hour de-duplication
- coverage report output
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

USERNAME = os.getenv("EPIAS_USERNAME")
PASSWORD = os.getenv("EPIAS_PASSWORD")

LOGIN_URL = "https://giris.epias.com.tr/cas/v1/tickets"
PTF_URL = "https://seffaflik.epias.com.tr/electricity-service/v1/markets/dam/data/mcp"
INTERIM_MCP_URL = "https://seffaflik.epias.com.tr/electricity-service/v1/markets/dam/data/interim-mcp"
CURVE_PTF_URL = "https://seffaflik.epias.com.tr/electricity-service/v1/markets/dam/data/supply-demand-chart-ptf-data"

CSV_PATH = PROJECT_ROOT / "data" / "ptf_dataset.csv"
REPORT_MD = PROJECT_ROOT / "reports" / "ptf_dataset_update_coverage.md"
REPORT_JSON = PROJECT_ROOT / "reports" / "ptf_dataset_update_coverage.json"

REQUEST_TIMEOUT = (10, 120)
MAX_RETRIES = 4
ROLLING_REFRESH_HOURS = 48
FORWARD_LOOK_DAYS = 7
CHUNK_DAYS = 90
TARGET_MIN_MAX_TS = pd.Timestamp("2026-06-02 23:00:00")


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def post_with_retries(url: str, **kwargs) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(url, timeout=REQUEST_TIMEOUT, **kwargs)
            if resp.status_code == 429:
                wait_s = min(120, 5 * 2 ** (attempt - 1))
                log(f"rate-limit retry={attempt}/{MAX_RETRIES} wait={wait_s}s url={url}")
                if attempt < MAX_RETRIES:
                    time.sleep(wait_s)
                    continue
            return resp
        except requests.exceptions.RequestException as exc:
            last_error = exc
            wait_s = min(120, 5 * 2 ** (attempt - 1))
            log(f"network retry={attempt}/{MAX_RETRIES} wait={wait_s}s err={exc}")
            if attempt < MAX_RETRIES:
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


def parse_ts(df: pd.DataFrame) -> pd.Series:
    if "date" not in df.columns:
        return pd.Series(dtype="datetime64[ns]")
    ts = pd.to_datetime(df["date"], errors="coerce")
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_localize(None)
    if "hour" in df.columns:
        hour = df["hour"].astype(str).str.extract(r"(\d{1,2})")[0]
        hour_num = pd.to_numeric(hour, errors="coerce")
        ts = ts.dt.normalize() + pd.to_timedelta(hour_num.fillna(0), unit="h")
    return ts


def normalize_ptf_items(items: list[dict[str, Any]]) -> pd.DataFrame:
    if not items:
        return pd.DataFrame(columns=["date", "hour", "price", "priceUsd", "priceEur"])
    df = pd.DataFrame(items)
    if "date" not in df.columns:
        return pd.DataFrame(columns=["date", "hour", "price", "priceUsd", "priceEur"])
    if "hour" not in df.columns:
        df["hour"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%H:00")
    for col in ["price", "priceUsd", "priceEur"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["ts_hour"] = parse_ts(df)
    df = df.dropna(subset=["ts_hour"])
    if "price" not in df.columns and "mcpPrice" in df.columns:
        df["price"] = pd.to_numeric(df["mcpPrice"], errors="coerce")
    return df[["date", "hour", "price", "priceUsd", "priceEur", "ts_hour"]].copy()


def normalize_interim_items(items: list[dict[str, Any]]) -> pd.DataFrame:
    if not items:
        return pd.DataFrame(columns=["date", "hour", "price", "priceUsd", "priceEur", "ts_hour"])
    df = pd.DataFrame(items)
    if "date" not in df.columns:
        return pd.DataFrame(columns=["date", "hour", "price", "priceUsd", "priceEur", "ts_hour"])
    if "hour" not in df.columns:
        df["hour"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%H:00")
    df["price"] = pd.to_numeric(df.get("marketTradePrice"), errors="coerce")
    df["priceUsd"] = np.nan
    df["priceEur"] = np.nan
    df["ts_hour"] = parse_ts(df)
    df = df.dropna(subset=["ts_hour"])
    return df[["date", "hour", "price", "priceUsd", "priceEur", "ts_hour"]].copy()


def load_existing() -> pd.DataFrame:
    if not CSV_PATH.exists():
        return pd.DataFrame(columns=["date", "hour", "price", "priceUsd", "priceEur"])
    old = pd.read_csv(CSV_PATH)
    if old.empty:
        return old
    old["ts_hour"] = parse_ts(old)
    old = old.dropna(subset=["ts_hour"])
    return old


def get_start_date_from_csv(old: pd.DataFrame) -> datetime:
    if old.empty or "ts_hour" not in old.columns:
        return datetime(2020, 1, 1)
    last_ts = pd.to_datetime(old["ts_hour"], errors="coerce").max()
    if pd.isna(last_ts):
        return datetime(2020, 1, 1)
    log(f"Son kayıt: {last_ts}")
    return last_ts.to_pydatetime().replace(tzinfo=None) - timedelta(hours=ROLLING_REFRESH_HOURS)


def fetch_chunk(start_date: datetime, end_date: datetime, tgt: str) -> pd.DataFrame:
    headers = {"Accept": "application/json", "Content-Type": "application/json", "TGT": tgt}
    payload = {
        "startDate": start_date.strftime("%Y-%m-%dT00:00:00+03:00"),
        "endDate": end_date.strftime("%Y-%m-%dT23:00:00+03:00"),
    }
    log(f"Çekiliyor: {payload['startDate']} → {payload['endDate']}")
    resp = post_with_retries(PTF_URL, json=payload, headers=headers)
    log(f"Durum: {resp.status_code}")
    if resp.status_code != 200:
        raise RuntimeError(resp.text[:2000])
    data = resp.json()
    items = data.get("items", [])
    return normalize_ptf_items(items)


def fetch_interim_chunk(start_date: datetime, end_date: datetime, tgt: str) -> pd.DataFrame:
    headers = {"Accept": "application/json", "Content-Type": "application/json", "TGT": tgt}
    payload = {
        "startDate": start_date.strftime("%Y-%m-%dT00:00:00+03:00"),
        "endDate": end_date.strftime("%Y-%m-%dT23:00:00+03:00"),
    }
    log(f"Interim çekiliyor: {payload['startDate']} → {payload['endDate']}")
    resp = post_with_retries(INTERIM_MCP_URL, json=payload, headers=headers)
    log(f"Interim durum: {resp.status_code}")
    if resp.status_code != 200:
        raise RuntimeError(resp.text[:2000])
    data = resp.json()
    items = data.get("items", [])
    return normalize_interim_items(items)


def fetch_curve_ptf_hour(ts_hour: datetime, tgt: str) -> pd.DataFrame:
    headers = {"Accept": "application/json", "Content-Type": "application/json", "TGT": tgt}
    payload = {"date": ts_hour.strftime("%Y-%m-%dT%H:00:00+03:00")}
    log(f"Curve PTF çekiliyor: {payload['date']}")
    resp = post_with_retries(CURVE_PTF_URL, json=payload, headers=headers)
    log(f"Curve PTF durum: {resp.status_code}")
    if resp.status_code != 200:
        raise RuntimeError(resp.text[:2000])
    obj = resp.json()
    row = None
    if isinstance(obj, dict) and "body" in obj:
        body = obj.get("body") or {}
        if isinstance(body, dict):
            content = body.get("content") or {}
            if isinstance(content, dict):
                row = content.get("response")
    elif isinstance(obj, dict) and "items" in obj and obj["items"]:
        row = obj["items"][0]
    if not isinstance(row, dict):
        return pd.DataFrame(columns=["date", "hour", "price", "priceUsd", "priceEur", "ts_hour"])
    out = pd.DataFrame(
        [
            {
                "date": row.get("date"),
                "hour": pd.to_datetime(row.get("date"), errors="coerce").strftime("%H:00") if row.get("date") else ts_hour.strftime("%H:00"),
                "price": pd.to_numeric(pd.Series([row.get("mcpPrice")]), errors="coerce").iloc[0],
                "priceUsd": np.nan,
                "priceEur": np.nan,
            }
        ]
    )
    out["ts_hour"] = parse_ts(out)
    return out.dropna(subset=["ts_hour"])[["date", "hour", "price", "priceUsd", "priceEur", "ts_hour"]]


def dedupe_sort(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "ts_hour" not in df.columns:
        df["ts_hour"] = parse_ts(df)
    df = df.dropna(subset=["ts_hour"]).sort_values("ts_hour").drop_duplicates("ts_hour", keep="last")
    df["date"] = pd.to_datetime(df["ts_hour"]).dt.strftime("%Y-%m-%dT%H:%M:%S+03:00")
    if "hour" not in df.columns:
        df["hour"] = pd.to_datetime(df["ts_hour"]).dt.strftime("%H:00")
    return df.drop(columns=["ts_hour"], errors="ignore").reset_index(drop=True)


def coverage_report(df: pd.DataFrame, existing_rows: int, fetched_rows: int, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    ts = parse_ts(df) if not df.empty else pd.Series(dtype="datetime64[ns]")
    full_hours = pd.date_range(ts.min(), ts.max(), freq="h") if not ts.empty else pd.DatetimeIndex([])
    missing = []
    if not ts.empty:
        missing = [str(x) for x in full_hours.difference(pd.DatetimeIndex(ts)).tolist()]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "csv_path": str(CSV_PATH.relative_to(PROJECT_ROOT)),
        "existing_rows": int(existing_rows),
        "fetched_rows": int(fetched_rows),
        "supplemental_interim_rows": int(max(0, fetched_rows - sum(c["rows"] for c in chunks if c.get("source") != "interim-mcp"))),
        "final_rows": int(len(df)),
        "max_timestamp": str(ts.max()) if not ts.empty else None,
        "min_timestamp": str(ts.min()) if not ts.empty else None,
        "duplicate_hours": int(df["date"].duplicated().sum()) if "date" in df.columns else None,
        "missing_hours_count": int(len(missing)),
        "missing_hours": missing[:200],
        "price_non_null_ratio": float(df["price"].notna().mean()) if "price" in df.columns and not df.empty else None,
        "coverage_target": ">= 2026-06-02 23:00:00",
        "coverage_met": bool(ts.max() >= TARGET_MIN_MAX_TS) if not ts.empty else False,
        "chunks": chunks,
    }


def main() -> None:
    if not USERNAME or not PASSWORD:
        raise SystemExit("Missing EPIAS credentials in .env")

    existing = load_existing()
    existing_rows = len(existing)
    start_date = get_start_date_from_csv(existing)
    start_date = start_date.replace(minute=0, second=0, microsecond=0)
    end_date = datetime.now().replace(tzinfo=None) + timedelta(days=FORWARD_LOOK_DAYS)

    tgt = obtain_tgt(USERNAME, PASSWORD)
    log(f"TGT alındı: {tgt[:20]}...")

    all_chunks: list[pd.DataFrame] = []
    chunk_meta: list[dict[str, Any]] = []
    current_start = start_date
    while current_start <= end_date:
        current_end = min(current_start + timedelta(days=CHUNK_DAYS - 1), end_date)
        chunk = fetch_chunk(current_start, current_end, tgt)
        chunk_meta.append(
            {
                "start": current_start.strftime("%Y-%m-%d"),
                "end": current_end.strftime("%Y-%m-%d"),
                "rows": int(len(chunk)),
            }
        )
        all_chunks.append(chunk)
        current_start = current_end + timedelta(days=1)
        time.sleep(1)

    new_df = pd.concat(all_chunks, ignore_index=True) if all_chunks else pd.DataFrame()
    fetched_rows = len(new_df)
    df = pd.concat([existing.drop(columns=["ts_hour"], errors="ignore"), new_df.drop(columns=["ts_hour"], errors="ignore")], ignore_index=True)
    df = dedupe_sort(df)

    supplemental_rows = 0
    max_ts = pd.to_datetime(parse_ts(df), errors="coerce").max() if not df.empty else pd.NaT
    if pd.notna(max_ts) and max_ts < TARGET_MIN_MAX_TS:
        missing_start = (max_ts + timedelta(days=1)).to_pydatetime().replace(tzinfo=None)
        missing_end = TARGET_MIN_MAX_TS.to_pydatetime().replace(tzinfo=None)
        interim_chunk = fetch_interim_chunk(missing_start, missing_end, tgt)
        supplemental_rows = len(interim_chunk)
        if not interim_chunk.empty:
            chunk_meta.append(
                {
                    "source": "interim-mcp",
                    "start": missing_start.strftime("%Y-%m-%d"),
                    "end": missing_end.strftime("%Y-%m-%d"),
                    "rows": int(len(interim_chunk)),
                }
            )
            df = pd.concat([df, interim_chunk.drop(columns=["ts_hour"], errors="ignore")], ignore_index=True)
            df = dedupe_sort(df)
            max_ts = pd.to_datetime(parse_ts(df), errors="coerce").max() if not df.empty else pd.NaT
        if pd.notna(max_ts) and max_ts < TARGET_MIN_MAX_TS:
            curve_rows = []
            supplement_day = TARGET_MIN_MAX_TS.to_pydatetime().replace(hour=0, minute=0, second=0, microsecond=0)
            for hour in range(24):
                curve_rows.append(fetch_curve_ptf_hour(supplement_day + timedelta(hours=hour), tgt))
                time.sleep(0.5)
            curve_chunk = pd.concat(curve_rows, ignore_index=True) if curve_rows else pd.DataFrame()
            if not curve_chunk.empty:
                chunk_meta.append(
                    {
                        "source": "curve-ptf",
                        "start": supplement_day.strftime("%Y-%m-%d"),
                        "end": TARGET_MIN_MAX_TS.strftime("%Y-%m-%d"),
                        "rows": int(len(curve_chunk)),
                    }
                )
                supplemental_rows += len(curve_chunk)
                df = pd.concat([df, curve_chunk.drop(columns=["ts_hour"], errors="ignore")], ignore_index=True)
                df = dedupe_sort(df)

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = CSV_PATH.with_suffix(".csv.tmp")
    df.to_csv(tmp_csv, index=False)
    tmp_csv.replace(CSV_PATH)

    report = coverage_report(df, existing_rows, fetched_rows + supplemental_rows, chunk_meta)
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    lines = [
        "# PTF Dataset Update Coverage",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Existing rows: `{report['existing_rows']}`",
        f"- Fetched rows: `{report['fetched_rows']}`",
        f"- Final rows: `{report['final_rows']}`",
        f"- Min timestamp: `{report['min_timestamp']}`",
        f"- Max timestamp: `{report['max_timestamp']}`",
        f"- Coverage target: `{report['coverage_target']}`",
        f"- Coverage met: `{report['coverage_met']}`",
        f"- Duplicate hours: `{report['duplicate_hours']}`",
        f"- Missing hours count: `{report['missing_hours_count']}`",
        f"- Price non-null ratio: `{report['price_non_null_ratio']}`",
        "",
        "## Chunks",
        "",
    ]
    for chunk in report["chunks"]:
        lines.append(f"- `{chunk['start']}` → `{chunk['end']}`: `{chunk['rows']}` rows")
    if report["missing_hours_count"]:
        lines.extend(["", "## Missing Hours", ""])
        lines.extend([f"- `{x}`" for x in report["missing_hours"][:50]])
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    log(f"Eski satır: {existing_rows}")
    log(f"Yeni çekilen satır: {fetched_rows}")
    log(f"Toplam satır: {len(df)}")
    log(f"CSV kaydedildi: {CSV_PATH}")
    log(f"Coverage raporu yazıldı: {REPORT_MD}")


if __name__ == "__main__":
    main()
