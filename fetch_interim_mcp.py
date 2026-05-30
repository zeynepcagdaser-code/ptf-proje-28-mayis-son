#!/usr/bin/env python3
"""
Fetch Kesinleşmemiş PTF (K.PTF) from EPİAŞ interim-mcp endpoint.
Does not modify data/ptf_dataset.csv.
"""

from __future__ import annotations

import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

LOGIN_URL = "https://giris.epias.com.tr/cas/v1/tickets"
INTERIM_MCP_URL = (
    "https://seffaflik.epias.com.tr/electricity-service/v1/markets/dam/data/interim-mcp"
)
CSV_PATH = PROJECT_ROOT / "data" / "raw" / "interim_mcp.csv"
REQUEST_TIMEOUT = (10, 120)
MAX_RETRIES = 3
ROLLING_REFRESH_HOURS = 48
DEFAULT_START = datetime(2020, 1, 1)
FORWARD_LOOK_DAYS = 7
KEEP_COLUMNS = ["date", "hour", "marketTradePrice"]


def post_with_retries(url: str, **kwargs) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return requests.post(url, timeout=REQUEST_TIMEOUT, **kwargs)
        except requests.exceptions.RequestException as exc:
            last_error = exc
            print(f"İstek hatası ({attempt}/{MAX_RETRIES}): {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(attempt * 10)
    raise last_error  # type: ignore[misc]


def obtain_tgt(username: str, password: str) -> str:
    tgt_response = post_with_retries(
        LOGIN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "text/plain"},
        data={"username": username, "password": password},
    )
    print("TGT durum kodu:", tgt_response.status_code)
    tgt_text = tgt_response.text.strip()
    if tgt_text.startswith("TGT-"):
        return tgt_text
    match = re.search(r"/cas/v1/tickets/([^\" ]+)", tgt_text)
    if not match:
        raise RuntimeError(f"TGT alınamadı: {tgt_text[:500]}")
    return match.group(1)


def get_start_date_from_csv(csv_path: Path) -> datetime:
    if not csv_path.exists():
        return DEFAULT_START
    old_df = pd.read_csv(csv_path)
    if old_df.empty or "date" not in old_df.columns:
        return DEFAULT_START
    parsed = pd.to_datetime(old_df["date"], errors="coerce")
    last_date = parsed.max()
    if pd.isna(last_date):
        return DEFAULT_START
    print("Son kayıt:", last_date)
    return last_date.to_pydatetime().replace(tzinfo=None) - timedelta(hours=ROLLING_REFRESH_HOURS)


def fetch_range(headers: dict, start: datetime, end: datetime) -> list[dict]:
    payload = {
        "startDate": start.strftime("%Y-%m-%dT00:00:00+03:00"),
        "endDate": end.strftime("%Y-%m-%dT00:00:00+03:00"),
    }
    for attempt in range(1, 8):
        response = post_with_retries(INTERIM_MCP_URL, json=payload, headers=headers)
        if response.status_code == 200:
            return response.json().get("items", [])
        if response.status_code == 429:
            wait_s = 65 * attempt
            print(f"429 rate limit — {wait_s}s bekleniyor ({payload['startDate']})")
            time.sleep(wait_s)
            continue
        raise RuntimeError(f"{response.status_code}: {response.text[:1500]}")
    raise RuntimeError(f"Rate limit aşıldı: {payload['startDate']}")


def normalize_output(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in KEEP_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"API yanıtında eksik kolonlar: {missing}; mevcut: {list(df.columns)}")
    out = df[KEEP_COLUMNS].copy()
    if "date" in out.columns and "hour" in out.columns:
        out = out.drop_duplicates(subset=["date", "hour"], keep="last")
        out = out.sort_values(["date", "hour"])
    return out


def main() -> None:
    username = os.getenv("EPIAS_USERNAME")
    password = os.getenv("EPIAS_PASSWORD")
    if not username or not password:
        print("EPIAS_USERNAME / EPIAS_PASSWORD .env içinde tanımlı olmalı.", file=sys.stderr)
        sys.exit(1)

    tgt = obtain_tgt(username, password)
    print("TGT alındı:", tgt[:20] + "...")
    headers = {"Accept": "application/json", "Content-Type": "application/json", "TGT": tgt}

    old_df = pd.DataFrame()
    if CSV_PATH.exists():
        old_df = pd.read_csv(CSV_PATH)
        print("Eski interim satır sayısı:", len(old_df))

    start_date = get_start_date_from_csv(CSV_PATH).replace(minute=0, second=0, microsecond=0)
    end_date = datetime.now() + timedelta(days=FORWARD_LOOK_DAYS)

    # interim-mcp returns at most 24 rows (one calendar day) per request regardless of range.
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = old_df.copy()
    current_start = start_date
    day_count = 0
    new_rows = 0

    while current_start < end_date:
        current_end = current_start + timedelta(days=1)
        items = fetch_range(headers, current_start, current_end)
        if items:
            chunk = normalize_output(pd.DataFrame(items))
            df = pd.concat([df, chunk], ignore_index=True)
            df = normalize_output(df)
            new_rows += len(chunk)
        current_start = current_end
        day_count += 1
        if day_count % 30 == 0:
            df.to_csv(CSV_PATH, index=False)
            print(f"  checkpoint: {day_count} gün, toplam {len(df)} satır ({current_start.date()})")
        time.sleep(0.65)

    print("Yeni gelen satır:", new_rows)
    df.to_csv(CSV_PATH, index=False)
    print("Bitti. Toplam satır:", len(df))
    print("Kaydedildi:", CSV_PATH)


if __name__ == "__main__":
    main()
