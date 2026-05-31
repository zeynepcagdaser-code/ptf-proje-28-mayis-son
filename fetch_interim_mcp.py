#!/usr/bin/env python3
"""
Fetch Kesinleşmemiş PTF (K.PTF) from the EPİAŞ interim-mcp endpoint.
Does not modify data/ptf_dataset.csv.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

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
STATE_PATH = PROJECT_ROOT / "data" / "raw" / "interim_mcp_fetch_state.json"
LOCK_PATH = PROJECT_ROOT / "data" / "raw" / "interim_mcp_fetch.lock"

REQUEST_TIMEOUT = (10, 120)
MAX_NETWORK_RETRIES = 4
MAX_HTTP_RETRIES = 8
MIN_REQUEST_SLEEP_SECONDS = 2.15
ROLLING_REFRESH_HOURS = 48
DEFAULT_START = datetime(2020, 1, 1)
FORWARD_LOOK_DAYS = 7
KEEP_COLUMNS = ["date", "hour", "marketTradePrice"]


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def acquire_lock() -> None:
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    if LOCK_PATH.exists():
        try:
            lock_data = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
            lock_pid = int(lock_data.get("pid", -1))
        except Exception:
            lock_pid = -1

        if lock_pid > 0 and process_is_alive(lock_pid):
            raise RuntimeError(f"Başka bir interim fetch çalışıyor olabilir: pid={lock_pid}")

        log(f"Stale lock temizleniyor: {LOCK_PATH}")
        LOCK_PATH.unlink(missing_ok=True)

    lock_payload = {"pid": os.getpid(), "created_at": datetime.now().isoformat()}
    try:
        fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(lock_payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except FileExistsError as exc:
        raise RuntimeError(f"Lock alınamadı: {LOCK_PATH}") from exc


def release_lock() -> None:
    try:
        if LOCK_PATH.exists():
            lock_data = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
            if int(lock_data.get("pid", -1)) == os.getpid():
                LOCK_PATH.unlink()
    except Exception as exc:
        log(f"Lock temizlenemedi: {exc}")


def post_with_network_retries(url: str, **kwargs) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(1, MAX_NETWORK_RETRIES + 1):
        try:
            return requests.post(url, timeout=REQUEST_TIMEOUT, **kwargs)
        except requests.exceptions.Timeout as exc:
            last_error = exc
            wait_s = min(120, 10 * 2 ** (attempt - 1))
            log(f"timeout retry={attempt}/{MAX_NETWORK_RETRIES} wait={wait_s}s")
        except requests.exceptions.RequestException as exc:
            last_error = exc
            wait_s = min(120, 10 * 2 ** (attempt - 1))
            log(f"request error retry={attempt}/{MAX_NETWORK_RETRIES}: {exc} wait={wait_s}s")

        if attempt < MAX_NETWORK_RETRIES:
            time.sleep(wait_s)

    raise last_error  # type: ignore[misc]


def obtain_tgt(username: str, password: str) -> str:
    response = post_with_network_retries(
        LOGIN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "text/plain"},
        data={"username": username, "password": password},
    )
    log(f"TGT status={response.status_code}")
    tgt_text = response.text.strip()
    if tgt_text.startswith("TGT-"):
        return tgt_text
    match = re.search(r"/cas/v1/tickets/([^\" ]+)", tgt_text)
    if not match:
        raise RuntimeError(f"TGT alınamadı: {tgt_text[:500]}")
    return match.group(1)


def normalize_output(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=KEEP_COLUMNS)

    missing = [column for column in KEEP_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"API yanıtında eksik kolonlar: {missing}; mevcut: {list(df.columns)}")

    out = df[KEEP_COLUMNS].copy()
    out = out.dropna(subset=["date", "hour"])
    out = out.drop_duplicates(subset=["date", "hour"], keep="last")
    out = out.sort_values(["date", "hour"]).reset_index(drop=True)
    return out


def load_existing_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=KEEP_COLUMNS)
    df = pd.read_csv(path)
    return normalize_output(df)


def get_start_date_from_csv(df: pd.DataFrame) -> datetime:
    if df.empty or "date" not in df.columns:
        return DEFAULT_START
    parsed = pd.to_datetime(df["date"], errors="coerce", utc=True)
    last_date = parsed.max()
    if pd.isna(last_date):
        return DEFAULT_START
    log(f"Son kayıt: {last_date}")
    local_last = last_date.tz_convert("Europe/Istanbul").to_pydatetime().replace(tzinfo=None)
    return local_last - timedelta(hours=ROLLING_REFRESH_HOURS)


def fetch_day(headers: dict[str, str], day_start: datetime) -> tuple[list[dict], dict[str, Any]]:
    day_end = day_start + timedelta(days=1)
    payload = {
        "startDate": day_start.strftime("%Y-%m-%dT00:00:00+03:00"),
        "endDate": day_end.strftime("%Y-%m-%dT00:00:00+03:00"),
    }
    stats: dict[str, Any] = {"http_retries": 0, "rate_limits": 0, "timeouts": 0}

    for attempt in range(1, MAX_HTTP_RETRIES + 1):
        try:
            response = post_with_network_retries(INTERIM_MCP_URL, json=payload, headers=headers)
        except requests.exceptions.Timeout:
            stats["timeouts"] += 1
            raise

        if response.status_code == 200:
            return response.json().get("items", []), stats

        stats["http_retries"] += 1

        if response.status_code == 429:
            stats["rate_limits"] += 1
            wait_s = min(300, 65 * attempt)
            log(f"429 rate_limit day={day_start.date()} retry={attempt}/{MAX_HTTP_RETRIES} wait={wait_s}s")
            time.sleep(wait_s)
            continue

        if response.status_code in {500, 502, 503, 504}:
            wait_s = min(240, 15 * 2 ** (attempt - 1))
            log(f"http {response.status_code} day={day_start.date()} retry={attempt}/{MAX_HTTP_RETRIES} wait={wait_s}s")
            time.sleep(wait_s)
            continue

        text = response.text[:1500]
        if response.status_code == 400 and "geçmiş zaman olmalıdır" in text:
            raise StopIteration(f"Endpoint ileri endDate kabul etmedi: {payload['endDate']}")

        raise RuntimeError(f"{response.status_code}: {text}")

    raise RuntimeError(f"HTTP retry limiti aşıldı: {payload['startDate']}")


def write_state(state: dict[str, Any]) -> None:
    atomic_write_text(STATE_PATH, json.dumps(state, ensure_ascii=False, indent=2, default=str) + "\n")


def main() -> None:
    username = os.getenv("EPIAS_USERNAME")
    password = os.getenv("EPIAS_PASSWORD")
    if not username or not password:
        print("EPIAS_USERNAME / EPIAS_PASSWORD .env içinde tanımlı olmalı.", file=sys.stderr)
        sys.exit(1)

    acquire_lock()
    try:
        tgt = obtain_tgt(username, password)
        log(f"TGT alındı: {tgt[:20]}...")
        headers = {"Accept": "application/json", "Content-Type": "application/json", "TGT": tgt}

        df = load_existing_csv(CSV_PATH)
        log(f"Mevcut interim satır sayısı: {len(df)}")

        start_date = get_start_date_from_csv(df).replace(minute=0, second=0, microsecond=0)
        end_date = (datetime.now() + timedelta(days=FORWARD_LOOK_DAYS)).replace(
            minute=0, second=0, microsecond=0
        )
        log(f"Fetch aralığı: {start_date} -> {end_date}")

        current_start = start_date
        total_new_rows = 0
        total_days = 0
        total_rate_limits = 0
        total_retries = 0

        while current_start < end_date:
            try:
                items, stats = fetch_day(headers, current_start)
            except StopIteration as exc:
                log(str(exc))
                break

            chunk = normalize_output(pd.DataFrame(items))
            if not chunk.empty:
                df = normalize_output(pd.concat([df, chunk], ignore_index=True))
                total_new_rows += len(chunk)

            total_days += 1
            total_rate_limits += int(stats["rate_limits"])
            total_retries += int(stats["http_retries"])

            atomic_write_csv(df, CSV_PATH)
            write_state(
                {
                    "last_completed_day": current_start.strftime("%Y-%m-%d"),
                    "rows": len(df),
                    "total_days_this_run": total_days,
                    "new_rows_this_run": total_new_rows,
                    "rate_limits_this_run": total_rate_limits,
                    "http_retries_this_run": total_retries,
                    "updated_at": datetime.now().isoformat(),
                }
            )

            log(
                "day=%s rows=%d total=%d retries=%d rate_limits=%d"
                % (
                    current_start.date(),
                    len(chunk),
                    len(df),
                    stats["http_retries"],
                    stats["rate_limits"],
                )
            )

            current_start += timedelta(days=1)
            time.sleep(MIN_REQUEST_SLEEP_SECONDS)

        log(
            "Bitti. days=%d new_rows=%d total_rows=%d retries=%d rate_limits=%d"
            % (total_days, total_new_rows, len(df), total_retries, total_rate_limits)
        )
        log(f"Kaydedildi: {CSV_PATH}")
    finally:
        release_lock()


if __name__ == "__main__":
    main()
