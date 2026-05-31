#!/usr/bin/env python3
"""
Append-only point-in-time snapshots for EPİAŞ interim MCP (K.PTF).

This script intentionally does not perform historical backfill and does not join
with finalized MCP. It captures only the currently visible today/tomorrow
interim MCP response so future research can use leak-free snapshots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

LOGIN_URL = "https://giris.epias.com.tr/cas/v1/tickets"
INTERIM_MCP_URL = (
    "https://seffaflik.epias.com.tr/electricity-service/v1/markets/dam/data/interim-mcp"
)
PUBLISHED_STATUS_URL = (
    "https://seffaflik.epias.com.tr/electricity-service/v1/markets/dam/data/"
    "interim-mcp-published-status"
)

CSV_PATH = PROJECT_ROOT / "data" / "snapshots" / "interim_mcp_snapshots.csv"
LOCK_PATH = PROJECT_ROOT / "data" / "snapshots" / "interim_mcp_snapshots.lock"

REQUEST_TIMEOUT = (10, 120)
MAX_NETWORK_RETRIES = 4
MAX_HTTP_RETRIES = 6
MIN_REQUEST_SLEEP_SECONDS = 1.5
TIMEZONE_NAME = "Europe/Istanbul"

SNAPSHOT_COLUMNS = [
    "snapshot_ts",
    "fetch_run_id",
    "delivery_date",
    "delivery_hour",
    "marketTradePrice",
    "published_status_completed",
    "response_hash",
    "source_endpoint",
]


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def istanbul_now() -> datetime:
    if ZoneInfo is not None:
        return datetime.now(ZoneInfo(TIMEZONE_NAME))
    return datetime.now(timezone(timedelta(hours=3)))


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
            raise RuntimeError(f"Başka bir snapshot süreci çalışıyor olabilir: pid={lock_pid}")

        log(f"Stale snapshot lock temizleniyor: {LOCK_PATH}")
        LOCK_PATH.unlink(missing_ok=True)

    payload = {"pid": os.getpid(), "created_at": datetime.now().isoformat()}
    try:
        fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except FileExistsError as exc:
        raise RuntimeError(f"Snapshot lock alınamadı: {LOCK_PATH}") from exc


def release_lock() -> None:
    try:
        if LOCK_PATH.exists():
            lock_data = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
            if int(lock_data.get("pid", -1)) == os.getpid():
                LOCK_PATH.unlink()
    except Exception as exc:
        log(f"Snapshot lock temizlenemedi: {exc}")


def request_with_network_retries(method: str, url: str, **kwargs: Any) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(1, MAX_NETWORK_RETRIES + 1):
        try:
            return requests.request(method, url, timeout=REQUEST_TIMEOUT, **kwargs)
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
    response = request_with_network_retries(
        "POST",
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


def response_hash(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def get_published_status(headers: dict[str, str]) -> bool | None:
    for attempt in range(1, MAX_HTTP_RETRIES + 1):
        response = request_with_network_retries("GET", PUBLISHED_STATUS_URL, headers=headers)

        if response.status_code == 200:
            body = response.json()
            content = body.get("content") or body.get("body", {}).get("content", {})
            completed = content.get("completed")
            return bool(completed) if completed is not None else None

        if response.status_code == 429:
            wait_s = min(300, 45 * attempt)
            log(f"429 status retry={attempt}/{MAX_HTTP_RETRIES} wait={wait_s}s")
            time.sleep(wait_s)
            continue

        if response.status_code in {500, 502, 503, 504}:
            wait_s = min(240, 10 * 2 ** (attempt - 1))
            log(f"http {response.status_code} status retry={attempt}/{MAX_HTTP_RETRIES} wait={wait_s}s")
            time.sleep(wait_s)
            continue

        log(f"Published status alınamadı: {response.status_code} {response.text[:500]}")
        return None

    log("Published status retry limiti aşıldı.")
    return None


def fetch_delivery_day(headers: dict[str, str], delivery_day: datetime) -> tuple[list[dict[str, Any]], str]:
    day_start = delivery_day.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start + timedelta(days=1)
    payload = {
        "startDate": day_start.strftime("%Y-%m-%dT00:00:00+03:00"),
        "endDate": day_end.strftime("%Y-%m-%dT00:00:00+03:00"),
    }

    for attempt in range(1, MAX_HTTP_RETRIES + 1):
        response = request_with_network_retries(
            "POST", INTERIM_MCP_URL, json=payload, headers=headers
        )

        if response.status_code == 200:
            body = response.json()
            items = body.get("items") or body.get("body", {}).get("items", [])
            return items, response_hash({"payload": payload, "items": items})

        if response.status_code == 429:
            wait_s = min(300, 60 * attempt)
            log(
                "429 rate_limit delivery_date=%s retry=%d/%d wait=%ss"
                % (day_start.date(), attempt, MAX_HTTP_RETRIES, wait_s)
            )
            time.sleep(wait_s)
            continue

        if response.status_code in {500, 502, 503, 504}:
            wait_s = min(240, 15 * 2 ** (attempt - 1))
            log(
                "http %s delivery_date=%s retry=%d/%d wait=%ss"
                % (response.status_code, day_start.date(), attempt, MAX_HTTP_RETRIES, wait_s)
            )
            time.sleep(wait_s)
            continue

        raise RuntimeError(
            f"Interim MCP snapshot alınamadı: {response.status_code} {response.text[:1500]}"
        )

    raise RuntimeError(f"Interim MCP retry limiti aşıldı: {day_start.date()}")


def normalize_item_hour(item: dict[str, Any]) -> str:
    hour = item.get("hour")
    if hour is None:
        return ""
    parsed = pd.to_datetime(hour, errors="coerce")
    if pd.isna(parsed):
        return str(hour)
    return parsed.isoformat()


def rows_from_items(
    items: list[dict[str, Any]],
    *,
    snapshot_ts: str,
    fetch_run_id: str,
    completed: bool | None,
    digest: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        delivery_date = item.get("date")
        delivery_hour = normalize_item_hour(item)
        price = item.get("marketTradePrice")
        if delivery_date is None and not delivery_hour:
            continue

        rows.append(
            {
                "snapshot_ts": snapshot_ts,
                "fetch_run_id": fetch_run_id,
                "delivery_date": delivery_date,
                "delivery_hour": delivery_hour,
                "marketTradePrice": price,
                "published_status_completed": completed,
                "response_hash": digest,
                "source_endpoint": INTERIM_MCP_URL,
            }
        )
    return rows


def load_existing_snapshots(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)

    df = pd.read_csv(path)
    for column in SNAPSHOT_COLUMNS:
        if column not in df.columns:
            df[column] = None
    return df[SNAPSHOT_COLUMNS].copy()


def append_snapshot_rows(existing: pd.DataFrame, new_rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not new_rows:
        return existing

    new_df = pd.DataFrame(new_rows, columns=SNAPSHOT_COLUMNS)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["fetch_run_id", "delivery_date", "delivery_hour", "response_hash"],
        keep="last",
    )
    combined = combined.sort_values(
        ["snapshot_ts", "delivery_date", "delivery_hour"], kind="stable"
    ).reset_index(drop=True)
    return combined[SNAPSHOT_COLUMNS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch current visible snapshot but do not write the CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    username = os.getenv("EPIAS_USERNAME")
    password = os.getenv("EPIAS_PASSWORD")
    if not username or not password:
        print("EPIAS_USERNAME / EPIAS_PASSWORD .env veya GitHub Secrets içinde tanımlı olmalı.", file=sys.stderr)
        sys.exit(1)

    acquire_lock()
    try:
        fetch_run_id = str(uuid.uuid4())
        snapshot_ts = istanbul_now().isoformat()
        today = istanbul_now().replace(hour=0, minute=0, second=0, microsecond=0)
        delivery_days = [today, today + timedelta(days=1)]

        tgt = obtain_tgt(username, password)
        log(f"TGT alındı: {tgt[:20]}...")
        headers = {"Accept": "application/json", "Content-Type": "application/json", "TGT": tgt}

        completed = get_published_status(headers)
        log(f"published_status_completed={completed}")

        all_rows: list[dict[str, Any]] = []
        for delivery_day in delivery_days:
            items, digest = fetch_delivery_day(headers, delivery_day)
            rows = rows_from_items(
                items,
                snapshot_ts=snapshot_ts,
                fetch_run_id=fetch_run_id,
                completed=completed,
                digest=digest,
            )
            all_rows.extend(rows)
            log(
                "snapshot delivery_date=%s items=%d rows=%d hash=%s"
                % (delivery_day.date(), len(items), len(rows), digest[:12])
            )
            time.sleep(MIN_REQUEST_SLEEP_SECONDS)

        existing = load_existing_snapshots(CSV_PATH)
        final_df = append_snapshot_rows(existing, all_rows)

        log(
            "snapshot_run=%s new_rows=%d existing_rows=%d final_rows=%d dry_run=%s"
            % (fetch_run_id, len(all_rows), len(existing), len(final_df), args.dry_run)
        )

        if not args.dry_run:
            atomic_write_csv(final_df, CSV_PATH)
            log(f"Snapshot CSV kaydedildi: {CSV_PATH}")
    finally:
        release_lock()


if __name__ == "__main__":
    main()
