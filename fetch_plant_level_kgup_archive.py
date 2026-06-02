#!/usr/bin/env python3
"""
Fetch a resumable plant-level KGUP archive from EPİAŞ.

This script discovers power plants, resolves their UEVÇB IDs, and fetches the
daily dpp-bulk output in rate-limit-aware batches. It preserves the raw bodies,
stores normalized chunk CSV files under `data/plant_level_kgup/raw_archive/`,
and writes a combined archive parquet at the end of the run.

Important:
    The dpp-bulk response currently exposes hourly generation by UEVÇB but does
    not appear to publish a dedicated publication timestamp field. We preserve
    any timestamp-like field if EPİAŞ exposes one, otherwise it is written as
    null and surfaced in the audit report.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import date as date_cls
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_ARCHIVE_ROOT = PROJECT_ROOT / "data" / "plant_level_kgup" / "raw_archive"
COMBINED_ARCHIVE_PATH = PROJECT_ROOT / "data" / "plant_level_kgup" / "plant_level_kgup_archive.parquet"
STATE_BASENAME = "plant_level_kgup_archive_state"
REPORT_MD = PROJECT_ROOT / "reports" / "plant_level_kgup_archive_fetch.md"
REPORT_JSON = PROJECT_ROOT / "reports" / "plant_level_kgup_archive_fetch.json"
COMBINED_INDEX_PATH = PROJECT_ROOT / "data" / "plant_level_kgup" / "plant_level_kgup_archive_index.csv"

LOGIN_URL = "https://giris.epias.com.tr/cas/v1/tickets"
POWERPLANT_LIST_URL = "https://seffaflik.epias.com.tr/electricity-service/v1/generation/data/organization-list"
UEVCB_BY_PLANT_URL = "https://seffaflik.epias.com.tr/electricity-service/v1/generation/data/uevcb-list-bulk"
DPP_BULK_URL = "https://seffaflik.epias.com.tr/electricity-service/v1/generation/data/dpp-bulk"

REQUEST_TIMEOUT = (10, 120)
MAX_RETRIES = 4
DEFAULT_SLEEP_SECONDS = 8.0
DEFAULT_BATCH_SIZE = 25

load_dotenv(PROJECT_ROOT / ".env")


def parse_date(value: str) -> pd.Timestamp:
    ts = pd.to_datetime(value, errors="raise")
    if getattr(ts, "tzinfo", None) is None:
        ts = ts.tz_localize("Europe/Istanbul")
    return ts


def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def post_with_retries(url: str, **kwargs: Any) -> requests.Response:
    last_exc: Exception | None = None
    sleep = 5.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return requests.post(url, timeout=REQUEST_TIMEOUT, **kwargs)
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                time.sleep(sleep)
                sleep *= 1.8
    assert last_exc is not None
    raise last_exc


def get_with_retries(url: str, **kwargs: Any) -> requests.Response:
    last_exc: Exception | None = None
    sleep = 5.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return requests.get(url, timeout=REQUEST_TIMEOUT, **kwargs)
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                time.sleep(sleep)
                sleep *= 1.8
    assert last_exc is not None
    raise last_exc


def login() -> str:
    username = os.getenv("EPIAS_USERNAME")
    password = os.getenv("EPIAS_PASSWORD")
    if not username or not password:
        raise RuntimeError("EPIAS_USERNAME and EPIAS_PASSWORD must be set.")
    resp = post_with_retries(
        LOGIN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "text/plain"},
        data={"username": username, "password": password},
    )
    text = resp.text.strip()
    if text.startswith("TGT-"):
        return text
    match = re.search(r"/cas/v1/tickets/([^\" ]+)", text)
    if not match:
        raise RuntimeError(f"Could not parse TGT from response: {text[:500]}")
    return match.group(1)


def api_get(url: str, tgt: str) -> requests.Response:
    return get_with_retries(url, headers={"Accept": "application/json", "TGT": tgt})


def api_post(url: str, tgt: str, payload: dict[str, Any]) -> requests.Response:
    return post_with_retries(
        url,
        headers={"Accept": "application/json", "Content-Type": "application/json", "TGT": tgt},
        json=payload,
    )


def extract_items(body: Any) -> list[dict[str, Any]]:
    if isinstance(body, dict):
        for key in ("items", "data", "content", "result"):
            value = body.get(key)
            if isinstance(value, list):
                return value
        return []
    if isinstance(body, list):
        return body
    return []


def slug(value: Any) -> str:
    text = str(value).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()


def detect_value(item: dict[str, Any], candidates: list[str]) -> Any:
    key_map = {slug(k): k for k in item.keys()}
    for candidate in candidates:
        found = key_map.get(slug(candidate))
        if found is not None:
            return item.get(found)
    return None


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n")
    tmp.replace(path)


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.parquet")
    from src.utils.safe_io import atomic_parquet_write
    atomic_parquet_write(frame, str(tmp), index=False)
    tmp.replace(path)


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "completed_dates": [],
            "completed_chunks": [],
            "discovered_plants": 0,
            "discovered_uevcbs": 0,
            "last_updated_at": None,
        }
    return json.loads(path.read_text())


def save_state(path: Path, state: dict[str, Any]) -> None:
    state["last_updated_at"] = utc_now().isoformat()
    write_json(path, state)


@dataclass
class PlantRecord:
    power_plant_id: int
    entsoe_code: str
    plant_name: str
    short_name: str


def discover_powerplants(tgt: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> list[PlantRecord]:
    payload = {
        "startDate": start_date.strftime("%Y-%m-%dT00:00:00+03:00"),
        "endDate": end_date.strftime("%Y-%m-%dT23:59:59+03:00"),
    }
    resp = api_post(POWERPLANT_LIST_URL, tgt, payload)
    body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {"raw": resp.text}
    items = extract_items(body)
    print(f"[discover] organization-list returned {len(items)} rows", flush=True)
    plants: list[PlantRecord] = []
    for item in items:
        plant_id = detect_value(item, ["organizationId", "id", "powerPlantId", "power_plant_id"])
        eic = detect_value(item, ["organizationCode", "organizationEtsoCode", "eic", "entsoe_code", "entsoeCode"])
        name = detect_value(item, ["organizationName", "name", "plantName", "powerPlantName"]) or detect_value(item, ["organizationShortName", "shortName"]) or ""
        short_name = detect_value(item, ["organizationShortName", "shortName"]) or name or ""
        if plant_id is None or eic is None:
            continue
        plants.append(
            PlantRecord(
                power_plant_id=int(plant_id),
                entsoe_code=str(eic).strip(),
                plant_name=str(name).strip(),
                short_name=str(short_name).strip(),
            )
        )
    return plants


def discover_uevcbs_bulk(
    tgt: str,
    organization_ids: list[int],
    start_date: pd.Timestamp,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payloads = [
        {"organizationIds": organization_ids, "startDate": start_date.strftime("%Y-%m-%dT00:00:00+03:00")},
        {"organizationIds": organization_ids},
    ]
    debug_entries = []
    print(f"[discover] UEVÇB bulk lookup for {len(organization_ids)} organizations", flush=True)
    for payload in payloads:
        try:
            resp = api_post(UEVCB_BY_PLANT_URL, tgt, payload)
            body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text
            debug_entries.append(
                {"organization_ids": organization_ids[:25], "payload": payload, "status_code": resp.status_code, "body": body}
            )
            if resp.status_code != 200:
                continue
            items = extract_items(body)
            if items:
                return items, debug_entries
        except Exception as exc:
            debug_entries.append({"organization_ids": organization_ids[:25], "payload": payload, "error": str(exc)})
    return [], debug_entries


def parse_kgup_item(
    item: dict[str, Any],
    uevcb_to_plant: dict[str, PlantRecord],
    fetched_at: pd.Timestamp,
    source_file: str,
) -> dict[str, Any] | None:
    uevcb_id = detect_value(item, ["uevcbId", "uevcb_id", "id"])
    if uevcb_id is None:
        return None
    uevcb_key = str(int(uevcb_id))
    plant = uevcb_to_plant.get(uevcb_key)
    date_value = detect_value(item, ["date"])
    time_value = detect_value(item, ["time"])
    delivery_hour = pd.to_datetime(date_value, errors="coerce")
    if pd.isna(delivery_hour):
        return None
    if getattr(delivery_hour, "tzinfo", None) is None:
        delivery_hour = delivery_hour.tz_localize("Europe/Istanbul")
    if isinstance(time_value, str):
        try:
            hour_match = re.match(r"^(\d{1,2})", time_value)
            if hour_match:
                delivery_hour = delivery_hour.normalize() + pd.to_timedelta(int(hour_match.group(1)), unit="h")
                if getattr(delivery_hour, "tzinfo", None) is None:
                    delivery_hour = delivery_hour.tz_localize("Europe/Istanbul")
        except Exception:
            pass
    publication_timestamp = detect_value(
        item,
        ["publicationTimestamp", "publication_timestamp", "publishTime", "publishedAt", "publicationDate", "createdAt"],
    )
    if publication_timestamp is not None:
        publication_timestamp = pd.to_datetime(publication_timestamp, errors="coerce")
    else:
        publication_timestamp = pd.NaT
    kgup_mwh = detect_value(item, ["toplam", "kgup_mwh", "kgup", "value"])
    try:
        kgup_mwh = float(kgup_mwh) if kgup_mwh is not None else None
    except Exception:
        kgup_mwh = None
    if kgup_mwh is None:
        return None
    return {
        "entsoe_code": plant.entsoe_code if plant else None,
        "plant_name": plant.plant_name if plant else detect_value(item, ["uevcbName", "plantName", "name"]),
        "fuel_type": "unknown",
        "delivery_hour": delivery_hour,
        "publication_timestamp": publication_timestamp,
        "kgup_mwh": kgup_mwh,
        "source_file": source_file,
        "source_api": "POST /v1/generation/data/dpp-bulk",
        "forecast_timestamp": fetched_at,
        "archive_snapshot_timestamp": fetched_at,
        "publication_timestamp_source": "missing_in_source",
        "uevcb_id": int(uevcb_id),
        "uevcb_name": detect_value(item, ["uevcbName", "name"]),
        "org_id": detect_value(item, ["orgId", "organizationId"]),
        "date_raw": date_value,
        "time_raw": time_value,
    }


def fetch_day_batch(
    tgt: str,
    day: pd.Timestamp,
    uevcb_ids: list[int],
    uevcb_to_plant: dict[str, PlantRecord],
    out_dir: Path,
    batch_idx: int,
    sleep_seconds: float,
) -> tuple[int, Path | None, Path | None, Path | None]:
    request_ts = utc_now()
    payload = {
        "date": day.strftime("%Y-%m-%dT00:00:00+03:00"),
        "region": "TR1",
        "uevcbIds": uevcb_ids,
    }
    print(
        f"[fetch] {day.date().isoformat()} batch={batch_idx:04d} uevcb_count={len(uevcb_ids)}",
        flush=True,
    )
    raw_response_path = out_dir / f"chunk_{batch_idx:04d}.json"
    raw_meta_path = out_dir / f"chunk_{batch_idx:04d}.meta.json"
    normalized_path = out_dir / f"chunk_{batch_idx:04d}.csv"
    resp = api_post(DPP_BULK_URL, tgt, payload)
    body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text
    write_json(raw_response_path, body)
    write_json(
        raw_meta_path,
        {
            "requested_at": request_ts.isoformat(),
            "status_code": resp.status_code,
            "payload": payload,
            "source_endpoint": "POST /v1/generation/data/dpp-bulk",
        },
    )
    if resp.status_code != 200:
        return 0, raw_response_path, raw_meta_path, None
    items = extract_items(body)
    rows: list[dict[str, Any]] = []
    for item in items:
        parsed = parse_kgup_item(item, uevcb_to_plant, request_ts, raw_response_path.name)
        if parsed is not None:
            rows.append(parsed)
    if rows:
        frame = pd.DataFrame(rows)
        frame["delivery_hour"] = pd.to_datetime(frame["delivery_hour"], errors="coerce")
        frame["publication_timestamp"] = pd.to_datetime(frame["publication_timestamp"], errors="coerce")
        frame["forecast_timestamp"] = pd.to_datetime(frame["forecast_timestamp"], errors="coerce", utc=True)
        frame["archive_snapshot_timestamp"] = pd.to_datetime(frame["archive_snapshot_timestamp"], errors="coerce", utc=True)
        write_csv_atomic(normalized_path, frame)
    else:
        normalized_path = None
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)
    return len(rows), raw_response_path, raw_meta_path, normalized_path


def collect_uevcb_map(
    tgt: str,
    plants: list[PlantRecord],
    start_date: pd.Timestamp,
    max_plants: int = 0,
) -> tuple[dict[str, PlantRecord], list[dict[str, Any]]]:
    debug = []
    mapping: dict[str, PlantRecord] = {}
    selected = plants if max_plants <= 0 else plants[:max_plants]
    batch_size = 1000
    for idx in range(0, len(selected), batch_size):
        batch = selected[idx : idx + batch_size]
        ids = [plant.power_plant_id for plant in batch]
        bulk_items, entries = discover_uevcbs_bulk(tgt, ids, start_date)
        debug.extend(entries)
        for item in bulk_items:
            uevcb_id = detect_value(item, ["uevcbId", "uevcb_id", "id"])
            org_id = detect_value(item, ["orgId", "organizationId"])
            eic = detect_value(item, ["eic", "entsoe_code", "entsoeCode", "organizationCode"])
            name = detect_value(item, ["name", "organizationName", "organizationShortName"])
            if uevcb_id is None or org_id is None:
                continue
            mapping[str(int(uevcb_id))] = PlantRecord(
                power_plant_id=int(org_id),
                entsoe_code=str(eic).strip() if eic is not None else "",
                plant_name=str(name).strip() if name is not None else "",
                short_name=str(name).strip() if name is not None else "",
            )
    return mapping, debug


def list_new_dates(start_date: pd.Timestamp, end_date: pd.Timestamp, completed_dates: set[str], only_missing: bool) -> list[pd.Timestamp]:
    dates = []
    current = start_date.normalize()
    end = end_date.normalize()
    while current <= end:
        key = current.date().isoformat()
        if not (only_missing and key in completed_dates):
            dates.append(current)
        current += pd.Timedelta(days=1)
    return dates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", required=False, default="2026-05-31", help="Inclusive start date, YYYY-MM-DD.")
    parser.add_argument("--end-date", required=False, default="2026-06-01", help="Inclusive end date, YYYY-MM-DD.")
    parser.add_argument("--sleep-seconds", type=float, default=DEFAULT_SLEEP_SECONDS, help="Sleep between requests.")
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES, help="Reserved for future tuning; retries are built-in.")
    parser.add_argument("--resume", action="store_true", help="Resume from state file.")
    parser.add_argument("--only-missing", action="store_true", help="Skip chunks already materialized on disk.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="UEVCB ids per dpp-bulk request.")
    parser.add_argument("--max-plants", type=int, default=0, help="Optional cap for plant discovery. 0 means all.")
    args = parser.parse_args()

    del args.max_retries  # retries are fixed inside helpers; the flag keeps the CLI stable.

    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date)
    if end_date < start_date:
        raise ValueError("--end-date must be >= --start-date")

    run_dir = RAW_ARCHIVE_ROOT / f"{start_date.date().isoformat()}_{end_date.date().isoformat()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / f"{STATE_BASENAME}_{start_date.date().isoformat()}_{end_date.date().isoformat()}.json"
    lock_path = run_dir / ".lock"
    if lock_path.exists():
        raise RuntimeError(f"Lock file exists, another run may be active: {lock_path}")
    lock_path.write_text(f"locked_at={utc_now().isoformat()}\n")

    try:
        state = load_state(state_path) if args.resume else {
            "completed_dates": [],
            "completed_chunks": [],
            "discovered_plants": 0,
            "discovered_uevcbs": 0,
            "last_updated_at": None,
        }
        completed_dates = set(state.get("completed_dates", []))
        completed_chunks = set(state.get("completed_chunks", []))

        tgt = login()
        plants = discover_powerplants(tgt, start_date, end_date)
        if not plants:
            raise RuntimeError("No power plants discovered from powerplant-list.")
        print(f"[discover] accepted {len(plants)} power plants", flush=True)
        uevcb_map, discovery_debug = collect_uevcb_map(tgt, plants, start_date, max_plants=args.max_plants)
        if not uevcb_map:
            raise RuntimeError("No UEVÇB ids discovered from power plants.")
        print(f"[discover] accepted {len(uevcb_map)} unique UEVÇB ids", flush=True)

        state["discovered_plants"] = len(plants)
        state["discovered_uevcbs"] = len(uevcb_map)
        state["powerplants_sample"] = [plant.__dict__ for plant in plants[:25]]
        state["uevcb_debug_rows"] = len(discovery_debug)
        save_state(state_path, state)

        # Persist discovery artifacts for auditing.
        write_json(run_dir / "powerplant_list.json", [plant.__dict__ for plant in plants])
        write_json(run_dir / "uevcb_discovery_debug.json", discovery_debug)
        write_json(run_dir / "uevcb_map.json", {k: v.__dict__ for k, v in uevcb_map.items()})

        uevcb_ids = sorted(int(k) for k in uevcb_map.keys())
        all_rows: list[pd.DataFrame] = []
        requested_chunks = 0
        successful_chunks = 0
        failed_chunks: list[dict[str, Any]] = []
        daily_summary: list[dict[str, Any]] = []

        for day in list_new_dates(start_date, end_date, completed_dates, args.only_missing):
            print(f"[day] starting {day.date().isoformat()}", flush=True)
            day_dir = run_dir / day.date().isoformat()
            day_dir.mkdir(parents=True, exist_ok=True)
            day_rows = 0
            day_success = 0
            day_fail = 0

            for chunk_idx, i in enumerate(range(0, len(uevcb_ids), args.batch_size)):
                chunk_ids = uevcb_ids[i : i + args.batch_size]
                chunk_key = f"{day.date().isoformat()}::{chunk_idx:04d}::{','.join(map(str, chunk_ids))}"
                if args.only_missing and chunk_key in completed_chunks:
                    continue
                requested_chunks += 1
                try:
                    rows, raw_path, meta_path, normalized_path = fetch_day_batch(
                        tgt=tgt,
                        day=day,
                        uevcb_ids=chunk_ids,
                        uevcb_to_plant=uevcb_map,
                        out_dir=day_dir,
                        batch_idx=chunk_idx,
                        sleep_seconds=args.sleep_seconds,
                    )
                    if normalized_path is not None and normalized_path.exists():
                        all_rows.append(pd.read_csv(normalized_path))
                    if rows > 0:
                        successful_chunks += 1
                        day_success += 1
                        day_rows += rows
                    else:
                        day_fail += 1
                        failed_chunks.append(
                            {
                                "day": day.date().isoformat(),
                                "chunk_idx": chunk_idx,
                                "chunk_ids": chunk_ids,
                                "reason": f"zero rows or non-200 response; raw={raw_path.name if raw_path else None}",
                            }
                        )
                    completed_chunks.add(chunk_key)
                    save_state(
                        state_path,
                        {
                            **state,
                            "completed_dates": sorted(completed_dates | {day.date().isoformat()}),
                            "completed_chunks": sorted(completed_chunks),
                            "discovered_plants": len(plants),
                            "discovered_uevcbs": len(uevcb_map),
                        },
                    )
                except Exception as exc:
                    day_fail += 1
                    failed_chunks.append(
                        {
                            "day": day.date().isoformat(),
                            "chunk_idx": chunk_idx,
                            "chunk_ids": chunk_ids,
                            "error": str(exc),
                        }
                    )
                    save_state(
                        state_path,
                        {
                            **state,
                            "completed_dates": sorted(completed_dates),
                            "completed_chunks": sorted(completed_chunks),
                            "discovered_plants": len(plants),
                            "discovered_uevcbs": len(uevcb_map),
                        },
                    )

            completed_dates.add(day.date().isoformat())
            daily_summary.append(
                {
                    "date": day.date().isoformat(),
                    "chunk_count": int((len(uevcb_ids) + args.batch_size - 1) // args.batch_size),
                    "successful_chunks": day_success,
                    "failed_chunks": day_fail,
                    "rows": day_rows,
                }
            )
            save_state(
                state_path,
                {
                    **state,
                    "completed_dates": sorted(completed_dates),
                    "completed_chunks": sorted(completed_chunks),
                    "discovered_plants": len(plants),
                    "discovered_uevcbs": len(uevcb_map),
                },
            )

        # Combine all chunk CSVs written in this run and any prior resumed chunks.
        chunk_csvs = sorted(run_dir.rglob("chunk_*.csv"))
        if chunk_csvs:
            combined = pd.concat((pd.read_csv(path) for path in chunk_csvs), ignore_index=True)
            for col in ["delivery_hour", "publication_timestamp", "forecast_timestamp"]:
                if col in combined.columns:
                    combined[col] = pd.to_datetime(combined[col], errors="coerce")
            if "archive_snapshot_timestamp" in combined.columns:
                combined["archive_snapshot_timestamp"] = pd.to_datetime(
                    combined["archive_snapshot_timestamp"], errors="coerce", utc=True
                )
            combined = combined.drop_duplicates(
                subset=["entsoe_code", "delivery_hour", "publication_timestamp", "forecast_timestamp", "archive_snapshot_timestamp"],
                keep="last",
            )
            combined = combined.sort_values(["delivery_hour", "entsoe_code"])
            write_parquet_atomic(COMBINED_ARCHIVE_PATH, combined)
            COMBINED_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
            combined_index = combined[[
                c for c in [
                    "delivery_hour",
                    "entsoe_code",
                    "plant_name",
                    "uevcb_id",
                    "kgup_mwh",
                    "publication_timestamp",
                    "forecast_timestamp",
                    "archive_snapshot_timestamp",
                    "source_file",
                ]
                if c in combined.columns
            ]].copy()
            combined_index.to_csv(COMBINED_INDEX_PATH, index=False)
            combined_rows = int(len(combined))
        else:
            combined_rows = 0

        report = {
            "generated_at": utc_now().isoformat(),
            "start_date": start_date.date().isoformat(),
            "end_date": end_date.date().isoformat(),
            "powerplants": len(plants),
            "uevcbs": len(uevcb_map),
            "requested_chunks": requested_chunks,
            "successful_chunks": successful_chunks,
            "failed_chunks": len(failed_chunks),
            "combined_rows": combined_rows,
            "completed_dates": sorted(completed_dates),
            "completed_chunks": sorted(completed_chunks),
            "failed_chunk_examples": failed_chunks[:25],
            "publication_timestamp_present": bool(
                chunk_csvs
                and any(
                    pd.read_csv(path).get("publication_timestamp", pd.Series(dtype="object")).notna().any()
                    for path in chunk_csvs[: min(len(chunk_csvs), 10)]
                )
            ),
            "archive_snapshot_timestamp_present": bool(
                chunk_csvs
                and any(
                    pd.read_csv(path).get("archive_snapshot_timestamp", pd.Series(dtype="object")).notna().any()
                    for path in chunk_csvs[: min(len(chunk_csvs), 10)]
                )
            ),
            "note": "The source endpoint does not expose publication timestamp in the fetched payload; archive_snapshot_timestamp is preserved as the fetch time.",
            "source_endpoints": {
                "powerplant_list": POWERPLANT_LIST_URL,
                "uevcb_by_power_plant": UEVCB_BY_PLANT_URL,
                "dpp_bulk": DPP_BULK_URL,
            },
            "raw_archive_root": str(RAW_ARCHIVE_ROOT.relative_to(PROJECT_ROOT)),
            "combined_archive_path": str(COMBINED_ARCHIVE_PATH.relative_to(PROJECT_ROOT)),
        }
        write_json(REPORT_JSON, report)
        REPORT_MD.write_text(
            "\n".join(
                [
                    "# Plant-Level KGUP Archive Fetch",
                    "",
                    f"- Period: `{report['start_date']}` → `{report['end_date']}`",
                    f"- Power plants discovered: `{report['powerplants']}`",
                    f"- UEVÇB ids discovered: `{report['uevcbs']}`",
                    f"- Requested chunks: `{report['requested_chunks']}`",
                    f"- Successful chunks: `{report['successful_chunks']}`",
                    f"- Failed chunks: `{report['failed_chunks']}`",
                    f"- Combined archive rows: `{report['combined_rows']}`",
                    "",
                    "## Notes",
                    "",
                    "- EPİAŞ dpp-bulk gives hourly plant-level KGÜP by UEVÇB.",
                    "- Power plant discovery uses `powerplant-list` and then `uevcb-list-by-power-plant-id`.",
                    "- If the source does not expose a publication timestamp, the field is preserved as null and flagged in the audit.",
                    "",
                    "## Paths",
                    "",
                    f"- Raw archive root: `{report['raw_archive_root']}`",
                    f"- Combined parquet: `{report['combined_archive_path']}`",
                    f"- State file: `{state_path.relative_to(PROJECT_ROOT)}`",
                    "",
                    "## Next Step",
                    "",
                    "Run `python3 build_plant_level_kgup_pipeline.py` to convert the archive into leakage-audited must-run features.",
                    "",
                ]
            )
            + "\n"
        )
        print(f"Wrote {REPORT_MD}")
        print(f"Wrote {REPORT_JSON}")
        print(f"Wrote {COMBINED_ARCHIVE_PATH}")
        print(f"Wrote {state_path}")
    finally:
        if lock_path.exists():
            lock_path.unlink()


if __name__ == "__main__":
    main()
