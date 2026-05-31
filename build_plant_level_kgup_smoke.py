#!/usr/bin/env python3
"""
Plant-level KGUP smoke fetch.

This script tries the EPİAŞ dpp-bulk endpoint on a tiny sample:
1 day, 5-10 UEVÇB IDs, and writes raw response artifacts for inspection.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_SMOKE_DIR = PROJECT_ROOT / "data" / "plant_level_kgup" / "raw_smoke"
REPORT_PATH = PROJECT_ROOT / "reports" / "plant_level_kgup_api_smoke.md"
REPORT_JSON = PROJECT_ROOT / "reports" / "plant_level_kgup_api_smoke.json"
LOGIN_URL = "https://giris.epias.com.tr/cas/v1/tickets"
POWERPLANT_LIST_URL = "https://seffaflik.epias.com.tr/electricity-service/v1/generation/data/powerplant-list"
UEVCB_BY_PLANT_URL = "https://seffaflik.epias.com.tr/electricity-service/v1/markets/data/uevcb-list-by-power-plant-id"
DPP_BULK_URL = "https://seffaflik.epias.com.tr/electricity-service/v1/generation/data/dpp-bulk"

REQUEST_TIMEOUT = (10, 120)
MAX_RETRIES = 3


def post_with_retries(url: str, **kwargs: Any) -> requests.Response:
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return requests.post(url, timeout=REQUEST_TIMEOUT, **kwargs)
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                time.sleep(5 * attempt)
    raise last_exc


def get_with_retries(url: str, **kwargs: Any) -> requests.Response:
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return requests.get(url, timeout=REQUEST_TIMEOUT, **kwargs)
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                time.sleep(5 * attempt)
    raise last_exc


def login() -> str:
    username = os.getenv("EPIAS_USERNAME")
    password = os.getenv("EPIAS_PASSWORD")
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


def extract_items(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        for key in ["items", "data", "content", "result"]:
            if key in data and isinstance(data[key], list):
                return data[key]
        return []
    if isinstance(data, list):
        return data
    return []


def find_value(item: dict[str, Any], candidates: list[str]) -> Any:
    keys = {str(k).lower(): k for k in item.keys()}
    for cand in candidates:
        if cand.lower() in keys:
            return item[keys[cand.lower()]]
    return None


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="Smoke date in YYYY-MM-DD, default today.")
    parser.add_argument("--plant-limit", type=int, default=3, help="How many power plants with UEVÇB ids to sample.")
    parser.add_argument("--plant-scan-limit", type=int, default=25, help="How many power plants to scan before giving up.")
    parser.add_argument("--uevcb-limit", type=int, default=10, help="How many UEVÇB IDs to sample in total.")
    args = parser.parse_args()

    smoke_date = args.date or datetime.now(timezone.utc).astimezone().date().isoformat()
    raw_dir = RAW_SMOKE_DIR / smoke_date
    raw_dir.mkdir(parents=True, exist_ok=True)

    report = [
        "# Plant-Level KGUP API Smoke",
        "",
        f"- Smoke date: `{smoke_date}`",
        f"- Sampled power plants: `0`",
        f"- Sampled UEVÇB IDs: `0`",
        f"- dpp-bulk payload: `{{}}`",
        "",
        "## Observations",
        "",
        "- ENTSO-E / plant ids present: `false`",
        "- UEVÇB ids present: `false`",
        "- Publication timestamp check: `no`",
        "- Smoke status: `failed`",
        "",
    ]
    try:
        tgt = login()
    except Exception as exc:
        report.extend([f"- Login error: `{exc}`", "", "Smoke stopped before API calls because TGT login could not be obtained."])
        REPORT_PATH.write_text("\n".join(report) + "\n")
        print(f"Wrote {REPORT_PATH}")
        return

    powerplant_resp = api_get(POWERPLANT_LIST_URL, tgt)
    powerplant_body = powerplant_resp.json() if powerplant_resp.headers.get("content-type", "").startswith("application/json") else powerplant_resp.text
    powerplants = powerplant_body if isinstance(powerplant_body, dict) else {"raw": powerplant_body}
    plants = extract_items(powerplants)
    sampled_plants = []
    plant_debug = []
    for item in plants:
        plant_id = find_value(item, ["powerPlantId", "plantId", "id"])
        if plant_id is None:
            continue
        sampled_plants.append(str(plant_id))
        plant_debug.append(item)
        if len(sampled_plants) >= args.plant_scan_limit:
            break

    sampled_uevcb_ids: list[int] = []
    uevcb_debug = []
    start_date = f"{smoke_date}T00:00:00+03:00"
    end_date = f"{smoke_date}T23:59:59+03:00"
    successful_plants = []
    scan_count = 0
    for plant_id in sampled_plants:
        scan_count += 1
        for payload in [
            {"powerPlantId": plant_id, "startDate": start_date, "endDate": end_date},
            {"powerplantId": plant_id, "startDate": start_date, "endDate": end_date},
            {"id": plant_id, "startDate": start_date, "endDate": end_date},
        ]:
            try:
                resp = api_post(UEVCB_BY_PLANT_URL, tgt, payload)
            except Exception as exc:
                uevcb_debug.append({"plant_id": plant_id, "payload": payload, "error": str(exc)})
                continue
            body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text
            uevcb_debug.append(
                {
                    "plant_id": plant_id,
                    "payload": payload,
                    "status_code": resp.status_code,
                    "body": body,
                }
            )
            if resp.status_code != 200:
                continue
            items = extract_items(body)
            for item in items:
                uevcb_id = find_value(item, ["uevcbId", "id", "uevcb_id"])
                if uevcb_id is None:
                    continue
                sampled_uevcb_ids.append(int(uevcb_id))
                if plant_id not in successful_plants:
                    successful_plants.append(plant_id)
                if len(sampled_uevcb_ids) >= args.uevcb_limit:
                    break
            if len(sampled_uevcb_ids) >= args.uevcb_limit:
                break
        if len(successful_plants) >= args.plant_limit or scan_count >= args.plant_scan_limit or len(sampled_uevcb_ids) >= args.uevcb_limit:
            break

    payload = {
        "date": start_date,
        "region": "TR1",
        "uevcbIds": sampled_uevcb_ids[: args.uevcb_limit],
    }
    smoke_result = None
    smoke_error = None
    if payload["uevcbIds"]:
        try:
            smoke_resp = api_post(DPP_BULK_URL, tgt, payload)
            smoke_result = smoke_resp.json() if smoke_resp.headers.get("content-type", "").startswith("application/json") else smoke_resp.text
            smoke_error = None if smoke_resp.status_code == 200 else f"HTTP {smoke_resp.status_code}: {str(smoke_result)[:500]}"
        except Exception as exc:
            smoke_error = str(exc)

    write_json(raw_dir / "powerplant_list.json", powerplants)
    write_json(raw_dir / "sampled_powerplants.json", plant_debug)
    write_json(raw_dir / "uevcb_by_powerplant.json", uevcb_debug)
    if smoke_result is not None:
        write_json(raw_dir / "dpp_bulk.json", smoke_result)
        items = extract_items(smoke_result)
        pd.DataFrame(items).to_json(raw_dir / "dpp_bulk_items.json", orient="records", force_ascii=False, indent=2)

    report = [
        "# Plant-Level KGUP API Smoke",
        "",
        f"- Smoke date: `{smoke_date}`",
        f"- Sampled power plants: `{len(sampled_plants)}`",
        f"- Successful plants with UEVÇB ids: `{len(successful_plants)}`",
        f"- Sampled UEVÇB IDs: `{len(sampled_uevcb_ids)}`",
        f"- dpp-bulk payload: `{json.dumps(payload, ensure_ascii=False)}`",
        "",
        "## Observations",
        "",
        f"- ENTSO-E / plant ids present: `{bool(sampled_plants)}`",
        f"- UEVÇB ids present: `{bool(sampled_uevcb_ids)}`",
        f"- Publication timestamp check: {'`yes`' if smoke_result and any(find_value(item, ['publicationTimestamp', 'publication_timestamp']) is not None for item in extract_items(smoke_result)) else '`no`'}",
        f"- Smoke status: `{('ok' if smoke_result is not None else 'failed')}`",
        "",
    ]
    if smoke_error:
        report.extend([f"- Smoke error: `{smoke_error}`", ""])
    report.extend(
        [
            "## Raw Artifacts",
            "",
            f"- `{(raw_dir / 'powerplant_list.json').relative_to(PROJECT_ROOT)}`",
            f"- `{(raw_dir / 'sampled_powerplants.json').relative_to(PROJECT_ROOT)}`",
            f"- `{(raw_dir / 'uevcb_by_powerplant.json').relative_to(PROJECT_ROOT)}`",
        ]
    )
    if smoke_result is not None:
        report.append(f"- `{(raw_dir / 'dpp_bulk.json').relative_to(PROJECT_ROOT)}`")
    REPORT_PATH.write_text("\n".join(report) + "\n")
    REPORT_JSON.write_text(
        json.dumps(
            {
                "smoke_date": smoke_date,
                "sampled_powerplants": sampled_plants,
                "successful_plants": successful_plants,
                "sampled_uevcb_ids": sampled_uevcb_ids,
                "scan_count": scan_count,
                "payload": payload,
                "date_window": {"startDate": start_date, "endDate": end_date},
                "powerplant_list_status": powerplant_resp.status_code,
                "powerplant_list_keys": list(powerplants.keys()) if isinstance(powerplants, dict) else None,
                "uevcb_debug": uevcb_debug,
                "smoke_status": "ok" if smoke_result is not None else "failed",
                "smoke_error": smoke_error,
                "publication_timestamp_present": bool(
                    smoke_result and any(find_value(item, ["publicationTimestamp", "publication_timestamp"]) is not None for item in extract_items(smoke_result))
                ),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        + "\n"
    )

    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {REPORT_JSON}")
    print(f"Wrote {raw_dir}")


if __name__ == "__main__":
    main()
