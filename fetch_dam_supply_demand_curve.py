#!/usr/bin/env python3
"""
Smoke test for EPİAŞ DAM supply-demand curve endpoints.

Calls:
- /v1/markets/dam/data/supply-demand-chart-data
- /v1/markets/dam/data/supply-demand-chart-ptf-data

Only a single hour smoke test is performed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

LOGIN_URL = "https://giris.epias.com.tr/cas/v1/tickets"
SUPPLY_DEMAND_URL = "https://seffaflik.epias.com.tr/electricity-service/v1/markets/dam/data/supply-demand-chart-data"
PTF_URL = "https://seffaflik.epias.com.tr/electricity-service/v1/markets/dam/data/supply-demand-chart-ptf-data"

OUT_DIR = PROJECT_ROOT / "data" / "raw" / "dam_supply_demand_curve_smoke"
REPORT_MD = PROJECT_ROOT / "reports" / "dam_supply_demand_curve_smoke.md"
REPORT_JSON = PROJECT_ROOT / "reports" / "dam_supply_demand_curve_smoke.json"
NORMALIZED_CSV = OUT_DIR / "normalized_curve.csv"
NORMALIZED_PQ = OUT_DIR / "normalized_curve.parquet"

REQUEST_TIMEOUT = (10, 120)
MAX_RETRIES = 4
SMOKE_DATE = "2026-06-01"
SMOKE_HOUR = "00:00"
SMOKE_DATETIME = "2026-06-01T00:00:00+03:00"


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def post_with_retries(url: str, **kwargs) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return requests.post(url, timeout=REQUEST_TIMEOUT, **kwargs)
        except requests.exceptions.RequestException as exc:
            last_error = exc
            wait_s = min(60, 5 * 2 ** (attempt - 1))
            log(f"network retry={attempt}/{MAX_RETRIES} wait={wait_s}s err={exc}")
            if attempt < MAX_RETRIES:
                time.sleep(wait_s)
    raise last_error  # type: ignore[misc]


def obtain_tgt(username: str, password: str) -> str:
    response = post_with_retries(
        LOGIN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "text/plain"},
        data={"username": username, "password": password},
    )
    log(f"TGT status={response.status_code}")
    text = response.text.strip()
    if text.startswith("TGT-"):
        return text
    match = re.search(r"/cas/v1/tickets/([^\" ]+)", text)
    if not match:
        raise RuntimeError(f"TGT alınamadı: {text[:500]}")
    return match.group(1)


def normalize_response(response: requests.Response) -> tuple[pd.DataFrame, dict[str, Any]]:
    content_type = response.headers.get("Content-Type", "")
    body_text = response.text
    meta: dict[str, Any] = {
        "status_code": response.status_code,
        "content_type": content_type,
        "response_hash": hashlib.sha256(body_text.encode("utf-8", errors="ignore")).hexdigest(),
        "body_preview": body_text[:2000],
    }
    try:
        payload = response.json()
    except Exception:
        return pd.DataFrame(), meta

    items = []
    if isinstance(payload, dict):
        nested_response = payload.get("body", {})
        if isinstance(nested_response, dict):
            nested_content = nested_response.get("content", {})
            if isinstance(nested_content, dict) and isinstance(nested_content.get("response"), dict):
                items = [nested_content["response"]]
                meta["nested_response"] = True
                meta["nested_keys"] = list(nested_content["response"].keys())
                meta["json_top_level_type"] = type(payload).__name__
                meta["items_count"] = len(items)
                return pd.DataFrame(items), meta
        if isinstance(payload.get("items"), list):
            items = payload["items"]
        elif isinstance(payload.get("data"), list):
            items = payload["data"]
        elif isinstance(payload.get("result"), list):
            items = payload["result"]
        else:
            # Flatten common dict shapes
            for key in ["price", "prices", "curve", "points", "rows"]:
                if isinstance(payload.get(key), list):
                    items = payload[key]
                    break
            if not items and any(isinstance(v, list) for v in payload.values()):
                for v in payload.values():
                    if isinstance(v, list):
                        items = v
                        break
    elif isinstance(payload, list):
        items = payload

    meta["json_top_level_type"] = type(payload).__name__
    meta["items_count"] = len(items)

    if not items:
        return pd.DataFrame(), meta

    df = pd.DataFrame(items)
    return df, meta


def save_raw_response(path: Path, response: requests.Response, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.with_suffix(".body.txt")).write_text(response.text, encoding="utf-8")
    (path.with_suffix(".meta.json")).write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str) + "\n")


def try_payload(tgt: str, url: str, payloads: list[dict[str, Any]], tag: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    headers = {"Accept": "application/json", "Content-Type": "application/json", "TGT": tgt}
    last_df = pd.DataFrame()
    last_meta: dict[str, Any] = {}
    for idx, payload in enumerate(payloads, start=1):
        log(f"{tag} attempt={idx}/{len(payloads)} payload={payload}")
        response = post_with_retries(url, json=payload, headers=headers)
        df, meta = normalize_response(response)
        meta["endpoint"] = url
        meta["payload"] = payload
        save_raw_response(OUT_DIR / f"{tag}_attempt_{idx:02d}", response, meta)
        last_df, last_meta = df, meta
        if response.status_code == 200:
            return df, meta
        log(f"{tag} HTTP {response.status_code}: {response.text[:800]}")
    return last_df, last_meta


def main() -> None:
    username = os.getenv("EPIAS_USERNAME")
    password = os.getenv("EPIAS_PASSWORD")
    if not username or not password:
        print("EPIAS_USERNAME / EPIAS_PASSWORD .env içinde tanımlı olmalı.", file=sys.stderr)
        sys.exit(1)

    tgt = obtain_tgt(username, password)
    log(f"TGT alındı: {tgt[:18]}...")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    payload_variants = [
        {"date": SMOKE_DATETIME},
    ]

    supply_df, supply_meta = try_payload(tgt, SUPPLY_DEMAND_URL, payload_variants, "supply_demand")
    ptf_df, ptf_meta = try_payload(tgt, PTF_URL, payload_variants, "ptf")

    normalized_rows = []
    if not supply_df.empty:
        cols = {c.lower(): c for c in supply_df.columns}
        price_col = next((cols[k] for k in ["supplyprice", "price", "fiyat"] if k in cols), None)
        supply_col = next((cols[k] for k in ["amount", "supply_mwh", "supply", "quantity", "mwh", "volume"] if k in cols), None)
        demand_col = next((cols[k] for k in ["demand_mwh", "demand", "requirement", "need"] if k in cols), None)
        for _, row in supply_df.iterrows():
            normalized_rows.append(
                {
                    "delivery_hour": SMOKE_DATETIME,
                    "price": row.get(price_col) if price_col else None,
                    "supply_mwh": row.get(supply_col) if supply_col else None,
                    "demand_mwh": row.get(demand_col) if demand_col else None,
                    "source_endpoint": SUPPLY_DEMAND_URL,
                }
            )

    if not ptf_df.empty:
        cols = {c.lower(): c for c in ptf_df.columns}
        clearing_price_col = next((cols[k] for k in ["mcpprice", "clearing_price", "markettradeprice", "ptf", "price"] if k in cols), None)
        clearing_volume_col = next((cols[k] for k in ["matchingquantity", "clearing_volume", "volume", "mwh", "quantity"] if k in cols), None)
        for _, row in ptf_df.iterrows():
            normalized_rows.append(
                {
                    "delivery_hour": SMOKE_DATETIME,
                    "price": row.get(clearing_price_col) if clearing_price_col else None,
                    "supply_mwh": row.get(clearing_volume_col) if clearing_volume_col else None,
                    "demand_mwh": None,
                    "source_endpoint": PTF_URL,
                }
            )

    normalized = pd.DataFrame(normalized_rows)
    if not normalized.empty:
        normalized["delivery_hour"] = pd.to_datetime(normalized["delivery_hour"], errors="coerce")
        normalized = normalized.sort_values(["source_endpoint", "delivery_hour"]).reset_index(drop=True)
        normalized.to_csv(NORMALIZED_CSV, index=False)
        try:
            normalized.to_parquet(NORMALIZED_PQ, index=False)
        except Exception:
            pass

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "smoke_date": SMOKE_DATE,
        "smoke_hour": SMOKE_HOUR,
        "supply_demand": {
            "status_code": supply_meta.get("status_code"),
            "items_count": supply_meta.get("items_count", 0),
            "content_type": supply_meta.get("content_type"),
            "response_hash": supply_meta.get("response_hash"),
            "body_preview": supply_meta.get("body_preview"),
            "available_columns": list(supply_df.columns),
            "raw_file": str((OUT_DIR / "supply_demand_attempt_01.body.txt").relative_to(PROJECT_ROOT)) if (OUT_DIR / "supply_demand_attempt_01.body.txt").exists() else None,
        },
        "ptf": {
            "status_code": ptf_meta.get("status_code"),
            "items_count": ptf_meta.get("items_count", 0),
            "content_type": ptf_meta.get("content_type"),
            "response_hash": ptf_meta.get("response_hash"),
            "body_preview": ptf_meta.get("body_preview"),
            "available_columns": list(ptf_df.columns),
            "raw_file": str((OUT_DIR / "ptf_attempt_01.body.txt").relative_to(PROJECT_ROOT)) if (OUT_DIR / "ptf_attempt_01.body.txt").exists() else None,
        },
        "normalized": {
            "rows": int(len(normalized)),
            "columns": list(normalized.columns),
            "clearing_price_column_found": any(c in {c.lower() for c in ptf_df.columns} for c in ["mcpprice", "clearing_price", "markettradeprice", "ptf", "price"]),
            "supply_demand_columns_present": {
                "price": any(c in {c.lower() for c in supply_df.columns} for c in ["supplyprice", "price", "fiyat", "offerprice", "bidprice"]),
                "supply_mwh": any(c in {c.lower() for c in supply_df.columns} for c in ["amount", "supply_mwh", "supply", "quantity", "mwh", "volume"]),
                "demand_mwh": any(c in {c.lower() for c in supply_df.columns} for c in ["demandprice", "demand_mwh", "demand", "requirement", "need"]),
            },
        },
        "usable_for_curve_extraction": bool(len(normalized) > 0),
        "notes": [
            "Raw response bodies and meta JSON saved per endpoint attempt.",
            "Only a single-hour smoke test was executed.",
            "Proxy fallback was not used.",
        ],
    }

    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n")
    REPORT_MD.write_text(
        "\n".join(
            [
                "# DAM Supply-Demand Curve Smoke Test",
                "",
                f"Generated: `{report['generated_at']}`",
                "",
                "## Endpoint Status",
                "",
                f"- Supply-demand endpoint status: `{report['supply_demand']['status_code']}`",
                f"- PTF endpoint status: `{report['ptf']['status_code']}`",
                f"- Supply-demand rows/items: `{report['supply_demand']['items_count']}`",
                f"- PTF rows/items: `{report['ptf']['items_count']}`",
                f"- Normalized rows: `{report['normalized']['rows']}`",
                "",
                "## Coverage",
                "",
                f"- Smoke date: `{SMOKE_DATE}`",
                f"- Smoke hour: `{SMOKE_HOUR}`",
                f"- Supply-demand columns: `{report['supply_demand']['available_columns']}`",
                f"- PTF columns: `{report['ptf']['available_columns']}`",
                "",
                "## Normalized Columns",
                "",
                f"- `{', '.join(report['normalized']['columns'])}`",
                "",
                "## Findings",
                "",
        f"- Clearing price column found: `{report['normalized']['clearing_price_column_found']}`",
        f"- Supply column present: `{report['normalized']['supply_demand_columns_present']['supply_mwh']}`",
        f"- Demand column present: `{report['normalized']['supply_demand_columns_present']['demand_mwh']}`",
        f"- Curve extraction usable: `{report['usable_for_curve_extraction']}`",
                "",
                "## Raw Artifacts",
                "",
                f"- Directory: `{OUT_DIR.relative_to(PROJECT_ROOT)}`",
                f"- PTF raw body: `{report['ptf']['raw_file']}`",
                f"- Supply-demand raw body: `{report['supply_demand']['raw_file']}`",
                "",
                "## Notes",
                "",
                "The smoke test only hits one date/hour. Raw response bodies and headers are preserved so the schema can be inspected before building any full historical pipeline.",
            ]
        )
        + "\n"
    )
    print(f"Wrote {REPORT_MD}")
    print(f"Wrote {REPORT_JSON}")
    if not normalized.empty:
        print(f"Wrote {NORMALIZED_CSV}")
        print(f"Wrote {NORMALIZED_PQ}")


if __name__ == "__main__":
    main()
