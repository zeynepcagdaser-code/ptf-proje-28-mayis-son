#!/usr/bin/env python3
"""
Fetch and reconstruct DAM supply-demand curves for a single day (24 hours).

No historical backfill. No proxy fallback. No training.
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

LOGIN_URL = "https://giris.epias.com.tr/cas/v1/tickets"
SUPPLY_DEMAND_URL = "https://seffaflik.epias.com.tr/electricity-service/v1/markets/dam/data/supply-demand-chart-data"
PTF_URL = "https://seffaflik.epias.com.tr/electricity-service/v1/markets/dam/data/supply-demand-chart-ptf-data"

TARGET_DATE = "2026-06-01"
OUT_DIR = PROJECT_ROOT / "data" / "raw" / "dam_supply_demand_curve_daily" / TARGET_DATE
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / f"reconstructed_daily_curve_features_{TARGET_DATE}.parquet"
REPORT_MD = PROJECT_ROOT / "reports" / f"reconstructed_daily_curve_{TARGET_DATE}.md"
REPORT_JSON = PROJECT_ROOT / "reports" / f"reconstructed_daily_curve_{TARGET_DATE}.json"
DEBUG_DIR = PROJECT_ROOT / "reports" / "curve_debug_examples" / TARGET_DATE

REQUEST_TIMEOUT = (10, 120)
MAX_RETRIES = 4


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


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


def save_raw(path: Path, response: requests.Response, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".body.txt").write_text(response.text, encoding="utf-8")
    meta = {
        "status_code": response.status_code,
        "content_type": response.headers.get("Content-Type"),
        "response_hash": hashlib.sha256(response.text.encode("utf-8", errors="ignore")).hexdigest(),
        "payload": payload,
        "endpoint": payload.get("endpoint"),
    }
    path.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str) + "\n")


def normalize_supply_demand(raw_items: list[dict[str, Any]], delivery_hour: str, endpoint: str) -> pd.DataFrame:
    supply_rows = [x for x in raw_items if "supplyPrice" in x]
    demand_rows = [x for x in raw_items if "demandPrice" in x]
    sup = pd.DataFrame(supply_rows).rename(columns={"amount": "supply_mwh", "supplyPrice": "price"})
    dem = pd.DataFrame(demand_rows).rename(columns={"amount": "demand_mwh", "demandPrice": "price"})

    out_rows = []
    for _, row in sup.iterrows():
        out_rows.append(
            {
                "delivery_hour": delivery_hour,
                "price": row.get("price"),
                "supply_mwh": row.get("supply_mwh"),
                "demand_mwh": np.nan,
                "source_endpoint": endpoint,
            }
        )
    for _, row in dem.iterrows():
        out_rows.append(
            {
                "delivery_hour": delivery_hour,
                "price": row.get("price"),
                "supply_mwh": np.nan,
                "demand_mwh": row.get("demand_mwh"),
                "source_endpoint": endpoint,
            }
        )
    return pd.DataFrame(out_rows)


def build_axis_tables(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    supply = raw[raw["supply_mwh"].notna()].copy()
    demand = raw[raw["demand_mwh"].notna()].copy()
    supply = supply.rename(columns={"supply_mwh": "qty"}).sort_values("price").reset_index(drop=True)
    demand = demand.rename(columns={"demand_mwh": "qty"}).sort_values("price").reset_index(drop=True)
    return supply, demand


def piecewise_step(x: np.ndarray, xp: np.ndarray, yp: np.ndarray, *, side: str) -> np.ndarray:
    if len(xp) == 0:
        return np.full_like(x, np.nan, dtype=float)
    if side == "left":
        idx = np.searchsorted(xp, x, side="right") - 1
    else:
        idx = np.searchsorted(xp, x, side="left")
    idx = np.clip(idx, 0, len(yp) - 1)
    return yp[idx]


def reconstruct_curve(raw: pd.DataFrame) -> tuple[float, float, dict[str, Any]]:
    supply, demand = build_axis_tables(raw)
    if supply.empty or demand.empty:
        raise RuntimeError("Supply or demand curve is empty.")

    price_axis = np.unique(np.concatenate([supply["price"].to_numpy(float), demand["price"].to_numpy(float)]))
    price_axis.sort()
    s_on = piecewise_step(price_axis, supply["price"].to_numpy(float), supply["qty"].to_numpy(float), side="left")
    d_on = piecewise_step(price_axis, demand["price"].to_numpy(float), demand["qty"].to_numpy(float), side="right")
    diff = s_on - d_on
    crossings = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(crossings):
        i = crossings[0]
        x0, x1 = price_axis[i], price_axis[i + 1]
        y0, y1 = diff[i], diff[i + 1]
        clearing_price = float(x0 if y1 == y0 else x0 - y0 * (x1 - x0) / (y1 - y0))
    else:
        clearing_price = float(price_axis[int(np.nanargmin(np.abs(diff)))])

    supply_at = float(np.interp(clearing_price, supply["price"].to_numpy(float), supply["qty"].to_numpy(float)))
    demand_at = float(np.interp(clearing_price, demand["price"].to_numpy(float), demand["qty"].to_numpy(float)))
    clearing_volume = float((supply_at + demand_at) / 2.0)

    idx = int(np.nanargmin(np.abs(price_axis - clearing_price)))
    prev_i = max(idx - 1, 0)
    next_i = min(idx + 1, len(price_axis) - 1)
    slope = float((s_on[next_i] - s_on[prev_i]) / max(price_axis[next_i] - price_axis[prev_i], 1e-9))
    elasticity = float(abs((d_on[next_i] - d_on[prev_i]) / max(price_axis[next_i] - price_axis[prev_i], 1e-9)))
    fragility = float(np.clip(abs(slope) + abs(elasticity), 0, 1e6))
    vol_100 = float(abs(np.interp(clearing_price + 100, price_axis, s_on) - np.interp(clearing_price, price_axis, s_on)))
    vol_500 = float(abs(np.interp(clearing_price + 500, price_axis, s_on) - np.interp(clearing_price, price_axis, s_on)))
    oversupply = float(np.clip((supply_at - demand_at) / max(clearing_volume, 1.0), 0, 1))
    cap_risk = float(np.clip((clearing_price - 3500) / 800.0, 0, 1))

    meta = {
        "price_axis": price_axis,
        "s_on": s_on,
        "d_on": d_on,
        "crossings": int(len(crossings)),
    }
    features = {
        "slope_near_clearing": slope,
        "elasticity_near_clearing": elasticity,
        "curve_fragility_score": fragility,
        "volume_needed_for_100TL_move": vol_100,
        "volume_needed_for_500TL_move": vol_500,
        "oversupply_pressure": oversupply,
        "cap_risk_score": cap_risk,
    }
    return clearing_price, clearing_volume, {**meta, **features}


def plot_curve(price_axis: np.ndarray, s_on: np.ndarray, d_on: np.ndarray, clearing_price: float, clearing_volume: float) -> None:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.step(price_axis, s_on, where="post", color="#4cc9f0", label="Supply")
    ax.step(price_axis, d_on, where="post", color="#f94144", label="Demand")
    ax.scatter([clearing_price], [clearing_volume], color="#f9c74f", s=70, label="Clearing")
    ax.set_xlabel("Price")
    ax.set_ylabel("MWh")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(DEBUG_DIR / "daily_2026-06-01_curve.png", dpi=180)
    plt.close(fig)


def main() -> None:
    user = os.getenv("EPIAS_USERNAME")
    pw = os.getenv("EPIAS_PASSWORD")
    if not user or not pw:
        print("Missing EPIAS credentials.", file=sys.stderr)
        sys.exit(1)

    tgt = obtain_tgt(user, pw)
    log(f"TGT alındı: {tgt[:18]}...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    headers = {"Accept": "application/json", "Content-Type": "application/json", "TGT": tgt}
    all_rows = []
    daily_reports = []
    for hour in range(24):
        hour_dt = f"{TARGET_DATE}T{hour:02d}:00:00+03:00"
        payload = {"date": hour_dt, "endpoint": SUPPLY_DEMAND_URL}

        s_resp = post_with_retries(SUPPLY_DEMAND_URL, json={"date": hour_dt}, headers=headers)
        p_resp = post_with_retries(PTF_URL, json={"date": hour_dt}, headers=headers)
        save_raw(OUT_DIR / f"hour_{hour:02d}_supply", s_resp, payload | {"endpoint": SUPPLY_DEMAND_URL})
        save_raw(OUT_DIR / f"hour_{hour:02d}_ptf", p_resp, payload | {"endpoint": PTF_URL})

        s_items = s_resp.json().get("items", []) if s_resp.status_code == 200 else []
        p_obj = p_resp.json() if p_resp.status_code == 200 else {}
        if isinstance(p_obj, dict) and "body" in p_obj:
            nested = p_obj["body"]
            if isinstance(nested, dict):
                content = nested.get("content", {})
                if isinstance(content, dict) and isinstance(content.get("response"), dict):
                    p_items = [content["response"]]
                else:
                    p_items = []
            else:
                p_items = []
        elif isinstance(p_obj, dict) and "items" in p_obj:
            p_items = p_obj.get("items", [])
        else:
            p_items = []

        raw = normalize_supply_demand(s_items, hour_dt, SUPPLY_DEMAND_URL)
        if raw.empty:
            daily_reports.append({"hour": hour_dt, "status": "empty_supply"})
            continue

        try:
            clearing_price, clearing_volume, meta = reconstruct_curve(raw)
        except Exception as exc:
            daily_reports.append({"hour": hour_dt, "status": "reconstruction_failed", "error": str(exc)})
            continue

        ptf_row = p_items[0] if p_items else {}
        mcp_price = float(ptf_row.get("mcpPrice", np.nan))
        matching_quantity = float(ptf_row.get("matchingQuantity", np.nan))
        row = {
            "delivery_hour": hour_dt,
            "mcpPrice": mcp_price,
            "matchingQuantity": matching_quantity,
            "reconstructed_clearing_price": clearing_price,
            "reconstructed_clearing_volume": clearing_volume,
            "reconstruction_price_error": clearing_price - mcp_price,
            "reconstruction_volume_error": clearing_volume - matching_quantity,
            **{k: meta[k] for k in ["slope_near_clearing", "elasticity_near_clearing", "curve_fragility_score", "volume_needed_for_100TL_move", "volume_needed_for_500TL_move", "oversupply_pressure", "cap_risk_score"]},
            "supply_points": int(raw["supply_mwh"].notna().sum()) if "supply_mwh" in raw.columns else 0,
            "demand_points": int(raw["demand_mwh"].notna().sum()) if "demand_mwh" in raw.columns else 0,
            "status": "ok" if p_resp.status_code == 200 and s_resp.status_code == 200 else "partial",
            "supply_status": s_resp.status_code,
            "ptf_status": p_resp.status_code,
        }
        all_rows.append(row)

        plot_curve(meta["price_axis"], meta["s_on"], meta["d_on"], clearing_price, clearing_volume)
        daily_reports.append(
            {
                "hour": hour_dt,
                "status": "ok",
                "mcp_price": mcp_price,
                "reconstructed_clearing_price": clearing_price,
                "price_error": clearing_price - mcp_price,
            }
        )

    features = pd.DataFrame(all_rows)
    if not features.empty:
        from src.utils.safe_io import atomic_parquet_write
        atomic_parquet_write(features, str(FEATURES_PATH), index=False)

    ok = features[features["status"] == "ok"] if not features.empty else pd.DataFrame()
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "date": TARGET_DATE,
        "hours_requested": 24,
        "hours_successful": int(len(ok)),
        "mean_reconstruction_error": float(ok["reconstruction_price_error"].abs().mean()) if not ok.empty else None,
        "max_reconstruction_error": float(ok["reconstruction_price_error"].abs().max()) if not ok.empty else None,
        "hours_high_error": ok.loc[ok["reconstruction_price_error"].abs() > 10, "delivery_hour"].tolist() if not ok.empty else [],
        "zero_hours": ok.loc[ok["mcpPrice"] <= 0, "delivery_hour"].tolist() if not ok.empty else [],
        "low_price_hours": ok.loc[ok["mcpPrice"] <= 50, "delivery_hour"].tolist() if not ok.empty else [],
        "spike_hours": ok.loc[ok["mcpPrice"] >= 4000, "delivery_hour"].tolist() if not ok.empty else [],
        "hourly_status": daily_reports,
    }
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n")
    REPORT_MD.write_text(
        "\n".join(
            [
                f"# Reconstructed Daily DAM Curves - {TARGET_DATE}",
                "",
                f"Generated: `{report['generated_at']}`",
                "",
                "## Summary",
                "",
                f"- Hours requested: `{report['hours_requested']}`",
                f"- Hours successful: `{report['hours_successful']}`",
                f"- Mean reconstruction error: `{report['mean_reconstruction_error']}`",
                f"- Max reconstruction error: `{report['max_reconstruction_error']}`",
                "",
                "## Regime Checks",
                "",
                f"- Zero-price hours: `{report['zero_hours']}`",
                f"- Low-price hours: `{report['low_price_hours']}`",
                f"- Spike hours: `{report['spike_hours']}`",
                f"- High-error hours (>10 TL): `{report['hours_high_error']}`",
                "",
                "## Files",
                "",
                f"- Raw directory: `{OUT_DIR.relative_to(PROJECT_ROOT)}`",
                f"- Features: `{FEATURES_PATH.relative_to(PROJECT_ROOT)}`",
                f"- Debug plot directory: `{DEBUG_DIR.relative_to(PROJECT_ROOT)}`",
            ]
        )
        + "\n"
    )
    print(f"Wrote {FEATURES_PATH}")
    print(f"Wrote {REPORT_MD}")
    print(f"Wrote {REPORT_JSON}")


if __name__ == "__main__":
    main()
