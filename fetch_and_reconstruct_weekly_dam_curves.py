#!/usr/bin/env python3
"""
Fetch and reconstruct DAM supply-demand curves for a week (7 days, 24 hours each).

No historical backfill. No proxy fallback. No training.
"""

from __future__ import annotations

import hashlib
import argparse
import json
import os
import re
import sys
import time
from contextlib import contextmanager
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

START_DATE = "2026-06-01"
END_DATE = "2026-06-07"

OUT_DIR = PROJECT_ROOT / "data" / "raw" / "dam_supply_demand_curve_weekly" / f"{START_DATE}_{END_DATE}"
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / f"reconstructed_weekly_curve_features_{START_DATE}_{END_DATE}.parquet"
REPORT_MD = PROJECT_ROOT / "reports" / f"reconstructed_weekly_curve_{START_DATE}_{END_DATE}.md"
REPORT_JSON = PROJECT_ROOT / "reports" / f"reconstructed_weekly_curve_{START_DATE}_{END_DATE}.json"
DEBUG_DIR = PROJECT_ROOT / "reports" / "curve_debug_examples" / f"weekly_{START_DATE}_{END_DATE}"

REQUEST_TIMEOUT = (10, 120)
MAX_RETRIES = 4
RATE_LIMIT_BACKOFF_BASE = 5
DEFAULT_SLEEP_SECONDS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch and reconstruct weekly DAM curves with resume support.")
    parser.add_argument("--start-date", default=START_DATE)
    parser.add_argument("--end-date", default=END_DATE)
    parser.add_argument("--sleep-seconds", type=float, default=DEFAULT_SLEEP_SECONDS)
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--read-timeout", type=float, default=120.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only-missing", action="store_true")
    parser.add_argument(
        "--max-hours",
        type=int,
        default=0,
        help="Stop after this many newly completed hours. 0 means no limit.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=0,
        help="Stop after this many attempted hours, successful or failed. 0 means no limit.",
    )
    return parser.parse_args()


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def post_with_retries(
    url: str,
    max_retries: int = MAX_RETRIES,
    request_timeout: tuple[float, float] = REQUEST_TIMEOUT,
    **kwargs,
) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, timeout=request_timeout, **kwargs)
            if resp.status_code == 429:
                wait_s = min(120, RATE_LIMIT_BACKOFF_BASE * 2 ** (attempt - 1))
                log(f"rate-limit retry={attempt}/{max_retries} wait={wait_s}s url={url}")
                if attempt < max_retries:
                    time.sleep(wait_s)
                    continue
            return resp
        except requests.exceptions.RequestException as exc:
            last_error = exc
            wait_s = min(120, RATE_LIMIT_BACKOFF_BASE * 2 ** (attempt - 1))
            log(f"network retry={attempt}/{max_retries} wait={wait_s}s err={exc}")
            if attempt < max_retries:
                time.sleep(wait_s)
    raise last_error  # type: ignore[misc]


def obtain_tgt(username: str, password: str) -> str:
    resp = post_with_retries(
        LOGIN_URL,
        max_retries=MAX_RETRIES,
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
    zero_pressure = float(np.clip((50 - clearing_price) / 50.0, 0, 1) * np.clip(oversupply + max(0.0, 1.0 - fragility / 200.0), 0, 1))
    spike_pressure = float(np.clip((clearing_price - 1500) / 2500.0, 0, 1) * np.clip(fragility / 200.0, 0, 1))

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
        "zero_pressure_from_curve": zero_pressure,
        "spike_pressure_from_curve": spike_pressure,
    }
    return clearing_price, clearing_volume, {**meta, **features}


def plot_curve(price_axis: np.ndarray, s_on: np.ndarray, d_on: np.ndarray, clearing_price: float, clearing_volume: float, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.step(price_axis, s_on, where="post", color="#4cc9f0", label="Supply")
    ax.step(price_axis, d_on, where="post", color="#f94144", label="Demand")
    ax.scatter([clearing_price], [clearing_volume], color="#f9c74f", s=70, label="Clearing")
    ax.set_xlabel("Price")
    ax.set_ylabel("MWh")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def safe_ratio(num: int, den: int) -> float | None:
    return float(num / den) if den else None


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "start_date": None,
            "end_date": None,
            "completed_hours": [],
            "failed_hours": [],
            "pending_hours": [],
            "completed_hours_meta": {},
            "updated_at": None,
        }
    return json.loads(path.read_text(encoding="utf-8"))


def update_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(path, state)


def append_feature_row(features_path: Path, row: dict[str, Any]) -> None:
    features_path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame([row])
    if features_path.exists():
        from src.utils.io_utils import read_parquet_with_normalized_ts
        existing = read_parquet_with_normalized_ts(features_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    if "delivery_hour" in combined.columns:
        combined = combined.drop_duplicates("delivery_hour", keep="last").sort_values("delivery_hour")
    tmp_parquet = features_path.with_suffix(".parquet.tmp")
    from src.utils.safe_io import atomic_parquet_write
    atomic_parquet_write(combined, str(tmp_parquet), index=False)
    tmp_parquet.replace(features_path)
    csv_path = features_path.with_suffix(".csv")
    tmp_csv = csv_path.with_suffix(".csv.tmp")
    combined.to_csv(tmp_csv, index=False)
    tmp_csv.replace(csv_path)


@contextmanager
def file_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        raise RuntimeError(f"Lock file already exists: {lock_path}")
    lock_path.write_text(str(os.getpid()), encoding="utf-8")
    try:
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def main() -> None:
    args = parse_args()
    user = os.getenv("EPIAS_USERNAME")
    pw = os.getenv("EPIAS_PASSWORD")
    if not user or not pw:
        print("Missing EPIAS credentials.", file=sys.stderr)
        sys.exit(1)

    start_date = args.start_date
    end_date = args.end_date
    out_dir = PROJECT_ROOT / "data" / "raw" / "dam_supply_demand_curve_weekly" / f"{start_date}_{end_date}"
    features_path = PROJECT_ROOT / "data" / "features" / f"reconstructed_weekly_curve_features_{start_date}_{end_date}.parquet"
    report_md = PROJECT_ROOT / "reports" / f"reconstructed_weekly_curve_{start_date}_{end_date}.md"
    report_json = PROJECT_ROOT / "reports" / f"reconstructed_weekly_curve_{start_date}_{end_date}.json"
    debug_dir = PROJECT_ROOT / "reports" / "curve_debug_examples" / f"weekly_{start_date}_{end_date}"
    state_path = out_dir / f"state_{start_date}_{end_date}.json"
    lock_path = out_dir / "weekly_reconstruction.lock"

    with file_lock(lock_path):
        tgt = obtain_tgt(user, pw)
        log(f"TGT alındı: {tgt[:18]}...")
        out_dir.mkdir(parents=True, exist_ok=True)
        debug_dir.mkdir(parents=True, exist_ok=True)
        state = load_state(state_path)
        state.setdefault("start_date", start_date)
        state.setdefault("end_date", end_date)

        headers = {"Accept": "application/json", "Content-Type": "application/json", "TGT": tgt}
        all_rows: list[dict[str, Any]] = []
        hourly_reports: list[dict[str, Any]] = []
        failed_hours: list[str] = list(state.get("failed_hours", []))
        completed_hours: set[str] = set(state.get("completed_hours", []))
        pending_hours: list[str] = []
        completed_hours_meta: dict[str, Any] = state.get("completed_hours_meta", {})
        hourly_curve_errors: list[dict[str, Any]] = []
        newly_completed_this_run = 0
        attempted_this_run = 0
        stop_requested = False
        day_count = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days + 1

        for day_offset in range(day_count):
            if stop_requested:
                break
            current_day = (pd.Timestamp(start_date) + pd.Timedelta(days=day_offset)).date().isoformat()
            day_dir = out_dir / current_day
            day_dir.mkdir(parents=True, exist_ok=True)
            daily_dirs = [day_dir]

            for hour in range(24):
                if args.max_hours and newly_completed_this_run >= args.max_hours:
                    stop_requested = True
                    break
                if args.max_attempts and attempted_this_run >= args.max_attempts:
                    stop_requested = True
                    break
                hour_dt = f"{current_day}T{hour:02d}:00:00+03:00"
                if args.only_missing and hour_dt in completed_hours:
                    continue

                payload = {"date": hour_dt}
                hourly_raw_dir = day_dir / f"hour_{hour:02d}"
                supply_body = hourly_raw_dir.with_name(f"hour_{hour:02d}_supply.body.txt")
                ptf_body = hourly_raw_dir.with_name(f"hour_{hour:02d}_ptf.body.txt")
                if args.resume and supply_body.exists() and ptf_body.exists() and hour_dt in completed_hours:
                    continue

                try:
                    attempted_this_run += 1
                    s_resp = post_with_retries(
                        SUPPLY_DEMAND_URL,
                        max_retries=args.max_retries,
                        request_timeout=(args.connect_timeout, args.read_timeout),
                        json=payload,
                        headers=headers,
                    )
                    time.sleep(float(args.sleep_seconds))
                    p_resp = post_with_retries(
                        PTF_URL,
                        max_retries=args.max_retries,
                        request_timeout=(args.connect_timeout, args.read_timeout),
                        json=payload,
                        headers=headers,
                    )
                except Exception as exc:
                    failed_hours.append(hour_dt)
                    hourly_reports.append({"hour": hour_dt, "status": "request_failed", "error": str(exc)})
                    update_state(state_path, {
                        **state,
                        "start_date": start_date,
                        "end_date": end_date,
                        "completed_hours": sorted(completed_hours),
                        "failed_hours": failed_hours,
                        "pending_hours": pending_hours + [hour_dt],
                        "completed_hours_meta": completed_hours_meta,
                        "attempted_this_run": attempted_this_run,
                    })
                    continue

                save_raw(day_dir / f"hour_{hour:02d}_supply", s_resp, payload | {"endpoint": SUPPLY_DEMAND_URL})
                save_raw(day_dir / f"hour_{hour:02d}_ptf", p_resp, payload | {"endpoint": PTF_URL})

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
                    failed_hours.append(hour_dt)
                    hourly_reports.append({"hour": hour_dt, "status": "empty_supply"})
                    update_state(state_path, {
                        **state,
                        "start_date": start_date,
                        "end_date": end_date,
                        "completed_hours": sorted(completed_hours),
                        "failed_hours": failed_hours,
                        "pending_hours": pending_hours + [hour_dt],
                        "completed_hours_meta": completed_hours_meta,
                        "attempted_this_run": attempted_this_run,
                    })
                    continue

                try:
                    clearing_price, clearing_volume, meta = reconstruct_curve(raw)
                except Exception as exc:
                    failed_hours.append(hour_dt)
                    hourly_reports.append({"hour": hour_dt, "status": "reconstruction_failed", "error": str(exc)})
                    update_state(state_path, {
                        **state,
                        "start_date": start_date,
                        "end_date": end_date,
                        "completed_hours": sorted(completed_hours),
                        "failed_hours": failed_hours,
                        "pending_hours": pending_hours + [hour_dt],
                        "completed_hours_meta": completed_hours_meta,
                        "attempted_this_run": attempted_this_run,
                    })
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
                    **{k: meta[k] for k in ["slope_near_clearing", "elasticity_near_clearing", "curve_fragility_score", "volume_needed_for_100TL_move", "volume_needed_for_500TL_move", "oversupply_pressure", "cap_risk_score", "zero_pressure_from_curve", "spike_pressure_from_curve"]},
                    "supply_points": int(raw["supply_mwh"].notna().sum()) if "supply_mwh" in raw.columns else 0,
                    "demand_points": int(raw["demand_mwh"].notna().sum()) if "demand_mwh" in raw.columns else 0,
                    "status": "ok" if p_resp.status_code == 200 and s_resp.status_code == 200 else "partial",
                    "supply_status": s_resp.status_code,
                    "ptf_status": p_resp.status_code,
                    "raw_dir": str(day_dir.relative_to(PROJECT_ROOT)),
                }
                all_rows.append(row)
                append_feature_row(features_path, row)

                completed_hours.add(hour_dt)
                newly_completed_this_run += 1
                completed_hours_meta[hour_dt] = {
                    "mcpPrice": mcp_price,
                    "reconstructed_clearing_price": clearing_price,
                    "reconstruction_price_error": clearing_price - mcp_price,
                    "reconstruction_volume_error": clearing_volume - matching_quantity,
                }
                hourly_curve_errors.append(
                    {
                        "delivery_hour": hour_dt,
                        "mcpPrice": mcp_price,
                        "reconstructed_clearing_price": clearing_price,
                        "reconstruction_price_error": clearing_price - mcp_price,
                        "reconstruction_volume_error": clearing_volume - matching_quantity,
                        "status": "ok",
                    }
                )
                hourly_reports.append(
                    {
                        "hour": hour_dt,
                        "status": "ok",
                        "mcp_price": mcp_price,
                        "reconstructed_clearing_price": clearing_price,
                        "price_error": clearing_price - mcp_price,
                    }
                )

                plot_curve(
                    meta["price_axis"],
                    meta["s_on"],
                    meta["d_on"],
                    clearing_price,
                    clearing_volume,
                    debug_dir / current_day / f"hour_{hour:02d}_curve.png",
                )

                update_state(state_path, {
                    **state,
                    "start_date": start_date,
                    "end_date": end_date,
                        "completed_hours": sorted(completed_hours),
                        "failed_hours": sorted(set(failed_hours)),
                        "pending_hours": pending_hours,
                        "completed_hours_meta": completed_hours_meta,
                        "newly_completed_this_run": newly_completed_this_run,
                        "attempted_this_run": attempted_this_run,
                    })

                time.sleep(float(args.sleep_seconds))

        state["completed_hours"] = sorted(completed_hours)
        state["failed_hours"] = sorted(set(failed_hours) - completed_hours)
        state["pending_hours"] = pending_hours
        state["completed_hours_meta"] = completed_hours_meta
        state["newly_completed_last_run"] = newly_completed_this_run
        state["attempted_last_run"] = attempted_this_run
        update_state(state_path, state)

        features = read_parquet_with_normalized_ts(features_path) if features_path.exists() else pd.DataFrame(all_rows)
        ok = features[features["status"] == "ok"] if not features.empty else pd.DataFrame()
        abs_errors = ok["reconstruction_price_error"].abs() if not ok.empty else pd.Series(dtype=float)
        pending_hours_final = [f"{(pd.Timestamp(start_date) + pd.Timedelta(days=day_offset)).date().isoformat()}T{hour:02d}:00:00+03:00"
                               for day_offset in range(day_count)
                               for hour in range(24)
                               if f"{(pd.Timestamp(start_date) + pd.Timedelta(days=day_offset)).date().isoformat()}T{hour:02d}:00:00+03:00" not in completed_hours]
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "start_date": start_date,
            "end_date": end_date,
            "days_requested": int(day_count),
            "hours_requested": int(day_count * 24),
            "completed_hours": int(len(completed_hours)),
            "newly_completed_this_run": int(newly_completed_this_run),
            "attempted_this_run": int(attempted_this_run),
            "failed_hours_count": int(len(set(failed_hours) - completed_hours)),
            "pending_hours_count": int(len(pending_hours_final)),
            "failed_hours": sorted(set(failed_hours) - completed_hours),
            "pending_hours": pending_hours_final,
            "mean_reconstruction_error": float(abs_errors.mean()) if not abs_errors.empty else None,
            "max_reconstruction_error": float(abs_errors.max()) if not abs_errors.empty else None,
            "p50_reconstruction_error": float(abs_errors.quantile(0.5)) if not abs_errors.empty else None,
            "p90_reconstruction_error": float(abs_errors.quantile(0.9)) if not abs_errors.empty else None,
            "high_error_completed_hours": ok.loc[ok["reconstruction_price_error"].abs() > 10, "delivery_hour"].tolist() if not ok.empty else [],
            "zero_hours": ok.loc[ok["mcpPrice"] <= 0, "delivery_hour"].tolist() if not ok.empty else [],
            "low_price_hours": ok.loc[ok["mcpPrice"] <= 50, "delivery_hour"].tolist() if not ok.empty else [],
            "spike_hours": ok.loc[ok["mcpPrice"] >= 4000, "delivery_hour"].tolist() if not ok.empty else [],
            "average_error_over_completed": float(abs_errors.mean()) if not abs_errors.empty else None,
            "hourly_status": hourly_reports,
            "raw_directory": str(out_dir.relative_to(PROJECT_ROOT)),
            "features_path": str(features_path.relative_to(PROJECT_ROOT)),
            "debug_directory": str(debug_dir.relative_to(PROJECT_ROOT)),
            "state_path": str(state_path.relative_to(PROJECT_ROOT)),
            "command": f"python3 fetch_and_reconstruct_weekly_dam_curves.py --start-date {start_date} --end-date {end_date} --sleep-seconds {args.sleep_seconds} --resume --only-missing --max-hours {args.max_hours} --max-attempts {args.max_attempts}",
        }
        atomic_write_json(report_json, report)
        atomic_write_text(
            report_md,
            "\n".join(
                [
                f"# Reconstructed Weekly DAM Curves - {start_date} to {end_date}",
                "",
                f"Generated: `{report['generated_at']}`",
                "",
                "## Summary",
                "",
                f"- Days requested: `{report['days_requested']}`",
                f"- Hours requested: `{report['hours_requested']}`",
                f"- Completed hours: `{report['completed_hours']}`",
                f"- Newly completed this run: `{report['newly_completed_this_run']}`",
                f"- Attempted this run: `{report['attempted_this_run']}`",
                f"- Failed hours: `{report['failed_hours_count']}`",
                f"- Pending hours: `{report['pending_hours_count']}`",
                f"- Mean reconstruction error: `{report['mean_reconstruction_error']}`",
                f"- Max reconstruction error: `{report['max_reconstruction_error']}`",
                f"- P50 reconstruction error: `{report['p50_reconstruction_error']}`",
                f"- P90 reconstruction error: `{report['p90_reconstruction_error']}`",
                f"- Average error over completed: `{report['average_error_over_completed']}`",
                "",
                "## Regime Checks",
                "",
                f"- Zero-price hours: `{report['zero_hours']}`",
                f"- Low-price hours: `{report['low_price_hours']}`",
                f"- Spike hours: `{report['spike_hours']}`",
                f"- High-error completed hours (>10 TL): `{report['high_error_completed_hours']}`",
                f"- Failed hours: `{report['failed_hours']}`",
                f"- Pending hours: `{report['pending_hours']}`",
                "",
                "## Files",
                "",
                f"- Raw directory: `{report['raw_directory']}`",
                f"- Features: `{report['features_path']}`",
                f"- Debug plot directory: `{report['debug_directory']}`",
                f"- State file: `{report['state_path']}`",
                "",
                "## Resume",
                "",
                f"- Command: `{report['command']}`",
            ]
        )
        + "\n"
        )

        # Weekly chart: actual vs reconstructed PTF and error
        if not ok.empty:
            ok_sorted = ok.copy()
            ok_sorted["delivery_hour_dt"] = pd.to_datetime(ok_sorted["delivery_hour"])
            ok_sorted = ok_sorted.sort_values("delivery_hour_dt")
            fig, ax1 = plt.subplots(figsize=(13, 6))
            ax1.plot(ok_sorted["delivery_hour_dt"], ok_sorted["mcpPrice"], label="EPİAŞ mcpPrice", color="#f94144", linewidth=2)
            ax1.plot(
                ok_sorted["delivery_hour_dt"],
                ok_sorted["reconstructed_clearing_price"],
                label="Reconstructed Clearing Price",
                color="#4cc9f0",
                linewidth=2,
            )
            ax1.set_ylabel("Price")
            ax1.set_xlabel("Delivery Hour")
            ax1.grid(alpha=0.2)
            ax1.legend(loc="upper left")
            ax2 = ax1.twinx()
            ax2.bar(
                ok_sorted["delivery_hour_dt"],
                ok_sorted["reconstruction_price_error"].abs(),
                width=0.03,
                color="#f9c74f",
                alpha=0.25,
                label="Abs Error",
            )
            ax2.set_ylabel("Absolute Error")
            fig.autofmt_xdate()
            fig.tight_layout()
            debug_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(debug_dir / "weekly_actual_vs_reconstructed_ptf.png", dpi=180)
            plt.close(fig)

        print(f"Wrote {features_path}")
        print(f"Wrote {report_md}")
        print(f"Wrote {report_json}")
        if pending_hours_final:
            log(f"Run incomplete. Pending hours: {len(pending_hours_final)}")


if __name__ == "__main__":
    main()
