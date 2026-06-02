#!/usr/bin/env python3
"""
Enrich D+2 feature rows with plant must-run, real DAM curve lags, and proxy curve signals.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from build_d2_ptf_features import build_rows, write_reports as write_d2_reports

PROJECT_ROOT = Path(__file__).resolve().parent
D2_PATH = PROJECT_ROOT / "data" / "features" / "d2_ptf_features.parquet"
MUST_RUN_PATH = PROJECT_ROOT / "data" / "features" / "must_run_supply_features.parquet"
MUST_RUN_PROXY_PATH = PROJECT_ROOT / "data" / "features" / "must_run_proxy_v2.parquet"
PROXY_CURVE_PATH = PROJECT_ROOT / "data" / "features" / "real_supply_demand_curve_features.parquet"
CURVE_GLOB = "reconstructed_weekly_curve_features_*.parquet"

OUT_PATH = PROJECT_ROOT / "data" / "features" / "premium_d2_features.parquet"
REPORT_JSON = PROJECT_ROOT / "reports" / "premium_d2_feature_builder.json"
REPORT_MD = PROJECT_ROOT / "reports" / "premium_d2_feature_builder.md"

CURVE_COLS = [
    "slope_near_clearing",
    "elasticity_near_clearing",
    "curve_fragility_score",
    "volume_needed_for_100TL_move",
    "oversupply_pressure",
    "cap_risk_score",
    "zero_pressure_from_curve",
    "spike_pressure_from_curve",
    "reconstructed_clearing_price",
]


def parse_ts(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce")
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert("Europe/Istanbul").dt.tz_localize(None)
    return ts


def load_real_curves() -> pd.DataFrame:
    frames = []
    for path in sorted((PROJECT_ROOT / "data" / "features").glob(CURVE_GLOB)):
        from src.utils.io_utils import read_parquet_with_normalized_ts
        frame = read_parquet_with_normalized_ts(path)
        frame["delivery_hour"] = parse_ts(frame["delivery_hour"])
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    curve = pd.concat(frames, ignore_index=True)
    return curve.drop_duplicates("delivery_hour", keep="last").sort_values("delivery_hour")


def load_must_run() -> pd.DataFrame:
    for path in [MUST_RUN_PATH, MUST_RUN_PROXY_PATH]:
        if path.exists():
            frame = read_parquet_with_normalized_ts(path)
            frame["delivery_hour"] = parse_ts(frame["delivery_hour"])
            return frame
    return pd.DataFrame()


def attach_curve_lags(frame: pd.DataFrame, curves: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    stats = {"curve_lag48": 0, "curve_lag24": 0, "curve_lag168": 0}
    if frame.empty:
        return frame, stats
    out = frame.copy()
    out["ts_hour"] = parse_ts(out["ts_hour"])
    if curves.empty:
        return out, stats

    curve = curves.copy()
    curve = curve.rename(columns={"delivery_hour": "curve_hour"})
    for lag_hours, prefix in [(48, "curve_lag48"), (24, "curve_lag24"), (168, "curve_lag168")]:
        keys = out[["ts_hour"]].copy()
        keys["curve_hour"] = keys["ts_hour"] - pd.to_timedelta(lag_hours, unit="h")
        merged = keys.merge(curve[["curve_hour"] + [c for c in CURVE_COLS if c in curve.columns]], on="curve_hour", how="left")
        for col in CURVE_COLS:
            if col in merged.columns:
                out[f"{prefix}_{col}"] = merged[col].to_numpy()
        stats[prefix] = int(merged[[c for c in CURVE_COLS if c in merged.columns]].notna().any(axis=1).sum())
    return out, stats


def attach_proxy_curve(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or not PROXY_CURVE_PATH.exists():
        return frame
    proxy = read_parquet_with_normalized_ts(PROXY_CURVE_PATH)
    proxy["ts_hour"] = parse_ts(proxy["ts_hour"])
    keep = [
        "ts_hour",
        "slope_near_clearing",
        "elasticity_near_clearing",
        "curve_fragility_score",
        "oversupply_pressure",
        "cap_risk_score",
        "zero_pressure_from_curve",
        "spike_pressure_from_curve",
    ]
    keep = [c for c in keep if c in proxy.columns]
    if len(keep) <= 1:
        return frame
    lag = proxy[keep].copy()
    lag["ts_hour"] = lag["ts_hour"] + pd.to_timedelta(48, unit="h")
    lag = lag.rename(columns={c: f"proxy_curve_lag48_{c}" for c in keep if c != "ts_hour"})
    return frame.merge(lag, on="ts_hour", how="left")


def attach_must_run(frame: pd.DataFrame, must_run: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if frame.empty:
        return frame, 0
    out = frame.copy()
    matched = 0
    if not must_run.empty:
        out = out.merge(
            must_run.rename(columns={"delivery_hour": "ts_hour"}),
            on="ts_hour",
            how="left",
            suffixes=("", "_must_run"),
        )
        matched = int(out["must_run_supply"].notna().sum()) if "must_run_supply" in out.columns else 0
    if matched == 0 and not must_run.empty:
        lagged = must_run.copy()
        lagged["ts_hour"] = parse_ts(lagged["delivery_hour"]) + pd.to_timedelta(48, unit="h")
        lagged = lagged.drop(columns=["delivery_hour"], errors="ignore")
        out = out.drop(columns=[c for c in out.columns if c.endswith("_must_run")], errors="ignore")
        out = out.merge(lagged, on="ts_hour", how="left", suffixes=("", "_must_run_lag"))
        matched = int(out["must_run_supply"].notna().sum()) if "must_run_supply" in out.columns else 0
    if "must_run_share" in out.columns and "load_forecast" in out.columns:
        out["premium_must_run_pressure"] = out["must_run_share"].fillna(0) * out["load_forecast"].fillna(0)
    if "must_run_solar" in out.columns:
        out["premium_solar_must_run_ramp"] = out["must_run_solar"].fillna(0)
    return out, matched


def enrich(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    curves = load_real_curves()
    must_run = load_must_run()
    out, curve_stats = attach_curve_lags(frame, curves)
    out = attach_proxy_curve(out)
    out, must_run_matched = attach_must_run(out, must_run)

    if "anchor_d1_ptf" in out.columns:
        out["base_pred"] = out["anchor_d1_ptf"]
        out["ptf_lag_24"] = out["anchor_d1_ptf"]
    if "hour" in out.columns:
        h = out["hour"].astype(int)
        groups = pd.Series("other", index=out.index, dtype="object")
        groups[h.between(0, 5)] = "night"
        groups[h.between(6, 10)] = "morning"
        groups[h.between(11, 16)] = "solar_window"
        groups[h.between(17, 22)] = "evening_ramp"
        out["hour_group"] = groups

    for col in ["residual_load_forecast", "load_minus_kgup", "kgup_solar_mw", "must_run_supply"]:
        if col in out.columns:
            std = out[col].std()
            std = std if std and std > 0 else 1.0
            out[f"{col}_premium_z"] = ((out[col] - out[col].median()) / std).fillna(0)

    audit = {
        "rows": int(len(out)),
        "curve_lag_stats": curve_stats,
        "must_run_matched_rows": must_run_matched,
        "real_curve_max": str(curves["delivery_hour"].max()) if not curves.empty else None,
        "must_run_source": str(MUST_RUN_PATH if MUST_RUN_PATH.exists() else MUST_RUN_PROXY_PATH),
    }
    return out.sort_values("ts_hour"), audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-date", default=None)
    parser.add_argument("--reuse-d2", action="store_true", help="Read existing d2_ptf_features.parquet instead of rebuilding.")
    args = parser.parse_args()

    if args.reuse_d2 and D2_PATH.exists():
        base = read_parquet_with_normalized_ts(D2_PATH)
        base["ts_hour"] = parse_ts(base["ts_hour"])
        diagnostics = {"reused_d2_path": str(D2_PATH)}
    else:
        target = pd.to_datetime(args.target_date).date() if args.target_date else None
        if target is None:
            from datetime import date, timedelta

            target = date.today() + timedelta(days=2)
        base, diagnostics = build_rows(target)
        write_d2_reports(base, diagnostics)

    premium, audit = enrich(base)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    from src.utils.safe_io import atomic_parquet_write
    atomic_parquet_write(premium, str(OUT_PATH), index=False)
    audit["generated_at"] = datetime.now(timezone.utc).isoformat()
    audit["output_path"] = str(OUT_PATH.relative_to(PROJECT_ROOT))
    audit["input_diagnostics"] = diagnostics
    REPORT_JSON.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str) + "\n")
    REPORT_MD.write_text(
        "\n".join(
            [
                "# Premium D+2 Feature Builder",
                "",
                f"Generated: `{audit['generated_at']}`",
                f"- Rows: `{audit['rows']}`",
                f"- Must-run matched: `{audit['must_run_matched_rows']}`",
                f"- Curve lag stats: `{audit['curve_lag_stats']}`",
            ]
        )
        + "\n"
    )
    print(f"Wrote {OUT_PATH} rows={len(premium)}")


if __name__ == "__main__":
    main()
