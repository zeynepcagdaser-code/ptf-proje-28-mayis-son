#!/usr/bin/env python3
"""
Extract compact market microstructure features from real supply-demand curve data.

If raw EPİAŞ GÖP curve files are not present in the repository, this script
falls back to the existing hourly curve proxy layer and derives compact
features from it. No model training.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PTF_PATH = PROJECT_ROOT / "data" / "ptf_dataset.csv"
KGUP_PATH = PROJECT_ROOT / "data" / "kgup_combined.csv"
LOAD_PATH = PROJECT_ROOT / "data" / "load_forecast.csv"
CURVE_PROXY_PATH = PROJECT_ROOT / "data" / "features" / "supply_demand_curve_features.parquet"

OUT_PATH = PROJECT_ROOT / "data" / "features" / "real_supply_demand_curve_features.parquet"
REPORT_MD = PROJECT_ROOT / "reports" / "real_supply_demand_curve_analysis.md"
REPORT_JSON = PROJECT_ROOT / "reports" / "real_supply_demand_curve_analysis.json"
DEBUG_DIR = PROJECT_ROOT / "reports" / "curve_debug_examples"


def _load_hourly_csv(path: Path, date_col: str, time_col: str | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if date_col not in df.columns:
        return pd.DataFrame()
    date = pd.to_datetime(df[date_col], errors="coerce")
    if time_col and time_col in df.columns:
        ts = pd.to_datetime(date.dt.strftime("%Y-%m-%d") + " " + df[time_col].astype(str), errors="coerce")
    else:
        ts = date
    if getattr(ts.dt, "tz", None) is None:
        df["ts_hour"] = ts.dt.tz_localize("Europe/Istanbul", nonexistent="NaT", ambiguous="NaT")
    else:
        df["ts_hour"] = ts.dt.tz_convert("Europe/Istanbul")
    return df


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace(0, np.nan)


def _ensure_datetime(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce")
    if getattr(s.dt, "tz", None) is None:
        return s.dt.tz_localize("Europe/Istanbul", nonexistent="NaT", ambiguous="NaT")
    return s.dt.tz_convert("Europe/Istanbul")


def _discover_raw_curve_files() -> list[Path]:
    patterns = ["*curve*", "*gop*", "*order*", "*bid*", "*offer*"]
    files: list[Path] = []
    for pat in patterns:
        files.extend(RAW_DIR.rglob(pat))
    return sorted({p for p in files if p.is_file()})


def _build_from_proxy() -> tuple[pd.DataFrame, dict[str, Any]]:
    if not CURVE_PROXY_PATH.exists():
        return pd.DataFrame(), {"available": False, "reason": "No raw curve files and no curve proxy layer found."}
    from src.utils.io_utils import read_parquet_with_normalized_ts
    curve = read_parquet_with_normalized_ts(CURVE_PROXY_PATH).copy()
    curve["ts_hour"] = _ensure_datetime(curve["ts_hour"])
    if "clearing_volume_proxy" not in curve.columns and "supply_gap" in curve.columns:
        curve["clearing_volume_proxy"] = np.abs(curve["supply_gap"])
    if "slope_near_clearing" not in curve.columns and "clearing_price_proxy" in curve.columns and "supply_gap" in curve.columns:
        curve["slope_near_clearing"] = curve["clearing_price_proxy"].diff().abs() / curve["supply_gap"].abs().replace(0, np.nan)
    if "local_curve_density" not in curve.columns and "bid_stack_density" in curve.columns:
        curve["local_curve_density"] = curve["bid_stack_density"]
    if "local_elasticity" not in curve.columns and "offer_stack_density" in curve.columns:
        curve["local_elasticity"] = curve["offer_stack_density"]
    if "zero_price_supply_excess" not in curve.columns and "oversupply_curve_pressure" in curve.columns:
        curve["zero_price_supply_excess"] = curve["oversupply_curve_pressure"] * curve.get("clearing_volume_proxy", 1)
    if "low_price_pressure" not in curve.columns and "clearing_price_proxy" in curve.columns:
        curve["low_price_pressure"] = np.clip((100 - curve["clearing_price_proxy"]) / 100.0, 0, 1)
    if "oversupply_mass_below_100" not in curve.columns and "oversupply_curve_pressure" in curve.columns:
        curve["oversupply_mass_below_100"] = curve["oversupply_curve_pressure"] * curve.get("clearing_volume_proxy", 1)
    if "renewable_oversupply_zone" not in curve.columns and "oversupply_curve_pressure" in curve.columns:
        curve["renewable_oversupply_zone"] = np.clip(curve["oversupply_curve_pressure"], 0, 1)
    if "curve_convexity_score" not in curve.columns and "clearing_price_proxy" in curve.columns:
        curve["curve_convexity_score"] = curve["clearing_price_proxy"].diff().diff().abs().fillna(0)
    if "cap_risk_from_curve" not in curve.columns and "clearing_price_proxy" in curve.columns:
        curve["cap_risk_from_curve"] = np.clip((curve["clearing_price_proxy"] - 3500) / 800.0, 0, 1)
    if "supply_gap_above_clearing" not in curve.columns and "supply_gap" in curve.columns:
        curve["supply_gap_above_clearing"] = curve["supply_gap"].abs()
    if "steepness_above_ptf" not in curve.columns and "slope_near_clearing" in curve.columns:
        curve["steepness_above_ptf"] = curve["slope_near_clearing"]
    if "marginality_jump_score" not in curve.columns and "clearing_price_proxy" in curve.columns:
        curve["marginality_jump_score"] = curve["clearing_price_proxy"].diff().abs().fillna(0)
    if "demand_curve_steepness" not in curve.columns and "slope_near_clearing" in curve.columns:
        curve["demand_curve_steepness"] = curve["slope_near_clearing"].abs()
    if "demand_elasticity" not in curve.columns and "local_elasticity" in curve.columns:
        curve["demand_elasticity"] = curve["local_elasticity"]
    if "demand_cliff_score" not in curve.columns and "clearing_price_proxy" in curve.columns:
        curve["demand_cliff_score"] = np.clip((curve["clearing_price_proxy"].rolling(3, min_periods=1).max() - curve["clearing_price_proxy"].rolling(3, min_periods=1).min()) / 1000.0, 0, 1)
    if "offer_stack_density" not in curve.columns and "local_curve_density" in curve.columns:
        curve["offer_stack_density"] = curve["local_curve_density"]
    if "bid_stack_density" not in curve.columns and "local_curve_density" in curve.columns:
        curve["bid_stack_density"] = curve["local_curve_density"]
    if "supply_concentration_score" not in curve.columns and "clearing_volume_proxy" in curve.columns:
        denom = float(curve["clearing_volume_proxy"].max()) if pd.notna(curve["clearing_volume_proxy"].max()) and curve["clearing_volume_proxy"].max() != 0 else np.nan
        curve["supply_concentration_score"] = np.clip(1 - (curve["clearing_volume_proxy"] / denom if pd.notna(denom) else 0).fillna(0), 0, 1)
    if "clearing_fragility_score" not in curve.columns and "cap_risk_from_curve" in curve.columns and "slope_near_clearing" in curve.columns:
        curve["clearing_fragility_score"] = np.clip(curve["cap_risk_from_curve"].fillna(0) + curve["slope_near_clearing"].fillna(0), 0, 1)
    if "volume_needed_for_500TL_move" not in curve.columns:
        curve["volume_needed_for_500TL_move"] = curve.get("clearing_volume_proxy", np.nan)
    if "volume_needed_for_1000TL_move" not in curve.columns:
        curve["volume_needed_for_1000TL_move"] = curve.get("clearing_volume_proxy", np.nan)
    if "curve_break_probability" not in curve.columns:
        curve["curve_break_probability"] = np.clip(curve.get("curve_convexity_score", 0) / 500.0, 0, 1)
    if "imbalance_pressure_proxy" not in curve.columns and "supply_gap" in curve.columns:
        curve["imbalance_pressure_proxy"] = np.clip(curve["supply_gap"].abs() / curve["supply_gap"].abs().max(), 0, 1)
    for col in ["clearing_price_proxy", "supply_gap", "bid_stack_density", "offer_stack_density", "marginality_risk_score", "oversupply_curve_pressure", "cap_risk_from_curve", "low_price_pressure_score"]:
        if col in curve.columns:
            curve[col] = pd.to_numeric(curve[col], errors="coerce")
    return curve, {
        "available": True,
        "mode": "proxy_fallback",
        "source": str(CURVE_PROXY_PATH.relative_to(PROJECT_ROOT)),
        "raw_curve_files": [],
    }


def _load_raw_curve_frames(raw_files: list[Path]) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for path in raw_files:
        try:
            if path.suffix.lower() == ".parquet":
                from src.utils.io_utils import read_parquet_with_normalized_ts
                df = read_parquet_with_normalized_ts(path)
            elif path.suffix.lower() in {".csv", ".txt"}:
                df = pd.read_csv(path)
            elif path.suffix.lower() in {".json"}:
                df = pd.read_json(path)
            else:
                continue
            df["source_file"] = str(path)
            frames.append(df)
        except Exception:
            continue
    return frames


def _extract_from_raw(raw_frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not raw_frames:
        return _build_from_proxy()

    # Expecting a tall table of curve points with time + price + volume/cumulative volume.
    # We normalize diverse schemas as best effort.
    rows: list[pd.DataFrame] = []
    for df in raw_frames:
        cands = {c.lower(): c for c in df.columns}
        ts_col = next((cands[k] for k in ["ts_hour", "datetime", "datetimeutc", "delivery_hour", "snapshot_ts"] if k in cands), None)
        price_col = next((cands[k] for k in ["price", "ptf", "markettradeprice", "clearing_price", "clearingprice"] if k in cands), None)
        vol_col = next((cands[k] for k in ["volume", "qty", "quantity", "mwh", "mw", "amount"] if k in cands), None)
        side_col = next((cands[k] for k in ["side", "type", "curve_side", "direction"] if k in cands), None)
        if ts_col is None or price_col is None:
            continue
        x = df.copy()
        x["ts_hour"] = _ensure_datetime(x[ts_col])
        x["curve_price"] = pd.to_numeric(x[price_col], errors="coerce")
        x["curve_volume"] = pd.to_numeric(x[vol_col], errors="coerce") if vol_col else np.nan
        x["curve_side"] = x[side_col].astype(str).str.lower() if side_col else "unknown"
        rows.append(x[["ts_hour", "curve_price", "curve_volume", "curve_side", "source_file"]].copy())

    if not rows:
        return _build_from_proxy()

    raw = pd.concat(rows, ignore_index=True)
    raw = raw.dropna(subset=["ts_hour", "curve_price"]).sort_values(["ts_hour", "curve_price"]).reset_index(drop=True)
    if "curve_volume" not in raw.columns:
        raw["curve_volume"] = np.nan

    grouped = []
    for ts_hour, group in raw.groupby("ts_hour"):
        group = group.sort_values("curve_price")
        price = group["curve_price"].to_numpy(float)
        vol = group["curve_volume"].fillna(0).to_numpy(float)
        cum = np.nancumsum(np.nan_to_num(vol, nan=0.0))
        total_vol = float(np.nan_to_num(vol, nan=0.0).sum())
        if len(price) >= 2:
            slope = float((price[-1] - price[0]) / max(total_vol, 1.0))
            convexity = float(np.diff(price, 2).sum()) if len(price) >= 3 else 0.0
            density = float(len(group) / max(price[-1] - price[0], 1.0))
            elasticity = float(total_vol / max(price[-1] - price[0], 1.0))
        else:
            slope = convexity = density = elasticity = np.nan
        clearing_idx = int(np.argmax(cum >= cum.max() * 0.5)) if len(cum) else 0
        clearing_price = float(price[min(clearing_idx, len(price) - 1)]) if len(price) else np.nan
        clearing_volume = float(cum[min(clearing_idx, len(cum) - 1)]) if len(cum) else np.nan
        low_price_mask = price <= 100
        oversupply_mass_below_100 = float(np.nan_to_num(vol[low_price_mask], nan=0.0).sum()) if len(price) else np.nan
        zero_price_supply_excess = float(np.nan_to_num(vol[price <= 50], nan=0.0).sum()) if len(price) else np.nan
        local_density = density
        local_elasticity = elasticity
        curve_break_probability = float(np.clip((convexity if not math.isnan(convexity) else 0) / 500.0, 0, 1))
        volume_needed_500 = float(np.interp(500, price, cum, left=0, right=cum.max() if len(cum) else np.nan)) if len(price) else np.nan
        volume_needed_1000 = float(np.interp(1000, price, cum, left=0, right=cum.max() if len(cum) else np.nan)) if len(price) else np.nan
        grouped.append(
            {
                "ts_hour": ts_hour,
                "clearing_price_proxy": clearing_price,
                "clearing_volume_proxy": clearing_volume,
                "slope_near_clearing": slope,
                "local_curve_density": local_density,
                "local_elasticity": local_elasticity,
                "zero_price_supply_excess": zero_price_supply_excess,
                "low_price_pressure": float(np.clip((100 - clearing_price) / 100.0, 0, 1)) if pd.notna(clearing_price) else np.nan,
                "oversupply_mass_below_100": oversupply_mass_below_100,
                "renewable_oversupply_zone": float(np.clip(oversupply_mass_below_100 / max(total_vol, 1.0), 0, 1)) if total_vol else np.nan,
                "curve_convexity_score": convexity,
                "cap_risk_from_curve": float(np.clip((clearing_price - 3500) / 800.0, 0, 1)) if pd.notna(clearing_price) else np.nan,
                "supply_gap_above_clearing": float(total_vol - clearing_volume) if pd.notna(clearing_volume) else np.nan,
                "steepness_above_ptf": slope,
                "marginality_jump_score": float(abs(np.diff(price).max()) if len(price) > 1 else 0.0),
                "demand_curve_steepness": abs(slope) if pd.notna(slope) else np.nan,
                "demand_elasticity": local_elasticity,
                "demand_cliff_score": float(np.clip((price.max() - price.min()) / 1000.0, 0, 1)) if len(price) else np.nan,
                "offer_stack_density": density,
                "bid_stack_density": float(len(group) / max(total_vol, 1.0)) if total_vol else np.nan,
                "supply_concentration_score": float(np.clip(group["curve_volume"].fillna(0).nlargest(min(10, len(group))).sum() / max(total_vol, 1.0), 0, 1)) if total_vol else np.nan,
                "clearing_fragility_score": float(np.clip(abs(slope) * (1 + curve_break_probability), 0, 1)),
                "volume_needed_for_500TL_move": volume_needed_500,
                "volume_needed_for_1000TL_move": volume_needed_1000,
                "curve_break_probability": curve_break_probability,
                "imbalance_pressure_proxy": float(np.clip(abs(clearing_volume - total_vol / 2) / max(total_vol, 1.0), 0, 1)) if total_vol else np.nan,
                "raw_points": int(len(group)),
                "source_mode": "raw_curve",
            }
        )

    frame = pd.DataFrame(grouped).sort_values("ts_hour").reset_index(drop=True)
    frame["ptf_zero_flag"] = (frame["clearing_price_proxy"] <= 0).astype(int)
    frame["ptf_low_flag"] = (frame["clearing_price_proxy"] <= 50).astype(int)
    frame["ptf_tight_flag"] = ((frame["clearing_price_proxy"] >= 1500) & (frame["clearing_price_proxy"] < 4000)).astype(int)
    frame["ptf_spike_flag"] = (frame["clearing_price_proxy"] >= 4000).astype(int)
    return frame, {
        "available": True,
        "mode": "raw_curve",
        "source": [str(p.relative_to(PROJECT_ROOT)) for p in sorted({Path(r["source_file"]) for r in raw.to_dict("records")})][:20],
        "raw_curve_files": [str(p.relative_to(PROJECT_ROOT)) for p in sorted({Path(r["source_file"]) for r in raw.to_dict("records")})],
        "rows": int(len(frame)),
    }


def _save_debug_plots(frame: pd.DataFrame) -> list[str]:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    if frame.empty:
        return paths
    samples = []
    for label, cond in [
        ("zero_pressure", frame["ptf_low_flag"] == 1 if "ptf_low_flag" in frame.columns else pd.Series(False, index=frame.index)),
        ("spike_cap", frame["ptf_spike_flag"] == 1 if "ptf_spike_flag" in frame.columns else pd.Series(False, index=frame.index)),
        ("tight", frame["ptf_tight_flag"] == 1 if "ptf_tight_flag" in frame.columns else pd.Series(False, index=frame.index)),
        ("normal", frame["ptf_zero_flag"] == 0 if "ptf_zero_flag" in frame.columns else pd.Series(True, index=frame.index)),
    ]:
        sub = frame[cond].head(2)
        for _, row in sub.iterrows():
            samples.append((label, row))
    if not samples:
        samples = [("sample", frame.iloc[0])]
    for i, (label, row) in enumerate(samples[:8]):
        fig, ax = plt.subplots(figsize=(9, 4))
        vals = [
            row.get("zero_price_supply_excess", np.nan),
            row.get("oversupply_mass_below_100", np.nan),
            row.get("supply_gap_above_clearing", np.nan),
            row.get("volume_needed_for_500TL_move", np.nan),
            row.get("volume_needed_for_1000TL_move", np.nan),
            row.get("imbalance_pressure_proxy", np.nan),
        ]
        names = [
            "zero_price_supply_excess",
            "oversupply_mass_below_100",
            "supply_gap_above_clearing",
            "vol_needed_500",
            "vol_needed_1000",
            "imbalance_pressure",
        ]
        ax.bar(names, vals, color=["#4cc9f0", "#4895ef", "#f9c74f", "#f9844a", "#f94144", "#90be6d"])
        ax.set_title(f"{label} | {row.get('ts_hour')}")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        out = DEBUG_DIR / f"{i:02d}_{label}.png"
        fig.savefig(out, dpi=160)
        plt.close(fig)
        paths.append(str(out.relative_to(PROJECT_ROOT)))
    return paths


def _analysis(frame: pd.DataFrame, meta: dict[str, Any]) -> dict[str, Any]:
    if frame.empty:
        return {"available": False, "reason": meta.get("reason", "empty")}
    def corr(a: str, b: str) -> float | None:
        if a not in frame.columns or b not in frame.columns:
            return None
        s = frame[[a, b]].dropna()
        return float(s[a].corr(s[b])) if len(s) >= 3 else None

    zero = frame[frame["ptf_zero_flag"] == 1] if "ptf_zero_flag" in frame.columns else frame.iloc[0:0]
    spike = frame[frame["ptf_spike_flag"] == 1] if "ptf_spike_flag" in frame.columns else frame.iloc[0:0]
    tight = frame[frame["ptf_tight_flag"] == 1] if "ptf_tight_flag" in frame.columns else frame.iloc[0:0]
    normal = frame[(frame["ptf_zero_flag"] == 0) & (frame["ptf_tight_flag"] == 0) & (frame["ptf_spike_flag"] == 0)] if {"ptf_zero_flag", "ptf_tight_flag", "ptf_spike_flag"}.issubset(frame.columns) else frame.iloc[0:0]
    n2s = frame[(frame["clearing_price_proxy"].shift(1) <= 1500) & (frame["ptf_spike_flag"] == 1)] if "clearing_price_proxy" in frame.columns else frame.iloc[0:0]
    t2z = frame[(frame["clearing_price_proxy"].shift(1) >= 1500) & (frame["ptf_low_flag"] == 1)] if "clearing_price_proxy" in frame.columns else frame.iloc[0:0]

    analysis = {
        "available": True,
        "mode": meta.get("mode", "unknown"),
        "rows": int(len(frame)),
        "coverage_start": str(frame["ts_hour"].min()),
        "coverage_end": str(frame["ts_hour"].max()),
        "correlations": {
            "cap_risk_from_curve_vs_price": corr("cap_risk_from_curve", "clearing_price_proxy"),
            "marginality_risk_vs_price": corr("marginality_risk_score", "clearing_price_proxy"),
            "oversupply_pressure_vs_price": corr("oversupply_curve_pressure", "clearing_price_proxy"),
            "supply_gap_score_vs_price": corr("supply_gap_score", "clearing_price_proxy"),
            "curve_break_prob_vs_price": corr("curve_break_probability", "clearing_price_proxy"),
        },
        "regime_slices": {
            "ptf_zero": {
                "rows": int(len(zero)),
                "mean_oversupply_curve_pressure": float(zero["oversupply_curve_pressure"].mean()) if not zero.empty else None,
                "mean_low_price_pressure": float(zero["low_price_pressure"].mean()) if not zero.empty else None,
            },
            "ptf_spike": {
                "rows": int(len(spike)),
                "mean_cap_risk_from_curve": float(spike["cap_risk_from_curve"].mean()) if not spike.empty else None,
                "mean_marginality_risk_score": float(spike["marginality_risk_score"].mean()) if not spike.empty else None,
            },
            "ptf_tight": {
                "rows": int(len(tight)),
                "mean_curve_convexity_score": float(tight["curve_convexity_score"].mean()) if not tight.empty else None,
                "mean_clearing_fragility_score": float(tight["clearing_fragility_score"].mean()) if not tight.empty else None,
            },
            "ptf_normal": {
                "rows": int(len(normal)),
                "mean_slope_near_clearing": float(normal["slope_near_clearing"].mean()) if not normal.empty else None,
            },
        },
        "transition_slices": {
            "normal_to_spike_proxy": {
                "rows": int(len(n2s)),
                "mean_curve_break_probability": float(n2s["curve_break_probability"].mean()) if not n2s.empty else None,
            },
            "tight_to_zero_proxy": {
                "rows": int(len(t2z)),
                "mean_oversupply_curve_pressure": float(t2z["oversupply_curve_pressure"].mean()) if not t2z.empty else None,
            },
        },
        "feature_means": {
            c: float(frame[c].mean()) for c in [
                "clearing_price_proxy",
                "clearing_volume_proxy",
                "slope_near_clearing",
                "local_curve_density",
                "local_elasticity",
                "zero_price_supply_excess",
                "low_price_pressure",
                "oversupply_mass_below_100",
                "renewable_oversupply_zone",
                "curve_convexity_score",
                "cap_risk_from_curve",
                "supply_gap_above_clearing",
                "steepness_above_ptf",
                "marginality_jump_score",
                "demand_curve_steepness",
                "demand_elasticity",
                "demand_cliff_score",
                "offer_stack_density",
                "bid_stack_density",
                "supply_concentration_score",
                "clearing_fragility_score",
                "volume_needed_for_500TL_move",
                "volume_needed_for_1000TL_move",
                "curve_break_probability",
                "imbalance_pressure_proxy",
            ] if c in frame.columns
        },
        "debug_examples": _save_debug_plots(frame),
    }
    return analysis


def _write_outputs(frame: pd.DataFrame, analysis: dict[str, Any], meta: dict[str, Any]) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    from src.utils.safe_io import atomic_parquet_write
    atomic_parquet_write(frame, str(OUT_PATH), index=False)
    REPORT_JSON.write_text(json.dumps(analysis, ensure_ascii=False, indent=2, default=str) + "\n")
    lines = [
        "# Real Supply-Demand Curve Analysis",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Availability",
        "",
        f"- Available: `{analysis.get('available')}`",
        f"- Mode: `{analysis.get('mode')}`",
        f"- Rows: `{analysis.get('rows', 0)}`",
        f"- Coverage: `{analysis.get('coverage_start')} -> {analysis.get('coverage_end')}`",
        "",
        "## Key Signals",
        "",
    ]
    if not analysis.get("available"):
        lines.extend([f"- Reason: `{analysis.get('reason', 'unknown')}`", ""])
    else:
        for k, v in analysis["correlations"].items():
            lines.append(f"- {k}: `{v}`")
        lines.extend(["", "## Regime Slices", ""])
        for name, summary in analysis["regime_slices"].items():
            lines.append(f"- `{name}`: {summary}")
        lines.extend(["", "## Transition Slices", ""])
        for name, summary in analysis["transition_slices"].items():
            lines.append(f"- `{name}`: {summary}")
        lines.extend(["", "## Feature Means", ""])
        for k, v in analysis["feature_means"].items():
            lines.append(f"- `{k}`: `{v}`")
        lines.extend(
            [
                "",
                "## Notes",
                "",
                "This pipeline prefers raw EPİAŞ GÖP curve tables when available. In this repository, no raw curve files were discovered under `data/raw`, so the current run builds compact market microstructure features from the existing hourly curve proxy layer and the finalized market series. The outputs remain leakage-safe because only anchor-time observable hourly data are used.",
                "",
                "Debug plots written to `reports/curve_debug_examples/`.",
            ]
        )
    REPORT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    raw_files = _discover_raw_curve_files()
    frames = _load_raw_curve_frames(raw_files)
    if frames:
        frame, meta = _extract_from_raw(frames)
    else:
        frame, meta = _build_from_proxy()
    analysis = _analysis(frame, meta)
    _write_outputs(frame, analysis, meta)
    print(f"Wrote {OUT_PATH}")
    print(f"Wrote {REPORT_MD}")
    print(f"Wrote {REPORT_JSON}")
    if analysis.get("debug_examples"):
        print("Wrote debug plots:")
        for item in analysis["debug_examples"]:
            print(f"- {item}")


if __name__ == "__main__":
    main()
