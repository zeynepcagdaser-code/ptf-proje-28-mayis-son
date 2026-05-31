#!/usr/bin/env python3
"""
Build supply-demand curve proxy features from existing hourly market datasets.

This is analysis-only. No model training.

Important:
The repo does not currently contain raw EPİAŞ supply-demand curve files, so this
script constructs market microstructure proxies from finalized hourly PTF, KGÜP,
and load forecast data. The outputs should be treated as explanatory curve
intelligence, not direct bid-curve reconstruction.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
PTF_PATH = PROJECT_ROOT / "data" / "ptf_dataset.csv"
KGUP_PATH = PROJECT_ROOT / "data" / "kgup_combined.csv"
LOAD_PATH = PROJECT_ROOT / "data" / "load_forecast.csv"
REGIME_LABELS_PATH = PROJECT_ROOT / "data" / "regime_labels.csv"
OUT_PATH = PROJECT_ROOT / "data" / "features" / "supply_demand_curve_features.parquet"
REPORT_MD = PROJECT_ROOT / "reports" / "supply_demand_curve_analysis.md"
REPORT_JSON = PROJECT_ROOT / "reports" / "supply_demand_curve_analysis.json"


def read_hourly(path: Path, date_col: str, time_col: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if date_col not in df.columns:
        return pd.DataFrame()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if time_col and time_col in df.columns:
        dt = pd.to_datetime(df[date_col].dt.strftime("%Y-%m-%d") + " " + df[time_col].astype(str), errors="coerce")
    else:
        dt = pd.to_datetime(df[date_col], errors="coerce")
    if getattr(dt.dt, "tz", None) is None:
        df["ts_hour"] = dt.dt.tz_localize("Europe/Istanbul", nonexistent="NaT", ambiguous="NaT")
    else:
        df["ts_hour"] = dt.dt.tz_convert("Europe/Istanbul")
    return df


def safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.replace(0, np.nan)
    return num / den


def regime_from_price(price: pd.Series) -> pd.Series:
    return pd.cut(
        price,
        bins=[-np.inf, 50, 1500, 4000, np.inf],
        labels=["negative_zero_pressure", "normal", "tight", "spike_cap"],
        right=False,
    )


def build_features() -> tuple[pd.DataFrame, dict[str, Any]]:
    ptf = read_hourly(PTF_PATH, "date")
    kgup = read_hourly(KGUP_PATH, "date", "time")
    load = read_hourly(LOAD_PATH, "date", "time")
    labels = pd.read_csv(REGIME_LABELS_PATH) if REGIME_LABELS_PATH.exists() else pd.DataFrame()

    if ptf.empty or kgup.empty or load.empty:
        return pd.DataFrame(), {
            "available": False,
            "reason": "One or more required hourly inputs are missing.",
            "ptf_rows": int(len(ptf)),
            "kgup_rows": int(len(kgup)),
            "load_rows": int(len(load)),
        }

    ptf = ptf.rename(columns={"price": "clearing_price_proxy"})
    kgup = kgup.rename(columns={"toplam": "kgup_total"})
    load = load.rename(columns={"lep": "load_forecast"})

    frame = (
        ptf[["ts_hour", "clearing_price_proxy"]]
        .merge(kgup[["ts_hour", "kgup_total", "dogalgaz", "ruzgar", "gunes", "barajli", "akarsu", "biokutle", "linyit", "tasKomur", "ithalKomur", "fuelOil", "nafta", "diger"]], on="ts_hour", how="inner")
        .merge(load[["ts_hour", "load_forecast"]], on="ts_hour", how="inner")
        .sort_values("ts_hour")
        .reset_index(drop=True)
    )

    frame["delivery_hour"] = frame["ts_hour"]
    frame["hour"] = frame["ts_hour"].dt.hour
    frame["weekday"] = frame["ts_hour"].dt.dayofweek

    renewable = frame[["ruzgar", "gunes", "barajli", "akarsu", "biokutle"]].fillna(0).sum(axis=1)
    thermal = frame[["dogalgaz", "linyit", "tasKomur", "ithalKomur", "fuelOil", "nafta", "diger"]].fillna(0).sum(axis=1)
    frame["renewable_total"] = renewable
    frame["thermal_total"] = thermal
    frame["supply_gap"] = frame["kgup_total"] - frame["load_forecast"]
    frame["supply_gap_1h"] = frame["supply_gap"].diff()
    frame["supply_gap_24h"] = frame["supply_gap"].diff(24)
    frame["supply_gap_zscore_24h"] = (frame["supply_gap"] - frame["supply_gap"].rolling(24, min_periods=12).mean()) / frame["supply_gap"].rolling(24, min_periods=12).std().replace(0, np.nan)

    frame["clearing_price_proxy"] = frame["clearing_price_proxy"].astype(float)
    frame["curve_slope_near_ptf"] = frame["clearing_price_proxy"].diff().abs().fillna(0) / frame["supply_gap"].abs().replace(0, np.nan)
    frame["demand_elasticity_proxy"] = -frame["load_forecast"].diff().fillna(0).abs() / frame["clearing_price_proxy"].diff().abs().replace(0, np.nan)
    frame["supply_elasticity_proxy"] = frame["kgup_total"].diff().fillna(0) / frame["clearing_price_proxy"].diff().abs().replace(0, np.nan)
    frame["supply_gap_score"] = safe_ratio(frame["supply_gap"], frame["load_forecast"])
    frame["marginality_risk_score"] = (
        np.clip(frame["load_forecast"] / frame["kgup_total"].replace(0, np.nan) - 1.0, 0, None)
        + np.clip(frame["thermal_total"] / frame["kgup_total"].replace(0, np.nan), 0, 1)
    ).fillna(0)
    frame["curve_convexity_score"] = frame["clearing_price_proxy"].diff().diff().abs().fillna(0)
    frame["low_price_pressure_score"] = np.clip((50 - frame["clearing_price_proxy"]) / 50.0, 0, 1)
    frame["cap_risk_from_curve"] = np.clip((frame["clearing_price_proxy"] - 3500) / 800.0, 0, 1)
    frame["oversupply_curve_pressure"] = np.clip((-frame["supply_gap"]) / frame["load_forecast"].replace(0, np.nan), 0, 1).fillna(0)
    frame["bid_stack_density"] = safe_ratio(frame["kgup_total"], frame["load_forecast"])
    frame["offer_stack_density"] = safe_ratio(frame["load_forecast"], frame["kgup_total"])
    frame["marginality_proxy"] = frame["thermal_total"] / frame["kgup_total"].replace(0, np.nan)
    frame["ptf_zero_flag"] = (frame["clearing_price_proxy"] <= 0).astype(int)
    frame["ptf_low_flag"] = (frame["clearing_price_proxy"] <= 50).astype(int)
    frame["ptf_spike_flag"] = (frame["clearing_price_proxy"] >= 4000).astype(int)
    frame["ptf_tight_flag"] = ((frame["clearing_price_proxy"] >= 1500) & (frame["clearing_price_proxy"] < 4000)).astype(int)

    if not labels.empty and "ts_hour" in labels.columns:
        labels["ts_hour"] = pd.to_datetime(labels["ts_hour"], errors="coerce")
        if getattr(labels["ts_hour"].dt, "tz", None) is None:
            labels["ts_hour"] = labels["ts_hour"].dt.tz_localize("Europe/Istanbul", nonexistent="NaT", ambiguous="NaT")
        else:
            labels["ts_hour"] = labels["ts_hour"].dt.tz_convert("Europe/Istanbul")
        frame = frame.merge(labels[["ts_hour", "target_regime"]], on="ts_hour", how="left")
    else:
        frame["target_regime"] = regime_from_price(frame["clearing_price_proxy"])
    frame["target_regime"] = frame["target_regime"].fillna(regime_from_price(frame["clearing_price_proxy"]))

    return frame, {
        "available": True,
        "rows": int(len(frame)),
        "coverage_start": str(frame["ts_hour"].min()),
        "coverage_end": str(frame["ts_hour"].max()),
        "price_zero_hours": int(frame["ptf_zero_flag"].sum()),
        "price_low_hours": int(frame["ptf_low_flag"].sum()),
        "price_tight_hours": int(frame["ptf_tight_flag"].sum()),
        "price_spike_hours": int(frame["ptf_spike_flag"].sum()),
    }


def analysis_block(frame: pd.DataFrame, meta: dict[str, Any]) -> dict[str, Any]:
    if frame.empty:
        return {
            "available": False,
            "reason": meta.get("reason", "No data"),
        }

    def corr(a: str, b: str) -> float | None:
        s = frame[[a, b]].dropna()
        if len(s) < 3:
            return None
        return float(s[a].corr(s[b]))

    regimes = {}
    for regime, group in frame.groupby("target_regime"):
        regimes[str(regime)] = {
            "rows": int(len(group)),
            "mean_price": float(group["clearing_price_proxy"].mean()),
            "mean_supply_gap": float(group["supply_gap"].mean()),
            "mean_curve_slope_near_ptf": float(group["curve_slope_near_ptf"].replace([np.inf, -np.inf], np.nan).dropna().mean()) if group["curve_slope_near_ptf"].notna().any() else None,
            "mean_marginality_risk_score": float(group["marginality_risk_score"].mean()),
            "mean_oversupply_curve_pressure": float(group["oversupply_curve_pressure"].mean()),
            "mean_cap_risk_from_curve": float(group["cap_risk_from_curve"].mean()),
            "mean_bid_stack_density": float(group["bid_stack_density"].replace([np.inf, -np.inf], np.nan).dropna().mean()) if group["bid_stack_density"].notna().any() else None,
        }

    low_price = frame[frame["ptf_low_flag"] == 1]
    spike = frame[frame["ptf_spike_flag"] == 1]
    zero = frame[frame["ptf_zero_flag"] == 1]
    tight = frame[frame["ptf_tight_flag"] == 1]

    return {
        "available": True,
        "rows": int(len(frame)),
        "coverage": {
            "start": str(frame["ts_hour"].min()),
            "end": str(frame["ts_hour"].max()),
        },
        "correlations": {
            "supply_gap_vs_price": corr("supply_gap", "clearing_price_proxy"),
            "bid_stack_density_vs_price": corr("bid_stack_density", "clearing_price_proxy"),
            "offer_stack_density_vs_price": corr("offer_stack_density", "clearing_price_proxy"),
            "oversupply_pressure_vs_price": corr("oversupply_curve_pressure", "clearing_price_proxy"),
            "marginality_risk_vs_price": corr("marginality_risk_score", "clearing_price_proxy"),
            "curve_slope_vs_price": corr("curve_slope_near_ptf", "clearing_price_proxy"),
        },
        "regime_summary": regimes,
        "slices": {
            "ptf_zero": {
                "rows": int(len(zero)),
                "mean_supply_gap": float(zero["supply_gap"].mean()) if not zero.empty else None,
                "mean_oversupply_curve_pressure": float(zero["oversupply_curve_pressure"].mean()) if not zero.empty else None,
                "mean_bid_stack_density": float(zero["bid_stack_density"].mean()) if not zero.empty else None,
            },
            "ptf_low": {
                "rows": int(len(low_price)),
                "mean_supply_gap": float(low_price["supply_gap"].mean()) if not low_price.empty else None,
                "mean_oversupply_curve_pressure": float(low_price["oversupply_curve_pressure"].mean()) if not low_price.empty else None,
                "mean_bid_stack_density": float(low_price["bid_stack_density"].mean()) if not low_price.empty else None,
            },
            "ptf_tight": {
                "rows": int(len(tight)),
                "mean_supply_gap": float(tight["supply_gap"].mean()) if not tight.empty else None,
                "mean_marginality_risk_score": float(tight["marginality_risk_score"].mean()) if not tight.empty else None,
                "mean_curve_slope_near_ptf": float(tight["curve_slope_near_ptf"].mean()) if not tight.empty else None,
            },
            "ptf_spike": {
                "rows": int(len(spike)),
                "mean_supply_gap": float(spike["supply_gap"].mean()) if not spike.empty else None,
                "mean_marginality_risk_score": float(spike["marginality_risk_score"].mean()) if not spike.empty else None,
                "mean_cap_risk_from_curve": float(spike["cap_risk_from_curve"].mean()) if not spike.empty else None,
            },
        },
        "feature_summary": {
            "mean_curve_convexity": float(frame["curve_convexity_score"].mean()),
            "mean_low_price_pressure": float(frame["low_price_pressure_score"].mean()),
            "mean_cap_risk_from_curve": float(frame["cap_risk_from_curve"].mean()),
            "mean_oversupply_curve_pressure": float(frame["oversupply_curve_pressure"].mean()),
        },
    }


def write_outputs(frame: pd.DataFrame, meta: dict[str, Any], audit: dict[str, Any]) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT_PATH, index=False)
    REPORT_JSON.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str) + "\n")
    lines = [
        "# Supply-Demand Curve Analysis",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Data Availability",
        "",
        f"- Available: `{meta.get('available')}`",
        f"- Rows: `{meta.get('rows', 0)}`",
        f"- Coverage: `{meta.get('coverage_start')} -> {meta.get('coverage_end')}`",
        f"- PTF zero hours: `{meta.get('price_zero_hours', 0)}`",
        f"- PTF low-price hours: `{meta.get('price_low_hours', 0)}`",
        f"- PTF tight hours: `{meta.get('price_tight_hours', 0)}`",
        f"- PTF spike hours: `{meta.get('price_spike_hours', 0)}`",
        "",
    ]
    if not audit.get("available"):
        lines.extend([f"- Reason: `{audit.get('reason', 'unknown')}`", ""])
    else:
        lines.extend(
            [
                "## Key Correlations",
                "",
                f"- supply_gap vs price: `{audit['correlations']['supply_gap_vs_price']}`",
                f"- bid_stack_density vs price: `{audit['correlations']['bid_stack_density_vs_price']}`",
                f"- offer_stack_density vs price: `{audit['correlations']['offer_stack_density_vs_price']}`",
                f"- oversupply_curve_pressure vs price: `{audit['correlations']['oversupply_pressure_vs_price']}`",
                f"- marginality_risk_score vs price: `{audit['correlations']['marginality_risk_vs_price']}`",
                f"- curve_slope_near_ptf vs price: `{audit['correlations']['curve_slope_vs_price']}`",
                "",
                "## Regime Summary",
                "",
            ]
        )
        for regime, summary in audit["regime_summary"].items():
            lines.append(
                f"- `{regime}`: rows={summary['rows']}, mean_price={summary['mean_price']:.2f}, mean_supply_gap={summary['mean_supply_gap']:.2f}, mean_cap_risk_from_curve={summary['mean_cap_risk_from_curve']}"
            )
        lines.extend(
            [
                "",
                "## Slice Diagnostics",
                "",
            ]
        )
        for name, summary in audit["slices"].items():
            lines.append(f"- `{name}`: {summary}")
        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                "These features are curve proxies built from hourly supply, load, and finalized PTF because raw EPİAŞ supply-demand curve files are not present in the repository. The strongest signal should be read as structural pressure, not literal bid-curve elasticity.",
                "",
                "## Missing Raw Curve Note",
                "",
                "No raw supply-demand curve snapshot file was found in the repo, so `clearing_price_proxy` is anchored to finalized PTF and the slope/elasticity terms are derived proxies.",
            ]
        )
    REPORT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    frame, meta = build_features()
    audit = analysis_block(frame, meta)
    write_outputs(frame, meta, audit)
    print(f"Wrote {OUT_PATH}")
    print(f"Wrote {REPORT_MD}")
    print(f"Wrote {REPORT_JSON}")


if __name__ == "__main__":
    main()
