#!/usr/bin/env python3
"""
Build leakage-safe feature rows for D+2 PTF forecast (calendar day +2).

Scenario: today is D, target delivery is D+2 (e.g. 1 Jun -> 3 Jun).
Anchor price for each hour h: PTF(D+1, h) — available on EPİAŞ when D+1 is published.
Fundamentals: load / KGÜP / wind for D+2 delivery hours.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
FEATURE_STORE_PATH = PROJECT_ROOT / "data" / "features" / "regime_feature_store.parquet"
REASONING_PATH = PROJECT_ROOT / "data" / "features" / "market_reasoning_features.parquet"
PTF_PATH = PROJECT_ROOT / "data" / "ptf_dataset.csv"
LOAD_PATH = PROJECT_ROOT / "data" / "load_forecast.csv"
KGUP_PATH = PROJECT_ROOT / "data" / "kgup_combined.csv"
WIND_PATH = PROJECT_ROOT / "data" / "wind_forecast.csv"
REGIME_PRED_PATH = PROJECT_ROOT / "data" / "predictions" / "regime_classifier_predictions.csv"
SPIKE_PRED_PATH = PROJECT_ROOT / "data" / "predictions" / "spike_cap_detector_predictions.csv"
TRANSITION_PRED_PATH = PROJECT_ROOT / "data" / "predictions" / "spike_transition_detector_predictions.csv"

OUT_PATH = PROJECT_ROOT / "data" / "features" / "d2_ptf_features.parquet"
REPORT_JSON = PROJECT_ROOT / "reports" / "d2_ptf_feature_builder.json"
REPORT_MD = PROJECT_ROOT / "reports" / "d2_ptf_feature_builder.md"


def parse_ts(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce")
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert("Europe/Istanbul").dt.tz_localize(None)
    return ts


def load_ptf() -> pd.DataFrame:
    ptf = pd.read_csv(PTF_PATH)
    ptf["ts_hour"] = parse_ts(ptf["date"])
    ptf["price"] = pd.to_numeric(ptf["price"], errors="coerce")
    return ptf.dropna(subset=["ts_hour", "price"]).sort_values("ts_hour").drop_duplicates("ts_hour")


def load_hourly_table(path: Path, value_col: str, rename: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["ts_hour", rename])
    df = pd.read_csv(path)
    df["ts_hour"] = parse_ts(df["date"])
    df[rename] = pd.to_numeric(df[value_col], errors="coerce")
    return df[["ts_hour", rename]].dropna(subset=["ts_hour"])


def load_kgup_hourly() -> pd.DataFrame:
    if not KGUP_PATH.exists():
        return pd.DataFrame()
    kg = pd.read_csv(KGUP_PATH)
    kg["ts_hour"] = parse_ts(kg["date"])
    for col in [
        "toplam",
        "dogalgaz",
        "ruzgar",
        "gunes",
        "linyit",
        "tasKomur",
        "ithalKomur",
        "fuelOil",
        "jeotermal",
        "barajli",
        "nafta",
        "biokutle",
        "akarsu",
        "diger",
    ]:
        if col in kg.columns:
            kg[col] = pd.to_numeric(kg[col], errors="coerce")
    hydro_cols = [c for c in ["barajli", "akarsu"] if c in kg.columns]
    if hydro_cols:
        kg["kgup_hydro_mw"] = kg[hydro_cols].sum(axis=1)
    renew_cols = [c for c in ["ruzgar", "gunes", "barajli", "akarsu", "biokutle", "jeotermal"] if c in kg.columns]
    if renew_cols:
        kg["kgup_renewable_mw"] = kg[renew_cols].sum(axis=1)
    return kg.sort_values("ts_hour").drop_duplicates("ts_hour", keep="last")


def _apply_kgup_row(base: dict[str, Any], row: pd.Series) -> None:
    base["kgup_total"] = float(row.get("toplam", np.nan))
    base["kgup_wind_mw"] = float(row.get("ruzgar", np.nan))
    base["kgup_solar_mw"] = float(row.get("gunes", np.nan))
    base["kgup_gas_mw"] = float(row.get("dogalgaz", np.nan))
    if "kgup_hydro_mw" in row.index and pd.notna(row.get("kgup_hydro_mw")):
        base["kgup_hydro_mw"] = float(row["kgup_hydro_mw"])
    if "kgup_renewable_mw" in row.index and pd.notna(row.get("kgup_renewable_mw")):
        base["kgup_renewable_mw"] = float(row["kgup_renewable_mw"])


def refresh_delivery_market_features(frame: pd.DataFrame, load_df: pd.DataFrame) -> pd.DataFrame:
    """Recompute fuel-switch features from delivery-hour KGUP/load (not stale feature store)."""
    if frame.empty:
        return frame
    out = frame.copy()
    total = pd.to_numeric(out.get("kgup_total"), errors="coerce")
    has_kgup = total.notna() & (total > 0)
    if not has_kgup.any():
        return out

    gas = pd.to_numeric(out.get("kgup_gas_mw"), errors="coerce")
    wind = pd.to_numeric(out.get("kgup_wind_mw"), errors="coerce").fillna(0)
    solar = pd.to_numeric(out.get("kgup_solar_mw"), errors="coerce").fillna(0)
    hydro = pd.to_numeric(out.get("kgup_hydro_mw"), errors="coerce").fillna(0)
    renewable = pd.to_numeric(out.get("kgup_renewable_mw"), errors="coerce")
    renewable = renewable.fillna(wind + solar + hydro)

    safe_total = total.where(total > 0, np.nan)
    out.loc[has_kgup, "gas_share_of_generation"] = (gas / safe_total)[has_kgup]
    out.loc[has_kgup, "hydro_share_of_generation"] = (hydro / safe_total)[has_kgup]
    out.loc[has_kgup, "renewable_share_of_generation"] = (renewable / safe_total)[has_kgup]
    out.loc[has_kgup, "renewable_minus_gas_shift"] = (
        out["renewable_share_of_generation"] - out["gas_share_of_generation"]
    )[has_kgup]

    load_norm = load_df.sort_values("ts_hour").copy()
    if not load_norm.empty and "load_forecast" in load_norm.columns:
        load_norm["load_roll_7d"] = (
            load_norm["load_forecast"].shift(24).rolling(24 * 7, min_periods=24 * 5).mean()
        )
        load_norm["load_roll_30d"] = (
            load_norm["load_forecast"].shift(24).rolling(24 * 30, min_periods=24 * 14).mean()
        )
        anchor_ts = out["ts_hour"] - pd.Timedelta(hours=24)
        norms = load_norm.set_index("ts_hour")[["load_roll_7d", "load_roll_30d"]]
        anchored = norms.reindex(anchor_ts.values)
        anchored.index = out.index
        weekly_norm = anchored["load_roll_7d"]
        monthly_norm = anchored["load_roll_30d"]
        load_fc = pd.to_numeric(out.get("load_forecast"), errors="coerce")
        out["load_deviation_from_weekly_norm"] = load_fc - weekly_norm
        out["load_deviation_from_monthly_norm"] = load_fc - monthly_norm
        weekly_gap = ((weekly_norm - load_fc) / weekly_norm.replace(0, np.nan)).clip(lower=0, upper=2).fillna(0)
        monthly_gap = ((monthly_norm - load_fc) / monthly_norm.replace(0, np.nan)).clip(lower=0, upper=2).fillna(0)
        weakness_raw = 0.60 * weekly_gap + 0.40 * monthly_gap
        out["demand_weakness_score"] = (weakness_raw / 1.5 * 100).clip(0, 100)
        out["low_demand_flag"] = (
            (out["demand_weakness_score"] >= 60) | ((weekly_gap > 0.10) & (monthly_gap > 0.05))
        ).astype(int)
        load_pressure = ((load_fc - weekly_norm) / weekly_norm.replace(0, np.nan)).fillna(0).clip(-0.5, 1.5)
        load_pressure = ((load_pressure + 0.5) / 2.0).clip(0, 1)
    else:
        load_pressure = pd.Series(0.5, index=out.index)
        out["low_demand_flag"] = 0

    gas_dep = out["gas_share_of_generation"].fillna(0).clip(0, 1)
    renew_relief = out["renewable_share_of_generation"].fillna(0).clip(0, 1)
    hydro_presence = out["hydro_share_of_generation"].fillna(0).clip(0, 1)
    out.loc[has_kgup, "gas_marginality_proxy"] = (
        100
        * (
            0.45 * gas_dep
            + 0.25 * load_pressure
            + 0.20 * (1 - renew_relief)
            + 0.10 * (1 - hydro_presence)
        )
    ).clip(0, 100)[has_kgup]
    out.loc[has_kgup, "cheap_supply_pressure"] = (
        100
        * (
            0.40 * renew_relief
            + 0.25 * hydro_presence
            + 0.20 * (1 - load_pressure)
            + 0.15 * (1 - gas_dep)
        )
    ).clip(0, 100)[has_kgup]
    out.loc[has_kgup, "gas_off_flag"] = (gas_dep <= 0.10).astype(int)[has_kgup]
    out.loc[has_kgup, "renewable_share_high_flag"] = (renew_relief >= 0.55).astype(int)[has_kgup]
    out.loc[has_kgup, "hydro_high_flag"] = (hydro_presence >= 0.20).astype(int)[has_kgup]
    out.loc[has_kgup, "zero_price_pressure_score"] = (
        100
        * (
            0.32 * out["low_demand_flag"].astype(float)
            + 0.22 * out["gas_off_flag"].astype(float)
            + 0.22 * out["renewable_share_high_flag"].astype(float)
            + 0.14 * out["hydro_high_flag"].astype(float)
            + 0.10 * (1 - load_pressure)
        )
    ).clip(0, 100)[has_kgup]

    if "load_forecast" in out.columns:
        out["load_vs_renewable_balance"] = pd.to_numeric(out["load_forecast"], errors="coerce") - renewable
    return out


def delivery_hours(target_date: date) -> list[pd.Timestamp]:
    start = datetime.combine(target_date, datetime.min.time())
    return [pd.Timestamp(start + timedelta(hours=h)) for h in range(24)]


def build_rows(target_date: date) -> tuple[pd.DataFrame, dict[str, Any]]:
    ptf = load_ptf()
    ptf_map = ptf.set_index("ts_hour")["price"]
    feature_store = pd.read_parquet(FEATURE_STORE_PATH) if FEATURE_STORE_PATH.exists() else pd.DataFrame()
    if not feature_store.empty:
        feature_store["ts_hour"] = parse_ts(feature_store["ts_hour"])

    reasoning = pd.read_parquet(REASONING_PATH) if REASONING_PATH.exists() else pd.DataFrame()
    if not reasoning.empty:
        reasoning["ts_hour"] = parse_ts(reasoning["ts_hour"])

    load = load_hourly_table(LOAD_PATH, "lep", "load_forecast")
    kgup = load_kgup_hourly()
    wind = load_hourly_table(WIND_PATH, "forecast", "wind_forecast_mw")

    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {
        "target_date": target_date.isoformat(),
        "requested_hours": 24,
        "produced_rows": 0,
        "fallbacks": {},
        "missing_anchor_hours": [],
    }

    for ts in delivery_hours(target_date):
        d1 = ts - timedelta(hours=24)
        d2 = ts - timedelta(hours=48)
        d7 = ts - timedelta(hours=168)

        if d1 not in ptf_map.index:
            # D+1 PTF not published yet: fall back to D+0 same hour (weaker anchor).
            d0 = ts - timedelta(hours=48)
            if d0 not in ptf_map.index:
                diagnostics["missing_anchor_hours"].append(str(ts))
                continue
            anchor = float(ptf_map.loc[d0])
            diagnostics["fallbacks"]["anchor_d1_from_d0"] = diagnostics["fallbacks"].get("anchor_d1_from_d0", 0) + 1
        else:
            anchor = float(ptf_map.loc[d1])

        base: dict[str, Any] = {"ts_hour": ts, "is_d2_forecast": 1}
        base["anchor_d1_ptf"] = anchor
        base["anchor_source"] = "d1_actual" if d1 in ptf_map.index else "d0_fallback"
        base["anchor_d2_ptf"] = float(ptf_map.loc[d2]) if d2 in ptf_map.index else np.nan
        base["anchor_d7_ptf"] = float(ptf_map.loc[d7]) if d7 in ptf_map.index else np.nan
        base["ptf_lag_24"] = base["anchor_d1_ptf"]
        base["ptf_lag_48"] = base["anchor_d2_ptf"]
        base["ptf_lag_168"] = base["anchor_d7_ptf"]
        base["ptf_momentum_d1_d2"] = base["anchor_d1_ptf"] - base["anchor_d2_ptf"]
        base["hour"] = ts.hour
        base["weekday"] = ts.weekday()
        base["weekend"] = int(ts.weekday() >= 5)
        base["month"] = ts.month
        base["evening_ramp_flag"] = int(ts.hour in range(17, 23))
        base["sunset_window_flag"] = int(ts.hour in range(16, 20))

        lag_hist = d1 - timedelta(hours=24)
        if not feature_store.empty and lag_hist in set(feature_store["ts_hour"]):
            hist = feature_store.loc[feature_store["ts_hour"] == lag_hist].iloc[0].to_dict()
            for key, val in hist.items():
                if key != "ts_hour" and key not in base:
                    base[key] = val

        if not load.empty and ts in set(load["ts_hour"]):
            base["load_forecast"] = float(load.loc[load["ts_hour"] == ts, "load_forecast"].iloc[0])
        elif not load.empty and d1 in set(load["ts_hour"]):
            base["load_forecast"] = float(load.loc[load["ts_hour"] == d1, "load_forecast"].iloc[0])
            diagnostics["fallbacks"]["load_forecast"] = diagnostics["fallbacks"].get("load_forecast", 0) + 1
        elif not feature_store.empty:
            lag168 = ts - timedelta(hours=168)
            if lag168 in set(feature_store["ts_hour"]) and "load_forecast" in feature_store.columns:
                base["load_forecast"] = float(feature_store.loc[feature_store["ts_hour"] == lag168, "load_forecast"].iloc[0])
                diagnostics["fallbacks"]["load_forecast_lag168"] = diagnostics["fallbacks"].get("load_forecast_lag168", 0) + 1

        if not kgup.empty and ts in set(kgup["ts_hour"]):
            _apply_kgup_row(base, kgup.loc[kgup["ts_hour"] == ts].iloc[0])
        elif not kgup.empty and d1 in set(kgup["ts_hour"]):
            _apply_kgup_row(base, kgup.loc[kgup["ts_hour"] == d1].iloc[0])
            diagnostics["fallbacks"]["kgup"] = diagnostics["fallbacks"].get("kgup", 0) + 1
        elif not feature_store.empty:
            lag168 = ts - timedelta(hours=168)
            hist168 = feature_store.loc[feature_store["ts_hour"] == lag168]
            if not hist168.empty:
                for col in ["kgup_total", "kgup_wind_mw", "kgup_solar_mw", "kgup_gas_mw"]:
                    if col in hist168.columns:
                        base[col] = float(hist168.iloc[0][col])
                diagnostics["fallbacks"]["kgup_lag168"] = diagnostics["fallbacks"].get("kgup_lag168", 0) + 1

        if not wind.empty and ts in set(wind["ts_hour"]):
            base["wind_forecast_mw"] = float(wind.loc[wind["ts_hour"] == ts, "wind_forecast_mw"].iloc[0])

        if base.get("load_forecast") and base.get("kgup_total"):
            base["load_minus_kgup"] = base["load_forecast"] - base["kgup_total"]
            base["residual_load_forecast"] = base["load_forecast"] - (
                float(base.get("kgup_renewable_mw", 0) or 0) + float(base.get("kgup_gas_mw", 0) or 0)
            )

        rows.append(base)

    frame = pd.DataFrame(rows)
    if frame.empty:
        diagnostics["missing_reason"] = "No rows produced — anchor D+1 PTF missing."
        return frame, diagnostics

    frame = refresh_delivery_market_features(frame, load)
    diagnostics["refreshed_fuel_switch_from_delivery_kgup"] = True

    if not reasoning.empty:
        frame = frame.merge(reasoning, on="ts_hour", how="left", suffixes=("", "_reason"))

    for path in [REGIME_PRED_PATH, SPIKE_PRED_PATH, TRANSITION_PRED_PATH]:
        if not path.exists():
            continue
        pred = pd.read_csv(path)
        pred["ts_hour"] = parse_ts(pred["ts_hour"])
        cols = [c for c in pred.columns if c != "ts_hour"]
        frame = frame.merge(pred[["ts_hour"] + cols], on="ts_hour", how="left")

    diagnostics["produced_rows"] = int(len(frame))
    diagnostics["ptf_max"] = str(ptf["ts_hour"].max())
    diagnostics["load_max"] = str(load["ts_hour"].max()) if not load.empty else None
    diagnostics["kgup_max"] = str(kgup["ts_hour"].max()) if not kgup.empty else None
    return frame.sort_values("ts_hour"), diagnostics


def write_reports(frame: pd.DataFrame, diagnostics: dict[str, Any]) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    if not frame.empty:
        frame.to_parquet(OUT_PATH, index=False)
    diagnostics["generated_at"] = datetime.now(timezone.utc).isoformat()
    diagnostics["output_path"] = str(OUT_PATH.relative_to(PROJECT_ROOT))
    REPORT_JSON.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2, default=str) + "\n")
    REPORT_MD.write_text(
        "\n".join(
            [
                "# D+2 PTF Feature Builder",
                "",
                f"Generated: `{diagnostics['generated_at']}`",
                "",
                f"- Target date: `{diagnostics.get('target_date')}`",
                f"- Rows: `{diagnostics.get('produced_rows', 0)}`",
                f"- PTF max: `{diagnostics.get('ptf_max')}`",
                f"- Load max: `{diagnostics.get('load_max')}`",
                f"- KGUP max: `{diagnostics.get('kgup_max')}`",
                f"- Missing anchor hours: `{len(diagnostics.get('missing_anchor_hours', []))}`",
                f"- Fallbacks: `{diagnostics.get('fallbacks', {})}`",
            ]
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-date",
        default=None,
        help="Delivery date YYYY-MM-DD (default: today + 2 days).",
    )
    args = parser.parse_args()
    if args.target_date:
        target = pd.to_datetime(args.target_date).date()
    else:
        target = datetime.now().date() + timedelta(days=2)

    frame, diagnostics = build_rows(target)
    write_reports(frame, diagnostics)
    print(f"Wrote {OUT_PATH} rows={len(frame)}")
    print(f"Wrote {REPORT_MD}")


if __name__ == "__main__":
    main()
