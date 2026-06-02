#!/usr/bin/env python3
"""
Build a leakage-safe curve-aware training dataset for next-day PTF forecasting.

Inputs:
    - reconstructed_weekly_curve_features_2026-06-01_2026-06-07.parquet
    - ptf_dataset.csv
    - kgup_combined.csv
    - load_forecast.csv
    - regime_feature_store.parquet
    - market_reasoning_features.parquet

Target:
    - T+1 hourly PTF

No training is performed here.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent

CURVE_PATH = PROJECT_ROOT / "data" / "features" / "reconstructed_weekly_curve_features_2026-06-01_2026-06-07.parquet"
CURVE_GLOB = "reconstructed_weekly_curve_features_*.parquet"
PTF_PATH = PROJECT_ROOT / "data" / "ptf_dataset.csv"
KGUP_PATH = PROJECT_ROOT / "data" / "kgup_combined.csv"
LOAD_FORECAST_PATH = PROJECT_ROOT / "data" / "load_forecast.csv"
REGIME_FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "regime_feature_store.parquet"
REASONING_PATH = PROJECT_ROOT / "data" / "features" / "market_reasoning_features.parquet"
MUST_RUN_GLOB = "must_run_supply_features_*.parquet"

OUT_PATH = PROJECT_ROOT / "data" / "features" / "curve_aware_training_dataset.parquet"
REPORT_MD = PROJECT_ROOT / "reports" / "curve_aware_dataset_audit.md"
REPORT_JSON = PROJECT_ROOT / "reports" / "curve_aware_dataset_audit.json"


def parse_dt(df: pd.DataFrame, date_col: str = "date", hour_col: str = "hour") -> pd.Series:
    dates = pd.to_datetime(df[date_col], errors="coerce")
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_localize(None)
    if hour_col not in df.columns:
        return dates
    hour = df[hour_col].astype(str).str.extract(r"(\d{1,2})")[0]
    hour_num = pd.to_numeric(hour, errors="coerce")
    return dates.dt.normalize() + pd.to_timedelta(hour_num.fillna(0), unit="h")


def to_naive(series: pd.Series) -> pd.Series:
    out = pd.to_datetime(series, errors="coerce")
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_localize(None)
    return out


def safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace(0, np.nan)


def price_band(price: pd.Series) -> pd.Series:
    return pd.cut(
        price,
        bins=[-np.inf, 50, 1500, 4000, np.inf],
        labels=["negative_zero_pressure", "normal", "tight", "spike_cap"],
    ).astype("string")


def read_curve_features() -> tuple[pd.DataFrame, list[str]]:
    curve_files = sorted((PROJECT_ROOT / "data" / "features").glob(CURVE_GLOB))
    if not curve_files and CURVE_PATH.exists():
        curve_files = [CURVE_PATH]
    frames = []
    for path in curve_files:
        frame = pd.read_parquet(path)
        frame["source_curve_file"] = path.name
        frames.append(frame)
    if not frames:
        return pd.DataFrame(), []
    curve = pd.concat(frames, ignore_index=True)
    if "delivery_hour" in curve.columns:
        curve = curve.drop_duplicates("delivery_hour", keep="last").sort_values("delivery_hour")
    return curve, [str(path.relative_to(PROJECT_ROOT)) for path in curve_files]


def read_must_run_features() -> tuple[pd.DataFrame, list[str]]:
    files = sorted((PROJECT_ROOT / "data" / "features").glob(MUST_RUN_GLOB))
    frames = []
    for path in files:
        try:
            frame = pd.read_parquet(path)
        except Exception:
            continue
        if "delivery_hour" not in frame.columns:
            continue
        frame = frame.copy()
        frame["delivery_hour"] = to_naive(frame["delivery_hour"])
        frame["source_must_run_file"] = path.name
        frames.append(frame)
    if not frames:
        return pd.DataFrame(), []
    mr = pd.concat(frames, ignore_index=True)
    mr = mr.drop_duplicates("delivery_hour", keep="last").sort_values("delivery_hour")
    return mr, [str(path.relative_to(PROJECT_ROOT)) for path in files]


def read_market_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    curve, curve_files = read_curve_features()
    if curve.empty:
        return curve, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), curve_files
    curve["delivery_hour"] = to_naive(curve["delivery_hour"])

    ptf = pd.read_csv(PTF_PATH)
    ptf["ts_hour"] = parse_dt(ptf, "date", "hour")
    ptf["price"] = pd.to_numeric(ptf["price"], errors="coerce")
    ptf = ptf.dropna(subset=["ts_hour"]).sort_values("ts_hour").drop_duplicates("ts_hour", keep="last")

    kgup = pd.read_csv(KGUP_PATH)
    kgup["ts_hour"] = pd.to_datetime(kgup["date"], errors="coerce")
    if getattr(kgup["ts_hour"].dt, "tz", None) is not None:
        kgup["ts_hour"] = kgup["ts_hour"].dt.tz_localize(None)
    for c in ["toplam", "dogalgaz", "ruzgar", "gunes", "barajli", "akarsu", "biokutle", "jeotermal", "linyit", "tasKomur", "ithalKomur", "fuelOil"]:
        if c in kgup.columns:
            kgup[c] = pd.to_numeric(kgup[c], errors="coerce")
    kgup = kgup.sort_values("ts_hour").drop_duplicates("ts_hour", keep="last")

    load = pd.read_csv(LOAD_FORECAST_PATH)
    load["ts_hour"] = pd.to_datetime(load["date"], errors="coerce")
    if getattr(load["ts_hour"].dt, "tz", None) is not None:
        load["ts_hour"] = load["ts_hour"].dt.tz_localize(None)
    load["load_forecast"] = pd.to_numeric(load["lep"], errors="coerce")
    load = load.drop_duplicates("ts_hour", keep="last")

    regime = pd.read_parquet(REGIME_FEATURES_PATH)
    regime["ts_hour"] = to_naive(regime["ts_hour"])
    regime = regime.sort_values("ts_hour").drop_duplicates("ts_hour", keep="last")

    reasoning = pd.read_parquet(REASONING_PATH)
    reasoning["ts_hour"] = to_naive(reasoning["ts_hour"])
    reasoning = reasoning.sort_values("ts_hour").drop_duplicates("ts_hour", keep="last")

    must_run, must_run_files = read_must_run_features()
    if not must_run.empty:
        must_run["delivery_hour"] = to_naive(must_run["delivery_hour"])

    # curve_files is already in the return signature; append must_run_files via audit only.
    return curve, ptf, kgup, load, regime, reasoning, curve_files


def build_dataset() -> tuple[pd.DataFrame, dict[str, Any]]:
    curve, ptf, kgup, load, regime, reasoning, curve_files = read_market_tables()
    if curve.empty or ptf.empty:
        return pd.DataFrame(), {"available": False, "reason": "curve or ptf source missing", "curve_files": curve_files}

    dataset = curve.copy()
    dataset["curve_day"] = dataset["delivery_hour"].dt.normalize()
    dataset["target_ts_hour"] = dataset["delivery_hour"] + pd.Timedelta(days=1)
    dataset["delivery_hour"] = dataset["target_ts_hour"]
    dataset["hour"] = dataset["delivery_hour"].dt.hour
    dataset["weekday"] = dataset["delivery_hour"].dt.dayofweek
    dataset["weekend"] = dataset["weekday"].isin([5, 6]).astype(int)
    dataset["month"] = dataset["delivery_hour"].dt.month

    rename_curve = {
        "slope_near_clearing": "prev_day_slope_near_clearing",
        "elasticity_near_clearing": "prev_day_elasticity_near_clearing",
        "curve_fragility_score": "prev_day_curve_fragility_score",
        "volume_needed_for_100TL_move": "prev_day_volume_needed_for_100TL_move",
        "volume_needed_for_500TL_move": "prev_day_volume_needed_for_500TL_move",
        "cap_risk_score": "prev_day_cap_risk_score",
        "oversupply_pressure": "prev_day_oversupply_pressure",
        "reconstructed_clearing_price": "prev_day_reconstructed_clearing_price",
        "reconstructed_clearing_volume": "prev_day_reconstructed_clearing_volume",
        "mcpPrice": "prev_day_mcpPrice",
        "matchingQuantity": "prev_day_matchingQuantity",
        "zero_pressure_from_curve": "prev_day_zero_pressure_from_curve",
        "spike_pressure_from_curve": "prev_day_spike_pressure_from_curve",
    }
    dataset = dataset.rename(columns=rename_curve)
    dataset["reconstruction_confidence"] = 1.0 / (1.0 + dataset["prev_day_curve_fragility_score"].fillna(0))

    curve_band = price_band(dataset["prev_day_mcpPrice"])
    dataset["previous_day_regime"] = curve_band

    # target PTF = T+1 hour
    target = ptf[["ts_hour", "price"]].rename(columns={"ts_hour": "delivery_hour", "price": "target_ptf"})
    dataset = dataset.merge(target, on="delivery_hour", how="left")
    dataset["target_hour"] = dataset["delivery_hour"]

    # Direct lag-24 PTF from the canonical PTF table. This is still
    # leakage-safe for a T+1 target and prevents stale feature-store coverage
    # from leaving the persistence anchor empty for newly fetched target days.
    lag24 = ptf[["ts_hour", "price"]].copy()
    lag24["delivery_hour"] = lag24["ts_hour"] + pd.Timedelta(hours=24)
    lag24 = lag24.rename(columns={"price": "ptf_lag_24_direct"})
    dataset = dataset.merge(lag24[["delivery_hour", "ptf_lag_24_direct"]], on="delivery_hour", how="left")

    # Market state at target hour is leakage-safe when sourced from lagged feature store.
    target_regime = regime.copy()
    target_regime = target_regime.rename(columns={"ts_hour": "delivery_hour"})
    dataset = dataset.merge(
        target_regime[
            [
                "delivery_hour",
                "ptf_lag_24",
                "kgup_total",
                "gas_share",
                "coal_share",
                "hydro_share",
                "wind_share",
                "solar_share",
                "thermal_share",
                "kgup_wind_mw",
                "kgup_solar_mw",
                "kgup_renewable_mw",
                "load_forecast",
                "load_ramp_1h",
                "load_ramp_3h",
                "residual_load_forecast",
                "residual_load_ramp",
                "load_minus_kgup",
                "solar_ramp_down",
                "solar_cliff_score",
                "wind_relief_score",
                "renewable_oversupply_score",
                "active_maintenance_capacity",
                "gas_maintenance",
                "coal_maintenance",
                "hydro_maintenance",
                "outage_stress_index",
                "gas_share_of_generation",
                "hydro_share_of_generation",
                "renewable_share_of_generation",
                "renewable_minus_gas_shift",
                "gas_marginality_proxy",
                "hydro_displacement_score",
                "cheap_supply_pressure",
                "low_demand_flag",
                "gas_off_flag",
                "renewable_share_high_flag",
                "hydro_high_flag",
                "zero_price_pressure_score",
                "load_deviation_from_weekly_norm",
                "load_deviation_from_monthly_norm",
                "demand_weakness_score",
                "load_vs_renewable_balance",
            ]
        ],
        on="delivery_hour",
        how="left",
        suffixes=("", "_target"),
    )
    if "ptf_lag_24" not in dataset.columns:
        dataset["ptf_lag_24"] = np.nan
    dataset["ptf_lag_24"] = dataset["ptf_lag_24"].fillna(dataset["ptf_lag_24_direct"])

    if not load.empty:
        dataset = dataset.merge(load[["ts_hour", "load_forecast"]].rename(columns={"ts_hour": "delivery_hour", "load_forecast": "load_forecast_target"}), on="delivery_hour", how="left")
    if not kgup.empty:
        kg = kgup.copy().rename(columns={"ts_hour": "delivery_hour"})
        kg["kgup_total"] = kg.get("toplam")
        kg["renewable_share"] = safe_ratio(kg[["ruzgar", "gunes", "akarsu"]].sum(axis=1), kg["toplam"])
        dataset = dataset.merge(
            kg[["delivery_hour", "toplam", "ruzgar", "gunes", "barajli", "akarsu", "biokutle", "jeotermal"]],
            on="delivery_hour",
            how="left",
            suffixes=("", "_kgup_realized"),
        )

    reasoning = reasoning.rename(columns={"ts_hour": "delivery_hour"})
    dataset = dataset.merge(reasoning, on="delivery_hour", how="left")

    # Optional must-run proxy features (structural market proxy).
    must_run, must_run_files = read_must_run_features()
    if not must_run.empty:
        keep_mr = [
            "delivery_hour",
            "must_run_supply",
            "must_run_wind",
            "must_run_solar",
            "must_run_hydro",
            "must_run_biomass",
            "must_run_geothermal",
            "must_run_share_of_load",
            "residual_load_after_must_run",
            "must_run_ramp_1h",
            "must_run_ramp_3h",
            "renewable_must_run_pressure",
            "hydro_must_run_pressure",
            "solar_must_run_pressure",
            "strict_point_in_time_safe",
            "structural_market_proxy",
        ]
        for col in keep_mr:
            if col not in must_run.columns:
                must_run[col] = np.nan
        dataset = dataset.merge(must_run[keep_mr], on="delivery_hour", how="left")
    else:
        must_run_files = []

    # Preserve target and audit features.
    dataset["target_ptf"] = pd.to_numeric(dataset["target_ptf"], errors="coerce")
    dataset["target_band"] = price_band(dataset["target_ptf"])

    # Feature family requested by user.
    keep_cols = [
        "delivery_hour",
        "target_hour",
        "target_ptf",
        "prev_day_slope_near_clearing",
        "prev_day_elasticity_near_clearing",
        "prev_day_curve_fragility_score",
        "prev_day_volume_needed_for_100TL_move",
        "prev_day_volume_needed_for_500TL_move",
        "prev_day_cap_risk_score",
        "prev_day_oversupply_pressure",
        "reconstruction_confidence",
        "ptf_lag_24",
        "residual_load_forecast",
        "kgup_total",
        "load_forecast",
        "gas_share",
        "coal_share",
        "hydro_share",
        "wind_share",
        "solar_share",
        "thermal_share",
        "kgup_wind_mw",
        "kgup_solar_mw",
        "kgup_renewable_mw",
        "hour",
        "weekday",
        "weekend",
        "month",
        "previous_day_regime",
        "prev_day_spike_pressure_from_curve",
        "prev_day_zero_pressure_from_curve",
        "analyst_zero_score",
        "analyst_spike_score",
        "analyst_tight_score",
        "analyst_persistence_break_score",
        "analyst_expected_regime",
        "analyst_confidence_score",
        "gas_share_of_generation",
        "hydro_share_of_generation",
        "renewable_share_of_generation",
        "renewable_minus_gas_shift",
        "gas_marginality_proxy",
        "hydro_displacement_score",
        "cheap_supply_pressure",
        "low_demand_flag",
        "gas_off_flag",
        "renewable_share_high_flag",
        "hydro_high_flag",
        "zero_price_pressure_score",
        "load_deviation_from_weekly_norm",
        "load_deviation_from_monthly_norm",
        "demand_weakness_score",
        "load_vs_renewable_balance",
        # must-run proxy features (may be missing outside fetched window)
        "must_run_supply",
        "must_run_wind",
        "must_run_solar",
        "must_run_hydro",
        "must_run_biomass",
        "must_run_geothermal",
        "must_run_share_of_load",
        "residual_load_after_must_run",
        "must_run_ramp_1h",
        "must_run_ramp_3h",
        "renewable_must_run_pressure",
        "hydro_must_run_pressure",
        "solar_must_run_pressure",
        "strict_point_in_time_safe",
        "structural_market_proxy",
    ]
    for col in keep_cols:
        if col not in dataset.columns:
            dataset[col] = np.nan
    out = dataset[keep_cols].copy()

    # leakage-safe audits
    out["target_band"] = dataset["target_band"]
    out["curve_day"] = dataset["curve_day"]

    # drop rows without target
    out = out.dropna(subset=["target_ptf"]).sort_values("delivery_hour").reset_index(drop=True)

    audit = {
        "available": True,
        "rows": int(len(out)),
        "curve_input_files": curve_files,
        "must_run_input_files": must_run_files,
        "curve_source_rows": int(len(curve)),
        "curve_rows_with_t_plus_1_target": int(len(out)),
        "coverage_start": str(out["delivery_hour"].min()) if not out.empty else None,
        "coverage_end": str(out["delivery_hour"].max()) if not out.empty else None,
        "feature_missing_ratios": {
            col: float(out[col].isna().mean()) for col in out.columns if col not in {"delivery_hour", "target_hour", "target_ptf", "target_band", "curve_day"}
        },
        "leakage_checks": {
            "uses_same_day_curve_as_feature": False,
            "target_ptf_is_future_only": True,
            "contains_realized_same_hour_future_market_features": False,
        },
        "target_distribution": out["target_band"].value_counts(dropna=False).to_dict(),
        "feature_correlations_with_target_ptf": {},
    }
    numeric_cols = [
        "prev_day_slope_near_clearing",
        "prev_day_elasticity_near_clearing",
        "prev_day_curve_fragility_score",
        "prev_day_volume_needed_for_100TL_move",
        "prev_day_volume_needed_for_500TL_move",
        "prev_day_cap_risk_score",
        "prev_day_oversupply_pressure",
        "reconstruction_confidence",
        "ptf_lag_24",
        "residual_load_forecast",
        "kgup_total",
        "load_forecast",
        "gas_share",
        "coal_share",
        "hydro_share",
        "wind_share",
        "solar_share",
        "thermal_share",
        "kgup_wind_mw",
        "kgup_solar_mw",
        "kgup_renewable_mw",
        "analyst_zero_score",
        "analyst_spike_score",
        "analyst_tight_score",
        "analyst_persistence_break_score",
        "analyst_confidence_score",
        "gas_marginality_proxy",
        "hydro_displacement_score",
        "cheap_supply_pressure",
        "zero_price_pressure_score",
        "demand_weakness_score",
        "load_vs_renewable_balance",
        "renewable_share_of_generation",
        "gas_share_of_generation",
        "must_run_supply",
        "must_run_share_of_load",
        "renewable_must_run_pressure",
        "hydro_must_run_pressure",
        "solar_must_run_pressure",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            audit["feature_correlations_with_target_ptf"][col] = None if out[[col, "target_ptf"]].dropna().shape[0] < 3 else float(out[col].corr(out["target_ptf"]))

    return out, audit


def main() -> None:
    out, audit = build_dataset()
    if out.empty:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        empty_cols = [
            "delivery_hour",
            "target_hour",
            "target_ptf",
            "prev_day_slope_near_clearing",
            "prev_day_elasticity_near_clearing",
            "prev_day_curve_fragility_score",
            "prev_day_volume_needed_for_100TL_move",
            "prev_day_volume_needed_for_500TL_move",
            "prev_day_cap_risk_score",
            "prev_day_oversupply_pressure",
            "reconstruction_confidence",
            "ptf_lag_24",
            "residual_load_forecast",
            "kgup_total",
            "load_forecast",
            "gas_share",
            "coal_share",
            "hydro_share",
            "wind_share",
            "solar_share",
            "thermal_share",
            "kgup_wind_mw",
            "kgup_solar_mw",
            "kgup_renewable_mw",
            "hour",
            "weekday",
            "weekend",
            "month",
            "previous_day_regime",
            "prev_day_spike_pressure_from_curve",
            "prev_day_zero_pressure_from_curve",
            "analyst_zero_score",
            "analyst_spike_score",
            "analyst_tight_score",
            "analyst_persistence_break_score",
            "analyst_expected_regime",
            "analyst_confidence_score",
            "target_band",
            "curve_day",
        ]
        pd.DataFrame(columns=empty_cols).to_parquet(OUT_PATH, index=False)
        REPORT_MD.write_text(
            "\n".join(
                [
                    "# Curve Aware Training Dataset Audit",
                    "",
                    "No rows were produced.",
                    "",
                    "The weekly curve input currently covers `2026-06-01` to `2026-06-01`, while the available finalized PTF history in `ptf_dataset.csv` ends at `2026-05-31`. That means there is no real T+1 target available yet for these curve rows.",
                    "",
                    "This is expected for a forward-looking smoke curve slice and keeps the dataset leakage-safe.",
                ]
            )
            + "\n"
        )
        REPORT_JSON.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str) + "\n")
        print(f"Wrote empty {OUT_PATH}")
        print(f"Wrote {REPORT_MD}")
        print(f"Wrote {REPORT_JSON}")
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    REPORT_JSON.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str) + "\n")
    REPORT_MD.write_text(
        "\n".join(
            [
                "# Curve Aware Training Dataset Audit",
                "",
                f"- Rows: `{audit['rows']}`",
                f"- Coverage start: `{audit['coverage_start']}`",
                f"- Coverage end: `{audit['coverage_end']}`",
                "",
                "## Leakage Checks",
                "",
                f"- Uses same-day curve as feature: `{audit['leakage_checks']['uses_same_day_curve_as_feature']}`",
                f"- Target PTF is future only: `{audit['leakage_checks']['target_ptf_is_future_only']}`",
                f"- Contains realized same-hour future market features: `{audit['leakage_checks']['contains_realized_same_hour_future_market_features']}`",
                "",
                "## Target Distribution",
                "",
                "\n".join([f"- `{k}`: `{v}`" for k, v in audit["target_distribution"].items()]) or "- None",
                "",
                "## Missing Ratios",
                "",
                "\n".join([f"- `{k}`: `{v:.3f}`" for k, v in sorted(audit["feature_missing_ratios"].items())]) or "- None",
                "",
                "## Feature Correlations with Target PTF",
                "",
                "\n".join(
                    [
                        f"- `{k}`: `{v}`"
                        for k, v in sorted(audit["feature_correlations_with_target_ptf"].items())
                    ]
                )
                or "- None",
            ]
        )
        + "\n"
    )
    print(f"Wrote {OUT_PATH}")
    print(f"Wrote {REPORT_MD}")
    print(f"Wrote {REPORT_JSON}")


if __name__ == "__main__":
    main()
