#!/usr/bin/env python3
"""
Build a second-generation must-run / renewable pressure proxy feature set.

Inputs:
    - data/plant_level_kgup/raw_smoke/
    - YEKDEM registry Excel files
    - aggregate KGUP
    - load forecast
    - regime labels

This script does not train a model. It produces structural, not strict point-
in-time-safe, proxy features from smoke plant-level KGUP and audits which
regimes they appear to explain best.
"""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from market_aware_ptf_pipeline_skeleton import YEKDEM_Registry

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_SMOKE_ROOT = PROJECT_ROOT / "data" / "plant_level_kgup" / "raw_smoke"
FEATURE_PATH = PROJECT_ROOT / "data" / "features" / "must_run_proxy_v2.parquet"
REPORT_MD = PROJECT_ROOT / "reports" / "must_run_proxy_v2_analysis.md"
REPORT_JSON = PROJECT_ROOT / "reports" / "must_run_proxy_v2_analysis.json"
LABELS_PATH = PROJECT_ROOT / "data" / "regime_labels.csv"
LOAD_FORECAST_PATH = PROJECT_ROOT / "data" / "load_forecast.csv"
AGG_KGUP_PATH = PROJECT_ROOT / "data" / "clean" / "kgup_hourly.parquet"


def slug(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()


def load_latest_smoke(raw_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not raw_root.exists():
        return pd.DataFrame(), {"raw_smoke_dir_exists": False}
    dated = sorted([p for p in raw_root.iterdir() if p.is_dir()])
    if not dated:
        return pd.DataFrame(), {"raw_smoke_dir_exists": True, "available_dates": []}
    latest = dated[-1]
    items_path = latest / "dpp_bulk_items.json"
    if not items_path.exists():
        items_path = latest / "dpp_bulk.json"
    if not items_path.exists():
        return pd.DataFrame(), {"raw_smoke_dir_exists": True, "available_dates": [p.name for p in dated], "latest": latest.name}
    data = json.loads(items_path.read_text())
    if isinstance(data, dict):
        data = data.get("items", [])
    frame = pd.DataFrame(data)
    meta = {
        "raw_smoke_dir_exists": True,
        "available_dates": [p.name for p in dated],
        "latest": latest.name,
        "raw_rows": int(len(frame)),
        "raw_keys": list(frame.columns),
    }
    return frame, meta


def load_labels() -> pd.DataFrame:
    labels = pd.read_csv(LABELS_PATH)
    labels["ts_hour"] = pd.to_datetime(labels["ts_hour"], errors="coerce")
    return labels


def load_load_forecast() -> pd.DataFrame:
    load = pd.read_csv(LOAD_FORECAST_PATH)
    load["ts_hour"] = pd.to_datetime(load["date"], errors="coerce")
    if getattr(load["ts_hour"].dt, "tz", None) is not None:
        load["ts_hour"] = load["ts_hour"].dt.tz_localize(None)
    load["load_forecast"] = pd.to_numeric(load["lep"], errors="coerce")
    return load[["ts_hour", "load_forecast"]]


def load_aggregate_kgup() -> pd.DataFrame:
    if AGG_KGUP_PATH.exists():
        from src.utils.io_utils import read_parquet_with_normalized_ts
        agg = read_parquet_with_normalized_ts(AGG_KGUP_PATH)
        agg["ts_hour"] = pd.to_datetime(agg["ts_hour"], errors="coerce")
        if getattr(agg["ts_hour"].dt, "tz", None) is not None:
            agg["ts_hour"] = agg["ts_hour"].dt.tz_localize(None)
        return agg[["ts_hour", "toplam", "ruzgar", "gunes", "barajli", "biokutle", "jeotermal"]].rename(
            columns={
                "toplam": "kgup_total",
                "ruzgar": "kgup_wind",
                "gunes": "kgup_solar",
                "barajli": "kgup_hydro",
                "biokutle": "kgup_biomass",
                "jeotermal": "kgup_geothermal",
            }
        )
    agg = pd.read_csv(AGG_KGUP_PATH.with_suffix(".csv"))
    agg["ts_hour"] = pd.to_datetime(agg["date"], errors="coerce")
    if getattr(agg["ts_hour"].dt, "tz", None) is not None:
        agg["ts_hour"] = agg["ts_hour"].dt.tz_localize(None)
    return agg[["ts_hour", "toplam", "ruzgar", "gunes", "barajli", "biokutle", "jeotermal"]].rename(
        columns={
            "toplam": "kgup_total",
            "ruzgar": "kgup_wind",
            "gunes": "kgup_solar",
            "barajli": "kgup_hydro",
            "biokutle": "kgup_biomass",
            "jeotermal": "kgup_geothermal",
        }
    )


def normalize_smoke(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["ts_hour", "uevcbId", "uevcbName", "orgId", "toplam", "dogalgaz", "ruzgar", "gunes", "barajli", "biokutle", "jeotermal", "akarsu", "diger"])
    out = frame.copy()
    out["ts_hour"] = pd.to_datetime(out["date"].astype(str) + " " + out["time"].astype(str), errors="coerce")
    if getattr(out["ts_hour"].dt, "tz", None) is not None:
        out["ts_hour"] = out["ts_hour"].dt.tz_localize(None)
    for col in ["toplam", "dogalgaz", "ruzgar", "gunes", "barajli", "biokutle", "jeotermal", "akarsu", "diger"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ["uevcbId", "orgId"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=["ts_hour"])


def build_proxy_features(smoke: pd.DataFrame, registry: YEKDEM_Registry, labels: pd.DataFrame, load: pd.DataFrame, agg: pd.DataFrame) -> pd.DataFrame:
    if smoke.empty:
        return pd.DataFrame()

    smoke = normalize_smoke(smoke)
    smoke["year"] = smoke["ts_hour"].dt.year
    smoke["hour"] = smoke["ts_hour"].dt.hour
    smoke["is_daylight"] = smoke["hour"].between(7, 18).astype(int)
    smoke["is_midday"] = smoke["hour"].between(10, 15).astype(int)
    smoke["is_evening"] = smoke["hour"].between(17, 22).astype(int)

    registry_lookup = registry.registry[[registry.registry_id_col, "registry_year", "registry_source_file"]].copy()
    registry_lookup = registry_lookup.rename(columns={registry.registry_id_col: "eic_code"})
    registry_lookup["eic_code"] = registry_lookup["eic_code"].astype(str).str.strip()
    registry_lookup["registry_year"] = pd.to_numeric(registry_lookup["registry_year"], errors="coerce")

    if "uevcbId" in smoke.columns:
        smoke["uevcbId"] = smoke["uevcbId"].astype("Int64")

    # Structural proxy assumes that observed UEVÇB-level production can represent
    # must-run pressure, especially for renewable-heavy plants.
    hourly = smoke.groupby("ts_hour", as_index=False).agg(
        must_run_supply_proxy=("toplam", "sum"),
        must_run_wind_proxy=("ruzgar", "sum"),
        must_run_solar_proxy=("gunes", "sum"),
        must_run_hydro_proxy=("barajli", "sum"),
        must_run_biomass_proxy=("biokutle", "sum"),
        must_run_geothermal_proxy=("jeotermal", "sum"),
        raw_uevcb_count=("uevcbId", "nunique"),
    )
    hourly["renewable_concentration_score"] = (
        hourly["must_run_wind_proxy"]
        + hourly["must_run_solar_proxy"]
        + hourly["must_run_hydro_proxy"]
        + hourly["must_run_biomass_proxy"]
        + hourly["must_run_geothermal_proxy"]
    ) / hourly["must_run_supply_proxy"].replace(0, np.nan)
    hourly["solar_oversupply_score"] = hourly["must_run_solar_proxy"] / hourly["must_run_supply_proxy"].replace(0, np.nan)
    hourly["hydro_pressure_score"] = hourly["must_run_hydro_proxy"] / hourly["must_run_supply_proxy"].replace(0, np.nan)
    hourly["renewable_ramp_score"] = hourly["must_run_supply_proxy"].diff().abs()
    hourly["renewable_ramp_1h"] = hourly["must_run_supply_proxy"].diff()
    hourly["renewable_ramp_24h"] = hourly["must_run_supply_proxy"].diff(24)
    hourly["renewable_share_of_load"] = hourly["must_run_supply_proxy"] / load.set_index("ts_hour")["load_forecast"].reindex(hourly["ts_hour"]).to_numpy()
    hourly["residual_load_after_renewables"] = (
        load.set_index("ts_hour")["load_forecast"].reindex(hourly["ts_hour"]).to_numpy() - hourly["must_run_supply_proxy"].to_numpy()
    )
    hourly["renewable_curtailment_pressure_proxy"] = (
        hourly["renewable_concentration_score"].fillna(0)
        * (hourly["renewable_share_of_load"].fillna(0))
        * (1 + hourly["ts_hour"].dt.hour.between(10, 15).astype(int))
    )
    hourly["evening_solar_collapse"] = -hourly["must_run_solar_proxy"].diff(-1)
    hourly["same_hour_renewable_ramp"] = hourly["must_run_supply_proxy"].diff()
    hourly["strict_point_in_time_safe"] = 0
    hourly["structural_market_proxy"] = 1
    hourly["hour"] = hourly["ts_hour"].dt.hour

    # Join aggregate KGUP and labels for audit.
    hourly = hourly.merge(load, on="ts_hour", how="left").merge(agg, on="ts_hour", how="left")
    hourly = hourly.merge(labels[["ts_hour", "price", "target_regime"]], on="ts_hour", how="left")
    return hourly.sort_values("ts_hour")


def safe_corr(a: pd.Series, b: pd.Series) -> float | None:
    frame = pd.concat([a, b], axis=1).dropna()
    if len(frame) < 3:
        return None
    return float(frame.iloc[:, 0].corr(frame.iloc[:, 1]))


def regime_audit(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"available": False, "reason": "No smoke proxy rows were built."}
    rows = []
    for regime, group in frame.groupby("target_regime", observed=False):
        rows.append(
            {
                "target_regime": str(regime),
                "rows": int(len(group)),
                "must_run_supply_proxy_mean": float(group["must_run_supply_proxy"].mean()),
                "must_run_supply_proxy_median": float(group["must_run_supply_proxy"].median()),
                "renewable_concentration_score_mean": float(group["renewable_concentration_score"].mean()),
                "solar_oversupply_score_mean": float(group["solar_oversupply_score"].mean()),
                "hydro_pressure_score_mean": float(group["hydro_pressure_score"].mean()),
                "renewable_curtailment_pressure_proxy_mean": float(group["renewable_curtailment_pressure_proxy"].mean()),
                "price_mean": float(group["price"].mean()),
            }
        )
    low_price = frame[frame["price"] <= 50]
    zero_pressure = frame[frame["target_regime"] == "negative_zero_pressure"]
    midday = frame[frame["hour"].between(10, 15)]
    return {
        "available": True,
        "rows": int(len(frame)),
        "coverage_start": str(frame["ts_hour"].min()),
        "coverage_end": str(frame["ts_hour"].max()),
        "correlations": {
            "must_run_supply_proxy_vs_price": safe_corr(frame["must_run_supply_proxy"], frame["price"]),
            "renewable_concentration_vs_price": safe_corr(frame["renewable_concentration_score"], frame["price"]),
            "solar_oversupply_vs_price_midday": safe_corr(midday["solar_oversupply_score"], midday["price"]),
            "renewable_curtailment_pressure_vs_price": safe_corr(frame["renewable_curtailment_pressure_proxy"], frame["price"]),
        },
        "slices": {
            "low_price_hours": {
                "rows": int(len(low_price)),
                "must_run_supply_proxy_mean": float(low_price["must_run_supply_proxy"].mean()) if not low_price.empty else None,
                "renewable_concentration_score_mean": float(low_price["renewable_concentration_score"].mean()) if not low_price.empty else None,
            },
            "zero_pressure_hours": {
                "rows": int(len(zero_pressure)),
                "must_run_supply_proxy_mean": float(zero_pressure["must_run_supply_proxy"].mean()) if not zero_pressure.empty else None,
                "renewable_concentration_score_mean": float(zero_pressure["renewable_concentration_score"].mean()) if not zero_pressure.empty else None,
            },
            "midday_hours": {
                "rows": int(len(midday)),
                "solar_oversupply_score_mean": float(midday["solar_oversupply_score"].mean()) if not midday.empty else None,
                "renewable_curtailment_pressure_proxy_mean": float(midday["renewable_curtailment_pressure_proxy"].mean()) if not midday.empty else None,
            },
        },
        "regime_summary": rows,
    }


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    work = frame.fillna("")
    headers = [str(col) for col in work.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in work.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in work.columns) + " |")
    return "\n".join(lines)


def write_reports(frame: pd.DataFrame, audit: dict[str, Any]) -> None:
    FEATURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEATURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    FEATURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    from src.utils.safe_io import atomic_parquet_write
    atomic_parquet_write(frame, str(FEATURE_PATH), index=False)
    REPORT_JSON.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")

    lines = [
        "# Must Run Proxy V2 Analysis",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat()}`",
        "",
    ]
    if not audit.get("available"):
        lines.extend(
            [
                "No rows available for proxy analysis.",
                "",
                f"Reason: `{audit.get('reason')}`",
            ]
        )
    else:
        lines.extend(
            [
                f"- Rows analyzed: `{audit['rows']}`",
                f"- Coverage: `{audit['coverage_start']}` → `{audit['coverage_end']}`",
                "",
                "## Correlations",
                "",
                f"- must_run_supply_proxy vs price: `{audit['correlations']['must_run_supply_proxy_vs_price']}`",
                f"- renewable_concentration_score vs price: `{audit['correlations']['renewable_concentration_vs_price']}`",
                f"- solar_oversupply_score vs price midday: `{audit['correlations']['solar_oversupply_vs_price_midday']}`",
                f"- renewable_curtailment_pressure_proxy vs price: `{audit['correlations']['renewable_curtailment_pressure_vs_price']}`",
                "",
                "## Regime Summary",
                "",
            ]
        )
        regime_df = pd.DataFrame(audit["regime_summary"])
        lines.append(markdown_table(regime_df))
        lines.extend(
            [
                "",
                "## Slices",
                "",
                json.dumps(audit["slices"], ensure_ascii=False, indent=2),
            ]
        )
    REPORT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    smoke, smoke_meta = load_latest_smoke(RAW_SMOKE_ROOT)
    registry_files = YEKDEM_Registry.discover_files(PROJECT_ROOT)
    registry = YEKDEM_Registry(registry_files)
    labels = load_labels()
    load = load_load_forecast()
    agg = load_aggregate_kgup()

    frame = build_proxy_features(smoke, registry, labels, load, agg)
    audit = regime_audit(frame)
    audit["smoke_meta"] = smoke_meta
    audit["feature_columns"] = list(frame.columns)
    audit["strict_point_in_time_safe"] = False
    audit["structural_market_proxy"] = True

    write_reports(frame, audit)
    print(f"Wrote {FEATURE_PATH}")
    print(f"Wrote {REPORT_MD}")
    print(f"Wrote {REPORT_JSON}")


if __name__ == "__main__":
    main()
