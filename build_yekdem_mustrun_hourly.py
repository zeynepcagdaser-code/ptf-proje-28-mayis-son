#!/usr/bin/env python3
"""
Build a YEKDEM / must-run hourly proxy table.

The repository currently does not contain a dense plant-level KGUP archive with
publication timestamps, so this script constructs a structurally grounded
hourly proxy table from the available must-run proxy layer and aggregates.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent

PROXY_PATH = PROJECT_ROOT / "data" / "features" / "must_run_proxy_v2.parquet"
REGIME_FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "regime_feature_store.parquet"
REGIME_LABELS_PATH = PROJECT_ROOT / "data" / "regime_labels.csv"
PTF_PATH = PROJECT_ROOT / "data" / "ptf_dataset.csv"

OUT_PATH = PROJECT_ROOT / "data" / "features" / "yekdem_mustrun_hourly.parquet"
REPORT_MD = PROJECT_ROOT / "reports" / "yekdem_mustrun_hourly_report.md"
REPORT_JSON = PROJECT_ROOT / "reports" / "yekdem_mustrun_hourly_report.json"


def to_naive(s: pd.Series) -> pd.Series:
    out = pd.to_datetime(s, errors="coerce")
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_localize(None)
    return out


def price_band(price: pd.Series) -> pd.Series:
    return pd.cut(
        price,
        bins=[-np.inf, 50, 1500, 4000, np.inf],
        labels=["negative_zero_pressure", "normal", "tight", "spike_cap"],
    ).astype("string")


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    proxy = pd.read_parquet(PROXY_PATH)
    if "ts_hour" in proxy.columns:
        proxy["ts_hour"] = to_naive(proxy["ts_hour"])

    regime = pd.read_parquet(REGIME_FEATURES_PATH)
    regime["ts_hour"] = to_naive(regime["ts_hour"])

    labels = pd.read_csv(REGIME_LABELS_PATH)
    labels["ts_hour"] = to_naive(labels["ts_hour"])

    ptf = pd.read_csv(PTF_PATH)
    ptf["delivery_hour"] = to_naive(ptf["date"])
    if "hour" in ptf.columns:
        hour_num = pd.to_numeric(ptf["hour"].astype(str).str.extract(r"(\d{1,2})")[0], errors="coerce")
        ptf["delivery_hour"] = ptf["delivery_hour"].dt.normalize() + pd.to_timedelta(hour_num.fillna(0), unit="h")
    ptf["price"] = pd.to_numeric(ptf["price"], errors="coerce")

    return proxy, regime, labels, ptf


def build_hourly(proxy: pd.DataFrame, regime: pd.DataFrame, labels: pd.DataFrame, ptf: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if proxy.empty:
        return pd.DataFrame(), {"available": False, "reason": "must_run_proxy_v2 is empty"}

    proxy = proxy.copy()
    if "ts_hour" not in proxy.columns:
        proxy["ts_hour"] = pd.to_datetime(proxy.index, errors="coerce")
    proxy["delivery_hour"] = proxy["ts_hour"]

    out = proxy.rename(
        columns={
            "must_run_supply_proxy": "must_run_supply",
            "must_run_wind_proxy": "must_run_wind",
            "must_run_solar_proxy": "must_run_solar",
            "must_run_hydro_proxy": "must_run_hydro",
            "must_run_biomass_proxy": "must_run_biomass",
            "must_run_geothermal_proxy": "must_run_geothermal",
        }
    ).copy()

    for col in [
        "must_run_supply",
        "must_run_wind",
        "must_run_solar",
        "must_run_hydro",
        "must_run_biomass",
        "must_run_geothermal",
        "renewable_concentration_score",
        "solar_oversupply_score",
        "hydro_pressure_score",
        "renewable_ramp_score",
        "renewable_share_of_load",
        "residual_load_after_renewables",
        "renewable_curtailment_pressure_proxy",
    ]:
        if col not in out.columns:
            out[col] = np.nan

    out["must_run_share"] = out["must_run_share"] if "must_run_share" in out.columns else out["must_run_supply"] / out["load_forecast"].replace(0, np.nan)
    out["residual_load_after_must_run"] = out["load_forecast"] - out["must_run_supply"]
    out["previous_day_regime"] = price_band(out.get("price", pd.Series(dtype=float)))

    out = out.merge(regime[["ts_hour", "ptf_lag_24", "residual_load_forecast", "residual_load_ramp", "load_minus_kgup", "kgup_total"]], left_on="ts_hour", right_on="ts_hour", how="left")
    out = out.merge(
        labels[["ts_hour", "target_regime"]].rename(columns={"target_regime": "target_regime_label"}),
        on="ts_hour",
        how="left",
    )
    out = out.merge(ptf[["delivery_hour", "price"]].rename(columns={"delivery_hour": "ts_hour", "price": "final_ptf"}), on="ts_hour", how="left")

    out["strict_point_in_time_safe"] = 0
    out["structural_market_proxy"] = 1
    out["source_type"] = "proxy_v2"
    out["ts_hour"] = pd.to_datetime(out["ts_hour"], errors="coerce")
    out = out.sort_values("ts_hour").reset_index(drop=True)

    out["target_regime"] = out.get("target_regime_label")
    audit = {
        "available": True,
        "rows": int(len(out)),
        "coverage_start": str(out["ts_hour"].min()) if not out.empty else None,
        "coverage_end": str(out["ts_hour"].max()) if not out.empty else None,
        "rows_with_price": int(out["final_ptf"].notna().sum()),
        "rows_with_regime": int(out["target_regime"].notna().sum()) if "target_regime" in out.columns else 0,
        "proxy_columns": [c for c in proxy.columns if c.startswith("must_run_") or c.endswith("_score")],
        "leakage_policy": "structural_proxy_only",
        "missing_real_plant_level_kgup_reason": "No dense plant-level KGUP archive with publication timestamps exists in the repository yet.",
    }
    return out, audit


def main() -> None:
    proxy, regime, labels, ptf = load_tables()
    out, audit = build_hourly(proxy, regime, labels, ptf)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    REPORT_JSON.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str) + "\n")
    REPORT_MD.write_text(
        "\n".join(
            [
                "# YEKDEM Must-Run Hourly Proxy",
                "",
                f"- Rows: `{audit['rows']}`",
                f"- Coverage: `{audit['coverage_start']}` → `{audit['coverage_end']}`",
                f"- Rows with finalized PTF: `{audit['rows_with_price']}`",
                f"- Rows with regime label: `{audit['rows_with_regime']}`",
                "",
                "## Interpretation",
                "",
                "This table is a structural must-run proxy, not a strict plant-level YEKDEM archive.",
                "It is built from `must_run_proxy_v2` because the repository currently lacks a dense plant-level KGUP archive with publication timestamps.",
                "",
                "## Leakage Policy",
                "",
                f"- `{audit['leakage_policy']}`",
                f"- Missing real plant-level reason: `{audit['missing_real_plant_level_kgup_reason']}`",
                "",
                "## Useful Columns",
                "",
                "- `must_run_supply`",
                "- `must_run_wind`",
                "- `must_run_solar`",
                "- `must_run_hydro`",
                "- `must_run_biomass`",
                "- `must_run_geothermal`",
                "- `must_run_share`",
                "- `residual_load_after_must_run`",
                "- `renewable_concentration_score`",
                "- `solar_oversupply_score`",
                "- `renewable_curtailment_pressure_proxy`",
                "",
            ]
        )
        + "\n"
    )
    print(f"Wrote {OUT_PATH}")
    print(f"Wrote {REPORT_MD}")
    print(f"Wrote {REPORT_JSON}")


if __name__ == "__main__":
    main()
