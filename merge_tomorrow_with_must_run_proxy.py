#!/usr/bin/env python3
"""
Enrich tomorrow morning features with must-run proxy v2 features.

This is an analysis-only merge. No model training.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
TOMORROW_PATH = PROJECT_ROOT / "data" / "features" / "tomorrow_morning_features.parquet"
PROXY_PATH = PROJECT_ROOT / "data" / "features" / "must_run_proxy_v2.parquet"
OUT_PATH = PROJECT_ROOT / "data" / "features" / "tomorrow_morning_features_enriched.parquet"
REPORT_MD = PROJECT_ROOT / "reports" / "tomorrow_must_run_enrichment_audit.md"
REPORT_JSON = PROJECT_ROOT / "reports" / "tomorrow_must_run_enrichment_audit.json"


def load_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "ts_hour" in frame.columns:
        frame["ts_hour"] = pd.to_datetime(frame["ts_hour"], errors="coerce")
    if "delivery_hour" in frame.columns:
        frame["delivery_hour"] = pd.to_datetime(frame["delivery_hour"], errors="coerce")
    return frame


def build_enrichment(tomorrow: pd.DataFrame, proxy: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if tomorrow.empty:
        return tomorrow.copy(), {"available": False, "reason": "tomorrow_morning_features is empty"}

    proxy = proxy.copy()
    if not proxy.empty:
        proxy["ts_hour"] = pd.to_datetime(proxy["ts_hour"], errors="coerce")
        if getattr(proxy["ts_hour"].dt, "tz", None) is not None:
            proxy["ts_hour"] = proxy["ts_hour"].dt.tz_localize(None)
        proxy["hour"] = proxy["ts_hour"].dt.hour
    tomorrow = tomorrow.copy()
    tomorrow["ts_hour"] = pd.to_datetime(tomorrow["ts_hour"], errors="coerce")
    if getattr(tomorrow["ts_hour"].dt, "tz", None) is not None:
        tomorrow["ts_hour"] = tomorrow["ts_hour"].dt.tz_localize(None)
    tomorrow["hour"] = tomorrow["ts_hour"].dt.hour
    direct_merge = tomorrow.merge(proxy, on="ts_hour", how="left", suffixes=("", "_proxy"))
    if not proxy.empty and "hour" in proxy.columns:
        hour_proxy = proxy.drop(columns=[c for c in ["ts_hour"] if c in proxy.columns]).copy()
        hour_proxy = hour_proxy.sort_values("hour").drop_duplicates("hour", keep="last")
        merged = direct_merge.drop(columns=[c for c in hour_proxy.columns if c in direct_merge.columns and c != "hour"], errors="ignore")
        merged = merged.merge(hour_proxy, on="hour", how="left", suffixes=("", "_proxy_hod"))
        merged["proxy_match_mode"] = np.where(direct_merge["must_run_supply_proxy"].notna(), "direct_ts", np.where(merged["must_run_supply_proxy"].notna(), "hour_of_day", "fallback"))
        for col in [c for c in proxy.columns if c not in {"ts_hour", "hour"}]:
            if col in direct_merge.columns and col not in merged.columns:
                merged[col] = direct_merge[col]
    else:
        merged = direct_merge.copy()
        merged["proxy_match_mode"] = np.where(merged["must_run_supply_proxy"].notna(), "direct_ts", "fallback")

    fallback_cols = [
        "must_run_supply_proxy",
        "must_run_wind_proxy",
        "must_run_solar_proxy",
        "must_run_hydro_proxy",
        "must_run_biomass_proxy",
        "must_run_geothermal_proxy",
        "renewable_concentration_score",
        "solar_oversupply_score",
        "hydro_pressure_score",
        "renewable_ramp_score",
        "renewable_ramp_1h",
        "renewable_ramp_24h",
        "renewable_share_of_load",
        "residual_load_after_renewables",
        "renewable_curtailment_pressure_proxy",
        "evening_solar_collapse",
        "same_hour_renewable_ramp",
        "strict_point_in_time_safe",
        "structural_market_proxy",
    ]

    merged["proxy_rows_matched"] = merged["must_run_supply_proxy"].notna().astype(int)
    merged["proxy_rows_missing"] = merged["must_run_supply_proxy"].isna().astype(int)
    merged["proxy_source"] = np.where(merged["must_run_supply_proxy"].notna(), "must_run_proxy_v2", "fallback")

    merged["must_run_supply_proxy_missing"] = merged["must_run_supply_proxy"].isna().astype(int)
    merged["must_run_wind_proxy_missing"] = merged["must_run_wind_proxy"].isna().astype(int)
    merged["must_run_solar_proxy_missing"] = merged["must_run_solar_proxy"].isna().astype(int)
    merged["must_run_hydro_proxy_missing"] = merged["must_run_hydro_proxy"].isna().astype(int)
    merged["renewable_concentration_score_missing"] = merged["renewable_concentration_score"].isna().astype(int)
    merged["solar_oversupply_score_missing"] = merged["solar_oversupply_score"].isna().astype(int)
    merged["hydro_pressure_score_missing"] = merged["hydro_pressure_score"].isna().astype(int)
    merged["renewable_ramp_score_missing"] = merged["renewable_ramp_score"].isna().astype(int)
    merged["renewable_share_of_load_missing"] = merged["renewable_share_of_load"].isna().astype(int)
    merged["residual_load_after_renewables_missing"] = merged["residual_load_after_renewables"].isna().astype(int)
    merged["renewable_curtailment_pressure_proxy_missing"] = merged["renewable_curtailment_pressure_proxy"].isna().astype(int)
    merged["strict_point_in_time_safe"] = merged.get("strict_point_in_time_safe", 0).fillna(0).astype(int)
    merged["structural_market_proxy"] = merged.get("structural_market_proxy", 1).fillna(1).astype(int)

    if "must_run_supply" in merged.columns:
        merged["must_run_supply_final"] = merged["must_run_supply"]
    else:
        merged["must_run_supply_final"] = np.nan
    for src, final in [
        ("must_run_supply_proxy", "must_run_supply_final"),
        ("must_run_wind_proxy", "must_run_wind_final"),
        ("must_run_solar_proxy", "must_run_solar_final"),
        ("must_run_hydro_proxy", "must_run_hydro_final"),
    ]:
        if final not in merged.columns:
            merged[final] = merged[src]

    for col in fallback_cols:
        if col not in merged.columns:
            continue
        merged[f"{col}_used_fallback"] = merged[col].isna().astype(int)
        if col in ["must_run_supply_proxy", "must_run_wind_proxy", "must_run_solar_proxy", "must_run_hydro_proxy"]:
            merged[col] = merged[col].fillna(merged.get(col.replace("_proxy", ""), merged[col]))
        elif col in ["renewable_concentration_score", "solar_oversupply_score", "hydro_pressure_score", "renewable_ramp_score"]:
            merged[col] = merged[col].fillna(0.0)
        elif col in ["renewable_share_of_load", "residual_load_after_renewables", "renewable_curtailment_pressure_proxy"]:
            merged[col] = merged[col].fillna(merged.get("load_forecast"))
        else:
            merged[col] = merged[col].fillna(0)

    diagnostics = {
        "available": True,
        "tomorrow_rows": int(len(tomorrow)),
        "proxy_rows": int(len(proxy)),
        "matched_rows": int(merged["proxy_rows_matched"].sum()),
        "fallback_rows": int(merged["proxy_rows_missing"].sum()),
        "match_mode_counts": merged["proxy_match_mode"].value_counts(dropna=False).to_dict(),
        "missing_feature_columns": [
            col for col in fallback_cols if merged.get(col) is not None and merged[col].isna().any()
        ],
        "merge_keys": ["ts_hour"],
        "strict_point_in_time_safe": int(merged["strict_point_in_time_safe"].sum()),
        "structural_market_proxy": int(merged["structural_market_proxy"].sum()),
        "leakage_risk": "medium" if int(merged["strict_point_in_time_safe"].sum()) < len(merged) else "low",
        "feature_columns": list(merged.columns),
    }
    return merged.sort_values("ts_hour"), diagnostics


def write_reports(frame: pd.DataFrame, audit: dict[str, Any]) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT_PATH, index=False)
    REPORT_JSON.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str) + "\n")

    lines = [
        "# Tomorrow Must-Run Enrichment Audit",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        f"- Tomorrow rows: `{audit.get('tomorrow_rows', 0)}`",
        f"- Proxy rows available: `{audit.get('proxy_rows', 0)}`",
        f"- Proxy rows matched: `{audit.get('matched_rows', 0)}`",
        f"- Fallback rows: `{audit.get('fallback_rows', 0)}`",
        f"- strict_point_in_time_safe rows: `{audit.get('strict_point_in_time_safe', 0)}`",
        f"- structural_market_proxy rows: `{audit.get('structural_market_proxy', 0)}`",
        f"- Leakage risk: `{audit.get('leakage_risk', 'unknown')}`",
        "",
        "## Missing Features",
        "",
    ]
    missing = audit.get("missing_feature_columns", [])
    lines.append("\n".join(f"- `{col}`" for col in missing) if missing else "- None")
    lines.extend(
        [
            "",
            "## Can this be used for tomorrow?",
            "",
            "Yes, for analysis and directional feature enrichment. The merge is usable for tomorrow morning inference, but not strict point-in-time safe because the proxy source is structural and comes from smoke data without publication timestamps.",
            "",
            "## Leakage Notes",
            "",
            "The enrichment preserves `strict_point_in_time_safe` and `structural_market_proxy`. The risk is structural rather than accidental label leakage: the proxy comes from smoke raw generation with no publication timestamp, so it should not be treated as a historical live-as-of feature.",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    tomorrow = load_frame(TOMORROW_PATH)
    proxy = load_frame(PROXY_PATH)
    enriched, audit = build_enrichment(tomorrow, proxy)
    write_reports(enriched, audit)
    print(f"Wrote {OUT_PATH}")
    print(f"Wrote {REPORT_MD}")
    print(f"Wrote {REPORT_JSON}")


if __name__ == "__main__":
    main()
