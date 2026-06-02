#!/usr/bin/env python3
"""
Feature inventory + missing feature audit for lstm_next24_v1.parquet.

Produces:
  - reports/feature_inventory_report.md/json
  - reports/missing_feature_report.md/json
  - reports/feature_sanity_report.md/json

No model training.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.config import (
    EXCLUDED_FROM_MAIN_REGRESSION,
    LOW_PRICE_CLASSIFIER_FEATURES,
    MAIN_REGRESSION_FEATURES,
    RISK_DASHBOARD_FEATURES,
    THESIS_DATA_DEBT_GROUPS,
    resolve_feature_list,
)

_MAIN_REGRESSION_SET = set(MAIN_REGRESSION_FEATURES)
_LOW_PRICE_CLASSIFIER_SET = set(LOW_PRICE_CLASSIFIER_FEATURES)
_RISK_DASHBOARD_SET = set(RISK_DASHBOARD_FEATURES)
_EXCLUDED_FROM_MAIN_SET = set(EXCLUDED_FROM_MAIN_REGRESSION)
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "lstm_next24_v1.parquet"
MASTER_PATH = PROJECT_ROOT / "data" / "master" / "master_hourly_v1.parquet"

OUT_INV_JSON = PROJECT_ROOT / "reports" / "feature_inventory_report.json"
OUT_INV_MD = PROJECT_ROOT / "reports" / "feature_inventory_report.md"
OUT_MISSING_JSON = PROJECT_ROOT / "reports" / "missing_feature_report.json"
OUT_MISSING_MD = PROJECT_ROOT / "reports" / "missing_feature_report.md"
OUT_SANITY_JSON = PROJECT_ROOT / "reports" / "feature_sanity_report.json"
OUT_SANITY_MD = PROJECT_ROOT / "reports" / "feature_sanity_report.md"

# Notes about timing / operational usage modes (see docs/feature_usage_modes.md)
USAGE_MODE_NOTES = [
    "current FİBA/FİBS (dam_price_independent_* ve fiba_fibs_*) post_dam_publication_mode için kullanılabilir.",
    "strict_forecast_mode için current FİBA/FİBS yerine lagged (örn. *_lag_24 / *_lag_168) tercih edilmeli.",
    "current GRF (grf_tl_1000sm3) yayın zamanı belirsizliği nedeniyle main modelde lagged/türetilmiş versiyonlar öncelikli (örn. grf_tl_lag_1d ve *_pressure_lag_1d).",
]


FAMILY_ORDER = [
    "ptf_lag_rolling",
    "calendar_holiday",
    "load_demand",
    "kgup_source_mix",
    "renewable_pressure",
    "thermal_price_setting",
    "smf_yal_yat_lagged",
    "wind_forecast",
    "outage",
    "fuel_currency",
    "cap_imbalance",
    "yekdem_merchant_proxy",
    "low_zero_price_risk",
    "other",
]


REQUESTED_FEATURES = [
    "low_load_flag",
    "holiday_low_load_flag",
    "renewable_pressure",
    "hydro_pressure",
    "res_ges_hes_pressure",
    "zero_price_risk_proxy",
    "low_price_regime_score",
    "gas_share",
    "coal_share",
    "gas_coal_competition_index",
    "thermal_price_setting_share",
    "gas_marginal_proxy",
    "merchant_proxy_share",
    "non_merchant_proxy_share",
    "ttf_gas_price",
    "brent_oil_price",
    "usd_try",
    "eur_try",
    "ttf_try_proxy",
    "fuel_cost_index",
    "gas_cost_pressure",
    "gas_marginal_pressure",
    "price_cap",
    "ptf_to_cap_ratio",
    "smf_to_cap_ratio",
    "spread_risk_flag",
    "imbalance_cost_proxy",
    "yekdem_unit_price",
    "kgup_excess_generation",
    "yekdem_revenue_loss_proxy",
]


def _is_target(col: str) -> bool:
    return col.startswith("target_")


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in {"ts_hour", "split"} and not _is_target(c)]


def infer_family(col: str) -> str:
    if col.startswith("ptf_lag_") or col.startswith("ptf_roll_") or col.startswith("ptf_rolling_") or col.startswith("ptf_change_"):
        return "ptf_lag_rolling"
    if col in {"hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "hour_of_day", "month", "is_weekend", "is_summer", "is_winter"}:
        return "calendar_holiday"
    if col.startswith("is_holiday"):
        return "calendar_holiday"
    if col.startswith("load_") or col.endswith("_minus_load") or col in {"kgup_total_minus_load"}:
        return "load_demand"
    if col.startswith("kgup_"):
        return "kgup_source_mix"
    if col in {"res_share", "solar_share", "hydro_share", "renewable_pressure", "renewable_suppression_pressure", "wind_forecast_share"}:
        return "renewable_pressure"
    if col in {"thermal_price_setting_share", "kgup_thermal_share", "kgup_renewable_share", "gas_share", "coal_share", "gas_coal_balance", "gas_coal_competition_index"}:
        return "thermal_price_setting"
    if col.startswith("smf_") or col.startswith("yal_yat_") or col.endswith("_lag_24") or col.endswith("_lag_48") or col.endswith("_lag_168") or col.startswith("smf_ptf_spread_"):
        return "smf_yal_yat_lagged"
    if col.startswith("wind_"):
        return "wind_forecast"
    if col.startswith("outage_"):
        return "outage"
    if "usd" in col.lower() or "eur" in col.lower():
        return "fuel_currency"
    if "cap" in col.lower() or "spike" in col.lower():
        return "cap_imbalance"
    if "yekdem" in col.lower() or "merchant" in col.lower() or "euas" in col.lower():
        return "yekdem_merchant_proxy"
    if col in {"low_load_flag", "holiday_low_load_flag", "solar_peak_hour_flag", "zero_price_risk_proxy"}:
        return "low_zero_price_risk"
    return "other"


def infer_sources(col: str) -> list[str]:
    # Best-effort mapping based on our engineering code.
    if col.startswith("ptf_"):
        return ["ptf_price"]
    if col.startswith("smf_") or col.startswith("smf_ptf_spread"):
        return ["smf_systemMarginalPrice", "ptf_price"]
    if col.startswith("yal_yat_"):
        return ["yal_yat_* (master)"]
    if col.startswith("wind_"):
        return ["wind_* (master)"]
    if col.startswith("outage_"):
        return ["outages_* (master)"]
    if col.startswith("kgup_"):
        return [col]
    if col in {"kgup_total_minus_load"}:
        return ["kgup_toplam", "load_lep"]
    if col.endswith("_minus_load") or col.startswith("load_") or col == "low_load_flag":
        return ["load_lep"]
    if col == "holiday_low_load_flag":
        # holiday/weekend flag is engineered from calendar; not a master column
        return ["load_lep", "ts_hour (holiday/weekend flag)"]
    if col in {"res_share", "solar_share", "hydro_share", "renewable_pressure", "renewable_suppression_pressure"}:
        return ["kgup_toplam", "kgup_ruzgar", "kgup_gunes", "kgup_barajli", "kgup_akarsu", "kgup_biokutle", "kgup_jeotermal"]
    if col in {"gas_share", "coal_share", "gas_coal_balance", "gas_coal_competition_index", "thermal_price_setting_share"}:
        return ["kgup_toplam", "kgup_dogalgaz", "kgup_ithalKomur", "kgup_linyit", "kgup_tasKomur"] + (["kgup_ruzgar", "kgup_gunes", "kgup_barajli", "kgup_akarsu", "kgup_biokutle", "kgup_jeotermal"] if col == "thermal_price_setting_share" else [])
    if col in {"hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "hour_of_day", "month", "is_weekend", "is_summer", "is_winter"}:
        return ["ts_hour"]
    if col.startswith("is_holiday"):
        return ["ts_hour", "holidays.Turkey()"]
    if col in {"solar_peak_hour_flag"}:
        return ["ts_hour"]
    if col == "zero_price_risk_proxy":
        return ["kgup_* (renewable/gas shares)", "load_lep", "ts_hour (holiday/weekend flag)"]
    if col == "price_cap":
        return ["engineered_constant"]
    return []


def infer_availability(col: str) -> str:
    # same-hour safe by default for planned KGUP/load/wind/outage; lag required for realized / balancing.
    if col.startswith("smf_") or col.startswith("yal_yat_") or col.endswith("_lag_24") or col.endswith("_lag_48") or col.endswith("_lag_168") or col.startswith("smf_ptf_spread_"):
        return "lag_required"
    if col.startswith("ptf_lag_") or col.startswith("ptf_roll_") or col.startswith("ptf_rolling_"):
        return "lag_required"
    return "same_hour_ok"


def infer_leakage_risk(col: str) -> str:
    # This is about *leakage to future target*, not about general modeling risk.
    fam = infer_family(col)
    if fam in {"smf_yal_yat_lagged", "ptf_lag_rolling"}:
        return "low"  # we only include lagged versions in this dataset
    if fam in {"kgup_source_mix", "load_demand", "wind_forecast", "outage", "calendar_holiday", "renewable_pressure", "thermal_price_setting", "low_zero_price_risk"}:
        return "low"
    if fam in {"fuel_currency", "cap_imbalance", "yekdem_merchant_proxy"}:
        return "medium"
    return "medium"


def recommend_usage(col: str) -> str:
    """Map columns to model buckets using features.config contract lists."""
    if col in _RISK_DASHBOARD_SET:
        return "risk_dashboard_only"
    if col in _MAIN_REGRESSION_SET:
        return "main_regression"
    if col in _LOW_PRICE_CLASSIFIER_SET:
        return "low_price_classifier"
    return "exclude"


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load feature dataset + a light master frame for sanity checks.

    IMPORTANT: For "missing source data" checks we should not rely on a
    column-restricted master read (it would create false "missing" flags).
    We handle source-availability via parquet schema in `build_missing_report`.
    """
    from src.utils.io_utils import read_parquet_with_normalized_ts
    feat = read_parquet_with_normalized_ts(FEATURES_PATH)
    master = read_parquet_with_normalized_ts(
        MASTER_PATH,
        columns=["ts_hour", "ptf_price", "smf_systemMarginalPrice"],
    )
    return feat, master


def build_inventory(feat: pd.DataFrame) -> dict[str, Any]:
    cols = _feature_cols(feat)
    rows: list[dict[str, Any]] = []
    for c in cols:
        rows.append(
            {
                "feature": c,
                "family": infer_family(c),
                "sources": infer_sources(c),
                "availability": infer_availability(c),  # same_hour_ok / lag_required
                "leakage_risk": infer_leakage_risk(c),
                "recommended_usage": recommend_usage(c),
                "null_pct": float(feat[c].isna().mean()) if c in feat.columns else None,
            }
        )
    rows.sort(key=lambda r: (FAMILY_ORDER.index(r["family"]) if r["family"] in FAMILY_ORDER else 999, r["feature"]))
    return {
        "features_path": str(FEATURES_PATH),
        "row_count": int(len(feat)),
        "feature_count": int(len(cols)),
        "families": FAMILY_ORDER,
        "rows": rows,
        "counts_by_family": {
            fam: int(sum(1 for r in rows if r["family"] == fam)) for fam in FAMILY_ORDER
        },
        "counts_by_recommended_usage": {
            k: int(sum(1 for r in rows if r["recommended_usage"] == k))
            for k in ["main_regression", "low_price_classifier", "risk_dashboard_only", "exclude"]
        },
        "high_leakage_risk_features": [r["feature"] for r in rows if r["leakage_risk"] == "high"],
    }


def build_missing_report(feat: pd.DataFrame, master: pd.DataFrame) -> dict[str, Any]:
    feat_cols = set(_feature_cols(feat))
    # Use parquet schema for source-availability checks to avoid false negatives.
    try:
        import pyarrow.parquet as pq

        master_cols = set(pq.ParquetFile(MASTER_PATH).schema.names)
        master_cols_source = "parquet_schema"
    except Exception:
        master_cols = set(master.columns)
        master_cols_source = "loaded_columns"
    items = []
    for name in REQUESTED_FEATURES:
        present = name in feat_cols
        sources = infer_sources(name)
        # If we know expected sources, check if they exist in master.
        source_status = "unknown"
        missing_sources = []
        if sources:
            # Expand wildcard markers loosely.
            concrete = []
            for s in sources:
                # Non-concrete / engineered markers: don't treat as missing master columns.
                if (
                    s.endswith("_* (master)")
                    or "*" in s
                    or "(" in s
                    or "holidays." in s
                    or s.startswith("engineered_")
                ):
                    source_status = "partial_unknown"
                    continue
                concrete.append(s)
            if concrete:
                missing_sources = [s for s in concrete if s not in master_cols]
                source_status = "ok" if not missing_sources else "missing_source_data"
            else:
                # Only wildcard/engineered sources were provided.
                source_status = "partial_unknown"
        else:
            # Try a direct master lookup.
            source_status = "ok" if name in master_cols else "missing_source_data"
        items.append(
            {
                "feature": name,
                "present_in_dataset": bool(present),
                "expected_sources": sources,
                "source_status": source_status,
                "missing_sources": missing_sources,
                "master_cols_source": master_cols_source,
            }
        )
    feat_cols_list = sorted(feat_cols)
    main_present, main_missing = resolve_feature_list(MAIN_REGRESSION_FEATURES, feat_cols_list)
    low_present, low_missing = resolve_feature_list(LOW_PRICE_CLASSIFIER_FEATURES, feat_cols_list)
    risk_present, risk_missing = resolve_feature_list(RISK_DASHBOARD_FEATURES, feat_cols_list)
    excluded_present, _ = resolve_feature_list(EXCLUDED_FROM_MAIN_REGRESSION, feat_cols_list)

    bucket_resolution = {
        "main_regression": {
            "requested_count": len(MAIN_REGRESSION_FEATURES),
            "present_count": len(main_present),
            "present": main_present,
            "missing": main_missing,
        },
        "low_price_classifier": {
            "requested_count": len(LOW_PRICE_CLASSIFIER_FEATURES),
            "present_count": len(low_present),
            "present": low_present,
            "missing": low_missing,
        },
        "risk_dashboard": {
            "requested_count": len(RISK_DASHBOARD_FEATURES),
            "present_count": len(risk_present),
            "present": risk_present,
            "missing": risk_missing,
        },
        "excluded_from_main_present_count": len(excluded_present),
    }

    return {
        "requested_features": REQUESTED_FEATURES,
        "present_count": int(sum(1 for x in items if x["present_in_dataset"])),
        "missing_count": int(sum(1 for x in items if not x["present_in_dataset"])),
        "missing_source_data_count": int(sum(1 for x in items if x["source_status"] == "missing_source_data")),
        "items": items,
        "bucket_resolution": bucket_resolution,
        "thesis_data_debt_groups": THESIS_DATA_DEBT_GROUPS,
    }


def _slice_mask(actual: pd.Series, name: str) -> pd.Series:
    if name == "actual_eq_0":
        return actual == 0
    if name == "actual_le_50":
        return actual <= 50
    if name == "actual_le_100":
        return actual <= 100
    if name == "normal_price":
        return (actual > 100) & (actual < 4000)
    if name == "spike_price":
        return actual >= 4000
    raise ValueError(name)


def build_sanity_report(feat: pd.DataFrame) -> dict[str, Any]:
    # We only need a proxy "actual" to define slices. Use target_1h as next-hour realized PTF.
    if "target_1h" not in feat.columns:
        raise ValueError("Expected target_1h in features parquet")
    actual = pd.to_numeric(feat["target_1h"], errors="coerce")

    features = _feature_cols(feat)
    # For sanity metrics, focus on engineered "risk" features + a small set of key shares.
    focus = [c for c in features if c in {
        "low_load_flag",
        "holiday_low_load_flag",
        "renewable_pressure",
        "renewable_suppression_pressure",
        "zero_price_risk_proxy",
        "gas_share",
        "coal_share",
        "gas_coal_competition_index",
        "thermal_price_setting_share",
    }]
    # Always include ptf_lag_24 as baseline context if present.
    if "ptf_lag_24" in features and "ptf_lag_24" not in focus:
        focus.append("ptf_lag_24")

    out: dict[str, Any] = {"slices": {}}
    slice_names = ["actual_eq_0", "actual_le_50", "actual_le_100", "normal_price", "spike_price"]
    for sname in slice_names:
        mask = _slice_mask(actual, sname)
        sub = feat.loc[mask, focus].copy()
        stats = {}
        for c in focus:
            x = pd.to_numeric(sub[c], errors="coerce")
            stats[c] = {
                "rows": int(x.notna().sum()),
                "mean": float(x.mean()) if x.notna().any() else None,
                "p50": float(x.median()) if x.notna().any() else None,
                "p90": float(x.quantile(0.9)) if x.notna().any() else None,
            }
        out["slices"][sname] = {"rows": int(mask.sum()), "feature_stats": stats}

    # Simple correlation with target_1h for focus features (over all rows).
    corr = {}
    for c in focus:
        x = pd.to_numeric(feat[c], errors="coerce")
        ok = x.notna() & actual.notna()
        if ok.sum() < 1000:
            corr[c] = None
        else:
            corr[c] = float(np.corrcoef(x[ok].to_numpy(dtype=float), actual[ok].to_numpy(dtype=float))[0, 1])
    out["focus_features"] = focus
    out["corr_with_target_1h"] = corr
    return out


def write_md_inventory(inv: dict[str, Any]) -> str:
    lines = [
        "# Feature Inventory Report",
        "",
        f"- Dataset: `{inv['features_path']}`",
        f"- Rows: {inv['row_count']}",
        f"- Feature count: {inv['feature_count']}",
        "",
        "## Counts by Family",
        "",
    ]
    for fam in FAMILY_ORDER:
        lines.append(f"- {fam}: {inv['counts_by_family'].get(fam,0)}")
    lines += ["", "## Counts by Recommended Usage", ""]
    for k, v in inv["counts_by_recommended_usage"].items():
        lines.append(f"- {k}: {v}")
    lines += [
        "",
        "## Inventory",
        "",
        "| Feature | Family | Sources | Availability | Leakage | Suggested | Null % |",
        "|---------|--------|---------|--------------|---------|----------|-------:|",
    ]
    for r in inv["rows"]:
        src = ", ".join(r["sources"]) if r["sources"] else "-"
        lines.append(
            f"| `{r['feature']}` | {r['family']} | {src} | {r['availability']} | {r['leakage_risk']} | {r['recommended_usage']} | {r['null_pct']*100:.2f} |"
        )
    return "\n".join(lines) + "\n"


def write_md_missing(rep: dict[str, Any]) -> str:
    lines = [
        "# Missing Feature Report",
        "",
        f"- Requested: {len(rep['requested_features'])}",
        f"- Present: {rep['present_count']}",
        f"- Missing: {rep['missing_count']}",
        f"- Missing source data: {rep['missing_source_data_count']}",
        "",
        "## Tez / piyasa veri borcu (ham veri yok — pipeline'a eklenmez)",
        "",
        "Aşağıdaki gruplar `features.config.THESIS_DATA_DEBT_GROUPS` ile tanımlıdır.",
        "Feature engineering bu kaynaklar gelene kadar üretilmez; yalnızca raporlanır.",
        "",
        "| Grup | Hedef feature'lar | Ham veri kaynağı |",
        "|------|-------------------|------------------|",
    ]
    for g in THESIS_DATA_DEBT_GROUPS:
        lines.append(
            f"| {g['group']} | `{g['target_features']}` | {g['raw_sources']} |"
        )
    lines += [
        "",
        "## İstenen feature'lar (REQUESTED_FEATURES)",
        "",
        "| Feature | Present | Source status | Missing sources |",
        "|---------|:-------:|--------------|----------------|",
    ]
    for it in rep["items"]:
        miss = ", ".join(it["missing_sources"]) if it["missing_sources"] else "-"
        lines.append(f"| `{it['feature']}` | {int(it['present_in_dataset'])} | {it['source_status']} | {miss} |")

    bucket = rep.get("bucket_resolution", {})
    if bucket:
        lines += [
            "",
            "## Model kovaları — parquet'te mevcut / eksik",
            "",
            f"- MAIN_REGRESSION: {bucket['main_regression']['present_count']}/{bucket['main_regression']['requested_count']} mevcut"
            f" ({len(bucket['main_regression']['missing'])} eksik)",
            f"- LOW_PRICE_CLASSIFIER: {bucket['low_price_classifier']['present_count']}/{bucket['low_price_classifier']['requested_count']} mevcut"
            f" ({len(bucket['low_price_classifier']['missing'])} eksik)",
            f"- RISK_DASHBOARD: {bucket['risk_dashboard']['present_count']}/{bucket['risk_dashboard']['requested_count']} mevcut"
            f" ({len(bucket['risk_dashboard']['missing'])} eksik)",
        ]
        for key, label in (
            ("main_regression", "MAIN eksik"),
            ("low_price_classifier", "LOW_PRICE eksik"),
            ("risk_dashboard", "RISK eksik"),
        ):
            miss = bucket[key]["missing"]
            if miss:
                lines.append(f"- {label}: `{', '.join(miss)}`")

    return "\n".join(lines) + "\n"


def write_md_sanity(rep: dict[str, Any]) -> str:
    lines = [
        "# Feature Sanity Report",
        "",
        "Slices are defined using `target_1h` (next-hour realized PTF) to approximate regime buckets.",
        "",
        "## Correlation with target_1h (focus features)",
        "",
        "| Feature | corr(target_1h) |",
        "|---------|-----------------:|",
    ]
    for c, v in rep["corr_with_target_1h"].items():
        lines.append(f"| `{c}` | {'' if v is None else f'{v:.4f}'} |")
    lines += ["", "## Slice summaries", ""]
    for sname, blob in rep["slices"].items():
        lines.append(f"### {sname} (rows={blob['rows']})")
        lines.append("")
        lines.append("| Feature | mean | p50 | p90 | non-null |")
        lines.append("|---------|-----:|----:|----:|--------:|")
        for feat, st in blob["feature_stats"].items():
            if st["mean"] is None:
                lines.append(f"| `{feat}` |  |  |  | {st['rows']} |")
            else:
                lines.append(f"| `{feat}` | {st['mean']:.4f} | {st['p50']:.4f} | {st['p90']:.4f} | {st['rows']} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    feat, master = load_frames()

    inv = build_inventory(feat)
    missing = build_missing_report(feat, master)
    sanity = build_sanity_report(feat)

    inv["notes"] = USAGE_MODE_NOTES
    missing["notes"] = USAGE_MODE_NOTES
    sanity["notes"] = USAGE_MODE_NOTES

    OUT_INV_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_INV_JSON.write_text(json.dumps(inv, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_INV_MD.write_text(write_md_inventory(inv), encoding="utf-8")

    OUT_MISSING_JSON.write_text(json.dumps(missing, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MISSING_MD.write_text(write_md_missing(missing), encoding="utf-8")

    OUT_SANITY_JSON.write_text(json.dumps(sanity, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_SANITY_MD.write_text(write_md_sanity(sanity), encoding="utf-8")

    # Terminal summary
    total_features = inv["feature_count"]
    missing_source = missing["missing_source_data_count"]
    bucket = missing["bucket_resolution"]
    main_req = bucket["main_regression"]["requested_count"]
    main_pres = bucket["main_regression"]["present_count"]
    main_miss = len(bucket["main_regression"]["missing"])
    low_req = bucket["low_price_classifier"]["requested_count"]
    low_pres = bucket["low_price_classifier"]["present_count"]
    low_miss = len(bucket["low_price_classifier"]["missing"])
    risk_req = bucket["risk_dashboard"]["requested_count"]
    risk_pres = bucket["risk_dashboard"]["present_count"]
    risk_miss = len(bucket["risk_dashboard"]["missing"])
    excluded_from_main_n = bucket["excluded_from_main_present_count"]
    inv_main_n = inv["counts_by_recommended_usage"]["main_regression"]
    inv_low_n = inv["counts_by_recommended_usage"]["low_price_classifier"]
    inv_dash_n = inv["counts_by_recommended_usage"]["risk_dashboard_only"]
    inv_exclude_n = inv["counts_by_recommended_usage"]["exclude"]
    high_risk = inv["high_leakage_risk_features"]

    print("=== Feature Audit Summary ===")
    print("Toplam parquet feature sayisi:", total_features)
    print(
        f"MAIN_REGRESSION_FEATURES: {main_pres}/{main_req} mevcut, {main_miss} eksik"
    )
    if bucket["main_regression"]["missing"]:
        print("  MAIN eksik:", ", ".join(bucket["main_regression"]["missing"]))
    print(
        f"LOW_PRICE_CLASSIFIER_FEATURES: {low_pres}/{low_req} mevcut, {low_miss} eksik"
    )
    if bucket["low_price_classifier"]["missing"]:
        print("  LOW_PRICE eksik:", ", ".join(bucket["low_price_classifier"]["missing"]))
    print(
        f"RISK_DASHBOARD_FEATURES: {risk_pres}/{risk_req} mevcut, {risk_miss} eksik"
    )
    if bucket["risk_dashboard"]["missing"]:
        print("  RISK eksik:", ", ".join(bucket["risk_dashboard"]["missing"]))
    print(
        "Ana regression'dan cikarilan (EXCLUDED_FROM_MAIN, parquet'te):",
        excluded_from_main_n,
    )
    print("Eksik ham veri gerektiren feature sayisi (REQUESTED):", missing_source)
    print("Envanter recommended_usage — main:", inv_main_n, "| low:", inv_low_n, "| risk:", inv_dash_n, "| exclude:", inv_exclude_n)
    print("Leakage riski high olan feature'lar:", high_risk)
    print("Wrote:")
    print(" ", OUT_INV_MD)
    print(" ", OUT_INV_JSON)
    print(" ", OUT_MISSING_MD)
    print(" ", OUT_MISSING_JSON)
    print(" ", OUT_SANITY_MD)
    print(" ", OUT_SANITY_JSON)


if __name__ == "__main__":
    main()
