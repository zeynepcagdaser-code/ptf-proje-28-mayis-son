#!/usr/bin/env python3
"""
Build a leakage-safe plant-level KGUP -> YEKDEM must-run feature pipeline.

Default behavior is local ingestion only:
    data/plant_level_kgup/raw/*.{csv,parquet,xls,xlsx}

The EPİAŞ API layer is deliberately explicit and opt-in because plant/UEVCB
bulk KGUP requires an ID universe and rate-limit-safe batching. This script
will not start broad historical API fetching unless a future `--fetch-api`
implementation is wired intentionally.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from market_aware_ptf_pipeline_skeleton import YEKDEM_Registry

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "plant_level_kgup" / "raw"
RAW_ARCHIVE_DIR = PROJECT_ROOT / "data" / "plant_level_kgup" / "raw_archive"
OUTPUT_DIR = PROJECT_ROOT / "data" / "plant_level_kgup"
FEATURE_PATH = PROJECT_ROOT / "data" / "features" / "must_run_supply_features.parquet"
NORMALIZED_PATH = OUTPUT_DIR / "normalized_plant_level_kgup.parquet"
YEKDEM_MATCHED_PATH = OUTPUT_DIR / "yekdem_matched_plant_level_kgup.parquet"
AUDIT_JSON = OUTPUT_DIR / "plant_level_kgup_audit.json"
AUDIT_MD = PROJECT_ROOT / "reports" / "plant_level_kgup_audit.md"
ANALYSIS_MD = PROJECT_ROOT / "reports" / "must_run_supply_analysis.md"

LOAD_FORECAST_PATH = PROJECT_ROOT / "data" / "load_forecast.csv"
REGIME_LABEL_PATH = PROJECT_ROOT / "data" / "regime_labels.csv"
AGG_KGUP_PATH = PROJECT_ROOT / "data" / "clean" / "kgup_hourly.parquet"

REQUIRED_SCHEMA = [
    "entsoe_code",
    "plant_name",
    "fuel_type",
    "delivery_hour",
    "publication_timestamp",
    "archive_snapshot_timestamp",
    "kgup_mwh",
    "source_file",
    "source_api",
    "forecast_timestamp",
]

EPİAS_API_NOTES = {
    "dpp_bulk_endpoint": "POST /v1/generation/data/dpp-bulk",
    "dpp_bulk_semantics": "Verilen güne ait, topluca verilen UEVÇB'lerin KGÜP değerlerini döner.",
    "powerplant_list_endpoint": "GET /v1/generation/data/powerplant-list",
    "uevcb_bulk_endpoint": "POST /v1/generation/data/uevcb-list-bulk",
}


def slug(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()


def detect_col(columns: pd.Index, candidates: list[str]) -> str | None:
    slug_map = {slug(col): str(col) for col in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        found = slug_map.get(slug(candidate))
        if found:
            return found
    return None


def read_any(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xls", ".xlsx", ".xlsm"}:
        frames = []
        for sheet in pd.ExcelFile(path).sheet_names:
            frames.append(pd.read_excel(path, sheet_name=sheet))
        return pd.concat(frames, ignore_index=True)
    raise ValueError(f"Unsupported raw KGUP file type: {path}")


def normalize_fuel_type(value: Any) -> str:
    s = slug(value)
    if any(token in s for token in ["ruzgar", "wind"]):
        return "wind"
    if any(token in s for token in ["gunes", "solar"]):
        return "solar"
    if any(token in s for token in ["hidro", "hes", "akarsu", "baraj"]):
        return "hydro"
    if any(token in s for token in ["bio", "biyokutle", "biokutle", "biomass"]):
        return "biomass"
    if any(token in s for token in ["jeotermal", "geothermal"]):
        return "geothermal"
    if any(token in s for token in ["dogalgaz", "gas"]):
        return "gas"
    if any(token in s for token in ["komur", "linyit", "coal"]):
        return "coal"
    return s or "unknown"


def plant_name_key(value: Any) -> str:
    text = slug(value)
    stopwords = {
        "reg",
        "regulatoru",
        "regulator",
        "ve",
        "hes",
        "ges",
        "res",
        "bes",
        "jes",
        "santrali",
        "enerji",
        "uretim",
        "uretimi",
        "as",
        "a",
        "s",
    }
    parts = [part for part in text.split("_") if part and part not in stopwords]
    return "_".join(parts)


@dataclass
class NormalizeReport:
    source_file: str
    rows_in: int
    rows_out: int
    missing_columns: list[str]
    detected_columns: dict[str, str | None]


class PlantLevelKgupNormalizer:
    COLUMN_CANDIDATES = {
        "entsoe_code": ["entsoe_code", "ENTSO-E Kodu", "ENTSO-E Kodu [1]", "eic", "eic_code", "uevcb_eic"],
        "plant_name": ["plant_name", "Tesis Adı", "Santral Adı", "powerPlantName", "organizationName"],
        "fuel_type": ["fuel_type", "Ana Kaynak Türü", "Kaynak Türü", "resourceType", "fuelType"],
        "delivery_hour": ["delivery_hour", "date", "Tarih", "deliveryDate", "period", "datetime"],
        "publication_timestamp": ["publication_timestamp", "publishTime", "publishedAt", "publicationDate"],
        "archive_snapshot_timestamp": ["archive_snapshot_timestamp", "snapshot_ts", "fetched_at", "requested_at"],
        "kgup_mwh": ["kgup_mwh", "KGÜP", "kgup", "dpp", "value", "miktar", "quantity", "toplam"],
        "forecast_timestamp": ["forecast_timestamp", "forecast_as_of", "as_of", "snapshot_ts"],
    }

    def normalize(self, frame: pd.DataFrame, source_file: str) -> tuple[pd.DataFrame, NormalizeReport]:
        detected = {target: detect_col(frame.columns, candidates) for target, candidates in self.COLUMN_CANDIDATES.items()}
        missing = [target for target, col in detected.items() if col is None and target not in {"forecast_timestamp"}]

        out = pd.DataFrame(index=frame.index)
        for target in REQUIRED_SCHEMA:
            source = detected.get(target)
            out[target] = frame[source] if source in frame.columns else pd.NA

        out["source_file"] = source_file
        out["source_api"] = out["source_api"].fillna("local_raw")
        out["entsoe_code"] = out["entsoe_code"].astype("string").str.strip()
        out["plant_name"] = out["plant_name"].astype("string").str.strip()
        out["fuel_type"] = out["fuel_type"].map(normalize_fuel_type)
        out["delivery_hour"] = pd.to_datetime(out["delivery_hour"], errors="coerce")
        out["publication_timestamp"] = pd.to_datetime(out["publication_timestamp"], errors="coerce")
        out["archive_snapshot_timestamp"] = pd.to_datetime(out["archive_snapshot_timestamp"], errors="coerce")
        out["forecast_timestamp"] = pd.to_datetime(out["forecast_timestamp"], errors="coerce")
        out["kgup_mwh"] = pd.to_numeric(out["kgup_mwh"], errors="coerce")

        out = out.dropna(subset=["entsoe_code", "delivery_hour", "kgup_mwh"])
        report = NormalizeReport(
            source_file=source_file,
            rows_in=int(len(frame)),
            rows_out=int(len(out)),
            missing_columns=missing,
            detected_columns=detected,
        )
        return out[REQUIRED_SCHEMA], report


def load_raw_plant_level_kgup(raw_dir: Path) -> tuple[pd.DataFrame, list[NormalizeReport]]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = RAW_ARCHIVE_DIR
    archive_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(
        path
        for root in {raw_dir, archive_dir}
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".csv", ".parquet", ".xls", ".xlsx", ".xlsm"}
    )
    normalizer = PlantLevelKgupNormalizer()
    frames = []
    reports = []
    for path in files:
        raw = read_any(path)
        normalized, report = normalizer.normalize(raw, path.name)
        frames.append(normalized)
        reports.append(report)
    if not frames:
        return pd.DataFrame(columns=REQUIRED_SCHEMA), reports
    data = pd.concat(frames, ignore_index=True)
    data = data.drop_duplicates(
        subset=["entsoe_code", "delivery_hour", "publication_timestamp", "forecast_timestamp"],
        keep="last",
    )
    return data.sort_values(["delivery_hour", "entsoe_code"]), reports


def attach_yekdem_mapping(data: pd.DataFrame, registry: YEKDEM_Registry) -> tuple[pd.DataFrame, dict[str, Any]]:
    if data.empty:
        return data.assign(is_yekdem=pd.Series(dtype=bool)), {
            "matched_plants": 0,
            "unmatched_plants": 0,
            "duplicate_mappings": 0,
            "low_confidence_matches": 0,
        }

    registry_cols = [registry.registry_id_col, "registry_year", "registry_source_file"]
    if registry.registry_name_col:
        registry_cols.append(registry.registry_name_col)
    fuel_col = registry._detect_col(registry.registry.columns, registry.DEFAULT_SOURCE_CANDIDATES)
    if fuel_col:
        registry_cols.append(fuel_col)

    lookup_base = registry.registry[registry_cols].copy()
    lookup_base = lookup_base.rename(
        columns={
            registry.registry_id_col: "entsoe_code",
            registry.registry_name_col or "": "yekdem_plant_name",
            fuel_col or "": "yekdem_fuel_type_raw",
        }
    )
    lookup_base["entsoe_code"] = lookup_base["entsoe_code"].astype("string").str.strip()
    lookup_base["registry_year"] = pd.to_numeric(lookup_base["registry_year"], errors="coerce")
    lookup_base["yekdem_fuel_type"] = lookup_base.get(
        "yekdem_fuel_type_raw", pd.Series(index=lookup_base.index, dtype="object")
    ).map(normalize_fuel_type)

    data = data.copy()
    data["delivery_year"] = data["delivery_hour"].dt.year
    lookup_frames = []
    for year in sorted(data["delivery_year"].dropna().astype(int).unique()):
        eligible = lookup_base[lookup_base["registry_year"].fillna(-1) <= year].copy()
        if eligible.empty:
            eligible = lookup_base.copy()
        eligible = (
            eligible.sort_values(["entsoe_code", "registry_year"])
            .drop_duplicates("entsoe_code", keep="last")
            .copy()
        )
        eligible["delivery_year"] = year
        lookup_frames.append(eligible)
    lookup = pd.concat(lookup_frames, ignore_index=True) if lookup_frames else lookup_base.assign(delivery_year=pd.NA)
    lookup["registry_name_key"] = lookup.get("yekdem_plant_name", pd.Series(index=lookup.index, dtype="object")).map(plant_name_key)
    data["plant_name_key"] = data["plant_name"].map(plant_name_key)

    code_lookup = lookup.rename(
        columns={
            "registry_year": "registry_year_code",
            "registry_source_file": "registry_source_file_code",
            "yekdem_plant_name": "yekdem_plant_name_code",
            "yekdem_fuel_type_raw": "yekdem_fuel_type_raw_code",
            "yekdem_fuel_type": "yekdem_fuel_type_code",
            "registry_name_key": "registry_name_key_code",
        }
    )
    name_lookup = lookup.drop_duplicates(["registry_name_key", "delivery_year"], keep="last").rename(
        columns={
            "entsoe_code": "registry_entsoe_code_name",
            "registry_year": "registry_year_name",
            "registry_source_file": "registry_source_file_name",
            "yekdem_plant_name": "yekdem_plant_name_name",
            "yekdem_fuel_type_raw": "yekdem_fuel_type_raw_name",
            "yekdem_fuel_type": "yekdem_fuel_type_name",
            "registry_name_key": "plant_name_key",
        }
    )

    mapped = data.merge(code_lookup, on=["entsoe_code", "delivery_year"], how="left")
    mapped["code_name_consistent"] = (
        mapped["registry_year_code"].notna()
        & mapped["plant_name_key"].notna()
        & mapped["registry_name_key_code"].notna()
        & mapped["plant_name_key"].eq(mapped["registry_name_key_code"])
    )
    mapped = mapped.merge(name_lookup, on=["plant_name_key", "delivery_year"], how="left")
    use_code = mapped["code_name_consistent"]
    use_name = ~use_code & mapped["registry_year_name"].notna()
    mapped["registry_year"] = np.where(use_code, mapped["registry_year_code"], mapped["registry_year_name"])
    mapped["registry_source_file"] = np.where(
        use_code, mapped["registry_source_file_code"], mapped["registry_source_file_name"]
    )
    mapped["yekdem_plant_name"] = np.where(use_code, mapped["yekdem_plant_name_code"], mapped["yekdem_plant_name_name"])
    mapped["yekdem_fuel_type_raw"] = np.where(
        use_code, mapped["yekdem_fuel_type_raw_code"], mapped["yekdem_fuel_type_raw_name"]
    )
    mapped["yekdem_fuel_type"] = np.where(use_code, mapped["yekdem_fuel_type_code"], mapped["yekdem_fuel_type_name"])
    mapped["yekdem_match_method"] = np.select(
        [use_code, use_name],
        ["entsoe_code_name_consistent", "normalized_name_exact"],
        default="unmatched",
    )
    mapped["is_yekdem"] = pd.notna(mapped["registry_year"])
    mapped["fuel_type_effective"] = pd.Series(mapped["yekdem_fuel_type"]).fillna(mapped["fuel_type"]).map(normalize_fuel_type)

    raw_plants = set(data["entsoe_code"].dropna().astype(str))
    matched = set(mapped.loc[mapped["is_yekdem"], "entsoe_code"].dropna().astype(str))
    duplicate_mappings = int(
        lookup_base.groupby(["entsoe_code", "registry_year"], dropna=False).size().gt(1).sum()
    )
    audit = {
        "matched_plants": int(len(matched)),
        "unmatched_plants": int(len(raw_plants - matched)),
        "duplicate_mappings": duplicate_mappings,
        "low_confidence_matches": 0,
        "match_method_counts": mapped.loc[mapped["is_yekdem"], "yekdem_match_method"].value_counts().to_dict(),
        "code_name_mismatch_rows": int(
            mapped["registry_year_code"].notna().sum() - mapped["code_name_consistent"].sum()
        ),
        "unmatched_plant_sample": sorted(raw_plants - matched)[:25],
    }
    return mapped, audit


def latest_asof_rows(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if data.empty:
        return data, {
            "eligible_rows": 0,
            "leakage_rows": 0,
            "missing_publication_timestamp": 0,
            "missing_forecast_timestamp": 0,
            "missing_snapshot_timestamp": 0,
        }
    frame = data.copy()
    frame["publication_timestamp"] = pd.to_datetime(frame["publication_timestamp"], errors="coerce", utc=True)
    frame["forecast_timestamp"] = pd.to_datetime(frame["forecast_timestamp"], errors="coerce", utc=True)
    if "archive_snapshot_timestamp" in frame.columns:
        frame["archive_snapshot_timestamp"] = pd.to_datetime(frame["archive_snapshot_timestamp"], errors="coerce", utc=True)
    else:
        frame["archive_snapshot_timestamp"] = pd.NaT
    frame["archive_snapshot_timestamp"] = frame["archive_snapshot_timestamp"].fillna(frame["forecast_timestamp"])
    missing_pub = int(frame["publication_timestamp"].isna().sum())
    missing_forecast = int(frame["forecast_timestamp"].isna().sum())
    missing_snapshot = int(frame["archive_snapshot_timestamp"].isna().sum())
    frame["strict_point_in_time_safe"] = (
        frame["publication_timestamp"].notna()
        & frame["forecast_timestamp"].notna()
        & (frame["publication_timestamp"] <= frame["forecast_timestamp"])
    )
    frame["structural_market_proxy"] = (
        ~frame["strict_point_in_time_safe"]
        & frame["archive_snapshot_timestamp"].notna()
        & frame["forecast_timestamp"].notna()
    )
    frame["leakage_safe"] = frame["strict_point_in_time_safe"] | frame["structural_market_proxy"]
    leakage_rows = int((~frame["leakage_safe"]).sum())
    eligible = frame[frame["leakage_safe"]].copy()
    if eligible.empty:
        return eligible, {
            "eligible_rows": 0,
            "leakage_rows": leakage_rows,
            "strict_point_in_time_rows": int(frame["strict_point_in_time_safe"].sum()),
            "structural_proxy_rows": int(frame["structural_market_proxy"].sum()),
            "missing_publication_timestamp": missing_pub,
            "missing_forecast_timestamp": missing_forecast,
            "missing_snapshot_timestamp": missing_snapshot,
        }
    eligible["asof_sort_timestamp"] = (
        eligible["publication_timestamp"]
        .fillna(eligible["archive_snapshot_timestamp"])
        .fillna(eligible["forecast_timestamp"])
    )
    idx = (
        eligible.sort_values("asof_sort_timestamp")
        .groupby(["entsoe_code", "delivery_hour"], observed=False)
        .tail(1)
        .index
    )
    latest = eligible.loc[idx].copy()
    return latest, {
        "eligible_rows": int(len(latest)),
        "leakage_rows": leakage_rows,
        "strict_point_in_time_rows": int(latest["strict_point_in_time_safe"].sum()),
        "structural_proxy_rows": int(latest["structural_market_proxy"].sum()),
        "missing_publication_timestamp": missing_pub,
        "missing_forecast_timestamp": missing_forecast,
        "missing_snapshot_timestamp": missing_snapshot,
    }


def load_load_forecast() -> pd.DataFrame:
    if not LOAD_FORECAST_PATH.exists():
        return pd.DataFrame(columns=["delivery_hour", "load_forecast"])
    load = pd.read_csv(LOAD_FORECAST_PATH)
    load["delivery_hour"] = pd.to_datetime(load["date"], errors="coerce").dt.tz_localize(None)
    return load[["delivery_hour", "lep"]].rename(columns={"lep": "load_forecast"})


def load_aggregate_kgup() -> pd.DataFrame:
    if AGG_KGUP_PATH.exists():
        agg = pd.read_parquet(AGG_KGUP_PATH)
        agg["delivery_hour"] = pd.to_datetime(agg["ts_hour"], errors="coerce").dt.tz_localize(None)
        return agg[["delivery_hour", "toplam"]].rename(columns={"toplam": "kgup_total"})
    if AGG_KGUP_PATH.with_suffix(".csv").exists():
        agg = pd.read_csv(AGG_KGUP_PATH.with_suffix(".csv"))
        agg["delivery_hour"] = pd.to_datetime(agg["date"], errors="coerce").dt.tz_localize(None)
        return agg[["delivery_hour", "toplam"]].rename(columns={"toplam": "kgup_total"})
    return pd.DataFrame(columns=["delivery_hour", "kgup_total"])


def build_must_run_features(latest_rows: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "delivery_hour",
        "must_run_supply",
        "must_run_wind",
        "must_run_solar",
        "must_run_hydro",
        "must_run_biomass",
        "must_run_geothermal",
        "must_run_share",
        "residual_load_after_must_run",
    ]
    if latest_rows.empty:
        return pd.DataFrame(columns=columns)

    yekdem = latest_rows[latest_rows["is_yekdem"]].copy()
    yekdem["delivery_hour"] = pd.to_datetime(yekdem["delivery_hour"], errors="coerce")
    if getattr(yekdem["delivery_hour"].dt, "tz", None) is not None:
        yekdem["delivery_hour"] = yekdem["delivery_hour"].dt.tz_localize(None)
    if yekdem.empty:
        return pd.DataFrame(columns=columns)

    hourly = yekdem.groupby("delivery_hour", as_index=False)["kgup_mwh"].sum().rename(
        columns={"kgup_mwh": "must_run_supply"}
    )
    flags = yekdem.groupby("delivery_hour", as_index=False).agg(
        strict_point_in_time_safe=("strict_point_in_time_safe", "all"),
        structural_market_proxy=("structural_market_proxy", "any"),
        missing_publication_timestamp_share=("publication_timestamp", lambda s: float(s.isna().mean())),
    )
    fuel_pivot = (
        yekdem.pivot_table(
            index="delivery_hour",
            columns="fuel_type_effective",
            values="kgup_mwh",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    features = hourly.merge(fuel_pivot, on="delivery_hour", how="left").merge(flags, on="delivery_hour", how="left")
    for source, target in {
        "wind": "must_run_wind",
        "solar": "must_run_solar",
        "hydro": "must_run_hydro",
        "biomass": "must_run_biomass",
        "geothermal": "must_run_geothermal",
    }.items():
        features[target] = features[source] if source in features.columns else 0.0
    features = features[
        [
            "delivery_hour",
            "must_run_supply",
            "must_run_wind",
            "must_run_solar",
            "must_run_hydro",
            "must_run_biomass",
            "must_run_geothermal",
            "strict_point_in_time_safe",
            "structural_market_proxy",
            "missing_publication_timestamp_share",
        ]
    ]

    load = load_load_forecast()
    agg = load_aggregate_kgup()
    features = features.merge(load, on="delivery_hour", how="left").merge(agg, on="delivery_hour", how="left")
    features["must_run_share"] = features["must_run_supply"] / features["load_forecast"].replace(0, np.nan)
    features["residual_load_after_must_run"] = features["load_forecast"] - features["must_run_supply"]
    return features.sort_values("delivery_hour")


def analyze_must_run(features: pd.DataFrame) -> dict[str, Any]:
    if features.empty or not REGIME_LABEL_PATH.exists():
        return {"available": False, "reason": "Must-run features or regime labels are missing."}
    labels = pd.read_csv(REGIME_LABEL_PATH)
    labels["delivery_hour"] = pd.to_datetime(labels["ts_hour"], errors="coerce")
    joined = features.merge(labels[["delivery_hour", "price", "target_regime"]], on="delivery_hour", how="inner")
    if joined.empty:
        return {"available": False, "reason": "No overlap with regime labels."}
    joined["negative_or_zero_pressure"] = joined["target_regime"].eq("negative_zero_pressure")
    joined["solar_window"] = joined["delivery_hour"].dt.hour.between(10, 16)
    rows = []
    for regime, group in joined.groupby("target_regime", observed=False):
        rows.append(
            {
                "target_regime": str(regime),
                "rows": int(len(group)),
                "must_run_supply_mean": float(group["must_run_supply"].mean()),
                "must_run_share_mean": float(group["must_run_share"].mean()),
                "residual_load_after_must_run_mean": float(group["residual_load_after_must_run"].mean()),
                "price_mean": float(group["price"].mean()),
            }
        )
    return {
        "available": True,
        "rows": int(len(joined)),
        "coverage_start": str(joined["delivery_hour"].min()),
        "coverage_end": str(joined["delivery_hour"].max()),
        "correlations": {
            "must_run_share_vs_price": safe_corr(joined["must_run_share"], joined["price"]),
            "residual_load_after_must_run_vs_price": safe_corr(joined["residual_load_after_must_run"], joined["price"]),
            "must_run_solar_vs_price_solar_window": safe_corr(
                joined.loc[joined["solar_window"], "must_run_solar"],
                joined.loc[joined["solar_window"], "price"],
            ),
        },
        "regime_summary": rows,
    }


def safe_corr(a: pd.Series, b: pd.Series) -> float | None:
    frame = pd.concat([a, b], axis=1).dropna()
    if len(frame) < 3:
        return None
    return float(frame.iloc[:, 0].corr(frame.iloc[:, 1]))


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


def write_reports(
    normalized: pd.DataFrame,
    matched: pd.DataFrame,
    latest: pd.DataFrame,
    features: pd.DataFrame,
    normalize_reports: list[NormalizeReport],
    mapping_audit: dict[str, Any],
    leakage_audit: dict[str, Any],
    analysis: dict[str, Any],
    registry: YEKDEM_Registry,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_MD.parent.mkdir(parents=True, exist_ok=True)

    audit = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "api_notes": EPİAS_API_NOTES,
        "raw_dir": str(RAW_DIR.relative_to(PROJECT_ROOT)),
        "normalized_rows": int(len(normalized)),
        "matched_rows": int(len(matched[matched.get("is_yekdem", False)])) if not matched.empty else 0,
        "latest_asof_rows": int(len(latest)),
        "feature_rows": int(len(features)),
        "normalization": [report.__dict__ for report in normalize_reports],
        "registry_summary": registry.summary().to_dict(orient="records"),
        "mapping_audit": mapping_audit,
        "leakage_audit": leakage_audit,
        "must_run_analysis": analysis,
    }
    AUDIT_JSON.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str) + "\n")

    matched_plant_count = mapping_audit.get("matched_plants", 0)
    unmatched_plant_count = mapping_audit.get("unmatched_plants", 0)
    total_kgup = float(latest["kgup_mwh"].sum()) if not latest.empty else 0.0
    yekdem_kgup = float(latest.loc[latest.get("is_yekdem", False), "kgup_mwh"].sum()) if not latest.empty else 0.0
    yekdem_share = yekdem_kgup / total_kgup if total_kgup else None

    AUDIT_MD.write_text(
        "\n".join(
            [
                "# Plant-Level KGUP Audit",
                "",
                f"Generated: `{audit['generated_at']}`",
                "",
                "## Source Status",
                "",
                f"- Raw directory: `{audit['raw_dir']}`",
                f"- Raw normalized rows: `{len(normalized)}`",
                f"- Leakage-safe latest rows: `{len(latest)}`",
                f"- Feature rows: `{len(features)}`",
                "",
                "## API Semantics",
                "",
                "- EPİAŞ documentation lists `POST /v1/generation/data/dpp-bulk` for bulk UEVÇB KGÜP by day.",
                "- Plant/UEVÇB ID discovery should use `powerplant-list`, `uevcb-list`, or `uevcb-list-bulk` before historical fetching.",
                "- This run did not start broad API fetching; it ingested local raw files only.",
                "",
                "## YEKDEM Matching",
                "",
                f"- Matched plants: `{matched_plant_count}`",
                f"- Unmatched plants: `{unmatched_plant_count}`",
                f"- Duplicate mappings: `{mapping_audit.get('duplicate_mappings', 0)}`",
                f"- Low-confidence matches: `{mapping_audit.get('low_confidence_matches', 0)}`",
                f"- YEKDEM share of leakage-safe plant KGUP: `{yekdem_share if yekdem_share is not None else 'n/a'}`",
                "",
                "## Leakage Audit",
                "",
                f"- Eligible as-of rows: `{leakage_audit.get('eligible_rows', 0)}`",
                f"- Rows failing publication <= forecast rule or missing timestamps: `{leakage_audit.get('leakage_rows', 0)}`",
                f"- Missing publication timestamp: `{leakage_audit.get('missing_publication_timestamp', 0)}`",
                f"- Missing forecast timestamp: `{leakage_audit.get('missing_forecast_timestamp', 0)}`",
                f"- Missing archive snapshot timestamp: `{leakage_audit.get('missing_snapshot_timestamp', 0)}`",
                "",
                "## Coverage By Registry Year",
                "",
                markdown_table(registry.summary()),
                "",
            ]
        )
        + "\n"
    )

    analysis_lines = [
        "# Must-Run Supply Analysis",
        "",
        f"Generated: `{audit['generated_at']}`",
        "",
    ]
    if not analysis.get("available"):
        analysis_lines.extend(
            [
                "Must-run feature informativeness could not be measured yet.",
                "",
                f"Reason: `{analysis.get('reason')}`",
                "",
                "Next requirement: add plant-level KGÜP raw files with `entsoe_code`, `delivery_hour`, `publication_timestamp`, `forecast_timestamp`, and `kgup_mwh` under `data/plant_level_kgup/raw/`.",
            ]
        )
    else:
        analysis_lines.extend(
            [
                f"- Rows analyzed: `{analysis['rows']}`",
                f"- Coverage: `{analysis['coverage_start']}` → `{analysis['coverage_end']}`",
                f"- Corr must_run_share vs price: `{analysis['correlations']['must_run_share_vs_price']}`",
                f"- Corr residual_load_after_must_run vs price: `{analysis['correlations']['residual_load_after_must_run_vs_price']}`",
                f"- Corr must_run_solar vs price during solar window: `{analysis['correlations']['must_run_solar_vs_price_solar_window']}`",
                "",
                "## Regime Summary",
                "",
                markdown_table(pd.DataFrame(analysis["regime_summary"])),
            ]
        )
    ANALYSIS_MD.write_text("\n".join(analysis_lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=str(RAW_DIR), help="Directory containing plant-level KGUP raw files.")
    parser.add_argument("--fetch-api", action="store_true", help="Reserved for explicit future EPİAŞ dpp-bulk fetching.")
    args = parser.parse_args()

    if args.fetch_api:
        raise NotImplementedError(
            "API fetch is intentionally not started yet. Wire powerplant/UEVCB ID discovery and dpp-bulk batching first."
        )

    raw_dir = Path(args.raw_dir)
    registry_files = YEKDEM_Registry.discover_files(PROJECT_ROOT)
    registry = YEKDEM_Registry(registry_files)

    normalized, normalize_reports = load_raw_plant_level_kgup(raw_dir)
    matched, mapping_audit = attach_yekdem_mapping(normalized, registry)
    latest, leakage_audit = latest_asof_rows(matched)
    features = build_must_run_features(latest)
    analysis = analyze_must_run(features)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_parquet(NORMALIZED_PATH, index=False)
    matched.to_parquet(YEKDEM_MATCHED_PATH, index=False)
    features.to_parquet(FEATURE_PATH, index=False)

    write_reports(
        normalized=normalized,
        matched=matched,
        latest=latest,
        features=features,
        normalize_reports=normalize_reports,
        mapping_audit=mapping_audit,
        leakage_audit=leakage_audit,
        analysis=analysis,
        registry=registry,
    )

    print(f"Wrote {NORMALIZED_PATH}")
    print(f"Wrote {YEKDEM_MATCHED_PATH}")
    print(f"Wrote {FEATURE_PATH}")
    print(f"Wrote {AUDIT_MD}")
    print(f"Wrote {ANALYSIS_MD}")


if __name__ == "__main__":
    main()
