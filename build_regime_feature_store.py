#!/usr/bin/env python3
"""
Build leakage-safe regime feature store for PTF regime research.

No training is performed. This script only produces anchor-time safe feature
columns and audits their leakage risk.
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
LOAD_FORECAST_PATH = PROJECT_ROOT / "data" / "load_forecast.csv"
OUTAGES_PATH = PROJECT_ROOT / "data" / "outages.csv"
SMF_PATH = PROJECT_ROOT / "data" / "smf.csv"
YAL_YAT_PATH = PROJECT_ROOT / "data" / "yal_yat.csv"
SNAPSHOT_PATH = PROJECT_ROOT / "data" / "snapshots" / "interim_mcp_snapshots.csv"

OUTPUT_PATH = PROJECT_ROOT / "data" / "features" / "regime_feature_store.parquet"
AUDIT_JSON = PROJECT_ROOT / "reports" / "regime_feature_store_audit.json"
AUDIT_MD = PROJECT_ROOT / "reports" / "regime_feature_store_audit.md"

FORBIDDEN_COLUMNS = {
    "price",
    "target_regime",
    "transition_label",
    "persistence_error",
    "systemMarginalPrice",
    "upRegulationDelivered",
    "downRegulationDelivered",
    "marketTradePrice",
}


def parse_datetime_with_hour(
    df: pd.DataFrame, date_col: str = "date", hour_col: str | None = None
) -> pd.Series:
    dates = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.tz_convert(
        "Europe/Istanbul"
    )
    if hour_col is None or hour_col not in df.columns:
        return dates.dt.tz_localize(None)

    hour_text = df[hour_col].astype(str)
    if hour_text.str.match(r"^\d{2}:\d{2}").any():
        return pd.to_datetime(dates.dt.strftime("%Y-%m-%d") + " " + hour_text, errors="coerce")

    parsed_hour = pd.to_datetime(df[hour_col], errors="coerce", utc=True)
    if parsed_hour.notna().any():
        return parsed_hour.dt.tz_convert("Europe/Istanbul").dt.tz_localize(None)

    return dates.dt.tz_localize(None)


def price_band(price: pd.Series) -> pd.Series:
    return pd.cut(
        price,
        bins=[-np.inf, 50, 1500, 4000, np.inf],
        labels=["negative_zero_pressure", "normal", "tight", "spike_cap"],
    ).astype("string")


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def load_ptf_history() -> pd.DataFrame:
    ptf = pd.read_csv(PTF_PATH)
    ptf["ts_hour"] = parse_datetime_with_hour(ptf, "date", "hour")
    ptf["ptf_value_internal"] = pd.to_numeric(ptf["price"], errors="coerce")
    ptf = (
        ptf.dropna(subset=["ts_hour", "ptf_value_internal"])
        .sort_values("ts_hour")
        .drop_duplicates("ts_hour", keep="last")
        .reset_index(drop=True)
    )

    out = pd.DataFrame({"ts_hour": ptf["ts_hour"]})
    out["ptf_lag_1"] = ptf["ptf_value_internal"].shift(1)
    out["ptf_lag_24"] = ptf["ptf_value_internal"].shift(24)
    out["ptf_lag_168"] = ptf["ptf_value_internal"].shift(168)

    abs_delta = ptf["ptf_value_internal"].diff().abs()
    out["rolling_volatility"] = abs_delta.shift(1).rolling(24, min_periods=6).mean()
    vol_q75 = abs_delta.shift(1).rolling(168, min_periods=24).quantile(0.75)
    out["volatility_cluster_score"] = safe_ratio(abs_delta.shift(1), vol_q75).clip(0, 3)

    band_lag_24 = price_band(ptf["ptf_value_internal"].shift(24))
    band_lag_168 = price_band(ptf["ptf_value_internal"].shift(168))
    out["price_band_lag_24"] = band_lag_24
    out["price_band_lag_168"] = band_lag_168
    out["price_band_persistence"] = (band_lag_24 == band_lag_168).astype("float")
    out.loc[band_lag_24.isna() | band_lag_168.isna(), "price_band_persistence"] = np.nan
    return out


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = out["ts_hour"].dt.hour
    out["weekday"] = out["ts_hour"].dt.dayofweek
    out["weekend"] = out["weekday"].isin([5, 6]).astype(int)
    out["month"] = out["ts_hour"].dt.month
    out["evening_ramp_flag"] = out["hour"].between(17, 22).astype(int)
    out["sunset_window_flag"] = out["hour"].between(16, 20).astype(int)
    return out


def load_kgup() -> pd.DataFrame:
    kgup = pd.read_csv(KGUP_PATH)
    kgup["ts_hour"] = parse_datetime_with_hour(kgup, "date", "time")
    numeric_columns = [
        "toplam",
        "dogalgaz",
        "ruzgar",
        "linyit",
        "tasKomur",
        "ithalKomur",
        "fuelOil",
        "barajli",
        "akarsu",
        "gunes",
    ]
    for column in numeric_columns:
        kgup[column] = pd.to_numeric(kgup[column], errors="coerce")
    kgup = kgup.sort_values(["ts_hour", "source_type"]).drop_duplicates("ts_hour", keep="last")

    out = pd.DataFrame({"ts_hour": kgup["ts_hour"]})
    out["kgup_total"] = kgup["toplam"]
    coal = kgup[["linyit", "tasKomur", "ithalKomur"]].sum(axis=1)
    hydro = kgup[["barajli", "akarsu"]].sum(axis=1)
    thermal = kgup[["dogalgaz", "linyit", "tasKomur", "ithalKomur", "fuelOil"]].sum(axis=1)
    out["gas_share"] = safe_ratio(kgup["dogalgaz"], kgup["toplam"])
    out["coal_share"] = safe_ratio(coal, kgup["toplam"])
    out["hydro_share"] = safe_ratio(hydro, kgup["toplam"])
    out["wind_share"] = safe_ratio(kgup["ruzgar"], kgup["toplam"])
    out["solar_share"] = safe_ratio(kgup["gunes"], kgup["toplam"])
    out["thermal_share"] = safe_ratio(thermal, kgup["toplam"])
    out["kgup_wind_mw"] = kgup["ruzgar"]
    out["kgup_solar_mw"] = kgup["gunes"]
    out["kgup_renewable_mw"] = kgup[["ruzgar", "gunes", "akarsu"]].sum(axis=1)
    return out


def load_load_forecast() -> pd.DataFrame:
    load = pd.read_csv(LOAD_FORECAST_PATH)
    load["ts_hour"] = parse_datetime_with_hour(load, "date", "time")
    load["load_forecast"] = pd.to_numeric(load["lep"], errors="coerce")
    load = load[["ts_hour", "load_forecast"]].drop_duplicates("ts_hour", keep="last")
    load = load.sort_values("ts_hour")
    load["load_ramp_1h"] = load["load_forecast"].diff()
    load["load_ramp_3h"] = load["load_forecast"] - load["load_forecast"].shift(3)
    return load


def add_residual_and_renewable_pressure(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("ts_hour").copy()
    out["residual_load_forecast"] = (
        out["load_forecast"] - out[["kgup_wind_mw", "kgup_solar_mw"]].sum(axis=1)
    )
    out["residual_load_ramp"] = out["residual_load_forecast"].diff()
    out["load_minus_kgup"] = out["load_forecast"] - out["kgup_total"]

    out["solar_ramp_down"] = (out["kgup_solar_mw"].shift(1) - out["kgup_solar_mw"]).clip(lower=0)
    solar_q90 = out["solar_ramp_down"].rolling(168, min_periods=24).quantile(0.90)
    out["solar_cliff_score"] = safe_ratio(out["solar_ramp_down"], solar_q90).clip(0, 3)

    out["wind_relief_score"] = safe_ratio(out["kgup_wind_mw"], out["load_forecast"]).clip(0, 1)
    out["renewable_oversupply_score"] = safe_ratio(
        out[["kgup_wind_mw", "kgup_solar_mw"]].sum(axis=1), out["load_forecast"]
    ).clip(0, 1)
    return out


def load_outage_proxy(base_hours: pd.Series) -> pd.DataFrame:
    hours = pd.DatetimeIndex(pd.Series(base_hours).dropna().sort_values().unique())
    if hours.empty or not OUTAGES_PATH.exists():
        return pd.DataFrame(columns=["ts_hour"])

    outages = pd.read_csv(OUTAGES_PATH, low_memory=False)
    outages["start"] = pd.to_datetime(
        outages["detailStartHour"].fillna(outages["caseStartDate"]),
        errors="coerce",
        utc=True,
    ).dt.tz_convert("Europe/Istanbul").dt.tz_localize(None).dt.floor("h")
    outages["end"] = pd.to_datetime(
        outages["detailEndHour"].fillna(outages["caseEndDate"]),
        errors="coerce",
        utc=True,
    ).dt.tz_convert("Europe/Istanbul").dt.tz_localize(None).dt.floor("h")
    outages["operatorPower"] = pd.to_numeric(outages["operatorPower"], errors="coerce").fillna(0)

    start = hours.min()
    end = hours.max()
    active = outages[
        outages["start"].notna()
        & outages["end"].notna()
        & (outages["operatorPower"] > 0)
        & (outages["end"] >= start)
        & (outages["start"] <= end)
    ].copy()

    out = pd.DataFrame({"ts_hour": hours})
    for column in [
        "active_maintenance_capacity",
        "gas_maintenance",
        "coal_maintenance",
        "hydro_maintenance",
    ]:
        out[column] = 0.0
    if active.empty:
        out["outage_stress_index"] = np.nan
        return out

    text = (
        active["powerPlantName"].fillna("")
        + " "
        + active["uevcbName"].fillna("")
        + " "
        + active["reason"].fillna("")
    ).str.lower()
    active["gas_maintenance"] = np.where(
        text.str.contains("doğal|dogal|gaz|dgkç|dgkc|kombi|ccgt|bandırma|tekirdağ", regex=True, na=False),
        active["operatorPower"],
        0.0,
    )
    active["coal_maintenance"] = np.where(
        text.str.contains(
            "kömür|komur|linyit|termik|ithal|zonguldak|zetes|atlas|cenal|içdaş|icdas|iskenderun|tunçbilek|tuncbilek|yunus emre|hunutlu|çayırhan|cayirhan|bekirli",
            regex=True,
            na=False,
        ),
        active["operatorPower"],
        0.0,
    )
    active["hydro_maintenance"] = np.where(
        text.str.contains("hes|hidro|baraj|obruk|gezende|hirfanlı|sarıyar|atatürk", regex=True, na=False),
        active["operatorPower"],
        0.0,
    )

    n_hours = len(hours)
    start_idx = ((active["start"] - start) / pd.Timedelta(hours=1)).apply(np.floor).astype("int64")
    end_idx = ((active["end"] - start) / pd.Timedelta(hours=1)).apply(np.floor).astype("int64") + 1
    start_idx = start_idx.clip(0, n_hours).to_numpy()
    end_idx = end_idx.clip(0, n_hours).to_numpy()

    def sweep(values: pd.Series) -> np.ndarray:
        diff = np.zeros(n_hours + 1)
        np.add.at(diff, start_idx, values.to_numpy(float))
        np.add.at(diff, end_idx, -values.to_numpy(float))
        return np.cumsum(diff[:-1])

    out["active_maintenance_capacity"] = sweep(active["operatorPower"])
    out["gas_maintenance"] = sweep(active["gas_maintenance"])
    out["coal_maintenance"] = sweep(active["coal_maintenance"])
    out["hydro_maintenance"] = sweep(active["hydro_maintenance"])
    return out


def add_outage_stress(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["outage_stress_index"] = safe_ratio(
        out["active_maintenance_capacity"], out["load_forecast"]
    ).clip(0, 3)
    return out


def load_lagged_balancing() -> pd.DataFrame:
    smf = pd.read_csv(SMF_PATH)
    smf["ts_hour"] = parse_datetime_with_hour(smf, "date", "hour")
    smf["smf_internal"] = pd.to_numeric(smf["systemMarginalPrice"], errors="coerce")
    smf = smf[["ts_hour", "smf_internal"]].drop_duplicates("ts_hour", keep="last")

    ptf = pd.read_csv(PTF_PATH)
    ptf["ts_hour"] = parse_datetime_with_hour(ptf, "date", "hour")
    ptf["ptf_internal"] = pd.to_numeric(ptf["price"], errors="coerce")
    ptf = ptf[["ts_hour", "ptf_internal"]].drop_duplicates("ts_hour", keep="last")
    smf = smf.merge(ptf, on="ts_hour", how="left").sort_values("ts_hour")
    smf["smf_lag_24"] = smf["smf_internal"].shift(24)
    smf["smf_spread_lagged"] = (smf["smf_internal"] - smf["ptf_internal"]).shift(24)

    yy = pd.read_csv(YAL_YAT_PATH)
    yy["ts_hour"] = parse_datetime_with_hour(yy, "date", "hour")
    yy["yal_internal"] = pd.to_numeric(yy["upRegulationDelivered"], errors="coerce")
    yy["yat_internal"] = pd.to_numeric(yy["downRegulationDelivered"], errors="coerce").abs()
    yy = yy[["ts_hour", "yal_internal", "yat_internal"]].drop_duplicates(
        "ts_hour", keep="last"
    )
    yy = yy.sort_values("ts_hour")
    yy["yal_lagged"] = yy["yal_internal"].shift(24)
    yy["yat_lagged"] = yy["yat_internal"].shift(24)

    return smf[["ts_hour", "smf_lag_24", "smf_spread_lagged"]].merge(
        yy[["ts_hour", "yal_lagged", "yat_lagged"]], on="ts_hour", how="outer"
    )


def load_snapshot_features() -> pd.DataFrame:
    if not SNAPSHOT_PATH.exists():
        return pd.DataFrame(
            columns=[
                "ts_hour",
                "snapshot_marketTradePrice",
                "snapshot_publish_state",
                "snapshot_age_minutes",
            ]
        )

    snap = pd.read_csv(SNAPSHOT_PATH)
    if snap.empty:
        return pd.DataFrame(columns=["ts_hour"])

    snap["delivery_ts"] = pd.to_datetime(snap["delivery_hour"], errors="coerce")
    snap["snapshot_ts"] = pd.to_datetime(snap["snapshot_ts"], errors="coerce", utc=True).dt.tz_convert(
        "Europe/Istanbul"
    ).dt.tz_localize(None)
    snap["snapshot_marketTradePrice"] = pd.to_numeric(
        snap["marketTradePrice"], errors="coerce"
    )
    # Only keep snapshots observed before or at the delivery hour. This avoids
    # using a snapshot taken after the delivery hour as a historical feature.
    snap = snap[
        snap["delivery_ts"].notna()
        & snap["snapshot_ts"].notna()
        & (snap["snapshot_ts"] <= snap["delivery_ts"])
    ].copy()
    if snap.empty:
        return pd.DataFrame(columns=["ts_hour"])

    snap["snapshot_age_minutes"] = (
        snap["delivery_ts"] - snap["snapshot_ts"]
    ).dt.total_seconds() / 60.0
    snap = snap.sort_values(["delivery_ts", "snapshot_ts"]).drop_duplicates(
        "delivery_ts", keep="last"
    )
    snap["snapshot_publish_state"] = snap["published_status_completed"].astype("string")
    return snap.rename(columns={"delivery_ts": "ts_hour"})[
        [
            "ts_hour",
            "snapshot_marketTradePrice",
            "snapshot_publish_state",
            "snapshot_age_minutes",
        ]
    ]


def build_feature_store() -> pd.DataFrame:
    features = load_ptf_history()
    features = add_calendar(features)
    features = features.merge(load_kgup(), on="ts_hour", how="left")
    features = features.merge(load_load_forecast(), on="ts_hour", how="left")
    features = add_residual_and_renewable_pressure(features)

    outage_proxy = load_outage_proxy(features["ts_hour"])
    features = features.merge(outage_proxy, on="ts_hour", how="left")
    features = add_outage_stress(features)

    features = features.merge(load_lagged_balancing(), on="ts_hour", how="left")
    features = features.merge(load_snapshot_features(), on="ts_hour", how="left")

    forbidden_present = sorted(FORBIDDEN_COLUMNS.intersection(features.columns))
    if forbidden_present:
        raise RuntimeError(f"Forbidden leakage-prone columns present: {forbidden_present}")

    return features.sort_values("ts_hour").reset_index(drop=True)


def audit_feature_store(features: pd.DataFrame) -> dict[str, Any]:
    feature_risks = {
        "ptf_lag_1/24/168": "low: lagged finalized PTF only",
        "rolling_volatility": "low: shifted historical price volatility",
        "price_band_persistence": "low: lagged price bands only",
        "calendar": "low: deterministic calendar",
        "KGUP stack": "medium: assumes schedule version is available before anchor; revision timing should be snapshotted later",
        "load forecast": "medium: assumes latest forecast is available before anchor; revision timing should be snapshotted later",
        "renewable pressure": "medium: derived from KGUP renewable schedule, not realized generation",
        "maintenance/outage": "medium-high: publication/revision timing requires audit; uses active operatorPower proxy",
        "lagged SMF/YAL/YAT": "low-medium: lag24 only; same-hour realized values excluded",
        "snapshot KPTF": "low if snapshot_ts <= delivery_hour; sparse until archive matures",
    }
    missing_rates = {
        column: float(features[column].isna().mean())
        for column in features.columns
        if column != "ts_hour"
    }
    audit = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output": str(OUTPUT_PATH.relative_to(PROJECT_ROOT)),
        "rows": int(len(features)),
        "columns": list(features.columns),
        "coverage_start": features["ts_hour"].min().isoformat() if not features.empty else None,
        "coverage_end": features["ts_hour"].max().isoformat() if not features.empty else None,
        "forbidden_columns_present": sorted(FORBIDDEN_COLUMNS.intersection(features.columns)),
        "feature_risks": feature_risks,
        "missing_rates_top20": dict(
            sorted(missing_rates.items(), key=lambda item: item[1], reverse=True)[:20]
        ),
        "snapshot_rows_with_safe_kptf": int(features["snapshot_marketTradePrice"].notna().sum())
        if "snapshot_marketTradePrice" in features.columns
        else 0,
        "excluded_sources": [
            "historical raw interim_mcp oracle dataset",
            "same-hour finalized PTF",
            "same-hour realized SMF",
            "same-hour realized YAL/YAT",
            "same-hour realized generation",
        ],
    }
    return audit


def write_outputs(features: pd.DataFrame, audit: dict[str, Any]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_JSON.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUTPUT_PATH, index=False)
    AUDIT_JSON.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")

    lines = [
        "# Regime Feature Store Audit",
        "",
        f"Generated: `{audit['generated_at']}`",
        "",
        "No training is performed. This table contains anchor-time safe features only.",
        "",
        f"- Output: `{audit['output']}`",
        f"- Rows: `{audit['rows']}`",
        f"- Coverage: `{audit['coverage_start']}` -> `{audit['coverage_end']}`",
        f"- Safe K.PTF snapshot rows: `{audit['snapshot_rows_with_safe_kptf']}`",
        f"- Forbidden columns present: `{audit['forbidden_columns_present']}`",
        "",
        "## Leakage Risk By Feature Family",
        "",
        "| Feature family | Risk |",
        "|---|---|",
    ]
    for family, risk in audit["feature_risks"].items():
        lines.append(f"| `{family}` | {risk} |")
    lines.extend(["", "## Highest Missing Rates", "", "| Column | Missing rate |", "|---|---:|"])
    for column, rate in audit["missing_rates_top20"].items():
        lines.append(f"| `{column}` | {rate:.3f} |")
    lines.extend(
        [
            "",
            "## Explicitly Excluded",
            "",
        ]
    )
    for item in audit["excluded_sources"]:
        lines.append(f"- {item}")
    AUDIT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    features = build_feature_store()
    audit = audit_feature_store(features)
    write_outputs(features, audit)
    print(f"Wrote {OUTPUT_PATH} rows={len(features)} columns={len(features.columns)}")
    print(f"Wrote {AUDIT_JSON}")
    print(f"Wrote {AUDIT_MD}")


if __name__ == "__main__":
    main()
