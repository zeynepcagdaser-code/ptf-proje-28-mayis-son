#!/usr/bin/env python3
"""
Build fuel-switch / marginality feature family and merge it into the regime
feature store.

This is a leakage-safe, anchor-time feature builder. It only uses contemporaneous
planned/forecast generation and load tables plus trailing rolling norms.
No training is performed.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
FEATURE_STORE_PATH = PROJECT_ROOT / "data" / "features" / "regime_feature_store.parquet"
PTF_PATH = PROJECT_ROOT / "data" / "ptf_dataset.csv"
KGUP_PATH = PROJECT_ROOT / "data" / "kgup_combined.csv"
LOAD_FORECAST_PATH = PROJECT_ROOT / "data" / "load_forecast.csv"
REGIME_LABELS_PATH = PROJECT_ROOT / "data" / "regime_labels.csv"

OUTPUT_PATH = PROJECT_ROOT / "data" / "features" / "regime_feature_store.parquet"
FUEL_SWITCH_PATH = PROJECT_ROOT / "data" / "features" / "fuel_switch_market_features.parquet"
AUDIT_JSON = PROJECT_ROOT / "reports" / "fuel_switch_market_features_audit.json"
AUDIT_MD = PROJECT_ROOT / "reports" / "fuel_switch_market_features_audit.md"

NEW_COLUMNS = [
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


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def robust_score(series: pd.Series, high_is_risky: bool = True) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    q10 = numeric.quantile(0.10)
    q90 = numeric.quantile(0.90)
    if pd.isna(q10) or pd.isna(q90) or q90 == q10:
        return pd.Series(0.0, index=series.index)
    score = (numeric - q10) / (q90 - q10)
    if not high_is_risky:
        score = 1 - score
    return (score.clip(0, 1) * 100).fillna(0)


def parse_datetime_with_hour(df: pd.DataFrame, date_col: str = "date", hour_col: str | None = None) -> pd.Series:
    dates = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.tz_convert("Europe/Istanbul")
    if hour_col is None or hour_col not in df.columns:
        return dates.dt.tz_localize(None)
    hour_text = df[hour_col].astype(str)
    if hour_text.str.match(r"^\d{2}:\d{2}").any():
        return pd.to_datetime(dates.dt.strftime("%Y-%m-%d") + " " + hour_text, errors="coerce")
    parsed_hour = pd.to_datetime(df[hour_col], errors="coerce", utc=True)
    if parsed_hour.notna().any():
        return parsed_hour.dt.tz_convert("Europe/Istanbul").dt.tz_localize(None)
    return dates.dt.tz_localize(None)


def load_base_store() -> pd.DataFrame:
    if not FEATURE_STORE_PATH.exists():
        raise FileNotFoundError(f"Missing feature store: {FEATURE_STORE_PATH}")
    frame = pd.read_parquet(FEATURE_STORE_PATH)
    frame["ts_hour"] = pd.to_datetime(frame["ts_hour"], errors="coerce")
    return frame.sort_values("ts_hour").reset_index(drop=True)


def load_generation_stack() -> pd.DataFrame:
    kgup = pd.read_csv(KGUP_PATH, low_memory=False)
    kgup["ts_hour"] = parse_datetime_with_hour(kgup, "date", "time")
    for col in [
        "toplam",
        "dogalgaz",
        "ruzgar",
        "linyit",
        "tasKomur",
        "ithalKomur",
        "fuelOil",
        "jeotermal",
        "barajli",
        "nafta",
        "biokutle",
        "akarsu",
        "gunes",
        "diger",
    ]:
        kgup[col] = pd.to_numeric(kgup[col], errors="coerce")

    kgup = kgup.sort_values(["ts_hour", "source_type"]).drop_duplicates("ts_hour", keep="last")

    out = pd.DataFrame({"ts_hour": kgup["ts_hour"]})
    out["gas_share_of_generation"] = safe_ratio(kgup["dogalgaz"], kgup["toplam"])
    out["hydro_share_of_generation"] = safe_ratio(kgup[["barajli", "akarsu"]].sum(axis=1), kgup["toplam"])
    renewable_total = kgup[["ruzgar", "gunes", "barajli", "akarsu", "biokutle", "jeotermal"]].sum(axis=1)
    out["renewable_share_of_generation"] = safe_ratio(renewable_total, kgup["toplam"])
    out["renewable_minus_gas_shift"] = out["renewable_share_of_generation"] - out["gas_share_of_generation"]
    out["load_vs_renewable_balance"] = np.nan  # filled after load merge
    return out


def load_forecast_stack() -> pd.DataFrame:
    load = pd.read_csv(LOAD_FORECAST_PATH, low_memory=False)
    load["ts_hour"] = parse_datetime_with_hour(load, "date", "time")
    load["load_forecast"] = pd.to_numeric(load["lep"], errors="coerce")
    load = load[["ts_hour", "load_forecast"]].drop_duplicates("ts_hour", keep="last").sort_values("ts_hour")

    load["load_roll_7d"] = load["load_forecast"].shift(24).rolling(24 * 7, min_periods=24 * 5).mean()
    load["load_roll_30d"] = load["load_forecast"].shift(24).rolling(24 * 30, min_periods=24 * 14).mean()
    load["load_deviation_from_weekly_norm"] = load["load_forecast"] - load["load_roll_7d"]
    load["load_deviation_from_monthly_norm"] = load["load_forecast"] - load["load_roll_30d"]
    return load


def compute_fuel_switch_features(base: pd.DataFrame) -> pd.DataFrame:
    base = base.drop(
        columns=[
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
        ],
        errors="ignore",
    )
    gen = load_generation_stack()
    load = load_forecast_stack()
    out = base.merge(gen, on="ts_hour", how="left")
    out = out.merge(load, on="ts_hour", how="left")

    if "load_forecast_x" in out.columns or "load_forecast_y" in out.columns:
        left = out["load_forecast_x"] if "load_forecast_x" in out.columns else pd.Series(np.nan, index=out.index)
        right = out["load_forecast_y"] if "load_forecast_y" in out.columns else pd.Series(np.nan, index=out.index)
        out["load_forecast"] = left.combine_first(right)
        out = out.drop(columns=[c for c in ["load_forecast_x", "load_forecast_y"] if c in out.columns])
    if "load_forecast" not in out.columns and "load_forecast_y" not in out.columns:
        out["load_forecast"] = base.get("load_forecast", pd.Series(np.nan, index=out.index))
    if "kgup_renewable_mw" not in out.columns and "kgup_renewable_mw_x" in out.columns:
        out["kgup_renewable_mw"] = out["kgup_renewable_mw_x"]
    if "kgup_renewable_mw" not in out.columns and "kgup_renewable_mw_y" in out.columns:
        out["kgup_renewable_mw"] = out["kgup_renewable_mw_y"]

    if "kgup_renewable_mw" not in out.columns:
        raise RuntimeError("Base feature store missing kgup_renewable_mw")

    out["load_vs_renewable_balance"] = out["load_forecast"] - out["kgup_renewable_mw"]

    # Demand weakness: negative deviations below past weekly/monthly norm.
    weekly_norm = out["load_roll_7d"].replace(0, np.nan)
    monthly_norm = out["load_roll_30d"].replace(0, np.nan)
    weekly_gap = ((weekly_norm - out["load_forecast"]) / weekly_norm).clip(lower=0, upper=2)
    monthly_gap = ((monthly_norm - out["load_forecast"]) / monthly_norm).clip(lower=0, upper=2)
    weakness_raw = 0.60 * weekly_gap.fillna(0) + 0.40 * monthly_gap.fillna(0)
    out["demand_weakness_score"] = (weakness_raw / 1.5 * 100).clip(0, 100)
    out["low_demand_flag"] = (
        (out["demand_weakness_score"] >= 60) | ((weekly_gap > 0.10) & (monthly_gap > 0.05))
    ).astype(int)

    # Gas marginality: gas share becomes more important when demand is strong and
    # renewable share is lower.
    load_pressure = ((out["load_forecast"] - weekly_norm) / weekly_norm).fillna(0).clip(-0.5, 1.5)
    load_pressure = ((load_pressure + 0.5) / 2.0).clip(0, 1)
    renewable_relief = out["renewable_share_of_generation"].fillna(0).clip(0, 1)
    gas_dependency = out["gas_share_of_generation"].fillna(0).clip(0, 1)
    hydro_presence = out["hydro_share_of_generation"].fillna(0).clip(0, 1)
    out["gas_marginality_proxy"] = (
        100
        * (
            0.45 * gas_dependency
            + 0.25 * load_pressure
            + 0.20 * (1 - renewable_relief)
            + 0.10 * (1 - hydro_presence)
        )
    ).clip(0, 100)

    out["hydro_displacement_score"] = (
        100
        * (
            0.45 * hydro_presence
            + 0.30 * out["renewable_minus_gas_shift"].fillna(0).clip(-1, 1).add(1).div(2)
            + 0.25
            * (
                1
                - (
                    (out["load_vs_renewable_balance"] / out["load_forecast"].replace(0, np.nan))
                    .fillna(0)
                    .clip(0, 2)
                    / 2
                )
            )
        )
    ).clip(0, 100)

    out["cheap_supply_pressure"] = (
        100
        * (
            0.40 * renewable_relief
            + 0.25 * hydro_presence
            + 0.20 * (1 - load_pressure)
            + 0.15 * (1 - gas_dependency)
        )
    ).clip(0, 100)

    # Flags use fixed, interpretable thresholds to remain point-in-time safe.
    out["gas_off_flag"] = (gas_dependency <= 0.10).astype(int)
    out["renewable_share_high_flag"] = (renewable_relief >= 0.55).astype(int)
    out["hydro_high_flag"] = (hydro_presence >= 0.20).astype(int)

    out["zero_price_pressure_score"] = (
        100
        * (
            0.32 * out["low_demand_flag"].astype(float)
            + 0.22 * out["gas_off_flag"].astype(float)
            + 0.22 * out["renewable_share_high_flag"].astype(float)
            + 0.14 * out["hydro_high_flag"].astype(float)
            + 0.10 * (1 - load_pressure)
        )
    ).clip(0, 100)

    # Keep the output compact and explicit.
    return out[
        [
            "ts_hour",
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
    ]


def load_regime_labels() -> pd.DataFrame:
    if not REGIME_LABELS_PATH.exists():
        return pd.DataFrame(columns=["ts_hour", "price", "target_regime"])
    labels = pd.read_csv(REGIME_LABELS_PATH, low_memory=False)
    if "ts_hour" in labels.columns:
        labels["ts_hour"] = pd.to_datetime(labels["ts_hour"], errors="coerce")
    if "price" in labels.columns:
        labels["price"] = pd.to_numeric(labels["price"], errors="coerce")
    return labels[[c for c in ["ts_hour", "price", "target_regime"] if c in labels.columns]].dropna(subset=["ts_hour"])


def analyze_features(features: pd.DataFrame) -> dict[str, Any]:
    labels = load_regime_labels()
    joined = features.merge(labels, on="ts_hour", how="left")
    joined["zero_price_flag"] = joined["price"].le(50)
    joined["low_price_flag"] = joined["price"].le(1500)
    joined["spike_flag"] = joined["price"].ge(4000)
    joined["tight_flag"] = joined["price"].between(1500, 3999.99)

    def corr(a: str, b: str) -> float | None:
        x = pd.to_numeric(joined[a], errors="coerce")
        y = pd.to_numeric(joined[b], errors="coerce")
        if x.notna().sum() < 3 or y.notna().sum() < 3:
            return None
        return float(x.corr(y))

    zero_slice = joined[joined["zero_price_flag"]]
    spike_slice = joined[joined["spike_flag"]]
    low_slice = joined[joined["low_price_flag"]]

    regime_means = {}
    if "target_regime" in joined.columns:
        for regime in ["negative_zero_pressure", "normal", "tight", "spike_cap"]:
            grp = joined[joined["target_regime"] == regime]
            if grp.empty:
                continue
            regime_means[regime] = {
                "rows": int(len(grp)),
                "gas_marginality_proxy_mean": float(grp["gas_marginality_proxy"].mean()),
                "hydro_displacement_score_mean": float(grp["hydro_displacement_score"].mean()),
                "renewable_share_of_generation_mean": float(grp["renewable_share_of_generation"].mean()),
                "gas_share_of_generation_mean": float(grp["gas_share_of_generation"].mean()),
                "demand_weakness_score_mean": float(grp["demand_weakness_score"].mean()),
                "zero_price_pressure_score_mean": float(grp["zero_price_pressure_score"].mean()),
            }

    high_zero = joined.sort_values("zero_price_pressure_score", ascending=False).head(20)
    high_gas = joined.sort_values("gas_marginality_proxy", ascending=False).head(20)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(features)),
        "coverage_start": features["ts_hour"].min().isoformat() if not features.empty else None,
        "coverage_end": features["ts_hour"].max().isoformat() if not features.empty else None,
        "missing_rates": {
            col: float(features[col].isna().mean())
            for col in NEW_COLUMNS
            if col in features.columns
        },
        "correlations": {
            "gas_marginality_proxy_vs_price": corr("gas_marginality_proxy", "price"),
            "hydro_displacement_score_vs_price": corr("hydro_displacement_score", "price"),
            "renewable_share_of_generation_vs_price": corr("renewable_share_of_generation", "price"),
            "gas_share_of_generation_vs_price": corr("gas_share_of_generation", "price"),
            "zero_price_pressure_score_vs_price": corr("zero_price_pressure_score", "price"),
            "demand_weakness_score_vs_price": corr("demand_weakness_score", "price"),
            "load_vs_renewable_balance_vs_price": corr("load_vs_renewable_balance", "price"),
        },
        "regime_means": regime_means,
        "slice_means": {
            "zero_price": {
                "rows": int(len(zero_slice)),
                "gas_marginality_proxy_mean": float(zero_slice["gas_marginality_proxy"].mean()) if not zero_slice.empty else None,
                "hydro_displacement_score_mean": float(zero_slice["hydro_displacement_score"].mean()) if not zero_slice.empty else None,
                "renewable_share_of_generation_mean": float(zero_slice["renewable_share_of_generation"].mean()) if not zero_slice.empty else None,
                "gas_share_of_generation_mean": float(zero_slice["gas_share_of_generation"].mean()) if not zero_slice.empty else None,
                "zero_price_pressure_score_mean": float(zero_slice["zero_price_pressure_score"].mean()) if not zero_slice.empty else None,
            },
            "low_price": {
                "rows": int(len(low_slice)),
                "gas_marginality_proxy_mean": float(low_slice["gas_marginality_proxy"].mean()) if not low_slice.empty else None,
                "hydro_displacement_score_mean": float(low_slice["hydro_displacement_score"].mean()) if not low_slice.empty else None,
                "renewable_share_of_generation_mean": float(low_slice["renewable_share_of_generation"].mean()) if not low_slice.empty else None,
                "gas_share_of_generation_mean": float(low_slice["gas_share_of_generation"].mean()) if not low_slice.empty else None,
                "zero_price_pressure_score_mean": float(low_slice["zero_price_pressure_score"].mean()) if not low_slice.empty else None,
            },
            "spike": {
                "rows": int(len(spike_slice)),
                "gas_marginality_proxy_mean": float(spike_slice["gas_marginality_proxy"].mean()) if not spike_slice.empty else None,
                "hydro_displacement_score_mean": float(spike_slice["hydro_displacement_score"].mean()) if not spike_slice.empty else None,
                "renewable_share_of_generation_mean": float(spike_slice["renewable_share_of_generation"].mean()) if not spike_slice.empty else None,
                "gas_share_of_generation_mean": float(spike_slice["gas_share_of_generation"].mean()) if not spike_slice.empty else None,
                "zero_price_pressure_score_mean": float(spike_slice["zero_price_pressure_score"].mean()) if not spike_slice.empty else None,
            },
        },
        "top_zero_pressure_hours": [
            {"ts_hour": row.ts_hour.isoformat() if pd.notna(row.ts_hour) else None, "score": float(row.zero_price_pressure_score), "price": float(row.price) if pd.notna(row.price) else None}
            for row in high_zero.itertuples(index=False)
        ],
        "top_gas_marginality_hours": [
            {"ts_hour": row.ts_hour.isoformat() if pd.notna(row.ts_hour) else None, "score": float(row.gas_marginality_proxy), "price": float(row.price) if pd.notna(row.price) else None}
            for row in high_gas.itertuples(index=False)
        ],
        "leakage_notes": [
            "Uses same-hour planned/forecast load and generation shares only.",
            "Load deviation baselines are trailing rolling windows shifted by 24h.",
            "No realized future PTF is used in feature construction.",
        ],
        "target_availability": {
            "price_rows_available": int(joined["price"].notna().sum()),
            "zero_price_rows": int(zero_slice.shape[0]),
            "spike_rows": int(spike_slice.shape[0]),
        },
    }


def write_outputs(features: pd.DataFrame, report: dict[str, Any]) -> None:
    FUEL_SWITCH_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_JSON.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(FUEL_SWITCH_PATH, index=False)

    # Merge into the main feature store, replacing or appending the new columns.
    base = load_base_store().drop(columns=[c for c in NEW_COLUMNS if c in load_base_store().columns], errors="ignore")
    merged = base.merge(features, on="ts_hour", how="left")
    merged.to_parquet(OUTPUT_PATH, index=False)

    AUDIT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    lines = [
        "# Fuel Switch / Marginality Feature Audit",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "This layer adds explicit gas marginality, hydro displacement, and demand weakness features to the regime feature store.",
        "",
        f"- Rows: `{report['rows']}`",
        f"- Coverage: `{report['coverage_start']}` -> `{report['coverage_end']}`",
        "",
        "## New Columns",
        "",
    ]
    for column in NEW_COLUMNS:
        lines.append(f"- `{column}`")
    lines.extend([
        "",
        "## Correlations With Price",
        "",
        "| Feature | Corr(price) |",
        "|---|---:|",
    ])
    for key, value in report["correlations"].items():
        lines.append(f"| `{key}` | {value:.4f} |" if value is not None else f"| `{key}` | n/a |")
    lines.extend([
        "",
        "## Regime Means",
        "",
        "| Regime | Rows | Gas marginality | Hydro displacement | Renewable share | Gas share | Demand weakness | Zero-pressure score |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for regime, stats in report["regime_means"].items():
        lines.append(
            f"| `{regime}` | {stats['rows']} | {stats['gas_marginality_proxy_mean']:.2f} | {stats['hydro_displacement_score_mean']:.2f} | {stats['renewable_share_of_generation_mean']:.3f} | {stats['gas_share_of_generation_mean']:.3f} | {stats['demand_weakness_score_mean']:.2f} | {stats['zero_price_pressure_score_mean']:.2f} |"
        )
    lines.extend([
        "",
        "## Price Slices",
        "",
        "| Slice | Rows | Gas marginality | Hydro displacement | Renewable share | Gas share | Zero-pressure score |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for name, stats in report["slice_means"].items():
        lines.append(
            f"| `{name}` | {stats['rows']} | {stats['gas_marginality_proxy_mean']:.2f} | {stats['hydro_displacement_score_mean']:.2f} | {stats['renewable_share_of_generation_mean']:.3f} | {stats['gas_share_of_generation_mean']:.3f} | {stats['zero_price_pressure_score_mean']:.2f} |"
        )
    lines.extend([
        "",
        "## Top Zero-Pressure Hours",
        "",
    ])
    for row in report["top_zero_pressure_hours"][:10]:
        lines.append(f"- `{row['ts_hour']}` score={row['score']:.2f} price={row['price']}")
    lines.extend([
        "",
        "## Top Gas Marginality Hours",
        "",
    ])
    for row in report["top_gas_marginality_hours"][:10]:
        lines.append(f"- `{row['ts_hour']}` score={row['score']:.2f} price={row['price']}")
    lines.extend([
        "",
        "## Leakage Notes",
        "",
    ])
    for note in report["leakage_notes"]:
        lines.append(f"- {note}")
    AUDIT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    base = load_base_store()
    features = compute_fuel_switch_features(base)
    report = analyze_features(features)
    write_outputs(features, report)
    print(f"Wrote {FUEL_SWITCH_PATH} rows={len(features)}")
    print(f"Wrote {OUTPUT_PATH}")
    print(f"Wrote {AUDIT_JSON}")
    print(f"Wrote {AUDIT_MD}")


if __name__ == "__main__":
    main()
