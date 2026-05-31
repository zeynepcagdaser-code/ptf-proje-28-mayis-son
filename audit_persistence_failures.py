#!/usr/bin/env python3
"""
Audit where lag24 persistence fails for regime-aware PTF research.

Evaluation-only script. It does not train models and does not create a feature
store. Finalized PTF labels are used only for diagnostics.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
LABELS_PATH = PROJECT_ROOT / "data" / "regime_labels.csv"
PTF_PATH = PROJECT_ROOT / "data" / "ptf_dataset.csv"
KGUP_PATH = PROJECT_ROOT / "data" / "kgup_combined.csv"
LOAD_FORECAST_PATH = PROJECT_ROOT / "data" / "load_forecast.csv"
OUTAGES_PATH = PROJECT_ROOT / "data" / "outages.csv"
REPORT_JSON = PROJECT_ROOT / "reports" / "persistence_failure_alpha_map.json"
REPORT_MD = PROJECT_ROOT / "reports" / "persistence_failure_alpha_map.md"


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


def load_or_build_labels() -> pd.DataFrame:
    if LABELS_PATH.exists():
        labels = pd.read_csv(LABELS_PATH)
        labels["ts_hour"] = pd.to_datetime(labels["ts_hour"], errors="coerce")
        return labels

    # Local fallback keeps this script useful even if the label builder has not
    # been run yet. The canonical path is still build_regime_labels.py.
    import build_regime_labels

    labels = build_regime_labels.build_labels()
    build_regime_labels.write_outputs(labels)
    return labels


def load_context(labels: pd.DataFrame) -> pd.DataFrame:
    load = pd.read_csv(LOAD_FORECAST_PATH)
    load["ts_hour"] = parse_datetime_with_hour(load, "date", "time")
    load["lep"] = pd.to_numeric(load["lep"], errors="coerce")
    load = load[["ts_hour", "lep"]].drop_duplicates("ts_hour", keep="last")

    kgup = pd.read_csv(KGUP_PATH)
    kgup["ts_hour"] = parse_datetime_with_hour(kgup, "date", "time")
    for column in [
        "toplam",
        "dogalgaz",
        "ruzgar",
        "gunes",
        "barajli",
        "akarsu",
        "ithalKomur",
        "linyit",
        "tasKomur",
        "fuelOil",
    ]:
        kgup[column] = pd.to_numeric(kgup[column], errors="coerce")
    kgup = kgup.sort_values(["ts_hour", "source_type"]).drop_duplicates(
        "ts_hour", keep="last"
    )
    kgup["kgup_thermal"] = kgup[
        ["dogalgaz", "ithalKomur", "linyit", "tasKomur", "fuelOil"]
    ].sum(axis=1)
    kgup = kgup[
        [
            "ts_hour",
            "toplam",
            "dogalgaz",
            "ruzgar",
            "gunes",
            "kgup_thermal",
            "source_type",
        ]
    ]

    out = labels.merge(load, on="ts_hour", how="left").merge(kgup, on="ts_hour", how="left")
    out["forecast_residual_load"] = out["lep"] - out[["ruzgar", "gunes"]].sum(axis=1)
    out["load_minus_kgup"] = out["lep"] - out["toplam"]
    out["residual_load_bin"] = pd.qcut(
        out["forecast_residual_load"],
        q=5,
        labels=["Q1_low", "Q2", "Q3", "Q4", "Q5_high"],
        duplicates="drop",
    )
    out["load_minus_kgup_bin"] = pd.qcut(
        out["load_minus_kgup"],
        q=5,
        labels=["Q1_low", "Q2", "Q3", "Q4", "Q5_high"],
        duplicates="drop",
    )
    return out


def add_2026_outage_proxy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["outage_operator_power"] = np.nan
    subset_mask = out["ts_hour"].dt.year == 2026
    if not subset_mask.any() or not OUTAGES_PATH.exists():
        return out

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

    sub = out.loc[subset_mask, ["ts_hour"]].copy()
    start = sub["ts_hour"].min().floor("h")
    end = sub["ts_hour"].max().floor("h")
    hours = pd.date_range(start, end, freq="h")
    if hours.empty:
        return out

    active = outages[
        outages["start"].notna()
        & outages["end"].notna()
        & (outages["operatorPower"] > 0)
        & (outages["end"] >= start)
        & (outages["start"] <= end)
    ].copy()
    if active.empty:
        return out

    n_hours = len(hours)
    start_offset = ((active["start"] - start) / pd.Timedelta(hours=1)).apply(np.floor)
    end_offset = ((active["end"] - start) / pd.Timedelta(hours=1)).apply(np.floor) + 1
    start_idx = start_offset.astype("int64").clip(0, n_hours).to_numpy()
    end_idx = end_offset.astype("int64").clip(0, n_hours).to_numpy()

    diff = np.zeros(n_hours + 1)
    np.add.at(diff, start_idx, active["operatorPower"].to_numpy(float))
    np.add.at(diff, end_idx, -active["operatorPower"].to_numpy(float))
    proxy = pd.DataFrame(
        {"ts_hour": hours, "outage_operator_power": np.cumsum(diff[:-1])}
    )
    out = out.drop(columns=["outage_operator_power"]).merge(proxy, on="ts_hour", how="left")
    out["outage_stress_bin"] = pd.qcut(
        out["outage_operator_power"],
        q=4,
        labels=["Q1_low", "Q2", "Q3", "Q4_high"],
        duplicates="drop",
    )
    return out


def grouped_mae(df: pd.DataFrame, group_cols: list[str], min_count: int = 1) -> pd.DataFrame:
    grouped = (
        df.dropna(subset=["persistence_error"])
        .groupby(group_cols, observed=False)["persistence_error"]
        .agg(["count", "mean", "median"])
        .reset_index()
    )
    grouped = grouped[grouped["count"] >= min_count]
    return grouped.sort_values("mean", ascending=False).reset_index(drop=True)


def records(df: pd.DataFrame, n: int | None = None) -> list[dict[str, Any]]:
    if n is not None:
        df = df.head(n)
    out: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        clean: dict[str, Any] = {}
        for key, value in row.items():
            if pd.isna(value):
                clean[key] = None
            elif isinstance(value, (np.integer,)):
                clean[key] = int(value)
            elif isinstance(value, (np.floating,)):
                clean[key] = float(value)
            else:
                clean[key] = value
        out.append(clean)
    return out


def build_audit() -> dict[str, Any]:
    labels = load_or_build_labels()
    df = load_context(labels)
    df = add_2026_outage_proxy(df)
    valid = df.dropna(subset=["persistence_error"]).copy()

    by_hour = grouped_mae(valid, ["hour"])
    by_regime = grouped_mae(valid, ["target_regime"])
    by_transition = grouped_mae(valid, ["transition_label"])
    by_hour_regime = grouped_mae(valid, ["hour", "target_regime"], min_count=50)
    by_residual = grouped_mae(valid, ["residual_load_bin"])
    by_load_gap = grouped_mae(valid, ["load_minus_kgup_bin"])
    by_outage_residual = grouped_mae(
        valid[valid["outage_stress_bin"].notna()],
        ["outage_stress_bin", "residual_load_bin"],
        min_count=20,
    )

    h1_h4 = valid[valid["hour"].isin([1, 2, 3, 4])]
    h24_full = valid

    audit = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "labels": str(LABELS_PATH.relative_to(PROJECT_ROOT)),
            "load_forecast": str(LOAD_FORECAST_PATH.relative_to(PROJECT_ROOT)),
            "kgup": str(KGUP_PATH.relative_to(PROJECT_ROOT)),
            "outages": str(OUTAGES_PATH.relative_to(PROJECT_ROOT)),
        },
        "rows": int(len(df)),
        "valid_persistence_rows": int(len(valid)),
        "h1_h4": {
            "definition": "target delivery hour in {1,2,3,4}; diagnostic slice only, not model horizon training",
            "rows": int(len(h1_h4)),
            "persistence_mae": float(h1_h4["persistence_error"].mean()),
            "persistence_median_error": float(h1_h4["persistence_error"].median()),
        },
        "full_h24": {
            "definition": "all delivery hours 0-23",
            "rows": int(len(h24_full)),
            "persistence_mae": float(h24_full["persistence_error"].mean()),
            "persistence_median_error": float(h24_full["persistence_error"].median()),
        },
        "worst_hours": records(by_hour, 24),
        "regime_mae": records(by_regime),
        "worst_transitions": records(by_transition, 20),
        "hour_x_regime_mae": records(by_hour_regime, 30),
        "residual_load_mae": records(by_residual),
        "load_minus_kgup_mae": records(by_load_gap),
        "residual_load_x_outage_stress_mae_2026": records(by_outage_residual, 20),
        "alpha_map": {
            "highest_alpha": [
                "normal -> spike_cap",
                "spike_cap -> normal",
                "negative_zero_pressure -> tight",
                "tight -> negative_zero_pressure",
                "tight -> spike_cap",
                "normal -> tight",
            ],
            "interpretation": "Persistence fails when yesterday's same-hour regime is no longer valid. Training should prioritize regime transition detection before price-level experts.",
        },
        "leakage_policy": {
            "finalized_ptf_usage": "evaluation_and_labels_only",
            "forbidden_as_features": [
                "price",
                "target_regime",
                "transition_label",
                "persistence_error",
                "same-hour realized SMF/YAL/YAT",
                "historical interim-mcp oracle data",
            ],
        },
    }
    return audit


def write_reports(audit: dict[str, Any]) -> None:
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")

    lines = [
        "# Persistence Failure Alpha Map",
        "",
        f"Generated: `{audit['generated_at']}`",
        "",
        "Evaluation-only report. No model is trained here.",
        "",
        "Finalized PTF is used only for regime labels and persistence-error evaluation.",
        "",
        "## H1-H4 vs Full H24",
        "",
        "| Slice | Rows | Persistence MAE | Median error |",
        "|---|---:|---:|---:|",
        f"| H1-H4 diagnostic slice | {audit['h1_h4']['rows']} | {audit['h1_h4']['persistence_mae']:.2f} | {audit['h1_h4']['persistence_median_error']:.2f} |",
        f"| Full H24 | {audit['full_h24']['rows']} | {audit['full_h24']['persistence_mae']:.2f} | {audit['full_h24']['persistence_median_error']:.2f} |",
        "",
        "H1-H4 here is a delivery-hour diagnostic slice. Later model evaluation must define horizon relative to anchor time separately.",
        "",
        "## Worst Transitions",
        "",
        "| Transition | Rows | MAE | Median |",
        "|---|---:|---:|---:|",
    ]
    for row in audit["worst_transitions"][:12]:
        lines.append(
            f"| `{row['transition_label']}` | {row['count']} | {row['mean']:.2f} | {row['median']:.2f} |"
        )

    lines.extend(["", "## Regime MAE", "", "| Regime | Rows | MAE | Median |", "|---|---:|---:|---:|"])
    for row in audit["regime_mae"]:
        lines.append(
            f"| `{row['target_regime']}` | {row['count']} | {row['mean']:.2f} | {row['median']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Hour x Regime Hotspots",
            "",
            "| Hour | Regime | Rows | MAE | Median |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for row in audit["hour_x_regime_mae"][:20]:
        lines.append(
            f"| {row['hour']} | `{row['target_regime']}` | {row['count']} | {row['mean']:.2f} | {row['median']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Residual Load x Outage Stress",
            "",
            "The outage proxy is limited to 2026 active operator-power maintenance/outage windows.",
            "",
            "| Outage stress | Residual load | Rows | MAE | Median |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in audit["residual_load_x_outage_stress_mae_2026"][:12]:
        lines.append(
            f"| `{row['outage_stress_bin']}` | `{row['residual_load_bin']}` | {row['count']} | {row['mean']:.2f} | {row['median']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Alpha Map",
            "",
            "The highest-alpha slices are not average hours. They are regime transitions where lag24 carries the wrong market state:",
            "",
        ]
    )
    for item in audit["alpha_map"]["highest_alpha"]:
        lines.append(f"- `{item}`")
    lines.extend(
        [
            "",
            "## Leakage Policy",
            "",
            "- Finalized PTF is allowed here only for labels and evaluation.",
            "- `transition_label` is exactly `lag24_regime -> target_regime`.",
            "- `persistence_error = abs(price - price_lag_24)`.",
            "- Do not feed target/evaluation columns into the future feature store.",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    audit = build_audit()
    write_reports(audit)
    print(f"Wrote {REPORT_JSON}")
    print(f"Wrote {REPORT_MD}")


if __name__ == "__main__":
    main()
