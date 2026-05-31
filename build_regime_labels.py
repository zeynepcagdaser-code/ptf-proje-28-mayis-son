#!/usr/bin/env python3
"""
Build finalized-PTF regime labels for regime-aware research.

This script creates labels only. Finalized PTF is used as the target/evaluation
source and must not be used as a feature in downstream forecasting.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
PTF_PATH = PROJECT_ROOT / "data" / "ptf_dataset.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "regime_labels.csv"
SUMMARY_JSON = PROJECT_ROOT / "reports" / "regime_label_summary.json"
SUMMARY_MD = PROJECT_ROOT / "reports" / "regime_label_summary.md"


def parse_ptf_datetime(df: pd.DataFrame) -> pd.Series:
    dates = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_convert(
        "Europe/Istanbul"
    )
    if "hour" not in df.columns:
        return dates

    hour_text = df["hour"].astype(str)
    if hour_text.str.match(r"^\d{2}:\d{2}").any():
        date_part = dates.dt.strftime("%Y-%m-%d")
        return pd.to_datetime(date_part + " " + hour_text, errors="coerce")

    parsed_hour = pd.to_datetime(df["hour"], errors="coerce", utc=True)
    if parsed_hour.notna().any():
        return parsed_hour.dt.tz_convert("Europe/Istanbul").dt.tz_localize(None)

    return dates.dt.tz_localize(None)


def assign_regime(price: float) -> str:
    if price <= 50:
        return "negative_zero_pressure"
    if price >= 4000:
        return "spike_cap"
    if price >= 1500:
        return "tight"
    return "normal"


def build_labels() -> pd.DataFrame:
    df = pd.read_csv(PTF_PATH)
    df["ts_hour"] = parse_ptf_datetime(df)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = (
        df.dropna(subset=["ts_hour", "price"])
        .sort_values("ts_hour")
        .drop_duplicates("ts_hour", keep="last")
        .reset_index(drop=True)
    )

    df["target_regime"] = df["price"].map(assign_regime)
    df["price_lag_24"] = df["price"].shift(24)
    df["lag24_regime"] = df["target_regime"].shift(24)
    df["transition_label"] = df["lag24_regime"] + " -> " + df["target_regime"]
    df["persistence_error"] = (df["price"] - df["price_lag_24"]).abs()
    df["hour"] = df["ts_hour"].dt.hour
    df["weekday"] = df["ts_hour"].dt.dayofweek
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)

    out = df[
        [
            "ts_hour",
            "price",
            "target_regime",
            "price_lag_24",
            "lag24_regime",
            "transition_label",
            "persistence_error",
            "hour",
            "weekday",
            "is_weekend",
        ]
    ].copy()
    return out


def write_outputs(labels: pd.DataFrame) -> dict[str, Any]:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(OUTPUT_PATH, index=False)

    valid = labels.dropna(subset=["price_lag_24", "transition_label"])
    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input": str(PTF_PATH.relative_to(PROJECT_ROOT)),
        "output": str(OUTPUT_PATH.relative_to(PROJECT_ROOT)),
        "rows": int(len(labels)),
        "valid_lag24_rows": int(len(valid)),
        "coverage_start": labels["ts_hour"].min().isoformat() if not labels.empty else None,
        "coverage_end": labels["ts_hour"].max().isoformat() if not labels.empty else None,
        "regime_counts": labels["target_regime"].value_counts().to_dict(),
        "transition_counts_top20": valid["transition_label"].value_counts().head(20).to_dict(),
        "persistence_error_mae": float(valid["persistence_error"].mean()),
        "persistence_error_median": float(valid["persistence_error"].median()),
        "leakage_policy": {
            "finalized_ptf_usage": "labels_and_evaluation_only",
            "forbidden_downstream_feature_columns": [
                "price",
                "target_regime",
                "transition_label",
                "persistence_error",
            ],
        },
    }

    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# Regime Label Summary",
        "",
        f"Generated: `{summary['generated_at']}`",
        "",
        "Finalized PTF is used here only to create labels and evaluation fields.",
        "These target columns must not enter the forecasting feature matrix.",
        "",
        f"- Rows: `{summary['rows']}`",
        f"- Valid lag24 rows: `{summary['valid_lag24_rows']}`",
        f"- Coverage: `{summary['coverage_start']}` -> `{summary['coverage_end']}`",
        f"- Persistence MAE, lag24: `{summary['persistence_error_mae']:.2f}`",
        f"- Persistence median absolute error, lag24: `{summary['persistence_error_median']:.2f}`",
        "",
        "## Regime Counts",
        "",
        "| Regime | Rows |",
        "|---|---:|",
    ]
    for regime, count in summary["regime_counts"].items():
        lines.append(f"| `{regime}` | {count} |")
    lines.extend(["", "## Top Transition Counts", "", "| Transition | Rows |", "|---|---:|"])
    for transition, count in summary["transition_counts_top20"].items():
        lines.append(f"| `{transition}` | {count} |")
    lines.extend(
        [
            "",
            "## Leakage Policy",
            "",
            "- `price`, `target_regime`, `transition_label`, and `persistence_error` are target/evaluation columns.",
            "- Downstream feature stores must use only anchor-time safe inputs.",
            "- Historical `interim-mcp` oracle data must not be used as a feature.",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines) + "\n")
    return summary


def main() -> None:
    labels = build_labels()
    summary = write_outputs(labels)
    print(f"Wrote {OUTPUT_PATH} rows={summary['rows']}")
    print(f"Wrote {SUMMARY_JSON}")
    print(f"Wrote {SUMMARY_MD}")


if __name__ == "__main__":
    main()
