#!/usr/bin/env python3
"""
Build tomorrow morning feature rows without training.

This script synthesizes 00:00-12:00 tomorrow feature rows from leakage-safe
lagged history and available forecast tables. It is intentionally conservative:
if a future source is missing, it falls back to lagged same-hour values and
documents the fallback in the report.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
FEATURE_STORE_PATH = PROJECT_ROOT / "data" / "features" / "regime_feature_store.parquet"
MUST_RUN_PATH = PROJECT_ROOT / "data" / "features" / "must_run_supply_features.parquet"
LOAD_FORECAST_PATH = PROJECT_ROOT / "data" / "load_forecast.csv"
REGIME_LABELS_PATH = PROJECT_ROOT / "data" / "regime_labels.csv"
REGIME_PRED_PATH = PROJECT_ROOT / "data" / "predictions" / "regime_classifier_predictions.csv"
SPIKE_PRED_PATH = PROJECT_ROOT / "data" / "predictions" / "spike_cap_detector_predictions.csv"
TRANSITION_PRED_PATH = PROJECT_ROOT / "data" / "predictions" / "spike_transition_detector_predictions.csv"

OUT_PATH = PROJECT_ROOT / "data" / "features" / "tomorrow_morning_features.parquet"
REPORT_PATH = PROJECT_ROOT / "reports" / "tomorrow_morning_feature_builder.md"
REPORT_JSON = PROJECT_ROOT / "reports" / "tomorrow_morning_feature_builder.json"


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def load_inputs() -> tuple[pd.DataFrame, dict[str, Any]]:
    feature_store = pd.read_parquet(FEATURE_STORE_PATH)
    feature_store["ts_hour"] = pd.to_datetime(feature_store["ts_hour"], errors="coerce")

    labels = _read_csv(REGIME_LABELS_PATH)
    if not labels.empty:
        labels["ts_hour"] = pd.to_datetime(labels["ts_hour"], errors="coerce")

    load_forecast = _read_csv(LOAD_FORECAST_PATH)
    if not load_forecast.empty:
        load_forecast["ts_hour"] = pd.to_datetime(load_forecast["date"], errors="coerce").dt.tz_localize(None)
        load_forecast["load_forecast"] = pd.to_numeric(load_forecast["lep"], errors="coerce")
        load_forecast = load_forecast[["ts_hour", "load_forecast"]]

    must_run = pd.read_parquet(MUST_RUN_PATH) if MUST_RUN_PATH.exists() else pd.DataFrame()
    if not must_run.empty:
        must_run["delivery_hour"] = pd.to_datetime(must_run["delivery_hour"], errors="coerce")

    predictions = {}
    for name, path, col in [
        ("regime_classifier", REGIME_PRED_PATH, None),
        ("spike_detector", SPIKE_PRED_PATH, None),
        ("spike_transition", TRANSITION_PRED_PATH, None),
    ]:
        frame = _read_csv(path)
        if not frame.empty and "ts_hour" in frame.columns:
            frame["ts_hour"] = pd.to_datetime(frame["ts_hour"], errors="coerce")
        predictions[name] = frame

    metadata = {
        "feature_store_max_ts": str(feature_store["ts_hour"].max()),
        "load_forecast_max_ts": str(load_forecast["ts_hour"].max()) if not load_forecast.empty else None,
        "regime_labels_max_ts": str(labels["ts_hour"].max()) if not labels.empty else None,
        "must_run_rows": int(len(must_run)),
    }
    return (feature_store, labels, load_forecast, must_run, predictions, metadata)


def tomorrow_hours() -> list[pd.Timestamp]:
    today = datetime.now().date()
    tomorrow = today + timedelta(days=1)
    return [pd.Timestamp(datetime.combine(tomorrow, datetime.min.time()) + timedelta(hours=h)) for h in range(13)]


def build_rows(
    feature_store: pd.DataFrame,
    labels: pd.DataFrame,
    load_forecast: pd.DataFrame,
    must_run: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    hours = tomorrow_hours()
    rows = []
    diagnostics = {
        "requested_rows": len(hours),
        "produced_rows": 0,
        "missing_reason": None,
        "fallbacks": {},
    }

    if feature_store.empty or labels.empty:
        diagnostics["missing_reason"] = "feature_store or labels missing"
        return pd.DataFrame(), diagnostics

    for ts in hours:
        lag = ts - timedelta(hours=24)
        hist = feature_store.loc[feature_store["ts_hour"] == lag]
        if hist.empty:
            continue
        base = hist.iloc[0].to_dict()
        base["ts_hour"] = ts

        if not load_forecast.empty and ts in set(load_forecast["ts_hour"]):
            load_row = load_forecast.loc[load_forecast["ts_hour"] == ts].iloc[0]
            base["load_forecast"] = float(load_row["load_forecast"])
            diagnostics["fallbacks"].setdefault("load_forecast", 0)
        elif not load_forecast.empty and lag in set(load_forecast["ts_hour"]):
            load_row = load_forecast.loc[load_forecast["ts_hour"] == lag].iloc[0]
            base["load_forecast"] = float(load_row["load_forecast"])
            diagnostics["fallbacks"]["load_forecast"] = diagnostics["fallbacks"].get("load_forecast", 0) + 1

        if not must_run.empty and ts in set(must_run["delivery_hour"]):
            must_row = must_run.loc[must_run["delivery_hour"] == ts].iloc[0]
            base["must_run_supply"] = float(must_row.get("must_run_supply", np.nan))
            base["must_run_share"] = float(must_row.get("must_run_share", np.nan))
            base["residual_load_after_must_run"] = float(must_row.get("residual_load_after_must_run", np.nan))
        else:
            base["must_run_supply"] = float(base.get("kgup_renewable_mw", 0.0) or 0.0)
            base["must_run_share"] = (
                base["must_run_supply"] / base["load_forecast"] if base.get("load_forecast") else np.nan
            )
            base["residual_load_after_must_run"] = (
                base["load_forecast"] - base["must_run_supply"] if base.get("load_forecast") else np.nan
            )
            diagnostics["fallbacks"]["must_run_supply"] = diagnostics["fallbacks"].get("must_run_supply", 0) + 1

        lag_label = labels.loc[labels["ts_hour"] == lag]
        if not lag_label.empty:
            base["lag24_ptf"] = float(lag_label.iloc[0]["price"])
            base["lag24_regime"] = lag_label.iloc[0]["target_regime"]
        else:
            base["lag24_ptf"] = float(base.get("ptf_lag_24", np.nan))
            base["lag24_regime"] = base.get("price_band_lag_24", None)

        if "hour" not in base or pd.isna(base.get("hour")):
            base["hour"] = ts.hour

        for name, pred in predictions.items():
            if pred.empty or "ts_hour" not in pred.columns:
                continue
            pred_row = pred.loc[pred["ts_hour"] == lag]
            if pred_row.empty:
                continue
            row = pred_row.iloc[0]
            if name == "regime_classifier":
                for col in ["prob_negative_zero_pressure", "prob_normal", "prob_tight", "prob_spike_cap"]:
                    if col in row:
                        base[col] = row[col]
            elif name == "spike_detector":
                if "spike_probability" in row:
                    base["binary_spike_probability"] = row["spike_probability"]
            elif name == "spike_transition":
                if "new_spike_probability" in row:
                    base["spike_transition_probability"] = row["new_spike_probability"]

        rows.append(base)

    out = pd.DataFrame(rows)
    diagnostics["produced_rows"] = int(len(out))
    if out.empty:
        diagnostics["missing_reason"] = "No lag-24 aligned feature rows for tomorrow morning"
        return out, diagnostics

    if "binary_spike_probability" not in out.columns:
        out["binary_spike_probability"] = 0.0
    if "spike_transition_probability" not in out.columns:
        out["spike_transition_probability"] = 0.0

    out["is_tomorrow_morning"] = 1
    out = out.sort_values("ts_hour")
    return out, diagnostics


def write_reports(out: pd.DataFrame, diag: dict[str, Any], metadata: dict[str, Any]) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    REPORT_JSON.write_text(json.dumps({"diagnostics": diag, "metadata": metadata}, ensure_ascii=False, indent=2) + "\n")
    report = [
        "# Tomorrow Morning Feature Builder",
        "",
        f"- Requested rows: `{diag['requested_rows']}`",
        f"- Produced rows: `{diag['produced_rows']}`",
        f"- Feature store max ts: `{metadata['feature_store_max_ts']}`",
        f"- Load forecast max ts: `{metadata['load_forecast_max_ts']}`",
        f"- Regime labels max ts: `{metadata['regime_labels_max_ts']}`",
        f"- Must-run rows: `{metadata['must_run_rows']}`",
        "",
        "## Why 0 rows happened before",
        "",
        "The earlier builder only looked for an exact tomorrow-date match inside historical tables. Those tables stop at the latest observed day, so tomorrow had no direct rows to copy.",
        "",
        "## Fallbacks",
        "",
    ]
    if diag["fallbacks"]:
        for key, count in sorted(diag["fallbacks"].items()):
            report.append(f"- `{key}` fallback used `{count}` times")
    else:
        report.append("- None")
    report.extend(
        [
            "",
            "## Notes",
            "",
            "The table is leakage-safe because it only uses lag-24 labels and available forecast/history rows.",
        ]
    )
    REPORT_PATH.write_text("\n".join(report) + "\n")


def main() -> None:
    feature_store, labels, load_forecast, must_run, predictions, metadata = load_inputs()
    out, diag = build_rows(feature_store, labels, load_forecast, must_run, predictions)
    write_reports(out, diag, metadata)
    print(f"Wrote {OUT_PATH}")
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {REPORT_JSON}")
    print(f"Rows: {len(out)}")


if __name__ == "__main__":
    main()
