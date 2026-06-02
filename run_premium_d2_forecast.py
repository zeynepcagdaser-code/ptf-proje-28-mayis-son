#!/usr/bin/env python3
"""Run premium D+2 PTF forecast (plant must-run + curve + hybrid proprietary stack)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from premium_d2_inference import predict_premium

PROJECT_ROOT = Path(__file__).resolve().parent
FEATURE_PATH = PROJECT_ROOT / "data" / "features" / "premium_d2_features.parquet"
OUT_CSV = PROJECT_ROOT / "data" / "predictions" / "premium_d2_forecast.csv"
REPORT_JSON = PROJECT_ROOT / "reports" / "premium_d2_forecast_run.json"
REPORT_MD = PROJECT_ROOT / "reports" / "premium_d2_forecast_run.md"


def write_reports(pred: pd.DataFrame, target_date: str) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(OUT_CSV, index=False)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_date": target_date,
        "rows": int(len(pred)),
        "mean_premium_pred": float(pred["premium_pred"].mean()),
        "min_premium_pred": float(pred["premium_pred"].min()),
        "max_premium_pred": float(pred["premium_pred"].max()),
        "stack": ["high_precision", "fuel_switch_hybrid", "d2_forecaster", "must_run_adjustment", "curve_adjustment"],
    }
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n")
    lines = [
        "# Premium D+2 PTF Forecast",
        "",
        f"Generated: `{report['generated_at']}`",
        f"Target: `{target_date}`",
        "",
        f"- Mean premium PTF: `{report['mean_premium_pred']:.2f}` TL/MWh",
        f"- Range: `{report['min_premium_pred']:.2f}` – `{report['max_premium_pred']:.2f}` TL/MWh",
        "",
        "| Hour | Persistence | Hybrid | D2 | **Premium** |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in pred.itertuples():
        lines.append(
            f"| {row.ts_hour} | {row.persistence_pred:.2f} | {row.hybrid_pred:.2f} | {row.d2_pred:.2f} | **{row.premium_pred:.2f}** |"
        )
    REPORT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-date", default=None)
    args = parser.parse_args()
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(f"Missing {FEATURE_PATH}. Run build_premium_d2_features.py first.")

    from src.utils.io_utils import read_parquet_with_normalized_ts
    frame = read_parquet_with_normalized_ts(FEATURE_PATH)
    pred = predict_premium(frame)
    target_date = args.target_date or str(pred["ts_hour"].dt.date.iloc[0])
    write_reports(pred, target_date)
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {REPORT_MD}")


if __name__ == "__main__":
    main()
