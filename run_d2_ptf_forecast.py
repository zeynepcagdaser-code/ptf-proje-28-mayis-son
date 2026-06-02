#!/usr/bin/env python3
"""Run D+2 PTF forecast for target delivery date (default: today + 2)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
FEATURE_PATH = PROJECT_ROOT / "data" / "features" / "d2_ptf_features.parquet"
MODEL_DIR = PROJECT_ROOT / "models" / "d2_ptf_forecaster"
HP_MODEL_DIR = PROJECT_ROOT / "models" / "high_precision_ptf_model"

OUT_CSV = PROJECT_ROOT / "data" / "predictions" / "d2_ptf_forecast.csv"
REPORT_JSON = PROJECT_ROOT / "reports" / "d2_ptf_forecast_run.json"
REPORT_MD = PROJECT_ROOT / "reports" / "d2_ptf_forecast_run.md"


def load_models() -> tuple[object, dict, dict[int, float], list[str]]:
    global_model = joblib.load(MODEL_DIR / "global_model.joblib")
    hour_models = joblib.load(MODEL_DIR / "hour_models.joblib")
    hour_alphas = json.loads((MODEL_DIR / "hour_alphas.json").read_text())
    hour_alphas = {int(k): float(v) for k, v in hour_alphas.items()}
    feature_cols = json.loads((MODEL_DIR / "feature_columns.json").read_text())
    return global_model, hour_models, hour_alphas, feature_cols


def predict_residual(
    frame: pd.DataFrame,
    global_model: object,
    hour_models: dict,
    feature_cols: list[str],
) -> np.ndarray:
    x = pd.DataFrame(index=frame.index)
    for col in feature_cols:
        x[col] = frame[col] if col in frame.columns else 0.0
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    global_res = global_model.predict(x)
    blended = global_res.copy()
    for hour, model in hour_models.items():
        mask = frame["hour"].astype(int).to_numpy() == hour
        if mask.any():
            blended[mask] = 0.35 * global_res[mask] + 0.65 * model.predict(x.loc[mask])
    return blended


def predict(frame: pd.DataFrame) -> pd.DataFrame:
    global_model, hour_models, hour_alphas, feature_cols = load_models()
    out = frame.copy()
    anchor = out["anchor_d1_ptf"].to_numpy(float)
    raw = predict_residual(out, global_model, hour_models, feature_cols)
    hours = out["hour"].astype(int).to_numpy()
    alpha = np.array([hour_alphas.get(int(h), 0.5) for h in hours], dtype=float)
    out["persistence_pred"] = anchor
    out["model_residual"] = raw
    out["predicted_ptf"] = np.clip(anchor + alpha * raw, 0, 5000)
    out["hour_alpha"] = alpha
    out["abs_diff_vs_persistence"] = np.abs(out["predicted_ptf"] - out["persistence_pred"])
    return out.sort_values("ts_hour")


def write_reports(pred: pd.DataFrame, target_date: str) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pred[
        [
            "ts_hour",
            "hour",
            "anchor_d1_ptf",
            "persistence_pred",
            "predicted_ptf",
            "model_residual",
            "hour_alpha",
            "abs_diff_vs_persistence",
        ]
    ].to_csv(OUT_CSV, index=False)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_date": target_date,
        "rows": int(len(pred)),
        "mean_predicted_ptf": float(pred["predicted_ptf"].mean()),
        "min_predicted_ptf": float(pred["predicted_ptf"].min()),
        "max_predicted_ptf": float(pred["predicted_ptf"].max()),
        "hours": pred[["ts_hour", "predicted_ptf", "persistence_pred"]].to_dict(orient="records"),
    }
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n")
    lines = [
        "# D+2 PTF Forecast",
        "",
        f"Generated: `{report['generated_at']}`",
        f"Target delivery date: `{target_date}`",
        "",
        f"- Hours forecasted: `{report['rows']}`",
        f"- Mean PTF: `{report['mean_predicted_ptf']:.2f}` TL/MWh",
        f"- Range: `{report['min_predicted_ptf']:.2f}` – `{report['max_predicted_ptf']:.2f}` TL/MWh",
        "",
        "## Hourly Forecast",
        "",
        "| Hour | Persistence (D+1) | Predicted PTF |",
        "| --- | ---: | ---: |",
    ]
    for row in pred.itertuples():
        lines.append(f"| {row.ts_hour} | {row.persistence_pred:.2f} | {row.predicted_ptf:.2f} |")
    REPORT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-date", default=None)
    args = parser.parse_args()
    target_date = args.target_date or (datetime.now().date() + timedelta(days=2)).isoformat()
    target_date_obj = pd.to_datetime(target_date).date()

    if not FEATURE_PATH.exists():
        raise FileNotFoundError(f"Missing {FEATURE_PATH}. Run build_d2_ptf_features.py first.")
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Missing {MODEL_DIR}. Run train_d2_ptf_forecaster.py first.")

    from src.utils.io_utils import read_parquet_with_normalized_ts
    frame = read_parquet_with_normalized_ts(FEATURE_PATH)
    frame["ts_hour"] = pd.to_datetime(frame["ts_hour"], errors="coerce")
    frame = frame[frame["ts_hour"].dt.date == target_date_obj].copy()
    if frame.empty:
        raise ValueError(f"No feature rows found for target date {target_date_obj}")
    pred = predict(frame)
    write_reports(pred, target_date)
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {REPORT_MD}")


if __name__ == "__main__":
    main()
