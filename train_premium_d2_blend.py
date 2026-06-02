#!/usr/bin/env python3
"""Tune premium D+2 blend weights on 2025 validation using the full model stack."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from premium_d2_inference import (
    apply_curve_cap_nudge,
    apply_must_run_zero_pull,
    load_blend_weights,
    predict_d2,
    predict_fuel_switch,
    predict_high_precision,
)
from build_hybrid_fuel_switch_forecast import build_prediction as hybrid_build_prediction
from premium_d2_inference import load_hybrid_params
from train_d2_ptf_forecaster import load_training_frame

PROJECT_ROOT = Path(__file__).resolve().parent
OUT_DIR = PROJECT_ROOT / "models" / "premium_d2_forecaster"
REPORT_JSON = PROJECT_ROOT / "reports" / "premium_d2_blend_metrics.json"


def build_validation_sample(max_rows: int = 168) -> pd.DataFrame:
    data = load_training_frame()
    val = data[data["split"] == "validation"].copy()
    val = val.rename(columns={"price": "actual_price"})
    val["anchor_d1_ptf"] = val["ptf_lag_24"]
    val["anchor_source"] = "historical_validation"
    if len(val) > max_rows:
        val = val.iloc[-max_rows:].copy()
    return val


def score_weights(frame: pd.DataFrame, weights: dict[str, float]) -> float:
    hp = predict_high_precision(frame)
    fs = predict_fuel_switch(frame)
    merged = hp.merge(
        fs[["fuel_switch_pred", "zero_rule_gate", "cheap_supply_gate", "gas_rule_gate", "zero_pressure_state_prob", "high_price_state_prob"]],
        left_index=True,
        right_index=True,
        how="left",
    )
    hybrid = hybrid_build_prediction(merged, load_hybrid_params())
    d2 = predict_d2(frame)
    persistence = merged["base_pred"].to_numpy(float)
    pred = (
        weights["hybrid_weight"] * hybrid.to_numpy(float)
        + weights["d2_weight"] * d2.to_numpy(float)
        + weights["persistence_weight"] * persistence
    )
    pred = apply_must_run_zero_pull(pred, frame, weights.get("must_run_zero_pull", 0.15))
    pred = apply_curve_cap_nudge(pred, frame, weights.get("curve_cap_nudge", 0.10))
    pred = np.clip(pred, 0, 5000)
    return float(mean_absolute_error(frame["actual_price"], pred))


def main() -> None:
    frame = build_validation_sample()
    best = load_blend_weights()
    best_mae = float("inf")
    records = []
    for hybrid_w in [0.25, 0.35, 0.45, 0.55]:
        for d2_w in [0.20, 0.30, 0.40, 0.50]:
            pers_w = max(0.0, 1.0 - hybrid_w - d2_w)
            if pers_w > 0.35:
                continue
            for zero_pull in [0.0, 0.10, 0.20]:
                for curve_nudge in [0.0, 0.10, 0.20]:
                    weights = {
                        "hybrid_weight": hybrid_w,
                        "d2_weight": d2_w,
                        "persistence_weight": pers_w,
                        "must_run_zero_pull": zero_pull,
                        "curve_cap_nudge": curve_nudge,
                    }
                    mae = score_weights(frame, weights)
                    records.append({"weights": weights, "validation_mae": mae})
                    if mae < best_mae:
                        best_mae = mae
                        best = weights

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "blend_weights.json").write_text(json.dumps(best, indent=2) + "\n")
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "validation_rows": int(len(frame)),
        "best_weights": best,
        "best_validation_mae": best_mae,
        "top_candidates": sorted(records, key=lambda r: r["validation_mae"])[:10],
    }
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Best validation MAE: {best_mae:.2f} TL")
    print(f"Wrote {OUT_DIR / 'blend_weights.json'}")


if __name__ == "__main__":
    main()
