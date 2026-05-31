#!/usr/bin/env python3
"""Blend the fuel-switch routed model with the existing high precision model.

The fuel-switch model is good when cheap-supply / zero-pressure mechanics are
active, but it weakens spike/cap hours. This script learns a validation-selected
operating rule that uses fuel-switch predictions only when the fuel-switch gates
are strong enough, and falls back to the high precision forecaster otherwise.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parent
FUEL_PATH = PROJECT_ROOT / "data" / "predictions" / "fuel_switch_routed_ptf_predictions.csv"
HIGH_PRECISION_PATH = PROJECT_ROOT / "data" / "predictions" / "high_precision_ptf_predictions.csv"
OUT_PATH = PROJECT_ROOT / "data" / "predictions" / "hybrid_fuel_switch_ptf_predictions.csv"
REPORT_JSON = PROJECT_ROOT / "reports" / "hybrid_fuel_switch_ptf_metrics.json"
REPORT_MD = PROJECT_ROOT / "reports" / "hybrid_fuel_switch_ptf_metrics.md"


def abs_error(y: pd.Series, pred: pd.Series) -> pd.Series:
    return (y.astype(float) - pred.astype(float)).abs()


def metric_block(y: pd.Series, pred: pd.Series) -> dict[str, Any]:
    valid = y.notna() & pred.notna()
    y = y.loc[valid]
    pred = pred.loc[valid]
    err = abs_error(y, pred)
    return {
        "rows": int(len(err)),
        "mae": float(err.mean()) if len(err) else None,
        "rmse": float(math.sqrt(mean_squared_error(y, pred))) if len(err) else None,
        "median_ae": float(err.median()) if len(err) else None,
        "p90_ae": float(err.quantile(0.90)) if len(err) else None,
        "pct_error_le_2": float((err <= 2).mean()) if len(err) else None,
        "pct_error_le_10": float((err <= 10).mean()) if len(err) else None,
        "pct_error_le_50": float((err <= 50).mean()) if len(err) else None,
    }


def grouped_mae(frame: pd.DataFrame, group_col: str, pred_col: str) -> list[dict[str, Any]]:
    rows = []
    for key, group in frame.groupby(group_col, dropna=False, observed=False):
        rows.append(
            {
                group_col: str(key),
                "rows": int(len(group)),
                "mae": float(abs_error(group["price"], group[pred_col]).mean()),
                "persistence_mae": float(abs_error(group["price"], group["persistence_pred"]).mean()),
                "high_precision_mae": float(abs_error(group["price"], group["high_precision_pred"]).mean()),
                "fuel_switch_mae": float(abs_error(group["price"], group["fuel_switch_pred"]).mean()),
            }
        )
    return sorted(rows, key=lambda item: item["mae"], reverse=True)


def load_inputs() -> pd.DataFrame:
    fuel = pd.read_csv(FUEL_PATH)
    high = pd.read_csv(HIGH_PRECISION_PATH)
    for frame in [fuel, high]:
        frame["ts_hour"] = pd.to_datetime(frame["ts_hour"], errors="coerce")
    high_keep = high[
        [
            "ts_hour",
            "high_precision_pred",
            "binary_spike_probability",
            "spike_transition_probability",
            "cap_floor_risk",
        ]
    ]
    data = fuel.merge(high_keep, on="ts_hour", how="inner")
    return data.sort_values("ts_hour").reset_index(drop=True)


def build_prediction(frame: pd.DataFrame, params: dict[str, float]) -> pd.Series:
    zero_signal = np.maximum(
        frame["zero_rule_gate"].fillna(0),
        frame["zero_pressure_state_prob"].fillna(0) / max(params["zero_prob_scale"], 1e-6),
    ).clip(0, 1)
    cheap_signal = frame["cheap_supply_gate"].fillna(0).clip(0, 1)
    spike_signal = np.maximum(
        frame["binary_spike_probability"].fillna(0),
        frame["spike_transition_probability"].fillna(0) * params["transition_scale"],
    ).clip(0, 1)
    gas_signal = frame["gas_rule_gate"].fillna(0).clip(0, 1)

    use_fuel = (
        ((zero_signal >= params["zero_threshold"]) | (cheap_signal >= params["cheap_threshold"]))
        & (spike_signal < params["spike_block_threshold"])
        & (gas_signal < params["gas_block_threshold"])
        & (frame["high_price_state_prob"].fillna(0) < params["high_price_block_threshold"])
        & (frame["cap_floor_risk"].fillna(0) < params["cap_risk_block_threshold"])
    )
    blend = np.where(use_fuel, params["fuel_weight"], 0.0)
    pred = blend * frame["fuel_switch_pred"] + (1 - blend) * frame["high_precision_pred"]
    return pd.Series(np.clip(pred, 0, 5000), index=frame.index)


def tune(frame: pd.DataFrame) -> tuple[dict[str, float], list[dict[str, Any]]]:
    candidates = []
    for zero_threshold in [0.55, 0.65, 0.75, 0.85, 0.95]:
        for cheap_threshold in [0.45, 0.55, 0.65, 0.75]:
            for fuel_weight in [0.35, 0.5, 0.65, 0.8, 1.0]:
                for spike_block_threshold in [0.001, 0.005, 0.02, 0.08, 0.20]:
                    for gas_block_threshold in [0.12, 0.18, 0.25, 0.40, 0.82]:
                        for high_price_block_threshold in [0.45, 0.65, 0.85, 0.98]:
                            for cap_risk_block_threshold in [0.001, 0.005, 0.02, 0.08, 0.20]:
                                candidates.append(
                                    {
                                        "zero_threshold": zero_threshold,
                                        "cheap_threshold": cheap_threshold,
                                        "fuel_weight": fuel_weight,
                                        "spike_block_threshold": spike_block_threshold,
                                        "gas_block_threshold": gas_block_threshold,
                                        "high_price_block_threshold": high_price_block_threshold,
                                        "cap_risk_block_threshold": cap_risk_block_threshold,
                                        "zero_prob_scale": 0.10,
                                        "transition_scale": 10.0,
                                    }
                                )
    records = []
    validation = frame[frame["split"] == "validation"].copy()
    best = candidates[0]
    best_score = float("inf")
    for params in candidates:
        pred = build_prediction(validation, params)
        mae = float(abs_error(validation["price"], pred).mean())
        zero = validation[validation["price"] <= 50]
        zero_mae = float(abs_error(zero["price"], pred.loc[zero.index]).mean()) if len(zero) else None
        # Keep a small zero-pressure bonus without letting validation overfit
        # exclusively to rare zero events.
        score = mae + (zero_mae or 0) * 0.02
        records.append({"params": params, "validation_mae": mae, "validation_zero_mae": zero_mae, "score": score})
        if score < best_score:
            best_score = score
            best = params
    return best, sorted(records, key=lambda item: item["score"])[:20]


def evaluate(frame: pd.DataFrame, params: dict[str, float]) -> dict[str, Any]:
    frame = frame.copy()
    frame["hybrid_pred"] = build_prediction(frame, params)
    metrics: dict[str, Any] = {"params": params}
    for split in ["validation", "test"]:
        subset = frame[frame["split"] == split].copy()
        metrics[split] = {
            "hybrid_pred": metric_block(subset["price"], subset["hybrid_pred"]),
            "high_precision_pred": metric_block(subset["price"], subset["high_precision_pred"]),
            "fuel_switch_pred": metric_block(subset["price"], subset["fuel_switch_pred"]),
            "persistence_pred": metric_block(subset["price"], subset["persistence_pred"]),
            "regime_wise": grouped_mae(subset, "target_regime", "hybrid_pred"),
            "fuel_usage_rate": float((subset["hybrid_pred"].round(8) != subset["high_precision_pred"].round(8)).mean()),
        }
        zero = subset[subset["price"] <= 50]
        spike = subset[subset["price"] >= 4000]
        metrics[split]["zero_price_hours"] = metric_block(zero["price"], zero["hybrid_pred"]) if len(zero) else None
        metrics[split]["spike_cap_hours"] = metric_block(spike["price"], spike["hybrid_pred"]) if len(spike) else None
    return metrics, frame


def write_outputs(frame: pd.DataFrame, metrics: dict[str, Any], top_candidates: list[dict[str, Any]]) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT_PATH, index=False)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "predictions": str(OUT_PATH.relative_to(PROJECT_ROOT)),
        "inputs": {
            "fuel_switch": str(FUEL_PATH.relative_to(PROJECT_ROOT)),
            "high_precision": str(HIGH_PRECISION_PATH.relative_to(PROJECT_ROOT)),
        },
        "top_validation_candidates": top_candidates,
        "metrics": metrics,
        "mechanism": "Fuel-switch is used only when zero/cheap-supply signals are high and spike/gas-risk blockers are low.",
    }
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    test = metrics["test"]
    rows = [
        ("hybrid_pred", test["hybrid_pred"]),
        ("high_precision_pred", test["high_precision_pred"]),
        ("fuel_switch_pred", test["fuel_switch_pred"]),
        ("persistence_pred", test["persistence_pred"]),
    ]
    lines = [
        "# Hybrid Fuel Switch PTF Metrics",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "This hybrid uses the fuel-switch model when the zero/cheap-supply mechanism is active and falls back to the high precision model when spike/gas risk is present.",
        "",
        "## Test Summary",
        "",
        "| Model | MAE | RMSE | Median AE | P90 AE | <=2 TL | <=10 TL | <=50 TL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in rows:
        lines.append(
            f"| `{name}` | {item['mae']:.2f} | {item['rmse']:.2f} | {item['median_ae']:.2f} | {item['p90_ae']:.2f} | {item['pct_error_le_2']:.4f} | {item['pct_error_le_10']:.4f} | {item['pct_error_le_50']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"- Delta vs high precision: `{test['hybrid_pred']['mae'] - test['high_precision_pred']['mae']:.2f}` TL/MWh",
            f"- Delta vs persistence: `{test['hybrid_pred']['mae'] - test['persistence_pred']['mae']:.2f}` TL/MWh",
            f"- Fuel-switch usage rate on test: `{test['fuel_usage_rate']:.4f}`",
            "",
            "## Regime-Wise Test MAE",
            "",
            "| Regime | Rows | Hybrid MAE | Persistence MAE | High precision MAE | Fuel-switch MAE |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(test["regime_wise"], key=lambda item: item["target_regime"]):
        lines.append(
            f"| `{row['target_regime']}` | {row['rows']} | {row['mae']:.2f} | {row['persistence_mae']:.2f} | {row['high_precision_mae']:.2f} | {row['fuel_switch_mae']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Selected Parameters",
            "",
            "```json",
            json.dumps(metrics["params"], indent=2),
            "```",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    frame = load_inputs()
    params, top_candidates = tune(frame)
    metrics, final_frame = evaluate(frame, params)
    write_outputs(final_frame, metrics, top_candidates)
    print(f"Wrote {OUT_PATH}")
    print(f"Wrote {REPORT_JSON}")
    print(f"Wrote {REPORT_MD}")


if __name__ == "__main__":
    main()
