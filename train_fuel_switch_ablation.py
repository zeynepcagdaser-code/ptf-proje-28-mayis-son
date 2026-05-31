#!/usr/bin/env python3
"""
Measure the effect of fuel-switch / marginality features on the high precision
PTF residual model.

This script intentionally does not replace the production model artifacts. It
trains comparable models with and without the explicit fuel-switch columns and
reports the delta on the same time-based split.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from train_high_precision_ptf_model import (
    PROJECT_ROOT,
    build_feature_matrix,
    choose_prediction_config,
    evaluate_predictions,
    load_dataset,
    predict_residuals,
    train_models,
)

FUEL_SWITCH_COLUMNS = [
    "gas_marginality_proxy",
    "hydro_displacement_score",
    "renewable_share_of_generation",
    "gas_share_of_generation",
    "renewable_minus_gas_shift",
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

MODEL_DIR = PROJECT_ROOT / "models" / "fuel_switch_ablation"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "predictions" / "fuel_switch_ablation_predictions.csv"
REPORT_JSON = PROJECT_ROOT / "reports" / "fuel_switch_ablation_metrics.json"
REPORT_MD = PROJECT_ROOT / "reports" / "fuel_switch_ablation_metrics.md"


def train_scenario(data: pd.DataFrame, scenario_name: str, drop_fuel_switch: bool) -> dict[str, Any]:
    scenario_data = data.copy()
    dropped = [col for col in FUEL_SWITCH_COLUMNS if col in scenario_data.columns]
    if drop_fuel_switch:
        scenario_data = scenario_data.drop(columns=dropped)

    features, feature_cols, forbidden_present = build_feature_matrix(scenario_data)
    models = train_models(scenario_data, features)
    residuals = predict_residuals(models, features)
    selected_config, validation_scores, selected_predictions = choose_prediction_config(
        scenario_data, residuals
    )
    frame = scenario_data.join(selected_predictions)
    pred_cols = [
        "high_precision_pred",
        "persistence_pred",
        "global_residual_pred",
        "regime_soft_pred_no_floor",
        "regime_classifier_routing_only_pred",
    ]
    evaluation = evaluate_predictions(frame, pred_cols, "high_precision_pred")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    scenario_dir = MODEL_DIR / scenario_name
    scenario_dir.mkdir(parents=True, exist_ok=True)
    for model_name, model in models.items():
        joblib.dump(model, scenario_dir / f"{model_name}.joblib")
    (scenario_dir / "feature_columns.json").write_text(
        json.dumps(feature_cols, ensure_ascii=False, indent=2) + "\n"
    )

    pred_out = frame[
        [
            "ts_hour",
            "split",
            "price",
            "target_regime",
            "lag24_regime",
            "transition_label",
            "persistence_error",
            "persistence_pred",
            "high_precision_pred",
            "global_residual_pred",
            "regime_soft_pred_no_floor",
            "regime_classifier_routing_only_pred",
            "selected_prediction_config",
        ]
    ].copy()
    pred_out["scenario"] = scenario_name

    return {
        "scenario": scenario_name,
        "drop_fuel_switch": drop_fuel_switch,
        "dropped_columns": dropped if drop_fuel_switch else [],
        "used_fuel_switch_columns": [col for col in FUEL_SWITCH_COLUMNS if col in data.columns]
        if not drop_fuel_switch
        else [],
        "feature_count": len(feature_cols),
        "forbidden_present": forbidden_present,
        "selected_config": selected_config,
        "validation_scores": validation_scores,
        "evaluation": evaluation,
        "predictions": pred_out,
    }


def extract_summary(result: dict[str, Any]) -> dict[str, Any]:
    test = result["evaluation"]["test"]
    selected = test["high_precision_pred"]
    persistence = test["persistence_pred"]
    detail = test["selected_detail"]
    return {
        "scenario": result["scenario"],
        "feature_count": result["feature_count"],
        "selected_config": result["selected_config"],
        "test_mae": selected["mae"],
        "test_rmse": selected["rmse"],
        "test_median_ae": selected["median_ae"],
        "test_p90_ae": selected["p90_ae"],
        "delta_vs_persistence": selected["mae"] - persistence["mae"],
        "pct_error_le_50": selected["pct_error_le_50"],
        "regime_wise": detail["regime_wise"],
        "cap_miss_penalty": detail["cap_miss_penalty"],
        "persistence_failure_hours": detail["persistence_failure_hours"],
        "delivery_hour_1_4_mae": detail["delivery_hour_1_4_mae"],
    }


def write_report(results: list[dict[str, Any]], predictions: pd.DataFrame) -> None:
    summaries = [extract_summary(result) for result in results]
    by_name = {summary["scenario"]: summary for summary in summaries}
    base = by_name["without_explicit_fuel_switch"]
    full = by_name["with_explicit_fuel_switch"]
    delta = {
        "mae_delta_full_minus_without": full["test_mae"] - base["test_mae"],
        "rmse_delta_full_minus_without": full["test_rmse"] - base["test_rmse"],
        "p90_delta_full_minus_without": full["test_p90_ae"] - base["test_p90_ae"],
        "delta_vs_persistence_change": full["delta_vs_persistence"]
        - base["delta_vs_persistence"],
    }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scenarios": summaries,
        "delta": delta,
        "fuel_switch_columns": FUEL_SWITCH_COLUMNS,
        "notes": [
            "Both scenarios use the same time-based split and model family.",
            "The without scenario drops explicit fuel-switch columns from the feature matrix.",
            "Updated analyst scores remain available in both scenarios, so this isolates direct column contribution rather than the full reasoning-layer contribution.",
        ],
    }

    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    lines = [
        "# Fuel Switch Ablation Metrics",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "This compares the same high precision residual model with and without explicit fuel-switch / marginality columns.",
        "",
        "## Test Summary",
        "",
        "| Scenario | Features | Test MAE | RMSE | Median AE | P90 AE | Delta vs persistence | <=50 TL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            f"| `{summary['scenario']}` | {summary['feature_count']} | {summary['test_mae']:.2f} | {summary['test_rmse']:.2f} | {summary['test_median_ae']:.2f} | {summary['test_p90_ae']:.2f} | {summary['delta_vs_persistence']:.2f} | {summary['pct_error_le_50']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Fuel-Switch Delta",
            "",
            f"- MAE delta, full minus without: `{delta['mae_delta_full_minus_without']:.2f}` TL/MWh",
            f"- RMSE delta, full minus without: `{delta['rmse_delta_full_minus_without']:.2f}` TL/MWh",
            f"- P90 AE delta, full minus without: `{delta['p90_delta_full_minus_without']:.2f}` TL/MWh",
            "",
            "Negative delta means the explicit fuel-switch columns improved the model.",
            "",
            "## Regime-Wise MAE",
            "",
        ]
    )
    for summary in summaries:
        lines.extend(
            [
                f"### `{summary['scenario']}`",
                "",
                "| Regime | Rows | Model MAE | Persistence MAE | Delta |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in sorted(summary["regime_wise"], key=lambda item: item["target_regime"]):
            lines.append(
                f"| `{row['target_regime']}` | {row['rows']} | {row['mae']:.2f} | {row['persistence_mae']:.2f} | {row['delta_vs_persistence']:.2f} |"
            )
        cap = summary["cap_miss_penalty"]
        lines.extend(
            [
                "",
                f"- Cap miss rate: `{cap['cap_miss_rate_pred_below_4000']}`",
                f"- Spike/cap MAE: `{cap['mae_spike_cap']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Columns Tested",
            "",
        ]
    )
    for column in FUEL_SWITCH_COLUMNS:
        lines.append(f"- `{column}`")
    lines.extend(["", "## Notes", ""])
    for note in report["notes"]:
        lines.append(f"- {note}")
    REPORT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    data, _ = load_dataset()
    results = [
        train_scenario(data, "without_explicit_fuel_switch", drop_fuel_switch=True),
        train_scenario(data, "with_explicit_fuel_switch", drop_fuel_switch=False),
    ]
    predictions = pd.concat([result["predictions"] for result in results], ignore_index=True)
    write_report(results, predictions)
    print(f"Wrote {REPORT_JSON}")
    print(f"Wrote {REPORT_MD}")
    print(f"Wrote {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
