#!/usr/bin/env python3
"""
Naive seasonal persistence baseline for 24h-ahead PTF.

For delivery at t+h: prediction = PTF at (t+h-24) — same clock hour, previous day.
Uses test split only; real TL/MWh; no scaler.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "lstm_next24_v1.parquet"
ANCHOR_TEST_PATH = PROJECT_ROOT / "data" / "model" / "anchor_test.csv"
MASTER_PATH = PROJECT_ROOT / "data" / "master" / "master_hourly_v1.parquet"
LSTM_METRICS_PATH = PROJECT_ROOT / "reports" / "lstm_baseline_metrics.json"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "predictions" / "persistence_predictions.csv"
METRICS_JSON = PROJECT_ROOT / "reports" / "persistence_metrics.json"
METRICS_MD = PROJECT_ROOT / "reports" / "persistence_metrics.md"

TARGET_HORIZONS = list(range(1, 25))
MAPE_MASK_THRESHOLD = 100.0


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mape(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
    mask = np.abs(a) > eps
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((a[mask] - b[mask]) / a[mask])) * 100)


def masked_mape(
    actual: np.ndarray,
    pred: np.ndarray,
    threshold: float = MAPE_MASK_THRESHOLD,
) -> float:
    mask = np.abs(actual) > threshold
    if not mask.any():
        return float("nan")
    return mape(actual[mask], pred[mask])


def horizon_mae_df(df: pd.DataFrame) -> dict[int, float]:
    grouped = df.groupby("target_hour")["absolute_error"].mean()
    return {int(h): float(v) for h, v in grouped.items()}


def load_ptf_lookup() -> pd.Series:
    from src.utils.io_utils import read_parquet_with_normalized_ts
    master = read_parquet_with_normalized_ts(MASTER_PATH, columns=["ts_hour", "ptf_price"])
    master = master.sort_values("ts_hour")
    ts = pd.to_datetime(master["ts_hour"], utc=True).dt.tz_convert("Europe/Istanbul")
    return pd.Series(master["ptf_price"].values, index=ts)


def build_persistence_predictions(
    test_df: pd.DataFrame,
    ptf_lookup: pd.Series,
) -> pd.DataFrame:
    target_cols = [f"target_{h}h" for h in TARGET_HORIZONS]
    rows: list[dict] = []

    for _, row in test_df.iterrows():
        anchor = pd.Timestamp(row["ts_hour"])
        if anchor.tzinfo is None:
            anchor = anchor.tz_localize("Europe/Istanbul")
        else:
            anchor = anchor.tz_convert("Europe/Istanbul")

        for h, tcol in zip(TARGET_HORIZONS, target_cols):
            actual = float(row[tcol])
            # PTF at delivery (t+h), lagged 24h => PTF at t+h-24
            pred_ts = anchor + pd.Timedelta(hours=h) - pd.Timedelta(hours=24)
            predicted = ptf_lookup.get(pred_ts, np.nan)

            rows.append(
                {
                    "anchor_ts_hour": str(anchor),
                    "target_hour": h,
                    "actual_price": actual,
                    "predicted_price": predicted,
                    "absolute_error": abs(actual - predicted)
                    if pd.notna(predicted)
                    else np.nan,
                }
            )

    return pd.DataFrame(rows)


def compute_metrics(df: pd.DataFrame) -> dict:
    valid = df.dropna(subset=["predicted_price", "actual_price"])
    actual = valid["actual_price"].to_numpy()
    pred = valid["predicted_price"].to_numpy()
    errors = valid["absolute_error"].to_numpy()

    zero_mask = actual == 0
    h_mae = horizon_mae_df(valid)

    return {
        "test_samples_anchors": int(df["anchor_ts_hour"].nunique()),
        "aligned_with_lstm_anchors": ANCHOR_TEST_PATH.exists(),
        "prediction_rows": int(len(df)),
        "valid_prediction_rows": int(len(valid)),
        "dropped_missing_lookup": int(len(df) - len(valid)),
        "mae": mae(actual, pred),
        "rmse": rmse(actual, pred),
        "mape": mape(actual, pred),
        "masked_mape_actual_gt_100": masked_mape(actual, pred),
        "horizon_mae": {str(k): v for k, v in h_mae.items()},
        "worst_horizon": max(h_mae, key=h_mae.get) if h_mae else None,
        "worst_horizon_mae": max(h_mae.values()) if h_mae else None,
        "zero_price_mae": mae(actual[zero_mask], pred[zero_mask])
        if zero_mask.any()
        else None,
        "zero_price_hours": int(zero_mask.sum()),
    }


def compare_with_lstm(persistence: dict) -> dict | None:
    if not LSTM_METRICS_PATH.exists():
        return None

    lstm = json.loads(LSTM_METRICS_PATH.read_text(encoding="utf-8"))
    p_mae = persistence["mae"]
    l_mae = lstm["test_mae"]
    delta = l_mae - p_mae
    pct = (delta / p_mae * 100) if p_mae else float("nan")

    return {
        "lstm_mae": l_mae,
        "persistence_mae": p_mae,
        "mae_delta_lstm_minus_persistence": delta,
        "mae_pct_change_vs_persistence": pct,
        "lstm_better_than_persistence": l_mae < p_mae,
        "interpretation": (
            "LSTM beats persistence baseline"
            if l_mae < p_mae
            else "Persistence baseline beats or matches LSTM"
        ),
        "lstm_rmse": lstm.get("test_rmse"),
        "persistence_rmse": persistence["rmse"],
        "rmse_delta_lstm_minus_persistence": lstm.get("test_rmse", 0) - persistence["rmse"],
        "lstm_masked_mape_unavailable": "masked MAPE not in LSTM report",
        "persistence_masked_mape_actual_gt_100": persistence["masked_mape_actual_gt_100"],
    }


def write_markdown(report: dict) -> str:
    lines = [
        "# Persistence Baseline Metrics",
        "",
        "Naive seasonal rule: `prediction(t+h) = PTF(t+h-24)` (same clock hour, previous day).",
        "",
        f"- Test anchors: {report['test_samples_anchors']}",
        f"- Prediction rows: {report['prediction_rows']}",
        f"- Valid rows: {report['valid_prediction_rows']}",
        "",
        "## Metrics (TL/MWh)",
        "",
        f"- MAE: {report['mae']:.4f}",
        f"- RMSE: {report['rmse']:.4f}",
        f"- MAPE (all): {report['mape']:.4f}%",
        f"- MAPE (actual > {MAPE_MASK_THRESHOLD}): {report['masked_mape_actual_gt_100']:.4f}%",
        f"- Zero-price MAE: {report.get('zero_price_mae', 'n/a')}",
        f"- Worst horizon: h{report.get('worst_horizon')} (MAE {report.get('worst_horizon_mae'):.4f})",
        "",
        "## Horizon MAE",
        "",
        "| Hour | MAE |",
        "|-----:|----:|",
    ]
    for h, v in sorted(report["horizon_mae"].items(), key=lambda x: int(x[0])):
        lines.append(f"| {h} | {v:.4f} |")

    cmp_ = report.get("lstm_comparison")
    if cmp_:
        lines.extend(
            [
                "",
                "## LSTM vs persistence",
                "",
                f"- Persistence MAE: {cmp_['persistence_mae']:.4f}",
                f"- LSTM MAE: {cmp_['lstm_mae']:.4f}",
                f"- Delta (LSTM − persistence): {cmp_['mae_delta_lstm_minus_persistence']:.4f}",
                f"- Relative change: {cmp_['mae_pct_change_vs_persistence']:.2f}%",
                f"- **{cmp_['interpretation']}**",
            ]
        )
        if cmp_["mae_delta_lstm_minus_persistence"] > 0:
            lines.append(
                "- Persistence is stronger → revisit LSTM features/architecture."
            )
        else:
            lines.append(
                "- LSTM improves over persistence → market is hard; model adds value."
            )

    return "\n".join(lines)


def load_test_anchors() -> pd.DataFrame:
    """Same test rows as LSTM (sequence anchors with 168h in-split history)."""
    from src.utils.io_utils import read_parquet_with_normalized_ts
    features = read_parquet_with_normalized_ts(FEATURES_PATH)
    test_df = features[features["split"] == "test"].copy()
    test_df["ts_hour"] = pd.to_datetime(test_df["ts_hour"], utc=True).dt.tz_convert(
        "Europe/Istanbul"
    )

    if ANCHOR_TEST_PATH.exists():
        anchors = pd.read_csv(ANCHOR_TEST_PATH)
        anchors["anchor_ts_hour"] = pd.to_datetime(
            anchors["anchor_ts_hour"], utc=True
        ).dt.tz_convert("Europe/Istanbul")
        test_df = test_df.merge(
            anchors[["anchor_ts_hour"]],
            left_on="ts_hour",
            right_on="anchor_ts_hour",
            how="inner",
        ).drop(columns=["anchor_ts_hour"])

    return test_df.sort_values("ts_hour").reset_index(drop=True)


def main() -> dict:
    test_df = load_test_anchors()

    ptf_lookup = load_ptf_lookup()
    pred_df = build_persistence_predictions(test_df, ptf_lookup)

    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(PREDICTIONS_PATH, index=False)

    metrics = compute_metrics(pred_df)
    metrics["method"] = "seasonal_persistence_lag24"
    metrics["predictions_path"] = str(PREDICTIONS_PATH)
    metrics["lstm_comparison"] = compare_with_lstm(metrics)

    METRICS_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    METRICS_MD.write_text(write_markdown(metrics), encoding="utf-8")

    print("Persistence baseline complete.")
    print(f"MAE:  {metrics['mae']:.4f} TL/MWh")
    print(f"RMSE: {metrics['rmse']:.4f} TL/MWh")
    print(f"MAPE (actual > {MAPE_MASK_THRESHOLD}): {metrics['masked_mape_actual_gt_100']:.4f}%")

    if metrics["lstm_comparison"]:
        c = metrics["lstm_comparison"]
        print(f"\nLSTM MAE:        {c['lstm_mae']:.4f}")
        print(f"Persistence MAE: {c['persistence_mae']:.4f}")
        print(f"Delta:           {c['mae_delta_lstm_minus_persistence']:.4f} "
              f"({c['mae_pct_change_vs_persistence']:.2f}% vs persistence)")
        print(c["interpretation"])

    print(f"\nMetrics: {METRICS_JSON}")
    return metrics


if __name__ == "__main__":
    main()
