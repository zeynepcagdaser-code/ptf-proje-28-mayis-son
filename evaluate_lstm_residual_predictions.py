#!/usr/bin/env python3
"""Evaluate residual LSTM from existing predictions CSV only (no training / model load)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
PREDICTIONS_CSV = PROJECT_ROOT / "data" / "predictions" / "lstm_residual_test_predictions.csv"
PERSISTENCE_METRICS_JSON = PROJECT_ROOT / "reports" / "persistence_metrics.json"
METRICS_JSON = PROJECT_ROOT / "reports" / "lstm_residual_metrics.json"
METRICS_MD = PROJECT_ROOT / "reports" / "lstm_residual_metrics.md"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
DIRECT_LSTM_METRICS = PROJECT_ROOT / "reports" / "lstm_baseline_metrics.json"
MAPE_MASK_THRESHOLD = 100.0


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mape_masked(actual: np.ndarray, pred: np.ndarray, threshold: float) -> float:
    mask = np.abs(actual) > threshold
    if not mask.any():
        return float("nan")
    a, p = actual[mask], pred[mask]
    return float(np.mean(np.abs((a - p) / a)) * 100)


def horizon_mae_df(df: pd.DataFrame, pred_col: str) -> dict[str, float]:
    err = (df["actual_price"] - df[pred_col]).abs()
    grouped = df.assign(_err=err).groupby("target_hour")["_err"].mean()
    return {str(int(h)): float(v) for h, v in grouped.items()}


def plot_horizon_mae(horizon_errors: dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hours = sorted(horizon_errors.keys(), key=int)
    values = [horizon_errors[h] for h in hours]
    plt.figure(figsize=(9, 5))
    plt.bar([int(h) for h in hours], values, color="steelblue")
    plt.xlabel("Forecast horizon (hours)")
    plt.ylabel("MAE (TL/MWh)")
    plt.title("Residual LSTM — final prediction MAE by horizon")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_vs_persistence(
    persistence_mae: float,
    final_mae: float,
    direct_lstm_mae: float | None,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["Persistence", "LSTM residual"]
    values = [persistence_mae, final_mae]
    colors = ["#6baed6", "#2171b5"]
    if direct_lstm_mae is not None:
        labels.append("LSTM direct")
        values.append(direct_lstm_mae)
        colors.append("#969696")

    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, values, color=colors)
    plt.ylabel("Test MAE (TL/MWh)")
    plt.title("Test MAE comparison")
    plt.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.1f}",
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def write_metrics_md(report: dict) -> str:
    cmp_ = report["persistence_comparison"]
    lines = [
        "# LSTM Residual Metrics",
        "",
        "_Evaluated from predictions CSV only (no model load)._",
        "",
        f"- **Source:** `{report['predictions_path']}`",
        f"- **Test anchors:** {report['test_samples_anchors']}",
        f"- **Prediction rows:** {report['prediction_rows']}",
        "",
        "## Final prediction metrics (TL/MWh)",
        "",
        f"- MAE: {report['test_mae']:.4f}",
        f"- RMSE: {report['test_rmse']:.4f}",
        f"- MAPE (actual > {MAPE_MASK_THRESHOLD}): {report['masked_mape_actual_gt_100']:.4f}%",
        f"- Zero-price MAE: {report.get('zero_price_mae', 'n/a')}",
        f"- Zero-price hours: {report.get('zero_price_hours', 'n/a')}",
        "",
        f"- **Worst horizon:** h{report['worst_horizon']} (MAE {report['worst_horizon_mae']:.4f})",
        "",
        "## vs persistence",
        "",
        f"- Persistence MAE (this CSV): {cmp_['persistence_mae']:.4f}",
        f"- Residual LSTM MAE: {cmp_['final_mae']:.4f}",
        f"- Improvement vs persistence: {cmp_['improvement_pct']:.2f}%",
        f"- **{cmp_['interpretation']}**",
        "",
    ]
    if cmp_.get("persistence_metrics_json_mae") is not None:
        lines.extend(
            [
                f"- Persistence MAE (`persistence_metrics.json`): {cmp_['persistence_metrics_json_mae']:.4f}",
                "",
            ]
        )
    lines.extend(
        [
            "## Horizon MAE (final prediction)",
            "",
            "| Hour | MAE |",
            "|-----:|----:|",
        ]
    )
    for h, v in sorted(report["horizon_mae"].items(), key=lambda x: int(x[0])):
        lines.append(f"| {h} | {v:.4f} |")
    return "\n".join(lines)


def main() -> dict:
    if not PREDICTIONS_CSV.exists():
        raise FileNotFoundError(f"Missing predictions: {PREDICTIONS_CSV}")

    df = pd.read_csv(PREDICTIONS_CSV)
    actual = df["actual_price"].to_numpy(dtype=np.float64)
    final = df["predicted_price"].to_numpy(dtype=np.float64)
    persistence = df["persistence_price"].to_numpy(dtype=np.float64)

    zero_mask = actual == 0
    h_mae = horizon_mae_df(df, "predicted_price")
    worst_h = max(h_mae, key=h_mae.get)

    persistence_mae = mae(actual, persistence)
    final_mae = mae(actual, final)
    improvement_pct = (persistence_mae - final_mae) / persistence_mae * 100

    interpretation = (
        "Residual LSTM beats persistence baseline"
        if final_mae < persistence_mae
        else "Persistence baseline still better than residual LSTM"
    )

    persistence_json_mae = None
    if PERSISTENCE_METRICS_JSON.exists():
        persistence_json_mae = json.loads(
            PERSISTENCE_METRICS_JSON.read_text(encoding="utf-8")
        ).get("mae")

    direct_lstm_mae = None
    if DIRECT_LSTM_METRICS.exists():
        direct_lstm_mae = json.loads(DIRECT_LSTM_METRICS.read_text())["test_mae"]

    report = {
        "evaluation_source": "predictions_csv_only",
        "predictions_path": str(PREDICTIONS_CSV),
        "test_samples_anchors": int(df["anchor_ts_hour"].nunique()),
        "prediction_rows": int(len(df)),
        "test_mae": final_mae,
        "test_rmse": rmse(actual, final),
        "masked_mape_actual_gt_100": mape_masked(actual, final, MAPE_MASK_THRESHOLD),
        "zero_price_mae": mae(actual[zero_mask], final[zero_mask]) if zero_mask.any() else None,
        "zero_price_hours": int(zero_mask.sum()),
        "horizon_mae": h_mae,
        "worst_horizon": int(worst_h),
        "worst_horizon_mae": h_mae[worst_h],
        "persistence_comparison": {
            "persistence_mae": persistence_mae,
            "final_mae": final_mae,
            "mae_delta_final_minus_persistence": final_mae - persistence_mae,
            "improvement_pct": improvement_pct,
            "lstm_better_than_persistence": final_mae < persistence_mae,
            "interpretation": interpretation,
            "persistence_metrics_json_mae": persistence_json_mae,
            "direct_lstm_mae": direct_lstm_mae,
        },
        "model_path": str(PROJECT_ROOT / "models" / "lstm_residual.pt"),
        "note": "Metrics recomputed from CSV; training may have exited before report write.",
    }

    METRICS_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    METRICS_MD.write_text(write_metrics_md(report), encoding="utf-8")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_horizon_mae(h_mae, FIGURES_DIR / "lstm_residual_horizon_mae.png")
    plot_vs_persistence(
        persistence_mae,
        final_mae,
        direct_lstm_mae,
        FIGURES_DIR / "lstm_residual_vs_persistence.png",
    )

    print("Residual LSTM evaluation (CSV only)")
    print(f"MAE:  {final_mae:.4f} TL/MWh")
    print(f"RMSE: {report['test_rmse']:.4f} TL/MWh")
    print(f"MAPE (actual > {MAPE_MASK_THRESHOLD}): {report['masked_mape_actual_gt_100']:.4f}%")
    print(f"Persistence MAE: {persistence_mae:.4f}")
    print(f"Improvement vs persistence: {improvement_pct:.2f}%")
    print(f"{interpretation}")
    print(f"Metrics: {METRICS_JSON}")
    return report


if __name__ == "__main__":
    main()
