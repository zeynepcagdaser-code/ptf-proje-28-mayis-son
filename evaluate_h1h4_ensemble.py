#!/usr/bin/env python3
"""Post-hoc h1–h4 ensemble from existing prediction CSVs (no training)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
ANCHOR_TEST_PATH = PROJECT_ROOT / "data" / "model" / "anchor_test.csv"

PRED_SOURCES = {
    "advanced_tree": PROJECT_ROOT / "data" / "predictions" / "tree_test_predictions.csv",
    "microstructure": PROJECT_ROOT
    / "data"
    / "predictions"
    / "microstructure_h1h4_predictions.csv",
    "short_expert": PROJECT_ROOT
    / "data"
    / "predictions"
    / "short_horizon_expert_predictions.csv",
    "persistence": PROJECT_ROOT / "data" / "predictions" / "persistence_predictions.csv",
}

METRICS_JSON = PROJECT_ROOT / "reports" / "h1h4_ensemble_metrics.json"
METRICS_MD = PROJECT_ROOT / "reports" / "h1h4_ensemble_metrics.md"
PREDICTIONS_CSV = PROJECT_ROOT / "data" / "predictions" / "h1h4_ensemble_predictions.csv"
FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "h1h4_ensemble_comparison.png"

HORIZONS = [1, 2, 3, 4]
ADVANCED_TREE_BASELINE_MAE = 453.0947792677989
PRIMARY_WEIGHTS = (0.7, 0.3)  # advanced, micro — fixed; not tuned on test


def mae_series(actual: pd.Series, pred: pd.Series) -> float:
    return float(np.mean(np.abs(actual - pred)))


def horizon_mae(df: pd.DataFrame, pred_col: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for h in HORIZONS:
        sub = df[df["target_hour"] == h]
        out[str(h)] = mae_series(sub["actual_price"], sub[pred_col])
    out["mean_h1_h4"] = float(np.mean([out[str(h)] for h in HORIZONS]))
    return out


def filter_lstm_anchors(pred_df: pd.DataFrame) -> pd.DataFrame:
    if not ANCHOR_TEST_PATH.exists():
        return pred_df
    anchors = pd.read_csv(ANCHOR_TEST_PATH)
    anchors["anchor_ts_hour"] = pd.to_datetime(
        anchors["anchor_ts_hour"], utc=True
    ).dt.tz_convert("Europe/Istanbul")
    out = pred_df.copy()
    out["anchor_ts_hour"] = pd.to_datetime(out["anchor_ts_hour"], utc=True).dt.tz_convert(
        "Europe/Istanbul"
    )
    return out.merge(anchors[["anchor_ts_hour"]], on="anchor_ts_hour", how="inner")


def load_horizon_preds(path: Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["target_hour"].isin(HORIZONS)].copy()
    df["anchor_ts_hour"] = pd.to_datetime(df["anchor_ts_hour"], utc=True).dt.tz_convert(
        "Europe/Istanbul"
    )
    df = filter_lstm_anchors(df)
    cols = ["anchor_ts_hour", "target_hour", "actual_price", "predicted_price"]
    df = df[cols].rename(columns={"predicted_price": name})
    return df


def merge_predictions() -> pd.DataFrame:
    base = load_horizon_preds(PRED_SOURCES["advanced_tree"], "advanced_tree")
    for key in ("microstructure", "short_expert", "persistence"):
        other = load_horizon_preds(PRED_SOURCES[key], key)
        base = base.merge(
            other[["anchor_ts_hour", "target_hour", key]],
            on=["anchor_ts_hour", "target_hour"],
            how="inner",
        )
    return base.sort_values(["anchor_ts_hour", "target_hour"]).reset_index(drop=True)


def add_ensemble_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    a, m = out["advanced_tree"], out["microstructure"]
    out["advanced_tree_only"] = a
    out["microstructure_only"] = m
    out["short_expert_only"] = out["short_expert"]
    out["persistence_only"] = out["persistence"]
    out["avg_advanced_micro"] = (a + m) / 2.0
    for w_adv, tag in [
        (0.8, "weighted_0.8_0.2"),
        (0.7, "weighted_0.7_0.3"),
        (0.6, "weighted_0.6_0.4"),
        (0.5, "weighted_0.5_0.5"),
    ]:
        out[tag] = w_adv * a + (1.0 - w_adv) * m
    w_adv, w_micro = PRIMARY_WEIGHTS
    out["primary_weighted_ensemble"] = w_adv * a + w_micro * m

    candidates = np.column_stack(
        [out["advanced_tree"].values, out["microstructure"].values, out["short_expert"].values]
    )
    errs = np.abs(out["actual_price"].values[:, None] - candidates)
    best_idx = np.argmin(errs, axis=1)
    out["horizon_best_oracle_test"] = candidates[np.arange(len(out)), best_idx]
    out["oracle_model"] = np.where(
        best_idx == 0, "advanced_tree", np.where(best_idx == 1, "microstructure", "short_expert")
    )
    return out


def evaluate_strategies(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    strategies = {
        "advanced_tree_only": "advanced_tree_only",
        "microstructure_only": "microstructure_only",
        "short_expert_only": "short_expert_only",
        "persistence_only": "persistence_only",
        "avg_advanced_micro": "avg_advanced_micro",
        "weighted_0.8_0.2": "weighted_0.8_0.2",
        "weighted_0.7_0.3": "weighted_0.7_0.3",
        "weighted_0.6_0.4": "weighted_0.6_0.4",
        "weighted_0.5_0.5": "weighted_0.5_0.5",
        "primary_weighted_ensemble": "primary_weighted_ensemble",
        "horizon_best_oracle_test": "horizon_best_oracle_test",
    }
    return {name: horizon_mae(df, col) for name, col in strategies.items()}


def oracle_model_counts(df: pd.DataFrame) -> dict[str, int]:
    return {str(k): int(v) for k, v in df["oracle_model"].value_counts().items()}


def write_md(payload: dict[str, Any]) -> None:
    lines = [
        "# h1–h4 prediction ensemble (no training)",
        "",
        "Combines existing test predictions on LSTM-anchor aligned rows.",
        "",
        f"- **Rows:** {payload['aligned_rows']}",
        f"- **Advanced tree baseline (mean h1–h4 MAE):** {payload['advanced_tree_baseline_mae']:.2f}",
        f"- **Primary result:** `{payload['primary_strategy']}` "
        f"({payload['primary_weights']['advanced_tree']:.1f} advanced + "
        f"{payload['primary_weights']['microstructure']:.1f} micro, fixed — not tuned on test)",
        "",
        "## Strategy MAE (TL/MWh)",
        "",
        "| Strategy | h1 | h2 | h3 | h4 | Mean h1–h4 | vs advanced |",
        "|----------|-----:|-----:|-----:|-----:|-----:|-----:|",
    ]
    primary = payload["strategies"][payload["primary_strategy"]]
    for name, m in payload["strategies"].items():
        delta = m["mean_h1_h4"] - payload["advanced_tree_baseline_mae"]
        tag = " **PRIMARY**" if name == payload["primary_strategy"] else ""
        oracle = " *(test oracle — leakage)*" if name == "horizon_best_oracle_test" else ""
        lines.append(
            f"| {name}{tag}{oracle} | {m['1']:.2f} | {m['2']:.2f} | {m['3']:.2f} | "
            f"{m['4']:.2f} | **{m['mean_h1_h4']:.2f}** | {delta:+.2f} |"
        )

    lines.extend(
        [
            "",
            "## Horizon-specific best selector",
            "",
            payload["horizon_selector_note"],
            "",
            f"Oracle test MAE mean: **{payload['strategies']['horizon_best_oracle_test']['mean_h1_h4']:.2f}**",
            f"Oracle model picks: {payload['oracle_model_counts']}",
            "",
            "## Verdict",
            "",
            payload["verdict"],
            "",
            f"Primary ensemble mean MAE: **{primary['mean_h1_h4']:.2f}**",
        ]
    )
    METRICS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_comparison(strategies: dict[str, dict[str, float]], primary: str) -> None:
    order = [
        "advanced_tree_only",
        "microstructure_only",
        "short_expert_only",
        "avg_advanced_micro",
        "weighted_0.8_0.2",
        "weighted_0.7_0.3",
        "weighted_0.6_0.4",
        "weighted_0.5_0.5",
        "primary_weighted_ensemble",
    ]
    labels = [s.replace("_", "\n") for s in order]
    means = [strategies[s]["mean_h1_h4"] for s in order]
    colors = ["#4C72B0" if s != primary else "#C44E52" for s in order]
    colors[0] = "#55A868"

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(order))
    bars = ax.bar(x, means, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(ADVANCED_TREE_BASELINE_MAE, color="#55A868", linestyle="--", linewidth=2, label="advanced tree baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Mean MAE h1–h4 (TL/MWh)")
    ax.set_title("h1–h4 ensemble vs advanced tree baseline")
    ax.legend(loc="upper right")
    for bar, val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=150)
    plt.close(fig)


def main() -> None:
    df = merge_predictions()
    df = add_ensemble_columns(df)
    strategies = evaluate_strategies(df)

    primary_name = "primary_weighted_ensemble"
    primary_mae = strategies[primary_name]["mean_h1_h4"]
    adv_mae = strategies["advanced_tree_only"]["mean_h1_h4"]
    beats = primary_mae < adv_mae

    if beats:
        verdict = (
            f"Primary weighted ensemble ({PRIMARY_WEIGHTS[0]:.1f}/{PRIMARY_WEIGHTS[1]:.1f}) "
            f"beats advanced tree on mean h1–h4 MAE ({primary_mae:.2f} vs {adv_mae:.2f}, "
            f"Δ {primary_mae - adv_mae:+.2f})."
        )
    else:
        verdict = (
            f"Primary weighted ensemble does NOT beat advanced tree "
            f"({primary_mae:.2f} vs {adv_mae:.2f})."
        )

    payload: dict[str, Any] = {
        "method": "post_hoc_prediction_ensemble",
        "prediction_sources": {k: str(v) for k, v in PRED_SOURCES.items()},
        "anchor_filter": str(ANCHOR_TEST_PATH),
        "horizons": HORIZONS,
        "aligned_rows": int(len(df)),
        "aligned_anchors": int(df["anchor_ts_hour"].nunique()),
        "advanced_tree_baseline_mae": ADVANCED_TREE_BASELINE_MAE,
        "primary_strategy": primary_name,
        "primary_weights": {"advanced_tree": PRIMARY_WEIGHTS[0], "microstructure": PRIMARY_WEIGHTS[1]},
        "strategies": strategies,
        "delta_vs_advanced_tree": {
            name: {k: strategies[name][k] - strategies["advanced_tree_only"][k] for k in ("1", "2", "3", "4", "mean_h1_h4")}
            for name in strategies
        },
        "beats_advanced_tree": {
            name: strategies[name]["mean_h1_h4"] < adv_mae for name in strategies
        },
        "primary_beats_advanced_tree": beats,
        "horizon_selector_note": (
            "No validation prediction CSVs available; horizon-specific selector uses test-set "
            "oracle only (labeled horizon_best_oracle_test — not used as primary result)."
        ),
        "oracle_model_counts": oracle_model_counts(df),
        "verdict": verdict,
        "predictions_path": str(PREDICTIONS_CSV),
        "figure_path": str(FIGURE_PATH),
    }

    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_md(payload)
    plot_comparison(strategies, primary_name)

    out_cols = [
        "anchor_ts_hour",
        "target_hour",
        "actual_price",
        "advanced_tree",
        "microstructure",
        "short_expert",
        "persistence",
        "primary_weighted_ensemble",
        "avg_advanced_micro",
        "weighted_0.7_0.3",
        "horizon_best_oracle_test",
        "oracle_model",
    ]
    PREDICTIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df[out_cols].to_csv(PREDICTIONS_CSV, index=False)

    print(f"Aligned rows: {len(df)}")
    print(f"Advanced tree only: {adv_mae:.2f}")
    print(f"Primary weighted ({PRIMARY_WEIGHTS[0]}/{PRIMARY_WEIGHTS[1]}): {primary_mae:.2f}")
    print(verdict)
    print(f"Metrics: {METRICS_JSON}")


if __name__ == "__main__":
    main()
