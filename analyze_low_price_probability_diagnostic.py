#!/usr/bin/env python3
"""Probability diagnostics for low-price classifier predictions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent
PRED_PATH = PROJECT_ROOT / "data" / "predictions" / "low_price_probabilities.csv"
METRICS_JSON = PROJECT_ROOT / "reports" / "low_price_classifier_metrics.json"
OUT_JSON = PROJECT_ROOT / "reports" / "low_price_probability_diagnostic.json"
OUT_MD = PROJECT_ROOT / "reports" / "low_price_probability_diagnostic.md"


def _dist_stats(values: np.ndarray) -> dict[str, float | None]:
    if len(values) == 0:
        return {"count": 0, "mean": None, "median": None, "p10": None, "p90": None, "max": None}
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
        "max": float(np.max(values)),
    }


def _segment_mask(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    low = df["is_low_actual"].astype(int) == 1
    zero = df["is_zero_actual"].fillna(0).astype(int) == 1
    normal = (~low) & (~zero)
    return low, zero, normal


def _threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray) -> list[dict[str, Any]]:
    grid = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    rows = []
    for thr in grid:
        pred = (y_prob >= thr).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append(
            {
                "threshold": thr,
                "precision": prec,
                "recall": rec,
                "f1": f1,
            }
        )
    return rows


def run_diagnostic(
    *,
    pred_path: Path | None = None,
    metrics_path: Path | None = None,
) -> dict[str, Any]:
    pred_path = pred_path or PRED_PATH
    metrics_path = metrics_path or METRICS_JSON

    pred = pd.read_csv(pred_path)
    pred["low_prob"] = pd.to_numeric(pred["low_prob"], errors="coerce")
    pred["is_low_actual"] = pd.to_numeric(pred["is_low_actual"], errors="coerce").fillna(0).astype(int)
    if "is_zero_actual" in pred.columns:
        pred["is_zero_actual"] = pd.to_numeric(pred["is_zero_actual"], errors="coerce")

    prior_metrics = None
    if metrics_path.exists():
        prior_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "predictions_path": str(pred_path),
        "prior_classifier_metrics_summary": (
            prior_metrics.get("summary") if prior_metrics else None
        ),
        "overall": {},
        "by_split": {},
        "by_horizon": {},
        "diagnosis": {},
    }

    low_mask, zero_mask, normal_mask = _segment_mask(pred)
    probs = pred["low_prob"].to_numpy(dtype=float)

    report["overall"]["probability_by_segment"] = {
        "actual_low": _dist_stats(probs[low_mask.to_numpy()]),
        "actual_zero": _dist_stats(probs[zero_mask.to_numpy()]),
        "normal": _dist_stats(probs[normal_mask.to_numpy()]),
    }

    for split in sorted(pred["split"].unique()):
        sub = pred[pred["split"] == split]
        lm, zm, nm = _segment_mask(sub)
        p = sub["low_prob"].to_numpy(dtype=float)
        y = sub["is_low_actual"].to_numpy(dtype=int)
        auc = None
        if len(np.unique(y)) > 1:
            auc = float(roc_auc_score(y, p))
        sweep = _threshold_sweep(y, p)
        max_f1 = max(sweep, key=lambda r: r["f1"])
        max_recall = max(sweep, key=lambda r: r["recall"])
        report["by_split"][split] = {
            "rows": int(len(sub)),
            "roc_auc": auc,
            "probability_by_segment": {
                "actual_low": _dist_stats(p[lm.to_numpy()]),
                "actual_zero": _dist_stats(p[zm.to_numpy()]),
                "normal": _dist_stats(p[nm.to_numpy()]),
            },
            "threshold_sweep": sweep,
            "max_f1_threshold": max_f1,
            "max_recall_threshold": max_recall,
        }

    horizon_rows = []
    for h in sorted(pred["horizon"].unique()):
        sub = pred[pred["horizon"] == h]
        test_sub = sub[sub["split"] == "test"]
        if test_sub.empty:
            continue
        lm, zm, nm = _segment_mask(test_sub)
        p = test_sub["low_prob"].to_numpy(dtype=float)
        y = test_sub["is_low_actual"].to_numpy(dtype=int)
        auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None
        horizon_rows.append(
            {
                "horizon": int(h),
                "test_roc_auc": auc,
                "p_low_median_actual_low": _dist_stats(p[lm.to_numpy()])["median"],
                "p_low_median_actual_zero": _dist_stats(p[zm.to_numpy()])["median"],
                "p_low_median_normal": _dist_stats(p[nm.to_numpy()])["median"],
                "p_low_mean_actual_low": _dist_stats(p[lm.to_numpy()])["mean"],
            }
        )
    report["by_horizon"]["test"] = horizon_rows

    test_blob = report["by_split"].get("test", {})
    test_auc = test_blob.get("roc_auc")
    low_med = report["overall"]["probability_by_segment"]["actual_low"]["median"]
    norm_med = report["overall"]["probability_by_segment"]["normal"]["median"]
    separation = None
    if low_med is not None and norm_med is not None:
        separation = float(low_med - norm_med)

    if test_auc is not None and test_auc >= 0.65 and separation is not None and separation < 0.05:
        verdict = (
            "AUC moderate/high but probability segments overlap — threshold tuning likely "
            "the main issue (model ranks somewhat but scores are compressed)."
        )
    elif test_auc is not None and test_auc < 0.60:
        verdict = "Low AUC — model discrimination is weak; feature engineering needed."
    elif separation is not None and separation >= 0.10:
        verdict = "Segments separate reasonably — threshold/recall trade-off is primary lever."
    else:
        verdict = "Mixed signal — review horizon-level AUC and segment overlap."

    report["diagnosis"] = {
        "test_roc_auc_overall": test_auc,
        "median_p_low_actual_low": low_med,
        "median_p_low_normal": norm_med,
        "median_gap_low_minus_normal": separation,
        "verdict": verdict,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    OUT_MD.write_text(_to_markdown(report), encoding="utf-8")
    report["report_json"] = str(OUT_JSON)
    report["report_md"] = str(OUT_MD)
    return report


def _to_markdown(report: dict[str, Any]) -> str:
    seg = report["overall"]["probability_by_segment"]
    lines = [
        "# Low-Price Probability Diagnostic",
        "",
        f"- **Generated (UTC):** {report['generated_at_utc']}",
        f"- **Predictions:** `{report['predictions_path']}`",
        "",
        "## Overall p_low by segment",
        "",
        "| Segment | count | mean | median | p10 | p90 |",
        "|---------|------:|-----:|-------:|----:|----:|",
    ]
    for name, stats in seg.items():
        lines.append(
            f"| {name} | {stats['count']} | "
            f"{'' if stats['mean'] is None else f'{stats['mean']:.4f}'} | "
            f"{'' if stats['median'] is None else f'{stats['median']:.4f}'} | "
            f"{'' if stats['p10'] is None else f'{stats['p10']:.4f}'} | "
            f"{'' if stats['p90'] is None else f'{stats['p90']:.4f}'} |"
        )
    lines += ["", "## Diagnosis", "", report["diagnosis"]["verdict"], ""]
    for split, blob in report["by_split"].items():
        lines += [
            f"### Split: {split}",
            "",
            f"- ROC-AUC: `{blob.get('roc_auc')}`",
            f"- Max-F1 threshold: `{blob.get('max_f1_threshold')}`",
            f"- Max-recall threshold: `{blob.get('max_recall_threshold')}`",
            "",
        ]
    lines += ["## Test horizon p_low medians", "", "| h | AUC | med(low) | med(zero) | med(normal) |", "|--:|----:|---------:|----------:|------------:|"]
    for row in report["by_horizon"].get("test", []):
        lines.append(
            f"| {row['horizon']} | {'' if row['test_roc_auc'] is None else f'{row['test_roc_auc']:.4f}'} "
            f"| {row['p_low_median_actual_low']} | {row['p_low_median_actual_zero']} "
            f"| {row['p_low_median_normal']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    report = run_diagnostic()
    d = report["diagnosis"]
    print("=== Low-Price Probability Diagnostic ===")
    print("Verdict:", d["verdict"])
    print("Test ROC-AUC:", d.get("test_roc_auc_overall"))
    print("Median p_low | low:", d.get("median_p_low_actual_low"), "| normal:", d.get("median_p_low_normal"))
    print("Wrote:", report["report_md"])


if __name__ == "__main__":
    main()
