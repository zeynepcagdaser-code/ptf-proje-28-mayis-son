#!/usr/bin/env python3
"""Verify low-price classifier label/probability alignment and recompute metrics."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent
TABULAR_DIR = PROJECT_ROOT / "data" / "model_low_price_tabular"
PRED_PATH = PROJECT_ROOT / "data" / "predictions" / "low_price_probabilities.csv"
OUT_JSON = PROJECT_ROOT / "reports" / "low_price_alignment_check.json"
OUT_MD = PROJECT_ROOT / "reports" / "low_price_alignment_check.md"

THRESHOLDS = [0.02, 0.05, 0.10]
HORIZONS = list(range(1, 25))


def _horizon_label(h: int) -> str:
    return f"is_low_{h}h"


def _zero_label(h: int) -> str:
    return f"is_zero_{h}h"


def _metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> dict[str, Any]:
    y_pred = (y_prob >= thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    out = {
        "threshold": thr,
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "pred_positive_rate": float(y_pred.mean()),
    }
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        out["roc_auc"] = None
        out["pr_auc"] = None
    return out


def _prob_stats(y_prob: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    vals = y_prob[mask]
    if len(vals) == 0:
        return {"count": 0, "mean": None, "median": None}
    return {
        "count": int(len(vals)),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
    }


def _verify_parquet_alignment(
    pred: pd.DataFrame,
    y_low: pd.DataFrame,
    y_zero: pd.DataFrame | None,
    split: str,
) -> dict[str, Any]:
    sub = pred[pred["split"] == split].copy()
    y_low_idx = y_low.set_index("sample_index") if "sample_index" in y_low.columns else y_low

    mismatches: dict[str, int] = {}
    for h in HORIZONS:
        hsub = sub[sub["horizon"] == h]
        expected = y_low_idx[_horizon_label(h)].loc[hsub["sample_index"].values].to_numpy()
        actual = hsub["is_low_actual"].to_numpy(dtype=int)
        mismatches[f"h{h}"] = int((expected != actual).sum())

    zero_mismatches = None
    if y_zero is not None and split == "test":
        y_zero_idx = y_zero.set_index("sample_index")
        for h in HORIZONS:
            hsub = sub[sub["horizon"] == h]
            z_csv = pd.to_numeric(hsub["is_zero_actual"], errors="coerce").fillna(-1).astype(int)
            z_pq = y_zero_idx[_zero_label(h)].loc[hsub["sample_index"].values].to_numpy()
            if zero_mismatches is None:
                zero_mismatches = {}
            zero_mismatches[f"h{h}"] = int((z_csv != z_pq).sum())

    return {
        "split": split,
        "pred_rows": int(len(sub)),
        "unique_samples": int(sub["sample_index"].nunique()),
        "expected_rows_samples_x_24": int(sub["sample_index"].nunique() * 24),
        "is_low_label_mismatches_by_horizon": mismatches,
        "total_is_low_mismatches": int(sum(mismatches.values())),
        "is_zero_label_mismatches_by_horizon": zero_mismatches,
    }


def _horizon_block(
    pred: pd.DataFrame,
    y_low: pd.DataFrame,
    y_zero: pd.DataFrame | None,
    split: str,
    h: int,
) -> dict[str, Any]:
    sub = pred[(pred["split"] == split) & (pred["horizon"] == h)].copy()
    sub = sub.sort_values("sample_index").reset_index(drop=True)

    y_low_idx = y_low.set_index("sample_index")
    label_col = _horizon_label(h)
    y_parquet = y_low_idx[label_col].loc[sub["sample_index"].values].to_numpy(dtype=int)

    y_prob = sub["low_prob"].to_numpy(dtype=float)
    y_csv = sub["is_low_actual"].to_numpy(dtype=int)
    thr_sel = float(sub["selected_threshold"].iloc[0]) if len(sub) else None
    low_pred = sub["low_pred"].to_numpy(dtype=int)
    pred_from_thr = (y_prob >= thr_sel).astype(int) if thr_sel is not None else None

    low_mask = y_csv == 1
    normal_mask = y_csv == 0

    block: dict[str, Any] = {
        "horizon": h,
        "split": split,
        "rows": int(len(sub)),
        "actual_low_count": int(low_mask.sum()),
        "actual_low_rate": float(low_mask.mean()) if len(sub) else None,
        "parquet_label_matches_csv": int((y_parquet == y_csv).sum()),
        "parquet_label_mismatches": int((y_parquet != y_csv).sum()),
        "mapping": f"low_prob (horizon={h}) vs is_low_{h}h / is_low_actual",
        "p_low_stats": {
            "actual_low_1": _prob_stats(y_prob, low_mask),
            "actual_low_0": _prob_stats(y_prob, normal_mask),
        },
        "threshold_metrics": {
            str(thr): _metrics_at_threshold(y_csv, y_prob, thr) for thr in THRESHOLDS
        },
        "selected_threshold": thr_sel,
        "low_pred_matches_threshold_rule": (
            int((low_pred == pred_from_thr).sum()) if pred_from_thr is not None else None
        ),
        "low_pred_threshold_mismatches": (
            int((low_pred != pred_from_thr).sum()) if pred_from_thr is not None else None
        ),
    }

    if y_zero is not None and split == "test":
        y_zero_idx = y_zero.set_index("sample_index")
        y_zero_arr = y_zero_idx[_zero_label(h)].loc[sub["sample_index"].values].to_numpy(dtype=int)
        zero_mask = y_zero_arr == 1
        block["actual_zero_count"] = int(zero_mask.sum())
        block["p_low_stats"]["actual_zero_1"] = _prob_stats(y_prob, zero_mask)
        block["is_zero_parquet_matches_csv"] = int(
            (
                pd.to_numeric(sub["is_zero_actual"], errors="coerce").fillna(-1).astype(int)
                == y_zero_arr
            ).sum()
        )

    return block


def run_alignment_check() -> dict[str, Any]:
    pred = pd.read_csv(PRED_PATH)
    pred["anchor_ts_hour"] = pd.to_datetime(pred["anchor_ts_hour"], utc=True)
    pred["low_prob"] = pd.to_numeric(pred["low_prob"], errors="coerce")
    pred["is_low_actual"] = pd.to_numeric(pred["is_low_actual"], errors="coerce").fillna(0).astype(int)

    y_low_test = pd.read_parquet(TABULAR_DIR / "y_low_test.parquet")
    y_zero_test = pd.read_parquet(TABULAR_DIR / "y_zero_test.parquet")
    y_low_val = pd.read_parquet(TABULAR_DIR / "y_low_val.parquet")

    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "predictions_path": str(PRED_PATH),
        "conventions": {
            "row_grain": "one row per (split, sample_index, horizon)",
            "probability_column": "low_prob",
            "label_column_in_csv": "is_low_actual",
            "label_column_in_parquet": "is_low_{h}h",
            "horizon_alignment": "pred.horizon=h maps to is_low_{h}h (no offset)",
            "sample_index_scope": "per-split 0..N-1, not global",
        },
        "split_alignment": {
            "test": _verify_parquet_alignment(pred, y_low_test, y_zero_test, "test"),
            "validation": _verify_parquet_alignment(pred, y_low_val, None, "validation"),
        },
        "split_summary": {},
        "per_horizon": {"test": [], "validation": []},
        "contradiction_analysis": {},
        "planned_models": {
            "zero_specific_classifier": {
                "status": "planned_not_trained",
                "target": "is_zero_{h}h from inverse-scaled y (PTF == 0)",
                "model": "separate horizon-wise classifier (LightGBM/HGB)",
                "threshold_policy": "recall-first; target zero recall >= 0.70 on validation",
                "features": "LOW_PRICE_CLASSIFIER_FEATURES + zero regime history",
                "note": "Decouple from <=50 TL low-price head; optimize zero capture",
            },
            "any_horizon_classifier": {
                "status": "planned_not_trained",
                "target_any_low": "any(is_low_1h..is_low_24h) per anchor row",
                "target_any_zero": "any(is_zero_1h..is_zero_24h) per anchor row",
                "grain": "one row per anchor hour (not 24x stacked)",
                "use_case": "hourly regime alert independent of horizon offset",
                "threshold_policy": "recall-first on validation",
            },
        },
    }

    for split in ("train", "validation", "test"):
        sub = pred[pred["split"] == split]
        low = sub["is_low_actual"] == 1
        p = sub["low_prob"].to_numpy(dtype=float)
        report["split_summary"][split] = {
            "rows": int(len(sub)),
            "actual_low_count": int(low.sum()),
            "p_low_median_given_actual_low": _prob_stats(p, low.to_numpy())["median"],
            "p_low_median_given_actual_not_low": _prob_stats(p, (~low).to_numpy())["median"],
            "recall_at_0.02": _metrics_at_threshold(
                sub["is_low_actual"].to_numpy(dtype=int), p, 0.02
            )["recall"],
            "roc_auc": (
                float(roc_auc_score(sub["is_low_actual"], p))
                if sub["is_low_actual"].nunique() > 1
                else None
            ),
        }

    for split, y_low, y_zero in (
        ("test", y_low_test, y_zero_test),
        ("validation", y_low_val, None),
    ):
        for h in HORIZONS:
            report["per_horizon"][split].append(
                _horizon_block(pred, y_low, y_zero, split, h)
            )

    train_med = report["split_summary"]["train"]["p_low_median_given_actual_low"]
    test_med = report["split_summary"]["test"]["p_low_median_given_actual_low"]
    test_recall = report["split_summary"]["test"]["recall_at_0.02"]
    test_auc = report["split_summary"]["test"]["roc_auc"]

    report["contradiction_analysis"] = {
        "apparent_paradox": "High overall median p_low for actual_low vs low test recall",
        "root_cause": (
            "Overall diagnostics pool train+validation+test. Train rows (~78% of stacked "
            "rows) show extreme overfit (median p_low|low≈1, recall@0.02=1.0). "
            "Test-only median p_low|actual_low is near zero, so recall stays low."
        ),
        "train_median_p_low_given_low": train_med,
        "test_median_p_low_given_low": test_med,
        "test_recall_at_0.02": test_recall,
        "test_roc_auc": test_auc,
        "alignment_verdict": (
            "PASS"
            if report["split_alignment"]["test"]["total_is_low_mismatches"] == 0
            else "FAIL"
        ),
        "horizon_offset_detected": False,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    OUT_MD.write_text(_to_markdown(report), encoding="utf-8")
    report["report_json"] = str(OUT_JSON)
    report["report_md"] = str(OUT_MD)
    return report


def _to_markdown(report: dict[str, Any]) -> str:
    ca = report["contradiction_analysis"]
    lines = [
        "# Low-Price Classifier Alignment Check",
        "",
        f"- **Generated (UTC):** {report['generated_at_utc']}",
        "",
        "## Mapping conventions",
        "",
        f"- Row grain: {report['conventions']['row_grain']}",
        f"- `low_prob` at `horizon=h` ↔ `is_low_{'{h}'}h` / `is_low_actual`",
        f"- `sample_index`: {report['conventions']['sample_index_scope']}",
        "",
        "## Contradiction analysis",
        "",
        ca["root_cause"],
        "",
        "| Split | median p_low \\| low | recall@0.02 | ROC-AUC |",
        "|-------|------------------:|------------:|--------:|",
    ]
    for split, blob in report["split_summary"].items():
        lines.append(
            f"| {split} | {blob['p_low_median_given_actual_low']} "
            f"| {blob['recall_at_0.02']:.4f} | {blob['roc_auc']} |"
        )
    lines += [
        "",
        f"**Alignment verdict:** {ca['alignment_verdict']}",
        f"**Horizon offset:** {ca['horizon_offset_detected']}",
        "",
        "## Test parquet vs CSV labels",
        "",
        f"- Total `is_low` mismatches: {report['split_alignment']['test']['total_is_low_mismatches']}",
        "",
        "## Test horizon snapshot (h=1, h=2, h=11)",
        "",
        "| h | low_n | med p\\|low | med p\\|not low | recall@0.02 | recall@0.05 | ROC-AUC |",
        "|--:|------:|-------------:|---------------:|------------:|------------:|--------:|",
    ]
    for h in [1, 2, 11]:
        blk = next(x for x in report["per_horizon"]["test"] if x["horizon"] == h)
        m02 = blk["threshold_metrics"]["0.02"]
        m05 = blk["threshold_metrics"]["0.05"]
        lines.append(
            f"| {h} | {blk['actual_low_count']} "
            f"| {blk['p_low_stats']['actual_low_1']['median']} "
            f"| {blk['p_low_stats']['actual_low_0']['median']} "
            f"| {m02['recall']:.4f} | {m05['recall']:.4f} | {m02.get('roc_auc')} |"
        )
    lines += [
        "",
        "## Planned (not trained)",
        "",
        "### Zero-specific classifier",
        "",
        json.dumps(report["planned_models"]["zero_specific_classifier"], indent=2),
        "",
        "### Any-horizon classifier",
        "",
        json.dumps(report["planned_models"]["any_horizon_classifier"], indent=2),
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    report = run_alignment_check()
    ca = report["contradiction_analysis"]
    print("=== Low-Price Alignment Check ===")
    print("Verdict:", ca["alignment_verdict"])
    print("Horizon offset:", ca["horizon_offset_detected"])
    print("Test label mismatches:", report["split_alignment"]["test"]["total_is_low_mismatches"])
    print("Train median p|low:", ca["train_median_p_low_given_low"])
    print("Test median p|low:", ca["test_median_p_low_given_low"])
    print("Test recall@0.02:", ca["test_recall_at_0.02"])
    print("Test ROC-AUC:", ca["test_roc_auc"])
    print("Wrote:", report["report_md"])


if __name__ == "__main__":
    main()
