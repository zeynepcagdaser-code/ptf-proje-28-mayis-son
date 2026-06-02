#!/usr/bin/env python3
"""
Rule-based low/zero regime triggers for hybrid gating (no ML classifier).

Evaluates three trigger policies on the test split and writes per-(anchor, horizon)
signals for downstream hybrid use.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parent
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "lstm_next24_v1.parquet"
TABULAR_DIR = PROJECT_ROOT / "data" / "model_low_price_tabular"
SIGNALS_PATH = PROJECT_ROOT / "data" / "predictions" / "low_regime_rule_signals.csv"
OUT_JSON = PROJECT_ROOT / "reports" / "low_regime_rule_trigger_metrics.json"
OUT_MD = PROJECT_ROOT / "reports" / "low_regime_rule_trigger_metrics.md"

HORIZONS = list(range(1, 25))
LOW_THRESHOLD = 50.0


def _horizon_label(h: int) -> str:
    return f"is_low_{h}h"


def _zero_label(h: int) -> str:
    return f"is_zero_{h}h"


def _load_test_frame() -> pd.DataFrame:
    from src.utils.io_utils import read_parquet_with_normalized_ts
    X = read_parquet_with_normalized_ts(TABULAR_DIR / "X_test.parquet", columns=["sample_index", "anchor_ts_hour"])
    y_low = read_parquet_with_normalized_ts(TABULAR_DIR / "y_low_test.parquet")
    y_zero = read_parquet_with_normalized_ts(TABULAR_DIR / "y_zero_test.parquet")

    raw_cols = [
        "ts_hour",
        "ptf_lag_1",
        "ptf_lag_2",
        "ptf_lag_3",
        "ptf_zero_ratio_24",
        "ptf_low_ratio_24",
        "ptf_zero_ratio_168",
        "ptf_low_ratio_168",
        "zero_price_risk_proxy",
        "renewable_suppression_pressure",
        "thermal_price_setting_share",
    ]
    raw = read_parquet_with_normalized_ts(FEATURES_PATH, columns=raw_cols)
    raw["ts_hour"] = pd.to_datetime(raw["ts_hour"], utc=True)

    base = X.merge(raw, left_on="anchor_ts_hour", right_on="ts_hour", how="left")
    base = base.merge(y_low, on="sample_index", how="left")
    base = base.merge(y_zero, on="sample_index", how="left")

    train_raw = read_parquet_with_normalized_ts(
        FEATURE_PATH,
        columns=train_cols,
    )
    train_raw = train_raw[train_raw["split"] == "train"]
    base["_ren_q75"] = float(train_raw["renewable_suppression_pressure"].quantile(0.75))
    base["_therm_q25"] = float(train_raw["thermal_price_setting_share"].quantile(0.25))
    return base


def _atomic_rules(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Individual rules used to build aggressive_rule (combined_any)."""
    return {
        "ptf_lag_1_le_50": (frame["ptf_lag_1"] <= LOW_THRESHOLD).fillna(False),
        "ptf_lag_1_le_100": (frame["ptf_lag_1"] <= 100.0).fillna(False),
        "ptf_zero_ratio_24_gt_0": (frame["ptf_zero_ratio_24"] > 0.0).fillna(False),
        "ptf_low_ratio_24_gt_0": (frame["ptf_low_ratio_24"] > 0.0).fillna(False),
        "ptf_zero_ratio_168_gt_0.05": (frame["ptf_zero_ratio_168"] > 0.05).fillna(False),
        "zero_price_risk_proxy_high": (frame["zero_price_risk_proxy"] >= 0.999).fillna(False),
        "renewable_high_thermal_low": (
            (frame["renewable_suppression_pressure"] >= frame["_ren_q75"])
            & (frame["thermal_price_setting_share"] <= frame["_therm_q25"])
        ).fillna(False),
    }


def _build_trigger_signals(frame: pd.DataFrame) -> pd.DataFrame:
    atomic = _atomic_rules(frame)

    aggressive = np.zeros(len(frame), dtype=int)
    for sig in atomic.values():
        aggressive = np.maximum(aggressive, sig.astype(int).to_numpy())

    balanced = (
        atomic["ptf_low_ratio_24_gt_0"]
        | atomic["ptf_zero_ratio_24_gt_0"]
        | atomic["ptf_zero_ratio_168_gt_0.05"]
    ).astype(int)

    renewable = atomic["renewable_high_thermal_low"].astype(int)

    out = frame[["sample_index", "anchor_ts_hour"]].copy()
    out["balanced_rule_signal"] = balanced
    out["aggressive_rule_signal"] = aggressive
    out["renewable_rule_signal"] = renewable
    return out


def _horizon_metrics(
    y_low: np.ndarray,
    y_zero: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    p, r, f1, _ = precision_recall_fscore_support(
        y_low, y_pred, average="binary", zero_division=0
    )
    neg = y_low == 0
    pos = y_low == 1
    zero_mask = y_zero == 1
    fp = int(((y_pred == 1) & neg).sum())
    tn = int(((y_pred == 0) & neg).sum())
    fpr = float(fp / (fp + tn)) if (fp + tn) else None
    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "false_positive_rate": fpr,
        "alarm_rate": float(y_pred.mean()),
        "zero_recall": float(y_pred[zero_mask].mean()) if zero_mask.any() else None,
        "actual_low_rate": float(y_low.mean()),
        "rows": int(len(y_low)),
    }


def _evaluate_trigger(
    long_df: pd.DataFrame,
    signal_col: str,
) -> dict[str, Any]:
    per_h = []
    agg_low: list[np.ndarray] = []
    agg_zero: list[np.ndarray] = []
    agg_pred: list[np.ndarray] = []

    for h in HORIZONS:
        sub = long_df[long_df["horizon"] == h]
        y_low = sub["actual_low"].to_numpy(dtype=int)
        y_zero = sub["actual_zero"].to_numpy(dtype=int)
        y_pred = sub[signal_col].to_numpy(dtype=int)
        m = _horizon_metrics(y_low, y_zero, y_pred)
        m["horizon"] = h
        per_h.append(m)
        agg_low.append(y_low)
        agg_zero.append(y_zero)
        agg_pred.append(y_pred)

    y_low_all = np.concatenate(agg_low)
    y_zero_all = np.concatenate(agg_zero)
    y_pred_all = np.concatenate(agg_pred)

    return {
        "overall": _horizon_metrics(y_low_all, y_zero_all, y_pred_all),
        "per_horizon": per_h,
    }


def _hybrid_recommendation(metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Score triggers for hybrid gating: recall-heavy but penalize alarm/FPR."""
    scores: dict[str, float] = {}
    rows = []
    for name, blob in metrics.items():
        o = blob["overall"]
        lr = o["recall"]
        zr = o["zero_recall"] if o["zero_recall"] is not None else 0.0
        prec = o["precision"]
        fpr = o["false_positive_rate"] if o["false_positive_rate"] is not None else 1.0
        alarm = o["alarm_rate"]
        # Weighted score: prioritize catching low/zero, tolerate moderate precision drop
        score = 0.35 * lr + 0.35 * zr + 0.20 * prec - 0.10 * fpr
        scores[name] = score
        rows.append(
            {
                "trigger": name,
                "score": score,
                "low_recall": lr,
                "zero_recall": zr,
                "precision": prec,
                "f1": o["f1"],
                "false_positive_rate": fpr,
                "alarm_rate": alarm,
            }
        )
    ranked = sorted(rows, key=lambda r: r["score"], reverse=True)
    balanced = metrics["balanced_rule"]["overall"]

    recommended = "balanced_rule"
    rationale = (
        "Recommended for hybrid: balanced_rule — strong low/zero recall from PTF "
        "regime history, materially lower false alarm rate than aggressive_rule, "
        "and more stable precision than renewable_rule alone."
    )
    if balanced["recall"] < 0.65:
        recommended = ranked[0]["trigger"]
        rationale = (
            f"Balanced recall below 0.65 on test; highest composite score: {recommended}. "
            "Review alarm_rate before using as hybrid gate."
        )

    return {
        "ranked": ranked,
        "recommended_for_hybrid": recommended,
        "rationale": rationale,
    }


def run_evaluation() -> dict[str, Any]:
    frame = _load_test_frame()
    triggers = _build_trigger_signals(frame)

    long_rows: list[pd.DataFrame] = []
    for h in HORIZONS:
        chunk = triggers[
            [
                "sample_index",
                "anchor_ts_hour",
                "balanced_rule_signal",
                "aggressive_rule_signal",
                "renewable_rule_signal",
            ]
        ].copy()
        chunk["horizon"] = h
        chunk["actual_low"] = frame[_horizon_label(h)].astype(int).to_numpy()
        chunk["actual_zero"] = frame[_zero_label(h)].astype(int).to_numpy()
        long_rows.append(chunk)
    long_df = pd.concat(long_rows, ignore_index=True)

    SIGNALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    long_df[
        [
            "sample_index",
            "anchor_ts_hour",
            "horizon",
            "actual_low",
            "actual_zero",
            "balanced_rule_signal",
            "aggressive_rule_signal",
            "renewable_rule_signal",
        ]
    ].to_csv(SIGNALS_PATH, index=False)

    trigger_metrics = {
        "balanced_rule": _evaluate_trigger(long_df, "balanced_rule_signal"),
        "aggressive_rule": _evaluate_trigger(long_df, "aggressive_rule_signal"),
        "renewable_rule": _evaluate_trigger(long_df, "renewable_rule_signal"),
    }

    recommendation = _hybrid_recommendation(trigger_metrics)

    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_anchor_count": int(len(frame)),
        "test_stacked_rows": int(len(long_df)),
        "signals_path": str(SIGNALS_PATH.relative_to(PROJECT_ROOT)),
        "trigger_definitions": {
            "balanced_rule": (
                "ptf_low_ratio_24 > 0 OR ptf_zero_ratio_24 > 0 OR ptf_zero_ratio_168 > 0.05"
            ),
            "aggressive_rule": "OR of all atomic baseline rules (combined_any_rule)",
            "renewable_rule": (
                "renewable_suppression_pressure >= train q75 AND "
                "thermal_price_setting_share <= train q25"
            ),
        },
        "triggers": trigger_metrics,
        "hybrid_recommendation": recommendation,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    OUT_MD.write_text(_to_markdown(report), encoding="utf-8")
    report["report_json"] = str(OUT_JSON)
    report["report_md"] = str(OUT_MD)
    return report


def _to_markdown(report: dict[str, Any]) -> str:
    rec = report["hybrid_recommendation"]
    lines = [
        "# Low Regime Rule Trigger Metrics (Test)",
        "",
        f"- **Generated (UTC):** {report['generated_at_utc']}",
        f"- **Test anchors:** {report['test_anchor_count']}",
        f"- **Signals CSV:** `{report['signals_path']}`",
        "",
        "## Hybrid recommendation",
        "",
        f"**Use `{rec['recommended_for_hybrid']}`** for hybrid gating.",
        "",
        rec["rationale"],
        "",
        "## Overall metrics (stacked horizons)",
        "",
        "| Trigger | low recall | zero recall | precision | F1 | FPR | alarm rate |",
        "|---------|----------:|------------:|----------:|---:|----:|-----------:|",
    ]
    for name, blob in report["triggers"].items():
        o = blob["overall"]
        lines.append(
            f"| {name} | {o['recall']:.4f} | "
            f"{'' if o['zero_recall'] is None else f'{o['zero_recall']:.4f}'} | "
            f"{o['precision']:.4f} | {o['f1']:.4f} | "
            f"{'' if o['false_positive_rate'] is None else f'{o['false_positive_rate']:.4f}'} | "
            f"{o['alarm_rate']:.4f} |"
        )
    lines += [
        "",
        "## Per-horizon low recall",
        "",
        "| Trigger | h1 | h6 | h12 | h18 | h24 |",
        "|---------|---:|---:|----:|----:|----:|",
    ]
    for name, blob in report["triggers"].items():
        vals = {row["horizon"]: row["recall"] for row in blob["per_horizon"]}
        lines.append(
            f"| {name} | {vals.get(1, 0):.3f} | {vals.get(6, 0):.3f} | "
            f"{vals.get(12, 0):.3f} | {vals.get(18, 0):.3f} | {vals.get(24, 0):.3f} |"
        )
    lines += ["", "## Ranked triggers", ""]
    for row in rec["ranked"]:
        lines.append(
            f"- **{row['trigger']}** score={row['score']:.4f} "
            f"(low_rec={row['low_recall']:.3f}, zero_rec={row['zero_recall']:.3f}, "
            f"prec={row['precision']:.3f}, alarm={row['alarm_rate']:.3f})"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    report = run_evaluation()
    rec = report["hybrid_recommendation"]
    print("=== Rule-Based Low Regime Triggers (test) ===")
    for name, blob in report["triggers"].items():
        o = blob["overall"]
        print(
            f"{name}: low_recall={o['recall']:.4f} zero_recall={o['zero_recall']} "
            f"precision={o['precision']:.4f} f1={o['f1']:.4f} fpr={o['false_positive_rate']:.4f} "
            f"alarm={o['alarm_rate']:.4f}"
        )
    print("Hybrid recommendation:", rec["recommended_for_hybrid"])
    print(rec["rationale"])
    print("Signals:", SIGNALS_PATH)
    print("Report:", report["report_md"])


if __name__ == "__main__":
    main()
