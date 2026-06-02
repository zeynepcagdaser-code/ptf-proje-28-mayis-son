#!/usr/bin/env python3
"""Rule-based baselines for low/zero price detection on test split."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parent
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "lstm_next24_v1.parquet"
TABULAR_DIR = PROJECT_ROOT / "data" / "model_low_price_tabular"
OUT_JSON = PROJECT_ROOT / "reports" / "low_price_rule_baseline.json"
OUT_MD = PROJECT_ROOT / "reports" / "low_price_rule_baseline.md"

HORIZONS = list(range(1, 25))
LOW_THRESHOLD = 50.0


def _horizon_label(h: int) -> str:
    return f"is_low_{h}h"


def _zero_label(h: int) -> str:
    return f"is_zero_{h}h"


def _metrics(y_low: np.ndarray, y_zero: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    p, r, f1, _ = precision_recall_fscore_support(
        y_low, y_pred, average="binary", zero_division=0
    )
    zero_mask = y_zero == 1
    zero_recall = (
        float(y_pred[zero_mask].mean()) if zero_mask.any() else None
    )
    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "pred_positive_rate": float(y_pred.mean()),
        "low_positive_rate": float(y_low.mean()),
        "zero_recall": zero_recall,
        "zero_rows": int(zero_mask.sum()),
    }


def _load_test_frame() -> pd.DataFrame:
    X = pd.read_parquet(TABULAR_DIR / "X_test.parquet", columns=["sample_index", "anchor_ts_hour"])
    y_low = pd.read_parquet(TABULAR_DIR / "y_low_test.parquet")
    y_zero = pd.read_parquet(TABULAR_DIR / "y_zero_test.parquet")

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
    raw = pd.read_parquet(FEATURES_PATH, columns=raw_cols)
    raw["ts_hour"] = pd.to_datetime(raw["ts_hour"], utc=True)

    base = X.merge(raw, left_on="anchor_ts_hour", right_on="ts_hour", how="left")
    base = base.merge(y_low, on="sample_index", how="left")
    base = base.merge(y_zero, on="sample_index", how="left")

    train_raw = pd.read_parquet(
        FEATURES_PATH,
        columns=[
            "ts_hour",
            "split",
            "renewable_suppression_pressure",
            "thermal_price_setting_share",
        ],
    )
    train_raw = train_raw[train_raw["split"] == "train"]
    base["_ren_q75"] = float(train_raw["renewable_suppression_pressure"].quantile(0.75))
    base["_therm_q25"] = float(train_raw["thermal_price_setting_share"].quantile(0.25))
    return base


def _evaluate_rules(frame: pd.DataFrame) -> dict[str, Any]:
    rules: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
        "rule_1_ptf_lag_1_le_50": lambda d: d["ptf_lag_1"] <= LOW_THRESHOLD,
        "rule_2_ptf_lag_1_le_100": lambda d: d["ptf_lag_1"] <= 100.0,
        "rule_3_ptf_zero_ratio_24_gt_0": lambda d: d["ptf_zero_ratio_24"] > 0.0,
        "rule_4_ptf_low_ratio_24_gt_0": lambda d: d["ptf_low_ratio_24"] > 0.0,
        "rule_5_ptf_zero_ratio_168_gt_0.05": lambda d: d["ptf_zero_ratio_168"] > 0.05,
        "rule_6_zero_price_risk_proxy_eq_1": lambda d: d["zero_price_risk_proxy"] >= 0.999,
        "rule_7_ren_high_therm_low": lambda d: (
            (d["renewable_suppression_pressure"] >= d["_ren_q75"])
            & (d["thermal_price_setting_share"] <= d["_therm_q25"])
        ),
    }

    signals = {name: fn(frame).fillna(False).astype(int) for name, fn in rules.items()}
    combined = np.zeros(len(frame), dtype=int)
    for sig in signals.values():
        combined = np.maximum(combined, sig.to_numpy(dtype=int))
    signals["combined_any_rule"] = pd.Series(combined, index=frame.index)

    report_rules: dict[str, Any] = {}
    for name, pred in signals.items():
        per_h = []
        agg_low = []
        agg_zero = []
        for h in HORIZONS:
            y_low = frame[_horizon_label(h)].to_numpy(dtype=int)
            y_zero = frame[_zero_label(h)].to_numpy(dtype=int)
            y_pred = pred.to_numpy(dtype=int)
            m = _metrics(y_low, y_zero, y_pred)
            m["horizon"] = h
            per_h.append(m)
            agg_low.append(y_low)
            agg_zero.append(y_zero)

        y_low_all = np.concatenate(agg_low)
        y_zero_all = np.concatenate(agg_zero)
        y_pred_all = np.tile(pred.to_numpy(dtype=int), len(HORIZONS))
        report_rules[name] = {
            "overall_stacked_horizons": _metrics(y_low_all, y_zero_all, y_pred_all),
            "per_horizon": per_h,
        }
    return report_rules


def run_rule_baselines() -> dict[str, Any]:
    frame = _load_test_frame()
    missing_raw = int(frame["ptf_lag_1"].isna().sum())
    rules = _evaluate_rules(frame)

    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_samples": int(len(frame)),
        "raw_feature_join_missing_rows": missing_raw,
        "rules": rules,
        "notes": {
            "raw_features_source": str(FEATURES_PATH),
            "rule_6": "zero_price_risk_proxy >= 0.999 (proxy is continuous score)",
            "rule_7_quantiles": "fit on train split only",
            "evaluation_grain": "per horizon, plus stacked 24x rows for overall",
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    OUT_MD.write_text(_to_markdown(report), encoding="utf-8")
    report["report_json"] = str(OUT_JSON)
    report["report_md"] = str(OUT_MD)
    return report


def _to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Low-Price Rule Baselines (Test)",
        "",
        f"- **Generated (UTC):** {report['generated_at_utc']}",
        f"- **Test anchors:** {report['test_samples']}",
        "",
        "## Overall (stacked horizons)",
        "",
        "| Rule | low recall | zero recall | precision | F1 |",
        "|------|----------:|------------:|----------:|---:|",
    ]
    for name, blob in report["rules"].items():
        o = blob["overall_stacked_horizons"]
        lines.append(
            f"| {name} | {o['recall']:.4f} | "
            f"{'' if o['zero_recall'] is None else f'{o['zero_recall']:.4f}'} "
            f"| {o['precision']:.4f} | {o['f1']:.4f} |"
        )
    lines += [
        "",
        "## Per-horizon low recall (h1, h6, h12, h24)",
        "",
        "| Rule | h1 | h6 | h12 | h24 |",
        "|------|---:|---:|----:|----:|",
    ]
    for name, blob in report["rules"].items():
        vals = {row["horizon"]: row["recall"] for row in blob["per_horizon"]}
        lines.append(
            f"| {name} | {vals.get(1, 0):.3f} | {vals.get(6, 0):.3f} "
            f"| {vals.get(12, 0):.3f} | {vals.get(24, 0):.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    report = run_rule_baselines()
    print("=== Low-Price Rule Baselines (test) ===")
    for name, blob in report["rules"].items():
        o = blob["overall_stacked_horizons"]
        print(
            f"{name}: recall={o['recall']:.4f} zero_recall={o['zero_recall']} "
            f"precision={o['precision']:.4f} f1={o['f1']:.4f}"
        )
    print("Wrote:", report["report_md"])


if __name__ == "__main__":
    main()
