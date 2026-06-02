#!/usr/bin/env python3
"""
Train a leakage-safe spike classifier from the master/features pipeline.

Definition:
  PTF spike = price >= 4800 TL/MWh

We train a binary LightGBM classifier to predict whether the future target hour
will be a spike. The training table is built from the existing leakage-safe
feature dataset `data/features/lstm_next24_v1.parquet`:

  - Features at anchor time t (built from t and earlier only)
  - Labels derived from target_{h}h (t+h future PTF), for h=1..24

We stack horizons into one training set and include `target_hour` as a feature.

Outputs:
  - models/spike_classifier.pkl
  - data/predictions/spike_probability.csv (anchor_ts_hour, target_hour, spike_prob)
  - reports/spike_classifier_metrics.json
  - reports/spike_classifier_metrics.md
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parent

MASTER_PATH = PROJECT_ROOT / "data" / "master" / "master_hourly_v1.parquet"
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "lstm_next24_v1.parquet"

MODEL_PATH = PROJECT_ROOT / "models" / "spike_classifier.pkl"
PRED_PATH = PROJECT_ROOT / "data" / "predictions" / "spike_probability.csv"
REPORT_JSON = PROJECT_ROOT / "reports" / "spike_classifier_metrics.json"
REPORT_MD = PROJECT_ROOT / "reports" / "spike_classifier_metrics.md"

SPIKE_THRESHOLD = 4800.0
HORIZONS = list(range(1, 25))


def build_long_table(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    targets = [f"target_{h}h" for h in HORIZONS]
    if any(t not in df.columns for t in targets):
        missing = [t for t in targets if t not in df.columns]
        raise ValueError(f"Missing target columns in features parquet: {missing}")

    feature_cols = [
        c
        for c in df.columns
        if c not in {"ts_hour", "split"} and not c.startswith("target_")
    ]
    if not feature_cols:
        raise ValueError("No feature columns found (expected non-target columns).")

    base = df[["ts_hour", "split"] + feature_cols].copy()
    base = base.replace([np.inf, -np.inf], np.nan)

    rows = []
    for h in HORIZONS:
        tcol = f"target_{h}h"
        tmp = base.copy()
        tmp["target_hour"] = h
        tmp["is_spike"] = (df[tcol].to_numpy(dtype=float) >= SPIKE_THRESHOLD).astype(int)
        rows.append(tmp)
    long_df = pd.concat(rows, ignore_index=True)
    # Past-only forward-fill already applied upstream, but keep training robust to residual NaNs.
    long_df[feature_cols] = long_df[feature_cols].ffill()
    long_df[feature_cols] = long_df[feature_cols].fillna(0.0)
    return long_df, feature_cols


def metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {
        "threshold": float(threshold),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "positive_rate": float(np.mean(y_true)),
        "pred_positive_rate": float(np.mean(y_pred)),
        "rows": int(len(y_true)),
    }


def per_horizon_metrics(frame: pd.DataFrame, prob_col: str) -> list[dict[str, Any]]:
    out = []
    for h in sorted(frame["target_hour"].unique()):
        sub = frame[frame["target_hour"] == h]
        if sub.empty:
            continue
        m = metrics(sub["is_spike"].to_numpy(dtype=int), sub[prob_col].to_numpy(dtype=float))
        m["target_hour"] = int(h)
        out.append(m)
    return out


def main() -> None:
    # 1) Read master (audit only; training uses leakage-safe feature parquet)
    master = pd.read_parquet(MASTER_PATH, columns=["ts_hour", "ptf_price"])
    master["ts_hour"] = pd.to_datetime(master["ts_hour"], errors="coerce")
    master["ptf_price"] = pd.to_numeric(master["ptf_price"], errors="coerce")
    master_spikes = int((master["ptf_price"] >= SPIKE_THRESHOLD).sum())

    # 2) Read features parquet and build stacked horizon training table
    df = pd.read_parquet(FEATURES_PATH)
    df["ts_hour"] = pd.to_datetime(df["ts_hour"], errors="coerce")
    long_df, feature_cols = build_long_table(df)

    # Basic split sanity
    if "split" not in long_df.columns:
        raise ValueError("Features parquet missing split column.")
    if long_df["split"].isna().any():
        raise ValueError("Found null split values.")

    train = long_df[long_df["split"] == "train"].copy()
    val = long_df[long_df["split"] == "validation"].copy()

    X_train = train[feature_cols + ["target_hour"]]
    y_train = train["is_spike"].to_numpy(dtype=int)
    X_val = val[feature_cols + ["target_hour"]]
    y_val = val["is_spike"].to_numpy(dtype=int)

    pos = max(int(y_train.sum()), 1)
    neg = max(int((1 - y_train).sum()), 1)
    scale_pos_weight = neg / pos

    model = LGBMClassifier(
        objective="binary",
        n_estimators=700,
        learning_rate=0.03,
        num_leaves=64,
        min_child_samples=60,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="binary_logloss")

    val_prob = model.predict_proba(X_val)[:, 1]
    val_metrics = metrics(y_val, val_prob, threshold=0.5)
    val_by_h = per_horizon_metrics(val.assign(spike_prob=val_prob), "spike_prob")

    # 6) Write spike probability predictions for all splits/horizons
    X_all = long_df[feature_cols + ["target_hour"]]
    all_prob = model.predict_proba(X_all)[:, 1]
    out_pred = long_df[["ts_hour", "target_hour"]].copy()
    out_pred = out_pred.rename(columns={"ts_hour": "anchor_ts_hour"})
    out_pred["spike_prob"] = all_prob
    PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_pred.to_csv(PRED_PATH, index=False)

    # 5) Save model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_cols": feature_cols,
            "uses_target_hour_feature": True,
            "spike_threshold": SPIKE_THRESHOLD,
            "horizons": HORIZONS,
        },
        MODEL_PATH,
    )

    report: dict[str, Any] = {
        "spike_threshold": SPIKE_THRESHOLD,
        "master_rows": int(len(master)),
        "master_spike_count": master_spikes,
        "features_rows": int(len(df)),
        "stacked_rows": int(len(long_df)),
        "feature_count": int(len(feature_cols) + 1),  # + target_hour
        "scale_pos_weight": float(scale_pos_weight),
        "validation": {
            "overall": val_metrics,
            "per_horizon": val_by_h,
        },
        "outputs": {
            "model_path": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
            "predictions_path": str(PRED_PATH.relative_to(PROJECT_ROOT)),
        },
        "note": "Model is trained on stacked horizons (h=1..24) with target_hour as a feature.",
    }

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n")

    lines = [
        "# Spike Classifier Metrics",
        "",
        f"- Spike threshold: `{SPIKE_THRESHOLD}` TL/MWh",
        f"- Master spikes (>= threshold): `{master_spikes}`",
        f"- Validation rows: `{val_metrics['rows']}` (stacked horizons)",
        f"- scale_pos_weight: `{scale_pos_weight:.3f}`",
        "",
        "## Validation (overall @ 0.5 threshold)",
        "",
        f"- precision: `{val_metrics['precision']:.4f}`",
        f"- recall: `{val_metrics['recall']:.4f}`",
        f"- F1: `{val_metrics['f1']:.4f}`",
        "",
        "## Validation by Horizon (0.5 threshold)",
        "",
        "| h | pos_rate | precision | recall | f1 | rows |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for m in val_by_h:
        lines.append(
            f"| {m['target_hour']} | {m['positive_rate']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['rows']} |"
        )
    lines += [
        "",
        f"Model: `{MODEL_PATH.relative_to(PROJECT_ROOT)}`",
        f"Predictions: `{PRED_PATH.relative_to(PROJECT_ROOT)}`",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n")

    print(f"Saved model: {MODEL_PATH}")
    print(f"Wrote predictions: {PRED_PATH} ({len(out_pred)} rows)")
    print(
        f"Validation F1: {val_metrics['f1']:.4f} (precision {val_metrics['precision']:.4f}, recall {val_metrics['recall']:.4f})"
    )


if __name__ == "__main__":
    main()
