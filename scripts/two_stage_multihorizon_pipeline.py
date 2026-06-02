#!/usr/bin/env python3
"""
Multi-horizon two-stage PTF pipeline for T+1..T+12.

This script trains a separate two-stage pipeline for each horizon.
Each horizon uses a spike classifier + non-spike/spike regressors to predict delta vs. a baseline.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import joblib

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMClassifier

from src.utils.feature_utils import (
    clean_and_engineer_features,
    feature_cols,
    make_regressor,
    target_column_name,
)
from src.utils.io_utils import read_parquet_with_normalized_ts

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "lstm_next24_v1.parquet"
PRED_DIR = PROJECT_ROOT / "data" / "predictions"
MODEL_DIR = PROJECT_ROOT / "models" / "realtime_two_stage"
SPIKE_THRESHOLD = 100.0
DEFAULT_HORIZONS = list(range(1, 13))


def load_features() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        return pd.DataFrame()
    return read_parquet_with_normalized_ts(FEATURES_PATH)


def baseline_column(df: pd.DataFrame, horizon: int) -> str:
    candidate = f"ptf_lag_{24 + horizon}"
    return candidate if candidate in df.columns else "ptf_lag_24"


def train_horizon(
    df: pd.DataFrame,
    horizon: int,
    objective: str = "regression",
    quantile_alpha: float | None = None,
):
    target_col = target_column_name(horizon)
    if target_col not in df.columns:
        print(f"Skipping horizon {horizon}, missing target column {target_col}")
        return

    frame = df.copy()
    frame = frame.dropna(subset=[target_col]).reset_index(drop=True)
    base_col = baseline_column(frame, horizon)
    frame["delta_h"] = frame[target_col] - frame[base_col]
    frame["is_spike"] = (frame["delta_h"].abs() >= SPIKE_THRESHOLD).astype(int)

    cols = feature_cols(frame)
    train = frame[frame["split"] == "train"].copy()
    val = frame[frame["split"] == "validation"].copy() if "validation" in frame["split"].unique() else pd.DataFrame()
    test = frame[frame["split"] == "test"].copy() if "test" in frame["split"].unique() else pd.DataFrame()

    # classifier
    clf = LGBMClassifier(n_estimators=200, random_state=42)
    X_train = train[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    y_train_clf = train["is_spike"]
    if len(y_train_clf.unique()) == 1:
        print(f"Horizon {horizon} has a single spike class; skipping classifier.")
        clf = None
    else:
        clf.fit(X_train, y_train_clf)

    # regressors
    nonspike_train = train[train["is_spike"] == 0]
    X_train_ns = nonspike_train[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    y_train_ns = nonspike_train["delta_h"]
    reg_ns = None
    if len(nonspike_train) >= 10 and not y_train_ns.isna().all():
        reg_ns = make_regressor(objective=objective, alpha=quantile_alpha)
        reg_ns.fit(X_train_ns, y_train_ns.fillna(0))

    spike_train = train[train["is_spike"] == 1]
    reg_spike = None
    if len(spike_train) >= 20:
        X_train_sp = spike_train[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        y_train_sp = spike_train["delta_h"].fillna(0)
        reg_spike = make_regressor(objective=objective, alpha=quantile_alpha)
        reg_spike.fit(X_train_sp, y_train_sp)
        spike_mean = float(y_train_sp.median())
    else:
        spike_mean = float(spike_train["delta_h"].median()) if not spike_train.empty else 0.0

    eval_frame = val if not val.empty else (test if not test.empty else frame)
    X_eval = eval_frame[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    preds = []
    for idx, row in eval_frame.iterrows():
        x = X_eval.loc[idx:idx]
        baseline = float(row[base_col])
        true = float(row[target_col])
        is_spike_pred = clf.predict(x)[0] if clf is not None else 0
        if is_spike_pred == 1:
            if reg_spike is not None:
                d = float(reg_spike.predict(x)[0])
                pred = baseline + d
            else:
                pred = baseline + spike_mean
        else:
            if reg_ns is not None:
                d = float(reg_ns.predict(x)[0])
                pred = baseline + d
            else:
                pred = baseline
        preds.append({
            "ts_hour": str(row["ts_hour"]),
            "target": true,
            "pred": pred,
            "horizon": horizon,
            "is_spike_true": int(row["is_spike"]),
            "is_spike_pred": int(is_spike_pred),
        })

    out = pd.DataFrame(preds)
    out["abs_err"] = (out["pred"] - out["target"]).abs()
    mae_overall = float(out["abs_err"].mean())
    mae_spike = float(out[out["is_spike_true"] == 1]["abs_err"].mean()) if not out[out["is_spike_true"] == 1].empty else None
    mae_nonspike = float(out[out["is_spike_true"] == 0]["abs_err"].mean()) if not out[out["is_spike_true"] == 0].empty else None
    acc_100 = float((out["abs_err"] <= 100).mean())

    model_root = MODEL_DIR / f"h{horizon}"
    model_root.mkdir(parents=True, exist_ok=True)
    if clf is not None:
        joblib.dump(clf, model_root / "classifier.pkl")
    if reg_ns is not None:
        joblib.dump(reg_ns, model_root / f"regressor_nonspike_{objective}.pkl")
    if reg_spike is not None:
        joblib.dump(reg_spike, model_root / f"regressor_spike_{objective}.pkl")

    suffix = objective if objective != "quantile" else f"quantile_{quantile_alpha:.2f}"
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PRED_DIR / f"ptf_two_stage_h{horizon}_latest_{suffix}.parquet"
    csv_path = PRED_DIR / f"ptf_two_stage_h{horizon}_latest_{suffix}.csv"
    out.to_parquet(out_path, index=False)
    out.to_csv(csv_path, index=False)

    diag = {
        "horizon": horizon,
        "n_rows_eval": int(len(out)),
        "mae_overall": mae_overall,
        "mae_spike": mae_spike,
        "mae_nonspike": mae_nonspike,
        "accuracy_within_100TL": acc_100,
        "spike_threshold": SPIKE_THRESHOLD,
        "spike_mean_delta_train": spike_mean,
        "objective": objective,
        "quantile_alpha": quantile_alpha,
        "output_parquet": str(out_path),
        "output_csv": str(csv_path),
    }
    (PRED_DIR / f"ptf_two_stage_h{horizon}_latest_diag_{suffix}.json").write_text(json.dumps(diag, ensure_ascii=False, indent=2) + "\n")

    print(f"Horizon {horizon}: MAE {mae_overall:.3f} (spike={mae_spike}, nonspike={mae_nonspike})")


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-horizon two-stage PTF pipeline")
    parser.add_argument("--horizons", type=int, nargs="*", default=None, help="Horizon numbers to build. Default is 1..12")
    parser.add_argument("--threshold", type=float, default=SPIKE_THRESHOLD, help="Spike threshold in TL")
    parser.add_argument("--objective", choices=["regression", "huber", "quantile"], default="regression", help="Regressor objective")
    parser.add_argument("--quantile-alpha", type=float, default=0.5, help="Quantile alpha for quantile regression")
    parser.add_argument("--run-all", action="store_true", help="Run all horizons 1..12")
    return parser.parse_args()


def main():
    args = parse_args()
    df = load_features()
    if df.empty:
        print("No features data found; aborting.")
        return
    df = clean_and_engineer_features(df)

    horizons = args.horizons if args.horizons else DEFAULT_HORIZONS
    if args.run_all:
        horizons = DEFAULT_HORIZONS

    for h in horizons:
        train_horizon(
            df,
            h,
            objective=args.objective,
            quantile_alpha=args.quantile_alpha,
        )


if __name__ == '__main__':
    main()
