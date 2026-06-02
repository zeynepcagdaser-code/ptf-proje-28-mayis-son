#!/usr/bin/env python3
"""
Two-stage PTF pipeline: spike classifier + regressor for non-spike hours.

Outputs:
- models/realtime_two_stage/classifier.pkl
- models/realtime_two_stage/regressor_nonspike.pkl
- data/predictions/ptf_two_stage_latest.csv/.parquet
- data/predictions/ptf_two_stage_latest_diag.json

Usage: run hourly or ad-hoc. Threshold default is 100 TL for spike.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import json
import joblib

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, accuracy_score
from lightgbm import LGBMClassifier, LGBMRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "lstm_next24_v1.parquet"
PRED_DIR = PROJECT_ROOT / "data" / "predictions"
MODEL_DIR = PROJECT_ROOT / "models" / "realtime_two_stage"
SPIKE_THRESHOLD = 100.0


def load_features():
    from src.utils.io_utils import read_parquet_with_normalized_ts
    if not FEATURES_PATH.exists():
        return pd.DataFrame()
    df = read_parquet_with_normalized_ts(FEATURES_PATH)
    return df


def feature_cols(df: pd.DataFrame):
    return [c for c in df.columns if not c.startswith("target_") and c not in {"ts_hour", "split"}]


def safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.replace(0, np.nan)
    return num / den


def winsorize_numeric(df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    numeric = df.select_dtypes(include=["number"]).columns
    for col in numeric:
        q = df[col].quantile([lower, upper])
        if pd.isna(q.iloc[0]) or pd.isna(q.iloc[1]):
            continue
        df[col] = df[col].clip(lower=q.iloc[0], upper=q.iloc[1])
    return df


def clean_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ts_hour"] = pd.to_datetime(df["ts_hour"], errors="coerce")
    if "hour" not in df.columns:
        df["hour"] = df["ts_hour"].dt.hour
    if "weekday" not in df.columns:
        df["weekday"] = df["ts_hour"].dt.dayofweek
    if "is_weekend2" not in df.columns:
        df["is_weekend2"] = (df["weekday"] >= 5).astype(int)
    if "delta_1h" not in df.columns and "target_1h" in df.columns and "ptf_lag_24" in df.columns:
        df["delta_1h"] = pd.to_numeric(df["target_1h"], errors="coerce") - pd.to_numeric(df["ptf_lag_24"], errors="coerce")
    if "ptf_lag_48" in df.columns and "ptf_lag_24" in df.columns:
        df["ptf_lag_24_diff_48"] = df["ptf_lag_24"] - df["ptf_lag_48"]
        df["ptf_lag_24_pct_48"] = safe_ratio(df["ptf_lag_24_diff_48"], df["ptf_lag_48"])
    if "ptf_lag_168" in df.columns and "ptf_lag_24" in df.columns:
        df["ptf_lag_24_diff_168"] = df["ptf_lag_24"] - df["ptf_lag_168"]
        df["ptf_lag_24_pct_168"] = safe_ratio(df["ptf_lag_24_diff_168"], df["ptf_lag_168"])
    if "kgup_renewable_share" in df.columns and "gas_share" in df.columns:
        df["renewable_vs_gas"] = df["kgup_renewable_share"] - df["gas_share"]
    if "renewable_pressure" in df.columns and "kgup_renewable_share" in df.columns:
        df["renewable_pressure_gap"] = df["kgup_renewable_share"] - df["renewable_pressure"]
    split_values = df["split"].copy() if "split" in df.columns else None
    df = df.apply(pd.to_numeric, errors="coerce")
    if split_values is not None:
        df["split"] = split_values
    df = winsorize_numeric(df)
    return df


def make_regressor(objective: str = "regression", alpha: float | None = None) -> LGBMRegressor:
    params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "random_state": 42,
        "objective": objective,
    }
    if objective == "quantile" and alpha is not None:
        params["alpha"] = alpha
    return LGBMRegressor(**params)


def train_and_eval(threshold: float = SPIKE_THRESHOLD, objective: str = "regression", quantile_alpha: float | None = None):
    df = load_features()
    if df.empty:
        print("No features data; aborting")
        return
    if "target_1h" not in df.columns or "ptf_lag_24" not in df.columns:
        raise RuntimeError("Expected target_1h and ptf_lag_24 in features parquet")

    df = clean_and_engineer_features(df)
    df = df.dropna(subset=["ptf_lag_24", "target_1h"]).reset_index(drop=True)

    df["delta_1h"] = df["target_1h"] - df["ptf_lag_24"]
    df["is_spike"] = (df["delta_1h"].abs() >= threshold).astype(int)

    cols = feature_cols(df)

    # splits
    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "validation"].copy() if "validation" in df["split"].unique() else pd.DataFrame()
    test = df[df["split"] == "test"].copy() if "test" in df["split"].unique() else pd.DataFrame()

    # Train classifier
    clf = LGBMClassifier(n_estimators=200, random_state=42)
    X_train = train[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    y_train_clf = train["is_spike"]
    if len(y_train_clf.unique()) == 1:
        print("Only one class in spike label in training; skipping classifier train")
        clf = None
    else:
        clf.fit(X_train, y_train_clf)

    # Train regressor on non-spike
    nonspike_train = train[train["is_spike"] == 0]
    X_train_ns = nonspike_train[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    y_train_ns = nonspike_train["delta_1h"]
    reg_ns = None
    if len(nonspike_train) >= 10 and not y_train_ns.isna().all():
        reg_ns = make_regressor(objective=objective, alpha=quantile_alpha)
        reg_ns.fit(X_train_ns, y_train_ns.fillna(0))

    # train spike regressor if enough examples, otherwise use median delta fallback
    spike_train = train[train["is_spike"] == 1]
    reg_spike = None
    if len(spike_train) >= 20:
        X_train_sp = spike_train[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        y_train_sp = spike_train["delta_1h"].fillna(0)
        reg_spike = make_regressor(objective=objective, alpha=quantile_alpha)
        reg_spike.fit(X_train_sp, y_train_sp)
        spike_mean_delta = float(y_train_sp.median())
    else:
        reg_spike = None
        spike_mean_delta = float(spike_train["delta_1h"].median()) if not spike_train.empty else 0.0

    # Evaluation on validation (or test if not present) — choose val > test
    eval_frame = val if not val.empty else (test if not test.empty else df)
    X_eval = eval_frame[cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    preds = []
    for idx, row in eval_frame.iterrows():
        x = X_eval.loc[idx:idx]
        true = float(row["target_1h"])
        lag24 = float(row["ptf_lag_24"])
        # classify
        is_spike_pred = clf.predict(x)[0] if clf is not None else 0
        if is_spike_pred == 1:
            if reg_spike is not None:
                d = float(reg_spike.predict(x)[0])
                pred = lag24 + d
            else:
                pred = lag24 + spike_mean_delta
        else:
            if reg_ns is None:
                pred = lag24
            else:
                d = float(reg_ns.predict(x)[0])
                pred = lag24 + d
        preds.append({"ts_hour": str(row["ts_hour"]), "target_1h": true, "pred": pred, "is_spike_true": int(row["is_spike"]), "is_spike_pred": int(is_spike_pred)})

    out = pd.DataFrame(preds)
    out["abs_err"] = (out["pred"] - out["target_1h"]).abs()

    mae_overall = float(out["abs_err"].mean())
    mae_spike = float(out[out["is_spike_true"] == 1]["abs_err"].mean()) if not out[out["is_spike_true"] == 1].empty else None
    mae_nonspike = float(out[out["is_spike_true"] == 0]["abs_err"].mean()) if not out[out["is_spike_true"] == 0].empty else None
    acc_100 = float((out["abs_err"] <= 100).mean())

    # save models
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if clf is not None:
        joblib.dump(clf, MODEL_DIR / "classifier.pkl")
    if reg_ns is not None:
        joblib.dump(reg_ns, MODEL_DIR / f"regressor_nonspike_{objective}.pkl")
    if reg_spike is not None:
        joblib.dump(reg_spike, MODEL_DIR / f"regressor_spike_{objective}.pkl")

    suffix = objective if objective != "quantile" else f"quantile_{quantile_alpha:.2f}"
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PRED_DIR / f"ptf_two_stage_latest_{suffix}.parquet"
    csv_path = PRED_DIR / f"ptf_two_stage_latest_{suffix}.csv"
    out.to_parquet(out_path, index=False)
    out.to_csv(csv_path, index=False)

    diag = {
        "n_rows_eval": int(len(out)),
        "mae_overall": mae_overall,
        "mae_spike": mae_spike,
        "mae_nonspike": mae_nonspike,
        "accuracy_within_100TL": acc_100,
        "spike_threshold": threshold,
        "spike_mean_delta_train": spike_mean_delta,
        "regressor_objective": objective,
        "quantile_alpha": quantile_alpha,
        "output_parquet": str(out_path),
        "output_csv": str(csv_path),
    }
    (PRED_DIR / f"ptf_two_stage_latest_diag_{suffix}.json").write_text(json.dumps(diag, ensure_ascii=False, indent=2)+"\n")

    print("Done. MAE overall:", mae_overall)
    print(json.dumps(diag, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Two-stage PTF pipeline with robust regression options")
    parser.add_argument("--threshold", type=float, default=SPIKE_THRESHOLD, help="Spike threshold in TL")
    parser.add_argument("--objective", choices=["regression", "huber", "quantile"], default="huber", help="Regressor objective")
    parser.add_argument("--quantile-alpha", type=float, default=0.5, help="Quantile alpha for quantile regression")
    parser.add_argument("--run-all", action="store_true", help="Run all objectives and compare results")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.run_all:
        results = {}
        for obj in ["regression", "huber", "quantile"]:
            alpha = args.quantile_alpha if obj == "quantile" else None
            print(f"\nRunning objective: {obj}, alpha={alpha}")
            train_and_eval(threshold=args.threshold, objective=obj, quantile_alpha=alpha)
            results[obj] = True
        print("Completed all objectives.")
    else:
        train_and_eval(threshold=args.threshold, objective=args.objective, quantile_alpha=args.quantile_alpha)
