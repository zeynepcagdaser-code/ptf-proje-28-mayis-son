#!/usr/bin/env python3
"""
Hourly retrainable PTF regressor + realtime 12-hour predictor.

- Trains one LightGBM model per horizon (1..12) on `data/features/lstm_next24_v1.parquet`.
- On run, finds latest EPIAŞ/ptf timestamp and predicts t+1..t+12 using the feature row
  anchored at the latest timestamp (or the newest available anchor before it).
- Saves models to `models/realtime_hourly/` and predictions to `data/predictions/`.

Usage: run hourly (cron / systemd / CI). This script is conservative: if data
is missing it will exit cleanly and not overwrite previous predictions.
"""

from __future__ import annotations

from pathlib import Path
from typing import List
import json
import os
import joblib

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "lstm_next24_v1.parquet"
PTF_CSV = PROJECT_ROOT / "data" / "ptf_dataset.csv"
MODEL_DIR = PROJECT_ROOT / "models" / "realtime_hourly"
PRED_DIR = PROJECT_ROOT / "data" / "predictions"

HORIZONS = list(range(1, 13))  # 1..12


def load_features() -> pd.DataFrame:
    from src.utils.io_utils import read_parquet_with_normalized_ts

    if not FEATURES_PATH.exists():
        return pd.DataFrame()
    return read_parquet_with_normalized_ts(FEATURES_PATH)


def latest_ptf_ts() -> pd.Timestamp | None:
    if not PTF_CSV.exists():
        return None
    ptf = pd.read_csv(PTF_CSV)
    if "date" in ptf.columns and "hour" in ptf.columns:
        ts = pd.to_datetime(ptf["date"], errors="coerce")
        hour = pd.to_numeric(ptf["hour"], errors="coerce").fillna(0).astype(int)
        ts = ts.dt.normalize() + pd.to_timedelta(hour, unit="h")
        return ts.max()
    if "ts_hour" in ptf.columns:
        ts = pd.to_datetime(ptf["ts_hour"], errors="coerce")
        return ts.max()
    return None


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    # drop all target_* columns and meta cols
    cols = [c for c in df.columns if not c.startswith("target_") and c not in {"ts_hour", "split"}]
    return df[cols].copy()


def train_models(df: pd.DataFrame, features: List[str]) -> dict[int, LGBMRegressor]:
    models: dict[int, LGBMRegressor] = {}
    train_df = df[df["split"] == "train"].dropna(subset=features, how="all")
    val_df = df[df["split"] == "validation"] if "validation" in df["split"].unique() else pd.DataFrame()

    X_train = train_df[features]
    # ensure numeric dtypes for LightGBM
    X_train = X_train.apply(pd.to_numeric, errors="coerce")
    for h in HORIZONS:
        target_col = f"target_{h}h" if f"target_{h}h" in df.columns else f"target_{h}"
        if target_col not in df.columns:
            continue
        y_train = train_df[target_col]
        if len(X_train) == 0 or y_train.isna().all():
            continue
        model = LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
        model.fit(X_train.fillna(0), y_train.fillna(0))
        models[h] = model
    return models


def eval_models(models: dict[int, LGBMRegressor], df: pd.DataFrame, features: List[str]):
    out = {}
    val_df = df[df["split"] == "validation"] if "validation" in df["split"].unique() else pd.DataFrame()
    if val_df.empty:
        return out
    X_val = val_df[features]
    X_val = X_val.apply(pd.to_numeric, errors="coerce")
    for h, model in models.items():
        target_col = f"target_{h}h" if f"target_{h}h" in df.columns else f"target_{h}"
        if target_col not in val_df.columns:
            continue
        y_true = val_df[target_col].values
        y_pred = model.predict(X_val.fillna(0))
        out[h] = {"mae": float(mean_absolute_error(y_true, y_pred))}
    return out


def save_models(models: dict[int, LGBMRegressor]):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for h, m in models.items():
        joblib.dump(m, MODEL_DIR / f"ptf_h{h}.pkl")


def load_models() -> dict[int, LGBMRegressor]:
    out = {}
    if not MODEL_DIR.exists():
        return out
    for h in HORIZONS:
        p = MODEL_DIR / f"ptf_h{h}.pkl"
        if p.exists():
            out[h] = joblib.load(p)
    return out


def predict_from_anchor(models: dict[int, LGBMRegressor], anchor_row: pd.Series, anchor_ts: pd.Timestamp, features: List[str]) -> pd.DataFrame:
    rows = []
    X = anchor_row[features].to_frame().T
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    for h, model in models.items():
        pred = float(model.predict(X)[0])
        rows.append({"delivery_hour": anchor_ts + pd.Timedelta(hours=h), "predicted_ptf": pred, "horizon": h})
    return pd.DataFrame(rows)


def main():
    df = load_features()
    if df.empty:
        print("No features parquet found; aborting.")
        return

    ts = latest_ptf_ts()
    if ts is None or pd.isna(ts):
        print("No PTF CSV timestamp found; aborting.")
        return

    # Find the feature anchor row at ts (or the nearest prior)
    # normalize dataframe ts_hour to naive and make the scalar ts comparable
    df["ts_hour"] = pd.to_datetime(df["ts_hour"], errors="coerce")
    try:
        # if ts is tz-aware, convert to Europe/Istanbul then drop tz
        if getattr(ts, "tzinfo", None) is not None:
            ts = ts.tz_convert("Europe/Istanbul").tz_localize(None)
    except Exception:
        try:
            ts = ts.tz_localize(None)
        except Exception:
            ts = pd.to_datetime(ts)

    anchor_row = df.loc[df["ts_hour"] == ts]
    if anchor_row.empty:
        prior = df[df["ts_hour"] < ts]
        if prior.empty:
            print("No anchor row available for prediction; aborting.")
            return
        anchor_row = prior.sort_values("ts_hour").iloc[[-1]]
    else:
        anchor_row = anchor_row.iloc[[0]]

    features = list(feature_matrix(df).columns)

    # Train
    models = train_models(df, features)
    save_models(models)

    # Eval
    evals = eval_models(models, df, features)

    # Predict
    preds = predict_from_anchor(models, anchor_row.iloc[0], ts, features)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PRED_DIR / "ptf_realtime_latest.parquet"
    preds.to_parquet(out_path, index=False)
    csv_path = PRED_DIR / "ptf_realtime_latest.csv"
    preds.to_csv(csv_path, index=False)

    # Save diagnostics
    diag = {
        "anchor_ts": str(ts),
        "n_models": len(models),
        "eval": evals,
        "output_parquet": str(out_path),
        "output_csv": str(csv_path),
    }
    (PRED_DIR / "ptf_realtime_latest.json").write_text(json.dumps(diag, ensure_ascii=False, indent=2) + "\n")

    print("Done. Predictions written to:", out_path)


if __name__ == "__main__":
    main()
