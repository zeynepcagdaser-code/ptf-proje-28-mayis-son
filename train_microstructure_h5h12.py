#!/usr/bin/env python3
"""h5–h12 LightGBM on microstructure features (persistence + residual). Does not touch h1–h4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "lstm_microstructure_next24_v1.parquet"
MODEL_DIR = PROJECT_ROOT / "models" / "microstructure_h5h12"
ANCHOR_TEST_PATH = PROJECT_ROOT / "data" / "model" / "anchor_test.csv"
PREDICTIONS_CSV = PROJECT_ROOT / "data" / "predictions" / "microstructure_h5h12_predictions.csv"

HORIZONS = list(range(5, 13))
EARLY_STOPPING_ROUNDS = 50
MAX_BOOST_ROUNDS = 2000


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def add_persistence(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("ts_hour").reset_index(drop=True)
    for h in HORIZONS:
        out[f"persistence_{h}h"] = out[f"target_{h}h"].shift(24)
    return out


def resolve_base_features(df: pd.DataFrame) -> list[str]:
    target_cols = [c for c in df.columns if c.startswith("target_")]
    exclude = {"ts_hour", "split", *target_cols}
    exclude.update(c for c in df.columns if c.startswith("persistence_"))
    return [c for c in df.columns if c not in exclude]


def horizon_features(base: list[str], h: int) -> list[str]:
    return base + [f"persistence_{h}h"]


def train_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
) -> Any:
    import lightgbm as lgb

    train_set = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
    return lgb.train(
        {
            "objective": "regression",
            "metric": "mae",
            "verbosity": -1,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_data_in_leaf": 50,
            "lambda_l1": 0.1,
            "lambda_l2": 1.0,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "seed": 42,
        },
        train_set,
        num_boost_round=MAX_BOOST_ROUNDS,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS), lgb.log_evaluation(0)],
    )


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


def run(*, smoke: bool = False) -> None:
    import lightgbm  # noqa: F401

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing {FEATURES_PATH}")

    from src.utils.io_utils import read_parquet_with_normalized_ts
    df = add_persistence(read_parquet_with_normalized_ts(FEATURES_PATH))
    base_features = resolve_base_features(df)
    horizons = HORIZONS[:2] if smoke else HORIZONS

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for h in horizons:
        tcol, pcol = f"target_{h}h", f"persistence_{h}h"
        fcols = horizon_features(base_features, h)
        cols = fcols + [tcol, pcol, "ts_hour", "split"]
        train = df[df["split"] == "train"].dropna(subset=cols)
        val = df[df["split"] == "validation"].dropna(subset=cols)
        test = df[df["split"] == "test"].dropna(subset=cols)

        X_train = train[fcols].to_numpy(dtype=np.float64)
        y_train = (train[tcol] - train[pcol]).to_numpy(dtype=np.float64)
        X_val = val[fcols].to_numpy(dtype=np.float64)
        y_val = (val[tcol] - val[pcol]).to_numpy(dtype=np.float64)

        booster = train_lgbm(X_train, y_train, X_val, y_val, fcols)
        booster.save_model(str(MODEL_DIR / f"horizon_{h:02d}.txt"))
        (MODEL_DIR / f"horizon_{h:02d}_features.json").write_text(
            json.dumps(fcols, indent=2), encoding="utf-8"
        )

        res_pred = booster.predict(test[fcols].to_numpy(dtype=np.float64))
        print(f"h{h} val_residual_mae={mae(y_val, booster.predict(X_val)):.2f}")

        for i, idx in enumerate(test.index):
            row = test.loc[idx]
            actual = float(row[tcol])
            persistence = float(row[pcol])
            pred = persistence + float(res_pred[i])
            rows.append(
                {
                    "anchor_ts_hour": str(row["ts_hour"]),
                    "target_hour": h,
                    "actual_price": actual,
                    "persistence_price": persistence,
                    "predicted_residual": float(res_pred[i]),
                    "predicted_price": pred,
                    "absolute_error": abs(actual - pred),
                }
            )

    pred_df = pd.DataFrame(rows)
    PREDICTIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(PREDICTIONS_CSV, index=False)

    aligned = filter_lstm_anchors(pred_df)
    mean_mae = float(
        aligned.groupby("target_hour")["absolute_error"].mean().mean()
    )
    print(f"Test mean h5–h12 MAE (aligned): {mean_mae:.2f}")
    print(f"Predictions: {PREDICTIONS_CSV}")


def main() -> None:
    p = argparse.ArgumentParser(description="Train microstructure LGBM h5–h12")
    p.add_argument("--smoke-test", action="store_true")
    run(smoke=p.parse_args().smoke_test)


if __name__ == "__main__":
    main()
