#!/usr/bin/env python3
"""
Train and evaluate a D+2 PTF forecaster.

Prediction form:
    pred(h) = anchor_d1_ptf(h) + alpha_h * model_residual(h)

anchor_d1_ptf = PTF at D+1 same hour (known on EPİAŞ when D+1 is published).
Training target at delivery hour t: price(t) - ptf_lag_24(t).
Per-hour shrinkage alpha_h is tuned on validation to minimize MAE.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parent
FEATURE_STORE = PROJECT_ROOT / "data" / "features" / "regime_feature_store.parquet"
REASONING = PROJECT_ROOT / "data" / "features" / "market_reasoning_features.parquet"
LABELS = PROJECT_ROOT / "data" / "regime_labels.csv"

MODEL_DIR = PROJECT_ROOT / "models" / "d2_ptf_forecaster"
METRICS_JSON = PROJECT_ROOT / "reports" / "d2_ptf_forecaster_metrics.json"
METRICS_MD = PROJECT_ROOT / "reports" / "d2_ptf_forecaster_metrics.md"

SPLIT_RANGES = {"train": (2020, 2024), "validation": (2025, 2025), "test": (2026, 2026)}

FORBIDDEN = {
    "price",
    "target_regime",
    "transition_label",
    "persistence_error",
    "price_lag_24",
    "target_ptf",
    "systemMarginalPrice",
    "marketTradePrice",
}


def assign_split(ts: pd.Series) -> pd.Series:
    years = ts.dt.year
    split = pd.Series(index=ts.index, dtype="object")
    for name, (a, b) in SPLIT_RANGES.items():
        split[(years >= a) & (years <= b)] = name
    return split


def hour_group(hour: pd.Series) -> pd.Series:
    h = hour.astype(int)
    groups = pd.Series("other", index=hour.index, dtype="object")
    groups[h.between(0, 5)] = "night"
    groups[h.between(6, 10)] = "morning"
    groups[h.between(11, 16)] = "solar_window"
    groups[h.between(17, 22)] = "evening_ramp"
    return groups


def load_training_frame() -> pd.DataFrame:
    from src.utils.io_utils import read_parquet_with_normalized_ts
    features = read_parquet_with_normalized_ts(FEATURE_STORE)
    reasoning = read_parquet_with_normalized_ts(REASONING)
    labels = pd.read_csv(LABELS)
    for frame in [features, reasoning, labels]:
        frame["ts_hour"] = pd.to_datetime(frame["ts_hour"], errors="coerce")

    data = features.merge(reasoning, on="ts_hour", how="inner").merge(
        labels[["ts_hour", "price", "target_regime", "price_lag_24"]],
        on="ts_hour",
        how="inner",
    )
    data = data.dropna(subset=["price", "ptf_lag_24"]).sort_values("ts_hour")
    data["split"] = assign_split(data["ts_hour"])
    data = data[data["split"].isin(SPLIT_RANGES)].copy()
    data["anchor_d1_ptf"] = data["ptf_lag_24"]
    data["residual_target"] = data["price"] - data["anchor_d1_ptf"]
    data["hour_group"] = hour_group(data["hour"])
    if "ptf_lag_48" in data.columns and "ptf_lag_24" in data.columns:
        data["ptf_momentum_d1_d2"] = data["ptf_lag_24"] - data["ptf_lag_48"]
    return data


def feature_columns(data: pd.DataFrame) -> list[str]:
    drop = {
        "ts_hour",
        "split",
        "price",
        "target_regime",
        "price_lag_24",
        "residual_target",
        "anchor_d1_ptf",
        "hour_group",
    }
    cols = [c for c in data.columns if c not in drop and c not in FORBIDDEN]
    numeric = []
    for col in cols:
        if pd.api.types.is_numeric_dtype(data[col]) or data[col].dtype == bool:
            numeric.append(col)
    return numeric


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    err = np.abs(y_true - y_pred)
    return {
        "rows": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "median_ae": float(np.median(err)),
        "p90_ae": float(np.quantile(err, 0.9)),
        "pct_le_100": float((err <= 100).mean()),
        "pct_le_50": float((err <= 50).mean()),
    }


def tune_hour_shrinkage(
    val: pd.DataFrame,
    raw_residual: np.ndarray,
    anchor: np.ndarray,
    y_true: np.ndarray,
) -> dict[int, float]:
    alphas: dict[int, float] = {}
    for hour in range(24):
        mask = val["hour"].astype(int) == hour
        if mask.sum() == 0:
            alphas[hour] = 0.5
            continue
        best_alpha = 0.0
        best_mae = float("inf")
        for alpha in np.linspace(0, 1, 21):
            pred = anchor[mask] + alpha * raw_residual[mask]
            mae = mean_absolute_error(y_true[mask], pred)
            if mae < best_mae:
                best_mae = mae
                best_alpha = float(alpha)
        alphas[hour] = best_alpha
    return alphas


def train_models(data: pd.DataFrame, feature_cols: list[str]) -> dict[str, Any]:
    train = data[data["split"] == "train"]
    val = data[data["split"] == "validation"]
    test = data[data["split"] == "test"]

    x_train = train[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_train = train["residual_target"].to_numpy(float)
    weights = np.ones(len(train), dtype=float)
    weights[train["target_regime"] == "spike_cap"] *= 3.0
    weights[train["target_regime"] == "negative_zero_pressure"] *= 1.5

    global_model = LGBMRegressor(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        verbose=-1,
    )
    global_model.fit(x_train, y_train, sample_weight=weights)

    hour_models: dict[int, LGBMRegressor] = {}
    for hour in range(24):
        mask = train["hour"].astype(int) == hour
        if mask.sum() < 200:
            continue
        model = LGBMRegressor(
            n_estimators=400,
            learning_rate=0.04,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42 + hour,
            verbose=-1,
        )
        model.fit(x_train.loc[mask], y_train[mask])
        hour_models[hour] = model

    def predict_split(split_frame: pd.DataFrame) -> np.ndarray:
        x = split_frame[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        global_res = global_model.predict(x)
        blended = global_res.copy()
        for hour, model in hour_models.items():
            mask = split_frame["hour"].astype(int).to_numpy() == hour
            if mask.any():
                blended[mask] = 0.35 * global_res[mask] + 0.65 * model.predict(x.loc[mask])
        return blended

    val_res = predict_split(val)
    val_alpha = tune_hour_shrinkage(
        val,
        val_res,
        val["anchor_d1_ptf"].to_numpy(float),
        val["price"].to_numpy(float),
    )

    def final_predict(split_frame: pd.DataFrame, alphas: dict[int, float]) -> np.ndarray:
        raw = predict_split(split_frame)
        anchor = split_frame["anchor_d1_ptf"].to_numpy(float)
        hours = split_frame["hour"].astype(int).to_numpy()
        alpha_vec = np.array([alphas.get(int(h), 0.5) for h in hours], dtype=float)
        pred = anchor + alpha_vec * raw
        return np.clip(pred, 0, 5000)

    val_pred = final_predict(val, val_alpha)
    test_pred = final_predict(test, val_alpha) if not test.empty else np.array([])
    persistence_val = val["anchor_d1_ptf"].to_numpy(float)
    persistence_test = test["anchor_d1_ptf"].to_numpy(float) if not test.empty else np.array([])

    return {
        "global_model": global_model,
        "hour_models": hour_models,
        "hour_alphas": val_alpha,
        "feature_cols": feature_cols,
        "evaluation": {
            "validation": {
                "model": metrics(val["price"].to_numpy(float), val_pred),
                "persistence": metrics(val["price"].to_numpy(float), persistence_val),
            },
            "test": {
                "model": metrics(test["price"].to_numpy(float), test_pred) if len(test_pred) else None,
                "persistence": metrics(test["price"].to_numpy(float), persistence_test) if len(persistence_test) else None,
            },
            "validation_by_hour_group": _group_metrics(val, val_pred),
        },
    }


def _group_metrics(frame: pd.DataFrame, pred: np.ndarray) -> list[dict[str, Any]]:
    work = frame.copy()
    work["pred"] = pred
    rows = []
    for group, grp in work.groupby("hour_group", observed=False):
        rows.append({"hour_group": str(group), **metrics(grp["price"].to_numpy(float), grp["pred"].to_numpy(float))})
    return rows


def save_artifacts(result: dict[str, Any]) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(result["global_model"], MODEL_DIR / "global_model.joblib")
    joblib.dump(result["hour_models"], MODEL_DIR / "hour_models.joblib")
    (MODEL_DIR / "hour_alphas.json").write_text(json.dumps(result["hour_alphas"], indent=2) + "\n")
    (MODEL_DIR / "feature_columns.json").write_text(json.dumps(result["feature_cols"], indent=2) + "\n")


def write_reports(result: dict[str, Any]) -> None:
    val_mae = result["evaluation"]["validation"]["model"]["mae"]
    test_mae = result["evaluation"]["test"]["model"]
    test_mae_val = test_mae["mae"] if test_mae else None
    target_met_val = val_mae <= 100
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": "train_d2_ptf_forecaster.py",
        "target_mae_tl": 100,
        "target_met_validation": target_met_val,
        "evaluation": result["evaluation"],
        "hour_alphas": result["hour_alphas"],
    }
    METRICS_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# D+2 PTF Forecaster Metrics",
        "",
        f"Generated: `{payload['generated_at']}`",
        "",
        f"- Validation MAE: `{val_mae:.2f}` TL/MWh",
        f"- Test MAE: `{test_mae_val:.2f}` TL/MWh" if test_mae_val is not None else "- Test MAE: n/a",
        f"- Target (100 TL) met on validation: `{target_met_val}`",
        "",
        "## Validation vs Persistence",
        "",
        f"- Model: `{result['evaluation']['validation']['model']}`",
        f"- Persistence (D+1 anchor): `{result['evaluation']['validation']['persistence']}`",
    ]
    METRICS_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-save", action="store_true")
    args = parser.parse_args()

    data = load_training_frame()
    feature_cols = feature_columns(data)
    result = train_models(data, feature_cols)
    if not args.skip_save:
        save_artifacts(result)
    write_reports(result)
    val_mae = result["evaluation"]["validation"]["model"]["mae"]
    print(f"Validation MAE: {val_mae:.2f} TL")
    print(f"Wrote {METRICS_JSON}")


if __name__ == "__main__":
    main()
