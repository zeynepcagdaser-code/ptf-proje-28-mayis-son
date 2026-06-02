#!/usr/bin/env python3
"""
Perform Optuna hyperparameter search for the two-stage PTF regressor.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json

import joblib
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMRegressor

from src.utils.feature_utils import clean_and_engineer_features, feature_cols, make_regressor, target_column_name
from src.utils.io_utils import read_parquet_with_normalized_ts

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "lstm_next24_v1.parquet"
OUTPUT_JSON = PROJECT_ROOT / "reports" / "optuna_two_stage_ptf.json"
MODEL_DIR = PROJECT_ROOT / "models" / "realtime_two_stage_optuna"


def load_features() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        return pd.DataFrame()
    return read_parquet_with_normalized_ts(FEATURES_PATH)


def objective_for_horizon(
    trial: optuna.Trial,
    df: pd.DataFrame,
    horizon: int,
    objective: str,
    quantile_alpha: float | None,
) -> float:
    target_col = target_column_name(horizon)
    if target_col not in df.columns:
        raise ValueError(f"Missing target column {target_col}")

    base_col = f"ptf_lag_{24 + horizon}" if f"ptf_lag_{24 + horizon}" in df.columns else "ptf_lag_24"
    df = df.dropna(subset=[target_col, base_col]).reset_index(drop=True)
    df["delta_h"] = df[target_col] - df[base_col]
    df = df[df["split"] == "train"].copy()
    if df.empty:
        raise ValueError("No training rows for Optuna search")

    X = df[feature_cols(df)].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = df["delta_h"].fillna(0)
    if len(X) < 20:
        raise ValueError("Not enough rows for Optuna sampling")

    def create_regressor(trial: optuna.Trial) -> LGBMRegressor:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 5.0),
            "objective": objective,
            "random_state": 42,
        }
        if objective == "quantile" and quantile_alpha is not None:
            params["alpha"] = quantile_alpha
        return LGBMRegressor(**params)

    tscv = TimeSeriesSplit(n_splits=3)
    scores = []
    for train_idx, valid_idx in tscv.split(X):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        model = create_regressor(trial)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        scores.append(mean_absolute_error(y_valid, y_pred))

    return float(pd.Series(scores).mean())


def run_optuna(
    df: pd.DataFrame,
    horizon: int,
    objective: str,
    quantile_alpha: float | None,
    n_trials: int,
):
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: objective_for_horizon(trial, df, horizon, objective, quantile_alpha), n_trials=n_trials)
    best = study.best_params
    best["objective"] = objective
    if objective == "quantile" and quantile_alpha is not None:
        best["alpha"] = quantile_alpha
    best["horizon"] = horizon
    best["best_value"] = study.best_value
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(best, ensure_ascii=False, indent=2) + "\n")
    print(f"Optuna best params for horizon {horizon}: {best}")
    print(f"Saved best params to {OUTPUT_JSON}")

    model_df = df[df["split"] == "train"].copy()
    model_df["delta_h"] = model_df[target_column_name(horizon)] - model_df["ptf_lag_24"]
    X = model_df[feature_cols(model_df)].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = model_df["delta_h"].fillna(0)
    model = LGBMRegressor(**{k: v for k, v in best.items() if k not in {"horizon", "best_value"}})
    model.fit(X, y)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"best_two_stage_h{horizon}.pkl"
    joblib.dump(model, model_path)
    print(f"Saved best model to {model_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Optuna tuning for two-stage PTF regression")
    parser.add_argument("--horizon", type=int, default=1, help="Horizon to tune")
    parser.add_argument("--objective", choices=["regression", "huber", "quantile"], default="regression")
    parser.add_argument("--quantile-alpha", type=float, default=0.5, help="Quantile alpha when using quantile objective")
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials")
    return parser.parse_args()


def main():
    args = parse_args()
    df = load_features()
    if df.empty:
        print("No features parquet found; aborting.")
        return
    df = clean_and_engineer_features(df)
    if target_column_name(args.horizon) not in df.columns:
        raise RuntimeError(f"Missing target column for horizon {args.horizon}")
    run_optuna(df, args.horizon, args.objective, args.quantile_alpha, args.trials)


if __name__ == '__main__':
    main()
