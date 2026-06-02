#!/usr/bin/env python3
"""
Walk-forward (time series) validation runner.

Goals:
  - enforce chronological splits (no random split)
  - 5-fold walk-forward evaluation
  - report MAE / RMSE per fold + mean/std

Default dataset:
  data/master/master_hourly_v1.parquet

Default target:
  ptf_price

Leakage guardrails in this script:
  - strictly time-ordered splits (future rows never included in train)
  - drop obvious target-derived columns (all `ptf_` prefixed columns except target)

NOTE:
This script does NOT guarantee feature-level point-in-time safety for every column
in the master parquet (e.g., some columns may be realized ex-post). It only ensures
that evaluation uses strictly forward splits (no test->train leakage).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = PROJECT_ROOT / "data" / "master" / "master_hourly_v1.parquet"
DEFAULT_TIME_COL = "ts_hour"
DEFAULT_TARGET_COL = "ptf_price"
REPORT_JSON = PROJECT_ROOT / "reports" / "walk_forward_metrics.json"


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    return float(np.mean(np.abs(a - b)))


def resolve_feature_cols(df: pd.DataFrame, *, target_col: str) -> list[str]:
    # Drop time + target + obvious target-derived columns.
    drop = {target_col, DEFAULT_TIME_COL}
    # Many master tables contain ptf-derived flags/currency prices which would be direct leakage.
    drop.update(c for c in df.columns if c.startswith("ptf_") and c != target_col)

    # Keep numeric/bool only (LightGBM-friendly). Convert bool later.
    numeric_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    return [c for c in numeric_cols if c not in drop]


def make_folds(
    n: int,
    *,
    n_folds: int = 5,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
) -> list[dict[str, int]]:
    """
    5-fold walk-forward with a fixed global validation window size (val_frac of full data),
    and test window split into n_folds equal chunks.

    Fold k:
      train: [0, train_end)
      val:   [train_end, val_end)
      test:  [val_end, test_end)
    where train_end expands forward each fold.
    """
    if not (0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1):
        raise ValueError("Invalid train/val fractions.")
    base_train = int(n * train_frac)
    val_size = int(n * val_frac)
    remaining_test = n - (base_train + val_size)
    if remaining_test <= 0:
        raise ValueError(f"Not enough rows for test after train/val split (n={n}).")

    fold_test = max(1, remaining_test // n_folds)
    folds: list[dict[str, int]] = []
    for k in range(n_folds):
        train_end = base_train + k * fold_test
        val_end = train_end + val_size
        test_end = min(val_end + fold_test, n)
        if val_end >= n or test_end <= val_end:
            break
        folds.append(
            {
                "fold": k + 1,
                "train_start": 0,
                "train_end": train_end,
                "val_start": train_end,
                "val_end": val_end,
                "test_start": val_end,
                "test_end": test_end,
            }
        )
    if len(folds) < n_folds:
        raise ValueError(f"Could only construct {len(folds)}/{n_folds} folds with n={n}.")
    return folds


def train_lgbm_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    *,
    seed: int = 42,
) -> Any:
    import lightgbm as lgb

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    dval = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=dtrain)

    params: dict[str, Any] = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": seed,
    }

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    return booster


def run(
    *,
    data_path: Path,
    time_col: str,
    target_col: str,
    n_folds: int,
) -> dict[str, Any]:
    if not data_path.exists():
        raise FileNotFoundError(f"Missing dataset: {data_path}")

    df = pd.read_parquet(data_path)
    if time_col not in df.columns:
        raise KeyError(f"Missing time_col={time_col} in {data_path.name}")
    if target_col not in df.columns:
        raise KeyError(f"Missing target_col={target_col} in {data_path.name}")

    # Ensure strict chronological order.
    df = df.sort_values(time_col).reset_index(drop=True)

    # Resolve usable feature columns.
    feature_cols = resolve_feature_cols(df, target_col=target_col)
    if not feature_cols:
        raise ValueError("No usable numeric feature columns found after leakage guardrails.")

    # LightGBM wants numeric arrays. Convert bool -> int.
    X_all = df[feature_cols].copy()
    for c in X_all.columns:
        if X_all[c].dtype == bool:
            X_all[c] = X_all[c].astype(np.int8)

    # Drop rows with NA in target/features.
    keep = ~df[target_col].isna()
    keep &= ~X_all.isna().any(axis=1)
    df = df.loc[keep].reset_index(drop=True)
    X_all = X_all.loc[keep].reset_index(drop=True)

    n = len(df)
    folds = make_folds(n, n_folds=n_folds, train_frac=0.60, val_frac=0.20)

    fold_reports: list[dict[str, Any]] = []
    maes: list[float] = []
    rmses: list[float] = []

    for f in folds:
        tr = slice(f["train_start"], f["train_end"])
        va = slice(f["val_start"], f["val_end"])
        te = slice(f["test_start"], f["test_end"])

        X_train = X_all.iloc[tr].to_numpy(dtype=np.float64)
        y_train = df.iloc[tr][target_col].to_numpy(dtype=np.float64)
        X_val = X_all.iloc[va].to_numpy(dtype=np.float64)
        y_val = df.iloc[va][target_col].to_numpy(dtype=np.float64)
        X_test = X_all.iloc[te].to_numpy(dtype=np.float64)
        y_test = df.iloc[te][target_col].to_numpy(dtype=np.float64)

        booster = train_lgbm_regressor(X_train, y_train, X_val, y_val, feature_cols)
        pred = booster.predict(X_test, num_iteration=booster.best_iteration)

        fold_mae = mae(y_test, pred)
        fold_rmse = rmse(y_test, pred)
        maes.append(fold_mae)
        rmses.append(fold_rmse)

        fold_reports.append(
            {
                "fold": f["fold"],
                "sizes": {
                    "train_rows": int(len(y_train)),
                    "val_rows": int(len(y_val)),
                    "test_rows": int(len(y_test)),
                },
                "time_range": {
                    "train_start": str(df.iloc[f["train_start"]][time_col]),
                    "train_end_exclusive": str(df.iloc[f["train_end"] - 1][time_col]),
                    "val_start": str(df.iloc[f["val_start"]][time_col]),
                    "val_end_exclusive": str(df.iloc[f["val_end"] - 1][time_col]),
                    "test_start": str(df.iloc[f["test_start"]][time_col]),
                    "test_end_exclusive": str(df.iloc[f["test_end"] - 1][time_col]),
                },
                "metrics": {
                    "mae": fold_mae,
                    "rmse": fold_rmse,
                    "best_iteration": int(getattr(booster, "best_iteration", 0) or 0),
                },
            }
        )

    report = {
        "data_path": str(data_path),
        "time_col": time_col,
        "target_col": target_col,
        "n_rows_used": n,
        "n_features_used": len(feature_cols),
        "n_folds": len(fold_reports),
        "split_policy": "chronological_walk_forward_60_20_20",
        "leakage_checks": [
            "Rows are sorted by time_col and split chronologically.",
            "No test rows are included in train/val in any fold (index ranges are disjoint and forward).",
            "Dropped all ptf_* columns except target to avoid direct target-derived leakage.",
        ],
        "folds": fold_reports,
        "summary": {
            "mae_mean": float(np.mean(maes)),
            "mae_std": float(np.std(maes, ddof=1)) if len(maes) > 1 else 0.0,
            "rmse_mean": float(np.mean(rmses)),
            "rmse_std": float(np.std(rmses, ddof=1)) if len(rmses) > 1 else 0.0,
        },
    }

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="5-fold walk-forward validation (chronological) for PTF regression.")
    p.add_argument("--data-path", type=str, default=str(DEFAULT_DATA))
    p.add_argument("--time-col", type=str, default=DEFAULT_TIME_COL)
    p.add_argument("--target-col", type=str, default=DEFAULT_TARGET_COL)
    p.add_argument("--n-folds", type=int, default=5)
    args = p.parse_args()

    report = run(
        data_path=Path(args.data_path),
        time_col=args.time_col,
        target_col=args.target_col,
        n_folds=args.n_folds,
    )
    s = report["summary"]
    print("=== Walk-forward validation ===")
    print(f"Folds: {report['n_folds']}")
    print(f"MAE:  mean={s['mae_mean']:.2f}  std={s['mae_std']:.2f}")
    print(f"RMSE: mean={s['rmse_mean']:.2f}  std={s['rmse_std']:.2f}")
    print(f"Report: {REPORT_JSON}")


if __name__ == "__main__":
    main()

