#!/usr/bin/env python3
"""Horizon-wise tree models for 24h-ahead PTF (LightGBM > XGBoost > HistGradientBoosting)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Protocol

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "lstm_next24_v1.parquet"
MODEL_DIR = PROJECT_ROOT / "models" / "tree_horizon"
ANCHOR_TEST_PATH = PROJECT_ROOT / "data" / "model" / "anchor_test.csv"
PREDICTIONS_CSV = PROJECT_ROOT / "data" / "predictions" / "tree_test_predictions.csv"
METRICS_JSON = PROJECT_ROOT / "reports" / "tree_baseline_metrics.json"
METRICS_MD = PROJECT_ROOT / "reports" / "tree_baseline_metrics.md"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
RESIDUAL_PRED_CSV = PROJECT_ROOT / "data" / "predictions" / "lstm_residual_test_predictions.csv"
PERSISTENCE_METRICS_JSON = PROJECT_ROOT / "reports" / "persistence_metrics.json"
RESIDUAL_METRICS_JSON = PROJECT_ROOT / "reports" / "lstm_residual_metrics.json"

HORIZONS = list(range(1, 25))
MAPE_MASK_THRESHOLD = 100.0
EARLY_STOPPING_ROUNDS = 50
MAX_BOOST_ROUNDS = 2000


class HorizonModel(Protocol):
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> None: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def save(self, path: Path) -> None: ...


def pick_backend() -> str:
    try:
        import lightgbm  # noqa: F401

        return "lightgbm"
    except ImportError:
        pass
    try:
        import xgboost  # noqa: F401

        return "xgboost"
    except ImportError:
        pass
    return "sklearn"


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mape_masked(actual: np.ndarray, pred: np.ndarray, threshold: float) -> float:
    mask = np.abs(actual) > threshold
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100)


def resolve_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    target_cols = sorted(
        [c for c in df.columns if c.startswith("target_") and not c.startswith("target_residual_")],
        key=lambda c: int(c.replace("target_", "").replace("h", "")),
    )
    exclude = {"ts_hour", "split", *target_cols}
    exclude.update(c for c in df.columns if c.startswith("persistence_"))
    base_feature_cols = [c for c in df.columns if c not in exclude]
    return base_feature_cols, target_cols


def horizon_feature_cols(base_feature_cols: list[str], horizon: int) -> list[str]:
    return base_feature_cols + [f"persistence_{horizon}h"]


def add_persistence_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("ts_hour").reset_index(drop=True)
    for h in HORIZONS:
        tcol = f"target_{h}h"
        out[f"persistence_{h}h"] = out[tcol].shift(24)
    return out


def load_splits(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    persistence_col: str,
    *,
    residual_target: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cols = feature_cols + [target_col, persistence_col]
    train = df[df["split"] == "train"].dropna(subset=cols)
    val = df[df["split"] == "validation"].dropna(subset=cols)
    test = df[df["split"] == "test"].dropna(subset=cols)

    X_train = train[feature_cols].to_numpy(dtype=np.float64)
    y_train = train[target_col].to_numpy(dtype=np.float64)
    X_val = val[feature_cols].to_numpy(dtype=np.float64)
    y_val = val[target_col].to_numpy(dtype=np.float64)
    X_test = test[feature_cols].to_numpy(dtype=np.float64)
    y_test = test[target_col].to_numpy(dtype=np.float64)

    if residual_target:
        y_train = y_train - train[persistence_col].to_numpy(dtype=np.float64)
        y_val = y_val - val[persistence_col].to_numpy(dtype=np.float64)

    return X_train, y_train, X_val, y_val, X_test, y_test


class LightGBMHorizonModel:
    def __init__(self) -> None:
        import lightgbm as lgb

        self._lgb = lgb
        self._model: Any = None

    def fit(self, X_train, y_train, X_val, y_val) -> None:
        lgb = self._lgb
        train_set = lgb.Dataset(X_train, label=y_train)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
        params = {
            "objective": "regression",
            "metric": "mae",
            "verbosity": -1,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_data_in_leaf": 80,
            "lambda_l1": 0.1,
            "lambda_l2": 1.0,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "seed": 42,
        }
        self._model = lgb.train(
            params,
            train_set,
            num_boost_round=MAX_BOOST_ROUNDS,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS), lgb.log_evaluation(0)],
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(path))


class XGBoostHorizonModel:
    def __init__(self) -> None:
        import xgboost as xgb

        self._xgb = xgb
        self._model: Any = None

    def fit(self, X_train, y_train, X_val, y_val) -> None:
        xgb = self._xgb
        self._model = xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="mae",
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.9,
            n_estimators=MAX_BOOST_ROUNDS,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            random_state=42,
            n_jobs=-1,
        )
        self._model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path)


class SklearnHorizonModel:
    def __init__(self) -> None:
        from sklearn.ensemble import HistGradientBoostingRegressor

        self._cls = HistGradientBoostingRegressor
        self._model: Any = None
        self._best_iter = 200

    def fit(self, X_train, y_train, X_val, y_val) -> None:
        best_mae = float("inf")
        best_model = None
        stale = 0
        step = 50

        for n_iter in range(step, MAX_BOOST_ROUNDS + 1, step):
            model = self._cls(
                max_iter=n_iter,
                learning_rate=0.05,
                max_depth=12,
                early_stopping=False,
                random_state=42,
            )
            model.fit(X_train, y_train)
            val_pred = model.predict(X_val)
            val_mae = mae(y_val, val_pred)

            if val_mae < best_mae:
                best_mae = val_mae
                best_model = model
                self._best_iter = n_iter
                stale = 0
            else:
                stale += 1
                if stale >= EARLY_STOPPING_ROUNDS // step:
                    break

        self._model = best_model

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path)


def create_model(backend: str) -> HorizonModel:
    if backend == "lightgbm":
        return LightGBMHorizonModel()
    if backend == "xgboost":
        return XGBoostHorizonModel()
    return SklearnHorizonModel()


def model_extension(backend: str) -> str:
    return ".txt" if backend == "lightgbm" else ".joblib"


def build_test_predictions(
    df: pd.DataFrame,
    base_feature_cols: list[str],
    target_cols: list[str],
    backend: str,
    *,
    smoke: bool = False,
    residual_target: bool = True,
) -> pd.DataFrame:
    test_df = df[df["split"] == "test"].copy()
    rows: list[dict] = []

    horizons = HORIZONS[:2] if smoke else HORIZONS
    for h in horizons:
        tcol = target_cols[h - 1]
        pcol = f"persistence_{h}h"
        feature_cols = horizon_feature_cols(base_feature_cols, h)
        model_path = MODEL_DIR / f"horizon_{h:02d}{model_extension(backend)}"

        valid = test_df.dropna(subset=feature_cols + [tcol, pcol])
        X = valid[feature_cols].to_numpy(dtype=np.float64)

        if backend == "lightgbm":
            import lightgbm as lgb

            booster = lgb.Booster(model_file=str(model_path))
            pred = booster.predict(X)
        else:
            model = joblib.load(model_path)
            pred = model.predict(X)

        for i, idx in enumerate(valid.index):
            row = valid.loc[idx]
            actual = float(row[tcol])
            persistence = float(row[pcol])
            predicted = float(pred[i]) + (persistence if residual_target else 0.0)
            rows.append(
                {
                    "anchor_ts_hour": str(row["ts_hour"]),
                    "target_hour": h,
                    "actual_price": actual,
                    "persistence_price": persistence,
                    "predicted_price": predicted,
                    "absolute_error": abs(actual - predicted),
                    "persistence_error": abs(actual - persistence),
                }
            )

    return pd.DataFrame(rows)


def filter_aligned(df: pd.DataFrame) -> pd.DataFrame:
    if not ANCHOR_TEST_PATH.exists():
        return df
    anchors = pd.read_csv(ANCHOR_TEST_PATH)
    anchors["anchor_ts_hour"] = pd.to_datetime(
        anchors["anchor_ts_hour"], utc=True
    ).dt.tz_convert("Europe/Istanbul")
    out = df.copy()
    out["anchor_ts_hour"] = pd.to_datetime(out["anchor_ts_hour"], utc=True).dt.tz_convert(
        "Europe/Istanbul"
    )
    return out.merge(anchors[["anchor_ts_hour"]], on="anchor_ts_hour", how="inner")


def compute_metrics(pred_df: pd.DataFrame) -> dict[str, Any]:
    actual = pred_df["actual_price"].to_numpy()
    pred = pred_df["predicted_price"].to_numpy()
    persistence = pred_df["persistence_price"].to_numpy()
    zero_mask = actual == 0

    h_mae = (
        pred_df.assign(_err=(pred_df["actual_price"] - pred_df["predicted_price"]).abs())
        .groupby("target_hour")["_err"]
        .mean()
    )
    horizon_mae = {str(int(h)): float(v) for h, v in h_mae.items()}
    worst_h = max(horizon_mae, key=horizon_mae.get)

    persistence_mae = mae(actual, persistence)
    final_mae = mae(actual, pred)
    improvement_pct = (persistence_mae - final_mae) / persistence_mae * 100

    return {
        "test_samples_anchors": int(pred_df["anchor_ts_hour"].nunique()),
        "prediction_rows": int(len(pred_df)),
        "mae": final_mae,
        "rmse": rmse(actual, pred),
        "masked_mape_actual_gt_100": mape_masked(actual, pred, MAPE_MASK_THRESHOLD),
        "zero_price_mae": mae(actual[zero_mask], pred[zero_mask]) if zero_mask.any() else None,
        "zero_price_hours": int(zero_mask.sum()),
        "horizon_mae": horizon_mae,
        "worst_horizon": int(worst_h),
        "worst_horizon_mae": horizon_mae[worst_h],
        "persistence_comparison": {
            "persistence_mae": persistence_mae,
            "tree_mae": final_mae,
            "mae_delta_tree_minus_persistence": final_mae - persistence_mae,
            "improvement_pct_vs_persistence": improvement_pct,
            "tree_better_than_persistence": final_mae < persistence_mae,
        },
    }


def residual_mae_on_subset(aligned_df: pd.DataFrame) -> float | None:
    if not RESIDUAL_PRED_CSV.exists():
        return None
    residual = pd.read_csv(RESIDUAL_PRED_CSV)
    residual["anchor_ts_hour"] = pd.to_datetime(
        residual["anchor_ts_hour"], utc=True
    ).dt.tz_convert("Europe/Istanbul")
    merged = aligned_df.merge(
        residual[["anchor_ts_hour", "target_hour", "predicted_price"]],
        on=["anchor_ts_hour", "target_hour"],
        how="inner",
        suffixes=("", "_residual"),
    )
    if merged.empty:
        return None
    return mae(
        merged["actual_price"].to_numpy(),
        merged["predicted_price_residual"].to_numpy(),
    )


def plot_horizon_mae(horizon_errors: dict[str, float], path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hours = sorted(horizon_errors.keys(), key=int)
    values = [horizon_errors[h] for h in hours]
    plt.figure(figsize=(9, 5))
    plt.bar([int(h) for h in hours], values, color="forestgreen")
    plt.xlabel("Forecast horizon (hours)")
    plt.ylabel("MAE (TL/MWh)")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_vs_baselines(
    persistence_mae: float,
    tree_mae: float,
    residual_mae: float | None,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["Persistence", "Tree"]
    values = [persistence_mae, tree_mae]
    colors = ["#6baed6", "#31a354"]
    if residual_mae is not None:
        labels.insert(1, "LSTM residual")
        values.insert(1, residual_mae)
        colors.insert(1, "#2171b5")

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=colors)
    plt.ylabel("Test MAE (TL/MWh)")
    plt.title("Test MAE — tree vs baselines (LSTM-anchor aligned)")
    plt.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.1f}",
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def write_metrics_md(report: dict) -> str:
    full = report["full_test_split"]
    aligned = report["lstm_anchor_aligned"]
    cmp_a = aligned["persistence_comparison"]
    lines = [
        "# Tree Horizon Baseline Metrics",
        "",
        f"- **Backend:** {report['backend']}",
        f"- **Training target:** {'residual (target − persistence)' if report.get('residual_target_training') else 'direct target_h'}",
        f"- **Features:** `{report['features_path']}`",
        f"- **Feature count (per horizon):** {report.get('feature_count_per_horizon', report.get('feature_count', 'n/a'))}",
        "",
        "## Full test split",
        "",
        f"- MAE: {full['mae']:.4f}",
        f"- RMSE: {full['rmse']:.4f}",
        f"- MAPE (actual > {MAPE_MASK_THRESHOLD}): {full['masked_mape_actual_gt_100']:.4f}%",
        f"- Zero-price MAE: {full.get('zero_price_mae')}",
        f"- vs persistence: {full['persistence_comparison']['improvement_pct_vs_persistence']:.2f}%",
        "",
        "## LSTM-anchor aligned (compare to persistence 545.81 / residual 535.97)",
        "",
        f"- MAE: {aligned['mae']:.4f}",
        f"- RMSE: {aligned['rmse']:.4f}",
        f"- Persistence MAE: {cmp_a['persistence_mae']:.4f}",
        f"- Improvement vs persistence: {cmp_a['improvement_pct_vs_persistence']:.2f}%",
        f"- Residual LSTM MAE (same subset): {aligned.get('residual_lstm_mae', 'n/a')}",
        f"- Improvement vs residual LSTM: {aligned.get('improvement_pct_vs_residual_lstm', 'n/a')}",
        "",
        f"- **Tree beats persistence (aligned):** {cmp_a['tree_better_than_persistence']}",
        f"- **Tree beats residual LSTM (aligned):** {aligned.get('tree_better_than_residual_lstm')}",
        "",
        "## Horizon MAE (aligned)",
        "",
        "| Hour | MAE |",
        "|-----:|----:|",
    ]
    for h, v in sorted(aligned["horizon_mae"].items(), key=lambda x: int(x[0])):
        lines.append(f"| {h} | {v:.4f} |")
    return "\n".join(lines)


def train_all(*, smoke: bool = False, residual_target: bool = True) -> dict:
    backend = pick_backend()
    print(f"Backend: {backend}")

    df = pd.read_parquet(FEATURES_PATH)
    df = add_persistence_columns(df)
    base_feature_cols, target_cols = resolve_columns(df)
    print(f"Base features: {len(base_feature_cols)} (+ persistence_h per horizon)")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    horizons = HORIZONS[:2] if smoke else HORIZONS
    ext = model_extension(backend)
    training_log: list[dict] = []

    t0_all = time.time()
    for h in horizons:
        tcol = target_cols[h - 1]
        pcol = f"persistence_{h}h"
        feature_cols = horizon_feature_cols(base_feature_cols, h)
        X_train, y_train, X_val, y_val, _, _ = load_splits(
            df, feature_cols, tcol, pcol, residual_target=residual_target
        )

        model = create_model(backend)
        t0 = time.time()
        model.fit(X_train, y_train, X_val, y_val)
        out_path = MODEL_DIR / f"horizon_{h:02d}{ext}"
        model.save(out_path)

        val_pred = model.predict(X_val)
        training_log.append(
            {
                "horizon": h,
                "val_mae": mae(y_val, val_pred),
                "train_rows": len(y_train),
                "val_rows": len(y_val),
                "seconds": round(time.time() - t0, 2),
            }
        )
        print(f"h{h:02d} val_mae={training_log[-1]['val_mae']:.2f} ({training_log[-1]['seconds']}s)")

    pred_df = build_test_predictions(
        df,
        base_feature_cols,
        target_cols,
        backend,
        smoke=smoke,
        residual_target=residual_target,
    )
    PREDICTIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(PREDICTIONS_CSV, index=False)

    full_metrics = compute_metrics(pred_df)
    aligned_df = filter_aligned(pred_df)
    aligned_metrics = compute_metrics(aligned_df)

    residual_aligned = residual_mae_on_subset(aligned_df)
    if residual_aligned is not None:
        aligned_metrics["residual_lstm_mae"] = residual_aligned
        tree_mae = aligned_metrics["mae"]
        aligned_metrics["improvement_pct_vs_residual_lstm"] = (
            (residual_aligned - tree_mae) / residual_aligned * 100
        )
        aligned_metrics["tree_better_than_residual_lstm"] = tree_mae < residual_aligned

    ref_persistence = None
    ref_residual = None
    if PERSISTENCE_METRICS_JSON.exists():
        ref_persistence = json.loads(PERSISTENCE_METRICS_JSON.read_text())["mae"]
    if RESIDUAL_METRICS_JSON.exists():
        ref_residual = json.loads(RESIDUAL_METRICS_JSON.read_text())["test_mae"]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_horizon_mae(
        aligned_metrics["horizon_mae"],
        FIGURES_DIR / "tree_horizon_mae.png",
        "Tree horizon model — MAE by horizon (LSTM-anchor aligned)",
    )
    plot_vs_baselines(
        aligned_metrics["persistence_comparison"]["persistence_mae"],
        aligned_metrics["mae"],
        residual_aligned,
        FIGURES_DIR / "tree_vs_persistence.png",
    )

    (MODEL_DIR / "training_metadata.json").write_text(
        json.dumps(
            {
                "backend": backend,
                "base_feature_columns": base_feature_cols,
                "target_columns": target_cols,
                "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
                "residual_target": residual_target,
                "final_prediction": "persistence_h + predicted_residual"
                if residual_target
                else "predicted_price_direct",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    aligned_cmp = aligned_metrics["persistence_comparison"]
    report = {
        "backend": backend,
        "features_path": str(FEATURES_PATH),
        "feature_count_base": len(base_feature_cols),
        "feature_count_per_horizon": len(base_feature_cols) + 1,
        "base_feature_columns": base_feature_cols,
        "uses_persistence_as_feature": True,
        "residual_target_training": residual_target,
        "training_seconds_total": round(time.time() - t0_all, 2),
        "training_log": training_log,
        "model_dir": str(MODEL_DIR),
        "predictions_path": str(PREDICTIONS_CSV),
        "full_test_split": full_metrics,
        "lstm_anchor_aligned": aligned_metrics,
        "reference_metrics": {
            "persistence_mae_reported": ref_persistence,
            "residual_lstm_mae_reported": ref_residual,
        },
        "beats_persistence_545": aligned_cmp["tree_better_than_persistence"],
        "beats_residual_536": aligned_metrics.get("tree_better_than_residual_lstm"),
        "smoke_test": smoke,
    }

    METRICS_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    METRICS_MD.write_text(write_metrics_md(report), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train horizon-wise tree PTF models")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Train only h1-h2 for a quick sanity check",
    )
    parser.add_argument(
        "--direct-target",
        action="store_true",
        help="Train on raw target_h (default: residual = target_h - persistence_h)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = train_all(smoke=args.smoke_test, residual_target=not args.direct_target)
    aligned = report["lstm_anchor_aligned"]
    cmp_ = aligned["persistence_comparison"]
    print("\n=== Tree horizon training complete ===")
    print(f"Backend: {report['backend']}")
    print(f"Full test MAE: {report['full_test_split']['mae']:.4f}")
    print(f"Aligned MAE:   {aligned['mae']:.4f}")
    print(f"Persistence:   {cmp_['persistence_mae']:.4f} (improvement {cmp_['improvement_pct_vs_persistence']:.2f}%)")
    if "residual_lstm_mae" in aligned:
        print(f"Residual LSTM: {aligned['residual_lstm_mae']:.4f}")
    print(f"Metrics: {METRICS_JSON}")


if __name__ == "__main__":
    main()
