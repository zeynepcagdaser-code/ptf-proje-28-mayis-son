#!/usr/bin/env python3
"""
Train horizon-wise low-price classifiers (PTF <= 50 TL/MWh).

Uses tabular features from data/model_low_price_tabular/.
Does NOT train LSTM or main regression models.
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_sample_weight

PROJECT_ROOT = Path(__file__).resolve().parent
TABULAR_DIR = PROJECT_ROOT / "data" / "model_low_price_tabular"
MODEL_PATH = PROJECT_ROOT / "models" / "low_price_classifier_horizon_models.pkl"
PRED_PATH = PROJECT_ROOT / "data" / "predictions" / "low_price_probabilities.csv"
REPORT_JSON = PROJECT_ROOT / "reports" / "low_price_classifier_metrics.json"
REPORT_MD = PROJECT_ROOT / "reports" / "low_price_classifier_metrics.md"

META_COLUMNS = {"sample_index", "anchor_ts_hour", "split"}
HORIZONS = list(range(1, 25))
THRESHOLD_GRID = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
RECALL_TARGET_PRIMARY = 0.70
RECALL_TARGET_SECONDARY = 0.80


def _horizon_label(h: int) -> str:
    return f"is_low_{h}h"


def _zero_label(h: int) -> str:
    return f"is_zero_{h}h"


def _load_split(split: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    key = {"train": "train", "validation": "val", "test": "test"}[split]
    X = pd.read_parquet(TABULAR_DIR / f"X_{key}.parquet")
    y_low = pd.read_parquet(TABULAR_DIR / f"y_low_{key}.parquet")
    y_zero = None
    if split == "test":
        y_zero = pd.read_parquet(TABULAR_DIR / "y_zero_test.parquet")
    return X, y_low, y_zero


def _align_frames(
    X: pd.DataFrame,
    y_low: pd.DataFrame,
    y_zero: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    if "sample_index" in y_low.columns:
        y_low = y_low.set_index("sample_index")
    if y_zero is not None and "sample_index" in y_zero.columns:
        y_zero = y_zero.set_index("sample_index")

    idx = X["sample_index"].to_numpy()
    y_low = y_low.loc[idx].reset_index(drop=True)
    if y_zero is not None:
        y_zero = y_zero.loc[idx].reset_index(drop=True)
    return X, y_low, y_zero


def _feature_matrix(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    feature_cols = [c for c in X.columns if c not in META_COLUMNS]
    features = X[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return features, feature_cols


def _make_classifier(backend: str, y_train: np.ndarray):
    if backend == "random_forest":
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=20,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )

    if backend == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError("lightgbm not installed") from exc
        pos = max(int(y_train.sum()), 1)
        neg = max(int((1 - y_train).sum()), 1)
        return LGBMClassifier(
            objective="binary",
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=40,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=neg / pos,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )

    if backend == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("xgboost not installed") from exc
        pos = max(int(y_train.sum()), 1)
        neg = max(int((1 - y_train).sum()), 1)
        return XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=neg / pos,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

    # default
    return HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.05,
        max_depth=None,
        min_samples_leaf=40,
        class_weight="balanced",
        random_state=42,
    )


def _fit_model(model: Any, X_train: np.ndarray, y_train: np.ndarray, backend: str) -> Any:
    if backend == "hist_gradient_boosting":
        model.fit(X_train, y_train)
        return model
    if backend == "random_forest":
        model.fit(X_train, y_train)
        return model
    # tree boosters with optional sample_weight fallback
    sw = compute_sample_weight(class_weight="balanced", y=y_train)
    model.fit(X_train, y_train, sample_weight=sw)
    return model


def _predict_proba(model: Any, X: np.ndarray, backend: str) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    # HistGradientBoosting returns proba
    return model.predict_proba(X)[:, 1]


def _binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    y_true = y_true.astype(int)
    y_pred = (y_prob >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))
    out: dict[str, Any] = {
        "threshold": float(threshold),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "positive_rate": float(y_true.mean()) if len(y_true) else None,
        "pred_positive_rate": float(y_pred.mean()) if len(y_pred) else None,
        "rows": int(len(y_true)),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }
    if len(np.unique(y_true)) > 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        out["roc_auc"] = None
        out["pr_auc"] = None
    return out


def _pick_best_precision_at_recall(
    grid: list[dict[str, Any]],
    min_recall: float,
) -> dict[str, Any] | None:
    candidates = [row for row in grid if row["recall"] >= min_recall]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (row["precision"], row["recall"], -row["threshold"]),
    )


def _threshold_search(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    grid: list[dict[str, Any]] = []
    for thr in THRESHOLD_GRID:
        m = _binary_metrics(y_true, y_prob, thr)
        grid.append(
            {
                "threshold": thr,
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "roc_auc": m.get("roc_auc"),
                "pr_auc": m.get("pr_auc"),
            }
        )

    max_f1_row = max(grid, key=lambda row: (row["f1"], row["recall"]))
    recall_70 = _pick_best_precision_at_recall(grid, RECALL_TARGET_PRIMARY)
    recall_80 = _pick_best_precision_at_recall(grid, RECALL_TARGET_SECONDARY)

    if recall_70 is not None:
        selected = recall_70
        selection_policy = f"recall>={RECALL_TARGET_PRIMARY:.2f} with max precision"
    else:
        selected = max(grid, key=lambda row: (row["recall"], row["precision"], -row["threshold"]))
        selection_policy = "max validation recall (no threshold met recall target)"

    selected_threshold = float(selected["threshold"])
    return {
        "grid": grid,
        "max_f1_threshold": float(max_f1_row["threshold"]),
        "max_f1_metrics": max_f1_row,
        "recall_ge_0.70_best_precision": recall_70,
        "recall_ge_0.80_best_precision": recall_80,
        "selected_threshold": selected_threshold,
        "selection_policy": selection_policy,
        "selected_metrics": _binary_metrics(y_true, y_prob, selected_threshold),
    }


def _zero_capture_metrics(
    y_zero: np.ndarray,
    y_low_pred: np.ndarray,
    y_low_prob: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Among actual zero hours, how often did the low classifier fire?"""
    mask = y_zero.astype(int) == 1
    n_zero = int(mask.sum())
    if n_zero == 0:
        return {
            "zero_rows": 0,
            "zero_recall_pred_low": None,
            "zero_recall_prob_ge_threshold": None,
            "mean_prob_on_zero": None,
        }
    return {
        "zero_rows": n_zero,
        "zero_recall_pred_low": float(y_low_pred[mask].mean()),
        "zero_recall_prob_ge_threshold": float((y_low_prob[mask] >= threshold).mean()),
        "mean_prob_on_zero": float(y_low_prob[mask].mean()),
    }


def train_low_price_classifiers(*, backend: str = "hist_gradient_boosting") -> dict[str, Any]:
    X_train_df, y_train_df, _ = _load_split("train")
    X_val_df, y_val_df, _ = _load_split("validation")
    X_test_df, y_test_df, y_zero_test_df = _load_split("test")

    X_train_df, y_train_df, _ = _align_frames(X_train_df, y_train_df, None)
    X_val_df, y_val_df, _ = _align_frames(X_val_df, y_val_df, None)
    X_test_df, y_test_df, y_zero_test_df = _align_frames(
        X_test_df, y_test_df, y_zero_test_df
    )

    X_train, feature_cols = _feature_matrix(X_train_df)
    X_val, _ = _feature_matrix(X_val_df)
    X_test, _ = _feature_matrix(X_test_df)

    X_train_np = X_train.to_numpy(dtype=np.float32)
    X_val_np = X_val.to_numpy(dtype=np.float32)
    X_test_np = X_test.to_numpy(dtype=np.float32)

    horizon_models: dict[int, Any] = {}
    horizon_report: dict[int, Any] = {}
    pred_frames: list[pd.DataFrame] = []

    for h in HORIZONS:
        label = _horizon_label(h)
        y_tr = y_train_df[label].to_numpy(dtype=int)
        y_va = y_val_df[label].to_numpy(dtype=int)
        y_te = y_test_df[label].to_numpy(dtype=int)
        y_ze = y_zero_test_df[_zero_label(h)].to_numpy(dtype=int)

        model = _make_classifier(backend, y_tr)
        model = _fit_model(model, X_train_np, y_tr, backend)

        val_prob = _predict_proba(model, X_val_np, backend)
        test_prob = _predict_proba(model, X_test_np, backend)

        thr_info = _threshold_search(y_va, val_prob)
        thr = thr_info["selected_threshold"]

        test_metrics = _binary_metrics(y_te, test_prob, thr)
        val_metrics = thr_info["selected_metrics"]

        test_pred = (test_prob >= thr).astype(int)
        zero_metrics = _zero_capture_metrics(y_ze, test_pred, test_prob, thr)

        horizon_models[h] = {
            "model": model,
            "threshold": thr,
            "label": label,
        }
        horizon_report[h] = {
            "horizon": h,
            "train_positive_rate": float(y_tr.mean()),
            "validation_threshold_search": thr_info,
            "validation_at_selected_threshold": val_metrics,
            "test_at_selected_threshold": test_metrics,
            "test_zero_capture": zero_metrics,
        }

        for split_name, X_df, y_low_df, probs, y_z_df in (
            ("train", X_train_df, y_train_df, _predict_proba(model, X_train_np, backend), None),
            ("validation", X_val_df, y_val_df, val_prob, None),
            ("test", X_test_df, y_test_df, test_prob, y_zero_test_df),
        ):
            frame = pd.DataFrame(
                {
                    "sample_index": X_df["sample_index"].astype(int),
                    "anchor_ts_hour": pd.to_datetime(X_df["anchor_ts_hour"], utc=True),
                    "split": split_name,
                    "horizon": h,
                    "low_prob": probs,
                    "low_pred": (probs >= thr).astype(int),
                    "is_low_actual": y_low_df[label].astype(int),
                    "selected_threshold": thr,
                }
            )
            if y_z_df is not None:
                frame["is_zero_actual"] = y_z_df[_zero_label(h)].astype(int)
            else:
                frame["is_zero_actual"] = np.nan
            pred_frames.append(frame)

    pred_all = pd.concat(pred_frames, ignore_index=True)
    PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    pred_all.to_csv(PRED_PATH, index=False)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "backend": backend,
            "feature_columns": feature_cols,
            "horizons": HORIZONS,
            "threshold_grid": THRESHOLD_GRID,
            "horizon_models": horizon_models,
        },
        MODEL_PATH,
    )

    val_f1s = [horizon_report[h]["validation_at_selected_threshold"]["f1"] for h in HORIZONS]
    test_f1s = [horizon_report[h]["test_at_selected_threshold"]["f1"] for h in HORIZONS]
    test_recalls = [horizon_report[h]["test_at_selected_threshold"]["recall"] for h in HORIZONS]
    thresholds = [
        horizon_report[h]["validation_threshold_search"]["selected_threshold"] for h in HORIZONS
    ]
    max_f1_thresholds = [
        horizon_report[h]["validation_threshold_search"]["max_f1_threshold"] for h in HORIZONS
    ]
    zero_recalls = [
        horizon_report[h]["test_zero_capture"]["zero_recall_pred_low"]
        for h in HORIZONS
        if horizon_report[h]["test_zero_capture"]["zero_recall_pred_low"] is not None
    ]

    worst = sorted(HORIZONS, key=lambda h: horizon_report[h]["test_at_selected_threshold"]["f1"])[:5]
    best = sorted(HORIZONS, key=lambda h: horizon_report[h]["test_at_selected_threshold"]["f1"], reverse=True)[:5]

    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "feature_count": len(feature_cols),
        "threshold_grid": THRESHOLD_GRID,
        "horizons": HORIZONS,
        "summary": {
            "mean_validation_f1": float(np.mean(val_f1s)),
            "mean_test_f1": float(np.mean(test_f1s)),
            "mean_test_recall": float(np.mean(test_recalls)),
            "mean_selected_threshold": float(np.mean(thresholds)),
            "mean_max_f1_threshold": float(np.mean(max_f1_thresholds)),
            "mean_zero_recall_pred_low": float(np.mean(zero_recalls)) if zero_recalls else None,
            "threshold_selection_policy": "recall-first on validation",
            "worst_test_f1_horizons": [
                {
                    "horizon": h,
                    "test_f1": horizon_report[h]["test_at_selected_threshold"]["f1"],
                    "test_recall": horizon_report[h]["test_at_selected_threshold"]["recall"],
                }
                for h in worst
            ],
            "best_test_f1_horizons": [
                {
                    "horizon": h,
                    "test_f1": horizon_report[h]["test_at_selected_threshold"]["f1"],
                }
                for h in best
            ],
        },
        "per_horizon": {str(h): horizon_report[h] for h in HORIZONS},
        "outputs": {
            "model_path": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
            "predictions_path": str(PRED_PATH.relative_to(PROJECT_ROOT)),
        },
    }

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    REPORT_MD.write_text(_report_markdown(report), encoding="utf-8")
    report["report_json"] = str(REPORT_JSON)
    report["report_md"] = str(REPORT_MD)
    return report


def _report_markdown(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines = [
        "# Low-Price Classifier Metrics",
        "",
        f"- **Generated (UTC):** {report['generated_at_utc']}",
        f"- **Backend:** `{report['backend']}`",
        f"- **Features:** {report['feature_count']}",
        f"- **Threshold grid:** {report['threshold_grid']}",
        "",
        "## Summary",
        "",
        f"- Mean validation F1: **{s['mean_validation_f1']:.4f}**",
        f"- Mean test F1: **{s['mean_test_f1']:.4f}**",
        f"- Mean test recall: **{s['mean_test_recall']:.4f}**",
        f"- Mean selected threshold (recall-first): **{s['mean_selected_threshold']:.3f}**",
        f"- Mean max-F1 threshold: **{s.get('mean_max_f1_threshold', 0):.3f}**",
        f"- Mean zero-price recall (pred low | actual zero): **{s['mean_zero_recall_pred_low']:.4f}**"
        if s["mean_zero_recall_pred_low"] is not None
        else "- Mean zero-price recall: n/a",
        "",
        "## Threshold policy",
        "",
        f"- Selection: {s.get('threshold_selection_policy', 'recall-first')}",
        f"- Grid: {report['threshold_grid']}",
        "",
        "## Worst horizons (test F1)",
        "",
        "| h | test F1 | test recall |",
        "|--:|--------:|------------:|",
    ]
    for row in s["worst_test_f1_horizons"]:
        lines.append(f"| {row['horizon']} | {row['test_f1']:.4f} | {row['test_recall']:.4f} |")
    lines += [
        "",
        "## Per-horizon test metrics (selected validation threshold)",
        "",
        "| h | thr | precision | recall | F1 | ROC-AUC | PR-AUC | zero recall |",
        "|--:|----:|----------:|-------:|---:|--------:|-------:|------------:|",
    ]
    for h in report["horizons"]:
        ph = report["per_horizon"][str(h)]
        tm = ph["test_at_selected_threshold"]
        zm = ph["test_zero_capture"]
        roc = tm.get("roc_auc")
        pr = tm.get("pr_auc")
        zr = zm.get("zero_recall_pred_low")
        lines.append(
            f"| {h} | {ph['validation_threshold_search']['selected_threshold']:.2f} "
            f"| {tm['precision']:.4f} | {tm['recall']:.4f} | {tm['f1']:.4f} "
            f"| {'' if roc is None else f'{roc:.4f}'} "
            f"| {'' if pr is None else f'{pr:.4f}'} "
            f"| {'' if zr is None else f'{zr:.4f}'} |"
        )
    lines += [
        "",
        f"Model: `{report['outputs']['model_path']}`",
        f"Predictions: `{report['outputs']['predictions_path']}`",
    ]
    return "\n".join(lines) + "\n"


def _print_summary(report: dict[str, Any]) -> None:
    s = report["summary"]
    print("=== Low-Price Classifier Training Summary ===")
    print("Backend:", report["backend"])
    print("Features:", report["feature_count"])
    print("Mean validation F1:", f"{s['mean_validation_f1']:.4f}")
    print("Mean test F1:", f"{s['mean_test_f1']:.4f}")
    print("Mean test recall:", f"{s['mean_test_recall']:.4f}")
    print("Mean selected threshold (recall-first):", f"{s['mean_selected_threshold']:.3f}")
    print("Mean max-F1 threshold:", f"{s.get('mean_max_f1_threshold', 0):.3f}")
    if s["mean_zero_recall_pred_low"] is not None:
        print("Mean zero-price recall (pred low | actual zero):", f"{s['mean_zero_recall_pred_low']:.4f}")
    print("Worst horizons (test F1):")
    for row in s["worst_test_f1_horizons"]:
        print(f"  h{row['horizon']}: F1={row['test_f1']:.4f}, recall={row['test_recall']:.4f}")
    print("Saved model:", MODEL_PATH)
    print("Predictions:", PRED_PATH)
    print("Report:", report["report_md"])


def _default_backend() -> str:
    try:
        import lightgbm  # noqa: F401

        return "lightgbm"
    except ImportError:
        return "hist_gradient_boosting"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train low-price horizon classifiers.")
    parser.add_argument(
        "--backend",
        choices=[
            "hist_gradient_boosting",
            "random_forest",
            "lightgbm",
            "xgboost",
        ],
        default=None,
    )
    args = parser.parse_args()
    backend = args.backend or _default_backend()
    report = train_low_price_classifiers(backend=backend)
    _print_summary(report)


if __name__ == "__main__":
    main()
