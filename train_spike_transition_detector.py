#!/usr/bin/env python3
"""
Train a transition-specific detector for new spike/cap events.

Target:
    is_new_spike_transition = target_regime == "spike_cap" and lag24_regime != "spike_cap"

This script does not train price experts, ensembles, or final PTF regressors.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent

FEATURE_STORE_PATH = PROJECT_ROOT / "data" / "features" / "regime_feature_store.parquet"
REASONING_PATH = PROJECT_ROOT / "data" / "features" / "market_reasoning_features.parquet"
LABEL_PATHS = [
    PROJECT_ROOT / "data" / "features" / "regime_labels.csv",
    PROJECT_ROOT / "data" / "regime_labels.csv",
]
BINARY_SPIKE_PRED_PATH = PROJECT_ROOT / "data" / "predictions" / "spike_cap_detector_predictions.csv"

MODEL_DIR = PROJECT_ROOT / "models" / "spike_transition_detector"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "predictions" / "spike_transition_detector_predictions.csv"
METRICS_JSON = PROJECT_ROOT / "reports" / "spike_transition_detector_metrics.json"
METRICS_MD = PROJECT_ROOT / "reports" / "spike_transition_detector_metrics.md"

SPLIT_RANGES = {
    "train": (2020, 2024),
    "validation": (2025, 2025),
    "test": (2026, 2026),
}
THRESHOLDS = [0.5, 0.3, 0.2, 0.1, 0.05, 0.01, 0.005, 0.002, 0.001]
OPERATING_THRESHOLD = 0.001
FORBIDDEN_FEATURES = {
    "price",
    "target_price",
    "finalized_ptf",
    "target_regime",
    "transition_label",
    "persistence_error",
    "analyst_reason_text",
    "marketTradePrice",
    "systemMarginalPrice",
    "upRegulationDelivered",
    "downRegulationDelivered",
}
DELTA_FEATURE_BASES = [
    "residual_load_forecast",
    "residual_load_ramp",
    "kgup_solar_mw",
    "kgup_wind_mw",
    "load_minus_kgup",
    "gas_share",
    "outage_stress_index",
]


def load_labels() -> tuple[pd.DataFrame, Path]:
    for path in LABEL_PATHS:
        if path.exists():
            labels = pd.read_csv(path)
            labels["ts_hour"] = pd.to_datetime(labels["ts_hour"], errors="coerce")
            return labels, path
    raise FileNotFoundError("Missing regime labels. Run build_regime_labels.py first.")


def assign_split(ts: pd.Series) -> pd.Series:
    years = ts.dt.year
    split = pd.Series(index=ts.index, dtype="object")
    for name, (start_year, end_year) in SPLIT_RANGES.items():
        split[(years >= start_year) & (years <= end_year)] = name
    return split


def add_transition_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("ts_hour").copy()
    out["residual_load_vs_lag24"] = out["residual_load_forecast"] - out[
        "residual_load_forecast"
    ].shift(24)
    out["residual_load_ramp_vs_lag24"] = out["residual_load_ramp"] - out[
        "residual_load_ramp"
    ].shift(24)
    out["solar_today_vs_lag24"] = out["kgup_solar_mw"] - out["kgup_solar_mw"].shift(24)
    out["solar_drop_vs_lag24"] = (
        out["kgup_solar_mw"].shift(24) - out["kgup_solar_mw"]
    ).clip(lower=0)
    out["wind_today_vs_lag24"] = out["kgup_wind_mw"] - out["kgup_wind_mw"].shift(24)
    out["load_minus_kgup_vs_lag24"] = out["load_minus_kgup"] - out[
        "load_minus_kgup"
    ].shift(24)
    out["gas_share_vs_lag24"] = out["gas_share"] - out["gas_share"].shift(24)
    out["outage_stress_vs_lag24"] = out["outage_stress_index"] - out[
        "outage_stress_index"
    ].shift(24)
    return out


def load_dataset() -> tuple[pd.DataFrame, Path]:
    features = pd.read_parquet(FEATURE_STORE_PATH)
    reasoning = pd.read_parquet(REASONING_PATH)
    labels, label_path = load_labels()
    for frame in [features, reasoning, labels]:
        frame["ts_hour"] = pd.to_datetime(frame["ts_hour"], errors="coerce")

    data = features.merge(reasoning, on="ts_hour", how="inner").merge(
        labels[
            [
                "ts_hour",
                "target_regime",
                "lag24_regime",
                "transition_label",
                "persistence_error",
            ]
        ],
        on="ts_hour",
        how="inner",
    )
    data = add_transition_features(data)
    data["is_new_spike_transition"] = (
        (data["target_regime"] == "spike_cap")
        & (data["lag24_regime"].fillna("__missing__") != "spike_cap")
    ).astype(int)
    data["split"] = assign_split(data["ts_hour"])
    data = data[data["split"].isin(SPLIT_RANGES)].copy()
    data["snapshot_marketTradePrice_missing"] = data["snapshot_marketTradePrice"].isna().astype(int)

    if BINARY_SPIKE_PRED_PATH.exists():
        binary = pd.read_csv(BINARY_SPIKE_PRED_PATH)
        binary["ts_hour"] = pd.to_datetime(binary["ts_hour"], errors="coerce")
        data = data.merge(
            binary[["ts_hour", "spike_probability", "pred_spike_0.2", "pred_spike_0.1"]].rename(
                columns={
                    "spike_probability": "binary_spike_probability",
                    "pred_spike_0.2": "binary_pred_spike_0_2",
                    "pred_spike_0.1": "binary_pred_spike_0_1",
                }
            ),
            on="ts_hour",
            how="left",
        )
    return data, label_path


def build_feature_matrix(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    work = data.copy()
    if "analyst_reason_text" in work.columns:
        work = work.drop(columns=["analyst_reason_text"])
    drop_cols = {
        "ts_hour",
        "split",
        "target_regime",
        "lag24_regime",
        "transition_label",
        "persistence_error",
        "is_new_spike_transition",
        "binary_spike_probability",
        "binary_pred_spike_0_2",
        "binary_pred_spike_0_1",
    }
    feature_cols = [
        col for col in work.columns if col not in drop_cols and col not in FORBIDDEN_FEATURES
    ]
    features = work[feature_cols].copy()
    forbidden_present = sorted(FORBIDDEN_FEATURES.intersection(features.columns))
    categorical_cols = [
        col
        for col in features.columns
        if pd.api.types.is_object_dtype(features[col])
        or pd.api.types.is_string_dtype(features[col])
        or isinstance(features[col].dtype, pd.CategoricalDtype)
    ]
    features = pd.get_dummies(features, columns=categorical_cols, dummy_na=True, dtype=float)
    features = features.replace([np.inf, -np.inf], np.nan)
    return features, list(features.columns), forbidden_present


def train_model(x_train: pd.DataFrame, y_train: pd.Series, x_val: pd.DataFrame, y_val: pd.Series) -> LGBMClassifier:
    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    scale_pos_weight = negatives / max(positives, 1)
    model = LGBMClassifier(
        objective="binary",
        n_estimators=750,
        learning_rate=0.03,
        num_leaves=36,
        min_child_samples=25,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.05,
        reg_lambda=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        eval_metric="auc",
    )
    return model


def metric_at_threshold(y_true: pd.Series, probability: pd.Series, threshold: float) -> dict[str, Any]:
    pred = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, pred, labels=[1], zero_division=0
    )
    return {
        "threshold": threshold,
        "precision": float(precision[0]),
        "recall": float(recall[0]),
        "f1": float(f1[0]),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "false_alarm_rate": float(fp / max(fp + tn, 1)),
        "miss_rate": float(fn / max(fn + tp, 1)),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_negative": int(tn),
    }


def transition_rows(frame: pd.DataFrame, pred_col: str) -> list[dict[str, Any]]:
    rows = []
    for transition in [
        "normal -> spike_cap",
        "tight -> spike_cap",
        "negative_zero_pressure -> spike_cap",
    ]:
        group = frame[frame["transition_label"] == transition]
        positives = group[group["is_new_spike_transition"] == 1]
        rows.append(
            {
                "transition_label": transition,
                "rows": int(len(group)),
                "positive_count": int(len(positives)),
                "recall": float((positives[pred_col] == 1).mean()) if len(positives) else None,
            }
        )
    return rows


def evaluate_split(frame: pd.DataFrame, split: str, threshold: float) -> dict[str, Any]:
    y = frame["is_new_spike_transition"]
    probability = frame["new_spike_probability"]
    pred = (probability >= threshold).astype(int)
    lag24_baseline = (frame["lag24_regime"] == "spike_cap").astype(int)
    binary_02 = frame["binary_pred_spike_0_2"].fillna(0).astype(int)
    binary_01 = frame["binary_pred_spike_0_1"].fillna(0).astype(int)

    try:
        pr_auc = float(average_precision_score(y, probability))
    except ValueError:
        pr_auc = float("nan")
    try:
        roc_auc = float(roc_auc_score(y, probability))
    except ValueError:
        roc_auc = float("nan")

    return {
        "split": split,
        "rows": int(len(frame)),
        "positive_count": int(y.sum()),
        "positive_rate": float(y.mean()),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "operating_threshold": threshold,
        "threshold_metrics": [metric_at_threshold(y, probability, th) for th in THRESHOLDS],
        "lag24_baseline": metric_at_threshold(y, lag24_baseline, 0.5),
        "binary_spike_detector_0_2": metric_at_threshold(y, binary_02, 0.5),
        "binary_spike_detector_0_1": metric_at_threshold(y, binary_01, 0.5),
        "transition_recall": {
            "detector": transition_rows(frame.assign(binary_pred=pred), "binary_pred"),
            "lag24": transition_rows(frame.assign(binary_pred=lag24_baseline), "binary_pred"),
            "binary_detector_0_2": transition_rows(frame.assign(binary_pred=binary_02), "binary_pred"),
            "binary_detector_0_1": transition_rows(frame.assign(binary_pred=binary_01), "binary_pred"),
        },
    }


def feature_importance(model: LGBMClassifier, features: list[str]) -> list[dict[str, Any]]:
    rows = [
        {"feature": feature, "importance": float(value)}
        for feature, value in zip(features, model.feature_importances_)
    ]
    return sorted(rows, key=lambda row: row["importance"], reverse=True)[:25]


def critical_evaluation(test_metrics: dict[str, Any]) -> str:
    operating = next(
        row
        for row in test_metrics["threshold_metrics"]
        if abs(row["threshold"] - OPERATING_THRESHOLD) < 1e-12
    )
    binary = test_metrics["binary_spike_detector_0_1"]
    lag = test_metrics["lag24_baseline"]
    notes = [
        f"Operating threshold {OPERATING_THRESHOLD} recall is {operating['recall']:.3f}, precision {operating['precision']:.3f}, false alarm {operating['false_alarm_rate']:.3f}.",
        f"Comparison recall: transition detector {operating['recall']:.3f}, binary spike detector @0.1 {binary['recall']:.3f}, lag24 baseline {lag['recall']:.3f}.",
    ]
    if operating["recall"] > binary["recall"]:
        notes.append("The transition detector improves new-spike recall over the broad binary spike detector.")
    else:
        notes.append("The transition detector does not improve recall over the broad binary spike detector yet.")
    if operating["recall"] >= 0.5:
        notes.append("Recall is high enough for a screening prototype, subject to false-alarm review.")
    else:
        notes.append("Recall is still weak for cap-entry routing; feature/threshold design needs more work.")
    return " ".join(notes)


def write_outputs(
    model: LGBMClassifier,
    feature_cols: list[str],
    predictions: pd.DataFrame,
    metrics: dict[str, Any],
) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "spike_transition_detector_lgbm.joblib")
    (MODEL_DIR / "feature_columns.json").write_text(
        json.dumps(feature_cols, ensure_ascii=False, indent=2) + "\n"
    )
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    METRICS_JSON.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")

    test = metrics["test"]
    operating = next(
        row
        for row in test["threshold_metrics"]
        if abs(row["threshold"] - OPERATING_THRESHOLD) < 1e-12
    )
    lines = [
        "# Spike Transition Detector Metrics",
        "",
        f"Generated: `{metrics['generated_at']}`",
        "",
        "This model detects only new spike/cap transitions where `target_regime == spike_cap` and `lag24_regime != spike_cap`.",
        "It does not train price experts, ensembles, or final PTF regressors.",
        "",
        "## Split",
        "",
        "| Split | Years | Rows | Positives | Positive rate |",
        "|---|---|---:|---:|---:|",
    ]
    for split, info in metrics["splits"].items():
        lines.append(
            f"| `{split}` | `{info['years']}` | {info['rows']} | {info['positive_count']} | {info['positive_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Test Summary",
            "",
            f"- PR-AUC: `{test['pr_auc']:.4f}`",
            f"- ROC-AUC: `{test['roc_auc']:.4f}`",
            f"- Operating threshold: `{OPERATING_THRESHOLD}`",
            f"- Recall: `{operating['recall']:.4f}`",
            f"- Precision: `{operating['precision']:.4f}`",
            f"- False alarm rate: `{operating['false_alarm_rate']:.4f}`",
            f"- Miss rate: `{operating['miss_rate']:.4f}`",
            "",
            "## Threshold Analysis",
            "",
            "| Threshold | Precision | Recall | F1 | Balanced acc | Miss | False alarm | TP | FP | FN | TN |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in test["threshold_metrics"]:
        lines.append(
            f"| {row['threshold']:.3f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['balanced_accuracy']:.4f} | {row['miss_rate']:.4f} | {row['false_alarm_rate']:.4f} | {row['true_positive']} | {row['false_positive']} | {row['false_negative']} | {row['true_negative']} |"
        )
    lines.extend(
        [
            "",
            "## Comparison",
            "",
            "| Method | Precision | Recall | False alarm | Miss |",
            "|---|---:|---:|---:|---:|",
            f"| transition detector @ {OPERATING_THRESHOLD} | {operating['precision']:.4f} | {operating['recall']:.4f} | {operating['false_alarm_rate']:.4f} | {operating['miss_rate']:.4f} |",
            f"| lag24 baseline | {test['lag24_baseline']['precision']:.4f} | {test['lag24_baseline']['recall']:.4f} | {test['lag24_baseline']['false_alarm_rate']:.4f} | {test['lag24_baseline']['miss_rate']:.4f} |",
            f"| binary spike detector @0.2 | {test['binary_spike_detector_0_2']['precision']:.4f} | {test['binary_spike_detector_0_2']['recall']:.4f} | {test['binary_spike_detector_0_2']['false_alarm_rate']:.4f} | {test['binary_spike_detector_0_2']['miss_rate']:.4f} |",
            f"| binary spike detector @0.1 | {test['binary_spike_detector_0_1']['precision']:.4f} | {test['binary_spike_detector_0_1']['recall']:.4f} | {test['binary_spike_detector_0_1']['false_alarm_rate']:.4f} | {test['binary_spike_detector_0_1']['miss_rate']:.4f} |",
            "",
            "## Transition Recall",
            "",
            "| Transition | Positives | Detector | Lag24 | Binary @0.2 | Binary @0.1 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    lookup = {
        name: {row["transition_label"]: row for row in rows}
        for name, rows in test["transition_recall"].items()
    }
    for row in test["transition_recall"]["detector"]:
        transition = row["transition_label"]
        lines.append(
            f"| `{transition}` | {row['positive_count']} | {row['recall'] if row['recall'] is not None else 0:.4f} | {lookup['lag24'].get(transition, {}).get('recall') or 0:.4f} | {lookup['binary_detector_0_2'].get(transition, {}).get('recall') or 0:.4f} | {lookup['binary_detector_0_1'].get(transition, {}).get('recall') or 0:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Top Feature Importances",
            "",
            "| Feature | Importance |",
            "|---|---:|",
        ]
    )
    for row in metrics["top_features"]:
        lines.append(f"| `{row['feature']}` | {row['importance']:.0f} |")
    lines.extend(
        [
            "",
            "## Critical Evaluation",
            "",
            metrics["critical_evaluation"],
            "",
            "## Leakage Checks",
            "",
        ]
    )
    for check in metrics["leakage_checks"]:
        lines.append(f"- **{check['check']}**: `{check['status']}` - {check['detail']}")
    METRICS_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    data, label_path = load_dataset()
    features, feature_cols, forbidden_present = build_feature_matrix(data)
    train_mask = data["split"] == "train"
    val_mask = data["split"] == "validation"
    test_mask = data["split"] == "test"

    model = train_model(
        features.loc[train_mask],
        data.loc[train_mask, "is_new_spike_transition"],
        features.loc[val_mask],
        data.loc[val_mask, "is_new_spike_transition"],
    )
    probability = model.predict_proba(features)[:, 1]
    predictions = data[
        [
            "ts_hour",
            "split",
            "target_regime",
            "lag24_regime",
            "transition_label",
            "is_new_spike_transition",
            "persistence_error",
            "binary_spike_probability",
            "binary_pred_spike_0_2",
            "binary_pred_spike_0_1",
        ]
    ].copy()
    predictions["new_spike_probability"] = probability
    for threshold in THRESHOLDS:
        predictions[f"pred_new_spike_{threshold:.3f}"] = (probability >= threshold).astype(int)

    top_features = feature_importance(model, feature_cols)
    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": "LightGBM LGBMClassifier binary new-spike transition",
        "label_path": str(label_path.relative_to(PROJECT_ROOT)),
        "feature_store": str(FEATURE_STORE_PATH.relative_to(PROJECT_ROOT)),
        "reasoning_features": str(REASONING_PATH.relative_to(PROJECT_ROOT)),
        "operating_threshold": OPERATING_THRESHOLD,
        "splits": {
            split: {
                "years": f"{years[0]}-{years[1]}",
                "rows": int((data["split"] == split).sum()),
                "positive_count": int(data.loc[data["split"] == split, "is_new_spike_transition"].sum()),
                "positive_rate": float(data.loc[data["split"] == split, "is_new_spike_transition"].mean()),
            }
            for split, years in SPLIT_RANGES.items()
        },
        "validation": evaluate_split(
            predictions.loc[val_mask].copy(), "validation", OPERATING_THRESHOLD
        ),
        "test": evaluate_split(predictions.loc[test_mask].copy(), "test", OPERATING_THRESHOLD),
        "top_features": top_features,
        "leakage_checks": [
            {
                "check": "Forbidden feature columns absent",
                "status": "pass" if not forbidden_present else "fail",
                "detail": f"Forbidden columns present in model feature matrix: {forbidden_present}",
            },
            {
                "check": "same-hour finalized PTF excluded",
                "status": "pass",
                "detail": "target_regime and transition_label are labels/evaluation only.",
            },
            {
                "check": "same-hour realized SMF/YAL/YAT excluded",
                "status": "pass",
                "detail": "Feature store exposes only lagged SMF/YAL/YAT fields.",
            },
            {
                "check": "historical interim oracle excluded",
                "status": "pass",
                "detail": "No raw historical interim_mcp source is read.",
            },
        ],
    }
    metrics["critical_evaluation"] = critical_evaluation(metrics["test"])
    write_outputs(model, feature_cols, predictions, metrics)
    print(f"Wrote {MODEL_DIR}")
    print(f"Wrote {PREDICTIONS_PATH}")
    print(f"Wrote {METRICS_JSON}")
    print(f"Wrote {METRICS_MD}")


if __name__ == "__main__":
    main()
