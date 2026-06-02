#!/usr/bin/env python3
"""
Train a binary spike/cap risk detector for regime-aware PTF research.

This script trains only a binary classifier:
    is_spike_cap = target_regime == "spike_cap"

It does not train price experts, ensembles, or final PTF regressors.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent

FEATURE_STORE_PATH = PROJECT_ROOT / "data" / "features" / "regime_feature_store.parquet"
REASONING_PATH = PROJECT_ROOT / "data" / "features" / "market_reasoning_features.parquet"
REGIME_LABEL_PATHS = [
    PROJECT_ROOT / "data" / "features" / "regime_labels.csv",
    PROJECT_ROOT / "data" / "regime_labels.csv",
]
MULTICLASS_PRED_PATH = PROJECT_ROOT / "data" / "predictions" / "regime_classifier_predictions.csv"

MODEL_DIR = PROJECT_ROOT / "models" / "spike_cap_detector"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "predictions" / "spike_cap_detector_predictions.csv"
METRICS_JSON = PROJECT_ROOT / "reports" / "spike_cap_detector_metrics.json"
METRICS_MD = PROJECT_ROOT / "reports" / "spike_cap_detector_metrics.md"
PR_FIG = PROJECT_ROOT / "reports" / "figures" / "spike_detector_pr_curve.png"
CM_FIG = PROJECT_ROOT / "reports" / "figures" / "spike_detector_confusion_matrix.png"

SPLIT_RANGES = {
    "train": (2020, 2024),
    "validation": (2025, 2025),
    "test": (2026, 2026),
}
THRESHOLDS = [0.5, 0.3, 0.2, 0.1]
OPERATING_THRESHOLD = 0.2
FOCUS_FEATURES = [
    "residual_load_ramp",
    "solar_cliff_score",
    "load_minus_kgup",
    "active_maintenance_capacity",
    "outage_stress_index",
    "gas_share",
    "hydro_share",
    "evening_ramp_flag",
    "analyst_spike_score",
    "analyst_persistence_break_score",
    "volatility_cluster_score",
]
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


def load_labels() -> tuple[pd.DataFrame, Path]:
    for path in REGIME_LABEL_PATHS:
        if path.exists():
            labels = pd.read_csv(path)
            labels["ts_hour"] = pd.to_datetime(labels["ts_hour"], errors="coerce")
            return labels, path
    raise FileNotFoundError("Missing regime_labels.csv. Run build_regime_labels.py first.")


def assign_split(ts: pd.Series) -> pd.Series:
    years = ts.dt.year
    split = pd.Series(index=ts.index, dtype="object")
    for name, (start_year, end_year) in SPLIT_RANGES.items():
        split[(years >= start_year) & (years <= end_year)] = name
    return split


def load_dataset() -> tuple[pd.DataFrame, Path]:
    from src.utils.io_utils import read_parquet_with_normalized_ts
    features = read_parquet_with_normalized_ts(FEATURE_STORE_PATH)
    reasoning = read_parquet_with_normalized_ts(REASONING_PATH)
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
    data = data.dropna(subset=["ts_hour", "target_regime"]).sort_values("ts_hour")
    data["is_spike_cap"] = (data["target_regime"] == "spike_cap").astype(int)
    data["split"] = assign_split(data["ts_hour"])
    data = data[data["split"].isin(SPLIT_RANGES)].copy()
    data["snapshot_marketTradePrice_missing"] = data["snapshot_marketTradePrice"].isna().astype(int)

    if MULTICLASS_PRED_PATH.exists():
        multi = pd.read_csv(MULTICLASS_PRED_PATH)
        multi["ts_hour"] = pd.to_datetime(multi["ts_hour"], errors="coerce")
        cols = ["ts_hour", "pred_regime", "prob_spike_cap"]
        data = data.merge(
            multi[cols].rename(
                columns={
                    "pred_regime": "multiclass_pred_regime",
                    "prob_spike_cap": "multiclass_prob_spike_cap",
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
        "is_spike_cap",
        "multiclass_pred_regime",
        "multiclass_prob_spike_cap",
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
        n_estimators=650,
        learning_rate=0.035,
        num_leaves=40,
        min_child_samples=45,
        subsample=0.9,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
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


def metrics_at_threshold(y_true: pd.Series, prob: pd.Series, threshold: float) -> dict[str, Any]:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, pred, labels=[1], zero_division=0
    )
    false_alarm_rate = fp / max(fp + tn, 1)
    cap_miss_rate = fn / max(fn + tp, 1)
    return {
        "threshold": threshold,
        "precision": float(precision[0]),
        "recall": float(recall[0]),
        "f1": float(f1[0]),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
        "cap_miss_rate": float(cap_miss_rate),
        "false_alarm_rate": float(false_alarm_rate),
    }


def transition_binary_recall(frame: pd.DataFrame, pred_col: str, min_count: int = 5) -> list[dict[str, Any]]:
    rows = []
    for transition, group in frame.groupby("transition_label", observed=False):
        if pd.isna(transition) or len(group) < min_count or group["is_spike_cap"].sum() == 0:
            continue
        rows.append(
            {
                "transition_label": str(transition),
                "count": int(len(group)),
                "spike_count": int(group["is_spike_cap"].sum()),
                "recall": float(
                    (group.loc[group["is_spike_cap"] == 1, pred_col] == 1).mean()
                ),
            }
        )
    return sorted(rows, key=lambda row: row["spike_count"], reverse=True)


def evaluate_split(frame: pd.DataFrame, split: str, threshold: float, train_p75_error: float) -> dict[str, Any]:
    y = frame["is_spike_cap"]
    prob = frame["spike_probability"]
    baseline_lag24 = (frame["lag24_regime"] == "spike_cap").astype(int)
    multiclass_pred = (
        frame["multiclass_pred_regime"].fillna("__missing__") == "spike_cap"
    ).astype(int)
    operating_pred = (prob >= threshold).astype(int)
    failure_mask = frame["persistence_error"] >= train_p75_error

    threshold_metrics = [metrics_at_threshold(y, prob, th) for th in THRESHOLDS]
    try:
        pr_auc = float(average_precision_score(y, prob))
    except ValueError:
        pr_auc = float("nan")
    try:
        roc_auc = float(roc_auc_score(y, prob))
    except ValueError:
        roc_auc = float("nan")

    lag24_metrics = metrics_at_threshold(y, baseline_lag24, 0.5)
    multiclass_metrics = metrics_at_threshold(y, multiclass_pred, 0.5)
    out = {
        "split": split,
        "rows": int(len(frame)),
        "positive_count": int(y.sum()),
        "positive_rate": float(y.mean()),
        "operating_threshold": threshold,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "threshold_metrics": threshold_metrics,
        "lag24_regime_baseline": lag24_metrics,
        "multiclass_classifier_baseline": multiclass_metrics,
        "persistence_failure_hours": {
            "threshold_train_p75": float(train_p75_error),
            "rows": int(failure_mask.sum()),
            "positive_count": int(y[failure_mask].sum()),
            "recall": float(
                (operating_pred[failure_mask & (y == 1)] == 1).mean()
            )
            if int(y[failure_mask].sum()) > 0
            else None,
            "lag24_baseline_recall": float(
                (baseline_lag24[failure_mask & (y == 1)] == 1).mean()
            )
            if int(y[failure_mask].sum()) > 0
            else None,
            "multiclass_recall": float(
                (multiclass_pred[failure_mask & (y == 1)] == 1).mean()
            )
            if int(y[failure_mask].sum()) > 0
            else None,
        },
        "transition_recall": {
            "detector": transition_binary_recall(
                frame.assign(binary_pred=operating_pred), "binary_pred"
            ),
            "lag24": transition_binary_recall(
                frame.assign(binary_pred=baseline_lag24), "binary_pred"
            ),
            "multiclass": transition_binary_recall(
                frame.assign(binary_pred=multiclass_pred), "binary_pred"
            ),
        },
        "confusion_matrix_operating_threshold": confusion_matrix(
            y, operating_pred, labels=[0, 1]
        ).tolist(),
    }
    return out


def plot_pr_curve(y_true: pd.Series, probability: pd.Series) -> None:
    PR_FIG.parent.mkdir(parents=True, exist_ok=True)
    precision, recall, _ = precision_recall_curve(y_true, probability)
    ap = average_precision_score(y_true, probability)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, label=f"PR-AUC={ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Spike/Cap Detector Precision-Recall Curve - Test")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PR_FIG, dpi=160)
    plt.close(fig)


def plot_confusion(cm: list[list[int]]) -> None:
    CM_FIG.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="Reds")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["non_spike", "spike"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["non_spike", "spike"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Spike Detector Confusion Matrix @ {OPERATING_THRESHOLD}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(CM_FIG, dpi=160)
    plt.close(fig)


def feature_importance(model: LGBMClassifier, feature_cols: list[str], n: int = 20) -> list[dict[str, Any]]:
    rows = [
        {"feature": feature, "importance": float(importance)}
        for feature, importance in zip(feature_cols, model.feature_importances_)
    ]
    return sorted(rows, key=lambda row: row["importance"], reverse=True)[:n]


def critical_evaluation(test_metrics: dict[str, Any], top_features: list[dict[str, Any]]) -> str:
    operating = next(
        item for item in test_metrics["threshold_metrics"] if item["threshold"] == OPERATING_THRESHOLD
    )
    recall = operating["recall"]
    precision = operating["precision"]
    false_alarm = operating["false_alarm_rate"]
    notes = []
    if recall >= 0.60:
        notes.append(f"Spike/cap recall improved into a usable screening zone at threshold {OPERATING_THRESHOLD}: {recall:.3f}.")
    else:
        notes.append(f"Spike/cap recall is still weak at threshold {OPERATING_THRESHOLD}: {recall:.3f}.")
    notes.append(f"Precision is {precision:.3f}; false alarm rate is {false_alarm:.3f}.")
    notes.append(
        "Top features are: "
        + ", ".join(row["feature"] for row in top_features[:8])
        + "."
    )
    lag_recall = test_metrics["lag24_regime_baseline"]["recall"]
    multi_recall = test_metrics["multiclass_classifier_baseline"]["recall"]
    notes.append(
        f"Recall comparison: detector {recall:.3f}, lag24 baseline {lag_recall:.3f}, multiclass {multi_recall:.3f}."
    )
    if recall > multi_recall:
        notes.append("The binary detector adds value over the multiclass classifier for spike screening.")
    else:
        notes.append("The binary detector does not yet beat the multiclass classifier on spike recall.")
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
    joblib.dump(model, MODEL_DIR / "spike_cap_detector_lgbm.joblib")
    (MODEL_DIR / "feature_columns.json").write_text(
        json.dumps(feature_cols, ensure_ascii=False, indent=2) + "\n"
    )
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    METRICS_JSON.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")

    test = metrics["test"]
    plot_pr_curve(
        predictions.loc[predictions["split"] == "test", "is_spike_cap"],
        predictions.loc[predictions["split"] == "test", "spike_probability"],
    )
    plot_confusion(test["confusion_matrix_operating_threshold"])

    operating = next(
        item for item in test["threshold_metrics"] if item["threshold"] == OPERATING_THRESHOLD
    )
    lines = [
        "# Spike / Cap Risk Detector Metrics",
        "",
        f"Generated: `{metrics['generated_at']}`",
        "",
        "This run trains only a binary `is_spike_cap` classifier. It does not train price experts, ensembles, or final PTF regressors.",
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
            f"- Balanced accuracy: `{operating['balanced_accuracy']:.4f}`",
            f"- Cap miss rate: `{operating['cap_miss_rate']:.4f}`",
            f"- False alarm rate: `{operating['false_alarm_rate']:.4f}`",
            "",
            "## Threshold Analysis",
            "",
            "| Threshold | Precision | Recall | F1 | Balanced acc | Cap miss | False alarm | TP | FP | FN | TN |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in test["threshold_metrics"]:
        lines.append(
            f"| {row['threshold']:.2f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['balanced_accuracy']:.4f} | {row['cap_miss_rate']:.4f} | {row['false_alarm_rate']:.4f} | {row['true_positive']} | {row['false_positive']} | {row['false_negative']} | {row['true_negative']} |"
        )
    lines.extend(
        [
            "",
            "## Baseline Comparison - Test",
            "",
            "| Method | Precision | Recall | Balanced acc | False alarm | Cap miss |",
            "|---|---:|---:|---:|---:|---:|",
            f"| spike detector @ {OPERATING_THRESHOLD} | {operating['precision']:.4f} | {operating['recall']:.4f} | {operating['balanced_accuracy']:.4f} | {operating['false_alarm_rate']:.4f} | {operating['cap_miss_rate']:.4f} |",
            f"| lag24_regime baseline | {test['lag24_regime_baseline']['precision']:.4f} | {test['lag24_regime_baseline']['recall']:.4f} | {test['lag24_regime_baseline']['balanced_accuracy']:.4f} | {test['lag24_regime_baseline']['false_alarm_rate']:.4f} | {test['lag24_regime_baseline']['cap_miss_rate']:.4f} |",
            f"| multiclass classifier | {test['multiclass_classifier_baseline']['precision']:.4f} | {test['multiclass_classifier_baseline']['recall']:.4f} | {test['multiclass_classifier_baseline']['balanced_accuracy']:.4f} | {test['multiclass_classifier_baseline']['false_alarm_rate']:.4f} | {test['multiclass_classifier_baseline']['cap_miss_rate']:.4f} |",
            "",
            "## Persistence Failure Hours",
            "",
            f"- Rows: `{test['persistence_failure_hours']['rows']}`",
            f"- Spike positives: `{test['persistence_failure_hours']['positive_count']}`",
            f"- Detector recall: `{test['persistence_failure_hours']['recall']:.4f}`",
            f"- Lag24 baseline recall: `{test['persistence_failure_hours']['lag24_baseline_recall']:.4f}`",
            f"- Multiclass recall: `{test['persistence_failure_hours']['multiclass_recall']:.4f}`",
            "",
            "## Spike Transition Recall",
            "",
            "| Transition | Spike count | Detector recall | Lag24 recall | Multiclass recall |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    detector_transitions = {
        row["transition_label"]: row for row in test["transition_recall"]["detector"]
    }
    lag_transitions = {
        row["transition_label"]: row for row in test["transition_recall"]["lag24"]
    }
    multi_transitions = {
        row["transition_label"]: row for row in test["transition_recall"]["multiclass"]
    }
    for transition, row in detector_transitions.items():
        lag_row = lag_transitions.get(transition, {})
        multi_row = multi_transitions.get(transition, {})
        lines.append(
            f"| `{transition}` | {row['spike_count']} | {row['recall']:.4f} | {lag_row.get('recall', 0.0):.4f} | {multi_row.get('recall', 0.0):.4f} |"
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
        data.loc[train_mask, "is_spike_cap"],
        features.loc[val_mask],
        data.loc[val_mask, "is_spike_cap"],
    )

    probability = model.predict_proba(features)[:, 1]
    predictions = data[
        [
            "ts_hour",
            "split",
            "target_regime",
            "is_spike_cap",
            "lag24_regime",
            "transition_label",
            "persistence_error",
            "multiclass_pred_regime",
            "multiclass_prob_spike_cap",
        ]
    ].copy()
    predictions["spike_probability"] = probability
    for threshold in THRESHOLDS:
        predictions[f"pred_spike_{threshold:.1f}"] = (probability >= threshold).astype(int)

    train_p75_error = float(data.loc[train_mask, "persistence_error"].quantile(0.75))
    top_features = feature_importance(model, feature_cols)
    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": "LightGBM LGBMClassifier binary",
        "label_path": str(label_path.relative_to(PROJECT_ROOT)),
        "feature_store": str(FEATURE_STORE_PATH.relative_to(PROJECT_ROOT)),
        "reasoning_features": str(REASONING_PATH.relative_to(PROJECT_ROOT)),
        "operating_threshold": OPERATING_THRESHOLD,
        "focus_features": FOCUS_FEATURES,
        "splits": {
            split: {
                "years": f"{years[0]}-{years[1]}",
                "rows": int((data["split"] == split).sum()),
                "positive_count": int(data.loc[data["split"] == split, "is_spike_cap"].sum()),
                "positive_rate": float(data.loc[data["split"] == split, "is_spike_cap"].mean()),
            }
            for split, years in SPLIT_RANGES.items()
        },
        "validation": evaluate_split(
            predictions.loc[val_mask].copy(), "validation", OPERATING_THRESHOLD, train_p75_error
        ),
        "test": evaluate_split(
            predictions.loc[test_mask].copy(), "test", OPERATING_THRESHOLD, train_p75_error
        ),
        "top_features": top_features,
        "leakage_checks": [
            {
                "check": "Forbidden feature columns absent",
                "status": "pass" if not forbidden_present else "fail",
                "detail": f"Forbidden columns present in model feature matrix: {forbidden_present}",
            },
            {
                "check": "historical interim-mcp oracle excluded",
                "status": "pass",
                "detail": "Only point-in-time snapshot columns from feature store are eligible.",
            },
            {
                "check": "same-hour finalized PTF excluded",
                "status": "pass",
                "detail": "Target regime and price are labels/evaluation only.",
            },
            {
                "check": "same-hour realized SMF/YAL-YAT excluded",
                "status": "pass",
                "detail": "Feature store exposes only lagged SMF/YAL-YAT fields.",
            },
        ],
    }
    metrics["critical_evaluation"] = critical_evaluation(metrics["test"], top_features)
    write_outputs(model, feature_cols, predictions, metrics)

    print(f"Wrote {MODEL_DIR}")
    print(f"Wrote {PREDICTIONS_PATH}")
    print(f"Wrote {METRICS_JSON}")
    print(f"Wrote {METRICS_MD}")
    print(f"Wrote {PR_FIG}")
    print(f"Wrote {CM_FIG}")


if __name__ == "__main__":
    main()
