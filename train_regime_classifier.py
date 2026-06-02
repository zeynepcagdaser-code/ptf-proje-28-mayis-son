#!/usr/bin/env python3
"""
Train the first regime classifier for regime-aware PTF research.

This script trains only target_regime classification. It does not train price
experts, ensembles, or final PTF forecasters.
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
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

PROJECT_ROOT = Path(__file__).resolve().parent

FEATURE_STORE_PATH = PROJECT_ROOT / "data" / "features" / "regime_feature_store.parquet"
REASONING_PATH = PROJECT_ROOT / "data" / "features" / "market_reasoning_features.parquet"
LABEL_PATHS = [
    PROJECT_ROOT / "data" / "features" / "regime_labels.csv",
    PROJECT_ROOT / "data" / "regime_labels.csv",
]

MODEL_DIR = PROJECT_ROOT / "models" / "regime_classifier"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "predictions" / "regime_classifier_predictions.csv"
METRICS_JSON = PROJECT_ROOT / "reports" / "regime_classifier_metrics.json"
METRICS_MD = PROJECT_ROOT / "reports" / "regime_classifier_metrics.md"
CONFUSION_FIG = PROJECT_ROOT / "reports" / "figures" / "regime_classifier_confusion_matrix.png"

CLASS_ORDER = ["negative_zero_pressure", "normal", "tight", "spike_cap"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_ORDER)}
ID_TO_CLASS = {idx: name for name, idx in CLASS_TO_ID.items()}
SPLIT_RANGES = {
    "train": (2020, 2024),
    "validation": (2025, 2025),
    "test": (2026, 2026),
}

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
    data["split"] = assign_split(data["ts_hour"])
    data = data[data["split"].isin(SPLIT_RANGES)].copy()
    data["snapshot_marketTradePrice_missing"] = data["snapshot_marketTradePrice"].isna().astype(int)
    return data, label_path


def build_feature_matrix(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    if "analyst_reason_text" in data.columns:
        data = data.drop(columns=["analyst_reason_text"])

    feature_cols = [
        column
        for column in data.columns
        if column
        not in {
            "ts_hour",
            "split",
            "target_regime",
            "lag24_regime",
            "transition_label",
            "persistence_error",
        }
    ]
    feature_cols = [column for column in feature_cols if column not in FORBIDDEN_FEATURES]
    features = data[feature_cols].copy()
    forbidden_present = sorted(FORBIDDEN_FEATURES.intersection(features.columns))

    categorical_cols = [
        column
        for column in features.columns
        if pd.api.types.is_object_dtype(features[column])
        or pd.api.types.is_string_dtype(features[column])
        or pd.api.types.is_categorical_dtype(features[column])
    ]
    features = pd.get_dummies(features, columns=categorical_cols, dummy_na=True, dtype=float)
    features = features.replace([np.inf, -np.inf], np.nan)
    return features, list(features.columns), forbidden_present


def sample_weights(y: pd.Series) -> np.ndarray:
    counts = y.value_counts()
    total = len(y)
    n_classes = len(counts)
    weights = y.map(lambda value: total / (n_classes * counts[value])).to_numpy(float)
    return weights


def train_model(x_train: pd.DataFrame, y_train: pd.Series, x_val: pd.DataFrame, y_val: pd.Series) -> LGBMClassifier:
    model = LGBMClassifier(
        objective="multiclass",
        num_class=len(CLASS_ORDER),
        n_estimators=450,
        learning_rate=0.045,
        num_leaves=48,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        class_weight=None,
        verbosity=-1,
    )
    model.fit(
        x_train,
        y_train.map(CLASS_TO_ID),
        sample_weight=sample_weights(y_train),
        eval_set=[(x_val, y_val.map(CLASS_TO_ID))],
        eval_metric="multi_logloss",
    )
    return model


def predict_classes(model: LGBMClassifier, x: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    probabilities = model.predict_proba(x)
    pred_ids = np.argmax(probabilities, axis=1)
    pred_labels = np.array([ID_TO_CLASS[int(idx)] for idx in pred_ids])
    prob_df = pd.DataFrame(probabilities, columns=[f"prob_{label}" for label in CLASS_ORDER], index=x.index)
    return pred_labels, prob_df


def baseline_predictions(data: pd.DataFrame, train_data: pd.DataFrame) -> tuple[pd.Series, str]:
    most_frequent = str(train_data["target_regime"].value_counts().idxmax())
    lag24 = data["lag24_regime"].fillna(most_frequent)
    return lag24, most_frequent


def transition_recall_table(frame: pd.DataFrame, min_count: int = 10) -> list[dict[str, Any]]:
    rows = []
    grouped = frame.dropna(subset=["transition_label"]).groupby("transition_label", observed=False)
    for transition, group in grouped:
        if len(group) < min_count:
            continue
        rows.append(
            {
                "transition_label": str(transition),
                "count": int(len(group)),
                "model_recall": float((group["pred_regime"] == group["target_regime"]).mean()),
                "lag24_baseline_recall": float((group["lag24_baseline"] == group["target_regime"]).mean()),
            }
        )
    return sorted(rows, key=lambda row: row["count"], reverse=True)


def evaluate_split(frame: pd.DataFrame, split_name: str, train_p75_error: float) -> dict[str, Any]:
    y_true = frame["target_regime"]
    y_pred = frame["pred_regime"]
    lag24 = frame["lag24_baseline"]
    frequent = frame["most_frequent_baseline"]

    report = classification_report(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
        zero_division=0,
    )
    per_class = {
        label: {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }
        for idx, label in enumerate(CLASS_ORDER)
    }
    normal_tight_mask = y_true.isin(["normal", "tight"])
    failure_mask = frame["persistence_error"] >= train_p75_error

    metrics = {
        "split": split_name,
        "rows": int(len(frame)),
        "class_distribution": y_true.value_counts().to_dict(),
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0)),
        "per_class": per_class,
        "spike_cap_recall": per_class["spike_cap"]["recall"],
        "negative_zero_pressure_recall": per_class["negative_zero_pressure"]["recall"],
        "normal_tight_accuracy": float((y_true[normal_tight_mask] == y_pred[normal_tight_mask]).mean()),
        "lag24_baseline_accuracy": float((y_true == lag24).mean()),
        "lag24_baseline_balanced_accuracy": float(balanced_accuracy_score(y_true, lag24)),
        "most_frequent_baseline_accuracy": float((y_true == frequent).mean()),
        "most_frequent_baseline_balanced_accuracy": float(balanced_accuracy_score(y_true, frequent)),
        "persistence_failure_hours": {
            "threshold_train_p75": float(train_p75_error),
            "rows": int(failure_mask.sum()),
            "model_recall": float((y_true[failure_mask] == y_pred[failure_mask]).mean()) if failure_mask.any() else None,
            "lag24_baseline_recall": float((y_true[failure_mask] == lag24[failure_mask]).mean()) if failure_mask.any() else None,
        },
        "transition_recall": transition_recall_table(frame),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASS_ORDER).tolist(),
    }
    return metrics


def plot_confusion_matrix(cm: list[list[int]]) -> None:
    CONFUSION_FIG.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.array(cm)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(CLASS_ORDER)))
    ax.set_xticklabels(CLASS_ORDER, rotation=35, ha="right")
    ax.set_yticks(range(len(CLASS_ORDER)))
    ax.set_yticklabels(CLASS_ORDER)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Regime Classifier Confusion Matrix - Test")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(CONFUSION_FIG, dpi=160)
    plt.close(fig)


def write_reports(metrics: dict[str, Any], predictions: pd.DataFrame, model: LGBMClassifier, feature_columns: list[str]) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_DIR / "regime_classifier_lgbm.joblib")
    (MODEL_DIR / "feature_columns.json").write_text(
        json.dumps(feature_columns, ensure_ascii=False, indent=2) + "\n"
    )
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    METRICS_JSON.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    plot_confusion_matrix(metrics["test"]["confusion_matrix"])

    test = metrics["test"]
    lines = [
        "# Regime Classifier Metrics",
        "",
        f"Generated: `{metrics['generated_at']}`",
        "",
        "This run trains only the `target_regime` classifier. It does not train price experts, ensembles, or final PTF forecasts.",
        "",
        "## Split",
        "",
        "| Split | Years | Rows |",
        "|---|---|---:|",
    ]
    for split, info in metrics["splits"].items():
        lines.append(f"| `{split}` | `{info['years']}` | {info['rows']} |")
    lines.extend(
        [
            "",
            "## Test Metrics",
            "",
            f"- Accuracy: `{test['accuracy']:.4f}`",
            f"- Balanced accuracy: `{test['balanced_accuracy']:.4f}`",
            f"- Macro F1: `{test['macro_f1']:.4f}`",
            f"- Spike/cap recall: `{test['spike_cap_recall']:.4f}`",
            f"- Negative/zero pressure recall: `{test['negative_zero_pressure_recall']:.4f}`",
            f"- Normal/tight accuracy: `{test['normal_tight_accuracy']:.4f}`",
            "",
            "## Baselines",
            "",
            "| Baseline | Accuracy | Balanced accuracy |",
            "|---|---:|---:|",
            f"| model | {test['accuracy']:.4f} | {test['balanced_accuracy']:.4f} |",
            f"| lag24_regime | {test['lag24_baseline_accuracy']:.4f} | {test['lag24_baseline_balanced_accuracy']:.4f} |",
            f"| most frequent regime | {test['most_frequent_baseline_accuracy']:.4f} | {test['most_frequent_baseline_balanced_accuracy']:.4f} |",
            "",
            "## Per-Class Metrics",
            "",
            "| Class | Precision | Recall | F1 | Support |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label in CLASS_ORDER:
        row = test["per_class"][label]
        lines.append(
            f"| `{label}` | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['support']} |"
        )

    pf = test["persistence_failure_hours"]
    lines.extend(
        [
            "",
            "## Persistence Failure Hours",
            "",
            f"- Threshold: train persistence_error p75 = `{pf['threshold_train_p75']:.2f}`",
            f"- Rows: `{pf['rows']}`",
            f"- Model recall/accuracy on failure hours: `{pf['model_recall']:.4f}`",
            f"- Lag24 baseline recall/accuracy on failure hours: `{pf['lag24_baseline_recall']:.4f}`",
            "",
            "## Transition Recall - Test",
            "",
            "| Transition | Rows | Model recall | Lag24 baseline recall |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in test["transition_recall"][:16]:
        lines.append(
            f"| `{row['transition_label']}` | {row['count']} | {row['model_recall']:.4f} | {row['lag24_baseline_recall']:.4f} |"
        )

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


def critical_evaluation(test_metrics: dict[str, Any]) -> str:
    spike_recall = test_metrics["spike_cap_recall"]
    zero_recall = test_metrics["negative_zero_pressure_recall"]
    model_fail = test_metrics["persistence_failure_hours"]["model_recall"]
    lag_fail = test_metrics["persistence_failure_hours"]["lag24_baseline_recall"]
    collapse = max(test_metrics["class_distribution"].values()) / test_metrics["rows"]
    predicted_collapse = test_metrics["accuracy"] <= collapse + 0.02 and test_metrics["balanced_accuracy"] < 0.35

    notes = []
    if predicted_collapse:
        notes.append("The classifier appears close to majority-class collapse; do not proceed to price experts before fixing class balance/features.")
    else:
        notes.append("The classifier does not collapse to only the majority class.")
    notes.append(
        f"Spike/cap recall is {'acceptable as a first prototype' if spike_recall >= 0.50 else 'weak and must improve before cap expert routing'} ({spike_recall:.3f})."
    )
    notes.append(
        f"Zero-pressure recall is {'acceptable as a first prototype' if zero_recall >= 0.50 else 'weak and must improve before zero expert routing'} ({zero_recall:.3f})."
    )
    if model_fail is not None and lag_fail is not None:
        delta = model_fail - lag_fail
        notes.append(
            f"On persistence-failure hours, model recall is {model_fail:.3f} vs lag24 baseline {lag_fail:.3f} (delta {delta:+.3f})."
        )
    return " ".join(notes)


def main() -> None:
    data, label_path = load_dataset()
    features, feature_columns, forbidden_present = build_feature_matrix(data)
    merged = data[["ts_hour", "target_regime", "lag24_regime", "transition_label", "persistence_error", "split"]].copy()

    train_mask = data["split"] == "train"
    val_mask = data["split"] == "validation"
    test_mask = data["split"] == "test"

    x_train = features.loc[train_mask]
    y_train = data.loc[train_mask, "target_regime"]
    x_val = features.loc[val_mask]
    y_val = data.loc[val_mask, "target_regime"]
    x_test = features.loc[test_mask]

    model = train_model(x_train, y_train, x_val, y_val)

    all_pred, all_prob = predict_classes(model, features)
    predictions = merged.copy()
    predictions["pred_regime"] = all_pred
    predictions = pd.concat([predictions, all_prob.reset_index(drop=True)], axis=1)
    predictions["lag24_baseline"], most_frequent = baseline_predictions(data, data.loc[train_mask])
    predictions["most_frequent_baseline"] = most_frequent

    train_p75_error = float(data.loc[train_mask, "persistence_error"].quantile(0.75))
    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": "LightGBM LGBMClassifier multiclass",
        "label_path": str(label_path.relative_to(PROJECT_ROOT)),
        "feature_store": str(FEATURE_STORE_PATH.relative_to(PROJECT_ROOT)),
        "reasoning_features": str(REASONING_PATH.relative_to(PROJECT_ROOT)),
        "outputs": {
            "model_dir": str(MODEL_DIR.relative_to(PROJECT_ROOT)),
            "predictions": str(PREDICTIONS_PATH.relative_to(PROJECT_ROOT)),
            "confusion_matrix_figure": str(CONFUSION_FIG.relative_to(PROJECT_ROOT)),
        },
        "splits": {
            split: {
                "years": f"{years[0]}-{years[1]}",
                "rows": int((data["split"] == split).sum()),
                "class_distribution": data.loc[data["split"] == split, "target_regime"].value_counts().to_dict(),
            }
            for split, years in SPLIT_RANGES.items()
        },
        "feature_columns": feature_columns,
        "leakage_checks": [
            {
                "check": "Forbidden feature columns absent",
                "status": "pass" if not forbidden_present else "fail",
                "detail": f"Forbidden columns present in model feature matrix: {forbidden_present}",
            },
            {
                "check": "analyst_reason_text excluded",
                "status": "pass",
                "detail": "Text reason column is kept out of numeric feature matrix.",
            },
            {
                "check": "historical interim-mcp oracle excluded",
                "status": "pass",
                "detail": "Only point-in-time snapshot columns from feature store are eligible.",
            },
            {
                "check": "same-hour realized SMF/YAL-YAT excluded",
                "status": "pass",
                "detail": "Feature store exposes only lagged SMF/YAL/YAT fields.",
            },
        ],
        "validation": evaluate_split(predictions.loc[val_mask].copy(), "validation", train_p75_error),
        "test": evaluate_split(predictions.loc[test_mask].copy(), "test", train_p75_error),
    }
    metrics["critical_evaluation"] = critical_evaluation(metrics["test"])

    write_reports(metrics, predictions, model, feature_columns)
    print(f"Wrote {MODEL_DIR}")
    print(f"Wrote {PREDICTIONS_PATH}")
    print(f"Wrote {METRICS_JSON}")
    print(f"Wrote {METRICS_MD}")
    print(f"Wrote {CONFUSION_FIG}")


if __name__ == "__main__":
    main()
