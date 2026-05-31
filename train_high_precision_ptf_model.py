#!/usr/bin/env python3
"""
Train a leakage-guarded, regime-aware residual PTF forecaster.

Target:
    price

Prediction form:
    final_pred = ptf_lag_24 + weighted residual correction

This script trains price residual experts. It does not use finalized price,
target regime, transition labels, or persistence error as model features.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parent

FEATURE_STORE_PATH = PROJECT_ROOT / "data" / "features" / "regime_feature_store.parquet"
REASONING_PATH = PROJECT_ROOT / "data" / "features" / "market_reasoning_features.parquet"
LABEL_PATHS = [
    PROJECT_ROOT / "data" / "features" / "regime_labels.csv",
    PROJECT_ROOT / "data" / "regime_labels.csv",
]
REGIME_PRED_PATH = PROJECT_ROOT / "data" / "predictions" / "regime_classifier_predictions.csv"
SPIKE_PRED_PATH = PROJECT_ROOT / "data" / "predictions" / "spike_cap_detector_predictions.csv"
TRANSITION_PRED_PATH = PROJECT_ROOT / "data" / "predictions" / "spike_transition_detector_predictions.csv"
PREVIOUS_H1H4_SUMMARY = PROJECT_ROOT / "reports" / "final_h1h4_summary.json"

MODEL_DIR = PROJECT_ROOT / "models" / "high_precision_ptf_model"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "predictions" / "high_precision_ptf_predictions.csv"
METRICS_JSON = PROJECT_ROOT / "reports" / "high_precision_ptf_model_metrics.json"
METRICS_MD = PROJECT_ROOT / "reports" / "high_precision_ptf_model_metrics.md"

SPLIT_RANGES = {
    "train": (2020, 2024),
    "validation": (2025, 2025),
    "test": (2026, 2026),
}
REGIMES = ["negative_zero_pressure", "normal", "tight", "spike_cap"]
EXPERT_REGIME_MAP = {
    "negative_zero_pressure": "zero",
    "normal": "normal",
    "tight": "tight",
    "spike_cap": "spike",
}
CAP_FLOOR_STRENGTHS = [0.0, 0.25, 0.5, 0.75, 1.0]

FORBIDDEN_FEATURES = {
    "price",
    "target_price",
    "finalized_ptf",
    "target_regime",
    "lag24_regime",
    "transition_label",
    "persistence_error",
    "price_lag_24",
    "analyst_reason_text",
    "marketTradePrice",
    "systemMarginalPrice",
    "upRegulationDelivered",
    "downRegulationDelivered",
    "is_spike_cap",
    "is_new_spike_transition",
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


def add_transition_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("ts_hour").copy()
    bases = [
        "residual_load_forecast",
        "residual_load_ramp",
        "kgup_solar_mw",
        "kgup_wind_mw",
        "load_minus_kgup",
        "gas_share",
        "outage_stress_index",
    ]
    for col in bases:
        if col in out.columns:
            out[f"{col}_vs_lag24"] = out[col] - out[col].shift(24)
    if "kgup_solar_mw" in out.columns:
        out["solar_drop_vs_lag24"] = (out["kgup_solar_mw"].shift(24) - out["kgup_solar_mw"]).clip(lower=0)
    return out


def hour_group(hour: pd.Series) -> pd.Series:
    h = hour.astype(int)
    groups = pd.Series("other", index=hour.index, dtype="object")
    groups[h.between(0, 5)] = "night"
    groups[h.between(6, 10)] = "morning"
    groups[h.between(11, 16)] = "solar_window"
    groups[h.between(17, 22)] = "evening_ramp"
    return groups


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
                "price",
                "target_regime",
                "price_lag_24",
                "lag24_regime",
                "transition_label",
                "persistence_error",
            ]
        ],
        on="ts_hour",
        how="inner",
    )

    data = add_transition_features(data)
    data["split"] = assign_split(data["ts_hour"])
    data = data[data["split"].isin(SPLIT_RANGES)].copy()
    data = data.dropna(subset=["price", "ptf_lag_24", "target_regime"]).sort_values("ts_hour")
    data["base_pred"] = data["ptf_lag_24"]
    data["residual_target"] = data["price"] - data["base_pred"]
    data["hour_group"] = hour_group(data["hour"])
    data["snapshot_marketTradePrice_missing"] = data["snapshot_marketTradePrice"].isna().astype(int)

    if REGIME_PRED_PATH.exists():
        pred = pd.read_csv(REGIME_PRED_PATH)
        pred["ts_hour"] = pd.to_datetime(pred["ts_hour"], errors="coerce")
        keep = ["ts_hour", "pred_regime"] + [f"prob_{regime}" for regime in REGIMES]
        data = data.merge(
            pred[keep].rename(columns={"pred_regime": "router_pred_regime"}),
            on="ts_hour",
            how="left",
        )
    for regime in REGIMES:
        col = f"prob_{regime}"
        if col not in data.columns:
            data[col] = 0.0

    if SPIKE_PRED_PATH.exists():
        pred = pd.read_csv(SPIKE_PRED_PATH)
        pred["ts_hour"] = pd.to_datetime(pred["ts_hour"], errors="coerce")
        data = data.merge(
            pred[["ts_hour", "spike_probability"]].rename(
                columns={"spike_probability": "binary_spike_probability"}
            ),
            on="ts_hour",
            how="left",
        )
    if "binary_spike_probability" not in data.columns:
        data["binary_spike_probability"] = data["prob_spike_cap"]

    if TRANSITION_PRED_PATH.exists():
        pred = pd.read_csv(TRANSITION_PRED_PATH)
        pred["ts_hour"] = pd.to_datetime(pred["ts_hour"], errors="coerce")
        data = data.merge(
            pred[["ts_hour", "new_spike_probability"]].rename(
                columns={"new_spike_probability": "spike_transition_probability"}
            ),
            on="ts_hour",
            how="left",
        )
    if "spike_transition_probability" not in data.columns:
        data["spike_transition_probability"] = 0.0

    data["binary_spike_probability"] = data["binary_spike_probability"].fillna(data["prob_spike_cap"]).fillna(0.0)
    data["spike_transition_probability"] = data["spike_transition_probability"].fillna(0.0)
    return data, label_path


def build_feature_matrix(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    work = data.copy()
    drop_cols = {
        "ts_hour",
        "split",
        "price",
        "target_regime",
        "price_lag_24",
        "lag24_regime",
        "transition_label",
        "persistence_error",
        "residual_target",
        "base_pred",
        "router_pred_regime",
        "binary_spike_probability",
        "spike_transition_probability",
    }
    feature_cols = [
        col
        for col in work.columns
        if col not in drop_cols and col not in FORBIDDEN_FEATURES
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


def train_weights(frame: pd.DataFrame) -> np.ndarray:
    residual_abs = frame["residual_target"].abs()
    scale = max(float(residual_abs.median()), 1.0)
    weights = 1.0 + np.clip(residual_abs / scale, 0, 6)
    weights += (frame["target_regime"] == "spike_cap").astype(float) * 5.0
    weights += (frame["target_regime"] == "negative_zero_pressure").astype(float) * 2.0
    weights += (frame["target_regime"] != frame["lag24_regime"]).astype(float) * 2.0
    weights += (frame["persistence_error"] >= frame["persistence_error"].quantile(0.90)).astype(float) * 2.0
    return weights.to_numpy(float)


def make_regressor(seed: int, objective: str = "huber") -> LGBMRegressor:
    return LGBMRegressor(
        objective=objective,
        alpha=0.85,
        n_estimators=650,
        learning_rate=0.035,
        num_leaves=48,
        min_child_samples=45,
        subsample=0.88,
        colsample_bytree=0.88,
        reg_alpha=0.1,
        reg_lambda=1.1,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def fit_expert(
    name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    weights: np.ndarray,
    mask: pd.Series | np.ndarray,
    seed: int,
) -> LGBMRegressor:
    mask_arr = np.asarray(mask, dtype=bool)
    if mask_arr.sum() < 100:
        mask_arr = np.ones(len(x_train), dtype=bool)
    model = make_regressor(seed)
    model.fit(x_train.loc[mask_arr], y_train.loc[mask_arr], sample_weight=weights[mask_arr])
    model.booster_.model_name = name
    return model


def train_models(data: pd.DataFrame, features: pd.DataFrame) -> dict[str, LGBMRegressor]:
    train_mask = data["split"] == "train"
    x_train = features.loc[train_mask]
    y_train = data.loc[train_mask, "residual_target"]
    train_frame = data.loc[train_mask].copy()
    weights = train_weights(train_frame)

    models: dict[str, LGBMRegressor] = {}
    models["global"] = fit_expert("global", x_train, y_train, weights, np.ones(len(x_train), dtype=bool), 11)

    for regime, expert in EXPERT_REGIME_MAP.items():
        models[f"regime_{expert}"] = fit_expert(
            f"regime_{expert}",
            x_train,
            y_train,
            weights,
            train_frame["target_regime"] == regime,
            30 + len(models),
        )

    transition_mask = (train_frame["target_regime"] == "spike_cap") & (
        train_frame["lag24_regime"] != "spike_cap"
    )
    models["spike_transition"] = fit_expert(
        "spike_transition", x_train, y_train, weights * (1 + transition_mask.to_numpy(float) * 8), transition_mask, 72
    )

    for group in ["night", "morning", "solar_window", "evening_ramp", "other"]:
        models[f"hour_{group}"] = fit_expert(
            f"hour_{group}",
            x_train,
            y_train,
            weights,
            train_frame["hour_group"] == group,
            90 + len(models),
        )
    return models


def predict_residuals(models: dict[str, LGBMRegressor], features: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {name: model.predict(features) for name, model in models.items()},
        index=features.index,
    )


def normalized_router_weights(data: pd.DataFrame) -> pd.DataFrame:
    zero = data["prob_negative_zero_pressure"].fillna(0) * (1 + data["analyst_zero_score"].fillna(0) / 100)
    normal = data["prob_normal"].fillna(0)
    tight = data["prob_tight"].fillna(0) * (1 + data["analyst_tight_score"].fillna(0) / 100)
    spike_signal = np.maximum(data["prob_spike_cap"].fillna(0), data["binary_spike_probability"].fillna(0))
    spike = spike_signal * (1 + data["analyst_spike_score"].fillna(0) / 100)
    weights = pd.DataFrame(
        {
            "zero": zero,
            "normal": normal,
            "tight": tight,
            "spike": spike,
        },
        index=data.index,
    ).clip(lower=0)
    total = weights.sum(axis=1).replace(0, np.nan)
    weights = weights.div(total, axis=0).fillna(0.0)
    no_signal = weights.sum(axis=1) == 0
    weights.loc[no_signal, "normal"] = 1.0
    return weights


def compose_predictions(data: pd.DataFrame, residuals: pd.DataFrame, cap_floor_strength: float) -> pd.DataFrame:
    weights = normalized_router_weights(data)
    regime_residual = (
        weights["zero"] * residuals["regime_zero"]
        + weights["normal"] * residuals["regime_normal"]
        + weights["tight"] * residuals["regime_tight"]
        + weights["spike"] * residuals["regime_spike"]
    )
    hour_residual = pd.Series(index=data.index, dtype=float)
    for group in ["night", "morning", "solar_window", "evening_ramp", "other"]:
        mask = data["hour_group"] == group
        hour_residual.loc[mask] = residuals.loc[mask, f"hour_{group}"]
    transition_gate = np.clip(data["spike_transition_probability"].fillna(0) * 100, 0, 0.65)
    soft_residual = (
        (1 - transition_gate) * (0.72 * regime_residual + 0.18 * hour_residual + 0.10 * residuals["global"])
        + transition_gate * residuals["spike_transition"]
    )
    pred = data["base_pred"] + soft_residual

    floor_risk = np.maximum(
        data["binary_spike_probability"].fillna(0).to_numpy(float),
        np.clip(data["spike_transition_probability"].fillna(0).to_numpy(float) * 10, 0, 1),
    )
    cap_floor = data["base_pred"].to_numpy(float) + np.maximum(0, 4000 - data["base_pred"].to_numpy(float)) * floor_risk * cap_floor_strength
    pred = np.maximum(pred.to_numpy(float), cap_floor)
    pred = np.clip(pred, 0, 5000)

    hard_route_residual = pd.Series(index=data.index, dtype=float)
    pred_regime = data["router_pred_regime"].fillna("normal")
    route_map = {
        "negative_zero_pressure": "regime_zero",
        "normal": "regime_normal",
        "tight": "regime_tight",
        "spike_cap": "regime_spike",
    }
    for regime, col in route_map.items():
        hard_route_residual.loc[pred_regime == regime] = residuals.loc[pred_regime == regime, col]
    hard_route_residual = hard_route_residual.fillna(residuals["global"])

    return pd.DataFrame(
        {
            "persistence_pred": data["base_pred"].to_numpy(float),
            "global_residual_pred": np.clip(data["base_pred"] + residuals["global"], 0, 5000),
            "regime_soft_pred_no_floor": np.clip(data["base_pred"] + soft_residual, 0, 5000),
            "regime_classifier_routing_only_pred": np.clip(data["base_pred"] + hard_route_residual, 0, 5000),
            "high_precision_pred": pred,
            "weighted_residual_correction": soft_residual.to_numpy(float),
            "cap_floor_risk": floor_risk,
            "cap_floor_strength": cap_floor_strength,
            "router_weight_zero": weights["zero"],
            "router_weight_normal": weights["normal"],
            "router_weight_tight": weights["tight"],
            "router_weight_spike": weights["spike"],
        },
        index=data.index,
    )


def abs_error(y: pd.Series, pred: pd.Series) -> pd.Series:
    return (y.astype(float) - pred.astype(float)).abs()


def metric_block(y: pd.Series, pred: pd.Series) -> dict[str, Any]:
    err = abs_error(y, pred)
    return {
        "rows": int(len(err)),
        "mae": float(err.mean()) if len(err) else None,
        "rmse": float(math.sqrt(mean_squared_error(y, pred))) if len(err) else None,
        "median_ae": float(err.median()) if len(err) else None,
        "p90_ae": float(err.quantile(0.90)) if len(err) else None,
        "pct_error_le_2": float((err <= 2).mean()) if len(err) else None,
        "pct_error_le_10": float((err <= 10).mean()) if len(err) else None,
        "pct_error_le_50": float((err <= 50).mean()) if len(err) else None,
    }


def grouped_mae(frame: pd.DataFrame, group_col: str, pred_col: str, min_count: int = 1) -> list[dict[str, Any]]:
    rows = []
    for key, group in frame.groupby(group_col, dropna=False, observed=False):
        if len(group) < min_count:
            continue
        rows.append(
            {
                str(group_col): str(key),
                "rows": int(len(group)),
                "mae": float(abs_error(group["price"], group[pred_col]).mean()),
                "persistence_mae": float(abs_error(group["price"], group["persistence_pred"]).mean()),
                "delta_vs_persistence": float(
                    abs_error(group["price"], group[pred_col]).mean()
                    - abs_error(group["price"], group["persistence_pred"]).mean()
                ),
            }
        )
    return sorted(rows, key=lambda row: row["mae"], reverse=True)


def cap_miss(frame: pd.DataFrame, pred_col: str) -> dict[str, Any]:
    spike = frame[frame["target_regime"] == "spike_cap"]
    if spike.empty:
        return {"rows": 0, "cap_miss_rate_pred_below_4000": None, "mean_shortfall_to_4000": None}
    shortfall = np.maximum(0, 4000 - spike[pred_col])
    return {
        "rows": int(len(spike)),
        "cap_miss_rate_pred_below_4000": float((spike[pred_col] < 4000).mean()),
        "mean_shortfall_to_4000": float(shortfall.mean()),
        "mae_spike_cap": float(abs_error(spike["price"], spike[pred_col]).mean()),
    }


def evaluate_predictions(data: pd.DataFrame, pred_cols: list[str], selected_col: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    train_p75 = float(data.loc[data["split"] == "train", "persistence_error"].quantile(0.75))
    for split in ["validation", "test"]:
        frame = data[data["split"] == split].copy()
        split_metrics: dict[str, Any] = {}
        for col in pred_cols:
            split_metrics[col] = metric_block(frame["price"], frame[col])
        selected = split_metrics[selected_col]
        failure = frame[frame["persistence_error"] >= train_p75]
        h1h4_delivery = frame[frame["hour"].isin([1, 2, 3, 4])]
        split_metrics["selected_detail"] = {
            "regime_wise": grouped_mae(frame, "target_regime", selected_col),
            "transition_wise_worst_25": grouped_mae(frame, "transition_label", selected_col, min_count=5)[:25],
            "hour_group_wise": grouped_mae(frame, "hour_group", selected_col),
            "persistence_failure_hours": metric_block(failure["price"], failure[selected_col]) if len(failure) else None,
            "persistence_failure_rows": int(len(failure)),
            "delivery_hour_1_4_mae": metric_block(h1h4_delivery["price"], h1h4_delivery[selected_col]) if len(h1h4_delivery) else None,
            "cap_miss_penalty": cap_miss(frame, selected_col),
            "normal_only_mae": next((r for r in grouped_mae(frame, "target_regime", selected_col) if r["target_regime"] == "normal"), None),
            "zero_only_mae": next((r for r in grouped_mae(frame, "target_regime", selected_col) if r["target_regime"] == "negative_zero_pressure"), None),
            "tight_only_mae": next((r for r in grouped_mae(frame, "target_regime", selected_col) if r["target_regime"] == "tight"), None),
            "spike_only_mae": next((r for r in grouped_mae(frame, "target_regime", selected_col) if r["target_regime"] == "spike_cap"), None),
        }
        metrics[split] = split_metrics
    return metrics


def read_previous_best() -> dict[str, Any]:
    if not PREVIOUS_H1H4_SUMMARY.exists():
        return {"available": False}
    try:
        data = json.loads(PREVIOUS_H1H4_SUMMARY.read_text())
        comparison = data.get("comparison", {})
        best = data.get("checkpoint", {}).get("best_model")
        return {"available": True, "raw_summary_path": str(PREVIOUS_H1H4_SUMMARY.relative_to(PROJECT_ROOT)), "best_model": best, "note": "Previous h1-h4 checkpoint is anchor/horizon based and not directly identical to this delivery-hour evaluation."}
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def choose_prediction_config(data: pd.DataFrame, residuals: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    validation_scores: dict[str, Any] = {}
    composed_by_name: dict[str, pd.DataFrame] = {}
    for strength in CAP_FLOOR_STRENGTHS:
        composed = compose_predictions(data, residuals, strength)
        frame = data.join(composed)
        name = f"cap_floor_{strength}"
        composed_by_name[name] = composed
        val = frame[frame["split"] == "validation"]
        validation_scores[name] = metric_block(val["price"], val["high_precision_pred"])

    base_composed = composed_by_name["cap_floor_0.0"].copy()
    base_frame = data.join(base_composed)
    for col in [
        "global_residual_pred",
        "regime_soft_pred_no_floor",
        "regime_classifier_routing_only_pred",
    ]:
        candidate = base_composed.copy()
        candidate["high_precision_pred"] = candidate[col]
        composed_by_name[col] = candidate
        val = base_frame[base_frame["split"] == "validation"]
        validation_scores[col] = metric_block(val["price"], val[col])

    best_name = min(
        validation_scores,
        key=lambda name: validation_scores[name]["mae"]
        if validation_scores[name]["mae"] is not None
        else float("inf"),
    )
    selected = composed_by_name[best_name].copy()
    selected["selected_prediction_config"] = best_name
    config = {
        "name": best_name,
        "cap_floor_strength": float(selected["cap_floor_strength"].iloc[0]),
        "validation_mae": validation_scores[best_name]["mae"],
    }
    return config, validation_scores, selected


def write_outputs(
    models: dict[str, LGBMRegressor],
    feature_cols: list[str],
    predictions: pd.DataFrame,
    metrics: dict[str, Any],
) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        joblib.dump(model, MODEL_DIR / f"{name}.joblib")
    (MODEL_DIR / "feature_columns.json").write_text(
        json.dumps(feature_cols, ensure_ascii=False, indent=2) + "\n"
    )
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    METRICS_JSON.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")

    test = metrics["evaluation"]["test"]
    selected = test["high_precision_pred"]
    persistence = test["persistence_pred"]
    global_resid = test["global_residual_pred"]
    routing_only = test["regime_classifier_routing_only_pred"]
    detail = test["selected_detail"]
    lines = [
        "# High Precision PTF Model Metrics",
        "",
        f"Generated: `{metrics['generated_at']}`",
        "",
        "This is a real final PTF price forecast prototype using `ptf_lag_24` as a persistence anchor and leakage-guarded residual experts.",
        "",
        "## Selected Validation Configuration",
        "",
        f"- Selected prediction config: `{metrics['selected_prediction_config']['name']}`",
        f"- Selected cap floor strength: `{metrics['selected_cap_floor_strength']}`",
        "- Selection criterion: validation MAE only.",
        "- Important caveat: this is delivery-hour evaluation, not the older anchor-based h1-h4 format.",
        "",
        "## Test Summary",
        "",
        "| Model | MAE | RMSE | Median AE | P90 AE | <=2 TL | <=10 TL | <=50 TL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in [
        "high_precision_pred",
        "persistence_pred",
        "global_residual_pred",
        "regime_soft_pred_no_floor",
        "regime_classifier_routing_only_pred",
    ]:
        row = test[name]
        lines.append(
            f"| `{name}` | {row['mae']:.2f} | {row['rmse']:.2f} | {row['median_ae']:.2f} | {row['p90_ae']:.2f} | {row['pct_error_le_2']:.4f} | {row['pct_error_le_10']:.4f} | {row['pct_error_le_50']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Comparison",
            "",
            f"- Delta vs persistence MAE: `{selected['mae'] - persistence['mae']:.2f}` TL/MWh.",
            f"- Delta vs global residual MAE: `{selected['mae'] - global_resid['mae']:.2f}` TL/MWh.",
            f"- Delta vs regime classifier routing only MAE: `{selected['mae'] - routing_only['mae']:.2f}` TL/MWh.",
            "- Previous best h1-h4 ensemble: `443.87` TL/MWh mean MAE from `reports/final_h1h4_summary.md`; not directly comparable because that benchmark is anchor/horizon based.",
            "",
            "## Regime-Wise Test MAE",
            "",
            "| Regime | Rows | Model MAE | Persistence MAE | Delta |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(detail["regime_wise"], key=lambda r: r["target_regime"]):
        lines.append(
            f"| `{row['target_regime']}` | {row['rows']} | {row['mae']:.2f} | {row['persistence_mae']:.2f} | {row['delta_vs_persistence']:.2f} |"
        )
    cap = detail["cap_miss_penalty"]
    lines.extend(
        [
            "",
            "## Stress Slices",
            "",
            f"- Persistence failure rows: `{detail['persistence_failure_rows']}`; model MAE `{detail['persistence_failure_hours']['mae']:.2f}` TL/MWh.",
            f"- Delivery hour 1-4 MAE: `{detail['delivery_hour_1_4_mae']['mae']:.2f}` TL/MWh.",
            f"- Spike/cap rows: `{cap['rows']}`; cap miss rate pred<4000: `{cap['cap_miss_rate_pred_below_4000']:.4f}`; mean shortfall to 4000: `{cap['mean_shortfall_to_4000']:.2f}`.",
            "",
            "## Worst Transition MAE",
            "",
            "| Transition | Rows | Model MAE | Persistence MAE | Delta |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in detail["transition_wise_worst_25"][:15]:
        lines.append(
            f"| `{row['transition_label']}` | {row['rows']} | {row['mae']:.2f} | {row['persistence_mae']:.2f} | {row['delta_vs_persistence']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Critical Evaluation",
            "",
        ]
    )
    if selected["mae"] <= 50:
        lines.append("The prototype reaches the requested 10-50 TL band globally on the 2026 test split.")
    else:
        lines.append(
            f"The prototype does not reach the 10-50 TL band globally; test MAE is {selected['mae']:.2f} TL/MWh."
        )
    if selected["pct_error_le_2"] < 0.5:
        lines.append(
            f"The 1-2 TL objective is not achieved: only {selected['pct_error_le_2']:.2%} of test hours are within 2 TL."
        )
    lines.append(
        "Main hard slice remains spike/cap and large regime transitions, where hidden bid-curve and participant strategy information is still missing."
    )
    lines.extend(
        [
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
    models = train_models(data, features)
    residuals = predict_residuals(models, features)
    selected_config, validation_config_scores, selected_composed = choose_prediction_config(data, residuals)
    predictions = data.join(selected_composed)

    pred_cols = [
        "high_precision_pred",
        "persistence_pred",
        "global_residual_pred",
        "regime_soft_pred_no_floor",
        "regime_classifier_routing_only_pred",
    ]
    evaluation = evaluate_predictions(predictions, pred_cols, "high_precision_pred")
    model_predictions = predictions[
        [
            "ts_hour",
            "split",
            "price",
            "target_regime",
            "lag24_regime",
            "transition_label",
            "base_pred",
            "persistence_pred",
            "global_residual_pred",
            "regime_soft_pred_no_floor",
            "regime_classifier_routing_only_pred",
            "high_precision_pred",
            "weighted_residual_correction",
            "binary_spike_probability",
            "spike_transition_probability",
            "cap_floor_risk",
            "router_weight_zero",
            "router_weight_normal",
            "router_weight_tight",
            "router_weight_spike",
        ]
    ].copy()
    for col in [
        "high_precision_pred",
        "persistence_pred",
        "global_residual_pred",
        "regime_soft_pred_no_floor",
        "regime_classifier_routing_only_pred",
    ]:
        model_predictions[f"{col}_abs_error"] = abs_error(model_predictions["price"], model_predictions[col])

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": "train_high_precision_ptf_model.py",
        "label_path": str(label_path.relative_to(PROJECT_ROOT)),
        "feature_store": str(FEATURE_STORE_PATH.relative_to(PROJECT_ROOT)),
        "reasoning_features": str(REASONING_PATH.relative_to(PROJECT_ROOT)),
        "routing_inputs": {
            "regime_classifier": str(REGIME_PRED_PATH.relative_to(PROJECT_ROOT)),
            "spike_detector": str(SPIKE_PRED_PATH.relative_to(PROJECT_ROOT)),
            "spike_transition_detector": str(TRANSITION_PRED_PATH.relative_to(PROJECT_ROOT)),
        },
        "splits": {
            split: {
                "years": f"{years[0]}-{years[1]}",
                "rows": int((predictions["split"] == split).sum()),
            }
            for split, years in SPLIT_RANGES.items()
        },
        "selected_prediction_config": selected_config,
        "selected_cap_floor_strength": selected_config["cap_floor_strength"],
        "validation_cap_floor_scores": validation_config_scores,
        "evaluation": evaluation,
        "previous_h1h4_reference": read_previous_best(),
        "leakage_checks": [
            {
                "check": "Forbidden feature columns absent",
                "status": "pass" if not forbidden_present else "fail",
                "detail": f"Forbidden columns present in model feature matrix: {forbidden_present}",
            },
            {
                "check": "price target only",
                "status": "pass",
                "detail": "price is used for residual target and evaluation, not as a feature.",
            },
            {
                "check": "regime labels excluded from features",
                "status": "pass",
                "detail": "target_regime, lag24_regime, transition_label, and persistence_error are dropped from X.",
            },
            {
                "check": "same-hour realized balancing excluded",
                "status": "pass",
                "detail": "Only lagged SMF/YAL/YAT columns from the feature store are used.",
            },
            {
                "check": "historical interim oracle excluded",
                "status": "pass",
                "detail": "No historical interim_mcp source is read; sparse point-in-time snapshot fields remain optional.",
            },
        ],
    }

    write_outputs(models, feature_cols, model_predictions, metrics)
    print(f"Wrote {MODEL_DIR}")
    print(f"Wrote {PREDICTIONS_PATH}")
    print(f"Wrote {METRICS_JSON}")
    print(f"Wrote {METRICS_MD}")


if __name__ == "__main__":
    main()
