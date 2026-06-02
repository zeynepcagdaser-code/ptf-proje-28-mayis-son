#!/usr/bin/env python3
"""
Train a fuel-switch aware PTF forecaster.

This model treats the relationship discussed in the market analysis as a
routing problem:

    weak demand + high renewable/hydro + gas off -> zero/low price expert
    high demand + high gas marginality -> tight/spike expert

It does not replace the existing high-precision model. It produces a separate
set of predictions and reports so we can measure whether this market mechanism
helps beyond ordinary regression features.
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
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent
FEATURE_STORE_PATH = PROJECT_ROOT / "data" / "features" / "regime_feature_store.parquet"
REASONING_PATH = PROJECT_ROOT / "data" / "features" / "market_reasoning_features.parquet"
LABEL_PATH = PROJECT_ROOT / "data" / "regime_labels.csv"

MODEL_DIR = PROJECT_ROOT / "models" / "fuel_switch_routed_ptf_model"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "predictions" / "fuel_switch_routed_ptf_predictions.csv"
REPORT_JSON = PROJECT_ROOT / "reports" / "fuel_switch_routed_ptf_metrics.json"
REPORT_MD = PROJECT_ROOT / "reports" / "fuel_switch_routed_ptf_metrics.md"

SPLIT_RANGES = {
    "train": (2020, 2024),
    "validation": (2025, 2025),
    "test": (2026, 2026),
}

FUEL_SWITCH_COLUMNS = [
    "gas_marginality_proxy",
    "hydro_displacement_score",
    "renewable_share_of_generation",
    "gas_share_of_generation",
    "renewable_minus_gas_shift",
    "cheap_supply_pressure",
    "low_demand_flag",
    "gas_off_flag",
    "renewable_share_high_flag",
    "hydro_high_flag",
    "zero_price_pressure_score",
    "load_deviation_from_weekly_norm",
    "load_deviation_from_monthly_norm",
    "demand_weakness_score",
    "load_vs_renewable_balance",
]

FORBIDDEN_FEATURES = {
    "price",
    "target_price",
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
    "is_zero_pressure",
    "is_low_price",
    "is_high_price",
    "residual_target",
    "base_pred",
    "split",
}


def assign_split(ts: pd.Series) -> pd.Series:
    years = ts.dt.year
    split = pd.Series(index=ts.index, dtype="object")
    for name, (start_year, end_year) in SPLIT_RANGES.items():
        split[(years >= start_year) & (years <= end_year)] = name
    return split


def hour_group(hour: pd.Series) -> pd.Series:
    h = hour.astype(int)
    groups = pd.Series("other", index=hour.index, dtype="object")
    groups[h.between(0, 5)] = "night"
    groups[h.between(6, 10)] = "morning"
    groups[h.between(11, 16)] = "solar_window"
    groups[h.between(17, 22)] = "evening_ramp"
    return groups


def load_dataset() -> pd.DataFrame:
    from src.utils.io_utils import read_parquet_with_normalized_ts
    features = read_parquet_with_normalized_ts(FEATURE_STORE_PATH)
    reasoning = read_parquet_with_normalized_ts(REASONING_PATH)
    labels = pd.read_csv(LABEL_PATH)
    for frame in [features, reasoning, labels]:
        frame["ts_hour"] = pd.to_datetime(frame["ts_hour"], errors="coerce")

    data = (
        features.merge(reasoning, on="ts_hour", how="inner")
        .merge(
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
        .sort_values("ts_hour")
        .reset_index(drop=True)
    )
    data["split"] = assign_split(data["ts_hour"])
    data = data[data["split"].isin(SPLIT_RANGES)].copy()
    data = data.dropna(subset=["price", "ptf_lag_24", "target_regime"])
    data["base_pred"] = data["ptf_lag_24"]
    data["residual_target"] = data["price"] - data["base_pred"]
    data["is_zero_pressure"] = data["price"].le(50).astype(int)
    data["is_low_price"] = data["price"].le(1500).astype(int)
    data["is_high_price"] = data["price"].ge(1500).astype(int)
    data["is_spike_cap"] = data["price"].ge(4000).astype(int)
    data["hour_group"] = hour_group(data["hour"])

    for col in [
        "residual_load_forecast",
        "residual_load_ramp",
        "kgup_solar_mw",
        "kgup_wind_mw",
        "load_minus_kgup",
        "gas_marginality_proxy",
        "hydro_displacement_score",
        "zero_price_pressure_score",
        "load_vs_renewable_balance",
    ]:
        if col in data.columns:
            data[f"{col}_vs_lag24"] = data[col] - data[col].shift(24)
    if "kgup_solar_mw" in data.columns:
        data["solar_drop_vs_lag24"] = (
            data["kgup_solar_mw"].shift(24) - data["kgup_solar_mw"]
        ).clip(lower=0)
    return data


def build_matrix(
    data: pd.DataFrame,
    extra_drop: set[str] | None = None,
    selected_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    work = data.copy()
    drop_cols = set(FORBIDDEN_FEATURES)
    drop_cols.update({"ts_hour"})
    if extra_drop:
        drop_cols.update(extra_drop)
    if selected_columns is not None:
        feature_cols = [col for col in selected_columns if col in work.columns and col not in drop_cols]
    else:
        feature_cols = [col for col in work.columns if col not in drop_cols]
    frame = work[feature_cols].copy()
    cat_cols = [
        col
        for col in frame.columns
        if pd.api.types.is_object_dtype(frame[col])
        or pd.api.types.is_string_dtype(frame[col])
        or isinstance(frame[col].dtype, pd.CategoricalDtype)
    ]
    frame = pd.get_dummies(frame, columns=cat_cols, dummy_na=True, dtype=float)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame, list(frame.columns)


def class_weight(y: pd.Series) -> np.ndarray:
    pos = max(int(y.sum()), 1)
    neg = max(int(len(y) - y.sum()), 1)
    ratio = neg / pos
    return np.where(y == 1, min(ratio, 30.0), 1.0)


def make_classifier(seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        n_estimators=450,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=35,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.05,
        reg_lambda=0.8,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


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


def train_state_classifiers(data: pd.DataFrame) -> tuple[dict[str, LGBMClassifier], pd.DataFrame, list[str]]:
    state_features = [
        "hour",
        "weekday",
        "weekend",
        "month",
        "ptf_lag_24",
        "ptf_lag_168",
        "price_band_persistence",
        "rolling_volatility",
        "volatility_cluster_score",
        "load_forecast",
        "load_ramp_1h",
        "load_ramp_3h",
        "residual_load_forecast",
        "residual_load_ramp",
        "load_minus_kgup",
        "kgup_total",
        "kgup_wind_mw",
        "kgup_solar_mw",
        "kgup_renewable_mw",
        "solar_cliff_score",
        "wind_relief_score",
        "renewable_oversupply_score",
        "active_maintenance_capacity",
        "outage_stress_index",
        "analyst_zero_score",
        "analyst_spike_score",
        "analyst_tight_score",
        "analyst_persistence_break_score",
    ] + FUEL_SWITCH_COLUMNS
    state_features += [c for c in data.columns if c.endswith("_vs_lag24")]
    x, cols = build_matrix(data, selected_columns=state_features)
    train = data["split"].eq("train")
    models: dict[str, LGBMClassifier] = {}
    preds = pd.DataFrame(index=data.index)

    targets = {
        "zero_pressure_state": "is_zero_pressure",
        "low_price_state": "is_low_price",
        "high_price_state": "is_high_price",
        "spike_state": "is_spike_cap",
    }
    for name, target_col in targets.items():
        model = make_classifier(seed=100 + len(models))
        y = data[target_col].astype(int)
        model.fit(x.loc[train], y.loc[train], sample_weight=class_weight(y.loc[train]))
        models[name] = model
        preds[f"{name}_prob"] = model.predict_proba(x)[:, 1]
    return models, preds, cols


def residual_weights(frame: pd.DataFrame, expert: str) -> np.ndarray:
    err = frame["residual_target"].abs()
    scale = max(float(err.median()), 1.0)
    weights = 1.0 + np.clip(err / scale, 0, 6)
    if expert == "zero":
        weights += frame["is_zero_pressure"].astype(float) * 10
        weights += frame["zero_price_pressure_score"].fillna(0).to_numpy(float) / 20
    elif expert == "low":
        weights += frame["is_low_price"].astype(float) * 4
        weights += frame["cheap_supply_pressure"].fillna(0).to_numpy(float) / 25
    elif expert == "gas":
        weights += frame["is_high_price"].astype(float) * 4
        weights += frame["gas_marginality_proxy"].fillna(0).to_numpy(float) / 25
    elif expert == "spike":
        weights += frame["is_spike_cap"].astype(float) * 8
        weights += frame["analyst_spike_score"].fillna(0).to_numpy(float) / 20
    weights += (frame["target_regime"] != frame["lag24_regime"]).astype(float) * 2
    return np.asarray(weights, dtype=float)


def train_regressors(data: pd.DataFrame) -> tuple[dict[str, LGBMRegressor], pd.DataFrame, list[str]]:
    x, cols = build_matrix(data)
    train = data["split"].eq("train")
    train_frame = data.loc[train].copy()
    x_train = x.loc[train]
    y_train = train_frame["residual_target"]

    expert_masks = {
        "global": np.ones(len(train_frame), dtype=bool),
        "zero": train_frame["is_zero_pressure"].eq(1)
        | train_frame["zero_price_pressure_score"].ge(65),
        "low": train_frame["is_low_price"].eq(1)
        | train_frame["cheap_supply_pressure"].ge(65),
        "gas": train_frame["is_high_price"].eq(1)
        | train_frame["gas_marginality_proxy"].ge(60),
        "spike": train_frame["is_spike_cap"].eq(1)
        | train_frame["analyst_spike_score"].ge(65),
    }
    models: dict[str, LGBMRegressor] = {}
    for i, (name, mask) in enumerate(expert_masks.items()):
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.sum() < 100:
            mask_arr = np.ones(len(train_frame), dtype=bool)
        model = make_regressor(200 + i)
        model.fit(
            x_train.loc[mask_arr],
            y_train.loc[mask_arr],
            sample_weight=residual_weights(train_frame, name)[mask_arr],
        )
        models[name] = model

    residuals = pd.DataFrame(
        {name: model.predict(x) for name, model in models.items()},
        index=data.index,
    )
    return models, residuals, cols


def compose(data: pd.DataFrame, residuals: pd.DataFrame, state_probs: pd.DataFrame, params: dict[str, float]) -> pd.DataFrame:
    residuals = residuals.copy()
    for col in residuals.columns:
        residuals[col] = residuals[col].fillna(residuals.get("global", pd.Series(0, index=residuals.index))).fillna(0)
    zero_prob = state_probs["zero_pressure_state_prob"].fillna(0).clip(0, 1)
    low_prob = state_probs["low_price_state_prob"].fillna(0).clip(0, 1)
    high_prob = state_probs["high_price_state_prob"].fillna(0).clip(0, 1)
    spike_prob = state_probs["spike_state_prob"].fillna(0).clip(0, 1)

    zero_rule = (
        data["zero_price_pressure_score"].fillna(0) / 100
        * (0.6 + 0.4 * data["renewable_share_high_flag"].fillna(0))
        * (0.7 + 0.3 * data["gas_off_flag"].fillna(0))
    ).clip(0, 1)
    residual_rank = data["residual_load_forecast"].rank(pct=True).fillna(0.5)
    gas_rule = (data["gas_marginality_proxy"].fillna(0) / 100 * (0.5 + 0.5 * residual_rank)).clip(0, 1)
    cheap_rule = (data["cheap_supply_pressure"].fillna(0) / 100).clip(0, 1)

    w_zero = params["zero_scale"] * np.maximum(zero_prob, zero_rule)
    w_low = params["low_scale"] * np.maximum(low_prob * 0.5, cheap_rule * 0.6)
    w_gas = params["gas_scale"] * np.maximum(high_prob, gas_rule)
    w_spike = params["spike_scale"] * np.maximum(spike_prob, data["analyst_spike_score"].fillna(0) / 100)
    w_global = pd.Series(params["global_scale"], index=data.index, dtype=float)

    weights = pd.DataFrame(
        {
            "zero": w_zero,
            "low": w_low,
            "gas": w_gas,
            "spike": w_spike,
            "global": w_global,
        },
        index=data.index,
    ).clip(lower=0)
    weights = weights.div(weights.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    residual = (
        weights["zero"] * residuals["zero"]
        + weights["low"] * residuals["low"]
        + weights["gas"] * residuals["gas"]
        + weights["spike"] * residuals["spike"]
        + weights["global"] * residuals["global"]
    )
    pred = data["base_pred"] + residual
    pred = pred.fillna(data["base_pred"]).fillna(0)

    # When the learned/state gate says zero pressure, softly pull toward the
    # low-price expert instead of letting yesterday's price dominate.
    zero_gate = np.maximum(zero_prob, zero_rule) ** params["zero_pull_power"]
    low_anchor = np.minimum(pred, params["zero_anchor_price"])
    pred = (1 - zero_gate * params["zero_pull_strength"]) * pred + (
        zero_gate * params["zero_pull_strength"]
    ) * low_anchor

    # In gas marginal / spike states, avoid predicting a normal price when the
    # system is near cap. This is a risk-aware floor, not a target lookup.
    high_gate = np.maximum(spike_prob, gas_rule * high_prob)
    if params["cap_floor_strength"] > 0:
        floor = (
            data["base_pred"]
            + np.maximum(0, params["cap_reference"] - data["base_pred"])
            * high_gate
            * params["cap_floor_strength"]
        )
        pred = np.maximum(pred, floor)
    pred = pd.Series(np.clip(pred, 0, 5000), index=data.index).fillna(data["base_pred"]).fillna(0)

    return pd.DataFrame(
        {
            "fuel_switch_pred": pred,
            "fuel_switch_residual": residual,
            "persistence_pred": data["base_pred"],
            "zero_pressure_state_prob": zero_prob,
            "low_price_state_prob": low_prob,
            "high_price_state_prob": high_prob,
            "spike_state_prob": spike_prob,
            "zero_rule_gate": zero_rule,
            "gas_rule_gate": gas_rule,
            "cheap_supply_gate": cheap_rule,
            "router_weight_zero": weights["zero"],
            "router_weight_low": weights["low"],
            "router_weight_gas": weights["gas"],
            "router_weight_spike": weights["spike"],
            "router_weight_global": weights["global"],
        },
        index=data.index,
    )


def abs_error(y: pd.Series, pred: pd.Series) -> pd.Series:
    return (y.astype(float) - pred.astype(float)).abs()


def metric_block(y: pd.Series, pred: pd.Series) -> dict[str, Any]:
    valid = y.notna() & pred.notna()
    y = y.loc[valid]
    pred = pred.loc[valid]
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


def grouped_mae(frame: pd.DataFrame, group_col: str, pred_col: str) -> list[dict[str, Any]]:
    rows = []
    for key, group in frame.groupby(group_col, dropna=False, observed=False):
        rows.append(
            {
                group_col: str(key),
                "rows": int(len(group)),
                "mae": float(abs_error(group["price"], group[pred_col]).mean()),
                "persistence_mae": float(abs_error(group["price"], group["persistence_pred"]).mean()),
                "delta_vs_persistence": float(
                    abs_error(group["price"], group[pred_col]).mean()
                    - abs_error(group["price"], group["persistence_pred"]).mean()
                ),
            }
        )
    return sorted(rows, key=lambda item: item["mae"], reverse=True)


def state_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, target in [
        ("zero_pressure_state", "is_zero_pressure"),
        ("low_price_state", "is_low_price"),
        ("high_price_state", "is_high_price"),
        ("spike_state", "is_spike_cap"),
    ]:
        prob = frame[f"{name}_prob"]
        pred = prob.ge(0.5).astype(int)
        y = frame[target].astype(int)
        item = {
            "positive_rows": int(y.sum()),
            "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y, pred)),
            "confusion_matrix_at_0_5": confusion_matrix(y, pred).tolist(),
            "classification_report_at_0_5": classification_report(y, pred, output_dict=True, zero_division=0),
        }
        if y.nunique() == 2:
            item["pr_auc"] = float(average_precision_score(y, prob))
            item["roc_auc"] = float(roc_auc_score(y, prob))
        else:
            item["pr_auc"] = None
            item["roc_auc"] = None
        out[name] = item
    return out


def evaluate(frame: pd.DataFrame, params: dict[str, float]) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "selected_params": params,
        "validation": {},
        "test": {},
    }
    for split in ["validation", "test"]:
        subset = frame[frame["split"] == split].copy()
        metrics[split]["fuel_switch_pred"] = metric_block(subset["price"], subset["fuel_switch_pred"])
        metrics[split]["persistence_pred"] = metric_block(subset["price"], subset["persistence_pred"])
        metrics[split]["regime_wise"] = grouped_mae(subset, "target_regime", "fuel_switch_pred")
        metrics[split]["hour_group_wise"] = grouped_mae(subset, "hour_group", "fuel_switch_pred")
        zero = subset[subset["is_zero_pressure"] == 1]
        spike = subset[subset["is_spike_cap"] == 1]
        failure = subset[
            subset["persistence_error"]
            >= frame.loc[frame["split"] == "train", "persistence_error"].quantile(0.75)
        ]
        metrics[split]["zero_price_hours"] = metric_block(zero["price"], zero["fuel_switch_pred"]) if len(zero) else None
        metrics[split]["spike_cap_hours"] = metric_block(spike["price"], spike["fuel_switch_pred"]) if len(spike) else None
        metrics[split]["persistence_failure_hours"] = metric_block(failure["price"], failure["fuel_switch_pred"]) if len(failure) else None
        metrics[split]["state_classifiers"] = state_metrics(subset)
    return metrics


def tune_params(data: pd.DataFrame, residuals: pd.DataFrame, state_probs: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame, list[dict[str, Any]]]:
    candidates = []
    for zero_scale in [1.0, 1.5, 2.0]:
        for zero_pull_strength in [0.15, 0.35, 0.55]:
            for cap_floor_strength in [0.0, 0.25, 0.5]:
                candidates.append(
                    {
                        "zero_scale": zero_scale,
                        "low_scale": 0.8,
                        "gas_scale": 1.0,
                        "spike_scale": 0.7,
                        "global_scale": 0.7,
                        "zero_pull_strength": zero_pull_strength,
                        "zero_pull_power": 1.0,
                        "zero_anchor_price": 80.0,
                        "cap_reference": 4000.0,
                        "cap_floor_strength": cap_floor_strength,
                    }
                )
    records: list[dict[str, Any]] = []
    best_params = candidates[0]
    best_frame = pd.DataFrame()
    best_mae = float("inf")
    for params in candidates:
        pred = compose(data, residuals, state_probs, params)
        frame = data.join(pred)
        val = frame[frame["split"] == "validation"]
        mae = float(abs_error(val["price"], val["fuel_switch_pred"]).mean())
        zero_val = val[val["is_zero_pressure"] == 1]
        zero_mae = float(abs_error(zero_val["price"], zero_val["fuel_switch_pred"]).mean()) if len(zero_val) else None
        score = mae + (0 if zero_mae is None else zero_mae * 0.03)
        records.append({"params": params, "validation_mae": mae, "validation_zero_mae": zero_mae, "selection_score": score})
        if score < best_mae:
            best_mae = score
            best_params = params
            best_frame = frame
    return best_params, best_frame, sorted(records, key=lambda row: row["selection_score"])[:10]


def write_outputs(
    data: pd.DataFrame,
    state_models: dict[str, LGBMClassifier],
    state_cols: list[str],
    regressors: dict[str, LGBMRegressor],
    regressor_cols: list[str],
    frame: pd.DataFrame,
    metrics: dict[str, Any],
    top_candidates: list[dict[str, Any]],
) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)

    for name, model in state_models.items():
        joblib.dump(model, MODEL_DIR / f"{name}.joblib")
    for name, model in regressors.items():
        joblib.dump(model, MODEL_DIR / f"expert_{name}.joblib")
    (MODEL_DIR / "state_feature_columns.json").write_text(json.dumps(state_cols, indent=2) + "\n")
    (MODEL_DIR / "regressor_feature_columns.json").write_text(json.dumps(regressor_cols, indent=2) + "\n")

    pred_cols = [
        "ts_hour",
        "split",
        "price",
        "target_regime",
        "lag24_regime",
        "transition_label",
        "persistence_error",
        "persistence_pred",
        "fuel_switch_pred",
        "zero_pressure_state_prob",
        "low_price_state_prob",
        "high_price_state_prob",
        "spike_state_prob",
        "zero_rule_gate",
        "gas_rule_gate",
        "cheap_supply_gate",
        "router_weight_zero",
        "router_weight_low",
        "router_weight_gas",
        "router_weight_spike",
        "router_weight_global",
        "gas_marginality_proxy",
        "hydro_displacement_score",
        "renewable_share_of_generation",
        "gas_share_of_generation",
        "zero_price_pressure_score",
        "demand_weakness_score",
    ]
    frame[[c for c in pred_cols if c in frame.columns]].to_csv(PREDICTIONS_PATH, index=False)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(frame)),
        "feature_store": str(FEATURE_STORE_PATH.relative_to(PROJECT_ROOT)),
        "reasoning_features": str(REASONING_PATH.relative_to(PROJECT_ROOT)),
        "predictions": str(PREDICTIONS_PATH.relative_to(PROJECT_ROOT)),
        "models": str(MODEL_DIR.relative_to(PROJECT_ROOT)),
        "top_validation_candidates": top_candidates,
        "metrics": metrics,
        "mechanism": {
            "zero_pressure": "low demand + high renewable/hydro + gas off routes to zero/low expert",
            "gas_marginality": "high residual load + gas marginality routes to gas/spike expert",
            "prediction": "base ptf_lag_24 + fuel-switch weighted residual correction",
        },
        "leakage_policy": {
            "same_hour_final_ptf_feature": False,
            "target_regime_feature": False,
            "same_hour_realized_smf_yal_yat": False,
            "uses_planned_forecast_stack": True,
        },
    }
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    test = metrics["test"]
    selected = test["fuel_switch_pred"]
    persistence = test["persistence_pred"]
    lines = [
        "# Fuel Switch Routed PTF Model Metrics",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "This model explicitly routes price residual experts through zero-pressure and gas-marginality state probabilities.",
        "",
        "## Test Summary",
        "",
        "| Model | MAE | RMSE | Median AE | P90 AE | <=2 TL | <=10 TL | <=50 TL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| `fuel_switch_pred` | {selected['mae']:.2f} | {selected['rmse']:.2f} | {selected['median_ae']:.2f} | {selected['p90_ae']:.2f} | {selected['pct_error_le_2']:.4f} | {selected['pct_error_le_10']:.4f} | {selected['pct_error_le_50']:.4f} |",
        f"| `persistence_pred` | {persistence['mae']:.2f} | {persistence['rmse']:.2f} | {persistence['median_ae']:.2f} | {persistence['p90_ae']:.2f} | {persistence['pct_error_le_2']:.4f} | {persistence['pct_error_le_10']:.4f} | {persistence['pct_error_le_50']:.4f} |",
        "",
        f"- Delta vs persistence: `{selected['mae'] - persistence['mae']:.2f}` TL/MWh",
        "",
        "## Regime-Wise Test MAE",
        "",
        "| Regime | Rows | Model MAE | Persistence MAE | Delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in sorted(test["regime_wise"], key=lambda item: item["target_regime"]):
        lines.append(
            f"| `{row['target_regime']}` | {row['rows']} | {row['mae']:.2f} | {row['persistence_mae']:.2f} | {row['delta_vs_persistence']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## State Classifiers",
            "",
            "| State | Positives | PR-AUC | ROC-AUC | Balanced acc @0.5 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, item in test["state_classifiers"].items():
        pr = item["pr_auc"]
        roc = item["roc_auc"]
        lines.append(
            f"| `{name}` | {item['positive_rows']} | {pr:.4f} | {roc:.4f} | {item['balanced_accuracy_at_0_5']:.4f} |"
            if pr is not None and roc is not None
            else f"| `{name}` | {item['positive_rows']} | n/a | n/a | {item['balanced_accuracy_at_0_5']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Selected Validation Parameters",
            "",
            "```json",
            json.dumps(metrics["selected_params"], indent=2),
            "```",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    data = load_dataset()
    state_models, state_probs, state_cols = train_state_classifiers(data)
    regressors, residuals, regressor_cols = train_regressors(data)
    best_params, frame, candidates = tune_params(data, residuals, state_probs)
    metrics = evaluate(frame, best_params)
    write_outputs(data, state_models, state_cols, regressors, regressor_cols, frame, metrics, candidates)
    print(f"Wrote {REPORT_JSON}")
    print(f"Wrote {REPORT_MD}")
    print(f"Wrote {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
