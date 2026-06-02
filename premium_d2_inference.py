#!/usr/bin/env python3
"""Shared inference for premium D+2 PTF stack: HP + fuel-switch hybrid + D2 + adjustments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from build_hybrid_fuel_switch_forecast import build_prediction as hybrid_build_prediction
from train_fuel_switch_routed_ptf_model import compose as fuel_switch_compose
from train_high_precision_ptf_model import (
    add_transition_features,
    build_feature_matrix,
    compose_predictions as hp_compose_predictions,
    hour_group,
    predict_residuals as hp_predict_residuals,
)

PROJECT_ROOT = Path(__file__).resolve().parent
HP_DIR = PROJECT_ROOT / "models" / "high_precision_ptf_model"
FS_DIR = PROJECT_ROOT / "models" / "fuel_switch_routed_ptf_model"
D2_DIR = PROJECT_ROOT / "models" / "d2_ptf_forecaster"
HYBRID_REPORT = PROJECT_ROOT / "reports" / "hybrid_fuel_switch_ptf_metrics.json"
FS_REPORT = PROJECT_ROOT / "reports" / "fuel_switch_routed_ptf_metrics.json"
HP_REPORT = PROJECT_ROOT / "reports" / "high_precision_ptf_model_metrics.json"
BLEND_WEIGHTS_PATH = PROJECT_ROOT / "models" / "premium_d2_forecaster" / "blend_weights.json"


def _load_json(path: Path, key: str | None = None) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    if key and key in payload:
        return payload[key]
    return payload


def load_hybrid_params() -> dict[str, float]:
    report = _load_json(HYBRID_REPORT)
    metrics = report.get("metrics", {})
    return metrics.get("params") or report.get("top_validation_candidates", [{}])[0].get("params", {})


def load_fuel_switch_params() -> dict[str, float]:
    report = _load_json(FS_REPORT)
    return report.get("metrics", {}).get("selected_params", {})


def load_hp_cap_floor_strength() -> float:
    report = _load_json(HP_REPORT)
    return float(report.get("selected_cap_floor_strength", 0.0))


def load_blend_weights() -> dict[str, float]:
    if BLEND_WEIGHTS_PATH.exists():
        return json.loads(BLEND_WEIGHTS_PATH.read_text())
    return {
        "hybrid_weight": 0.40,
        "d2_weight": 0.35,
        "persistence_weight": 0.25,
        "must_run_zero_pull": 0.15,
        "curve_cap_nudge": 0.10,
    }


def _matrix_from_columns(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    x = pd.DataFrame(index=frame.index)
    for col in feature_cols:
        x[col] = frame[col] if col in frame.columns else 0.0
    return x.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["ts_hour"] = pd.to_datetime(out["ts_hour"], errors="coerce")
    if "anchor_d1_ptf" in out.columns:
        out["base_pred"] = pd.to_numeric(out["anchor_d1_ptf"], errors="coerce")
        out["ptf_lag_24"] = out["base_pred"]
    elif "ptf_lag_24" in out.columns:
        out["base_pred"] = pd.to_numeric(out["ptf_lag_24"], errors="coerce")
    else:
        out["base_pred"] = 0.0
        out["ptf_lag_24"] = 0.0

    if "hour" not in out.columns:
        out["hour"] = out["ts_hour"].dt.hour
    out["hour_group"] = hour_group(out["hour"])

    fuel_defaults = {
        "zero_price_pressure_score": 0.0,
        "cheap_supply_pressure": 0.0,
        "gas_marginality_proxy": 0.0,
        "hydro_displacement_score": 0.0,
        "renewable_share_of_generation": 0.0,
        "gas_share_of_generation": 0.0,
        "renewable_minus_gas_shift": 0.0,
        "low_demand_flag": 0.0,
        "gas_off_flag": 0.0,
        "renewable_share_high_flag": 0.0,
        "hydro_high_flag": 0.0,
        "load_deviation_from_weekly_norm": 0.0,
        "load_deviation_from_monthly_norm": 0.0,
        "demand_weakness_score": 0.0,
        "load_vs_renewable_balance": 0.0,
        "analyst_spike_score": 0.0,
    }
    for col, default in fuel_defaults.items():
        if col not in out.columns:
            out[col] = default

    for col, default in [
        ("prob_negative_zero_pressure", 0.05),
        ("prob_normal", 0.70),
        ("prob_tight", 0.20),
        ("prob_spike_cap", 0.05),
        ("binary_spike_probability", 0.0),
        ("spike_transition_probability", 0.0),
        ("analyst_zero_score", 0.0),
        ("analyst_spike_score", 0.0),
        ("analyst_tight_score", 0.0),
        ("analyst_persistence_break_score", 0.0),
    ]:
        if col not in out.columns:
            out[col] = default

    if "router_pred_regime" not in out.columns:
        out["router_pred_regime"] = np.select(
            [
                out["prob_negative_zero_pressure"] >= out[["prob_spike_cap", "prob_tight", "prob_normal"]].max(axis=1),
                out["prob_spike_cap"] >= out[["prob_negative_zero_pressure", "prob_tight", "prob_normal"]].max(axis=1),
                out["prob_tight"] >= out[["prob_negative_zero_pressure", "prob_spike_cap", "prob_normal"]].max(axis=1),
            ],
            ["negative_zero_pressure", "spike_cap", "tight"],
            default="normal",
        )

    out = add_transition_features(out)
    return out


def predict_high_precision(frame: pd.DataFrame) -> pd.DataFrame:
    data = prepare_frame(frame)
    feature_cols = json.loads((HP_DIR / "feature_columns.json").read_text())
    x, _, _ = build_feature_matrix(data.assign(split="inference"))
    for col in feature_cols:
        if col not in x.columns:
            x[col] = 0.0
    x = x[feature_cols].fillna(0.0)

    models = {}
    for path in HP_DIR.glob("*.joblib"):
        models[path.stem] = joblib.load(path)
    residuals = hp_predict_residuals(models, x)
    composed = hp_compose_predictions(data, residuals, load_hp_cap_floor_strength())
    return data.join(composed)


def predict_fuel_switch(frame: pd.DataFrame) -> pd.DataFrame:
    data = prepare_frame(frame)
    state_cols = json.loads((FS_DIR / "state_feature_columns.json").read_text())
    regressor_cols = json.loads((FS_DIR / "regressor_feature_columns.json").read_text())

    state_probs = pd.DataFrame(index=data.index)
    for name in ["zero_pressure_state", "low_price_state", "high_price_state", "spike_state"]:
        model = joblib.load(FS_DIR / f"{name}.joblib")
        x = _matrix_from_columns(data, state_cols)
        state_probs[f"{name}_prob"] = model.predict_proba(x)[:, 1]

    residuals = pd.DataFrame(index=data.index)
    for name in ["global", "zero", "low", "gas", "spike"]:
        model = joblib.load(FS_DIR / f"expert_{name}.joblib")
        x = _matrix_from_columns(data, regressor_cols)
        residuals[name] = model.predict(x)

    fs_params = load_fuel_switch_params()
    if not fs_params:
        fs_params = {
            "zero_scale": 1.5,
            "low_scale": 0.8,
            "gas_scale": 1.0,
            "spike_scale": 0.7,
            "global_scale": 0.7,
            "zero_pull_strength": 0.35,
            "zero_pull_power": 1.0,
            "zero_anchor_price": 80.0,
            "cap_reference": 4000.0,
            "cap_floor_strength": 0.25,
        }
    composed = fuel_switch_compose(data, residuals, state_probs, fs_params)
    return data.join(composed)


def predict_d2(frame: pd.DataFrame) -> pd.Series:
    global_model = joblib.load(D2_DIR / "global_model.joblib")
    hour_models = joblib.load(D2_DIR / "hour_models.joblib")
    hour_alphas = json.loads((D2_DIR / "hour_alphas.json").read_text())
    hour_alphas = {int(k): float(v) for k, v in hour_alphas.items()}
    feature_cols = json.loads((D2_DIR / "feature_columns.json").read_text())

    data = prepare_frame(frame)
    x = _matrix_from_columns(data, feature_cols)
    global_res = global_model.predict(x)
    blended = global_res.copy()
    for hour, model in hour_models.items():
        mask = data["hour"].astype(int).to_numpy() == hour
        if mask.any():
            blended[mask] = 0.35 * global_res[mask] + 0.65 * model.predict(x.loc[mask])
    alpha = np.array([hour_alphas.get(int(h), 0.5) for h in data["hour"].astype(int)], dtype=float)
    return pd.Series(np.clip(data["base_pred"].to_numpy(float) + alpha * blended, 0, 5000), index=data.index)


def apply_must_run_zero_pull(pred: np.ndarray, frame: pd.DataFrame, strength: float) -> np.ndarray:
    if "must_run_share" not in frame.columns:
        return pred
    share = frame["must_run_share"].fillna(0).to_numpy(float)
    solar = frame.get("must_run_solar", pd.Series(0, index=frame.index)).fillna(0).to_numpy(float)
    gate = np.clip(share * 2.0 + solar / np.maximum(frame.get("load_forecast", pd.Series(1, index=frame.index)).fillna(1).to_numpy(float), 1) * 5000, 0, 1)
    low_anchor = np.minimum(pred, 80.0)
    return (1 - gate * strength) * pred + gate * strength * low_anchor


def apply_curve_cap_nudge(pred: np.ndarray, frame: pd.DataFrame, strength: float) -> np.ndarray:
    work = prepare_frame(frame)
    cap = work.get("curve_lag48_cap_risk_score", pd.Series(0, index=work.index)).fillna(0)
    if cap.sum() == 0:
        cap = work.get("proxy_curve_lag48_cap_risk_score", pd.Series(0, index=work.index)).fillna(0)
    oversupply = work.get("curve_lag48_oversupply_pressure", pd.Series(0, index=work.index)).fillna(0)
    cap = cap.to_numpy(float)
    oversupply = oversupply.to_numpy(float)
    base = work["base_pred"].to_numpy(float)
    floor = base + np.maximum(0, 4000 - base) * cap * strength
    zero_pull = pred * (1 - np.clip(oversupply, 0, 1) * strength * 0.5)
    return np.maximum(np.minimum(zero_pull, 5000), floor)


def predict_premium(frame: pd.DataFrame) -> pd.DataFrame:
    weights = load_blend_weights()
    hp_frame = predict_high_precision(frame)
    fs_frame = predict_fuel_switch(frame)
    hybrid_input = hp_frame.merge(
        fs_frame[
            [
                "fuel_switch_pred",
                "zero_rule_gate",
                "cheap_supply_gate",
                "gas_rule_gate",
                "zero_pressure_state_prob",
                "high_price_state_prob",
            ]
        ],
        left_index=True,
        right_index=True,
        how="left",
    )
    hybrid_params = load_hybrid_params()
    if not hybrid_params:
        hybrid_params = {
            "zero_threshold": 0.55,
            "cheap_threshold": 0.45,
            "fuel_weight": 1.0,
            "spike_block_threshold": 0.005,
            "gas_block_threshold": 0.4,
            "high_price_block_threshold": 0.98,
            "cap_risk_block_threshold": 0.005,
            "zero_prob_scale": 0.1,
            "transition_scale": 10.0,
        }
    hybrid_pred = hybrid_build_prediction(hybrid_input, hybrid_params)
    d2_pred = predict_d2(frame)
    persistence = hybrid_input["base_pred"].to_numpy(float)

    premium = (
        weights.get("hybrid_weight", 0.4) * hybrid_pred.to_numpy(float)
        + weights.get("d2_weight", 0.35) * d2_pred.to_numpy(float)
        + weights.get("persistence_weight", 0.25) * persistence
    )
    premium = apply_must_run_zero_pull(premium, frame, weights.get("must_run_zero_pull", 0.15))
    premium = apply_curve_cap_nudge(premium, frame, weights.get("curve_cap_nudge", 0.10))
    premium = np.clip(premium, 0, 5000)

    out = prepare_frame(frame)
    cols = ["ts_hour", "hour", "hour_group", "base_pred"]
    if "anchor_source" in frame.columns:
        out["anchor_source"] = frame["anchor_source"].values
        cols.append("anchor_source")
    out = out[cols].copy()
    out["persistence_pred"] = persistence
    out["high_precision_pred"] = hybrid_input["high_precision_pred"].to_numpy(float)
    out["fuel_switch_pred"] = hybrid_input["fuel_switch_pred"].to_numpy(float)
    out["hybrid_pred"] = hybrid_pred.to_numpy(float)
    out["d2_pred"] = d2_pred.to_numpy(float)
    out["premium_pred"] = premium
    out["abs_diff_vs_persistence"] = np.abs(out["premium_pred"] - out["persistence_pred"])
    if "must_run_share" in frame.columns:
        out["must_run_share"] = frame["must_run_share"].to_numpy(float)
    if "curve_lag48_cap_risk_score" in frame.columns:
        out["curve_lag48_cap_risk_score"] = frame["curve_lag48_cap_risk_score"].to_numpy(float)
    return out.sort_values("ts_hour")
