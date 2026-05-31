#!/usr/bin/env python3
"""
Train a fast regime-aware LightGBM PTF regressor with a spike-direction-aware loss.

Requested core inputs:
    - MustRunProxy
    - Load_Forecast
    - Lag24_PTF
    - Hour_of_Day

Design:
    1) Regime splitter (Normal vs Spike_Risk) via binary LightGBM classifier.
    2) Normal regressor (MAE objective).
    3) Spike-risk regressor (custom objective prioritizing spike direction).
    4) Soft routing to blend normal/spike predictions.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_recall_fscore_support, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent

FEATURE_STORE = PROJECT_ROOT / "data" / "features" / "regime_feature_store.parquet"
MUST_RUN_FEATURES = PROJECT_ROOT / "data" / "features" / "must_run_supply_features.parquet"
LABELS = PROJECT_ROOT / "data" / "regime_labels.csv"

MODEL_DIR = PROJECT_ROOT / "models" / "regime_aware_regressor"
REPORT_MD = PROJECT_ROOT / "reports" / "regime_aware_regressor_metrics.md"
REPORT_JSON = PROJECT_ROOT / "reports" / "regime_aware_regressor_metrics.json"
PREDICTIONS = PROJECT_ROOT / "data" / "predictions" / "regime_aware_regressor_predictions.csv"
TOMORROW_FORECAST = PROJECT_ROOT / "data" / "predictions" / "tomorrow_morning_ptf_forecast.csv"

SPLIT_RANGES = {
    "train": (2020, 2024),
    "validation": (2025, 2025),
    "test": (2026, 2026),
}

SPIKE_RISK_REGIMES = {"tight", "spike_cap"}


def assign_split(ts: pd.Series) -> pd.Series:
    years = ts.dt.year
    split = pd.Series(index=ts.index, dtype="object")
    for name, (start_year, end_year) in SPLIT_RANGES.items():
        split[(years >= start_year) & (years <= end_year)] = name
    return split


def load_base() -> tuple[pd.DataFrame, dict[str, Any]]:
    features = pd.read_parquet(FEATURE_STORE)
    labels = pd.read_csv(LABELS)
    features["ts_hour"] = pd.to_datetime(features["ts_hour"], errors="coerce")
    labels["ts_hour"] = pd.to_datetime(labels["ts_hour"], errors="coerce")

    frame = features.merge(
        labels[["ts_hour", "price", "target_regime"]],
        on="ts_hour",
        how="left",
    )
    frame["split"] = assign_split(frame["ts_hour"])
    frame["hour_of_day"] = frame["ts_hour"].dt.hour

    must_run_status: dict[str, Any] = {
        "must_run_source": "fallback_proxy",
        "must_run_rows": 0,
        "must_run_non_null": 0,
        "must_run_ratio": 0.0,
    }

    if MUST_RUN_FEATURES.exists():
        mr = pd.read_parquet(MUST_RUN_FEATURES)
        if not mr.empty:
            mr["delivery_hour"] = pd.to_datetime(mr["delivery_hour"], errors="coerce")
            frame = frame.merge(
                mr[["delivery_hour", "must_run_supply", "must_run_share", "residual_load_after_must_run"]],
                left_on="ts_hour",
                right_on="delivery_hour",
                how="left",
            ).drop(columns=["delivery_hour"])
            must_run_status["must_run_source"] = "must_run_supply_features.parquet"
            must_run_status["must_run_rows"] = int(len(mr))
            must_run_status["must_run_non_null"] = int(mr["must_run_supply"].notna().sum())
            must_run_status["must_run_ratio"] = float(mr["must_run_supply"].notna().mean())
        else:
            frame["must_run_supply"] = np.nan
    else:
        frame["must_run_supply"] = np.nan

    # Controlled fallback proxy when plant-level must-run is missing.
    frame["must_run_proxy"] = frame["must_run_supply"]
    frame["must_run_proxy"] = frame["must_run_proxy"].fillna(frame.get("kgup_renewable_mw", pd.Series(index=frame.index)))
    frame["must_run_proxy"] = frame["must_run_proxy"].fillna(
        frame.get("kgup_wind_mw", pd.Series(index=frame.index, dtype=float)).fillna(0)
        + frame.get("kgup_solar_mw", pd.Series(index=frame.index, dtype=float)).fillna(0)
    )

    frame["load_forecast"] = pd.to_numeric(frame["load_forecast"], errors="coerce")
    frame["lag24_ptf"] = pd.to_numeric(frame["ptf_lag_24"], errors="coerce")
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame["residual_target"] = frame["price"] - frame["lag24_ptf"]
    frame["is_spike_risk"] = frame["target_regime"].isin(SPIKE_RISK_REGIMES).astype(int)
    frame["must_run_proxy"] = pd.to_numeric(frame["must_run_proxy"], errors="coerce")
    frame["must_run_proxy_share"] = frame["must_run_proxy"] / frame["load_forecast"].replace(0, np.nan)

    return frame, must_run_status


FEATURE_COLS = [
    "must_run_proxy",
    "load_forecast",
    "lag24_ptf",
    "hour_of_day",
]


def prep_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    x = frame[FEATURE_COLS].copy()
    x = x.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    x = x.fillna(0.0)
    y_price = frame["price"].astype(float)
    y_residual = frame["residual_target"].astype(float)
    return x, y_price, y_residual


def train_splitter(train: pd.DataFrame, val: pd.DataFrame) -> lgb.LGBMClassifier:
    x_train, _, _ = prep_xy(train)
    x_val, _, _ = prep_xy(val)
    y_train = train["is_spike_risk"].astype(int)
    y_val = val["is_spike_risk"].astype(int)
    pos = max(int(y_train.sum()), 1)
    neg = max(int((1 - y_train).sum()), 1)
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=450,
        learning_rate=0.045,
        num_leaves=48,
        min_child_samples=50,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=neg / pos,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], eval_metric="auc")
    return model


def spike_direction_objective(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Custom objective on residual target:
    - penalize wrong direction (sign mismatch),
    - add extra weight when true residual is strongly positive (spike-up bias).
    """
    error = y_pred - y_true
    sign_mismatch = np.sign(y_pred) != np.sign(y_true)
    spike_up = y_true > 300
    weight = np.ones_like(y_true, dtype=float)
    weight += 4.0 * sign_mismatch.astype(float)
    weight += 3.0 * spike_up.astype(float)
    grad = 2.0 * weight * error
    hess = 2.0 * weight
    return grad, hess


def train_regressor(
    train: pd.DataFrame,
    val: pd.DataFrame,
    target_col: str,
    objective: str | Any,
    custom_obj: Any = None,
) -> lgb.LGBMRegressor:
    x_train, _, y_train_res = prep_xy(train)
    x_val, _, y_val_res = prep_xy(val)
    y_train = y_train_res if target_col == "residual_target" else train[target_col].astype(float)
    y_val = y_val_res if target_col == "residual_target" else val[target_col].astype(float)

    model = lgb.LGBMRegressor(
        objective=objective if objective is not None else custom_obj,
        n_estimators=700,
        learning_rate=0.04,
        num_leaves=48,
        min_child_samples=40,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(60, verbose=False)],
    )
    return model


def route_predict(
    frame: pd.DataFrame,
    splitter: lgb.LGBMClassifier,
    normal_model: lgb.LGBMRegressor,
    spike_model: lgb.LGBMRegressor,
) -> pd.DataFrame:
    x, _, _ = prep_xy(frame)
    spike_prob = splitter.predict_proba(x)[:, 1]
    normal_resid = normal_model.predict(x)
    spike_resid = spike_model.predict(x)
    blended_resid = (1 - spike_prob) * normal_resid + spike_prob * spike_resid
    pred = frame["lag24_ptf"].to_numpy(float) + blended_resid
    out = frame[["ts_hour", "split", "price", "target_regime", "lag24_ptf", "hour_of_day"]].copy()
    out["spike_risk_prob"] = spike_prob
    out["normal_residual_pred"] = normal_resid
    out["spike_residual_pred"] = spike_resid
    out["blended_residual_pred"] = blended_resid
    out["pred_price"] = np.clip(pred, 0, 5000)
    out["persistence_pred"] = frame["lag24_ptf"]
    out["abs_error"] = (out["price"] - out["pred_price"]).abs()
    out["persistence_abs_error"] = (out["price"] - out["persistence_pred"]).abs()
    return out


def metrics_block(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"rows": 0}
    mae = float(mean_absolute_error(df["price"], df["pred_price"]))
    rmse = float(np.sqrt(mean_squared_error(df["price"], df["pred_price"])))
    p_mae = float(mean_absolute_error(df["price"], df["persistence_pred"]))
    return {
        "rows": int(len(df)),
        "mae": mae,
        "rmse": rmse,
        "persistence_mae": p_mae,
        "delta_vs_persistence": float(mae - p_mae),
        "pct_le_10": float((df["abs_error"] <= 10).mean()),
        "pct_le_50": float((df["abs_error"] <= 50).mean()),
    }


def build_tomorrow_forecast(
    base_frame: pd.DataFrame,
    splitter: lgb.LGBMClassifier,
    normal_model: lgb.LGBMRegressor,
    spike_model: lgb.LGBMRegressor,
) -> pd.DataFrame:
    now_local = datetime.now()
    tomorrow = (now_local + timedelta(days=1)).date()
    candidate = base_frame[
        (base_frame["ts_hour"].dt.date == tomorrow)
        & (base_frame["ts_hour"].dt.hour.between(0, 11))
    ].copy()
    if candidate.empty:
        return pd.DataFrame(
            columns=[
                "ts_hour",
                "must_run_proxy",
                "load_forecast",
                "lag24_ptf",
                "hour_of_day",
                "spike_risk_prob",
                "pred_price",
            ]
        )
    x, _, _ = prep_xy(candidate)
    spike_prob = splitter.predict_proba(x)[:, 1]
    normal_resid = normal_model.predict(x)
    spike_resid = spike_model.predict(x)
    blended_resid = (1 - spike_prob) * normal_resid + spike_prob * spike_resid
    pred = np.clip(candidate["lag24_ptf"].to_numpy(float) + blended_resid, 0, 5000)
    out = candidate[["ts_hour", "must_run_proxy", "load_forecast", "lag24_ptf", "hour_of_day"]].copy()
    out["spike_risk_prob"] = spike_prob
    out["pred_price"] = pred
    return out.sort_values("ts_hour")


def main() -> None:
    frame, must_run_status = load_base()
    train = frame[(frame["split"] == "train") & frame["price"].notna() & frame["lag24_ptf"].notna()].copy()
    val = frame[(frame["split"] == "validation") & frame["price"].notna() & frame["lag24_ptf"].notna()].copy()
    test = frame[(frame["split"] == "test") & frame["price"].notna() & frame["lag24_ptf"].notna()].copy()

    splitter = train_splitter(train, val)
    normal_train = train[train["is_spike_risk"] == 0].copy()
    normal_val = val[val["is_spike_risk"] == 0].copy()
    spike_train = train[train["is_spike_risk"] == 1].copy()
    spike_val = val[val["is_spike_risk"] == 1].copy()

    if len(normal_train) < 300 or len(spike_train) < 200:
        raise RuntimeError("Not enough rows for regime-aware regression training.")

    normal_model = train_regressor(
        normal_train,
        normal_val if not normal_val.empty else normal_train.sample(min(500, len(normal_train)), random_state=42),
        target_col="residual_target",
        objective="l1",
    )
    spike_model = train_regressor(
        spike_train,
        spike_val if not spike_val.empty else spike_train.sample(min(300, len(spike_train)), random_state=42),
        target_col="residual_target",
        objective=None,
        custom_obj=spike_direction_objective,
    )

    pred_all = route_predict(frame[frame["lag24_ptf"].notna() & frame["price"].notna()].copy(), splitter, normal_model, spike_model)
    pred_all.to_csv(PREDICTIONS, index=False)

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "must_run_status": must_run_status,
        "features": FEATURE_COLS,
        "split_metrics": {
            split: metrics_block(pred_all[pred_all["split"] == split])
            for split in ["train", "validation", "test"]
        },
    }

    val_pred = pred_all[pred_all["split"] == "validation"].copy()
    if not val_pred.empty:
        y_true = val_pred["target_regime"].isin(SPIKE_RISK_REGIMES).astype(int)
        y_prob = val_pred["spike_risk_prob"]
        y_hat = (y_prob >= 0.5).astype(int)
        pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_hat, average="binary", zero_division=0)
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = None
        metrics["validation_splitter"] = {
            "precision": float(pr),
            "recall": float(rc),
            "f1": float(f1),
            "roc_auc": auc,
        }

    tomorrow = build_tomorrow_forecast(frame[frame["lag24_ptf"].notna()].copy(), splitter, normal_model, spike_model)
    tomorrow.to_csv(TOMORROW_FORECAST, index=False)
    metrics["tomorrow_morning_rows"] = int(len(tomorrow))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(splitter, MODEL_DIR / "splitter.joblib")
    joblib.dump(normal_model, MODEL_DIR / "normal_residual_lgb.joblib")
    joblib.dump(spike_model, MODEL_DIR / "spike_residual_lgb_custom.joblib")
    (MODEL_DIR / "feature_columns.json").write_text(json.dumps(FEATURE_COLS, ensure_ascii=False, indent=2) + "\n")
    REPORT_JSON.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")

    test_m = metrics["split_metrics"]["test"]
    REPORT_MD.write_text(
        "\n".join(
            [
                "# Regime-Aware Regressor (LightGBM)",
                "",
                f"Generated: `{metrics['generated_at']}`",
                "",
                "## Setup",
                "",
                f"- Features: `{', '.join(FEATURE_COLS)}`",
                "- Regime split: `Normal` vs `Spike_Risk` (`tight` + `spike_cap`).",
                "- Spike expert objective: custom directional residual loss (sign mismatch penalty).",
                "",
                "## Must-Run Source",
                "",
                f"- Source mode: `{must_run_status['must_run_source']}`",
                f"- Non-null must-run ratio: `{must_run_status['must_run_ratio']}`",
                "",
                "## Test Metrics",
                "",
                f"- MAE: `{test_m.get('mae')}`",
                f"- RMSE: `{test_m.get('rmse')}`",
                f"- Persistence MAE: `{test_m.get('persistence_mae')}`",
                f"- Delta vs persistence: `{test_m.get('delta_vs_persistence')}`",
                f"- Error <= 10 TL: `{test_m.get('pct_le_10')}`",
                f"- Error <= 50 TL: `{test_m.get('pct_le_50')}`",
                "",
                "## Tomorrow Morning Forecast",
                "",
                f"- Rows produced: `{len(tomorrow)}`",
                f"- File: `{TOMORROW_FORECAST.relative_to(PROJECT_ROOT)}`",
            ]
        )
        + "\n"
    )

    print(f"Wrote {PREDICTIONS}")
    print(f"Wrote {TOMORROW_FORECAST}")
    print(f"Wrote {REPORT_MD}")
    print(f"Wrote {REPORT_JSON}")
    print(f"Wrote {MODEL_DIR}")


if __name__ == "__main__":
    main()
