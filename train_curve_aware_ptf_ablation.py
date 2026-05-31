#!/usr/bin/env python3
"""
Train a small leakage-safe curve-aware PTF ablation model.

This is not the production high-precision model. It answers one concrete
question: when previous-day real DAM supply-demand curve features are added to
the existing market/fuel-switch state, does next-day PTF error improve on the
available historical curve slice?
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "features" / "curve_aware_training_dataset.parquet"
PRED_PATH = PROJECT_ROOT / "data" / "predictions" / "curve_aware_ablation_predictions.csv"
REPORT_MD = PROJECT_ROOT / "reports" / "curve_aware_model_ablation.md"
REPORT_JSON = PROJECT_ROOT / "reports" / "curve_aware_model_ablation.json"
MODEL_DIR = PROJECT_ROOT / "models" / "curve_aware_ptf_ablation"


BASE_FEATURES = [
    "ptf_lag_24",
    "hour",
    "weekday",
    "weekend",
    "month",
    "residual_load_forecast",
    "kgup_total",
    "load_forecast",
    "gas_share",
    "coal_share",
    "hydro_share",
    "wind_share",
    "solar_share",
    "thermal_share",
    "kgup_wind_mw",
    "kgup_solar_mw",
    "kgup_renewable_mw",
    "analyst_zero_score",
    "analyst_spike_score",
    "analyst_tight_score",
    "analyst_persistence_break_score",
    "analyst_confidence_score",
    "gas_share_of_generation",
    "hydro_share_of_generation",
    "renewable_share_of_generation",
    "renewable_minus_gas_shift",
    "gas_marginality_proxy",
    "hydro_displacement_score",
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
    "previous_day_regime",
    "analyst_expected_regime",
]

CURVE_FEATURES = [
    "prev_day_slope_near_clearing",
    "prev_day_elasticity_near_clearing",
    "prev_day_curve_fragility_score",
    "prev_day_volume_needed_for_100TL_move",
    "prev_day_volume_needed_for_500TL_move",
    "prev_day_cap_risk_score",
    "prev_day_oversupply_pressure",
    "reconstruction_confidence",
    "prev_day_spike_pressure_from_curve",
    "prev_day_zero_pressure_from_curve",
]

CATEGORICAL = ["previous_day_regime", "analyst_expected_regime"]


def price_band(price: pd.Series) -> pd.Series:
    return pd.cut(
        price,
        bins=[-np.inf, 50, 1500, 4000, np.inf],
        labels=["negative_zero_pressure", "normal", "tight", "spike_cap"],
    ).astype("string")


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df["delivery_hour"] = pd.to_datetime(df["delivery_hour"], errors="coerce")
    df["target_ptf"] = pd.to_numeric(df["target_ptf"], errors="coerce")
    df["ptf_lag_24"] = pd.to_numeric(df["ptf_lag_24"], errors="coerce")
    df = df.dropna(subset=["delivery_hour", "target_ptf", "ptf_lag_24"]).sort_values("delivery_hour")
    df["target_band"] = price_band(df["target_ptf"])
    df["lag24_band"] = price_band(df["ptf_lag_24"])
    df["is_transition"] = (df["target_band"] != df["lag24_band"]).astype(int)
    df["persistence_error"] = (df["target_ptf"] - df["ptf_lag_24"]).abs()
    return df.reset_index(drop=True)


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["delivery_hour"] < "2026-05-29"].copy()
    val = df[(df["delivery_hour"] >= "2026-05-29") & (df["delivery_hour"] < "2026-05-30")].copy()
    test = df[df["delivery_hour"] >= "2026-05-30"].copy()
    if train.empty or test.empty:
        n = len(df)
        train = df.iloc[: int(n * 0.70)].copy()
        val = df.iloc[int(n * 0.70) : int(n * 0.85)].copy()
        test = df.iloc[int(n * 0.85) :].copy()
    return train, val, test


def design_matrix(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    for frame in (train, val, test):
        for col in features:
            if col not in frame.columns:
                frame[col] = np.nan
    full = pd.concat([train[features], val[features], test[features]], axis=0)
    full = pd.get_dummies(full, columns=[c for c in CATEGORICAL if c in features], dummy_na=True)
    full = full.replace([np.inf, -np.inf], np.nan)
    full = full.apply(lambda s: pd.to_numeric(s, errors="coerce") if s.dtype == "object" else s)
    feature_names = full.columns.tolist()
    n_train, n_val = len(train), len(val)
    x_train = full.iloc[:n_train].reset_index(drop=True)
    x_val = full.iloc[n_train : n_train + n_val].reset_index(drop=True)
    x_test = full.iloc[n_train + n_val :].reset_index(drop=True)
    return x_train, x_val, x_test, feature_names


def sample_weights(frame: pd.DataFrame) -> np.ndarray:
    residual = (frame["target_ptf"] - frame["ptf_lag_24"]).abs()
    weights = np.ones(len(frame), dtype=float)
    weights += (frame["target_band"] == "spike_cap").to_numpy(dtype=float) * 4.0
    weights += (frame["target_band"] == "tight").to_numpy(dtype=float) * 2.0
    weights += (frame["target_band"] == "negative_zero_pressure").to_numpy(dtype=float) * 1.5
    weights += frame["is_transition"].to_numpy(dtype=float) * 1.5
    weights += (residual > residual.quantile(0.85)).to_numpy(dtype=float) * 2.0
    return weights


def train_model(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, features: list[str], name: str) -> tuple[np.ndarray, lgb.LGBMRegressor, pd.DataFrame]:
    x_train, x_val, x_test, feature_names = design_matrix(train, val, test, features)
    y_train = train["target_ptf"] - train["ptf_lag_24"]
    y_val = val["target_ptf"] - val["ptf_lag_24"]

    model = lgb.LGBMRegressor(
        objective="huber",
        alpha=0.85,
        n_estimators=350,
        learning_rate=0.035,
        num_leaves=15,
        min_child_samples=8,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.5,
        reg_lambda=1.5,
        random_state=42,
        verbosity=-1,
    )
    fit_kwargs: dict[str, Any] = {
        "sample_weight": sample_weights(train),
    }
    if not val.empty:
        fit_kwargs["eval_set"] = [(x_val, y_val)]
        fit_kwargs["eval_sample_weight"] = [sample_weights(val)]
        fit_kwargs["callbacks"] = [lgb.early_stopping(40, verbose=False)]
    model.fit(x_train, y_train, **fit_kwargs)
    residual_pred = model.predict(x_test)
    pred = test["ptf_lag_24"].to_numpy() + residual_pred
    pred = np.clip(pred, 0, 5000)

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
            "model": name,
        }
    ).sort_values("importance", ascending=False)
    return pred, model, importance


def metrics(y_true: pd.Series, pred: np.ndarray, frame: pd.DataFrame) -> dict[str, Any]:
    frame = frame.reset_index(drop=True)
    y_true = y_true.reset_index(drop=True)
    pred = np.asarray(pred)
    ae = np.abs(y_true.to_numpy() - pred)
    out: dict[str, Any] = {
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(mean_squared_error(y_true, pred) ** 0.5),
        "median_ae": float(np.median(ae)),
        "p90_ae": float(np.quantile(ae, 0.90)),
        "within_10_tl_pct": float((ae <= 10).mean()),
        "within_50_tl_pct": float((ae <= 50).mean()),
        "rows": int(len(frame)),
    }
    regime_mae: dict[str, float] = {}
    for regime in frame["target_band"].dropna().unique():
        mask = (frame["target_band"] == regime).to_numpy()
        if mask.any():
            regime_mae[str(regime)] = float(mean_absolute_error(y_true[mask], pred[mask]))
    out["regime_mae"] = regime_mae
    transition_mask = frame["is_transition"].to_numpy(dtype=bool)
    out["transition_mae"] = None if not transition_mask.any() else float(mean_absolute_error(y_true[transition_mask], pred[transition_mask]))
    high_fail_mask = frame["persistence_error"] >= frame["persistence_error"].quantile(0.75)
    out["persistence_failure_mae"] = float(mean_absolute_error(y_true[high_fail_mask], pred[high_fail_mask]))
    return out


def main() -> None:
    df = load_data()
    train, val, test = split_data(df)

    y_test = test["target_ptf"]
    persistence_pred = test["ptf_lag_24"].to_numpy()

    base_pred, base_model, base_imp = train_model(train, val, test, BASE_FEATURES, "base_market_fuel_switch")
    curve_pred, curve_model, curve_imp = train_model(train, val, test, BASE_FEATURES + CURVE_FEATURES, "curve_aware")

    pred_df = test[
        [
            "delivery_hour",
            "target_ptf",
            "target_band",
            "ptf_lag_24",
            "previous_day_regime",
            "prev_day_cap_risk_score",
            "prev_day_curve_fragility_score",
            "gas_marginality_proxy",
            "hydro_displacement_score",
            "zero_price_pressure_score",
        ]
    ].copy()
    pred_df["persistence_pred"] = persistence_pred
    pred_df["base_market_pred"] = base_pred
    pred_df["curve_aware_pred"] = curve_pred
    pred_df["persistence_abs_error"] = (pred_df["target_ptf"] - pred_df["persistence_pred"]).abs()
    pred_df["base_market_abs_error"] = (pred_df["target_ptf"] - pred_df["base_market_pred"]).abs()
    pred_df["curve_aware_abs_error"] = (pred_df["target_ptf"] - pred_df["curve_aware_pred"]).abs()

    persistence_metrics = metrics(y_test, persistence_pred, test)
    base_metrics = metrics(y_test, base_pred, test)
    curve_metrics = metrics(y_test, curve_pred, test)

    importance = pd.concat([base_imp.head(30), curve_imp.head(30)], ignore_index=True)

    report = {
        "dataset_rows": int(len(df)),
        "train_rows": int(len(train)),
        "validation_rows": int(len(val)),
        "test_rows": int(len(test)),
        "coverage": {
            "start": str(df["delivery_hour"].min()),
            "end": str(df["delivery_hour"].max()),
            "train_start": str(train["delivery_hour"].min()),
            "train_end": str(train["delivery_hour"].max()),
            "validation_start": str(val["delivery_hour"].min()) if not val.empty else None,
            "validation_end": str(val["delivery_hour"].max()) if not val.empty else None,
            "test_start": str(test["delivery_hour"].min()),
            "test_end": str(test["delivery_hour"].max()),
        },
        "class_distribution": {
            "train": train["target_band"].value_counts().to_dict(),
            "validation": val["target_band"].value_counts().to_dict(),
            "test": test["target_band"].value_counts().to_dict(),
        },
        "metrics": {
            "persistence": persistence_metrics,
            "base_market_fuel_switch": base_metrics,
            "curve_aware": curve_metrics,
        },
        "delta_vs_persistence": {
            "base_market_fuel_switch_mae": float(base_metrics["mae"] - persistence_metrics["mae"]),
            "curve_aware_mae": float(curve_metrics["mae"] - persistence_metrics["mae"]),
        },
        "delta_curve_vs_base": {
            "mae": float(curve_metrics["mae"] - base_metrics["mae"]),
            "rmse": float(curve_metrics["rmse"] - base_metrics["rmse"]),
        },
        "top_curve_aware_features": curve_imp.head(20).to_dict(orient="records"),
        "caveat": "This is a two-week curve-history smoke ablation, not a production-grade backtest. More historical DAM curve days are required before trusting small MAE differences.",
    }

    PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)

    pred_df.to_csv(PRED_PATH, index=False)
    importance.to_csv(MODEL_DIR / "feature_importance.csv", index=False)
    joblib.dump(base_model, MODEL_DIR / "base_market_fuel_switch_lgbm.pkl")
    joblib.dump(curve_model, MODEL_DIR / "curve_aware_lgbm.pkl")
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n")

    lines = [
        "# Curve-Aware PTF Ablation",
        "",
        "This is a limited historical smoke test using reconstructed real DAM supply-demand curve features as previous-day inputs for next-day PTF.",
        "",
        "## Split",
        "",
        f"- Train: `{report['coverage']['train_start']}` -> `{report['coverage']['train_end']}` (`{report['train_rows']}` rows)",
        f"- Validation: `{report['coverage']['validation_start']}` -> `{report['coverage']['validation_end']}` (`{report['validation_rows']}` rows)",
        f"- Test: `{report['coverage']['test_start']}` -> `{report['coverage']['test_end']}` (`{report['test_rows']}` rows)",
        "",
        "## Test MAE",
        "",
        f"- Persistence: `{persistence_metrics['mae']:.2f}` TL/MWh",
        f"- Base market + fuel-switch: `{base_metrics['mae']:.2f}` TL/MWh",
        f"- Curve-aware: `{curve_metrics['mae']:.2f}` TL/MWh",
        f"- Curve vs base delta: `{curve_metrics['mae'] - base_metrics['mae']:.2f}` TL/MWh",
        "",
        "## Regime MAE",
        "",
        "| Model | Regime | MAE |",
        "|---|---:|---:|",
    ]
    for model_name, metric_dict in report["metrics"].items():
        for regime, value in metric_dict["regime_mae"].items():
            lines.append(f"| {model_name} | {regime} | {value:.2f} |")
    lines += [
        "",
        "## Top Curve-Aware Features",
        "",
    ]
    for row in report["top_curve_aware_features"][:15]:
        lines.append(f"- `{row['feature']}`: `{row['importance']}`")
    lines += [
        "",
        "## Caveat",
        "",
        report["caveat"],
        "",
        "A reliable answer needs more historical curve coverage. This run is useful mainly to verify the data plumbing and the direction of signal.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n")

    print(f"Wrote {PRED_PATH}")
    print(f"Wrote {REPORT_MD}")
    print(f"Wrote {REPORT_JSON}")
    print(f"Wrote {MODEL_DIR}")
    print(json.dumps(report["metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
