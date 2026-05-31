#!/usr/bin/env python3
"""
Run tomorrow morning PTF inference using the trained regime-aware regressor.

No training, no feature engineering beyond joining already-produced tables.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent

FEATURE_PATH = PROJECT_ROOT / "data" / "features" / "tomorrow_morning_features_enriched.parquet"
REASONING_PATH = PROJECT_ROOT / "data" / "features" / "market_reasoning_features.parquet"
REGIME_PRED_PATH = PROJECT_ROOT / "data" / "predictions" / "regime_classifier_predictions.csv"
SPIKE_PRED_PATH = PROJECT_ROOT / "data" / "predictions" / "spike_cap_detector_predictions.csv"
SPIKE_TRANSITION_PRED_PATH = PROJECT_ROOT / "data" / "predictions" / "spike_transition_detector_predictions.csv"

MODEL_DIR = PROJECT_ROOT / "models" / "regime_aware_regressor"
REPORT_MD = PROJECT_ROOT / "reports" / "tomorrow_forecast_run.md"
REPORT_JSON = PROJECT_ROOT / "reports" / "tomorrow_forecast_run.json"
OUT_PREDICTIONS = PROJECT_ROOT / "data" / "predictions" / "tomorrow_morning_ptf_forecast.csv"


def spike_direction_objective(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility shim for joblib-loaded spike model pickles."""
    error = y_pred - y_true
    sign_mismatch = np.sign(y_pred) != np.sign(y_true)
    spike_up = y_true > 300
    weight = np.ones_like(y_true, dtype=float)
    weight += 4.0 * sign_mismatch.astype(float)
    weight += 3.0 * spike_up.astype(float)
    grad = 2.0 * weight * error
    hess = 2.0 * weight
    return grad, hess


def _load_features() -> pd.DataFrame:
    frame = pd.read_parquet(FEATURE_PATH)
    frame["ts_hour"] = pd.to_datetime(frame["ts_hour"], errors="coerce")
    return frame


def _merge_optional(base: pd.DataFrame, path: Path, keep_cols: list[str]) -> pd.DataFrame:
    if not path.exists():
        return base
    other = pd.read_csv(path)
    if "ts_hour" not in other.columns:
        return base
    other["ts_hour"] = pd.to_datetime(other["ts_hour"], errors="coerce")
    cols = [c for c in keep_cols if c in other.columns]
    if not cols:
        return base
    return base.merge(other[["ts_hour"] + cols], on="ts_hour", how="left")


def load_inputs() -> pd.DataFrame:
    frame = _load_features()

    if REASONING_PATH.exists():
        reasoning = pd.read_parquet(REASONING_PATH)
        reasoning["ts_hour"] = pd.to_datetime(reasoning["ts_hour"], errors="coerce")
        frame = frame.merge(reasoning, on="ts_hour", how="left", suffixes=("", "_reason"))

    frame = _merge_optional(
        frame,
        REGIME_PRED_PATH,
        [
            "pred_regime",
            "prob_negative_zero_pressure",
            "prob_normal",
            "prob_tight",
            "prob_spike_cap",
        ],
    )
    frame = _merge_optional(
        frame,
        SPIKE_PRED_PATH,
        [
            "spike_probability",
            "multiclass_prob_spike_cap",
        ],
    )
    frame = _merge_optional(
        frame,
        SPIKE_TRANSITION_PRED_PATH,
        [
            "new_spike_probability",
        ],
    )

    frame["ts_hour"] = pd.to_datetime(frame["ts_hour"], errors="coerce")
    frame["hour_of_day"] = frame["ts_hour"].dt.hour
    if "must_run_proxy" not in frame.columns:
        if "must_run_supply_proxy" in frame.columns:
            frame["must_run_proxy"] = frame["must_run_supply_proxy"]
        elif "must_run_supply" in frame.columns:
            frame["must_run_proxy"] = frame["must_run_supply"]
        else:
            frame["must_run_proxy"] = np.nan
    return frame


def load_models() -> tuple[object, object, object, list[str]]:
    splitter = joblib.load(MODEL_DIR / "splitter.joblib")
    normal_model = joblib.load(MODEL_DIR / "normal_residual_lgb.joblib")
    spike_model = joblib.load(MODEL_DIR / "spike_residual_lgb_custom.joblib")
    feature_cols = json.loads((MODEL_DIR / "feature_columns.json").read_text())
    return splitter, normal_model, spike_model, feature_cols


def build_analyst_scores(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["analyst_zero_score"] = (
        (1.0 - np.clip(out.get("lag24_ptf", 0).fillna(0) / 1500.0, 0, 1))
        + np.clip(out.get("renewable_oversupply_score", 0).fillna(0), 0, 1)
    ).clip(0, 1)
    out["analyst_spike_score"] = (
        np.clip(out.get("residual_load_ramp", 0).fillna(0) / 3000.0, 0, 1)
        + np.clip(out.get("outage_stress_index", 0).fillna(0) / 5.0, 0, 1)
        + out.get("evening_ramp_flag", 0).fillna(0).astype(float) * 0.25
        + np.clip(out.get("load_minus_kgup", 0).fillna(0) / 5000.0, 0, 1)
    ).clip(0, 1)
    out["analyst_tight_score"] = (
        np.clip(out.get("load_ramp_1h", 0).fillna(0) / 2000.0, 0, 1)
        + np.clip(out.get("outage_stress_index", 0).fillna(0) / 7.0, 0, 1)
        + np.clip(out.get("gas_share", 0).fillna(0), 0, 1) * 0.2
    ).clip(0, 1)
    out["analyst_persistence_break_score"] = (
        np.clip(out.get("solar_cliff_score", 0).fillna(0) / 1000.0, 0, 1)
        + np.clip(out.get("residual_load_ramp", 0).fillna(0) / 4000.0, 0, 1)
        + np.clip(out.get("volatility_cluster_score", 0).fillna(0), 0, 1) * 0.2
    ).clip(0, 1)
    out["analyst_confidence_score"] = (
        1.0 - 0.4 * out["analyst_persistence_break_score"] - 0.3 * out["analyst_spike_score"]
    ).clip(0, 1)
    out["analyst_expected_regime"] = np.select(
        [
            out["analyst_zero_score"] >= out[["analyst_spike_score", "analyst_tight_score"]].max(axis=1),
            out["analyst_spike_score"] >= out[["analyst_zero_score", "analyst_tight_score"]].max(axis=1),
            out["analyst_tight_score"] >= out[["analyst_zero_score", "analyst_spike_score"]].max(axis=1),
        ],
        ["negative_zero_pressure", "spike_cap", "tight"],
        default="normal",
    )
    return out


def predict(frame: pd.DataFrame, splitter: object, normal_model: object, spike_model: object, feature_cols: list[str]) -> pd.DataFrame:
    candidates = frame[frame["is_tomorrow_morning"].fillna(0).astype(int) == 1].copy()
    if candidates.empty:
        candidates = frame[(frame["ts_hour"].dt.date == (datetime.now().date())) | frame["ts_hour"].notna()].copy()
    if "must_run_proxy" not in candidates.columns:
        candidates["must_run_proxy"] = candidates.get("must_run_supply_proxy", candidates.get("must_run_supply", np.nan))
    candidates = build_analyst_scores(candidates)
    x = candidates[feature_cols].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    spike_prob = splitter.predict_proba(x)[:, 1]
    normal_resid = normal_model.predict(x)
    spike_resid = spike_model.predict(x)
    blended = (1 - spike_prob) * normal_resid + spike_prob * spike_resid
    pred = np.clip(candidates["lag24_ptf"].to_numpy(float) + blended, 0, 5000)

    out = candidates[
        [
            "ts_hour",
            "analyst_spike_score",
            "analyst_persistence_break_score",
            "analyst_zero_score",
            "analyst_tight_score",
            "analyst_confidence_score",
            "analyst_expected_regime",
            "lag24_ptf",
            "must_run_proxy",
        ]
    ].copy()
    out = out.rename(
        columns={
            "analyst_expected_regime": "predicted_regime",
            "lag24_ptf": "persistence_pred",
        }
    )
    out["must_run_supply_proxy"] = candidates.get("must_run_supply_proxy", candidates["must_run_proxy"]).to_numpy(float)
    out["predicted_ptf"] = pred
    out["persistence_pred"] = candidates["lag24_ptf"].to_numpy(float)
    out["routing_branch"] = np.where(spike_prob >= 0.5, "spike_route", "normal_route")
    out["spike_probability"] = spike_prob
    out["spike_transition_probability"] = np.clip(
        0.65 * spike_prob + 0.35 * out["analyst_persistence_break_score"].to_numpy(float),
        0,
        1,
    )
    out["abs_diff_vs_persistence"] = (out["predicted_ptf"] - out["persistence_pred"]).abs()
    return out.sort_values("ts_hour")


def write_reports(pred: pd.DataFrame) -> None:
    OUT_PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(OUT_PREDICTIONS, index=False)

    spike_hours = pred.loc[pred["spike_probability"] >= 0.5, "ts_hour"].astype(str).tolist()
    zero_hours = pred.loc[pred.get("predicted_regime", pd.Series(index=pred.index)).eq("negative_zero_pressure"), "ts_hour"].astype(str).tolist()
    cap_hours = pred.loc[pred["spike_probability"] >= 0.5, "ts_hour"].astype(str).tolist()
    separation = pred.sort_values("abs_diff_vs_persistence", ascending=False).head(5)[["ts_hour", "predicted_ptf", "persistence_pred", "abs_diff_vs_persistence"]]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(pred)),
        "routing_branch_counts": pred["routing_branch"].value_counts(dropna=False).to_dict(),
        "spike_hours": spike_hours,
        "zero_pressure_hours": zero_hours,
        "cap_risk_hours": cap_hours,
        "most_divergent_hours": separation.to_dict(orient="records"),
        "columns": list(pred.columns),
    }
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n")
    REPORT_MD.write_text(
        "\n".join(
            [
                "# Tomorrow Forecast Run",
                "",
                f"Generated: `{report['generated_at']}`",
                "",
                f"- Rows forecasted: `{report['rows']}`",
                f"- Routing branches: `{report['routing_branch_counts']}`",
                f"- Spike risk hours: `{', '.join(spike_hours) if spike_hours else 'none'}`",
                f"- Zero-pressure hours: `{', '.join(zero_hours) if zero_hours else 'none'}`",
                f"- Cap-risk hours: `{', '.join(cap_hours) if cap_hours else 'none'}`",
                "",
                "## Most Divergent Hours vs Persistence",
                "",
                *(f"- {row['ts_hour']}: pred={row['predicted_ptf']:.2f}, persistence={row['persistence_pred']:.2f}, diff={row['abs_diff_vs_persistence']:.2f}" for row in report["most_divergent_hours"]),
            ]
        )
        + "\n"
    )


def main() -> None:
    frame = load_inputs()
    splitter, normal_model, spike_model, feature_cols = load_models()
    pred = predict(frame, splitter, normal_model, spike_model, feature_cols)
    write_reports(pred)
    print(f"Wrote {OUT_PREDICTIONS}")
    print(f"Wrote {REPORT_MD}")
    print(f"Wrote {REPORT_JSON}")


if __name__ == "__main__":
    main()
