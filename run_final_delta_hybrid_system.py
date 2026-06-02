#!/usr/bin/env python3
"""
Final baseline system:
  persistence + delta-target regression + balanced_rule hybrid.

Data source:
  - data/model (main_regression sequences)
  - data/model/sequence_metadata.json must be feature_profile=main_regression and feature_count=73

No deep learning. No new data fetching.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


PROJECT_ROOT = Path(__file__).resolve().parent
SEQ_DIR = PROJECT_ROOT / "data" / "model"
MASTER_PATH = PROJECT_ROOT / "data" / "master" / "master_hourly_v1.parquet"
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "lstm_next24_v1.parquet"

REPORTS_DIR = PROJECT_ROOT / "reports"
PRED_DIR = PROJECT_ROOT / "data" / "predictions"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSON = REPORTS_DIR / "final_delta_hybrid_system_metrics.json"
OUT_MD = REPORTS_DIR / "final_delta_hybrid_system_metrics.md"
OUT_PRED = PRED_DIR / "final_delta_hybrid_predictions.csv"

H = 24
HORIZONS = list(range(1, H + 1))
LOW_TH = 50.0
NORMAL_TH = 100.0
SPIKE_TH = 1000.0

TABULAR_COLS_PER_FEATURE = 10  # last, trend, + 4 stats (24h) + 4 stats (168h)


def _log(msg: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if not m.any():
        return float("nan")
    return float(mean_absolute_error(y_true[m], y_pred[m]))


def _safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if not m.any():
        return float("nan")
    return float(math.sqrt(mean_squared_error(y_true[m], y_pred[m])))


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray, *, threshold: float) -> float | None:
    m = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > threshold)
    if not m.any():
        return None
    return float(np.mean(np.abs((y_true[m] - y_pred[m]) / y_true[m])) * 100.0)


def _safe_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if not m.any():
        return float("nan")
    denom = np.abs(y_true[m]) + np.abs(y_pred[m])
    denom = np.where(denom == 0, np.nan, denom)
    return float(np.nanmean(2.0 * np.abs(y_pred[m] - y_true[m]) / denom) * 100.0)


def _safe_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if not m.any():
        return float("nan")
    denom = np.sum(np.abs(y_true[m]))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true[m] - y_pred[m])) / denom * 100.0)


def _regime_masks(actual: np.ndarray) -> dict[str, np.ndarray]:
    a = np.asarray(actual, dtype=float)
    return {
        "zero_only": a == 0.0,
        "low_le_50": a <= LOW_TH,
        "normal_gt_100": a > NORMAL_TH,
        "spike_ge_1000": a >= SPIKE_TH,
    }


def _metrics_blob(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {
        "mae": _safe_mae(y_true, y_pred),
        "rmse": _safe_rmse(y_true, y_pred),
        "mape_gt_50": _safe_mape(y_true, y_pred, threshold=50.0),
        "mape_gt_100": _safe_mape(y_true, y_pred, threshold=100.0),
        "wape": _safe_wape(y_true, y_pred),
        "smape": _safe_smape(y_true, y_pred),
    }
    masks = _regime_masks(y_true)
    for k, m in masks.items():
        out[f"{k}_mae"] = _safe_mae(y_true[m], y_pred[m]) if m.any() else None
    return out


def _load_split(split: Literal["train", "validation", "test"]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    key = {"train": "train", "validation": "val", "test": "test"}[split]
    X = np.load(SEQ_DIR / f"X_{key}.npy", mmap_mode="r")
    y_scaled = np.load(SEQ_DIR / f"y_{key}.npy", mmap_mode="r")
    scaler = joblib.load(SEQ_DIR / "target_scaler.pkl")
    y = scaler.inverse_transform(np.asarray(y_scaled)).astype(np.float64)

    anchor_file = {"train": "anchor_train.csv", "validation": "anchor_val.csv", "test": "anchor_test.csv"}[split]
    anchor = pd.read_csv(SEQ_DIR / anchor_file)
    anchor["anchor_ts_hour"] = pd.to_datetime(anchor["anchor_ts_hour"], utc=True)
    if len(anchor) != y.shape[0]:
        raise ValueError(f"{split}: anchor rows {len(anchor)} != y rows {y.shape[0]}")
    return X, y, anchor


def _tabularize(X: np.ndarray, *, feature_count: int) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    n, w, f = X.shape
    if w < 168:
        raise ValueError(f"window {w} < 168")
    if f != feature_count:
        raise ValueError(f"feature dim {f} != expected {feature_count}")

    last = X[:, -1, :]
    win24 = X[:, -24:, :]
    win168 = X[:, -168:, :]
    mean24 = win24.mean(axis=1)

    out = np.empty((n, feature_count * TABULAR_COLS_PER_FEATURE), dtype=np.float32)
    out[:, 0 * feature_count : 1 * feature_count] = last
    out[:, 1 * feature_count : 2 * feature_count] = last - mean24
    out[:, 2 * feature_count : 3 * feature_count] = win24.mean(axis=1)
    out[:, 3 * feature_count : 4 * feature_count] = win24.std(axis=1)
    out[:, 4 * feature_count : 5 * feature_count] = win24.min(axis=1)
    out[:, 5 * feature_count : 6 * feature_count] = win24.max(axis=1)
    out[:, 6 * feature_count : 7 * feature_count] = win168.mean(axis=1)
    out[:, 7 * feature_count : 8 * feature_count] = win168.std(axis=1)
    out[:, 8 * feature_count : 9 * feature_count] = win168.min(axis=1)
    out[:, 9 * feature_count : 10 * feature_count] = win168.max(axis=1)
    return out


def _load_ptf_series() -> pd.Series:
    master = pd.read_parquet(MASTER_PATH, columns=["ts_hour", "ptf_price"])
    master["ts_hour"] = pd.to_datetime(master["ts_hour"], utc=True)
    master = master.dropna(subset=["ts_hour"]).set_index("ts_hour").sort_index()
    return master["ptf_price"]


def _load_ptf_fallback_lag24() -> pd.Series:
    feat = pd.read_parquet(FEATURES_PATH, columns=["ts_hour", "ptf_lag_24", "ptf_low_ratio_24", "ptf_zero_ratio_24", "ptf_zero_ratio_168"])
    feat["ts_hour"] = pd.to_datetime(feat["ts_hour"], utc=True)
    feat = feat.dropna(subset=["ts_hour"]).set_index("ts_hour").sort_index()
    return feat["ptf_lag_24"], feat[["ptf_low_ratio_24", "ptf_zero_ratio_24", "ptf_zero_ratio_168"]]


def _persistence(anchor_ts: pd.Series, ptf: pd.Series, fallback_lag24: pd.Series) -> np.ndarray:
    P = np.full((len(anchor_ts), H), np.nan, dtype=np.float64)
    fb = fallback_lag24.reindex(anchor_ts).to_numpy(dtype=float)
    for j, h in enumerate(HORIZONS):
        ts = anchor_ts + pd.to_timedelta(h - 24, unit="h")
        p = ptf.reindex(ts).to_numpy(dtype=float)
        m = np.isnan(p)
        if m.any():
            p = p.copy()
            p[m] = fb[m]
        P[:, j] = p
    return P


def _balanced_rule_signal(anchor_ts: pd.Series, rule_df: pd.DataFrame) -> np.ndarray:
    row = rule_df.reindex(anchor_ts)
    sig = (
        (row["ptf_low_ratio_24"].fillna(0) > 0)
        | (row["ptf_zero_ratio_24"].fillna(0) > 0)
        | (row["ptf_zero_ratio_168"].fillna(0) > 0.05)
    )
    return sig.fillna(False).astype(int).to_numpy()


@dataclass
class Backend:
    name: str
    model_factory: Any


def _choose_backend() -> Backend:
    try:
        from lightgbm import LGBMRegressor

        return Backend(
            name="lightgbm",
            model_factory=lambda: LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=63,
                min_child_samples=40,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=-1,
                verbosity=-1,
            ),
        )
    except Exception:
        from sklearn.ensemble import HistGradientBoostingRegressor

        return Backend(
            name="hist_gradient_boosting",
            model_factory=lambda: HistGradientBoostingRegressor(
                max_iter=120,
                learning_rate=0.05,
                min_samples_leaf=40,
                random_state=42,
            ),
        )


def _load_direct_main_predictions_if_any() -> pd.DataFrame | None:
    path = PRED_DIR / "main_tabular_predictions.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "split" in df.columns:
        df = df[df["split"] == "test"].copy()
    if "anchor_ts_hour" in df.columns:
        df["anchor_ts_hour"] = pd.to_datetime(df["anchor_ts_hour"], utc=True)
    return df


def main() -> None:
    # Preconditions
    meta = _read_json(SEQ_DIR / "sequence_metadata.json")
    if meta.get("feature_profile") != "main_regression":
        raise SystemExit(f"ERROR: feature_profile != main_regression ({meta.get('feature_profile')})")
    if int(meta.get("resolved_feature_count", -1)) != 73:
        raise SystemExit(f"ERROR: feature_count != 73 ({meta.get('resolved_feature_count')})")
    shape_train = tuple(meta["shapes"]["train"]["X"])
    if shape_train[-2:] != (168, 73):
        raise SystemExit(f"ERROR: X_train shape mismatch; got {shape_train}")

    backend = _choose_backend()
    _log(f"Backend selected: {backend.name}")

    # Load data
    X_tr, y_tr, a_tr = _load_split("train")
    X_va, y_va, a_va = _load_split("validation")
    X_te, y_te, a_te = _load_split("test")
    _log(f"Loaded splits: train={len(a_tr)} val={len(a_va)} test={len(a_te)}")

    ptf = _load_ptf_series()
    lag24, rule_df = _load_ptf_fallback_lag24()

    P_tr = _persistence(a_tr["anchor_ts_hour"], ptf, lag24)
    P_va = _persistence(a_va["anchor_ts_hour"], ptf, lag24)
    P_te = _persistence(a_te["anchor_ts_hour"], ptf, lag24)

    D_tr = y_tr - P_tr
    D_va = y_va - P_va
    D_te = y_te - P_te

    # Tabularize
    T_tr = _tabularize(X_tr, feature_count=73)
    T_va = _tabularize(X_va, feature_count=73)
    T_te = _tabularize(X_te, feature_count=73)
    _log(f"Tabular shapes: train={T_tr.shape} test={T_te.shape}")

    # Fit delta regressors per horizon
    delta_pred_te = np.empty_like(P_te)
    for j, h in enumerate(HORIZONS):
        _log(f"h{h} fit start")
        model = backend.model_factory()
        model.fit(T_tr, D_tr[:, j])
        delta_pred_te[:, j] = model.predict(T_te).astype(np.float64)
        _log(f"h{h} fit done")

    delta_final_te = P_te + delta_pred_te

    sig_te = _balanced_rule_signal(a_te["anchor_ts_hour"], rule_df)
    final_te = delta_final_te.copy()
    m = sig_te == 1
    if m.any():
        final_te[m, :] = np.minimum(final_te[m, :], P_te[m, :])

    # Metrics (test) – flatten over horizons
    actual_flat = y_te.reshape(-1)
    persistence_flat = P_te.reshape(-1)
    delta_flat = delta_final_te.reshape(-1)
    final_flat = final_te.reshape(-1)
    masks = _regime_masks(actual_flat)

    out = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_training_performed": False,
        "backend": backend.name,
        "dataset": {
            "sequence_dir": str(SEQ_DIR),
            "feature_profile": meta.get("feature_profile"),
            "feature_count": int(meta.get("resolved_feature_count")),
            "x_train_shape": list(X_tr.shape),
        },
        "balanced_rule": "ptf_low_ratio_24>0 OR ptf_zero_ratio_24>0 OR ptf_zero_ratio_168>0.05",
        "hybrid_policy": "if signal==1: final=min(delta_final,persistence) else final=delta_final",
        "test_metrics": {
            "persistence": _metrics_blob(actual_flat, persistence_flat),
            "delta_model": _metrics_blob(actual_flat, delta_flat),
            "final_delta_hybrid": _metrics_blob(actual_flat, final_flat),
        },
        "per_horizon_test_mae": [],
        "success_criteria": {},
        "direct_main_comparison": None,
        "outputs": {
            "report_json": str(OUT_JSON),
            "report_md": str(OUT_MD),
            "predictions_csv": str(OUT_PRED),
        },
    }

    # Per-horizon MAE (test)
    for j, h in enumerate(HORIZONS):
        out["per_horizon_test_mae"].append(
            {
                "horizon": h,
                "persistence_mae": _safe_mae(y_te[:, j], P_te[:, j]),
                "delta_model_mae": _safe_mae(y_te[:, j], delta_final_te[:, j]),
                "final_delta_hybrid_mae": _safe_mae(y_te[:, j], final_te[:, j]),
            }
        )

    # Compare vs prior hybrid (520) and persistence
    prev_hybrid = None
    prev_path = REPORTS_DIR / "hybrid_balanced_rule_metrics.json"
    if prev_path.exists():
        prev = _read_json(prev_path)
        try:
            prev_hybrid = float(prev["splits"]["test"]["hybrid"]["overall_mae"])
        except Exception:
            prev_hybrid = None
    out["success_criteria"] = {
        "final_better_than_persistence_mae": out["test_metrics"]["final_delta_hybrid"]["mae"] < out["test_metrics"]["persistence"]["mae"],
        "final_better_than_prev_hybrid_mae_520": out["test_metrics"]["final_delta_hybrid"]["mae"] < 520.0,
        "final_better_than_prev_hybrid_mae_from_report": None if prev_hybrid is None else out["test_metrics"]["final_delta_hybrid"]["mae"] < prev_hybrid,
        "mape_gt_100_close_to_8pct": (
            None
            if out["test_metrics"]["final_delta_hybrid"]["mape_gt_100"] is None
            else float(out["test_metrics"]["final_delta_hybrid"]["mape_gt_100"]) <= 9.0
        ),
    }

    # Optional direct main comparison from predictions file (test)
    direct = _load_direct_main_predictions_if_any()
    if direct is not None and {"actual_ptf", "main_pred"}.issubset(set(direct.columns)):
        y = direct["actual_ptf"].to_numpy(dtype=float)
        p = direct["main_pred"].to_numpy(dtype=float)
        out["direct_main_comparison"] = _metrics_blob(y, p)

    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # Write predictions (test only, long format)
    rows: list[pd.DataFrame] = []
    for j, h in enumerate(HORIZONS):
        rows.append(
            pd.DataFrame(
                {
                    "anchor_ts_hour": a_te["anchor_ts_hour"],
                    "horizon": h,
                    "actual": y_te[:, j],
                    "persistence_pred": P_te[:, j],
                    "delta_final_pred": delta_final_te[:, j],
                    "final_pred": final_te[:, j],
                    "balanced_rule_signal": sig_te,
                }
            )
        )
    pd.concat(rows, ignore_index=True).to_csv(OUT_PRED, index=False)

    # Markdown report
    tm = out["test_metrics"]
    lines = []
    lines.append("## Final delta-hybrid system metrics (test)")
    lines.append("")
    lines.append("**Model eğitimi yapılmadı (deep learning yok; yalnızca tabular baseline regressor fit).**")
    lines.append("")
    lines.append(f"- **Dataset**: feature_profile=`{out['dataset']['feature_profile']}`, feature_count=**{out['dataset']['feature_count']}**")
    lines.append(f"- **Backend**: `{backend.name}`")
    lines.append("")
    lines.append("### Overall metrics (test, flattened h1-h24)")
    lines.append("")
    lines.append("| model | MAE | RMSE | WAPE% | sMAPE% | MAPE% (actual>50) | MAPE% (actual>100) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name in ["persistence", "delta_model", "final_delta_hybrid"]:
        b = tm[name]
        lines.append(
            f"| {name} | {b['mae']:.2f} | {b['rmse']:.2f} | {b['wape']:.2f} | {b['smape']:.2f} | "
            f"{'' if b['mape_gt_50'] is None else f'{b['mape_gt_50']:.2f}'} | "
            f"{'' if b['mape_gt_100'] is None else f'{b['mape_gt_100']:.2f}'} |"
        )
    lines.append("")
    lines.append("### Regime MAE (test)")
    lines.append("")
    lines.append("| model | zero-only | low<=50 | normal>100 | spike>=1000 |")
    lines.append("|---|---:|---:|---:|---:|")
    for name in ["persistence", "delta_model", "final_delta_hybrid"]:
        b = tm[name]
        lines.append(
            f"| {name} | "
            f"{'' if b['zero_only_mae'] is None else f'{b['zero_only_mae']:.2f}'} | "
            f"{'' if b['low_le_50_mae'] is None else f'{b['low_le_50_mae']:.2f}'} | "
            f"{'' if b['normal_gt_100_mae'] is None else f'{b['normal_gt_100_mae']:.2f}'} | "
            f"{'' if b['spike_ge_1000_mae'] is None else f'{b['spike_ge_1000_mae']:.2f}'} |"
        )
    lines.append("")
    lines.append("### Success criteria")
    lines.append("")
    for k, v in out["success_criteria"].items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append("### Per-horizon MAE (test)")
    lines.append("")
    lines.append("| h | persistence | delta_model | final_delta_hybrid |")
    lines.append("|--:|------------:|-----------:|-------------------:|")
    for row in out["per_horizon_test_mae"]:
        lines.append(
            f"| {row['horizon']} | {row['persistence_mae']:.2f} | {row['delta_model_mae']:.2f} | {row['final_delta_hybrid_mae']:.2f} |"
        )
    lines.append("")
    lines.append(f"Predictions: `{OUT_PRED}`")
    lines.append("")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    _log(f"Wrote: {OUT_MD}")
    _log(f"Wrote: {OUT_JSON}")
    _log(f"Wrote: {OUT_PRED}")
    _log(f"Test MAE persistence={tm['persistence']['mae']:.2f} delta_model={tm['delta_model']['mae']:.2f} final={tm['final_delta_hybrid']['mae']:.2f}")


if __name__ == "__main__":
    main()

