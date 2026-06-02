#!/usr/bin/env python3
"""
Final deliverable: simple tabular PTF forecasting model (no hybrid, no classifier).

Uses the final 73-feature `data/model` main_regression dataset, converts sequences to
tabular, trains 24 horizon-specific regressors, compares to persistence baseline, and
writes thesis-ready reports.
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

OUT_JSON = REPORTS_DIR / "final_lightgbm_ptf_metrics.json"
OUT_MD = REPORTS_DIR / "final_lightgbm_ptf_metrics.md"
OUT_PRED = PRED_DIR / "final_lightgbm_ptf_predictions.csv"
OUT_THESIS = REPORTS_DIR / "thesis_ready_model_summary.md"

H = 24
HORIZONS = list(range(1, H + 1))
TABULAR_COLS_PER_FEATURE = 10  # last, trend, + 4 stats (24h) + 4 stats (168h)

LOW_TH = 50.0
NORMAL_TH = 100.0
SPIKE_TH = 1000.0


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


def _load_ptf_lag24_feature() -> pd.Series:
    feat = pd.read_parquet(FEATURES_PATH, columns=["ts_hour", "ptf_lag_24"])
    feat["ts_hour"] = pd.to_datetime(feat["ts_hour"], utc=True)
    feat = feat.dropna(subset=["ts_hour"]).set_index("ts_hour").sort_index()
    return feat["ptf_lag_24"]


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


@dataclass(frozen=True)
class Backend:
    name: str
    param_grid: list[dict[str, Any]]
    model_factory: Any


def _choose_backend() -> Backend:
    try:
        from lightgbm import LGBMRegressor

        grid: list[dict[str, Any]] = []
        for lr in [0.03, 0.05, 0.1]:
            for num_leaves in [31, 63]:
                for n_estimators in [300, 500]:
                    grid.append({"learning_rate": lr, "num_leaves": num_leaves, "n_estimators": n_estimators})
        return Backend(
            name="lightgbm",
            param_grid=grid,
            model_factory=lambda params: LGBMRegressor(
                **params,
                min_child_samples=40,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=-1,
                verbosity=-1,
            ),
        )
    except Exception:
        pass

    try:
        from xgboost import XGBRegressor

        grid = []
        for lr in [0.03, 0.05, 0.1]:
            for max_depth in [4, 6]:
                for n_estimators in [300, 500]:
                    grid.append({"learning_rate": lr, "max_depth": max_depth, "n_estimators": n_estimators})
        return Backend(
            name="xgboost",
            param_grid=grid,
            model_factory=lambda params: XGBRegressor(
                **params,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                objective="reg:squarederror",
            ),
        )
    except Exception:
        from sklearn.ensemble import HistGradientBoostingRegressor

        grid = []
        for lr in [0.03, 0.05, 0.1]:
            for max_leaf_nodes in [31, 63]:
                for max_iter in [120, 250]:
                    grid.append({"learning_rate": lr, "max_leaf_nodes": max_leaf_nodes, "max_iter": max_iter})
        return Backend(
            name="hist_gradient_boosting",
            param_grid=grid,
            model_factory=lambda params: HistGradientBoostingRegressor(
                **params,
                min_samples_leaf=40,
                random_state=42,
            ),
        )


def _tune_on_validation(
    backend: Backend,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    *,
    tune_horizons: list[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tune_idx = [h - 1 for h in tune_horizons]
    _log(f"Tuning on validation horizons={tune_horizons} (primary: MAPE>100, secondary: MAE)")

    results: list[dict[str, Any]] = []
    best_params: dict[str, Any] | None = None
    best_key: tuple[float, float] | None = None

    for i, params in enumerate(backend.param_grid, start=1):
        preds = []
        trues = []
        for j in tune_idx:
            m = backend.model_factory(params)
            m.fit(X_tr, y_tr[:, j])
            preds.append(m.predict(X_va).astype(np.float64))
            trues.append(y_va[:, j].astype(np.float64))

        y_true = np.concatenate(trues, axis=0)
        y_pred = np.concatenate(preds, axis=0)
        mape100 = _safe_mape(y_true, y_pred, threshold=100.0)
        mae = _safe_mae(y_true, y_pred)
        key = (float("inf") if mape100 is None else float(mape100), float(mae))

        results.append({"params": params, "val_mape_gt_100": mape100, "val_mae": mae})
        if best_key is None or key < best_key:
            best_key = key
            best_params = params

    assert best_params is not None
    _log(f"Selected params: {best_params} with key={best_key}")
    return best_params, results


def main() -> None:
    meta = _read_json(SEQ_DIR / "sequence_metadata.json")
    if meta.get("feature_profile") != "main_regression":
        raise SystemExit(f"ERROR: feature_profile != main_regression ({meta.get('feature_profile')})")
    if int(meta.get("resolved_feature_count", -1)) != 73:
        raise SystemExit(f"ERROR: feature_count != 73 ({meta.get('resolved_feature_count')})")

    backend = _choose_backend()
    _log(f"Backend selected: {backend.name}")

    X_tr, y_tr, a_tr = _load_split("train")
    X_va, y_va, a_va = _load_split("validation")
    X_te, y_te, a_te = _load_split("test")
    _log(f"Loaded splits: train={len(a_tr)} val={len(a_va)} test={len(a_te)}")

    T_tr = _tabularize(X_tr, feature_count=73)
    T_va = _tabularize(X_va, feature_count=73)
    T_te = _tabularize(X_te, feature_count=73)
    _log(f"Tabular shapes: train={T_tr.shape} val={T_va.shape} test={T_te.shape}")

    # Persistence baseline (test)
    ptf = _load_ptf_series()
    lag24 = _load_ptf_lag24_feature()
    P_te = _persistence(a_te["anchor_ts_hour"], ptf, lag24)

    # Tune on validation (bounded): representative horizons
    tune_h = [1, 12, 24]
    best_params, tune_results = _tune_on_validation(backend, T_tr, y_tr, T_va, y_va, tune_horizons=tune_h)

    # Final fit: train+val, predict test (24 models)
    T_trva = np.concatenate([T_tr, T_va], axis=0)
    y_trva = np.concatenate([y_tr, y_va], axis=0)

    pred_te = np.empty_like(y_te, dtype=np.float64)
    for h in HORIZONS:
        j = h - 1
        _log(f"h{h} fit start")
        m = backend.model_factory(best_params)
        m.fit(T_trva, y_trva[:, j])
        pred_te[:, j] = m.predict(T_te).astype(np.float64)
        _log(f"h{h} fit done")

    # Metrics (test flattened)
    y_true_flat = y_te.reshape(-1)
    y_pred_flat = pred_te.reshape(-1)
    p_flat = P_te.reshape(-1)

    persistence_metrics = _metrics_blob(y_true_flat, p_flat)
    model_metrics = _metrics_blob(y_true_flat, y_pred_flat)

    out: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": backend.name,
        "dataset": {
            "sequence_dir": str(SEQ_DIR),
            "feature_profile": meta.get("feature_profile"),
            "feature_count": int(meta.get("resolved_feature_count")),
            "tabular_dim": int(T_tr.shape[1]),
            "x_train_shape": list(X_tr.shape),
        },
        "tuning": {
            "split": "validation",
            "tune_horizons": tune_h,
            "primary_metric": "MAPE(actual>100)",
            "secondary_metric": "MAE",
            "grid_size": len(backend.param_grid),
            "selected_params": best_params,
            "results": tune_results,
        },
        "test_metrics": {"persistence": persistence_metrics, "model": model_metrics},
        "per_horizon_test": [],
        "decision": {
            "model_beats_persistence_mae": float(model_metrics["mae"]) < float(persistence_metrics["mae"]),
            "model_mape_gt_100_close_to_8pct": (
                None if model_metrics["mape_gt_100"] is None else float(model_metrics["mape_gt_100"]) <= 9.0
            ),
        },
        "outputs": {
            "report_json": str(OUT_JSON),
            "report_md": str(OUT_MD),
            "predictions_csv": str(OUT_PRED),
            "thesis_summary_md": str(OUT_THESIS),
        },
    }

    for h in HORIZONS:
        j = h - 1
        out["per_horizon_test"].append(
            {"horizon": h, "persistence": _metrics_blob(y_te[:, j], P_te[:, j]), "model": _metrics_blob(y_te[:, j], pred_te[:, j])}
        )

    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # Predictions CSV (test only, long format)
    rows = []
    for h in HORIZONS:
        j = h - 1
        rows.append(
            pd.DataFrame(
                {
                    "split": "test",
                    "anchor_ts_hour": a_te["anchor_ts_hour"],
                    "horizon": h,
                    "actual": y_te[:, j],
                    "prediction": pred_te[:, j],
                    "persistence_pred": P_te[:, j],
                }
            )
        )
    pd.concat(rows, ignore_index=True).to_csv(OUT_PRED, index=False)

    # Markdown metrics report
    lines = []
    lines.append("## Final tabular PTF model metrics (test)")
    lines.append("")
    lines.append("**Model eğitimi yapıldı (tabular baseline; deep learning yok; hybrid/classifier yok).**")
    lines.append("")
    lines.append(f"- **Dataset**: feature_profile=`{out['dataset']['feature_profile']}`, feature_count=**{out['dataset']['feature_count']}**")
    lines.append(f"- **Backend**: `{backend.name}`")
    lines.append(f"- **Selected params (val)**: `{best_params}`")
    lines.append("")
    lines.append("### Overall metrics (test, flattened h1-h24)")
    lines.append("")
    lines.append("| model | MAE | RMSE | WAPE% | sMAPE% | MAPE% (actual>50) | MAPE% (actual>100) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, b in [("persistence", persistence_metrics), ("model", model_metrics)]:
        m50 = "" if b["mape_gt_50"] is None else f"{b['mape_gt_50']:.2f}"
        m100 = "" if b["mape_gt_100"] is None else f"{b['mape_gt_100']:.2f}"
        lines.append(f"| {name} | {b['mae']:.2f} | {b['rmse']:.2f} | {b['wape']:.2f} | {b['smape']:.2f} | {m50} | {m100} |")
    lines.append("")
    lines.append("### Regime MAE (test)")
    lines.append("")
    lines.append("| model | zero-only | low<=50 | normal>100 | spike>=1000 |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, b in [("persistence", persistence_metrics), ("model", model_metrics)]:
        z = "" if b["zero_only_mae"] is None else f"{b['zero_only_mae']:.2f}"
        lo = "" if b["low_le_50_mae"] is None else f"{b['low_le_50_mae']:.2f}"
        no = "" if b["normal_gt_100_mae"] is None else f"{b['normal_gt_100_mae']:.2f}"
        sp = "" if b["spike_ge_1000_mae"] is None else f"{b['spike_ge_1000_mae']:.2f}"
        lines.append(f"| {name} | {z} | {lo} | {no} | {sp} |")
    lines.append("")
    lines.append("### Per-horizon (test)")
    lines.append("")
    lines.append("| h | persistence MAE | model MAE | persistence MAPE>100 | model MAPE>100 |")
    lines.append("|--:|---------------:|---------:|---------------------:|--------------:|")
    for row in out["per_horizon_test"]:
        p = row["persistence"]
        m = row["model"]
        p100 = "" if p["mape_gt_100"] is None else f"{p['mape_gt_100']:.2f}"
        m100 = "" if m["mape_gt_100"] is None else f"{m['mape_gt_100']:.2f}"
        lines.append(f"| {row['horizon']} | {p['mae']:.2f} | {m['mae']:.2f} | {p100} | {m100} |")
    lines.append("")
    lines.append(f"Predictions: `{OUT_PRED}`")
    lines.append("")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Thesis-ready summary
    thesis = []
    thesis.append("## Thesis-ready model summary")
    thesis.append("")
    thesis.append("### Used data")
    thesis.append("- `data/model`: main_regression sequence dataset (168h input window, 24h horizons)")
    thesis.append("- Feature set: **73** features (FİBA/FİBS + GRF + DAM microstructure entegre)")
    thesis.append("- Splits: train/validation/test with `anchor_*.csv` timestamps")
    thesis.append("")
    thesis.append("### Model")
    thesis.append(f"- Backend: **{backend.name}**")
    thesis.append("- Multi-horizon: 24 ayrı model (h1..h24)")
    thesis.append("- Tabularization: last timestep + 24h mean/std/min/max + 168h mean/std/min/max + trend(last-mean24)")
    thesis.append("")
    thesis.append("### Why LightGBM/XGBoost")
    thesis.append("- Ağaç tabanlı boosting modelleri tabular feature set’lerde güçlü ve hızlı baseline sağlar.")
    thesis.append("- Deep learning’e göre daha hızlı iterasyon, daha az operasyonel karmaşıklık.")
    thesis.append("")
    thesis.append("### Validation tuning")
    thesis.append("- Small grid search (learning_rate, num_leaves/max_depth, n_estimators) on validation.")
    thesis.append("- Primary metric: **MAPE(actual>100)**, secondary metric: **MAE**.")
    thesis.append(f"- Selected params: `{best_params}`")
    thesis.append("")
    thesis.append("### Persistence baseline comparison (test, flattened h1-h24)")
    thesis.append("- Persistence: dünkü aynı saat PTF (`anchor_ts + h - 24`), eksikse `ptf_lag_24` fallback.")
    thesis.append(f"- Persistence MAE: **{persistence_metrics['mae']:.2f}**")
    thesis.append(f"- Model MAE: **{model_metrics['mae']:.2f}**")
    if persistence_metrics["mape_gt_100"] is not None:
        thesis.append(f"- Persistence MAPE(actual>100): **{persistence_metrics['mape_gt_100']:.2f}%**")
    else:
        thesis.append("- Persistence MAPE(actual>100): N/A")
    if model_metrics["mape_gt_100"] is not None:
        thesis.append(f"- Model MAPE(actual>100): **{model_metrics['mape_gt_100']:.2f}%**")
    else:
        thesis.append("- Model MAPE(actual>100): N/A")
    thesis.append(f"- Model RMSE: **{model_metrics['rmse']:.2f}**")
    thesis.append(f"- Model WAPE: **{model_metrics['wape']:.2f}%**")
    thesis.append("")
    thesis.append("### 24-hour forecasting approach")
    thesis.append("- Her horizon için ayrı model eğitilir; bu sayede horizon’a özgü hata profili yakalanır.")
    thesis.append("")
    OUT_THESIS.write_text("\n".join(thesis) + "\n", encoding="utf-8")

    _log(f"Wrote: {OUT_MD}")
    _log(f"Wrote: {OUT_JSON}")
    _log(f"Wrote: {OUT_PRED}")
    _log(f"Wrote: {OUT_THESIS}")


if __name__ == "__main__":
    main()

