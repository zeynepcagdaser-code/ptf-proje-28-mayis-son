"""Train / predict advanced tree stack (hour × horizon, classifiers, rolling)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tree_advanced import config as cfg
from tree_advanced.backend import (
    load_model,
    model_suffix,
    pick_backend,
    predict_model,
    save_model,
    train_classifier,
    train_regressor,
)
from tree_advanced.weights import blend_recency_weights


def resolve_base_features(df: pd.DataFrame) -> list[str]:
    target_cols = {c for c in df.columns if c.startswith("target_") and "residual" not in c}
    exclude = {"ts_hour", "split", "anchor_hour", *target_cols}
    exclude.update(c for c in df.columns if c.startswith("persistence_"))
    exclude.update(c for c in df.columns if c.startswith("target_residual_"))
    return [c for c in df.columns if c not in exclude]


def horizon_features(base: list[str], horizon: int) -> list[str]:
    return base + [f"persistence_{horizon}h"]


def model_paths(horizon: int, hour: int, kind: str, backend: str) -> Path:
    ext = model_suffix(backend)
    hour_tag = "global" if hour < 0 else f"hour_{hour:02d}"
    return cfg.MODEL_DIR / kind / f"h{horizon:02d}" / f"{hour_tag}{ext}"


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mape_masked(actual: np.ndarray, pred: np.ndarray, threshold: float) -> float:
    mask = np.abs(actual) > threshold
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100)


def prepare_df(features_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(features_path).sort_values("ts_hour").reset_index(drop=True)
    if "anchor_hour" not in df.columns:
        ts = pd.to_datetime(df["ts_hour"], utc=True).dt.tz_convert("Europe/Istanbul")
        df["anchor_hour"] = ts.dt.hour.astype(int)
    return df


def split_mask(df: pd.DataFrame, split: str, hour: int | None = None) -> pd.Series:
    m = df["split"] == split
    if hour is not None:
        m &= df["anchor_hour"] == hour
    return m


def train_one_cell(
    df: pd.DataFrame,
    *,
    horizon: int,
    hour: int,
    base_features: list[str],
    backend: str,
    recency_days: int,
    recency_medium_days: int,
    recency_boost: float,
    rolling_refit: bool,
) -> dict[str, Any]:
    tcol = f"target_{horizon}h"
    pcol = f"persistence_{horizon}h"
    fcols = horizon_features(base_features, horizon)

    train = df[split_mask(df, "train", hour)].dropna(subset=fcols + [tcol, pcol])
    val = df[split_mask(df, "validation", hour)].dropna(subset=fcols + [tcol, pcol])

    if len(train) < cfg.MIN_ROWS_PER_HOUR or len(val) < 20:
        train = df[split_mask(df, "train")].dropna(subset=fcols + [tcol, pcol])
        val = df[split_mask(df, "validation")].dropna(subset=fcols + [tcol, pcol])
        hour = -1  # pooled across hours
        if len(train) < cfg.MIN_ROWS_PER_HOUR:
            return {"horizon": horizon, "hour": hour, "skipped": True, "reason": "insufficient_rows"}

    ref_time = pd.to_datetime(train["ts_hour"].max(), utc=True)
    w_train = blend_recency_weights(
        train["ts_hour"],
        reference_time=ref_time,
        recent_days=recency_days,
        medium_days=recency_medium_days,
        max_boost=recency_boost,
    )

    X_train = train[fcols].to_numpy(dtype=np.float64)
    y_res_train = (train[tcol] - train[pcol]).to_numpy(dtype=np.float64)
    X_val = val[fcols].to_numpy(dtype=np.float64)
    y_res_val = (val[tcol] - val[pcol]).to_numpy(dtype=np.float64)

    reg = train_regressor(
        X_train, y_res_train, X_val, y_res_val, sample_weight=w_train, backend=backend
    )
    save_model(reg, model_paths(horizon, hour, "regressor", backend), backend=backend, kind="regressor")

    y_zero_train = (train[tcol] <= cfg.ZERO_THRESHOLD).astype(int).to_numpy()
    y_zero_val = (val[tcol] <= cfg.ZERO_THRESHOLD).astype(int).to_numpy()
    if y_zero_train.sum() >= 5 and y_zero_val.sum() >= 1:
        clf_z = train_classifier(
            X_train, y_zero_train, X_val, y_zero_val, sample_weight=w_train, backend=backend
        )
        save_model(
            clf_z,
            model_paths(horizon, hour, "classifier_zero", backend),
            backend=backend,
            kind="classifier",
        )

    y_spike_train = (train[tcol] >= cfg.SPIKE_THRESHOLD).astype(int).to_numpy()
    y_spike_val = (val[tcol] >= cfg.SPIKE_THRESHOLD).astype(int).to_numpy()
    if y_spike_train.sum() >= 5 and y_spike_val.sum() >= 1:
        clf_s = train_classifier(
            X_train, y_spike_train, X_val, y_spike_val, sample_weight=w_train, backend=backend
        )
        save_model(
            clf_s,
            model_paths(horizon, hour, "classifier_spike", backend),
            backend=backend,
            kind="classifier",
        )

    info: dict[str, Any] = {
        "horizon": horizon,
        "hour": hour,
        "train_rows": len(train),
        "val_rows": len(val),
        "skipped": False,
    }

    if rolling_refit:
        pool = pd.concat([train, val], ignore_index=True)
        ref2 = pd.to_datetime(pool["ts_hour"].max(), utc=True)
        w_pool = blend_recency_weights(
            pool["ts_hour"],
            reference_time=ref2,
            recent_days=recency_days,
            medium_days=recency_medium_days,
            max_boost=recency_boost,
        )
        X_pool = pool[fcols].to_numpy(dtype=np.float64)
        y_pool = (pool[tcol] - pool[pcol]).to_numpy(dtype=np.float64)
        reg2 = train_regressor(
            X_pool,
            y_pool,
            X_val,
            y_res_val,
            sample_weight=w_pool,
            backend=backend,
        )
        save_model(
            reg2,
            model_paths(horizon, hour, "regressor_online", backend),
            backend=backend,
            kind="regressor",
        )
        info["rolling_refit"] = True

    return info


def predict_row(
    row: pd.Series,
    *,
    horizon: int,
    base_features: list[str],
    backend: str,
    use_online: bool,
) -> tuple[float, float, float]:
    fcols = horizon_features(base_features, horizon)
    hour = int(row["anchor_hour"])
    X = row[fcols].to_numpy(dtype=np.float64).reshape(1, -1)
    persistence = float(row[f"persistence_{horizon}h"])

    reg_kind = "regressor_online" if use_online else "regressor"
    reg_path = model_paths(horizon, hour, reg_kind, backend)
    if not reg_path.exists():
        reg_path = model_paths(horizon, hour, "regressor", backend)
    if not reg_path.exists():
        reg_path = model_paths(horizon, -1, reg_kind, backend)
    if not reg_path.exists():
        reg_path = model_paths(horizon, -1, "regressor", backend)
    if not reg_path.exists():
        return persistence, 0.0, 0.0

    residual = float(predict_model(load_model(reg_path, backend=backend), X, backend=backend)[0])
    price = persistence + residual

    p_zero, p_spike = 0.0, 0.0
    z_path = model_paths(horizon, hour, "classifier_zero", backend)
    if z_path.exists():
        p_zero = float(predict_model(load_model(z_path, backend=backend), X, backend=backend)[0])
    s_path = model_paths(horizon, hour, "classifier_spike", backend)
    if s_path.exists():
        p_spike = float(predict_model(load_model(s_path, backend=backend), X, backend=backend)[0])

    if cfg.APPLY_CLASSIFIER_OVERRIDES:
        if p_zero >= cfg.ZERO_CLASS_THRESHOLD and persistence < 800:
            price = 0.0
        elif p_spike >= cfg.SPIKE_CLASS_THRESHOLD:
            price = max(price, persistence + 400)

    return price, p_zero, p_spike


def build_predictions(
    df: pd.DataFrame,
    base_features: list[str],
    backend: str,
    *,
    smoke: bool = False,
    use_online: bool = True,
) -> pd.DataFrame:
    test = df[df["split"] == "test"].copy()
    horizons = cfg.HORIZONS[:2] if smoke else cfg.HORIZONS
    rows: list[dict] = []

    for h in horizons:
        tcol = f"target_{h}h"
        fcols = horizon_features(base_features, h)
        valid = test.dropna(subset=fcols + [tcol, f"persistence_{h}h"])
        for _, row in valid.iterrows():
            actual = float(row[tcol])
            persistence = float(row[f"persistence_{h}h"])
            pred, pz, ps = predict_row(
                row,
                horizon=h,
                base_features=base_features,
                backend=backend,
                use_online=use_online,
            )
            rows.append(
                {
                    "anchor_ts_hour": str(row["ts_hour"]),
                    "anchor_hour": int(row["anchor_hour"]),
                    "target_hour": h,
                    "actual_price": actual,
                    "persistence_price": persistence,
                    "predicted_residual": pred - persistence,
                    "predicted_price": pred,
                    "prob_zero": pz,
                    "prob_spike": ps,
                    "absolute_error": abs(actual - pred),
                    "persistence_error": abs(actual - persistence),
                }
            )
    return pd.DataFrame(rows)


def filter_aligned(pred_df: pd.DataFrame) -> pd.DataFrame:
    if not cfg.ANCHOR_TEST_PATH.exists():
        return pred_df
    anchors = pd.read_csv(cfg.ANCHOR_TEST_PATH)
    anchors["anchor_ts_hour"] = pd.to_datetime(
        anchors["anchor_ts_hour"], utc=True
    ).dt.tz_convert("Europe/Istanbul")
    out = pred_df.copy()
    out["anchor_ts_hour"] = pd.to_datetime(out["anchor_ts_hour"], utc=True).dt.tz_convert(
        "Europe/Istanbul"
    )
    return out.merge(anchors[["anchor_ts_hour"]], on="anchor_ts_hour", how="inner")


def compute_metrics(pred_df: pd.DataFrame) -> dict[str, Any]:
    actual = pred_df["actual_price"].to_numpy()
    pred = pred_df["predicted_price"].to_numpy()
    persistence = pred_df["persistence_price"].to_numpy()
    zero_mask = actual <= cfg.ZERO_THRESHOLD

    h_mae = (
        pred_df.assign(_e=(pred_df["actual_price"] - pred_df["predicted_price"]).abs())
        .groupby("target_hour")["_e"]
        .mean()
    )
    horizon_mae = {str(int(k)): float(v) for k, v in h_mae.items()}
    p_mae = mae(actual, persistence)
    t_mae = mae(actual, pred)

    return {
        "test_samples_anchors": int(pred_df["anchor_ts_hour"].nunique()),
        "prediction_rows": int(len(pred_df)),
        "mae": t_mae,
        "rmse": rmse(actual, pred),
        "masked_mape_actual_gt_100": mape_masked(actual, pred, cfg.MAPE_MASK_THRESHOLD),
        "zero_price_mae": mae(actual[zero_mask], pred[zero_mask]) if zero_mask.any() else None,
        "zero_price_hours": int(zero_mask.sum()),
        "horizon_mae": horizon_mae,
        "worst_horizon": int(max(horizon_mae, key=horizon_mae.get)),
        "persistence_comparison": {
            "persistence_mae": p_mae,
            "tree_mae": t_mae,
            "improvement_pct_vs_persistence": (p_mae - t_mae) / p_mae * 100 if p_mae else 0.0,
            "tree_better_than_persistence": t_mae < p_mae,
        },
    }


def residual_lstm_mae(aligned: pd.DataFrame) -> float | None:
    if not cfg.RESIDUAL_PRED_CSV.exists():
        return None
    r = pd.read_csv(cfg.RESIDUAL_PRED_CSV)
    r["anchor_ts_hour"] = pd.to_datetime(r["anchor_ts_hour"], utc=True).dt.tz_convert(
        "Europe/Istanbul"
    )
    m = aligned.merge(
        r[["anchor_ts_hour", "target_hour", "predicted_price"]],
        on=["anchor_ts_hour", "target_hour"],
        suffixes=("", "_res"),
    )
    if m.empty:
        return None
    return mae(m["actual_price"].to_numpy(), m["predicted_price_res"].to_numpy())


def plot_figures(aligned_metrics: dict, residual_mae_val: float | None) -> None:
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    h_mae = aligned_metrics["horizon_mae"]
    hours = sorted(h_mae.keys(), key=int)
    plt.figure(figsize=(9, 5))
    plt.bar([int(h) for h in hours], [h_mae[h] for h in hours], color="seagreen")
    plt.xlabel("Horizon (h)")
    plt.ylabel("MAE (TL/MWh)")
    plt.title("Tree advanced — horizon MAE (aligned)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(cfg.FIGURES_DIR / "tree_advanced_horizon_mae.png", dpi=150)
    plt.close()

    cmp_ = aligned_metrics["persistence_comparison"]
    labels = ["Persistence", "Tree adv."]
    values = [cmp_["persistence_mae"], cmp_["tree_mae"]]
    colors = ["#6baed6", "#31a354"]
    if residual_mae_val is not None:
        labels.insert(1, "LSTM residual")
        values.insert(1, residual_mae_val)
        colors.insert(1, "#2171b5")
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=colors)
    plt.ylabel("Test MAE (TL/MWh)")
    plt.title("Tree advanced vs baselines")
    plt.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.1f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(cfg.FIGURES_DIR / "tree_advanced_vs_persistence.png", dpi=150)
    plt.close()


def write_md(report: dict) -> str:
    a = report["lstm_anchor_aligned"]
    c = a["persistence_comparison"]
    lines = [
        "# Tree Advanced Metrics",
        "",
        f"- Backend: **{report['backend']}**",
        f"- Residual target: **yes** (final = persistence + residual)",
        f"- Hour-specific models: **24 × 24 horizons**",
        f"- Classifiers: zero-price + spike",
        f"- Rolling online refit: **{report['rolling_refit']}**",
        f"- Recency weight: last **{report['recency_days']}**d (boost **{report['recency_boost']}×**), "
        f"partial **{report['recency_medium_days']}**d",
        "",
        "## Aligned test (LSTM anchors)",
        "",
        f"- MAE: **{a['mae']:.2f}**",
        f"- RMSE: {a['rmse']:.2f}",
        f"- MAPE (actual > {cfg.MAPE_MASK_THRESHOLD}): {a['masked_mape_actual_gt_100']:.2f}%",
        f"- Zero-price MAE: {a.get('zero_price_mae')}",
        f"- Persistence MAE: {c['persistence_mae']:.2f}",
        f"- Improvement vs persistence: **{c['improvement_pct_vs_persistence']:.2f}%**",
        f"- Residual LSTM MAE: {a.get('residual_lstm_mae', 'n/a')}",
        f"- Improvement vs residual LSTM: {a.get('improvement_pct_vs_residual_lstm', 'n/a')}",
        "",
        "## Horizon MAE",
        "",
        "| h | MAE |",
        "|--:|----:|",
    ]
    for h, v in sorted(a["horizon_mae"].items(), key=lambda x: int(x[0])):
        lines.append(f"| {h} | {v:.2f} |")
    return "\n".join(lines)


def run_training(
    *,
    features_path: Path | None = None,
    smoke: bool = False,
    recency_days: int = cfg.DEFAULT_RECENCY_DAYS,
    recency_medium_days: int = 90,
    recency_boost: float = cfg.DEFAULT_RECENCY_BOOST,
    rolling_refit: bool = True,
    skip_train: bool = False,
) -> dict[str, Any]:
    features_path = Path(features_path) if features_path else cfg.FEATURES_PATH
    if not features_path.exists():
        raise FileNotFoundError(f"Run build_tree_features.py first. Missing {features_path}")

    backend = pick_backend()
    df = prepare_df(features_path)
    base_features = resolve_base_features(df)
    cfg.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    horizons = cfg.HORIZONS[:2] if smoke else cfg.HORIZONS
    hours = cfg.HOURS[:3] if smoke else cfg.HOURS
    log: list[dict] = []

    if not skip_train:
        t0 = time.time()
        for h in horizons:
            for hour in hours:
                info = train_one_cell(
                    df,
                    horizon=h,
                    hour=hour,
                    base_features=base_features,
                    backend=backend,
                    recency_days=recency_days,
                    recency_medium_days=recency_medium_days,
                    recency_boost=recency_boost,
                    rolling_refit=rolling_refit,
                )
                if not info.get("skipped"):
                    log.append(info)
        print(f"Trained {len(log)} cells in {time.time() - t0:.1f}s")

    pred_df = build_predictions(
        df, base_features, backend, smoke=smoke, use_online=rolling_refit
    )
    cfg.PREDICTIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(cfg.PREDICTIONS_CSV, index=False)

    full = compute_metrics(pred_df)
    aligned_df = filter_aligned(pred_df)
    aligned = compute_metrics(aligned_df)
    r_mae = residual_lstm_mae(aligned_df)
    if r_mae is not None:
        aligned["residual_lstm_mae"] = r_mae
        aligned["improvement_pct_vs_residual_lstm"] = (r_mae - aligned["mae"]) / r_mae * 100
        aligned["tree_better_than_residual_lstm"] = aligned["mae"] < r_mae

    report = {
        "backend": backend,
        "features_path": str(features_path),
        "feature_count": len(base_features),
        "model_dir": str(cfg.MODEL_DIR),
        "residual_target_training": True,
        "hour_specific_models": True,
        "classifiers": ["zero_price", "spike"],
        "rolling_refit": rolling_refit,
        "recency_days": recency_days,
        "recency_medium_days": recency_medium_days,
        "recency_boost": recency_boost,
        "training_cells": len(log),
        "predictions_path": str(cfg.PREDICTIONS_CSV),
        "full_test_split": full,
        "lstm_anchor_aligned": aligned,
        "reference": {
            "persistence_mae": 545.81,
            "residual_lstm_mae": 535.97,
        },
        "smoke_test": smoke,
    }

    cfg.METRICS_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    cfg.METRICS_MD.write_text(write_md(report), encoding="utf-8")
    plot_figures(aligned, r_mae)

    # Also write tree_baseline paths for user-requested filenames if desired
    _mirror_legacy_outputs(report, aligned)

    return report


def _mirror_legacy_outputs(report: dict, aligned: dict) -> None:
    """Copy metrics to tree_baseline_* paths requested in spec."""
    legacy_json = cfg.PROJECT_ROOT / "reports" / "tree_baseline_metrics.json"
    legacy_md = cfg.PROJECT_ROOT / "reports" / "tree_baseline_metrics.md"
    legacy_pred = cfg.PROJECT_ROOT / "data" / "predictions" / "tree_test_predictions.csv"
    legacy_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    legacy_md.write_text(write_md(report).replace("Tree Advanced", "Tree Baseline (Advanced Pipeline)"), encoding="utf-8")
    if cfg.PREDICTIONS_CSV.exists():
        legacy_pred.write_text(cfg.PREDICTIONS_CSV.read_text(encoding="utf-8"), encoding="utf-8")
    fig_dir = cfg.FIGURES_DIR
    import shutil

    for src, dst in [
        ("tree_advanced_horizon_mae.png", "tree_horizon_mae.png"),
        ("tree_advanced_vs_persistence.png", "tree_vs_persistence.png"),
    ]:
        s = fig_dir / src
        if s.exists():
            shutil.copy(s, fig_dir / dst)
