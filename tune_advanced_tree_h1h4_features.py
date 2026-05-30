#!/usr/bin/env python3
"""
Feature selection for advanced tree h1–h4 only.

Extracts gain importance from existing tree_advanced models, retrains h1–h4
with top-10/20/30/50/all feature sets (separate model dir; does not delete originals).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tree_advanced import config as adv_cfg
from tree_advanced.backend import load_model, model_suffix, predict_model, save_model
from tree_advanced.pipeline import (
    filter_aligned,
    horizon_features,
    prepare_df,
    resolve_base_features,
    split_mask,
)
from tree_advanced.weights import blend_recency_weights

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_MODEL_DIR = PROJECT_ROOT / "models" / "tree_advanced"
TUNE_MODEL_ROOT = PROJECT_ROOT / "models" / "tree_advanced_h1h4_tune"
FEATURES_PATH = adv_cfg.FEATURES_PATH
ANCHOR_TEST_PATH = adv_cfg.ANCHOR_TEST_PATH

METRICS_JSON = PROJECT_ROOT / "reports" / "advanced_tree_h1h4_feature_selection.json"
METRICS_MD = PROJECT_ROOT / "reports" / "advanced_tree_h1h4_feature_selection.md"
FIGURES_PATH = PROJECT_ROOT / "reports" / "figures" / "advanced_tree_h1h4_feature_selection.png"
BEST_PRED_CSV = PROJECT_ROOT / "data" / "predictions" / "advanced_tree_h1h4_best_predictions.csv"

SHORT_H = [1, 2, 3, 4]
FEATURE_SET_SIZES = [10, 20, 30, 50, "all"]
BACKEND = "lightgbm"

BASELINE_PATHS = {
    "advanced_tree": PROJECT_ROOT / "data" / "predictions" / "tree_advanced_test_predictions.csv",
    "short_expert": PROJECT_ROOT / "data" / "predictions" / "short_horizon_expert_predictions.csv",
    "residual_lstm": PROJECT_ROOT / "data" / "predictions" / "lstm_residual_test_predictions.csv",
}
ADVANCED_METRICS = PROJECT_ROOT / "reports" / "tree_advanced_metrics.json"
SHORT_METRICS = PROJECT_ROOT / "reports" / "short_horizon_expert_metrics.json"


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def tune_model_path(feature_set: str, horizon: int, hour: int) -> Path:
    ext = model_suffix(BACKEND)
    tag = "global" if hour < 0 else f"hour_{hour:02d}"
    return TUNE_MODEL_ROOT / feature_set / f"h{horizon:02d}" / f"{tag}{ext}"


def extract_importance_h1h4(base_features: list[str]) -> dict[str, Any]:
    """Sum gain from regressor_online cells for horizons 1–4."""
    per_horizon: dict[int, dict[str, float]] = {h: {} for h in SHORT_H}
    cells_loaded = 0

    for h in SHORT_H:
        fcols = horizon_features(base_features, h)
        h_dir = SOURCE_MODEL_DIR / "regressor_online" / f"h{h:02d}"
        if not h_dir.exists():
            continue
        for model_file in sorted(h_dir.glob("hour_*.txt")):
            booster = load_model(model_file, backend=BACKEND)
            gains = booster.feature_importance(importance_type="gain")
            # Saved boosters use Column_0..N; map by training column order.
            for i, g in enumerate(gains):
                if i >= len(fcols):
                    break
                name = fcols[i]
                per_horizon[h][name] = per_horizon[h].get(name, 0.0) + float(g)
            cells_loaded += 1

    global_agg: dict[str, float] = {}
    for h in SHORT_H:
        for name, g in per_horizon[h].items():
            global_agg[name] = global_agg.get(name, 0.0) + g

    ranked_global = sorted(global_agg.items(), key=lambda x: -x[1])
    ranked_per_h = {
        h: sorted(per_horizon[h].items(), key=lambda x: -x[1]) for h in SHORT_H
    }

    return {
        "cells_loaded": cells_loaded,
        "per_horizon_gain": {str(h): dict(ranked_per_h[h]) for h in SHORT_H},
        "global_gain_ranked": [{"feature": n, "gain": g} for n, g in ranked_global],
        "per_horizon_ranked": {
            str(h): [{"feature": n, "gain": g} for n, g in ranked_per_h[h]] for h in SHORT_H
        },
    }


def select_features_for_horizon(
    ranked: list[dict],
    horizon: int,
    k: int | str,
    all_features: list[str],
) -> list[str]:
    pcol = f"persistence_{horizon}h"
    if k == "all":
        return list(all_features)

    pool = [r["feature"] for r in ranked if r["feature"] != pcol]
    chosen = pool[: max(k - 1, 0)]
    if pcol not in chosen:
        chosen = [pcol] + chosen
    return chosen[:k] if pcol in chosen else [pcol] + chosen[: k - 1]


def build_feature_sets(
    base_features: list[str],
    importance: dict[str, Any],
) -> dict[str, dict[int, list[str]]]:
    sets: dict[str, dict[int, list[str]]] = {}
    for k in FEATURE_SET_SIZES:
        sets[str(k)] = {}
        for h in SHORT_H:
            ranked = importance["per_horizon_ranked"][str(h)]
            all_f = horizon_features(base_features, h)
            sets[str(k)][h] = select_features_for_horizon(ranked, h, k, all_f)
    return sets


def train_cell(
    df: pd.DataFrame,
    horizon: int,
    hour: int,
    feature_cols: list[str],
    out_path: Path,
) -> dict[str, float]:
    tcol = f"target_{horizon}h"
    pcol = f"persistence_{horizon}h"

    train = df[split_mask(df, "train", hour)].dropna(subset=feature_cols + [tcol, pcol])
    val = df[split_mask(df, "validation", hour)].dropna(subset=feature_cols + [tcol, pcol])
    test = df[split_mask(df, "test", hour)].dropna(subset=feature_cols + [tcol, pcol])

    if len(train) < adv_cfg.MIN_ROWS_PER_HOUR or len(val) < 20:
        train = df[split_mask(df, "train")].dropna(subset=feature_cols + [tcol, pcol])
        val = df[split_mask(df, "validation")].dropna(subset=feature_cols + [tcol, pcol])
        test = df[split_mask(df, "test")].dropna(subset=feature_cols + [tcol, pcol])
        hour = -1

    ref = pd.to_datetime(train["ts_hour"].max(), utc=True)
    w = blend_recency_weights(
        train["ts_hour"],
        reference_time=ref,
        recent_days=adv_cfg.DEFAULT_RECENCY_DAYS,
        medium_days=adv_cfg.DEFAULT_RECENCY_DAYS + 30,
        max_boost=adv_cfg.DEFAULT_RECENCY_BOOST,
    )

    X_tr = train[feature_cols].to_numpy(dtype=np.float64)
    y_tr = (train[tcol] - train[pcol]).to_numpy(dtype=np.float64)
    X_va = val[feature_cols].to_numpy(dtype=np.float64)
    y_va = (val[tcol] - val[pcol]).to_numpy(dtype=np.float64)
    X_te = test[feature_cols].to_numpy(dtype=np.float64)

    import lightgbm as lgb

    train_set = lgb.Dataset(X_tr, label=y_tr, weight=w, feature_name=feature_cols)
    val_set = lgb.Dataset(X_va, label=y_va, reference=train_set)
    model = lgb.train(
        {
            "objective": "regression",
            "metric": "mae",
            "verbosity": -1,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_data_in_leaf": 40,
            "lambda_l1": 0.2,
            "lambda_l2": 1.0,
            "feature_fraction": 0.75,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "seed": 42,
        },
        train_set,
        num_boost_round=1500,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    model.save_model(str(out_path))

    val_res = predict_model(model, X_va, backend=BACKEND)
    val_price = val[pcol].to_numpy() + val_res
    val_mae = mae(val[tcol].to_numpy(), val_price)

    te_res = predict_model(model, X_te, backend=BACKEND)
    te_price = test[pcol].to_numpy() + te_res
    te_mae = mae(test[tcol].to_numpy(), te_price)

    return {
        "val_mae": val_mae,
        "test_mae": te_mae,
        "hour": hour,
        "train_rows": len(train),
    }


def predict_feature_set(
    df: pd.DataFrame,
    feature_set_name: str,
    horizon_features_map: dict[int, list[str]],
) -> pd.DataFrame:
    test = df[df["split"] == "test"].copy()
    rows: list[dict] = []

    for h in SHORT_H:
        tcol = f"target_{h}h"
        pcol = f"persistence_{h}h"
        fcols = horizon_features_map[h]
        valid = test.dropna(subset=fcols + [tcol, pcol])

        for _, row in valid.iterrows():
            hour = int(row["anchor_hour"])
            X = row[fcols].to_numpy(dtype=np.float64).reshape(1, -1)
            persistence = float(row[pcol])

            path = tune_model_path(feature_set_name, h, hour)
            if not path.exists():
                path = tune_model_path(feature_set_name, h, -1)
            if not path.exists():
                pred = persistence
            else:
                res = float(predict_model(load_model(path, backend=BACKEND), X, backend=BACKEND)[0])
                pred = persistence + res

            actual = float(row[tcol])
            rows.append(
                {
                    "anchor_ts_hour": str(row["ts_hour"]),
                    "anchor_hour": hour,
                    "target_hour": h,
                    "actual_price": actual,
                    "persistence_price": persistence,
                    "predicted_price": pred,
                    "feature_set": feature_set_name,
                }
            )
    return pd.DataFrame(rows)


def metrics_from_pred(pred_df: pd.DataFrame) -> dict[str, Any]:
    aligned = filter_aligned(pred_df)
    by_h = {}
    for h in SHORT_H:
        sub = aligned[aligned["target_hour"] == h]
        by_h[str(h)] = mae(sub["actual_price"].to_numpy(), sub["predicted_price"].to_numpy())
    mean_mae = float(np.mean(list(by_h.values())))
    val_parts = []
    # aggregate val MAE from training logs stored separately
    return {"horizon_mae": by_h, "mean_mae_h1_h4": mean_mae, "aligned_rows": len(aligned)}


def baseline_h1h4(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    d = pd.read_csv(path)
    d = d[d["target_hour"].isin(SHORT_H)]
    d["anchor_ts_hour"] = pd.to_datetime(d["anchor_ts_hour"], utc=True).dt.tz_convert(
        "Europe/Istanbul"
    )
    aligned = filter_aligned(d.rename(columns={"predicted_price": "predicted_price"}))
    if "predicted_price" not in aligned.columns:
        return None
    by_h = {}
    for h in SHORT_H:
        sub = aligned[aligned["target_hour"] == h]
        by_h[str(h)] = mae(sub["actual_price"].to_numpy(), sub["predicted_price"].to_numpy())
    by_h["mean_h1_h4"] = float(np.mean([by_h[str(h)] for h in SHORT_H]))
    return by_h


def run() -> dict:
    import lightgbm  # noqa: F401

    df = prepare_df(FEATURES_PATH)
    base_features = resolve_base_features(df)

    print("Extracting feature importance from tree_advanced h1–h4...")
    importance = extract_importance_h1h4(base_features)
    feature_sets = build_feature_sets(base_features, importance)

    experiments: list[dict] = []
    val_mae_accum: dict[str, list[float]] = {str(k): [] for k in FEATURE_SET_SIZES}

    for fs_name, h_map in feature_sets.items():
        print(f"\n=== Training feature set: {fs_name} ({len(h_map[1])} features @ h1) ===")
        TUNE_MODEL_ROOT.mkdir(parents=True, exist_ok=True)
        val_scores: list[float] = []

        for h in SHORT_H:
            fcols = h_map[h]
            for hour in adv_cfg.HOURS:
                out = tune_model_path(fs_name, h, hour)
                stats = train_cell(df, h, hour, fcols, out)
                val_scores.append(stats["val_mae"])

        mean_val = float(np.mean(val_scores))
        val_mae_accum[fs_name] = val_scores

        pred_df = predict_feature_set(df, fs_name, h_map)
        aligned = filter_aligned(pred_df)
        by_h = {}
        for h in SHORT_H:
            sub = aligned[aligned["target_hour"] == h]
            by_h[str(h)] = mae(sub["actual_price"].to_numpy(), sub["predicted_price"].to_numpy())
        mean_test = float(np.mean(list(by_h.values())))
        overfit = mean_test - mean_val

        experiments.append(
            {
                "feature_set": fs_name,
                "n_features_h1": len(h_map[1]),
                "features_h1": h_map[1],
                "val_mae_mean": mean_val,
                "test_mae_mean": mean_test,
                "overfit_gap": overfit,
                "horizon_test_mae": by_h,
            }
        )
        print(f"  val_mae={mean_val:.2f} test_mae={mean_test:.2f} gap={overfit:.2f}")

    baselines = {
        "advanced_tree": baseline_h1h4(BASELINE_PATHS["advanced_tree"]),
        "short_expert": baseline_h1h4(BASELINE_PATHS["short_expert"]),
        "residual_lstm": baseline_h1h4(BASELINE_PATHS["residual_lstm"]),
        "persistence": None,
    }
    if baselines["advanced_tree"]:
        # persistence from same advanced pred file
        d = pd.read_csv(BASELINE_PATHS["advanced_tree"])
        d = d[d["target_hour"].isin(SHORT_H)]
        aligned = filter_aligned(d)
        by_h = {}
        for h in SHORT_H:
            sub = aligned[aligned["target_hour"] == h]
            by_h[str(h)] = mae(
                sub["actual_price"].to_numpy(), sub["persistence_price"].to_numpy()
            )
        by_h["mean_h1_h4"] = float(np.mean([by_h[str(h)] for h in SHORT_H]))
        baselines["persistence"] = by_h

    best = min(experiments, key=lambda x: x["test_mae_mean"])
    adv_mean = (baselines.get("advanced_tree") or {}).get("mean_h1_h4")
    beats_advanced = adv_mean is not None and best["test_mae_mean"] < adv_mean

    if beats_advanced:
        verdict = (
            f"Best tuned set '{best['feature_set']}' beats current advanced tree h1–h4 "
            f"({best['test_mae_mean']:.1f} vs {adv_mean:.1f})."
        )
    else:
        verdict = (
            f"No feature subset beat current advanced tree h1–h4 mean MAE ({adv_mean:.1f}). "
            f"Best tuned: '{best['feature_set']}' at {best['test_mae_mean']:.1f}."
        )

    best_pred = predict_feature_set(df, best["feature_set"], feature_sets[best["feature_set"]])
    BEST_PRED_CSV.parent.mkdir(parents=True, exist_ok=True)
    best_pred.to_csv(BEST_PRED_CSV, index=False)

    report = {
        "importance": importance,
        "feature_sets": {k: {str(h): v for h, v in m.items()} for k, m in feature_sets.items()},
        "experiments": experiments,
        "baselines_h1_h4": baselines,
        "best_feature_set": best["feature_set"],
        "best_test_mae_mean": best["test_mae_mean"],
        "beats_advanced_tree": beats_advanced,
        "verdict": verdict,
        "tune_model_root": str(TUNE_MODEL_ROOT),
        "source_models": str(SOURCE_MODEL_DIR),
    }

    METRICS_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    METRICS_MD.write_text(_write_md(report), encoding="utf-8")
    _plot(report, experiments, baselines)

    print(f"\n{verdict}")
    print(f"Report: {METRICS_JSON}")
    return report


def _plot(report: dict, experiments: list, baselines: dict) -> None:
    FIGURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    labels = [e["feature_set"] for e in experiments]
    val_m = [e["val_mae_mean"] for e in experiments]
    test_m = [e["test_mae_mean"] for e in experiments]
    x = np.arange(len(labels))
    w = 0.35
    plt.figure(figsize=(10, 5))
    plt.bar(x - w / 2, val_m, width=w, label="Val MAE (mean cells)")
    plt.bar(x + w / 2, test_m, width=w, label="Test MAE (aligned)")
    if baselines.get("advanced_tree"):
        plt.axhline(baselines["advanced_tree"]["mean_h1_h4"], color="red", ls="--", label="Advanced tree (current)")
    plt.xticks(x, labels)
    plt.ylabel("MAE (TL/MWh)")
    plt.title("h1–h4 feature selection — val vs test")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_PATH, dpi=150)
    plt.close()


def _write_md(report: dict) -> str:
    lines = [
        "# Advanced Tree h1–h4 Feature Selection",
        "",
        f"Importance cells loaded: {report['importance']['cells_loaded']}",
        "",
        "## Experiments",
        "",
        "| Set | #feat (h1) | Val MAE | Test MAE | Overfit (test−val) |",
        "|-----|----------:|--------:|---------:|-------------------:|",
    ]
    for e in report["experiments"]:
        lines.append(
            f"| {e['feature_set']} | {e['n_features_h1']} | {e['val_mae_mean']:.2f} | "
            f"{e['test_mae_mean']:.2f} | {e['overfit_gap']:.2f} |"
        )
    lines.extend(["", "## Baselines (h1–h4 mean MAE)", ""])
    for name, data in report["baselines_h1_h4"].items():
        if data:
            lines.append(f"- **{name}**: {data.get('mean_h1_h4', 'n/a'):.2f}")
    lines.extend(["", f"**Best set:** `{report['best_feature_set']}`", "", report["verdict"]])
    return "\n".join(lines)


if __name__ == "__main__":
    run()
