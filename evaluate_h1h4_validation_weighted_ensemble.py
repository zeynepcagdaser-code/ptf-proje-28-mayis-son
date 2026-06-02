#!/usr/bin/env python3
"""
Validation-based per-horizon ensemble weights (advanced tree + microstructure).
No training — inference only for validation CSVs; test weights frozen from validation.
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
from scipy.optimize import minimize

from tree_advanced.backend import pick_backend
from tree_advanced.pipeline import (
    horizon_features,
    predict_row,
    prepare_df,
    resolve_base_features,
)

PROJECT_ROOT = Path(__file__).resolve().parent
ANCHOR_VAL_PATH = PROJECT_ROOT / "data" / "model" / "anchor_val.csv"
ANCHOR_TEST_PATH = PROJECT_ROOT / "data" / "model" / "anchor_test.csv"

TREE_FEATURES = PROJECT_ROOT / "data" / "features" / "lstm_tree_micro_v1.parquet"
MICRO_FEATURES = PROJECT_ROOT / "data" / "features" / "lstm_microstructure_next24_v1.parquet"
MICRO_MODEL_DIR = PROJECT_ROOT / "models" / "microstructure_h1h4"

VAL_PRED_PATHS = {
    "advanced_tree": PROJECT_ROOT
    / "data"
    / "predictions"
    / "tree_advanced_h1h4_validation_predictions.csv",
    "microstructure": PROJECT_ROOT
    / "data"
    / "predictions"
    / "microstructure_h1h4_validation_predictions.csv",
}
TEST_PRED_PATHS = {
    "advanced_tree": PROJECT_ROOT / "data" / "predictions" / "tree_test_predictions.csv",
    "microstructure": PROJECT_ROOT
    / "data"
    / "predictions"
    / "microstructure_h1h4_predictions.csv",
}

METRICS_JSON = PROJECT_ROOT / "reports" / "h1h4_validation_weighted_ensemble_metrics.json"
METRICS_MD = PROJECT_ROOT / "reports" / "h1h4_validation_weighted_ensemble_metrics.md"
PREDICTIONS_CSV = (
    PROJECT_ROOT / "data" / "predictions" / "h1h4_validation_weighted_ensemble_predictions.csv"
)
FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "h1h4_validation_weighted_ensemble.png"
OPT_WEIGHTS_JSON = PROJECT_ROOT / "reports" / "h1h4_optimized_weights.json"

HORIZONS = [1, 2, 3, 4]
WEIGHT_GRID = [round(w, 1) for w in np.arange(0.0, 1.01, 0.1)]
ADVANCED_TREE_TEST_BASELINE = 453.0947792677989


def mae(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - pred)))


def horizon_mae(df: pd.DataFrame, pred_col: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for h in HORIZONS:
        sub = df[df["target_hour"] == h]
        out[str(h)] = mae(sub["actual_price"].to_numpy(), sub[pred_col].to_numpy())
    out["mean_h1_h4"] = float(np.mean([out[str(h)] for h in HORIZONS]))
    return out


def filter_anchors(pred_df: pd.DataFrame, anchor_path: Path) -> pd.DataFrame:
    if not anchor_path.exists():
        return pred_df
    anchors = pd.read_csv(anchor_path)
    anchors["anchor_ts_hour"] = pd.to_datetime(
        anchors["anchor_ts_hour"], utc=True
    ).dt.tz_convert("Europe/Istanbul")
    out = pred_df.copy()
    out["anchor_ts_hour"] = pd.to_datetime(out["anchor_ts_hour"], utc=True).dt.tz_convert(
        "Europe/Istanbul"
    )
    return out.merge(anchors[["anchor_ts_hour"]], on="anchor_ts_hour", how="inner")


def build_advanced_tree_split_predictions(
    df: pd.DataFrame,
    base_features: list[str],
    backend: str,
    *,
    split: str,
    use_online: bool,
) -> pd.DataFrame:
    subset = df[df["split"] == split].copy()
    rows: list[dict] = []
    for h in HORIZONS:
        tcol = f"target_{h}h"
        fcols = horizon_features(base_features, h)
        valid = subset.dropna(subset=fcols + [tcol, f"persistence_{h}h"])
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
                    "target_hour": h,
                    "actual_price": actual,
                    "persistence_price": persistence,
                    "predicted_price": pred,
                    "prob_zero": pz,
                    "prob_spike": ps,
                }
            )
    return pd.DataFrame(rows)


def add_micro_persistence(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("ts_hour").reset_index(drop=True)
    for h in HORIZONS:
        out[f"persistence_{h}h"] = out[f"target_{h}h"].shift(24)
    return out


def micro_base_features(df: pd.DataFrame) -> list[str]:
    target_cols = [c for c in df.columns if c.startswith("target_")]
    exclude = {"ts_hour", "split", *target_cols}
    exclude.update(c for c in df.columns if c.startswith("persistence_"))
    return [c for c in df.columns if c not in exclude]


def build_microstructure_split_predictions(df: pd.DataFrame, *, split: str) -> pd.DataFrame:
    import lightgbm as lgb

    base = micro_base_features(df)
    subset = df[df["split"] == split].copy()
    rows: list[dict] = []

    for h in HORIZONS:
        model_path = MICRO_MODEL_DIR / f"horizon_{h:02d}.txt"
        feat_path = MICRO_MODEL_DIR / f"horizon_{h:02d}_features.json"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model {model_path}")
        fcols = json.loads(feat_path.read_text(encoding="utf-8"))
        tcol, pcol = f"target_{h}h", f"persistence_{h}h"
        valid = subset.dropna(subset=fcols + [tcol, pcol])
        booster = lgb.Booster(model_file=str(model_path))
        X = valid[fcols].to_numpy(dtype=np.float64)
        res = booster.predict(X)
        for i, idx in enumerate(valid.index):
            row = valid.loc[idx]
            actual = float(row[tcol])
            persistence = float(row[pcol])
            pred = persistence + float(res[i])
            rows.append(
                {
                    "anchor_ts_hour": str(row["ts_hour"]),
                    "target_hour": h,
                    "actual_price": actual,
                    "persistence_price": persistence,
                    "predicted_price": pred,
                }
            )
    return pd.DataFrame(rows)


def ensure_validation_predictions(*, regenerate: bool = False) -> dict[str, Path]:
    backend = pick_backend()
    tree_df = prepare_df(TREE_FEATURES)
    tree_base = resolve_base_features(tree_df)

    from src.utils.io_utils import read_parquet_with_normalized_ts
    micro_df = add_micro_persistence(read_parquet_with_normalized_ts(MICRO_FEATURES))

    outputs: dict[str, Path] = {}
    for name, builder in [
        (
            "advanced_tree",
            lambda: build_advanced_tree_split_predictions(
                tree_df,
                tree_base,
                backend,
                split="validation",
                # regressor_online is fit on train+val → in-sample on validation; use train-only regressor
                use_online=False,
            ),
        ),
        ("microstructure", lambda: build_microstructure_split_predictions(micro_df, split="validation")),
    ]:
        path = VAL_PRED_PATHS[name]
        if regenerate or not path.exists():
            pred = builder()
            path.parent.mkdir(parents=True, exist_ok=True)
            pred.to_csv(path, index=False)
            print(f"Wrote validation predictions: {path} ({len(pred)} rows)")
        else:
            print(f"Using existing validation predictions: {path}")
        outputs[name] = path
    return outputs


def load_split_preds(path: Path, anchor_path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["target_hour"].isin(HORIZONS)].copy()
    df = filter_anchors(df, anchor_path)
    return df[
        ["anchor_ts_hour", "target_hour", "actual_price", "predicted_price"]
    ].rename(columns={"predicted_price": "pred"})


def merge_pair(val_paths: dict[str, Path], anchor_path: Path) -> pd.DataFrame:
    adv = load_split_preds(val_paths["advanced_tree"], anchor_path).rename(columns={"pred": "advanced_tree"})
    micro = load_split_preds(val_paths["microstructure"], anchor_path).rename(columns={"pred": "microstructure"})
    merged = adv.merge(
        micro[["anchor_ts_hour", "target_hour", "microstructure"]],
        on=["anchor_ts_hour", "target_hour"],
        how="inner",
    )
    return merged.sort_values(["anchor_ts_hour", "target_hour"]).reset_index(drop=True)

def optimize_primary_weights(val_df: pd.DataFrame) -> dict[str, Any]:
    """
    Optimize a single global weight pair for (advanced_tree, microstructure)
    on validation predictions only.

    Constraints:
      - w_adv in [0,1]
      - w_micro = 1 - w_adv (so w_adv + w_micro = 1)
    Objective:
      - minimize MAE over all validation rows (h1-h4 combined)
    """
    if val_df.empty:
        return {"available": False, "reason": "validation dataframe empty"}

    actual = val_df["actual_price"].to_numpy(dtype=float)
    adv = val_df["advanced_tree"].to_numpy(dtype=float)
    micro = val_df["microstructure"].to_numpy(dtype=float)

    def obj(x: np.ndarray) -> float:
        w_adv = float(x[0])
        pred = w_adv * adv + (1.0 - w_adv) * micro
        return mae(actual, pred)

    res = minimize(
        obj,
        x0=np.array([0.7], dtype=float),
        bounds=[(0.0, 1.0)],
        method="SLSQP",
        options={"maxiter": 200, "ftol": 1e-10},
    )
    w_adv = float(np.clip(res.x[0], 0.0, 1.0))
    w_micro = 1.0 - w_adv
    pred = w_adv * adv + w_micro * micro
    return {
        "available": True,
        "objective": "mae",
        "constraints": {"w_adv_in_[0,1]": True, "w_adv_plus_w_micro_eq_1": True},
        "optimized_weights": {"advanced_tree": w_adv, "microstructure": w_micro},
        "validation_mae": mae(actual, pred),
        "optimizer": {
            "method": "scipy.optimize.minimize",
            "solver": "SLSQP",
            "success": bool(res.success),
            "status": int(res.status),
            "message": str(res.message),
            "nfev": int(getattr(res, "nfev", -1)),
        },
    }


def select_weights_validation(val_df: pd.DataFrame) -> dict[str, Any]:
    selected: dict[str, float] = {}
    validation_mae_by_w: dict[str, dict[str, float]] = {}
    search_detail: dict[str, list[dict[str, float]]] = {}

    for h in HORIZONS:
        sub = val_df[val_df["target_hour"] == h]
        actual = sub["actual_price"].to_numpy()
        adv = sub["advanced_tree"].to_numpy()
        micro = sub["microstructure"].to_numpy()
        curve: list[dict[str, float]] = []
        best_w = 0.0
        best_mae = float("inf")
        for w in WEIGHT_GRID:
            pred = w * adv + (1.0 - w) * micro
            err = mae(actual, pred)
            curve.append({"w_advanced": w, "mae": err})
            if err <= best_mae:
                best_mae = err
                best_w = w
        selected[str(h)] = float(best_w)
        validation_mae_by_w[str(h)] = {str(c["w_advanced"]): c["mae"] for c in curve}
        search_detail[str(h)] = curve

    return {
        "selected_weights_advanced": selected,
        "validation_search": search_detail,
        "validation_mae_by_weight": validation_mae_by_w,
    }


def apply_horizon_weights(df: pd.DataFrame, weights: dict[str, float], col: str) -> pd.Series:
    out = np.empty(len(df), dtype=np.float64)
    for h in HORIZONS:
        w = weights[str(h)]
        mask = df["target_hour"] == h
        out[mask.to_numpy()] = w * df.loc[mask, "advanced_tree"].to_numpy() + (1.0 - w) * df.loc[
            mask, "microstructure"
        ].to_numpy()
    return pd.Series(out, index=df.index, name=col)


def oracle_weights_test(test_df: pd.DataFrame) -> dict[str, float]:
    oracle: dict[str, float] = {}
    for h in HORIZONS:
        sub = test_df[test_df["target_hour"] == h]
        actual = sub["actual_price"].to_numpy()
        adv = sub["advanced_tree"].to_numpy()
        micro = sub["microstructure"].to_numpy()
        best_w = 0.0
        best_mae = float("inf")
        for w in WEIGHT_GRID:
            err = mae(actual, w * adv + (1.0 - w) * micro)
            if err <= best_mae:
                best_mae = err
                best_w = w
        oracle[str(h)] = best_w
    return oracle


def write_md(payload: dict[str, Any]) -> None:
    sel = payload["selected_weights_validation"]
    test_m = payload["test_metrics"]
    lines = [
        "# h1–h4 validation-weighted ensemble",
        "",
        "Per-horizon blend weights chosen on **validation** only; applied to test.",
        "",
        f"- Advanced tree on validation: {payload.get('advanced_tree_validation_inference', 'regressor (train-only)')}",
        f"- {payload.get('note', '')}",
        "",
        f"- Validation rows (aligned): {payload['validation_rows']}",
        f"- Test rows (aligned): {payload['test_rows']}",
        "",
        "## Selected weights (w × advanced + (1−w) × microstructure)",
        "",
        "| Horizon | w (advanced) | w (micro) |",
        "|--------|-------------:|----------:|",
    ]
    for h in HORIZONS:
        w = sel[str(h)]
        lines.append(f"| h{h} | {w:.1f} | {1.0 - w:.1f} |")
    lines.extend(
        [
            "",
            "## Test MAE (TL/MWh)",
            "",
            "| Model | h1 | h2 | h3 | h4 | Mean h1–h4 | vs advanced |",
            "|-------|-----:|-----:|-----:|-----:|-----:|-----:|",
        ]
    )
    for name, m in test_m.items():
        delta = m["mean_h1_h4"] - test_m["advanced_tree_only"]["mean_h1_h4"]
        tag = ""
        if name == "validation_weighted_ensemble":
            tag = " **PRIMARY**"
        elif name == "test_oracle_weights":
            tag = " *(test oracle — leakage)*"
        lines.append(
            f"| {name}{tag} | {m['1']:.2f} | {m['2']:.2f} | {m['3']:.2f} | {m['4']:.2f} | "
            f"**{m['mean_h1_h4']:.2f}** | {delta:+.2f} |"
        )
    lines.extend(["", "## Verdict", "", payload["verdict"]])
    METRICS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_results(payload: dict[str, Any]) -> None:
    test_m = payload["test_metrics"]
    names = [
        "advanced_tree_only",
        "microstructure_only",
        "validation_weighted_ensemble",
        "fixed_0.7_0.3",
        "test_oracle_weights",
    ]
    labels = ["advanced", "micro", "val-weighted\n(PRIMARY)", "fixed 0.7/0.3", "test oracle\n(leakage)"]
    means = [test_m[n]["mean_h1_h4"] for n in names]
    colors = ["#55A868", "#4C72B0", "#C44E52", "#8172B2", "#CCB974"]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(names))
    bars = ax.bar(x, means, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(ADVANCED_TREE_TEST_BASELINE, linestyle="--", color="#55A868", label="advanced baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean MAE h1–h4 (TL/MWh)")
    ax.set_title("Test MAE: validation-selected ensemble weights")
    ax.legend()
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{val:.1f}", ha="center", fontsize=8)
    fig.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=150)
    plt.close(fig)


def run(*, regenerate_val: bool = False) -> dict[str, Any]:
    val_paths = ensure_validation_predictions(regenerate=regenerate_val)
    val_df = merge_pair(val_paths, ANCHOR_VAL_PATH)
    optimized = optimize_primary_weights(val_df)
    OPT_WEIGHTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    OPT_WEIGHTS_JSON.write_text(json.dumps(optimized, ensure_ascii=False, indent=2, default=str) + "\n")
    weight_info = select_weights_validation(val_df)
    weights = weight_info["selected_weights_advanced"]

    val_df["validation_weighted_ensemble"] = apply_horizon_weights(
        val_df, weights, "validation_weighted_ensemble"
    )
    validation_metrics = {
        "advanced_tree_only": horizon_mae(val_df.assign(pred=val_df["advanced_tree"]), "pred"),
        "microstructure_only": horizon_mae(val_df.assign(pred=val_df["microstructure"]), "pred"),
        "validation_weighted_ensemble": horizon_mae(val_df, "validation_weighted_ensemble"),
    }

    test_adv = load_split_preds(TEST_PRED_PATHS["advanced_tree"], ANCHOR_TEST_PATH).rename(
        columns={"pred": "advanced_tree"}
    )
    test_micro = load_split_preds(TEST_PRED_PATHS["microstructure"], ANCHOR_TEST_PATH).rename(
        columns={"pred": "microstructure"}
    )
    test_df = test_adv.merge(
        test_micro[["anchor_ts_hour", "target_hour", "microstructure"]],
        on=["anchor_ts_hour", "target_hour"],
        how="inner",
    )
    test_df["validation_weighted_ensemble"] = apply_horizon_weights(
        test_df, weights, "validation_weighted_ensemble"
    )
    test_df["fixed_0.7_0.3"] = 0.7 * test_df["advanced_tree"] + 0.3 * test_df["microstructure"]

    oracle_w = oracle_weights_test(test_df)
    test_df["test_oracle_weights"] = apply_horizon_weights(test_df, oracle_w, "test_oracle_weights")

    test_metrics = {
        "advanced_tree_only": horizon_mae(test_df.assign(pred=test_df["advanced_tree"]), "pred"),
        "microstructure_only": horizon_mae(test_df.assign(pred=test_df["microstructure"]), "pred"),
        "validation_weighted_ensemble": horizon_mae(test_df, "validation_weighted_ensemble"),
        "fixed_0.7_0.3": horizon_mae(test_df, "fixed_0.7_0.3"),
        "test_oracle_weights": horizon_mae(test_df, "test_oracle_weights"),
    }

    primary_mae = test_metrics["validation_weighted_ensemble"]["mean_h1_h4"]
    adv_mae = test_metrics["advanced_tree_only"]["mean_h1_h4"]
    beats_adv = primary_mae < adv_mae

    if beats_adv:
        verdict = (
            f"Validation-weighted ensemble beats advanced tree on test "
            f"({primary_mae:.2f} vs {adv_mae:.2f}, Δ {primary_mae - adv_mae:+.2f})."
        )
    else:
        verdict = (
            f"Validation-weighted ensemble does NOT beat advanced tree on test "
            f"({primary_mae:.2f} vs {adv_mae:.2f}, Δ {primary_mae - adv_mae:+.2f})."
        )

    payload: dict[str, Any] = {
        "method": "per_horizon_validation_weight_search",
        "advanced_tree_validation_inference": "regressor (train-only), not regressor_online (train+val)",
        "note": (
            "Test predictions use existing CSVs (online/advanced pipeline). "
            "Weights are chosen on validation with out-of-sample advanced tree regressors."
        ),
        "weight_grid_advanced": WEIGHT_GRID,
        "blend_formula": "w * advanced_tree + (1 - w) * microstructure",
        "validation_prediction_paths": {k: str(v) for k, v in val_paths.items()},
        "test_prediction_paths": {k: str(v) for k, v in TEST_PRED_PATHS.items()},
        "anchor_val": str(ANCHOR_VAL_PATH),
        "anchor_test": str(ANCHOR_TEST_PATH),
        "validation_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "optimized_primary_weights": optimized,
        "selected_weights_validation": weights,
        "validation_metrics": validation_metrics,
        "validation_weight_search": weight_info["validation_search"],
        "test_metrics": test_metrics,
        "test_oracle_weights": oracle_w,
        "advanced_tree_test_baseline": ADVANCED_TREE_TEST_BASELINE,
        "primary_result": "validation_weighted_ensemble",
        "primary_beats_advanced_tree": beats_adv,
        "verdict": verdict,
    }

    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_md(payload)
    plot_results(payload)

    out = test_df[
        [
            "anchor_ts_hour",
            "target_hour",
            "actual_price",
            "advanced_tree",
            "microstructure",
            "validation_weighted_ensemble",
            "fixed_0.7_0.3",
            "test_oracle_weights",
        ]
    ].copy()
    out["weight_advanced_h"] = out["target_hour"].map(lambda h: weights[str(int(h))])
    PREDICTIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(PREDICTIONS_CSV, index=False)

    print(f"Validation rows: {len(val_df)}")
    print(f"Selected weights: {weights}")
    print(f"Test primary MAE: {primary_mae:.2f} (advanced only: {adv_mae:.2f})")
    print(verdict)
    print(f"Metrics: {METRICS_JSON}")
    return payload


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Validation-weighted h1–h4 ensemble")
    p.add_argument(
        "--regenerate-val-predictions",
        action="store_true",
        help="Re-run inference on validation split",
    )
    args = p.parse_args()
    run(regenerate_val=args.regenerate_val_predictions)


if __name__ == "__main__":
    main()
