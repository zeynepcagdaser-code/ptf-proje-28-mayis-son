#!/usr/bin/env python3
"""Validation-weighted ensemble h5–h12 (advanced tree + microstructure)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
MICRO_MODEL_DIR = PROJECT_ROOT / "models" / "microstructure_h5h12"
TRAIN_MICRO_SCRIPT = PROJECT_ROOT / "train_microstructure_h5h12.py"

HORIZONS = list(range(5, 13))
WEIGHT_GRID = [round(w, 1) for w in np.arange(0.0, 1.01, 0.1)]

VAL_PRED_PATHS = {
    "advanced_tree": PROJECT_ROOT
    / "data"
    / "predictions"
    / "tree_advanced_h5h12_validation_predictions.csv",
    "microstructure": PROJECT_ROOT
    / "data"
    / "predictions"
    / "microstructure_h5h12_validation_predictions.csv",
}
TEST_PRED_PATHS = {
    "advanced_tree": PROJECT_ROOT / "data" / "predictions" / "tree_test_predictions.csv",
    "microstructure": PROJECT_ROOT / "data" / "predictions" / "microstructure_h5h12_predictions.csv",
    "persistence": PROJECT_ROOT / "data" / "predictions" / "persistence_predictions.csv",
}

METRICS_JSON = PROJECT_ROOT / "reports" / "h5h12_validation_weighted_ensemble_metrics.json"
METRICS_MD = PROJECT_ROOT / "reports" / "h5h12_validation_weighted_ensemble_metrics.md"
PREDICTIONS_CSV = (
    PROJECT_ROOT / "data" / "predictions" / "h5h12_validation_weighted_ensemble_predictions.csv"
)
FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "h5h12_validation_weighted_ensemble.png"


def mae(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - pred)))


def improvement_pct(model_mae: float, base_mae: float) -> float:
    if base_mae <= 0:
        return 0.0
    return (base_mae - model_mae) / base_mae * 100.0


def horizon_mae(df: pd.DataFrame, pred_col: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for h in HORIZONS:
        sub = df[df["target_hour"] == h]
        out[str(h)] = mae(sub["actual_price"].to_numpy(), sub[pred_col].to_numpy())
    out["mean_h5_h12"] = float(np.mean([out[str(h)] for h in HORIZONS]))
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


def ensure_micro_models(*, train_if_missing: bool) -> None:
    missing = [h for h in HORIZONS if not (MICRO_MODEL_DIR / f"horizon_{h:02d}.txt").exists()]
    if not missing:
        return
    if not train_if_missing:
        raise FileNotFoundError(
            f"Missing microstructure models for horizons {missing}. "
            f"Run: python3 {TRAIN_MICRO_SCRIPT.name}"
        )
    print(f"Training microstructure h5–h12 ({len(missing)} horizons)...")
    subprocess.run([sys.executable, str(TRAIN_MICRO_SCRIPT)], check=True, cwd=PROJECT_ROOT)


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
            pred, _, _ = predict_row(
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

    subset = df[df["split"] == split].copy()
    rows: list[dict] = []
    for h in HORIZONS:
        model_path = MICRO_MODEL_DIR / f"horizon_{h:02d}.txt"
        fcols = json.loads((MICRO_MODEL_DIR / f"horizon_{h:02d}_features.json").read_text(encoding="utf-8"))
        tcol, pcol = f"target_{h}h", f"persistence_{h}h"
        valid = subset.dropna(subset=fcols + [tcol, pcol])
        booster = lgb.Booster(model_file=str(model_path))
        res = booster.predict(valid[fcols].to_numpy(dtype=np.float64))
        for i, idx in enumerate(valid.index):
            row = valid.loc[idx]
            rows.append(
                {
                    "anchor_ts_hour": str(row["ts_hour"]),
                    "target_hour": h,
                    "actual_price": float(row[tcol]),
                    "persistence_price": float(row[pcol]),
                    "predicted_price": float(row[pcol]) + float(res[i]),
                }
            )
    return pd.DataFrame(rows)


def ensure_validation_predictions(*, regenerate: bool) -> dict[str, Path]:
    backend = pick_backend()
    tree_df = prepare_df(TREE_FEATURES)
    tree_base = resolve_base_features(tree_df)
    micro_df = add_micro_persistence(pd.read_parquet(MICRO_FEATURES))

    outputs: dict[str, Path] = {}
    builders = {
        "advanced_tree": lambda: build_advanced_tree_split_predictions(
            tree_df, tree_base, backend, split="validation", use_online=False
        ),
        "microstructure": lambda: build_microstructure_split_predictions(micro_df, split="validation"),
    }
    for name, builder in builders.items():
        path = VAL_PRED_PATHS[name]
        if regenerate or not path.exists():
            pred = builder()
            path.parent.mkdir(parents=True, exist_ok=True)
            pred.to_csv(path, index=False)
            print(f"Wrote {path} ({len(pred)} rows)")
        else:
            print(f"Using {path}")
        outputs[name] = path
    return outputs


def load_split_preds(path: Path, anchor_path: Path, pred_col: str = "predicted_price") -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["target_hour"].isin(HORIZONS)].copy()
    df = filter_anchors(df, anchor_path)
    if pred_col not in df.columns:
        pred_col = "predicted_price"
    return df[["anchor_ts_hour", "target_hour", "actual_price", pred_col]].rename(columns={pred_col: "pred"})


def merge_pair(val_paths: dict[str, Path], anchor_path: Path) -> pd.DataFrame:
    adv = load_split_preds(val_paths["advanced_tree"], anchor_path).rename(columns={"pred": "advanced_tree"})
    micro = load_split_preds(val_paths["microstructure"], anchor_path).rename(columns={"pred": "microstructure"})
    return adv.merge(
        micro[["anchor_ts_hour", "target_hour", "microstructure"]],
        on=["anchor_ts_hour", "target_hour"],
        how="inner",
    ).sort_values(["anchor_ts_hour", "target_hour"])


def select_weights_validation(val_df: pd.DataFrame) -> dict[str, Any]:
    selected: dict[str, float] = {}
    search_detail: dict[str, list[dict[str, float]]] = {}
    for h in HORIZONS:
        sub = val_df[val_df["target_hour"] == h]
        actual = sub["actual_price"].to_numpy()
        adv = sub["advanced_tree"].to_numpy()
        micro = sub["microstructure"].to_numpy()
        curve: list[dict[str, float]] = []
        best_w, best_mae = 0.0, float("inf")
        for w in WEIGHT_GRID:
            err = mae(actual, w * adv + (1.0 - w) * micro)
            curve.append({"w_advanced": w, "mae": err})
            if err <= best_mae:
                best_mae, best_w = err, w
        selected[str(h)] = float(best_w)
        search_detail[str(h)] = curve
    return {"selected_weights_advanced": selected, "validation_search": search_detail}


def apply_horizon_weights(df: pd.DataFrame, weights: dict[str, float], col: str) -> pd.Series:
    out = np.empty(len(df), dtype=np.float64)
    for h in HORIZONS:
        w = weights[str(h)]
        mask = df["target_hour"] == h
        out[mask.to_numpy()] = (
            w * df.loc[mask, "advanced_tree"].to_numpy()
            + (1.0 - w) * df.loc[mask, "microstructure"].to_numpy()
        )
    return pd.Series(out, index=df.index, name=col)


def oracle_weights_test(test_df: pd.DataFrame) -> dict[str, float]:
    oracle: dict[str, float] = {}
    for h in HORIZONS:
        sub = test_df[test_df["target_hour"] == h]
        actual, adv, micro = sub["actual_price"].to_numpy(), sub["advanced_tree"].to_numpy(), sub[
            "microstructure"
        ].to_numpy()
        best_w, best_mae = 0.0, float("inf")
        for w in WEIGHT_GRID:
            err = mae(actual, w * adv + (1.0 - w) * micro)
            if err <= best_mae:
                best_mae, best_w = err, w
        oracle[str(h)] = float(best_w)
    return oracle


def attach_improvement(metrics: dict[str, float], base: dict[str, float]) -> dict[str, Any]:
    out = dict(metrics)
    out["improvement_pct_vs_persistence"] = {
        k: improvement_pct(metrics[k], base[k]) for k in list(metrics.keys()) if k != "mean_h5_h12"
    }
    out["improvement_pct_vs_persistence"]["mean_h5_h12"] = improvement_pct(
        metrics["mean_h5_h12"], base["mean_h5_h12"]
    )
    return out


def write_md(payload: dict[str, Any]) -> None:
    sel = payload["selected_weights_validation"]
    tm = payload["test_metrics"]
    pers = tm["persistence"]
    lines = [
        "# h5–h12 validation-weighted ensemble",
        "",
        "Weights chosen on **validation** only; applied to test. h1–h4 checkpoint untouched.",
        "",
        f"- Validation rows: {payload['validation_rows']}",
        f"- Test rows: {payload['test_rows']}",
        f"- Advanced tree validation inference: {payload['advanced_tree_validation_inference']}",
        "",
        "## Selected weights",
        "",
        "| h | w_adv | w_micro |",
        "|--:|------:|--------:|",
    ]
    for h in HORIZONS:
        w = sel[str(h)]
        lines.append(f"| {h} | {w:.1f} | {1 - w:.1f} |")
    lines.extend(["", "## Test MAE (TL/MWh)", ""])
    hdr = "| Model | " + " | ".join(f"h{h}" for h in HORIZONS) + " | Mean h5–h12 | vs adv | vs pers % |"
    sep = "|-------|" + "|".join(["-----:"] * (len(HORIZONS) + 3)) + "|"
    lines.extend([hdr, sep])
    adv_mean = tm["advanced_tree"]["mean_h5_h12"]
    for name in [
        "persistence",
        "advanced_tree",
        "microstructure",
        "validation_weighted_ensemble",
        "test_oracle_weights",
    ]:
        m = tm[name]
        tag = " **PRIMARY**" if name == "validation_weighted_ensemble" else ""
        if name == "test_oracle_weights":
            tag = " *(test oracle)*"
        vs_adv = m["mean_h5_h12"] - adv_mean
        vs_p = m["improvement_pct_vs_persistence"]["mean_h5_h12"]
        row = f"| {name}{tag} | " + " | ".join(f"{m[str(h)]:.2f}" for h in HORIZONS)
        row += f" | **{m['mean_h5_h12']:.2f}** | {vs_adv:+.2f} | {vs_p:+.1f}% |"
        lines.append(row)
    lines.extend(["", "## Verdict", "", payload["verdict"]])
    METRICS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_results(test_metrics: dict[str, Any], adv_mean: float) -> None:
    names = ["persistence", "advanced_tree", "microstructure", "validation_weighted_ensemble", "test_oracle_weights"]
    labels = ["Persistence", "Advanced", "Micro", "Val-weighted\n(PRIMARY)", "Test oracle"]
    means = [test_metrics[n]["mean_h5_h12"] for n in names]
    colors = ["#888888", "#55A868", "#DD8452", "#C44E52", "#CCB974"]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    bars = ax.bar(x, means, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(adv_mean, linestyle="--", color="#55A868", label="advanced tree")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean MAE h5–h12 (TL/MWh)")
    ax.set_title("h5–h12 validation-weighted ensemble (test)")
    ax.legend()
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{val:.1f}", ha="center", fontsize=8)
    fig.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=150)
    plt.close(fig)


def run(*, regenerate_val: bool = False, train_micro: bool = True) -> dict[str, Any]:
    ensure_micro_models(train_if_missing=train_micro)
    if not TEST_PRED_PATHS["microstructure"].exists():
        subprocess.run([sys.executable, str(TRAIN_MICRO_SCRIPT)], check=True, cwd=PROJECT_ROOT)

    val_paths = ensure_validation_predictions(regenerate=regenerate_val)
    val_df = merge_pair(val_paths, ANCHOR_VAL_PATH)
    weight_info = select_weights_validation(val_df)
    weights = weight_info["selected_weights_advanced"]

    keys = ["anchor_ts_hour", "target_hour"]
    test_adv = load_split_preds(TEST_PRED_PATHS["advanced_tree"], ANCHOR_TEST_PATH).rename(
        columns={"pred": "advanced_tree"}
    )
    test_micro = load_split_preds(TEST_PRED_PATHS["microstructure"], ANCHOR_TEST_PATH).rename(
        columns={"pred": "microstructure"}
    )
    test_pers = load_split_preds(TEST_PRED_PATHS["persistence"], ANCHOR_TEST_PATH).rename(
        columns={"pred": "persistence"}
    )
    test_df = test_adv.merge(test_micro[keys + ["microstructure"]], on=keys, how="inner")
    test_df = test_df.merge(test_pers[keys + ["persistence"]], on=keys, how="inner")
    test_df["validation_weighted_ensemble"] = apply_horizon_weights(test_df, weights, "validation_weighted_ensemble")
    oracle_w = oracle_weights_test(test_df)
    test_df["test_oracle_weights"] = apply_horizon_weights(test_df, oracle_w, "test_oracle_weights")

    test_metrics_raw = {
        "persistence": horizon_mae(test_df, "persistence"),
        "advanced_tree": horizon_mae(test_df, "advanced_tree"),
        "microstructure": horizon_mae(test_df, "microstructure"),
        "validation_weighted_ensemble": horizon_mae(test_df, "validation_weighted_ensemble"),
        "test_oracle_weights": horizon_mae(test_df, "test_oracle_weights"),
    }
    pers_base = test_metrics_raw["persistence"]
    test_metrics = {k: attach_improvement(v, pers_base) for k, v in test_metrics_raw.items()}

    primary_mae = test_metrics["validation_weighted_ensemble"]["mean_h5_h12"]
    adv_mae = test_metrics["advanced_tree"]["mean_h5_h12"]
    beats_adv = primary_mae < adv_mae

    micro_mae = test_metrics["microstructure"]["mean_h5_h12"]
    if beats_adv:
        verdict = (
            f"Validation-weighted ensemble beats advanced tree on h5–h12 test "
            f"({primary_mae:.2f} vs {adv_mae:.2f}, Δ {primary_mae - adv_mae:+.2f})."
        )
    elif abs(primary_mae - adv_mae) < 1e-6:
        verdict = (
            f"Validation selected w=1.0 for all h5–h12 (advanced tree best on validation). "
            f"Test ensemble equals advanced tree (mean MAE {adv_mae:.2f}). "
            f"Microstructure alone is lower on test ({micro_mae:.2f}) but not used — no validation gain."
        )
    else:
        verdict = (
            f"Validation-weighted ensemble is WORSE than advanced tree on h5–h12 test "
            f"({primary_mae:.2f} vs {adv_mae:.2f}, Δ {primary_mae - adv_mae:+.2f})."
        )

    payload: dict[str, Any] = {
        "scope": "h5_h12_only",
        "h1_h4_checkpoint_untouched": True,
        "method": "per_horizon_validation_weight_search",
        "advanced_tree_validation_inference": "regressor (train-only), not regressor_online",
        "weight_grid_advanced": WEIGHT_GRID,
        "selected_weights_validation": weights,
        "validation_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "validation_prediction_paths": {k: str(v) for k, v in val_paths.items()},
        "test_prediction_paths": {k: str(v) for k, v in TEST_PRED_PATHS.items()},
        "validation_weight_search": weight_info["validation_search"],
        "test_metrics": test_metrics,
        "test_oracle_weights": oracle_w,
        "primary_result": "validation_weighted_ensemble",
        "primary_beats_advanced_tree": beats_adv,
        "verdict": verdict,
    }

    METRICS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_md(payload)
    plot_results(test_metrics, adv_mae)

    out = test_df[
        keys
        + ["actual_price", "persistence", "advanced_tree", "microstructure", "validation_weighted_ensemble", "test_oracle_weights"]
    ].copy()
    out["weight_advanced_h"] = out["target_hour"].map(lambda h: weights[str(int(h))])
    PREDICTIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(PREDICTIONS_CSV, index=False)

    print(f"Selected weights: {weights}")
    print(f"Test mean h5–h12: ensemble={primary_mae:.2f}, advanced={adv_mae:.2f}")
    print(verdict)
    print(f"Metrics: {METRICS_JSON}")
    return payload


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="h5–h12 validation-weighted ensemble")
    p.add_argument("--regenerate-val-predictions", action="store_true")
    p.add_argument("--skip-train-micro", action="store_true")
    args = p.parse_args()
    run(regenerate_val=args.regenerate_val_predictions, train_micro=not args.skip_train_micro)


if __name__ == "__main__":
    main()
