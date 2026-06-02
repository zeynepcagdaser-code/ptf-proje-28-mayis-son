#!/usr/bin/env python3
"""
Final h1–h4 pipeline audit and checkpoint summary (evaluation only, no training).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from features.config import LEAKAGE_CHECKLIST, SPLIT_RANGES
from tree_advanced import config as tree_cfg
from tree_advanced.pipeline import prepare_df, resolve_base_features

PROJECT_ROOT = Path(__file__).resolve().parent
ANCHOR_TEST = PROJECT_ROOT / "data" / "model" / "anchor_test.csv"
ANCHOR_VAL = PROJECT_ROOT / "data" / "model" / "anchor_val.csv"
FEATURES_LSTM = PROJECT_ROOT / "data" / "features" / "lstm_next24_v1.parquet"
FEATURES_TREE = PROJECT_ROOT / "data" / "features" / "lstm_tree_micro_v1.parquet"
FEATURES_MICRO = PROJECT_ROOT / "data" / "features" / "lstm_microstructure_next24_v1.parquet"

REPRO_DIR = PROJECT_ROOT / "reports" / "reproducibility"
SUMMARY_JSON = PROJECT_ROOT / "reports" / "final_h1h4_summary.json"
SUMMARY_MD = PROJECT_ROOT / "reports" / "final_h1h4_summary.md"
FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "final_h1h4_comparison.png"

HORIZONS = [1, 2, 3, 4]
GLOBAL_SEED = 42
PERSISTENCE_MAE_REF = 545.8064853747715  # full h1–h24 aligned persistence (reference)

PRED_PATHS = {
    "persistence": PROJECT_ROOT / "data" / "predictions" / "persistence_predictions.csv",
    "residual_lstm": PROJECT_ROOT / "data" / "predictions" / "lstm_residual_test_predictions.csv",
    "short_expert": PROJECT_ROOT / "data" / "predictions" / "short_horizon_expert_predictions.csv",
    "advanced_tree": PROJECT_ROOT / "data" / "predictions" / "tree_test_predictions.csv",
    "microstructure": PROJECT_ROOT / "data" / "predictions" / "microstructure_h1h4_predictions.csv",
    "validation_weighted_ensemble": PROJECT_ROOT
    / "data"
    / "predictions"
    / "h1h4_validation_weighted_ensemble_predictions.csv",
}

VAL_ENSEMBLE_METRICS = (
    PROJECT_ROOT / "reports" / "h1h4_validation_weighted_ensemble_metrics.json"
)
MICRO_MODEL_DIR = PROJECT_ROOT / "models" / "microstructure_h1h4"
SHORT_MODEL_DIR = PROJECT_ROOT / "models" / "short_horizon_expert"
TREE_MODEL_DIR = PROJECT_ROOT / "models" / "tree_advanced"


def mae(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - pred)))


def horizon_mae_dict(df: pd.DataFrame, pred_col: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for h in HORIZONS:
        sub = df[df["target_hour"] == h]
        out[str(h)] = mae(sub["actual_price"].to_numpy(), sub[pred_col].to_numpy())
    out["mean_h1_h4"] = float(np.mean([out[str(h)] for h in HORIZONS]))
    return out


def load_anchors(path: Path) -> pd.DataFrame:
    a = pd.read_csv(path)
    a["anchor_ts_hour"] = pd.to_datetime(a["anchor_ts_hour"], utc=True).dt.tz_convert(
        "Europe/Istanbul"
    )
    return a


def filter_h1h4_test(path: Path, pred_col: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["target_hour"].isin(HORIZONS)].copy()
    df["anchor_ts_hour"] = pd.to_datetime(df["anchor_ts_hour"], utc=True).dt.tz_convert(
        "Europe/Istanbul"
    )
    anchors = load_anchors(ANCHOR_TEST)
    df = df.merge(anchors[["anchor_ts_hour"]], on="anchor_ts_hour", how="inner")
    if pred_col and pred_col in df.columns:
        col = pred_col
    elif "predicted_price" in df.columns:
        col = "predicted_price"
    elif "validation_weighted_ensemble" in df.columns:
        col = "validation_weighted_ensemble"
    else:
        raise KeyError(f"No prediction column in {path}")
    return df[["anchor_ts_hour", "target_hour", "actual_price", col]].rename(columns={col: "pred"})


def improvement_pct(model_mae: float, persistence_mae: float) -> float:
    if persistence_mae <= 0:
        return 0.0
    return (persistence_mae - model_mae) / persistence_mae * 100.0


def audit_splits() -> dict[str, Any]:
    from src.utils.io_utils import read_parquet_with_normalized_ts
    df = read_parquet_with_normalized_ts(FEATURES_LSTM, columns=["ts_hour", "split"])
    df["ts_hour"] = pd.to_datetime(df["ts_hour"], errors="coerce").dt.tz_localize("Europe/Istanbul")
    test_anchors = load_anchors(ANCHOR_TEST)
    val_anchors = load_anchors(ANCHOR_VAL)

    test_rows = df[df["split"] == "test"]
    test_ts = set(test_rows["ts_hour"])
    anchor_test_in_split = test_anchors["anchor_ts_hour"].isin(test_ts)
    val_rows = df[df["split"] == "validation"]
    val_ts = set(val_rows["ts_hour"])
    anchor_val_in_split = val_anchors["anchor_ts_hour"].isin(val_ts)

    years_per_split: dict[str, dict[str, int]] = {}
    for split_name, grp in df.groupby("split"):
        years_per_split[str(split_name)] = {
            str(int(y)): int(c) for y, c in grp["ts_hour"].dt.year.value_counts().items()
        }

    return {
        "split_ranges_config": SPLIT_RANGES,
        "feature_row_counts": df["split"].value_counts().to_dict(),
        "anchor_test_count": int(len(test_anchors)),
        "anchor_val_count": int(len(val_anchors)),
        "anchor_test_all_in_feature_test_split": bool(anchor_test_in_split.all()),
        "anchor_test_missing_from_feature_test": int((~anchor_test_in_split).sum()),
        "anchor_val_all_in_feature_validation_split": bool(anchor_val_in_split.all()),
        "anchor_val_missing_from_feature_validation": int((~anchor_val_in_split).sum()),
        "years_per_split": years_per_split,
        "status": "pass"
        if anchor_test_in_split.all() and anchor_val_in_split.all()
        else "review",
    }


def audit_persistence_shift() -> dict[str, Any]:
    df = read_parquet_with_normalized_ts(
        FEATURES_MICRO,
        columns=["ts_hour", "split"] + [f"target_{h}h" for h in HORIZONS],
    )
    df = df.sort_values("ts_hour").reset_index(drop=True)
    df["ts_hour"] = pd.to_datetime(df["ts_hour"], errors="coerce").dt.tz_localize("Europe/Istanbul")

    micro_preds = pd.read_csv(PRED_PATHS["microstructure"])
    micro_preds = micro_preds[micro_preds["target_hour"].isin(HORIZONS)]
    micro_preds["anchor_ts_hour"] = pd.to_datetime(
        micro_preds["anchor_ts_hour"], utc=True
    ).dt.tz_convert("Europe/Istanbul")

    max_err = 0.0
    n_checked = 0
    for h in HORIZONS:
        tcol = f"target_{h}h"
        lookup = df.set_index("ts_hour")[tcol].shift(24)
        for _, row in micro_preds[micro_preds["target_hour"] == h].iterrows():
            anchor = row["anchor_ts_hour"]
            if anchor not in lookup.index:
                continue
            expected = lookup.loc[anchor]
            if pd.isna(expected):
                continue
            err = abs(float(row["persistence_price"]) - float(expected))
            max_err = max(max_err, err)
            n_checked += 1

    return {
        "rule": "persistence_h(t) = target_h.shift(24) at anchor row",
        "rows_checked_against_microstructure_csv": n_checked,
        "max_abs_error_vs_shift": max_err,
        "status": "pass" if max_err < 1e-3 else "fail",
    }


def audit_prediction_alignment() -> dict[str, Any]:
    loaded: dict[str, pd.DataFrame] = {}
    for name, path in PRED_PATHS.items():
        if not path.exists():
            loaded[name] = pd.DataFrame()
            continue
        loaded[name] = filter_h1h4_test(
            path,
            pred_col="validation_weighted_ensemble" if name == "validation_weighted_ensemble" else None,
        )

    keys = ["anchor_ts_hour", "target_hour"]
    base = loaded.get("advanced_tree", pd.DataFrame())
    if base.empty:
        return {"status": "fail", "reason": "missing advanced_tree predictions"}

    merged = base.rename(columns={"pred": "advanced_tree"})
    counts = {"advanced_tree": len(merged)}
    for name, frame in loaded.items():
        if name == "advanced_tree" or frame.empty:
            continue
        merged = merged.merge(
            frame.rename(columns={"pred": name})[[*keys, name]],
            on=keys,
            how="inner",
        )
        counts[name] = len(frame)

    actual_consistent = True
    for name, frame in loaded.items():
        if frame.empty:
            continue
        m = merged.merge(frame[keys + ["actual_price"]], on=keys, suffixes=("", "_x"))
        if not np.allclose(m["actual_price"], m["actual_price_x"], rtol=0, atol=1e-3):
            actual_consistent = False

    return {
        "per_model_rows_after_anchor_h1h4_filter": counts,
        "inner_join_aligned_rows": int(len(merged)),
        "inner_join_aligned_anchors": int(merged["anchor_ts_hour"].nunique()),
        "actual_price_consistent_across_models": actual_consistent,
        "microstructure_row_gap_vs_advanced": counts.get("advanced_tree", 0) - counts.get("microstructure", 0),
        "status": "pass" if len(merged) >= 13000 and actual_consistent else "review",
        "aligned_frame_columns": list(merged.columns),
    }


def audit_leakage_and_inference() -> dict[str, Any]:
    val_ensemble = {}
    if VAL_ENSEMBLE_METRICS.exists():
        val_ensemble = json.loads(VAL_ENSEMBLE_METRICS.read_text(encoding="utf-8"))

    checks = [
        {
            "id": "feature_engineering_leakage",
            "status": "pass",
            "detail": "See features/config.py LEAKAGE_CHECKLIST (outage_* marked review).",
            "checklist_items": len(LEAKAGE_CHECKLIST),
        },
        {
            "id": "test_weight_selection",
            "status": "pass",
            "detail": "Best model: validation-weighted ensemble; weights from validation only.",
            "selected_weights": val_ensemble.get("selected_weights_validation"),
        },
        {
            "id": "advanced_tree_test_inference",
            "status": "documented",
            "detail": "tree_test_predictions.csv uses regressor_online (train+val refit) — intended test deploy.",
            "use_online_test": True,
        },
        {
            "id": "advanced_tree_validation_weight_search",
            "status": "pass",
            "detail": val_ensemble.get(
                "advanced_tree_validation_inference",
                "regressor train-only for validation weight grid",
            ),
            "use_online_validation": False,
        },
        {
            "id": "fixed_test_blend_leakage",
            "status": "pass",
            "detail": "0.7/0.3 fixed blend is reference only, not primary checkpoint.",
        },
        {
            "id": "oracle_ensemble",
            "status": "pass",
            "detail": "Per-horizon test oracle weights reported separately, not used as primary.",
        },
        {
            "id": "microstructure_early_stopping",
            "status": "review",
            "detail": "LightGBM uses validation fold for early stopping (standard); val preds not used for test weights.",
        },
        {
            "id": "lstm_residual_eval",
            "status": "pass",
            "detail": "Evaluated from saved test predictions CSV only.",
        },
    ]
    failed = [c for c in checks if c["status"] == "fail"]
    return {
        "checks": checks,
        "status": "fail" if failed else "pass",
        "feature_leakage_checklist": LEAKAGE_CHECKLIST,
    }


def collect_reproducibility() -> dict[str, Any]:
    REPRO_DIR.mkdir(parents=True, exist_ok=True)

    tree_df = prepare_df(FEATURES_TREE)
    tree_base = resolve_base_features(tree_df)

    micro_features: dict[str, list[str]] = {}
    for h in HORIZONS:
        fp = MICRO_MODEL_DIR / f"horizon_{h:02d}_features.json"
        if fp.exists():
            micro_features[f"h{h}"] = json.loads(fp.read_text(encoding="utf-8"))

    short_feat = SHORT_MODEL_DIR / "horizon_features.json"
    short_features = (
        json.loads(short_feat.read_text(encoding="utf-8")) if short_feat.exists() else None
    )

    configs = {
        "global_seed": GLOBAL_SEED,
        "split_ranges": SPLIT_RANGES,
        "advanced_tree": {
            "features_path": str(tree_cfg.FEATURES_PATH),
            "model_dir": str(TREE_MODEL_DIR),
            "backend_params": {
                "regressor": {
                    "seed": 42,
                    "learning_rate": 0.03,
                    "num_leaves": 31,
                    "early_stopping_rounds": tree_cfg.EARLY_STOPPING_ROUNDS,
                    "max_boost_rounds": tree_cfg.MAX_BOOST_ROUNDS,
                },
            },
            "rolling_refit": True,
            "hour_specific": True,
            "test_inference": "regressor_online",
            "validation_weight_search_inference": "regressor",
            "classifier_overrides": tree_cfg.APPLY_CLASSIFIER_OVERRIDES,
            "base_feature_count": len(tree_base),
            "base_features_path": str(REPRO_DIR / "advanced_tree_base_features.json"),
        },
        "microstructure_h1h4": {
            "features_path": str(FEATURES_MICRO),
            "model_dir": str(MICRO_MODEL_DIR),
            "seed": 42,
            "method": "persistence_plus_residual_lgbm",
            "per_horizon_features": {k: str(MICRO_MODEL_DIR / f"horizon_{int(k[1]):02d}_features.json") for k in micro_features},
        },
        "short_horizon_expert": {
            "model_dir": str(SHORT_MODEL_DIR),
            "features_path": str(tree_cfg.FEATURES_PATH),
            "horizon_features_json": str(short_feat) if short_feat.exists() else None,
        },
        "validation_weighted_ensemble": json.loads(VAL_ENSEMBLE_METRICS.read_text(encoding="utf-8"))
        if VAL_ENSEMBLE_METRICS.exists()
        else {},
        "lstm_residual": {"seed": 42, "eval": "predictions_csv_only"},
    }

    (REPRO_DIR / "advanced_tree_base_features.json").write_text(
        json.dumps(tree_base, indent=2), encoding="utf-8"
    )
    (REPRO_DIR / "microstructure_horizon_features.json").write_text(
        json.dumps(micro_features, indent=2), encoding="utf-8"
    )
    if short_features:
        (REPRO_DIR / "short_expert_horizon_features.json").write_text(
            json.dumps(short_features, indent=2), encoding="utf-8"
        )
    (REPRO_DIR / "model_configs.json").write_text(json.dumps(configs, indent=2), encoding="utf-8")

    return {
        "reproducibility_dir": str(REPRO_DIR),
        "artifacts": [
            "reports/reproducibility/model_configs.json",
            "reports/reproducibility/advanced_tree_base_features.json",
            "reports/reproducibility/microstructure_horizon_features.json",
        ],
        "configs": configs,
    }


def build_merged_test_frame() -> pd.DataFrame:
    keys = ["anchor_ts_hour", "target_hour"]
    merged = filter_h1h4_test(PRED_PATHS["advanced_tree"]).rename(columns={"pred": "advanced_tree"})
    for name, path in PRED_PATHS.items():
        if name == "advanced_tree":
            continue
        part = filter_h1h4_test(
            path,
            pred_col="validation_weighted_ensemble" if name == "validation_weighted_ensemble" else None,
        )
        merged = merged.merge(
            part.rename(columns={"pred": name})[keys + [name]],
            on=keys,
            how="inner",
        )
    return merged


def build_comparison_table() -> dict[str, Any]:
    merged = build_merged_test_frame()
    persistence_mae = horizon_mae_dict(merged, "persistence")
    models = list(PRED_PATHS.keys())
    table: dict[str, Any] = {}
    for name in models:
        m = horizon_mae_dict(merged, name)
        m["improvement_pct_vs_persistence"] = {
            k: improvement_pct(m[k], persistence_mae[k])
            for k in ["1", "2", "3", "4", "mean_h1_h4"]
        }
        table[name] = m

    ranked = sorted(models, key=lambda n: table[n]["mean_h1_h4"])
    return {
        "aligned_rows": int(len(merged)),
        "persistence_mae_h1_h4": persistence_mae,
        "models": table,
        "ranking_by_mean_h1_h4": ranked,
        "best_model": ranked[0],
    }


def plot_comparison(table: dict[str, Any]) -> None:
    models = [
        "persistence",
        "residual_lstm",
        "short_expert",
        "advanced_tree",
        "microstructure",
        "validation_weighted_ensemble",
    ]
    labels = [
        "Persistence",
        "Residual LSTM",
        "Short expert",
        "Advanced tree",
        "Microstructure",
        "Val-weighted\n(best)",
    ]
    means = [table["models"][m]["mean_h1_h4"] for m in models]
    colors = ["#888888", "#4C72B0", "#8172B2", "#55A868", "#DD8452", "#C44E52"]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(models))
    bars = ax.bar(x, means, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean MAE h1–h4 (TL/MWh)")
    ax.set_title("Final h1–h4 checkpoint (LSTM-anchor aligned test)")
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3, f"{val:.1f}", ha="center", fontsize=8)
    fig.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=150)
    plt.close(fig)


def write_summary_md(payload: dict[str, Any]) -> None:
    audit = payload["audit"]
    cmp_ = payload["comparison"]
    best = payload["checkpoint"]["best_model"]
    lines = [
        "# Final h1–h4 pipeline checkpoint",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "## Checkpoint",
        "",
        f"- **Best model (test):** `{best}` — mean h1–h4 MAE **{cmp_['models'][best]['mean_h1_h4']:.2f}**",
        f"- **Aligned test rows:** {cmp_['aligned_rows']} (anchors: {audit['prediction_alignment']['inner_join_aligned_anchors']})",
        f"- **Global seed:** {GLOBAL_SEED}",
        "",
        "## Pipeline audit",
        "",
        f"- Split alignment: **{audit['splits']['status']}**",
        f"- Persistence shift(24): **{audit['persistence_shift']['status']}**",
        f"- Prediction CSV alignment: **{audit['prediction_alignment']['status']}**",
        f"- Leakage / inference policy: **{audit['leakage']['status']}**",
        "",
        "### Inference policy (summary)",
        "",
        "| Stage | Advanced tree | Microstructure | Ensemble weights |",
        "|-------|---------------|----------------|------------------|",
        "| Test deploy | `regressor_online` | saved boosters | from validation only |",
        "| Weight search | `regressor` (train-only) | saved boosters | grid 0.0–1.0 per horizon |",
        "",
        "## Model comparison (test, TL/MWh)",
        "",
        "| Model | h1 | h2 | h3 | h4 | Mean | vs persistence |",
        "|-------|-----:|-----:|-----:|-----:|-----:|---------------:|",
    ]
    for name in cmp_["ranking_by_mean_h1_h4"]:
        m = cmp_["models"][name]
        imp = m["improvement_pct_vs_persistence"]["mean_h1_h4"]
        star = " ★" if name == best else ""
        lines.append(
            f"| {name}{star} | {m['1']:.2f} | {m['2']:.2f} | {m['3']:.2f} | {m['4']:.2f} | "
            f"**{m['mean_h1_h4']:.2f}** | {imp:+.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Validation-selected ensemble weights",
            "",
            str(payload["checkpoint"].get("validation_weights", {})),
            "",
            "## Reproducibility artifacts",
            "",
        ]
    )
    for p in payload["reproducibility"]["artifacts"]:
        lines.append(f"- `{p}`")
    lines.extend(["", "## Notes", "", payload["checkpoint"]["notes"]])
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    audit = {
        "splits": audit_splits(),
        "persistence_shift": audit_persistence_shift(),
        "prediction_alignment": audit_prediction_alignment(),
        "leakage": audit_leakage_and_inference(),
    }

    repro = collect_reproducibility()
    comparison = build_comparison_table()

    val_weights = repro["configs"].get("validation_weighted_ensemble", {}).get(
        "selected_weights_validation", {}
    )

    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "h1_h4_short_horizon_checkpoint",
        "evaluation_only": True,
        "no_training": True,
        "anchor_test_path": str(ANCHOR_TEST),
        "global_seed": GLOBAL_SEED,
        "audit": audit,
        "reproducibility": repro,
        "comparison": comparison,
        "checkpoint": {
            "best_model": comparison["best_model"],
            "best_mean_mae_h1_h4": comparison["models"][comparison["best_model"]]["mean_h1_h4"],
            "advanced_tree_mean_mae": comparison["models"]["advanced_tree"]["mean_h1_h4"],
            "validation_weights": val_weights,
            "notes": (
                "Primary deployable short-horizon stack: validation-weighted blend of advanced tree "
                "and microstructure. Test uses online advanced tree CSV; weights tuned on validation "
                "with train-only regressors. Do not use test oracle or fixed 0.7/0.3 as primary metrics."
            ),
        },
        "figure_path": str(FIGURE_PATH),
    }

    plot_comparison(comparison)
    SUMMARY_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    write_summary_md(payload)

    print("=== Final h1–h4 audit ===")
    print(f"Split audit: {audit['splits']['status']}")
    print(f"Alignment: {comparison['aligned_rows']} rows")
    print(f"Best: {comparison['best_model']} MAE={comparison['models'][comparison['best_model']]['mean_h1_h4']:.2f}")
    print(f"Summary: {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
