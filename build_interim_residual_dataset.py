#!/usr/bin/env python3
"""Build interim_residual_next24_v1.parquet (interim baseline + final MCP residual targets)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from features.build_tree_features import build_tree_dataframe
from features.config import TARGET_HORIZONS
from features.interim_residual import (
    INTERIM_FEATURE_COLUMNS,
    add_interim_anchor_features,
    add_interim_baselines_and_residuals,
    exclude_from_model_features,
    list_interim_baseline_columns,
    list_interim_residual_columns,
    load_finalized_prices,
    load_interim_prices,
)

PROJECT_ROOT = Path(__file__).resolve().parent
INTERIM_CSV = PROJECT_ROOT / "data" / "raw" / "interim_mcp.csv"
FINAL_CSV = PROJECT_ROOT / "data" / "ptf_dataset.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "features" / "interim_residual_next24_v1.parquet"
REPORT_JSON = PROJECT_ROOT / "reports" / "interim_mcp_audit.json"
REPORT_MD = PROJECT_ROOT / "reports" / "interim_mcp_audit.md"
FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "interim_vs_final.png"


def build_dataset() -> tuple[pd.DataFrame, dict]:
    if not INTERIM_CSV.exists():
        raise FileNotFoundError(f"Run fetch_interim_mcp.py first. Missing {INTERIM_CSV}")

    interim = load_interim_prices(INTERIM_CSV)
    final = load_finalized_prices(FINAL_CSV)

    tree_df, meta = build_tree_dataframe()
    tree_df = tree_df.merge(interim, on="ts_hour", how="left", validate="one_to_one")
    tree_df = tree_df.merge(final, on="ts_hour", how="left", validate="one_to_one")

    missing_interim = int(tree_df["interim_mcp"].isna().sum())
    missing_final = int(tree_df["finalized_mcp"].isna().sum())

    tree_df = add_interim_anchor_features(tree_df)
    tree_df = add_interim_baselines_and_residuals(tree_df)

    baseline_cols = list_interim_baseline_columns()
    residual_cols = list_interim_residual_columns()
    target_cols = [f"target_{h}h" for h in TARGET_HORIZONS]

    required = (
        ["ts_hour", "anchor_hour", "split", "interim_mcp", "finalized_mcp"]
        + meta["feature_columns"]
        + INTERIM_FEATURE_COLUMNS
        + target_cols
        + baseline_cols
        + residual_cols
    )
    required = list(dict.fromkeys(required))
    result = tree_df[required].copy()

    before = len(result)
    result = result.dropna(subset=target_cols + baseline_cols + residual_cols)
    dropped = before - len(result)

    build_meta = {
        **meta,
        "interim_csv": str(INTERIM_CSV),
        "final_csv": str(FINAL_CSV),
        "missing_interim_at_join": missing_interim,
        "missing_final_at_join": missing_final,
        "rows_after_drop_na_targets": int(len(result)),
        "rows_dropped_missing_interim_residual": int(dropped),
        "interim_feature_columns": INTERIM_FEATURE_COLUMNS,
        "model_feature_columns": exclude_from_model_features(result.columns.tolist()),
        "baseline_columns": baseline_cols,
        "residual_target_columns": residual_cols,
    }
    return result, build_meta


def run_audit(df: pd.DataFrame, build_meta: dict) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    aligned = df.dropna(subset=["interim_mcp", "finalized_mcp"]).copy()
    im = aligned["interim_mcp"].to_numpy()
    fin = aligned["finalized_mcp"].to_numpy()
    residual = fin - im

    corr = float(np.corrcoef(im, fin)[0, 1]) if len(im) > 2 else float("nan")
    interim_mae = float(np.mean(np.abs(residual)))

    persistence_mae_by_h: dict[str, float] = {}
    interim_baseline_mae_by_h: dict[str, float] = {}
    for h in TARGET_HORIZONS:
        tcol = f"target_{h}h"
        bcol = f"interim_baseline_{h}h"
        sub = df.dropna(subset=[tcol, bcol])
        act = sub[tcol].to_numpy()
        pers = sub[tcol].shift(24).to_numpy()
        ib = sub[bcol].to_numpy()
        mask = ~np.isnan(pers)
        persistence_mae_by_h[str(h)] = float(np.mean(np.abs(act[mask] - pers[mask])))
        interim_baseline_mae_by_h[str(h)] = float(np.mean(np.abs(act - ib)))

    res_all = []
    for h in TARGET_HORIZONS:
        col = f"target_residual_{h}h"
        res_all.extend(aligned[col].dropna().tolist())
    res_arr = np.array(res_all)

    hourly = (
        aligned.assign(hour=pd.to_datetime(aligned["ts_hour"]).dt.hour)
        .groupby("hour")
        .apply(lambda g: float(np.mean(np.abs(g["finalized_mcp"] - g["interim_mcp"]))))
    )
    hourly_correction = {str(int(k)): float(v) for k, v in hourly.items()}

    by_split = {}
    for split_name, grp in aligned.groupby("split"):
        by_split[split_name] = {
            "interim_vs_final_mae": float(np.mean(np.abs(grp["finalized_mcp"] - grp["interim_mcp"]))),
            "correlation": float(np.corrcoef(grp["interim_mcp"], grp["finalized_mcp"])[0, 1])
            if len(grp) > 2
            else None,
        }

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(im[::50], fin[::50], alpha=0.15, s=8)
    mx = max(im.max(), fin.max())
    axes[0].plot([0, mx], [0, mx], "r--", lw=1)
    axes[0].set_xlabel("Interim MCP (K.PTF)")
    axes[0].set_ylabel("Final MCP")
    axes[0].set_title(f"Interim vs final (r={corr:.3f})")

    axes[1].bar(hourly.index, hourly.values, color="steelblue", edgecolor="black", linewidth=0.4)
    axes[1].set_xlabel("Hour")
    axes[1].set_ylabel("Mean |final − interim|")
    axes[1].set_title("Hourly correction magnitude")
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=150)
    plt.close(fig)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(OUTPUT_PATH),
        "rows": int(len(df)),
        "aligned_price_rows": int(len(aligned)),
        "interim_vs_final_correlation": corr,
        "interim_baseline_mae_same_hour": interim_mae,
        "persistence_baseline_mae_by_horizon": persistence_mae_by_h,
        "interim_baseline_mae_by_horizon": interim_baseline_mae_by_h,
        "mean_persistence_mae_h1_h24": float(np.mean(list(persistence_mae_by_h.values()))),
        "mean_interim_baseline_mae_h1_h24": float(np.mean(list(interim_baseline_mae_by_h.values()))),
        "residual_distribution": {
            "mean": float(res_arr.mean()),
            "std": float(res_arr.std()),
            "p05": float(np.percentile(res_arr, 5)),
            "p50": float(np.percentile(res_arr, 50)),
            "p95": float(np.percentile(res_arr, 95)),
            "min": float(res_arr.min()),
            "max": float(res_arr.max()),
        },
        "hourly_mean_abs_correction": hourly_correction,
        "by_split": by_split,
        "build_metadata": build_meta,
        "figure_path": str(FIGURE_PATH),
    }


def write_audit_md(report: dict) -> None:
    rd = report["residual_distribution"]
    lines = [
        "# Interim MCP (K.PTF) audit",
        "",
        f"Generated: {report['generated_at_utc']}",
        "",
        f"- Dataset: `{report['dataset_path']}`",
        f"- Rows: {report['rows']}",
        "",
        "## Interim vs finalized MCP",
        "",
        f"- Pearson correlation: **{report['interim_vs_final_correlation']:.4f}**",
        f"- Same-hour MAE |final − interim|: **{report['interim_baseline_mae_same_hour']:.2f}** TL/MWh",
        "",
        "## Baseline comparison (mean abs error by horizon)",
        "",
        "| Horizon | Persistence (final lag-24) | Interim baseline |",
        "|--------:|---------------------------:|-----------------:|",
    ]
    for h in TARGET_HORIZONS:
        hs = str(h)
        lines.append(
            f"| h{h} | {report['persistence_baseline_mae_by_horizon'][hs]:.2f} | "
            f"{report['interim_baseline_mae_by_horizon'][hs]:.2f} |"
        )
    lines.extend(
        [
            f"| **Mean h1–h24** | **{report['mean_persistence_mae_h1_h24']:.2f}** | "
            f"**{report['mean_interim_baseline_mae_h1_h24']:.2f}** |",
            "",
            "## Residual target (final − interim at delivery hour)",
            "",
            f"- Mean: {rd['mean']:.2f}",
            f"- Std: **{rd['std']:.2f}**",
            f"- p05 / p50 / p95: {rd['p05']:.2f} / {rd['p50']:.2f} / {rd['p95']:.2f}",
            "",
            f"Figure: `{report['figure_path']}`",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    df, build_meta = build_dataset()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    from src.utils.safe_io import atomic_parquet_write
    atomic_parquet_write(df, str(OUTPUT_PATH), index=False)
    print(f"Wrote {OUTPUT_PATH} ({len(df)} rows)")

    report = run_audit(df, build_meta)
    REPORT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    write_audit_md(report)
    print(f"Audit: {REPORT_JSON}")
    print(f"Mean |final-interim|: {report['interim_baseline_mae_same_hour']:.2f}")
    print(f"Residual std: {report['residual_distribution']['std']:.2f}")


if __name__ == "__main__":
    main()
