#!/usr/bin/env python3
"""
Build microstructure feature dataset from lstm_next24_v1 + master (leakage-safe).

No model training; feature engineering and validation reports only.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from features.microstructure_v2 import NEW_FEATURE_NAMES, build_microstructure_columns

PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_FEATURES = PROJECT_ROOT / "data" / "features" / "lstm_next24_v1.parquet"
MASTER_PATH = PROJECT_ROOT / "data" / "master" / "master_hourly_v1.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "features" / "lstm_microstructure_next24_v1.parquet"
REPORT_JSON = PROJECT_ROOT / "reports" / "microstructure_feature_report.json"
REPORT_MD = PROJECT_ROOT / "reports" / "microstructure_feature_report.md"
CORR_FIG = PROJECT_ROOT / "reports" / "figures" / "microstructure_correlation.png"

VOLATILITY_FEATURES = [
    "smf_volatility_24h",
    "ptf_volatility_24h",
    "ptf_change_1h",
    "ptf_change_3h",
    "ptf_change_24h",
]


def null_report(df: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    n = len(df)
    return {
        c: round(float(df[c].isna().sum() / n * 100), 4) if n else 0.0 for c in cols
    }


def feature_summary(df: pd.DataFrame, cols: list[str]) -> list[dict]:
    rows = []
    for c in cols:
        s = df[c]
        rows.append(
            {
                "feature": c,
                "count_non_null": int(s.notna().sum()),
                "mean": float(s.mean()) if s.notna().any() else None,
                "std": float(s.std()) if s.notna().any() else None,
                "min": float(s.min()) if s.notna().any() else None,
                "p50": float(s.median()) if s.notna().any() else None,
                "max": float(s.max()) if s.notna().any() else None,
            }
        )
    return rows


def top_volatility_features(df: pd.DataFrame, cols: list[str], top_n: int = 10) -> list[dict]:
    """Rank by std (scale-free volatility proxy)."""
    ranked = []
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c].dropna()
        if len(s) < 10:
            continue
        ranked.append({"feature": c, "std": float(s.std()), "mean_abs": float(s.abs().mean())})
    ranked.sort(key=lambda x: -x["std"])
    return ranked[:top_n]


def correlation_subset(df: pd.DataFrame, cols: list[str], max_cols: int = 24) -> pd.DataFrame:
    use = [c for c in cols if c in df.columns][:max_cols]
    return df[use].corr(method="pearson", min_periods=500)


def plot_correlation(corr: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(corr.columns)
    figsize = (max(10, n * 0.45), max(8, n * 0.4))
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.columns, fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Microstructure features — Pearson correlation")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def write_markdown(report: dict) -> str:
    lines = [
        "# Microstructure Feature Report",
        "",
        f"- **Generated (UTC):** {report['generated_at_utc']}",
        f"- **Input:** `{report['input_path']}`",
        f"- **Master (for raw lags):** `{report['master_path']}`",
        f"- **Output:** `{report['output_path']}`",
        f"- **Rows:** {report['row_count']}",
        f"- **Microstructure spec features:** {len(NEW_FEATURE_NAMES)}",
        f"- **Newly added columns:** {len(report.get('columns_added', []))}",
        f"- **Already in input (kept as-is):** {report.get('columns_already_in_input', [])}",
        "",
        "## Leakage rules",
        "",
        "- Realized SMF/PTF/YAL-YAT: only via `shift(1)` or longer lags",
        "- No interpolation / bfill",
        "- `target_*` and `split` preserved from input",
        "- KGÜP/load/wind forecast ramps use plan values; realized balancing not at same hour",
        "",
        "## Null % (new features)",
        "",
        "| Feature | Null % |",
        "|---------|-------:|",
    ]
    for feat, pct in sorted(report["null_pct"].items(), key=lambda x: x[1], reverse=True):
        lines.append(f"| `{feat}` | {pct:.2f} |")

    lines.extend(
        [
            "",
            "## Top volatility-related features (by std)",
            "",
            "| Rank | Feature | Std | Mean |abs| |",
            "|-----:|---------|----:|-----------:|",
        ]
    )
    for i, row in enumerate(report["top_volatility_features"], 1):
        lines.append(
            f"| {i} | `{row['feature']}` | {row['std']:.4f} | {row.get('mean_abs', 0):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Feature summary (new columns)",
            "",
            "| Feature | Mean | Std | Min | Median | Max |",
            "|---------|-----:|----:|----:|-------:|----:|",
        ]
    )
    for row in report["feature_summary"]:
        lines.append(
            f"| `{row['feature']}` | {row['mean']:.2f} | {row['std']:.2f} | "
            f"{row['min']:.2f} | {row['p50']:.2f} | {row['max']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## High |correlation| pairs (|r| > 0.85, new features only)",
            "",
        ]
    )
    pairs = report.get("high_correlation_pairs", [])
    if pairs:
        for p in pairs[:20]:
            lines.append(f"- `{p['a']}` ↔ `{p['b']}`: {p['r']:.3f}")
    else:
        lines.append("_None above threshold._")

    lines.extend(["", f"Correlation figure: `{report['correlation_figure']}`"])
    return "\n".join(lines)


def high_corr_pairs(corr: pd.DataFrame, threshold: float = 0.85) -> list[dict]:
    pairs = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            r = corr.loc[a, b]
            if pd.notna(r) and abs(r) >= threshold:
                pairs.append({"a": a, "b": b, "r": float(r)})
    pairs.sort(key=lambda x: -abs(x["r"]))
    return pairs


def run() -> dict:
    if not INPUT_FEATURES.exists():
        raise FileNotFoundError(f"Missing {INPUT_FEATURES}")
    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"Missing {MASTER_PATH}")

    base = pd.read_parquet(INPUT_FEATURES)
    master = pd.read_parquet(MASTER_PATH).sort_values("ts_hour").reset_index(drop=True)

    micro_full = build_microstructure_columns(master)
    micro_full["ts_hour"] = master["ts_hour"]

    already_present = [c for c in NEW_FEATURE_NAMES if c in base.columns]
    cols_to_add = [c for c in NEW_FEATURE_NAMES if c not in base.columns]
    micro_add = micro_full[["ts_hour"] + cols_to_add]

    merged = base.merge(micro_add, on="ts_hour", how="left", validate="one_to_one")
    if len(merged) != len(base):
        raise ValueError(f"Row count mismatch after merge: {len(merged)} vs {len(base)}")

    missing_new = [c for c in NEW_FEATURE_NAMES if c not in merged.columns]
    if missing_new:
        raise ValueError(f"Failed to create features: {missing_new}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUTPUT_PATH, index=False)

    new_cols = NEW_FEATURE_NAMES
    null_pct = null_report(merged, new_cols)
    summary = feature_summary(merged, new_cols)
    top_vol = top_volatility_features(merged, VOLATILITY_FEATURES + new_cols, top_n=10)
    corr = correlation_subset(merged, new_cols)
    plot_correlation(corr, CORR_FIG)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(INPUT_FEATURES),
        "master_path": str(MASTER_PATH),
        "output_path": str(OUTPUT_PATH),
        "row_count": len(merged),
        "new_feature_count": len(new_cols),
        "new_feature_names": new_cols,
        "columns_added": cols_to_add,
        "columns_already_in_input": already_present,
        "null_pct": null_pct,
        "feature_summary": summary,
        "top_volatility_features": top_vol,
        "high_correlation_pairs": high_corr_pairs(corr),
        "correlation_figure": str(CORR_FIG),
        "split_counts": {
            str(k): int(v) for k, v in merged["split"].value_counts().items()
        },
        "target_columns_preserved": [c for c in merged.columns if c.startswith("target_")],
    }

    REPORT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    REPORT_MD.write_text(write_markdown(report), encoding="utf-8")

    print("Microstructure features built.")
    print("Output:", OUTPUT_PATH)
    print("Rows:", report["row_count"], "| New features:", report["new_feature_count"])
    print("Report:", REPORT_JSON)
    return report


if __name__ == "__main__":
    run()
