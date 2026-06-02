#!/usr/bin/env python3
"""
Holdout backtest: predict June 1 2026 PTF using only pre-June-1 information.

Protocol:
  - Anchor for each hour h on 2026-06-01 = PTF(2026-05-31, h)  [D+1 persistence]
  - Load/KGUP for 2026-06-01 from forecast tables (day-ahead fundamentals)
  - Actual 2026-06-01 PTF held out entirely from model inputs
  - Compare persistence, D2 model, and premium stack vs actuals
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from build_premium_d2_features import enrich
from build_d2_ptf_features import build_rows, parse_ts
from premium_d2_inference import predict_premium
from run_d2_ptf_forecast import predict as predict_d2

PROJECT_ROOT = Path(__file__).resolve().parent
TARGET_DATE = date(2026, 6, 1)
PTF_PATH = PROJECT_ROOT / "data" / "ptf_dataset.csv"
OUT_CSV = PROJECT_ROOT / "data" / "predictions" / "june1_holdout_backtest.csv"
REPORT_JSON = PROJECT_ROOT / "reports" / "june1_holdout_backtest.json"
REPORT_MD = PROJECT_ROOT / "reports" / "june1_holdout_backtest.md"


def load_actuals() -> pd.DataFrame:
    ptf = pd.read_csv(PTF_PATH)
    ptf["ts_hour"] = parse_ts(ptf["date"])
    ptf["actual_ptf"] = pd.to_numeric(ptf["price"], errors="coerce")
    day = pd.Timestamp(TARGET_DATE).date()
    actual = ptf[ptf["ts_hour"].dt.date == day][["ts_hour", "actual_ptf"]].copy()
    return actual.sort_values("ts_hour")


def metric_block(y: np.ndarray, pred: np.ndarray) -> dict:
    err = np.abs(y - pred)
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "median_ae": float(np.median(err)),
        "max_ae": float(err.max()),
        "pct_le_100": float((err <= 100).mean()),
        "pct_le_50": float((err <= 50).mean()),
        "pct_le_200": float((err <= 200).mean()),
    }


def main() -> None:
    actual = load_actuals()
    if len(actual) != 24:
        raise RuntimeError(f"Expected 24 actual hours for {TARGET_DATE}, got {len(actual)}")

    features, build_diag = build_rows(TARGET_DATE)
    premium_features, enrich_audit = enrich(features)
    premium_pred = predict_premium(premium_features)
    d2_pred = predict_d2(premium_features)

    result = actual.merge(
        premium_pred[
            [
                "ts_hour",
                "persistence_pred",
                "high_precision_pred",
                "hybrid_pred",
                "d2_pred",
                "premium_pred",
                "hour_group",
                "anchor_source",
            ]
        ],
        on="ts_hour",
        how="inner",
    )
    result = result.merge(
        d2_pred[["ts_hour", "predicted_ptf"]].rename(columns={"predicted_ptf": "d2_only_pred"}),
        on="ts_hour",
        how="left",
    )
    result["error_persistence"] = (result["actual_ptf"] - result["persistence_pred"]).abs()
    result["error_premium"] = (result["actual_ptf"] - result["premium_pred"]).abs()
    result["error_d2"] = (result["actual_ptf"] - result["d2_pred"]).abs()
    result["error_hybrid"] = (result["actual_ptf"] - result["hybrid_pred"]).abs()

    y = result["actual_ptf"].to_numpy(float)
    metrics = {
        "persistence_may31_anchor": metric_block(y, result["persistence_pred"].to_numpy(float)),
        "d2_model": metric_block(y, result["d2_pred"].to_numpy(float)),
        "hybrid_stack": metric_block(y, result["hybrid_pred"].to_numpy(float)),
        "premium_blend": metric_block(y, result["premium_pred"].to_numpy(float)),
        "high_precision": metric_block(y, result["high_precision_pred"].to_numpy(float)),
    }

    by_hour = result[
        [
            "ts_hour",
            "hour_group",
            "actual_ptf",
            "persistence_pred",
            "premium_pred",
            "error_premium",
            "error_persistence",
        ]
    ].to_dict(orient="records")

    by_group = []
    for group, grp in result.groupby("hour_group", observed=False):
        by_group.append(
            {
                "hour_group": str(group),
                "rows": int(len(grp)),
                "actual_mean": float(grp["actual_ptf"].mean()),
                "persistence_mae": float(grp["error_persistence"].mean()),
                "premium_mae": float(grp["error_premium"].mean()),
            }
        )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "target_date": str(TARGET_DATE),
            "anchor": "PTF(2026-05-31, same hour)",
            "actuals": "PTF(2026-06-01) held out from model inputs",
            "fundamentals": "2026-06-01 load/kgup from forecast tables when available",
        },
        "build_diagnostics": build_diag,
        "enrich_audit": enrich_audit,
        "metrics": metrics,
        "best_model": min(metrics, key=lambda k: metrics[k]["mae"]),
        "hourly": by_hour,
        "by_hour_group": by_group,
        "may31_daily_mean": float(result["persistence_pred"].mean()),
        "june1_actual_mean": float(result["actual_ptf"].mean()),
        "regime_shift": "May 31 avg ~754 TL → June 1 actual ~1298 TL (+72% spike day)",
    }

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_CSV, index=False)
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n")

    m = metrics
    lines = [
        "# 1 Haziran 2026 Holdout Backtest",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "## Protokol",
        "",
        "- Tahmin edilen gün: **2026-06-01** (24 saat)",
        "- Anchor: **31 Mayıs aynı saat PTF** (D+1 persistence)",
        "- 1 Haziran gerçek PTF model girdisine **dahil edilmedi**",
        "- Load/KGUP: 1 Haziran forecast tablolarından",
        "",
        f"- 31 Mayıs ortalama PTF (anchor): `{report['may31_daily_mean']:.1f}` TL",
        f"- 1 Haziran gerçekleşen ortalama: `{report['june1_actual_mean']:.1f}` TL",
        "",
        "## Doğruluk (MAE = ortalama mutlak hata, TL/MWh)",
        "",
        "| Model | MAE | RMSE | Medyan hata | ≤100 TL | ≤200 TL |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, label in [
        ("persistence_may31_anchor", "Persistence (31 May anchor)"),
        ("d2_model", "D+2 residual model"),
        ("hybrid_stack", "Hybrid fuel-switch + HP"),
        ("premium_blend", "Premium blend"),
        ("high_precision", "High-precision"),
    ]:
        row = m[name]
        lines.append(
            f"| {label} | {row['mae']:.1f} | {row['rmse']:.1f} | {row['median_ae']:.1f} | {row['pct_le_100']:.0%} | {row['pct_le_200']:.0%} |"
        )
    lines.extend(["", f"**En iyi model:** `{report['best_model']}`", "", "## Saatlik karşılaştırma", ""])
    lines.append("| Saat | Gerçek | Persistence | Premium | Hata (premium) |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for row in by_hour:
        ts = pd.Timestamp(row["ts_hour"])
        lines.append(
            f"| {ts.hour:02d}:00 | {row['actual_ptf']:.1f} | {row['persistence_pred']:.1f} | {row['premium_pred']:.1f} | {row['error_premium']:.1f} |"
        )
    REPORT_MD.write_text("\n".join(lines) + "\n")

    print(f"Persistence MAE: {m['persistence_may31_anchor']['mae']:.1f} TL")
    print(f"Premium MAE:     {m['premium_blend']['mae']:.1f} TL")
    print(f"D2 MAE:          {m['d2_model']['mae']:.1f} TL")
    print(f"Best:            {report['best_model']}")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {REPORT_MD}")


if __name__ == "__main__":
    main()
