"""Build tree-ready feature set: base LSTM features + microstructure."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from features.build_features import _prepare_master
from features.config import FFILL_LIMIT, MASTER_PATH, REPORTS_DIR
from features.engineering import (
    add_persistence_and_residual_targets,
    assign_split,
    list_engineered_feature_columns,
    list_target_columns,
)
from features.microstructure import add_microstructure_features, list_microstructure_columns
from features.report import build_features_report, missing_pct, write_features_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "features" / "lstm_tree_micro_v1.parquet"


def build_tree_dataframe(master_path: Path | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    master_path = master_path or MASTER_PATH
    rows_master = len(pd.read_parquet(master_path, columns=["ts_hour"]))

    df = _prepare_master(master_path)
    from features.engineering import (
        add_calendar_features,
        add_holiday_features,
        add_lagged_realized_features,
        add_ptf_lag_features,
        add_spread_lag_features,
        add_supply_demand_features,
        add_targets,
    )

    df = add_targets(df)
    df = add_ptf_lag_features(df)
    df = add_calendar_features(df)
    df = add_holiday_features(df)
    df = add_spread_lag_features(df)
    df = add_supply_demand_features(df)
    df = add_lagged_realized_features(df)
    df = add_microstructure_features(df)

    base_features = list_engineered_feature_columns()
    micro_features = list_microstructure_columns()
    feature_columns = base_features + [c for c in micro_features if c in df.columns]
    target_columns = list_target_columns()

    before_targets = len(df)
    df = df.dropna(subset=target_columns)
    rows_dropped_targets = before_targets - len(df)

    missing_before_ffill = missing_pct(df, feature_columns)
    df[feature_columns] = df[feature_columns].ffill(limit=FFILL_LIMIT)
    missing_after_ffill = missing_pct(df, feature_columns)

    history_required = ["ptf_lag_168", "ptf_roll_mean_168", "ptf_roll_std_168"]
    before_history = len(df)
    df = df.dropna(subset=history_required)
    rows_dropped_history = before_history - len(df)

    df = add_persistence_and_residual_targets(df)
    df["split"] = assign_split(df["ts_hour"])
    ts = pd.to_datetime(df["ts_hour"], utc=True).dt.tz_convert("Europe/Istanbul")
    df["anchor_hour"] = ts.dt.hour.astype(int)

    persistence_cols = [f"persistence_{h}h" for h in range(1, 25)]
    residual_cols = [f"target_residual_{h}h" for h in range(1, 25)]

    out_columns = (
        ["ts_hour", "anchor_hour"]
        + feature_columns
        + target_columns
        + persistence_cols
        + residual_cols
        + ["split"]
    )
    result = df[out_columns].copy()

    metadata = {
        "rows_master": rows_master,
        "rows_dropped_targets": rows_dropped_targets,
        "rows_dropped_history": rows_dropped_history,
        "feature_columns": feature_columns,
        "base_feature_count": len(base_features),
        "microstructure_feature_count": len([c for c in micro_features if c in df.columns]),
        "target_columns": target_columns,
        "missing_before_ffill": missing_before_ffill,
        "missing_after_ffill": missing_after_ffill,
        "split_counts": {str(k): int(v) for k, v in result["split"].value_counts().items()},
    }
    return result, metadata


def run_build(
    *,
    master_path: Path | None = None,
    output_path: Path | None = None,
    reports_dir: Path | None = None,
) -> dict[str, Any]:
    output_path = output_path or OUTPUT_PATH
    reports_dir = reports_dir or REPORTS_DIR
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df, metadata = build_tree_dataframe(master_path)
    df.to_parquet(output_path, index=False)

    report = build_features_report(
        df=df,
        feature_columns=metadata["feature_columns"],
        target_columns=metadata["target_columns"],
        rows_master=metadata["rows_master"],
        rows_dropped_targets=metadata["rows_dropped_targets"],
        rows_dropped_history=metadata["rows_dropped_history"],
        missing_before_ffill=metadata["missing_before_ffill"],
        missing_after_ffill=metadata["missing_after_ffill"],
        split_counts=metadata["split_counts"],
        output_path=output_path,
        leakage_checklist=[],
        training_format={"dataset_kind": "tree_microstructure"},
    )
    report["microstructure_feature_count"] = metadata["microstructure_feature_count"]
    json_path, md_path = write_features_report(report, reports_dir, basename="tree_features_report")
    report["report_json"] = str(json_path)
    report["report_md"] = str(md_path)
    report["output_path"] = str(output_path)
    return report
