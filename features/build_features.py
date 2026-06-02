"""Build leakage-safe tabular feature dataset for next-24h PTF."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from features.config import (
    FFILL_LIMIT,
    INPUT_WINDOW,
    LEAKAGE_CHECKLIST,
    MASTER_PATH,
    OUTPUT_HORIZON,
    OUTPUT_PATH,
    REPORTS_DIR,
)
from features.engineering import (
    add_cap_and_ratio_features,
    add_calendar_features,
    add_fiba_fibs_features,
    add_grf_features,
    add_holiday_features,
    add_lagged_realized_features,
    add_ptf_lag_features,
    add_ptf_low_regime_history_features,
    add_spread_lag_features,
    add_ptf_downside_risk_features,
    add_supply_demand_features,
    add_targets,
    assign_split,
    list_engineered_feature_columns,
    list_target_columns,
)
from features.report import build_features_report, missing_pct, write_features_report


def _prepare_master(master_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(master_path)
    df = df.sort_values("ts_hour").reset_index(drop=True)
    if df["ts_hour"].duplicated().any():
        raise ValueError("master ts_hour must be unique")
    return df


def build_feature_dataframe(master_path: Path | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    master_path = master_path or MASTER_PATH
    rows_master = len(pd.read_parquet(master_path, columns=["ts_hour"]))

    df = _prepare_master(master_path)

    df = add_targets(df)
    df = add_ptf_lag_features(df)
    df = add_ptf_low_regime_history_features(df)
    df = add_calendar_features(df)
    df = add_holiday_features(df)
    df = add_spread_lag_features(df)
    df = add_supply_demand_features(df)
    df = add_ptf_downside_risk_features(df)
    df = add_fiba_fibs_features(df)
    df = add_grf_features(df)
    df = add_lagged_realized_features(df)
    df = add_cap_and_ratio_features(df)

    feature_columns = list_engineered_feature_columns()
    target_columns = list_target_columns()

    missing_cols = [c for c in feature_columns if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected feature columns: {missing_cols}")

    # Drop rows without full target horizon (last 24 hours of series).
    before_targets = len(df)
    df = df.dropna(subset=target_columns)
    rows_dropped_targets = before_targets - len(df)

    missing_before_ffill = missing_pct(df, feature_columns)

    # Past-only ffill on features (no bfill, no interpolation).
    df[feature_columns] = df[feature_columns].ffill(limit=FFILL_LIMIT)

    missing_after_ffill = missing_pct(df, feature_columns)

    # Require enough history for 168h input window and lag/roll features.
    history_required = [
        "ptf_lag_168",
        "ptf_roll_mean_168",
        "ptf_roll_std_168",
    ]
    before_history = len(df)
    df = df.dropna(subset=history_required)
    rows_dropped_history = before_history - len(df)

    df["split"] = assign_split(df["ts_hour"])

    out_columns = ["ts_hour"] + feature_columns + target_columns + ["split"]
    result = df[out_columns].copy()

    split_counts = {
        str(k): int(v)
        for k, v in result["split"].value_counts(dropna=False).items()
    }

    metadata = {
        "rows_master": rows_master,
        "rows_dropped_targets": rows_dropped_targets,
        "rows_dropped_history": rows_dropped_history,
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "missing_before_ffill": missing_before_ffill,
        "missing_after_ffill": missing_after_ffill,
        "split_counts": split_counts,
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

    df, metadata = build_feature_dataframe(master_path)
    df.to_parquet(output_path, index=False)

    training_format = {
        "input_window": INPUT_WINDOW,
        "output_horizon": OUTPUT_HORIZON,
        "index_mapping": "X[t-167:t] -> y[t+1:t+24] at anchor ts_hour=t",
        "note": "Tabular rows only; sequence tensors not materialized in this artifact",
    }

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
        leakage_checklist=LEAKAGE_CHECKLIST,
        training_format=training_format,
    )

    json_path, md_path = write_features_report(report, reports_dir)
    report["report_json"] = str(json_path)
    report["report_md"] = str(md_path)
    return report
