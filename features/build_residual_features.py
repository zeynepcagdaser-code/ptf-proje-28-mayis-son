"""Build residual-learning tabular dataset (persistence + residual targets)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from features.build_features import build_feature_dataframe
from features.config import INPUT_WINDOW, OUTPUT_HORIZON, REPORTS_DIR
from features.engineering import (
    add_persistence_and_residual_targets,
    list_persistence_columns,
    list_residual_target_columns,
    list_target_columns,
)
from features.report import build_features_report, write_features_report
from features.residual_config import OUTPUT_PATH, RESIDUAL_LEAKAGE_CHECKLIST


def build_residual_feature_dataframe(
    master_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df, metadata = build_feature_dataframe(master_path)
    df = add_persistence_and_residual_targets(df)

    feature_columns = metadata["feature_columns"]
    target_columns = list_target_columns()
    persistence_columns = list_persistence_columns()
    residual_columns = list_residual_target_columns()

    out_columns = (
        ["ts_hour"]
        + feature_columns
        + target_columns
        + persistence_columns
        + residual_columns
        + ["split"]
    )
    result = df[out_columns].copy()

    metadata["persistence_columns"] = persistence_columns
    metadata["residual_target_columns"] = residual_columns
    metadata["target_columns"] = target_columns
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

    df, metadata = build_residual_feature_dataframe(master_path)
    from src.utils.safe_io import atomic_parquet_write
    atomic_parquet_write(df, str(output_path), index=False)

    training_format = {
        "input_window": INPUT_WINDOW,
        "output_horizon": OUTPUT_HORIZON,
        "index_mapping": "X[t-167:t] -> y_residual[t+1:t+24] at anchor ts_hour=t",
        "final_prediction": "persistence_h + inverse_transform(predicted_residual_h)",
        "note": "Tabular rows; sequence tensors in data/model_residual/",
    }

    report = build_features_report(
        df=df,
        feature_columns=metadata["feature_columns"],
        target_columns=metadata["residual_target_columns"],
        rows_master=metadata["rows_master"],
        rows_dropped_targets=metadata["rows_dropped_targets"],
        rows_dropped_history=metadata["rows_dropped_history"],
        missing_before_ffill=metadata["missing_before_ffill"],
        missing_after_ffill=metadata["missing_after_ffill"],
        split_counts=metadata["split_counts"],
        output_path=output_path,
        leakage_checklist=RESIDUAL_LEAKAGE_CHECKLIST,
        training_format=training_format,
    )
    report["price_target_columns"] = metadata["target_columns"]
    report["persistence_columns"] = metadata["persistence_columns"]
    report["residual_target_columns"] = metadata["residual_target_columns"]
    report["dataset_kind"] = "residual_learning"

    json_path, md_path = write_features_report(
        report,
        reports_dir,
        basename="residual_features_report",
    )
    report["report_json"] = str(json_path)
    report["report_md"] = str(md_path)
    return report
