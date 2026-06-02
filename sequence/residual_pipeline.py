"""Build scaled numpy sequence datasets for residual LSTM training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sequence.config import INPUT_WINDOW, OUTPUT_HORIZON
from sequence.framing import build_split_sequences
from sequence.report import build_sequence_report, write_sequence_report
from sequence.residual_config import (
    ANCHOR_FILE_MAP,
    EXCLUDE_COLUMNS,
    FEATURE_COLUMNS_FILE,
    FEATURE_SCALER_FILE,
    FEATURES_PATH,
    LEAKAGE_CHECKLIST,
    METADATA_FILE,
    MODEL_DATA_DIR,
    PERSISTENCE_PREFIX,
    PRICE_TARGET_PREFIX,
    REPORTS_DIR,
    SPLIT_ORDER,
    TARGET_COLUMNS_FILE,
    TARGET_PREFIX,
    TARGET_SCALER_FILE,
)
from sequence.scaler import (
    fit_feature_scaler,
    fit_target_scaler,
    save_scaler,
    transform_features,
    transform_targets,
)


def _resolve_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    target_columns = sorted(
        [c for c in df.columns if c.startswith(TARGET_PREFIX)],
        key=lambda name: int(name.replace(TARGET_PREFIX, "").replace("h", "")),
    )
    exclude = set(EXCLUDE_COLUMNS)
    exclude.update(c for c in df.columns if c.startswith(PRICE_TARGET_PREFIX) and not c.startswith(TARGET_PREFIX))
    exclude.update(c for c in df.columns if c.startswith(PERSISTENCE_PREFIX))
    exclude.update(target_columns)

    feature_columns = [c for c in df.columns if c not in exclude]
    return feature_columns, target_columns


def _save_anchor_csv(
    anchor_ts_hours: list,
    split_name: str,
    output_dir: Path,
) -> Path:
    anchor_df = pd.DataFrame(
        {
            "sample_index": range(len(anchor_ts_hours)),
            "anchor_ts_hour": anchor_ts_hours,
            "split": split_name,
        }
    )
    path = output_dir / ANCHOR_FILE_MAP[split_name]
    anchor_df.to_csv(path, index=False)
    return path


def _tabular_nan_report(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
) -> dict[str, int]:
    return {
        "feature_nan_rows": int(df[feature_columns].isna().any(axis=1).sum()),
        "target_nan_rows": int(df[target_columns].isna().any(axis=1).sum()),
    }


def run_pipeline(
    *,
    features_path: Path | None = None,
    output_dir: Path | None = None,
    reports_dir: Path | None = None,
) -> dict[str, Any]:
    features_path = features_path or FEATURES_PATH
    output_dir = output_dir or MODEL_DATA_DIR
    reports_dir = reports_dir or REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    from src.utils.io_utils import read_parquet_with_normalized_ts
    df = read_parquet_with_normalized_ts(features_path).sort_values("ts_hour").reset_index(drop=True)
    feature_columns, target_columns = _resolve_columns(df)

    tabular_nan = _tabular_nan_report(df, feature_columns, target_columns)

    partitions: dict[str, pd.DataFrame] = {}
    for split in SPLIT_ORDER:
        part = df[df["split"] == split].copy()
        part = part.dropna(subset=feature_columns + target_columns)
        partitions[split] = part.sort_values("ts_hour").reset_index(drop=True)

    train_features = partitions["train"][feature_columns]
    train_targets = partitions["train"][target_columns]

    feature_scaler = fit_feature_scaler(train_features)
    target_scaler = fit_target_scaler(train_targets)

    scaled_partitions: dict[str, pd.DataFrame] = {}
    for split, part in partitions.items():
        scaled = part[["ts_hour", "split"]].copy()
        scaled[feature_columns] = transform_features(part[feature_columns], feature_scaler)
        scaled[target_columns] = transform_targets(part[target_columns], target_scaler)
        scaled_partitions[split] = scaled

    arrays: dict[str, dict[str, np.ndarray]] = {}
    sequence_counts: dict[str, int] = {}
    dropped_nan: dict[str, int] = {}
    dropped_insufficient: dict[str, int] = {}
    shapes: dict[str, dict[str, list[int]]] = {}
    anchor_paths: dict[str, str] = {}

    for split in SPLIT_ORDER:
        result = build_split_sequences(
            scaled_partitions[split],
            feature_columns,
            target_columns,
            window=INPUT_WINDOW,
        )
        arrays[split] = {"X": result.X, "y": result.y}
        sequence_counts[split] = int(result.X.shape[0])
        dropped_nan[split] = result.dropped_nan
        dropped_insufficient[split] = result.dropped_insufficient_rows
        shapes[split] = {
            "X": list(result.X.shape),
            "y": list(result.y.shape),
        }
        anchor_path = _save_anchor_csv(
            result.anchor_ts_hours,
            split,
            output_dir,
        )
        anchor_paths[split] = str(anchor_path)

    file_map = {
        "train": ("X_train", "y_train"),
        "validation": ("X_val", "y_val"),
        "test": ("X_test", "y_test"),
    }
    for split, (x_file, y_file) in file_map.items():
        np.save(output_dir / f"{x_file}.npy", arrays[split]["X"])
        np.save(output_dir / f"{y_file}.npy", arrays[split]["y"])

    save_scaler(feature_scaler, output_dir / FEATURE_SCALER_FILE)
    save_scaler(target_scaler, output_dir / TARGET_SCALER_FILE)

    (output_dir / FEATURE_COLUMNS_FILE).write_text(
        json.dumps(feature_columns, indent=2),
        encoding="utf-8",
    )
    (output_dir / TARGET_COLUMNS_FILE).write_text(
        json.dumps(target_columns, indent=2),
        encoding="utf-8",
    )

    metadata = {
        "dataset_kind": "residual_learning",
        "source_features": str(features_path),
        "input_window": INPUT_WINDOW,
        "output_horizon": OUTPUT_HORIZON,
        "feature_count": len(feature_columns),
        "target_count": len(target_columns),
        "target_kind": "residual",
        "index_mapping": f"X[t-{INPUT_WINDOW - 1}:t] -> y_residual[t+1:t+{OUTPUT_HORIZON}]",
        "final_prediction": "persistence_h + inverse_transform(predicted_residual_h)",
        "shapes": shapes,
        "sequence_counts": sequence_counts,
        "dropped_nan_sequences": dropped_nan,
        "tabular_nan_rows": tabular_nan,
        "scaler_fit_split": "train",
        "scaler_type": "MinMaxScaler(feature_range=(0, 1))",
        "split_order": SPLIT_ORDER,
        "anchor_files": anchor_paths,
    }
    (output_dir / METADATA_FILE).write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )

    report = build_sequence_report(
        feature_columns=feature_columns,
        target_columns=target_columns,
        window=INPUT_WINDOW,
        horizon=OUTPUT_HORIZON,
        shapes=shapes,
        sequence_counts=sequence_counts,
        dropped_nan=dropped_nan,
        dropped_insufficient=dropped_insufficient,
        scaler_fit_split="train",
        leakage_checklist=LEAKAGE_CHECKLIST,
        output_dir=output_dir,
        source_path=features_path,
    )
    report["dataset_kind"] = "residual_learning"
    report["tabular_nan_rows"] = tabular_nan
    report["metadata_file"] = str(output_dir / METADATA_FILE)
    report["anchor_files"] = anchor_paths

    json_path, md_path = write_sequence_report(
        report,
        reports_dir,
        basename="residual_sequence_report",
    )
    report["report_json"] = str(json_path)
    report["report_md"] = str(md_path)

    return report
