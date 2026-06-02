#!/usr/bin/env python3
"""
Build tabular low-price classifier datasets from model_low_price sequence tensors.

Reads scaled X/y .npy files, inverse-transforms y for TL/MWh labels, aggregates
per-sequence window statistics, and writes parquet artifacts under data/model_low_price_tabular/.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = Path(__file__).resolve().parent
SEQUENCE_DIR = PROJECT_ROOT / "data" / "model_low_price"
OUTPUT_DIR = PROJECT_ROOT / "data" / "model_low_price_tabular"
REPORTS_DIR = PROJECT_ROOT / "reports"

METADATA_FILE = "sequence_metadata.json"
FEATURE_COLUMNS_FILE = "feature_columns.json"
TARGET_SCALER_FILE = "target_scaler.pkl"
ANCHOR_FILES = {
    "train": "anchor_train.csv",
    "validation": "anchor_val.csv",
    "test": "anchor_test.csv",
}
SPLIT_FILE_PREFIX = {
    "train": "train",
    "validation": "val",
    "test": "test",
}

LOW_PRICE_THRESHOLD_TL = 50.0
HORIZON = 24
WINDOW_24 = 24
WINDOW_168 = 168


def _load_metadata(sequence_dir: Path) -> dict[str, Any]:
    path = sequence_dir / METADATA_FILE
    return json.loads(path.read_text(encoding="utf-8"))


def _load_feature_names(sequence_dir: Path, metadata: dict[str, Any]) -> list[str]:
    path = sequence_dir / FEATURE_COLUMNS_FILE
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return list(metadata["feature_columns"])


def _load_target_scaler(sequence_dir: Path) -> MinMaxScaler:
    return joblib.load(sequence_dir / TARGET_SCALER_FILE)


def _inverse_targets(y_scaled: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    return scaler.inverse_transform(y_scaled).astype(np.float64)


def _horizon_column_names(prefix: str) -> list[str]:
    return [f"{prefix}_{h}h" for h in range(1, HORIZON + 1)]


def _build_label_frames(
    y_scaled: np.ndarray,
    scaler: MinMaxScaler,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_tl = _inverse_targets(y_scaled, scaler)
    low_cols = _horizon_column_names("is_low")
    zero_cols = _horizon_column_names("is_zero")
    y_low = pd.DataFrame(
        (y_tl <= LOW_PRICE_THRESHOLD_TL).astype(np.int8),
        columns=low_cols,
    )
    y_zero = pd.DataFrame(
        (y_tl == 0.0).astype(np.int8),
        columns=zero_cols,
    )
    return y_low, y_zero


def _aggregate_window(
    window: np.ndarray,
    suffix: str,
    feature_names: list[str],
) -> dict[str, np.ndarray]:
    """window shape (n_samples, window_len, n_features)."""
    out: dict[str, np.ndarray] = {}
    for stat_name, reducer in (
        ("mean", np.mean),
        ("std", np.std),
        ("min", np.min),
        ("max", np.max),
    ):
        values = reducer(window, axis=1)
        for idx, name in enumerate(feature_names):
            out[f"{name}_{stat_name}_{suffix}"] = values[:, idx]
    return out


def _sequences_to_tabular_features(
    X: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    if X.ndim != 3:
        raise ValueError(f"Expected X.ndim==3, got {X.shape}")
    n_samples, window_len, n_features = X.shape
    if n_features != len(feature_names):
        raise ValueError(
            f"Feature dim mismatch: X has {n_features}, names have {len(feature_names)}"
        )
    if window_len < WINDOW_168:
        raise ValueError(f"Window length {window_len} < required {WINDOW_168}")

    last = X[:, -1, :]
    win_24 = X[:, -WINDOW_24:, :]
    win_168 = X[:, -WINDOW_168:, :]
    mean_24 = np.mean(win_24, axis=1)

    columns: dict[str, np.ndarray] = {}
    for idx, name in enumerate(feature_names):
        columns[f"{name}_last"] = last[:, idx]
        columns[f"{name}_trend_last_minus_mean_24h"] = last[:, idx] - mean_24[:, idx]

    columns.update(_aggregate_window(win_24, "24h", feature_names))
    columns.update(_aggregate_window(win_168, "168h", feature_names))

    return pd.DataFrame(columns)


def _load_split_arrays(
    sequence_dir: Path,
    split: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    file_key = SPLIT_FILE_PREFIX[split]
    X = np.load(sequence_dir / f"X_{file_key}.npy")
    y = np.load(sequence_dir / f"y_{file_key}.npy")
    anchor = pd.read_csv(sequence_dir / ANCHOR_FILES[split])
    if len(anchor) != len(X):
        raise ValueError(
            f"{split}: anchor rows ({len(anchor)}) != X rows ({len(X)})"
        )
    return X, y, anchor


def _class_rates(df: pd.DataFrame, prefix: str) -> dict[str, float]:
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        return {}
    values = df[cols].to_numpy()
    return {
        "any_horizon": float((values.any(axis=1)).mean()),
        "mean_over_horizons": float(values.mean()),
        "per_horizon": {
            col: float(df[col].mean()) for col in cols
        },
    }


def _attach_anchor_features(
    features: pd.DataFrame,
    anchor: pd.DataFrame,
    split: str,
) -> pd.DataFrame:
    out = features.copy()
    out.insert(0, "split", split)
    out.insert(0, "anchor_ts_hour", pd.to_datetime(anchor["anchor_ts_hour"], utc=True))
    out.insert(0, "sample_index", anchor["sample_index"].astype(int))
    return out


def build_low_price_tabular_dataset(
    *,
    sequence_dir: Path | None = None,
    output_dir: Path | None = None,
    reports_dir: Path | None = None,
) -> dict[str, Any]:
    sequence_dir = sequence_dir or SEQUENCE_DIR
    output_dir = output_dir or OUTPUT_DIR
    reports_dir = reports_dir or REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    metadata = _load_metadata(sequence_dir)
    feature_names = _load_feature_names(sequence_dir, metadata)
    target_scaler = _load_target_scaler(sequence_dir)

    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sequence_dir": str(sequence_dir),
        "output_dir": str(output_dir),
        "feature_profile": metadata.get("feature_profile", "low_price_classifier"),
        "base_feature_count": len(feature_names),
        "base_feature_columns": feature_names,
        "low_price_threshold_tl": LOW_PRICE_THRESHOLD_TL,
        "aggregation": {
            "last_timestep": True,
            "window_24h": ["mean", "std", "min", "max"],
            "window_168h": ["mean", "std", "min", "max"],
            "trend": "last_minus_mean_24h",
        },
        "splits": {},
    }

    for split in ("train", "validation", "test"):
        X, y, anchor = _load_split_arrays(sequence_dir, split)
        X_tab = _sequences_to_tabular_features(X, feature_names)
        X_tab = _attach_anchor_features(X_tab, anchor, split)
        y_low, y_zero = _build_label_frames(y, target_scaler)

        file_key = SPLIT_FILE_PREFIX[split]
        x_path = output_dir / f"X_{file_key}.parquet"
        X_tab.to_parquet(x_path, index=False)

        y_low_path = output_dir / f"y_low_{file_key}.parquet"
        y_low_out = y_low.copy()
        y_low_out.insert(0, "sample_index", anchor["sample_index"].astype(int))
        y_low_out.to_parquet(y_low_path, index=False)

        split_report: dict[str, Any] = {
            "X_path": str(x_path),
            "y_low_path": str(y_low_path),
            "n_rows": int(len(X_tab)),
            "X_shape": list(X.shape),
            "tabular_feature_count": int(X_tab.shape[1] - 3),  # minus index cols
            "y_low_shape": list(y_low.shape),
            "low_class_rate": _class_rates(y_low, "is_low"),
            "zero_class_rate": _class_rates(y_zero, "is_zero"),
        }

        if split == "test":
            y_zero_path = output_dir / "y_zero_test.parquet"
            y_zero_out = y_zero.copy()
            y_zero_out.insert(0, "sample_index", anchor["sample_index"].astype(int))
            y_zero_out.to_parquet(y_zero_path, index=False)
            split_report["y_zero_path"] = str(y_zero_path)
            split_report["y_zero_shape"] = list(y_zero.shape)

        report["splits"][split] = split_report

    report["tabular_feature_count"] = report["splits"]["train"]["tabular_feature_count"]
    expected = len(feature_names) * (1 + 4 + 4 + 1)
    report["expected_tabular_feature_count"] = expected

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = reports_dir / f"low_price_tabular_dataset_report_{stamp}.json"
    md_path = reports_dir / f"low_price_tabular_dataset_report_{stamp}.md"
    latest_json = reports_dir / "low_price_tabular_dataset_report_latest.json"
    latest_md = reports_dir / "low_price_tabular_dataset_report_latest.md"

    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    md_path.write_text(_report_to_markdown(report), encoding="utf-8")
    latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_md.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")

    report["report_json"] = str(json_path)
    report["report_md"] = str(md_path)
    return report


def _report_to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Low-Price Tabular Dataset Report",
        "",
        f"- **Generated (UTC):** {report['generated_at_utc']}",
        f"- **Sequence dir:** `{report['sequence_dir']}`",
        f"- **Output dir:** `{report['output_dir']}`",
        f"- **Base features:** {report['base_feature_count']} "
        f"→ **tabular features:** {report['tabular_feature_count']} "
        f"(expected {report['expected_tabular_feature_count']})",
        f"- **Low-price threshold:** {report['low_price_threshold_tl']} TL/MWh",
        "",
        "## Splits",
        "",
    ]
    for split, blob in report["splits"].items():
        lines.append(f"### {split}")
        lines.append("")
        lines.append(f"- Rows: {blob['n_rows']}")
        lines.append(f"- Sequence X shape: `{blob['X_shape']}`")
        lines.append(f"- Tabular feature count: {blob['tabular_feature_count']}")
        lines.append(f"- y_low shape: `{blob['y_low_shape']}`")
        lines.append(
            f"- Low class rate (any horizon): {blob['low_class_rate']['any_horizon']:.4f}"
        )
        lines.append(
            f"- Low class rate (mean over horizons): "
            f"{blob['low_class_rate']['mean_over_horizons']:.4f}"
        )
        lines.append(
            f"- Zero class rate (any horizon): {blob['zero_class_rate']['any_horizon']:.4f}"
        )
        lines.append(
            f"- Zero class rate (mean over horizons): "
            f"{blob['zero_class_rate']['mean_over_horizons']:.4f}"
        )
        lines.append(f"- X: `{blob['X_path']}`")
        lines.append(f"- y_low: `{blob['y_low_path']}`")
        if "y_zero_path" in blob:
            lines.append(f"- y_zero (test only): `{blob['y_zero_path']}`")
        lines.append("")
    return "\n".join(lines) + "\n"


def _print_summary(report: dict[str, Any]) -> None:
    train = report["splits"]["train"]
    val = report["splits"]["validation"]
    test = report["splits"]["test"]
    print("=== Low-Price Tabular Dataset Summary ===")
    print("Output dir:", report["output_dir"])
    print("X_train shape (sequence):", tuple(train["X_shape"]))
    print("X_train tabular rows x features:", train["n_rows"], "x", train["tabular_feature_count"])
    print("y_low_train shape:", tuple(train["y_low_shape"]))
    print("Tabular feature count:", report["tabular_feature_count"])
    print()
    print("Low class rate (any horizon <=50 TL):")
    print(f"  train: {train['low_class_rate']['any_horizon']:.4f}")
    print(f"  val:   {val['low_class_rate']['any_horizon']:.4f}")
    print(f"  test:  {test['low_class_rate']['any_horizon']:.4f}")
    print("Low class rate (mean over 24 horizons):")
    print(f"  train: {train['low_class_rate']['mean_over_horizons']:.4f}")
    print(f"  val:   {val['low_class_rate']['mean_over_horizons']:.4f}")
    print(f"  test:  {test['low_class_rate']['mean_over_horizons']:.4f}")
    print()
    print("Zero class rate (any horizon == 0 TL):")
    print(f"  train: {train['zero_class_rate']['any_horizon']:.4f}")
    print(f"  val:   {val['zero_class_rate']['any_horizon']:.4f}")
    print(f"  test:  {test['zero_class_rate']['any_horizon']:.4f}")
    print("Zero class rate (mean over 24 horizons):")
    print(f"  train: {train['zero_class_rate']['mean_over_horizons']:.4f}")
    print(f"  val:   {val['zero_class_rate']['mean_over_horizons']:.4f}")
    print(f"  test:  {test['zero_class_rate']['mean_over_horizons']:.4f}")
    print()
    print("Report JSON:", report["report_json"])
    print("Report MD:", report["report_md"])


def main() -> None:
    if not SEQUENCE_DIR.exists():
        raise FileNotFoundError(
            f"Sequence directory not found: {SEQUENCE_DIR}. "
            "Run: python3 run_sequence.py --feature-profile low_price_classifier"
        )
    report = build_low_price_tabular_dataset()
    _print_summary(report)


if __name__ == "__main__":
    main()
