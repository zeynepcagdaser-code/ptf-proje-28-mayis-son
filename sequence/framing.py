"""Build (N, window, features) and (N, horizon) arrays per split."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sequence.config import INPUT_WINDOW, OUTPUT_HORIZON


@dataclass
class FramingResult:
    X: np.ndarray
    y: np.ndarray
    anchor_ts_hours: list
    dropped_nan: int
    dropped_insufficient_rows: int


def build_split_sequences(
    df_split: pd.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
    *,
    window: int = INPUT_WINDOW,
) -> FramingResult:
    """
    Build sequences within a single split partition.

    For anchor index j (0-based within split):
      X = features[j-window+1 : j+1]   # 168 rows ending at t
      y = targets[j]                   # target_1h..target_24h at anchor t
    """
    df_split = df_split.sort_values("ts_hour").reset_index(drop=True)
    n = len(df_split)

    if n < window:
        return FramingResult(
            X=np.empty((0, window, len(feature_columns)), dtype=np.float64),
            y=np.empty((0, len(target_columns)), dtype=np.float64),
            anchor_ts_hours=[],
            dropped_nan=0,
            dropped_insufficient_rows=n,
        )

    features = df_split[feature_columns].to_numpy(dtype=np.float64)
    targets = df_split[target_columns].to_numpy(dtype=np.float64)
    ts_hours = df_split["ts_hour"].tolist()

    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    anchors: list = []
    dropped_nan = 0

    for j in range(window - 1, n):
        x_window = features[j - window + 1 : j + 1]
        y_row = targets[j]

        if np.isnan(x_window).any() or np.isnan(y_row).any():
            dropped_nan += 1
            continue

        X_list.append(x_window)
        y_list.append(y_row)
        anchors.append(ts_hours[j])

    if not X_list:
        return FramingResult(
            X=np.empty((0, window, len(feature_columns)), dtype=np.float64),
            y=np.empty((0, len(target_columns)), dtype=np.float64),
            anchor_ts_hours=[],
            dropped_nan=dropped_nan,
            dropped_insufficient_rows=window - 1,
        )

    return FramingResult(
        X=np.stack(X_list, axis=0),
        y=np.stack(y_list, axis=0),
        anchor_ts_hours=anchors,
        dropped_nan=dropped_nan,
        dropped_insufficient_rows=window - 1,
    )
