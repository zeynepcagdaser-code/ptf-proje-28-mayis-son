"""Leakage-safe MinMax scalers (train fit only)."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def fit_feature_scaler(train_features: pd.DataFrame) -> MinMaxScaler:
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_features)
    return scaler


def fit_target_scaler(train_targets: pd.DataFrame) -> MinMaxScaler:
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_targets)
    return scaler


def transform_features(df: pd.DataFrame, scaler: MinMaxScaler) -> pd.DataFrame:
    return pd.DataFrame(
        scaler.transform(df),
        columns=df.columns,
        index=df.index,
    )


def transform_targets(df: pd.DataFrame, scaler: MinMaxScaler) -> pd.DataFrame:
    return pd.DataFrame(
        scaler.transform(df),
        columns=df.columns,
        index=df.index,
    )


def save_scaler(scaler: MinMaxScaler, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)


def load_scaler(path: Path) -> MinMaxScaler:
    return joblib.load(path)
