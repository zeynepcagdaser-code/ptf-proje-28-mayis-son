"""Recency sample weights for rolling / online training."""

from __future__ import annotations

import numpy as np
import pandas as pd


def recency_weights(
    ts_hour: pd.Series,
    *,
    reference_time: pd.Timestamp,
    recent_days: int = 60,
    max_boost: float = 3.0,
) -> np.ndarray:
    """
    Weight 1.0 for old rows; linear ramp up to max_boost for last `recent_days`.
    """
    ts = pd.to_datetime(ts_hour, utc=True)
    if reference_time.tzinfo is None:
        reference_time = reference_time.tz_localize("UTC")
    else:
        reference_time = reference_time.tz_convert("UTC")

    age_days = (reference_time - ts).dt.total_seconds().to_numpy() / 86400.0
    w = np.ones(len(ts), dtype=np.float64)
    mask = (age_days >= 0) & (age_days <= recent_days)
    if mask.any():
        w[mask] = 1.0 + (max_boost - 1.0) * (1.0 - age_days[mask] / recent_days)
    return w


def blend_recency_weights(
    ts_hour: pd.Series,
    *,
    reference_time: pd.Timestamp,
    recent_days: int,
    medium_days: int,
    max_boost: float,
) -> np.ndarray:
    """Heavier tail: 30–90 day window gets partial boost, last `recent_days` full boost."""
    ts = pd.to_datetime(ts_hour, utc=True)
    ref = reference_time.tz_convert("UTC") if reference_time.tzinfo else reference_time.tz_localize("UTC")
    age = (ref - ts).dt.total_seconds().to_numpy() / 86400.0
    w = np.ones(len(ts), dtype=np.float64)
    recent_mask = (age >= 0) & (age <= recent_days)
    medium_mask = (age > recent_days) & (age <= medium_days)
    if recent_mask.any():
        w[recent_mask] = 1.0 + (max_boost - 1.0) * (1.0 - age[recent_mask] / recent_days)
    if medium_mask.any():
        mid_boost = 1.0 + (max_boost - 1.0) * 0.35
        w[medium_mask] = mid_boost
    return w
