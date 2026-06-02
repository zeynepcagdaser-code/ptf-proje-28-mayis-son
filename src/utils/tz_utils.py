from zoneinfo import ZoneInfo
import pandas as pd
from typing import Optional


def normalize_to_ts_hour(df: pd.DataFrame, col: str = "date", out: str = "ts_hour", tz: str = "Europe/Istanbul") -> pd.DataFrame:
    """Normalize a datetime column to naive hourly `ts_hour` in the given timezone.

    - If values are naive, they are localized to `tz`.
    - If values are timezone-aware, they are converted to `tz`.
    Returns the same DataFrame with a new column named `out`.
    """
    if col not in df.columns:
        raise KeyError(col)

    s = pd.to_datetime(df[col], errors="coerce")
    if s.dt.tz is None:
        s = s.dt.tz_localize(ZoneInfo(tz))
    else:
        s = s.dt.tz_convert(ZoneInfo(tz))

    df[out] = s.dt.tz_convert(ZoneInfo(tz)).dt.tz_localize(None).dt.floor("h")
    return df


def is_series_tz_aware(s: pd.Series) -> Optional[bool]:
    """Return True if the series has a timezone-aware dtype, False if naive, None if unknown."""
    try:
        return s.dt.tz is not None
    except Exception:
        return None
