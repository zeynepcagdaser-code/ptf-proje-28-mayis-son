from typing import Optional, Any
from pathlib import Path

import pandas as pd

from .tz_utils import normalize_to_ts_hour


def read_parquet_with_normalized_ts(path: Any, ts_col: str = "ts_hour", columns: Optional[list[str]] = None, **kwargs) -> pd.DataFrame:
    """Read a parquet (or path-like) and normalize `ts_col` to naive hourly Europe/Istanbul if present.

    `path` may be a Path, string, or object accepted by `pd.read_parquet`. Additional kwargs
    are forwarded to `pd.read_parquet`.
    """
    df = pd.read_parquet(path, columns=columns, **kwargs)
    if ts_col in df.columns:
        df = normalize_to_ts_hour(df, col=ts_col, out=ts_col)
    return df
