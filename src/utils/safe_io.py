import os
from typing import Any


def atomic_parquet_write(df: Any, path: str, index: bool = False, **to_parquet_kwargs) -> None:
    """Write a DataFrame to Parquet atomically by writing a temp file then renaming.

    Keeps API minimal to avoid depending on calling code changes.
    """
    tmp_path = path + ".tmp"
    # ensure target dir exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # write to temp file then atomically replace
    df.to_parquet(tmp_path, index=index, **to_parquet_kwargs)
    os.replace(tmp_path, path)
