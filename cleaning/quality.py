"""Quality flags only — no imputation, no price modification."""

from __future__ import annotations

import pandas as pd

from cleaning.config import PRICE_CAP_TL


def add_price_flags(df: pd.DataFrame, price_columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in price_columns:
        if col not in out.columns:
            continue
        out[f"is_{col}_zero"] = out[col].eq(0)
        out[f"is_{col}_capped"] = out[col].eq(PRICE_CAP_TL)
    return out


def clip_with_flag(
    df: pd.DataFrame,
    columns: list[str],
    lower: float = 0.0,
) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        flag_col = f"was_{col}_clipped"
        out[flag_col] = out[col] < lower
        out[col] = out[col].clip(lower=lower)
    return out
