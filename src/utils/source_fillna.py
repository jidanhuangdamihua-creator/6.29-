from __future__ import annotations

import pandas as pd


def fill_source_numeric_na(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN values in numeric source columns only, preserving attrs and schema."""
    attrs = df.attrs.copy()
    out = df.copy()

    numeric_cols = out.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        out.loc[:, numeric_cols] = out.loc[:, numeric_cols].fillna(0)

    out.attrs.update(attrs)
    return out
