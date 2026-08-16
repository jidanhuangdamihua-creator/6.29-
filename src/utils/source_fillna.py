from __future__ import annotations

from typing import Sequence

import pandas as pd

from src.protocols.transformation_identity import (
    FILL_POLICY_IDENTITY_ATTR,
    FillPolicyIdentity,
)
from src.utils.dataframe_attrs import copy_frame_with_lightweight_attrs


def _is_numeric_like_model_column(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
        return False
    if pd.api.types.is_numeric_dtype(series):
        return True

    non_null = series.dropna()
    if non_null.empty:
        return False
    converted = pd.to_numeric(non_null, errors="coerce")
    return bool(converted.notna().all())


def fill_source_numeric_na(
    df: pd.DataFrame,
    feature_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Fill source NaNs for numeric model columns, preserving attrs and schema."""
    out = copy_frame_with_lightweight_attrs(df)

    filled_columns: list[str] = []
    coerced_columns: list[str] = []
    if feature_columns is None:
        numeric_cols = out.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 0:
            filled_columns.extend(
                str(column) for column in numeric_cols if bool(out[column].isna().any())
            )
            out.loc[:, numeric_cols] = out.loc[:, numeric_cols].fillna(0)
    else:
        for col in dict.fromkeys(str(col) for col in feature_columns):
            if col not in out.columns:
                continue
            if pd.api.types.is_bool_dtype(out[col]) or isinstance(out[col].dtype, pd.CategoricalDtype):
                continue
            if pd.api.types.is_numeric_dtype(out[col]):
                if bool(out[col].isna().any()):
                    filled_columns.append(col)
                out[col] = out[col].fillna(0)
            elif _is_numeric_like_model_column(out[col]):
                coerced_columns.append(col)
                if bool(out[col].isna().any()):
                    filled_columns.append(col)
                out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    out.attrs[FILL_POLICY_IDENTITY_ATTR] = FillPolicyIdentity(
        schema_version="fill-policy-v1",
        helper_identity="src.utils.source_fillna.fill_source_numeric_na_v1",
        filled_columns=tuple(dict.fromkeys(filled_columns)),
        fill_value_identity="numeric_zero_exact",
        numeric_coercion_identity=(
            "pd.to_numeric_errors_coerce_then_zero:" + ",".join(dict.fromkeys(coerced_columns))
            if coerced_columns
            else "preserve_numeric_dtype"
        ),
    )

    return out
