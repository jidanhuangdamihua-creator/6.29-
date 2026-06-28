import numpy as np
import pandas as pd

from src.utils.source_fillna import fill_source_numeric_na


def test_numeric_nan_filled_only_for_numeric_columns():
    df = pd.DataFrame(
        {
            "sales": [1.0, np.nan, 2.0],
            "oil_price": [np.nan, 10.0, 11.0],
            "tag": pd.Series([None, "x", "y"], dtype="string"),
            "group": pd.Series([None, "a", "b"], dtype="category"),
            "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
        }
    )

    result = fill_source_numeric_na(df)

    assert result["sales"].isna().sum() == 0
    assert result["oil_price"].isna().sum() == 0
    assert result["sales"].iloc[1] == 0.0
    assert result["oil_price"].iloc[0] == 0.0

    assert pd.isna(result["tag"].iloc[0])
    assert pd.isna(result["group"].iloc[0])
    assert result["date"].tolist() == df["date"].tolist()


def test_schema_and_attrs_preserved():
    df = pd.DataFrame(
        {
            "sales": [1.0, np.nan],
            "tag": [None, "x"],
        }
    )
    df.attrs["split_role"] = "source"
    original_columns = list(df.columns)
    original_shape = df.shape
    original_dtypes = df.dtypes.copy()

    result = fill_source_numeric_na(df)

    assert list(result.columns) == original_columns
    assert result.shape == original_shape
    assert result.dtypes.equals(original_dtypes)
    assert result.attrs.get("split_role") == "source"


def test_original_dataframe_not_mutated():
    df = pd.DataFrame(
        {
            "sales": [1.0, np.nan],
            "tag": [None, "x"],
        }
    )

    result = fill_source_numeric_na(df)

    assert df["sales"].isna().sum() == 1
    assert result["sales"].isna().sum() == 0
    assert df is not result
