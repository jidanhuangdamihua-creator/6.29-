from __future__ import annotations

import pandas as pd
import pytest

from src.utils.sealed_parquet import SealedParquetProjectionError, read_sealed_projection


def test_sealed_projection_preserves_requested_order_and_date_pushdown(tmp_path) -> None:
    path = tmp_path / "target.parquet"
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "sales": [1.0, 2.0, 3.0, 4.0, 5.0],
            "unused": [10, 11, 12, 13, 14],
        }
    ).to_parquet(path, index=False)

    frame = read_sealed_projection(
        path,
        columns=("sales", "date"),
        date_start="2024-01-02",
        date_end="2024-01-04",
    )

    assert list(frame.columns) == ["sales", "date"]
    assert frame["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
    ]


def test_sealed_projection_fails_closed_on_unknown_or_missing_columns(tmp_path) -> None:
    path = tmp_path / "target.parquet"
    pd.DataFrame({"date": pd.date_range("2024-01-01", periods=2), "sales": [1.0, 2.0]}).to_parquet(
        path, index=False
    )

    with pytest.raises(SealedParquetProjectionError, match="unknown"):
        read_sealed_projection(path, columns=("date", "truth"))
    with pytest.raises(SealedParquetProjectionError, match="missing"):
        read_sealed_projection(path, columns=("date", "y_true"))
