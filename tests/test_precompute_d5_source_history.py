from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.precompute_d5_source_history import (
    _atomic_write_parquet,
    build_output_manifest,
    validate_precomputed_output,
)


def test_manifest_records_frame_schema_and_self_check(tmp_path: Path) -> None:
    output_path = tmp_path / "prepared_180day_source.parquet"
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2017-01-01", periods=2, freq="D"),
            "store_nbr": pd.Series([48, 48], dtype="int64"),
            "item_nbr": pd.Series([1159415, 1159415], dtype="int64"),
            "sales": pd.Series([1.0, 2.0], dtype="float64"),
        }
    )
    frame.to_parquet(tmp_path / "source.parquet", index=False)
    (tmp_path / "oil.csv").write_text("date,dcoilwtico\n2017-01-01,1\n", encoding="utf-8")
    frame.to_parquet(output_path, index=False)

    manifest = build_output_manifest(
        source_path=tmp_path / "source.parquet",
        auxiliary_files={"oil": tmp_path / "oil.csv"},
        output_path=output_path,
        frame=frame,
        source_history_start=pd.Timestamp("2017-01-01"),
        source_history_end=pd.Timestamp("2017-01-02"),
        source_history_days=2,
        key_fields=("store_nbr", "item_nbr"),
        generated_at="2026-07-26T00:00:00Z",
        source_history_frame_digest="digest",
        synthetic_row_count=0,
    )

    assert manifest["artifact"]["row_count"] == 2
    assert manifest["artifact"]["schema"] == [
        {"name": "date", "dtype": "datetime64[ns]"},
        {"name": "store_nbr", "dtype": "int64"},
        {"name": "item_nbr", "dtype": "int64"},
        {"name": "sales", "dtype": "float64"},
    ]
    validate_precomputed_output(output_path, manifest)


def test_self_check_rejects_manifest_row_count_mismatch(tmp_path: Path) -> None:
    output_path = tmp_path / "prepared.parquet"
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2017-01-01", periods=2, freq="D"),
            "store_nbr": [48, 48],
            "item_nbr": [1159415, 1159415],
        }
    )
    frame.to_parquet(tmp_path / "source.parquet", index=False)
    frame.to_parquet(output_path, index=False)
    manifest = build_output_manifest(
        source_path=tmp_path / "source.parquet",
        auxiliary_files={},
        output_path=output_path,
        frame=frame,
        source_history_start=pd.Timestamp("2017-01-01"),
        source_history_end=pd.Timestamp("2017-01-02"),
        source_history_days=2,
        key_fields=("store_nbr", "item_nbr"),
        generated_at="2026-07-26T00:00:00Z",
        source_history_frame_digest="digest",
        synthetic_row_count=0,
    )
    manifest["artifact"]["row_count"] = 3

    with pytest.raises(ValueError, match="row count mismatch"):
        validate_precomputed_output(output_path, manifest)


def test_atomic_parquet_write_drops_runtime_attrs(tmp_path: Path) -> None:
    output_path = tmp_path / "prepared.parquet"
    frame = pd.DataFrame({"date": pd.date_range("2017-01-01", periods=1)})
    frame.attrs["source_history_start"] = pd.Timestamp("2017-01-01")

    _atomic_write_parquet(frame, output_path)

    restored = pd.read_parquet(output_path)
    assert restored.equals(frame)
