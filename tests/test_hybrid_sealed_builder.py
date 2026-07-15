from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.adopt_and_seal_d3_d6 import adopt_and_seal_dataset
from scripts.regenerate_d1_d2_parquets import build_and_seal_dataset
from src.data_processing.sealed_daily import (
    calendarize_and_fill,
    canonicalize_source_sales,
)


def _d2_long_frame(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [date for date in dates for brand in range(1, 4) for item in range(1, 11)],
            "brand": [brand for _date in dates for brand in range(1, 4) for _item in range(1, 11)],
            "item": [item for _date in dates for _brand in range(1, 4) for item in range(1, 11)],
            "sales": [
                float(brand + item)
                for _date in dates
                for brand in range(1, 4)
                for item in range(1, 11)
            ],
            "promo": 0,
        }
    )


def test_calendarize_and_fill_adds_only_calendar_rows_and_preserves_unused_nulls() -> None:
    frame = pd.DataFrame(
        {
            "entity": ["A", "A"],
            "date": ["2024-01-01", "2024-01-03"],
            "sales": [3.0, 5.0],
            "unused_audit": [None, None],
        }
    )

    result = calendarize_and_fill(
        frame,
        group_cols=("entity",),
        start="2024-01-01",
        end="2024-01-03",
        fill_rules={"sales": "zero"},
    )

    assert result["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-01",
        "2024-01-02",
        "2024-01-03",
    ]
    assert result["sales"].tolist() == [3.0, 0.0, 5.0]
    assert result["unused_audit"].isna().all()
    assert result.attrs["fill_policy_engine_version"] == "calendarize-fill/v1"
    assert result.attrs["synthetic_date_count"] == 1


def test_source_sales_canonicalization_records_closed_repair_reasons() -> None:
    frame = pd.DataFrame(
        {
            "entity": ["A", "A", "A", "A"],
            "date": pd.date_range("2024-01-01", periods=4, freq="D"),
            "sales": [np.nan, -2.0, np.inf, np.nan],
        }
    )

    with pytest.raises(ValueError, match="infinity"):
        canonicalize_source_sales(frame, calendar_row_missing=[False, False, False, True])

    frame.loc[2, "sales"] = 4.0
    repaired, audit = canonicalize_source_sales(
        frame,
        calendar_row_missing=[False, False, False, True],
    )

    assert repaired["sales"].tolist() == [0.0, 0.0, 4.0, 0.0]
    assert audit["repair_reason_counts"] == {
        "original_nan": 1,
        "original_negative": 1,
        "calendar_row_missing": 1,
    }
    assert len(audit["repair_mask_sha256"]) == 64


def test_d2_raw_rebuild_publishes_approved_june_calendar_row(tmp_path: Path) -> None:
    dates = pd.date_range("2018-05-30", "2018-06-30", freq="D").difference(
        pd.DatetimeIndex([pd.Timestamp("2018-06-02")])
    )
    output = tmp_path / "d1_d6_sealed_v1"

    dataset_dir = build_and_seal_dataset(
        2,
        _d2_long_frame(dates),
        output_dir=output,
        raw_input_path=tmp_path / "d2-raw.csv",
    )

    assert dataset_dir == output / "dataset2"
    assert (dataset_dir / "source.parquet").is_file()
    assert (dataset_dir / "target.parquet").is_file()
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["provenance_level"] == "raw_rebuilt"
    assert manifest["fill_policy_engine_version"] == "calendarize-fill/v1"
    assert manifest["dataset_canonicalization"]["rule_id"] == "d2_june_absent_transaction_day_v1"

    target = pd.read_parquet(dataset_dir / "target.parquet")
    june_2 = target[target["date"] == pd.Timestamp("2018-06-02")]
    assert len(june_2) == 1
    assert june_2.iloc[0]["sales"] == 0.0
    assert june_2.iloc[0]["promo"] == 0


def test_adoption_records_parent_identity_and_structural_only_disclosure(tmp_path: Path) -> None:
    source = tmp_path / "dataset3-source.parquet"
    target = tmp_path / "dataset3-target.parquet"
    source_frame = pd.DataFrame(
        {
            "entity_id": ["1", "1"],
            "store_id": [1, 1],
            "item_id": [1, 1],
            "date": pd.date_range("2015-01-01", periods=2, freq="D"),
            "sales": [1.0, 2.0],
        }
    )
    target_frame = source_frame.assign(entity_id="10", store_id=10)
    source_frame.to_parquet(source, index=False)
    target_frame.to_parquet(target, index=False)

    dataset_dir = adopt_and_seal_dataset(
        3,
        source_path=source,
        target_path=target,
        output_dir=tmp_path / "sealed",
    )

    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["provenance_level"] == "adopted_solidified"
    assert manifest["content_validation_level"] == "structural_only"
    assert manifest["adopted_content_validated"] is False
    assert manifest["parent_artifacts"]["source"]["sha256"]
    assert manifest["parent_artifacts"]["source"]["size_bytes"] == source.stat().st_size
    assert manifest["parent_artifacts"]["source"]["first_seen_at"] is None
    assert manifest["parent_artifacts"]["source"]["first_seen_reliability"] == "unavailable"


def test_adoption_failure_does_not_publish_dataset_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        adopt_and_seal_dataset(
            3,
            source_path=tmp_path / "missing-source.parquet",
            target_path=tmp_path / "missing-target.parquet",
            output_dir=tmp_path / "sealed",
        )

    assert not (tmp_path / "sealed" / "dataset3").exists()
