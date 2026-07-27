from __future__ import annotations

import pandas as pd


_REPAIR_DATES = (
    "2018-08-15",
    "2018-11-01",
    "2018-12-08",
    "2018-12-25",
    "2018-12-26",
)


def _target_frame(*, include_repair_dates: bool = False) -> pd.DataFrame:
    dates = pd.date_range("2014-01-02", "2018-12-31", freq="D")
    if not include_repair_dates:
        dates = dates.difference(
            pd.DatetimeIndex(
                [*pd.date_range("2014-02-01", periods=18, freq="D"), *_REPAIR_DATES]
            )
        )
    frame = pd.DataFrame(
        {
            "date": dates,
            "brand_id": pd.Series(1, index=range(len(dates)), dtype="int64"),
            "item_id": pd.Series(10, index=range(len(dates)), dtype="int64"),
            "sales": pd.Series(2.0, index=range(len(dates)), dtype="float64"),
            "promo": pd.Series(1.0, index=range(len(dates)), dtype="float64"),
            "entity_id": pd.Series(1, index=range(len(dates)), dtype="object"),
            "year": pd.Series(dates.year.to_numpy(), index=range(len(dates)), dtype="float64"),
            "month": pd.Series(dates.month.to_numpy(), index=range(len(dates)), dtype="float64"),
            "week": pd.Series(dates.isocalendar().week.to_numpy(), index=range(len(dates)), dtype="float64"),
            "day": pd.Series(dates.day.to_numpy(), index=range(len(dates)), dtype="float64"),
        }
    )
    frame.attrs["split_role"] = "target"
    return frame


def test_d2_target_producer_adds_only_authorized_zero_demand_rows() -> None:
    from src.protocols.d2_target_calendarization import calendarize_d2_target_frame

    original = _target_frame()
    repaired, evidence = calendarize_d2_target_frame(original)

    assert len(repaired) == len(original) + 5
    inserted = repaired[repaired["date"].isin(pd.to_datetime(_REPAIR_DATES))]
    assert len(inserted) == 5
    assert inserted[["brand_id", "item_id"]].drop_duplicates().to_records(index=False).tolist() == [(1, 10)]
    assert inserted["sales"].tolist() == [0.0] * 5
    assert inserted["promo"].tolist() == [0.0] * 5
    assert inserted["entity_id"].tolist() == [1] * 5
    assert inserted["year"].tolist() == [2018.0] * 5
    assert inserted["month"].tolist() == [8.0, 11.0, 12.0, 12.0, 12.0]
    assert inserted["day"].tolist() == [15.0, 1.0, 8.0, 25.0, 26.0]
    assert evidence["inserted_dates"] == list(_REPAIR_DATES)
    assert evidence["inserted_count"] == 5
    assert list(repaired.columns) == list(original.columns)
    assert repaired.dtypes.astype(str).to_dict() == original.dtypes.astype(str).to_dict()


def test_d2_target_producer_preserves_original_rows_and_is_idempotent() -> None:
    from src.protocols.d2_target_calendarization import (
        calendarize_d2_target_frame,
        target_semantic_digest,
    )

    original = _target_frame()
    repaired, first_evidence = calendarize_d2_target_frame(original)
    rerun, second_evidence = calendarize_d2_target_frame(repaired)

    assert target_semantic_digest(repaired) == target_semantic_digest(rerun)
    assert len(rerun) == 1807
    assert second_evidence["inserted_count"] == 0
    assert second_evidence["inserted_dates"] == []
    original_keys = set(zip(original["date"], original["brand_id"], original["item_id"]))
    rerun_original = rerun[
        rerun.apply(
            lambda row: (row["date"], row["brand_id"], row["item_id"]) in original_keys,
            axis=1,
        )
    ]
    assert target_semantic_digest(rerun_original) == target_semantic_digest(original)
    assert first_evidence["policy"] == "closed_day_zero_demand"
