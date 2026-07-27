from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.protocols.d2_source_calendarization import (
    D2_SOURCE_MISSING_DATES,
    repair_d2_source_calendar_fields,
    slice_d2_source_frame,
    verify_d2_source_frame,
)
from src.protocols.experiment_protocol import ProtocolViolation


def _source_frame() -> pd.DataFrame:
    dates = pd.date_range("2018-01-02", "2018-06-30", freq="D")
    frame = pd.DataFrame(
        {
            "date": dates,
            "brand_id": 1,
            "item_id": 4,
            "entity_id": "B1",
            "sales": np.arange(len(dates), dtype=float),
            "promo": np.ones(len(dates), dtype=float),
            "year": dates.year,
            "month": dates.month,
            "week": dates.isocalendar().week.astype("int64"),
            "day": dates.day,
        }
    )
    frame.attrs["split_role"] = "source"
    return frame


def test_repairs_calendar_fields_only_on_approved_dates() -> None:
    frame = _source_frame()
    approved = pd.to_datetime(D2_SOURCE_MISSING_DATES)
    frame.loc[frame["date"].isin(approved), ["year", "month", "week", "day"]] = np.nan
    original_sales = frame["sales"].copy()
    original_promo = frame["promo"].copy()

    repaired, evidence = repair_d2_source_calendar_fields(frame)

    repaired_rows = repaired[repaired["date"].isin(approved)]
    assert repaired_rows["year"].tolist() == [2018] * len(approved)
    assert repaired_rows["month"].tolist() == [4, 4, 5, 6]
    assert repaired_rows["week"].tolist() == [13, 17, 18, 22]
    assert repaired_rows["day"].tolist() == [1, 25, 1, 2]
    assert repaired["sales"].equals(original_sales)
    assert repaired["promo"].equals(original_promo)
    assert evidence["repair_dates"] == list(D2_SOURCE_MISSING_DATES)
    assert evidence["changed_cell_count"] == len(approved) * 4


def test_rejects_nonfinite_calendar_fields_outside_approved_dates() -> None:
    frame = _source_frame()
    frame.loc[frame["date"].eq(pd.Timestamp("2018-06-03")), "year"] = np.nan

    with pytest.raises(ProtocolViolation, match="outside approved repair dates"):
        repair_d2_source_calendar_fields(frame)


def test_sealed_source_verifier_rejects_unrepaired_approved_date_fields() -> None:
    frame = _source_frame()
    approved = pd.to_datetime(D2_SOURCE_MISSING_DATES)
    frame.loc[frame["date"].isin(approved), ["year", "month", "week", "day"]] = np.nan
    frame.loc[frame["date"].isin(approved), "sales"] = 0.0

    with pytest.raises(ProtocolViolation, match="requires producer repair"):
        verify_d2_source_frame(
            slice_d2_source_frame(frame),
            candidate_keys=(("1", "4"),),
        )
