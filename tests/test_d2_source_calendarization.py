from __future__ import annotations

import inspect

import pandas as pd
import pytest

from src.protocols.d2_source_calendarization import (
    D2_FROZEN_SOURCE_CANDIDATE_KEYS,
    D2_SOURCE_CALENDARIZATION_RULE_VERSION,
    D2_SOURCE_MISSING_DATES,
    calendarize_d2_source_frame,
    slice_d2_source_frame,
)
from src.protocols.experiment_protocol import ProtocolViolation


def _source_frame(*, missing_dates: tuple[str, ...] = D2_SOURCE_MISSING_DATES) -> pd.DataFrame:
    dates = pd.date_range("2018-01-02", "2018-06-30", freq="D")
    missing = set(pd.to_datetime(list(missing_dates)))
    rows = []
    for date in dates:
        if date in missing:
            continue
        rows.append(
            {
                "date": date,
                "brand_id": 1,
                "item_id": 1,
                "entity_id": "B1",
                "sales": float(date.day),
                "promo": int(date.day % 2 == 0),
                "year": 2018,
                "month": date.month,
                "week": int(date.isocalendar().week),
                "day": date.day,
            }
        )
    frame = pd.DataFrame(rows)
    frame.attrs["split_role"] = "source"
    return frame


def test_frozen_source_candidate_keys_are_exactly_three_brands_and_nine_items() -> None:
    assert len(D2_FROZEN_SOURCE_CANDIDATE_KEYS) == 27
    assert D2_FROZEN_SOURCE_CANDIDATE_KEYS[0] == ("1", "1")
    assert D2_FROZEN_SOURCE_CANDIDATE_KEYS[-1] == ("3", "9")


def test_calendarizer_fills_only_the_four_frozen_dates_and_rebuilds_calendar_fields() -> None:
    result, report = calendarize_d2_source_frame(
        slice_d2_source_frame(_source_frame()),
        candidate_keys=(("1", "1"),),
    )

    assert len(result) == 180
    assert result["date"].nunique() == 180
    synthetic = result[result["date"].isin(pd.to_datetime(list(D2_SOURCE_MISSING_DATES)))]
    assert synthetic["sales"].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert synthetic["brand_id"].tolist() == [1, 1, 1, 1]
    assert synthetic["item_id"].tolist() == [1, 1, 1, 1]
    assert synthetic["entity_id"].tolist() == ["B1"] * 4
    assert synthetic["year"].tolist() == [2018] * 4
    assert synthetic["month"].tolist() == [4, 4, 5, 6]
    assert synthetic["week"].tolist() == [13, 17, 18, 22]
    assert synthetic["day"].tolist() == [1, 25, 1, 2]
    assert synthetic["promo"].isna().all()
    assert report.synthetic_row_count == 4
    assert report.rule_version == D2_SOURCE_CALENDARIZATION_RULE_VERSION
    assert len(report.source_authority_digest) == 64
    assert len(report.consumer_frame_fingerprint) == 64


def test_calendarizer_never_interpolates_sales() -> None:
    source = _source_frame()
    result, _ = calendarize_d2_source_frame(
        slice_d2_source_frame(source),
        candidate_keys=(("1", "1"),),
    )

    assert result.loc[result["date"].eq(pd.Timestamp("2018-04-01")), "sales"].item() == 0.0
    assert result.loc[result["date"].eq(pd.Timestamp("2018-03-31")), "sales"].item() == 31.0
    assert result.loc[result["date"].eq(pd.Timestamp("2018-04-02")), "sales"].item() == 2.0


def test_calendarizer_rejects_any_missing_date_outside_the_allowlist() -> None:
    source = _source_frame(missing_dates=D2_SOURCE_MISSING_DATES + ("2018-03-01",))

    with pytest.raises(ProtocolViolation, match="unsupported missing source dates"):
        calendarize_d2_source_frame(
            slice_d2_source_frame(source),
            candidate_keys=(("1", "1"),),
        )


def test_calendarizer_rejects_duplicate_entity_date_keys() -> None:
    source = pd.concat([_source_frame(), _source_frame().iloc[[0]]], ignore_index=True)

    with pytest.raises(ProtocolViolation, match="duplicate"):
        calendarize_d2_source_frame(
            slice_d2_source_frame(source),
            candidate_keys=(("1", "1"),),
        )


def test_calendarizer_requires_source_role_and_has_no_target_parameter() -> None:
    source = _source_frame()
    source.attrs["split_role"] = "target"

    with pytest.raises(ProtocolViolation, match="source role"):
        calendarize_d2_source_frame(source, candidate_keys=(("1", "1"),))

    parameters = inspect.signature(calendarize_d2_source_frame).parameters
    assert "target_df" not in parameters
    assert "validation_df" not in parameters
    assert "blind_df" not in parameters


def test_calendarizer_rejects_missing_candidate_entity() -> None:
    with pytest.raises(ProtocolViolation, match="candidate entity"):
        calendarize_d2_source_frame(
            slice_d2_source_frame(_source_frame()),
            candidate_keys=(("2", "1"),),
        )


def test_slice_d2_source_frame_keeps_only_the_frozen_interval() -> None:
    source = pd.concat(
        [
            _source_frame(),
            pd.DataFrame(
                [
                    {
                        "date": pd.Timestamp("2017-12-31"),
                        "brand_id": 1,
                        "item_id": 1,
                        "entity_id": "B1",
                        "sales": 99.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    source.attrs["split_role"] = "source"

    sliced = slice_d2_source_frame(source)

    assert sliced["date"].min() == pd.Timestamp("2018-01-02")
    assert sliced["date"].max() == pd.Timestamp("2018-06-30")
    assert sliced["date"].nunique() == 176
