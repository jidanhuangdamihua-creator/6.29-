from __future__ import annotations

import pandas as pd

from scripts.validate_d1_d6_protocol_inputs import build_preflight_reports
from src.protocols.runner_adapter import configure_protocol_frames


_MISSING_DATES = pd.to_datetime(
    ["2018-04-01", "2018-04-25", "2018-05-01", "2018-06-02"]
)


def _d2_source_with_four_missing_dates() -> pd.DataFrame:
    dates = pd.date_range("2018-01-02", "2018-06-30", freq="D")
    rows = []
    for brand in range(1, 4):
        for item in range(1, 10):
            for date in dates:
                if date in set(_MISSING_DATES):
                    continue
                rows.append(
                    {
                        "date": date,
                        "brand_id": brand,
                        "item_id": item,
                        "entity_id": f"B{brand}",
                        "sales": float(item),
                        "year": date.year,
                        "month": date.month,
                        "week": int(date.isocalendar().week),
                        "day": date.day,
                    }
                )
    source = pd.DataFrame(rows)
    source.attrs["split_role"] = "source"
    return source


def _d2_target_after_observed_window() -> pd.DataFrame:
    observed = pd.date_range("2018-06-01", "2018-06-30", freq="D")
    target = pd.DataFrame(
        {
            "date": observed.append(pd.DatetimeIndex([pd.Timestamp("2018-07-01")])),
            "brand_id": [1] * 31,
            "item_id": [10] * 31,
            "sales": [1.0] * 31,
        }
    )
    target.attrs["split_role"] = "target"
    return target


def test_configure_protocol_frames_calendarizes_d2_source_before_pool_creation() -> None:
    source, target = configure_protocol_frames(
        _d2_source_with_four_missing_dates(),
        _d2_target_after_observed_window(),
        dataset_id="D2",
        scenario="with",
        group_cols=("brand_id", "item_id"),
        observed_start="2018-06-01",
    )

    assert source.groupby(["brand_id", "item_id"]).date.nunique().eq(180).all()
    assert source.attrs["d2_source_calendarization_rule_version"]
    assert source.attrs["d2_source_authority_digest"] == target.attrs[
        "d2_source_authority_digest"
    ]
    assert source.attrs["d2_consumer_frame_fingerprint"] == target.attrs[
        "d2_consumer_frame_fingerprint"
    ]


def test_d2_preflight_prepares_pool_from_calendarized_source() -> None:
    reports = build_preflight_reports(
        _d2_source_with_four_missing_dates(),
        _d2_target_after_observed_window(),
        dataset_id="D2",
        scenario="with",
        group_cols=("brand_id", "item_id"),
        observed_start="2018-06-01",
        k=1,
    )

    assert reports[0]["status"] == "passed"
    assert reports[0]["candidate_count"] == 27


def test_d2_without_preflight_calendarizes_only_the_without_candidates() -> None:
    reports = build_preflight_reports(
        _d2_source_with_four_missing_dates(),
        _d2_target_after_observed_window(),
        dataset_id="D2",
        scenario="without",
        group_cols=("brand_id", "item_id"),
        observed_start="2018-06-01",
        k=1,
    )

    assert reports[0]["status"] == "passed"
    assert reports[0]["candidate_count"] == 9
