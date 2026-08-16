from __future__ import annotations

import pandas as pd
import pytest

from scripts.run_full_paper_experiments import _bind_protocol_cell_identity
from scripts.validate_d1_d6_protocol_inputs import build_preflight_reports
from src.data_processing.data_preprocessing import (
    normalize_features,
    temporal_split_by_ratio_or_dates,
)
from src.protocols.experiment_protocol import ProtocolViolation
from src.protocols.d2_source_calendarization import repair_d2_source_entity_identity
from src.protocols.runner_adapter import configure_protocol_frames


_MISSING_DATES = pd.to_datetime(
    ["2018-04-01", "2018-04-25", "2018-05-01", "2018-06-02"]
)


def _d2_source_precalendarized() -> pd.DataFrame:
    dates = pd.date_range("2018-01-02", "2018-06-30", freq="D")
    rows = []
    for brand in range(1, 4):
        for item in range(1, 10):
            for date in dates:
                rows.append(
                    {
                        "date": date,
                        "brand_id": brand,
                        "item_id": item,
                        "entity_id": str(brand),
                        "sales": 0.0 if date in set(_MISSING_DATES) else float(item),
                        "promo": float(date.day % 2),
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
            "promo": [1.0] * 31,
        }
    )
    target.attrs["split_role"] = "target"
    return target


def test_configure_protocol_frames_verifies_precalendarized_d2_source_before_pool_creation() -> None:
    source, target = configure_protocol_frames(
        _d2_source_precalendarized(),
        _d2_target_after_observed_window(),
        dataset_id="D2",
        scenario="with",
        group_cols=("brand_id", "item_id"),
        observed_start="2018-06-01",
    )

    assert source.groupby(["brand_id", "item_id"]).date.nunique().eq(180).all()
    assert source.attrs["d2_source_calendarization_rule_version"]
    assert source.attrs["d2_synthetic_source_row_count"] == 0
    assert source.attrs["d2_source_authority_digest"] == target.attrs[
        "d2_source_authority_digest"
    ]
    assert source.attrs["d2_consumer_frame_fingerprint"] == target.attrs[
        "d2_consumer_frame_fingerprint"
    ]


def test_d2_preflight_prepares_pool_from_precalendarized_source() -> None:
    reports = build_preflight_reports(
        _d2_source_precalendarized(),
        _d2_target_after_observed_window(),
        dataset_id="D2",
        scenario="with",
        group_cols=("brand_id", "item_id"),
        observed_start="2018-06-01",
        k=1,
    )

    assert reports[0]["status"] == "passed"
    assert reports[0]["candidate_count"] == 27


def test_d2_without_preflight_uses_only_the_without_candidates() -> None:
    reports = build_preflight_reports(
        _d2_source_precalendarized(),
        _d2_target_after_observed_window(),
        dataset_id="D2",
        scenario="without",
        group_cols=("brand_id", "item_id"),
        observed_start="2018-06-01",
        k=1,
    )

    assert reports[0]["status"] == "passed"
    assert reports[0]["candidate_count"] == 9


def test_d2_runtime_rejects_missing_frozen_date_instead_of_calendarizing() -> None:
    source = _d2_source_precalendarized()
    source = source.loc[
        ~source["date"].eq(pd.Timestamp("2018-04-01"))
    ].copy()
    with pytest.raises(ProtocolViolation, match="exact 180"):
        configure_protocol_frames(
            source,
            _d2_target_after_observed_window(),
            dataset_id="D2",
            scenario="with",
            group_cols=("brand_id", "item_id"),
            observed_start="2018-06-01",
        )


def test_sealed_producer_repairs_all_27_missing_entity_ids_canonically() -> None:
    source = _d2_source_precalendarized()
    repair_date = pd.Timestamp("2018-06-02")
    source.loc[source["date"].eq(repair_date), "entity_id"] = pd.NA

    repaired, evidence = repair_d2_source_entity_identity(source)

    repaired_rows = repaired.loc[repaired["date"].eq(repair_date)]
    assert len(repaired_rows) == 27
    assert repaired_rows["entity_id"].isna().sum() == 0
    assert repaired_rows["entity_id"].tolist() == repaired_rows["brand_id"].astype(str).tolist()
    assert evidence["repaired_row_count"] == 27
    assert evidence["changed_cell_count"] == 27


def test_configured_d2_target_preserves_180_forecast_promo_masks() -> None:
    target = _d2_target_after_observed_window()
    extra_dates = pd.date_range("2018-07-02", "2018-12-27", freq="D")
    extra = pd.DataFrame(
        {
            "date": extra_dates,
            "brand_id": 1,
            "item_id": 10,
            "sales": 1.0,
            "promo": 1.0,
        }
    )
    target = pd.concat([target, extra], ignore_index=True)
    target.attrs["split_role"] = "target"

    _, configured_target = configure_protocol_frames(
        _d2_source_precalendarized(),
        target,
        dataset_id="D2",
        scenario="without",
        group_cols=("brand_id", "item_id"),
        observed_start="2018-06-01",
    )

    assert configured_target.loc[
        configured_target["date"] > pd.Timestamp("2018-06-30"), "promo"
    ].isna().sum() == 180


def test_d2_notl_production_wiring_normalizes_with_masked_forecast_promo() -> None:
    source = _d2_source_precalendarized()
    target = _d2_target_after_observed_window()
    extra_dates = pd.date_range("2018-07-02", "2018-12-27", freq="D")
    target = pd.concat(
        [
            target,
            pd.DataFrame(
                {
                    "date": extra_dates,
                    "brand_id": 1,
                    "item_id": 10,
                    "sales": 1.0,
                    "promo": 1.0,
                }
            ),
        ],
        ignore_index=True,
    )
    target["entity_id"] = target["brand_id"].astype(str)
    target["year"] = target["date"].dt.year
    target["month"] = target["date"].dt.month
    target["week"] = target["date"].dt.isocalendar().week.astype("int64")
    target["day"] = target["date"].dt.day
    target.attrs.update(
        {
            "split_role": "target",
            "split_mode": "days",
            "split_config": {"train_days": 15, "val_days": 15, "test_days": 180},
        }
    )
    lifecycle = ("D2", "without", 1, 42, ("1", "10"))
    _bind_protocol_cell_identity(source, lifecycle)
    _bind_protocol_cell_identity(target, lifecycle)

    _, configured_target = configure_protocol_frames(
        source,
        target,
        dataset_id="D2",
        scenario="without",
        group_cols=("brand_id", "item_id"),
        observed_start="2018-06-01",
    )
    partitions = temporal_split_by_ratio_or_dates(configured_target)
    normalized = normalize_features(
        *partitions,
        feature_columns=("sales", "year", "month", "week", "day"),
    )

    assert configured_target.loc[
        configured_target["date"] > pd.Timestamp("2018-06-30"), "promo"
    ].isna().sum() == 180
    assert normalized[2]["promo"].isna().sum() == 180
