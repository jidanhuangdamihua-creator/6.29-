from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.protocols.experiment_protocol import ProtocolViolation
from src.protocols.runner_adapter import configure_protocol_frames
from src.source_selection.source_selector import SourceSelector


SOURCE_DATES = pd.date_range("2023-07-15", periods=190, freq="D")
OBSERVED_DATES = SOURCE_DATES[150:180]


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    source_dates = SOURCE_DATES
    observed_dates = OBSERVED_DATES
    target_dates = source_dates[150:190]
    target = pd.DataFrame(
        {
            "store_id": "T1",
            "product_id": "P0",
            "second_category_id": 20,
            "date": target_dates,
            "sales": np.r_[np.zeros(30), np.full(10, 5000.0)],
        }
    )
    source = pd.concat(
        [
            pd.DataFrame(
                {
                    "store_id": store,
                    "product_id": item,
                    "second_category_id": 20,
                    "date": source_dates,
                    "sales": np.r_[np.full(150, 50.0), np.full(30, observed), np.full(10, future)],
                    "year": source_dates.year,
                    "month": source_dates.month,
                    "week": source_dates.isocalendar().week.to_numpy(dtype=np.int64),
                    "day": source_dates.day,
                    "holiday_flag": 0,
                }
            )
            for store, item, observed, future in (
                ("S1", "P1", 1.0, 1000.0),
                ("S2", "P2", 4.0, 2000.0),
                ("S3", "P3", 10.0, 3000.0),
            )
        ],
        ignore_index=True,
    )
    return source, target


def _select(source: pd.DataFrame, target: pd.DataFrame):
    configured_source, configured_target = configure_protocol_frames(
        source,
        target,
        dataset_id="D4",
        scenario="with",
        group_cols=("store_id", "product_id"),
        grouping_col="second_category_id",
        observed_start=OBSERVED_DATES[0],
    )
    return SourceSelector().select_top_k_sources(
        configured_target,
        configured_source,
        feature_cols=("sales",),
        k=2,
        group_cols=("store_id", "product_id"),
    )


def test_future_perturbation_preserves_order_distances_weights_and_digests() -> None:
    source, target = _frames()
    baseline = _select(source, target)
    changed_source = source.copy()
    changed_target = target.copy()
    changed_source.loc[changed_source["date"] > OBSERVED_DATES[-1], "sales"] = 1e12
    changed_target.loc[changed_target["date"] > OBSERVED_DATES[-1], "sales"] = -1e12
    changed = _select(changed_source, changed_target)

    assert baseline["sources"] == changed["sources"]
    assert baseline["meta"]["candidate_pool_digest"] == changed["meta"]["candidate_pool_digest"]
    assert baseline["meta"]["selection_result_digest"] == changed["meta"]["selection_result_digest"]


def test_observed_perturbation_has_deterministic_positive_control() -> None:
    source, target = _frames()
    baseline = _select(source, target)
    changed_source = source.copy()
    mask = (changed_source["store_id"] == "S3") & changed_source["date"].isin(OBSERVED_DATES)
    changed_source.loc[mask, "sales"] = 0.0
    changed = _select(changed_source, target)

    assert tuple(baseline["sources"][0]["source_key"]) == ("S1", "P1")
    assert tuple(changed["sources"][0]["source_key"]) == ("S3", "P3")


def test_named_d1_d6_frame_without_shared_metadata_fails() -> None:
    source, target = _frames()
    source.attrs["dataset_name"] = "Dataset4"
    target.attrs["dataset_name"] = "Dataset4"
    with pytest.raises(ProtocolViolation, match="shared protocol metadata"):
        SourceSelector().select_top_k_sources(
            target,
            source,
            feature_cols=("sales",),
            k=1,
            group_cols=("store_id", "product_id"),
        )
