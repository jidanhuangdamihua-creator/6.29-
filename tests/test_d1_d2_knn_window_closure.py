from __future__ import annotations

import pandas as pd
import pytest

from src.protocols.experiment_protocol import ProtocolViolation
from src.protocols.knn_frames import get_configured_knn_frame
from src.protocols.runner_adapter import configure_protocol_frames


def _strict_frames(dataset_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if dataset_id == "D1":
        observed_start = pd.Timestamp("2017-06-01")
        origin = pd.Timestamp("2017-06-30")
        group_cols = ("store_id", "item_id")
        source_start = observed_start
        source_keys = [(str(store), str(item)) for store in range(1, 4) for item in range(1, 10)]
    else:
        observed_start = pd.Timestamp("2018-06-01")
        origin = pd.Timestamp("2018-06-30")
        group_cols = ("brand_id", "item_id")
        source_start = pd.Timestamp("2018-01-02")
        source_keys = [(str(brand), str(item)) for brand in range(1, 4) for item in range(1, 10)]

    source_dates = pd.date_range(source_start, origin, freq="D")
    future_dates = pd.DatetimeIndex([origin + pd.Timedelta(days=1), origin + pd.Timedelta(days=4)])
    source_rows = []
    for key_index, key in enumerate(source_keys):
        for timestamp in source_dates.append(future_dates):
            frozen_zero = dataset_id == "D2" and timestamp.strftime("%Y-%m-%d") in {
                "2018-04-01",
                "2018-04-25",
                "2018-05-01",
                "2018-06-02",
            }
            row = {
                group_cols[0]: key[0],
                group_cols[1]: key[1],
                "date": timestamp,
                "sales": 0.0 if frozen_zero else float(key_index + 1),
            }
            if dataset_id == "D2":
                row.update({"promo": 0.0, "year": timestamp.year, "month": timestamp.month, "week": int(timestamp.isocalendar().week), "day": timestamp.day})
            source_rows.append(row)

    target_dates = pd.date_range(observed_start, origin + pd.Timedelta(days=4), freq="D")
    target_rows = []
    for timestamp in target_dates:
        row = {
            group_cols[0]: "1",
            group_cols[1]: "10",
            "date": timestamp,
            "sales": 0.0 if timestamp <= origin else 999999.0,
        }
        if dataset_id == "D2":
            row.update({"promo": 0.0, "year": timestamp.year, "month": timestamp.month, "week": int(timestamp.isocalendar().week), "day": timestamp.day})
        target_rows.append(row)

    source = pd.DataFrame(source_rows)
    target = pd.DataFrame(target_rows)
    source.attrs["split_role"] = "source"
    target.attrs["split_role"] = "target"
    return source, target


@pytest.mark.parametrize("dataset_id", ["D1", "D2"])
def test_configure_protocol_frames_builds_origin_bounded_knn_frames(dataset_id: str) -> None:
    source, target = _strict_frames(dataset_id)
    group_cols = ("store_id", "item_id") if dataset_id == "D1" else ("brand_id", "item_id")
    configured_source, configured_target = configure_protocol_frames(
        source,
        target,
        dataset_id=dataset_id,
        scenario="with",
        group_cols=group_cols,
        observed_start=None,
    )

    origin = pd.Timestamp("2017-06-30" if dataset_id == "D1" else "2018-06-30")
    observed_start = origin - pd.Timedelta(days=29)
    source_knn = get_configured_knn_frame(configured_source, "source")
    target_knn = get_configured_knn_frame(configured_target, "target")

    assert source_knn["date"].between(observed_start, origin, inclusive="both").all()
    assert target_knn["date"].between(observed_start, origin, inclusive="both").all()
    assert source_knn["date"].max() <= origin
    assert target_knn["date"].max() == origin
    assert not target_knn["date"].isin(
        [origin + pd.Timedelta(days=1), origin + pd.Timedelta(days=4)]
    ).any()
    assert configured_target["date"].max() == origin + pd.Timedelta(days=4)


def test_configure_protocol_frames_rejects_invalid_dates_before_knn_frame_build() -> None:
    source, target = _strict_frames("D1")
    source.loc[source.index[0], "date"] = pd.NaT

    with pytest.raises(ProtocolViolation, match="invalid dates"):
        configure_protocol_frames(
            source,
            target,
            dataset_id="D1",
            scenario="with",
            group_cols=("store_id", "item_id"),
            observed_start=None,
        )
