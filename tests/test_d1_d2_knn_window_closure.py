from __future__ import annotations

import pandas as pd
import pytest

from src.protocols.experiment_protocol import ProtocolViolation
from src.protocols.experiment_protocol import PROTOCOL_VERSION
from src.protocols.knn_frames import get_configured_knn_frame
from src.protocols.runner_adapter import configure_protocol_frames
from scripts.regenerate_solidified_knn import (
    _build_regenerated_payload,
    _select_d1_d2_shared_protocol,
)
from src.source_selection import source_selector as source_selector_module
from src.source_selection.source_selector import SourceSelector


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
                row.update({"entity_id": key[0], "promo": 0.0, "year": timestamp.year, "month": timestamp.month, "week": int(timestamp.isocalendar().week), "day": timestamp.day})
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
            row.update({"entity_id": "1", "promo": 0.0, "year": timestamp.year, "month": timestamp.month, "week": int(timestamp.isocalendar().week), "day": timestamp.day})
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


def _select_d1(source: pd.DataFrame, target: pd.DataFrame) -> dict[str, object]:
    configured_source, configured_target = configure_protocol_frames(
        source,
        target,
        dataset_id="D1",
        scenario="with",
        group_cols=("store_id", "item_id"),
        observed_start=None,
    )
    return SourceSelector().select_top_k_sources(
        configured_target,
        configured_source,
        feature_cols=("sales",),
        k=3,
        group_cols=("store_id", "item_id"),
        weight_mode="inverse_distance",
    )


def test_shared_selector_receives_only_the_configured_observed_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, target = _strict_frames("D1")
    captured: dict[str, object] = {}
    original = source_selector_module.select_daily_sequence_sources

    def capture(**kwargs: object):
        captured.update(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(
        source_selector_module,
        "select_daily_sequence_sources",
        capture,
    )
    selection = _select_d1(source, target)

    captured_source = captured["source_df"]
    captured_target = captured["target_df"]
    assert isinstance(captured_source, pd.DataFrame)
    assert isinstance(captured_target, pd.DataFrame)
    assert captured_source["date"].min() == pd.Timestamp("2017-06-01")
    assert captured_source["date"].max() == pd.Timestamp("2017-06-30")
    assert captured_target["date"].min() == pd.Timestamp("2017-06-01")
    assert captured_target["date"].max() == pd.Timestamp("2017-06-30")
    assert not (captured_source["date"] > pd.Timestamp("2017-06-30")).any()
    assert not (captured_target["date"] > pd.Timestamp("2017-06-30")).any()
    assert selection["meta"]["boundary"] == "inclusive"
    assert selection["meta"]["selection_digest"] == selection["meta"]["selection_result_digest"]


def test_d1_future_sentinels_do_not_change_any_knn_identity() -> None:
    source, target = _strict_frames("D1")
    baseline = _select_d1(source, target)
    changed_source = source.copy()
    changed_target = target.copy()
    future = changed_source["date"] > pd.Timestamp("2017-06-30")
    changed_source.loc[future, "sales"] = 10**12
    changed_target.loc[changed_target["date"] > pd.Timestamp("2017-06-30"), "sales"] = -10**12

    changed = _select_d1(changed_source, changed_target)

    for field in (
        "source_frame_digest",
        "target_frame_digest",
        "candidate_pool_digest",
        "selection_digest",
        "selection_result_digest",
    ):
        assert baseline["meta"][field] == changed["meta"][field]
    assert baseline["sources"] == changed["sources"]


@pytest.mark.parametrize("dataset_id", [1, 2])
def test_regeneration_d1_d2_uses_shared_origin_bounded_selector(dataset_id: int) -> None:
    source, target = _strict_frames(f"D{dataset_id}")
    group_cols = ("store_id", "item_id") if dataset_id == 1 else ("brand_id", "item_id")
    target_entity = target.loc[
        (target[group_cols[0]].astype(str) == "1")
        & (target[group_cols[1]].astype(str) == "10")
    ].copy()

    selected = _select_d1_d2_shared_protocol(
        dataset_id=dataset_id,
        source_df=source,
        target_entity_df=target_entity,
        scenario="with",
        feature_cols=("sales",),
        k=3,
        group_cols=group_cols,
    )

    metadata = selected["meta"]
    assert metadata["selection_authority"] == "shared_protocol"
    assert metadata["selection_path"] == "shared_protocol"
    assert metadata["protocol_version"] == PROTOCOL_VERSION
    assert metadata["knn_observed_start"] == (
        "2017-06-01" if dataset_id == 1 else "2018-06-01"
    )
    assert metadata["knn_observed_end"] == (
        "2017-06-30" if dataset_id == 1 else "2018-06-30"
    )
    assert metadata["feature_cols"] == (["sales"] if dataset_id == 1 else ["sales", "promo"])
    assert metadata["knn_feature_columns"] == metadata["feature_cols"]
    assert len(metadata["source_frame_digest"]) == 64
    assert len(metadata["target_frame_digest"]) == 64


def test_d1_d2_regenerated_payload_binds_shared_authority() -> None:
    old_payload = {
        "dataset_id": 1,
        "dataset": "D1",
        "info_sharing": "with",
        "k": 3,
        "target_train_window": {"start": "2017-06-05", "end": "2017-06-19"},
        "group_cols": ["store_id", "item_id"],
    }
    metadata = {
        "selection_authority": "shared_protocol",
        "protocol_version": PROTOCOL_VERSION,
        "selection_digest": "a" * 64,
    }

    regenerated = _build_regenerated_payload(
        dataset_id=1,
        old_payload=old_payload,
        feature_cols=["sales"],
        feature_info={"selected_features": ["sales"]},
        source_pool_size=10,
        results={"1_10": []},
        selection_metadata={"1_10": metadata},
    )

    assert regenerated["selection_authority"] == "shared_protocol"
    assert regenerated["protocol_version"] == PROTOCOL_VERSION
    assert regenerated["training_selection_authority"] == "shared_protocol_selector"
    assert regenerated["feature_cols"] == ["sales"]
    assert regenerated["selection_metadata"] == {"1_10": metadata}
