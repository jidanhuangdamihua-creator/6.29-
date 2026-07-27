from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.run_full_paper_experiments import _project_modeling_frames
from src.protocols.candidate_pool import select_daily_sequence_sources
from src.protocols.experiment_protocol import (
    D1_KNN_FEATURES,
    D2_KNN_FEATURES,
    ProtocolViolation,
    get_experiment_protocol,
)
from src.protocols.knn_frames import get_configured_knn_frame
from src.protocols.runner_adapter import configure_protocol_frames
from src.source_selection.source_selector import SourceSelector


ORIGIN = pd.Timestamp("2018-06-30")
DATES = pd.date_range("2018-06-01", ORIGIN, freq="D")
FUTURE = pd.date_range("2018-07-01", periods=2, freq="D")
GROUP_COLS = ("brand_id", "item_id")
CANDIDATES = (("1", "1"), ("1", "2"))


def _frames(
    *,
    dataset_id: str = "D2",
    source_promo: tuple[float, float] = (1.0, 9.0),
    target_promo: float = 1.0,
    future_promo: float = 3.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    origin = ORIGIN if dataset_id == "D2" else pd.Timestamp("2017-06-30")
    dates = pd.date_range(origin - pd.Timedelta(days=29), origin, freq="D")
    future = pd.date_range(origin + pd.Timedelta(days=1), periods=2, freq="D")
    group_cols = GROUP_COLS if dataset_id == "D2" else ("store_id", "item_id")
    source_rows: list[dict[str, object]] = []
    for index, key in enumerate(CANDIDATES):
        for date in dates.append(future):
            source_rows.append(
                {
                    group_cols[0]: key[0],
                    group_cols[1]: key[1],
                    "date": date,
                    "sales": 10.0,
                    "promo": source_promo[index] if date <= origin else future_promo,
                }
            )
    target_rows = [
        {
            group_cols[0]: "1",
            group_cols[1]: "10",
            "date": date,
            "sales": 10.0,
            "promo": target_promo if date <= origin else future_promo,
        }
        for date in dates.append(future)
    ]
    source = pd.DataFrame(source_rows)
    target = pd.DataFrame(target_rows)
    for frame in (source, target):
        frame.attrs.update(
            {
                "d2_source_calendarization_rule_version": "d2_source_calendarization_v1",
                "d2_source_authority_digest": "a" * 64,
                "d2_consumer_frame_fingerprint": "b" * 64,
            }
        )
    return source, target


def _select(
    source: pd.DataFrame,
    target: pd.DataFrame,
    *,
    dataset_id: str = "D2",
    feature_cols: tuple[str, ...] | None = None,
) -> object:
    protocol = get_experiment_protocol(dataset_id)
    return select_daily_sequence_sources(
        target_df=target,
        source_df=source,
        protocol=protocol,
        scenario="with",
        target_key=("1", "10"),
        candidate_keys=CANDIDATES,
        group_cols=GROUP_COLS if dataset_id == "D2" else ("store_id", "item_id"),
        observed_start="2018-06-01" if dataset_id == "D2" else "2017-06-01",
        feature_cols=feature_cols or protocol.knn_feature_columns,
        k=1,
    )


def test_protocol_declares_distinct_ordered_knn_features() -> None:
    assert D1_KNN_FEATURES == ("sales",)
    assert D2_KNN_FEATURES == ("sales", "promo")
    assert get_experiment_protocol("D1").knn_feature_columns == D1_KNN_FEATURES
    assert get_experiment_protocol("D2").knn_feature_columns == D2_KNN_FEATURES
    assert D1_KNN_FEATURES != D2_KNN_FEATURES


def test_d2_historical_promo_changes_distance_and_selection() -> None:
    source, target = _frames(source_promo=(1.0, 9.0), target_promo=9.0)
    baseline = _select(source, target)

    changed_source = source.copy()
    changed_source.loc[
        (changed_source["brand_id"] == "1")
        & (changed_source["item_id"] == "2")
        & (changed_source["date"] <= ORIGIN),
        "promo",
    ] = 1.0
    changed = _select(changed_source, target)

    assert baseline.ordered_source_keys == (("1", "2"),)
    assert changed.ordered_source_keys == (("1", "1"),)
    assert baseline.distances.tolist() != changed.distances.tolist()
    assert baseline.selection_result_digest != changed.selection_result_digest


def test_d1_selection_and_digest_ignore_unrequested_promo() -> None:
    source, target = _frames(dataset_id="D1", source_promo=(1.0, 9.0), target_promo=1.0)
    baseline = _select(source, target, dataset_id="D1", feature_cols=D1_KNN_FEATURES)

    changed_source = source.copy()
    changed_target = target.copy()
    changed_source["promo"] = 10_000.0
    changed_target["promo"] = -10_000.0
    changed = _select(
        changed_source,
        changed_target,
        dataset_id="D1",
        feature_cols=D1_KNN_FEATURES,
    )

    assert baseline.ordered_source_keys == changed.ordered_source_keys
    assert baseline.distances.tolist() == changed.distances.tolist()
    assert baseline.selection_result_digest == changed.selection_result_digest
    assert baseline.candidate_pool_digest == changed.candidate_pool_digest


@pytest.mark.parametrize("dataset_id", ["D1", "D3", "D4", "D5", "D6"])
def test_projection_passthrough_uses_each_formal_protocol_without_promo_drift(
    dataset_id: str,
) -> None:
    protocol = get_experiment_protocol(dataset_id)
    source = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=2),
            "sales": [1.0, 2.0],
            "promo": [7.0, 8.0],
            "onpromotion": [0.0, 1.0],
            "oil_price": [90.0, 91.0],
        }
    )
    target = source.copy()

    projected_source, projected_target = _project_modeling_frames(
        source,
        target,
        modeling_feature_cols=["sales"],
        required_passthrough_cols=protocol.knn_feature_columns,
    )

    assert set(protocol.knn_feature_columns).issubset(projected_source.columns)
    assert set(protocol.knn_feature_columns).issubset(projected_target.columns)
    assert "promo" not in protocol.knn_feature_columns
    assert "promo" not in projected_source.columns
    assert "promo" not in projected_target.columns


def test_d2_future_promo_and_post_origin_rows_do_not_change_selection_or_digest() -> None:
    source, target = _frames(source_promo=(1.0, 9.0), target_promo=1.0, future_promo=3.0)
    baseline = _select(source, target)

    changed_source = source.copy()
    changed_target = target.copy()
    changed_source.loc[changed_source["date"] > ORIGIN, "promo"] = -1_000_000.0
    changed_target.loc[changed_target["date"] > ORIGIN, "promo"] = 1_000_000.0
    changed = _select(changed_source, changed_target)

    assert baseline.ordered_source_keys == changed.ordered_source_keys
    assert baseline.distances.tolist() == changed.distances.tolist()
    assert baseline.candidate_pool_digest == changed.candidate_pool_digest
    assert baseline.selection_result_digest == changed.selection_result_digest
    assert baseline.observed_end == ORIGIN.strftime("%Y-%m-%d")


def test_d2_configured_forecast_consumer_excludes_real_promo() -> None:
    from tests.test_d1_d2_knn_window_closure import _strict_frames

    source, target = _strict_frames("D2")
    configured_source, configured_target = configure_protocol_frames(
        source,
        target,
        dataset_id="D2",
        scenario="with",
        group_cols=GROUP_COLS,
        observed_start=None,
    )

    historical = get_configured_knn_frame(configured_target, "target")
    forecast = configured_target.attrs["forecast_consumer_frame"]
    assert historical.attrs["knn_feature_columns"] == list(D2_KNN_FEATURES)
    assert historical["date"].max() == ORIGIN
    assert "promo" in historical.columns
    assert "promo" not in forecast.columns
    assert configured_target.loc[configured_target["date"] > ORIGIN, "promo"].isna().all()


@pytest.mark.parametrize("scenario", ["with", "without"])
def test_d2_projection_and_configured_frames_preserve_promo_only_for_history(
    scenario: str,
) -> None:
    from tests.test_d1_d2_knn_window_closure import _strict_frames

    source, target = _strict_frames("D2")
    modeling_feature_cols = ["sales", "year", "month", "week", "day"]
    knn_required_cols = get_experiment_protocol("D2").knn_feature_columns
    projected_source, projected_target = _project_modeling_frames(
        source,
        target,
        modeling_feature_cols=modeling_feature_cols,
        required_passthrough_cols=knn_required_cols,
    )

    configured_source, configured_target = configure_protocol_frames(
        projected_source,
        projected_target,
        dataset_id="D2",
        scenario=scenario,
        group_cols=GROUP_COLS,
        observed_start=None,
    )
    source_observed = get_configured_knn_frame(configured_source, "source")
    target_observed = get_configured_knn_frame(configured_target, "target")
    forecast = configured_target.attrs["forecast_consumer_frame"]

    assert "promo" not in modeling_feature_cols
    assert list(source_observed.attrs["knn_feature_columns"]) == list(D2_KNN_FEATURES)
    assert list(target_observed.attrs["knn_feature_columns"]) == list(D2_KNN_FEATURES)
    assert set(D2_KNN_FEATURES).issubset(source_observed.columns)
    assert set(D2_KNN_FEATURES).issubset(target_observed.columns)
    assert source_observed["date"].max() <= ORIGIN
    assert target_observed["date"].max() == ORIGIN
    assert "promo" not in forecast.columns
    assert configured_target.loc[configured_target["date"] > ORIGIN, "promo"].isna().all()


def test_d2_selection_metadata_keeps_window_days_separate_from_signature_dim() -> None:
    from tests.test_d1_d2_knn_window_closure import _strict_frames

    source, target = _strict_frames("D2")
    configured_source, configured_target = configure_protocol_frames(
        source,
        target,
        dataset_id="D2",
        scenario="with",
        group_cols=GROUP_COLS,
        observed_start=None,
    )
    selected = SourceSelector().select_top_k_sources(
        configured_target,
        configured_source,
        feature_cols=D2_KNN_FEATURES,
        k=3,
        group_cols=GROUP_COLS,
    )

    metadata = selected["meta"]
    assert metadata["observed_days"] == 30
    assert metadata["protocol_observed_days"] == 30
    assert metadata["target_signature_dim"] == 60


def test_d2_sales_only_configured_metadata_fails_closed() -> None:
    from tests.test_d1_d2_knn_window_closure import _strict_frames

    source, target = _strict_frames("D2")
    configured_source, configured_target = configure_protocol_frames(
        source,
        target,
        dataset_id="D2",
        scenario="with",
        group_cols=GROUP_COLS,
        observed_start=None,
    )
    configured_knn = configured_source.attrs["protocol_knn_observed_frame"].copy()
    configured_knn.attrs = configured_knn.attrs.copy()
    configured_knn.attrs["knn_feature_columns"] = ["sales"]
    configured_source.attrs["protocol_knn_observed_frame"] = configured_knn

    with pytest.raises(ProtocolViolation, match="feature metadata"):
        SourceSelector().select_top_k_sources(
            configured_target,
            configured_source,
            feature_cols=D2_KNN_FEATURES,
            k=3,
            group_cols=GROUP_COLS,
        )


def test_d2_missing_or_misordered_features_fail_closed() -> None:
    source, target = _frames()
    missing_promo = source.drop(columns=["promo"])
    with pytest.raises(ProtocolViolation, match="promo|feature"):
        _select(missing_promo, target)
    with pytest.raises(ProtocolViolation, match="protocol"):
        _select(source, target, feature_cols=("promo", "sales"))
