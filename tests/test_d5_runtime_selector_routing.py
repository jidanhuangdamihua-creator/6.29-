from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.constants import D4_D6_RUNTIME_KNN_PROTOCOL_VERSION
from src.protocols.candidate_pool import build_candidate_pool_digest
from src.protocols.experiment_protocol import get_experiment_protocol
import src.source_selection.source_selector as source_selector_module
from src.source_selection.source_selector import SourceSelector


def _runtime_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    history_dates = pd.date_range("2020-01-02", periods=180, freq="D")
    observed_dates = history_dates[-30:]
    source = pd.concat(
        [
            pd.DataFrame(
                {
                    "store_nbr": store,
                    "item_nbr": item,
                    "date": history_dates,
                    "sales": float(value),
                }
            )
            for store, item, value in (("S1", "I1", 1.0), ("S2", "I2", 2.0), ("S3", "I3", 3.0))
        ],
        ignore_index=True,
    )
    target = pd.DataFrame(
        {
            "store_nbr": "T1",
            "item_nbr": "I0",
            "date": observed_dates,
            "sales": np.zeros(len(observed_dates)),
        }
    )
    attrs = {
        "dataset_name": "Dataset5",
        "selection_authority": "runtime",
        "protocol_version": D4_D6_RUNTIME_KNN_PROTOCOL_VERSION,
        "target_observed_start": observed_dates[0],
        "target_observed_end": observed_dates[-1],
        "source_history_start": history_dates[0],
        "source_history_end": history_dates[-1],
        "source_history_days": 180,
        "source_history_expected_date_count": 180,
        "source_history_completeness_policy": "exact_expected_date_set",
        "source_history_calendar": "Gregorian daily",
        "source_history_inclusive_end": True,
        "source_history_calendarization_rule": "not_applicable",
        "source_history_synthetic_row_count": 0,
        "source_history_frame_digest": "a" * 64,
        "target_test_excluded": True,
        "source_future_excluded": True,
        "source_alignment_mode": "exact_target_observed_dates",
        "representation": "mean_std_min_max_last",
        "scaling": "none",
        "scaler_fit_scope": "not_applicable",
    }
    source.attrs = attrs.copy()
    target.attrs = attrs.copy()
    return source, target


def test_d5_runtime_metadata_routes_before_legacy_d1_d6_rejection(monkeypatch) -> None:
    source, target = _runtime_frames()
    selector = SourceSelector()
    signature_lengths: list[int] = []
    real_signature = selector._signature_from_df

    def tracked_signature(frame, feature_cols, static_feature_cols=None):
        signature_lengths.append(len(frame))
        return real_signature(frame, feature_cols, static_feature_cols=static_feature_cols)

    monkeypatch.setattr(selector, "_signature_from_df", tracked_signature)

    result = selector.select_top_k_sources(
        target,
        source,
        feature_cols=("sales",),
        k=2,
        group_cols=("store_nbr", "item_nbr"),
    )

    assert result["meta"]["selection_authority"] == "runtime"
    assert len(result["sources"]) == 2
    assert signature_lengths == [30]
    assert result["meta"]["consumer_frame_rows"] == 2 * 180
    assert result["meta"]["selected_count"] == 2
    assert len(result["meta"]["consumer_frame_digest"]) == 64


def test_d5_runtime_consumer_frame_normalizes_numeric_source_keys() -> None:
    source, target = _runtime_frames()
    source["store_nbr"] = source["store_nbr"].map({"S1": 1, "S2": 2, "S3": 3})
    source["item_nbr"] = source["item_nbr"].map({"I1": 11, "I2": 12, "I3": 13})

    result = SourceSelector().select_top_k_sources(
        target,
        source,
        feature_cols=("sales",),
        k=2,
        group_cols=("store_nbr", "item_nbr"),
    )

    assert result["meta"]["consumer_frame_rows"] == 2 * 180


def test_d5_runtime_producer_emits_protocol_canonical_metadata() -> None:
    source, target = _runtime_frames()
    source.attrs["knn_feature_columns"] = ["legacy"]
    target.attrs["knn_feature_columns"] = ["legacy"]
    source.attrs["feature_cols"] = ["legacy"]
    target.attrs["feature_cols"] = ["legacy"]

    result = SourceSelector().select_top_k_sources(
        target,
        source,
        feature_cols=("sales",),
        k=2,
        group_cols=("store_nbr", "item_nbr"),
    )

    protocol = get_experiment_protocol("D5")
    window = protocol.observation_window(target.attrs["target_observed_start"])
    metadata = result["meta"]
    assert metadata["knn_feature_columns"] == list(protocol.knn_feature_columns)
    assert metadata["historical_feature_columns"] == list(protocol.knn_feature_columns)
    assert metadata["forecast_excluded_columns"] == []
    assert metadata["feature_scope"] == "historical_observed"
    assert metadata["max_allowed_date_relation"] == "date<=origin"
    assert metadata["knn_observed_start"] == window.knn_observed_start.isoformat()
    assert metadata["knn_observed_end"] == window.knn_observed_end.isoformat()
    assert metadata["candidate_pool_digest"] == build_candidate_pool_digest(
        **metadata["candidate_pool_digest_input"]
    )
    assert metadata["candidate_pool_digest_input"]["target_key"] == ["T1", "I0"]


def test_d5_runtime_rejects_observed_end_that_is_not_protocol_origin() -> None:
    source, target = _runtime_frames()
    target.attrs["target_observed_end"] = pd.Timestamp("2020-06-28")
    with pytest.raises(ValueError, match="target_observed_end"):
        SourceSelector().select_top_k_sources(
            target,
            source,
            feature_cols=("sales",),
            k=2,
            group_cols=("store_nbr", "item_nbr"),
        )


def test_d5_runtime_source_signatures_are_cached_for_repeated_targets(monkeypatch) -> None:
    source, target = _runtime_frames()
    selector = SourceSelector()
    calls: list[int] = []
    real_vectorized = selector._runtime_vectorized_source_signatures

    def tracked_vectorized(*args, **kwargs):
        calls.append(1)
        return real_vectorized(*args, **kwargs)

    monkeypatch.setattr(selector, "_runtime_vectorized_source_signatures", tracked_vectorized)

    for _ in range(2):
        selector.select_top_k_sources(
            target,
            source,
            feature_cols=("sales",),
            k=2,
            group_cols=("store_nbr", "item_nbr"),
        )

    assert calls == [1]
    assert len(selector._runtime_source_selection_metadata_cache) == 1


def test_d5_runtime_source_eligibility_is_cached_for_repeated_targets(monkeypatch) -> None:
    source, target = _runtime_frames()
    selector = SourceSelector()
    calls: list[int] = []
    real_eligibility = source_selector_module.build_exact_source_history_candidate_frame

    def tracked_eligibility(*args, **kwargs):
        calls.append(1)
        return real_eligibility(*args, **kwargs)

    monkeypatch.setattr(
        source_selector_module,
        "build_exact_source_history_candidate_frame",
        tracked_eligibility,
    )

    for _ in range(2):
        selector.select_top_k_sources(
            target,
            source,
            feature_cols=("sales",),
            k=2,
            group_cols=("store_nbr", "item_nbr"),
        )

    assert calls == [1]


def test_d5_runtime_explicit_features_skip_full_frame_inference(monkeypatch) -> None:
    source, target = _runtime_frames()

    def fail_inference(*args, **kwargs):
        raise AssertionError("runtime explicit features must not infer over the full source frame")

    monkeypatch.setattr(
        source_selector_module,
        "infer_source_selection_feature_columns",
        fail_inference,
    )

    result = SourceSelector().select_top_k_sources(
        target,
        source,
        feature_cols=("sales",),
        k=2,
        group_cols=("store_nbr", "item_nbr"),
    )

    assert result["meta"]["feature_cols"] == ["sales"]
    assert result["meta"]["runtime_infer_error"] == ""
