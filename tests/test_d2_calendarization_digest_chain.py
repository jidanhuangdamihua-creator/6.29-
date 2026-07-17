from __future__ import annotations

from src.protocols.candidate_pool import build_candidate_pool_digest
from src.source_selection.source_selector import SourceSelector

from tests.test_d2_source_calendarization_integration import (
    _d2_source_precalendarized,
    _d2_target_after_observed_window,
)
from src.protocols.runner_adapter import configure_protocol_frames


def _configured_d2_frames():
    return configure_protocol_frames(
        _d2_source_precalendarized(),
        _d2_target_after_observed_window(),
        dataset_id="D2",
        scenario="with",
        group_cols=("brand_id", "item_id"),
        observed_start="2018-06-01",
    )


def test_d2_calendarization_identity_changes_candidate_and_sealed_digests() -> None:
    source, target = _configured_d2_frames()
    baseline = SourceSelector().select_top_k_sources(
        target,
        source,
        feature_cols=("sales",),
        k=1,
        group_cols=("brand_id", "item_id"),
        weight_mode="inverse_distance",
    )

    mutated = source.copy()
    mutated.attrs = source.attrs.copy()
    mutated.attrs["d2_source_calendarization_rule_version"] = "d2_source_calendarization_v2"
    mutated_target = target.copy()
    mutated_target.attrs = target.attrs.copy()
    mutated_target.attrs["d2_source_calendarization_rule_version"] = "d2_source_calendarization_v2"
    changed = SourceSelector().select_top_k_sources(
        mutated_target,
        mutated,
        feature_cols=("sales",),
        k=1,
        group_cols=("brand_id", "item_id"),
        weight_mode="inverse_distance",
    )

    assert baseline["meta"]["d2_source_authority_digest"]
    assert baseline["meta"]["d2_consumer_frame_fingerprint"]
    assert baseline["meta"]["d2_sealed_identity"]
    assert baseline["meta"]["candidate_pool_digest"] != changed["meta"]["candidate_pool_digest"]
    assert baseline["meta"]["d2_sealed_identity"] != changed["meta"]["d2_sealed_identity"]


def test_d2_candidate_digest_includes_source_identity_fields() -> None:
    payload = {
        "protocol_version": "d1_d6_protocol_v1",
        "dataset_id": "D2",
        "scenario": "with",
        "target_key": ("1", "10"),
        "group_cols": ("brand_id", "item_id"),
        "candidate_keys": (("1", "1"),),
        "observed_start": "2018-06-01",
        "observed_end": "2018-06-30",
        "feature_cols": ("sales",),
    }
    baseline = build_candidate_pool_digest(**payload)
    with_identity = build_candidate_pool_digest(
        **payload,
        calendarization_rule_version="d2_source_calendarization_v1",
        source_authority_digest="a" * 64,
        consumer_frame_fingerprint="b" * 64,
    )
    assert baseline != with_identity


def test_legacy_candidate_digest_fixture_remains_unchanged() -> None:
    assert (
        build_candidate_pool_digest(
            protocol_version="d1_d6_protocol_v1",
            dataset_id="D1",
            scenario="without",
            target_key=("Store1", "Item10"),
            group_cols=("store", "item"),
            candidate_keys=(("Store1", "Item1"), ("Store1", "Item2")),
            observed_start="2017-06-05",
            observed_end="2017-07-04",
            feature_cols=("sales",),
        )
        == "7d7e0e0d6a08841426df0cea2273e420ae5d4b4dbc12c4c36e5cbf21e1328c72"
    )
