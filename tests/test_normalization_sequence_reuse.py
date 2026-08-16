from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler

from src.data_processing.data_preprocessing import build_tabular_sequence, normalize_features
from src.protocols.experiment_protocol import ProtocolViolation
from src.protocols.transformation_identity import (
    CELL_IDENTITY_ATTR,
    NORMALIZATION_EVIDENCE_ATTR,
    RFE_STAGE_IDENTITY_ATTR,
    RFEStageIdentity,
    build_normalization_identity,
    exact_array_digest,
    scaler_parameter_evidence,
)
from src.protocols.transformation_reuse import TargetTransformationReuseContext
from src.protocols.provenance import (
    assert_actual_cnn_training_validated,
    bind_actual_cnn_source_frame,
)


FEATURES = ("sales", "year", "constant")
LIFECYCLE = ("D1", "with_information_sharing", 1, 42, ("1", "10"))


def _parts(
    *,
    dataset_id: str = "D1",
    source_key: tuple[str, ...] | None = None,
    target_key: tuple[str, ...] = ("1", "10"),
    value_offset: float = 0.0,
    rfe_token: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    days = 18
    frame = pd.DataFrame(
        {
            "entity_id": [1] * days,
            "item_id": [10] * days,
            "date": pd.date_range("2024-01-01", periods=days, freq="D"),
            "sales": np.arange(1, days + 1, dtype=np.float64) + value_offset,
            "year": np.asarray([2024 + (i % 2) for i in range(days)], dtype=np.float64),
            "constant": np.asarray([5] * days, dtype=np.float64),
        }
    )
    lifecycle = (dataset_id, "with_information_sharing", 1, 42, target_key)
    frame.attrs.update(
        {
            CELL_IDENTITY_ATTR: lifecycle,
            "protocol_dataset_id": dataset_id,
            "protocol_scenario": "with_information_sharing",
            "protocol_target_key": target_key,
            "protocol_group_cols": ("entity_id", "item_id"),
            "protocol_version": "strict_paper_v1",
            "protocol_track": "strict_paper",
            "split_role": "source" if source_key is not None else "target",
            "split_mode": "ratio" if source_key is not None else "days",
            "split_config": (
                {"train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1}
                if source_key is not None
                else {"train_days": 10, "val_days": 4, "test_days": 4}
            ),
        }
    )
    if rfe_token is not None:
        frame.attrs[RFE_STAGE_IDENTITY_ATTR] = RFEStageIdentity(
            schema_version="rfe-stage-v1",
            stage="POST_RFE",
            selected_feature_cols=FEATURES,
            rfe_protocol_identity=f"rfe-protocol-{rfe_token}",
            joint_train_identity=f"joint-{rfe_token}",
            estimator_identity="random_forest",
            estimator_config=(("n_estimators", 10), ("step", 1)),
            random_state=42,
            keep_ratio=0.5,
            selection_evidence_digest=f"selection-{rfe_token}",
        )
    result = []
    for role, slc in (
        ("train", slice(0, 10)),
        ("validation", slice(10, 14)),
        ("test", slice(14, 18)),
    ):
        part = frame.iloc[slc].copy()
        part.attrs = dict(frame.attrs)
        part.attrs["temporal_partition"] = role
        result.append(part)
    return tuple(result)  # type: ignore[return-value]


def _normalization_request(
    context: TargetTransformationReuseContext,
    parts: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
):
    return normalize_features(
        *parts,
        feature_columns=FEATURES,
        reuse_context=context,
    )


def test_normalization_context_builds_once_and_returns_private_frames_and_scalers() -> None:
    context = TargetTransformationReuseContext(LIFECYCLE)
    raw = _parts()
    legacy = normalize_features(*raw, feature_columns=FEATURES)
    first = _normalization_request(context, raw)
    second = _normalization_request(context, raw)
    assert context.counts() == {
        "normalization_requests": 2,
        "normalization_heavy_builds": 1,
        "normalization_hits": 1,
        "normalization_misses": 1,
        "normalization_consumer_frame_copies": 6,
        "normalization_consumer_scaler_copies": 2,
        "sequence_requests": 0,
        "sequence_heavy_builds": 0,
        "sequence_hits": 0,
        "sequence_misses": 0,
        "sequence_consumer_array_copies": 0,
    }
    for left, right in zip(first[:3], second[:3]):
        assert left is not right
        assert not np.shares_memory(left.to_numpy(), right.to_numpy())
        pd.testing.assert_frame_equal(left, right, check_exact=True)
    canonical = next(iter(context._normalizations.values()))
    for consumer, template in zip(
        first[:3],
        (
            canonical._train_template,
            canonical._validation_template,
            canonical._test_template,
        ),
    ):
        assert consumer is not template
        assert not np.shares_memory(consumer.to_numpy(), template.to_numpy())
    for expected, actual in zip(legacy[:3], first[:3]):
        pd.testing.assert_frame_equal(expected, actual, check_exact=True)
        assert tuple(actual.columns) == tuple(expected.columns)
        assert tuple(str(dtype) for dtype in actual.dtypes) == tuple(str(dtype) for dtype in expected.dtypes)
    assert np.array_equal(first[0]["constant"].to_numpy(), np.zeros(len(first[0])))
    scaler_a, scaler_b = first[3], second[3]
    assert scaler_a is not scaler_b
    assert scaler_a is not canonical._scaler_template
    assert scaler_parameter_evidence(scaler_a) == scaler_parameter_evidence(scaler_b)
    assert scaler_parameter_evidence(scaler_a) == scaler_parameter_evidence(legacy[3])
    scaler_b_before = scaler_parameter_evidence(scaler_b)
    probe = raw[0][list(FEATURES)]
    assert np.array_equal(scaler_a.transform(probe), scaler_b.transform(probe))
    transformed = scaler_a.transform(probe)
    assert np.array_equal(scaler_a.inverse_transform(transformed), scaler_b.inverse_transform(transformed))

    first[0].iloc[0, first[0].columns.get_loc("sales")] = -999.0
    first[0]["consumer_only"] = 1
    first[0].drop(first[0].index[-1], inplace=True)
    first[0].attrs["method"] = "mutated"
    scaler_a.scale_[0] = -999.0
    scaler_a.min_[0] = -999.0
    scaler_a.data_min_[0] = -999.0
    third = _normalization_request(context, raw)
    assert "consumer_only" not in third[0]
    assert len(third[0]) == len(raw[0])
    assert third[0].iloc[0]["sales"] != -999.0
    assert third[0].attrs.get("method") != "mutated"
    assert third[3].scale_[0] != -999.0
    assert third[3].min_[0] != -999.0
    assert third[3].data_min_[0] != -999.0
    assert scaler_parameter_evidence(scaler_b) == scaler_b_before


def test_normalization_different_identity_misses_and_canonical_owner_is_not_exposed() -> None:
    context = TargetTransformationReuseContext(LIFECYCLE)
    base = _parts()
    changed = _parts(value_offset=100.0)
    first = _normalization_request(context, base)
    _normalization_request(context, changed)
    assert context.normalization_heavy_builds == 2
    assert context.normalization_hits == 0

    canonical = context._normalizations[first[0].attrs[NORMALIZATION_EVIDENCE_ATTR].identity]
    assert all(
        consumer is not template
        and not np.shares_memory(consumer.to_numpy(), template.to_numpy())
        for consumer, template in zip(
            first[:3],
            (
                canonical._train_template,
                canonical._validation_template,
                canonical._test_template,
            ),
        )
    )
    assert first[3] is not canonical._scaler_template
    for name in ("data_min_", "data_max_", "data_range_", "scale_", "min_"):
        assert not np.shares_memory(
            np.asarray(getattr(first[3], name)),
            np.asarray(getattr(canonical._scaler_template, name)),
        )


def test_normalization_miss_exactly_validates_builder_output_before_insertion() -> None:
    context = TargetTransformationReuseContext(LIFECYCLE)
    raw = _parts()
    identity = build_normalization_identity(
        *raw,
        feature_cols=FEATURES,
        scaler=MinMaxScaler(),
    )

    def corrupted_builder():
        result = normalize_features(*raw, feature_columns=FEATURES)
        result[0].iloc[0, result[0].columns.get_loc("sales")] = -123.0
        return result

    with pytest.raises(ProtocolViolation, match="normalized frame evidence mismatch"):
        context.normalize(
            identity=identity,
            raw_frames=raw,
            heavy_builder=corrupted_builder,
        )
    assert not context._normalizations


def test_sequence_context_is_readonly_canonical_and_writable_copy_on_consume() -> None:
    context = TargetTransformationReuseContext(LIFECYCLE)
    train, _, _, _, _ = _normalization_request(context, _parts())
    legacy_x, legacy_y = build_tabular_sequence(
        train, horizon=1, window_size=2, feature_columns=FEATURES
    )
    first = build_tabular_sequence(
        train,
        horizon=1,
        window_size=2,
        feature_columns=FEATURES,
        reuse_context=context,
    )
    second = build_tabular_sequence(
        train,
        horizon=1,
        window_size=2,
        feature_columns=FEATURES,
        reuse_context=context,
    )
    assert context.sequence_requests == 2
    assert context.sequence_heavy_builds == 1
    assert context.sequence_hits == 1
    canonical = next(iter(context._sequences.values()))
    assert canonical.template_writeable_flags == (False, False)
    assert first[0].flags.writeable and first[1].flags.writeable
    assert second[0].flags.writeable and second[1].flags.writeable
    assert not np.shares_memory(first[0], second[0])
    assert not np.shares_memory(first[1], second[1])
    assert np.array_equal(first[0], legacy_x)
    assert np.array_equal(first[1], legacy_y)
    x_digest = exact_array_digest(second[0])
    y_digest = exact_array_digest(second[1])
    first[0][0, 0, 0] = -999.0
    first[1][0] = -999.0
    third = build_tabular_sequence(
        train,
        horizon=1,
        window_size=2,
        feature_columns=FEATURES,
        reuse_context=context,
    )
    assert exact_array_digest(third[0]) == x_digest
    assert exact_array_digest(third[1]) == y_digest


def test_sequence_detects_normalized_working_frame_and_canonical_corruption() -> None:
    context = TargetTransformationReuseContext(LIFECYCLE)
    train, _, _, _, _ = _normalization_request(context, _parts())
    build_tabular_sequence(
        train, horizon=1, window_size=2, feature_columns=FEATURES, reuse_context=context
    )
    corrupted_work = deepcopy(train)
    corrupted_work.loc[corrupted_work.index[0], "sales"] = -5.0
    with pytest.raises(ProtocolViolation, match="normalized frame evidence mismatch"):
        build_tabular_sequence(
            corrupted_work,
            horizon=1,
            window_size=2,
            feature_columns=FEATURES,
            reuse_context=context,
        )
    canonical = next(iter(context._sequences.values()))
    canonical._x_template.flags.writeable = True
    canonical._x_template[0, 0, 0] = -7.0
    canonical._x_template.flags.writeable = False
    with pytest.raises(ProtocolViolation, match="SEQUENCE_REUSE_EVIDENCE_MISMATCH"):
        build_tabular_sequence(
            train, horizon=1, window_size=2, feature_columns=FEATURES, reuse_context=context
        )


def test_sequence_hit_preserves_actual_cnn_provenance_without_rebuilding_windows() -> None:
    context = TargetTransformationReuseContext(LIFECYCLE)
    raw = _parts(source_key=("1", "10"))
    for part in raw:
        bind_actual_cnn_source_frame(
            part,
            source_key=("1", "10"),
            group_cols=("entity_id", "item_id"),
            feature_cols=FEATURES,
        )
    first_train, _, _, _, _ = _normalization_request(context, raw)
    build_tabular_sequence(
        first_train, horizon=1, window_size=2,
        feature_columns=FEATURES, reuse_context=context,
    )
    assert_actual_cnn_training_validated(first_train, source_key=("1", "10"))

    second_train, _, _, _, _ = _normalization_request(context, raw)
    build_tabular_sequence(
        second_train, horizon=1, window_size=2,
        feature_columns=FEATURES, reuse_context=context,
    )
    assert context.sequence_heavy_builds == 1
    assert context.sequence_hits == 1
    assert_actual_cnn_training_validated(second_train, source_key=("1", "10"))


def _run_full_transfer_count_fixture(
    *,
    overlap_ss_with_k3_a: bool,
    dataset_id: str = "D1",
    target_key: tuple[str, ...] = ("1", "10"),
):
    lifecycle = (dataset_id, "with_information_sharing", 1, 42, target_key)
    context = TargetTransformationReuseContext(lifecycle)
    groups = {
        "target": _parts(dataset_id=dataset_id, target_key=target_key),
        "k3_a": _parts(dataset_id=dataset_id, target_key=target_key, source_key=("A",), value_offset=10.0),
        "k3_b": _parts(dataset_id=dataset_id, target_key=target_key, source_key=("B",), value_offset=20.0),
        "k3_c": _parts(dataset_id=dataset_id, target_key=target_key, source_key=("C",), value_offset=30.0),
        "ss": _parts(
            dataset_id=dataset_id,
            target_key=target_key,
            source_key=("A",) if overlap_ss_with_k3_a else ("SS",),
            value_offset=10.0 if overlap_ss_with_k3_a else 40.0,
        ),
        "rfe_a": _parts(dataset_id=dataset_id, target_key=target_key, source_key=("A",), value_offset=10.0, rfe_token="shared"),
        "rfe_b": _parts(dataset_id=dataset_id, target_key=target_key, source_key=("B",), value_offset=20.0, rfe_token="shared"),
        "rfe_c": _parts(dataset_id=dataset_id, target_key=target_key, source_key=("C",), value_offset=30.0, rfe_token="shared"),
        "rfe_target": _parts(dataset_id=dataset_id, target_key=target_key, rfe_token="shared"),
    }
    request_multiplicity = {
        "target": 8,
        "k3_a": 3,
        "k3_b": 3,
        "k3_c": 3,
        "ss": 1,
        "rfe_a": 1,
        "rfe_b": 1,
        "rfe_c": 1,
        "rfe_target": 1,
    }
    normalized = {}
    for name, multiplicity in request_multiplicity.items():
        for _ in range(multiplicity):
            normalized[name] = _normalization_request(context, groups[name])

    for role_index in range(3):
        for _ in range(8):
            build_tabular_sequence(
                normalized["target"][role_index],
                horizon=1,
                window_size=2,
                feature_columns=FEATURES,
                reuse_context=context,
            )
    for name in ("k3_a", "k3_b", "k3_c"):
        for _ in range(3):
            build_tabular_sequence(
                normalized[name][0],
                horizon=1,
                window_size=2,
                feature_columns=FEATURES,
                reuse_context=context,
            )
    build_tabular_sequence(
        normalized["ss"][0], horizon=1, window_size=2,
        feature_columns=FEATURES, reuse_context=context,
    )
    for name in ("rfe_a", "rfe_b", "rfe_c"):
        build_tabular_sequence(
            normalized[name][0], horizon=1, window_size=2,
            feature_columns=FEATURES, reuse_context=context,
        )
    for role_index in range(3):
        build_tabular_sequence(
            normalized["rfe_target"][role_index], horizon=1, window_size=2,
            feature_columns=FEATURES, reuse_context=context,
        )
    return context, groups


@pytest.mark.parametrize("dataset_id", ("D1", "D2", "D3"))
def test_conservative_transfer_counts_are_exactly_22_to_9_and_40_to_13(
    monkeypatch,
    dataset_id: str,
) -> None:
    fit_calls = 0
    transform_calls = 0
    original_fit = MinMaxScaler.fit
    original_transform = MinMaxScaler.transform

    def counted_fit(self, *args, **kwargs):
        nonlocal fit_calls
        fit_calls += 1
        return original_fit(self, *args, **kwargs)

    def counted_transform(self, *args, **kwargs):
        nonlocal transform_calls
        transform_calls += 1
        return original_transform(self, *args, **kwargs)

    monkeypatch.setattr(MinMaxScaler, "fit", counted_fit)
    monkeypatch.setattr(MinMaxScaler, "transform", counted_transform)
    context, groups = _run_full_transfer_count_fixture(
        overlap_ss_with_k3_a=False,
        dataset_id=dataset_id,
    )
    assert context.normalization_requests == 22
    assert context.normalization_heavy_builds == 9
    assert context.normalization_hits == 13
    assert context.normalization_misses == 9
    assert fit_calls == 9
    assert transform_calls == 27
    assert context.normalization_consumer_frame_copies == 66
    assert context.normalization_consumer_scaler_copies == 22
    assert context.sequence_requests == 40
    assert context.sequence_heavy_builds == 13
    assert context.sequence_hits == 27
    assert context.sequence_misses == 13
    assert context.sequence_consumer_array_copies == 80

    notl = normalize_features(*groups["target"], feature_columns=FEATURES)
    for frame in notl[:3]:
        build_tabular_sequence(frame, horizon=1, window_size=2, feature_columns=FEATURES)
    assert fit_calls == 10
    assert transform_calls == 30
    assert context.normalization_requests == 22
    assert context.sequence_requests == 40


def test_runtime_identity_overlap_naturally_reduces_to_8_and_12() -> None:
    context, _ = _run_full_transfer_count_fixture(overlap_ss_with_k3_a=True)
    assert context.normalization_requests == 22
    assert context.normalization_heavy_builds == 8
    assert context.normalization_hits == 14
    assert context.sequence_requests == 40
    assert context.sequence_heavy_builds == 12
    assert context.sequence_hits == 28


def test_rfe_cross_target_and_cross_cell_isolation_fail_closed() -> None:
    context = TargetTransformationReuseContext(LIFECYCLE)
    ordinary = _normalization_request(context, _parts())
    rfe = _normalization_request(context, _parts(rfe_token="same-features"))
    assert context.normalization_heavy_builds == 2
    assert ordinary[0].attrs[NORMALIZATION_EVIDENCE_ATTR].identity != rfe[0].attrs[NORMALIZATION_EVIDENCE_ATTR].identity

    other_target = _parts(target_key=("1", "11"))
    with pytest.raises(ProtocolViolation, match="CONTEXT_OWNER_MISMATCH"):
        _normalization_request(context, other_target)
    for changed in (
        ("D1", "without_information_sharing", 1, 42, ("1", "10")),
        ("D1", "with_information_sharing", 5, 42, ("1", "10")),
        ("D1", "with_information_sharing", 1, 43, ("1", "10")),
    ):
        assert TargetTransformationReuseContext(changed).lifecycle_identity != context.lifecycle_identity


@pytest.mark.parametrize("dataset_id", ("D4", "D5", "D6"))
def test_five_target_cell_counts_use_five_independent_contexts(dataset_id: str) -> None:
    contexts = []
    for index in range(5):
        context, _ = _run_full_transfer_count_fixture(
            overlap_ss_with_k3_a=False,
            dataset_id=dataset_id,
            target_key=("target", str(index)),
        )
        contexts.append(context)
    assert len({id(context) for context in contexts}) == 5
    assert sum(context.normalization_requests for context in contexts) == 110
    assert sum(context.normalization_heavy_builds for context in contexts) == 45
    assert sum(context.sequence_requests for context in contexts) == 200
    assert sum(context.sequence_heavy_builds for context in contexts) == 65
    assert 110 + 5 == 115
    assert 45 + 5 == 50
    assert 200 + 15 == 215
    assert 65 + 15 == 80


def test_context_has_no_global_persistent_or_attrs_cache_contract() -> None:
    first = TargetTransformationReuseContext(LIFECYCLE)
    second = TargetTransformationReuseContext(LIFECYCLE)
    normalized = _normalization_request(first, _parts())
    assert first._normalizations is not second._normalizations
    assert first._sequences is not second._sequences
    assert not any("cache" in key or "reuse_context" in key for frame in normalized[:3] for key in frame.attrs)
    assert not hasattr(TargetTransformationReuseContext, "__wrapped__")
