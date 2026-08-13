from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, is_dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler

from src.data_processing.data_preprocessing import (
    build_tabular_sequence,
    normalize_features,
    to_cnn_tensor,
)
from src.protocols.experiment_protocol import ProtocolViolation
from src.protocols.raw_preprocessing import RawPreprocessingIdentity
from src.protocols.transformation_identity import (
    CELL_IDENTITY_ATTR,
    FILL_POLICY_IDENTITY_ATTR,
    KNN_FEATURE_CONTRACTS,
    MODEL_FEATURE_CONTRACTS,
    NORMALIZATION_EVIDENCE_ATTR,
    RFE_STAGE_IDENTITY_ATTR,
    SEQUENCE_EVIDENCE_ATTR,
    FillPolicyIdentity,
    NormalizationEvidence,
    RFEStageIdentity,
    RawPartitionIdentity,
    ScalerAlgorithmIdentity,
    SequenceEvidence,
    build_normalization_identity,
    build_raw_partition_identity,
    canonical_serialize,
    exact_array_digest,
    require_same_identity,
    scaler_algorithm_identity,
    semantic_digest,
    validate_runtime_feature_contract,
)
from src.utils.source_fillna import fill_source_numeric_na


FEATURES = ("sales", "year", "constant")


def _frame(*, dtype: str = "float64", days: int = 18) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "entity_id": [1] * days,
            "item_id": [10] * days,
            "date": pd.date_range("2024-01-01", periods=days, freq="D"),
            "sales": np.arange(1, days + 1, dtype=dtype),
            "year": np.asarray([2024 + (i % 2) for i in range(days)], dtype=dtype),
            "constant": np.asarray([5] * days, dtype=dtype),
        }
    )
    frame.attrs.update(
        {
            CELL_IDENTITY_ATTR: ("D1", "with_information_sharing", 1, 42, ("1", "10")),
            "protocol_dataset_id": "D1",
            "protocol_scenario": "with_information_sharing",
            "protocol_target_key": ("1", "10"),
            "protocol_group_cols": ("entity_id", "item_id"),
            "protocol_version": "strict_paper_v1",
            "protocol_track": "strict_paper",
            "split_role": "target",
            "split_mode": "days",
            "split_config": {"train_days": 10, "val_days": 4, "test_days": 4},
        }
    )
    return frame


def _partitions(*, dtype: str = "float64") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = _frame(dtype=dtype)
    inherited = dict(raw.attrs)
    result = []
    for role, slc in (("train", slice(0, 10)), ("validation", slice(10, 14)), ("test", slice(14, 18))):
        part = raw.iloc[slc].copy()
        part.attrs = dict(inherited)
        part.attrs["temporal_partition"] = role
        result.append(part)
    return tuple(result)  # type: ignore[return-value]


def _normalized(
    *,
    dtype: str = "float64",
    features: tuple[str, ...] = FEATURES,
) -> tuple[
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    MinMaxScaler,
    NormalizationEvidence,
]:
    raw = _partitions(dtype=dtype)
    train, val, test, scaler, _ = normalize_features(*raw, feature_columns=features)
    scaled = (train, val, test)
    evidence = train.attrs[NORMALIZATION_EVIDENCE_ATTR]
    assert evidence is val.attrs[NORMALIZATION_EVIDENCE_ATTR]
    assert evidence is test.attrs[NORMALIZATION_EVIDENCE_ATTR]
    return raw, scaled, scaler, evidence


def _assert_no_mutable_payload(value: object) -> None:
    assert not isinstance(value, (list, dict, np.ndarray, pd.DataFrame, MinMaxScaler))
    if is_dataclass(value):
        for info in fields(value):
            _assert_no_mutable_payload(getattr(value, info.name))
    elif isinstance(value, tuple):
        for item in value:
            _assert_no_mutable_payload(item)


def _post_rfe(features: tuple[str, ...] = FEATURES) -> RFEStageIdentity:
    return RFEStageIdentity(
        schema_version="rfe-stage-v1",
        stage="POST_RFE",
        selected_feature_cols=features,
        rfe_protocol_identity="sklearn.feature_selection.RFE_step1_v1",
        joint_train_identity="joint-digest",
        estimator_identity="random_forest",
        estimator_config=(("n_estimators", 10), ("step", 1)),
        random_state=42,
        keep_ratio=0.5,
        selection_evidence_digest="selection-digest",
    )


def test_canonical_serialization_is_stable_versioned_and_mapping_order_independent() -> None:
    left = {"z": (2, 3), "a": {"n": 1}}
    right = {"a": {"n": 1}, "z": (2, 3)}
    assert canonical_serialize(left) == canonical_serialize(left)
    assert canonical_serialize(left) == canonical_serialize(right)
    assert semantic_digest(left) == semantic_digest(right)
    assert canonical_serialize(("a", "b")) != canonical_serialize(["a", "b"])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_serialization_rejects_nonfinite_values(value: float) -> None:
    with pytest.raises(ProtocolViolation, match="NaN and infinity"):
        canonical_serialize(value)
    with pytest.raises(ProtocolViolation, match="NaN and infinity"):
        canonical_serialize(np.asarray([value], dtype=np.float64))


def test_raw_partition_identity_canonicalizes_incoming_order_and_rejects_mutations() -> None:
    train, _, _ = _partitions()
    shuffled = train.sample(frac=1.0, random_state=7)
    shuffled.attrs = dict(train.attrs)
    identity = build_raw_partition_identity(train, feature_cols=FEATURES)
    shuffled_identity = build_raw_partition_identity(shuffled, feature_cols=FEATURES)
    assert identity == shuffled_identity

    less = train.iloc[:-1].copy()
    less.attrs = dict(train.attrs)
    assert build_raw_partition_identity(less, feature_cols=FEATURES) != identity

    changed_entity = train.copy()
    changed_entity.attrs = dict(train.attrs)
    changed_entity.loc[changed_entity.index[0], "entity_id"] = 2
    assert build_raw_partition_identity(changed_entity, feature_cols=FEATURES) != identity

    changed_date = train.copy()
    changed_date.attrs = dict(train.attrs)
    changed_date.loc[changed_date.index[0], "date"] = pd.Timestamp("2023-12-31")
    assert build_raw_partition_identity(changed_date, feature_cols=FEATURES) != identity


def test_raw_partition_identity_rejects_duplicate_missing_and_invalid_dates() -> None:
    train, _, _ = _partitions()
    duplicate = pd.concat([train, train.iloc[[0]]], ignore_index=True)
    duplicate.attrs = dict(train.attrs)
    with pytest.raises(ProtocolViolation, match="duplicate group/date"):
        build_raw_partition_identity(duplicate, feature_cols=FEATURES)

    missing = train.copy()
    missing.attrs = dict(train.attrs)
    missing.loc[missing.index[0], "date"] = None
    with pytest.raises(ProtocolViolation, match="missing dates"):
        build_raw_partition_identity(missing, feature_cols=FEATURES)

    invalid = train.copy()
    invalid.attrs = dict(train.attrs)
    invalid["date"] = invalid["date"].astype(object)
    invalid.loc[invalid.index[0], "date"] = "not-a-date"
    with pytest.raises(ProtocolViolation, match="invalid dates"):
        build_raw_partition_identity(invalid, feature_cols=FEATURES)


def test_column_feature_dtype_fill_split_and_cell_contracts_are_identity_fields() -> None:
    train, _, _ = _partitions()
    base = build_raw_partition_identity(train, feature_cols=FEATURES)

    reordered = train.loc[:, ["entity_id", "item_id", "date", "year", "sales", "constant"]].copy()
    reordered.attrs = dict(train.attrs)
    assert build_raw_partition_identity(reordered, feature_cols=FEATURES) != base
    assert build_raw_partition_identity(train, feature_cols=("sales", "constant", "year")) != base

    float32 = train.copy()
    float32.attrs = dict(train.attrs)
    float32["sales"] = float32["sales"].astype(np.float32)
    assert build_raw_partition_identity(float32, feature_cols=FEATURES) != base

    filled = train.copy()
    filled.attrs = dict(train.attrs)
    filled.attrs[FILL_POLICY_IDENTITY_ATTR] = FillPolicyIdentity(
        "fill-policy-v1", "different_helper", (), "numeric_zero_exact", "preserve_numeric_dtype"
    )
    assert build_raw_partition_identity(filled, feature_cols=FEATURES) != base

    validation_role = train.copy()
    validation_role.attrs = dict(train.attrs)
    validation_role.attrs["temporal_partition"] = "validation"
    assert build_raw_partition_identity(validation_role, feature_cols=FEATURES) != base

    boundary = train.copy()
    boundary.attrs = dict(train.attrs)
    boundary.attrs["split_config"] = {"train_days": 9, "val_days": 5, "test_days": 4}
    assert build_raw_partition_identity(boundary, feature_cols=FEATURES) != base

    other_cell = train.copy()
    other_cell.attrs = dict(train.attrs)
    other_cell.attrs[CELL_IDENTITY_ATTR] = ("D1", "without_information_sharing", 5, 46, ("1", "10"))
    assert build_raw_partition_identity(other_cell, feature_cols=FEATURES) != base


def test_source_fill_policy_sidecar_records_real_fill_and_coercion() -> None:
    frame = _frame(days=3)
    frame.loc[frame.index[0], "sales"] = np.nan
    frame["numeric_text"] = pd.Series(["1", None, "3"], dtype=object)
    filled = fill_source_numeric_na(frame, feature_columns=("sales", "numeric_text"))
    policy = filled.attrs[FILL_POLICY_IDENTITY_ATTR]
    assert policy.filled_columns == ("sales", "numeric_text")
    assert "numeric_text" in policy.numeric_coercion_identity
    assert filled[["sales", "numeric_text"]].isna().sum().sum() == 0


def test_scaler_algorithm_identity_binds_config_and_versions() -> None:
    base = scaler_algorithm_identity(MinMaxScaler())
    clipped = scaler_algorithm_identity(MinMaxScaler(clip=True))
    ranged = scaler_algorithm_identity(MinMaxScaler(feature_range=(-1, 1)))
    versioned = scaler_algorithm_identity(MinMaxScaler(), sklearn_version="future-version")
    numpy_versioned = scaler_algorithm_identity(MinMaxScaler(), numpy_version="future-numpy")
    assert len({base, clipped, ranged, versioned, numpy_versioned}) == 5


def test_normalization_evidence_exact_parity_and_constant_feature_contract() -> None:
    raw, scaled, scaler, evidence = _normalized()
    assert evidence.fit_row_count == len(raw[0])
    assert evidence.constant_feature_cols == ("constant",)
    assert evidence.scaler_parameters.data_min == (1.0, 2024.0, 5.0)
    assert evidence.scaler_parameters.data_max == (10.0, 2025.0, 5.0)
    assert evidence.scaler_parameters.data_range == (9.0, 1.0, 0.0)
    assert evidence.scaler_parameters.scale[-1] == 1.0
    assert evidence.scaler_parameters.min_offset[-1] == -5.0
    assert np.array_equal(scaled[0]["constant"].to_numpy(), np.zeros(10))
    assert evidence.scaler_parameters.n_samples_seen == 10
    assert evidence.scaler_parameters.n_features_in == 3
    assert evidence.scaler_parameters.feature_names_in == FEATURES
    expected_train = scaled[0].loc[:, list(FEATURES)].to_numpy()
    assert evidence.train.exact_values_digest == exact_array_digest(expected_train)
    assert evidence.finite is True
    assert scaler is not evidence.scaler_parameters


def test_normalization_identity_is_order_stable_but_feature_dtype_and_fit_scope_sensitive() -> None:
    raw, _, scaler, evidence = _normalized()
    shuffled = []
    for part in raw:
        copy = part.sample(frac=1.0, random_state=9)
        copy.attrs = dict(part.attrs)
        shuffled.append(copy)
    shuffled_identity = build_normalization_identity(
        *shuffled, feature_cols=FEATURES, scaler=MinMaxScaler().fit(shuffled[0][list(FEATURES)])
    )
    assert shuffled_identity == evidence.identity

    reordered = build_normalization_identity(
        *raw,
        feature_cols=("sales", "constant", "year"),
        scaler=MinMaxScaler().fit(raw[0][["sales", "constant", "year"]]),
    )
    assert reordered != evidence.identity

    raw32 = _partitions(dtype="float32")
    dtype_identity = build_normalization_identity(
        *raw32, feature_cols=FEATURES, scaler=MinMaxScaler().fit(raw32[0][list(FEATURES)])
    )
    assert dtype_identity != evidence.identity

    train_plus_val = replace(evidence.identity, fit_scope="train_plus_validation")
    assert train_plus_val != evidence.identity
    with pytest.raises(ProtocolViolation, match="train_only"):
        build_normalization_identity(
            *raw, feature_cols=FEATURES, scaler=scaler, fit_scope="train_plus_validation"
        )


def test_rfe_stage_identity_separates_non_pre_post_and_all_post_contract_fields() -> None:
    non = RFEStageIdentity.non_rfe()
    pre = RFEStageIdentity.pre_rfe(protocol_identity="rfe-protocol-v1")
    post = _post_rfe()
    assert len({non, pre, post}) == 3
    assert replace(post, selected_feature_cols=("sales", "constant")) != post
    assert replace(post, random_state=43) != post
    assert replace(post, keep_ratio=0.75) != post
    assert replace(post, estimator_config=(("n_estimators", 20), ("step", 1))) != post
    assert replace(post, joint_train_identity="other-joint") != post
    with pytest.raises(ProtocolViolation, match="complete selection evidence"):
        RFEStageIdentity("rfe-stage-v1", "POST_RFE", selected_feature_cols=("sales",))


def test_normalization_rejects_mixed_rfe_stages_and_post_never_matches_ordinary() -> None:
    train, val, test = _partitions()
    scaler = MinMaxScaler().fit(train[list(FEATURES)])
    ordinary = build_normalization_identity(train, val, test, feature_cols=FEATURES, scaler=scaler)
    post = _post_rfe()
    for part in (train, val, test):
        part.attrs[RFE_STAGE_IDENTITY_ATTR] = post
    rfe = build_normalization_identity(train, val, test, feature_cols=FEATURES, scaler=scaler)
    assert rfe != ordinary
    val.attrs[RFE_STAGE_IDENTITY_ATTR] = RFEStageIdentity.non_rfe()
    with pytest.raises(ProtocolViolation, match="mismatched RFE"):
        build_normalization_identity(train, val, test, feature_cols=FEATURES, scaler=scaler)


def test_sequence_evidence_exact_X_y_sample_and_label_alignment() -> None:
    _, scaled, _, _ = _normalized()
    train = scaled[0]
    X, y = build_tabular_sequence(train, horizon=2, window_size=3, feature_columns=FEATURES)
    evidence = train.attrs[SEQUENCE_EVIDENCE_ATTR]
    assert isinstance(evidence, SequenceEvidence)
    values = train.loc[:, list(FEATURES)].to_numpy(dtype=np.float32)
    expected_X = np.asarray([values[i : i + 3] for i in range(0, 6)], dtype=np.float32)
    expected_y = train["sales"].to_numpy(dtype=np.float32)[4:10]
    assert np.array_equal(X, expected_X)
    assert np.array_equal(y, expected_y)
    assert evidence.x_exact_digest == exact_array_digest(X)
    assert evidence.y_exact_digest == exact_array_digest(y)
    assert evidence.sample_count == 6
    assert evidence.first_sample.window_start == "2024-01-01"
    assert evidence.first_sample.window_end == "2024-01-03"
    assert evidence.first_sample.label_date == "2024-01-05"
    assert evidence.last_sample.label_date == "2024-01-10"
    assert evidence.first_sample.group_key == (1, 10)


def test_sequence_identity_changes_for_window_horizon_feature_target_dtype_and_partition() -> None:
    _, scaled, _, _ = _normalized()
    train = scaled[0]
    build_tabular_sequence(train, horizon=1, window_size=2, feature_columns=FEATURES)
    base = train.attrs[SEQUENCE_EVIDENCE_ATTR].identity
    build_tabular_sequence(train, horizon=2, window_size=2, feature_columns=FEATURES)
    horizon = train.attrs[SEQUENCE_EVIDENCE_ATTR].identity
    build_tabular_sequence(train, horizon=1, window_size=3, feature_columns=FEATURES)
    window = train.attrs[SEQUENCE_EVIDENCE_ATTR].identity
    build_tabular_sequence(train, horizon=1, window_size=2, feature_columns=("sales", "constant", "year"))
    feature_order = train.attrs[SEQUENCE_EVIDENCE_ATTR].identity
    assert len({base, horizon, window, feature_order}) == 4
    assert replace(base, target_column="other_target") != base
    assert replace(base, x_dtype="float64") != base
    assert replace(base, partition_role="validation") != base


def test_sequence_incoming_order_is_canonical_and_duplicate_date_fails_closed() -> None:
    raw = list(_partitions())
    for index, part in enumerate(raw):
        shuffled = part.sample(frac=1.0, random_state=11)
        shuffled.attrs = dict(part.attrs)
        raw[index] = shuffled
    train, _, _, _, _ = normalize_features(*raw, feature_columns=FEATURES)
    X, y = build_tabular_sequence(train, horizon=1, window_size=2, feature_columns=FEATURES)
    ordered_raw, ordered_scaled, _, _ = _normalized()
    expected_X, expected_y = build_tabular_sequence(
        ordered_scaled[0], horizon=1, window_size=2, feature_columns=FEATURES
    )
    assert np.array_equal(X, expected_X)
    assert np.array_equal(y, expected_y)
    assert train.attrs[SEQUENCE_EVIDENCE_ATTR].identity == ordered_scaled[0].attrs[SEQUENCE_EVIDENCE_ATTR].identity

    duplicate = pd.concat([ordered_raw[0], ordered_raw[0].iloc[[0]]], ignore_index=True)
    duplicate.attrs = dict(ordered_raw[0].attrs)
    with pytest.raises(ProtocolViolation, match="duplicate group/date"):
        normalize_features(duplicate, ordered_raw[1], ordered_raw[2], feature_columns=FEATURES)


def test_identity_and_evidence_are_frozen_and_contain_no_mutable_payloads() -> None:
    _, scaled, _, evidence = _normalized()
    build_tabular_sequence(scaled[0], horizon=1, window_size=2, feature_columns=FEATURES)
    sequence = scaled[0].attrs[SEQUENCE_EVIDENCE_ATTR]
    with pytest.raises(FrozenInstanceError):
        evidence.fit_row_count = 999  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        sequence.sample_count = 999  # type: ignore[misc]
    _assert_no_mutable_payload(evidence)
    _assert_no_mutable_payload(sequence)


def test_equal_semantics_propagate_equal_distinct_evidence_without_sharing_scaler_or_arrays() -> None:
    _, scaled_a, scaler_a, evidence_a = _normalized()
    _, scaled_b, scaler_b, evidence_b = _normalized()
    assert evidence_a == evidence_b
    assert evidence_a is not evidence_b
    assert scaler_a is not scaler_b
    assert not np.shares_memory(
        scaled_a[0][list(FEATURES)].to_numpy(), scaled_b[0][list(FEATURES)].to_numpy()
    )
    require_same_identity(evidence_a.identity, evidence_b.identity, contract="normalization")
    with pytest.raises(ProtocolViolation, match="identity mismatch"):
        require_same_identity(
            evidence_a.identity,
            replace(evidence_b.identity, fit_scope="train_plus_validation"),
            contract="normalization",
        )


def test_to_cnn_tensor_alias_is_observed_without_mutation_and_copy_on_consume_isolated() -> None:
    X = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    before = exact_array_digest(X)
    tensor = to_cnn_tensor(X)
    assert np.shares_memory(X, tensor)
    assert exact_array_digest(X) == before
    consumer = tensor.copy()
    consumer[0, 0, 0] = -999
    assert exact_array_digest(X) == before


def test_tiny_keras_fit_does_not_mutate_input_ndarrays() -> None:
    tf = pytest.importorskip("tensorflow")
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(123)
    X = np.arange(24, dtype=np.float32).reshape(4, 3, 2) / np.float32(24.0)
    y = np.arange(4, dtype=np.float32)
    x_before = exact_array_digest(X)
    y_before = exact_array_digest(y)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(3, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="sgd", loss="mse")
    model.fit(X, y, epochs=1, batch_size=2, shuffle=False, verbose=0)
    assert exact_array_digest(X) == x_before
    assert exact_array_digest(y) == y_before


def test_legacy_calls_without_explicit_context_remain_private_22_40_and_with_notl_23_43() -> None:
    raw = _partitions()
    scaler_ids = []
    scalers = []
    evidence_ids = []
    normalized_trains = []
    for _ in range(22):
        train, _, _, scaler, _ = normalize_features(*raw, feature_columns=FEATURES)
        scaler_ids.append(id(scaler))
        scalers.append(scaler)
        evidence_ids.append(id(train.attrs[NORMALIZATION_EVIDENCE_ATTR]))
        normalized_trains.append(train)
    assert len(set(scaler_ids)) == 22
    assert len(scalers) == 22
    assert len(set(evidence_ids)) == 22
    assert len({train.attrs[NORMALIZATION_EVIDENCE_ATTR].exact_digest for train in normalized_trains}) == 1

    notl_train, _, _, notl_scaler, _ = normalize_features(*raw, feature_columns=FEATURES)
    scalers.append(notl_scaler)
    normalized_trains.append(notl_train)
    assert len({id(scaler) for scaler in scalers}) == 23

    arrays = []
    for index in range(40):
        frame = normalized_trains[index % len(normalized_trains)]
        X, y = build_tabular_sequence(frame, horizon=1, window_size=2, feature_columns=FEATURES)
        assert X.flags.writeable and y.flags.writeable
        arrays.append((X, y))
    assert len(arrays) == 40
    assert len({id(pair[0]) for pair in arrays}) == 40
    assert len({id(pair[1]) for pair in arrays}) == 40
    assert all(not np.shares_memory(arrays[0][0], pair[0]) for pair in arrays[1:])

    for frame in normalized_trains[-3:]:
        arrays.append(
            build_tabular_sequence(frame, horizon=1, window_size=2, feature_columns=FEATURES)
        )
    assert len(arrays) == 43
    assert len({id(pair[0]) for pair in arrays}) == 43
    assert len({id(pair[1]) for pair in arrays}) == 43


@pytest.mark.parametrize("dataset_id", tuple(MODEL_FEATURE_CONTRACTS))
def test_d1_d6_runtime_model_and_knn_feature_contracts(dataset_id: str) -> None:
    validate_runtime_feature_contract(
        dataset_id,
        model_feature_cols=MODEL_FEATURE_CONTRACTS[dataset_id],
        knn_feature_cols=KNN_FEATURE_CONTRACTS[dataset_id],
    )


def test_d2_promo_and_d5_onpromotion_hard_guards_fail_closed() -> None:
    assert "promo" in KNN_FEATURE_CONTRACTS["D2"]
    assert "promo" not in MODEL_FEATURE_CONTRACTS["D2"]
    assert "onpromotion" in KNN_FEATURE_CONTRACTS["D5"]
    assert "onpromotion" not in MODEL_FEATURE_CONTRACTS["D5"]
    assert "oil_price" in KNN_FEATURE_CONTRACTS["D5"]
    assert "oil_price" in MODEL_FEATURE_CONTRACTS["D5"]
    with pytest.raises(ProtocolViolation, match="model feature identity mismatch"):
        validate_runtime_feature_contract(
            "D2",
            model_feature_cols=(*MODEL_FEATURE_CONTRACTS["D2"], "promo"),
            knn_feature_cols=KNN_FEATURE_CONTRACTS["D2"],
        )
    with pytest.raises(ProtocolViolation, match="KNN feature identity mismatch"):
        validate_runtime_feature_contract(
            "D2",
            model_feature_cols=MODEL_FEATURE_CONTRACTS["D2"],
            knn_feature_cols=("sales",),
        )
    with pytest.raises(ProtocolViolation, match="model feature identity mismatch"):
        validate_runtime_feature_contract(
            "D5",
            model_feature_cols=(*MODEL_FEATURE_CONTRACTS["D5"], "onpromotion"),
            knn_feature_cols=KNN_FEATURE_CONTRACTS["D5"],
        )


def test_d3_runtime_contract_remains_current_sales_only_knn_without_authority_edits() -> None:
    assert KNN_FEATURE_CONTRACTS["D3"] == ("sales",)
    assert MODEL_FEATURE_CONTRACTS["D3"] == (
        "sales", "year", "month", "week", "day", "customers", "open", "promo", "school_holiday"
    )
    repository = Path(__file__).resolve().parents[1]
    assert not any(
        path.name == "test_normalization_sequence_identity.py"
        for path in (repository / "数据集" / "固化数据").glob("**/*")
    )


@pytest.mark.parametrize("dataset_id", tuple(MODEL_FEATURE_CONTRACTS))
def test_c1_representative_normalization_9_and_sequence_13_group_contracts(dataset_id: str) -> None:
    features = MODEL_FEATURE_CONTRACTS[dataset_id]
    days = 18
    data = {
        "entity_id": [1] * days,
        "item_id": [10] * days,
        "date": pd.date_range("2024-01-01", periods=days, freq="D"),
    }
    for offset, feature in enumerate(features):
        data[feature] = np.arange(1 + offset, days + 1 + offset, dtype=np.float64)
    frame = pd.DataFrame(data)
    frame.attrs.update(
        {
            CELL_IDENTITY_ATTR: (dataset_id, "with_information_sharing", 1, 42, ("target",)),
            "protocol_dataset_id": dataset_id,
            "protocol_scenario": "with_information_sharing",
            "protocol_target_key": ("target",),
            "protocol_group_cols": ("entity_id", "item_id"),
            "protocol_version": "strict_paper_v1",
            "protocol_track": "strict_paper" if dataset_id in {"D1", "D2", "D3"} else "extended",
            "split_role": "target",
            "split_mode": "days",
            "split_config": {"train_days": 10, "val_days": 4, "test_days": 4},
        }
    )
    raw_parts = []
    for role, slc in (("train", slice(0, 10)), ("validation", slice(10, 14)), ("test", slice(14, 18))):
        part = frame.iloc[slc].copy()
        part.attrs = dict(frame.attrs)
        part.attrs["temporal_partition"] = role
        raw_parts.append(part)
    train, val, test, _, _ = normalize_features(*raw_parts, feature_columns=features)
    normalization = train.attrs[NORMALIZATION_EVIDENCE_ATTR].identity

    def raw_variant(raw: RawPartitionIdentity, token: str) -> RawPartitionIdentity:
        return replace(
            raw,
            source_key=(token,),
            source_role="source",
            upstream_identity_digest=semantic_digest((dataset_id, token)),
            raw_values_digest=semantic_digest((raw.raw_values_digest, token)),
        )

    def normalization_variant(token: str, *, rfe: bool = False):
        train_identity = raw_variant(normalization.train_partition_identity, token)
        validation_identity = raw_variant(normalization.validation_partition_identity, token)
        test_identity = raw_variant(normalization.test_partition_identity, token)
        return replace(
            normalization,
            train_partition_identity=train_identity,
            validation_partition_identity=validation_identity,
            test_partition_identity=test_identity,
            fit_partition_digest=semantic_digest(train_identity),
            rfe_stage_identity=_post_rfe(features) if rfe else RFEStageIdentity.non_rfe(),
        )

    normalization_groups = [normalization]
    normalization_groups.extend(normalization_variant(key) for key in ("K3-A", "K3-B", "K3-C"))
    normalization_groups.append(normalization_variant("SS"))
    normalization_groups.extend(
        normalization_variant(key, rfe=True) for key in ("K3-A", "K3-B", "K3-C")
    )
    normalization_groups.append(replace(normalization, rfe_stage_identity=_post_rfe(features)))
    assert len(normalization_groups) == 9
    assert len(set(normalization_groups)) == 9

    build_tabular_sequence(train, horizon=1, window_size=2, feature_columns=features)
    base_sequence = train.attrs[SEQUENCE_EVIDENCE_ATTR].identity
    normalized_evidence = train.attrs[NORMALIZATION_EVIDENCE_ATTR]

    def sequence_partition(partition, *, token: str | None = None, rfe: bool = False):
        normalized = partition
        if token is not None:
            raw = raw_variant(partition.partition_identity, token)
            normalized = replace(
                partition,
                partition_identity=raw,
                exact_values_digest=semantic_digest((partition.exact_values_digest, token)),
            )
        return replace(
            base_sequence,
            normalized_partition_evidence=normalized,
            partition_role=normalized.partition_identity.partition_role,
            rfe_stage_identity=_post_rfe(features) if rfe else RFEStageIdentity.non_rfe(),
        )

    sequence_groups = [
        sequence_partition(normalized_evidence.train),
        sequence_partition(normalized_evidence.validation),
        sequence_partition(normalized_evidence.test),
    ]
    sequence_groups.extend(
        sequence_partition(normalized_evidence.train, token=token)
        for token in ("K3-A", "K3-B", "K3-C")
    )
    sequence_groups.append(sequence_partition(normalized_evidence.train, token="SS"))
    sequence_groups.extend(
        sequence_partition(normalized_evidence.train, token=token, rfe=True)
        for token in ("K3-A", "K3-B", "K3-C")
    )
    sequence_groups.extend(
        sequence_partition(partition, rfe=True)
        for partition in (
            normalized_evidence.train,
            normalized_evidence.validation,
            normalized_evidence.test,
        )
    )
    assert len(sequence_groups) == 13
    assert len(set(sequence_groups)) == 13

    k3_a = normalization_variant("K3-A")
    ss_same_raw = normalization_variant("K3-A")
    assert k3_a == ss_same_raw


def test_target_and_cross_cell_lifecycle_isolation() -> None:
    train, _, _ = _partitions()
    identities = []
    for target in (("166", "258"), ("166", "432"), ("166", "433"), ("166", "313"), ("166", "311")):
        current = train.copy()
        current.attrs = dict(train.attrs)
        current.attrs[CELL_IDENTITY_ATTR] = ("D4", "with_information_sharing", 1, 42, target)
        current.attrs["protocol_target_key"] = target
        identities.append(build_raw_partition_identity(current, feature_cols=FEATURES))
    assert len(set(identities)) == 5
    assert replace(identities[0], horizon=5) != identities[0]
    assert replace(identities[0], seed=43) != identities[0]
    assert replace(identities[0], scenario="without_information_sharing") != identities[0]


def test_raw_preprocessing_upstream_identity_is_propagated_not_recomputed() -> None:
    train, _, _ = _partitions()
    upstream = RawPreprocessingIdentity(
        lifecycle_identity=("D1", "with_information_sharing", 1, 42, ("1", "10")),
        dataset_id="D1",
        scenario="with_information_sharing",
        horizon=1,
        seed=42,
        target_key=("1", "10"),
        source_key=("2", "20"),
        source_cutoff="2023-12-31",
        source_window=("2023-01-01", "2023-12-31"),
        group_cols=("entity_id", "item_id"),
        model_feature_cols=FEATURES,
        row_count=len(train),
        row_identity_digest="upstream-row",
        date_identity_digest="upstream-date",
        columns=tuple(train.columns),
        dtypes=tuple(map(str, train.dtypes)),
        candidate_pool_digest="candidate",
        selection_result_digest="selection",
        source_protocol_digest="source-protocol",
        raw_frame_digest="raw-frame",
    )
    train.attrs["protocol_raw_preprocessing_identity"] = upstream
    identity = build_raw_partition_identity(train, feature_cols=FEATURES)
    assert identity.upstream_preprocessing_identity is upstream
    assert identity.upstream_identity_digest == semantic_digest(upstream)
