"""Fail-closed identities and evidence for normalization and sequences.

The objects in this module are immutable sidecars.  They never own pandas
frames, sklearn estimators, or numpy arrays and they do not perform lookup,
memoization, or reuse.  Target-local reuse is implemented separately and uses
these sidecars as its only lookup/evidence gate.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from datetime import date, datetime
import hashlib
import math
import struct
from typing import Any, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler

from src.protocols.experiment_protocol import ProtocolViolation, normalize_source_key


CANONICAL_SERIALIZATION_VERSION = "p1.2-c2-canonical-v1"
RAW_PARTITION_SCHEMA_VERSION = "raw-partition-v2-consumed-columns"
NORMALIZATION_SCHEMA_VERSION = "normalization-v1"
SEQUENCE_SCHEMA_VERSION = "sequence-v1"
CANONICAL_SORT_IDENTITY = "group_then_utc_date_mergesort_v1"
DATE_IDENTITY_VERSION = "utc_normalized_day_v1"

RAW_PREPROCESSING_IDENTITY_ATTR = "protocol_raw_preprocessing_identity"
FILL_POLICY_IDENTITY_ATTR = "protocol_fill_policy_identity"
RFE_STAGE_IDENTITY_ATTR = "protocol_rfe_stage_identity"
NORMALIZATION_EVIDENCE_ATTR = "protocol_normalization_evidence"
SEQUENCE_EVIDENCE_ATTR = "protocol_sequence_evidence"
CELL_IDENTITY_ATTR = "protocol_cell_identity"


class _ImmutableSidecar:
    def __deepcopy__(self, memo: dict[int, Any]) -> "_ImmutableSidecar":
        memo[id(self)] = self
        return self


def _length_prefix(size: int) -> bytes:
    if size < 0:
        raise ProtocolViolation("canonical serialization length must be non-negative")
    return struct.pack("<Q", size)


def _chunk(tag: bytes, payload: bytes) -> bytes:
    return tag + _length_prefix(len(payload)) + payload


def _scalar_from_numpy(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _encode_canonical(value: Any) -> bytes:
    value = _scalar_from_numpy(value)
    if value is None or value is pd.NA:
        return _chunk(b"N", b"")
    if isinstance(value, (bool, np.bool_)):
        return _chunk(b"B", b"\x01" if bool(value) else b"\x00")
    if isinstance(value, int) and not isinstance(value, bool):
        return _chunk(b"I", str(value).encode("ascii"))
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProtocolViolation("canonical serialization rejects NaN and infinity")
        return _chunk(b"F", struct.pack("<d", value))
    if isinstance(value, str):
        return _chunk(b"S", value.encode("utf-8"))
    if isinstance(value, bytes):
        return _chunk(b"Y", value)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp):
            raise ProtocolViolation("canonical serialization rejects missing dates")
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        return _chunk(b"D", timestamp.isoformat().encode("ascii"))
    if isinstance(value, np.dtype):
        return _chunk(b"P", value.str.encode("ascii"))
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ProtocolViolation("canonical ndarray serialization rejects object dtype")
        if np.issubdtype(value.dtype, np.number) and not bool(np.isfinite(value).all()):
            raise ProtocolViolation("canonical ndarray serialization rejects NaN and infinity")
        dtype = value.dtype
        canonical_dtype = dtype.newbyteorder("<") if dtype.byteorder not in {"|", "<"} else dtype
        array = np.ascontiguousarray(value.astype(canonical_dtype, copy=False))
        payload = b"".join(
            (
                _encode_canonical(canonical_dtype.str),
                _encode_canonical(tuple(int(part) for part in array.shape)),
                _chunk(b"R", array.tobytes(order="C")),
            )
        )
        return _chunk(b"A", payload)
    if is_dataclass(value) and not isinstance(value, type):
        type_name = f"{type(value).__module__}.{type(value).__qualname__}"
        items = []
        for field_info in fields(value):
            items.append(_encode_canonical(field_info.name))
            items.append(_encode_canonical(getattr(value, field_info.name)))
        return _chunk(b"C", _encode_canonical(type_name) + b"".join(items))
    if isinstance(value, tuple):
        return _chunk(b"T", b"".join(_encode_canonical(item) for item in value))
    if isinstance(value, list):
        return _chunk(b"L", b"".join(_encode_canonical(item) for item in value))
    if isinstance(value, Mapping):
        encoded_items = [
            (_encode_canonical(key), _encode_canonical(item))
            for key, item in value.items()
        ]
        encoded_items.sort(key=lambda pair: pair[0])
        return _chunk(
            b"M",
            b"".join(_chunk(b"K", key) + _chunk(b"V", item) for key, item in encoded_items),
        )
    raise ProtocolViolation(
        f"unsupported canonical serialization type: {type(value).__module__}.{type(value).__qualname__}"
    )


def canonical_serialize(value: Any) -> bytes:
    """Serialize supported semantic values with one versioned binary format."""

    return _chunk(b"H", CANONICAL_SERIALIZATION_VERSION.encode("ascii")) + _encode_canonical(value)


def semantic_digest(value: Any) -> str:
    return hashlib.sha256(canonical_serialize(value)).hexdigest()


def exact_array_digest(array: np.ndarray) -> str:
    return semantic_digest(np.asarray(array))


def _canonical_scalar(value: Any) -> Any:
    value = _scalar_from_numpy(value)
    if value is pd.NA or value is None:
        raise ProtocolViolation("identity columns contain missing values")
    try:
        if bool(pd.isna(value)):
            raise ProtocolViolation("identity columns contain missing values")
    except (TypeError, ValueError):
        pass
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(value)
    if isinstance(value, (str, bytes, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ProtocolViolation("identity columns contain NaN or infinity")
        return value
    raise ProtocolViolation(f"unsupported identity scalar: {type(value).__name__}")


def canonical_date_tokens(values: pd.Series) -> Tuple[str, ...]:
    """Canonicalize dates as UTC days; null and invalid values fail closed."""

    if bool(values.isna().any()):
        raise ProtocolViolation("date identity contains missing dates")
    try:
        parsed = pd.to_datetime(values, errors="raise", utc=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProtocolViolation(f"date identity contains invalid dates: {exc}") from exc
    if bool(parsed.isna().any()):
        raise ProtocolViolation("date identity contains invalid dates")
    return tuple(parsed.dt.normalize().dt.strftime("%Y-%m-%d"))


def _identity_rows(
    frame: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    date_col: str,
) -> Tuple[Tuple[Any, ...], ...]:
    columns = (*tuple(group_cols), date_col)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ProtocolViolation(f"row identity columns are missing: {missing!r}")
    date_tokens = canonical_date_tokens(frame[date_col])
    rows = tuple(
        tuple(_canonical_scalar(frame.iloc[index][column]) for column in group_cols)
        + (date_tokens[index],)
        for index in range(len(frame))
    )
    encoded = tuple(canonical_serialize(row) for row in rows)
    if len(set(encoded)) != len(encoded):
        raise ProtocolViolation("row identity contains duplicate group/date rows")
    return rows


def _canonical_row_positions(rows: Sequence[Tuple[Any, ...]]) -> Tuple[int, ...]:
    return tuple(sorted(range(len(rows)), key=lambda index: canonical_serialize(rows[index])))


def canonical_frame_digest(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
    group_cols: Sequence[str],
    date_col: str = "date",
) -> str:
    """Digest ordered columns/dtypes and values in canonical group/date order."""

    selected = tuple(str(column) for column in columns)
    if len(set(selected)) != len(selected):
        raise ProtocolViolation("canonical frame digest rejects duplicate requested columns")
    if not frame.columns.is_unique:
        raise ProtocolViolation("canonical frame digest rejects duplicate frame columns")
    missing = [column for column in selected if column not in frame.columns]
    if missing:
        raise ProtocolViolation(f"canonical frame digest missing columns: {missing!r}")
    rows = _identity_rows(frame, group_cols=group_cols, date_col=date_col)
    positions = _canonical_row_positions(rows)
    records = []
    for position in positions:
        record = []
        for column in selected:
            if column == date_col:
                record.append(rows[position][-1])
            else:
                record.append(_canonical_scalar(frame.iloc[position][column]))
        records.append(tuple(record))
    return semantic_digest(
        (
            "canonical-frame-v1",
            selected,
            tuple(str(frame[column].dtype) for column in selected),
            tuple(records),
        )
    )


@dataclass(frozen=True)
class FillPolicyIdentity(_ImmutableSidecar):
    schema_version: str
    helper_identity: str
    filled_columns: Tuple[str, ...]
    fill_value_identity: str
    numeric_coercion_identity: str

    @classmethod
    def none(cls) -> "FillPolicyIdentity":
        return cls("fill-policy-v1", "no_fill_v1", (), "none", "none")


@dataclass(frozen=True)
class SplitIdentity(_ImmutableSidecar):
    schema_version: str
    partition_role: str
    split_role: str
    split_mode: str
    split_config_digest: str
    boundary_identity: Tuple[str, str, int]


@dataclass(frozen=True)
class DateIdentity(_ImmutableSidecar):
    schema_version: str
    date_min: str
    date_max: str
    date_count: int
    date_digest: str
    duplicate_status: str
    missing_status: str
    invalid_status: str


@dataclass(frozen=True)
class RawPartitionIdentity(_ImmutableSidecar):
    schema_version: str
    lifecycle_identity: Tuple[Any, ...]
    dataset_id: str
    scenario: str
    horizon: int
    seed: int
    target_key: Tuple[Any, ...]
    source_key: Tuple[Any, ...]
    source_role: str
    partition_role: str
    split_identity: SplitIdentity
    group_cols: Tuple[str, ...]
    date_col: str
    row_count: int
    row_membership_digest: str
    canonical_row_order_digest: str
    date_identity: DateIdentity
    columns: Tuple[str, ...]
    dtypes: Tuple[str, ...]
    model_feature_cols: Tuple[str, ...]
    model_feature_dtypes: Tuple[str, ...]
    target_column: str
    target_dtype: str
    fill_policy_identity: FillPolicyIdentity
    upstream_preprocessing_identity: Any
    upstream_identity_digest: str
    source_authority_identity: str
    raw_protocol_identity: str
    raw_values_digest: str


RFE_STAGES = ("NON_RFE", "PRE_RFE", "POST_RFE")


@dataclass(frozen=True)
class RFEStageIdentity(_ImmutableSidecar):
    schema_version: str
    stage: str
    selected_feature_cols: Tuple[str, ...] = ()
    rfe_protocol_identity: str = ""
    joint_train_identity: str = ""
    estimator_identity: str = ""
    estimator_config: Tuple[Tuple[str, Any], ...] = ()
    random_state: int = 0
    keep_ratio: float = 0.0
    selection_evidence_digest: str = ""

    def __post_init__(self) -> None:
        if self.stage not in RFE_STAGES:
            raise ProtocolViolation(f"unsupported RFE stage: {self.stage!r}")
        if self.stage == "POST_RFE":
            required = (
                self.selected_feature_cols,
                self.rfe_protocol_identity,
                self.joint_train_identity,
                self.estimator_identity,
                self.estimator_config,
                self.selection_evidence_digest,
            )
            if not all(required) or not (0.0 < self.keep_ratio <= 1.0):
                raise ProtocolViolation("POST_RFE identity requires complete selection evidence")
        elif self.stage == "PRE_RFE":
            if not self.rfe_protocol_identity or any(
                (
                    self.selected_feature_cols,
                    self.joint_train_identity,
                    self.estimator_identity,
                    self.estimator_config,
                    self.selection_evidence_digest,
                    self.random_state,
                    self.keep_ratio,
                )
            ):
                raise ProtocolViolation("PRE_RFE identity requires only its protocol identity")
        elif any(
            (
                self.selected_feature_cols,
                self.rfe_protocol_identity,
                self.joint_train_identity,
                self.estimator_identity,
                self.estimator_config,
                self.selection_evidence_digest,
                self.random_state,
                self.keep_ratio,
            )
        ):
            raise ProtocolViolation("NON_RFE identity cannot carry RFE evidence")

    @classmethod
    def non_rfe(cls) -> "RFEStageIdentity":
        return cls("rfe-stage-v1", "NON_RFE")

    @classmethod
    def pre_rfe(cls, *, protocol_identity: str) -> "RFEStageIdentity":
        return cls("rfe-stage-v1", "PRE_RFE", rfe_protocol_identity=protocol_identity)


@dataclass(frozen=True)
class ScalerAlgorithmIdentity(_ImmutableSidecar):
    schema_version: str
    class_identity: str
    feature_range: Tuple[float, float]
    copy: bool
    clip: bool
    sklearn_version: str
    numpy_version: str


@dataclass(frozen=True)
class NormalizationIdentity(_ImmutableSidecar):
    schema_version: str
    train_partition_identity: RawPartitionIdentity
    validation_partition_identity: RawPartitionIdentity
    test_partition_identity: RawPartitionIdentity
    actual_feature_cols: Tuple[str, ...]
    target_column: str
    fit_scope: str
    fit_partition_digest: str
    scaler_algorithm_identity: ScalerAlgorithmIdentity
    rfe_stage_identity: RFEStageIdentity


@dataclass(frozen=True)
class ScalerParameterEvidence(_ImmutableSidecar):
    data_min: Tuple[float, ...]
    data_max: Tuple[float, ...]
    data_range: Tuple[float, ...]
    scale: Tuple[float, ...]
    min_offset: Tuple[float, ...]
    n_features_in: int
    n_samples_seen: int
    feature_names_in: Tuple[str, ...]
    exact_digest: str


@dataclass(frozen=True)
class NormalizedPartitionEvidence(_ImmutableSidecar):
    partition_identity: RawPartitionIdentity
    shape: Tuple[int, int]
    feature_dtype: Tuple[str, ...]
    row_membership_digest: str
    canonical_row_order_digest: str
    date_identity_digest: str
    feature_cols: Tuple[str, ...]
    finite: bool
    exact_values_digest: str


@dataclass(frozen=True)
class NormalizationEvidence(_ImmutableSidecar):
    schema_version: str
    identity: NormalizationIdentity
    fit_row_count: int
    fit_membership_digest: str
    fit_order_digest: str
    fit_values_digest: str
    scaler_parameters: ScalerParameterEvidence
    constant_feature_cols: Tuple[str, ...]
    train: NormalizedPartitionEvidence
    validation: NormalizedPartitionEvidence
    test: NormalizedPartitionEvidence
    finite: bool
    exact_digest: str


@dataclass(frozen=True)
class SequenceIdentity(_ImmutableSidecar):
    schema_version: str
    normalized_partition_evidence: NormalizedPartitionEvidence
    partition_role: str
    group_cols: Tuple[str, ...]
    date_col: str
    canonical_sort_identity: str
    actual_feature_cols: Tuple[str, ...]
    target_column: str
    window_size: int
    horizon: int
    x_dtype: str
    y_dtype: str
    rfe_stage_identity: RFEStageIdentity
    lifecycle_identity: Tuple[Any, ...]


@dataclass(frozen=True)
class SampleBoundaryEvidence(_ImmutableSidecar):
    group_key: Tuple[Any, ...]
    window_start: str
    window_end: str
    label_date: str
    horizon: int
    partition_role: str


@dataclass(frozen=True)
class SequenceEvidence(_ImmutableSidecar):
    schema_version: str
    identity: SequenceIdentity
    x_shape: Tuple[int, ...]
    y_shape: Tuple[int, ...]
    x_dtype: str
    y_dtype: str
    sample_count: int
    sample_key_date_digest: str
    window_alignment_digest: str
    label_alignment_digest: str
    x_exact_digest: str
    y_exact_digest: str
    first_sample: SampleBoundaryEvidence | None
    last_sample: SampleBoundaryEvidence | None
    finite: bool
    exact_digest: str


def resolve_group_cols(frame: pd.DataFrame) -> Tuple[str, ...]:
    configured = tuple(str(column) for column in frame.attrs.get("protocol_group_cols", ()))
    if configured and all(column in frame.columns for column in configured):
        return configured
    if all(column in frame.columns for column in ("entity_id", "item_id")):
        return ("entity_id", "item_id")
    raise ProtocolViolation("cannot resolve runtime group columns for transformation identity")


def transformation_consumed_identity_columns(
    frame: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    date_col: str,
    feature_cols: Sequence[str],
) -> Tuple[str, ...]:
    """Bind raw identity to row/entity/date identity and consumed features only."""

    requested = [*map(str, group_cols), str(date_col)]
    if "entity_id" in frame.columns:
        requested.append("entity_id")
    requested.extend(map(str, feature_cols))
    scoped = tuple(dict.fromkeys(requested))
    missing = [column for column in scoped if column not in frame.columns]
    if missing:
        raise ProtocolViolation(
            f"transformation-consumed identity columns are missing: {missing!r}"
        )
    return scoped


def transformation_identity_requested(frame: pd.DataFrame) -> bool:
    """Return whether a frame belongs to the formal identity-aware lifecycle.

    Small legacy/unit callers that provide only numeric columns retain the
    original normalization API.  Once any formal lifecycle marker is present,
    missing identity inputs are errors rather than a reason to skip evidence.
    """

    return any(
        marker in frame.attrs
        for marker in (
            CELL_IDENTITY_ATTR,
            RAW_PREPROCESSING_IDENTITY_ATTR,
            "protocol_version",
            "protocol_dataset_id",
        )
    )


def _cell_identity(frame: pd.DataFrame) -> Tuple[Any, ...]:
    raw = frame.attrs.get(CELL_IDENTITY_ATTR)
    upstream = frame.attrs.get(RAW_PREPROCESSING_IDENTITY_ATTR)
    if raw is None and upstream is not None:
        raw = getattr(upstream, "lifecycle_identity", None)
    if raw is None:
        raw = (
            str(frame.attrs.get("protocol_dataset_id", frame.attrs.get("dataset_name", ""))),
            str(frame.attrs.get("protocol_scenario", frame.attrs.get("information_sharing_scenario", ""))),
            int(frame.attrs.get("model_horizon", 0)),
            int(frame.attrs.get("protocol_seed", 0)),
            tuple(frame.attrs.get("protocol_target_key", ())),
        )
    result = tuple(raw)
    if len(result) != 5:
        raise ProtocolViolation("cell/lifecycle identity must have five fields")
    return (
        str(result[0]),
        str(result[1]),
        int(result[2]),
        int(result[3]),
        normalize_source_key(result[4]),
    )


def build_raw_partition_identity(
    frame: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    target_column: str = "sales",
    group_cols: Sequence[str] | None = None,
    date_col: str = "date",
) -> RawPartitionIdentity:
    features = tuple(str(column) for column in feature_cols)
    if not features or len(set(features)) != len(features):
        raise ProtocolViolation("raw partition identity requires unique runtime features")
    if target_column not in features or target_column not in frame.columns:
        raise ProtocolViolation("raw partition target column must be an actual model feature")
    missing = [column for column in features if column not in frame.columns]
    if missing:
        raise ProtocolViolation(f"raw partition model features are missing: {missing!r}")
    groups = tuple(group_cols or resolve_group_cols(frame))
    rows = _identity_rows(frame, group_cols=groups, date_col=date_col)
    positions = _canonical_row_positions(rows)
    canonical_rows = tuple(rows[position] for position in positions)
    date_tokens = tuple(row[-1] for row in rows)
    canonical_dates = tuple(row[-1] for row in canonical_rows)
    partition_role = str(frame.attrs.get("temporal_partition", "unsplit"))
    if partition_role not in {"train", "validation", "test", "unsplit"}:
        raise ProtocolViolation(f"unsupported partition role: {partition_role!r}")
    split_config = frame.attrs.get("split_config", {}) or {}
    if not isinstance(split_config, Mapping):
        raise ProtocolViolation("split_config must be a mapping")
    split_identity = SplitIdentity(
        "split-identity-v1",
        partition_role,
        str(frame.attrs.get("split_role", frame.attrs.get("role", "unknown"))),
        str(frame.attrs.get("split_mode", "unknown")),
        semantic_digest(split_config),
        (min(date_tokens), max(date_tokens), len(set(date_tokens))),
    )
    date_identity = DateIdentity(
        DATE_IDENTITY_VERSION,
        min(date_tokens),
        max(date_tokens),
        len(set(date_tokens)),
        semantic_digest(tuple(sorted(date_tokens))),
        "none",
        "none",
        "none",
    )
    lifecycle = _cell_identity(frame)
    upstream = frame.attrs.get(RAW_PREPROCESSING_IDENTITY_ATTR)
    if upstream is not None:
        params = getattr(type(upstream), "__dataclass_params__", None)
        if not is_dataclass(upstream) or params is None or not bool(params.frozen):
            raise ProtocolViolation("upstream preprocessing identity must be a frozen dataclass")
    upstream_digest = semantic_digest(upstream) if upstream is not None else ""
    fill = frame.attrs.get(FILL_POLICY_IDENTITY_ATTR, FillPolicyIdentity.none())
    if not isinstance(fill, FillPolicyIdentity):
        raise ProtocolViolation("fill policy sidecar has an invalid type")
    source_key_raw = frame.attrs.get("protocol_actual_source_key", ())
    source_key = normalize_source_key(source_key_raw) if source_key_raw else ()
    source_role = "source" if source_key else str(frame.attrs.get("split_role", "target"))
    raw_columns = transformation_consumed_identity_columns(
        frame,
        group_cols=groups,
        date_col=date_col,
        feature_cols=features,
    )
    raw_dtypes = tuple(str(frame[column].dtype) for column in raw_columns)
    authority = str(
        frame.attrs.get(
            "source_history_frame_digest",
            frame.attrs.get("source_frame_digest", frame.attrs.get("target_frame_digest", "")),
        )
    )
    protocol_identity = semantic_digest(
        (
            frame.attrs.get("protocol_version", ""),
            frame.attrs.get("protocol_track", ""),
            frame.attrs.get("selection_authority", ""),
        )
    )
    return RawPartitionIdentity(
        RAW_PARTITION_SCHEMA_VERSION,
        lifecycle,
        str(lifecycle[0]),
        str(lifecycle[1]),
        int(lifecycle[2]),
        int(lifecycle[3]),
        tuple(lifecycle[4]),
        source_key,
        source_role,
        partition_role,
        split_identity,
        groups,
        date_col,
        int(len(frame)),
        semantic_digest(tuple(sorted(canonical_serialize(row) for row in rows))),
        semantic_digest((CANONICAL_SORT_IDENTITY, canonical_rows)),
        date_identity,
        raw_columns,
        raw_dtypes,
        features,
        tuple(str(frame[column].dtype) for column in features),
        target_column,
        str(frame[target_column].dtype),
        fill,
        upstream,
        upstream_digest,
        authority,
        protocol_identity,
        canonical_frame_digest(
            frame,
            columns=raw_columns,
            group_cols=groups,
            date_col=date_col,
        ),
    )


def scaler_algorithm_identity(
    scaler: MinMaxScaler,
    *,
    sklearn_version: str | None = None,
    numpy_version: str | None = None,
) -> ScalerAlgorithmIdentity:
    if not isinstance(scaler, MinMaxScaler):
        raise ProtocolViolation("normalization identity requires sklearn MinMaxScaler")
    params = scaler.get_params(deep=False)
    return ScalerAlgorithmIdentity(
        "minmax-scaler-algorithm-v1",
        "sklearn.preprocessing.MinMaxScaler",
        tuple(float(value) for value in params["feature_range"]),
        bool(params["copy"]),
        bool(params["clip"]),
        str(sklearn_version or sklearn.__version__),
        str(numpy_version or np.__version__),
    )


def _rfe_identity(frame: pd.DataFrame) -> RFEStageIdentity:
    result = frame.attrs.get(RFE_STAGE_IDENTITY_ATTR, RFEStageIdentity.non_rfe())
    if not isinstance(result, RFEStageIdentity):
        raise ProtocolViolation("RFE stage sidecar has an invalid type")
    return result


def build_normalization_identity(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    scaler: MinMaxScaler,
    target_column: str = "sales",
    fit_scope: str = "train_only",
) -> NormalizationIdentity:
    if fit_scope != "train_only":
        raise ProtocolViolation("production normalization fit scope must be train_only")
    train = build_raw_partition_identity(train_df, feature_cols=feature_cols, target_column=target_column)
    validation = build_raw_partition_identity(val_df, feature_cols=feature_cols, target_column=target_column)
    test = build_raw_partition_identity(test_df, feature_cols=feature_cols, target_column=target_column)
    if (train.partition_role, validation.partition_role, test.partition_role) != (
        "train", "validation", "test"
    ):
        raise ProtocolViolation("normalization requires exact train/validation/test partition roles")
    rfe = _rfe_identity(train_df)
    if _rfe_identity(val_df) != rfe or _rfe_identity(test_df) != rfe:
        raise ProtocolViolation("normalization partitions have mismatched RFE stage identity")
    features = tuple(str(column) for column in feature_cols)
    return NormalizationIdentity(
        NORMALIZATION_SCHEMA_VERSION,
        train,
        validation,
        test,
        features,
        target_column,
        fit_scope,
        semantic_digest(train),
        scaler_algorithm_identity(scaler),
        rfe,
    )


def _parameter_tuple(scaler: MinMaxScaler, name: str) -> Tuple[float, ...]:
    value = np.asarray(getattr(scaler, name), dtype=np.float64)
    if not bool(np.isfinite(value).all()):
        raise ProtocolViolation(f"fitted scaler parameter is non-finite: {name}")
    return tuple(float(item) for item in value)


def scaler_parameter_evidence(scaler: MinMaxScaler) -> ScalerParameterEvidence:
    required = ("data_min_", "data_max_", "data_range_", "scale_", "min_", "n_features_in_", "n_samples_seen_")
    missing = [name for name in required if not hasattr(scaler, name)]
    if missing:
        raise ProtocolViolation(f"MinMaxScaler is not fitted: missing {missing!r}")
    values = (
        _parameter_tuple(scaler, "data_min_"),
        _parameter_tuple(scaler, "data_max_"),
        _parameter_tuple(scaler, "data_range_"),
        _parameter_tuple(scaler, "scale_"),
        _parameter_tuple(scaler, "min_"),
    )
    names = tuple(str(item) for item in getattr(scaler, "feature_names_in_", ()))
    payload = (*values, int(scaler.n_features_in_), int(scaler.n_samples_seen_), names)
    return ScalerParameterEvidence(
        *values,
        int(scaler.n_features_in_),
        int(scaler.n_samples_seen_),
        names,
        semantic_digest(payload),
    )


def _normalized_partition_evidence(
    frame: pd.DataFrame,
    identity: RawPartitionIdentity,
    *,
    feature_cols: Sequence[str],
) -> NormalizedPartitionEvidence:
    features = tuple(str(column) for column in feature_cols)
    rows = _identity_rows(frame, group_cols=identity.group_cols, date_col=identity.date_col)
    positions = _canonical_row_positions(rows)
    values = frame.loc[:, list(features)].to_numpy()[list(positions)]
    if not bool(np.isfinite(values).all()):
        raise ProtocolViolation("normalized partition contains NaN or infinity")
    canonical_rows = tuple(rows[position] for position in positions)
    current_membership = semantic_digest(tuple(sorted(canonical_serialize(row) for row in rows)))
    current_order = semantic_digest((CANONICAL_SORT_IDENTITY, canonical_rows))
    date_tokens = tuple(row[-1] for row in rows)
    current_date = semantic_digest(tuple(sorted(date_tokens)))
    if (
        current_membership != identity.row_membership_digest
        or current_order != identity.canonical_row_order_digest
        or current_date != identity.date_identity.date_digest
    ):
        raise ProtocolViolation("normalization changed row/date identity")
    return NormalizedPartitionEvidence(
        identity,
        (int(len(frame)), len(features)),
        tuple(str(frame[column].dtype) for column in features),
        identity.row_membership_digest,
        identity.canonical_row_order_digest,
        identity.date_identity.date_digest,
        features,
        True,
        exact_array_digest(values),
    )


def build_normalization_evidence(
    train_raw: pd.DataFrame,
    val_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    train_scaled: pd.DataFrame,
    val_scaled: pd.DataFrame,
    test_scaled: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    scaler: MinMaxScaler,
    target_column: str = "sales",
) -> NormalizationEvidence:
    identity = build_normalization_identity(
        train_raw,
        val_raw,
        test_raw,
        feature_cols=feature_cols,
        scaler=scaler,
        target_column=target_column,
    )
    params = scaler_parameter_evidence(scaler)
    train = _normalized_partition_evidence(
        train_scaled, identity.train_partition_identity, feature_cols=feature_cols
    )
    validation = _normalized_partition_evidence(
        val_scaled, identity.validation_partition_identity, feature_cols=feature_cols
    )
    test = _normalized_partition_evidence(
        test_scaled, identity.test_partition_identity, feature_cols=feature_cols
    )
    constant = tuple(
        feature
        for feature, value_range in zip(feature_cols, params.data_range)
        if value_range == 0.0
    )
    fit_values = canonical_frame_digest(
        train_raw,
        columns=feature_cols,
        group_cols=identity.train_partition_identity.group_cols,
        date_col=identity.train_partition_identity.date_col,
    )
    payload = (
        identity,
        len(train_raw),
        identity.train_partition_identity.row_membership_digest,
        identity.train_partition_identity.canonical_row_order_digest,
        fit_values,
        params,
        constant,
        train,
        validation,
        test,
        True,
    )
    return NormalizationEvidence(
        NORMALIZATION_SCHEMA_VERSION,
        identity,
        int(len(train_raw)),
        identity.train_partition_identity.row_membership_digest,
        identity.train_partition_identity.canonical_row_order_digest,
        fit_values,
        params,
        constant,
        train,
        validation,
        test,
        True,
        semantic_digest(payload),
    )


def normalized_partition_from_frame(frame: pd.DataFrame) -> NormalizedPartitionEvidence:
    evidence = frame.attrs.get(NORMALIZATION_EVIDENCE_ATTR)
    if not isinstance(evidence, NormalizationEvidence):
        raise ProtocolViolation("normalized frame is missing normalization evidence")
    role = str(frame.attrs.get("temporal_partition", ""))
    lookup = {
        "train": evidence.train,
        "validation": evidence.validation,
        "test": evidence.test,
    }
    if role not in lookup:
        raise ProtocolViolation(f"normalized frame has unsupported partition role: {role!r}")
    evidence_for_role = lookup[role]
    validate_normalized_partition_evidence(frame, evidence_for_role)
    return evidence_for_role


def validate_normalized_partition_evidence(
    frame: pd.DataFrame,
    evidence: NormalizedPartitionEvidence,
) -> None:
    """Fail closed when a normalized working frame no longer matches its sidecar."""

    actual = _normalized_partition_evidence(
        frame,
        evidence.partition_identity,
        feature_cols=evidence.feature_cols,
    )
    if actual != evidence:
        raise ProtocolViolation("normalized frame evidence mismatch")


def make_sample_boundary(
    *,
    group_key: Sequence[Any],
    window_start: Any,
    window_end: Any,
    label_date: Any,
    horizon: int,
    partition_role: str,
) -> SampleBoundaryEvidence:
    tokens = canonical_date_tokens(pd.Series([window_start, window_end, label_date]))
    return SampleBoundaryEvidence(
        tuple(_canonical_scalar(value) for value in group_key),
        tokens[0],
        tokens[1],
        tokens[2],
        int(horizon),
        str(partition_role),
    )


def build_sequence_evidence(
    frame: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    *,
    feature_cols: Sequence[str],
    horizon: int,
    window_size: int,
    samples: Sequence[SampleBoundaryEvidence],
    target_column: str = "sales",
    group_cols: Sequence[str] = ("entity_id", "item_id"),
    date_col: str = "date",
) -> SequenceEvidence:
    x = np.asarray(X)
    labels = np.asarray(y)
    features = tuple(str(column) for column in feature_cols)
    if x.ndim != 3 or labels.ndim != 1 or len(x) != len(labels):
        raise ProtocolViolation("sequence arrays violate X/y rank or sample-count contract")
    if x.shape[1:] != (int(window_size), len(features)):
        raise ProtocolViolation("sequence X shape violates window/feature contract")
    identity = build_sequence_identity(
        frame,
        feature_cols=features,
        horizon=horizon,
        window_size=window_size,
        target_column=target_column,
        group_cols=group_cols,
        date_col=date_col,
        x_dtype=str(x.dtype),
        y_dtype=str(labels.dtype),
    )
    normalized = identity.normalized_partition_evidence
    if len(samples) != len(x):
        raise ProtocolViolation("sequence sample map count does not match X/y")
    if not bool(np.isfinite(x).all()) or not bool(np.isfinite(labels).all()):
        raise ProtocolViolation("sequence arrays contain NaN or infinity")
    if any(
        sample.horizon != int(horizon)
        or sample.partition_role != normalized.partition_identity.partition_role
        for sample in samples
    ):
        raise ProtocolViolation("sequence sample map horizon/partition mismatch")
    sample_tuple = tuple(samples)
    x_digest = exact_array_digest(x)
    y_digest = exact_array_digest(labels)
    sample_digest = semantic_digest(sample_tuple)
    window_digest = semantic_digest(
        (
            tuple((sample.group_key, sample.window_start, sample.window_end) for sample in sample_tuple),
            x_digest,
        )
    )
    label_digest = semantic_digest(
        (
            tuple((sample.group_key, sample.label_date, sample.horizon) for sample in sample_tuple),
            y_digest,
        )
    )
    first = sample_tuple[0] if sample_tuple else None
    last = sample_tuple[-1] if sample_tuple else None
    payload = (
        identity,
        tuple(int(part) for part in x.shape),
        tuple(int(part) for part in labels.shape),
        str(x.dtype),
        str(labels.dtype),
        len(x),
        sample_digest,
        window_digest,
        label_digest,
        x_digest,
        y_digest,
        first,
        last,
        True,
    )
    return SequenceEvidence(
        SEQUENCE_SCHEMA_VERSION,
        identity,
        tuple(int(part) for part in x.shape),
        tuple(int(part) for part in labels.shape),
        str(x.dtype),
        str(labels.dtype),
        int(len(x)),
        sample_digest,
        window_digest,
        label_digest,
        x_digest,
        y_digest,
        first,
        last,
        True,
        semantic_digest(payload),
    )


def build_sequence_identity(
    frame: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    horizon: int,
    window_size: int,
    target_column: str = "sales",
    group_cols: Sequence[str] = ("entity_id", "item_id"),
    date_col: str = "date",
    x_dtype: str = "float32",
    y_dtype: str = "float32",
) -> SequenceIdentity:
    """Build the exact pre-computation lookup identity for one sequence request."""

    if int(horizon) <= 0 or int(window_size) <= 0:
        raise ProtocolViolation("sequence identity requires positive horizon/window_size")
    normalized = normalized_partition_from_frame(frame)
    normalization = frame.attrs.get(NORMALIZATION_EVIDENCE_ATTR)
    if not isinstance(normalization, NormalizationEvidence):
        raise ProtocolViolation("sequence frame is missing normalization evidence")
    features = tuple(str(column) for column in feature_cols)
    if target_column not in features:
        raise ProtocolViolation("sequence target column must be an actual model feature")
    rfe = _rfe_identity(frame)
    if rfe != normalization.identity.rfe_stage_identity:
        raise ProtocolViolation("sequence frame RFE stage differs from normalized evidence")
    return SequenceIdentity(
        SEQUENCE_SCHEMA_VERSION,
        normalized,
        normalized.partition_identity.partition_role,
        tuple(str(column) for column in group_cols),
        date_col,
        CANONICAL_SORT_IDENTITY,
        features,
        target_column,
        int(window_size),
        int(horizon),
        str(np.dtype(x_dtype)),
        str(np.dtype(y_dtype)),
        rfe,
        normalized.partition_identity.lifecycle_identity,
    )


def require_same_identity(left: Any, right: Any, *, contract: str) -> None:
    """Fail-closed equality gate for future reuse code."""

    if type(left) is not type(right) or left != right:
        raise ProtocolViolation(f"{contract} identity mismatch")


MODEL_FEATURE_CONTRACTS = {
    "D1": ("sales", "year", "month", "week", "day"),
    "D2": ("sales", "year", "month", "week", "day"),
    "D3": (
        "sales", "year", "month", "week", "day", "customers", "open", "promo", "school_holiday"
    ),
    "D4": (
        "sales", "stock_hour6_22_cnt", "activity_flag", "discount", "holiday_flag", "precpt",
        "avg_temperature", "avg_humidity", "avg_wind_level", "year", "month", "week", "day"
    ),
    "D5": (
        "sales", "year", "month", "week", "day", "class", "perishable", "cluster",
        "transactions", "oil_price", "is_holiday"
    ),
    "D6": (
        "sales", "year", "month", "week", "day", "wm_yr_wk", "weekday", "is_event_1",
        "is_event_2", "snap", "sell_price"
    ),
}


KNN_FEATURE_CONTRACTS = {
    "D1": ("sales",),
    "D2": ("sales", "promo"),
    "D3": ("sales",),
    "D4": ("sales",),
    "D5": ("sales", "onpromotion", "oil_price"),
    "D6": ("sales",),
}


def validate_runtime_feature_contract(
    dataset_id: str,
    *,
    model_feature_cols: Sequence[str],
    knn_feature_cols: Sequence[str],
) -> None:
    normalized = str(dataset_id).strip().upper().replace("DATASET", "D")
    if normalized not in MODEL_FEATURE_CONTRACTS:
        raise ProtocolViolation(f"unsupported runtime feature contract: {dataset_id!r}")
    if tuple(model_feature_cols) != MODEL_FEATURE_CONTRACTS[normalized]:
        raise ProtocolViolation(f"{normalized} runtime model feature identity mismatch")
    if tuple(knn_feature_cols) != KNN_FEATURE_CONTRACTS[normalized]:
        raise ProtocolViolation(f"{normalized} runtime KNN feature identity mismatch")


__all__ = [
    "CANONICAL_SERIALIZATION_VERSION",
    "CANONICAL_SORT_IDENTITY",
    "CELL_IDENTITY_ATTR",
    "FILL_POLICY_IDENTITY_ATTR",
    "KNN_FEATURE_CONTRACTS",
    "MODEL_FEATURE_CONTRACTS",
    "NORMALIZATION_EVIDENCE_ATTR",
    "RAW_PREPROCESSING_IDENTITY_ATTR",
    "RFE_STAGE_IDENTITY_ATTR",
    "SEQUENCE_EVIDENCE_ATTR",
    "DateIdentity",
    "FillPolicyIdentity",
    "NormalizationEvidence",
    "NormalizationIdentity",
    "NormalizedPartitionEvidence",
    "RFEStageIdentity",
    "RawPartitionIdentity",
    "SampleBoundaryEvidence",
    "ScalerAlgorithmIdentity",
    "ScalerParameterEvidence",
    "SequenceEvidence",
    "SequenceIdentity",
    "SplitIdentity",
    "build_normalization_evidence",
    "build_normalization_identity",
    "build_raw_partition_identity",
    "build_sequence_evidence",
    "build_sequence_identity",
    "canonical_date_tokens",
    "canonical_frame_digest",
    "canonical_serialize",
    "exact_array_digest",
    "make_sample_boundary",
    "normalized_partition_from_frame",
    "require_same_identity",
    "scaler_algorithm_identity",
    "scaler_parameter_evidence",
    "semantic_digest",
    "transformation_identity_requested",
    "validate_normalized_partition_evidence",
    "validate_runtime_feature_contract",
]
