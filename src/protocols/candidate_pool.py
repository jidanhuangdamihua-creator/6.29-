"""Canonical candidate-pool auditing and leak-free daily-sequence KNN."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from .experiment_protocol import (
    ExperimentProtocol,
    ProtocolViolation,
    SourceKey,
    normalize_scenario,
    normalize_source_key,
)
from .knn_frames import canonical_knn_frame_digest
from src.utils.dataframe_attrs import (
    copy_frame_with_lightweight_attrs,
    temporarily_detached_attrs,
)


def _iso_date(value: object) -> str:
    converted = pd.Timestamp(value)
    if pd.isna(converted):
        raise ProtocolViolation(f"invalid digest date: {value!r}")
    return converted.normalize().strftime("%Y-%m-%d")


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _canonical_candidate_pool_input(
    protocol_version: object,
    dataset_id: object,
    scenario: object,
    target_key: Sequence[object],
    group_cols: Sequence[object],
    candidate_keys: Iterable[Sequence[object]],
    observed_start: object,
    observed_end: object,
    feature_cols: Sequence[object],
    calendarization_rule_version: object | None = None,
    source_authority_digest: object | None = None,
    consumer_frame_fingerprint: object | None = None,
    source_frame_digest: object | None = None,
    target_frame_digest: object | None = None,
    source_history_days: object | None = None,
    source_history_start: object | None = None,
    source_history_end: object | None = None,
    source_history_completeness_policy: object | None = None,
    source_history_frame_digest: object | None = None,
) -> Dict[str, Any]:
    normalized_candidates = [normalize_source_key(key) for key in candidate_keys]
    if len(set(normalized_candidates)) != len(normalized_candidates):
        raise ProtocolViolation("candidate pool contains duplicate source keys")
    normalized_candidates.sort()
    payload = {
        "protocol_version": str(protocol_version).strip(),
        "dataset_id": str(dataset_id).strip().upper(),
        "scenario": normalize_scenario(scenario),
        "target_key": list(normalize_source_key(target_key)),
        "group_cols": [str(column).strip() for column in group_cols],
        "candidate_keys": [list(key) for key in normalized_candidates],
        "observed_start": _iso_date(observed_start),
        "observed_end": _iso_date(observed_end),
        "feature_cols": [str(column).strip() for column in feature_cols],
    }
    identity_values = (
        calendarization_rule_version,
        source_authority_digest,
        consumer_frame_fingerprint,
    )
    if any(value is not None for value in identity_values):
        if any(value is None for value in identity_values):
            raise ProtocolViolation(
                "D2 candidate digest requires complete source calendarization identity"
            )
        payload.update(
            {
                "calendarization_rule_version": str(calendarization_rule_version),
                "source_authority_digest": str(source_authority_digest),
                "consumer_frame_fingerprint": str(consumer_frame_fingerprint),
            }
        )
    frame_identity = (source_frame_digest, target_frame_digest)
    if any(value is not None for value in frame_identity):
        if any(value is None for value in frame_identity):
            raise ProtocolViolation(
                "KNN candidate digest requires both source and target frame digests"
            )
        payload.update(
            {
                "source_frame_digest": str(source_frame_digest),
                "target_frame_digest": str(target_frame_digest),
            }
        )
    history_identity = (
        source_history_days,
        source_history_start,
        source_history_end,
        source_history_completeness_policy,
        source_history_frame_digest,
    )
    if any(value is not None for value in history_identity):
        if any(value is None for value in history_identity):
            raise ProtocolViolation(
                "D4-D6 candidate digest requires complete source history identity"
            )
        payload.update(
            {
                "source_history_days": int(source_history_days),
                "source_history_start": _iso_date(source_history_start),
                "source_history_end": _iso_date(source_history_end),
                "source_history_completeness_policy": str(
                    source_history_completeness_policy
                ),
                "source_history_frame_digest": str(source_history_frame_digest),
            }
        )
    return payload


def build_candidate_pool_digest(
    protocol_version: object,
    dataset_id: object,
    scenario: object,
    target_key: Sequence[object],
    group_cols: Sequence[object],
    candidate_keys: Iterable[Sequence[object]],
    observed_start: object,
    observed_end: object,
    feature_cols: Sequence[object],
    calendarization_rule_version: object | None = None,
    source_authority_digest: object | None = None,
    consumer_frame_fingerprint: object | None = None,
    source_frame_digest: object | None = None,
    target_frame_digest: object | None = None,
    source_history_days: object | None = None,
    source_history_start: object | None = None,
    source_history_end: object | None = None,
    source_history_completeness_policy: object | None = None,
    source_history_frame_digest: object | None = None,
) -> str:
    """Return the one production SHA-256 digest for a candidate pool."""

    payload = _canonical_candidate_pool_input(
        protocol_version,
        dataset_id,
        scenario,
        target_key,
        group_cols,
        candidate_keys,
        observed_start,
        observed_end,
        feature_cols,
        calendarization_rule_version,
        source_authority_digest,
        consumer_frame_fingerprint,
        source_frame_digest,
        target_frame_digest,
        source_history_days,
        source_history_start,
        source_history_end,
        source_history_completeness_policy,
        source_history_frame_digest,
    )
    return _sha256_payload(payload)


def build_source_pool_fingerprint(
    *,
    protocol_version: object,
    dataset_id: object,
    scenario: object,
    target_key: Sequence[object],
    group_cols: Sequence[object],
    candidate_keys: Iterable[Sequence[object]],
) -> str:
    """Bind the exact ordered composite-key candidate identity for one target."""

    normalized_candidates = [normalize_source_key(key) for key in candidate_keys]
    if len(set(normalized_candidates)) != len(normalized_candidates):
        raise ProtocolViolation("source pool fingerprint contains duplicate source keys")
    normalized_candidates.sort()
    return _sha256_payload(
        {
            "protocol_version": str(protocol_version).strip(),
            "dataset_id": str(dataset_id).strip().upper(),
            "scenario": normalize_scenario(scenario),
            "target_key": list(normalize_source_key(target_key)),
            "group_cols": [str(column).strip() for column in group_cols],
            "candidate_keys": [list(key) for key in normalized_candidates],
        }
    )


@dataclass(frozen=True)
class RankedDistance:
    source_key: SourceKey
    distance: float
    tie_group: int


def rank_source_distances(
    source_keys: Sequence[Sequence[object]],
    distances: np.ndarray,
    *,
    tie_tolerance: float,
) -> Tuple[RankedDistance, ...]:
    """Rank distances with anchored, non-chaining tie groups."""

    keys = tuple(normalize_source_key(key) for key in source_keys)
    values = np.asarray(distances, dtype=np.float64).reshape(-1)
    if len(keys) != values.size:
        raise ProtocolViolation("source key and distance counts differ")
    if len(set(keys)) != len(keys):
        raise ProtocolViolation("distance ranking contains duplicate source keys")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ProtocolViolation("source distances must be finite and non-negative")
    if not np.isfinite(tie_tolerance) or tie_tolerance < 0:
        raise ProtocolViolation("tie_tolerance must be finite and non-negative")

    ordered = sorted(zip(keys, values.tolist()), key=lambda item: (item[1], item[0]))
    ranked = []
    tie_group = 1
    start = 0
    while start < len(ordered):
        anchor = float(ordered[start][1])
        stop = start + 1
        while (
            stop < len(ordered)
            and float(ordered[stop][1]) - anchor <= tie_tolerance
        ):
            stop += 1
        current = [
            (key, float(raw_distance))
            for key, raw_distance in ordered[start:stop]
        ]
        current.sort(key=lambda item: item[0])
        ranked.extend(
            RankedDistance(key, raw_distance, tie_group)
            for key, raw_distance in current
        )
        start = stop
        tie_group += 1
    return tuple(ranked)


@dataclass(frozen=True)
class SelectionEntry:
    rank: int
    source_key: SourceKey
    distance: float
    weight: float
    tie_group: int
    observed_start: str
    observed_end: str
    raw_vector: Tuple[float, ...]
    scaled_vector: Tuple[float, ...]


@dataclass(frozen=True)
class SelectionResult:
    protocol_version: str
    dataset_id: str
    scenario: str
    target_key: SourceKey
    group_cols: Tuple[str, ...]
    feature_cols: Tuple[str, ...]
    observed_start: str
    observed_end: str
    source_observation_cutoff: str
    candidate_pool_digest: str
    candidate_pool_digest_input: Mapping[str, Any]
    selection_result_digest: str
    entries: Tuple[SelectionEntry, ...]
    excluded_candidates: Tuple[Mapping[str, Any], ...]
    scaler_min: float
    scaler_max: float

    @property
    def ordered_source_keys(self) -> Tuple[SourceKey, ...]:
        return tuple(entry.source_key for entry in self.entries)

    @property
    def distances(self) -> np.ndarray:
        return np.asarray([entry.distance for entry in self.entries], dtype=np.float64)

    @property
    def weights(self) -> np.ndarray:
        return np.asarray([entry.weight for entry in self.entries], dtype=np.float64)


def _float_text(value: float) -> str:
    converted = np.float64(value)
    if not np.isfinite(converted):
        raise ProtocolViolation("selection digest received a non-finite float")
    return format(float(converted), ".17g")


def build_selection_result_digest(
    *,
    protocol_version: str,
    candidate_pool_digest: str,
    k: int,
    weight_mode: str,
    weight_epsilon: float,
    entries: Sequence[SelectionEntry],
) -> str:
    payload = {
        "protocol_version": protocol_version,
        "candidate_pool_digest": candidate_pool_digest,
        "k": int(k),
        "weight_mode": weight_mode,
        "weight_epsilon": _float_text(weight_epsilon),
        "ordered_top_k": [
            {
                "rank": int(entry.rank),
                "source_key": list(entry.source_key),
                "distance": _float_text(entry.distance),
                "weight": _float_text(entry.weight),
                "tie_group": int(entry.tie_group),
            }
            for entry in entries
        ],
    }
    return _sha256_payload(payload)


def build_consumer_fingerprint(
    *,
    protocol_version: object,
    dataset_id: object,
    scenario: object,
    target_key: Sequence[object],
    source_pool_fingerprint: object,
    candidate_pool_digest: object,
    selection_result_digest: object,
    ordered_top_k: Sequence[Mapping[str, Any]],
) -> str:
    """Bind the exact candidate authority and ordered formal-consumer result."""

    normalized_top_k = []
    for row in ordered_top_k:
        normalized_top_k.append(
            {
                "source_rank": int(row["source_rank"]),
                "source_key": list(normalize_source_key(row["source_key"])),
                "distance": _float_text(float(row["distance"])),
                "weight": _float_text(float(row["weight"])),
                "tie_group": int(row["tie_group"]),
            }
        )
    return _sha256_payload(
        {
            "protocol_version": str(protocol_version).strip(),
            "dataset_id": str(dataset_id).strip().upper(),
            "scenario": normalize_scenario(scenario),
            "target_key": list(normalize_source_key(target_key)),
            "source_pool_fingerprint": str(source_pool_fingerprint),
            "candidate_pool_digest": str(candidate_pool_digest),
            "selection_result_digest": str(selection_result_digest),
            "ordered_top_k": normalized_top_k,
        }
    )


def _normalize_key_column(series: pd.Series) -> np.ndarray:
    """Normalize one key column without Python row-wise dataframe apply."""
    if series.isna().any():
        raise ProtocolViolation("source key components may not be null")
    inferred = pd.api.types.infer_dtype(series, skipna=False)
    if pd.api.types.is_bool_dtype(series) or inferred == "boolean":
        return np.where(series.to_numpy(dtype=bool), "true", "false").astype(str)
    if pd.api.types.is_integer_dtype(series) or inferred == "integer":
        return np.char.mod("%d", pd.to_numeric(series, errors="raise").to_numpy(dtype=np.int64))
    if pd.api.types.is_float_dtype(series) or inferred in {"floating", "mixed-integer-float"}:
        values = pd.to_numeric(series, errors="raise").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ProtocolViolation("source key components must be finite")
        normalized = np.char.mod("%.17g", values)
        normalized[values == 0.0] = "0"
        return normalized
    if inferred not in {"string", "unicode", "bytes", "categorical"}:
        # This runs only on the already de-duplicated raw-key table, never on
        # the million-row observation frame, and preserves mixed Python types
        # exactly as normalize_source_key specifies.
        return np.asarray(
            [normalize_source_key((value,))[0] for value in series.to_numpy()],
            dtype=str,
        )
    normalized = series.astype("string").str.strip()
    if normalized.isna().any() or normalized.eq("").any():
        raise ProtocolViolation("source key components may not be empty")
    return normalized.to_numpy(dtype=str)


class InsufficientCandidatePoolError(ProtocolViolation):
    """Strict insufficient-K failure carrying complete internal exclusions."""

    def __init__(
        self,
        *,
        valid_count: int,
        required_k: int,
        exclusions: Sequence[Mapping[str, Any]],
        eligible_count: int | None = None,
        target_key: Sequence[object] | None = None,
        scenario: str | None = None,
        sample_limit: int = 20,
    ) -> None:
        self.valid_count = int(valid_count)
        self.required_k = int(required_k)
        self.eligible_count = (
            int(eligible_count) if eligible_count is not None else self.valid_count + len(exclusions)
        )
        self.target_key = (
            normalize_source_key(target_key) if target_key is not None else None
        )
        self.scenario = str(scenario) if scenario is not None else None
        self.exclusions = tuple(dict(item) for item in exclusions)
        samples = self.exclusions[: int(sample_limit)]
        super().__init__(
            f"scenario={self.scenario!r} target_key={self.target_key!r} "
            f"eligible candidates={self.eligible_count} valid candidates={self.valid_count} "
            f"is below required K={self.required_k}; "
            f"excluded_count={len(self.exclusions)} excluded_samples={samples!r}"
        )


@dataclass(frozen=True)
class CanonicalSourceIndex:
    """One canonical key/date index for an immutable source context."""

    group_cols: Tuple[str, ...]
    row_count: int
    source_keys: Tuple[SourceKey, ...]
    keys_by_position: Tuple[SourceKey, ...]
    normalized_dates: Tuple[pd.Timestamp, ...]
    row_labels_by_key: Mapping[SourceKey, Tuple[object, ...]]
    duplicate_date_keys: frozenset[SourceKey]
    metadata_by_col: Mapping[str, Mapping[SourceKey, str]]
    keys_by_metadata_value: Mapping[str, Mapping[str, Tuple[SourceKey, ...]]]
    unique_row_labels: bool

    def __deepcopy__(self, memo: Dict[int, Any]) -> "CanonicalSourceIndex":
        memo[id(self)] = self
        return self

    def mask_for_normalized_key(
        self,
        frame: pd.DataFrame,
        key: SourceKey,
    ) -> pd.Series | None:
        if not self.unique_row_labels or not frame.index.is_unique:
            return None
        labels = self.row_labels_by_key.get(key)
        if labels is None:
            return pd.Series(False, index=frame.index)
        return pd.Series(frame.index.isin(labels), index=frame.index)


def build_canonical_source_index(
    source_df: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    metadata_cols: Sequence[str] = (),
) -> CanonicalSourceIndex:
    """Normalize every full-source key exactly once using protocol semantics."""

    normalized_group_cols = tuple(str(column) for column in group_cols)
    normalized_metadata_cols = tuple(str(column) for column in metadata_cols)
    required = [*normalized_group_cols, "date", *normalized_metadata_cols]
    missing = [column for column in required if column not in source_df.columns]
    if missing:
        raise ProtocolViolation(f"source dataframe missing canonical-index columns: {missing}")
    if not source_df.columns.is_unique:
        duplicates = list(
            dict.fromkeys(source_df.columns[source_df.columns.duplicated(keep=False)].tolist())
        )
        raise ProtocolViolation(
            f"source provenance dataframe contains duplicate columns: {duplicates!r}"
        )
    with temporarily_detached_attrs(source_df):
        dates = pd.to_datetime(source_df["date"], errors="coerce").dt.normalize()
        key_rows = tuple(
            source_df.loc[:, list(normalized_group_cols)].itertuples(
                index=False, name=None
            )
        )
        metadata_arrays = {
            column: source_df[column].to_numpy(copy=False)
            for column in normalized_metadata_cols
        }
    if dates.isna().any():
        raise ProtocolViolation("source provenance dataframe contains invalid dates")

    keys_by_position = tuple(
        normalize_source_key(tuple(row)) for row in key_rows
    )
    source_keys = tuple(sorted(set(keys_by_position)))
    labels_by_key: Dict[SourceKey, list[object]] = {key: [] for key in source_keys}
    dates_by_key: Dict[SourceKey, set[pd.Timestamp]] = {key: set() for key in source_keys}
    duplicate_date_keys: set[SourceKey] = set()
    for label, key, raw_date in zip(source_df.index, keys_by_position, dates):
        labels_by_key[key].append(label)
        normalized_date = pd.Timestamp(raw_date)
        if normalized_date in dates_by_key[key]:
            duplicate_date_keys.add(key)
        dates_by_key[key].add(normalized_date)

    metadata_maps: Dict[str, Mapping[SourceKey, str]] = {}
    metadata_indexes: Dict[str, Mapping[str, Tuple[SourceKey, ...]]] = {}
    for column in normalized_metadata_cols:
        values_by_key: Dict[SourceKey, str] = {}
        for key, raw_value in zip(keys_by_position, metadata_arrays[column]):
            if pd.isna(raw_value) or not str(raw_value).strip():
                raise ProtocolViolation(
                    f"source grouping metadata {column!r} contains null/empty values"
                )
            value = str(raw_value).strip()
            previous = values_by_key.setdefault(key, value)
            if previous != value:
                raise ProtocolViolation(
                    f"source key {key!r} maps to multiple {column} values"
                )
        keys_by_value: Dict[str, list[SourceKey]] = {}
        for key in source_keys:
            keys_by_value.setdefault(values_by_key[key], []).append(key)
        metadata_maps[column] = MappingProxyType(dict(values_by_key))
        metadata_indexes[column] = MappingProxyType(
            {value: tuple(sorted(keys)) for value, keys in keys_by_value.items()}
        )

    return CanonicalSourceIndex(
        group_cols=normalized_group_cols,
        row_count=int(len(source_df)),
        source_keys=source_keys,
        keys_by_position=keys_by_position,
        normalized_dates=tuple(pd.Timestamp(value) for value in dates),
        row_labels_by_key=MappingProxyType(
            {key: tuple(labels_by_key[key]) for key in source_keys}
        ),
        duplicate_date_keys=frozenset(duplicate_date_keys),
        metadata_by_col=MappingProxyType(metadata_maps),
        keys_by_metadata_value=MappingProxyType(metadata_indexes),
        unique_row_labels=bool(source_df.index.is_unique),
    )


@dataclass(frozen=True)
class PreparedDailySequencePool:
    """One immutable vectorized source observation index reusable across targets."""

    group_cols: Tuple[str, ...]
    required_dates: Tuple[str, ...]
    source_keys: Tuple[SourceKey, ...]
    sales_matrix: np.ndarray
    feature_matrices: Mapping[str, np.ndarray]
    date_presence_matrix: np.ndarray
    key_to_index: Mapping[SourceKey, int]
    duplicate_date_keys: frozenset[SourceKey]
    nonfinite_sales_keys: frozenset[SourceKey]
    nonfinite_feature_keys: Mapping[str, frozenset[SourceKey]]
    metadata_by_col: Mapping[str, Mapping[SourceKey, str]]
    keys_by_metadata_value: Mapping[str, Mapping[str, Tuple[SourceKey, ...]]]
    source_index: CanonicalSourceIndex

    def __deepcopy__(self, memo: Dict[int, Any]) -> "PreparedDailySequencePool":
        """Keep pandas attrs operations from copying the immutable pool."""
        memo[id(self)] = self
        return self

    def validate_for(
        self,
        *,
        group_cols: Sequence[str],
        required_dates: pd.DatetimeIndex,
    ) -> None:
        expected_dates = tuple(required_dates.strftime("%Y-%m-%d"))
        if tuple(group_cols) != self.group_cols:
            raise ProtocolViolation(
                f"prepared pool group_cols mismatch: {self.group_cols!r} != {tuple(group_cols)!r}"
            )
        if expected_dates != self.required_dates:
            raise ProtocolViolation("prepared pool observation dates differ from selection window")

    def source_identities(self, grouping_col: str) -> Tuple[Tuple[SourceKey, str], ...]:
        if grouping_col not in self.metadata_by_col:
            raise ProtocolViolation(
                f"prepared pool does not contain grouping metadata {grouping_col!r}"
            )
        values = self.metadata_by_col[grouping_col]
        return tuple((key, values[key]) for key in self.source_keys)

    def keys_for_metadata_value(
        self,
        grouping_col: str,
        group_value: object,
    ) -> Tuple[SourceKey, ...]:
        if grouping_col not in self.keys_by_metadata_value:
            raise ProtocolViolation(
                f"prepared pool does not contain grouping index {grouping_col!r}"
            )
        normalized_value = str(group_value).strip()
        return self.keys_by_metadata_value[grouping_col].get(normalized_value, ())

    def missing_dates_for(self, raw_key: Sequence[object]) -> Tuple[str, ...]:
        """Expand missing dates only for a candidate that needs diagnostics."""
        key = normalize_source_key(raw_key)
        index = self.key_to_index.get(key)
        if index is None:
            return self.required_dates
        missing_indices = np.flatnonzero(~self.date_presence_matrix[index])
        return tuple(self.required_dates[position] for position in missing_indices)

    def selected_sales_frame(self, keys: Sequence[Sequence[object]]) -> pd.DataFrame:
        """Materialize only selected 30-day sales rows for shared provenance checks."""
        return self.selected_frame(keys, feature_cols=("sales",))

    def selected_frame(
        self,
        keys: Sequence[Sequence[object]],
        *,
        feature_cols: Sequence[str],
    ) -> pd.DataFrame:
        """Materialize selected aligned observations using the declared KNN features."""
        normalized_features = tuple(str(column) for column in feature_cols)
        if not normalized_features:
            raise ProtocolViolation("selected frame requires non-empty feature columns")
        missing = [column for column in normalized_features if column not in self.feature_matrices]
        if missing:
            raise ProtocolViolation(
                f"prepared pool is missing declared feature columns: {missing!r}"
            )
        frames = []
        dates = pd.to_datetime(list(self.required_dates))
        for raw_key in keys:
            key = normalize_source_key(raw_key)
            index = self.key_to_index.get(key)
            if index is None:
                raise ProtocolViolation(f"selected source key is absent from prepared pool: {key!r}")
            payload: Dict[str, Any] = {
                column: key[position] for position, column in enumerate(self.group_cols)
            }
            payload["date"] = dates
            for column in normalized_features:
                payload[column] = self.feature_matrices[column][index].copy()
            frames.append(pd.DataFrame(payload))
        if not frames:
            return pd.DataFrame(columns=[*self.group_cols, "date", *normalized_features])
        return pd.concat(frames, ignore_index=True)


@dataclass(frozen=True)
class PreparedCandidateDateEligibility:
    """Read-only candidate date facts bound to one prepared-pool window."""

    pool_id: int
    group_cols: Tuple[str, ...]
    required_dates: Tuple[str, ...]
    feature_cols: Tuple[str, ...]
    candidate_scope: Tuple[SourceKey, ...]
    complete_candidate_scope: Tuple[SourceKey, ...]
    excluded_candidate_scope: Tuple[SourceKey, ...]
    missing_dates_by_candidate: Tuple[Tuple[SourceKey, Tuple[str, ...]], ...]

    def __deepcopy__(self, memo: Dict[int, Any]) -> "PreparedCandidateDateEligibility":
        memo[id(self)] = self
        return self

    def exclusion_records(self) -> Tuple[Mapping[str, Any], ...]:
        return tuple(
            {
                "source_key": key,
                "reason": "missing_observed_dates",
                "missing_dates": missing_dates,
            }
            for key, missing_dates in self.missing_dates_by_candidate
        )


def classify_prepared_candidate_dates(
    pool: PreparedDailySequencePool,
    candidate_keys: Iterable[Sequence[object]],
    *,
    group_cols: Sequence[str],
    required_dates: pd.DatetimeIndex,
    feature_cols: Sequence[str],
) -> PreparedCandidateDateEligibility:
    """Classify duplicate/missing dates before any candidate numeric validation."""

    normalized_group_cols = tuple(str(column) for column in group_cols)
    normalized_features = tuple(str(column) for column in feature_cols)
    normalized_candidates = tuple(
        normalize_source_key(key) for key in candidate_keys
    )
    if len(set(normalized_candidates)) != len(normalized_candidates):
        raise ProtocolViolation("candidate pool contains duplicate source keys")
    pool.validate_for(
        group_cols=normalized_group_cols,
        required_dates=required_dates,
    )
    missing_features = [
        column for column in normalized_features if column not in pool.feature_matrices
    ]
    if missing_features:
        raise ProtocolViolation(
            f"prepared pool is missing declared feature columns: {missing_features!r}"
        )

    complete = []
    missing_by_candidate = []
    for candidate_key in normalized_candidates:
        if candidate_key in pool.duplicate_date_keys:
            raise ProtocolViolation(
                f"source {candidate_key!r} contains duplicate observed dates"
            )
        missing_dates = pool.missing_dates_for(candidate_key)
        if missing_dates:
            missing_by_candidate.append((candidate_key, tuple(missing_dates)))
            continue
        complete.append(candidate_key)

    excluded = tuple(key for key, _ in missing_by_candidate)
    return PreparedCandidateDateEligibility(
        pool_id=id(pool),
        group_cols=normalized_group_cols,
        required_dates=tuple(required_dates.strftime("%Y-%m-%d")),
        feature_cols=normalized_features,
        candidate_scope=normalized_candidates,
        complete_candidate_scope=tuple(complete),
        excluded_candidate_scope=excluded,
        missing_dates_by_candidate=tuple(missing_by_candidate),
    )


def validate_prepared_candidate_date_eligibility(
    pool: PreparedDailySequencePool,
    proof: PreparedCandidateDateEligibility,
    candidate_keys: Iterable[Sequence[object]],
    *,
    group_cols: Sequence[str],
    required_dates: pd.DatetimeIndex,
    feature_cols: Sequence[str],
) -> PreparedCandidateDateEligibility:
    """Fail closed unless a proof exactly matches current pool date facts."""

    if not isinstance(proof, PreparedCandidateDateEligibility):
        raise ProtocolViolation("prepared-pool date eligibility proof is invalid")
    expected = classify_prepared_candidate_dates(
        pool,
        candidate_keys,
        group_cols=group_cols,
        required_dates=required_dates,
        feature_cols=feature_cols,
    )
    if proof != expected:
        raise ProtocolViolation(
            "prepared-pool date eligibility proof does not match current pool identity"
        )
    return proof


def prepare_daily_sequence_pool(
    source_df: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    observed_start: object,
    observed_end: object | None = None,
    metadata_cols: Sequence[str] = (),
    feature_cols: Sequence[str] | None = None,
    source_index: CanonicalSourceIndex | None = None,
) -> PreparedDailySequencePool:
    """Prepare all source keys and aligned 30-day feature matrices exactly once."""
    normalized_group_cols = tuple(str(column) for column in group_cols)
    if not normalized_group_cols:
        raise ProtocolViolation("prepared pool group_cols may not be empty")
    normalized_features = tuple(
        str(column) for column in (("sales",) if feature_cols is None else feature_cols)
    )
    if not normalized_features or len(set(normalized_features)) != len(normalized_features):
        raise ProtocolViolation("prepared pool feature columns must be non-empty and unique")
    required_columns = [*normalized_group_cols, "date", *normalized_features, *metadata_cols]
    missing = [column for column in required_columns if column not in source_df.columns]
    if missing:
        raise ProtocolViolation(f"source dataframe missing prepared-pool columns: {missing}")

    start = pd.Timestamp(observed_start).normalize()
    end = start + pd.Timedelta(days=29) if observed_end is None else pd.Timestamp(observed_end).normalize()
    required_dates = pd.date_range(start, end, freq="D")
    if len(required_dates) != 30:
        raise ProtocolViolation("prepared source observation window must contain exactly 30 days")

    if source_index is None:
        source_index = build_canonical_source_index(
            source_df,
            group_cols=normalized_group_cols,
            metadata_cols=metadata_cols,
        )
    elif (
        source_index.group_cols != normalized_group_cols
        or source_index.row_count != len(source_df)
    ):
        raise ProtocolViolation("canonical source index does not match prepared-pool frame")
    missing_metadata = [
        str(column)
        for column in metadata_cols
        if str(column) not in source_index.metadata_by_col
    ]
    if missing_metadata:
        raise ProtocolViolation(
            f"canonical source index is missing metadata columns: {missing_metadata!r}"
        )
    parsed_dates = pd.Series(source_index.normalized_dates, index=source_df.index)
    working_source = copy_frame_with_lightweight_attrs(source_df, deep=False)
    normalized_keys = source_index.source_keys
    key_to_index = {key: index for index, key in enumerate(normalized_keys)}

    observed_mask = parsed_dates.isin(required_dates)
    observed = working_source.loc[
        observed_mask, [*normalized_group_cols, *normalized_features]
    ].copy()
    observed["date"] = parsed_dates.loc[observed_mask].to_numpy()
    observed_positions = np.flatnonzero(observed_mask.to_numpy())
    observed["__protocol_source_key__"] = [
        source_index.keys_by_position[int(position)]
        for position in observed_positions
    ]
    observed["__protocol_key_index__"] = observed[
        "__protocol_source_key__"
    ].map(key_to_index)
    for column in normalized_features:
        observed[column] = pd.to_numeric(observed[column], errors="coerce")

    duplicate_rows = observed.duplicated(["__protocol_key_index__", "date"], keep=False)
    duplicate_indices = set(observed.loc[duplicate_rows, "__protocol_key_index__"].astype(int))
    nonfinite_feature_indices: Dict[str, set[int]] = {}
    for column in normalized_features:
        nonfinite_rows = ~np.isfinite(observed[column].to_numpy(dtype=np.float64))
        nonfinite_feature_indices[column] = set(
            observed.loc[nonfinite_rows, "__protocol_key_index__"].astype(int)
        )
    nonfinite_indices = nonfinite_feature_indices.get("sales", set())

    first_rows = observed.drop_duplicates(["__protocol_key_index__", "date"], keep="first")
    presence = first_rows.assign(__protocol_date_present__=True).pivot(
        index="__protocol_key_index__",
        columns="date",
        values="__protocol_date_present__",
    )
    presence = presence.reindex(index=range(len(normalized_keys)), columns=required_dates)
    date_presence_matrix = presence.notna().to_numpy(dtype=bool, copy=True)
    date_presence_matrix.setflags(write=False)

    feature_matrices: Dict[str, np.ndarray] = {}
    for column in normalized_features:
        pivot = first_rows.pivot(
            index="__protocol_key_index__", columns="date", values=column
        )
        pivot = pivot.reindex(index=range(len(normalized_keys)), columns=required_dates)
        matrix = pivot.to_numpy(dtype=np.float64, copy=True)
        matrix.setflags(write=False)
        feature_matrices[column] = matrix
    sales_matrix = feature_matrices.get("sales")
    if sales_matrix is None:
        raise ProtocolViolation("prepared pool feature columns must include sales")

    return PreparedDailySequencePool(
        group_cols=normalized_group_cols,
        required_dates=tuple(required_dates.strftime("%Y-%m-%d")),
        source_keys=normalized_keys,
        sales_matrix=sales_matrix,
        feature_matrices=MappingProxyType(feature_matrices),
        date_presence_matrix=date_presence_matrix,
        key_to_index=MappingProxyType(key_to_index),
        duplicate_date_keys=frozenset(normalized_keys[index] for index in duplicate_indices),
        nonfinite_sales_keys=frozenset(normalized_keys[index] for index in nonfinite_indices),
        nonfinite_feature_keys=MappingProxyType(
            {
                column: frozenset(
                    normalized_keys[index] for index in indices
                )
                for column, indices in nonfinite_feature_indices.items()
            }
        ),
        metadata_by_col=source_index.metadata_by_col,
        keys_by_metadata_value=source_index.keys_by_metadata_value,
        source_index=source_index,
    )


def _prepare_dates(frame: pd.DataFrame, *, role: str) -> pd.DataFrame:
    if "date" not in frame.columns:
        raise ProtocolViolation(f"{role} dataframe requires date column")
    prepared = frame.copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce").dt.normalize()
    if prepared["date"].isna().any():
        raise ProtocolViolation(f"{role} dataframe contains invalid dates")
    return prepared


def _exact_observed_vector(
    frame: pd.DataFrame,
    required_dates: pd.DatetimeIndex,
    *,
    role: str,
    feature_cols: Sequence[str],
) -> np.ndarray:
    observed = frame[frame["date"].isin(required_dates)].sort_values("date")
    if observed["date"].duplicated().any():
        raise ProtocolViolation(f"{role} contains duplicate observed dates")
    actual_dates = pd.DatetimeIndex(observed["date"])
    if not actual_dates.equals(required_dates):
        missing = required_dates.difference(actual_dates).strftime("%Y-%m-%d").tolist()
        raise ProtocolViolation(f"{role} missing observed dates: {missing}")
    vectors = []
    for column in feature_cols:
        values = pd.to_numeric(observed[column], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ProtocolViolation(
                f"{role} observed feature {column!r} contains non-finite values"
            )
        vectors.append(values)
    return np.concatenate(vectors).astype(np.float64, copy=False)


def select_daily_sequence_sources(
    *,
    target_df: pd.DataFrame,
    source_df: pd.DataFrame,
    prepared_pool: PreparedDailySequencePool | None = None,
    protocol: ExperimentProtocol,
    scenario: object,
    target_key: Sequence[object],
    candidate_keys: Iterable[Sequence[object]],
    group_cols: Sequence[str],
    observed_start: object,
    feature_cols: Sequence[str],
    k: int,
) -> SelectionResult:
    """Select exactly K sources using protocol-declared aligned 30-day features."""

    declared_features = tuple(protocol.knn_feature_columns)
    requested_features = tuple(str(column) for column in feature_cols)
    if requested_features != declared_features:
        raise ProtocolViolation(
            f"feature_cols must be exactly the {protocol.dataset_id} protocol KNN features: "
            f"expected {declared_features!r}, got {requested_features!r}"
        )
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ProtocolViolation("K must be a positive integer")
    normalized_scenario = normalize_scenario(scenario)
    normalized_target = normalize_source_key(target_key)
    normalized_candidates = tuple(normalize_source_key(key) for key in candidate_keys)
    if len(set(normalized_candidates)) != len(normalized_candidates):
        raise ProtocolViolation("candidate pool contains duplicate source keys")
    if normalized_target in normalized_candidates:
        raise ProtocolViolation("target key may not enter candidate pool")
    if len(tuple(group_cols)) != len(normalized_target):
        raise ProtocolViolation("group_cols do not match target key arity")
    if prepared_pool is None:
        missing_group_cols = [column for column in group_cols if column not in source_df.columns]
        if missing_group_cols:
            raise ProtocolViolation(f"source dataframe missing group columns: {missing_group_cols}")
        if "sales" not in source_df.columns:
            raise ProtocolViolation("source dataframe requires sales column")
    missing_target_features = [
        column for column in declared_features if column not in target_df.columns
    ]
    missing_source_features = [
        column for column in declared_features if column not in source_df.columns
    ]
    if missing_target_features:
        raise ProtocolViolation(
            f"target dataframe is missing protocol KNN features: {missing_target_features!r}"
        )
    if missing_source_features and prepared_pool is None:
        raise ProtocolViolation(
            f"source dataframe is missing protocol KNN features: {missing_source_features!r}"
        )

    calendarization_rule_version = source_df.attrs.get(
        "d2_source_calendarization_rule_version"
    )
    source_authority_digest = source_df.attrs.get("d2_source_authority_digest")
    consumer_frame_fingerprint = source_df.attrs.get("d2_consumer_frame_fingerprint")
    if protocol.dataset_id == "D2" and any(
        value is None
        for value in (
            calendarization_rule_version,
            source_authority_digest,
            consumer_frame_fingerprint,
        )
    ):
        raise ProtocolViolation(
            "D2 source calendarization identity is required before KNN"
        )
    source_history_identity = {}
    if "source_history_days" in source_df.attrs:
        source_history_identity = {
            "source_history_days": source_df.attrs.get("source_history_days"),
            "source_history_start": source_df.attrs.get("source_history_start"),
            "source_history_end": source_df.attrs.get("source_history_end"),
            "source_history_completeness_policy": source_df.attrs.get(
                "source_history_completeness_policy"
            ),
            "source_history_frame_digest": source_df.attrs.get(
                "source_history_frame_digest"
            ),
        }
        if any(value is None for value in source_history_identity.values()):
            raise ProtocolViolation(
                "D4-D6 source history identity is incomplete before KNN"
            )

    window = protocol.observation_window(observed_start)
    observed_start_iso = window.knn_observed_start.isoformat()
    observed_end_iso = window.knn_observed_end.isoformat()
    required_dates = pd.date_range(observed_start_iso, observed_end_iso, freq="D")

    target = _prepare_dates(target_df, role="target")
    target_legal = target.loc[target["date"].isin(required_dates)].copy()
    target_vector = _exact_observed_vector(
        target,
        required_dates,
        role="target",
        feature_cols=declared_features,
    )
    pool = prepared_pool or prepare_daily_sequence_pool(
        source_df,
        group_cols=group_cols,
        observed_start=observed_start_iso,
        observed_end=observed_end_iso,
        feature_cols=declared_features,
    )
    pool.validate_for(group_cols=group_cols, required_dates=required_dates)
    date_eligibility = classify_prepared_candidate_dates(
        pool,
        normalized_candidates,
        group_cols=group_cols,
        required_dates=required_dates,
        feature_cols=declared_features,
    )

    source_frame_digest = None
    target_frame_digest = None
    if protocol.dataset_id in {"D1", "D2"}:
        digest_feature_cols = (
            None if declared_features == ("sales",) else declared_features
        )
        digest_ignored_columns = ("promo",) if digest_feature_cols is None else ()
        target_digest_frame = (
            target_df
            if len(target_df) == len(target_legal)
            else target_legal
        )
        source_dates = pd.to_datetime(source_df["date"], errors="coerce").dt.normalize()
        source_is_observed = bool(
            not source_dates.isna().any()
            and source_dates.isin(required_dates).all()
        )
        source_digest_frame = (
            source_df
            if source_is_observed
            else pool.selected_frame(
                normalized_candidates,
                feature_cols=declared_features,
            )
        )
        target_frame_digest = canonical_knn_frame_digest(
            target_digest_frame,
            group_cols=group_cols,
            feature_cols=digest_feature_cols,
            ignore_columns=digest_ignored_columns,
        )
        source_frame_digest = canonical_knn_frame_digest(
            source_digest_frame,
            group_cols=group_cols,
            feature_cols=digest_feature_cols,
            ignore_columns=digest_ignored_columns,
        )

    valid_keys = []
    raw_vectors = []
    excluded = list(date_eligibility.exclusion_records())
    for candidate_key in date_eligibility.complete_candidate_scope:
        for column in declared_features:
            if candidate_key in pool.nonfinite_feature_keys.get(column, frozenset()):
                raise ProtocolViolation(
                    f"source {candidate_key!r} observed feature {column!r} "
                    "contains non-finite values"
                )
        pool_index = pool.key_to_index.get(candidate_key)
        if pool_index is None:
            raise ProtocolViolation(f"prepared pool key lookup failed for {candidate_key!r}")
        values = np.concatenate(
            [pool.feature_matrices[column][pool_index] for column in declared_features]
        ).astype(np.float64, copy=False)
        valid_keys.append(candidate_key)
        raw_vectors.append(values)

    if len(valid_keys) < k:
        raise InsufficientCandidatePoolError(
            valid_count=len(valid_keys),
            required_k=k,
            exclusions=excluded,
            eligible_count=len(normalized_candidates),
            target_key=normalized_target,
            scenario=normalized_scenario,
        )

    source_matrix = np.vstack(raw_vectors).astype(np.float64)
    scaler_values = np.concatenate((target_vector, source_matrix.reshape(-1)))
    scaler_min = float(np.min(scaler_values))
    scaler_max = float(np.max(scaler_values))
    scaler_range = scaler_max - scaler_min
    if scaler_range == 0.0:
        scaled_target = np.zeros_like(target_vector, dtype=np.float64)
        scaled_sources = np.zeros_like(source_matrix, dtype=np.float64)
    else:
        scaled_target = (target_vector - scaler_min) / scaler_range
        scaled_sources = (source_matrix - scaler_min) / scaler_range
    distances = np.linalg.norm(scaled_sources - scaled_target, axis=1).astype(np.float64)
    ranked = rank_source_distances(
        valid_keys,
        distances,
        tie_tolerance=protocol.tie_tolerance,
    )
    selected_ranked = ranked[:k]
    selected_distances = np.asarray(
        [entry.distance for entry in selected_ranked], dtype=np.float64
    )
    scores = 1.0 / (selected_distances + np.float64(protocol.weight_epsilon))
    weights = scores / np.sum(scores, dtype=np.float64)
    if not np.isfinite(weights).all():
        raise ProtocolViolation("source weights are non-finite")

    raw_by_key = {key: vector for key, vector in zip(valid_keys, raw_vectors)}
    scaled_by_key = {key: vector for key, vector in zip(valid_keys, scaled_sources)}
    entries = tuple(
        SelectionEntry(
            rank=index,
            source_key=ranked_entry.source_key,
            distance=float(ranked_entry.distance),
            weight=float(weights[index - 1]),
            tie_group=ranked_entry.tie_group,
            observed_start=observed_start_iso,
            observed_end=observed_end_iso,
            raw_vector=tuple(float(value) for value in raw_by_key[ranked_entry.source_key]),
            scaled_vector=tuple(float(value) for value in scaled_by_key[ranked_entry.source_key]),
        )
        for index, ranked_entry in enumerate(selected_ranked, start=1)
    )

    digest_input = _canonical_candidate_pool_input(
        protocol.protocol_version,
        protocol.dataset_id,
        normalized_scenario,
        normalized_target,
        tuple(group_cols),
        normalized_candidates,
        observed_start_iso,
        observed_end_iso,
        tuple(feature_cols),
        calendarization_rule_version,
        source_authority_digest,
        consumer_frame_fingerprint,
        source_frame_digest,
        target_frame_digest,
        **source_history_identity,
    )
    candidate_digest = _sha256_payload(digest_input)
    selection_digest = build_selection_result_digest(
        protocol_version=protocol.protocol_version,
        candidate_pool_digest=candidate_digest,
        k=k,
        weight_mode=protocol.weight_mode,
        weight_epsilon=protocol.weight_epsilon,
        entries=entries,
    )
    return SelectionResult(
        protocol_version=protocol.protocol_version,
        dataset_id=protocol.dataset_id,
        scenario=normalized_scenario,
        target_key=normalized_target,
        group_cols=tuple(group_cols),
        feature_cols=tuple(feature_cols),
        observed_start=observed_start_iso,
        observed_end=observed_end_iso,
        source_observation_cutoff=observed_end_iso,
        candidate_pool_digest=candidate_digest,
        candidate_pool_digest_input=digest_input,
        selection_result_digest=selection_digest,
        entries=entries,
        excluded_candidates=tuple(excluded),
        scaler_min=scaler_min,
        scaler_max=scaler_max,
    )
