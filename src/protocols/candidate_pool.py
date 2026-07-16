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

from src.data_processing.sealed_daily import calendarize_and_fill, canonicalize_source_sales

from .experiment_protocol import (
    ExperimentProtocol,
    ProtocolViolation,
    SourceKey,
    normalize_scenario,
    normalize_source_key,
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
) -> Dict[str, Any]:
    normalized_candidates = [normalize_source_key(key) for key in candidate_keys]
    if len(set(normalized_candidates)) != len(normalized_candidates):
        raise ProtocolViolation("candidate pool contains duplicate source keys")
    normalized_candidates.sort()
    return {
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
    )
    return _sha256_payload(payload)


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
    vector_shape: Tuple[int, int] = (30, 1)
    vector_digest: str = ""
    source_repair_digest: str = ""
    source_training_digest: str = ""


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
    source_window_start: str = ""
    source_window_end: str = ""
    knn_window_start: str = ""
    knn_window_end: str = ""
    knn_schema_digest: str = ""
    source_repair_digest: str = ""
    source_training_digest: str = ""
    selection_identity_digest: str = ""

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
class PreparedDailySequencePool:
    """One immutable vectorized source observation index reusable across targets."""

    group_cols: Tuple[str, ...]
    required_dates: Tuple[str, ...]
    source_keys: Tuple[SourceKey, ...]
    sales_matrix: np.ndarray
    date_presence_matrix: np.ndarray
    key_to_index: Mapping[SourceKey, int]
    duplicate_date_keys: frozenset[SourceKey]
    nonfinite_sales_keys: frozenset[SourceKey]
    metadata_by_col: Mapping[str, Mapping[SourceKey, str]]
    keys_by_metadata_value: Mapping[str, Mapping[str, Tuple[SourceKey, ...]]]
    pretrain_dates: Tuple[str, ...] = ()
    knn_feature_cols: Tuple[str, ...] = ("sales",)
    required_feature_cols: Tuple[str, ...] = ("sales",)
    knn_matrix: np.ndarray | None = None
    source_frames: Mapping[SourceKey, pd.DataFrame] = MappingProxyType({})
    repair_audits: Mapping[SourceKey, Mapping[str, Any]] = MappingProxyType({})
    ineligible_reasons: Mapping[SourceKey, Tuple[str, ...]] = MappingProxyType({})
    vector_digests: Mapping[SourceKey, str] = MappingProxyType({})
    training_digests: Mapping[SourceKey, str] = MappingProxyType({})

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
        """Materialize canonical selected source rows for provenance checks."""
        if self.source_frames:
            return self.selected_source_frame(keys)
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
            payload["sales"] = self.sales_matrix[index].copy()
            frames.append(pd.DataFrame(payload))
        if not frames:
            return pd.DataFrame(columns=[*self.group_cols, "date", "sales"])
        return pd.concat(frames, ignore_index=True)

    def selected_source_frame(self, keys: Sequence[Sequence[object]]) -> pd.DataFrame:
        frames = []
        for raw_key in keys:
            key = normalize_source_key(raw_key)
            frame = self.source_frames.get(key)
            if frame is None:
                raise ProtocolViolation(f"selected source key is absent from prepared pool: {key!r}")
            frames.append(frame.copy())
        if not frames:
            return pd.DataFrame(columns=[*self.group_cols, "date", *self.required_feature_cols])
        return pd.concat(frames, ignore_index=True)

    def repair_audit_for(self, raw_key: Sequence[object]) -> Mapping[str, Any]:
        key = normalize_source_key(raw_key)
        if key not in self.repair_audits:
            raise ProtocolViolation(f"source repair audit is absent for {key!r}")
        return self.repair_audits[key]

    def ineligible_reasons_for(self, raw_key: Sequence[object]) -> Tuple[str, ...]:
        return self.ineligible_reasons.get(normalize_source_key(raw_key), ())


def prepare_daily_sequence_pool(
    source_df: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    observed_start: object,
    observed_end: object | None = None,
    metadata_cols: Sequence[str] = (),
    pretrain_start: object | None = None,
    pretrain_end: object | None = None,
    knn_feature_cols: Sequence[str] = ("sales",),
    required_feature_cols: Sequence[str] = ("sales",),
) -> PreparedDailySequencePool:
    """Calendarize and validate sources before exposing final-30-day KNN vectors."""
    normalized_group_cols = tuple(str(column) for column in group_cols)
    if not normalized_group_cols:
        raise ProtocolViolation("prepared pool group_cols may not be empty")
    normalized_knn_features = tuple(str(column) for column in knn_feature_cols)
    normalized_required_features = tuple(
        dict.fromkeys(str(column) for column in required_feature_cols)
    )
    if not normalized_knn_features or normalized_knn_features[0] != "sales":
        raise ProtocolViolation("KNN feature schema must start with sales")
    if "sales" not in normalized_required_features:
        normalized_required_features = ("sales", *normalized_required_features)
    required_columns = [
        *normalized_group_cols,
        "date",
        *dict.fromkeys((*normalized_knn_features, *normalized_required_features)),
        *metadata_cols,
    ]
    missing = [column for column in required_columns if column not in source_df.columns]
    if missing:
        raise ProtocolViolation(f"source dataframe missing prepared-pool columns: {missing}")

    start = pd.Timestamp(observed_start).normalize()
    end = start + pd.Timedelta(days=29) if observed_end is None else pd.Timestamp(observed_end).normalize()
    required_dates = pd.date_range(start, end, freq="D")
    if len(required_dates) != 30:
        raise ProtocolViolation("prepared source observation window must contain exactly 30 days")

    full_end = end if pretrain_end is None else pd.Timestamp(pretrain_end).normalize()
    full_start = start if pretrain_start is None else pd.Timestamp(pretrain_start).normalize()
    full_dates = pd.date_range(full_start, full_end, freq="D")
    if full_end != end:
        raise ProtocolViolation("source pretrain window must end with the KNN window")
    if pretrain_start is not None and len(full_dates) != 180:
        raise ProtocolViolation("source pretrain window must contain exactly 180 days")
    if not required_dates.equals(full_dates[-30:]):
        raise ProtocolViolation("KNN window must be the final 30 source pretrain days")

    parsed_dates = pd.to_datetime(source_df["date"], errors="coerce").dt.normalize()
    if parsed_dates.isna().any():
        raise ProtocolViolation("source dataframe contains invalid dates")

    raw_key_table = source_df.loc[:, list(normalized_group_cols)].drop_duplicates().reset_index(drop=True)
    normalized_arrays = [_normalize_key_column(raw_key_table[column]) for column in normalized_group_cols]
    raw_key_table["__protocol_source_key__"] = list(zip(*normalized_arrays))
    normalized_keys = tuple(sorted(set(raw_key_table["__protocol_source_key__"])))
    key_to_index = {key: index for index, key in enumerate(normalized_keys)}
    raw_key_table["__protocol_key_index__"] = raw_key_table["__protocol_source_key__"].map(key_to_index)

    metadata_maps: Dict[str, Mapping[SourceKey, str]] = {}
    metadata_key_indexes: Dict[str, Mapping[str, Tuple[SourceKey, ...]]] = {}
    if metadata_cols:
        metadata_table = source_df.loc[:, [*normalized_group_cols, *metadata_cols]].drop_duplicates()
        metadata_table = metadata_table.merge(raw_key_table, on=list(normalized_group_cols), how="left", validate="many_to_one")
        for column in metadata_cols:
            normalized_values = metadata_table[column].astype("string").str.strip()
            if normalized_values.isna().any() or normalized_values.eq("").any():
                raise ProtocolViolation(f"source grouping metadata {column!r} contains null/empty values")
            metadata_table[f"__meta_{column}"] = normalized_values
            grouped = metadata_table.groupby("__protocol_source_key__", sort=False)[f"__meta_{column}"]
            counts = grouped.nunique(dropna=False)
            conflicts = counts[counts != 1]
            if not conflicts.empty:
                raise ProtocolViolation(
                    f"source key {conflicts.index[0]!r} maps to multiple {column} values"
                )
            values = grouped.first().to_dict()
            metadata_maps[str(column)] = MappingProxyType(
                {key: str(values[key]) for key in normalized_keys}
            )
            keys_by_value: Dict[str, list[SourceKey]] = {}
            for key in normalized_keys:
                keys_by_value.setdefault(str(values[key]), []).append(key)
            metadata_key_indexes[str(column)] = MappingProxyType(
                {
                    value: tuple(sorted(keys))
                    for value, keys in keys_by_value.items()
                }
            )

    working = source_df.copy()
    working["date"] = parsed_dates
    working = working.merge(
        raw_key_table.loc[:, [*normalized_group_cols, "__protocol_source_key__"]],
        on=list(normalized_group_cols),
        how="left",
        validate="many_to_one",
    )
    duplicate_rows = working.duplicated(
        ["__protocol_source_key__", "date"], keep=False
    )
    duplicate_keys = set(working.loc[duplicate_rows, "__protocol_source_key__"])
    working = working.drop_duplicates(
        ["__protocol_source_key__", "date"], keep="first"
    )
    working = calendarize_and_fill(
        working.drop(columns="__protocol_source_key__"),
        group_cols=normalized_group_cols,
        start=full_start,
        end=full_end,
        fill_rules={},
    )
    working["__protocol_calendar_row_missing__"] = tuple(
        working.attrs["calendar_row_missing_mask"]
    )
    working = working.merge(
        raw_key_table.loc[:, [*normalized_group_cols, "__protocol_source_key__"]],
        on=list(normalized_group_cols),
        how="left",
        validate="many_to_one",
    )
    grouped_working = {
        key: group.drop(columns="__protocol_source_key__")
        for key, group in working.groupby("__protocol_source_key__", sort=False)
    }
    source_frames: Dict[SourceKey, pd.DataFrame] = {}
    repair_audits: Dict[SourceKey, Mapping[str, Any]] = {}
    ineligible_reasons: Dict[SourceKey, Tuple[str, ...]] = {}
    vector_digests: Dict[SourceKey, str] = {}
    training_digests: Dict[SourceKey, str] = {}
    nonfinite_sales_keys: set[SourceKey] = set()
    knn_vectors = []
    sales_vectors = []
    presence_rows = []

    for key in normalized_keys:
        raw = grouped_working.get(key, working.iloc[0:0]).copy()
        raw = raw[raw["date"].between(full_start, full_end, inclusive="both")]
        if key in duplicate_keys:
            ineligible_reasons[key] = ("duplicate_source_date",)
        missing_mask = raw.pop("__protocol_calendar_row_missing__").to_numpy(dtype=bool)
        indexed = raw.set_index("date").reindex(full_dates)
        indexed["date"] = full_dates
        for column, value in zip(normalized_group_cols, key):
            indexed[column] = value
        canonical_input = indexed.reset_index(drop=True)
        reasons = list(ineligible_reasons.get(key, ()))
        if missing_mask.any():
            reasons.append(
                "missing_observed_dates" if missing_mask[-30:].any()
                else "missing_source_history_dates"
            )
        try:
            canonical, repair = canonicalize_source_sales(
                canonical_input,
                calendar_row_missing=missing_mask,
            )
        except ValueError as exc:
            if "infinity" not in str(exc):
                raise
            reasons.append("source_sales_infinity")
            nonfinite_sales_keys.add(key)
            canonical = canonical_input
            repair = {
                "version": "source_sales_canonicalization/v1",
                "repair_reason_counts": {},
                "affected_date_digest": "",
                "repair_mask_sha256": "",
                "affected_rows": [],
                "rows_examined": len(canonical_input),
            }

        for calendar_column, values in {
            "year": full_dates.year,
            "month": full_dates.month,
            "week": full_dates.isocalendar().week.to_numpy(dtype=np.int64),
            "day": full_dates.day,
            "weekday": full_dates.weekday,
        }.items():
            if calendar_column in normalized_required_features and calendar_column in canonical:
                canonical[calendar_column] = np.asarray(values)

        unresolved = False
        for column in normalized_required_features:
            numeric = pd.to_numeric(canonical[column], errors="coerce").to_numpy(dtype=np.float64)
            if not np.isfinite(numeric).all():
                unresolved = True
                break
            canonical[column] = numeric
        for column in normalized_knn_features:
            numeric = pd.to_numeric(canonical[column], errors="coerce")
            canonical[column] = numeric
            knn_numeric = numeric[canonical["date"].isin(required_dates)].to_numpy(
                dtype=np.float64
            )
            if not np.isfinite(knn_numeric).all():
                unresolved = True
                break
        if unresolved and "source_sales_infinity" not in reasons:
            reasons.append("unresolved_required_feature")
        ineligible_reasons[key] = tuple(dict.fromkeys(reasons))
        source_frames[key] = canonical
        repair_audits[key] = MappingProxyType(dict(repair))
        presence_rows.append((~missing_mask)[-30:])

        if reasons:
            knn_vectors.append(np.full(30 * len(normalized_knn_features), np.nan))
            sales_vectors.append(np.full(30, np.nan))
            continue
        knn_rows = canonical[canonical["date"].isin(required_dates)].sort_values("date")
        matrix = knn_rows.loc[:, list(normalized_knn_features)].to_numpy(dtype=np.float64)
        vector = matrix.reshape(-1)
        training_matrix = canonical.loc[:, list(normalized_required_features)].to_numpy(
            dtype=np.float64
        )
        vector_payload = {
            "dates": list(required_dates.strftime("%Y-%m-%d")),
            "feature_cols": list(normalized_knn_features),
            "shape": [30, len(normalized_knn_features)],
            "values": [_float_text(value) for value in vector],
        }
        training_payload = {
            "dates": list(full_dates.strftime("%Y-%m-%d")),
            "feature_cols": list(normalized_required_features),
            "shape": list(training_matrix.shape),
            "values": [_float_text(value) for value in training_matrix.reshape(-1)],
        }
        vector_digests[key] = _sha256_payload(vector_payload)
        training_digests[key] = _sha256_payload(training_payload)
        knn_vectors.append(vector)
        sales_vectors.append(matrix[:, normalized_knn_features.index("sales")])

    knn_matrix = np.asarray(knn_vectors, dtype=np.float64)
    knn_matrix.setflags(write=False)
    sales_matrix = np.asarray(sales_vectors, dtype=np.float64)
    sales_matrix.setflags(write=False)
    date_presence_matrix = np.asarray(presence_rows, dtype=bool)
    date_presence_matrix.setflags(write=False)

    return PreparedDailySequencePool(
        group_cols=normalized_group_cols,
        required_dates=tuple(required_dates.strftime("%Y-%m-%d")),
        source_keys=normalized_keys,
        sales_matrix=sales_matrix,
        date_presence_matrix=date_presence_matrix,
        key_to_index=MappingProxyType(key_to_index),
        duplicate_date_keys=frozenset(duplicate_keys),
        nonfinite_sales_keys=frozenset(nonfinite_sales_keys),
        metadata_by_col=MappingProxyType(metadata_maps),
        keys_by_metadata_value=MappingProxyType(metadata_key_indexes),
        pretrain_dates=tuple(full_dates.strftime("%Y-%m-%d")),
        knn_feature_cols=normalized_knn_features,
        required_feature_cols=normalized_required_features,
        knn_matrix=knn_matrix,
        source_frames=MappingProxyType(source_frames),
        repair_audits=MappingProxyType(repair_audits),
        ineligible_reasons=MappingProxyType(ineligible_reasons),
        vector_digests=MappingProxyType(vector_digests),
        training_digests=MappingProxyType(training_digests),
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
    feature_cols: Sequence[str] = ("sales",),
) -> np.ndarray:
    observed = frame[frame["date"].isin(required_dates)].sort_values("date")
    if observed["date"].duplicated().any():
        raise ProtocolViolation(f"{role} contains duplicate observed dates")
    actual_dates = pd.DatetimeIndex(observed["date"])
    if not actual_dates.equals(required_dates):
        missing = required_dates.difference(actual_dates).strftime("%Y-%m-%d").tolist()
        raise ProtocolViolation(f"{role} missing observed dates: {missing}")
    missing_features = [column for column in feature_cols if column not in observed.columns]
    if missing_features:
        raise ProtocolViolation(f"{role} missing KNN features: {missing_features!r}")
    values = observed.loc[:, list(feature_cols)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ProtocolViolation(f"{role} observed KNN features contain non-finite values")
    return values.reshape(-1)


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
    feature_cols: Sequence[str] = ("sales",),
    model_feature_cols: Sequence[str] | None = None,
    knn_schema_digest: str | None = None,
    k: int,
) -> SelectionResult:
    """Validate canonical 180-day sources, then rank on their final 30 days."""

    normalized_features = tuple(str(column) for column in feature_cols)
    if not normalized_features or normalized_features[0] != "sales":
        raise ProtocolViolation("feature_cols must start with sales in frozen KNN order")
    # Candidate eligibility is defined exclusively by the frozen KNN safe
    # view. Predictor completeness belongs to the independent model preflight.
    normalized_model_features = normalized_features
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
    if "sales" not in target_df.columns:
        raise ProtocolViolation("target dataframe requires sales column")

    window = protocol.observation_window(observed_start)
    observed_start_iso = window.knn_observed_start.isoformat()
    observed_end_iso = window.knn_observed_end.isoformat()
    required_dates = pd.date_range(observed_start_iso, observed_end_iso, freq="D")

    target = _prepare_dates(target_df, role="target")
    target_vector = _exact_observed_vector(
        target,
        required_dates,
        role="target",
        feature_cols=normalized_features,
    )

    pretrain_end = pd.Timestamp(observed_end_iso).normalize()
    pretrain_start = pretrain_end - pd.Timedelta(days=179)

    pool = prepared_pool or prepare_daily_sequence_pool(
        source_df,
        group_cols=group_cols,
        observed_start=observed_start_iso,
        observed_end=observed_end_iso,
        pretrain_start=pretrain_start,
        pretrain_end=pretrain_end,
        knn_feature_cols=normalized_features,
        required_feature_cols=normalized_model_features,
    )
    pool.validate_for(group_cols=group_cols, required_dates=required_dates)
    if pool.pretrain_dates != tuple(pd.date_range(pretrain_start, pretrain_end).strftime("%Y-%m-%d")):
        raise ProtocolViolation("WINDOW_FAILURE: prepared pool source pretrain dates differ from exact 180-day window")
    if pool.knn_feature_cols != normalized_features:
        raise ProtocolViolation("KNN_SCHEMA_FAILURE: prepared pool KNN feature order differs from frozen schema")
    if pool.required_feature_cols != normalized_model_features:
        raise ProtocolViolation("KNN_SCHEMA_FAILURE: prepared pool required fields differ from frozen KNN schema")

    valid_keys = []
    raw_vectors = []
    excluded = []
    for candidate_key in normalized_candidates:
        reasons = pool.ineligible_reasons_for(candidate_key)
        if reasons:
            excluded.append(
                {
                    "source_key": candidate_key,
                    "reason": reasons[0],
                    "reasons": reasons,
                    "missing_dates": tuple(pool.missing_dates_for(candidate_key)),
                }
            )
            continue
        pool_index = pool.key_to_index.get(candidate_key)
        if pool_index is None:
            excluded.append(
                {
                    "source_key": candidate_key,
                    "reason": "source_key_absent",
                    "reasons": ("source_key_absent",),
                    "missing_dates": tuple(required_dates.strftime("%Y-%m-%d")),
                }
            )
            continue
        if pool.knn_matrix is None:
            raise ProtocolViolation("prepared pool has no canonical KNN matrix")
        values = pool.knn_matrix[pool_index]
        if not np.isfinite(values).all():
            excluded.append(
                {
                    "source_key": candidate_key,
                    "reason": "nonfinite_knn_vector",
                    "reasons": ("nonfinite_knn_vector",),
                    "missing_dates": (),
                }
            )
            continue
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
    if scaler_max == scaler_min:
        scaled_sources = np.zeros_like(source_matrix)
        scaled_target = np.zeros_like(target_vector)
    else:
        scale = np.float64(scaler_max - scaler_min)
        scaled_sources = (source_matrix - np.float64(scaler_min)) / scale
        scaled_target = (target_vector - np.float64(scaler_min)) / scale
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
            vector_shape=(30, len(normalized_features)),
            vector_digest=pool.vector_digests[ranked_entry.source_key],
            source_repair_digest=str(
                pool.repair_audits[ranked_entry.source_key]["repair_mask_sha256"]
            ).removeprefix("sha256:"),
            source_training_digest=pool.training_digests[ranked_entry.source_key],
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
        normalized_features,
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
    resolved_knn_schema_digest = str(knn_schema_digest or _sha256_payload({
        "feature_cols": list(normalized_features),
        "dtypes": ["float64"] * len(normalized_features),
    })).removeprefix("sha256:")
    repair_digest = _sha256_payload(
        {
            "selected": [
                [list(entry.source_key), entry.source_repair_digest] for entry in entries
            ]
        }
    )
    training_digest = _sha256_payload(
        {
            "selected": [
                [list(entry.source_key), entry.source_training_digest] for entry in entries
            ]
        }
    )
    selection_identity = _sha256_payload(
        {
            "knn_schema_digest": resolved_knn_schema_digest,
            "vector_shape": [30, len(normalized_features)],
            "knn_date_range": [observed_start_iso, observed_end_iso],
            "source_repair_digest": repair_digest,
            "vectors": [
                [list(entry.source_key), entry.vector_digest] for entry in entries
            ],
        }
    )
    return SelectionResult(
        protocol_version=protocol.protocol_version,
        dataset_id=protocol.dataset_id,
        scenario=normalized_scenario,
        target_key=normalized_target,
        group_cols=tuple(group_cols),
        feature_cols=normalized_features,
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
        source_window_start=pretrain_start.strftime("%Y-%m-%d"),
        source_window_end=pretrain_end.strftime("%Y-%m-%d"),
        knn_window_start=observed_start_iso,
        knn_window_end=observed_end_iso,
        knn_schema_digest=resolved_knn_schema_digest,
        source_repair_digest=repair_digest,
        source_training_digest=training_digest,
        selection_identity_digest=selection_identity,
    )
