"""Canonical candidate-pool auditing and leak-free daily-sequence KNN."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime
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

    remaining = sorted(zip(keys, values.tolist()), key=lambda item: (item[1], item[0]))
    ranked = []
    tie_group = 1
    while remaining:
        anchor = float(remaining[0][1])
        current = []
        later = []
        for key, raw_distance in remaining:
            if float(raw_distance) - anchor <= tie_tolerance:
                current.append((key, float(raw_distance)))
            else:
                later.append((key, float(raw_distance)))
        current.sort(key=lambda item: item[0])
        ranked.extend(
            RankedDistance(key, raw_distance, tie_group)
            for key, raw_distance in current
        )
        remaining = later
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


def _normalized_key_series(frame: pd.DataFrame, group_cols: Sequence[str]) -> pd.Series:
    return frame.loc[:, list(group_cols)].apply(
        lambda row: normalize_source_key(tuple(row.tolist())), axis=1
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
) -> np.ndarray:
    observed = frame[frame["date"].isin(required_dates)].sort_values("date")
    if observed["date"].duplicated().any():
        raise ProtocolViolation(f"{role} contains duplicate observed dates")
    actual_dates = pd.DatetimeIndex(observed["date"])
    if not actual_dates.equals(required_dates):
        missing = required_dates.difference(actual_dates).strftime("%Y-%m-%d").tolist()
        raise ProtocolViolation(f"{role} missing observed dates: {missing}")
    values = observed["sales"].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ProtocolViolation(f"{role} observed sales contain non-finite values")
    return values


def select_daily_sequence_sources(
    *,
    target_df: pd.DataFrame,
    source_df: pd.DataFrame,
    protocol: ExperimentProtocol,
    scenario: object,
    target_key: Sequence[object],
    candidate_keys: Iterable[Sequence[object]],
    group_cols: Sequence[str],
    observed_start: object,
    feature_cols: Sequence[str] = ("sales",),
    k: int,
) -> SelectionResult:
    """Select exactly K sources using only aligned legal 30-day sales sequences."""

    if tuple(feature_cols) != ("sales",):
        raise ProtocolViolation("feature_cols must be exactly ('sales',) for strict KNN")
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
    missing_group_cols = [column for column in group_cols if column not in source_df.columns]
    if missing_group_cols:
        raise ProtocolViolation(f"source dataframe missing group columns: {missing_group_cols}")
    if "sales" not in target_df.columns or "sales" not in source_df.columns:
        raise ProtocolViolation("target and source dataframes require sales column")

    window = protocol.observation_window(observed_start)
    observed_start_iso = window.knn_observed_start.isoformat()
    observed_end_iso = window.knn_observed_end.isoformat()
    required_dates = pd.date_range(observed_start_iso, observed_end_iso, freq="D")

    target = _prepare_dates(target_df, role="target")
    source = _prepare_dates(source_df, role="source")
    target_vector = _exact_observed_vector(
        target,
        required_dates,
        role="target",
    )
    source["__protocol_source_key__"] = _normalized_key_series(source, group_cols)

    valid_keys = []
    raw_vectors = []
    excluded = []
    for candidate_key in normalized_candidates:
        candidate_frame = source[source["__protocol_source_key__"] == candidate_key]
        observed = candidate_frame[candidate_frame["date"].isin(required_dates)].sort_values("date")
        if observed["date"].duplicated().any():
            raise ProtocolViolation(
                f"source {candidate_key!r} contains duplicate observed dates"
            )
        actual_dates = pd.DatetimeIndex(observed["date"])
        if not actual_dates.equals(required_dates):
            missing = required_dates.difference(actual_dates).strftime("%Y-%m-%d").tolist()
            excluded.append(
                {
                    "source_key": candidate_key,
                    "reason": "missing_observed_dates",
                    "missing_dates": tuple(missing),
                }
            )
            continue
        values = observed["sales"].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ProtocolViolation(
                f"source {candidate_key!r} observed sales contain non-finite values"
            )
        valid_keys.append(candidate_key)
        raw_vectors.append(values)

    if len(valid_keys) < k:
        raise ProtocolViolation(
            f"valid candidates={len(valid_keys)} is below required K={k}; excluded={excluded!r}"
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
