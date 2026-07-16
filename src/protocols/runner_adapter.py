"""Bind dataset runner frames to the immutable D1-D6 protocol."""

from __future__ import annotations

from typing import Sequence, Tuple

import pandas as pd

from .candidate_pool import PreparedDailySequencePool
from .feature_schema import get_knn_schema, get_predictor_schema
from .sealing_protocol import SourcePretrainWindow
from .experiment_protocol import (
    EXTENDED_TRACK,
    STRICT_PAPER_TRACK,
    ProtocolViolation,
    SourceIdentity,
    get_experiment_protocol,
    normalize_canonical_target_key,
    normalize_scenario,
    normalize_source_key,
    validate_canonical_target_key,
)


def source_key_mask(
    frame: pd.DataFrame,
    group_cols: Sequence[str],
    source_key: Sequence[object],
) -> pd.Series:
    """Match normalized selector keys back to raw typed source columns exactly."""
    normalized_key = normalize_source_key(source_key)
    if len(group_cols) != len(normalized_key):
        raise ProtocolViolation("source key arity differs from group columns")
    missing = [column for column in group_cols if column not in frame.columns]
    if missing:
        raise ProtocolViolation(f"source key lookup is missing columns: {missing}")
    mask = pd.Series(True, index=frame.index)
    for column, expected in zip(group_cols, normalized_key):
        mask &= frame[column].map(lambda value: normalize_source_key((value,))[0]).eq(expected)
    return mask


def _unique_key(frame: pd.DataFrame, group_cols: Sequence[str], *, role: str) -> Tuple[str, ...]:
    missing = [column for column in group_cols if column not in frame.columns]
    if missing:
        raise ProtocolViolation(f"{role} frame missing protocol key columns: {missing}")
    keys = {
        normalize_source_key(tuple(row))
        for row in frame.loc[:, list(group_cols)].drop_duplicates().itertuples(index=False, name=None)
    }
    if len(keys) != 1:
        raise ProtocolViolation(f"{role} frame must contain exactly one target key, got {sorted(keys)!r}")
    return next(iter(keys))


def _available_keys(frame: pd.DataFrame, group_cols: Sequence[str]) -> Tuple[Tuple[str, ...], ...]:
    missing = [column for column in group_cols if column not in frame.columns]
    if missing:
        raise ProtocolViolation(f"source frame missing protocol key columns: {missing}")
    keys = {
        normalize_source_key(tuple(row))
        for row in frame.loc[:, list(group_cols)].drop_duplicates().itertuples(index=False, name=None)
    }
    return tuple(sorted(keys))


def validate_predictor_safe_view(
    frame: pd.DataFrame,
    *,
    dataset_id: object,
    passthrough_cols: Sequence[str] = ("date",),
) -> None:
    """Fail closed on predictor schema independently of KNN selection."""
    schema = get_predictor_schema(dataset_id)
    expected = set(schema.ordered_names)
    columns = set(str(column) for column in frame.columns)
    missing = sorted(expected - columns)
    if missing:
        raise ProtocolViolation(f"PREDICTOR_SCHEMA_FAILURE: missing predictor fields: {missing}")
    allowed = expected | {str(column) for column in passthrough_cols}
    extra = sorted(columns - allowed)
    if extra:
        raise ProtocolViolation(f"PREDICTOR_SCHEMA_FAILURE: unexpected predictor fields: {extra}")
    unavailable = tuple(frame.attrs.get("availability_failures", ()))
    if unavailable:
        raise ProtocolViolation(f"AVAILABILITY_FAILURE: {unavailable!r}")


def _strict_raw_candidates(
    dataset_id: str,
    scenario: str,
    target_key: Tuple[str, ...],
    available: Tuple[Tuple[str, ...], ...],
) -> Tuple[Tuple[str, ...], ...]:
    if dataset_id in {"D1", "D2"}:
        expected_target = ("1", "10")
        domains = range(1, 4) if scenario == "with" else range(1, 2)
        expected = tuple(
            (str(domain), str(item))
            for domain in domains
            for item in range(1, 10)
        )
    else:
        expected_target = ("10",)
        domains = range(1, 31) if scenario == "with" else range(1, 10)
        expected = tuple((str(domain),) for domain in domains if domain != 10)
    if target_key != expected_target:
        raise ProtocolViolation(
            f"{dataset_id} target must be {expected_target!r}, got {target_key!r}"
        )
    available_set = set(available)
    missing = tuple(key for key in expected if key not in available_set)
    if missing:
        raise ProtocolViolation(
            f"{dataset_id}/{scenario} missing required candidate keys: {missing!r}"
        )
    return expected


def _extended_candidates(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    *,
    scenario: str,
    target_key: Tuple[str, ...],
    group_cols: Sequence[str],
    grouping_col: str | None,
    require_same_group: bool,
    candidate_exclusion_positions: Sequence[int],
) -> Tuple[Tuple[str, ...], ...]:
    if not require_same_group:
        return _extended_candidates_from_identities(
            tuple(SourceIdentity(key) for key in _available_keys(source_df, group_cols)),
            scenario=scenario,
            target_key=target_key,
            target_group=None,
            require_same_group=False,
            candidate_exclusion_positions=candidate_exclusion_positions,
        )
    if grouping_col is None:
        raise ProtocolViolation("extended protocol requires a grouping column")
    if grouping_col not in source_df.columns or grouping_col not in target_df.columns:
        raise ProtocolViolation(f"extended protocol requires grouping column {grouping_col!r}")
    target_groups = {
        str(value).strip() for value in target_df[grouping_col].dropna().unique()
    }
    if len(target_groups) != 1:
        raise ProtocolViolation(
            f"extended target must contain exactly one {grouping_col}, got {sorted(target_groups)!r}"
        )
    target_group = next(iter(target_groups))
    identities = []
    grouped = source_df.groupby(list(group_cols), sort=False, dropna=False)
    for raw_key, rows in grouped:
        key = normalize_source_key(raw_key if isinstance(raw_key, tuple) else (raw_key,))
        group_values = {str(value).strip() for value in rows[grouping_col].dropna().unique()}
        if len(group_values) != 1:
            raise ProtocolViolation(
                f"source key {key!r} maps to multiple {grouping_col} values"
            )
        identities.append(SourceIdentity(key, next(iter(group_values))))
    return _extended_candidates_from_identities(
        identities,
        scenario=scenario,
        target_key=target_key,
        target_group=target_group,
        require_same_group=True,
        candidate_exclusion_positions=candidate_exclusion_positions,
    )


def _extended_candidates_from_identities(
    identities: Sequence[SourceIdentity],
    *,
    scenario: str,
    target_key: Tuple[str, ...],
    target_group: str | None,
    require_same_group: bool,
    candidate_exclusion_positions: Sequence[int],
) -> Tuple[Tuple[str, ...], ...]:
    target_identity = SourceIdentity(target_key, target_group)
    all_identities = tuple(identities) + (target_identity,)
    candidates = []
    target_store = target_key[0]
    for identity in all_identities:
        if identity.key == target_key:
            continue
        if any(
            identity.key[position] == target_key[position]
            for position in candidate_exclusion_positions
        ):
            continue
        if require_same_group and identity.group_value != target_group:
            continue
        if scenario == "without" and identity.key[0] != target_store:
            continue
        candidates.append(identity.key)
    candidates = sorted(set(candidates))
    if not candidates:
        raise ProtocolViolation("extended candidate pool is empty")
    if scenario == "with" and not any(key[0] != target_store for key in candidates):
        raise ProtocolViolation("with-sharing extended pool contains no cross-store candidate")
    return tuple(candidates)


def configure_protocol_frames(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    *,
    dataset_id: object,
    scenario: object,
    group_cols: Sequence[str],
    observed_start: object,
    grouping_col: str | None = None,
    prepared_pool: PreparedDailySequencePool | None = None,
    enforce_formal_target: bool | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Attach strict metadata and clip source history before any fitted transform."""

    protocol = get_experiment_protocol(dataset_id)
    normalized_scenario = normalize_scenario(scenario)
    normalized_group_cols = tuple(str(column) for column in group_cols)
    expected_arity = len(protocol.source_pool_rule.key_fields)
    if len(normalized_group_cols) != expected_arity:
        raise ProtocolViolation(
            f"{protocol.dataset_id} requires {expected_arity} group columns, got {normalized_group_cols!r}"
        )
    if "date" not in source_df.columns or "date" not in target_df.columns:
        raise ProtocolViolation("protocol frames require date columns")
    require_static_target = (
        protocol.track == STRICT_PAPER_TRACK
        if enforce_formal_target is None
        else bool(enforce_formal_target)
    )
    if require_static_target and normalized_group_cols != protocol.source_pool_rule.key_fields:
        raise ProtocolViolation(
            f"{protocol.dataset_id} formal runtime requires canonical group columns "
            f"{protocol.source_pool_rule.key_fields!r}, got {normalized_group_cols!r}"
        )
    runtime_target_key = _unique_key(target_df, normalized_group_cols, role="target")
    if require_static_target:
        target_key = validate_canonical_target_key(
            protocol.dataset_id,
            runtime_target_key,
        )
    else:
        target_key = normalize_canonical_target_key(
            runtime_target_key,
            expected_arity=expected_arity,
        )
    available = (
        prepared_pool.source_keys
        if prepared_pool is not None
        else _available_keys(source_df, normalized_group_cols)
    )
    if protocol.track == EXTENDED_TRACK:
        rule = protocol.source_pool_rule
        expected_group_col = grouping_col or rule.grouping_field
        exclusion_positions = rule.candidate_exclusion_positions()
        if prepared_pool is None:
            candidates = _extended_candidates(
                source_df,
                target_df,
                scenario=normalized_scenario,
                target_key=target_key,
                group_cols=normalized_group_cols,
                grouping_col=expected_group_col,
                require_same_group=rule.require_same_group,
                candidate_exclusion_positions=exclusion_positions,
            )
        else:
            if rule.require_same_group:
                if expected_group_col is None or expected_group_col not in target_df.columns:
                    raise ProtocolViolation(
                        f"extended target requires grouping column {expected_group_col!r}"
                    )
                target_groups = {
                    str(value).strip()
                    for value in target_df[expected_group_col].dropna().unique()
                }
                if len(target_groups) != 1:
                    raise ProtocolViolation(
                        f"extended target must contain exactly one {expected_group_col}, "
                        f"got {sorted(target_groups)!r}"
                    )
                target_group = next(iter(target_groups))
                identities = tuple(
                    SourceIdentity(key, target_group)
                    for key in prepared_pool.keys_for_metadata_value(
                        expected_group_col, target_group
                    )
                )
            else:
                target_group = None
                identities = tuple(SourceIdentity(key) for key in prepared_pool.source_keys)
            candidates = _extended_candidates_from_identities(
                identities,
                scenario=normalized_scenario,
                target_key=target_key,
                target_group=target_group,
                require_same_group=rule.require_same_group,
                candidate_exclusion_positions=exclusion_positions,
            )
    else:
        candidates = _strict_raw_candidates(
            protocol.dataset_id,
            normalized_scenario,
            target_key,
            available,
        )

    window = protocol.observation_window(observed_start)
    cutoff = pd.Timestamp(window.source_observation_cutoff).normalize()
    source_window = SourcePretrainWindow.ending_at(protocol.dataset_id, cutoff)
    source_start = pd.Timestamp(source_window.pretrain_start).normalize()
    if prepared_pool is None:
        source = source_df.copy()
        source_dates = pd.to_datetime(source["date"], errors="coerce").dt.normalize()
        if source_dates.isna().any():
            raise ProtocolViolation("source frame contains invalid dates")
        source = source.loc[source_dates.between(source_start, cutoff, inclusive="both")].copy()
        if source.empty:
            raise ProtocolViolation("source frame is empty at source_observation_cutoff")
    else:
        prepared_pool.validate_for(
            group_cols=normalized_group_cols,
            required_dates=pd.date_range(window.knn_observed_start, window.knn_observed_end),
        )
        source = source_df.iloc[0:0].copy()
    target = target_df.copy()
    target_dates = pd.to_datetime(target["date"], errors="coerce").dt.normalize()
    if target_dates.isna().any():
        raise ProtocolViolation("target frame contains invalid dates")
    if not (target_dates > cutoff).any():
        raise ProtocolViolation("target frame has no test dates after knn_observed_end")

    metadata = {
        "selection_authority": "shared_protocol",
        "protocol_version": protocol.protocol_version,
        "protocol_track": protocol.track,
        "protocol_dataset_id": protocol.dataset_id,
        "protocol_scenario": normalized_scenario,
        "protocol_target_key": target_key,
        "protocol_candidate_keys": candidates,
        "protocol_group_cols": normalized_group_cols,
        "protocol_observed_start": window.knn_observed_start.isoformat(),
        "protocol_observed_days": 30,
        "knn_observed_start": window.knn_observed_start.isoformat(),
        "knn_observed_end": window.knn_observed_end.isoformat(),
        "source_observation_cutoff": window.source_observation_cutoff.isoformat(),
        "source_pretrain_start": source_window.pretrain_start.isoformat(),
        "source_pretrain_end": source_window.pretrain_end.isoformat(),
        "source_pretrain_days": source_window.pretrain_days,
        "target_observed_start": window.knn_observed_start.isoformat(),
        "target_observed_end": window.knn_observed_end.isoformat(),
        "target_test_excluded": True,
        "source_future_excluded": True,
        "representation": protocol.knn_representation,
        "knn_representation": protocol.knn_representation,
        "scaling": "global_minmax_legal_observed_values",
        "scaler_fit_scope": "target_and_candidate_legal_observed_values",
        "source_alignment_mode": "exact_knn_observed_dates",
        "information_sharing_scenario": normalized_scenario,
        "knn_schema_digest": get_knn_schema(protocol.dataset_id).digest,
        "predictor_schema_digest": get_predictor_schema(protocol.dataset_id).digest,
    }
    source.attrs = {**source_df.attrs, **metadata}
    target.attrs = {**target_df.attrs, **metadata}
    if prepared_pool is not None:
        source.attrs["prepared_daily_sequence_pool"] = prepared_pool
    return source, target
