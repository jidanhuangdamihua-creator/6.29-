"""Bind dataset runner frames to the immutable D1-D6 protocol."""

from __future__ import annotations

from copy import deepcopy
from typing import Sequence, Tuple

import pandas as pd

from src.constants import SOURCE_HISTORY_DAYS

from .candidate_pool import (
    CanonicalSourceIndex,
    PreparedDailySequencePool,
    build_canonical_source_index,
    classify_prepared_candidate_dates,
    prepare_daily_sequence_pool,
    validate_prepared_candidate_date_eligibility,
)
from .d2_source_calendarization import (
    D2_SOURCE_CALENDARIZATION_RULE_VERSION,
    slice_d2_source_frame,
    verify_d2_source_frame,
)
from .knn_frames import (
    build_observed_knn_frame,
    build_prepared_pool_observed_knn_frame,
    canonical_knn_frame_digest,
    seal_canonical_knn_digest_evidence,
)
from .selection_metadata import build_selection_metadata_contract
from src.utils.dataframe_attrs import (
    ProtocolFrameContext,
    copy_frame_with_lightweight_attrs,
    get_protocol_frame_context,
    lightweight_frame_attrs,
    select_rows_with_lightweight_attrs,
    set_protocol_frame_context,
)
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
    context = get_protocol_frame_context(frame)
    source_index = context.source_index if context is not None else None
    if (
        isinstance(source_index, CanonicalSourceIndex)
        and source_index.group_cols == tuple(str(column) for column in group_cols)
    ):
        indexed_mask = source_index.mask_for_normalized_key(frame, normalized_key)
        if indexed_mask is not None:
            return indexed_mask
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


def _available_keys(
    frame: pd.DataFrame,
    group_cols: Sequence[str],
    source_index: CanonicalSourceIndex | None = None,
) -> Tuple[Tuple[str, ...], ...]:
    missing = [column for column in group_cols if column not in frame.columns]
    if missing:
        raise ProtocolViolation(f"source frame missing protocol key columns: {missing}")
    if source_index is not None and source_index.group_cols == tuple(group_cols):
        return source_index.source_keys
    safe_frame = copy_frame_with_lightweight_attrs(frame, deep=False)
    keys = {
        normalize_source_key(tuple(row))
        for row in safe_frame.loc[:, list(group_cols)].drop_duplicates().itertuples(index=False, name=None)
    }
    return tuple(sorted(keys))


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
    source_index: CanonicalSourceIndex | None = None,
) -> Tuple[Tuple[str, ...], ...]:
    if not require_same_group:
        return _extended_candidates_from_identities(
            tuple(
                SourceIdentity(key)
                for key in _available_keys(source_df, group_cols, source_index)
            ),
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
    if (
        source_index is not None
        and source_index.group_cols == tuple(group_cols)
        and grouping_col in source_index.metadata_by_col
    ):
        values = source_index.metadata_by_col[grouping_col]
        return _extended_candidates_from_identities(
            tuple(SourceIdentity(key, values[key]) for key in source_index.source_keys),
            scenario=scenario,
            target_key=target_key,
            target_group=target_group,
            require_same_group=True,
            candidate_exclusion_positions=candidate_exclusion_positions,
        )
    identities = []
    safe_source_df = copy_frame_with_lightweight_attrs(source_df, deep=False)
    grouped = safe_source_df.groupby(list(group_cols), sort=False, dropna=False)
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


def _verify_d2_source_for_candidates(
    source_df: pd.DataFrame,
    *,
    candidates: Sequence[Sequence[object]],
    source_index: CanonicalSourceIndex,
    group_cols: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Verify one D2 source carrier and preserve the legacy evidence contract."""

    source = copy_frame_with_lightweight_attrs(source_df)
    source_dates = pd.to_datetime(source["date"], errors="coerce").dt.normalize()
    if source_dates.isna().any():
        raise ProtocolViolation("source frame contains invalid dates")
    source.attrs.setdefault("split_role", "source")
    sealed_source = source.copy()
    verified_source, report = verify_d2_source_frame(
        slice_d2_source_frame(sealed_source),
        candidate_keys=candidates,
    )
    candidate_mask = pd.Series(False, index=source.index)
    for candidate_key in candidates:
        indexed = source_index.mask_for_normalized_key(source, candidate_key)
        if indexed is None:
            indexed = source_key_mask(source, group_cols, candidate_key)
        candidate_mask |= indexed
    source = select_rows_with_lightweight_attrs(source, candidate_mask)
    source.attrs = {**sealed_source.attrs, **verified_source.attrs}
    source.attrs.update(
        {
            "d2_source_verification_frame_fingerprint": report.consumer_frame_fingerprint,
            "d2_source_verified_candidate_row_count": int(len(verified_source)),
        }
    )
    metadata = {
        "d2_source_calendarization_rule_version": report.rule_version,
        "d2_source_authority_digest": report.source_authority_digest,
        "d2_consumer_frame_fingerprint": report.consumer_frame_fingerprint,
        "d2_synthetic_source_row_count": report.synthetic_row_count,
        "d2_source_calendarization_report": report.to_dict(),
    }
    return source, metadata


def _d2_verification_candidate_keys(
    scenario: str,
    available: Sequence[Sequence[object]],
) -> tuple[tuple[str, str], ...]:
    """Resolve the one cell-level D2 verification scope without building a context scope."""

    brands = range(1, 4) if scenario == "with" else range(1, 2)
    expected = tuple(
        (str(brand), str(item))
        for brand in brands
        for item in range(1, 10)
    )
    available_set = {normalize_source_key(key) for key in available}
    missing = tuple(key for key in expected if key not in available_set)
    if missing:
        raise ProtocolViolation(
            f"D2/{scenario} missing required candidate keys: {missing!r}"
        )
    return expected


def prepare_protocol_source_pool(
    source_df: pd.DataFrame,
    *,
    dataset_id: object,
    scenario: object,
    group_cols: Sequence[str],
    observed_start: object,
    grouping_col: str | None = None,
) -> tuple[pd.DataFrame, PreparedDailySequencePool]:
    """Prepare the immutable source index/pool once for one D1-D3 cell."""

    protocol = get_experiment_protocol(dataset_id)
    if protocol.dataset_id not in {"D1", "D2", "D3"}:
        raise ProtocolViolation("cell source preparation is restricted to D1-D3")
    normalized_scenario = normalize_scenario(scenario)
    normalized_group_cols = tuple(str(column) for column in group_cols)
    if len(normalized_group_cols) != len(protocol.source_pool_rule.key_fields):
        raise ProtocolViolation(
            f"{protocol.dataset_id} requires {len(protocol.source_pool_rule.key_fields)} "
            f"group columns, got {normalized_group_cols!r}"
        )
    if "date" not in source_df.columns:
        raise ProtocolViolation("protocol source frame requires a date column")
    window = protocol.observation_window(observed_start)
    metadata_cols = (
        (grouping_col,)
        if grouping_col is not None and grouping_col in source_df.columns
        else ()
    )
    source_index = build_canonical_source_index(
        source_df,
        group_cols=normalized_group_cols,
        metadata_cols=metadata_cols,
    )
    prepared_pool = prepare_daily_sequence_pool(
        source_df,
        group_cols=normalized_group_cols,
        observed_start=window.knn_observed_start,
        observed_end=window.knn_observed_end,
        metadata_cols=metadata_cols,
        feature_cols=protocol.knn_feature_columns,
        source_index=source_index,
    )
    prepared_source = copy_frame_with_lightweight_attrs(source_df)
    if protocol.dataset_id == "D2":
        candidates = _d2_verification_candidate_keys(
            normalized_scenario,
            prepared_pool.source_keys,
        )
        prepared_source, _ = _verify_d2_source_for_candidates(
            source_df,
            candidates=candidates,
            source_index=source_index,
            group_cols=normalized_group_cols,
        )
    return prepared_source, prepared_pool


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
    retain_source_frame: bool = False,
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
    window = protocol.observation_window(observed_start)
    knn_feature_columns = tuple(protocol.knn_feature_columns)
    digest_feature_columns = (
        None if knn_feature_columns == ("sales",) else knn_feature_columns
    )
    digest_ignored_columns = ("promo",) if digest_feature_columns is None else ()
    index_metadata_cols = (
        (grouping_col,)
        if grouping_col is not None and grouping_col in source_df.columns
        else ()
    )
    source_index = (
        prepared_pool.source_index
        if prepared_pool is not None
        else build_canonical_source_index(
            source_df,
            group_cols=normalized_group_cols,
            metadata_cols=index_metadata_cols,
        )
    )
    runtime_pool = prepared_pool or prepare_daily_sequence_pool(
        source_df,
        group_cols=normalized_group_cols,
        observed_start=window.knn_observed_start,
        observed_end=window.knn_observed_end,
        metadata_cols=index_metadata_cols,
        feature_cols=knn_feature_columns,
        source_index=source_index,
    )
    target_knn_frame = build_observed_knn_frame(
        target_df,
        window=window,
        role="target",
        group_cols=normalized_group_cols,
        feature_cols=knn_feature_columns,
    )
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
    target_key_frame = (
        target_knn_frame
        if protocol.dataset_id in {"D1", "D2"}
        else target_df
    )
    runtime_target_key = _unique_key(target_key_frame, normalized_group_cols, role="target")
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
    available = runtime_pool.source_keys
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
                source_index=source_index,
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

    prepared_date_eligibility = None
    if prepared_pool is not None:
        required_dates = pd.date_range(
            window.knn_observed_start,
            window.knn_observed_end,
        )
        prepared_date_eligibility = classify_prepared_candidate_dates(
            runtime_pool,
            candidates,
            group_cols=normalized_group_cols,
            required_dates=required_dates,
            feature_cols=knn_feature_columns,
        )
        validate_prepared_candidate_date_eligibility(
            runtime_pool,
            prepared_date_eligibility,
            candidates,
            group_cols=normalized_group_cols,
            required_dates=required_dates,
            feature_cols=knn_feature_columns,
        )

    cutoff = pd.Timestamp(window.source_observation_cutoff).normalize()
    d2_calendarization_metadata = {}
    if prepared_pool is None:
        source = copy_frame_with_lightweight_attrs(source_df)
        source_dates = pd.to_datetime(source["date"], errors="coerce").dt.normalize()
        if source_dates.isna().any():
            raise ProtocolViolation("source frame contains invalid dates")
        if protocol.dataset_id == "D2":
            source, d2_calendarization_metadata = _verify_d2_source_for_candidates(
                source_df,
                candidates=candidates,
                source_index=source_index,
                group_cols=normalized_group_cols,
            )
        else:
            source = select_rows_with_lightweight_attrs(source, source_dates <= cutoff)
            if source.empty:
                raise ProtocolViolation("source frame is empty at source_observation_cutoff")
    else:
        prepared_pool.validate_for(
            group_cols=normalized_group_cols,
            required_dates=pd.date_range(window.knn_observed_start, window.knn_observed_end),
        )
        if retain_source_frame:
            source = copy_frame_with_lightweight_attrs(source_df)
            source_dates = pd.to_datetime(source["date"], errors="coerce").dt.normalize()
            if source_dates.isna().any():
                raise ProtocolViolation("source frame contains invalid dates")
            source = select_rows_with_lightweight_attrs(source, source_dates <= cutoff)
            if source.empty:
                raise ProtocolViolation("source frame is empty at source_observation_cutoff")
        else:
            source_view = copy_frame_with_lightweight_attrs(source_df, deep=False)
            source = source_view.iloc[0:0].copy()
            source.attrs = source_view.attrs
        if protocol.dataset_id == "D2":
            required_d2_attrs = (
                "d2_source_calendarization_rule_version",
                "d2_source_authority_digest",
                "d2_consumer_frame_fingerprint",
                "d2_source_calendarization_report",
            )
            missing_d2_attrs = [
                name for name in required_d2_attrs if name not in source_df.attrs
            ]
            if missing_d2_attrs:
                raise ProtocolViolation(
                    "D2 prepared pool is missing calendarization metadata: "
                    f"{missing_d2_attrs!r}"
                )
            d2_calendarization_metadata = {
                name: source_df.attrs[name]
                for name in required_d2_attrs
            }
            if (
                d2_calendarization_metadata["d2_source_calendarization_rule_version"]
                != D2_SOURCE_CALENDARIZATION_RULE_VERSION
            ):
                raise ProtocolViolation("D2 source calendarization rule version is unsupported")
    target = target_df.copy()
    target_dates = pd.to_datetime(target["date"], errors="coerce").dt.normalize()
    if target_dates.isna().any():
        raise ProtocolViolation("target frame contains invalid dates")
    if not (target_dates > cutoff).any():
        raise ProtocolViolation("target frame has no test dates after knn_observed_end")

    retain_legacy_source_observed_domain = (
        prepared_pool is not None
        and retain_source_frame
        and protocol.dataset_id in {"D1", "D2", "D3"}
    )
    source_knn_input = (
        source
        if prepared_pool is None or retain_legacy_source_observed_domain
        else runtime_pool.selected_frame(candidates, feature_cols=knn_feature_columns)
    )
    if prepared_date_eligibility is None or retain_legacy_source_observed_domain:
        source_knn_frame = build_observed_knn_frame(
            source_knn_input,
            window=window,
            role="source",
            group_cols=normalized_group_cols,
            feature_cols=knn_feature_columns,
        )
    else:
        source_knn_frame = build_prepared_pool_observed_knn_frame(
            source_knn_input,
            window=window,
            role="source",
            group_cols=normalized_group_cols,
            feature_cols=knn_feature_columns,
            eligibility_proof=prepared_date_eligibility,
            pool_identity=id(runtime_pool),
        )
    metadata = {
        "selection_authority": "shared_protocol",
        "protocol_version": protocol.protocol_version,
        "protocol_track": protocol.track,
        "protocol_dataset_id": protocol.dataset_id,
        "protocol_scenario": normalized_scenario,
        "protocol_target_key": target_key,
        "protocol_group_cols": normalized_group_cols,
        "protocol_observed_start": window.knn_observed_start.isoformat(),
        "protocol_observed_days": window.observed_days,
        "origin": window.origin.isoformat(),
        "observed_days": window.observed_days,
        "boundary": "inclusive",
        "knn_observed_start": window.knn_observed_start.isoformat(),
        "knn_observed_end": window.knn_observed_end.isoformat(),
        "source_observation_cutoff": window.source_observation_cutoff.isoformat(),
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
        "source_frame_min_date": source_knn_frame["date"].min().strftime("%Y-%m-%d"),
        "source_frame_max_date": source_knn_frame["date"].max().strftime("%Y-%m-%d"),
        "target_frame_min_date": target_knn_frame["date"].min().strftime("%Y-%m-%d"),
        "target_frame_max_date": target_knn_frame["date"].max().strftime("%Y-%m-%d"),
    }
    metadata.update(build_selection_metadata_contract(protocol, window=window))
    if protocol.dataset_id in {"D4", "D5", "D6"} and "source_history_days" in source_df.attrs:
        source_history_fields = (
            "source_history_days",
            "source_history_start",
            "source_history_end",
            "source_history_expected_date_count",
            "source_history_completeness_policy",
            "source_history_calendar",
            "source_history_inclusive_end",
            "source_history_calendarization_rule",
            "source_history_synthetic_row_count",
            "source_history_frame_digest",
        )
        missing_history = [
            field for field in source_history_fields if field not in source_df.attrs
        ]
        if missing_history:
            raise ProtocolViolation(
                "D4-D6 source history metadata is incomplete: "
                f"{missing_history!r}"
            )
        if int(source_df.attrs["source_history_days"]) != SOURCE_HISTORY_DAYS:
            raise ProtocolViolation(
                "D4-D6 source history must be exactly "
                f"{SOURCE_HISTORY_DAYS} days"
            )
        metadata.update({field: source_df.attrs[field] for field in source_history_fields})
        if "source_history_eligible_key_count" in source_df.attrs:
            metadata["source_history_eligible_key_count"] = int(
                source_df.attrs["source_history_eligible_key_count"]
            )
        elif prepared_pool is not None:
            metadata["source_history_eligible_key_count"] = len(prepared_pool.source_keys)
    metadata.update(d2_calendarization_metadata)
    protocol_report = deepcopy(metadata.pop("d2_source_calendarization_report", None))
    source_knn_frame.attrs.update(metadata)
    target_knn_frame.attrs.update(metadata)
    shared_digest_identity = (
        ("dataset_id", protocol.dataset_id),
        ("scenario", normalized_scenario),
        ("target_key", target_key),
        ("candidate_scope", tuple(candidates)),
        ("group_cols", normalized_group_cols),
        ("observed_start", window.knn_observed_start.isoformat()),
        ("observed_end", window.knn_observed_end.isoformat()),
        ("feature_cols", knn_feature_columns),
        ("ignored_columns", digest_ignored_columns),
    )
    if prepared_date_eligibility is not None:
        shared_digest_identity = (
            *shared_digest_identity,
            (
                "complete_candidate_scope",
                prepared_date_eligibility.complete_candidate_scope,
            ),
            (
                "excluded_candidate_scope",
                prepared_date_eligibility.excluded_candidate_scope,
            ),
        )
    seal_canonical_knn_digest_evidence(
        source_knn_frame,
        context_identity=((*shared_digest_identity, ("role", "source"))),
    )
    seal_canonical_knn_digest_evidence(
        target_knn_frame,
        context_identity=((*shared_digest_identity, ("role", "target"))),
    )
    source_frame_digest = canonical_knn_frame_digest(
        source_knn_frame,
        group_cols=normalized_group_cols,
        feature_cols=digest_feature_columns,
        ignore_columns=digest_ignored_columns,
    )
    if (
        prepared_pool is not None
        and protocol.dataset_id in {"D4", "D6"}
        and knn_feature_columns == ("sales",)
        and runtime_pool.source_observed_frame_digest is not None
    ):
        source_frame_digest = runtime_pool.source_observed_frame_digest
    metadata.update(
        {
            "source_frame_digest": source_frame_digest,
            "target_frame_digest": canonical_knn_frame_digest(
                target_knn_frame,
                group_cols=normalized_group_cols,
                feature_cols=digest_feature_columns,
                ignore_columns=digest_ignored_columns,
            ),
        }
    )
    source_knn_frame.attrs.update(metadata)
    target_knn_frame.attrs.update(metadata)
    forecast_consumer = target.loc[target_dates > cutoff].copy()
    if protocol.dataset_id == "D2":
        forecast_consumer = forecast_consumer.drop(columns=["promo"], errors="ignore")
        target.loc[target_dates > cutoff, "promo"] = pd.NA
    forecast_consumer.attrs = {
        **metadata,
        "feature_scope": "forecast_consumer",
        "forecast_excluded_columns": ["promo"] if protocol.dataset_id == "D2" else [],
    }
    runtime_context = ProtocolFrameContext(
        source_index=source_index,
        observed_frames={"source": source_knn_frame, "target": target_knn_frame},
        observed_carrier_ids={"source": id(source), "target": id(target)},
        candidate_keys=tuple(candidates),
        forecast_frame=forecast_consumer,
        prepared_pool=runtime_pool,
        protocol_report=protocol_report,
    )
    for role, carrier in (("source", source), ("target", target)):
        if runtime_context.observed_carrier_ids.get(role) != id(carrier):
            raise ProtocolViolation(f"{role} configured carrier identity mismatch")
    source.attrs = {
        **lightweight_frame_attrs(source_df.attrs),
        **metadata,
    }
    target.attrs = {
        **lightweight_frame_attrs(target_df.attrs),
        **metadata,
    }
    set_protocol_frame_context(source, runtime_context)
    set_protocol_frame_context(target, runtime_context)
    return source, target
