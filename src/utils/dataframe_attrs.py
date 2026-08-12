"""Explicit pandas attrs boundary for protocol and modeling dataframes."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Iterator, Mapping

import pandas as pd


PROTOCOL_CONTEXT_ATTR = "protocol_frame_context"

HEAVY_SOURCE_HISTORY_ATTRS = frozenset(
    {
        "source_history_eligibility",
        "source_history_eligible_keys",
        "source_history_incomplete_keys",
        "source_history_duplicate_keys",
    }
)

HEAVY_PROTOCOL_ATTRS = frozenset(
    {
        *HEAVY_SOURCE_HISTORY_ATTRS,
        "protocol_candidate_keys",
        "protocol_knn_observed_frame",
        "forecast_consumer_frame",
        "prepared_daily_sequence_pool",
        "protocol_sample_manifest",
        "d2_source_calendarization_report",
        "protocol_actual_cnn_audit",
        "protocol_raw_partition",
        "protocol_fitted_scaler",
        "audit_metadata",
    }
)

# This is intentionally an allowlist.  Adding a new runtime attr requires an
# explicit decision that it is small and semantically required by a consumer.
WORKING_FRAME_ATTR_ALLOWLIST = frozenset(
    {
        PROTOCOL_CONTEXT_ATTR,
        "dataset_name",
        "strict_dataset_name",
        "role",
        "split_role",
        "split_mode",
        "split_config",
        "strict_paper_mode",
        "strict_paper_split",
        "paper_split_protocol",
        "train_days",
        "val_days",
        "test_days",
        "observed_days",
        "target_window_expected_days",
        "target_window_range_days",
        "target_window_unique_days",
        "target_train_window",
        "temporal_partition",
        "method",
        "model_window_size",
        "model_horizon",
        "information_sharing_scenario",
        "selection_authority",
        "protocol_version",
        "protocol_track",
        "protocol_dataset_id",
        "protocol_scenario",
        "protocol_target_key",
        "protocol_group_cols",
        "protocol_observed_start",
        "protocol_observed_days",
        "origin",
        "boundary",
        "knn_observed_start",
        "knn_observed_end",
        "source_observation_cutoff",
        "target_observed_start",
        "target_observed_end",
        "target_test_excluded",
        "source_future_excluded",
        "representation",
        "knn_representation",
        "scaling",
        "scaler_fit_scope",
        "source_alignment_mode",
        "source_frame_min_date",
        "source_frame_max_date",
        "target_frame_min_date",
        "target_frame_max_date",
        "source_frame_digest",
        "target_frame_digest",
        "knn_frame_role",
        "knn_observed_days",
        "knn_boundary",
        "knn_feature_columns",
        "historical_feature_columns",
        "forecast_excluded_columns",
        "feature_scope",
        "max_allowed_date_relation",
        "knn_frame_min_date",
        "knn_frame_max_date",
        "knn_frame_digest",
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
        "source_history_eligible_key_count",
        "source_history_incomplete_key_count",
        "source_history_duplicate_key_count",
        "source_history_outside_window_row_count",
        "source_history_validation_path",
        "source_history_prevalidated_exact",
        "source_history_canonical_order",
        "d2_source_calendarization_rule_version",
        "d2_source_authority_digest",
        "d2_consumer_frame_fingerprint",
        "d2_synthetic_source_row_count",
        "d2_source_verification_frame_fingerprint",
        "d2_source_verified_candidate_row_count",
        "d2_source_slicing_complete",
        "d2_source_interval_start",
        "d2_source_interval_end",
        "solidified_parquet_path",
        "protocol_actual_source_key",
        "protocol_scaler_feature_cols",
    }
)


@dataclass(frozen=True)
class ProtocolFrameContext:
    """Shared references kept outside ordinary pandas attrs propagation.

    The context is immutable as a binding object.  Objects referenced by it are
    builder-owned and must be treated as read-only by every accessor.
    """

    source_index: Any = None
    observed_frames: Mapping[str, pd.DataFrame] | None = None
    observed_carrier_ids: Mapping[str, int] | None = None
    candidate_keys: tuple[tuple[str, ...], ...] = ()
    forecast_frame: pd.DataFrame | None = None
    prepared_pool: Any = None
    sample_manifest: Any = None
    protocol_report: Any = None
    actual_source_key: tuple[str, ...] | None = None
    actual_cnn_audit: Mapping[tuple[str, ...], Mapping[str, Any]] | None = None
    raw_partition: pd.DataFrame | None = None
    fitted_scaler: Any = None
    scaler_feature_cols: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "observed_frames", _observed_mapping(self.observed_frames))
        object.__setattr__(
            self,
            "observed_carrier_ids",
            MappingProxyType(dict(self.observed_carrier_ids or {})),
        )
        object.__setattr__(
            self,
            "candidate_keys",
            tuple(tuple(component for component in key) for key in self.candidate_keys),
        )
        if self.actual_cnn_audit is not None and not isinstance(
            self.actual_cnn_audit, MappingProxyType
        ):
            object.__setattr__(
                self,
                "actual_cnn_audit",
                MappingProxyType(dict(self.actual_cnn_audit)),
            )

    def __deepcopy__(self, memo: dict[int, Any]) -> "ProtocolFrameContext":
        memo[id(self)] = self
        return self


def _observed_mapping(value: Mapping[str, pd.DataFrame] | None) -> Mapping[str, pd.DataFrame]:
    return MappingProxyType(dict(value or {}))


def context_with(
    context: ProtocolFrameContext | None,
    **updates: Any,
) -> ProtocolFrameContext:
    current = context or ProtocolFrameContext()
    if "observed_frames" in updates:
        updates["observed_frames"] = _observed_mapping(updates["observed_frames"])
    return replace(current, **updates)


def get_protocol_frame_context(frame: pd.DataFrame) -> ProtocolFrameContext | None:
    value = frame.attrs.get(PROTOCOL_CONTEXT_ATTR)
    return value if isinstance(value, ProtocolFrameContext) else None


def set_protocol_frame_context(frame: pd.DataFrame, context: ProtocolFrameContext) -> None:
    frame.attrs[PROTOCOL_CONTEXT_ATTR] = context


def _context_from_attrs(attrs: Mapping[str, Any]) -> ProtocolFrameContext | None:
    raw_context = attrs.get(PROTOCOL_CONTEXT_ATTR)
    context = raw_context if isinstance(raw_context, ProtocolFrameContext) else None
    updates: dict[str, Any] = {}
    if "protocol_candidate_keys" in attrs and not (context is not None and context.candidate_keys):
        updates["candidate_keys"] = tuple(tuple(key) for key in attrs["protocol_candidate_keys"])
    if "prepared_daily_sequence_pool" in attrs and not (
        context is not None and context.prepared_pool is not None
    ):
        updates["prepared_pool"] = attrs["prepared_daily_sequence_pool"]
    if "forecast_consumer_frame" in attrs:
        updates["forecast_frame"] = attrs["forecast_consumer_frame"]
    if "protocol_sample_manifest" in attrs:
        updates["sample_manifest"] = attrs["protocol_sample_manifest"]
    if "d2_source_calendarization_report" in attrs:
        updates["protocol_report"] = attrs["d2_source_calendarization_report"]
    if "audit_metadata" in attrs and "protocol_report" not in updates:
        updates["protocol_report"] = attrs["audit_metadata"]
    if "protocol_actual_source_key" in attrs:
        updates["actual_source_key"] = tuple(attrs["protocol_actual_source_key"])
    if "protocol_actual_cnn_audit" in attrs:
        updates["actual_cnn_audit"] = attrs["protocol_actual_cnn_audit"]
    if "protocol_raw_partition" in attrs:
        updates["raw_partition"] = attrs["protocol_raw_partition"]
    if "protocol_fitted_scaler" in attrs:
        updates["fitted_scaler"] = attrs["protocol_fitted_scaler"]
    if "protocol_scaler_feature_cols" in attrs:
        updates["scaler_feature_cols"] = tuple(attrs["protocol_scaler_feature_cols"])
    configured = attrs.get("protocol_knn_observed_frame")
    if isinstance(configured, pd.DataFrame):
        role = str(attrs.get("split_role", configured.attrs.get("knn_frame_role", ""))).lower()
        observed = dict(context.observed_frames or {}) if context is not None else {}
        if role in {"source", "target"}:
            observed[role] = configured
            updates["observed_frames"] = observed
    if not updates:
        return context
    return context_with(context, **updates)


def lightweight_frame_attrs(attrs: Mapping[str, Any]) -> dict[str, Any]:
    """Return only explicitly approved working-frame attrs and one context ref."""

    result = {
        key: value
        for key, value in attrs.items()
        if key in WORKING_FRAME_ATTR_ALLOWLIST and key != PROTOCOL_CONTEXT_ATTR
    }
    context = _context_from_attrs(attrs)
    if context is not None:
        result[PROTOCOL_CONTEXT_ATTR] = context
    return result


def promote_protocol_frame_context(frame: pd.DataFrame) -> ProtocolFrameContext | None:
    """Move known heavyweight attrs into the shared context without copying them."""

    attrs = lightweight_frame_attrs(frame.attrs)
    frame.attrs = attrs
    return get_protocol_frame_context(frame)


@contextmanager
def temporarily_detached_attrs(frame: pd.DataFrame) -> Iterator[None]:
    """Run a pandas operation without allowing it to deepcopy frame attrs."""

    original = frame.attrs
    frame.attrs = {}
    try:
        yield
    finally:
        frame.attrs = original


def copy_frame_with_lightweight_attrs(
    frame: pd.DataFrame,
    *,
    deep: bool = True,
) -> pd.DataFrame:
    attrs = lightweight_frame_attrs(frame.attrs)
    with temporarily_detached_attrs(frame):
        copied = frame.copy(deep=deep)
    copied.attrs = attrs
    return copied


def select_rows_with_lightweight_attrs(
    frame: pd.DataFrame,
    row_selector: Any,
    *,
    deep: bool = True,
) -> pd.DataFrame:
    attrs = lightweight_frame_attrs(frame.attrs)
    with temporarily_detached_attrs(frame):
        selected = frame.loc[row_selector].copy(deep=deep)
    selected.attrs = attrs
    return selected


__all__ = [
    "HEAVY_PROTOCOL_ATTRS",
    "HEAVY_SOURCE_HISTORY_ATTRS",
    "PROTOCOL_CONTEXT_ATTR",
    "ProtocolFrameContext",
    "WORKING_FRAME_ATTR_ALLOWLIST",
    "context_with",
    "copy_frame_with_lightweight_attrs",
    "get_protocol_frame_context",
    "lightweight_frame_attrs",
    "promote_protocol_frame_context",
    "select_rows_with_lightweight_attrs",
    "set_protocol_frame_context",
    "temporarily_detached_attrs",
]
