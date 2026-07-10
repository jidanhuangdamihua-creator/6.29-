"""Helpers for source-level failure tolerance in multi-source TL methods."""

from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.utils.finite_diagnostics import NonFiniteArrayError


SOURCE_LEVEL_EXCEPTIONS = (NonFiniteArrayError, FloatingPointError, ValueError, RuntimeError)
RUNTIME_SELECTION_META_FIELDS = (
    "selection_authority",
    "protocol_version",
    "target_observed_start",
    "target_observed_end",
    "source_history_start",
    "source_history_end",
    "target_test_excluded",
    "source_future_excluded",
    "source_alignment_mode",
    "feature_cols",
    "representation",
    "scaling",
    "scaler_fit_scope",
    "selected_sources_runtime",
    "candidate_pool_digest",
    "selection_result_digest",
    "source_skip_diagnostics",
)
SOURCE_FAILURE_MARKERS = (
    "ss-tl failed for source_key",
    "non-finite",
    "model weights contain non-finite values",
    "prediction contains",
    "nan",
    "inf",
    "floating point",
    "overflow",
    "invalid value encountered",
)
NON_SOURCE_FAILURE_MARKERS = (
    "sales must remain",
    "invalid source_key",
    "source_key not found",
    "selected source_key not found",
    "inconsistent target y_test",
    "target",
    "schema",
    "config",
    "feature columns",
    "sales not in feature",
    "sales column",
)


def runtime_selection_meta(selection_result: Mapping[str, object]) -> Dict[str, object]:
    """Extract the complete D4-D6 runtime selection trace from selector output."""
    raw_meta = selection_result.get("meta", {})
    if not isinstance(raw_meta, Mapping) or raw_meta.get("selection_authority") != "runtime":
        return {}
    missing = [field for field in RUNTIME_SELECTION_META_FIELDS if field not in raw_meta]
    if missing:
        raise ValueError(f"Runtime source selection metadata is incomplete: {missing}")
    return {field: raw_meta[field] for field in RUNTIME_SELECTION_META_FIELDS}


def _message(exc: BaseException) -> str:
    return str(exc).lower()


def _contains_marker(message: str, marker: str) -> bool:
    if marker in {"nan", "inf"}:
        return re.search(rf"(?<![a-z]){re.escape(marker)}(?![a-z])", message) is not None
    return marker in message


def is_nonfinite_source_failure(exc: BaseException) -> bool:
    """Return True when a skipped source failed due to detected NaN/Inf values."""
    if isinstance(exc, NonFiniteArrayError):
        return True
    message = _message(exc)
    return (
        "non-finite" in message
        or "nan_count" in message
        or "inf_count" in message
        or _contains_marker(message, "nan")
        or _contains_marker(message, "inf")
    )


def should_skip_source_exception(exc: BaseException) -> bool:
    """Return True when an exception is safe to treat as current-source failure."""
    message = _message(exc)
    if any(_contains_marker(message, marker) for marker in NON_SOURCE_FAILURE_MARKERS):
        return False
    if any(_contains_marker(message, marker) for marker in SOURCE_FAILURE_MARKERS):
        return True
    return isinstance(exc, (NonFiniteArrayError, FloatingPointError))


def make_failed_source(source_key: Tuple[Any, ...], exc: BaseException) -> Dict[str, object]:
    """Build a serializable failed-source diagnostic entry."""
    return {
        "failed_source_key": tuple(source_key),
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
    }


def source_failure_messages(failed_sources: Sequence[Dict[str, object]]) -> list[str]:
    """Return concise human-readable source failure summaries."""
    messages: list[str] = []
    for entry in failed_sources:
        messages.append(
            f"{entry.get('failed_source_key')}: "
            f"{entry.get('exception_type')}: "
            f"{entry.get('exception_message')}"
        )
    return messages


def source_failure_meta(
    *,
    requested_k: int,
    selected_sources: Sequence[Dict[str, object]],
    valid_source_count: int,
    failed_sources: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    """Summarize skipped source diagnostics for method metadata."""
    failed_source_keys = [entry["failed_source_key"] for entry in failed_sources]
    skipped_nonfinite_source_count = sum(
        1
        for entry in failed_sources
        if str(entry.get("exception_type", "")) == "NonFiniteArrayError"
        or "non-finite" in str(entry.get("exception_message", "")).lower()
        or "nan_count" in str(entry.get("exception_message", "")).lower()
        or "inf_count" in str(entry.get("exception_message", "")).lower()
    )
    skipped_source_count = len(failed_sources)
    return {
        "requested_k": int(requested_k),
        "effective_k": int(valid_source_count),
        "selected_source_count": int(len(selected_sources)),
        "valid_source_count": int(valid_source_count),
        "skipped_source_count": int(skipped_source_count),
        "failed_source_count": int(skipped_source_count),
        "failed_source_keys": failed_source_keys,
        "skipped_nonfinite_source_count": int(skipped_nonfinite_source_count),
        "selected_sources": list(selected_sources),
        "failed_sources": list(failed_sources),
        "source_failure_messages": source_failure_messages(failed_sources),
    }


def normalize_successful_source_weights(weights: Sequence[float]) -> list[float]:
    """Renormalize surviving source weights after failed sources are skipped."""
    arr = np.asarray(weights, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("No successful source weights to normalize.")
    total = float(arr.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"Successful source weights must sum to a positive finite value, got {total}")
    return [float(value) for value in (arr / total)]


def all_sources_failed_message(method_name: str, failed_sources: Sequence[Dict[str, object]]) -> str:
    """Return a clear error message for all-source failure."""
    messages = source_failure_messages(failed_sources)
    summary = "; ".join(messages[:3])
    if len(messages) > 3:
        summary = f"{summary}; ... {len(messages) - 3} more"
    return (
        f"{method_name}: all selected sources failed "
        f"(failed_source_count={len(failed_sources)}). "
        f"Reasons: {summary}"
    )


class AllSourcesFailedError(RuntimeError):
    """Raised when every selected source fails with source-level errors."""

    def __init__(
        self,
        method_name: str,
        failed_sources: Sequence[Dict[str, object]],
        *,
        selected_sources: Optional[Sequence[Dict[str, object]]] = None,
        selection_meta: Optional[Mapping[str, object]] = None,
    ) -> None:
        self.method_name = str(method_name)
        self.failed_sources = list(failed_sources)
        self.selected_sources = list(selected_sources or [])
        self.selection_meta = dict(selection_meta or {})
        super().__init__(all_sources_failed_message(self.method_name, self.failed_sources))


def is_all_sources_failed_error(exc: BaseException) -> bool:
    """Return True when an exception is the typed all-source failure."""
    return isinstance(exc, AllSourcesFailedError)


def error_row_from_all_sources_failed(
    exc: AllSourcesFailedError,
    *,
    requested_k: int,
    elapsed: float,
) -> Dict[str, object]:
    """Build raw result payload for entity-level all-source error rows."""
    return {
        "rmse": np.nan,
        "accuracy": np.nan,
        "mae": np.nan,
        "mape": np.nan,
        "smape": np.nan,
        "training_time": float(elapsed),
        "prediction_shape": "N/A",
        "error": str(exc),
        "meta": {
            **source_failure_meta(
                requested_k=requested_k,
                selected_sources=exc.selected_sources,
                valid_source_count=0,
                failed_sources=exc.failed_sources,
            ),
            **exc.selection_meta,
        },
    }
