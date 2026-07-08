"""Helpers for source-level failure tolerance in multi-source TL methods."""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np

from src.utils.finite_diagnostics import NonFiniteArrayError


SOURCE_LEVEL_EXCEPTIONS = (NonFiniteArrayError, FloatingPointError, ValueError, RuntimeError)
NON_SOURCE_FAILURE_MARKERS = (
    "target",
    "feature_cols",
    "source_df",
    "target_df",
    "missing feature",
    "shape mismatch",
    "cannot fuse",
    "selected source_key",
)


def is_nonfinite_source_failure(exc: BaseException) -> bool:
    """Return True when a skipped source failed due to detected NaN/Inf values."""
    if isinstance(exc, NonFiniteArrayError):
        return True
    message = str(exc).lower()
    return "non-finite" in message or "nan_count" in message or "inf_count" in message


def should_skip_source_exception(exc: BaseException) -> bool:
    """Return True when an exception is safe to treat as current-source failure."""
    if isinstance(exc, (NonFiniteArrayError, FloatingPointError)):
        return True
    if not isinstance(exc, (ValueError, RuntimeError)):
        return False
    message = str(exc).lower()
    return not any(marker in message for marker in NON_SOURCE_FAILURE_MARKERS)


def make_failed_source(source_key: Tuple[Any, ...], exc: BaseException) -> Dict[str, object]:
    """Build a serializable failed-source diagnostic entry."""
    return {
        "failed_source_key": tuple(source_key),
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
    }


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
        "effective_k": len(selected_sources),
        "valid_source_count": int(valid_source_count),
        "skipped_source_count": int(skipped_source_count),
        "failed_source_count": int(skipped_source_count),
        "failed_source_keys": failed_source_keys,
        "skipped_nonfinite_source_count": int(skipped_nonfinite_source_count),
        "failed_sources": list(failed_sources),
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
    return f"{method_name} failed for all selected sources: {list(failed_sources)}"
