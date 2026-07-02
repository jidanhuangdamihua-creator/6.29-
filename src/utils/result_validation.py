"""Result-row validation helpers for experiment outputs."""

from __future__ import annotations

from typing import Any, Dict

import math


def _is_missing_or_nonfinite(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    if not text:
        return True
    try:
        number = float(text)
    except (TypeError, ValueError):
        return True
    return not math.isfinite(number)


def row_has_silent_metric_failure(row: Dict[str, Any]) -> bool:
    """Detect rows that look successful but have no usable metrics."""
    error = str(row.get("error", "") or "").strip()
    if error:
        return False
    prediction_shape = str(row.get("prediction_shape", "") or "").strip()
    if not prediction_shape or prediction_shape == "N/A":
        return False
    return _is_missing_or_nonfinite(row.get("rmse")) or _is_missing_or_nonfinite(row.get("smape"))


def annotate_silent_metric_failure(row: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy with an explicit error if metrics are silently missing."""
    out = dict(row)
    if row_has_silent_metric_failure(out):
        out["error"] = (
            "silent_metric_failure: prediction_shape is present but rmse/smape "
            "are missing or non-finite"
        )
    return out
