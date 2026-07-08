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


def _missing_metric_reasons(row: Dict[str, Any]) -> str:
    reasons = []
    for metric in ("rmse", "smape"):
        value = row.get(metric)
        if value is None or not str(value).strip():
            reasons.append(f"{metric} is missing")
            continue
        try:
            number = float(str(value).strip())
        except (TypeError, ValueError):
            reasons.append(f"{metric} is non-numeric")
            continue
        if math.isnan(number):
            reasons.append(f"{metric} is NaN")
        elif math.isinf(number):
            reasons.append(f"{metric} is infinite")
    return "; ".join(reasons) or "rmse/smape are missing or non-finite"


def annotate_silent_metric_failure(row: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy with explicit result status and failure type."""
    out = dict(row)
    error = str(out.get("error", "") or "").strip()
    if error:
        out["result_status"] = "failed"
        out["failure_type"] = str(out.get("failure_type", "") or "training_exception")
        return out
    if row_has_silent_metric_failure(out):
        out["result_status"] = "failed"
        out["failure_type"] = "silent_metric_failure"
        out["error"] = f"silent_metric_failure: {_missing_metric_reasons(out)}"
        return out
    out["result_status"] = str(out.get("result_status", "") or "success")
    out["failure_type"] = str(out.get("failure_type", "") or "")
    return out
