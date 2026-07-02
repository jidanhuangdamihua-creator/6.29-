"""Finite-value diagnostics for model inputs, predictions, and labels."""

from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np


class NonFiniteArrayError(ValueError):
    """Raised when an array contains NaN or Inf values."""

    def __init__(self, message: str, diagnostics: Mapping[str, Any]):
        super().__init__(message)
        self.diagnostics = dict(diagnostics)


def summarize_finite_array(values: Any, name: str) -> Dict[str, Any]:
    """Return shape and NaN/Inf counts for an array-like value."""
    arr = np.asarray(values)
    try:
        numeric = arr.astype(np.float64, copy=False)
    except (TypeError, ValueError):
        numeric = np.asarray(arr, dtype=np.float64)
    return {
        f"{name}_shape": tuple(arr.shape),
        f"{name}_nan_count": int(np.isnan(numeric).sum()),
        f"{name}_inf_count": int(np.isinf(numeric).sum()),
    }


def validate_finite_array(
    values: Any,
    name: str,
    context: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return diagnostics when finite; raise with diagnostics when not finite."""
    diagnostics: Dict[str, Any] = dict(context or {})
    diagnostics.update(summarize_finite_array(values, name=name))
    nan_count = int(diagnostics[f"{name}_nan_count"])
    inf_count = int(diagnostics[f"{name}_inf_count"])
    if nan_count or inf_count:
        raise NonFiniteArrayError(
            f"{name} contains non-finite values: nan_count={nan_count} inf_count={inf_count}",
            diagnostics=diagnostics,
        )
    return diagnostics


def summarize_model_weights(model: Any) -> Dict[str, int]:
    """Count NaN/Inf values across Keras-style model weights."""
    nan_count = 0
    inf_count = 0
    if model is None or not hasattr(model, "get_weights"):
        return {"model_weight_nan_count": 0, "model_weight_inf_count": 0}
    for weights in model.get_weights():
        arr = np.asarray(weights, dtype=np.float64)
        nan_count += int(np.isnan(arr).sum())
        inf_count += int(np.isinf(arr).sum())
    return {
        "model_weight_nan_count": int(nan_count),
        "model_weight_inf_count": int(inf_count),
    }
