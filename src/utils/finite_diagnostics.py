"""Finite-value diagnostics for model inputs, predictions, and labels."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


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


def _sample_dates(df: pd.DataFrame) -> list[str]:
    if "date" not in df.columns:
        return []
    values = df["date"].dropna().head(5).tolist()
    return [str(value) for value in values]


def validate_feature_frame_finite(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    context: str,
    dataset_id: int | str | None = None,
    method: str | None = None,
    role: str | None = None,
    entity_id: str | None = None,
    stage: str | None = None,
    allow_fill: bool = False,
) -> Dict[str, Any] | Tuple[pd.DataFrame, Dict[str, Any]]:
    """Validate feature columns in a dataframe, optionally repairing numeric non-finites."""
    cols = [str(col) for col in feature_columns]
    missing = [col for col in cols if col not in df.columns]
    diagnostics: Dict[str, Any] = {
        "context": str(context),
        "dataset_id": dataset_id,
        "method": method,
        "role": role,
        "entity_id": entity_id,
        "stage": stage or str(context),
        "feature_columns": cols,
        "missing_columns": missing,
        "bad_columns": {},
    }
    if missing:
        raise NonFiniteArrayError(
            f"[{context}] missing feature columns: {missing}",
            diagnostics=diagnostics,
        )

    repaired = df.copy() if allow_fill else df
    repaired_columns: Dict[str, int] = {}
    bad_columns: Dict[str, Dict[str, Any]] = {}

    for col in cols:
        values = pd.to_numeric(repaired[col], errors="coerce")
        nan_count = int(values.isna().sum())
        posinf_count = int(np.isposinf(values.to_numpy(dtype=np.float64, copy=False)).sum())
        neginf_count = int(np.isneginf(values.to_numpy(dtype=np.float64, copy=False)).sum())
        if nan_count or posinf_count or neginf_count:
            bad_columns[col] = {
                "nan_count": nan_count,
                "posinf_count": posinf_count,
                "neginf_count": neginf_count,
                "sample_dates": _sample_dates(repaired),
            }
            if allow_fill:
                repaired[col] = values.replace([np.inf, -np.inf], np.nan).fillna(0)
                repaired_columns[col] = nan_count + posinf_count + neginf_count

    diagnostics["bad_columns"] = bad_columns
    diagnostics["nan_count"] = int(sum(item["nan_count"] for item in bad_columns.values()))
    diagnostics["posinf_count"] = int(sum(item["posinf_count"] for item in bad_columns.values()))
    diagnostics["neginf_count"] = int(sum(item["neginf_count"] for item in bad_columns.values()))
    diagnostics["repaired_columns"] = repaired_columns
    diagnostics["source_numeric_na_repaired"] = bool(allow_fill and repaired_columns)

    if bad_columns and not allow_fill:
        prefix_parts = [
            f"D{dataset_id}" if dataset_id is not None else None,
            method,
            role,
            entity_id,
            stage or context,
        ]
        prefix = "[" + "][".join(str(part) for part in prefix_parts if part) + "]"
        raise NonFiniteArrayError(
            f"{prefix} Non-finite values detected: bad_columns={bad_columns}",
            diagnostics=diagnostics,
        )

    if allow_fill:
        return repaired, diagnostics
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
