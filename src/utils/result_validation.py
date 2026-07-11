"""Result-row validation helpers for experiment outputs."""

from __future__ import annotations

from typing import Any, Dict

import math
import pandas as pd

from src.constants import STRICT_PROTOCOL_FIELDS
from src.protocols.experiment_protocol import FORMAL_HORIZONS, FORMAL_SEEDS, PROTOCOL_VERSION


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


def _missing_contract_value(value: Any) -> bool:
    if value is None:
        return True
    try:
        if bool(pd.isna(value)):
            return True
    except (TypeError, ValueError):
        pass
    return not str(value).strip()


def _as_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def classify_protocol_result(row: Dict[str, Any]) -> str:
    """Classify one row without fabricating any missing protocol evidence."""
    if any(_missing_contract_value(row.get(field)) for field in STRICT_PROTOCOL_FIELDS):
        return "legacy_unverified"
    if str(row.get("error", "") or "").strip():
        return "failed"
    try:
        start = pd.Timestamp(row["knn_observed_start"]).normalize()
        end = pd.Timestamp(row["knn_observed_end"]).normalize()
        valid = (
            str(row["protocol_version"]) == PROTOCOL_VERSION
            and str(row["protocol_track"]) in {"strict_paper", "extended"}
            and str(row["knn_representation"])
            in {"daily_sales_flattened_30d", "not_applicable_target_only"}
            and _as_true(row["target_test_excluded"])
            and _as_true(row["source_future_excluded"])
            and int(row["horizon"]) in FORMAL_HORIZONS
            and int(row["seed"]) in FORMAL_SEEDS
            and str(row["primary_metric_space"]) == "original_sales"
            and (end - start).days == 29
        )
    except (TypeError, ValueError):
        valid = False
    if not valid:
        return "protocol_invalid"
    current = str(row.get("result_status", "") or "").strip()
    return "confirmed_baseline" if current == "confirmed_baseline" else "trial"


def validate_confirmed_baseline_group(rows: pd.DataFrame) -> pd.DataFrame:
    """Promote exactly one complete 5-horizon x 5-seed group."""
    if rows.empty:
        raise ValueError("confirmed baseline group may not be empty")
    classifications = [classify_protocol_result(row) for row in rows.to_dict(orient="records")]
    invalid = [status for status in classifications if status not in {"trial", "confirmed_baseline"}]
    if invalid:
        raise ValueError(f"confirmed baseline group contains non-strict rows: {invalid}")
    pairs = set(zip(rows["horizon"].astype(int), rows["seed"].astype(int)))
    expected = {(horizon, seed) for horizon in FORMAL_HORIZONS for seed in FORMAL_SEEDS}
    if pairs != expected or len(rows) != len(expected):
        raise ValueError("confirmed baseline requires five horizons and five seeds exactly once")
    identity_cols = [
        column
        for column in ("dataset_id", "protocol_track", "scenario", "target_entity_key", "method")
        if column in rows.columns
    ]
    for column in identity_cols:
        if rows[column].astype(str).nunique() != 1:
            raise ValueError(f"confirmed baseline group mixes identity column {column}")
    if rows["sample_manifest_digest"].astype(str).nunique() != 1:
        raise ValueError("confirmed baseline group mixes sample manifests")
    for metric in ("rmse", "mae", "smape", "accuracy"):
        values = pd.to_numeric(rows[metric], errors="coerce")
        if values.isna().any() or not values.map(math.isfinite).all():
            raise ValueError(f"confirmed baseline metric {metric} is missing or non-finite")
    confirmed = rows.copy()
    confirmed["result_status"] = "confirmed_baseline"
    return confirmed


def confirmed_baseline_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Return only explicitly promoted baseline rows; legacy/trial rows never mix."""
    if "result_status" not in rows.columns:
        return rows.iloc[0:0].copy()
    return rows[rows["result_status"].astype(str) == "confirmed_baseline"].copy()


def promote_complete_baseline_groups(rows: pd.DataFrame) -> pd.DataFrame:
    """Promote complete groups while leaving incomplete trials and legacy rows isolated."""
    promoted = rows.copy()
    promoted["result_status"] = [
        classify_protocol_result(record)
        for record in promoted.to_dict(orient="records")
    ]
    identity_cols = [
        column
        for column in (
            "dataset_id",
            "protocol_track",
            "protocol_version",
            "scenario",
            "target_entity_key",
            "method",
            "candidate_pool_digest",
            "selection_result_digest",
            "sample_manifest_digest",
        )
        if column in promoted.columns
    ]
    strict = promoted[promoted["result_status"] == "trial"]
    if strict.empty or not identity_cols:
        return promoted
    for _, group in strict.groupby(identity_cols, dropna=False, sort=False):
        try:
            confirmed = validate_confirmed_baseline_group(group)
        except ValueError:
            continue
        promoted.loc[confirmed.index, "result_status"] = "confirmed_baseline"
    return promoted
