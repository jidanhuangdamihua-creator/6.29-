"""Shared result schema alignment helpers for D1-D6 experiment outputs."""

from __future__ import annotations

import math
import hashlib
import json
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

import pandas as pd

from src.constants import (
    RESULT_CONTRACT_VERSION,
    RESULT_SCHEMA_COLUMNS,
    SCHEMA_FAMILY_D1_D3,
    SCHEMA_FAMILY_D4_D6,
    preferred_columns_with_extras,
    stable_json_cell,
)
from src.utils.result_validation import annotate_silent_metric_failure, classify_protocol_result


TRACE_COLUMNS = [
    "dataset_id",
    "scenario",
    "target_entity_key",
    "source_identifier",
    "selected_sources",
]

RESULT_SCHEMA_REGISTRY_VERSION = "result_schema_registry_v1"
REGISTERED_RESULT_EXTRA_COLUMNS_BY_SCHEMA_FAMILY = MappingProxyType(
    {
        SCHEMA_FAMILY_D1_D3: (
            "sample_count",
            "source_identification",
            "feature_cols_final",
            "rfe_candidate_features",
            "rfe_selected_features",
            "signature_components",
        ),
        SCHEMA_FAMILY_D4_D6: (
            "sample_count",
            "domain_filter_applied_to_source",
            "domain_filter_scope",
            "domain_filter_column",
            "domain_filter_value",
            "target_domain_validation_passed",
            "target_domain_validation_target_count",
            "source_pool_policy",
        ),
    }
)


def result_schema_registry_digest() -> str:
    payload = {
        "version": RESULT_SCHEMA_REGISTRY_VERSION,
        "base_columns": list(RESULT_SCHEMA_COLUMNS),
        "extras": {
            family: list(columns)
            for family, columns in sorted(
                REGISTERED_RESULT_EXTRA_COLUMNS_BY_SCHEMA_FAMILY.items()
            )
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _first_seen_columns(records: Sequence[Mapping[str, Any]]) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for record in records:
        for column in record:
            if column not in seen:
                columns.append(column)
                seen.add(column)
    return columns


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, bool) else False


def _is_empty_string(value: Any) -> bool:
    return isinstance(value, str) and value == ""


def _fill_missing_or_empty(value: Any, replacement: str) -> Any:
    if _is_missing(value) or _is_empty_string(value):
        return replacement
    return value


def _replace_empty_only(value: Any, replacement: str) -> Any:
    if _is_empty_string(value):
        return replacement
    return value


def normalize_information_sharing_contract(value: Any) -> str:
    """Normalize only the D1-D3 information_sharing contract field."""
    text = str(value).strip().lower()
    normalized = "_".join(text.replace("-", " ").replace("_", " ").split())
    if normalized == "with":
        return "with"
    if normalized == "with_information_sharing":
        return "with"
    if normalized == "without":
        return "without"
    if normalized == "without_information_sharing":
        return "without"
    raise ValueError(f"Unsupported information_sharing contract value: {value!r}")


def _preferred_columns(
    preferred_columns: Sequence[str] | None,
    extra_preferred_columns: Sequence[str],
) -> list[str]:
    columns = list(RESULT_SCHEMA_COLUMNS if preferred_columns is None else preferred_columns)
    for column in extra_preferred_columns:
        if column not in columns:
            columns.append(column)
    return columns


def align_result_records(
    records: Sequence[Mapping[str, Any]],
    *,
    schema_family: str,
    preferred_columns: Sequence[str] | None = None,
    extra_preferred_columns: Sequence[str] = (),
    normalize_information_sharing: bool = False,
    fill_missing_contract_defaults: bool = False,
) -> pd.DataFrame:
    """Align result rows to the shared CSV schema using dict-first materialization."""
    record_list = [dict(record) for record in records]
    ordered_preferred = _preferred_columns(preferred_columns, extra_preferred_columns)
    columns = _first_seen_columns(record_list)

    rows = [
        {column: record[column] if column in record else float("nan") for column in columns}
        for record in record_list
    ]

    missing_columns = [column for column in ordered_preferred if column not in columns]
    columns.extend(missing_columns)
    for row in rows:
        for column in missing_columns:
            row[column] = ""

    for row in rows:
        if "result_contract_version" in columns:
            current = row.get("result_contract_version", "")
            if fill_missing_contract_defaults:
                row["result_contract_version"] = _fill_missing_or_empty(
                    current,
                    RESULT_CONTRACT_VERSION,
                )
            else:
                row["result_contract_version"] = _replace_empty_only(
                    current,
                    RESULT_CONTRACT_VERSION,
                )
        if "schema_family" in columns:
            current = row.get("schema_family", "")
            if fill_missing_contract_defaults:
                row["schema_family"] = _fill_missing_or_empty(current, schema_family)
            else:
                row["schema_family"] = _replace_empty_only(current, schema_family)

    if normalize_information_sharing and "information_sharing" in columns:
        for row in rows:
            row["information_sharing"] = normalize_information_sharing_contract(
                row["information_sharing"]
            )

    if rows:
        rows = [annotate_silent_metric_failure(row) for row in rows]
        for row in rows:
            row["result_status"] = classify_protocol_result(row)
        rows = [
            {column: stable_json_cell(value) for column, value in row.items()}
            for row in rows
        ]

    ordered_columns = preferred_columns_with_extras(columns, ordered_preferred)
    return pd.DataFrame(rows, columns=ordered_columns)


def align_d1_d3_result_records(
    records: Sequence[Mapping[str, Any]],
    *,
    schema_family: str = SCHEMA_FAMILY_D1_D3,
) -> pd.DataFrame:
    return align_result_records(
        records,
        schema_family=schema_family,
        preferred_columns=RESULT_SCHEMA_COLUMNS,
        normalize_information_sharing=True,
        fill_missing_contract_defaults=True,
    )


def align_d4_d6_result_records(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    return align_result_records(
        records,
        schema_family=SCHEMA_FAMILY_D4_D6,
        preferred_columns=RESULT_SCHEMA_COLUMNS,
        extra_preferred_columns=TRACE_COLUMNS,
        normalize_information_sharing=False,
        fill_missing_contract_defaults=False,
    )
