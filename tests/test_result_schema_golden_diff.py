from __future__ import annotations

import math
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from scripts import run_d4_experiment, run_d5_experiment, run_d6_experiment
from scripts import run_full_paper_experiments
from src.constants import (
    RESULT_CONTRACT_VERSION,
    RESULT_SCHEMA_COLUMNS,
    SCHEMA_FAMILY_D1_D3,
    SCHEMA_FAMILY_D4_D6,
    preferred_columns_with_extras,
    stable_json_cell,
)
from src.utils.result_validation import annotate_silent_metric_failure


MAX_DIFFS_TO_REPORT = 50


def _first_seen_columns(records: Iterable[dict[str, Any]]) -> list[str]:
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
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _fillna_empty_replace_empty(value: Any, replacement: str) -> Any:
    if _is_missing(value) or value == "":
        return replacement
    return value


def _replace_empty_only(value: Any, replacement: str) -> Any:
    if isinstance(value, str) and value == "":
        return replacement
    return value


def _normalize_information_sharing(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"with", "with_information_sharing"}:
        return "with"
    if text in {"without", "without_information_sharing"}:
        return "without"
    raise ValueError(f"Unsupported information_sharing contract value: {value!r}")


def _candidate_align_records(
    records: list[dict[str, Any]],
    *,
    schema_family: str,
    preferred_columns: list[str],
    normalize_information_sharing: bool,
    fill_missing_contract_defaults: bool,
) -> pd.DataFrame:
    columns = _first_seen_columns(records)
    rows = [
        {column: record[column] if column in record else np.nan for column in columns}
        for record in records
    ]

    missing_columns = [column for column in preferred_columns if column not in columns]
    columns.extend(missing_columns)
    for row in rows:
        for column in missing_columns:
            row[column] = ""

    if "result_contract_version" in columns:
        for row in rows:
            current = row.get("result_contract_version", "")
            if fill_missing_contract_defaults:
                row["result_contract_version"] = _fillna_empty_replace_empty(
                    current,
                    RESULT_CONTRACT_VERSION,
                )
            else:
                row["result_contract_version"] = _replace_empty_only(
                    current,
                    RESULT_CONTRACT_VERSION,
                )

    if "schema_family" in columns:
        for row in rows:
            current = row.get("schema_family", "")
            if fill_missing_contract_defaults:
                row["schema_family"] = _fillna_empty_replace_empty(current, schema_family)
            else:
                row["schema_family"] = _replace_empty_only(current, schema_family)

    if normalize_information_sharing and "information_sharing" in columns and rows:
        for row in rows:
            row["information_sharing"] = _normalize_information_sharing(
                row["information_sharing"]
            )

    if rows:
        rows = [annotate_silent_metric_failure(row) for row in rows]
        rows = [
            {column: stable_json_cell(value) for column, value in row.items()}
            for row in rows
        ]

    aligned = pd.DataFrame(rows, columns=columns)
    return aligned[
        preferred_columns_with_extras(aligned.columns, preferred_columns)
    ]


def _cell_values_match(old_value: Any, candidate_value: Any) -> bool:
    if _is_missing(old_value) and _is_missing(candidate_value):
        return True
    if _is_missing(old_value) or _is_missing(candidate_value):
        return False
    if isinstance(old_value, (bool, np.bool_)) or isinstance(candidate_value, (bool, np.bool_)):
        return type(old_value) is type(candidate_value) and old_value == candidate_value
    return type(old_value) is type(candidate_value) and old_value == candidate_value


def _assert_frames_equal_cell_by_cell(
    old_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
) -> None:
    failures: list[str] = []
    if old_df.shape != candidate_df.shape:
        failures.append(f"shape old={old_df.shape} candidate={candidate_df.shape}")
    if list(old_df.columns) != list(candidate_df.columns):
        failures.append(
            "columns differ\n"
            f"old={list(old_df.columns)}\n"
            f"candidate={list(candidate_df.columns)}"
        )

    if failures:
        raise AssertionError("\n".join(failures))

    for row_index in range(old_df.shape[0]):
        for column in old_df.columns:
            old_value = old_df.iloc[row_index][column]
            candidate_value = candidate_df.iloc[row_index][column]
            if _cell_values_match(old_value, candidate_value):
                continue
            failures.append(
                "row={row} column={column!r} "
                "old={old!r} ({old_type}) candidate={candidate!r} ({candidate_type})".format(
                    row=row_index,
                    column=column,
                    old=old_value,
                    old_type=type(old_value).__name__,
                    candidate=candidate_value,
                    candidate_type=type(candidate_value).__name__,
                )
            )
            if len(failures) >= MAX_DIFFS_TO_REPORT:
                raise AssertionError(
                    "DataFrame cell diff exceeded limit:\n" + "\n".join(failures)
                )

    if failures:
        raise AssertionError("DataFrame cell diff:\n" + "\n".join(failures))


def _d1_d3_fixture_records() -> list[dict[str, Any]]:
    return [
        {
            "dataset": "Dataset1",
            "dataset_id": 1,
            "method": "MSWA-TL",
            "information_sharing": "without_information_sharing",
            "scenario": "without_information_sharing",
            "target_entity_key": "store-1-item-2",
            "source_identifier": "source-store-3",
            "signature_components": {
                "dataset": "D1",
                "scenario": "without_information_sharing",
            },
            "diagnostics": {"source_pool_size": 3, "finite": True},
            "selected_sources": [
                {"source_key": "source-store-3", "distance": 0.0},
                {"source_key": "source-store-4", "distance": 1.23},
            ],
            "selected_source_ids": ["B1:4", "B2:3"],
            "selected_source_keys": ["B1 Item4", "B2 Item3"],
            "failed_sources": [],
            "source_failure_messages": [],
            "source_domain_filter": {"store_id": 1},
            "source_domain_filter_applied": True,
            "source_domain_filter_reason": "without_information_sharing_same_store",
            "source_pool_size_before_filter": 5,
            "source_pool_size_after_filter": 3,
            "rmse": 0,
            "smape": 1.23,
            "mae": 0.5,
            "mape": 0.25,
            "accuracy": 0.75,
            "prediction_shape": [2, 1],
            "inverse_transform_applied": False,
            "inverse_transform_available": True,
            "error": "",
            "extra_column": "keep-extra",
        },
        {
            "result_contract_version": "",
            "schema_family": None,
            "dataset": "Dataset2",
            "dataset_id": 2,
            "method": "MSML-TL-RFE",
            "information_sharing": "with",
            "scenario": "with_information_sharing",
            "target_entity_key": "brand-1",
            "source_identifier": "brand-2",
            "signature_components": {"scenario": "with_information_sharing"},
            "diagnostics": {"nan_count": 1, "ok": False},
            "selected_sources": [{"source_key": "brand-2", "distance": float("nan")}],
            "selected_source_ids": [],
            "selected_source_keys": [],
            "failed_sources": [
                {"failed_source_key": ["brand", 9], "exception_type": "ValueError"}
            ],
            "source_failure_messages": ["brand 9 failed"],
            "rmse": float("nan"),
            "smape": 1.23,
            "mae": "",
            "mape": None,
            "accuracy": True,
            "prediction_shape": [1, 1],
            "error": "",
            "extra_column": None,
        },
        {
            "dataset": "Dataset3",
            "dataset_id": 3,
            "method": "No-TL",
            "information_sharing": "without",
            "scenario": "without_information_sharing",
            "selected_sources": "not_applicable",
            "failed_sources": "",
            "rmse": 1.23,
            "smape": 0,
            "mae": 0,
            "mape": 0,
            "accuracy": False,
            "prediction_shape": "",
            "error": None,
            "empty_string_extra": "",
        },
    ]


def _d4_d6_fixture_records() -> list[dict[str, Any]]:
    return [
        {
            "result_contract_version": "",
            "schema_family": "",
            "dataset": "Dataset4",
            "dataset_id": 4,
            "method": "SS-TL",
            "information_sharing": "without",
            "scenario": "without_information_sharing",
            "target_entity_key": "store-a-product-b",
            "source_identifier": "store-c-product-d",
            "signature_components": {"target": "store-a-product-b"},
            "diagnostics": {"feature_consistency_status": "ok", "count": 0},
            "selected_sources": [{"source_key": "store-c-product-d", "distance": 1.23}],
            "selected_source_ids": ["store-c:product-d"],
            "selected_source_keys": ["store-c product-d"],
            "failed_sources": [],
            "source_failure_messages": [],
            "source_domain_filter": {"second_category_id": 20},
            "source_domain_filter_name": "without_information_sharing_same_domain_protocol",
            "source_domain_filter_applied": True,
            "source_pool_size_before_filter": 10,
            "source_pool_size_after_filter": 3,
            "rmse": 0,
            "smape": 1.23,
            "mae": 0.1,
            "mape": 0.2,
            "accuracy": 0.3,
            "prediction_shape": [3, 1],
            "paper_metric_aligned": False,
            "inverse_transform_available": True,
            "error": "",
            "extra_result_detail": "keep",
        },
        {
            "result_contract_version": None,
            "schema_family": None,
            "dataset": "Dataset5",
            "dataset_id": 5,
            "method": "MSWA-TL",
            "information_sharing": "with_information_sharing",
            "scenario": "with_information_sharing",
            "target_entity_key": "48_364606",
            "source_identifier": "48_564533",
            "signature_components": {"scenario": "with_information_sharing"},
            "diagnostics": {"nan_count": 1, "ok": False},
            "selected_sources": [{"source_key": "48_564533", "weight": True}],
            "selected_source_ids": [],
            "selected_source_keys": [],
            "failed_sources": [
                {"failed_source_key": ["48", "314384"], "exception_type": "RuntimeError"}
            ],
            "source_failure_messages": ["source failed"],
            "rmse": float("nan"),
            "smape": "",
            "mae": None,
            "mape": 1.23,
            "accuracy": True,
            "prediction_shape": [1, 1],
            "error": "",
            "extra_result_detail": None,
        },
        {
            "dataset": "Dataset6",
            "dataset_id": 6,
            "method": "No-TL",
            "information_sharing": "with",
            "scenario": "with_information_sharing",
            "target_entity_key": "CA_1_FOODS_3_586",
            "source_identifier": "not_applicable",
            "selected_sources": "not_applicable",
            "failed_sources": "",
            "rmse": 1.23,
            "smape": 0,
            "mae": 0,
            "mape": 0,
            "accuracy": False,
            "prediction_shape": "",
            "error": None,
            "empty_string_extra": "",
        },
    ]


def test_d1_d3_dict_first_candidate_matches_existing_frame_alignment_cell_by_cell() -> None:
    records = _d1_d3_fixture_records()

    old_df = run_full_paper_experiments._align_frame_to_preferred_columns(
        pd.DataFrame(records),
        SCHEMA_FAMILY_D1_D3,
    )
    candidate_df = _candidate_align_records(
        records,
        schema_family=SCHEMA_FAMILY_D1_D3,
        preferred_columns=list(RESULT_SCHEMA_COLUMNS),
        normalize_information_sharing=True,
        fill_missing_contract_defaults=True,
    )

    _assert_frames_equal_cell_by_cell(old_df, candidate_df)


def test_d4_d5_d6_existing_runner_alignment_outputs_are_identical() -> None:
    records = _d4_d6_fixture_records()
    raw = pd.DataFrame(records)

    d4_df = run_d4_experiment._align_results_to_reference_schema(raw)
    d5_df = run_d5_experiment._align_results_to_reference_schema(raw)
    d6_df = run_d6_experiment._align_results_to_reference_schema(raw)

    _assert_frames_equal_cell_by_cell(d4_df, d5_df)
    _assert_frames_equal_cell_by_cell(d4_df, d6_df)


def test_d4_d6_dict_first_candidate_matches_existing_frame_alignment_cell_by_cell() -> None:
    records = _d4_d6_fixture_records()
    preferred_columns = list(RESULT_SCHEMA_COLUMNS) + list(run_d4_experiment.TRACE_COLUMNS)
    candidate_df = _candidate_align_records(
        records,
        schema_family=SCHEMA_FAMILY_D4_D6,
        preferred_columns=preferred_columns,
        normalize_information_sharing=False,
        fill_missing_contract_defaults=False,
    )

    aligners: list[tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]] = [
        ("D4", run_d4_experiment._align_results_to_reference_schema),
        ("D5", run_d5_experiment._align_results_to_reference_schema),
        ("D6", run_d6_experiment._align_results_to_reference_schema),
    ]
    for runner_name, aligner in aligners:
        old_df = aligner(pd.DataFrame(records))
        try:
            _assert_frames_equal_cell_by_cell(old_df, candidate_df)
        except AssertionError as exc:
            raise AssertionError(f"{runner_name} alignment diff:\n{exc}") from exc
