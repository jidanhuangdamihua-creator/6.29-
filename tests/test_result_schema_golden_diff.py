from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import pandas as pd

from scripts import run_d4_experiment, run_d5_experiment, run_d6_experiment
from scripts import run_full_paper_experiments
from src.constants import (
    RESULT_SCHEMA_COLUMNS,
    SCHEMA_FAMILY_D1_D3,
    SCHEMA_FAMILY_D4_D6,
)
from src.utils.result_schema import (
    REGISTERED_RESULT_EXTRA_COLUMNS_BY_SCHEMA_FAMILY,
    RESULT_SCHEMA_REGISTRY_VERSION,
    TRACE_COLUMNS,
    align_d1_d3_result_records,
    align_d4_d6_result_records,
    align_result_records,
    result_schema_registry_digest,
)


MAX_DIFFS_TO_REPORT = 50


def test_result_extra_registry_is_explicit_and_deterministic() -> None:
    assert RESULT_SCHEMA_REGISTRY_VERSION == "result_schema_registry_v1"
    assert set(REGISTERED_RESULT_EXTRA_COLUMNS_BY_SCHEMA_FAMILY) == {
        SCHEMA_FAMILY_D1_D3,
        SCHEMA_FAMILY_D4_D6,
    }
    assert REGISTERED_RESULT_EXTRA_COLUMNS_BY_SCHEMA_FAMILY[SCHEMA_FAMILY_D1_D3] == (
        "sample_count",
        "source_identification",
        "feature_cols_final",
        "rfe_candidate_features",
        "rfe_selected_features",
        "signature_components",
    )
    assert len(result_schema_registry_digest()) == 64
    assert result_schema_registry_digest() == result_schema_registry_digest()


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


def _candidate_align_records(
    records: list[dict[str, Any]],
    *,
    schema_family: str,
    preferred_columns: list[str],
    extra_preferred_columns: tuple[str, ...] = (),
    normalize_information_sharing: bool,
    fill_missing_contract_defaults: bool,
) -> pd.DataFrame:
    return align_result_records(
        records,
        schema_family=schema_family,
        preferred_columns=preferred_columns,
        extra_preferred_columns=extra_preferred_columns,
        normalize_information_sharing=normalize_information_sharing,
        fill_missing_contract_defaults=fill_missing_contract_defaults,
    )


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
            "prediction_shape": (2, 1),
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
            "information_sharing": "with-information-sharing",
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
            "information_sharing": "without information sharing",
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
            "signature_components": '{"target":"store-a-product-b"}',
            "diagnostics": '{"feature_consistency_status":"ok","count":0}',
            "selected_sources": '[{"source_key":"store-c-product-d","distance":1.23}]',
            "selected_source_ids": ["store-c:product-d"],
            "selected_source_keys": ["store-c product-d"],
            "failed_sources": "[]",
            "source_failure_messages": [],
            "source_domain_filter": {"second_category_id": 20},
            "source_domain_filter_name": "without_information_sharing_same_domain_protocol",
            "source_domain_filter_applied": True,
            "source_pool_size_before_filter": np.int64(10),
            "source_pool_size_after_filter": 3,
            "rmse": np.float64(0),
            "smape": 1.23,
            "mae": 0.1,
            "mape": 0.2,
            "accuracy": np.bool_(True),
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
    candidate_df = align_d1_d3_result_records(records, schema_family=SCHEMA_FAMILY_D1_D3)

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
    candidate_df = _candidate_align_records(
        records,
        schema_family=SCHEMA_FAMILY_D4_D6,
        preferred_columns=list(RESULT_SCHEMA_COLUMNS),
        extra_preferred_columns=tuple(TRACE_COLUMNS),
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


def test_empty_records_and_empty_frames_align_to_preferred_columns() -> None:
    d1_empty = align_d1_d3_result_records([], schema_family=SCHEMA_FAMILY_D1_D3)
    d4_empty = align_d4_d6_result_records([])

    _assert_frames_equal_cell_by_cell(
        run_full_paper_experiments._align_frame_to_preferred_columns(
            pd.DataFrame(),
            SCHEMA_FAMILY_D1_D3,
        ),
        d1_empty,
    )
    _assert_frames_equal_cell_by_cell(
        run_d4_experiment._align_results_to_reference_schema(pd.DataFrame()),
        d4_empty,
    )
    assert list(d1_empty.columns) == list(RESULT_SCHEMA_COLUMNS)
    assert list(d4_empty.columns) == list(RESULT_SCHEMA_COLUMNS)


def test_full_paper_materialize_empty_inputs_returns_empty_schema_frames() -> None:
    paper_df, extended_df = run_full_paper_experiments._materialize_result_dataframes([], [])

    assert paper_df.empty
    assert extended_df.empty
    assert list(paper_df.columns) == list(RESULT_SCHEMA_COLUMNS)
    assert list(extended_df.columns) == list(RESULT_SCHEMA_COLUMNS)


def test_d4_d6_preserves_information_sharing_and_existing_contract_nulls() -> None:
    records = [
        {
            "result_contract_version": None,
            "schema_family": pd.NA,
            "dataset": "Dataset4",
            "method": "MSWA-TL",
            "information_sharing": "with_information_sharing",
            "scenario": "with_information_sharing",
            "rmse": 1.0,
            "smape": 2.0,
            "prediction_shape": (2, 1),
            "error": "",
        },
        {
            "result_contract_version": float("nan"),
            "schema_family": float("nan"),
            "dataset": "Dataset5",
            "method": "No-TL",
            "information_sharing": "without_information_sharing",
            "scenario": "without_information_sharing",
            "rmse": 1.0,
            "smape": 2.0,
            "prediction_shape": (2, 1),
            "error": "",
        },
    ]

    aligned = align_d4_d6_result_records(records)

    assert pd.isna(aligned.loc[0, "result_contract_version"])
    assert pd.isna(aligned.loc[0, "schema_family"])
    assert pd.isna(aligned.loc[1, "result_contract_version"])
    assert pd.isna(aligned.loc[1, "schema_family"])
    assert aligned.loc[0, "information_sharing"] == "with_information_sharing"
    assert aligned.loc[0, "scenario"] == "with_information_sharing"


def test_d4_d6_preserves_pre_stringified_json_cells_exactly() -> None:
    selected_sources = '[{"source":"A","distance":0.1}]'
    failed_sources = '[{"source":"B","error":"boom"}]'
    diagnostics = '{"ok":true}'
    signature_components = '{"scenario":"with_information_sharing"}'

    aligned = align_d4_d6_result_records(
        [
            {
                "dataset": "Dataset6",
                "method": "MSML-TL",
                "information_sharing": "with_information_sharing",
                "scenario": "with_information_sharing",
                "selected_sources": selected_sources,
                "failed_sources": failed_sources,
                "diagnostics": diagnostics,
                "signature_components": signature_components,
                "rmse": 1.0,
                "smape": 2.0,
                "prediction_shape": (2, 1),
                "error": "",
            }
        ]
    )

    assert aligned.loc[0, "selected_sources"] == selected_sources
    assert aligned.loc[0, "failed_sources"] == failed_sources
    assert aligned.loc[0, "diagnostics"] == diagnostics
    assert aligned.loc[0, "signature_components"] == signature_components


def test_d1_d3_normalizes_information_sharing_hyphen_and_space_only() -> None:
    aligned = align_d1_d3_result_records(
        [
            {
                "dataset": "Dataset1",
                "method": "MSWA-TL",
                "information_sharing": "with-information-sharing",
                "scenario": "with-information-sharing",
                "rmse": 1.0,
                "smape": 2.0,
                "prediction_shape": (2, 1),
                "error": "",
            },
            {
                "dataset": "Dataset1",
                "method": "MSWA-TL",
                "information_sharing": "without information sharing",
                "scenario": "without information sharing",
                "rmse": 1.0,
                "smape": 2.0,
                "prediction_shape": (2, 1),
                "error": "",
            },
        ],
        schema_family=SCHEMA_FAMILY_D1_D3,
    )

    assert list(aligned["information_sharing"]) == ["with", "without"]
    assert list(aligned["scenario"]) == [
        "with-information-sharing",
        "without information sharing",
    ]
