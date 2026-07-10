from __future__ import annotations

import json

from src.constants import RESULT_SCHEMA_COLUMNS
from src.transfer_methods.source_failure_tolerance import (
    AllSourcesFailedError,
    error_row_from_all_sources_failed,
    runtime_selection_meta,
)
from src.utils.entity_experiment import _row_from_result


RUNTIME_TOP_K = [
    {
        "source_rank": 1,
        "source_key": ("runtime-source", "item"),
        "distance": 0.25,
        "weight": 1.0,
    }
]


def _selection_result() -> dict[str, object]:
    return {
        "sources": list(RUNTIME_TOP_K),
        "meta": {
            "selection_authority": "runtime",
            "protocol_version": "runtime_knn_windowed_stats_v1",
            "target_observed_start": "2024-01-01",
            "target_observed_end": "2024-01-30",
            "source_history_start": "2023-04-06",
            "source_history_end": "2024-01-30",
            "target_test_excluded": True,
            "source_future_excluded": True,
            "source_alignment_mode": "exact_target_observed_dates",
            "feature_cols": ["sales", "promo"],
            "representation": "mean_std_min_max_last",
            "scaling": "none",
            "scaler_fit_scope": "not_applicable",
            "selected_sources_runtime": list(RUNTIME_TOP_K),
            "candidate_pool_digest": "candidate-sha256",
            "selection_result_digest": "result-sha256",
            "source_skip_diagnostics": [
                {
                    "source_key": ("incomplete", "item"),
                    "reason": "missing_target_observed_dates",
                    "missing_dates": ["2024-01-30"],
                }
            ],
        },
    }


def test_runtime_selection_meta_extracts_only_runtime_authority_fields() -> None:
    selection = _selection_result()

    extracted = runtime_selection_meta(selection)

    assert extracted["selection_authority"] == "runtime"
    assert extracted["selected_sources_runtime"] == RUNTIME_TOP_K
    assert extracted["candidate_pool_digest"] == "candidate-sha256"
    assert "requested_feature_cols" not in extracted


def test_d4_d6_result_row_records_runtime_top_k_not_json_top_k() -> None:
    runtime_meta = runtime_selection_meta(_selection_result())
    json_top_k = [{"source_key": ("json-source", "item"), "distance": 0.0}]
    raw = {
        "rmse": 1.0,
        "accuracy": 0.5,
        "mae": 0.5,
        "mape": 1.0,
        "smape": 2.0,
        "prediction_shape": (1, 1),
        "meta": {
            "selected_sources": list(RUNTIME_TOP_K),
            "requested_k": 1,
            "effective_k": 1,
            "selected_source_count": 1,
            "valid_source_count": 1,
            **runtime_meta,
        },
    }

    row = _row_from_result(
        raw,
        method="MSWA-TL",
        entity_key="target-a",
        config={
            "dataset_id": 4,
            "dataset_name": "Dataset4",
            "info_sharing": "without",
            "source_count": 1,
            "knn_json_selected_sources": json_top_k,
        },
        elapsed=0.1,
    )

    assert row["selection_authority"] == "runtime"
    assert row["protocol_version"] == "runtime_knn_windowed_stats_v1"
    assert json.loads(row["selected_sources_runtime"]) == json.loads(
        json.dumps(RUNTIME_TOP_K)
    )
    assert json.loads(row["selected_sources"]) == json.loads(json.dumps(RUNTIME_TOP_K))
    assert json.loads(row["selected_sources_runtime"]) != json.loads(json.dumps(json_top_k))
    assert row["candidate_pool_digest"] == "candidate-sha256"
    assert row["selection_result_digest"] == "result-sha256"
    assert row["target_test_excluded"] is True
    assert row["source_future_excluded"] is True


def test_runtime_authority_fields_are_part_of_result_schema() -> None:
    for field in (
        "selection_authority",
        "protocol_version",
        "target_observed_start",
        "target_observed_end",
        "source_history_start",
        "source_history_end",
        "target_test_excluded",
        "source_future_excluded",
        "source_alignment_mode",
        "representation",
        "scaling",
        "scaler_fit_scope",
        "selected_sources_runtime",
        "candidate_pool_digest",
        "selection_result_digest",
        "source_skip_diagnostics",
    ):
        assert field in RESULT_SCHEMA_COLUMNS


def test_all_sources_failed_row_preserves_runtime_selection_metadata() -> None:
    runtime_meta = runtime_selection_meta(_selection_result())
    exc = AllSourcesFailedError(
        "MSWA-TL",
        [
            {
                "failed_source_key": ("runtime-source", "item"),
                "exception_type": "RuntimeError",
                "exception_message": "training failed",
            }
        ],
        selected_sources=RUNTIME_TOP_K,
        selection_meta=runtime_meta,
    )

    raw = error_row_from_all_sources_failed(exc, requested_k=1, elapsed=0.1)

    assert raw["meta"]["selection_authority"] == "runtime"
    assert raw["meta"]["selected_sources_runtime"] == RUNTIME_TOP_K
