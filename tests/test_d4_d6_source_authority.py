from __future__ import annotations

from src.constants import RESULT_SCHEMA_COLUMNS
from src.transfer_methods.source_failure_tolerance import runtime_selection_meta


def _shared_selection() -> dict[str, object]:
    sources = [
        {
            "source_rank": 1,
            "source_key": ("S1", "I1"),
            "distance": 0.25,
            "weight": 1.0,
            "tie_group": 1,
        }
    ]
    return {
        "sources": sources,
        "meta": {
            "selection_authority": "shared_protocol",
            "protocol_track": "extended",
            "protocol_version": "d1_d6_protocol_v1",
            "knn_observed_start": "2024-01-01",
            "knn_observed_end": "2024-01-30",
            "source_observation_cutoff": "2024-01-30",
            "target_test_excluded": True,
            "source_future_excluded": True,
            "source_alignment_mode": "exact_knn_observed_dates",
            "feature_cols": ["sales"],
            "representation": "daily_sales_flattened_30d",
            "scaling": "global_minmax_legal_observed_values",
            "scaler_fit_scope": "target_and_candidate_legal_observed_values",
            "selected_sources_runtime": sources,
            "candidate_pool_digest": "a" * 64,
            "candidate_pool_digest_input": {"dataset_id": "D4"},
            "selection_result_digest": "b" * 64,
            "source_skip_diagnostics": [],
        },
    }


def test_shared_selection_metadata_is_preserved_for_transfer_methods() -> None:
    extracted = runtime_selection_meta(_shared_selection())
    assert extracted["selection_authority"] == "shared_protocol"
    assert extracted["protocol_version"] == "d1_d6_protocol_v1"
    assert extracted["representation"] == "daily_sales_flattened_30d"
    assert extracted["candidate_pool_digest_input"] == {"dataset_id": "D4"}


def test_shared_protocol_audit_fields_are_in_result_schema() -> None:
    for field in (
        "protocol_track",
        "protocol_version",
        "knn_observed_start",
        "knn_observed_end",
        "knn_representation",
        "source_observation_cutoff",
        "target_test_excluded",
        "source_future_excluded",
        "candidate_pool_digest",
        "candidate_pool_digest_input",
        "selection_result_digest",
        "horizon",
        "seed",
        "primary_metric_space",
        "sample_manifest_digest",
    ):
        assert field in RESULT_SCHEMA_COLUMNS
