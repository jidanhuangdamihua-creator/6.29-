"""Parity checks for D4-D6 regenerated KNN payload schema and diagnostics."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from scripts.regenerate_solidified_knn import (
    _build_regenerated_payload,
    _filter_source_for_scenario,
)
from src.constants import D4_D6_RUNTIME_KNN_PROTOCOL_VERSION


DATASET_CASES = {
    4: {
        "group_cols": ["store_id", "product_id"],
        "domain_filter": {"second_category_id": 20},
        "domain_column": "second_category_id",
        "domain_values": [20, 20, 30, 20],
    },
    5: {
        "group_cols": ["store_nbr", "item_nbr"],
        "domain_filter": {"family": "GROCERY I"},
        "domain_column": "family",
        "domain_values": ["GROCERY I", "GROCERY I", "BEVERAGES", "GROCERY I"],
    },
    6: {
        "group_cols": ["store_id", "item_id"],
        "domain_filter": {"dept_id": "FOODS_3"},
        "domain_column": "dept_id",
        "domain_values": ["FOODS_3", "FOODS_3", "FOODS_2", "FOODS_3"],
    },
}

EXPECTED_REGENERATED_PAYLOAD_FIELDS = {
    "dataset_id",
    "dataset",
    "info_sharing",
    "k",
    "window_size",
    "horizon",
    "target_train_window",
    "domain_filter",
    "group_cols",
    "selection_authority",
    "protocol_version",
    "results_semantics",
    "training_selection_authority",
    "json_results_used_for",
    "feature_cols",
    "feature_info",
    "source_pool_size",
    "source_domain_policy_diagnostics",
    "results",
    "selection_metadata",
}

def _source_df(dataset_id: int) -> pd.DataFrame:
    case = DATASET_CASES[dataset_id]
    first_group_col, second_group_col = case["group_cols"]
    return pd.DataFrame(
        {
            first_group_col: [1, 1, 2, 3],
            second_group_col: [10, 10, 20, 30],
            case["domain_column"]: case["domain_values"],
        }
    )


def _old_payload(dataset_id: int, scenario: str) -> dict[str, Any]:
    case = DATASET_CASES[dataset_id]
    return {
        "dataset_id": dataset_id,
        "dataset": f"D{dataset_id}",
        "info_sharing": scenario,
        "k": 1,
        "window_size": 30,
        "horizon": 1,
        "target_train_window": {"start": "2024-01-01", "end": "2024-01-30"},
        "domain_filter": case["domain_filter"],
        "group_cols": case["group_cols"],
    }


def _runtime_selection_metadata() -> dict[str, dict[str, Any]]:
    return {
        "target": {
            "selection_authority": "runtime",
            "protocol_version": D4_D6_RUNTIME_KNN_PROTOCOL_VERSION,
        }
    }


@pytest.mark.parametrize("scenario", ("without", "with"))
def test_d4_d6_regenerated_payload_schema_group_cols_and_domain_diagnostics_parity(
    scenario: str,
) -> None:
    regenerated_payloads = []
    for dataset_id in (4, 5, 6):
        old_payload = _old_payload(dataset_id, scenario)
        source_policy = _filter_source_for_scenario(
            _source_df(dataset_id),
            dataset_id=dataset_id,
            scenario=scenario,
            old_payload=old_payload,
        )
        regenerated = _build_regenerated_payload(
            old_payload=old_payload,
            feature_cols=["sales"],
            feature_info={"selected_features": ["sales"]},
            source_pool_size=len(source_policy.frame),
            source_domain_policy_diagnostics=source_policy.diagnostics,
            results={"target": []},
            selection_metadata=_runtime_selection_metadata(),
        )
        diagnostics = regenerated["source_domain_policy_diagnostics"]

        assert set(regenerated) == EXPECTED_REGENERATED_PAYLOAD_FIELDS
        assert regenerated["group_cols"] == old_payload["group_cols"]
        assert regenerated["group_cols"] is not old_payload["group_cols"]
        assert diagnostics == source_policy.diagnostics
        assert diagnostics is not source_policy.diagnostics
        assert not any("skip" in key or "missing" in key for key in diagnostics)
        assert regenerated["source_pool_size"] == diagnostics["source_pool_rows_after_filter"]

        entities_before = diagnostics["source_pool_entities_before_filter"]
        entities_after = diagnostics["source_pool_entities_after_filter"]
        if scenario == "with":
            assert diagnostics["source_domain_filter_applied"] is False
            assert diagnostics["source_pool_rows_before_filter"] == diagnostics[
                "source_pool_rows_after_filter"
            ]
            assert diagnostics["excluded_source_row_count"] == 0
            assert entities_before == entities_after
            assert diagnostics["excluded_source_entity_count"] in (0, None)
        else:
            assert diagnostics["source_domain_filter_applied"] is True
            assert diagnostics["source_pool_rows_after_filter"] <= diagnostics[
                "source_pool_rows_before_filter"
            ]
            assert diagnostics["excluded_source_row_count"] == (
                diagnostics["source_pool_rows_before_filter"]
                - diagnostics["source_pool_rows_after_filter"]
            )
            if entities_before is not None and entities_after is not None:
                assert diagnostics["excluded_source_entity_count"] == (
                    entities_before - entities_after
                )

        regenerated_payloads.append(regenerated)

    assert {frozenset(payload) for payload in regenerated_payloads} == {
        frozenset(EXPECTED_REGENERATED_PAYLOAD_FIELDS)
    }
