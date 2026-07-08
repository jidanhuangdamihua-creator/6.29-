from __future__ import annotations

import pandas as pd
import pytest

from src.utils.source_domain_filter import apply_source_domain_policy, normalize_domain_filter


def test_without_domain_filter_supports_equal_list_and_multi_field_and() -> None:
    source_df = pd.DataFrame(
        {
            "entity_id": ["a", "b", "c", "d"],
            "family": ["GROCERY I", "BEVERAGES", "GROCERY I", "GROCERY I"],
            "dept_id": ["FOODS_1", "FOODS_2", "FOODS_3", "FOODS_3"],
            "second_category_id": [10, 20, 20, 30],
        }
    )

    equal = apply_source_domain_policy(
        source_df,
        {"family": "GROCERY I"},
        information_sharing="without",
    )
    assert equal.frame["entity_id"].tolist() == ["a", "c", "d"]
    assert equal.diagnostics["source_domain_filter_applied"] is True
    assert equal.diagnostics["source_pool_size_before_filter"] == 4
    assert equal.diagnostics["source_pool_size_after_filter"] == 3

    list_filter = apply_source_domain_policy(
        source_df,
        {"dept_id": ["FOODS_2", "FOODS_3"]},
        information_sharing="without",
    )
    assert list_filter.frame["entity_id"].tolist() == ["b", "c", "d"]

    multi = apply_source_domain_policy(
        source_df,
        {"dept_id": ["FOODS_3"], "second_category_id": 20},
        information_sharing="without",
    )
    assert multi.frame["entity_id"].tolist() == ["c"]


def test_without_domain_filter_accepts_legacy_column_value_shape() -> None:
    normalized = normalize_domain_filter({"column": "second_category_id", "value": 20})

    assert normalized == {"second_category_id": 20}


def test_without_domain_filter_missing_field_fails_fast() -> None:
    source_df = pd.DataFrame({"entity_id": ["a"], "family": ["GROCERY I"]})

    with pytest.raises(ValueError, match="source_domain_filter missing columns"):
        apply_source_domain_policy(
            source_df,
            {"dept_id": "FOODS_3"},
            information_sharing="without",
        )


def test_with_domain_filter_records_metadata_without_filtering_or_missing_field_error() -> None:
    source_df = pd.DataFrame({"entity_id": ["a", "b"], "family": ["A", "B"]})

    result = apply_source_domain_policy(
        source_df,
        {"dept_id": "FOODS_3"},
        information_sharing="with",
    )

    assert result.frame["entity_id"].tolist() == ["a", "b"]
    assert result.diagnostics["knn_json_domain_filter"] == {"dept_id": "FOODS_3"}
    assert result.diagnostics["source_domain_filter"] is None
    assert result.diagnostics["source_domain_filter_applied"] is False
    assert result.diagnostics["source_pool_size_before_filter"] == 2
    assert result.diagnostics["source_pool_size_after_filter"] == 2
    assert result.diagnostics["source_domain_filter_reason"] == "with_information_sharing_all_source_pool"
    assert result.diagnostics["source_domain_filter_error"] == ""
