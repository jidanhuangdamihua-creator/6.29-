from __future__ import annotations

import pandas as pd
import pytest

from scripts.regenerate_solidified_knn import _filter_source_for_scenario
from src.utils.d4_d6_runtime import apply_runtime_source_domain_policy
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
    assert equal.diagnostics["source_pool_rows_before_filter"] == 4
    assert equal.diagnostics["source_pool_rows_after_filter"] == 3
    assert equal.diagnostics["excluded_source_row_count"] == 1
    assert equal.diagnostics["source_pool_entities_before_filter"] == 4
    assert equal.diagnostics["source_pool_entities_after_filter"] == 3
    assert equal.diagnostics["excluded_source_entity_count"] == 1

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
    assert result.diagnostics["source_pool_rows_before_filter"] == 2
    assert result.diagnostics["source_pool_rows_after_filter"] == 2
    assert result.diagnostics["excluded_source_row_count"] == 0
    assert result.diagnostics["source_pool_entities_before_filter"] == 2
    assert result.diagnostics["source_pool_entities_after_filter"] == 2
    assert result.diagnostics["excluded_source_entity_count"] == 0
    assert result.diagnostics["source_domain_filter_reason"] == "with_information_sharing_all_source_pool"
    assert result.diagnostics["source_domain_filter_error"] == ""


@pytest.mark.parametrize("dataset_id", [4, 5, 6])
@pytest.mark.parametrize(
    "domain_filter",
    [
        {"column": "dept_id", "value": "FOODS_3"},
        {"dept_id": ["FOODS_2", "FOODS_3"]},
        {"dept_id": "FOODS_3", "second_category_id": 20},
    ],
)
def test_regenerate_without_filter_matches_runtime_policy(
    dataset_id: int,
    domain_filter: dict[str, object],
) -> None:
    source_df = pd.DataFrame(
        {
            "entity_id": ["a", "b", "c", "d"],
            "dept_id": ["FOODS_1", "FOODS_2", "FOODS_3", "FOODS_3"],
            "second_category_id": [10, 20, 20, 30],
        }
    )
    source_df.attrs["protocol_version"] = "runtime_knn_windowed_stats_v1"

    expected = apply_source_domain_policy(
        source_df,
        domain_filter,
        information_sharing="without",
        entity_group_cols=("dept_id", "second_category_id"),
    ).frame
    actual = _filter_source_for_scenario(
        source_df,
        dataset_id=dataset_id,
        scenario="without",
        old_payload={"domain_filter": domain_filter},
    )

    assert actual.frame["entity_id"].tolist() == expected["entity_id"].tolist()
    assert actual.frame.attrs == expected.attrs
    assert actual.diagnostics == apply_source_domain_policy(
        source_df,
        domain_filter,
        information_sharing="without",
        entity_group_cols=("dept_id", "second_category_id"),
    ).diagnostics


@pytest.mark.parametrize(
    ("dataset_id", "group_cols", "domain_filter"),
    (
        (5, ["store_nbr", "item_nbr"], {"family": "GROCERY I"}),
        (6, ["store_id", "item_id"], {"dept_id": "FOODS_3"}),
    ),
)
@pytest.mark.parametrize("scenario", ("without", "with"))
def test_d5_d6_runtime_policy_diagnostics_track_rows_entities_and_config(
    dataset_id: int,
    group_cols: list[str],
    domain_filter: dict[str, str],
    scenario: str,
) -> None:
    source_df = pd.DataFrame(
        {
            "store_nbr": [1, 1, 2, 3],
            "item_nbr": [10, 10, 20, 30],
            "store_id": [1, 1, 2, 3],
            "item_id": [10, 10, 20, 30],
            "family": ["GROCERY I", "GROCERY I", "BEVERAGES", "GROCERY I"],
            "dept_id": ["FOODS_3", "FOODS_3", "FOODS_2", "FOODS_3"],
        }
    )
    config = {"info_sharing": scenario}

    result = apply_runtime_source_domain_policy(
        source_df,
        {"domain_filter": domain_filter, "group_cols": group_cols},
        config,
    )

    assert config["source_pool_rows_before_filter"] == 4
    assert config["source_pool_entities_before_filter"] == 3
    if scenario == "without":
        assert len(result) == 3
        assert config["source_domain_filter_applied"] is True
        assert config["source_domain_filter_reason"] == "without_information_sharing_same_domain_protocol"
        assert config["source_pool_rows_after_filter"] <= config["source_pool_rows_before_filter"]
        assert config["excluded_source_row_count"] == (
            config["source_pool_rows_before_filter"] - config["source_pool_rows_after_filter"]
        )
        assert config["source_pool_entities_after_filter"] <= config["source_pool_entities_before_filter"]
        assert config["excluded_source_entity_count"] == (
            config["source_pool_entities_before_filter"] - config["source_pool_entities_after_filter"]
        )
    else:
        assert len(result) == 4
        assert config["source_domain_filter_applied"] is False
        assert config["source_domain_filter_reason"] == "with_information_sharing_all_source_pool"
        assert config["source_pool_rows_after_filter"] == config["source_pool_rows_before_filter"]
        assert config["excluded_source_row_count"] == 0
        assert config["source_pool_entities_after_filter"] == config["source_pool_entities_before_filter"]
        assert config["excluded_source_entity_count"] == 0


def test_policy_leaves_entity_counts_null_without_a_reliable_entity_key() -> None:
    result = apply_source_domain_policy(
        pd.DataFrame({"family": ["A", "B"]}),
        {"family": "A"},
        information_sharing="without",
    )

    assert result.diagnostics["source_pool_entities_before_filter"] is None
    assert result.diagnostics["source_pool_entities_after_filter"] is None
    assert result.diagnostics["excluded_source_entity_count"] is None
