from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from scripts.run_full_paper_experiments import _build_error_row
from scripts.run_strict_protocol_baseline import build_mode_expected_contract
from src.constants import SCHEMA_FAMILY_D1_D3
from src.protocols.experiment_protocol import (
    FORMAL_METHODS,
    ProtocolViolation,
    formal_target_entity_keys,
    get_experiment_protocol,
    serialize_canonical_target_key,
    validate_canonical_target_key,
)
from src.protocols.runner_adapter import configure_protocol_frames
from src.utils.result_acceptance import (
    FORMAL_KEY_COLUMNS,
    accept_cell_csv,
    build_formal_cell_contract,
)
from test_strict_result_contract import _strict_row


def _daily_rows(
    first: object,
    second: object | None,
    *,
    periods: int = 35,
    start: str = "2020-01-01",
) -> pd.DataFrame:
    payload: dict[str, object] = {
        "date": pd.date_range(start, periods=periods, freq="D"),
        "sales": np.ones(periods),
    }
    if second is None:
        payload["store_id"] = first
    else:
        payload["store_id"] = first
        payload["item_id"] = second
    return pd.DataFrame(payload)


def _d1_source() -> pd.DataFrame:
    return pd.concat(
        [
            _daily_rows(store, item, periods=30, start="2017-06-01")
            for store in range(1, 4)
            for item in range(1, 10)
        ],
        ignore_index=True,
    )


def test_d1_static_and_runtime_target_are_the_same_canonical_key() -> None:
    protocol = get_experiment_protocol("D1")
    assert protocol.source_pool_rule.target_key == ("1", "10")

    _, target = configure_protocol_frames(
        _d1_source(),
        _daily_rows(1, 10, start="2017-06-01"),
        dataset_id="D1",
        scenario="with",
        group_cols=("store_id", "item_id"),
        observed_start="2017-06-01",
    )

    assert target.attrs["protocol_target_key"] == ("1", "10")
    assert serialize_canonical_target_key("D1", target.attrs["protocol_target_key"]) == "1/10"


def test_display_label_is_separate_from_formal_identity_columns() -> None:
    protocol = get_experiment_protocol("D1")

    assert protocol.target_display_label == "Store1/Item10"
    assert protocol.target_display_label != serialize_canonical_target_key(
        "D1", protocol.source_pool_rule.target_key
    )
    assert "target_display_label" not in FORMAL_KEY_COLUMNS


def test_d1_cell_expected_keys_equal_the_six_success_method_keys(tmp_path) -> None:
    rows = []
    for method in FORMAL_METHODS:
        row = _strict_row(horizon=1, seed=42)
        row.update(
            {
                "dataset_id": "D1",
                "target_entity_key": "1/10",
                "scenario": "without",
                "information_sharing": "without",
                "method": method,
                "schema_family": SCHEMA_FAMILY_D1_D3,
                "result_status": "trial",
                "error": "",
            }
        )
        rows.append(row)
    path = tmp_path / "dataset1_without_results.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    expected = build_formal_cell_contract(
        dataset_id=1,
        mode="without",
        targets=formal_target_entity_keys("D1"),
        horizon=1,
        seed=42,
    )

    outcome = accept_cell_csv(path, expected=expected)

    assert outcome.report.passed
    actual = set(
        zip(
            outcome.accepted_rows["target_entity_key"],
            outcome.accepted_rows["method"],
        )
    )
    assert actual == {("1/10", method) for method in FORMAL_METHODS}


def test_d2_real_key_is_canonical_and_brand_item_aliases_are_rejected() -> None:
    assert get_experiment_protocol("D2").source_pool_rule.target_key == ("1", "10")
    assert validate_canonical_target_key("D2", (1, 10)) == ("1", "10")

    with pytest.raises(ProtocolViolation, match="runtime canonical target key"):
        validate_canonical_target_key("D2", ("Brand1", "Item10"))
    with pytest.raises(ProtocolViolation, match="runtime canonical target key"):
        validate_canonical_target_key("D2", ("B1", "10"))


@pytest.mark.parametrize(
    ("dataset_id", "key", "serialized"),
    [
        ("D3", ("10",), "10"),
        ("D4", ("166", "258"), "166/258"),
        ("D5", ("48", "364606"), "48/364606"),
        ("D6", ("CA_1", "FOODS_3_586"), "CA_1/FOODS_3_586"),
    ],
)
def test_d3_d6_canonical_key_arity_and_serialization(
    dataset_id: str, key: tuple[str, ...], serialized: str
) -> None:
    assert validate_canonical_target_key(dataset_id, key) == key
    assert serialize_canonical_target_key(dataset_id, key) == serialized
    with pytest.raises(ProtocolViolation, match="arity"):
        validate_canonical_target_key(dataset_id, key + ("extra",))


@pytest.mark.parametrize(
    ("dataset_id", "group_cols"),
    [
        ("D1", ("store_id", "item_id")),
        ("D2", ("brand_id", "item_id")),
        ("D3", ("store_id",)),
        ("D4", ("store_id", "product_id")),
        ("D5", ("store_nbr", "item_nbr")),
        ("D6", ("store_id", "item_id")),
    ],
)
def test_static_protocol_names_the_real_canonical_group_columns(
    dataset_id: str,
    group_cols: tuple[str, ...],
) -> None:
    assert get_experiment_protocol(dataset_id).source_pool_rule.key_fields == group_cols


def test_formal_group_column_mismatch_fails_before_candidate_or_training_work() -> None:
    source = _daily_rows(166, 259, periods=30).rename(
        columns={"item_id": "alias_product"}
    )
    target = _daily_rows(166, 258).rename(columns={"item_id": "alias_product"})

    with patch(
        "src.protocols.runner_adapter._extended_candidates"
    ) as candidate_builder:
        with pytest.raises(ProtocolViolation, match="canonical group columns"):
            configure_protocol_frames(
                source,
                target,
                dataset_id="D4",
                scenario="without",
                group_cols=("store_id", "alias_product"),
                observed_start="2020-01-01",
                enforce_formal_target=True,
            )

    candidate_builder.assert_not_called()


def test_static_runtime_mismatch_fails_before_candidate_or_training_work() -> None:
    with patch(
        "src.protocols.runner_adapter._strict_raw_candidates"
    ) as candidate_builder:
        with pytest.raises(ProtocolViolation, match="runtime canonical target key"):
            configure_protocol_frames(
                _d1_source(),
                _daily_rows("Store1", "Item10", start="2017-06-01"),
                dataset_id="D1",
                scenario="with",
                group_cols=("store_id", "item_id"),
                observed_start="2017-06-01",
            )

    candidate_builder.assert_not_called()


def test_success_and_error_rows_share_the_same_canonical_target_key() -> None:
    runtime_success_key = serialize_canonical_target_key(
        "D1", validate_canonical_target_key("D1", (1, 10))
    )
    error = _build_error_row(
        dataset_name="Dataset1",
        method_name="MSWA-TL",
        source_count=3,
        information_sharing_scenario="without",
        protocol={},
        strict_paper_mode=True,
        exc=RuntimeError("boom"),
    )

    assert runtime_success_key == error["target_entity_key"] == "1/10"


@pytest.mark.parametrize(
    ("dataset", "canonical_targets"),
    [
        ("d1", ("1/10",)),
        ("d2", ("1/10",)),
        ("d3", ("10",)),
        ("d4", ("166/258", "166/432", "166/433", "166/313", "166/311")),
        (
            "d5",
            (
                "48/364606",
                "48/1159415",
                "48/1159414",
                "48/1349808",
                "48/320682",
            ),
        ),
        (
            "d6",
            (
                "CA_1/FOODS_3_586",
                "CA_1/FOODS_3_080",
                "CA_1/FOODS_3_555",
                "CA_1/FOODS_3_377",
                "CA_1/FOODS_3_668",
            ),
        ),
    ],
)
def test_acceptance_expected_targets_use_static_canonical_serialization(
    dataset: str,
    canonical_targets: tuple[str, ...],
) -> None:
    contract = build_mode_expected_contract(dataset=dataset, scenario="without")
    dataset_id = int(dataset[1:])

    assert formal_target_entity_keys(dataset) == canonical_targets
    assert contract.targets_by_dataset_mode[(dataset_id, "without")] == canonical_targets
    assert all(
        "Store" not in key and "Brand" not in key and "Item" not in key
        for key in canonical_targets
    )
