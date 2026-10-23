from datetime import date

import pytest

from src.constants import SOLIDIFIED_TARGET_WINDOWS, SOURCE_HISTORY_DAYS
from src.protocols.experiment_protocol import ProtocolViolation
from src.protocols.sealing_protocol import (
    SEALED_PROTOCOL_VERSION,
    SOURCE_PRETRAIN_DAYS,
    classify_feature_roles,
    get_source_pretrain_window,
    get_target_window,
)


def test_all_target_windows_are_exact_30_plus_180_natural_days():
    expected = {
        1: ("2017-06-01", "2017-12-27"),
        2: ("2018-06-01", "2018-12-27"),
        3: ("2015-01-03", "2015-07-31"),
        4: ("2024-12-16", "2025-07-13"),
        5: ("2017-01-17", "2017-08-14"),
        6: ("2015-10-26", "2016-05-22"),
    }
    for dataset_id, (start, end) in expected.items():
        window = get_target_window(dataset_id)
        assert window.start == date.fromisoformat(start)
        assert window.observed_days == 30
        assert window.test_days == 180
        assert window.end == date.fromisoformat(end)
        assert (window.end - window.start).days == 209


def test_source_pretrain_is_180_days_ending_at_observed_end():
    assert SOURCE_PRETRAIN_DAYS == 180
    window = get_source_pretrain_window(2)
    assert window.end.isoformat() == "2018-06-30"
    assert window.start.isoformat() == "2018-01-02"
    assert (window.end - window.start).days == 179


@pytest.mark.parametrize("dataset_id", range(1, 7))
def test_every_source_pretrain_window_is_exactly_180_natural_days(dataset_id):
    target = get_target_window(dataset_id)
    source = get_source_pretrain_window(dataset_id)

    assert source.dataset_id == dataset_id
    assert source.days == 180
    assert source.end == target.observed_end
    assert (source.end - source.start).days == 179


def test_feature_roles_are_ordered_and_fail_closed_for_blind_rows():
    roles = classify_feature_roles(
        2,
        [" item_id ", "sales", "promo", "month", "customers"],
        ["month", "promo", "customers", "sales"],
    )

    assert roles.knn_observed == ("sales", "promo", "month", "customers")
    assert roles.model_historical == ("sales", "promo", "month", "customers")
    assert roles.future_known == ("month",)
    assert roles.evaluation_only == ("sales", "promo", "customers")


def test_d3_roles_keep_identifiers_out_and_separate_future_truth():
    roles = classify_feature_roles(
        3,
        [
            "store_id",
            "sales",
            "customers",
            "open",
            "promo",
            "state_holiday",
            "school_holiday",
            "month",
        ],
        [
            "store_id",
            "sales",
            "customers",
            "open",
            "promo",
            "state_holiday",
            "school_holiday",
            "month",
        ],
    )

    assert "store_id" not in roles.knn_observed
    assert "store_id" not in roles.model_historical
    assert "store_id" not in roles.future_known
    assert "store_id" not in roles.evaluation_only
    assert roles.future_known == (
        "open",
        "promo",
        "state_holiday",
        "school_holiday",
        "month",
    )
    assert roles.evaluation_only == ("sales", "customers")


def test_declared_blind_calendar_column_can_be_added_after_observed_columns():
    roles = classify_feature_roles(1, ["sales", "year"], ["month"])

    assert roles.knn_observed == ("sales", "year")
    assert roles.model_historical == ("sales", "year")
    assert roles.future_known == ("year", "month")
    assert roles.evaluation_only == ("sales",)


def test_unclassified_new_blind_feature_raises_protocol_violation():
    with pytest.raises(ProtocolViolation, match="^unclassified blind feature: customers$"):
        classify_feature_roles(3, ["sales", "month"], ["customers"])


@pytest.mark.parametrize(
    ("dataset_id", "observed", "blind"),
    [
        (0, ["sales"], []),
        (7, ["sales"], []),
        (1, [""], []),
        (1, [" sales", "sales "], []),
        (1, ["sales"], [" month", "month "]),
    ],
)
def test_invalid_dataset_or_column_declarations_raise_protocol_violation(
    dataset_id, observed, blind
):
    with pytest.raises(ProtocolViolation):
        classify_feature_roles(dataset_id, observed, blind)


@pytest.mark.parametrize("dataset_id", [True, 1.0, "1", None])
@pytest.mark.parametrize(
    "accessor",
    [
        get_target_window,
        get_source_pretrain_window,
        lambda dataset_id: classify_feature_roles(dataset_id, ["sales"]),
    ],
)
def test_dataset_id_is_a_strict_integer(accessor, dataset_id):
    with pytest.raises(ProtocolViolation, match="unsupported sealed dataset_id"):
        accessor(dataset_id)


def test_sealed_protocol_version_is_frozen():
    assert SEALED_PROTOCOL_VERSION == "d1_d6_sealed_blind_v1"


def test_legacy_source_window_alias_and_d5_authority_follow_sealed_protocol():
    assert SOURCE_HISTORY_DAYS == SOURCE_PRETRAIN_DAYS == 180
    assert SOLIDIFIED_TARGET_WINDOWS[5] == {
        "train_start": "2017-01-17",
        "test_end": "2017-08-14",
    }
