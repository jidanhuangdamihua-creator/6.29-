from __future__ import annotations

from datetime import timedelta

import pytest

from src.protocols.sealing_protocol import (
    FORMAL_HORIZONS,
    FORMAL_SEEDS,
    SEALING_PROTOCOL_VERSION,
    ProtocolViolation,
    get_source_pretrain_window,
    get_target_window,
)


EXPECTED_WINDOWS = {
    "D1": ("2017-06-01", "2017-06-30", "2017-12-27"),
    "D2": ("2018-06-01", "2018-06-30", "2018-12-27"),
    "D3": ("2015-01-03", "2015-02-01", "2015-07-31"),
    "D4": ("2024-12-16", "2025-01-14", "2025-07-13"),
    "D5": ("2017-01-17", "2017-02-15", "2017-08-14"),
    "D6": ("2015-10-26", "2015-11-24", "2016-05-22"),
}


@pytest.mark.parametrize("dataset_id,expected", EXPECTED_WINDOWS.items())
def test_exact_target_and_source_windows_are_frozen(dataset_id, expected) -> None:
    target = get_target_window(dataset_id)
    source = get_source_pretrain_window(dataset_id)

    assert SEALING_PROTOCOL_VERSION == "d1_d6_sealed_v1"
    assert FORMAL_HORIZONS == (1, 2, 3, 4, 5)
    assert FORMAL_SEEDS == (42, 43, 44, 45, 46)
    assert (
        target.target_start.isoformat(),
        target.observed_end.isoformat(),
        target.blind_end.isoformat(),
    ) == expected
    assert target.train_days == 15
    assert target.validation_days == 15
    assert target.blind_days == 180
    assert target.validation_start == target.train_end + timedelta(days=1)
    assert target.blind_start == target.validation_end + timedelta(days=1)

    assert source.pretrain_days == 180
    assert source.knn_days == 30
    assert source.pretrain_end == target.observed_end
    assert source.knn_end == source.pretrain_end
    assert source.knn_start == source.pretrain_end - timedelta(days=29)
    assert source.pretrain_start == source.pretrain_end - timedelta(days=179)


def test_dataset_aliases_are_closed_and_normalized() -> None:
    assert get_target_window("dataset4") is get_target_window("D4")
    assert get_target_window(4) is get_target_window("D4")
    with pytest.raises(ProtocolViolation, match="unsupported dataset"):
        get_target_window("D7")
