from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.constants import SOURCE_HISTORY_DAYS
import src.protocols.source_history as source_history_module
from src.protocols.source_history import (
    build_exact_source_history_candidate_frame,
    expected_source_history_dates,
    source_history_frame_digest,
)


ORIGIN = pd.Timestamp("2025-01-14")
EXPECTED = expected_source_history_dates(ORIGIN, source_history_days=180)
KEY_FIELDS = ("store_id", "product_id")


def _frame(key: tuple[int, int], dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "store_id": key[0],
            "product_id": key[1],
            "date": dates,
            "sales": np.ones(len(dates), dtype=float),
        }
    )


def test_frozen_d4_d6_source_history_is_180_gregorian_days() -> None:
    assert SOURCE_HISTORY_DAYS == 180
    assert len(EXPECTED) == 180
    assert EXPECTED[0] == ORIGIN - pd.Timedelta(days=179)
    assert EXPECTED[-1] == ORIGIN


def test_exact_date_set_eligibility_excludes_window_outside_rows() -> None:
    source = _frame((1, 1), EXPECTED).pipe(
        lambda frame: pd.concat(
            [frame, _frame((1, 1), pd.DatetimeIndex([ORIGIN + pd.Timedelta(days=1)]))],
            ignore_index=True,
        )
    )

    result = build_exact_source_history_candidate_frame(
        source,
        key_fields=KEY_FIELDS,
        origin=ORIGIN,
        source_history_days=180,
    )

    assert result.eligible_keys == (("1", "1"),)
    candidate = result.candidate_frame
    assert len(candidate) == 180
    assert pd.DatetimeIndex(candidate["date"]).equals(EXPECTED)
    assert candidate["date"].max() == ORIGIN


def test_prevalidated_calendarized_source_uses_bounded_exact_check() -> None:
    source = _frame((1, 1), EXPECTED)
    source.attrs["source_history_prevalidated_exact"] = True

    result = build_exact_source_history_candidate_frame(
        source,
        key_fields=KEY_FIELDS,
        origin=ORIGIN,
        source_history_days=180,
    )

    assert result.eligible_keys == (("1", "1"),)
    assert result.candidate_frame.attrs["source_history_validation_path"] == "prevalidated_calendarized"


def test_source_history_digest_hashes_large_canonical_frames_one_column_at_a_time(monkeypatch) -> None:
    source = _frame((1, 1), EXPECTED[:5])
    source.attrs["source_history_canonical_order"] = True
    calls: list[int] = []
    real_hash = pd.util.hash_pandas_object

    def tracked_hash(frame, *args, **kwargs):
        calls.append(len(frame))
        return real_hash(frame, *args, **kwargs)

    monkeypatch.setattr(source_history_module, "SOURCE_HISTORY_DIGEST_CHUNK_ROWS", 2)
    monkeypatch.setattr(source_history_module, "SOURCE_HISTORY_DIGEST_SMALL_FRAME_ROWS", 0)
    monkeypatch.setattr(source_history_module, "SOURCE_HISTORY_DIGEST_LARGE_FRAME_ROWS", 4)
    monkeypatch.setattr(pd.util, "hash_pandas_object", tracked_hash)

    source_history_frame_digest(source, key_fields=KEY_FIELDS)

    assert calls == [len(source)] * len(source.columns)


def test_source_history_digest_hashes_small_canonical_frames_in_bounded_column_chunks(monkeypatch) -> None:
    source = _frame((1, 1), EXPECTED[:5])
    source.attrs["source_history_canonical_order"] = True
    calls: list[int] = []
    real_hash = pd.util.hash_pandas_object

    def tracked_hash(frame, *args, **kwargs):
        calls.append(len(frame))
        return real_hash(frame, *args, **kwargs)

    monkeypatch.setattr(source_history_module, "SOURCE_HISTORY_DIGEST_CHUNK_ROWS", 2)
    monkeypatch.setattr(source_history_module, "SOURCE_HISTORY_DIGEST_SMALL_FRAME_ROWS", 0)
    monkeypatch.setattr(source_history_module, "SOURCE_HISTORY_DIGEST_LARGE_FRAME_ROWS", 100)
    monkeypatch.setattr(pd.util, "hash_pandas_object", tracked_hash)

    source_history_frame_digest(source, key_fields=KEY_FIELDS)

    assert calls == [size for _column in source.columns for size in (2, 2, 1)]


def test_source_history_digest_uses_stable_json_for_small_canonical_frames(monkeypatch) -> None:
    source = _frame((1, 1), EXPECTED[:5])
    source.attrs["source_history_canonical_order"] = True
    hash_calls: list[int] = []
    json_calls: list[int] = []
    real_hash = pd.util.hash_pandas_object
    real_to_json = pd.DataFrame.to_json

    def tracked_hash(frame, *args, **kwargs):
        hash_calls.append(len(frame))
        return real_hash(frame, *args, **kwargs)

    def tracked_to_json(frame, *args, **kwargs):
        json_calls.append(len(frame))
        return real_to_json(frame, *args, **kwargs)

    monkeypatch.setattr(source_history_module, "SOURCE_HISTORY_DIGEST_SMALL_FRAME_ROWS", 10)
    monkeypatch.setattr(pd.util, "hash_pandas_object", tracked_hash)
    monkeypatch.setattr(pd.DataFrame, "to_json", tracked_to_json)

    source_history_frame_digest(source, key_fields=KEY_FIELDS)

    assert hash_calls == []
    assert json_calls == [len(source)]


@pytest.mark.parametrize(
    "dates, extra_rows",
    [
        (EXPECTED[:-1], 0),  # 179 legal dates
        (EXPECTED[:-1].append(pd.DatetimeIndex([EXPECTED[0]])), 0),  # duplicate, one missing
        (EXPECTED[:-1].append(pd.DatetimeIndex([ORIGIN + pd.Timedelta(days=1)])), 0),  # missing + outside
        (EXPECTED.delete(40), 0),  # an interior Gregorian gap
        (EXPECTED[:82], 0),  # the D4 729_424 fact
    ],
)
def test_incomplete_or_wrong_date_sets_are_not_eligible(
    dates: pd.DatetimeIndex, extra_rows: int
) -> None:
    del extra_rows
    result = build_exact_source_history_candidate_frame(
        _frame((1, 1), dates),
        key_fields=KEY_FIELDS,
        origin=ORIGIN,
        source_history_days=180,
    )

    assert result.eligible_keys == ()
    assert result.candidate_frame.empty


def test_duplicate_entity_date_is_recorded_and_rejected_even_with_all_dates() -> None:
    source = pd.concat(
        [
            _frame((1, 1), EXPECTED),
            _frame((1, 1), pd.DatetimeIndex([EXPECTED[0]])),
        ],
        ignore_index=True,
    )

    result = build_exact_source_history_candidate_frame(
        source,
        key_fields=KEY_FIELDS,
        origin=ORIGIN,
        source_history_days=180,
    )

    assert result.eligible_keys == ()
    assert result.duplicate_keys == (("1", "1"),)
