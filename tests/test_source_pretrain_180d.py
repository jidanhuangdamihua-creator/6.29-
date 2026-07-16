from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.protocols.candidate_pool import select_daily_sequence_sources
from src.protocols.experiment_protocol import ProtocolViolation, get_experiment_protocol


SOURCE_DATES = pd.date_range("2019-07-05", periods=180, freq="D")
KNN_DATES = SOURCE_DATES[-30:]


def _source(key: tuple[str, str], offset: float) -> pd.DataFrame:
    values = np.arange(180, dtype=np.float64) + offset
    return pd.DataFrame(
        {
            "store_id": key[0],
            "item_id": key[1],
            "date": SOURCE_DATES,
            "sales": values,
            "year": SOURCE_DATES.year,
            "month": SOURCE_DATES.month,
            "week": SOURCE_DATES.isocalendar().week.to_numpy(dtype=np.int64),
            "day": SOURCE_DATES.day,
            "holiday_flag": 0.0,
            "unused_audit": np.nan,
        }
    )


def _target() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": KNN_DATES,
            "sales": np.arange(150, 180, dtype=np.float64),
        }
    )


def _select(source: pd.DataFrame, *, k: int = 1):
    return select_daily_sequence_sources(
        target_df=_target(),
        source_df=source,
        protocol=get_experiment_protocol("D4"),
        scenario="with",
        target_key=("T", "I0"),
        candidate_keys=(("S1", "I1"), ("S2", "I2")),
        group_cols=("store_id", "item_id"),
        observed_start=KNN_DATES[0],
        feature_cols=("sales",),
        k=k,
    )


def test_predictor_only_fields_do_not_change_knn_identity() -> None:
    source = pd.concat([_source(("S1", "I1"), 0.0), _source(("S2", "I2"), 10.0)])
    baseline = _select(source)
    changed = source.copy()
    changed.loc[
        (changed["store_id"] == "S1") & (changed["date"] == SOURCE_DATES[0]),
        "holiday_flag",
    ] = np.nan

    selection = _select(changed)

    assert selection.ordered_source_keys == baseline.ordered_source_keys
    np.testing.assert_array_equal(selection.distances, baseline.distances)
    np.testing.assert_array_equal(selection.weights, baseline.weights)
    assert selection.candidate_pool_digest == baseline.candidate_pool_digest
    assert selection.selection_identity_digest == baseline.selection_identity_digest
    assert selection.source_training_digest == baseline.source_training_digest
    assert selection.source_window_start == SOURCE_DATES[0].strftime("%Y-%m-%d")
    assert selection.source_window_end == SOURCE_DATES[-1].strftime("%Y-%m-%d")
    assert selection.knn_window_start == KNN_DATES[0].strftime("%Y-%m-%d")
    assert selection.knn_window_end == KNN_DATES[-1].strftime("%Y-%m-%d")
    assert selection.entries[0].vector_shape == (30, 1)
    assert selection.excluded_candidates == ()


def test_first_150_day_mutation_changes_training_digest_but_not_knn() -> None:
    source = pd.concat([_source(("S1", "I1"), 0.0), _source(("S2", "I2"), 10.0)])
    before = _select(source)
    changed = source.copy()
    changed.loc[
        (changed["store_id"] == "S1") & (changed["date"] == SOURCE_DATES[20]),
        "sales",
    ] += 5000.0
    after = _select(changed)

    assert before.ordered_source_keys == after.ordered_source_keys
    np.testing.assert_array_equal(before.distances, after.distances)
    assert before.entries[0].vector_digest == after.entries[0].vector_digest
    assert before.selection_identity_digest == after.selection_identity_digest
    assert before.source_training_digest != after.source_training_digest


def test_source_training_window_cannot_be_shortened() -> None:
    source = pd.concat([_source(("S1", "I1"), 0.0), _source(("S2", "I2"), 10.0)])
    source = source[source["date"] != SOURCE_DATES[0]]

    with pytest.raises(ProtocolViolation, match="valid candidates.*required K=2"):
        _select(source, k=2)
