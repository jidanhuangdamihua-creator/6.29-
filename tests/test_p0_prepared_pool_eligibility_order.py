from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import src.protocols.knn_frames as knn_frames
from src.protocols.candidate_pool import (
    classify_prepared_candidate_dates,
    prepare_daily_sequence_pool,
)
from src.protocols.experiment_protocol import ObservationWindow, ProtocolViolation
from src.protocols.knn_frames import (
    build_observed_knn_frame,
    build_prepared_pool_observed_knn_frame,
)
from src.protocols.runner_adapter import configure_protocol_frames
from src.source_selection.source_selector import SourceSelector
from src.utils.dataframe_attrs import get_protocol_frame_context


GROUP_COLS = ("entity_id", "item_id")
FEATURE_COLS = ("sales", "onpromotion", "oil_price")
OBSERVED_DATES = pd.date_range("2020-01-01", periods=30, freq="D")
TARGET_DATES = pd.date_range("2020-01-01", periods=35, freq="D")
BASE_FULL_FRAME_DIGEST = (
    "04227d57ee345d49f5fd22048f330f1571bc0c2901b0cea7c75cf3d89a6ad83b"
)


def _rows(
    entity: str,
    item: str,
    *,
    dates: pd.DatetimeIndex = OBSERVED_DATES,
    sales: object = 1.0,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entity_id": entity,
            "item_id": item,
            "family": "F1",
            "date": dates,
            "sales": sales,
            "onpromotion": 0.0,
            "oil_price": 50.0,
        }
    )


def _target() -> pd.DataFrame:
    return _rows("T", "TARGET", dates=TARGET_DATES, sales=0.0)


def _configure(source: pd.DataFrame):
    pool = prepare_daily_sequence_pool(
        source,
        group_cols=GROUP_COLS,
        observed_start="2020-01-01",
        metadata_cols=("family",),
        feature_cols=FEATURE_COLS,
    )
    configured = configure_protocol_frames(
        source.iloc[0:0].copy(),
        _target(),
        dataset_id="D5",
        scenario="with",
        group_cols=GROUP_COLS,
        grouping_col="family",
        observed_start="2020-01-01",
        prepared_pool=pool,
    )
    return pool, configured


@pytest.mark.parametrize("explicit_bad_value", [False, True])
def test_incomplete_candidate_is_excluded_before_numeric_validation(
    explicit_bad_value: bool,
) -> None:
    complete = _rows("S0", "VALID", sales=1.0)
    incomplete = _rows("S1", "MISSING", sales=2.0).iloc[1:].copy()
    if explicit_bad_value:
        incomplete["sales"] = incomplete["sales"].astype(object)
        incomplete.loc[incomplete.index[0], "sales"] = "not-numeric"
    source = pd.concat([complete, incomplete], ignore_index=True)
    _, (configured_source, configured_target) = _configure(source)

    result = SourceSelector().select_top_k_sources(
        configured_target,
        configured_source,
        feature_cols=FEATURE_COLS,
        k=1,
        group_cols=GROUP_COLS,
    )

    assert result["meta"]["source_skip_diagnostics"] == [
        {
            "source_key": ("S1", "MISSING"),
            "reason": "missing_observed_dates",
            "missing_dates": ("2020-01-01",),
        }
    ]
    assert tuple(item["source_key"] for item in result["sources"]) == (
        ("S0", "VALID"),
    )


@pytest.mark.parametrize("bad_value", [np.nan, "not-numeric"])
def test_complete_candidate_numeric_values_remain_fail_closed(bad_value: object) -> None:
    complete = _rows("S0", "VALID", sales=1.0)
    if isinstance(bad_value, str):
        complete["sales"] = complete["sales"].astype(object)
    complete.loc[complete.index[0], "sales"] = bad_value

    with pytest.raises(ProtocolViolation, match="non-numeric|non-finite"):
        _configure(complete)


def test_duplicate_date_fails_before_missing_date_classification() -> None:
    duplicate = pd.concat(
        [
            _rows("S0", "VALID", sales=1.0),
            _rows("S0", "VALID", sales=1.0).iloc[[0]],
            _rows("S1", "MISSING", sales=2.0).iloc[1:],
        ],
        ignore_index=True,
    )

    with pytest.raises(ProtocolViolation, match="duplicate observed dates"):
        _configure(duplicate)


def test_full_candidate_frame_and_digest_remain_base_exact_and_hash_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = [_rows("S0", "VALID", sales=5.0)]
    for index in range(25):
        frames.append(
            _rows(f"S{index + 1}", f"I{index}", sales=float(len(f"I{index}")))[
                1:
            ].copy()
        )
    source = pd.concat(frames, ignore_index=True)
    calls = 0
    real_compute = knn_frames._compute_canonical_knn_frame_digest

    def counting_compute(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_compute(*args, **kwargs)

    monkeypatch.setattr(knn_frames, "_compute_canonical_knn_frame_digest", counting_compute)
    pool, (configured_source, configured_target) = _configure(source)
    context = get_protocol_frame_context(configured_source)
    observed = context.observed_frames["source"]
    expected = pool.selected_frame(
        context.candidate_keys,
        feature_cols=FEATURE_COLS,
    )

    pd.testing.assert_frame_equal(observed, expected, check_exact=True)
    assert observed.shape == (780, 6)
    assert observed.columns.tolist() == [*GROUP_COLS, "date", *FEATURE_COLS]
    assert int(observed["sales"].isna().sum()) == 25
    assert observed.attrs["knn_frame_digest"] == BASE_FULL_FRAME_DIGEST
    assert configured_source.attrs["source_frame_digest"] == BASE_FULL_FRAME_DIGEST
    evidence = observed.attrs[knn_frames._DIGEST_EVIDENCE_ATTR]
    assert evidence.candidate_scope == context.candidate_keys
    assert evidence.complete_candidate_scope == (("S0", "VALID"),)
    assert evidence.excluded_candidate_scope == context.candidate_keys[1:]
    assert calls == 2

    SourceSelector().select_top_k_sources(
        configured_target,
        configured_source,
        feature_cols=FEATURE_COLS,
        k=1,
        group_cols=GROUP_COLS,
    )
    assert calls == 2


def test_ordinary_observed_builder_remains_full_frame_strict() -> None:
    incomplete_aligned = pd.concat(
        [
            _rows("S0", "VALID", sales=1.0),
            _rows("S1", "MISSING", sales=2.0).assign(
                sales=lambda frame: frame["sales"].mask(frame.index == 0)
            ),
        ],
        ignore_index=True,
    )

    with pytest.raises(ProtocolViolation, match="non-numeric"):
        build_observed_knn_frame(
            incomplete_aligned,
            window=ObservationWindow.from_start("2020-01-01"),
            role="source",
            group_cols=GROUP_COLS,
            feature_cols=FEATURE_COLS,
        )


def test_prepared_builder_rejects_mismatched_eligibility_proof() -> None:
    source = pd.concat(
        [
            _rows("S0", "VALID", sales=1.0),
            _rows("S1", "MISSING", sales=2.0).iloc[1:],
        ],
        ignore_index=True,
    )
    pool = prepare_daily_sequence_pool(
        source,
        group_cols=GROUP_COLS,
        observed_start="2020-01-01",
        metadata_cols=("family",),
        feature_cols=FEATURE_COLS,
    )
    candidate_scope = (("S0", "VALID"), ("S1", "MISSING"))
    proof = classify_prepared_candidate_dates(
        pool,
        candidate_scope,
        group_cols=GROUP_COLS,
        required_dates=OBSERVED_DATES,
        feature_cols=FEATURE_COLS,
    )
    mismatched = replace(proof, excluded_candidate_scope=())

    with pytest.raises(ProtocolViolation, match="does not partition"):
        build_prepared_pool_observed_knn_frame(
            pool.selected_frame(candidate_scope, feature_cols=FEATURE_COLS),
            window=ObservationWindow.from_start("2020-01-01"),
            role="source",
            group_cols=GROUP_COLS,
            feature_cols=FEATURE_COLS,
            eligibility_proof=mismatched,
            pool_identity=id(pool),
        )
