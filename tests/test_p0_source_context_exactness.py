from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

import src.protocols.candidate_pool as candidate_pool
import src.protocols.provenance as provenance
from src.protocols.candidate_pool import (
    build_canonical_source_index,
    select_daily_sequence_sources,
)
from src.protocols.experiment_protocol import get_experiment_protocol
from src.protocols.provenance import extract_selected_source_slices
from src.protocols.runner_adapter import source_key_mask
from src.utils.dataframe_attrs import ProtocolFrameContext, set_protocol_frame_context


CASES = {
    "D1": (("store_id", "item_id"), ("1", "10"), ("sales",), "2017-06-01"),
    "D2": (("brand_id", "item_id"), ("1", "10"), ("sales", "promo"), "2018-06-01"),
    "D3": (("store_id",), ("10",), ("sales",), "2020-01-01"),
    "D4": (("store_id", "product_id"), ("166", "258"), ("sales",), "2020-01-01"),
    "D5": (
        ("store_nbr", "item_nbr"),
        ("48", "364606"),
        ("sales", "onpromotion", "oil_price"),
        "2020-01-01",
    ),
    "D6": (
        ("store_id", "item_id"),
        ("CA_1", "FOODS_3_586"),
        ("sales",),
        "2020-01-01",
    ),
}

METHOD_K = {
    "SS-TL": 1,
    "MSWA-TL": 2,
    "MSSB-TL": 2,
    "MSML-TL": 2,
    "MSML-TL-RFE": 2,
}

CANDIDATE_DIGESTS = {
    ("D1", "with"): "fcf4471d0fda5dc6219475a43d8c35587a585378d54f2535c133ba03e6c1bab4",
    ("D1", "without"): "6b1b81dee94e706ba273d9673544557cf2e26f178bd85d10ca07d22b5b3057b6",
    ("D2", "with"): "1bc9e013fdcabebd7a9af85d8bc18b706f861977d7f8aba7c26ae01faf0fceff",
    ("D2", "without"): "43bbeeec67c32f163c14af58ea3fef5eb3bd1af53b0a30f905479e24e4bed758",
    ("D3", "with"): "1be4e430e3921da4d453b6392cd486d1c31161bca06c4aee507b35aa12f4b13b",
    ("D3", "without"): "f40cfdc9f0b561db1c3ffa57540a8d357523322e50420c2647b59e014123f5cf",
    ("D4", "with"): "bb3a6d9591383d19f3d7424ca390b9966630c508ebb88000581f87e30e954232",
    ("D4", "without"): "b770d3de033a5c6644984bbb178a735487e981e6f9f1d483ec568a82cf33a8da",
    ("D5", "with"): "e41a32ce2d55765cf883db808214d09da04fafffc126ba73fa1df2454cbbc14b",
    ("D5", "without"): "d5643d800b1ccd96169e15c2c412c20bacc733bfb07014a2ab2a77591559882c",
    ("D6", "with"): "34f06c74509dc41a56a3402621dba5ee39afedafefd597550eb81d657633fa87",
    ("D6", "without"): "7f854760aae4aac6b8fa5ace980b942e80acea3196d2e06eb53fa73042e07bb5",
}

EXPECTED_DISTANCE_WEIGHT = {
    1: {
        1: ((1.8257418583505538,), (1.0,)),
        2: ((2.581988897471611,), (1.0,)),
        3: ((3.1622776601683795,), (1.0,)),
    },
    2: {
        1: (
            (1.8257418583505538, 3.6514837167011076),
            (0.6666666660580861, 0.33333333394191395),
        ),
        2: (
            (2.581988897471611, 5.163977794943222),
            (0.6666666662363352, 0.33333333376366486),
        ),
        3: (
            (3.1622776601683795, 6.324555320336759),
            (0.6666666663153025, 0.3333333336846975),
        ),
    },
}


def _frames(dataset_id: str) -> tuple[pd.DataFrame, pd.DataFrame, tuple[tuple[str, ...], ...]]:
    group_cols, target_key, feature_cols, start = CASES[dataset_id]
    dates = pd.date_range(start, periods=30, freq="D")
    keys = tuple(
        tuple([f"S{number}"] + [f"I{number}"] * (len(group_cols) - 1))
        for number in (1, 2, 3)
    )
    source_frames = []
    for number, key in enumerate(keys, start=1):
        payload: dict[str, object] = {"date": dates}
        payload.update(dict(zip(group_cols, key)))
        payload.update({column: float(number) for column in feature_cols})
        source_frames.append(pd.DataFrame(payload))
    source = pd.concat(source_frames, ignore_index=True)
    target_payload: dict[str, object] = {"date": dates}
    target_payload.update(dict(zip(group_cols, target_key)))
    target_payload.update({column: 0.0 for column in feature_cols})
    target = pd.DataFrame(target_payload)
    if dataset_id == "D2":
        for frame in (source, target):
            frame.attrs.update(
                d2_source_calendarization_rule_version="v",
                d2_source_authority_digest="a",
                d2_consumer_frame_fingerprint="b",
            )
    return source, target, keys


@pytest.mark.parametrize(
    ("dataset_id", "scenario", "method"),
    itertools.product(CASES, ("with", "without"), METHOD_K),
)
def test_d1_d6_selection_and_source_slices_match_literal_golden(
    dataset_id: str,
    scenario: str,
    method: str,
) -> None:
    group_cols, target_key, feature_cols, start = CASES[dataset_id]
    source, target, keys = _frames(dataset_id)
    k = METHOD_K[method]

    result = select_daily_sequence_sources(
        target_df=target,
        source_df=source,
        protocol=get_experiment_protocol(dataset_id),
        scenario=scenario,
        target_key=target_key,
        candidate_keys=keys,
        group_cols=group_cols,
        observed_start=start,
        feature_cols=feature_cols,
        k=k,
    )

    expected_keys = keys[:k]
    feature_width = len(feature_cols)
    expected_distances, expected_weights = EXPECTED_DISTANCE_WEIGHT[k][feature_width]
    assert result.ordered_source_keys == expected_keys
    assert tuple(entry.distance for entry in result.entries) == expected_distances
    assert tuple(entry.weight for entry in result.entries) == expected_weights
    assert result.candidate_pool_digest == CANDIDATE_DIGESTS[(dataset_id, scenario)]
    assert result.candidate_pool_digest_input["candidate_keys"] == [list(key) for key in keys]
    assert result.excluded_candidates == ()

    slices = extract_selected_source_slices(
        result,
        source,
        training_start=start,
        model_feature_cols=feature_cols,
    )
    assert tuple(item.source_key for item in slices) == expected_keys
    expected_dates = tuple(pd.date_range(start, periods=30).strftime("%Y-%m-%d"))
    for number, source_slice in enumerate(slices, start=1):
        assert source_slice.dates == expected_dates
        assert source_slice.feature_cols == feature_cols
        expected_values = np.full((30, feature_width), float(number), dtype=np.float64)
        np.testing.assert_array_equal(np.asarray(source_slice.values), expected_values)


def test_no_tl_has_no_source_selection_contract() -> None:
    assert "No-TL" not in METHOD_K


def test_full_source_key_index_is_built_once_and_reused_without_full_masks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row_count = 3000
    source = pd.DataFrame(
        {
            "store_id": np.repeat(np.arange(100), row_count // 100),
            "item_id": np.repeat(np.arange(1000, 1100), row_count // 100),
            "date": np.tile(pd.date_range("2020-01-01", periods=30), 100),
            "sales": np.arange(row_count, dtype=float),
        }
    )
    calls = 0
    axis_one_apply_calls = 0
    real_normalize = candidate_pool.normalize_source_key
    real_apply = pd.DataFrame.apply

    def counting_normalize(key):
        nonlocal calls
        calls += 1
        return real_normalize(key)

    def counting_apply(frame, function, axis=0, *args, **kwargs):
        nonlocal axis_one_apply_calls
        if axis in (1, "columns"):
            axis_one_apply_calls += 1
        return real_apply(frame, function, axis=axis, *args, **kwargs)

    monkeypatch.setattr(candidate_pool, "normalize_source_key", counting_normalize)
    monkeypatch.setattr(pd.DataFrame, "apply", counting_apply)
    source_index = build_canonical_source_index(
        source,
        group_cols=("store_id", "item_id"),
    )
    assert calls == row_count
    assert axis_one_apply_calls == 0

    set_protocol_frame_context(
        source,
        ProtocolFrameContext(source_index=source_index),
    )
    for _method in METHOD_K:
        mask = source_key_mask(source, ("store_id", "item_id"), ("1", "1001"))
        assert int(mask.sum()) == 30
    assert calls == row_count
    assert axis_one_apply_calls == 0


def test_selected_slice_validation_never_calls_full_prepare_with_context_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, target, keys = _frames("D4")
    result = select_daily_sequence_sources(
        target_df=target,
        source_df=source,
        protocol=get_experiment_protocol("D4"),
        scenario="with",
        target_key=CASES["D4"][1],
        candidate_keys=keys,
        group_cols=CASES["D4"][0],
        observed_start=CASES["D4"][3],
        feature_cols=CASES["D4"][2],
        k=2,
    )
    index = build_canonical_source_index(source, group_cols=CASES["D4"][0])
    set_protocol_frame_context(source, ProtocolFrameContext(source_index=index))

    def fail_full_prepare(*_args, **_kwargs):
        raise AssertionError("selected lookup must not prepare the full source frame")

    monkeypatch.setattr(provenance, "_prepare_source", fail_full_prepare)
    slices = extract_selected_source_slices(
        result,
        source,
        training_start=CASES["D4"][3],
        model_feature_cols=CASES["D4"][2],
    )
    assert tuple(item.source_key for item in slices) == result.ordered_source_keys


def test_lifecycle_accounting_is_per_context_not_cross_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, _, _ = _frames("D4")
    calls = 0
    real_normalize = candidate_pool.normalize_source_key

    def counting_normalize(key):
        nonlocal calls
        calls += 1
        return real_normalize(key)

    monkeypatch.setattr(candidate_pool, "normalize_source_key", counting_normalize)
    shared_index = build_canonical_source_index(source, group_cols=CASES["D4"][0])
    shared_context = ProtocolFrameContext(source_index=shared_index)
    for _method in METHOD_K:
        set_protocol_frame_context(source, shared_context)
        source_key_mask(source, CASES["D4"][0], ("S1", "I1"))
    assert calls == len(source)

    calls = 0
    for _method in METHOD_K:
        build_canonical_source_index(source, group_cols=CASES["D4"][0])
    assert calls == len(source) * len(METHOD_K)
