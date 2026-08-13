from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.protocols.experiment_protocol import ProtocolViolation
from src.protocols.raw_preprocessing import (
    TargetK3RawSourceContext,
    canonical_raw_frame_digest,
)
from src.protocols.runner_adapter import source_key_mask
from src.protocols.runner_adapter import configure_protocol_frames
from src.source_selection.source_selector import TargetK3SelectionContext


K3_METHODS = ("MSWA-TL", "MSSB-TL", "MSML-TL", "MSML-TL-RFE")


def _source_frame(*, dataset: str = "D1", scenario: str = "without_information_sharing") -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    rows = []
    for source_number in (1, 2, 3, 4):
        for offset, date in enumerate(dates):
            rows.append(
                {
                    "entity_id": f"source-{source_number}",
                    "item_id": "item-1",
                    "date": date,
                    "sales": float(source_number * 10 + offset),
                    "year": np.int64(date.year),
                    "month": np.int64(date.month),
                    "week": np.int64(date.isocalendar().week),
                    "day": np.int64(date.day),
                    "promo": np.float64(offset % 2),
                    "customers": np.float64(100 + offset),
                    "open": np.int64(1),
                    "school_holiday": np.int64(0),
                    "stock_hour6_22_cnt": np.float64(20 + offset),
                    "activity_flag": np.int64(offset % 2),
                    "discount": np.float64(0.1 * (offset % 3)),
                    "holiday_flag": np.int64(0),
                    "precpt": np.float64(0.0),
                    "avg_temperature": np.float64(20 + offset),
                    "avg_humidity": np.float64(50 + offset),
                    "avg_wind_level": np.float64(2),
                    "class": np.int64(1),
                    "perishable": np.int64(0),
                    "cluster": np.int64(2),
                    "transactions": np.float64(1000 + offset),
                    "oil_price": np.float64(40 + offset),
                    "is_holiday": np.int64(0),
                    "onpromotion": np.float64(offset % 2),
                    "wm_yr_wk": np.int64(1000 + offset),
                    "weekday": np.int64(date.weekday()),
                    "event_flag": np.int64(0),
                    "snap": np.int64(offset % 2),
                    "sell_price": np.float64(3.5),
                }
            )
    frame = pd.DataFrame(rows)
    frame.attrs.update(
        {
            "protocol_version": "test-protocol",
            "protocol_dataset_id": dataset,
            "protocol_scenario": scenario,
            "protocol_target_key": ("target-1", "item-1"),
            "source_observation_cutoff": "2019-12-31",
            "source_history_start": "2020-01-01",
            "source_history_end": "2020-01-06",
            "source_frame_digest": f"source-frame-{dataset}-{scenario}",
            "split_role": "source",
            "split_mode": "ratio",
            "split_config": {"train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1},
            "audit_metadata": {"heavy": ["must", "not", "propagate"]},
        }
    )
    return frame


def _selection(keys=(1, 2, 3)) -> dict[str, object]:
    return {
        "sources": [
            {
                "source_key": (f"source-{number}", "item-1"),
                "distance": float(number),
                "weight": 1.0 / 3.0,
            }
            for number in keys
        ],
        "meta": {
            "requested_k": 3,
            "effective_k": 3,
            "candidate_pool_digest": "candidate-pool-digest",
            "selection_result_digest": "selection-result-digest",
        },
    }


def _lifecycle(*, dataset: str = "D1", scenario: str = "without_information_sharing", horizon: int = 1, seed: int = 42, target: str = "target-1") -> tuple[object, ...]:
    return (dataset, scenario, horizon, seed, (target, "item-1"))


def _consume_all_methods(context: TargetK3RawSourceContext, source: pd.DataFrame) -> dict[str, list[pd.DataFrame]]:
    result: dict[str, list[pd.DataFrame]] = {}
    for method in K3_METHODS:
        result[method] = [
            context.working_source(
                selection_result=_selection(),
                source_df=source,
                source_key=selected["source_key"],
                group_cols=("entity_id", "item_id"),
                model_feature_cols=("sales", "year"),
            )
            for selected in _selection()["sources"]
        ]
    return result


def test_k3_materialization_count_is_three_and_each_consumer_gets_a_copy(monkeypatch) -> None:
    import src.protocols.raw_preprocessing as raw_preprocessing

    source = _source_frame()
    real_mask = source_key_mask
    calls: list[tuple[object, ...]] = []

    def counting_mask(frame, group_cols, source_key):
        calls.append(tuple(source_key))
        return real_mask(frame, group_cols, source_key)

    monkeypatch.setattr(raw_preprocessing, "source_key_mask", counting_mask)
    context = TargetK3RawSourceContext(_lifecycle())
    working = _consume_all_methods(context, source)

    assert calls == [
        ("source-1", "item-1"),
        ("source-2", "item-1"),
        ("source-3", "item-1"),
    ]
    assert context.materialization_count == 3
    assert len({id(frames[0]) for frames in working.values()}) == 4


@pytest.mark.parametrize("dataset", ["D1", "D2", "D3", "D4", "D5", "D6"])
def test_per_target_selected_raw_operation_count_is_exactly_thirteen_to_four(
    dataset,
    monkeypatch,
) -> None:
    import src.protocols.raw_preprocessing as raw_preprocessing

    source = _source_frame(dataset=dataset)
    real_mask = source_key_mask
    legacy_calls = 0
    for _method in K3_METHODS:
        for selected in _selection()["sources"]:
            legacy_calls += 1
            key = tuple(selected["source_key"])
            legacy = source[real_mask(source, ("entity_id", "item_id"), key)].copy()
            assert not legacy.empty
    legacy_ss_calls = 1

    canonical_calls: list[tuple[object, ...]] = []

    def counting_mask(frame, group_cols, source_key):
        canonical_calls.append(tuple(source_key))
        return real_mask(frame, group_cols, source_key)

    monkeypatch.setattr(raw_preprocessing, "source_key_mask", counting_mask)
    context = TargetK3RawSourceContext(_lifecycle(dataset=dataset))
    _consume_all_methods(context, source)
    new_ss_calls = 1

    assert legacy_calls == 12
    assert len(canonical_calls) == context.materialization_count == 3
    assert (legacy_ss_calls + legacy_calls, new_ss_calls + len(canonical_calls)) == (13, 4)


@pytest.mark.parametrize("dataset", ["D4", "D5", "D6"])
def test_five_target_cell_raw_operation_count_is_exactly_sixty_five_to_twenty(
    dataset,
    monkeypatch,
) -> None:
    import src.protocols.raw_preprocessing as raw_preprocessing

    real_mask = source_key_mask
    calls: list[tuple[object, ...]] = []

    def counting_mask(frame, group_cols, source_key):
        calls.append(tuple(source_key))
        return real_mask(frame, group_cols, source_key)

    monkeypatch.setattr(raw_preprocessing, "source_key_mask", counting_mask)
    contexts = []
    for target_number in range(1, 6):
        source = _source_frame(dataset=dataset)
        target_key = (f"target-{target_number}", "item-1")
        source.attrs["protocol_target_key"] = target_key
        context = TargetK3RawSourceContext(
            _lifecycle(dataset=dataset, target=f"target-{target_number}")
        )
        _consume_all_methods(context, source)
        contexts.append(context)

    legacy_total = 5 * (1 + 12)
    new_total = 5 + len(calls)
    assert legacy_total == 65
    assert len(calls) == sum(context.materialization_count for context in contexts) == 15
    assert new_total == 20


@pytest.mark.parametrize(
    ("dataset", "model_features"),
    [
        ("D1", ("sales", "year", "month", "week", "day")),
        ("D2", ("sales", "year", "month", "week", "day")),
        ("D3", ("sales", "customers", "open", "promo", "school_holiday", "year")),
        (
            "D4",
            (
                "sales",
                "stock_hour6_22_cnt",
                "activity_flag",
                "discount",
                "holiday_flag",
                "precpt",
                "avg_temperature",
                "avg_humidity",
                "avg_wind_level",
                "year",
                "month",
                "week",
                "day",
            ),
        ),
        ("D5", ("sales", "class", "perishable", "cluster", "transactions", "oil_price", "is_holiday")),
        ("D6", ("sales", "wm_yr_wk", "weekday", "event_flag", "snap", "sell_price")),
    ],
)
def test_legacy_and_canonical_raw_frames_are_exactly_equal(dataset, model_features) -> None:
    source = _source_frame(dataset=dataset)
    context = TargetK3RawSourceContext(_lifecycle(dataset=dataset))

    for selected in _selection()["sources"]:
        key = tuple(selected["source_key"])
        legacy = source[source_key_mask(source, ("entity_id", "item_id"), key)].copy()
        canonical = context.working_source(
            selection_result=_selection(),
            source_df=source,
            source_key=key,
            group_cols=("entity_id", "item_id"),
            model_feature_cols=model_features,
        )
        pd.testing.assert_frame_equal(legacy, canonical, check_exact=True, check_categorical=True)
        assert tuple(map(str, legacy.dtypes)) == tuple(map(str, canonical.dtypes))
        assert canonical_raw_frame_digest(legacy) == canonical_raw_frame_digest(canonical)
        provenance = context.provenance_for(key)
        assert provenance.source_key == key
        assert provenance.columns == tuple(legacy.columns)
        assert provenance.dtypes == tuple(map(str, legacy.dtypes))
        assert provenance.raw_frame_digest == canonical_raw_frame_digest(legacy)
        assert provenance.model_feature_cols == model_features


def test_copy_on_consume_isolates_values_rows_columns_and_attrs() -> None:
    source = _source_frame()
    source.attrs["method"] = "MSWA-TL"
    original = deepcopy(source)
    context = TargetK3RawSourceContext(_lifecycle())
    working = _consume_all_methods(context, source)
    mswa = working["MSWA-TL"][0]
    mssb = working["MSSB-TL"][0]

    mswa.iloc[0, mswa.columns.get_loc("sales")] = -999.0
    mswa["method_only"] = 1
    mswa.attrs["method"] = "MSWA-TL"
    mswa.attrs["split_config"]["train_ratio"] = 0.5
    mswa.drop(index=mswa.index[-1], inplace=True)

    assert mssb.iloc[0]["sales"] != -999.0
    assert "method_only" not in mssb
    assert "method" not in mssb.attrs
    assert mssb.attrs["split_config"]["train_ratio"] == 0.8
    pd.testing.assert_frame_equal(source, original, check_exact=True)
    fresh = context.working_source(
        selection_result=_selection(),
        source_df=source,
        source_key=("source-1", "item-1"),
        group_cols=("entity_id", "item_id"),
        model_feature_cols=("sales", "year"),
    )
    assert len(fresh) == 6
    assert fresh.iloc[0]["sales"] != -999.0
    assert "method_only" not in fresh
    assert "method" not in fresh.attrs
    assert "audit_metadata" not in fresh.attrs


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda frame: frame.rename(columns={"sales": "sales_changed"}), "schema"),
        (lambda frame: frame[["entity_id", "item_id", "date", "year", "sales", "promo"]], "schema"),
        (lambda frame: frame.assign(sales=frame["sales"].astype("float32")), "dtype"),
        (lambda frame: frame.iloc[::-1], "row"),
    ],
)
def test_bound_context_rejects_schema_dtype_and_row_identity_drift(mutation, message) -> None:
    source = _source_frame()
    context = TargetK3RawSourceContext(_lifecycle())
    context.working_source(
        selection_result=_selection(),
        source_df=source,
        source_key=("source-1", "item-1"),
        group_cols=("entity_id", "item_id"),
        model_feature_cols=("sales", "year"),
    )
    changed = mutation(source.copy())
    changed.attrs = deepcopy(source.attrs)
    with pytest.raises(ProtocolViolation, match=message):
        context.working_source(
            selection_result=_selection(),
            source_df=changed,
            source_key=("source-1", "item-1"),
            group_cols=("entity_id", "item_id"),
            model_feature_cols=("sales", "year"),
        )


def test_source_key_and_selection_order_mismatch_fail_closed() -> None:
    source = _source_frame()
    context = TargetK3RawSourceContext(_lifecycle())
    context.working_source(
        selection_result=_selection(),
        source_df=source,
        source_key=("source-1", "item-1"),
        group_cols=("entity_id", "item_id"),
        model_feature_cols=("sales", "year"),
    )
    with pytest.raises(ProtocolViolation, match="source key"):
        context.working_source(
            selection_result=_selection(),
            source_df=source,
            source_key=("source-4", "item-1"),
            group_cols=("entity_id", "item_id"),
            model_feature_cols=("sales", "year"),
        )
    with pytest.raises(ProtocolViolation, match="selection"):
        context.working_source(
            selection_result=_selection((3, 2, 1)),
            source_df=source,
            source_key=("source-1", "item-1"),
            group_cols=("entity_id", "item_id"),
            model_feature_cols=("sales", "year"),
        )


def test_duplicate_source_dates_fail_closed() -> None:
    source = _source_frame()
    duplicate = pd.concat([source, source.iloc[[0]]], ignore_index=True)
    duplicate.attrs = deepcopy(source.attrs)
    with pytest.raises(ProtocolViolation, match="duplicate date"):
        TargetK3RawSourceContext(_lifecycle()).working_source(
            selection_result=_selection(),
            source_df=duplicate,
            source_key=("source-1", "item-1"),
            group_cols=("entity_id", "item_id"),
            model_feature_cols=("sales", "year"),
        )


def test_cross_target_and_cross_cell_contexts_never_share_canonical_objects() -> None:
    source_a = _source_frame()
    source_b = _source_frame()
    source_b.attrs["protocol_target_key"] = ("target-2", "item-1")
    source_cell_b = _source_frame()
    target_a = TargetK3RawSourceContext(_lifecycle(target="target-1"))
    target_b = TargetK3RawSourceContext(_lifecycle(target="target-2"))
    cell_b = TargetK3RawSourceContext(_lifecycle(target="target-1", horizon=2, seed=7))

    frames = [
        context.working_source(
            selection_result=_selection(),
            source_df=source,
            source_key=("source-1", "item-1"),
            group_cols=("entity_id", "item_id"),
            model_feature_cols=("sales", "year"),
        )
        for context, source in zip(
            (target_a, target_b, cell_b),
            (source_a, source_b, source_cell_b),
        )
    ]
    assert len({context.canonical_identity_for(("source-1", "item-1")) for context in (target_a, target_b, cell_b)}) == 3
    frames[0].iloc[0, frames[0].columns.get_loc("sales")] = -1.0
    assert frames[1].iloc[0]["sales"] != -1.0
    assert frames[2].iloc[0]["sales"] != -1.0


@pytest.mark.parametrize(
    ("dataset", "model_features", "knn_only"),
    [
        ("D2", ("sales", "year"), "promo"),
        ("D3", ("sales", "year", "promo"), None),
        ("D5", ("sales", "year"), "promo"),
    ],
)
def test_model_feature_contract_does_not_absorb_knn_only_columns(dataset, model_features, knn_only) -> None:
    source = _source_frame(dataset=dataset)
    context = TargetK3RawSourceContext(_lifecycle(dataset=dataset))
    context.working_source(
        selection_result=_selection(),
        source_df=source,
        source_key=("source-1", "item-1"),
        group_cols=("entity_id", "item_id"),
        model_feature_cols=model_features,
    )
    provenance = context.provenance_for(("source-1", "item-1"))
    assert provenance.model_feature_cols == model_features
    if knn_only is not None:
        assert knn_only not in provenance.model_feature_cols


def test_level_one_context_has_no_fill_split_normalization_or_rfe_state() -> None:
    context = TargetK3RawSourceContext(_lifecycle())
    assert not hasattr(context, "filled_sources")
    assert not hasattr(context, "raw_splits")
    assert not hasattr(context, "scaler")
    assert not hasattr(context, "normalized_sources")
    assert not hasattr(context, "rfe_state")


def test_canonical_raw_context_accepts_real_shared_selection_evidence() -> None:
    dates = pd.date_range("2020-01-01", periods=35, freq="D")
    source = pd.concat(
        [
            pd.DataFrame(
                {
                    "store_id": f"S{number}",
                    "item_id": f"I{number}",
                    "second_category_id": 20,
                    "date": dates,
                    "sales": np.full(35, float(number), dtype=np.float64),
                    "model_feature": np.full(35, number * 10.0, dtype=np.float64),
                }
            )
            for number in range(1, 5)
        ],
        ignore_index=True,
    )
    target = pd.DataFrame(
        {
            "store_id": "T1",
            "item_id": "I0",
            "second_category_id": 20,
            "date": dates,
            "sales": np.zeros(35, dtype=np.float64),
            "model_feature": np.zeros(35, dtype=np.float64),
        }
    )
    configured_source, configured_target = configure_protocol_frames(
        source,
        target,
        dataset_id="D4",
        scenario="with",
        group_cols=("store_id", "item_id"),
        observed_start="2020-01-01",
    )
    lifecycle = (
        "D4",
        configured_target.attrs["protocol_scenario"],
        1,
        42,
        tuple(configured_target.attrs["protocol_target_key"]),
    )
    selection_context = TargetK3SelectionContext(lifecycle)
    evidence = selection_context.selection_for_method(
        target_df=configured_target,
        source_df=configured_source,
        feature_cols=("sales", "model_feature"),
        group_cols=("store_id", "item_id"),
        k=3,
        weight_mode="inverse_distance",
    )
    selection = evidence.method_wrapper(
        lifecycle_identity=lifecycle,
        target_df=configured_target,
        source_df=configured_source,
        feature_cols=("sales", "model_feature"),
        group_cols=("store_id", "item_id"),
        k=3,
        weight_mode="inverse_distance",
    )
    raw_context = TargetK3RawSourceContext(lifecycle)
    working = [
        raw_context.working_source(
            selection_result=selection,
            source_df=configured_source,
            source_key=selected["source_key"],
            group_cols=("store_id", "item_id"),
            model_feature_cols=("sales", "model_feature"),
        )
        for selected in selection["sources"]
    ]

    assert raw_context.materialization_count == 3
    assert len({id(frame) for frame in working}) == 3
    assert all(not frame.empty for frame in working)


@pytest.mark.parametrize(
    "runner",
    [
        pytest.param(
            __import__("src.transfer_methods.mswa_tl", fromlist=["run_mswa_tl"]).run_mswa_tl,
            id="MSWA-TL",
        ),
        pytest.param(
            __import__("src.transfer_methods.mssb_tl", fromlist=["run_mssb_tl"]).run_mssb_tl,
            id="MSSB-TL",
        ),
        pytest.param(
            __import__("src.transfer_methods.msml_tl", fromlist=["run_msml_tl"]).run_msml_tl,
            id="MSML-TL",
        ),
        pytest.param(
            __import__("src.transfer_methods.msml_tl_rfe", fromlist=["run_msml_tl_rfe"]).run_msml_tl_rfe,
            id="MSML-TL-RFE",
        ),
    ],
)
def test_each_k3_method_consumes_the_shared_raw_context_before_training(runner) -> None:
    class ExpectedRawConsume(RuntimeError):
        pass

    class Evidence:
        def method_wrapper(self, **_kwargs):
            return _selection()

    class SelectionContext:
        lifecycle_identity = _lifecycle()

        def selection_for_method(self, **_kwargs):
            return Evidence()

    class RawContext:
        def working_source(self, **kwargs):
            assert tuple(kwargs["source_key"]) == ("source-1", "item-1")
            assert tuple(kwargs["model_feature_cols"]) == ("sales", "year")
            raise ExpectedRawConsume

    source = _source_frame()
    target = source[source["entity_id"] == "source-4"].copy()
    target.loc[:, "entity_id"] = "target-1"
    target.attrs["split_role"] = "target"

    with pytest.raises(ExpectedRawConsume):
        runner(
            source_df=source,
            target_df=target,
            feature_cols=("sales", "year"),
            k=3,
            group_cols=("entity_id", "item_id"),
            k3_selection_context=SelectionContext(),
            k3_raw_source_context=RawContext(),
        )
