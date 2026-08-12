from __future__ import annotations

import numpy as np
import pandas as pd

import src.protocols.runner_adapter as runner_adapter
from src.protocols.candidate_pool import prepare_daily_sequence_pool
from src.protocols.runner_adapter import configure_protocol_frames
from src.protocols.source_history import build_exact_source_history_candidate_frame
from src.protocols.knn_frames import build_observed_knn_frame
from src.utils import entity_experiment
from src.utils.dataframe_attrs import get_protocol_frame_context


class _DeepcopyBomb:
    def __deepcopy__(self, memo):
        raise AssertionError("large sentinel attrs must not be deep-copied")


def _d5_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2020-01-01", periods=35, freq="D")
    source = pd.concat(
        [
            pd.DataFrame(
                {
                    "store_nbr": store,
                    "item_nbr": item,
                    "family": "F1",
                    "date": dates,
                    "sales": float(value),
                    "onpromotion": float(value),
                    "oil_price": float(value),
                    "entity_id": f"{store}_{item}",
                    "item_id": item,
                    "year": dates.year,
                    "month": dates.month,
                    "week": dates.isocalendar().week.astype(int),
                    "day": dates.day,
                    "class": 1,
                    "perishable": 0,
                    "cluster": 1,
                    "transactions": 1.0,
                    "is_holiday": 0,
                }
            )
            for store, item, value in (("48", "S1", 1.0), ("49", "S2", 2.0))
        ],
        ignore_index=True,
    )
    target = source.iloc[: len(dates)].copy()
    target.loc[:, "store_nbr"] = "48"
    target.loc[:, "item_nbr"] = "1159415"
    target.loc[:, "entity_id"] = "48_1159415"
    target.loc[:, "item_id"] = "1159415"
    target.loc[:, "sales"] = 0.0
    target.loc[:, "onpromotion"] = 0.0
    target.loc[:, "oil_price"] = 0.0
    return source, target


def test_entity_runner_forwards_the_same_prepared_pool_to_protocol_frames(monkeypatch):
    source, target = _d5_frames()
    source.attrs.update({"split_role": "source", "split_mode": "ratio"})
    target.attrs.update({"split_role": "target", "split_mode": "days"})
    captured: list[object] = []
    prepared: list[object] = []
    real_configure = entity_experiment.configure_protocol_frames
    real_prepare = entity_experiment.prepare_daily_sequence_pool

    def recording_configure(*args, **kwargs):
        captured.append(kwargs.get("prepared_pool"))
        return real_configure(*args, **kwargs)

    def fake_runner(**kwargs):
        result = {
            "rmse": 1.0,
            "accuracy": 1.0,
            "smape": 1.0,
            "error": "",
            "prediction_shape": (1, 1),
        }
        result.update(kwargs["expected_metric_identity"])
        return result

    def recording_prepare(*args, **kwargs):
        pool = real_prepare(*args, **kwargs)
        prepared.append(pool)
        return pool

    monkeypatch.setattr(entity_experiment, "configure_protocol_frames", recording_configure)
    monkeypatch.setattr(entity_experiment, "prepare_daily_sequence_pool", recording_prepare)
    monkeypatch.setattr(entity_experiment, "_method_runner", lambda method: fake_runner)

    entity_experiment.run_single_entity_experiment(
        entity_key="48_1159415",
        source_df=source,
        target_entity_df=target,
        feature_cols=["sales", "year", "month", "week", "day", "class", "perishable", "cluster", "transactions", "oil_price", "is_holiday"],
        config={
            "dataset_id": 5,
            "dataset_name": "Dataset5",
            "info_sharing": "without",
            "group_cols": ("store_nbr", "item_nbr"),
            "source_count": 1,
            "horizon": 1,
            "window_size": 3,
            "learning_rate": 0.001,
            "source_epochs": 1,
            "target_epochs": 1,
            "batch_size": 1,
        },
        enabled_methods=["No-TL"],
    )

    assert len(captured) == 1
    assert len(prepared) == 1
    assert captured[0] is prepared[0]
    assert captured[0] is not None


def test_prepared_pool_path_keeps_model_source_and_never_calls_full_groupby(monkeypatch):
    source, target = _d5_frames()
    pool = prepare_daily_sequence_pool(
        source,
        group_cols=("store_nbr", "item_nbr"),
        observed_start="2020-01-01",
        metadata_cols=("family",),
        feature_cols=("sales", "onpromotion", "oil_price"),
    )

    def fail_full_groupby(*args, **kwargs):
        raise AssertionError("prepared pool path must not call full source groupby")

    monkeypatch.setattr(runner_adapter, "_extended_candidates", fail_full_groupby)
    configured_source, _ = configure_protocol_frames(
        source,
        target,
        dataset_id=5,
        scenario="without",
        group_cols=("store_nbr", "item_nbr"),
        grouping_col="family",
        observed_start="2020-01-01",
        prepared_pool=pool,
        retain_source_frame=True,
    )

    expected_source_rows = int((source["date"] <= pd.Timestamp("2020-01-30")).sum())
    assert len(configured_source) == expected_source_rows
    context = get_protocol_frame_context(configured_source)
    assert context.prepared_pool is pool
    assert context.candidate_keys == (("48", "S1"),)


def test_fallback_candidate_groupby_also_ignores_large_source_attrs():
    source, target = _d5_frames()
    source.attrs["source_history_eligibility"] = {"eligible_keys": [_DeepcopyBomb()]}
    source.attrs["source_history_eligible_keys"] = [["48", "S1"], ["49", "S2"]]

    configured_source, _ = configure_protocol_frames(
        source,
        target,
        dataset_id=5,
        scenario="without",
        group_cols=("store_nbr", "item_nbr"),
        grouping_col="family",
        observed_start="2020-01-01",
    )

    assert not configured_source.empty
    assert "source_history_eligibility" not in configured_source.attrs
    assert "source_history_eligible_keys" not in configured_source.attrs


def test_d5_model_frame_keeps_knn_passthrough_separate_from_model_features():
    source, _ = _d5_frames()
    model_features = (
        "sales",
        "year",
        "month",
        "week",
        "day",
        "class",
        "perishable",
        "cluster",
        "transactions",
        "oil_price",
        "is_holiday",
    )

    model_frame = entity_experiment._build_model_dataframe(
        source,
        model_features,
        source_selection_group_cols=("store_nbr", "item_nbr"),
        required_passthrough_cols=("sales", "onpromotion", "oil_price"),
    )

    assert "onpromotion" in model_frame.columns
    assert set(model_frame.columns) - {
        "date",
        "entity_id",
        "item_id",
        "store_nbr",
        "item_nbr",
        "onpromotion",
    } == set(model_features)


def test_candidate_frame_drops_large_eligibility_attrs_but_keeps_protocol_summary():
    dates = pd.date_range("2025-01-14", periods=180, freq="D")
    source = pd.concat(
        [
            pd.DataFrame({"store_id": store, "product_id": product, "date": dates, "sales": 1.0})
            for store, product in ((1, 1), (1, 2))
        ],
        ignore_index=True,
    )
    result = build_exact_source_history_candidate_frame(
        source,
        key_fields=("store_id", "product_id"),
        origin=dates[-1],
        source_history_days=180,
    )

    attrs = result.candidate_frame.attrs
    assert "source_history_eligible_keys" not in attrs
    assert "source_history_eligibility" not in attrs
    assert attrs["source_history_days"] == 180
    assert attrs["source_history_eligible_key_count"] == 2


def test_observed_knn_frame_does_not_deepcopy_large_nested_attrs():
    source = pd.DataFrame(
        {
            "store_id": [1, 1],
            "product_id": [1, 1],
            "date": pd.date_range("2020-01-01", periods=2, freq="D"),
            "sales": [1.0, 2.0],
        }
    )
    source.attrs["source_history_eligibility"] = {"eligible_keys": [_DeepcopyBomb()]}
    source.attrs["source_history_eligible_keys"] = [["1", "1"]]

    observed = build_observed_knn_frame(
        source,
        window=type(
            "Window",
            (),
            {
                "knn_observed_start": pd.Timestamp("2020-01-01"),
                "knn_observed_end": pd.Timestamp("2020-01-02"),
                "observed_days": 2,
            },
        )(),
        role="source",
        group_cols=("store_id", "product_id"),
        feature_cols=("sales",),
    )

    assert "source_history_eligibility" not in observed.attrs
    assert "source_history_eligible_keys" not in observed.attrs
