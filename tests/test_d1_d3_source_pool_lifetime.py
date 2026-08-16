from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import src.protocols.runner_adapter as runner_adapter
from scripts import run_full_paper_experiments as full_runner
from src.data_processing.data_preprocessing import normalize_features
from src.protocols.experiment_protocol import (
    FORMAL_HORIZONS,
    FORMAL_SEEDS,
    ProtocolViolation,
    get_experiment_protocol,
)
from src.protocols.knn_frames import get_configured_knn_frame
from src.protocols.rolling_origin import build_sample_manifest
from src.protocols.transformation_reuse import TargetTransformationReuseContext
from src.source_selection.source_selector import SourceSelector, TargetK3SelectionContext
from src.utils.dataframe_attrs import get_protocol_frame_context


METHODS = ("No-TL", "SS-TL", "MSWA-TL", "MSSB-TL", "MSML-TL", "MSML-TL-RFE")
CANONICAL_D3_TARGET = ("10",)


def _calendar_fields(timestamp: pd.Timestamp) -> dict[str, int]:
    return {
        "year": int(timestamp.year),
        "month": int(timestamp.month),
        "week": int(timestamp.isocalendar().week),
        "day": int(timestamp.day),
    }


def _cell_frames(dataset_id: str) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...], object]:
    if dataset_id == "D1":
        group_cols = ("store_id", "item_id")
        observed_start = None
        source_dates = pd.date_range("2017-06-01", "2017-06-30")
        target_dates = pd.date_range("2017-06-01", periods=210)
        source_keys = tuple(
            (store, item) for store in range(1, 4) for item in range(1, 10)
        )
        target_key = (1, 10)
    elif dataset_id == "D2":
        group_cols = ("brand_id", "item_id")
        observed_start = None
        source_dates = pd.date_range("2018-01-02", "2018-06-30")
        target_dates = pd.date_range("2018-06-01", periods=210)
        source_keys = tuple(
            (brand, item) for brand in range(1, 4) for item in range(1, 10)
        )
        target_key = (1, 10)
    else:
        group_cols = ("store_id",)
        observed_start = "2015-01-03"
        source_dates = pd.date_range("2015-01-03", periods=30)
        target_dates = pd.date_range("2015-01-03", periods=210)
        source_keys = tuple((store,) for store in range(1, 31) if store != 10)
        target_key = (10,)

    source_rows: list[dict[str, object]] = []
    for key_index, key in enumerate(source_keys, start=1):
        for timestamp in source_dates:
            frozen_d2_date = dataset_id == "D2" and timestamp.strftime("%Y-%m-%d") in {
                "2018-04-01",
                "2018-04-25",
                "2018-05-01",
                "2018-06-02",
            }
            row = {
                group_cols[0]: key[0],
                "date": timestamp,
                "entity_id": f"source-{key_index}",
                "sales": 0.0 if frozen_d2_date else float(key_index),
                **_calendar_fields(timestamp),
            }
            if len(group_cols) == 2:
                row[group_cols[1]] = key[1]
            if dataset_id == "D2":
                row["promo"] = float((timestamp.day + key_index) % 2)
            elif dataset_id == "D3":
                row.update(
                    {
                        "entity_id": str(key[0]),
                        "customers": float(100 + key_index),
                        "open": 1,
                        "promo": int(timestamp.day % 2),
                        "school_holiday": 0,
                    }
                )
            source_rows.append(row)

    target_rows = []
    for offset, timestamp in enumerate(target_dates):
        row = {
            group_cols[0]: target_key[0],
            "date": timestamp,
            "entity_id": "target",
            "sales": float(100 + offset),
            **_calendar_fields(timestamp),
        }
        if len(group_cols) == 2:
            row[group_cols[1]] = target_key[1]
        if dataset_id == "D2":
            row["promo"] = float(offset % 2)
        elif dataset_id == "D3":
            row.update(
                {
                    "entity_id": "10",
                    "customers": float(200 + offset),
                    "open": 1,
                    "promo": int(offset % 2),
                    "school_holiday": 0,
                }
            )
        target_rows.append(row)

    source = pd.DataFrame(source_rows)
    target = pd.DataFrame(target_rows)
    source.attrs["split_role"] = "source"
    target.attrs["split_role"] = "target"
    return source, target, group_cols, observed_start


def _mode_source(dataset_id: str, source: pd.DataFrame, scenario: str) -> pd.DataFrame:
    if scenario == "with" or dataset_id == "D3":
        return source.copy()
    column = "store_id" if dataset_id == "D1" else "brand_id"
    filtered = source.loc[source[column].eq(1)].copy()
    filtered.attrs = deepcopy(source.attrs)
    return filtered


def _configured_six(
    dataset_id: str,
    source: pd.DataFrame,
    target: pd.DataFrame,
    group_cols: tuple[str, ...],
    observed_start: object,
    scenario: str,
    *,
    prepared_pool=None,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    configured_sources = []
    configured_targets = []
    for method in METHODS:
        source_carrier = source.copy()
        source_carrier.attrs = deepcopy(source.attrs)
        target_carrier = target.copy()
        target_carrier.attrs = deepcopy(target.attrs)
        source_carrier.attrs["method"] = method
        target_carrier.attrs["method"] = method
        configured_source, configured_target = runner_adapter.configure_protocol_frames(
            source_carrier,
            target_carrier,
            dataset_id=dataset_id,
            scenario=scenario,
            group_cols=group_cols,
            observed_start=observed_start,
            prepared_pool=prepared_pool,
            retain_source_frame=prepared_pool is not None,
            enforce_formal_target=True,
        )
        configured_sources.append(configured_source)
        configured_targets.append(configured_target)
    return configured_sources, configured_targets


def _manifest(frame: pd.DataFrame):
    return build_sample_manifest(
        frame,
        dataset_id=frame.attrs["protocol_dataset_id"],
        track=frame.attrs["protocol_track"],
        scenario=frame.attrs["protocol_scenario"],
        target_key=frame.attrs["protocol_target_key"],
        observed_end=frame.attrs["knn_observed_end"],
        first_forecast_origin=pd.Timestamp(frame.attrs["knn_observed_end"])
        + pd.Timedelta(days=10),
        input_window=10,
    )


def _assert_frame_equal_without_attrs(left: pd.DataFrame, right: pd.DataFrame) -> None:
    left_copy = left.copy()
    right_copy = right.copy()
    left_copy.attrs = {}
    right_copy.attrs = {}
    pd.testing.assert_frame_equal(left_copy, right_copy)


def _selection_snapshot(source: pd.DataFrame, target: pd.DataFrame, *, k: int) -> dict[str, object]:
    context = get_protocol_frame_context(source)
    selected = SourceSelector().select_top_k_sources(
        target,
        source,
        feature_cols=("sales", "year", "month", "week", "day"),
        k=k,
        group_cols=tuple(target.attrs["protocol_group_cols"]),
        consumer_source_df=source,
    )
    meta = selected["meta"]
    return {
        "candidate_keys": context.candidate_keys,
        "sources": selected["sources"],
        "candidate_pool_digest": meta["candidate_pool_digest"],
        "selection_digest": meta["selection_digest"],
        "source_pool_fingerprint": meta["source_pool_fingerprint"],
        "eligible_candidate_keys": meta["eligible_candidate_keys"],
        "valid_30d_candidate_keys": meta["valid_30d_candidate_keys"],
    }


@pytest.mark.parametrize("dataset_id", ["D1", "D2", "D3"])
@pytest.mark.parametrize("scenario", ["without", "with"])
def test_cell_hoist_reduces_only_source_constructors_and_preserves_exact_parity(
    monkeypatch: pytest.MonkeyPatch,
    dataset_id: str,
    scenario: str,
) -> None:
    raw_source, target, group_cols, observed_start = _cell_frames(dataset_id)
    source = _mode_source(dataset_id, raw_source, scenario)
    calls = {"index": 0, "pool": 0, "verify": 0, "candidate": 0}
    real_index = runner_adapter.build_canonical_source_index
    real_pool = runner_adapter.prepare_daily_sequence_pool
    real_verify = runner_adapter.verify_d2_source_frame
    real_candidate = runner_adapter._strict_raw_candidates

    def count_index(*args, **kwargs):
        calls["index"] += 1
        return real_index(*args, **kwargs)

    def count_pool(*args, **kwargs):
        calls["pool"] += 1
        return real_pool(*args, **kwargs)

    def count_verify(*args, **kwargs):
        calls["verify"] += 1
        return real_verify(*args, **kwargs)

    def count_candidate(*args, **kwargs):
        calls["candidate"] += 1
        return real_candidate(*args, **kwargs)

    monkeypatch.setattr(runner_adapter, "build_canonical_source_index", count_index)
    monkeypatch.setattr(runner_adapter, "prepare_daily_sequence_pool", count_pool)
    monkeypatch.setattr(runner_adapter, "verify_d2_source_frame", count_verify)
    monkeypatch.setattr(runner_adapter, "_strict_raw_candidates", count_candidate)

    legacy_sources, legacy_targets = _configured_six(
        dataset_id, source, target, group_cols, observed_start, scenario
    )
    assert calls["index"] == 6
    assert calls["pool"] == 6
    assert calls["verify"] == (6 if dataset_id == "D2" else 0)
    assert calls["candidate"] == 6

    calls.update(index=0, pool=0, verify=0, candidate=0)
    prepared_source, shared_pool = runner_adapter.prepare_protocol_source_pool(
        source,
        dataset_id=dataset_id,
        scenario=scenario,
        group_cols=group_cols,
        observed_start=observed_start,
    )
    new_sources, new_targets = _configured_six(
        dataset_id,
        prepared_source,
        target,
        group_cols,
        observed_start,
        scenario,
        prepared_pool=shared_pool,
    )
    assert calls["index"] == 1
    assert calls["pool"] == 1
    assert calls["verify"] == (1 if dataset_id == "D2" else 0)
    assert calls["candidate"] == 6

    legacy_contexts = [get_protocol_frame_context(frame) for frame in legacy_sources]
    new_contexts = [get_protocol_frame_context(frame) for frame in new_sources]
    assert len({id(context) for context in legacy_contexts}) == 6
    assert len({id(context) for context in new_contexts}) == 6
    assert all(context.prepared_pool is shared_pool for context in new_contexts)
    assert all(context.source_index is shared_pool.source_index for context in new_contexts)
    assert len({id(frame) for frame in new_sources}) == 6
    assert len({id(frame) for frame in new_targets}) == 6
    if dataset_id == "D2":
        assert len({id(context.protocol_report) for context in new_contexts}) == 6

    for legacy_source, legacy_target, new_source, new_target in zip(
        legacy_sources, legacy_targets, new_sources, new_targets
    ):
        legacy_context = get_protocol_frame_context(legacy_source)
        new_context = get_protocol_frame_context(new_source)
        assert legacy_context.candidate_keys == new_context.candidate_keys
        _assert_frame_equal_without_attrs(
            get_configured_knn_frame(legacy_source, "source"),
            get_configured_knn_frame(new_source, "source"),
        )
        _assert_frame_equal_without_attrs(
            get_configured_knn_frame(legacy_target, "target"),
            get_configured_knn_frame(new_target, "target"),
        )
        assert legacy_source.attrs["source_frame_digest"] == new_source.attrs[
            "source_frame_digest"
        ]
        assert legacy_target.attrs["target_frame_digest"] == new_target.attrs[
            "target_frame_digest"
        ]
        assert _manifest(legacy_target) == _manifest(new_target)
        _assert_frame_equal_without_attrs(
            legacy_context.forecast_frame,
            new_context.forecast_frame,
        )
        if dataset_id == "D2":
            assert legacy_source.attrs["d2_source_authority_digest"] == new_source.attrs[
                "d2_source_authority_digest"
            ]
            assert legacy_source.attrs["d2_consumer_frame_fingerprint"] == new_source.attrs[
                "d2_consumer_frame_fingerprint"
            ]

    for method_index, k in enumerate((1, 3, 3, 3, 3), start=1):
        assert _selection_snapshot(
            legacy_sources[method_index], legacy_targets[method_index], k=k
        ) == _selection_snapshot(new_sources[method_index], new_targets[method_index], k=k)

    original = float(shared_pool.sales_matrix[0, 0])
    new_sources[0].loc[new_sources[0].index[0], "sales"] = -999.0
    assert float(shared_pool.sales_matrix[0, 0]) == original
    assert not np.shares_memory(
        new_sources[0]["sales"].to_numpy(), new_sources[1]["sales"].to_numpy()
    )
    assert not shared_pool.sales_matrix.flags.writeable
    assert not shared_pool.date_presence_matrix.flags.writeable


def test_d3_without_keeps_29_store_source_domain_and_9_store_candidate_scope() -> None:
    source, target, group_cols, observed_start = _cell_frames("D3")
    prepared_source, pool = runner_adapter.prepare_protocol_source_pool(
        source,
        dataset_id="D3",
        scenario="without",
        group_cols=group_cols,
        observed_start=observed_start,
    )
    configured_source, _ = _configured_six(
        "D3",
        prepared_source,
        target,
        group_cols,
        observed_start,
        "without",
        prepared_pool=pool,
    )
    context = get_protocol_frame_context(configured_source[0])
    observed = get_configured_knn_frame(configured_source[0], "source")
    assert observed["store_id"].astype(str).nunique() == 29
    assert len(context.candidate_keys) == 9
    assert context.candidate_keys == tuple((str(store),) for store in range(1, 10))


def test_d2_knn_model_schema_and_forecast_boundary_remain_separate() -> None:
    source, target, group_cols, observed_start = _cell_frames("D2")
    prepared_source, pool = runner_adapter.prepare_protocol_source_pool(
        source,
        dataset_id="D2",
        scenario="with",
        group_cols=group_cols,
        observed_start=observed_start,
    )
    configured_sources, configured_targets = _configured_six(
        "D2",
        prepared_source,
        target,
        group_cols,
        observed_start,
        "with",
        prepared_pool=pool,
    )
    context = get_protocol_frame_context(configured_sources[0])
    assert tuple(pool.feature_matrices) == ("sales", "promo")
    assert full_runner._resolve_dataset_feature_cols(
        "Dataset2", source, target, {}
    ) == [
        "sales",
        "year",
        "month",
        "week",
        "day",
    ]
    assert "promo" in get_configured_knn_frame(configured_sources[0], "source").columns
    assert "promo" not in context.forecast_frame.columns
    assert configured_targets[0].loc[
        configured_targets[0]["date"] > pd.Timestamp("2018-06-30"), "promo"
    ].isna().all()


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("identity", "missing required candidate keys"),
        ("date", "exact 180 days"),
        ("calendarization", "outside approved"),
        ("knn_schema", "prepared-pool columns"),
        ("group_columns", "canonical-index columns"),
    ],
)
def test_d2_cell_preparation_fails_closed_on_incompatible_source(
    mutation: str,
    error: str,
) -> None:
    source, _, group_cols, observed_start = _cell_frames("D2")
    if mutation == "identity":
        source = source.loc[~((source["brand_id"] == 3) & (source["item_id"] == 9))].copy()
    elif mutation == "date":
        source = source.loc[source["date"] != pd.Timestamp("2018-04-01")].copy()
    elif mutation == "calendarization":
        source.loc[source.index[0], "day"] = 99
    elif mutation == "knn_schema":
        source = source.drop(columns=["promo"])
    else:
        source = source.drop(columns=["brand_id"])
    source.attrs["split_role"] = "source"

    with pytest.raises((ProtocolViolation, KeyError), match=error):
        runner_adapter.prepare_protocol_source_pool(
            source,
            dataset_id="D2",
            scenario="with",
            group_cols=group_cols,
            observed_start=observed_start,
        )


def test_d2_promo_availability_fails_closed_before_selection() -> None:
    source, target, group_cols, observed_start = _cell_frames("D2")
    source.loc[source["date"] == pd.Timestamp("2018-06-01"), "promo"] = np.nan
    prepared_source, pool = runner_adapter.prepare_protocol_source_pool(
        source,
        dataset_id="D2",
        scenario="with",
        group_cols=group_cols,
        observed_start=observed_start,
    )
    with pytest.raises(ProtocolViolation, match="promo.*non-numeric"):
        runner_adapter.configure_protocol_frames(
            prepared_source,
            target,
            dataset_id="D2",
            scenario="with",
            group_cols=group_cols,
            observed_start=observed_start,
            prepared_pool=pool,
            retain_source_frame=True,
            enforce_formal_target=True,
        )


def test_changed_d2_source_bytes_create_new_cell_dependencies() -> None:
    source, _, group_cols, observed_start = _cell_frames("D2")
    first_source, first_pool = runner_adapter.prepare_protocol_source_pool(
        source,
        dataset_id="D2",
        scenario="with",
        group_cols=group_cols,
        observed_start=observed_start,
    )
    changed = source.copy()
    changed.attrs = deepcopy(source.attrs)
    changed.loc[
        (changed["brand_id"] == 1)
        & (changed["item_id"] == 1)
        & (changed["date"] == pd.Timestamp("2018-06-01")),
        "sales",
    ] += 1.0
    second_source, second_pool = runner_adapter.prepare_protocol_source_pool(
        changed,
        dataset_id="D2",
        scenario="with",
        group_cols=group_cols,
        observed_start=observed_start,
    )
    assert first_pool is not second_pool
    assert first_pool.source_index is not second_pool.source_index
    assert not np.array_equal(first_pool.sales_matrix, second_pool.sales_matrix)
    assert first_source.attrs["d2_source_authority_digest"] != second_source.attrs[
        "d2_source_authority_digest"
    ]


def test_cell_identity_prevents_cross_mode_horizon_and_seed_reuse() -> None:
    source, target, _, _ = _cell_frames("D1")
    base = {"source_df": source, "target_df": target}
    protocol = {"strict_paper_mode": True}

    def prepare(mode: str, horizon: int, seed: int):
        return full_runner._prepare_cell_source_dependencies(
            dataset_name="Dataset1",
            information_sharing_scenario=mode,
            cfg={"single_experiment": {"horizon": horizon, "seed": seed}},
            protocol=protocol,
            strict_paper_mode=True,
            base_data=base,
        )

    cells = (
        prepare("without_information_sharing", 1, 42),
        prepare("with_information_sharing", 1, 42),
        prepare("without_information_sharing", 2, 42),
        prepare("without_information_sharing", 1, 43),
    )
    assert len({id(cell.prepared_pool) for cell in cells}) == 4
    assert len({id(cell.prepared_pool.source_index) for cell in cells}) == 4
    assert len({cell.cell_identity for cell in cells}) == 4


def test_six_method_dispatch_reuses_only_explicit_cell_source_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, target, _, _ = _cell_frames("D1")
    base = {"source_df": source, "target_df": target}
    cfg = {
        "dataset_paths": {},
        "single_experiment": {"horizon": 1, "seed": 42},
    }
    protocol = {"strict_paper_mode": True}
    preparation_counts = {"filter": 0, "source_projection": 0}
    real_filter = full_runner._apply_information_sharing_filter
    real_projection = full_runner._project_modeling_frame

    def count_filter(*args, **kwargs):
        preparation_counts["filter"] += 1
        return real_filter(*args, **kwargs)

    def count_projection(frame, keep_cols):
        if frame is not target:
            preparation_counts["source_projection"] += 1
        return real_projection(frame, keep_cols)

    monkeypatch.setattr(full_runner, "_apply_information_sharing_filter", count_filter)
    monkeypatch.setattr(full_runner, "_project_modeling_frame", count_projection)
    cell = full_runner._prepare_cell_source_dependencies(
        dataset_name="Dataset1",
        information_sharing_scenario="with_information_sharing",
        cfg=cfg,
        protocol=protocol,
        strict_paper_mode=True,
        base_data=base,
    )
    assert preparation_counts == {"filter": 1, "source_projection": 1}
    monkeypatch.setattr(full_runner, "_project_modeling_frame", real_projection)

    class StopAtConfigure(RuntimeError):
        pass

    forwarded = []

    def capture_configure(source_carrier, target_carrier, **kwargs):
        forwarded.append(
            (source_carrier, target_carrier, kwargs["prepared_pool"])
        )
        raise StopAtConfigure

    monkeypatch.setattr(full_runner, "_load_experiment_runners", lambda: None)
    monkeypatch.setattr(full_runner, "configure_protocol_frames", capture_configure)
    monkeypatch.setattr(
        full_runner,
        "_apply_information_sharing_filter",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("method dispatch repeated source filtering")
        ),
    )

    for method in METHODS:
        with pytest.raises(StopAtConfigure):
            full_runner.run_experiment(
                dataset_name="Dataset1",
                method_name=method,
                source_count=3 if method not in {"No-TL", "SS-TL"} else 1,
                information_sharing_scenario="with_information_sharing",
                cfg=cfg,
                protocol=protocol,
                strict_paper_mode=True,
                base_data=base,
                cell_source_preparation=cell,
            )

    assert len(forwarded) == 6
    assert all(pool is cell.prepared_pool for _, _, pool in forwarded)
    assert len({id(source_carrier) for source_carrier, _, _ in forwarded}) == 6
    assert len({id(target_carrier) for _, target_carrier, _ in forwarded}) == 6
    assert preparation_counts["filter"] == 1


def _d3_production_cfg() -> dict[str, object]:
    return {
        "dataset_paths": {},
        "single_experiment": {
            "horizon": 1,
            "seed": 42,
            "window_size": 10,
            "learning_rate": 0.001,
            "source_epochs": 1,
            "target_epochs": 1,
            "batch_size": 8,
            "weight_mode": "inverse_distance",
            "keep_ratio": 0.5,
        },
    }


def _d3_cell(scenario: str) -> tuple[dict[str, pd.DataFrame], full_runner.CellSourcePreparation]:
    source, target, _, _ = _cell_frames("D3")
    base = {"source_df": source, "target_df": target}
    cell = full_runner._prepare_cell_source_dependencies(
        dataset_name="Dataset3",
        information_sharing_scenario=scenario,
        cfg=_d3_production_cfg(),
        protocol={"strict_paper_mode": True},
        strict_paper_mode=True,
        base_data=base,
    )
    return base, cell


def _assert_canonical_d3_cell_attrs(frame: pd.DataFrame, scenario: str) -> None:
    expected = ("D3", scenario, 1, 42, CANONICAL_D3_TARGET)
    assert frame.attrs["protocol_dataset_id"] == "D3"
    assert frame.attrs["protocol_scenario"] == scenario
    assert tuple(frame.attrs["protocol_cell_identity"]) == expected
    assert frame.attrs["model_horizon"] == 1
    assert frame.attrs["protocol_seed"] == 42
    assert tuple(frame.attrs["protocol_target_key"]) == CANONICAL_D3_TARGET


@pytest.mark.parametrize(
    ("scenario_alias", "canonical_scenario"),
    [
        ("with_information_sharing", "with"),
        ("without_information_sharing", "without"),
    ],
)
def test_d3_ss_production_wiring_canonicalizes_cell_before_transformation_reuse(
    monkeypatch: pytest.MonkeyPatch,
    scenario_alias: str,
    canonical_scenario: str,
) -> None:
    base, cell = _d3_cell(scenario_alias)
    expected = ("D3", canonical_scenario, 1, 42, CANONICAL_D3_TARGET)

    class IdentityVerified(RuntimeError):
        pass

    def verify_ss(**kwargs):
        source_df = kwargs["source_df"]
        target_df = kwargs["target_df"]
        context = kwargs["transformation_reuse_context"]
        _assert_canonical_d3_cell_attrs(source_df, canonical_scenario)
        _assert_canonical_d3_cell_attrs(target_df, canonical_scenario)
        assert context.lifecycle_identity == expected

        one_source = source_df.loc[source_df["store_id"].eq(1)].iloc[:18].copy()
        one_source.attrs = deepcopy(source_df.attrs)
        one_source.attrs.update(
            {
                "split_role": "source",
                "split_mode": "ratio",
                "split_config": {
                    "train_ratio": 10 / 18,
                    "val_ratio": 4 / 18,
                    "test_ratio": 4 / 18,
                },
            }
        )
        partitions = []
        for role, bounds in (
            ("train", (0, 10)),
            ("validation", (10, 14)),
            ("test", (14, 18)),
        ):
            part = one_source.iloc[slice(*bounds)].copy()
            part.attrs = deepcopy(one_source.attrs)
            part.attrs["temporal_partition"] = role
            partitions.append(part)
        normalize_features(
            *partitions,
            feature_columns=("sales", "year"),
            reuse_context=context,
        )
        raise IdentityVerified

    monkeypatch.setattr(full_runner, "_load_experiment_runners", lambda: None)
    monkeypatch.setattr(full_runner, "run_ss_tl_experiment", verify_ss, raising=False)
    with pytest.raises(IdentityVerified):
        full_runner.run_experiment(
            dataset_name="Dataset3",
            method_name="SS-TL",
            source_count=1,
            information_sharing_scenario=scenario_alias,
            cfg=_d3_production_cfg(),
            protocol={"strict_paper_mode": True},
            strict_paper_mode=True,
            base_data=base,
            cell_source_preparation=cell,
        )


@pytest.mark.parametrize(
    ("scenario_alias", "canonical_scenario"),
    [
        ("with_information_sharing", "with"),
        ("without_information_sharing", "without"),
    ],
)
def test_d3_four_multisource_production_paths_compare_canonical_raw_scenario(
    monkeypatch: pytest.MonkeyPatch,
    scenario_alias: str,
    canonical_scenario: str,
) -> None:
    base, cell = _d3_cell(scenario_alias)

    class IdentityVerified(RuntimeError):
        pass

    def verify_raw(**kwargs):
        source_df = kwargs["source_df"]
        target_df = kwargs["target_df"]
        raw_context = kwargs["k3_raw_source_context"]
        _assert_canonical_d3_cell_attrs(source_df, canonical_scenario)
        _assert_canonical_d3_cell_attrs(target_df, canonical_scenario)
        selection = {
            "sources": [
                {"source_key": (str(store),), "distance": float(store), "weight": 1 / 3}
                for store in (1, 2, 3)
            ],
            "meta": {
                "requested_k": 3,
                "effective_k": 3,
                "candidate_pool_digest": "production-path-candidates",
                "selection_result_digest": "production-path-selection",
            },
        }
        raw_context.working_source(
            selection_result=selection,
            source_df=source_df,
            source_key=("1",),
            group_cols=("store_id",),
            model_feature_cols=kwargs["feature_cols"],
        )
        raise IdentityVerified

    monkeypatch.setattr(full_runner, "_load_experiment_runners", lambda: None)
    for runner_name in (
        "run_mswa_experiment",
        "run_mssb_experiment",
        "run_msml_experiment",
        "run_msml_rfe_experiment",
    ):
        monkeypatch.setattr(full_runner, runner_name, verify_raw, raising=False)

    for method in METHODS[2:]:
        with pytest.raises(IdentityVerified):
            full_runner.run_experiment(
                dataset_name="Dataset3",
                method_name=method,
                source_count=3,
                information_sharing_scenario=scenario_alias,
                cfg=_d3_production_cfg(),
                protocol={"strict_paper_mode": True},
                strict_paper_mode=True,
                base_data=base,
                cell_source_preparation=cell,
            )


@pytest.mark.parametrize("identity_index,bad_value", [(2, 2), (3, 43), (4, ("11",))])
def test_d3_production_wiring_rejects_wrong_transformation_owner_identity(
    monkeypatch: pytest.MonkeyPatch,
    identity_index: int,
    bad_value: object,
) -> None:
    base, cell = _d3_cell("with_information_sharing")
    lifecycle = list(cell.transformation_reuse_context.lifecycle_identity)
    lifecycle[identity_index] = bad_value
    bad_cell = replace(
        cell,
        transformation_reuse_context=TargetTransformationReuseContext(tuple(lifecycle)),
    )
    monkeypatch.setattr(full_runner, "_load_experiment_runners", lambda: None)

    with pytest.raises(ProtocolViolation, match="cell context identity mismatch"):
        full_runner.run_experiment(
            dataset_name="Dataset3",
            method_name="SS-TL",
            source_count=1,
            information_sharing_scenario="with_information_sharing",
            cfg=_d3_production_cfg(),
            protocol={"strict_paper_mode": True},
            strict_paper_mode=True,
            base_data=base,
            cell_source_preparation=bad_cell,
        )


def test_formal_lifecycle_cardinalities_are_unchanged() -> None:
    assert len(full_runner.DATASETS) == 3
    assert len(full_runner.INFO_SHARING_SCENARIOS) == 2
    assert FORMAL_HORIZONS == (1, 2, 3, 4, 5)
    assert FORMAL_SEEDS == (42, 43, 44, 45, 46)
    assert len(full_runner.METHODS) == 6
    assert 6 * 2 * len(FORMAL_HORIZONS) * len(FORMAL_SEEDS) == 300


@pytest.mark.parametrize("dataset_id", ["D1", "D2", "D3"])
@pytest.mark.parametrize("scenario", ["without", "with"])
def test_d1_d3_target_local_k3_core_matches_four_legacy_selectors_exactly(
    dataset_id: str,
    scenario: str,
) -> None:
    raw_source, target, group_cols, observed_start = _cell_frames(dataset_id)
    source = _mode_source(dataset_id, raw_source, scenario)
    prepared_source, pool = runner_adapter.prepare_protocol_source_pool(
        source,
        dataset_id=dataset_id,
        scenario=scenario,
        group_cols=group_cols,
        observed_start=observed_start,
    )
    configured_source, configured_target = runner_adapter.configure_protocol_frames(
        prepared_source,
        target,
        dataset_id=dataset_id,
        scenario=scenario,
        group_cols=group_cols,
        observed_start=observed_start,
        prepared_pool=pool,
        retain_source_frame=True,
        enforce_formal_target=True,
    )
    model_features = ("sales", "year", "month", "week", "day")

    def select(k: int):
        return SourceSelector().select_top_k_sources(
            configured_target,
            configured_source,
            feature_cols=model_features,
            k=k,
            group_cols=group_cols,
        )

    legacy_k3 = [select(3) for _ in range(4)]
    k1 = select(1)
    lifecycle = (
        dataset_id,
        scenario,
        1,
        42,
        tuple(configured_target.attrs["protocol_target_key"]),
    )
    context = TargetK3SelectionContext(lifecycle)
    wrappers = []
    for _method in METHODS[2:]:
        evidence = context.selection_for_method(
            target_df=configured_target,
            source_df=configured_source,
            feature_cols=model_features,
            group_cols=group_cols,
            k=3,
            weight_mode="inverse_distance",
        )
        wrappers.append(
            evidence.method_wrapper(
                lifecycle_identity=lifecycle,
                target_df=configured_target,
                source_df=configured_source,
                feature_cols=model_features,
                group_cols=group_cols,
                k=3,
                weight_mode="inverse_distance",
            )
        )

    assert legacy_k3[0] == legacy_k3[1] == legacy_k3[2] == legacy_k3[3]
    assert all(wrapper == legacy_k3[0] for wrapper in wrappers)
    assert len({id(wrapper) for wrapper in wrappers}) == 4
    assert k1["meta"]["selection_result_digest"] != legacy_k3[0]["meta"][
        "selection_result_digest"
    ]
    expected_candidates = {
        ("D1", "without"): 9,
        ("D1", "with"): 27,
        ("D2", "without"): 9,
        ("D2", "with"): 27,
        ("D3", "without"): 9,
        ("D3", "with"): 29,
    }[(dataset_id, scenario)]
    runtime_context = get_protocol_frame_context(configured_source)
    assert len(runtime_context.candidate_keys) == expected_candidates
    if dataset_id == "D3" and scenario == "without":
        assert len(pool.source_keys) == 29
        assert len(runtime_context.candidate_keys) == 9
    if dataset_id == "D2":
        assert tuple(pool.feature_matrices) == ("sales", "promo")
        assert legacy_k3[0]["meta"]["d2_source_calendarization_rule_version"]
        assert wrappers[0]["meta"]["d2_sealed_identity"] == legacy_k3[0]["meta"][
            "d2_sealed_identity"
        ]
