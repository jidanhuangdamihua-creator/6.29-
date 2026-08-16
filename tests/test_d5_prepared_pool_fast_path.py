from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

import src.protocols.runner_adapter as runner_adapter
from scripts import run_d4_experiment, run_d5_experiment, run_d6_experiment
from src.data_processing.data_preprocessing import normalize_features
from src.protocols.candidate_pool import prepare_daily_sequence_pool
from src.protocols.experiment_protocol import ProtocolViolation, get_experiment_protocol
from src.protocols.runner_adapter import configure_protocol_frames
from src.protocols.source_history import build_exact_source_history_candidate_frame
from src.protocols.knn_frames import build_observed_knn_frame
from src.protocols.rolling_origin import build_sample_manifest
from src.source_selection.source_selector import SourceSelector, TargetK3SelectionContext
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


@pytest.mark.parametrize("scenario", ["without", "with"])
def test_d5_entity_production_wiring_binds_canonical_transformation_owner(
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
) -> None:
    source, target = _d5_frames()
    expected = ("D5", scenario, 1, 42, ("48", "1159415"))

    class IdentityVerified(RuntimeError):
        pass

    def verify_ss(**kwargs):
        source_df = kwargs["source_df"]
        target_df = kwargs["target_df"]
        context = kwargs["transformation_reuse_context"]
        for frame in (source_df, target_df):
            assert frame.attrs["protocol_dataset_id"] == "D5"
            assert frame.attrs["protocol_scenario"] == scenario
            assert tuple(frame.attrs["protocol_cell_identity"]) == expected
            assert frame.attrs["model_horizon"] == 1
            assert frame.attrs["protocol_seed"] == 42
            assert tuple(frame.attrs["protocol_target_key"]) == ("48", "1159415")
        assert context.lifecycle_identity == expected

        first_group = source_df.loc[
            source_df["store_nbr"].astype(str).eq("48")
            & source_df["item_nbr"].astype(str).eq("S1")
        ].copy()
        first_group.attrs = deepcopy(source_df.attrs)
        first_group.attrs.update(
            {
                "split_role": "source",
                "split_mode": "ratio",
                "split_config": {
                    "train_ratio": 0.6,
                    "val_ratio": 0.2,
                    "test_ratio": 0.2,
                },
            }
        )
        partitions = []
        for role, bounds in (
            ("train", (0, 21)),
            ("validation", (21, 28)),
            ("test", (28, 35)),
        ):
            partition = first_group.iloc[slice(*bounds)].copy()
            partition.attrs = deepcopy(first_group.attrs)
            partition.attrs["temporal_partition"] = role
            partitions.append(partition)
        normalize_features(
            *partitions,
            feature_columns=("sales",),
            reuse_context=context,
        )
        raise IdentityVerified

    monkeypatch.setattr(entity_experiment, "_method_runner", lambda method: verify_ss)
    with pytest.raises(IdentityVerified):
        entity_experiment.run_single_entity_experiment(
            entity_key="48_1159415",
            source_df=source,
            target_entity_df=target,
            feature_cols=[
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
            ],
            config={
                "dataset_id": 5,
                "dataset_name": "Dataset5",
                "info_sharing": scenario,
                "group_cols": ("store_nbr", "item_nbr"),
                "source_count": 1,
                "horizon": 1,
                "seed": 42,
                "window_size": 3,
                "learning_rate": 0.001,
                "source_epochs": 1,
                "target_epochs": 1,
                "batch_size": 1,
            },
            enabled_methods=["SS-TL"],
        )


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


@pytest.mark.parametrize(
    ("dataset_id", "group_cols"),
    [
        (4, ("store_id", "product_id")),
        (6, ("store_id", "item_id")),
    ],
)
def test_d4_d6_entity_runner_forwards_explicit_cell_pool(
    monkeypatch: pytest.MonkeyPatch,
    dataset_id: int,
    group_cols: tuple[str, str],
) -> None:
    dates = pd.date_range("2020-01-01", periods=2, freq="D")
    source = pd.DataFrame(
        {
            group_cols[0]: ["S", "S"],
            group_cols[1]: ["I", "I"],
            "date": dates,
            "sales": [1.0, 2.0],
        }
    )
    target = source.copy()
    forwarded: list[object] = []
    shared_pool = object()

    class _StopAfterConfigure(Exception):
        pass

    def stop_after_configure(*args, **kwargs):
        forwarded.append(kwargs.get("prepared_pool"))
        raise _StopAfterConfigure

    monkeypatch.setattr(entity_experiment, "configure_protocol_frames", stop_after_configure)

    with pytest.raises(_StopAfterConfigure):
        entity_experiment.run_single_entity_experiment(
            entity_key="target",
            source_df=source,
            target_entity_df=target,
            feature_cols=["sales"],
            config={
                "dataset_id": dataset_id,
                "info_sharing": "without",
                "group_cols": group_cols,
            },
            enabled_methods=["No-TL"],
            prepared_pool=shared_pool,
        )

    assert forwarded == [shared_pool]


def test_d5_cell_builds_one_pool_for_five_targets_without_cross_mode_reuse(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    source, target_template = _d5_frames()
    target_keys = [
        "48_364606",
        "48_1159415",
        "48_1159414",
        "48_1349808",
        "48_320682",
    ]
    targets = []
    for entity_key in target_keys:
        item_nbr = entity_key.split("_", 1)[1]
        frame = target_template.copy()
        frame.loc[:, "item_nbr"] = item_nbr
        frame.loc[:, "item_id"] = item_nbr
        frame.loc[:, "entity_id"] = entity_key
        targets.append(frame)
    target = pd.concat(targets, ignore_index=True)
    target.attrs.update(target_template.attrs)

    prepare_calls: list[object] = []
    forwarded: list[object] = []
    real_prepare = run_d5_experiment.prepare_daily_sequence_pool
    cell = {"mode": "without", "output_dir": tmp_path / "without"}

    def recording_prepare(*args, **kwargs):
        pool = real_prepare(*args, **kwargs)
        prepare_calls.append(pool)
        return pool

    def recording_entity_runner(**kwargs):
        forwarded.append(kwargs["prepared_pool"])
        return [
                {
                "target_entity_key": kwargs["entity_key"],
                "method": "No-TL",
                "smape": 0.0,
                "rmse": 0.0,
            }
        ]

    monkeypatch.setattr(
        run_d5_experiment,
        "_parse_args",
        lambda: type(
            "Args",
            (),
            {
                "formal_source_path": None,
                "formal_target_path": None,
                    "info_sharing": cell["mode"],
                "smoke": True,
                "target_limit": 5,
                "target_keys": target_keys,
                "source_limit": None,
                "epochs": 1,
                    "output_dir": cell["output_dir"],
                "repair_source_numeric_na": False,
                "horizon": 1,
                "seed": 42,
            },
        )(),
    )
    monkeypatch.setattr(
        run_d5_experiment,
        "resolve_formal_dataset_paths",
        lambda *args, **kwargs: type(
            "Paths", (), {"source_path": tmp_path / "source", "target_path": tmp_path / "target"}
        )(),
    )
    monkeypatch.setattr(run_d5_experiment, "reserve_new_output_dir", lambda path: path.mkdir())
    monkeypatch.setattr(
        run_d5_experiment,
        "load_knn_results",
        lambda *args, **kwargs: {
            "group_cols": ["store_nbr", "item_nbr"],
            "results": {key: [] for key in target_keys},
        },
    )
    monkeypatch.setattr(run_d5_experiment, "read_dataset_windows", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        run_d5_experiment,
        "load_d5_runtime_inputs",
        lambda **kwargs: type(
            "Inputs",
            (),
            {
                "source_df": source,
                "target_df": target,
                "calendar_reconstruction": type("Report", (), {"to_dict": lambda self: {}})(),
            },
        )(),
    )
    monkeypatch.setattr(
        run_d5_experiment,
        "resolve_knn_feature_columns",
        lambda **kwargs: {
            "selected_features": [
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
            ],
            "feature_consistency_status": "aligned",
            "feature_source": "test",
            "knn_feature_mode": "test",
            "json_only_features": [],
            "runtime_only_features": [],
        },
    )
    monkeypatch.setattr(run_d5_experiment, "apply_runtime_source_domain_policy", lambda frame, *args: frame)
    monkeypatch.setattr(run_d5_experiment, "prepare_daily_sequence_pool", recording_prepare)
    monkeypatch.setattr(run_d5_experiment, "run_single_entity_experiment", recording_entity_runner)
    monkeypatch.setattr(run_d5_experiment, "_align_results_to_reference_schema", lambda frame: frame)
    monkeypatch.setattr(run_d5_experiment, "config", dict(run_d5_experiment.config))

    run_d5_experiment.main()
    cell.update({"mode": "with", "output_dir": tmp_path / "with"})
    run_d5_experiment.main()

    assert len(prepare_calls) == 2
    assert len(forwarded) == 10
    assert all(pool is prepare_calls[0] for pool in forwarded[:5])
    assert all(pool is prepare_calls[1] for pool in forwarded[5:])
    assert prepare_calls[0] is not prepare_calls[1]


def test_shared_d5_pool_preserves_per_target_selection_and_manifest_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, target_template = _d5_frames()
    targets = []
    for item_nbr in ("364606", "1159415", "1159414", "1349808", "320682"):
        target = target_template.copy()
        target.loc[:, "item_nbr"] = item_nbr
        target.loc[:, "item_id"] = item_nbr
        target.loc[:, "entity_id"] = f"48_{item_nbr}"
        targets.append(target)

    protocol = get_experiment_protocol(5)
    shared_pool = prepare_daily_sequence_pool(
        source,
        group_cols=("store_nbr", "item_nbr"),
        observed_start="2020-01-01",
        metadata_cols=("family",),
        feature_cols=protocol.knn_feature_columns,
    )
    proofs: list[object] = []
    candidate_scopes: list[object] = []
    real_classify = runner_adapter.classify_prepared_candidate_dates
    real_candidates = runner_adapter._extended_candidates_from_identities

    def recording_classify(*args, **kwargs):
        proof = real_classify(*args, **kwargs)
        proofs.append(proof)
        return proof

    def recording_candidates(*args, **kwargs):
        candidates = real_candidates(*args, **kwargs)
        candidate_scopes.append(candidates)
        return candidates

    monkeypatch.setattr(runner_adapter, "classify_prepared_candidate_dates", recording_classify)
    monkeypatch.setattr(runner_adapter, "_extended_candidates_from_identities", recording_candidates)

    def snapshot(target: pd.DataFrame, pool) -> dict[str, object]:
        configured_source, configured_target = configure_protocol_frames(
            source,
            target,
            dataset_id=5,
            scenario="without",
            group_cols=("store_nbr", "item_nbr"),
            grouping_col="family",
            observed_start="2020-01-01",
            prepared_pool=pool,
            retain_source_frame=True,
            enforce_formal_target=True,
        )
        selection = SourceSelector().select_top_k_sources(
            configured_target,
            configured_source,
            feature_cols=protocol.knn_feature_columns,
            k=1,
            group_cols=("store_nbr", "item_nbr"),
            consumer_source_df=source,
        )
        manifest = build_sample_manifest(
            configured_target,
            dataset_id="D5",
            track=configured_target.attrs["protocol_track"],
            scenario="without",
            target_key=configured_target.attrs["protocol_target_key"],
            observed_end=configured_target.attrs["knn_observed_end"],
            first_forecast_origin=pd.Timestamp(configured_target.attrs["knn_observed_end"])
            + pd.Timedelta(days=3),
            input_window=3,
        )
        return {
            "candidate_keys": get_protocol_frame_context(configured_source).candidate_keys,
            "selected": selection["meta"]["selected_sources_runtime"],
            "candidate_digest": selection["meta"]["candidate_pool_digest"],
            "selection_digest": selection["meta"]["selection_result_digest"],
            "target_digest": selection["meta"]["target_frame_digest"],
            "sample_manifest": manifest.digest,
        }

    legacy = []
    for target in targets:
        legacy.append(
            snapshot(
                target,
                prepare_daily_sequence_pool(
                    source,
                    group_cols=("store_nbr", "item_nbr"),
                    observed_start="2020-01-01",
                    metadata_cols=("family",),
                    feature_cols=protocol.knn_feature_columns,
                ),
            )
        )
    proofs.clear()
    candidate_scopes.clear()
    shared = [snapshot(target, shared_pool) for target in targets]

    assert shared == legacy
    assert len(proofs) == 5
    assert len(candidate_scopes) == 5
    assert len({id(proof) for proof in proofs}) == 5
    assert all(proof.pool_id == id(shared_pool) for proof in proofs)
    assert [proof.candidate_scope for proof in proofs] == [
        (("48", "S1"),),
    ] * 5
    assert candidate_scopes == [(("48", "S1"),)] * 5
    assert len({item["candidate_digest"] for item in shared}) == 5
    assert len({item["selection_digest"] for item in shared}) == 5
    assert len({item["target_digest"] for item in shared}) == 5
    assert len({item["sample_manifest"] for item in shared}) == 5


def test_prepared_pool_arrays_and_selected_consumers_are_isolated() -> None:
    source, _ = _d5_frames()
    pool = prepare_daily_sequence_pool(
        source,
        group_cols=("store_nbr", "item_nbr"),
        observed_start="2020-01-01",
        metadata_cols=("family",),
        feature_cols=("sales", "onpromotion", "oil_price"),
    )

    assert not pool.sales_matrix.flags.writeable
    assert not pool.date_presence_matrix.flags.writeable
    assert all(not matrix.flags.writeable for matrix in pool.feature_matrices.values())
    first = pool.selected_frame(
        (("48", "S1"),), feature_cols=("sales", "onpromotion", "oil_price")
    )
    second = pool.selected_frame(
        (("48", "S1"),), feature_cols=("sales", "onpromotion", "oil_price")
    )
    original = float(pool.sales_matrix[pool.key_to_index[("48", "S1")], 0])
    first.loc[0, "sales"] = -999.0

    assert float(pool.sales_matrix[pool.key_to_index[("48", "S1")], 0]) == original
    assert float(second.loc[0, "sales"]) == original


@pytest.mark.parametrize(
    ("group_cols", "observed_start", "error"),
    [
        (("item_nbr", "store_nbr"), "2020-01-01", "group_cols mismatch"),
        (("store_nbr", "item_nbr"), "2020-01-02", "observation dates differ"),
    ],
)
def test_prepared_pool_rejects_changed_cell_identity(
    group_cols,
    observed_start,
    error,
) -> None:
    source, _ = _d5_frames()
    pool = prepare_daily_sequence_pool(
        source,
        group_cols=("store_nbr", "item_nbr"),
        observed_start="2020-01-01",
        metadata_cols=("family",),
        feature_cols=("sales", "onpromotion", "oil_price"),
    )

    with pytest.raises(ProtocolViolation, match=error):
        pool.validate_for(
            group_cols=group_cols,
            required_dates=pd.date_range(observed_start, periods=30),
        )


def test_prepared_pool_rejects_changed_d5_feature_schema() -> None:
    source, target = _d5_frames()
    sales_only_pool = prepare_daily_sequence_pool(
        source,
        group_cols=("store_nbr", "item_nbr"),
        observed_start="2020-01-01",
        metadata_cols=("family",),
        feature_cols=("sales",),
    )

    with pytest.raises(ProtocolViolation, match="missing declared feature columns"):
        configure_protocol_frames(
            source,
            target,
            dataset_id=5,
            scenario="without",
            group_cols=("store_nbr", "item_nbr"),
            grouping_col="family",
            observed_start="2020-01-01",
            prepared_pool=sales_only_pool,
        )


def test_d5_three_feature_k3_core_is_shared_only_within_each_target() -> None:
    base_source, base_target = _d5_frames()
    additions = []
    for store, item, value in (("50", "S3", 3.0), ("51", "S4", 4.0)):
        frame = base_source.loc[base_source["item_nbr"].eq("S1")].copy()
        frame.loc[:, "store_nbr"] = store
        frame.loc[:, "item_nbr"] = item
        frame.loc[:, "entity_id"] = f"{store}_{item}"
        frame.loc[:, "item_id"] = item
        frame.loc[:, ["sales", "onpromotion", "oil_price"]] = value
        additions.append(frame)
    source = pd.concat([base_source, *additions], ignore_index=True)
    protocol = get_experiment_protocol(5)
    pool = prepare_daily_sequence_pool(
        source,
        group_cols=("store_nbr", "item_nbr"),
        observed_start="2020-01-01",
        metadata_cols=("family",),
        feature_cols=protocol.knn_feature_columns,
    )
    evidences = []
    for target_item in ("364606", "1159415"):
        target = base_target.copy()
        target.loc[:, "item_nbr"] = target_item
        target.loc[:, "item_id"] = target_item
        target.loc[:, "entity_id"] = f"48_{target_item}"
        configured_source, configured_target = configure_protocol_frames(
            source,
            target,
            dataset_id=5,
            scenario="with",
            group_cols=("store_nbr", "item_nbr"),
            grouping_col="family",
            observed_start="2020-01-01",
            prepared_pool=pool,
            retain_source_frame=True,
            enforce_formal_target=True,
        )
        legacy = [
            SourceSelector().select_top_k_sources(
                configured_target,
                configured_source,
                feature_cols=protocol.knn_feature_columns,
                k=3,
                group_cols=("store_nbr", "item_nbr"),
            )
            for _ in range(4)
        ]
        lifecycle = (
            "D5",
            "with",
            1,
            42,
            tuple(configured_target.attrs["protocol_target_key"]),
        )
        context = TargetK3SelectionContext(lifecycle)
        wrappers = []
        for _method in ("MSWA-TL", "MSSB-TL", "MSML-TL", "MSML-TL-RFE"):
            evidence = context.selection_for_method(
                target_df=configured_target,
                source_df=configured_source,
                feature_cols=protocol.knn_feature_columns,
                group_cols=("store_nbr", "item_nbr"),
                k=3,
                weight_mode="inverse_distance",
            )
            wrappers.append(
                evidence.method_wrapper(
                    lifecycle_identity=lifecycle,
                    target_df=configured_target,
                    source_df=configured_source,
                    feature_cols=protocol.knn_feature_columns,
                    group_cols=("store_nbr", "item_nbr"),
                    k=3,
                    weight_mode="inverse_distance",
                )
            )
        assert tuple(pool.feature_matrices) == (
            "sales",
            "onpromotion",
            "oil_price",
        )
        assert legacy[0] == legacy[1] == legacy[2] == legacy[3]
        assert all(wrapper == legacy[0] for wrapper in wrappers)
        assert wrappers[0]["meta"]["feature_cols"] == [
            "sales",
            "onpromotion",
            "oil_price",
        ]
        evidences.append(context._evidence)

    assert evidences[0] is not evidences[1]
    assert evidences[0].target_key != evidences[1].target_key


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
