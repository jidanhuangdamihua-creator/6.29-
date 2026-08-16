from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

import src.protocols.candidate_pool as candidate_pool
import src.protocols.runner_adapter as runner_adapter
from scripts import run_d4_experiment, run_d6_experiment
from src.data_processing.data_preprocessing import normalize_features
from src.protocols.candidate_pool import prepare_daily_sequence_pool
from src.protocols.experiment_protocol import ProtocolViolation, get_experiment_protocol
from src.protocols.rolling_origin import build_sample_manifest
from src.protocols.runner_adapter import configure_protocol_frames
from src.source_selection.source_selector import SourceSelector, TargetK3SelectionContext
from src.protocols.transformation_identity import MODEL_FEATURE_CONTRACTS
from src.utils import entity_experiment
from src.utils.dataframe_attrs import get_protocol_frame_context


def _dataset_case(dataset_id: int) -> dict[str, object]:
    if dataset_id == 4:
        return {
            "runner": run_d4_experiment,
            "group_cols": ("store_id", "product_id"),
            "grouping_col": None,
            "metadata_cols": (),
            "target_keys": (
                "166_258",
                "166_432",
                "166_433",
                "166_313",
                "166_311",
            ),
        }
    if dataset_id == 6:
        return {
            "runner": run_d6_experiment,
            "group_cols": ("store_id", "item_id"),
            "grouping_col": "dept_id",
            "metadata_cols": ("dept_id",),
            "target_keys": (
                "CA_1_FOODS_3_586",
                "CA_1_FOODS_3_080",
                "CA_1_FOODS_3_555",
                "CA_1_FOODS_3_377",
                "CA_1_FOODS_3_668",
            ),
        }
    raise AssertionError(dataset_id)


def _cell_frames(dataset_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    case = _dataset_case(dataset_id)
    first_col, second_col = case["group_cols"]
    dates = pd.date_range("2020-01-01", periods=35, freq="D")
    if dataset_id == 4:
        source_keys = (
            ("166", "900"),
            ("166", "901"),
            ("166", "902"),
            ("167", "903"),
            ("168", "904"),
        )
        target_pairs = (
            ("166", "258"),
            ("166", "432"),
            ("166", "433"),
            ("166", "313"),
            ("166", "311"),
        )
    else:
        source_keys = (
            ("CA_1", "FOODS_3_001"),
            ("CA_1", "FOODS_3_002"),
            ("CA_1", "FOODS_3_003"),
            ("CA_2", "FOODS_3_004"),
            ("CA_3", "FOODS_3_005"),
        )
        target_pairs = (
            ("CA_1", "FOODS_3_586"),
            ("CA_1", "FOODS_3_080"),
            ("CA_1", "FOODS_3_555"),
            ("CA_1", "FOODS_3_377"),
            ("CA_1", "FOODS_3_668"),
        )

    source_frames = []
    for index, (first, second) in enumerate(source_keys, start=1):
        payload = {
            first_col: first,
            second_col: second,
            "entity_id": f"{first}_{second}",
            "date": dates[:30],
            "sales": np.arange(30, dtype=np.float64) + float(index),
        }
        if dataset_id == 4:
            payload.update(
                {
                    "first_category_id": 15,
                    "second_category_id": 20 + index,
                }
            )
        else:
            payload["dept_id"] = "FOODS_3"
        for offset, feature in enumerate(MODEL_FEATURE_CONTRACTS[f"D{dataset_id}"], start=1):
            if feature not in payload:
                payload[feature] = np.arange(30, dtype=np.float64) + float(offset)
        source_frames.append(pd.DataFrame(payload))
    source = pd.concat(source_frames, ignore_index=True)

    target_frames = []
    for index, ((first, second), entity_key) in enumerate(
        zip(target_pairs, case["target_keys"]), start=1
    ):
        payload = {
            first_col: first,
            second_col: second,
            "entity_id": entity_key,
            "date": dates,
            "sales": np.arange(35, dtype=np.float64) + float(index) / 10.0,
        }
        if dataset_id == 4:
            payload.update({"first_category_id": 15, "second_category_id": 20})
        else:
            payload["dept_id"] = "FOODS_3"
        for offset, feature in enumerate(MODEL_FEATURE_CONTRACTS[f"D{dataset_id}"], start=1):
            if feature not in payload:
                payload[feature] = np.arange(35, dtype=np.float64) + float(offset)
        target_frames.append(pd.DataFrame(payload))
    target = pd.concat(target_frames, ignore_index=True)
    target.attrs["knn_observed_start"] = pd.Timestamp("2020-01-01")
    target.attrs["target_observed_start"] = pd.Timestamp("2020-01-01")
    return source, target


@pytest.mark.parametrize("dataset_id", [4, 6])
@pytest.mark.parametrize("scenario", ["without", "with"])
def test_d4_d6_entity_production_wiring_binds_canonical_transformation_owner(
    monkeypatch: pytest.MonkeyPatch,
    dataset_id: int,
    scenario: str,
) -> None:
    source, targets = _cell_frames(dataset_id)
    case = _dataset_case(dataset_id)
    entity_key = case["target_keys"][0]
    target = targets.loc[targets["entity_id"].eq(entity_key)].copy()
    expected_target_key = tuple(
        get_experiment_protocol(dataset_id).formal_target_keys[0]
    )
    expected = (f"D{dataset_id}", scenario, 1, 42, expected_target_key)

    class IdentityVerified(RuntimeError):
        pass

    def verify_ss(**kwargs):
        source_df = kwargs["source_df"]
        target_df = kwargs["target_df"]
        context = kwargs["transformation_reuse_context"]
        for frame in (source_df, target_df):
            assert frame.attrs["protocol_dataset_id"] == f"D{dataset_id}"
            assert frame.attrs["protocol_scenario"] == scenario
            assert tuple(frame.attrs["protocol_cell_identity"]) == expected
            assert frame.attrs["model_horizon"] == 1
            assert frame.attrs["protocol_seed"] == 42
            assert tuple(frame.attrs["protocol_target_key"]) == expected_target_key
        assert context.lifecycle_identity == expected

        first_group = source_df.loc[
            source_df[case["group_cols"][0]].astype(str).eq(
                str(source_df[case["group_cols"][0]].iloc[0])
            )
            & source_df[case["group_cols"][1]].astype(str).eq(
                str(source_df[case["group_cols"][1]].iloc[0])
            )
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
            ("train", (0, 18)),
            ("validation", (18, 24)),
            ("test", (24, 30)),
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
            entity_key=str(entity_key),
            source_df=source,
            target_entity_df=target,
            feature_cols=MODEL_FEATURE_CONTRACTS[f"D{dataset_id}"],
            config={
                "dataset_id": dataset_id,
                "dataset_name": f"Dataset{dataset_id}",
                "info_sharing": scenario,
                "group_cols": case["group_cols"],
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


@pytest.mark.parametrize(
    ("attr_name", "bad_value"),
    [
        ("model_horizon", 2),
        ("protocol_seed", 43),
        ("protocol_target_key", ("166", "432")),
    ],
)
def test_d4_d6_entity_identity_wiring_rejects_wrong_owner_fields(
    attr_name: str,
    bad_value: object,
) -> None:
    expected = ("D4", "with", 1, 42, ("166", "258"))
    frame = pd.DataFrame({"sales": [1.0]})
    entity_experiment._bind_protocol_cell_identity(frame, expected)
    frame.attrs[attr_name] = bad_value

    with pytest.raises(ProtocolViolation, match="protocol cell identity wiring mismatch"):
        entity_experiment._require_bound_protocol_cell_identity(frame, expected)


def _patch_cell_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    dataset_id: int,
    source: pd.DataFrame,
    target: pd.DataFrame,
    prepare_calls: list[object],
    forwarded: list[object],
    cell: dict[str, object],
) -> None:
    case = _dataset_case(dataset_id)
    runner = case["runner"]
    real_prepare = runner.prepare_daily_sequence_pool

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
        runner,
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
                "source_limit": None,
                "epochs": 1,
                "output_dir": cell["output_dir"],
                "repair_source_numeric_na": False,
                "horizon": cell["horizon"],
                "seed": cell["seed"],
            },
        )(),
    )
    monkeypatch.setattr(
        runner,
        "resolve_formal_dataset_paths",
        lambda *args, **kwargs: type(
            "Paths",
            (),
            {"source_path": tmp_path / "source", "target_path": tmp_path / "target"},
        )(),
    )
    monkeypatch.setattr(runner, "reserve_new_output_dir", lambda path: path.mkdir())
    monkeypatch.setattr(runner, "set_protocol_seed", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "load_default_metric_protocol", lambda *args: {})
    monkeypatch.setattr(
        runner,
        "load_knn_results",
        lambda *args, **kwargs: {
            "group_cols": list(case["group_cols"]),
            "results": {key: [] for key in case["target_keys"]},
        },
    )
    monkeypatch.setattr(runner, "read_dataset_windows", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        runner,
        "load_parquet_source_target",
        lambda *args, **kwargs: (source, target),
    )
    monkeypatch.setattr(
        runner,
        "resolve_knn_feature_columns",
        lambda **kwargs: {
            "selected_features": list(MODEL_FEATURE_CONTRACTS[f"D{dataset_id}"]),
            "feature_consistency_status": "aligned",
            "feature_source": "test",
            "knn_feature_mode": "test",
            "json_only_features": [],
            "runtime_only_features": [],
        },
    )
    monkeypatch.setattr(
        runner, "apply_runtime_source_domain_policy", lambda frame, *args: frame
    )
    if dataset_id == 4:
        monkeypatch.setattr(
            runner, "validate_runtime_target_domain", lambda *args, **kwargs: None
        )
    monkeypatch.setattr(runner, "prepare_daily_sequence_pool", recording_prepare)
    monkeypatch.setattr(runner, "run_single_entity_experiment", recording_entity_runner)
    monkeypatch.setattr(runner, "_align_results_to_reference_schema", lambda frame: frame)
    monkeypatch.setattr(runner, "config", dict(runner.config))


@pytest.mark.parametrize("dataset_id", [4, 6])
def test_d4_d6_cell_builds_pool_and_index_once_for_five_targets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    dataset_id: int,
) -> None:
    source, target = _cell_frames(dataset_id)
    case = _dataset_case(dataset_id)
    runner = case["runner"]
    prepare_calls: list[object] = []
    forwarded: list[object] = []
    cell = {
        "mode": "without",
        "horizon": 1,
        "seed": 42,
        "output_dir": tmp_path / f"d{dataset_id}_cell_a",
    }
    _patch_cell_runner(
        monkeypatch,
        tmp_path,
        dataset_id,
        source,
        target,
        prepare_calls,
        forwarded,
        cell,
    )

    runner.main()
    cell.update(
        {
            "mode": "with",
            "horizon": 2,
            "seed": 43,
            "output_dir": tmp_path / f"d{dataset_id}_cell_b",
        }
    )
    runner.main()

    assert len(prepare_calls) == 2
    assert len(forwarded) == 10
    assert all(pool is prepare_calls[0] for pool in forwarded[:5])
    assert all(pool.source_index is prepare_calls[0].source_index for pool in forwarded[:5])
    assert all(pool is prepare_calls[1] for pool in forwarded[5:])
    assert all(pool.source_index is prepare_calls[1].source_index for pool in forwarded[5:])
    assert prepare_calls[0] is not prepare_calls[1]
    assert prepare_calls[0].source_index is not prepare_calls[1].source_index


def _parity_snapshot(
    dataset_id: int,
    source: pd.DataFrame,
    target: pd.DataFrame,
    *,
    prepared_pool,
) -> tuple[dict[str, object], object, object]:
    case = _dataset_case(dataset_id)
    group_cols = case["group_cols"]
    configured_source, configured_target = configure_protocol_frames(
        source,
        target,
        dataset_id=dataset_id,
        scenario="without",
        group_cols=group_cols,
        grouping_col=case["grouping_col"],
        observed_start="2020-01-01",
        prepared_pool=prepared_pool,
        retain_source_frame=prepared_pool is not None,
        enforce_formal_target=True,
    )
    context = get_protocol_frame_context(configured_source)
    selection = SourceSelector().select_top_k_sources(
        configured_target,
        configured_source,
        feature_cols=("sales",),
        k=3,
        group_cols=group_cols,
        consumer_source_df=source,
    )
    manifest = build_sample_manifest(
        configured_target,
        dataset_id=f"D{dataset_id}",
        track=configured_target.attrs["protocol_track"],
        scenario="without",
        target_key=configured_target.attrs["protocol_target_key"],
        observed_end=configured_target.attrs["knn_observed_end"],
        first_forecast_origin=pd.Timestamp(configured_target.attrs["knn_observed_end"])
        + pd.Timedelta(days=3),
        input_window=3,
    )
    sources = selection["sources"]
    meta = selection["meta"]
    snapshot = {
        "target_key": configured_target.attrs["protocol_target_key"],
        "candidate_keys": context.candidate_keys,
        "candidate_ordering": context.candidate_keys,
        "candidate_count": len(context.candidate_keys),
        "distances": tuple(row["distance"] for row in sources),
        "weights": tuple(row["weight"] for row in sources),
        "selection_identity": tuple(
            (tuple(row["source_key"]), row["source_rank"]) for row in sources
        ),
        "selection_digest": meta["selection_result_digest"],
        "target_digest": meta["target_frame_digest"],
        "candidate_scope_digest": meta["candidate_pool_digest"],
        "sample_manifest": manifest.digest,
        "source_provenance": (
            meta["source_frame_digest"],
            meta["source_pool_fingerprint"],
            meta["consumer_frame_digest"],
            meta["consumer_fingerprint"],
        ),
        "selected_source_keys": tuple(tuple(row["source_key"]) for row in sources),
    }
    return snapshot, context, configured_target


@pytest.mark.parametrize("dataset_id", [4, 6])
def test_d4_d6_legacy_per_target_and_hoisted_pool_are_exactly_equal(
    monkeypatch: pytest.MonkeyPatch,
    dataset_id: int,
) -> None:
    source, targets = _cell_frames(dataset_id)
    case = _dataset_case(dataset_id)
    protocol = get_experiment_protocol(dataset_id)
    target_frames = [
        targets[targets["entity_id"] == entity_key].copy()
        for entity_key in case["target_keys"]
    ]

    pool_builds: list[object] = []
    index_builds: list[object] = []
    legacy_candidate_scopes: list[object] = []
    legacy_eligibility: list[object] = []
    real_prepare = runner_adapter.prepare_daily_sequence_pool
    real_build_index = candidate_pool.build_canonical_source_index
    real_legacy_candidates = runner_adapter._extended_candidates
    real_classify_dates = candidate_pool.classify_prepared_candidate_dates

    def recording_prepare(*args, **kwargs):
        pool = real_prepare(*args, **kwargs)
        pool_builds.append(pool)
        return pool

    def recording_build_index(*args, **kwargs):
        index = real_build_index(*args, **kwargs)
        index_builds.append(index)
        return index

    def recording_legacy_candidates(*args, **kwargs):
        candidates = real_legacy_candidates(*args, **kwargs)
        legacy_candidate_scopes.append(candidates)
        return candidates

    def recording_legacy_eligibility(*args, **kwargs):
        proof = real_classify_dates(*args, **kwargs)
        legacy_eligibility.append(proof)
        return proof

    monkeypatch.setattr(runner_adapter, "prepare_daily_sequence_pool", recording_prepare)
    monkeypatch.setattr(runner_adapter, "build_canonical_source_index", recording_build_index)
    monkeypatch.setattr(candidate_pool, "build_canonical_source_index", recording_build_index)
    monkeypatch.setattr(runner_adapter, "_extended_candidates", recording_legacy_candidates)
    monkeypatch.setattr(
        candidate_pool, "classify_prepared_candidate_dates", recording_legacy_eligibility
    )
    legacy_results = [
        _parity_snapshot(dataset_id, source, target, prepared_pool=None)
        for target in target_frames
    ]
    legacy = [item[0] for item in legacy_results]
    legacy_contexts = [item[1] for item in legacy_results]
    assert len(pool_builds) == 5
    assert len(index_builds) == 5
    assert len({id(context.prepared_pool) for context in legacy_contexts}) == 5
    assert len({id(context.source_index) for context in legacy_contexts}) == 5
    assert len(legacy_candidate_scopes) == 5
    assert len(legacy_eligibility) == 5

    pool_builds.clear()
    index_builds.clear()
    shared_pool = prepare_daily_sequence_pool(
        source,
        group_cols=case["group_cols"],
        observed_start="2020-01-01",
        feature_cols=protocol.knn_feature_columns,
        metadata_cols=case["metadata_cols"],
        capture_source_observed_frame_digest=True,
    )
    assert len(index_builds) == 1
    monkeypatch.setattr(
        candidate_pool, "classify_prepared_candidate_dates", real_classify_dates
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

    monkeypatch.setattr(
        runner_adapter, "classify_prepared_candidate_dates", recording_classify
    )
    monkeypatch.setattr(
        runner_adapter, "_extended_candidates_from_identities", recording_candidates
    )
    hoisted_results = [
        _parity_snapshot(dataset_id, source, target, prepared_pool=shared_pool)
        for target in target_frames
    ]
    hoisted = [item[0] for item in hoisted_results]
    contexts = [item[1] for item in hoisted_results]

    mismatches = [
        {
            key: (legacy_item[key], hoisted_item[key])
            for key in legacy_item
            if legacy_item[key] != hoisted_item[key]
        }
        for legacy_item, hoisted_item in zip(legacy, hoisted)
    ]
    assert mismatches == [{}] * 5
    assert len(proofs) == 5
    assert len(candidate_scopes) == 5
    assert pool_builds == []
    assert len(index_builds) == 1
    assert len({id(proof) for proof in proofs}) == 5
    assert all(proof.pool_id == id(shared_pool) for proof in proofs)
    assert all(context.prepared_pool is shared_pool for context in contexts)
    assert all(context.source_index is shared_pool.source_index for context in contexts)
    assert [proof.candidate_scope for proof in proofs] == candidate_scopes
    assert [item["target_key"] for item in hoisted] == list(
        protocol.formal_target_keys
    )
    assert len({item["target_digest"] for item in hoisted}) == 5
    assert len({item["selection_digest"] for item in hoisted}) == 5
    assert len({item["candidate_scope_digest"] for item in hoisted}) == 5
    assert len({item["sample_manifest"] for item in hoisted}) == 5


@pytest.mark.parametrize("dataset_id", [4, 6])
def test_d4_d6_pool_is_readonly_and_materialization_isolated(dataset_id: int) -> None:
    source, _ = _cell_frames(dataset_id)
    case = _dataset_case(dataset_id)
    pool = prepare_daily_sequence_pool(
        source,
        group_cols=case["group_cols"],
        observed_start="2020-01-01",
        feature_cols=("sales",),
        metadata_cols=case["metadata_cols"],
    )
    key = pool.source_keys[0]
    first = pool.selected_frame((key,), feature_cols=("sales",))
    second = pool.selected_frame((key,), feature_cols=("sales",))
    original = float(pool.sales_matrix[pool.key_to_index[key], 0])

    assert not pool.sales_matrix.flags.writeable
    assert not pool.date_presence_matrix.flags.writeable
    assert all(not matrix.flags.writeable for matrix in pool.feature_matrices.values())
    first.loc[0, "sales"] = -999.0
    assert float(pool.sales_matrix[pool.key_to_index[key], 0]) == original
    assert float(second.loc[0, "sales"]) == original


@pytest.mark.parametrize("dataset_id", [4, 6])
def test_d4_d6_incompatible_pool_configuration_is_not_silently_reused(
    dataset_id: int,
) -> None:
    source, targets = _cell_frames(dataset_id)
    case = _dataset_case(dataset_id)
    pool = prepare_daily_sequence_pool(
        source,
        group_cols=case["group_cols"],
        observed_start="2020-01-01",
        feature_cols=("sales",),
        metadata_cols=case["metadata_cols"],
    )

    with pytest.raises(ProtocolViolation, match="group_cols mismatch"):
        pool.validate_for(
            group_cols=tuple(reversed(case["group_cols"])),
            required_dates=pd.date_range("2020-01-01", periods=30),
        )
    with pytest.raises(ProtocolViolation, match="observation dates differ"):
        pool.validate_for(
            group_cols=case["group_cols"],
            required_dates=pd.date_range("2020-01-02", periods=30),
        )

    featureless_pool = replace(
        pool,
        feature_matrices=MappingProxyType({}),
    )
    first_target = targets[
        targets["entity_id"] == case["target_keys"][0]
    ].copy()
    with pytest.raises(ProtocolViolation, match="missing declared feature columns"):
        configure_protocol_frames(
            source,
            first_target,
            dataset_id=dataset_id,
            scenario="without",
            group_cols=case["group_cols"],
            grouping_col=case["grouping_col"],
            observed_start="2020-01-01",
            prepared_pool=featureless_pool,
            enforce_formal_target=True,
        )

    changed_source = source.copy()
    changed_source.loc[0, "sales"] += 1000.0
    rebuilt = prepare_daily_sequence_pool(
        changed_source,
        group_cols=case["group_cols"],
        observed_start="2020-01-01",
        feature_cols=("sales",),
        metadata_cols=case["metadata_cols"],
    )
    shifted_window = prepare_daily_sequence_pool(
        source,
        group_cols=case["group_cols"],
        observed_start="2020-01-02",
        feature_cols=("sales",),
        metadata_cols=case["metadata_cols"],
    )

    assert rebuilt is not pool
    assert rebuilt.source_index is not pool.source_index
    assert not np.array_equal(rebuilt.sales_matrix, pool.sales_matrix)
    assert shifted_window is not pool
    assert shifted_window.required_dates != pool.required_dates


@pytest.mark.parametrize("dataset_id", [4, 6])
def test_d4_d6_k3_reuse_is_once_per_target_and_never_once_per_cell(
    dataset_id: int,
) -> None:
    source, targets = _cell_frames(dataset_id)
    case = _dataset_case(dataset_id)
    protocol = get_experiment_protocol(dataset_id)
    pool = prepare_daily_sequence_pool(
        source,
        group_cols=case["group_cols"],
        observed_start="2020-01-01",
        feature_cols=protocol.knn_feature_columns,
        metadata_cols=case["metadata_cols"],
        capture_source_observed_frame_digest=True,
    )
    evidences = []
    target_digests = []
    for entity_key in case["target_keys"]:
        target = targets[targets["entity_id"] == entity_key].copy()
        configured_source, configured_target = configure_protocol_frames(
            source,
            target,
            dataset_id=dataset_id,
            scenario="without",
            group_cols=case["group_cols"],
            grouping_col=case["grouping_col"],
            observed_start="2020-01-01",
            prepared_pool=pool,
            retain_source_frame=True,
            enforce_formal_target=True,
        )
        legacy = [
            SourceSelector().select_top_k_sources(
                configured_target,
                configured_source,
                feature_cols=("sales",),
                k=3,
                group_cols=case["group_cols"],
            )
            for _ in range(4)
        ]
        lifecycle = (
            f"D{dataset_id}",
            "without",
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
                feature_cols=("sales",),
                group_cols=case["group_cols"],
                k=3,
                weight_mode="inverse_distance",
            )
            wrappers.append(
                evidence.method_wrapper(
                    lifecycle_identity=lifecycle,
                    target_df=configured_target,
                    source_df=configured_source,
                    feature_cols=("sales",),
                    group_cols=case["group_cols"],
                    k=3,
                    weight_mode="inverse_distance",
                )
            )
        assert legacy[0] == legacy[1] == legacy[2] == legacy[3]
        assert all(wrapper == legacy[0] for wrapper in wrappers)
        assert len({id(wrapper) for wrapper in wrappers}) == 4
        evidences.append(context._evidence)
        target_digests.append(legacy[0]["meta"]["target_frame_digest"])

    assert len(evidences) == 5
    assert len({id(evidence) for evidence in evidences}) == 5
    assert len({evidence.target_key for evidence in evidences}) == 5
    assert len(set(target_digests)) == 5
    assert all(evidence.prepared_pool_id == id(pool) for evidence in evidences)
