from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

import src.protocols.candidate_pool as candidate_pool_module
import src.source_selection.source_selector as selector_module
from src.transfer_methods import mssb_tl, msml_tl, msml_tl_rfe, mswa_tl
from src.experiment import experiment_runner
from src.protocols.candidate_pool import InsufficientCandidatePoolError
from src.protocols.experiment_protocol import ProtocolViolation
from src.protocols.runner_adapter import configure_protocol_frames
from src.source_selection.source_selector import (
    K3SelectionEvidence,
    SourceSelector,
    TargetK3SelectionContext,
)
from src.utils.dataframe_attrs import (
    context_with,
    get_protocol_frame_context,
    set_protocol_frame_context,
)


GROUP_COLS = ("store_id", "item_id")


def _configured_d4(
    source_values: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0),
    *,
    target_value: float = 0.0,
    target_key: tuple[str, str] = ("T1", "I0"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2020-01-01", periods=35, freq="D")
    source = pd.concat(
        [
            pd.DataFrame(
                {
                    "store_id": f"S{index}",
                    "item_id": f"I{index}",
                    "second_category_id": 20,
                    "date": dates,
                    "sales": np.full(35, value, dtype=np.float64),
                    "model_feature": np.full(35, index, dtype=np.float64),
                }
            )
            for index, value in enumerate(source_values, start=1)
        ],
        ignore_index=True,
    )
    target = pd.DataFrame(
        {
            "store_id": target_key[0],
            "item_id": target_key[1],
            "second_category_id": 20,
            "date": dates,
            "sales": np.full(35, target_value, dtype=np.float64),
            "model_feature": np.full(35, 99.0, dtype=np.float64),
        }
    )
    return configure_protocol_frames(
        source,
        target,
        dataset_id="D4",
        scenario="with",
        group_cols=GROUP_COLS,
        observed_start="2020-01-01",
    )


def _identity(target: pd.DataFrame, *, seed: int = 42) -> tuple[object, ...]:
    return (
        "D4",
        target.attrs["protocol_scenario"],
        1,
        seed,
        tuple(target.attrs["protocol_target_key"]),
    )


def _legacy(
    source: pd.DataFrame,
    target: pd.DataFrame,
    *,
    k: int,
) -> dict[str, object]:
    return SourceSelector().select_top_k_sources(
        target_df=target,
        source_df=source,
        feature_cols=("sales", "model_feature"),
        k=k,
        group_cols=GROUP_COLS,
        weight_mode="inverse_distance",
    )


def _shared_wrappers(
    source: pd.DataFrame,
    target: pd.DataFrame,
) -> tuple[TargetK3SelectionContext, list[dict[str, object]]]:
    context = TargetK3SelectionContext(_identity(target))
    wrappers = []
    for _method in ("MSWA-TL", "MSSB-TL", "MSML-TL", "MSML-TL-RFE"):
        evidence = context.selection_for_method(
            target_df=target,
            source_df=source,
            feature_cols=("sales", "model_feature"),
            group_cols=GROUP_COLS,
            k=3,
            weight_mode="inverse_distance",
        )
        wrappers.append(
            evidence.method_wrapper(
                lifecycle_identity=context.lifecycle_identity,
                target_df=target,
                source_df=source,
                feature_cols=("sales", "model_feature"),
                group_cols=GROUP_COLS,
                k=3,
                weight_mode="inverse_distance",
            )
        )
    return context, wrappers


@pytest.mark.parametrize(
    "source_values,target_value",
    [
        ((0.0, 1.0, 2.0, 3.0), 0.0),
        ((0.0, 0.0, 1.0, 2.0), 0.0),
        ((0.0, 0.0, 0.0, 0.0), 0.0),
        ((1.0, 1.0, 2.0, 2.0), 0.0),
    ],
)
def test_shared_k3_is_exact_legacy_parity_for_zero_and_equal_distances(
    source_values: tuple[float, ...],
    target_value: float,
) -> None:
    source, target = _configured_d4(source_values, target_value=target_value)
    legacy = [_legacy(source, target, k=3) for _ in range(4)]
    _context, wrappers = _shared_wrappers(source, target)

    assert legacy[0] == legacy[1] == legacy[2] == legacy[3]
    assert all(wrapper == legacy[0] for wrapper in wrappers)
    assert [row["tie_group"] for row in wrappers[0]["sources"]] == [
        row["tie_group"] for row in legacy[0]["sources"]
    ]


def test_k3_heavy_core_four_to_one_and_k1_remains_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, target = _configured_d4()
    calls: list[int] = []
    real_select = SourceSelector.select_top_k_sources

    def counted(self, *args, **kwargs):
        calls.append(int(kwargs["k"]))
        return real_select(self, *args, **kwargs)

    monkeypatch.setattr(SourceSelector, "select_top_k_sources", counted)

    k1 = _legacy(source, target, k=1)
    _context, wrappers = _shared_wrappers(source, target)

    assert calls == [1, 3]
    assert len(wrappers) == 4
    assert k1["meta"]["requested_k"] == 1
    assert wrappers[0]["meta"]["requested_k"] == 3
    assert k1["meta"]["selection_result_digest"] != wrappers[0]["meta"][
        "selection_result_digest"
    ]
    assert len(k1["sources"]) == 1
    assert len(wrappers[0]["sources"]) == 3


def test_method_wrappers_are_distinct_and_mutation_isolated() -> None:
    source, target = _configured_d4()
    context, wrappers = _shared_wrappers(source, target)
    evidence = context._evidence

    assert isinstance(evidence, K3SelectionEvidence)
    assert len({id(wrapper) for wrapper in wrappers}) == 4
    assert len({id(wrapper["meta"]) for wrapper in wrappers}) == 4
    assert len({id(wrapper["sources"]) for wrapper in wrappers}) == 4
    original_second_key = wrappers[1]["sources"][0]["source_key"]
    wrappers[0]["sources"][0]["source_key"] = ("MUTATED", "KEY")
    wrappers[0]["meta"]["runtime_knn_proof"]["distance_count"] = -1

    fresh = evidence.method_wrapper(
        lifecycle_identity=context.lifecycle_identity,
        target_df=target,
        source_df=source,
        feature_cols=("sales", "model_feature"),
        group_cols=GROUP_COLS,
        k=3,
        weight_mode="inverse_distance",
    )
    assert wrappers[1]["sources"][0]["source_key"] == original_second_key
    assert fresh["sources"][0]["source_key"] == original_second_key
    assert fresh["meta"]["runtime_knn_proof"]["distance_count"] >= 3
    with pytest.raises(FrozenInstanceError):
        evidence.selection_result_digest = "changed"  # type: ignore[misc]


def test_candidate_count_below_three_does_not_couple_k1_failure() -> None:
    source, target = _configured_d4((0.0, 1.0))

    k1 = _legacy(source, target, k=1)
    assert len(k1["sources"]) == 1
    with pytest.raises(InsufficientCandidatePoolError):
        TargetK3SelectionContext(_identity(target)).selection_for_method(
            target_df=target,
            source_df=source,
            feature_cols=("sales", "model_feature"),
            group_cols=GROUP_COLS,
            k=3,
            weight_mode="inverse_distance",
        )


def test_candidate_target_schema_pool_and_cell_mismatches_fail_closed() -> None:
    source, target = _configured_d4()
    context, _wrappers = _shared_wrappers(source, target)
    evidence = context._evidence
    assert evidence is not None

    source_context = get_protocol_frame_context(source)
    assert source_context is not None
    changed_source = source.copy()
    changed_source.attrs = source.attrs.copy()
    set_protocol_frame_context(
        changed_source,
        context_with(
            source_context,
            candidate_keys=tuple(reversed(source_context.candidate_keys)),
        ),
    )
    with pytest.raises(ProtocolViolation, match="candidate scope"):
        evidence.method_wrapper(
            lifecycle_identity=context.lifecycle_identity,
            target_df=target,
            source_df=changed_source,
            feature_cols=("sales", "model_feature"),
            group_cols=GROUP_COLS,
            k=3,
            weight_mode="inverse_distance",
        )

    other_source, other_target = _configured_d4(target_key=("T2", "I9"))
    with pytest.raises(ProtocolViolation, match="target key"):
        evidence.method_wrapper(
            lifecycle_identity=context.lifecycle_identity,
            target_df=other_target,
            source_df=other_source,
            feature_cols=("sales", "model_feature"),
            group_cols=GROUP_COLS,
            k=3,
            weight_mode="inverse_distance",
        )

    rebuilt_source, rebuilt_target = _configured_d4()
    with pytest.raises(ProtocolViolation, match="prepared-pool"):
        evidence.method_wrapper(
            lifecycle_identity=context.lifecycle_identity,
            target_df=rebuilt_target,
            source_df=rebuilt_source,
            feature_cols=("sales", "model_feature"),
            group_cols=GROUP_COLS,
            k=3,
            weight_mode="inverse_distance",
        )

    changed_vector_source, changed_vector_target = _configured_d4(target_value=9.0)
    with pytest.raises(ProtocolViolation, match="target KNN digest"):
        evidence.method_wrapper(
            lifecycle_identity=context.lifecycle_identity,
            target_df=changed_vector_target,
            source_df=changed_vector_source,
            feature_cols=("sales", "model_feature"),
            group_cols=GROUP_COLS,
            k=3,
            weight_mode="inverse_distance",
        )

    with pytest.raises(ProtocolViolation, match="model feature"):
        evidence.method_wrapper(
            lifecycle_identity=context.lifecycle_identity,
            target_df=target,
            source_df=source,
            feature_cols=("model_feature", "sales"),
            group_cols=GROUP_COLS,
            k=3,
            weight_mode="inverse_distance",
        )

    with pytest.raises(ProtocolViolation, match="lifecycle"):
        evidence.method_wrapper(
            lifecycle_identity=(*context.lifecycle_identity[:-2], 99, context.lifecycle_identity[-1]),
            target_df=target,
            source_df=source,
            feature_cols=("sales", "model_feature"),
            group_cols=GROUP_COLS,
            k=3,
            weight_mode="inverse_distance",
        )

    with pytest.raises(ProtocolViolation, match="non-K3"):
        evidence.method_wrapper(
            lifecycle_identity=context.lifecycle_identity,
            target_df=target,
            source_df=source,
            feature_cols=("sales", "model_feature"),
            group_cols=GROUP_COLS,
            k=1,
            weight_mode="inverse_distance",
        )


def test_distinct_targets_never_share_context_or_evidence() -> None:
    source_a, target_a = _configured_d4(target_key=("T1", "I0"))
    source_b, target_b = _configured_d4(target_key=("T2", "I9"))
    context_a, _ = _shared_wrappers(source_a, target_a)
    context_b, _ = _shared_wrappers(source_b, target_b)

    assert context_a is not context_b
    assert context_a._evidence is not context_b._evidence
    assert context_a._evidence.target_key != context_b._evidence.target_key


def test_four_actual_method_entrypoints_share_one_selector_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, target = _configured_d4()
    context = TargetK3SelectionContext(_identity(target))
    calls: list[int] = []
    real_select = SourceSelector.select_top_k_sources

    def counted(self, *args, **kwargs):
        calls.append(int(kwargs["k"]))
        return real_select(self, *args, **kwargs)

    class StopAfterSelection(RuntimeError):
        pass

    def stop(*args, **kwargs):
        raise StopAfterSelection

    monkeypatch.setattr(SourceSelector, "select_top_k_sources", counted)
    monkeypatch.setattr(mswa_tl, "temporal_split_by_ratio_or_dates", stop)
    monkeypatch.setattr(mssb_tl, "temporal_split_by_ratio_or_dates", stop)
    monkeypatch.setattr(msml_tl, "train_source_cnn_for_msml", stop)
    monkeypatch.setattr(msml_tl_rfe, "temporal_split_by_ratio_or_dates", stop)
    common = {
        "source_df": source,
        "target_df": target,
        "feature_cols": ("sales", "model_feature"),
        "k": 3,
        "group_cols": GROUP_COLS,
        "weight_mode": "inverse_distance",
        "k3_selection_context": context,
    }

    for runner in (
        mswa_tl.run_mswa_tl,
        mssb_tl.run_mssb_tl,
        msml_tl.run_msml_tl,
        msml_tl_rfe.run_msml_tl_rfe,
    ):
        with pytest.raises((StopAfterSelection, ProtocolViolation)):
            runner(**common)

    assert calls == [3]
    assert context._evidence is not None


def test_actual_ss_tl_entrypoint_executes_independent_legacy_k1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, target = _configured_d4()
    calls: list[int] = []
    real_select = SourceSelector.select_top_k_sources

    def counted(self, *args, **kwargs):
        calls.append(int(kwargs["k"]))
        return real_select(self, *args, **kwargs)

    class StopAfterK1(RuntimeError):
        pass

    monkeypatch.setattr(SourceSelector, "select_top_k_sources", counted)
    monkeypatch.setattr(
        experiment_runner,
        "temporal_split_by_ratio_or_dates",
        lambda *args, **kwargs: (_ for _ in ()).throw(StopAfterK1()),
    )
    with pytest.raises(StopAfterK1):
        experiment_runner.run_ss_tl_experiment(
            source_df=source,
            target_df=target,
            feature_cols=("sales", "model_feature"),
            group_cols=GROUP_COLS,
        )

    assert calls == [1]


def test_deterministic_heavy_operation_counts_are_five_to_two(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, target = _configured_d4()
    counts = {"core": 0, "ranking": 0, "digest": 0}
    real_core = selector_module.select_daily_sequence_sources
    real_ranking = candidate_pool_module.rank_source_distances
    real_digest = candidate_pool_module.build_selection_result_digest

    def count_core(*args, **kwargs):
        counts["core"] += 1
        return real_core(*args, **kwargs)

    def count_ranking(*args, **kwargs):
        counts["ranking"] += 1
        return real_ranking(*args, **kwargs)

    def count_digest(*args, **kwargs):
        counts["digest"] += 1
        return real_digest(*args, **kwargs)

    monkeypatch.setattr(selector_module, "select_daily_sequence_sources", count_core)
    monkeypatch.setattr(candidate_pool_module, "rank_source_distances", count_ranking)
    monkeypatch.setattr(
        candidate_pool_module,
        "build_selection_result_digest",
        count_digest,
    )

    _legacy(source, target, k=1)
    for _method in ("MSWA-TL", "MSSB-TL", "MSML-TL", "MSML-TL-RFE"):
        _legacy(source, target, k=3)
    assert counts == {"core": 5, "ranking": 5, "digest": 5}

    counts.update(core=0, ranking=0, digest=0)
    _legacy(source, target, k=1)
    _shared_wrappers(source, target)
    assert counts == {"core": 2, "ranking": 2, "digest": 2}
