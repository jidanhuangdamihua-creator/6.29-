"""Contract tests for SITA Arm F Random-3 / Arm G Bottom-3.

No training. Frozen Arm A–E identities and 3e-4 bundles are read-only.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tf_compat  # noqa: F401

import numpy as np
import pytest

from src.evaluation.metrics import smape
from src.transfer_methods.ms_katl_ae_ablation import (
    AE_ARMS,
    HORIZONS,
    UNIFORM_3E4_DATASETS,
    UNIFORM_3E4_LEARNING_RATE_LABEL,
    UNIFORM_3E4_SEEDS,
    UNIFORM_3E4_TARGETS,
    ae_arm_spec,
    ae_optimizer_contract,
    ae_training_recipe,
)
from src.transfer_methods.ms_katl_lite import CONV_LAYER_NAMES
from src.transfer_methods.ms_katl_primary_target_lr_screening import original_rmse
from src.transfer_methods.ms_katl_sita_source_selection_ablation import (
    ARM_F,
    ARM_F_IDENTITY,
    ARM_G,
    ARM_G_IDENTITY,
    CONFIG_DIFF_ALLOWED_FIELDS,
    EXPECTED_F_FULL_RUNS,
    EXPECTED_F_STAGE1_RUNS,
    EXPECTED_G_RUNS,
    FROZEN_ARM_C_TOP3,
    HOLM_FAMILY,
    HOLM_FAMILY_SIZE,
    OUTPUT_MARKER,
    SIMILARITY_DIRECTION,
    SOURCE_SELECTION_GRANULARITY,
    SOURCE_SET_IDS,
    SOURCE_SET_SEEDS,
    STAGE1_SOURCE_SETS,
    ForcedK3SelectionContext,
    RankedSource,
    SourceSelectionAblationError,
    arm_c_reference_payload,
    arm_fg_config_payload,
    assert_audit_gates,
    assert_output_isolated,
    assert_protocol_parity,
    config_diff,
    default_output_dir,
    enumerate_identities,
    format_source_id,
    frozen_3e4_root,
    frozen_arm_c_top3_from_bundles,
    holm_adjust,
    implementation_audit_flags,
    inject_forced_k3_selection_context,
    inverse_distance_weights,
    leakage_audit,
    load_fg_run_rows,
    load_frozen_3e4_arm_rows,
    make_selected_triple,
    materialize_source_sets,
    overlap_sensitivity_report,
    parse_source_id,
    sample_random_keys,
    sort_triple_by_distance,
    source_manifest_rows,
    validate_identity_matrix,
)


def _ranked_pool(n: int = 8) -> list[RankedSource]:
    rows = []
    for index in range(n):
        rows.append(
            RankedSource(
                source_key=("S", str(index + 1)),
                distance=float(index + 1) * 0.1,
                rank=index + 1,
            )
        )
    return rows


def _synthetic_rankings(n: int = 8) -> dict:
    ranked = tuple(_ranked_pool(n))
    top3 = tuple(item.source_key for item in ranked[:3])
    bottom3 = tuple(item.source_key for item in ranked[-3:])
    payload = {
        "dataset": "D1",
        "target_id": "1/10",
        "candidate_pool_size": n,
        "eligible_candidate_count": n,
        "ranked": ranked,
        "top3": top3,
        "bottom3": bottom3,
        "topk_meta": {},
        "raw_topk_payload": {"meta": {"candidate_pool_digest": "pool", "requested_k": 3, "effective_k": 3}, "sources": []},
    }
    return {
        "D1": payload,
        "D4": {**payload, "dataset": "D4", "target_id": "166/258"},
        "D5": {**payload, "dataset": "D5", "target_id": "48/1159414"},
        "D6": {**payload, "dataset": "D6", "target_id": "CA_1/FOODS_3_080"},
    }


def test_audit_gates_pass_on_frozen_authorities():
    flags = implementation_audit_flags(project_root=PROJECT_ROOT)
    assert_audit_gates(flags)
    assert flags["ARM_A_REFERENCE_CONFIG_FOUND"] == "YES"
    assert flags["ARM_C_REFERENCE_CONFIG_FOUND"] == "YES"
    assert flags["TOP3_SELECTION_IMPLEMENTATION_FOUND"] == "YES"
    assert flags["LEGAL_SOURCE_POOL_IMPLEMENTATION_FOUND"] == "YES"
    assert flags["SOURCE_FUSION_IMPLEMENTATION_FOUND"] == "YES"
    assert flags["FORMAL_TARGETS_MATCH"] == "YES"
    assert flags["FORMAL_SEEDS_MATCH"] == "YES"
    assert flags["FORMAL_HORIZONS_MATCH"] == "YES"
    assert flags["ABLATION_LR_MATCH_3E4"] == "YES"
    assert flags["SOURCE_SELECTION_GRANULARITY"] == SOURCE_SELECTION_GRANULARITY == "target-level"
    assert flags["SIMILARITY_DIRECTION"] == SIMILARITY_DIRECTION == "LOWER_IS_MORE_SIMILAR"


def test_arm_a_and_c_definitions_unchanged():
    spec_a = ae_arm_spec("A")
    spec_c = ae_arm_spec("C")
    assert spec_a.name == "TARGET_NATIVE_INIT"
    assert spec_a.init_mode == "native" and spec_a.target_backbone_trainable and not spec_a.revin_enabled
    assert spec_c.name == "FUSED_TRAINABLE"
    assert spec_c.init_mode == "fused" and spec_c.target_backbone_trainable and not spec_c.revin_enabled
    assert tuple(CONV_LAYER_NAMES) == ("conv1", "conv2", "conv3")
    recipe = ae_training_recipe()
    optimizer = ae_optimizer_contract()
    assert recipe["loss_name"] == "huber" and recipe["lr_schedule"] == "cosine" and recipe["use_ema"]
    assert optimizer["optimizer"] == "Adam"
    assert optimizer["epochs"] == 50
    assert optimizer["batch_size"] == 16
    assert optimizer["stage_b_early_stopping"] is False
    assert UNIFORM_3E4_LEARNING_RATE_LABEL == "3e-4"


def test_ae_arms_do_not_include_f_or_g():
    assert AE_ARMS == ("A", "B", "C", "D", "E")
    with pytest.raises(Exception):
        ae_arm_spec("F")
    with pytest.raises(Exception):
        ae_arm_spec("G")


def test_formal_targets_horizons_seeds_and_d4():
    assert dict(UNIFORM_3E4_TARGETS) == {
        "D1": "1/10",
        "D4": "166/258",
        "D5": "48/1159414",
        "D6": "CA_1/FOODS_3_080",
    }
    assert "166/311" not in UNIFORM_3E4_TARGETS.values()
    assert "166/313" not in UNIFORM_3E4_TARGETS.values()
    assert tuple(HORIZONS) == (1, 3, 5)
    assert tuple(UNIFORM_3E4_SEEDS) == (42, 43, 44, 45, 46)


def test_frozen_arm_c_top3_is_target_level_and_matches_runtime():
    resolved = frozen_arm_c_top3_from_bundles(PROJECT_ROOT)
    assert resolved == dict(FROZEN_ARM_C_TOP3)
    rows = load_frozen_3e4_arm_rows(project_root=PROJECT_ROOT, arms=("A", "C"))
    a_rows = [row for row in rows if row["arm"] == "A"]
    c_rows = [row for row in rows if row["arm"] == "C"]
    assert len(a_rows) == 60
    assert len(c_rows) == 60
    assert {row["target"] for row in rows if row["dataset"] == "D4"} == {"166/258"}
    for dataset in UNIFORM_3E4_DATASETS:
        identities = {
            tuple(tuple(item) for item in row["source_identities"])
            for row in c_rows
            if row["dataset"] == dataset
        }
        assert len(identities) == 1
        assert next(iter(identities)) == FROZEN_ARM_C_TOP3[dataset]


def test_arm_f_and_g_source_count_is_three():
    rankings = _synthetic_rankings()
    materialized = materialize_source_sets(rankings)
    for dataset in UNIFORM_3E4_DATASETS:
        for source_set_id, triple in materialized[dataset].items():
            assert len(triple.sources) == 3
            if triple.arm == ARM_F:
                assert source_set_id in SOURCE_SET_IDS
            else:
                assert source_set_id == "BOTTOM3"
                assert triple.arm == ARM_G


def test_target_not_in_source_list_and_no_duplicates():
    rankings = _synthetic_rankings()
    materialized = materialize_source_sets(rankings)
    for dataset, sets in materialized.items():
        target = UNIFORM_3E4_TARGETS[dataset]
        target_key = parse_source_id(target)
        for triple in sets.values():
            keys = [item.source_key for item in triple.sources]
            assert target_key not in keys
            assert len(set(keys)) == 3
            leak = leakage_audit(triple, target)
            assert leak["TARGET_NOT_IN_SOURCES"] == "YES"
            assert leak["NO_DUPLICATE_SOURCES"] == "YES"
            assert leak["SOURCE_COUNT_3"] == "YES"
            assert leak["TARGET_ONLY_INFERENCE"] == "YES"
            assert leak["NO_FUTURE_SOURCE_AT_PREDICTION"] == "YES"
            assert leak["NO_SOURCE_PREDICTION_INJECTION"] == "YES"
            assert leak["OBSERVED_TARGET_BUDGET_UNCHANGED"] == "YES"


def test_random_source_set_invariant_across_training_seeds():
    keys = [item.source_key for item in _ranked_pool()]
    sampled = [sample_random_keys(keys, source_sampling_seed=1001) for _ in range(5)]
    assert len(set(sampled)) == 1
    for seed in UNIFORM_3E4_SEEDS:
        reset = sample_random_keys(keys, source_sampling_seed=1001)
        assert reset == sampled[0]
        _ = seed
    different = sample_random_keys(keys, source_sampling_seed=1002)
    assert different != sampled[0]


def test_training_seed_does_not_affect_source_selection_and_sampling_seed_is_independent():
    identities = enumerate_identities(arms=(ARM_F,), source_sets=("R1",))
    by_dataset = {}
    for item in identities:
        by_dataset.setdefault(item.dataset, []).append(item)
    for dataset, items in by_dataset.items():
        seeds = {item.training_seed for item in items}
        sets = {item.source_set_id for item in items}
        sampling = {item.source_sampling_seed for item in items}
        assert seeds == set(UNIFORM_3E4_SEEDS)
        assert sets == {"R1"}
        assert sampling == {1001}
        assert all(item.source_sampling_seed != item.training_seed for item in items)


def test_r1_r5_are_reproducible_and_frozen():
    assert SOURCE_SET_SEEDS == {"R1": 1001, "R2": 1002, "R3": 1003, "R4": 1004, "R5": 1005}
    keys = [item.source_key for item in _ranked_pool(10)]
    first = {source_set_id: sample_random_keys(keys, source_sampling_seed=seed) for source_set_id, seed in SOURCE_SET_SEEDS.items()}
    second = {source_set_id: sample_random_keys(keys, source_sampling_seed=seed) for source_set_id, seed in SOURCE_SET_SEEDS.items()}
    assert first == second
    assert len({first[item] for item in SOURCE_SET_IDS}) == 5


def test_bottom3_matches_lowest_ranked_and_top3_matches_arm_c_contract():
    ranked = _ranked_pool()
    rankings = _synthetic_rankings()
    materialized = materialize_source_sets(rankings)
    bottom = materialized["D1"]["BOTTOM3"]
    assert tuple(item.source_key for item in bottom.sources) == tuple(item.source_key for item in ranked[-3:])
    assert bottom.top3 == tuple(item.source_key for item in ranked[:3])
    assert FROZEN_ARM_C_TOP3["D1"] == (("1", "7"), ("1", "8"), ("1", "2"))
    assert FROZEN_ARM_C_TOP3["D4"] == (("166", "184"), ("166", "530"), ("166", "229"))
    assert FROZEN_ARM_C_TOP3["D6"] == (("CA_1", "FOODS_3_202"), ("CA_1", "FOODS_3_288"), ("CA_1", "FOODS_3_694"))


def test_fusion_architecture_trainable_revin_and_lr_parity_with_arm_c():
    reference = arm_c_reference_payload()
    for arm in (ARM_F, ARM_G):
        payload = arm_fg_config_payload(arm, source_set_id="R1", source_sampling_seed=1001)
        diff = config_diff(reference, payload)
        unexpected = [key for key in diff if key not in CONFIG_DIFF_ALLOWED_FIELDS]
        assert unexpected == []
        report = assert_protocol_parity(arm)
        assert report["DIFF_COUNT"] == 0
        assert payload["init_mode"] == "fused"
        assert payload["target_backbone_trainable"] is True
        assert payload["revin_enabled"] is False
        assert payload["conv_transfer_scope"] == ["conv1", "conv2", "conv3"]
        assert payload["stage_b_learning_rate"] == "3e-4"
        assert payload["recipe"] == reference["recipe"]
        assert payload["optimizer"] == reference["optimizer"]
        assert payload["inference"] == "target_only"
    assert ARM_F_IDENTITY == "SITA_ABLATION_F_RANDOM3"
    assert ARM_G_IDENTITY == "SITA_ABLATION_G_BOTTOM3"


def test_expected_run_counts_and_unique_identities():
    full = enumerate_identities(arms=(ARM_F, ARM_G), source_sets=SOURCE_SET_IDS)
    report = validate_identity_matrix(full, arms=(ARM_F, ARM_G), source_sets=SOURCE_SET_IDS)
    assert report["DRY_RUN_TOTAL_RUNS"] == EXPECTED_F_FULL_RUNS + EXPECTED_G_RUNS == 360
    stage1_f = enumerate_identities(arms=(ARM_F,), source_sets=STAGE1_SOURCE_SETS)
    assert validate_identity_matrix(stage1_f, arms=(ARM_F,), source_sets=STAGE1_SOURCE_SETS)["DRY_RUN_TOTAL_RUNS"] == EXPECTED_F_STAGE1_RUNS == 180
    g_only = enumerate_identities(arms=(ARM_G,), source_sets=STAGE1_SOURCE_SETS)
    assert validate_identity_matrix(g_only, arms=(ARM_G,), source_sets=STAGE1_SOURCE_SETS)["DRY_RUN_TOTAL_RUNS"] == EXPECTED_G_RUNS == 60
    f_keys = {item.identity_tuple for item in full if item.arm == ARM_F}
    g_keys = {item.identity_tuple for item in full if item.arm == ARM_G}
    assert len(f_keys) == 300
    assert len(g_keys) == 60
    assert all(len(key) == 6 for key in f_keys)
    assert all(len(key) == 5 for key in g_keys)
    horizons = {item.horizon for item in full}
    seeds = {item.training_seed for item in full}
    assert horizons == {1, 3, 5}
    assert seeds == {42, 43, 44, 45, 46}
    assert all(item.dataset != "D4" or item.target_identity == "166/258" for item in full)


def test_same_target_reuses_source_identities_across_horizons():
    identities = enumerate_identities(arms=(ARM_F,), source_sets=("R1",))
    d1_r1 = [item for item in identities if item.dataset == "D1"]
    assert {item.horizon for item in d1_r1} == {1, 3, 5}
    assert {item.source_set_id for item in d1_r1} == {"R1"}
    assert {item.source_sampling_seed for item in d1_r1} == {1001}


def test_inverse_distance_weights_and_distance_sort():
    rows = [
        RankedSource(("B",), 0.5, 2),
        RankedSource(("A",), 0.2, 1),
        RankedSource(("C",), 0.8, 3),
    ]
    ordered = sort_triple_by_distance(rows)
    assert tuple(item.source_key for item in ordered) == (("A",), ("B",), ("C",))
    weights = inverse_distance_weights([item.distance for item in ordered])
    assert math.isclose(sum(weights), 1.0, abs_tol=1e-12)
    assert weights[0] > weights[1] > weights[2]


def test_no_replacement_sampling_when_pool_too_small():
    keys = [("A",), ("B",)]
    with pytest.raises(SourceSelectionAblationError, match="legal source pool size"):
        sample_random_keys(keys, source_sampling_seed=1001)


def test_metric_parity_from_raw_predictions():
    true = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    pred = np.asarray([1.1, 1.8, 3.2, 3.7], dtype=np.float64)
    rmse = original_rmse(true, pred)
    smape_value = float(smape(true, pred))
    assert rmse == pytest.approx(math.sqrt(np.mean((true - pred) ** 2)))
    assert smape_value > 0
    assert SIMILARITY_DIRECTION == "LOWER_IS_MORE_SIMILAR"


def test_holm_family_is_fixed_and_not_result_dependent():
    adjusted = holm_adjust({"C_vs_F": 0.01, "F_vs_A": 0.04, "C_vs_G": 0.20, "F_vs_G": 0.03})
    assert len(adjusted) == HOLM_FAMILY_SIZE == 4
    assert HOLM_FAMILY == "Holm"
    assert set(adjusted) == {"C_vs_F", "F_vs_A", "C_vs_G", "F_vs_G"}


def test_output_isolation_refuses_frozen_3e4_root():
    with pytest.raises(SourceSelectionAblationError):
        assert_output_isolated(frozen_3e4_root(PROJECT_ROOT), project_root=PROJECT_ROOT)
    isolated = default_output_dir(PROJECT_ROOT)
    assert OUTPUT_MARKER in isolated.as_posix()
    assert isolated == assert_output_isolated(isolated, project_root=PROJECT_ROOT)


def test_overlap_sensitivity_does_not_replace_main_random3():
    rankings = _synthetic_rankings()
    materialized = materialize_source_sets(rankings)
    report = overlap_sensitivity_report(materialized)
    assert report["n_random_sets"] == 20
    assert report["promote_to_formal"] is False
    assert set(report["overlap_counts"]) == {0, 1, 2, 3}
    rows = source_manifest_rows(materialized)
    assert len(rows) == 4 * 6
    assert all("top3_overlap_count" in row for row in rows)


def test_source_id_roundtrip_and_d6_format():
    assert format_source_id(("CA_1", "FOODS_3_202")) == "CA_1/FOODS_3_202"
    assert parse_source_id("CA_1/FOODS_3_202") == ("CA_1", "FOODS_3_202")
    assert parse_source_id("166_258") == ("166", "258")
    assert parse_source_id(["1", "7"]) == ("1", "7")


def test_forced_k3_context_exposes_lifecycle_identity_for_stage_a():
    identity = ("Dataset1", "without_information_sharing", 1, 42, ("1", "10"))
    payload = {"sources": [{"source_key": ["1", "7"]}], "meta": {"requested_k": 3, "effective_k": 3}}
    context = ForcedK3SelectionContext(payload, lifecycle_identity=identity)
    assert context.lifecycle_identity == identity
    evidence = context.selection_for_method()
    result = evidence.method_wrapper(
        lifecycle_identity=context.lifecycle_identity,
        target_df=None,
        source_df=None,
        feature_cols=[],
        group_cols=[],
        k=3,
        weight_mode="inverse_distance",
    )
    assert result["sources"] == payload["sources"]
    with pytest.raises(SourceSelectionAblationError):
        ForcedK3SelectionContext(payload, lifecycle_identity=("too-short",))


def test_inject_forced_k3_preserves_arm_c_lifecycle_and_rebinds_raw_sources():
    class OriginalContext:
        lifecycle_identity = ("Dataset6", "without_information_sharing", 5, 46, ("CA_1", "FOODS_3_080"))

    original_raw = object()
    kwargs = {
        "k3_selection_context": OriginalContext(),
        "k3_raw_source_context": original_raw,
    }
    payload = {"sources": [], "meta": {}}
    injected = inject_forced_k3_selection_context(kwargs, payload)
    assert injected.lifecycle_identity == OriginalContext.lifecycle_identity
    assert kwargs["k3_selection_context"] is injected
    assert kwargs["k3_raw_source_context"] is not original_raw
    assert tuple(kwargs["k3_raw_source_context"].lifecycle_identity) == OriginalContext.lifecycle_identity
    assert kwargs["k3_raw_source_context"].materialization_count == 0


def test_analyze_loads_all_cell_manifests_not_truncated_csv(tmp_path):
    cells = tmp_path / "cells"
    (cells / "F_only_r4").mkdir(parents=True)
    (cells / "G_bottom").mkdir(parents=True)
    (cells / "F_only_r4" / "run_manifest.json").write_text(
        '{"arm":"F","source_set_id":"R4","result_status":"OK","RMSE":1}\n',
        encoding="utf-8",
    )
    (cells / "G_bottom" / "run_manifest.json").write_text(
        '{"arm":"G","source_set_id":"BOTTOM3","result_status":"OK","RMSE":2}\n',
        encoding="utf-8",
    )
    truncated = tmp_path / "results"
    truncated.mkdir()
    (truncated / "all_runs.csv").write_text("arm,source_set_id,result_status\nF,R4,OK\n", encoding="utf-8")
    rows = load_fg_run_rows(tmp_path)
    arms = {row["arm"] for row in rows}
    assert arms == {"F", "G"}
    assert len(rows) == 2


def test_make_selected_triple_records_top3_overlap():
    ranked = _ranked_pool()
    top3 = tuple(item.source_key for item in ranked[:3])
    triple = make_selected_triple(
        dataset="D1",
        target_id="1/10",
        arm=ARM_F,
        source_set_id="R1",
        source_sampling_seed=1001,
        selected_keys=top3,
        ranked=ranked,
        top3=top3,
    )
    assert triple.overlap_count == 3
    assert triple.overlap_ratio == pytest.approx(1.0)
    mixed = make_selected_triple(
        dataset="D1",
        target_id="1/10",
        arm=ARM_F,
        source_set_id="R2",
        source_sampling_seed=1002,
        selected_keys=(ranked[0].source_key, ranked[4].source_key, ranked[7].source_key),
        ranked=ranked,
        top3=top3,
    )
    assert mixed.overlap_count == 1
