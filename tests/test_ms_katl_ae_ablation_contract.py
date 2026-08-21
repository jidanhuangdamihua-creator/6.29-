"""No-training contract tests for the V3 A–E controlled ablation protocol."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tf_compat  # noqa: F401

import numpy as np

from src.evaluation.metric_contract import ORIGINAL_SALES_SPACE, SALES_VALUE_POLICY
from src.protocols.experiment_protocol import FORMAL_METHODS
from src.transfer_methods.ms_katl_ae_ablation import (
    AE_ARM_NAMES,
    AE_ARMS,
    AE_EXECUTION_BODY_ENABLED,
    AE_FORMAL_EXECUTION_ENABLED,
    AE_FORMAL_EXECUTION_IMPLEMENTED,
    EXPECTED_CANONICAL_CELLS,
    EXPECTED_RUNS,
    FORMAL_TRAINING_UNLOCK_ENV,
    HORIZONS,
    LEGACY_MECHANISM_ARMS,
    PARENT_METHOD,
    SCENARIO,
    SEEDS,
    SYNTHETIC_DENSE_OUT_FILL,
    SYNTHETIC_SOURCE_FILLS,
    SYNTHETIC_SOURCE_WEIGHTS,
    V3_LR_MAP,
    AEAblationArmSpec,
    AEContractError,
    AETrainingBlocked,
    ae_arm_registry,
    ae_arm_spec,
    ae_training_recipe,
    analysis_contract,
    arm_factor_payload,
    assert_d_equals_frozen_v3,
    assert_formal_training_permitted,
    best_source_target_init_snapshot,
    build_synthetic_stage_a,
    construct_all_ae_arms,
    enumerate_canonical_cells,
    enumerate_planned_runs,
    load_ae_manifest,
    metric_contract,
    pairwise_single_factor_contracts,
    planned_result_row,
    provenance_schema_fields,
    require_v3_recipe,
    resolve_cell_lr,
    run_ae_formal_matrix,
    run_ae_preflight,
    verify_common_head_init,
    write_formal_plan,
)
from src.transfer_methods.ms_katl_fto_three_track_robustness import EXPECTED_V3_ARCHITECTURE_DIGEST
from src.transfer_methods.ms_katl_fto_v3_screening import v3_architecture_digest, v3_training_recipe
from src.transfer_methods.ms_katl_lite import CONV_LAYER_NAMES
from src.transfer_methods.ms_katl_mechanism_ablation import ABLATION_ARM_NAMES


@pytest.fixture(scope="module")
def synthetic_stage_a():
    return build_synthetic_stage_a(seed=42)


@pytest.fixture(scope="module")
def constructed_arms(synthetic_stage_a):
    lr_value, _ = resolve_cell_lr("D4", "166/311")
    return construct_all_ae_arms(
        synthetic_stage_a,
        seed=42,
        selected_lr=lr_value,
        recipe=ae_training_recipe(),
    )


def test_registry_identity_separated_from_legacy_a_f():
    registry = ae_arm_registry()
    assert tuple(registry) == AE_ARMS == ("A", "B", "C", "D", "E")
    assert [spec.name for spec in registry.values()] == [
        "TARGET_NATIVE_INIT",
        "CONTROLLED_FROZEN_FUSED",
        "FUSED_TRAINABLE",
        "CURRENT_V3",
        "BEST_SOURCE_TRAINABLE",
    ]
    assert set(AE_ARM_NAMES.values()).isdisjoint(set(ABLATION_ARM_NAMES))
    assert tuple(LEGACY_MECHANISM_ARMS) == ABLATION_ARM_NAMES
    for forbidden in LEGACY_MECHANISM_ARMS:
        with pytest.raises(AEContractError, match="legacy mechanism arm"):
            ae_arm_spec(forbidden)
    with pytest.raises(AEContractError, match="unknown AE arm"):
        ae_arm_spec("F")
    with pytest.raises(AEContractError, match="unknown init_mode"):
        AEAblationArmSpec(
            arm="A",
            name="TARGET_NATIVE_INIT",
            init_mode="glorot",
            target_backbone_trainable=True,
            revin_enabled=False,
            transfer_branch_enabled=False,
            gate_mode="zero",
            use_adversarial=False,
        )
    assert FORMAL_METHODS == ("No-TL", "SS-TL", "MSWA-TL", "MSSB-TL", "MSML-TL", "MSML-TL-RFE")
    assert PARENT_METHOD not in FORMAL_METHODS


def test_training_recipe_equals_frozen_v3():
    recipe = ae_training_recipe()
    assert recipe == v3_training_recipe()
    assert recipe == {
        "loss_name": "huber",
        "huber_delta": 1.0,
        "lr_schedule": "cosine",
        "use_ema": True,
        "ema_momentum": 0.99,
    }
    with pytest.raises(AEContractError, match="AE_RECIPE_MISMATCH"):
        require_v3_recipe(None)
    with pytest.raises(AEContractError, match="AE_RECIPE_MISMATCH"):
        require_v3_recipe({"loss_name": "mse", "huber_delta": 1.0, "lr_schedule": "constant", "use_ema": False, "ema_momentum": 0.99})


def test_lr_equality_matches_frozen_v3_map():
    identities = enumerate_planned_runs()
    by_cell = {}
    for item in identities:
        by_cell.setdefault(item.canonical_tuple, []).append((item.arm, item.selected_lr, item.selected_lr_label))
    assert len(by_cell) == EXPECTED_CANONICAL_CELLS
    for (_dataset, _target, _h, _s, _sc), rows in by_cell.items():
        assert [arm for arm, _, _ in rows] == list(AE_ARMS)
        lrs = {lr for _, lr, _ in rows}
        labels = {label for _, _, label in rows}
        assert len(lrs) == 1
        assert len(labels) == 1
    for (dataset, target), label in V3_LR_MAP.items():
        value, resolved = resolve_cell_lr(dataset, target)
        assert resolved == label
        for arm in AE_ARMS:
            assert resolve_cell_lr(dataset, target) == (value, label)


def test_transfer_disabled_on_every_arm():
    for spec in ae_arm_registry().values():
        assert spec.transfer_branch_enabled is False
        assert spec.gate_mode == "zero"
        assert spec.use_adversarial is False


def test_bc_same_fused_init(constructed_arms):
    assert constructed_arms["B"].target_initial_conv_digest == constructed_arms["C"].target_initial_conv_digest
    assert constructed_arms["B"].fused_init_digest == constructed_arms["C"].fused_init_digest
    assert constructed_arms["D"].target_initial_conv_digest == constructed_arms["C"].target_initial_conv_digest


def test_bc_trainable_variable_difference(constructed_arms):
    b = constructed_arms["B"]
    c = constructed_arms["C"]
    for name in CONV_LAYER_NAMES:
        assert b.target_cnn.get_layer(name).trainable is False
        assert c.target_cnn.get_layer(name).trainable is True
    assert b.trainable_includes_head_target is True
    assert c.trainable_includes_head_target is True
    assert b.trainable_includes_conv is False
    assert c.trainable_includes_conv is True


def test_ac_architecture_only_init_differs(constructed_arms):
    report = pairwise_single_factor_contracts(constructed_arms)
    assert report["A_C_SINGLE_FACTOR_CONTRACT"] == "PASS"
    a = arm_factor_payload(constructed_arms["A"])
    c = arm_factor_payload(constructed_arms["C"])
    assert a["init_mode"] == "native"
    assert c["init_mode"] == "fused"
    assert a["target_backbone_trainable"] is True
    assert c["target_backbone_trainable"] is True
    assert a["revin_enabled"] is False
    assert c["revin_enabled"] is False
    assert a["head_target_init_digest"] == c["head_target_init_digest"]
    assert a["target_initial_conv_digest"] != c["target_initial_conv_digest"]
    assert constructed_arms["A"].source_derived_conv_params is False


def test_e_rank1_only_does_not_copy_dense_out(synthetic_stage_a, constructed_arms):
    e_cnn = constructed_arms["E"].target_cnn
    c_cnn = constructed_arms["C"].target_cnn
    for name in CONV_LAYER_NAMES:
        for weight in e_cnn.get_layer(name).get_weights():
            np.testing.assert_allclose(weight, SYNTHETIC_SOURCE_FILLS[0], atol=1e-6)
        fused_fill = (
            SYNTHETIC_SOURCE_WEIGHTS[0] * SYNTHETIC_SOURCE_FILLS[0]
            + SYNTHETIC_SOURCE_WEIGHTS[1] * SYNTHETIC_SOURCE_FILLS[1]
            + SYNTHETIC_SOURCE_WEIGHTS[2] * SYNTHETIC_SOURCE_FILLS[2]
        )
        for weight in c_cnn.get_layer(name).get_weights():
            np.testing.assert_allclose(weight, fused_fill, atol=1e-6)
    e_dense = e_cnn.get_layer("dense_out").get_weights()
    native_dense = [
        array
        for array in synthetic_stage_a.native_target_snapshot[-2:]
    ]
    for left, right in zip(e_dense, native_dense):
        np.testing.assert_array_equal(left, right)
    for weight in e_dense:
        assert not np.allclose(weight, SYNTHETIC_DENSE_OUT_FILL)
    rank1 = {
        name: [np.array(array, copy=True) for array in synthetic_stage_a.source_conv_params[0][name]]
        for name in CONV_LAYER_NAMES
    }
    rebuilt = best_source_target_init_snapshot(
        synthetic_stage_a.input_shape,
        synthetic_stage_a.learning_rate,
        rank1,
        seed=synthetic_stage_a.seed,
    )
    assert constructed_arms["E"].init_snapshot[0].shape == rebuilt[0].shape


def test_c_weighted_fusion_matches_beta_sum(synthetic_stage_a, constructed_arms):
    expected = (
        SYNTHETIC_SOURCE_WEIGHTS[0] * SYNTHETIC_SOURCE_FILLS[0]
        + SYNTHETIC_SOURCE_WEIGHTS[1] * SYNTHETIC_SOURCE_FILLS[1]
        + SYNTHETIC_SOURCE_WEIGHTS[2] * SYNTHETIC_SOURCE_FILLS[2]
    )
    for name in CONV_LAYER_NAMES:
        arrays = constructed_arms["C"].target_initial_conv[name]
        for array in arrays:
            np.testing.assert_allclose(array, expected, atol=1e-6)
        for array in synthetic_stage_a.fused_params[name]:
            np.testing.assert_allclose(array, expected, atol=1e-6)


def test_e_source_identity_is_rank1(synthetic_stage_a, constructed_arms):
    assert constructed_arms["E"].best_source_identity == tuple(synthetic_stage_a.selected_sources[0]["source_key"])
    assert constructed_arms["E"].best_source_identity == synthetic_stage_a.source_identities[0]
    distances = np.asarray(synthetic_stage_a.source_distances)
    assert int(np.argmin(distances)) == 0
    assert constructed_arms["C"].best_source_identity == constructed_arms["E"].best_source_identity


def test_cd_same_fused_init(constructed_arms):
    assert constructed_arms["C"].target_initial_conv_digest == constructed_arms["D"].target_initial_conv_digest
    assert constructed_arms["C"].fused_init_digest == constructed_arms["D"].fused_init_digest


def test_cd_only_revin(constructed_arms):
    report = pairwise_single_factor_contracts(constructed_arms)
    assert report["C_D_SINGLE_FACTOR_CONTRACT"] == "PASS"
    c = arm_factor_payload(constructed_arms["C"])
    d = arm_factor_payload(constructed_arms["D"])
    assert c["revin_enabled"] is False
    assert d["revin_enabled"] is True
    assert c["target_backbone_trainable"] is True
    assert d["target_backbone_trainable"] is True
    assert c["init_mode"] == d["init_mode"] == "fused"
    assert c["selected_lr"] == d["selected_lr"]
    assert c["recipe"] == d["recipe"]
    assert c["head_target_init_digest"] == d["head_target_init_digest"]
    assert c["transfer_branch_enabled"] is False
    assert d["gate_mode"] == "zero"
    assert constructed_arms["D"].revin is not None
    assert constructed_arms["C"].revin is None
    n_features = int(constructed_arms["D"].target_cnn.input_shape[-1])
    assert constructed_arms["D"].revin_parameter_count == 2 * n_features
    assert constructed_arms["C"].revin_parameter_count == 0


def test_common_head_init(constructed_arms):
    assert verify_common_head_init(constructed_arms) == "PASS"
    digests = [constructed_arms[arm].head_target_init_digest for arm in AE_ARMS]
    assert len(set(digests)) == 1


def test_d_identity_equals_frozen_v3():
    flags = assert_d_equals_frozen_v3()
    assert flags["ARM_D_PARENT_V3_IDENTITY"] == "PASS"
    assert flags["parent_method"] == "FUSED_TARGET_ONLY_V3"
    assert flags["parent_v3_architecture_digest"] == EXPECTED_V3_ARCHITECTURE_DIGEST
    assert v3_architecture_digest() == EXPECTED_V3_ARCHITECTURE_DIGEST
    spec = ae_arm_spec("D")
    assert spec.init_mode == "fused"
    assert spec.target_backbone_trainable is True
    assert spec.revin_enabled is True


def test_target_split_identity(constructed_arms):
    splits = [tuple(sorted(constructed_arms[arm].target_split_digest.items())) for arm in AE_ARMS]
    assert len(set(splits)) == 1


def test_metric_protocol_identity():
    contract = metric_contract()
    assert contract["primary_metric"] == "rmse_original"
    assert contract["primary_metric_space"] == ORIGINAL_SALES_SPACE
    assert contract["sales_value_policy"] == SALES_VALUE_POLICY
    assert contract["revin_inverse_before_scaler"] == "D_only"
    assert "mae_original" in contract["secondary_metrics"]
    analysis = analysis_contract()
    assert analysis["planned_contrasts"] == ["C_vs_B", "C_vs_A", "C_vs_E", "D_vs_C"]
    assert analysis["condition_count"] == 18
    assert analysis["multiple_comparison"] == "Holm"
    assert analysis["forbid_post_hoc_contrast_changes"] is True


def test_matrix_enumeration_270_unique():
    cells = enumerate_canonical_cells()
    runs = enumerate_planned_runs()
    assert len(cells) == EXPECTED_CANONICAL_CELLS == 54
    assert len(runs) == EXPECTED_RUNS == 270
    assert HORIZONS == (1, 3, 5)
    assert SEEDS == (42, 43, 44)
    assert len({item.identity_tuple for item in runs}) == 270
    assert [item.arm for item in runs[:5]] == list(AE_ARMS)
    row = planned_result_row(runs[0])
    assert row["status"] == "PLANNED"
    assert row["rmse_original"] is None
    assert row["smape_original"] is None
    assert row["mae_original"] is None
    assert set(provenance_schema_fields()) >= {
        "rmse_original",
        "head_target_init_digest",
        "best_source_snapshot_digest",
        "parent_v3_architecture_digest",
    }


def test_scenario_frozen_without():
    assert SCENARIO == "without"
    for item in enumerate_planned_runs():
        assert item.scenario == "without"
    manifest = load_ae_manifest(PROJECT_ROOT)
    assert manifest["scenario"] == "without"


def test_training_guard_blocks_without_unlock():
    with pytest.raises(AETrainingBlocked, match="FORMAL_TRAINING_BLOCKED"):
        run_ae_formal_matrix(execute=False)
    with pytest.raises(AETrainingBlocked, match="FORMAL_TRAINING_BLOCKED"):
        run_ae_formal_matrix(execute=True, environ={})
    with pytest.raises(AETrainingBlocked, match="FORMAL_TRAINING_BLOCKED"):
        assert_formal_training_permitted(execute=True, environ={})
    with pytest.raises(AETrainingBlocked, match="FORMAL_TRAINING_BLOCKED"):
        assert_formal_training_permitted(
            execute=False, environ={FORMAL_TRAINING_UNLOCK_ENV: "YES"}
        )
    assert AE_EXECUTION_BODY_ENABLED is True
    assert AE_FORMAL_EXECUTION_ENABLED is True
    assert AE_FORMAL_EXECUTION_IMPLEMENTED is True


def test_preflight_does_not_call_training_entrypoints(monkeypatch, tmp_path):
    def _boom(*_args, **_kwargs):
        raise AssertionError("TRAINING_FORBIDDEN_DURING_AE_PREFLIGHT")

    import src.transfer_methods.ms_katl_lite as lite
    import src.models.cnn_model as cnn_model
    import tensorflow as tf

    monkeypatch.setattr(lite, "prepare_ms_katl_stage_a", _boom)
    monkeypatch.setattr(lite, "train_ms_katl_head", _boom)
    monkeypatch.setattr(cnn_model, "create_training_callbacks", _boom)

    class _ForbiddenOptimizer:
        def apply_gradients(self, *_args, **_kwargs):
            raise AssertionError("TRAINING_FORBIDDEN_DURING_AE_PREFLIGHT")

    monkeypatch.setattr(tf.keras.Model, "fit", _boom, raising=False)
    monkeypatch.setattr(tf.keras.optimizers.Optimizer, "apply_gradients", _boom, raising=False)

    result = run_ae_preflight(
        project_root=PROJECT_ROOT,
        write_plan_path=tmp_path / "AE_FORMAL_PLAN.json",
        include_construction_contracts=True,
    )
    assert result["TRAINING_STARTED"] == "NO"
    assert result["training_guard"] == "PASS"
    assert "FORMAL_TRAINING_BLOCKED=YES" in result["text"]
    assert result["READY_FOR_FORMAL_AE_RUN"] == "YES"
    plan = json.loads((tmp_path / "AE_FORMAL_PLAN.json").read_text(encoding="utf-8"))
    assert plan["status"] == "PLANNED"
    assert plan["expected_runs"] == 270
    assert all(row["rmse_original"] is None for row in plan["runs"])
    assert all(row["status"] == "PLANNED" for row in plan["runs"])


def test_pairwise_report_and_manifest_match_code():
    manifest = load_ae_manifest(PROJECT_ROOT)
    assert manifest["parent_v3_architecture_digest"] == EXPECTED_V3_ARCHITECTURE_DIGEST
    assert manifest["expected_canonical_cells"] == 54
    assert manifest["expected_runs"] == 270
    assert manifest["training_recipe"] == ae_training_recipe()
    assert manifest["execution_body_enabled"] is True
    assert manifest["formal_execution_implemented"] is True
    assert manifest["formal_execution_enabled"] is True


def test_write_formal_plan_never_writes_metrics(tmp_path):
    path = write_formal_plan(tmp_path / "plan.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "PLANNED"
    assert "OK" not in {row["status"] for row in payload["runs"]}
    assert all(row["rmse_original"] is None and row["mae_original"] is None for row in payload["runs"])


def test_real_cell_routing_stage_a_call_count_and_keys():
    from src.transfer_methods.ms_katl_ae_execution import (
        AEExecutionMode,
        AEExecutionPhase,
        FakeRealStageAFactory,
        assert_d_parent_identity,
        assert_lr_equality,
        assert_phase_allowed,
        assert_recipe_equality,
        classify_paired_bundle,
        dry_run_canonical_graph,
        enumerate_real_arm_plans,
        enumerate_real_canonical_cell_plans,
        validate_resume_root,
    )

    cells = enumerate_real_canonical_cell_plans(PROJECT_ROOT)
    assert len(cells) == 54
    assert len({cell.stage_a_key for cell in cells}) == 54
    assert all(cell.scenario == "without" for cell in cells)
    assert all(cell.source_ids_status == "SOURCE_IDS_DEFERRED_TO_FORMAL_STAGE_A" for cell in cells)
    factory = FakeRealStageAFactory()
    graph = dry_run_canonical_graph(cells, factory=factory)
    assert graph["stage_a_calls"] == 54
    assert len(graph["arm_plans"]) == 270
    assert len({item.run_key for item in graph["arm_plans"]}) == 270
    consumed = {item.arm: item.source_init_consumed for item in graph["arm_plans"] if item.cell.horizon == 1 and item.cell.seed == 42 and item.cell.dataset == "D4"}
    assert consumed["A"] is False
    assert consumed["B"] is True
    assert consumed["C"] is True
    assert consumed["D"] is True
    assert consumed["E"] is True
    assert assert_lr_equality(graph["arm_plans"]) == "PASS"
    assert assert_recipe_equality(graph["arm_plans"]) == "PASS"
    assert assert_d_parent_identity(graph["arm_plans"]) == "PASS"
    assert classify_paired_bundle({"A": "OK", "B": "OK", "C": "OK", "D": "OK", "E": "FAILED"}) == "PAIR_INCOMPLETE"
    assert_phase_allowed(AEExecutionMode.FORMAL, AEExecutionPhase.STAGE_A_TRAIN)
    with pytest.raises(AETrainingBlocked, match="TRAINING_FORBIDDEN_DURING_REAL_DRY_RUN"):
        assert_phase_allowed(AEExecutionMode.DRY_RUN, AEExecutionPhase.TRAIN_STAGE_B)
    with pytest.raises(AETrainingBlocked, match="TRAINING_FORBIDDEN_DURING_REAL_DRY_RUN"):
        assert_phase_allowed(AEExecutionMode.DRY_RUN, AEExecutionPhase.STAGE_A_TRAIN)
    with pytest.raises(AEContractError, match="resume root missing"):
        validate_resume_root(PROJECT_ROOT / "outputs" / "preflight" / "missing_resume")


def test_arm_build_order_invariance():
    from src.transfer_methods.ms_katl_ae_execution import arm_build_order_invariance

    assert arm_build_order_invariance() == "PASS"


def test_real_dry_run_does_not_call_training(monkeypatch, tmp_path):
    def _boom(*_args, **_kwargs):
        raise AssertionError("TRAINING_FORBIDDEN_DURING_REAL_DRY_RUN")

    import src.transfer_methods.ms_katl_lite as lite
    import src.transfer_methods.msml_tl as msml
    import tensorflow as tf
    from src.transfer_methods.ms_katl_ae_execution import run_ae_dry_run

    monkeypatch.setattr(lite, "prepare_ms_katl_stage_a", _boom)
    monkeypatch.setattr(lite, "train_ms_katl_head", _boom)
    monkeypatch.setattr(msml, "train_source_cnn_for_msml", _boom)
    monkeypatch.setattr(tf.keras.Model, "fit", _boom, raising=False)
    monkeypatch.setattr(tf.keras.optimizers.Optimizer, "apply_gradients", _boom, raising=False)

    result = run_ae_dry_run(
        project_root=PROJECT_ROOT,
        write_artifacts=False,
        include_construction=True,
    )
    assert result["STAGE_A_MOCK_CALLS"] == 54
    assert result["PLANNED_RUNS"] == 270
    assert result["UNIQUE_RUN_KEYS"] == 270
    assert result["UNIQUE_STAGE_A_KEYS"] == 54
    assert result["TRAINING_STARTED"] == "NO"
    assert result["REAL_DATA_ROUTING"] == "PASS"
    assert result["d4_trace"]["target_identity"] == "166/311"
    assert result["d4_trace"]["source_ids_status"] == "SOURCE_IDS_DEFERRED_TO_FORMAL_STAGE_A"
    assert result["READY_FOR_FORMAL_AE_EXECUTION_ENABLE"] == "YES"
