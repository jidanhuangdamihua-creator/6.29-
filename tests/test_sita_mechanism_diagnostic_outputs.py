"""Orchestration tests for mechanism diagnostic outputs. No A–E / 42-cell training."""

from __future__ import annotations

import inspect
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.transfer_methods.sita_mechanism_diagnostic_outputs import (
    CELL_STATUS_FAILED,
    FORMAL_TEST_REFERENCE_SPLIT,
    assert_one_condition_one_digest,
    assert_paired_sample_ids,
    audit_status_from_diffs,
    classify_layer_role,
    consistency_audit_row,
    create_run_root,
    d6_cross_seed_pair_count,
    epoch0_precedes_first_optimizer_update,
    error_by_sales_bin,
    generate_cka_comparison_matrix,
    mark_cell_complete,
    mark_cell_failed,
    parameter_drift_rows,
    pointwise_prediction_rows,
    prepare_run_root,
    primary_transferred_backbone_weights,
    record_diagnostic_epoch_eval,
    reference_scale_available,
    resolve_run_directory,
    resume_cell_decision,
    save_condition_reference,
    select_formal_test_reference_inputs,
    timestamped_run_root,
    unordered_seed_pairs,
    write_pointwise_predictions,
    write_training_dynamics,
    filesystem_safe_id,
)
from src.transfer_methods.sita_mechanism_diagnostics import reference_input_digest
from src.transfer_methods.sita_mechanism_diagnostics_capture import (
    ENABLE_MECHANISM_DIAGNOSTICS_KEY,
    ensure_session,
    select_reference_inputs,
    stage_b_diagnostics_train_kwargs,
)
from src.transfer_methods.sita_mechanism_diagnostics_contract import (
    CANONICAL_CKA_COMPARISONS,
    COMPARISON_A_C_MATCHED_SEED,
    COMPARISON_B_B_CROSS_SEED,
    COMPARISON_B_C_MATCHED_SEED,
    COMPARISON_C_C_CROSS_SEED,
    COMPARISON_C_INIT_B_FINAL_FROZEN_REFERENCE,
    COMPARISON_C_INIT_C_FINAL_ADAPTATION_DISPLACEMENT,
    FORMAL_EVALUATION_WEIGHT_VARIANT,
    MechanismDiagnosticsContractError,
)


def _cell():
    return SimpleNamespace(dataset="D1", target="1/10", horizon=1, seed=42)


def _stage_a():
    return SimpleNamespace(fused_target_digest="fused", source_identities=(), source_weights=())


def _bundle(n_test=6, n_train=99):
    return SimpleNamespace(
        X_test=np.arange(n_test * 2, dtype=np.float32).reshape(n_test, 2),
        X_train=np.ones((n_train, 2), dtype=np.float32),
        X_val=np.arange(8, dtype=np.float32).reshape(4, 2),
        y_val=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        f_t={"val": np.zeros((4, 1), dtype=np.float32)},
    )


def test_formal_test_inputs_enter_cka_reference_builder():
    bundle = _bundle()
    refs = select_formal_test_reference_inputs(bundle)
    np.testing.assert_array_equal(refs, bundle.X_test)
    context = {ENABLE_MECHANISM_DIAGNOSTICS_KEY: True}
    session = ensure_session(context, cell=_cell(), stage_a=_stage_a(), bundle=bundle)
    np.testing.assert_array_equal(session.reference_inputs, bundle.X_test)
    assert session.reference_input_digest == reference_input_digest(bundle.X_test)
    assert context["cka_condition_references"]["D1|1/10|h1"]["split"] == FORMAL_TEST_REFERENCE_SPLIT


def test_x_train_prefix_is_not_primary_cka_reference():
    with pytest.raises(MechanismDiagnosticsContractError, match="X_train_prefix"):
        select_reference_inputs(np.ones((8, 2)))
    bundle = _bundle()
    session = ensure_session(
        {ENABLE_MECHANISM_DIAGNOSTICS_KEY: True},
        cell=_cell(),
        stage_a=_stage_a(),
        bundle=bundle,
    )
    assert session.reference_inputs.shape[0] != bundle.X_train.shape[0]
    source = inspect.getsource(ensure_session)
    assert "select_formal_test_reference_inputs" in source
    assert "X_train_prefix" not in source


def test_one_condition_one_reference_digest_shared_across_arms_seeds():
    bundle = _bundle()
    context = {ENABLE_MECHANISM_DIAGNOSTICS_KEY: True}
    digests = []
    for seed, arm in ((42, "A"), (42, "B"), (43, "C")):
        cell = SimpleNamespace(dataset="D1", target="1/10", horizon=1, seed=seed)
        session = ensure_session(context, cell=cell, stage_a=_stage_a(), bundle=bundle)
        digests.append(session.reference_input_digest)
        context.pop("mechanism_diagnostics_session", None)
    assert len(set(digests)) == 1
    assert_one_condition_one_digest({"D1|1/10|h1": digests})
    with pytest.raises(MechanismDiagnosticsContractError, match="exactly one reference_input_digest"):
        assert_one_condition_one_digest({"D1|1/10|h1": ["aaa", "bbb"]})


def test_reference_npz_and_manifest_use_formal_test_split(tmp_path):
    x_test = np.arange(10, dtype=np.float32).reshape(5, 2)
    saved = save_condition_reference(
        tmp_path,
        "D1|1/10|h1",
        x_test,
        [
            {
                "sample_id": i,
                "date_or_prediction_index": i,
                "split": FORMAL_TEST_REFERENCE_SPLIT,
                "input_shape": "2",
            }
            for i in range(5)
        ],
    )
    assert Path(saved["npz"]).is_file()
    assert Path(saved["manifest"]).is_file()
    assert saved["digest"] == reference_input_digest(x_test)
    text = Path(saved["manifest"]).read_text(encoding="utf-8")
    assert "sample_id" in text and "date_or_prediction_index" in text
    assert FORMAL_TEST_REFERENCE_SPLIT in text


def test_epoch0_is_before_first_optimizer_update():
    lite = (PROJECT_ROOT / "src/transfer_methods/ms_katl_lite.py").read_text(encoding="utf-8")
    assert epoch0_precedes_first_optimizer_update(lite)
    rows = []
    record_diagnostic_epoch_eval(
        rows,
        payload={
            "epoch": 0,
            "before_first_target_optimizer_update": True,
            "optimizer_update_count": 0,
            "weight_variant": "EMA",
            "train_objective_loss": 0.5,
            "validation_objective_loss": 0.7,
            "y_true_original": np.array([10.0, 20.0, 30.0]),
            "y_pred_original": np.array([11.0, 19.0, 28.0]),
        },
        constructed=None,
        bundle=None,
        cell=_cell(),
        arm="C",
    )
    assert rows[0]["epoch"] == 0
    assert rows[0]["before_first_target_optimizer_update"] is True
    assert rows[0]["validation_rmse_original_scale"] > 0
    assert rows[0]["validation_smape_original_scale"] > 0


def test_validation_original_scale_metrics_are_generated():
    rows = []
    record_diagnostic_epoch_eval(
        rows,
        payload={
            "epoch": 1,
            "before_first_target_optimizer_update": False,
            "weight_variant": "EMA",
            "train_objective_loss": 0.2,
            "y_true_original": np.array([100.0, 200.0, 300.0]),
            "y_pred_original": np.array([110.0, 190.0, 310.0]),
        },
        constructed=None,
        bundle=None,
        cell=_cell(),
        arm="B",
    )
    assert rows[0]["validation_rmse_original_scale"] == pytest.approx(10.0)
    assert rows[0]["validation_smape_original_scale"] > 0
    assert rows[0]["weight_variant"] == "EMA"


def test_pointwise_prediction_is_ema_original_scale():
    rows = pointwise_prediction_rows(
        dataset="D6",
        target="CA_1/FOODS_3_080",
        horizon=1,
        seed=42,
        arm="C",
        y_true_original=np.array([10.0, 20.0, 30.0]),
        y_pred_original=np.array([12.0, 18.0, 29.0]),
        sample_ids=["s0", "s1", "s2"],
        dates=["d0", "d1", "d2"],
        weight_variant=FORMAL_EVALUATION_WEIGHT_VARIANT,
    )
    assert rows[0]["weight_variant"] == "EMA"
    assert rows[0]["y_true"] == 10.0
    assert rows[0]["signed_error"] == pytest.approx(2.0)
    assert rows[0]["absolute_error"] == pytest.approx(2.0)
    assert rows[0]["squared_error"] == pytest.approx(4.0)


def test_bc_prediction_sample_ids_align():
    true = np.array([1.0, 2.0, 3.0, 4.0])
    ids = ["a", "b", "c", "d"]
    b_rows = pointwise_prediction_rows(
        dataset="D6", target="t", horizon=1, seed=42, arm="B",
        y_true_original=true, y_pred_original=true + 0.1, sample_ids=ids,
    )
    c_rows = pointwise_prediction_rows(
        dataset="D6", target="t", horizon=1, seed=42, arm="C",
        y_true_original=true, y_pred_original=true - 0.2, sample_ids=ids,
    )
    assert_paired_sample_ids(b_rows, c_rows)
    with pytest.raises(MechanismDiagnosticsContractError, match="sample IDs"):
        assert_paired_sample_ids(b_rows, list(reversed(c_rows)))


def test_canonical_cka_comparison_types_are_generated():
    rng = np.random.default_rng(1)
    digest = "shared"
    records = []
    for seed in (42, 43, 44, 45, 46):
        for arm in ("B", "C"):
            for stage in ("INIT", "FINAL"):
                records.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "stage": stage,
                        "representations": rng.normal(size=(12, 3)),
                        "reference_input_digest": digest,
                    }
                )
        records.append(
            {
                "arm": "A",
                "seed": 42,
                "stage": "FINAL",
                "representations": rng.normal(size=(12, 3)),
                "reference_input_digest": digest,
            }
        )
    rows = generate_cka_comparison_matrix(records, reference_scale_available=True)
    types = {row["comparison_type"] for row in rows}
    assert set(CANONICAL_CKA_COMPARISONS) <= types
    for row in rows:
        assert row["weight_variant"] == "EMA"
        assert row["reference_input_digest"] == digest
        assert "n_samples" in row and "n_features_left" in row and "n_features_right" in row
        if row["n_samples"] < 100:
            assert row["flag"] == "LOW_REFERENCE_SAMPLE_COUNT"


def test_d6_horizon_bb_and_cc_are_10_unique_unordered_pairs():
    seeds = (42, 43, 44, 45, 46)
    pairs = unordered_seed_pairs(seeds)
    assert d6_cross_seed_pair_count(5) == 10
    assert len(pairs) == 10
    assert len(set(pairs)) == 10
    assert all(left < right for left, right in pairs)
    rng = np.random.default_rng(2)
    records = []
    for seed in seeds:
        for arm in ("B", "C"):
            records.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "stage": "FINAL",
                    "representations": rng.normal(size=(8, 2)),
                    "reference_input_digest": "d",
                }
            )
    rows = generate_cka_comparison_matrix(records, reference_scale_available=True)
    bb = {(row["seed_left"], row["seed_right"]) for row in rows if row["comparison_type"] == COMPARISON_B_B_CROSS_SEED}
    cc = {(row["seed_left"], row["seed_right"]) for row in rows if row["comparison_type"] == COMPARISON_C_C_CROSS_SEED}
    assert bb == set(pairs)
    assert cc == set(pairs)
    assert len(bb) == 10 and len(cc) == 10


def test_layer_role_excludes_head_target_from_primary_backbone():
    named = {
        "conv1": [np.eye(2, dtype=np.float32)],
        "conv2": [np.ones((2, 2), dtype=np.float32)],
        "conv3": [np.eye(2, dtype=np.float32) * 2],
        "head_target": [np.array([[99.0, 99.0]], dtype=np.float32)],
    }
    assert classify_layer_role("conv1") == "transferred_backbone"
    assert classify_layer_role("head_target") == "prediction_head"
    rows = parameter_drift_rows(named, named)
    roles = {row["layer"]: row["layer_role"] for row in rows}
    assert roles["head_target"] == "prediction_head"
    backbone = primary_transferred_backbone_weights(named)
    assert all(arr.shape != (1, 2) for arr in backbone)
    assert not any(np.allclose(arr, 99.0) for arr in backbone)


def test_consistency_pass_review_fail_boundaries():
    assert audit_status_from_diffs(0.49, 0.49) == "PASS"
    assert audit_status_from_diffs(0.5, 0.0) == "REVIEW"
    assert audit_status_from_diffs(2.0, 2.0) == "REVIEW"
    assert audit_status_from_diffs(2.01, 0.0) == "FAIL"
    assert audit_status_from_diffs(0.0, 2.01) == "FAIL"
    near_zero = consistency_audit_row(
        dataset="D1", target="1/10", horizon=1, seed=42, arm="C",
        formal_rmse=0.0, rerun_rmse=0.001, formal_smape=1.0, rerun_smape=1.0,
    )
    assert "near 0" in near_zero["notes"]
    assert "rounding" in near_zero["notes"]
    assert near_zero["replaces_frozen_results"] is False


def test_complete_cell_is_skipped_on_resume(tmp_path):
    from src.transfer_methods.sita_mechanism_diagnostic_execution import persist_synthetic_cell_artifacts
    from src.transfer_methods.sita_mechanism_diagnostic_rerun import DiagnosticRerunCell

    cell = DiagnosticRerunCell(dataset="D1", target="1/10", horizon=1, seed=42, arm="C")
    cid = cell.cell_key
    persist_synthetic_cell_artifacts(tmp_path, cell)
    mark_cell_complete(tmp_path, cid, {"cell_id": cid})
    decision = resume_cell_decision(
        tmp_path,
        cid,
        expected_config_digest="cfg",
        expected_matrix_digest="mat",
        stored_config_digest="cfg",
        stored_matrix_digest="mat",
    )
    assert decision == "SKIP"
    original = (tmp_path / "status" / f"{filesystem_safe_id(cid)}.COMPLETE").read_text(encoding="utf-8")
    decision2 = resume_cell_decision(
        tmp_path,
        cid,
        expected_config_digest="cfg",
        expected_matrix_digest="mat",
        stored_config_digest="cfg",
        stored_matrix_digest="mat",
    )
    assert decision2 == "SKIP"
    assert (tmp_path / "status" / f"{filesystem_safe_id(cid)}.COMPLETE").read_text(encoding="utf-8") == original


def test_failed_cell_can_resume(tmp_path):
    cid = "D1|1/10|h1|s42|B"
    mark_cell_failed(tmp_path, cid, {"cell_id": cid, "status": CELL_STATUS_FAILED})
    decision = resume_cell_decision(
        tmp_path,
        cid,
        expected_config_digest="cfg",
        expected_matrix_digest="mat",
        stored_config_digest="cfg",
        stored_matrix_digest="mat",
    )
    assert decision == "RERUN"


def test_config_and_matrix_digest_mismatch_fail_fast(tmp_path):
    with pytest.raises(MechanismDiagnosticsContractError, match="config digest mismatch"):
        resume_cell_decision(
            tmp_path, "x",
            expected_config_digest="a",
            expected_matrix_digest="b",
            stored_config_digest="z",
            stored_matrix_digest="b",
        )
    with pytest.raises(MechanismDiagnosticsContractError, match="matrix digest mismatch"):
        resume_cell_decision(
            tmp_path, "x",
            expected_config_digest="a",
            expected_matrix_digest="b",
            stored_config_digest="a",
            stored_matrix_digest="z",
        )
    prepare_run_root(tmp_path, config_digest="cfg", matrix_digest="mat", resume=False)
    with pytest.raises(MechanismDiagnosticsContractError, match="config digest mismatch"):
        prepare_run_root(tmp_path, config_digest="other", matrix_digest="mat", resume=True)


def test_new_run_root_does_not_overwrite_history(tmp_path):
    now = datetime(2026, 9, 16, 15, 51, 0)
    first = create_run_root(tmp_path, now=now)
    assert first.name.startswith("mechanism_rerun_")
    sentinel = first / "keep.txt"
    sentinel.write_text("history", encoding="utf-8")
    with pytest.raises(MechanismDiagnosticsContractError, match="already exists"):
        create_run_root(tmp_path, now=now)
    second = create_run_root(tmp_path, now=datetime(2026, 9, 16, 15, 51, 1))
    assert first != second
    assert sentinel.read_text(encoding="utf-8") == "history"
    assert resolve_run_directory(plan_only=True, resume=False, run_dir=None) is None
    with pytest.raises(MechanismDiagnosticsContractError, match="--resume requires explicit --run-dir"):
        resolve_run_directory(plan_only=False, resume=True, run_dir=None)
    with pytest.raises(MechanismDiagnosticsContractError, match="must not reuse --run-dir"):
        resolve_run_directory(plan_only=False, resume=False, run_dir=str(first))
    assert "rerun_42cell" not in str(timestamped_run_root(now))


def test_diagnostics_disabled_keeps_formal_train_kwargs():
    assert stage_b_diagnostics_train_kwargs({}, arm="C") == {}
    assert stage_b_diagnostics_train_kwargs(None, arm="C") == {}
    formal = (PROJECT_ROOT / "src/transfer_methods/ms_katl_ae_formal.py").read_text(encoding="utf-8")
    lite = (PROJECT_ROOT / "src/transfer_methods/ms_katl_lite.py").read_text(encoding="utf-8")
    assert "stage_b_diagnostics_train_kwargs" in formal
    assert "on_diagnostic_epoch_eval: Optional[Callable[[Mapping[str, object]], None]] = None" in lite
    assert "if on_diagnostic_epoch_eval is not None" in lite


def test_sales_bin_summary_does_not_force_three_bins():
    rows = pointwise_prediction_rows(
        dataset="D1", target="1/10", horizon=1, seed=42, arm="C",
        y_true_original=np.ones(6), y_pred_original=np.ones(6) + 0.1,
    )
    bins = error_by_sales_bin(rows)
    assert len(bins) >= 1
    assert "could not form 3 tertile bins" in bins[0]["notes"]


def test_pilot_reference_scale_unavailable_and_d6_available():
    assert reference_scale_available("D1") is False
    assert reference_scale_available("D4") is False
    assert reference_scale_available("D5") is False
    assert reference_scale_available("D6") is True
    assert COMPARISON_A_C_MATCHED_SEED
    assert COMPARISON_B_C_MATCHED_SEED
    assert COMPARISON_C_INIT_C_FINAL_ADAPTATION_DISPLACEMENT
    assert COMPARISON_C_INIT_B_FINAL_FROZEN_REFERENCE
    assert COMPARISON_B_B_CROSS_SEED
    assert COMPARISON_C_C_CROSS_SEED
