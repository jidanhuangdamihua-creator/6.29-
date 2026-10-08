"""Contract checks for the M1–M7 runner. These do not train the formal budget."""

from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd

from src.transfer_methods.ms_katl_mechanism_ablation import run_ablation_arm
from src.transfer_methods.sita_m1_m7_ablation import (
    GROUPS,
    RESUME_POLICY,
    M7ContractError,
    assert_output_isolation,
    assert_resume_contract,
    build_probe_metadata,
    default_output_root,
    evaluate_init_pairing,
    execution_order,
    group_is_complete,
    render_plan,
    safe_linear_cka,
    select_probe_indices,
    write_plan_file,
)
from src.transfer_methods.sita_mechanism_diagnostics_contract import (
    MechanismDiagnosticsContractError,
)


def test_execution_order_captures_m6_first():
    assert execution_order(GROUPS)[0] == "M6"
    assert execution_order(("M1", "M3", "M6")) == ("M6", "M1", "M3")


def test_resume_policy_is_group_level_only():
    assert RESUME_POLICY["granularity"] == "group"
    assert RESUME_POLICY["epoch_or_step_resume"] is False
    assert "Adam moments" in RESUME_POLICY["not_restored"]


def test_resume_contract_rejects_a_different_source_digest():
    saved = {
        "protocol": "SITA_M1_M7_ABLATION",
        "dataset": "D1",
        "target": "1/10",
        "horizon": 1,
        "seed": 42,
        "source_learning_rate": "1e-4",
        "target_learning_rate": "3e-5",
        "target_epochs": 50,
        "source_epochs": 50,
        "batch_size": 16,
        "source_weight_digests": ["a", "b", "c"],
        "fused_params_digest": "fused",
        "resume_policy": {"granularity": "group"},
    }
    current = dict(saved)
    current["fused_params_digest"] = "other"
    try:
        assert_resume_contract(saved, current)
    except M7ContractError as exc:
        assert "fused_params_digest" in str(exc)
    else:
        raise AssertionError("mismatched fusion digest was accepted")


def test_probe_metadata_records_partition_ids_and_shared_preprocessing():
    rows = [
        {"window_start": "2020-01-01", "forecast_origin_date": "2020-01-10", "target_date": "2020-01-11"},
        {"window_start": "2020-01-02", "forecast_origin_date": "2020-01-11", "target_date": "2020-01-12"},
        {"window_start": "2020-01-11", "forecast_origin_date": "2020-01-20", "target_date": "2020-01-21"},
    ]
    indices = select_probe_indices(len(rows), stride=2)
    meta = build_probe_metadata(rows, indices, stride=2)
    assert meta["partition"] == "test"
    assert meta["labels_used"] is False
    assert meta["window_ids"] == [0, 2]
    assert meta["n_samples"] == 2
    assert meta["date_range"]["forecast_origin_min"] == "2020-01-10"
    assert meta["date_range"]["target_date_max"] == "2020-01-21"
    assert "shared MinMax" in meta["input_preprocessing"]
    assert "RevIN" in meta["within_group_revin"]
    assert meta["cka_uses_checkpoint_selection"] is False
    assert meta["iid_samples"] is False


def test_degenerate_cka_is_recorded_without_a_filled_value():
    left = np.zeros((4, 3))
    right = np.ones((4, 3))
    result = safe_linear_cka(left, right)
    assert result["status"] == "UNDEFINED"
    assert result["value"] is None
    assert result["reason"]
    try:
        from src.transfer_methods.sita_mechanism_diagnostics import linear_cka

        linear_cka(left, right)
    except MechanismDiagnosticsContractError:
        pass
    else:
        raise AssertionError("zero-variance CKA should stay undefined")


def test_init_pairing_requires_m6_head_and_nearest_source():
    common_head = "head"
    native = "native"
    fused = "fused"
    nearest = "nearest"
    revin = "affine"
    snapshots = {}
    for group, conv, affine in (
        ("M1", native, None),
        ("M2", fused, None),
        ("M3", fused, None),
        ("M4", native, revin),
        ("M5", fused, revin),
        ("M6", fused, revin),
        ("M7", nearest, revin),
    ):
        snapshots[group] = {
            "conv": conv,
            "head_target": common_head,
            "revin_affine": affine,
            "nearest_source_conv": nearest if group == "M7" else None,
        }
    assert evaluate_init_pairing(snapshots)["status"] == "PASS"
    snapshots["M1"]["head_target"] = "other"
    assert evaluate_init_pairing(snapshots)["status"] == "FAIL"


def test_plan_is_not_complete(tmp_path):
    text = render_plan(dataset="D1", horizon=1, seed=42)
    assert "TRAINING_CALLED=NO" in text
    assert "EXECUTION_STATUS=PLANNED" in text
    assert "TARGET_LR_D1=3e-5" in text
    assert "TARGET_LR_D5=3e-4" in text
    output = tmp_path / "outputs" / "sita_m1_m7_ablation" / "plan-only"
    write_plan_file(
        output,
        text,
        dataset="D1",
        horizon=1,
        seed=42,
        groups=GROUPS,
        project_root=tmp_path,
    )
    payload = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    assert payload["status"] == "PLANNED"
    assert payload["training_called"] is False


def test_planned_directory_is_not_a_completed_group(tmp_path):
    group_dir = tmp_path / "M1"
    group_dir.mkdir()
    (group_dir / "status.json").write_text(json.dumps({"status": "PLANNED"}), encoding="utf-8")
    assert group_is_complete(group_dir) is False


def test_output_isolation_rejects_existing_result_roots():
    try:
        assert_output_isolation(default_output_root().parent / "runs" / "sita_m1_m7_ablation")
    except M7ContractError:
        pass
    else:
        raise AssertionError("historical runs root was accepted")
    isolated = assert_output_isolation(default_output_root() / "d1-h1-s42")
    assert isolated.name == "d1-h1-s42"


def test_run_ablation_arm_overrides_default_to_unchanged_behavior():
    signature = inspect.signature(run_ablation_arm)
    assert signature.parameters["init_weight_snapshot"].default is None
    assert signature.parameters["head_target_init_weights"].default is None
    assert signature.parameters["revin_affine_init"].default is None
    assert signature.parameters["capture_weight_states"].default is False
    assert signature.parameters["on_epoch_metric_record"].default is None
    assert signature.parameters["record_update_counters"].default is False


def test_update_contract_uses_weight_movement_when_names_have_no_layer_tokens():
    from src.transfer_methods.sita_m1_m7_ablation import GROUP_SPECS, assert_update_contract

    def layer(raw_move, ema_move, shadow):
        return {
            "raw": {"max_abs": raw_move},
            "ema": {"max_abs": ema_move},
            "ema_shadow_includes_layer": shadow,
        }

    drift = {
        "conv_names_in_optimizer": [],
        "trainable_names": ["kernel", "bias"],
        "layers": {
            "conv1": layer(0.001, 0.0002, True),
            "conv2": layer(0.001, 0.0002, True),
            "conv3": layer(0.001, 0.0002, True),
            "head_target": layer(0.001, 0.0002, True),
            "revin_affine": layer(0.001, 0.0002, True),
        },
    }
    assert_update_contract(GROUP_SPECS["M6"], drift)
    frozen = {
        "conv_names_in_optimizer": [],
        "trainable_names": ["kernel"],
        "layers": {
            "conv1": layer(0.0, 0.0, False),
            "conv2": layer(0.0, 0.0, False),
            "conv3": layer(0.0, 0.0, False),
            "head_target": layer(0.001, 0.0002, True),
        },
    }
    assert_update_contract(GROUP_SPECS["M2"], frozen)
    unchanged = {
        "layers": {
            "conv1": layer(0.0, 0.0, True),
            "conv2": layer(0.0, 0.0, True),
            "conv3": layer(0.0, 0.0, True),
            "head_target": layer(0.0, 0.0, True),
            "revin_affine": layer(0.0, 0.0, True),
        },
    }
    observed = assert_update_contract(GROUP_SPECS["M6"], unchanged)
    assert any("unchanged" in item for item in observed["observations"])
    from src.transfer_methods.sita_m1_m7_ablation import recompute_metrics

    frame = pd.DataFrame(
        {
            "y_true_original": [1.0, 2.0, 4.0],
            "y_pred_original": [1.0, 2.0, 5.0],
        }
    )
    metrics = recompute_metrics(frame)
    assert abs(metrics["rmse_original"] - (1.0 / 3.0) ** 0.5) < 1e-12
    assert metrics["smape_original"] > 0.0


def test_epoch_log_counters_and_default_training_path(tmp_path):
    import tensorflow as tf

    from src.transfer_methods.ms_katl_lite import MSKATLLite, train_ms_katl_head
    from src.transfer_methods.sita_m1_m7_ablation import EpochValidationLog

    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 4)).astype(np.float32)
    y = rng.normal(size=(4,)).astype(np.float32)
    features = [rng.normal(size=(4, 4)).astype(np.float32)]
    x_val = rng.normal(size=(2, 4)).astype(np.float32)
    y_val = rng.normal(size=(2,)).astype(np.float32)
    f_val = [rng.normal(size=(2, 4)).astype(np.float32)]
    prior = np.ones((1,), dtype=np.float32)
    log_path = tmp_path / "epoch_log.csv"
    writer = EpochValidationLog(log_path)
    tf.keras.backend.clear_session()
    recorded = train_ms_katl_head(
        MSKATLLite(d_model=4, num_sources=1, transfer_branch_enabled=False, gate_mode="zero"),
        x, features, y, features, prior,
        x_val=x_val, f_t_val=f_val, y_val=y_val,
        epochs=2, batch_size=2, learning_rate=1e-3, seed=7,
        loss_name="huber", lr_schedule="cosine", use_ema=True, ema_momentum=0.99,
        on_epoch_metric_record=writer, record_update_counters=True,
    )
    frame = pd.read_csv(log_path)
    assert len(frame) == 2
    assert list(frame["epoch"]) == [1, 2]
    assert frame["val_metric"].notna().all()
    assert set(frame["weight_variant"]) == {"RAW"}
    assert set(frame["metric_space"]) == {"scaled_target"}
    assert bool(frame["used_for_checkpoint_selection"].iloc[0]) is False
    counters = recorded["update_counters"]
    assert counters["optimizer_update_count_evidence"] == "observed_optimizer_iterations_delta"
    assert counters["ema_update_count_evidence"] == "observed_at_ema_update"
    assert counters["ema_initialization_count"] == 1
    assert counters["ema_update_count"] == counters["optimizer_update_count"]
    assert counters["optimizer_update_count"] == 4
    tf.keras.backend.clear_session()
    plain = train_ms_katl_head(
        MSKATLLite(d_model=4, num_sources=1, transfer_branch_enabled=False, gate_mode="zero"),
        x, features, y, features, prior,
        x_val=x_val, f_t_val=f_val, y_val=y_val,
        epochs=2, batch_size=2, learning_rate=1e-3, seed=7,
        loss_name="huber", lr_schedule="cosine", use_ema=True, ema_momentum=0.99,
    )
    assert "update_counters" not in plain
    assert len(plain["history"]["forecast_loss"]) == 2
    assert len(recorded["history"]["forecast_loss"]) == 2


def test_post_training_failure_keeps_written_epoch_log(tmp_path):
    from src.transfer_methods.sita_m1_m7_ablation import (
        EpochValidationLog,
        M7ContractError,
        mark_group_failed,
    )

    group_dir = tmp_path / "M6"
    group_dir.mkdir()
    writer = EpochValidationLog(group_dir / "epoch_log.csv")
    writer(
        {
            "epoch": 1,
            "global_step": 1,
            "train_loss": 0.2,
            "train_loss_name": "huber",
            "val_metric_name": "rmse",
            "val_metric": 0.5,
            "val_metric_missing_reason": "",
            "learning_rate": 1e-4,
            "metric_space": "scaled_target",
            "weight_variant": "RAW",
            "used_for_checkpoint_selection": False,
        }
    )
    try:
        raise M7ContractError("post-training audit failed")
    except M7ContractError as exc:
        mark_group_failed(group_dir, exc)
    assert "0.5" in (group_dir / "epoch_log.csv").read_text(encoding="utf-8")
    assert (group_dir / "epoch_log.csv").is_file()
    assert json.loads((group_dir / "status.json").read_text(encoding="utf-8"))["status"] == "FAILED"


def test_missing_validation_log_is_not_diagnostic_complete(tmp_path):
    from src.transfer_methods.sita_m1_m7_ablation import assess_epoch_validation_log

    missing = assess_epoch_validation_log(tmp_path / "absent.csv")
    assert missing["EPOCH_VALIDATION_LOG_COMPLETE"] is False
    blank = tmp_path / "epoch_log.csv"
    blank.write_text("epoch,val_rmse\n1,\n2,\n", encoding="utf-8")
    assessed = assess_epoch_validation_log(blank, expected_epochs=2)
    assert assessed["EPOCH_VALIDATION_LOG_COMPLETE"] is False
    assert assessed["reason"] == "validation_metric_absent_without_reason"


def test_parameter_map_uses_layer_objects_when_names_lack_semantic_tokens():
    from src.transfer_methods.ms_katl_mechanism_ablation import build_prediction_parameter_map

    class Variable:
        def __init__(self, name, shape):
            self.name = name
            self.shape = shape
            self.trainable = True

    class Layer:
        def __init__(self, weights):
            self.weights = weights

    conv_kernel = Variable("kernel", (2, 1, 3))
    conv_bias = Variable("bias", (3,))
    head_kernel = Variable("kernel", (3, 1))
    dense_kernel = Variable("kernel", (8, 1))
    mapping = build_prediction_parameter_map(
        conv_layers={
            "conv1": Layer([conv_kernel, conv_bias]),
            "conv2": Layer([]),
            "conv3": Layer([]),
        },
        head_target=Layer([head_kernel]),
        dense_out=Layer([dense_kernel]),
        trainable_variables=[conv_kernel, conv_bias, head_kernel],
        use_ema=True,
    )
    by_module = {}
    for entry in mapping["entries"]:
        by_module.setdefault(entry["semantic_module"], []).append(entry)
    assert by_module["conv1"][0]["variable_name"] == "kernel"
    assert by_module["conv1"][0]["in_optimizer"] is True
    assert by_module["conv1"][0]["ema_shadow"] is True
    assert by_module["head_target"][0]["role"] == "prediction_output"
    assert by_module["dense_out"][0]["role"] == "cnn_output_not_forecast_head"
    assert by_module["dense_out"][0]["on_prediction_path"] is False
    assert by_module["dense_out"][0]["in_optimizer"] is False


def test_saved_training_finalizes_without_calling_training(tmp_path):
    import inspect

    from src.transfer_methods.sita_m1_m7_ablation import finalize_saved_group, plan_group_execution, save_state

    assert "train_ms_katl_head" not in inspect.getsource(finalize_saved_group)
    group_dir = tmp_path / "M2"
    group_dir.mkdir()
    state = {
        "conv": {
            "conv1": [np.zeros((1, 1), dtype=np.float32)],
            "conv2": [np.zeros((1,), dtype=np.float32)],
            "conv3": [np.ones((1,), dtype=np.float32)],
        },
        "head_target": [np.zeros((1, 1), dtype=np.float32)],
        "cnn_full": [np.zeros((1,), dtype=np.float32)],
        "revin_affine": None,
    }
    for name in ("init_weights.npz", "raw_weights.npz", "ema_weights.npz"):
        save_state(group_dir / name, state)
    assert plan_group_execution(group_dir) == "train"
    (group_dir / "resolved_config.json").write_text("{}", encoding="utf-8")
    (group_dir / "init_snapshot.json").write_text("{}", encoding="utf-8")
    (group_dir / "training_evidence.json").write_text(
        json.dumps(
            {
                "training_finished": True,
                "optimizer_update_count_evidence": "observed_optimizer_iterations_delta",
                "ema_update_count_evidence": "observed_at_ema_update",
            }
        ),
        encoding="utf-8",
    )
    assert plan_group_execution(group_dir) == "finalize_saved_weights"
