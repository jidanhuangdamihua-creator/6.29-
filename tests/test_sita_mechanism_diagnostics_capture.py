"""Capture-hook wiring tests for SITA mechanism diagnostics.

No AE training. Numpy-only session assembly plus default-off hook kwargs.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.transfer_methods.sita_mechanism_diagnostics import digest_arrays
from src.transfer_methods.sita_mechanism_diagnostics_capture import (
    ENABLE_MECHANISM_DIAGNOSTICS_KEY,
    MECHANISM_DIAGNOSTICS_SESSION_KEY,
    finalize_stage_b_diagnostics,
    mechanism_diagnostics_enabled,
    session_from_cell,
    stage_b_diagnostics_train_kwargs,
)
from src.transfer_methods.sita_mechanism_diagnostics_contract import (
    COMPARISON_C_INIT_B_FINAL_FROZEN_REFERENCE,
    COMPARISON_C_INIT_C_FINAL,
    FORMAL_EVALUATION_WEIGHT_VARIANT,
    MechanismDiagnosticsContractError,
)


def _cell():
    return SimpleNamespace(dataset="D1", target="1/10", horizon=1, seed=42)


def _stage_a():
    return SimpleNamespace(
        fused_target_digest="fused-init-1",
        source_identities=(("src-a", "D1"), ("src-b", "D1"), ("src-c", "D1")),
        source_weights=(0.5, 0.3, 0.2),
    )


def _constructed(arm: str, *, backbone_trainable: bool):
    return SimpleNamespace(
        spec=SimpleNamespace(init_mode="fused", target_backbone_trainable=backbone_trainable),
        target_cnn=None,
        target_extractor=None,
        revin=None,
        target_initial_conv={"conv1": [np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)]},
    )


def _session():
    refs = np.arange(8, dtype=np.float32).reshape(4, 2)
    return session_from_cell(_cell(), _stage_a(), reference_inputs=refs)


def test_diagnostics_disabled_by_default_and_returns_no_train_kwargs():
    assert mechanism_diagnostics_enabled({}) is False
    assert mechanism_diagnostics_enabled(None) is False
    kwargs = stage_b_diagnostics_train_kwargs({}, arm="C")
    assert kwargs == {}
    finalize_stage_b_diagnostics({}, arm="C")
    from src.transfer_methods.ms_katl_lite import DEFAULT_LOSS_NAME, prepare_ms_katl_stage_a, train_ms_katl_head
    import inspect as _inspect

    stage_a_sig = _inspect.signature(prepare_ms_katl_stage_a)
    assert stage_a_sig.parameters["seed"].default is _inspect.Parameter.empty
    head_sig = _inspect.signature(train_ms_katl_head)
    assert head_sig.parameters["loss_name"].default == DEFAULT_LOSS_NAME
    assert head_sig.parameters["on_before_first_target_optimizer_update"].default is None
    assert head_sig.parameters["on_diagnostic_epoch_eval"].default is None


def test_disabled_context_does_not_create_a_session():
    context = {}
    assert stage_b_diagnostics_train_kwargs(context, arm="C", cell=_cell()) == {}
    assert MECHANISM_DIAGNOSTICS_SESSION_KEY not in context


def test_enabled_non_diagnostic_arm_does_not_attach_hook():
    context = {ENABLE_MECHANISM_DIAGNOSTICS_KEY: True}
    assert stage_b_diagnostics_train_kwargs(context, arm="D") == {}


def test_enabled_missing_runtime_objects_fail_closed():
    context = {ENABLE_MECHANISM_DIAGNOSTICS_KEY: True}
    with pytest.raises(MechanismDiagnosticsContractError, match="require cell/stage_a"):
        stage_b_diagnostics_train_kwargs(context, arm="C")


def test_session_assembles_primary_on_ema_and_skips_incomparable_b_final():
    session = _session()
    init_w = [np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)]
    final_w = [np.array([[1.0, 0.1], [0.0, 0.9]], dtype=np.float32)]
    rng = np.random.default_rng(0)
    init_rep = rng.normal(size=(4, 3))
    final_rep = init_rep + 0.05
    session.capture_stage(
        arm="C",
        stage="INIT",
        backbone_weights=init_w,
        representations=init_rep,
        optimizer_update_count=0,
        captured_before_first_target_optimizer_update=True,
        init_mode="fused",
        backbone_trainable=True,
    )
    session.capture_stage(
        arm="C",
        stage="FINAL",
        backbone_weights=final_w,
        representations=final_rep,
        optimizer_update_count=8,
        captured_before_first_target_optimizer_update=False,
        init_mode="fused",
        backbone_trainable=True,
    )
    session.maybe_assemble()
    assert session.primary is not None
    payload = session.primary.as_dict()
    assert payload["status"] == "GENERATED"
    assert payload["comparison"] == COMPARISON_C_INIT_C_FINAL
    assert payload["weight_variant"] == FORMAL_EVALUATION_WEIGHT_VARIANT
    assert payload["relative_parameter_drift"] > 0.0
    assert 0.0 <= payload["cka"] <= 1.0


def test_session_generates_frozen_b_reference_when_backbone_unmoved():
    session = _session()
    fused = [np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32)]
    rng = np.random.default_rng(1)
    reps = rng.normal(size=(4, 2))
    common = dict(
        backbone_weights=fused,
        representations=reps,
        init_mode="fused",
    )
    session.capture_stage(
        arm="C",
        stage="INIT",
        optimizer_update_count=0,
        captured_before_first_target_optimizer_update=True,
        backbone_trainable=True,
        **common,
    )
    session.capture_stage(
        arm="B",
        stage="INIT",
        optimizer_update_count=0,
        captured_before_first_target_optimizer_update=True,
        backbone_trainable=False,
        **common,
    )
    session.capture_stage(
        arm="B",
        stage="FINAL",
        optimizer_update_count=8,
        captured_before_first_target_optimizer_update=False,
        backbone_trainable=False,
        **common,
    )
    session.maybe_assemble()
    assert session.frozen_reference["status"] == "GENERATED"
    assert session.frozen_reference["comparison"] == COMPARISON_C_INIT_B_FINAL_FROZEN_REFERENCE
    assert session.frozen_reference["relative_parameter_drift"] == pytest.approx(0.0)


def test_session_skips_b_final_when_backbone_moved():
    session = _session()
    init_w = [np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)]
    moved = [np.array([[1.0, 0.2], [0.0, 0.8]], dtype=np.float32)]
    reps = np.eye(4, 2, dtype=np.float32)
    session.capture_stage(
        arm="C",
        stage="INIT",
        backbone_weights=init_w,
        representations=reps,
        optimizer_update_count=0,
        captured_before_first_target_optimizer_update=True,
        init_mode="fused",
        backbone_trainable=True,
    )
    session.capture_stage(
        arm="B",
        stage="INIT",
        backbone_weights=init_w,
        representations=reps,
        optimizer_update_count=0,
        captured_before_first_target_optimizer_update=True,
        init_mode="fused",
        backbone_trainable=False,
    )
    session.capture_stage(
        arm="B",
        stage="FINAL",
        backbone_weights=moved,
        representations=reps,
        optimizer_update_count=8,
        captured_before_first_target_optimizer_update=False,
        init_mode="fused",
        backbone_trainable=False,
    )
    session.maybe_assemble()
    assert session.frozen_reference["status"] == "SKIPPED"
    assert session.frozen_reference["reason"]


def test_enabled_hook_is_attached_and_finalize_requires_init():
    refs = np.arange(8, dtype=np.float32).reshape(4, 2)
    bundle = SimpleNamespace(
        X_test=refs,
        X_val=refs,
        y_val=np.zeros(4, dtype=np.float32),
        f_t={"val": np.zeros((4, 1), dtype=np.float32)},
    )
    constructed = _constructed("C", backbone_trainable=True)
    context = {ENABLE_MECHANISM_DIAGNOSTICS_KEY: True}
    kwargs = stage_b_diagnostics_train_kwargs(
        context,
        cell=_cell(),
        arm="C",
        stage_a=_stage_a(),
        constructed=constructed,
        bundle=bundle,
    )
    assert "on_before_first_target_optimizer_update" in kwargs
    assert "on_diagnostic_epoch_eval" in kwargs
    assert "x_val" in kwargs and "y_val" in kwargs and "f_t_val" in kwargs
    session = context[MECHANISM_DIAGNOSTICS_SESSION_KEY]
    assert session.reference_input_digest == digest_arrays([refs])

    with pytest.raises(MechanismDiagnosticsContractError, match="C_INIT was not captured"):
        finalize_stage_b_diagnostics(context, arm="C", constructed=constructed)


def test_enabled_hook_then_finalize_assembles_primary(monkeypatch):
    import src.transfer_methods.sita_mechanism_diagnostics_capture as capture_mod

    refs = np.arange(8, dtype=np.float32).reshape(4, 2)
    init_w = (np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),)
    final_w = (np.array([[1.1, 0.0], [0.0, 0.9]], dtype=np.float32),)
    rng = np.random.default_rng(3)
    init_rep = rng.normal(size=(4, 3))
    final_rep = init_rep + 0.2
    phase = {"name": "INIT"}
    named_init = {
        "conv1": (init_w[0].copy(),),
        "conv2": (np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32),),
        "conv3": (np.array([[3.0, 0.0], [0.0, 3.0]], dtype=np.float32),),
        "head_target": (np.array([[0.5]], dtype=np.float32),),
    }
    named_final = {
        "conv1": (final_w[0].copy(),),
        "conv2": (np.array([[2.1, 0.0], [0.0, 1.9]], dtype=np.float32),),
        "conv3": (np.array([[3.1, 0.0], [0.0, 2.9]], dtype=np.float32),),
        "head_target": (np.array([[0.8]], dtype=np.float32),),
    }

    monkeypatch.setattr(
        capture_mod,
        "snapshot_backbone_from_constructed",
        lambda _constructed: init_w if phase["name"] == "INIT" else final_w,
    )
    monkeypatch.setattr(
        capture_mod,
        "forward_reference_from_constructed",
        lambda _constructed, _refs: init_rep if phase["name"] == "INIT" else final_rep,
    )
    monkeypatch.setattr(
        capture_mod,
        "snapshot_named_layers_from_constructed",
        lambda _constructed: named_init if phase["name"] == "INIT" else named_final,
    )

    bundle = SimpleNamespace(X_test=refs)
    constructed = _constructed("C", backbone_trainable=True)
    context = {ENABLE_MECHANISM_DIAGNOSTICS_KEY: True}
    kwargs = stage_b_diagnostics_train_kwargs(
        context,
        cell=_cell(),
        arm="C",
        stage_a=_stage_a(),
        constructed=constructed,
        bundle=bundle,
    )
    kwargs["on_before_first_target_optimizer_update"](
        {
            "optimizer_update_count": 0,
            "captured_before_first_target_optimizer_update": True,
        }
    )
    session = context[MECHANISM_DIAGNOSTICS_SESSION_KEY]
    assert ("C", "INIT") in session.checkpoints
    assert session.checkpoints[("C", "INIT")].optimizer_update_count == 0
    assert session.checkpoints[("C", "INIT")].weight_variant == FORMAL_EVALUATION_WEIGHT_VARIANT

    phase["name"] = "FINAL"
    kwargs["on_before_ema_assign"]({"optimizer_update_count": 1})
    finalize_stage_b_diagnostics(context, arm="C", constructed=constructed)
    assert session.primary is not None
    payload = session.primary.as_dict()
    assert payload["comparison"] == COMPARISON_C_INIT_C_FINAL
    assert payload["weight_variant"] == FORMAL_EVALUATION_WEIGHT_VARIANT
    assert payload["relative_parameter_drift"] > 0.0
    assert not np.array_equal(
        session.named_layer_weights[("C", "INIT")]["head_target"][0],
        session.named_layer_weights[("C", "FINAL")]["head_target"][0],
    )
