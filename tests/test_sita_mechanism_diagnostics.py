"""Numerical and assembly tests for SITA mechanism diagnostics.

No training. Numpy-only: linear CKA, parameter drift, PRIMARY EMA assembly,
and C_INIT → B_FINAL skip/generate.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.transfer_methods.sita_mechanism_diagnostics import (
    assemble_c_init_b_final_frozen_reference,
    assemble_c_init_c_final_primary,
    digest_arrays,
    linear_cka,
    parameter_l2_distance,
    reference_input_digest,
    relative_parameter_drift,
)
from src.transfer_methods.sita_mechanism_diagnostics_contract import (
    COMPARISON_C_INIT_B_FINAL_FROZEN_REFERENCE,
    COMPARISON_C_INIT_C_FINAL,
    MechanismCheckpoint,
    MechanismDiagnosticsContractError,
)


def _c_init(**overrides) -> MechanismCheckpoint:
    weights = [np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)]
    base = MechanismCheckpoint(
        arm="C",
        stage="INIT",
        dataset="D1",
        target="1/10",
        horizon=1,
        seed=42,
        architecture="SITA_CNN_V3",
        reference_input_digest="ref",
        source_informed_init_digest="fused-init-1",
        source_identities=("src-a", "src-b", "src-c"),
        source_weights_digest="weights-1",
        backbone_digest=digest_arrays(weights),
        weight_variant="EMA",
        optimizer_update_count=0,
        captured_before_first_target_optimizer_update=True,
        init_mode="fused",
        backbone_trainable=True,
    )
    return replace(base, **overrides)


def test_linear_cka_identical_is_one():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(32, 8))
    assert linear_cka(x, x) == pytest.approx(1.0, abs=1e-10)
    assert linear_cka(x, -x) == pytest.approx(1.0, abs=1e-10)


def test_linear_cka_is_invariant_to_isotropic_scaling():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(48, 6))
    y = rng.normal(size=(48, 5))
    assert linear_cka(x, y) == pytest.approx(linear_cka(3.0 * x, 0.25 * y), rel=1e-10, abs=1e-10)


def test_linear_cka_near_zero_for_independent_directions():
    n = 64
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = np.stack([np.sin(t), np.sin(2.0 * t)], axis=1)
    y = np.stack([np.cos(t), np.cos(2.0 * t)], axis=1)
    assert linear_cka(x, y) == pytest.approx(0.0, abs=1e-10)


def test_linear_cka_requires_aligned_reference_samples():
    with pytest.raises(MechanismDiagnosticsContractError, match="sample axis mismatch"):
        linear_cka(np.ones((8, 3)), np.ones((7, 3)))
    with pytest.raises(MechanismDiagnosticsContractError, match="at least 2"):
        linear_cka(np.ones((1, 3)), np.ones((1, 3)))


def test_parameter_drift_zero_on_identical_weights():
    weights = [np.arange(6.0).reshape(2, 3), np.ones(3)]
    assert parameter_l2_distance(weights, weights) == pytest.approx(0.0, abs=1e-12)
    assert relative_parameter_drift(weights, weights) == pytest.approx(0.0, abs=1e-12)


def test_parameter_drift_positive_when_weights_move():
    initial = [np.zeros((2, 2), dtype=np.float64)]
    final = [np.eye(2, dtype=np.float64)]
    assert parameter_l2_distance(initial, final) == pytest.approx(np.sqrt(2.0))
    assert relative_parameter_drift(initial, final) > 0.0


def test_assemble_primary_c_init_c_final_on_ema():
    init_w = [np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)]
    final_w = [np.array([[1.0, 0.1], [0.0, 0.9]], dtype=np.float32)]
    rng = np.random.default_rng(2)
    ref = rng.normal(size=(16, 4))
    init_repr = ref @ rng.normal(size=(4, 5))
    final_repr = init_repr + 0.05 * rng.normal(size=init_repr.shape)
    c_init = _c_init(
        reference_input_digest=reference_input_digest(ref),
        backbone_digest=digest_arrays(init_w),
    )
    c_final = replace(
        c_init,
        stage="FINAL",
        optimizer_update_count=8,
        captured_before_first_target_optimizer_update=False,
        backbone_digest=digest_arrays(final_w),
    )
    bundle = assemble_c_init_c_final_primary(
        c_init,
        c_final,
        init_backbone_weights=init_w,
        final_backbone_weights=final_w,
        init_representations=init_repr,
        final_representations=final_repr,
        prediction={"space": "original_sales", "n": 16},
        weight_variant="EMA",
    )
    assert bundle.pairing.comparison == COMPARISON_C_INIT_C_FINAL
    assert bundle.weight_variant == "EMA"
    assert bundle.relative_parameter_drift > 0.0
    assert 0.0 <= bundle.cka <= 1.0
    assert {item.kind for item in bundle.artifacts} == {"prediction", "parameter_drift", "CKA"}
    assert {item.weight_variant for item in bundle.artifacts} == {"EMA"}


def test_assemble_primary_rejects_raw_when_formal_is_ema():
    init_w = [np.eye(2, dtype=np.float32)]
    c_init = _c_init(backbone_digest=digest_arrays(init_w), weight_variant="RAW")
    c_final = replace(
        c_init,
        stage="FINAL",
        optimizer_update_count=3,
        captured_before_first_target_optimizer_update=False,
    )
    x = np.eye(8, dtype=np.float64)
    with pytest.raises(MechanismDiagnosticsContractError, match="PRIMARY outputs must use EMA"):
        assemble_c_init_c_final_primary(
            c_init,
            c_final,
            init_backbone_weights=init_w,
            final_backbone_weights=init_w,
            init_representations=x,
            final_representations=x,
            prediction={},
            weight_variant="RAW",
        )


def test_c_init_b_final_generated_when_comparable_and_skipped_otherwise():
    init_w = [np.eye(2, dtype=np.float32)]
    c_init = _c_init(backbone_digest=digest_arrays(init_w))
    b_init = replace(c_init, arm="B", backbone_trainable=False)
    b_final = replace(
        b_init,
        stage="FINAL",
        optimizer_update_count=9,
        captured_before_first_target_optimizer_update=False,
    )
    generated = assemble_c_init_b_final_frozen_reference(
        c_init,
        b_final,
        b_init=b_init,
        init_backbone_weights=init_w,
        b_final_backbone_weights=init_w,
        force=True,
    )
    assert generated["status"] == "GENERATED"
    assert generated["comparison"] == COMPARISON_C_INIT_B_FINAL_FROZEN_REFERENCE
    assert generated["parameter_l2"] == pytest.approx(0.0, abs=1e-12)

    skipped = assemble_c_init_b_final_frozen_reference(
        c_init,
        replace(b_final, source_informed_init_digest="other-fused"),
        b_init=b_init,
        force=True,
    )
    assert skipped["status"] == "SKIPPED"
    assert "source-informed initialization identity not comparable" in skipped["reason"]
    assert "parameter_l2" not in skipped
