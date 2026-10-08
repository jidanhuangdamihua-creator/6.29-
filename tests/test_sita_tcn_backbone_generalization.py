from __future__ import annotations

import inspect

import numpy as np
import pytest

from scripts.baselines.bl7_tcn_revin import fit_bl7_revin, predict_bl7_revin
from scripts.sita_tcn_protocol import (
    DEFAULT_OUTPUT_ROOT,
    FORBIDDEN_D4_TARGETS,
    METHOD_NAME,
    PILOT_TARGET,
    ROOT,
    SOLIDIFIED_D4_258_DIAGNOSTIC_SOURCES,
    TARGETS,
    TIE_RULE,
    assert_identity_isolation,
    assert_isolated_output_dir,
    assert_paper_target,
    select_formal_sita_sources,
    solidified_diagnostic_sources,
    relative_improvement_pct,
    win_tie_loss,
)
from src.source_selection.source_selector import SourceSelector
from src.transfer_methods.sita_tcn import (
    EXPECTED_TARGET_ONLY_KEYS,
    EXPECTED_TRANSFERABLE_TCN_BACKBONE_KEYS,
    MATERIALIZED_FORWARD_PARITY_MAX_ABS_ERROR,
    SITATCNContractError,
    build_formal_bl7_tcn_revin,
    canonical_source_init_state,
    fit_tcn_revin_with_optional_backbone,
    fuse_materialized_backbones,
    hash_state_dict,
    materialize_effective_tcn_state,
    materialized_forward_parity_error,
    official_inverse_distance_weights as tcn_official_weights,
    overlay_backbone,
    predict_target_only,
    seed_torch,
    split_source_sales,
    train_source_tcn,
    transfer_scope_audit_payload,
    uses_weight_norm,
)


def _series(n: int = 40, start: float = 10.0, slope: float = 1.0) -> np.ndarray:
    return start + slope * np.arange(n, dtype=np.float64)


def test_source_weight_sum_is_one() -> None:
    weights = tcn_official_weights([0.394, 0.399, 0.500])
    assert weights.shape == (3,)
    assert abs(float(np.sum(weights)) - 1.0) < 1e-12
    reference = SourceSelector().compute_source_weights(np.asarray([0.394, 0.399, 0.500]), mode="inverse_distance")
    np.testing.assert_allclose(weights, reference)


def test_source_selection_identity_matches_formal_sita() -> None:
    diagnostic = solidified_diagnostic_sources("D4", PILOT_TARGET)
    assert diagnostic == SOLIDIFIED_D4_258_DIAGNOSTIC_SOURCES
    selection = select_formal_sita_sources("D4", PILOT_TARGET, horizon=3, seed=42)
    observed = tuple(item["source_id"] for item in selection["sources"])
    assert observed == diagnostic
    assert selection["selection_authority"] == "shared_protocol"
    assert selection["distance_metric"] == "euclidean"
    assert abs(selection["weight_sum"] - 1.0) < 1e-8


def test_source_shared_initialization_hash() -> None:
    canonical = canonical_source_init_state(seed=42)
    shared = hash_state_dict(canonical)
    rng = np.random.default_rng(0)
    sales_a = 20.0 + np.sin(np.linspace(0.0, 12.0, 180)) + 0.1 * rng.normal(size=180)
    sales_b = 50.0 + np.cos(np.linspace(0.0, 20.0, 180)) ** 2
    first = train_source_tcn(sales_a, horizon=1, init_state=canonical, seed=42)
    second = train_source_tcn(sales_b, horizon=1, init_state=canonical, seed=42)
    assert hash_state_dict(canonical) == shared
    assert hash_state_dict(first["state_dict"]) != hash_state_dict(second["state_dict"])


def test_transfer_scope_excludes_head() -> None:
    payload = transfer_scope_audit_payload()
    assert "head.weight" not in payload["TRANSFERABLE_TCN_BACKBONE_KEYS"]
    assert "head.bias" not in payload["TRANSFERABLE_TCN_BACKBONE_KEYS"]
    assert "head.weight" in payload["TARGET_ONLY_KEYS"]


def test_transfer_scope_excludes_revin() -> None:
    payload = transfer_scope_audit_payload()
    assert payload["TRANSFERABLE_TCN_BACKBONE_KEYS"] == list(EXPECTED_TRANSFERABLE_TCN_BACKBONE_KEYS)
    assert "revin.affine_weight" in payload["TARGET_ONLY_KEYS"]
    assert "revin.affine_bias" in payload["TARGET_ONLY_KEYS"]
    assert set(EXPECTED_TARGET_ONLY_KEYS) == set(payload["TARGET_ONLY_KEYS"])


def test_weight_norm_materialization_parity() -> None:
    seed_torch(0)
    model = build_formal_bl7_tcn_revin()
    assert uses_weight_norm(model.state_dict()) is False
    inputs = np.random.default_rng(0).normal(size=(4, 10, 1)).astype(np.float32)
    error = materialized_forward_parity_error(model.state_dict(), inputs)
    assert error <= MATERIALIZED_FORWARD_PARITY_MAX_ABS_ERROR
    g = np.array([2.0, 0.5], dtype=np.float64)
    v = np.array([[3.0, 4.0], [0.0, 1.0]], dtype=np.float64)
    fake = {"layer.weight_g": g, "layer.weight_v": v, "layer.bias": np.array([0.1, -0.2])}
    materialized = materialize_effective_tcn_state(fake)
    assert "layer.weight_g" not in materialized
    assert "layer.weight_v" not in materialized
    v_norm = np.sqrt((v**2).sum(axis=1, keepdims=True))
    expected = g.reshape(2, 1) * (v / v_norm)
    np.testing.assert_allclose(materialized["layer.weight"], expected)


def test_fused_weights_are_finite() -> None:
    seed_torch(1)
    left = build_formal_bl7_tcn_revin().state_dict()
    seed_torch(2)
    right = build_formal_bl7_tcn_revin().state_dict()
    fused = fuse_materialized_backbones([left, right], [0.6, 0.4])
    assert set(fused) == set(EXPECTED_TRANSFERABLE_TCN_BACKBONE_KEYS)
    assert all(np.isfinite(np.asarray(tensor)).all() for tensor in fused.values())
    with pytest.raises(SITATCNContractError):
        overlay_backbone(build_formal_bl7_tcn_revin(), {"head.weight": fused["blocks.0.conv1.weight"]}, keys=("head.weight",))


def test_forward_output_shape() -> None:
    seed_torch(3)
    model = build_formal_bl7_tcn_revin()
    import torch

    outputs = model(torch.zeros(5, 10, 1))
    assert tuple(outputs.shape) == (5, 1)


def test_forward_output_finite() -> None:
    seed_torch(4)
    model = build_formal_bl7_tcn_revin()
    import torch

    outputs = model(torch.ones(3, 10, 1))
    assert torch.isfinite(outputs).all()


def test_head_init_matches_bl7_for_same_seed() -> None:
    train = _series(15)
    validation = _series(15, start=15.0)
    random_model, random_diag = fit_tcn_revin_with_optional_backbone(train, validation, horizon=1, seed=42)
    seed_torch(7)
    donor = build_formal_bl7_tcn_revin()
    fused = {name: donor.state_dict()[name].detach().cpu().numpy() for name in EXPECTED_TRANSFERABLE_TCN_BACKBONE_KEYS}
    sita_model, sita_diag = fit_tcn_revin_with_optional_backbone(
        train, validation, horizon=1, seed=42, fused_backbone=fused
    )
    assert random_diag.target_head_init_hash == sita_diag.target_head_init_hash
    assert random_diag.target_revin_init_hash == sita_diag.target_revin_init_hash
    assert sita_diag.paired_head_initialization == "PASS"
    assert random_model.model is not None and sita_model.model is not None


def test_epoch0_loss_is_finite() -> None:
    fitted, diagnostics = fit_tcn_revin_with_optional_backbone(
        _series(20, start=8.0),
        _series(20, start=18.0),
        horizon=3,
        seed=43,
    )
    assert np.isfinite(diagnostics.train_loss_epoch0)
    assert np.isfinite(diagnostics.val_loss_epoch0)
    assert np.isfinite(diagnostics.activation_norm_epoch0)
    assert fitted.model is not None


def test_target_only_inference() -> None:
    fitted, _ = fit_tcn_revin_with_optional_backbone(_series(15), _series(15, start=20.0), horizon=1, seed=44)
    assert not hasattr(fitted, "source_models")
    assert not hasattr(fitted, "source_series")
    prediction = predict_target_only(fitted, np.arange(10.0, dtype=float))
    assert np.isfinite(prediction)
    signature = inspect.signature(predict_target_only)
    assert list(signature.parameters) == ["fitted_model", "input_sales"]


def test_original_sales_space_evaluation() -> None:
    from src.evaluation.metrics import compute_original_scale_metrics

    fitted = fit_bl7_revin(_series(15, start=100.0), _series(15, start=120.0), horizon=1, seed=42)
    prediction = predict_bl7_revin(fitted, np.linspace(130.0, 139.0, 10))
    metrics = compute_original_scale_metrics(np.asarray([140.0]), np.asarray([prediction]))
    assert metrics["primary_metric_space"] == "original_sales"
    assert 0.0 <= metrics["smape"] <= 200.0
    assert prediction > 50.0


def test_result_identity_isolated() -> None:
    assert METHOD_NAME == "SITA-TCN"
    assert_identity_isolation(["SITA-TCN", "TCN_RevIN"])
    with pytest.raises(SITATCNContractError):
        assert_identity_isolation(["TCN"])
    with pytest.raises(SITATCNContractError):
        assert_identity_isolation(["FUSED_TARGET_ONLY_V3"])
    assert TARGETS["D4"] == "166/258"
    assert "166/311" in FORBIDDEN_D4_TARGETS
    with pytest.raises(SITATCNContractError):
        assert_paper_target("D4", "166/311")


def test_no_existing_freeze_overwrite() -> None:
    allowed = assert_isolated_output_dir(DEFAULT_OUTPUT_ROOT / "pilot")
    assert "sita_tcn" in str(allowed)
    with pytest.raises(SITATCNContractError):
        assert_isolated_output_dir(ROOT / "outputs" / "baselines")
    with pytest.raises(SITATCNContractError):
        assert_isolated_output_dir(ROOT / "results")


def test_tie_rule_is_exact_numeric_no_tolerance() -> None:
    assert TIE_RULE == "exact_numeric_no_tolerance"
    assert win_tie_loss(1.0, 0.9) == "W"
    assert win_tie_loss(1.0, 1.0) == "T"
    assert win_tie_loss(1.0, 1.1) == "L"
    assert relative_improvement_pct(2.0, 1.5) == pytest.approx(25.0)


def test_source_split_matches_sita_ratios() -> None:
    train, validation, test = split_source_sales(_series(180))
    assert train.size == 144
    assert validation.size == 18
    assert test.size == 18
