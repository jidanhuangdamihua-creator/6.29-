"""Tests for FUSED_TARGET_ONLY_V3 (RevIN wrap + frozen V2 recipe)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tf_compat  # noqa: F401
import tensorflow as tf

from src.transfer_methods.ms_katl_fto_v2_screening import (
    V2_TRAINING_RECIPE,
    v2_architecture,
    v2_architecture_digest,
    v2_training_recipe,
)
from src.transfer_methods.ms_katl_fto_v3_screening import (
    V3_METHOD_DISPLAY,
    V3_REVIN_CONTRACT,
    V3_TRAINING_RECIPE,
    assert_v3_lr_arms_share_stage_a,
    attach_v1_deltas,
    attach_v2_deltas,
    format_v3_vs_frozen_summary,
    v3_architecture,
    v3_architecture_digest,
    v3_training_recipe,
)
from src.transfer_methods.ms_katl_lite import (
    MSKATLLite,
    compute_knn_prior,
    extract_features_maybe_revin,
    model_weight_digest,
    predict_ms_katl,
    train_ms_katl_head,
)
from src.transfer_methods.ms_katl_primary_target_lr_screening import (
    FUSED_TARGET_ONLY_CONTRACT,
    architecture_digest as v1_architecture_digest,
    ScreeningContractError,
)
from src.transfer_methods.revin import RevIN

WINDOW = 8
N_FEATURES = 3
N = 12
K = 2
D_MODEL = 6


def _windows(seed: int = 7, shift: float = 4.0):
    rng = np.random.default_rng(seed)
    x = rng.normal(loc=shift, scale=1.5, size=(N, WINDOW, N_FEATURES)).astype(np.float32)
    x[0] = 0.0  # zero-variance window: RevIN must stay finite
    y = (x[:, -1, 0] + rng.normal(scale=0.1, size=N)).astype(np.float32)
    f_t = [rng.normal(size=(N, D_MODEL)).astype(np.float32) for _ in range(K)]
    f_s_pools = [rng.normal(size=(16, D_MODEL)).astype(np.float32) for _ in range(K)]
    prior = compute_knn_prior([0.5, 1.0])
    return x, f_t, y, f_s_pools, prior


def _extractor(seed: int = 3):
    tf.keras.utils.set_random_seed(seed)
    inputs = tf.keras.Input(shape=(WINDOW, N_FEATURES))
    flat = tf.keras.layers.Flatten()(inputs)
    out = tf.keras.layers.Dense(D_MODEL)(flat)
    return tf.keras.Model(inputs, out, name="tiny_extractor")


def _head(seed: int = 3):
    tf.keras.utils.set_random_seed(seed)
    return MSKATLLite(
        d_model=D_MODEL,
        num_sources=K,
        transfer_branch_enabled=False,
        gate_mode="zero",
    )


# -------------------------------------------------------------------
# RevIN layer
# -------------------------------------------------------------------

def test_revin_roundtrip_recovers_sequence_with_identity_affine():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, WINDOW, N_FEATURES)).astype(np.float32)
    layer = RevIN(num_features=N_FEATURES, affine=True)
    x_norm, mean, stdev = layer.normalize(tf.constant(x))
    recovered = layer.denormalize_sequence(x_norm, mean, stdev).numpy()
    np.testing.assert_allclose(recovered, x, rtol=1e-5, atol=1e-5)


def test_revin_zero_variance_window_is_finite():
    x = np.zeros((2, WINDOW, N_FEATURES), dtype=np.float32)
    x[1] = 3.0
    layer = RevIN(num_features=N_FEATURES, affine=False)
    x_norm, mean, stdev = layer.normalize(tf.constant(x))
    y = layer.denormalize(tf.zeros((2, 1)), mean, stdev).numpy()
    assert np.isfinite(x_norm.numpy()).all()
    assert np.isfinite(y).all()
    np.testing.assert_allclose(y[0], 0.0, atol=1e-5)
    np.testing.assert_allclose(y[1], 3.0, atol=1e-5)


def test_revin_rejects_bad_rank_and_channel():
    layer = RevIN(num_features=N_FEATURES)
    with pytest.raises(ValueError):
        layer.normalize(tf.ones((4, N_FEATURES)))
    with pytest.raises(ValueError):
        RevIN(num_features=N_FEATURES, target_channel=9)
    with pytest.raises(ValueError):
        RevIN(num_features=N_FEATURES, eps=0.0)


def test_revin_denormalize_uses_target_channel_stats():
    x = np.zeros((1, WINDOW, N_FEATURES), dtype=np.float32)
    x[..., 0] = 10.0
    x[..., 1] = 2.0
    layer = RevIN(num_features=N_FEATURES, affine=False, target_channel=0)
    _, mean, stdev = layer.normalize(tf.constant(x))
    y = layer.denormalize(tf.zeros((1, 1)), mean, stdev).numpy()
    np.testing.assert_allclose(y, [[10.0]], atol=1e-5)
    layer1 = RevIN(num_features=N_FEATURES, affine=False, target_channel=1)
    _, mean1, stdev1 = layer1.normalize(tf.constant(x))
    y1 = layer1.denormalize(tf.zeros((1, 1)), mean1, stdev1).numpy()
    np.testing.assert_allclose(y1, [[2.0]], atol=1e-5)


# -------------------------------------------------------------------
# Training wiring
# -------------------------------------------------------------------

def test_default_training_leaves_revin_disabled():
    x, f_t, y, f_s_pools, prior = _windows()
    extractor = _extractor()
    summary = train_ms_katl_head(
        _head(),
        x_train=x,
        f_t_train=f_t,
        y_train=y,
        f_s_pools=f_s_pools,
        knn_prior=prior,
        target_extractor=extractor,
        epochs=2,
        batch_size=4,
        learning_rate=1e-2,
        seed=3,
        use_adversarial=False,
    )
    recipe = summary["training_recipe"]
    assert recipe["revin_enabled"] is False
    assert recipe["loss_name"] == "mse"
    assert recipe["use_ema"] is False


def test_revin_changes_weights_and_keeps_predictions_finite():
    x, f_t, y, f_s_pools, prior = _windows()
    extractor_a = _extractor(seed=3)
    extractor_b = _extractor(seed=3)
    model_a = _head(seed=3)
    model_b = _head(seed=3)
    kwargs = dict(
        x_train=x,
        f_t_train=f_t,
        y_train=y,
        f_s_pools=f_s_pools,
        knn_prior=prior,
        epochs=3,
        batch_size=4,
        learning_rate=1e-2,
        seed=3,
        use_adversarial=False,
        **V3_TRAINING_RECIPE,
    )
    train_ms_katl_head(model_a, target_extractor=extractor_a, **kwargs)
    revin = RevIN(num_features=N_FEATURES)
    summary = train_ms_katl_head(
        model_b, target_extractor=extractor_b, revin=revin, **kwargs
    )
    assert summary["training_recipe"]["revin_enabled"] is True
    assert summary["training_recipe"]["revin_affine"] is True
    assert model_weight_digest(model_a) != model_weight_digest(model_b)
    z, mean, stdev = extract_features_maybe_revin(extractor_b, x, revin=revin)
    pred_norm = predict_ms_katl(model_b, z, f_t, prior)
    pred = revin.denormalize_numpy(pred_norm, mean.numpy(), stdev.numpy())
    assert np.isfinite(pred).all()


def test_extract_without_revin_matches_plain_extractor():
    x, *_rest = _windows()
    extractor = _extractor()
    z_plain = extractor(tf.constant(x), training=False).numpy()
    z, mean, stdev = extract_features_maybe_revin(extractor, x, revin=None)
    np.testing.assert_allclose(z, z_plain)
    assert mean is None and stdev is None


# -------------------------------------------------------------------
# V3 contract
# -------------------------------------------------------------------

def test_v3_core_matches_v1_v2_but_digest_and_revin_differ():
    arch = v3_architecture()
    core = {key: arch[key] for key in FUSED_TARGET_ONLY_CONTRACT}
    assert core == FUSED_TARGET_ONLY_CONTRACT
    assert core == v2_architecture()
    assert arch["revin_enabled"] is True
    assert arch["revin_affine"] is True
    assert v3_training_recipe() == V2_TRAINING_RECIPE
    assert v3_training_recipe() == v2_training_recipe()
    digest = v3_architecture_digest()
    assert digest != v1_architecture_digest()
    assert digest != v2_architecture_digest()


def test_v3_recipe_cannot_drift_from_v2():
    assert V3_TRAINING_RECIPE == V2_TRAINING_RECIPE
    assert V3_REVIN_CONTRACT["revin_enabled"] is True


def test_assert_v3_lr_arms_requires_v3_method():
    base = {
        "target_identity": "166/258",
        "source_identities": [("166", "1")],
        "source_distances": [0.1],
        "source_model_digests": ["a"],
        "source_scaler_digests": ["b"],
        "target_split_digest": {"X_train": "x"},
        "target_scaler_digest": "s",
        "fused_init_digest": "f",
        "architecture_digest": "d",
    }
    rows = [
        {**base, "learning_rate": lr, "method": V3_METHOD_DISPLAY}
        for lr in (3e-5, 1e-4, 3e-4)
    ]
    flags = assert_v3_lr_arms_share_stage_a(rows)
    assert flags["LR_ARMS_FUSED_INIT_MATCH"] == "YES"
    bad = [{**row, "method": "FUSED_TARGET_ONLY_V2"} for row in rows]
    with pytest.raises(ScreeningContractError):
        assert_v3_lr_arms_share_stage_a(bad)


def test_attach_v1_and_v2_deltas():
    v1_index = {("D4", "166/258", 1e-4): {"rmse_original": 0.50, "smape_original": 40.0}}
    v2_index = {("D4", "166/258", 1e-4): {"rmse_original": 0.42, "smape_original": 30.0}}
    row = {
        "dataset": "D4",
        "target_identity": "166/258",
        "learning_rate": 1e-4,
        "rmse_original": 0.40,
        "smape_original": 25.0,
    }
    out = attach_v2_deltas(attach_v1_deltas(row, v1_index), v2_index)
    assert out["v1_rmse_original"] == pytest.approx(0.50)
    assert out["v2_rmse_original"] == pytest.approx(0.42)
    assert out["delta_rmse_vs_v1"] == pytest.approx(-0.10)
    assert out["delta_rmse_vs_v2"] == pytest.approx(-0.02)
    assert out["v3_beats_v1_rmse"] is True
    assert out["v3_beats_v2_rmse"] is True
    missing = attach_v2_deltas(
        {
            "dataset": "D2",
            "target_identity": "1/10",
            "learning_rate": 3e-4,
            "rmse_original": 4.0,
            "smape_original": 100.0,
        },
        v2_index,
    )
    assert missing["v2_status"] == "MISSING"


def test_format_v3_summary_counts_wins():
    rows = [
        {
            "dataset": "D4",
            "target_identity": "166/258",
            "learning_rate_label": "3e-4",
            "rmse_original": 0.4,
            "v2_rmse_original": 0.5,
            "delta_rmse_vs_v2": -0.1,
            "smape_original": 20.0,
            "v2_smape_original": 30.0,
            "delta_smape_vs_v2": -10.0,
            "delta_rmse_vs_v1": -0.2,
            "delta_smape_vs_v1": -15.0,
        }
    ]
    text = format_v3_vs_frozen_summary(rows)
    assert "V3_RMSE_WINS_VS_V2  = 1/1" in text
    assert "V3_SMAPE_WINS_VS_V2 = 1/1" in text
    assert "V3_RMSE_WINS_VS_V1  = 1/1" in text
