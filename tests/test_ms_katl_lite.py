"""Unit tests for MS-KATL-Lite layers, gradient semantics and training loop."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tf_compat  # noqa: F401  (must be imported before tensorflow)
import tensorflow as tf

from src.transfer_methods.ms_katl_lite import (
    DomainDiscriminator,
    GradientReversalLayer,
    KNNPriorCrossAttention,
    MSKATLLite,
    ResidualGate,
    SourceAdapter,
    compute_knn_prior,
    lambda_schedule,
    train_ms_katl_head,
    predict_ms_katl,
)


# -------------------------------------------------------------------
# KNN prior
# -------------------------------------------------------------------

def test_knn_prior_sums_to_one_and_orders_by_distance():
    prior = compute_knn_prior([0.5, 1.0, 2.0], tau_k=1.0)
    assert prior.shape == (3,)
    assert np.isclose(prior.sum(), 1.0)
    # 距离越小先验越大
    assert prior[0] > prior[1] > prior[2]


def test_knn_prior_fails_closed_on_empty_and_invalid():
    with pytest.raises(ValueError):
        compute_knn_prior([])
    with pytest.raises(ValueError):
        compute_knn_prior([1.0, float("nan")])
    with pytest.raises(ValueError):
        compute_knn_prior([1.0], tau_k=0.0)


def test_lambda_schedule_bounds_and_monotonicity():
    assert lambda_schedule(0.0) == pytest.approx(0.0)
    assert lambda_schedule(1.0) == pytest.approx(2.0 / (1.0 + np.exp(-10.0)) - 1.0)
    values = [lambda_schedule(p) for p in np.linspace(0, 1, 11)]
    assert all(b >= a for a, b in zip(values, values[1:]))
    assert 0.0 <= min(values) and max(values) < 1.0


# -------------------------------------------------------------------
# GRL
# -------------------------------------------------------------------

def test_grl_identity_forward_and_reversed_gradient():
    grl = GradientReversalLayer()
    x = tf.constant([[1.0, -2.0, 3.0]])
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = grl(x)
        loss = tf.reduce_sum(y)
    np.testing.assert_allclose(y.numpy(), x.numpy())
    grad = tape.gradient(loss, x)
    # 反向仅取反（-I），不含 lambda
    np.testing.assert_allclose(grad.numpy(), -np.ones_like(x.numpy()))


# -------------------------------------------------------------------
# Attention / Gate / Discriminator shapes and semantics
# -------------------------------------------------------------------

def test_cross_attention_shapes_softmax_and_mask():
    batch, k, d_model, d_adapter = 4, 3, 256, 32
    attn = KNNPriorCrossAttention(d_k=32, gamma=0.5)
    z_t = tf.random.normal((batch, d_model))
    tokens = tf.random.normal((batch, k, d_adapter))
    prior = tf.constant(np.repeat([[0.5, 0.3, 0.2]], batch, axis=0), dtype=tf.float32)

    z_f, alpha = attn(z_t, tokens, prior)
    assert z_f.shape == (batch, 32)
    assert alpha.shape == (batch, k)
    np.testing.assert_allclose(alpha.numpy().sum(axis=1), np.ones(batch), rtol=1e-5)

    mask = tf.constant(np.repeat([[True, True, False]], batch, axis=0))
    _, alpha_masked = attn(z_t, tokens, prior, source_mask=mask)
    np.testing.assert_allclose(alpha_masked.numpy()[:, 2], np.zeros(batch), atol=1e-6)
    np.testing.assert_allclose(alpha_masked.numpy().sum(axis=1), np.ones(batch), rtol=1e-5)


def test_strong_prior_dominates_when_gamma_positive():
    """gamma>0 时极端 KNN 先验应显著拉动 alpha。"""
    batch, k = 8, 3
    attn = KNNPriorCrossAttention(d_k=32, gamma=1.0)
    z_t = tf.zeros((batch, 256))
    tokens = tf.zeros((batch, k, 32))  # 打平 attention logits，仅剩 prior bias
    prior = tf.constant(np.repeat([[0.98, 0.01, 0.01]], batch, axis=0), dtype=tf.float32)
    _, alpha = attn(z_t, tokens, prior)
    assert float(alpha.numpy()[:, 0].mean()) > 0.8


def test_residual_gate_scalar_range_and_target_bias():
    gate = ResidualGate()
    z_t = tf.random.normal((5, 256))
    z_f = tf.random.normal((5, 8))
    g = gate(z_t, z_f)
    assert g.shape == (5, 1)
    g_np = g.numpy()
    assert np.all(g_np > 0.0) and np.all(g_np < 1.0)
    assert float(gate.gate_proj.bias.numpy()) == pytest.approx(-2.0)


def test_explicit_residual_prediction_recovers_target_head_at_zero_gate():
    z_t, f_t, prior = _tiny_inputs()
    model = MSKATLLite(d_model=256, num_sources=3)
    prior_b = tf.constant(np.repeat(prior.reshape(1, -1), 4, axis=0), dtype=tf.float32)
    y, _, g = model(
        tf.constant(z_t), [tf.constant(f) for f in f_t], prior_b, training=False
    )
    z_t_tf = tf.constant(z_t)
    adapted = [model.adapters[i](tf.constant(f_t[i])) for i in range(3)]
    tokens = tf.stack(adapted, axis=1)
    z_f, _ = model.cross_attention(z_t_tf, tokens, prior_b)
    y_expected = model.head_target(z_t_tf) + g * model.head_transfer(z_f)
    np.testing.assert_allclose(y.numpy(), y_expected.numpy(), rtol=1e-5, atol=1e-5)


def test_domain_discriminator_output_in_unit_interval():
    disc = DomainDiscriminator()
    out = disc(tf.random.normal((7, 32)))
    assert out.shape == (7, 1)
    assert np.all(out.numpy() >= 0.0) and np.all(out.numpy() <= 1.0)


# -------------------------------------------------------------------
# Full head model
# -------------------------------------------------------------------

def _tiny_inputs(batch=4, k=3, d_model=256, seed=0):
    rng = np.random.default_rng(seed)
    z_t = rng.normal(size=(batch, d_model)).astype(np.float32)
    f_t = [rng.normal(size=(batch, d_model)).astype(np.float32) for _ in range(k)]
    prior = compute_knn_prior([0.5, 1.0, 1.5][:k])
    return z_t, f_t, prior


def test_ms_katl_forward_shapes():
    z_t, f_t, prior = _tiny_inputs()
    model = MSKATLLite(d_model=256, num_sources=3)
    prior_b = tf.constant(np.repeat(prior.reshape(1, -1), 4, axis=0), dtype=tf.float32)
    y, alpha, g = model(
        tf.constant(z_t), [tf.constant(f) for f in f_t], prior_b, training=False
    )
    assert y.shape == (4, 1)
    assert alpha.shape == (4, 3)
    assert g.shape == (4, 1)


def test_k_zero_fails_closed_and_k_above_three_rejected():
    with pytest.raises(ValueError):
        MSKATLLite(num_sources=0)
    with pytest.raises(ValueError):
        MSKATLLite(num_sources=4)


def test_adapter_receives_adversarial_gradient_and_alpha_path_is_stopped():
    """MUST 约束：GRL 梯度必须传回 adapter；stop_gradient 仅作用于 alpha_bar。"""
    tf.random.set_seed(0)
    batch, k = 4, 2
    rng = np.random.default_rng(0)
    z_t = tf.constant(rng.normal(size=(batch, 256)).astype(np.float32))
    f_t = [tf.constant(rng.normal(size=(batch, 256)).astype(np.float32)) for _ in range(k)]
    f_s = [tf.constant(rng.normal(size=(batch, 256)).astype(np.float32)) for _ in range(k)]
    prior = tf.constant(
        np.repeat(compute_knn_prior([0.5, 1.0]).reshape(1, -1), batch, axis=0),
        dtype=tf.float32,
    )
    model = MSKATLLite(d_model=256, num_sources=k)
    bce = tf.keras.losses.BinaryCrossentropy()

    with tf.GradientTape() as tape:
        _, alpha, _ = model(z_t, f_t, prior, training=True)
        alpha_bar = tf.reduce_mean(alpha, axis=0)
        loss_adv = tf.constant(0.0)
        for i in range(k):
            u_t = model.adapters[i](f_t[i])
            u_s = model.adapters[i](f_s[i])
            pred_t = model.discriminators[i](model.grl(u_t))
            pred_s = model.discriminators[i](model.grl(u_s))
            loss_domain = bce(tf.zeros_like(pred_s), pred_s) + bce(
                tf.ones_like(pred_t), pred_t
            )
            loss_adv += tf.stop_gradient(alpha_bar[i]) * loss_domain

    adapter_vars = [v for a in model.adapters for v in a.trainable_variables]
    attention_vars = model.cross_attention.trainable_variables
    grads = tape.gradient(loss_adv, adapter_vars + attention_vars)
    adapter_grads = grads[: len(adapter_vars)]
    attention_grads = grads[len(adapter_vars):]

    # adapter 必须收到对抗梯度（GRL 传回）
    assert all(g is not None for g in adapter_grads)
    assert any(float(tf.reduce_max(tf.abs(g))) > 0 for g in adapter_grads)
    # alpha_bar 路径被 stop_gradient 截断：attention 参数不应从 L_adv 收到梯度
    assert all(g is None for g in attention_grads)


def test_train_ms_katl_head_runs_and_predicts():
    """极小规模端到端：训练循环可运行、损失有限、预测形状正确。"""
    tf.random.set_seed(0)
    k, n, m, d_model = 2, 5, 20, 256
    rng = np.random.default_rng(0)
    z_t_train = rng.normal(size=(n, d_model)).astype(np.float32)
    f_t_train = [rng.normal(size=(n, d_model)).astype(np.float32) for _ in range(k)]
    y_train = rng.normal(size=(n,)).astype(np.float32)
    f_s_pools = [rng.normal(size=(m, d_model)).astype(np.float32) for _ in range(k)]
    prior = compute_knn_prior([0.5, 1.0])

    model = MSKATLLite(d_model=d_model, num_sources=k)
    summary = train_ms_katl_head(
        model,
        x_train=z_t_train,
        f_t_train=f_t_train,
        y_train=y_train,
        f_s_pools=f_s_pools,
        knn_prior=prior,
        epochs=3,
        batch_size=16,
        learning_rate=1e-3,
        seed=42,
        use_adversarial=False,
    )
    history = summary["history"]
    assert len(history["total_loss"]) == 3
    assert all(np.isfinite(v) for v in history["total_loss"])
    assert summary["steps_per_epoch"] == 1
    assert all(v == 0.0 for v in history["lambda"])

    y_pred = predict_ms_katl(model, z_t_train, f_t_train, prior)
    assert y_pred.shape == (n, 1)
    assert np.all(np.isfinite(y_pred))


HEAD_TRAIN_MS_KATL_HEAD_RETURN_KEYS = {
    "history",
    "total_steps",
    "steps_per_epoch",
    "training_recipe",
    "gate_diagnostics",
}
DIAGNOSTIC_ONLY_TRAIN_MS_KATL_HEAD_RETURN_KEYS = {
    "before_first_target_optimizer_update_capture",
}


def test_train_ms_katl_head_disabled_omits_diagnostic_only_keys():
    tf.random.set_seed(2)
    k, n, m, d_model = 2, 4, 8, 256
    rng = np.random.default_rng(2)
    z_t_train = rng.normal(size=(n, d_model)).astype(np.float32)
    f_t_train = [rng.normal(size=(n, d_model)).astype(np.float32) for _ in range(k)]
    y_train = rng.normal(size=(n,)).astype(np.float32)
    f_s_pools = [rng.normal(size=(m, d_model)).astype(np.float32) for _ in range(k)]
    prior = compute_knn_prior([0.4, 0.9])
    model = MSKATLLite(d_model=d_model, num_sources=k)
    summary = train_ms_katl_head(
        model,
        x_train=z_t_train,
        f_t_train=f_t_train,
        y_train=y_train,
        f_s_pools=f_s_pools,
        knn_prior=prior,
        epochs=1,
        batch_size=16,
        learning_rate=1e-3,
        seed=3,
        use_adversarial=False,
        on_before_first_target_optimizer_update=None,
        on_diagnostic_epoch_eval=None,
    )
    assert set(summary) == HEAD_TRAIN_MS_KATL_HEAD_RETURN_KEYS
    for key in DIAGNOSTIC_ONLY_TRAIN_MS_KATL_HEAD_RETURN_KEYS:
        assert key not in summary


def test_train_ms_katl_head_captures_before_first_optimizer_update():
    tf.random.set_seed(1)
    k, n, m, d_model = 2, 4, 8, 256
    rng = np.random.default_rng(1)
    z_t_train = rng.normal(size=(n, d_model)).astype(np.float32)
    f_t_train = [rng.normal(size=(n, d_model)).astype(np.float32) for _ in range(k)]
    y_train = rng.normal(size=(n,)).astype(np.float32)
    f_s_pools = [rng.normal(size=(m, d_model)).astype(np.float32) for _ in range(k)]
    prior = compute_knn_prior([0.4, 0.9])
    model = MSKATLLite(d_model=d_model, num_sources=k)
    captures = []

    def _capture(payload):
        captures.append(dict(payload))

    summary = train_ms_katl_head(
        model,
        x_train=z_t_train,
        f_t_train=f_t_train,
        y_train=y_train,
        f_s_pools=f_s_pools,
        knn_prior=prior,
        epochs=1,
        batch_size=16,
        learning_rate=1e-3,
        seed=7,
        use_adversarial=False,
        on_before_first_target_optimizer_update=_capture,
    )
    assert len(captures) == 1
    assert captures[0]["optimizer_update_count"] == 0
    assert captures[0]["captured_before_first_target_optimizer_update"] is True
    capture = summary["before_first_target_optimizer_update_capture"]
    assert capture["requested"] is True
    assert capture["invoked"] is True
    assert capture["optimizer_update_count"] == 0
    assert capture["captured_before_first_target_optimizer_update"] is True


def test_source_adapter_output_dim():
    adapter = SourceAdapter(adapter_dim=8)
    out = adapter(tf.random.normal((6, 256)))
    assert out.shape == (6, 8)
