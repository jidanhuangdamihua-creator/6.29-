"""Synthetic contract tests for CNN, TCN, and RevIN.

Uses only synthetic float32 tensors. Does not read sealed data, fit scalers,
rank sources, or run training.
"""

from __future__ import annotations

import unittest

import numpy as np
import tensorflow as tf

from sita_ablation.models import (
    CNN_FUSIBLE_LAYERS,
    TCN_DILATIONS,
    TCN_FUSIBLE_LAYERS,
    TCN_FUSIBLE_SHAPES,
    build_source_cnn,
    build_target_cnn,
    build_tcn,
    cnn_fusible_weights,
    expected_cnn_fusible_shapes,
    tcn_fusible_weights,
)
from sita_ablation.revin import build_revin


def _finite(array: np.ndarray) -> None:
    if not np.isfinite(array).all():
        raise AssertionError("expected finite tensor")


class CnnContractTests(unittest.TestCase):
    def test_source_cnn_shapes_and_fusible_weights(self):
        for channels in (5, 13, 11):
            with self.subTest(channels=channels):
                model = build_source_cnn((10, channels))
                x = np.random.default_rng(0).standard_normal((4, 10, channels)).astype(np.float32)
                y = model(x, training=False).numpy()
                self.assertEqual(y.shape, (4, 1))
                _finite(y)

                flat = model.get_layer("flatten").output
                self.assertEqual(tuple(flat.shape[1:]), (256,))

                weights = cnn_fusible_weights(model)
                self.assertEqual(tuple(weights.keys()), CNN_FUSIBLE_LAYERS)
                expected = expected_cnn_fusible_shapes(channels)
                for name in CNN_FUSIBLE_LAYERS:
                    kernel, bias = weights[name]
                    self.assertEqual(kernel.shape, expected[name][0], name)
                    self.assertEqual(bias.shape, expected[name][1], name)

                # Head and dropout are not fusible.
                self.assertEqual(model.get_layer("dense_out").kernel.shape, (256, 1))
                self.assertEqual(model.get_layer("dense_out").bias.shape, (1,))
                self.assertNotIn("dense_out", weights)
                self.assertEqual(model.loss, "mse")
                self.assertIsNone(model.optimizer.clipnorm)

    def test_target_cnn_extractor_and_head(self):
        channels = 5
        full, extractor = build_target_cnn((10, channels))
        x = np.random.default_rng(1).standard_normal((3, 10, channels)).astype(np.float32)
        z = extractor(x, training=False).numpy()
        y = full(x, training=False).numpy()
        self.assertEqual(z.shape, (3, 256))
        self.assertEqual(y.shape, (3, 1))
        _finite(z)
        _finite(y)

        # Same backbone tensors; prediction equals head_target(z_T) under training=False.
        z_tf = extractor(x, training=False)
        y_from_head = full.get_layer("head_target")(full.get_layer("feature_dropout")(z_tf, training=False))
        np.testing.assert_allclose(y, y_from_head.numpy(), rtol=0, atol=0)

        weights = cnn_fusible_weights(full)
        self.assertEqual(tuple(weights.keys()), CNN_FUSIBLE_LAYERS)
        self.assertEqual(full.get_layer("head_target").kernel.shape, (256, 1))
        self.assertNotIn("head_target", weights)
        # Target path is not compiled for Stage A source MSE.
        self.assertFalse(getattr(full, "compiled", False))
        self.assertFalse(hasattr(full, "optimizer") and full.optimizer is not None)


class TcnContractTests(unittest.TestCase):
    def test_tcn_shapes_dilations_and_fusible_weights(self):
        model = build_tcn((10, 1))
        x = np.random.default_rng(2).standard_normal((5, 10, 1)).astype(np.float32)
        y = model(x, training=False).numpy()
        self.assertEqual(y.shape, (5, 1))
        _finite(y)
        self.assertTrue(str(model.inputs[0].name).startswith("x_sales"))
        self.assertEqual(TCN_DILATIONS, (1, 2))

        weights = tcn_fusible_weights(model)
        self.assertEqual(tuple(weights.keys()), TCN_FUSIBLE_LAYERS)
        for name in TCN_FUSIBLE_LAYERS:
            kernel, bias = weights[name]
            self.assertEqual(kernel.shape, TCN_FUSIBLE_SHAPES[name][0], name)
            self.assertEqual(bias.shape, TCN_FUSIBLE_SHAPES[name][1], name)

        head = model.get_layer("head")
        self.assertEqual(head.kernel.shape, (16, 1))
        self.assertEqual(head.bias.shape, (1,))
        self.assertNotIn("head", weights)

        # Block 1 has matching channels, so no residual layer.
        layer_names = {layer.name for layer in model.layers}
        self.assertIn("blocks.0.residual", layer_names)
        self.assertNotIn("blocks.1.residual", layer_names)
        self.assertFalse(getattr(model, "compiled", False))

    def test_tcn_rejects_wrong_channel_count(self):
        with self.assertRaises(ValueError):
            build_tcn((10, 2))


class RevINContractTests(unittest.TestCase):
    def test_affine_init_and_shapes(self):
        for num_features in (1, 5, 11, 13):
            with self.subTest(num_features=num_features):
                revin = build_revin(num_features)
                self.assertEqual(float(revin.eps), 1e-5)
                self.assertTrue(revin.affine)
                self.assertEqual(revin.target_channel, 0)
                np.testing.assert_array_equal(
                    revin.affine_weight.numpy(),
                    np.ones((num_features,), dtype=np.float32),
                )
                np.testing.assert_array_equal(
                    revin.affine_bias.numpy(),
                    np.zeros((num_features,), dtype=np.float32),
                )

    def test_normalize_shapes_and_roundtrip(self):
        revin = build_revin(3)
        rng = np.random.default_rng(3)
        x = rng.normal(loc=2.0, scale=3.0, size=(4, 10, 3)).astype(np.float32)
        x_norm, mean, stdev = revin.normalize(tf.constant(x))
        self.assertEqual(tuple(x_norm.shape), (4, 10, 3))
        self.assertEqual(tuple(mean.shape), (4, 1, 3))
        self.assertEqual(tuple(stdev.shape), (4, 1, 3))
        _finite(x_norm.numpy())
        _finite(mean.numpy())
        _finite(stdev.numpy())

        # Stats are stop_gradient: equal to numpy reductions along time.
        np.testing.assert_allclose(
            mean.numpy(),
            np.mean(x, axis=1, keepdims=True),
            rtol=1e-6,
            atol=1e-6,
        )
        variance = np.mean((x - mean.numpy()) ** 2, axis=1, keepdims=True)
        np.testing.assert_allclose(
            stdev.numpy(),
            np.sqrt(variance + 1e-5),
            rtol=1e-6,
            atol=1e-6,
        )

        # With identity affine, channel-0 prediction denormalizes by the frozen formula.
        y_scaled = rng.normal(size=(4, 1)).astype(np.float32)
        y_hat = revin.denormalize(tf.constant(y_scaled), mean, stdev).numpy()
        expected = (
            (y_scaled - 0.0) / (1.0 + 1e-10) * stdev.numpy()[:, 0:1, 0]
            + mean.numpy()[:, 0:1, 0]
        )
        np.testing.assert_allclose(y_hat, expected, rtol=1e-6, atol=1e-6)
        _finite(y_hat)

    def test_denormalize_uses_affine_and_eps_squared(self):
        revin = build_revin(2)
        revin.affine_weight.assign(np.array([2.0, 3.0], dtype=np.float32))
        revin.affine_bias.assign(np.array([0.5, -1.0], dtype=np.float32))
        mean = tf.constant([[[1.0, 9.0]]], dtype=tf.float32)
        stdev = tf.constant([[[4.0, 5.0]]], dtype=tf.float32)
        y = tf.constant([[1.5]], dtype=tf.float32)
        got = revin.denormalize(y, mean, stdev).numpy()
        expected = ((1.5 - 0.5) / (2.0 + 1e-10)) * 4.0 + 1.0
        np.testing.assert_allclose(got, [[expected]], rtol=0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
