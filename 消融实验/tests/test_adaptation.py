"""Synthetic target-adaptation tests.

The trainer and optimizer are mocked. These tests do not read sealed data,
call model.fit, fit a source model, or write a result file.
"""

from __future__ import annotations

import inspect
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from sita_ablation.adaptation import (
    DEFINED,
    UNDEFINED_ZERO_INIT_NORM,
    TargetTrainOutput,
    TargetWindows,
    WeightSnapshot,
    _recorded_original_errors,
    _snapshot_dict,
    adapt_target,
    parameter_shift,
    run_target_training,
)
from sita_ablation.arms import BLOCKED_GAP_TCN_SOURCE_FIT, expand_runs, frozen_tensors
from sita_ablation.data import ConditionFrames, ContractError, DateSpan, load_dataset_contract
from sita_ablation.models import TCN_FUSIBLE_SHAPES
from sita_ablation.scaling import fit_target_model_scaler, inverse_sales
from sita_ablation.stage_a import (
    NOT_APPLICABLE,
    SourceRecord,
    StageABundle,
    assess_tcn_source_fit,
    fuse_init,
    make_stage_a_key,
)
from sita_ablation.windows import (
    build_target_partition_windows,
    build_test_windows,
    expected_target_partition_count,
)

_PROTOCOL = Path(__file__).resolve().parents[1] / "configs" / "protocol_frozen.json"
_REAL_LOOP_SOURCE = inspect.getsource(run_target_training)


class MockTrainer:
    def __init__(self):
        self.requests = []
        self.validations = []
        self.revins = []
        self.conv = []
        self.heads = []
        self.layer_trainable = []
        self.models = []

    def __call__(self, request, model, revin, validation):
        self.requests.append(request)
        self.validations.append(validation)
        self.revins.append(revin)
        self.models.append(model)
        self.layer_trainable.append({layer.name: bool(layer.trainable) for layer in model.layers if layer.weights})
        conv_name = "conv1" if any(layer.name == "conv1" for layer in model.layers) else "blocks.0.conv1"
        head_name = "head_target" if any(layer.name == "head_target" for layer in model.layers) else "head"
        self.conv.append([np.array(array, copy=True) for array in model.get_layer(conv_name).get_weights()])
        self.heads.append([np.array(array, copy=True) for array in model.get_layer(head_name).get_weights()])
        epochs = int(request.protocol.epochs)
        snap = _snapshot_dict(model, revin)
        return TargetTrainOutput(
            train_loss=tuple(0.1 for _ in range(epochs)),
            val_loss=tuple(0.2 for _ in range(epochs)),
            val_rmse=tuple(0.3 for _ in range(epochs)),
            val_smape=tuple(0.4 for _ in range(epochs)),
            epoch0_val_rmse=1.25,
            gradient_steps=int(request.protocol.decay_steps),
            raw_weights=snap,
            ema_weights=_snapshot_dict(model, revin),
            used_validation_for_update=False,
            used_validation_for_selection=False,
        )


def _optimizer_variables(model, revin):
    """Trainable weights the optimizer would receive, identified as layer variables."""
    variables = [var for layer in model.layers for var in layer.trainable_weights]
    if revin is not None:
        variables.extend(revin.trainable_weights)
    return variables


def _contains_variable(variables, var) -> bool:
    return any(var is item for item in variables)


def _assert_optimizer_membership(model, revin, freeze_conv: bool) -> None:
    """P1 keeps conv1–conv3 out of the optimizer. The head and RevIN stay in."""
    optimizer = _optimizer_variables(model, revin)
    for name in ("conv1", "conv2", "conv3"):
        layer = model.get_layer(name)
        for var in (layer.kernel, layer.bias):
            present = _contains_variable(optimizer, var)
            if freeze_conv:
                if present:
                    raise AssertionError(f"{name} variable entered the optimizer")
            elif not present:
                raise AssertionError(f"{name} variable left the optimizer")
    head = model.get_layer("head_target")
    for var in (head.kernel, head.bias):
        if not _contains_variable(optimizer, var):
            raise AssertionError("head_target variable left the optimizer")
    if revin is None:
        return
    for var in (revin.affine_weight, revin.affine_bias):
        if not _contains_variable(optimizer, var):
            raise AssertionError("RevIN variable left the optimizer")


class AdaptationTests(unittest.TestCase):
    def setUp(self):
        self.fit = patch("tensorflow.keras.Model.fit", side_effect=AssertionError("model.fit"))
        self.source_fit = patch("sita_ablation.stage_a.fit_source_cnn", side_effect=AssertionError("source fit"))
        self.stage_a = patch("sita_ablation.stage_a.load_or_train_stage_a", side_effect=AssertionError("stage_a"))
        self.rank = patch("sita_ablation.source_selection.rank_candidates", side_effect=AssertionError("rank"))
        self.real = patch("sita_ablation.adaptation.run_target_training", side_effect=AssertionError("real training"))
        self.fit_mock = self.fit.start()
        self.source_fit_mock = self.source_fit.start()
        self.stage_a_mock = self.stage_a.start()
        self.rank_mock = self.rank.start()
        self.real_mock = self.real.start()

    def tearDown(self):
        for item in (self.real, self.rank, self.stage_a, self.source_fit, self.fit):
            item.stop()

    def test_real_loop_is_not_model_fit_and_tests_do_not_call_it(self):
        self.assertNotIn(".fit(", _REAL_LOOP_SOURCE)
        self.assertNotIn("validation_data", _REAL_LOOP_SOURCE)
        self.assertIn("CosineDecay", _REAL_LOOP_SOURCE)
        self.assertIn("Huber", _REAL_LOOP_SOURCE)
        self.assertEqual(self.real_mock.call_count, 0)
        self.assertEqual(self.fit_mock.call_count, 0)
        self.assertEqual(self.source_fit_mock.call_count, 0)

    def test_arm_initialization_trainable_state_and_revin(self):
        windows = _windows("D1", train_value=1.0, val_value=2.0, test_value=0.0)
        scaler = _scaler("D1")
        template = _cnn_template(5)
        held = np.array(template.tensors["conv1"][0], copy=True)
        bundle = _bundle(_key())
        inverse = fuse_init(bundle, "inverse_distance")["conv1"][0]
        uniform = fuse_init(bundle, "uniform")["conv1"][0]
        single = fuse_init(bundle, "single")["conv1"][0]
        self.assertFalse(np.allclose(inverse, uniform))
        expected = {
            "F1": template.tensors["conv1"][0],
            "F2": template.tensors["conv1"][0],
            "F3": inverse,
            "F4": inverse,
            "P1": inverse,
            "P2": single,
            "P3": uniform,
        }
        results = {}
        for arm in ("F1", "F2", "F3", "F4", "P1", "P2", "P3"):
            spec = _spec(arm)
            trainer = MockTrainer()
            result = adapt_target(spec, windows, scaler, template, None if spec.selection_rule == "none" else bundle, trainer=trainer)
            results[arm] = result
            with self.subTest(arm=arm):
                self.assertEqual(trainer.requests[0].protocol.learning_rate, spec.target_learning_rate)
                np.testing.assert_allclose(trainer.conv[0][0], expected[arm])
                np.testing.assert_array_equal(trainer.heads[0][0], template.tensors["head_target"][0])
                self.assertEqual(trainer.revins[0] is not None, spec.revin)
                self.assertEqual(result.revin_enabled, spec.revin)
                self.assertEqual(set(result.frozen_names), set(frozen_tensors(spec)))
                self.assertTrue(set(result.frozen_names).isdisjoint(result.trainable_names))
                _assert_optimizer_membership(trainer.models[0], trainer.revins[0], spec.freeze_conv)
                self.assertEqual(trainer.layer_trainable[0]["conv1"], not spec.freeze_conv)
                self.assertTrue(trainer.layer_trainable[0]["head_target"])
                if spec.revin:
                    self.assertEqual(trainer.revins[0].num_features, 5)
                    self.assertIsNone(result.ema_shift["revin"].relative)
                    self.assertEqual(result.ema_shift["revin"].status, UNDEFINED_ZERO_INIT_NORM)
                else:
                    self.assertNotIn("revin", result.raw_shift)
                self.assertEqual(result.raw_shift["conv1"].status, DEFINED)
                self.assertEqual(result.raw_shift["conv1"].relative, 0.0)
                self.assertEqual(result.raw_shift["head"].relative, 0.0)
                self.assertIsNone(result.signed_error)
                if spec.selection_rule == "none":
                    self.assertIsNone(result.alpha_entropy)
                else:
                    self.assertIsInstance(result.alpha_entropy, float)
                    self.assertNotEqual(result.alpha_entropy, result.effective_source_count)
                self.assertFalse(result.checkpoint_selection)
        np.testing.assert_array_equal(template.tensors["conv1"][0], held)
        bare = next(
            item
            for item in expand_runs(1, include_no_revin_variants=True)
            if item.arm == "P1" and item.revin is False and item.dataset == "D1" and item.horizon == 1 and item.seed == 42
        )
        bare_trainer = MockTrainer()
        bare_result = adapt_target(bare, windows, scaler, template, bundle, trainer=bare_trainer)
        self.assertIsNone(bare_trainer.revins[0])
        self.assertFalse(bare_trainer.layer_trainable[0]["conv1"])
        self.assertEqual(set(bare_result.frozen_names), set(frozen_tensors(bare)))
        _assert_optimizer_membership(bare_trainer.models[0], bare_trainer.revins[0], True)
        f1_result = results["F1"]
        f4_result = results["F4"]
        p2_result = results["P2"]
        p3_result = results["P3"]
        self.assertEqual(f1_result.source_info, "ABSENT")
        self.assertIsNone(f1_result.source_validation_metric)
        self.assertIsNone(f1_result.effective_source_count)
        self.assertEqual(f4_result.source_info, "PRESENT")
        self.assertEqual(f4_result.source_validation_metric, NOT_APPLICABLE)
        self.assertEqual(f4_result.source_validation_value_space, NOT_APPLICABLE)
        self.assertEqual(p2_result.effective_source_count, 1.0)
        self.assertAlmostEqual(p3_result.effective_source_count, 3.0)
        self.assertIsInstance(f4_result.effective_source_count, float)
        self.assertNotEqual(f4_result.alpha_entropy, f4_result.effective_source_count)
        self.assertEqual(self.source_fit_mock.call_count, 0)
        self.assertEqual(self.stage_a_mock.call_count, 0)
        self.assertEqual(self.rank_mock.call_count, 0)
        self.assertEqual(self.fit_mock.call_count, 0)

    def test_training_uses_only_the_target_train_slice(self):
        windows = _windows("D1", train_value=1.0, val_value=2.0, test_value=0.0)
        trainer = MockTrainer()
        result = adapt_target(_spec("F2"), windows, _scaler("D1"), _cnn_template(5), None, trainer=trainer)
        request = trainer.requests[0]
        validation = trainer.validations[0]
        protocol = json.loads(_PROTOCOL.read_text(encoding="utf-8"))
        contract = load_dataset_contract("D1")
        np.testing.assert_array_equal(request.x, windows.train.x)
        np.testing.assert_array_equal(request.y_scaled, windows.train.y_scaled)
        self.assertFalse(np.array_equal(request.x, windows.validation.x))
        self.assertFalse(np.array_equal(request.x, windows.test.x))
        self.assertEqual(request.x.shape[0], expected_target_partition_count(1))
        self.assertEqual(validation.x.shape[0], expected_target_partition_count(1))
        self.assertEqual(result.n_train, len(windows.train.label_date))
        self.assertEqual(result.n_validation, len(windows.validation.label_date))
        self.assertEqual(validation.role, "record_only")
        self.assertFalse(validation.participates_in_update)
        self.assertFalse(validation.participates_in_model_selection)
        np.testing.assert_array_equal(validation.x, windows.validation.x)
        train_labels = pd.to_datetime(request.label_date)
        val_labels = pd.to_datetime(validation.label_date)
        self.assertTrue((train_labels >= contract.target_train.start).all() and (train_labels <= contract.target_train.end).all())
        self.assertTrue((val_labels >= contract.target_val.start).all() and (val_labels <= contract.target_val.end).all())
        self._assert_inputs_inside(windows.validation, contract.target_val, horizon=1)
        self.assertTrue(set(train_labels).isdisjoint(val_labels))
        self.assertEqual(request.protocol.epochs, protocol["target_optimizer"]["epochs"]["value"])
        self.assertEqual(request.protocol.batch_size, protocol["target_optimizer"]["batch_size"]["value"])
        self.assertEqual(request.protocol.loss_name, "huber")
        self.assertEqual(request.protocol.huber_delta, protocol["huber"]["delta"]["value"])
        self.assertEqual(request.protocol.optimizer_name, protocol["target_optimizer"]["name"]["value"])
        self.assertEqual(request.protocol.schedule, protocol["target_optimizer"]["schedule"]["value"])
        self.assertFalse(request.protocol.schedule_passes_alpha)
        self.assertFalse(request.protocol.early_stopping)
        self.assertEqual(request.protocol.ema_momentum, protocol["ema"]["momentum"]["value"])
        self.assertEqual(request.protocol.final_weight_state, "EMA")
        self.assertEqual(request.protocol.validation_weight_variant, "RAW")
        self.assertEqual(request.protocol.steps_per_epoch, 1)
        self.assertEqual(request.protocol.decay_steps, 50)
        self.assertEqual(result.gradient_steps, 50)
        self.assertEqual(result.epoch0_val_rmse, 1.25)
        self.assertEqual(result.validation_rmse, 0.3)
        self.assertEqual(result.validation_smape, 0.4)
        self.assertEqual(result.validation_weight_variant, "RAW")
        self.assertEqual(result.test_weight_variant, "EMA")
        self.assertNotEqual(result.validation_rmse, result.epoch0_val_rmse)
        self.assertEqual(len(result.train_loss), 50)
        self.assertFalse(result.checkpoint_selection)

    def test_partition_counts_match_each_horizon(self):
        counts = {1: 5, 3: 3, 5: 1}
        contract = load_dataset_contract("D1")
        for horizon, count in counts.items():
            windows = _windows("D1", horizon=horizon, train_value=1.0, val_value=2.0, test_value=0.0)
            trainer = MockTrainer()
            result = adapt_target(
                _spec("F1", horizon=horizon),
                windows,
                _scaler("D1"),
                _cnn_template(5),
                None,
                trainer=trainer,
            )
            with self.subTest(horizon=horizon):
                self.assertEqual(expected_target_partition_count(horizon), count)
                self.assertEqual(len(windows.train.label_date), count)
                self.assertEqual(len(windows.validation.label_date), count)
                self.assertEqual(result.n_train, count)
                self.assertEqual(result.n_validation, len(windows.validation.label_date))
                self.assertEqual(trainer.requests[0].x.shape[0], count)
                self.assertEqual(trainer.validations[0].x.shape[0], count)
                self.assertFalse(trainer.validations[0].participates_in_update)
                self.assertFalse(trainer.validations[0].participates_in_model_selection)
                self.assertFalse(np.array_equal(trainer.requests[0].x, trainer.validations[0].x))
                np.testing.assert_array_equal(trainer.requests[0].x, windows.train.x)
                np.testing.assert_array_equal(trainer.requests[0].label_date, windows.train.label_date)
                self._assert_inputs_inside(windows.train, contract.target_train, horizon)
                self._assert_inputs_inside(windows.validation, contract.target_val, horizon)
                train_days = set(pd.to_datetime(windows.train.label_date))
                val_days = set(pd.to_datetime(windows.validation.label_date))
                self.assertTrue(train_days.isdisjoint(val_days))

    def _assert_inputs_inside(self, batch, span, horizon):
        starts = pd.to_datetime(batch.input_start)
        ends = pd.to_datetime(batch.input_end)
        labels = pd.to_datetime(batch.label_date)
        self.assertTrue((starts >= span.start).all() and (ends <= span.end).all())
        self.assertTrue((labels >= span.start).all() and (labels <= span.end).all())
        self.assertTrue(np.all((ends - starts).days == 9))
        self.assertTrue(np.all((labels - ends).days == horizon))

    def test_validation_cannot_update_or_select(self):
        windows = _windows("D1", train_value=1.0, val_value=2.0, test_value=0.0)
        spec = _spec("F1")
        template = _cnn_template(5)
        scaler = _scaler("D1")

        def select_on_validation(request, model, revin, validation):
            output = MockTrainer()(request, model, revin, validation)
            return TargetTrainOutput(
                train_loss=output.train_loss,
                val_loss=output.val_loss,
                val_rmse=output.val_rmse,
                val_smape=output.val_smape,
                epoch0_val_rmse=output.epoch0_val_rmse,
                gradient_steps=output.gradient_steps,
                raw_weights=output.raw_weights,
                ema_weights=output.ema_weights,
                used_validation_for_update=False,
                used_validation_for_selection=True,
            )

        with self.assertRaises(ContractError):
            adapt_target(spec, windows, scaler, template, None, trainer=select_on_validation)

        def stop_early(request, model, revin, validation):
            output = MockTrainer()(request, model, revin, validation)
            return TargetTrainOutput(
                train_loss=(0.1,),
                val_loss=output.val_loss,
                val_rmse=output.val_rmse,
                val_smape=output.val_smape,
                epoch0_val_rmse=output.epoch0_val_rmse,
                gradient_steps=output.gradient_steps,
                raw_weights=output.raw_weights,
                ema_weights=output.ema_weights,
                used_validation_for_update=False,
                used_validation_for_selection=False,
            )

        with self.assertRaises(ContractError):
            adapt_target(spec, windows, scaler, template, None, trainer=stop_early)

    def test_alt_lr_uses_the_runspec_mapping(self):
        windows_d1 = _windows("D1", train_value=1.0, val_value=2.0, test_value=0.0)
        windows_d5 = _windows("D5", train_value=1.0, val_value=2.0, test_value=0.0)
        bundle = _bundle(_key())
        cases = (
            (_spec("F1", "alt_lr", phase=2), windows_d1, _scaler("D1"), _cnn_template(5), None, 3e-4),
            (_spec("F4", "alt_lr", phase=2), windows_d1, _scaler("D1"), _cnn_template(5), bundle, 3e-4),
            (_spec("F1", "alt_lr", phase=2, dataset="D5"), windows_d5, _scaler("D5"), _cnn_template(11), None, 3e-5),
        )
        for spec, windows, scaler, template, stage_a, rate in cases:
            trainer = MockTrainer()
            result = adapt_target(spec, windows, scaler, template, stage_a, trainer=trainer)
            with self.subTest(arm=spec.arm, dataset=spec.dataset):
                self.assertEqual(spec.target_learning_rate, rate)
                self.assertEqual(trainer.requests[0].protocol.learning_rate, spec.target_learning_rate)
                self.assertEqual(result.learning_rate, spec.target_learning_rate)
                self.assertNotEqual(trainer.requests[0].protocol.learning_rate, 3e-4 if spec.dataset == "D5" else 3e-5)

    def test_test_inference_covers_180_dates_without_updating(self):
        windows = _windows("D1", train_value=1.0, val_value=2.0, test_value=0.0)
        scaler = _scaler("D1")
        scale = np.array(scaler.scale, copy=True)
        fit_dates = scaler.fit_dates
        template = _cnn_template(5, conv_value=0.0, conv_bias=0.0, head_kernel=0.0, head_bias=-4.0)
        trainer = MockTrainer()
        result = adapt_target(_spec("F1"), windows, scaler, template, None, trainer=trainer)
        expected = inverse_sales(scaler, np.array([-4.0], dtype=np.float32))
        contract = load_dataset_contract("D1")
        expected_days = pd.date_range(contract.target_test.start, contract.target_test.end, freq="D")
        self.assertEqual(result.test_predictions.shape, (180,))
        self.assertEqual(result.test_predictions.dtype, np.float64)
        self.assertLess(float(expected[0]), 0.0)
        np.testing.assert_allclose(result.test_predictions, np.full(180, expected[0]))
        self.assertEqual(tuple(pd.to_datetime(result.label_date)), tuple(expected_days))
        np.testing.assert_array_equal(result.label_date, windows.test.label_date)
        self.assertEqual(result.unscored_test_dates, ())
        self.assertEqual(result.test_weight_sha256_before, result.test_weight_sha256_after)
        self.assertNotEqual(trainer.requests[0].x.shape[0], 180)
        np.testing.assert_array_equal(scaler.scale, scale)
        self.assertEqual(scaler.fit_dates, fit_dates)
        self.assertEqual(scaler.role, "target_model")

    def test_frozen_conv_must_match_the_loaded_init(self):
        windows = _windows("D1", train_value=1.0, val_value=2.0, test_value=0.0)
        bundle = _bundle(_key())

        def tamper(request, model, revin, validation):
            if model.get_layer("conv1").trainable:
                raise AssertionError("P1 conv1 was trainable")
            output = MockTrainer()(request, model, revin, validation)
            changed = {name: [np.array(array, copy=True) for array in tensors] for name, tensors in output.raw_weights.items()}
            changed["conv1"][0] = changed["conv1"][0] + np.float32(1.0)
            return TargetTrainOutput(
                train_loss=output.train_loss,
                val_loss=output.val_loss,
                val_rmse=output.val_rmse,
                val_smape=output.val_smape,
                epoch0_val_rmse=output.epoch0_val_rmse,
                gradient_steps=output.gradient_steps,
                raw_weights=changed,
                ema_weights=changed,
                used_validation_for_update=False,
                used_validation_for_selection=False,
            )

        with self.assertRaises(ContractError):
            adapt_target(_spec("P1"), windows, _scaler("D1"), _cnn_template(5), bundle, trainer=tamper)

    def test_sita_tcn_is_blocked_and_tcn_revin_does_not_fit_a_source(self):
        windows = _windows("D1", train_value=1.0, val_value=2.0, test_value=0.0)
        scaler = _scaler("D1")
        incomplete = assess_tcn_source_fit({"loss": None})
        blocked_trainer = MockTrainer()
        with patch("sita_ablation.stage_a.load_tcn_source_fit", return_value=incomplete), patch(
            "sita_ablation.adaptation.fuse_init", side_effect=AssertionError("fuse")
        ) as fuse:
            blocked = adapt_target(
                _spec("SITA_TCN", phase=3), windows, scaler, _tcn_template(), _bundle(_key()), trainer=blocked_trainer
            )
        self.assertEqual(blocked_trainer.requests, [])
        self.assertEqual(blocked.status, BLOCKED_GAP_TCN_SOURCE_FIT)
        self.assertIsNone(blocked.signed_error)
        self.assertIsNone(blocked.alpha_entropy)
        self.assertIsNone(blocked.learning_rate)
        self.assertIsNone(blocked.gradient_steps)
        self.assertEqual(blocked.test_predictions.shape, (0,))
        self.assertEqual(blocked.source_validation_metric, None)
        self.assertFalse(blocked.revin_enabled)
        self.assertFalse(fuse.called)
        self.assertEqual(self.real_mock.call_count, 0)
        self.assertEqual(self.source_fit_mock.call_count, 0)
        self.assertEqual(self.stage_a_mock.call_count, 0)

        sourced = MockTrainer()
        tcn_bundle = _tcn_stage_bundle()
        sourced_result = adapt_target(
            _spec("SITA_TCN", phase=3), windows, scaler, _tcn_template(), tcn_bundle, trainer=sourced
        )
        self.assertEqual(sourced_result.status, "COMPLETED")
        self.assertEqual(sourced.requests[0].protocol.loss_name, "huber")
        self.assertEqual(sourced.requests[0].protocol.schedule, "cosine")
        self.assertEqual(sourced.requests[0].protocol.final_weight_state, "EMA")
        self.assertFalse(sourced.requests[0].protocol.early_stopping)
        self.assertFalse(sourced.validations[0].participates_in_update)
        self.assertFalse(sourced.validations[0].participates_in_model_selection)
        np.testing.assert_array_equal(sourced.requests[0].x, windows.train.x_sales)
        np.testing.assert_allclose(sourced.conv[0][0], fuse_init(tcn_bundle, "inverse_distance")["blocks.0.conv1"][0])
        self.assertEqual(sourced_result.source_validation_metric, NOT_APPLICABLE)
        self.assertIsInstance(sourced_result.alpha_entropy, float)
        self.assertIsNone(sourced_result.signed_error)
        self.assertEqual(self.source_fit_mock.call_count, 0)
        self.assertEqual(self.stage_a_mock.call_count, 0)
        self.assertEqual(self.real_mock.call_count, 0)

        trainer = MockTrainer()
        with patch("sita_ablation.adaptation.fuse_init", side_effect=AssertionError("fuse")) as fuse:
            result = adapt_target(_spec("TCN_REVIN", phase=3), windows, scaler, _tcn_template(), None, trainer=trainer)
        self.assertFalse(fuse.called)
        self.assertEqual(self.source_fit_mock.call_count, 0)
        self.assertEqual(result.status, "COMPLETED")
        self.assertEqual(trainer.requests[0].x.shape, windows.train.x_sales.shape)
        np.testing.assert_array_equal(trainer.requests[0].x, windows.train.x_sales)
        self.assertFalse(np.array_equal(trainer.requests[0].x, windows.train.x))
        self.assertIsNotNone(trainer.revins[0])
        self.assertEqual(trainer.revins[0].num_features, 1)
        self.assertEqual(trainer.requests[0].protocol.loss_name, "huber")
        self.assertEqual(trainer.requests[0].protocol.learning_rate, _spec("TCN_REVIN", phase=3).target_learning_rate)
        np.testing.assert_array_equal(trainer.conv[0][0], _tcn_template().tensors["blocks.0.conv1"][0])
        self.assertNotIn("conv1", result.raw_shift)
        self.assertIn("blocks.0.conv1", result.raw_shift)
        self.assertIn("head", result.raw_shift)
        self.assertEqual(result.ema_shift["revin"].status, UNDEFINED_ZERO_INIT_NORM)
        self.assertIsNone(result.source_validation_metric)
        self.assertEqual(result.test_predictions.shape, (180,))
        self.assertEqual(result.test_weight_sha256_before, result.test_weight_sha256_after)

    def test_bundle_identity_and_shift_rules(self):
        windows = _windows("D1", train_value=1.0, val_value=2.0, test_value=0.0)
        scaler = _scaler("D1")
        template = _cnn_template(5)
        top = _bundle(_key())
        with self.assertRaises(ContractError):
            adapt_target(_spec("F4"), windows, scaler, template, None, trainer=MockTrainer())
        with self.assertRaises(ContractError):
            adapt_target(_spec("F1"), windows, scaler, template, top, trainer=MockTrainer())
        bottom = _bundle(_key(rule="bottom"))
        with self.assertRaises(ContractError):
            adapt_target(_spec("F4"), windows, scaler, template, bottom, trainer=MockTrainer())
        wide = _bundle(_key(trained_k=5))
        with self.assertRaises(ContractError):
            adapt_target(_spec("P2"), windows, scaler, template, wide, trainer=MockTrainer())
        with self.assertRaises(ContractError):
            adapt_target(_spec("F4", "ell15", phase=2), windows, scaler, template, top, trainer=MockTrainer())
        short = _bundle(_key(comparison_days=15), fill_start=7)
        trainer = MockTrainer()
        adapt_target(_spec("F4", "ell15", phase=2), windows, scaler, template, short, trainer=trainer)
        np.testing.assert_allclose(trainer.conv[0][0], fuse_init(short, "inverse_distance")["conv1"][0])
        self.assertFalse(np.allclose(trainer.conv[0][0], fuse_init(top, "inverse_distance")["conv1"][0]))

        zero = parameter_shift([np.zeros(4)], [np.ones(4)])
        self.assertIsNone(zero.relative)
        self.assertEqual(zero.status, UNDEFINED_ZERO_INIT_NORM)
        self.assertGreater(zero.absolute, 0.0)
        untouched = parameter_shift([np.zeros(4)], [np.zeros(4)])
        self.assertIsNone(untouched.relative)
        self.assertEqual(untouched.absolute, 0.0)
        gamma = np.ones(3)
        beta = np.zeros(3)
        unchanged_revin = parameter_shift([gamma, beta], [gamma, beta], unchanged_is_undefined=True)
        self.assertIsNone(unchanged_revin.relative)
        self.assertEqual(unchanged_revin.status, UNDEFINED_ZERO_INIT_NORM)
        moved = parameter_shift([gamma, beta], [np.full(3, 2.0), beta], unchanged_is_undefined=True)
        self.assertEqual(moved.status, DEFINED)
        self.assertIsNotNone(moved.relative)
        rmse, _smape = _recorded_original_errors(np.array([-1.0, 0.0]), np.array([-2.0, 2.0]))
        self.assertAlmostEqual(rmse, float(np.sqrt(2.0)))


def _spec(arm, tag="main", phase=1, dataset="D1", horizon=1):
    return next(
        item
        for item in expand_runs(phase)
        if item.arm == arm
        and item.protocol_tag == tag
        and item.dataset == dataset
        and item.horizon == horizon
        and item.seed == 42
    )


def _frames():
    contract = load_dataset_contract("D1")
    return ConditionFrames(
        dataset="D1",
        source_rows=pd.DataFrame(),
        target_rows=pd.DataFrame(),
        key_fields=contract.key_fields,
        target_key=contract.target_key,
        model_features=contract.model_features,
        similarity_variables=contract.similarity_variables,
        source_history=contract.source_history,
        target_train=contract.target_train,
        target_val=contract.target_val,
        target_test=contract.target_test,
        comparison_30=DateSpan(contract.target_train.start, contract.target_val.end),
        comparison_15=DateSpan(contract.target_train.start, contract.target_train.end),
        data_sha256={},
        grouping_field=None,
    )


def _tcn_stage_bundle():
    key = make_stage_a_key(
        _frames(),
        horizon=1,
        seed=42,
        backbone="tcn",
        selection_rule="top",
        trained_k=3,
        comparison_days=30,
    )
    sources = []
    for rank in range(3):
        value = np.float32(rank + 1)
        sources.append(
            SourceRecord(
                rank=rank,
                source_key=("1", str(rank + 1)),
                weights={
                    name: [np.full(kernel, value), np.full(bias, value)]
                    for name, (kernel, bias) in TCN_FUSIBLE_SHAPES.items()
                },
                scaler=None,  # type: ignore[arg-type]
                sample_count=1,
            )
        )
    return StageABundle(
        key=key,
        status="COMPLETED",
        distances=np.arange(1, 4, dtype=np.float64),
        selected=tuple(source.source_key for source in sources),
        sources=tuple(sources),
        source_validation_metric=NOT_APPLICABLE,
        source_validation_value_space=NOT_APPLICABLE,
        cache_dir=Path("synthetic-not-written"),
    )


def _key(rule="top", trained_k=3, comparison_days=30):
    return make_stage_a_key(
        _frames(),
        horizon=1,
        seed=42,
        backbone="cnn",
        selection_rule=rule,
        trained_k=trained_k,
        comparison_days=comparison_days,
    )


def _bundle(key, fill_start=1):
    sources = []
    channels = len(key.model_features)
    for rank in range(key.trained_k):
        value = np.float32(fill_start + rank)
        sources.append(
            SourceRecord(
                rank=rank,
                source_key=("1", str(rank + 1)),
                weights={
                    "conv1": [np.full((3, channels, 32), value), np.full((32,), value)],
                    "conv2": [np.full((3, 32, 64), value), np.full((64,), value)],
                    "conv3": [np.full((3, 64, 128), value), np.full((128,), value)],
                },
                scaler=None,  # type: ignore[arg-type]
                sample_count=1,
            )
        )
    return StageABundle(
        key=key,
        status="COMPLETED",
        distances=np.arange(1, key.trained_k + 1, dtype=np.float64),
        selected=tuple(source.source_key for source in sources),
        sources=tuple(sources),
        source_validation_metric=NOT_APPLICABLE,
        source_validation_value_space=NOT_APPLICABLE,
        cache_dir=Path("synthetic-not-written"),
    )


def _cnn_template(channels, conv_value=0.5, conv_bias=0.1, head_kernel=0.2, head_bias=-0.25):
    return WeightSnapshot(
        tensors={
            "conv1": (np.full((3, channels, 32), conv_value, np.float32), np.full((32,), conv_bias, np.float32)),
            "conv2": (np.full((3, 32, 64), conv_value, np.float32), np.full((64,), conv_bias, np.float32)),
            "conv3": (np.full((3, 64, 128), conv_value, np.float32), np.full((128,), conv_bias, np.float32)),
            "head_target": (np.full((256, 1), head_kernel, np.float32), np.full((1,), head_bias, np.float32)),
            "affine_weight": (np.ones((channels,), np.float32),),
            "affine_bias": (np.zeros((channels,), np.float32),),
        }
    )


def _tcn_template():
    tensors = {
        name: (np.full(kernel, 0.3, np.float32), np.full(bias, 0.05, np.float32))
        for name, (kernel, bias) in TCN_FUSIBLE_SHAPES.items()
    }
    tensors["head"] = (np.full((16, 1), 0.2, np.float32), np.full((1,), -0.25, np.float32))
    tensors["affine_weight"] = (np.ones((1,), np.float32),)
    tensors["affine_bias"] = (np.zeros((1,), np.float32),)
    return WeightSnapshot(tensors=tensors)


def _scaler(dataset):
    contract = load_dataset_contract(dataset)
    start = contract.target_train.start
    columns = contract.model_features
    frame = {"date": [start, start + pd.Timedelta(days=1)], "sales": [0.0, 10.0]}
    for column in columns:
        if column != "sales":
            frame[column] = [0.0, 1.0]
    return fit_target_model_scaler(pd.DataFrame(frame), columns)


def _windows(dataset, *, train_value, val_value, test_value, horizon=1):
    """Build each partition from its own frozen slice. Validation does not reuse train days."""
    contract = load_dataset_contract(dataset)
    columns = contract.model_features
    train_rows = _slice_frame(contract.target_train, columns, train_value, contract)
    val_rows = _slice_frame(contract.target_val, columns, val_value, contract)
    test_rows = _slice_frame(contract.target_test, columns, test_value, contract)
    observed = pd.concat([val_rows, test_rows], ignore_index=True)
    test_start, test_end = contract.target_test.iso()
    return TargetWindows(
        train=build_target_partition_windows(
            train_rows,
            key_fields=contract.key_fields,
            key=contract.target_key,
            feature_columns=columns,
            horizon=horizon,
            partition="train",
        ),
        validation=build_target_partition_windows(
            val_rows,
            key_fields=contract.key_fields,
            key=contract.target_key,
            feature_columns=columns,
            horizon=horizon,
            partition="validation",
        ),
        test=build_test_windows(
            observed,
            test_start=test_start,
            test_end=test_end,
            feature_columns=columns,
            horizon=horizon,
        ),
    )


def _slice_frame(span, columns, value, contract):
    days = pd.date_range(span.start, span.end, freq="D")
    if len(days) not in (15, 180):
        raise AssertionError(f"frozen slice {span.iso()} has {len(days)} days")
    data = {
        "date": days,
        "sales": np.full(len(days), value, dtype=np.float64),
        "sales_original": np.full(len(days), value, dtype=np.float64),
    }
    for column in columns:
        if column not in data:
            data[column] = np.full(len(days), value, dtype=np.float64)
    for field, part in zip(contract.key_fields, contract.target_key):
        data[field] = part
    return pd.DataFrame(data)


if __name__ == "__main__":
    unittest.main()
