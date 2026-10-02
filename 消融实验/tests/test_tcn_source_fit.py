"""Synthetic checks for the closed error formulas and the TCN source-fit gate.

No sealed tables, no real ``model.fit``, and no formal run.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from sita_ablation.arms import BLOCKED_GAP_TCN_SOURCE_FIT, expand_runs, prepare_arm
from sita_ablation.data import (
    ConditionFrames,
    ContractError,
    DateSpan,
    clip_source_history,
    load_dataset_contract,
    split_source_frame,
)
from sita_ablation.metrics import alpha_entropy, effective_source_count, signed_error
from sita_ablation.runner import dry_run
from sita_ablation.scaling import fit_source_model_scaler, transform_model_frame
from sita_ablation.stage_a import (
    assess_tcn_source_fit,
    fit_source_tcn,
    inspect_stage_a,
    load_or_train_stage_a,
    load_tcn_source_fit,
    make_stage_a_key,
    stage_a_cache_dir,
    stage_a_key_sha256,
)
from sita_ablation.windows import build_partition_windows

_CONFIG = Path(__file__).resolve().parents[1] / "configs"


def _frames(items):
    contract = load_dataset_contract("D1")
    history = pd.date_range(contract.source_history.start, contract.source_history.end, freq="D")
    comparison = pd.date_range(contract.target_train.start, contract.target_val.end, freq="D")
    target_rows = [_point(contract, contract.target_key, day, 0.0) for day in comparison]
    source_rows = []
    for item in items:
        for index, day in enumerate(history):
            source_rows.append(_point(contract, ("1", str(item)), day, item * 10.0 + (index % 7)))
    return ConditionFrames(
        dataset="D1",
        source_rows=pd.DataFrame(source_rows),
        target_rows=pd.DataFrame(target_rows),
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


def _point(contract, key, day, sales):
    stamp = pd.Timestamp(day).normalize()
    row = {"date": stamp, "sales": float(sales)}
    for field, part in zip(contract.key_fields, key):
        row[field] = part
    row["year"] = int(stamp.year)
    row["month"] = int(stamp.month)
    row["week"] = int(stamp.isocalendar().week)
    row["day"] = int(stamp.day)
    return row


_FROZEN_FIELDS = (
    "loss",
    "optimizer",
    "learning_rate",
    "batch_size",
    "epochs",
    "clipnorm",
    "shuffle",
    "validation_used_for_fit",
    "validation_for_early_stopping",
    "validation_for_checkpoint_selection",
    "validation_for_source_ranking",
    "ema",
    "final_weights",
    "source_revin",
    "seed",
)


def _frozen_section() -> dict:
    payload = json.loads((_CONFIG / "tcn_frozen.json").read_text(encoding="utf-8"))
    section = payload["tcn_source_fit"]
    return {name: section[name] for name in _FROZEN_FIELDS}


def _old_section() -> dict:
    return {
        "loss": None,
        "optimizer": None,
        "learning_rate": None,
        "batch_size": None,
        "epochs": None,
        "ema": {"value": False},
        "final_weight_state": None,
        "validation": {
            "enters_fit": False,
            "participates_in_update": False,
            "early_stopping": False,
            "checkpoint_selection": False,
            "model_selection": False,
            "source_validation_metric": "NOT_APPLICABLE",
            "source_validation_value_space": "NOT_APPLICABLE",
        },
        "seed": {"value": "cell_seed", "reset": "once per StageAKey before sequential source fits"},
        "source_revin": {"value": "UNUSED"},
    }


def _tcn_weights():
    zero = np.float32(0)
    return {
        "blocks.0.conv1": [np.full((3, 1, 16), zero), np.full((16,), zero)],
        "blocks.0.conv2": [np.full((3, 16, 16), zero), np.full((16,), zero)],
        "blocks.0.residual": [np.full((1, 1, 16), zero), np.full((16,), zero)],
        "blocks.1.conv1": [np.full((3, 16, 16), zero), np.full((16,), zero)],
        "blocks.1.conv2": [np.full((3, 16, 16), zero), np.full((16,), zero)],
    }


class ClosedFormulaTests(unittest.TestCase):
    def test_signed_error_sign_matches_over_and_under_estimation(self):
        self.assertGreater(signed_error(np.array([1.0, 2.0]), np.array([4.0, 6.0])), 0.0)
        self.assertLess(signed_error(np.array([4.0, 6.0]), np.array([1.0, 2.0])), 0.0)
        self.assertEqual(signed_error(np.array([3.0]), np.array([3.0])), 0.0)
        self.assertEqual(signed_error(np.array([-5.0]), np.array([-1.0])), 0.0)

    def test_alpha_entropy_uses_natural_log_clip_and_records_neff(self):
        weights = np.array([0.5, 0.5, 0.0], dtype=np.float64)
        clipped = np.clip(weights, 1e-12, 1.0)
        terms = np.where(clipped == 0.0, 0.0, clipped * np.log(clipped))
        self.assertEqual(alpha_entropy(weights), float(-np.sum(terms)))
        self.assertNotEqual(alpha_entropy(weights), float(-np.sum(np.where(clipped == 0.0, 0.0, clipped * np.log2(clipped)))))
        self.assertTrue(np.isfinite(alpha_entropy(np.array([0.0, 0.0, 1.0]))))
        self.assertEqual(effective_source_count(weights), 1.0 / (0.5**2 + 0.5**2 + 0.0**2))
        self.assertNotEqual(alpha_entropy(weights), effective_source_count(weights))


class TcnSourceFitContractTests(unittest.TestCase):
    def test_frozen_contract_unblocks_sita_tcn(self):
        contract = load_tcn_source_fit()
        self.assertTrue(contract.complete)
        self.assertEqual(contract.missing, ())
        self.assertEqual(contract.loss, "mse")
        self.assertEqual(contract.optimizer, "adam")
        self.assertEqual(contract.learning_rate, 1e-4)
        self.assertEqual(contract.batch_size, 16)
        self.assertEqual(contract.epochs, 50)
        self.assertIsNone(contract.clipnorm)
        self.assertIs(contract.shuffle, True)
        self.assertIs(contract.validation_used_for_fit, False)
        self.assertIs(contract.validation_for_early_stopping, False)
        self.assertIs(contract.validation_for_checkpoint_selection, False)
        self.assertIs(contract.validation_for_source_ranking, False)
        self.assertIs(contract.ema, False)
        self.assertEqual(contract.final_weights, "last_epoch_raw")
        self.assertEqual(contract.source_revin, "UNUSED")
        self.assertEqual(contract.seed, "cell_seed_reset_per_stage_a_key")
        frames = _frames((1, 2, 3))
        spec = _sita_spec()
        with patch("sita_ablation.data.load_condition_frames", side_effect=AssertionError("sealed")), patch(
            "pandas.read_parquet", side_effect=AssertionError("parquet")
        ):
            preview = dry_run(3, workers=1, only_cell="D1:1:42", workspace_dirty=False)
        by_arm = {cell.spec.arm: cell for cell in preview.plan}
        self.assertEqual(by_arm["SITA_TCN"].status, "planned")
        self.assertNotEqual(by_arm["SITA_TCN"].status, BLOCKED_GAP_TCN_SOURCE_FIT)
        self.assertEqual(by_arm["TCN_REVIN"].status, "planned")
        self.assertIsNone(by_arm["TCN_REVIN"].stage_a_key_sha256)
        self.assertEqual(preview.preflight.gap_tcn_source_fit, "CLOSED")
        self.assertNotIn("GAP-TCN-SOURCE-FIT", preview.preflight.reasons)
        self.assertEqual(by_arm["TCN_REVIN"].spec.selection_rule, "none")
        self.assertEqual(by_arm["SITA_TCN"].spec.selection_rule, "top")
        with self.assertRaises(ContractError) as caught:
            prepare_arm(spec, frames, {}, bundle=None)
        self.assertNotIn("GAP-TCN-SOURCE-FIT", str(caught.exception))

    def test_missing_or_null_field_stays_blocked(self):
        frozen = _frozen_section()
        for name in _FROZEN_FIELDS:
            removed = dict(frozen)
            del removed[name]
            self._assert_blocked(removed, label=f"missing {name}")
            if name == "clipnorm":
                continue
            nulled = dict(frozen)
            nulled[name] = None
            self._assert_blocked(nulled, label=f"null {name}")
        self._assert_blocked(dict(frozen, clipnorm=1.0), label="clipnorm number")
        self._assert_blocked(_old_section(), label="old contract")
        old = assess_tcn_source_fit(_old_section())
        self.assertIsNone(old.loss)
        self.assertIsNone(old.optimizer)
        self.assertIsNone(old.learning_rate)
        self.assertIsNone(old.epochs)
        self.assertIsNone(old.final_weights)
        self.assertNotEqual(old.seed, "cell_seed_reset_per_stage_a_key")

    def test_wrong_type_or_inherited_target_settings_stay_blocked(self):
        frozen = _frozen_section()
        for section in (
            dict(frozen, loss="huber"),
            dict(frozen, optimizer="Adam"),
            dict(frozen, learning_rate="1e-4"),
            dict(frozen, batch_size=16.0),
            dict(frozen, epochs=True),
            dict(frozen, shuffle=1),
            dict(frozen, ema=0),
            dict(frozen, final_weights="last_epoch"),
            dict(frozen, seed="cell_seed"),
            dict(frozen, source_revin={"value": "UNUSED"}),
            {**frozen, "patience": 5},
            {**frozen, "early_stopping": False},
            {**frozen, "best_state": True},
            {**frozen, "huber": 1.0},
            {**frozen, "cosine": True},
        ):
            self._assert_blocked(section, label=str(sorted(section)))

    def test_mse_uses_source_minmax_sales_and_validation_stays_out(self):
        frames = _frames((1, 2, 3))
        key = _tcn_key(frames)
        calls = []

        def trainer(request):
            calls.append(request)
            self.assertEqual(request.x.shape[-1], 1)
            self.assertFalse(hasattr(request, "validation"))
            clipped = clip_source_history(frames, request.source_key)
            split = split_source_frame(clipped, ("sales",))
            scaler = fit_source_model_scaler(split.train, ("sales",))
            transformed = transform_model_frame(scaler, split.train)
            windows = build_partition_windows(
                transformed,
                key_fields=frames.key_fields,
                key=request.source_key,
                feature_columns=("sales",),
                horizon=1,
                partition="train",
            )
            np.testing.assert_array_equal(request.x, windows.x_sales)
            np.testing.assert_array_equal(request.y_scaled, windows.y_scaled)
            raw = split.train["sales"].to_numpy(dtype=np.float64)
            low, high = float(raw.min()), float(raw.max())
            by_day = {
                pd.Timestamp(day).strftime("%Y-%m-%d"): float(value)
                for day, value in zip(split.train["date"], raw)
            }
            for day, observed in zip(request.label_date, request.y_scaled):
                stamp = pd.Timestamp(day).strftime("%Y-%m-%d")
                expected = (by_day[stamp] - low) / (high - low)
                self.assertAlmostEqual(float(observed), expected, places=5)
            start = pd.Timestamp(windows.input_start[0])
            end = pd.Timestamp(windows.input_end[0])
            dates = pd.to_datetime(split.train["date"])
            raw_days = split.train.loc[(dates >= start) & (dates <= end), "sales"].to_numpy(dtype=np.float64)
            self.assertEqual(raw_days.shape, (10,))
            expected_window = (raw_days - low) / (high - low)
            np.testing.assert_allclose(request.x[0, :, 0], expected_window, atol=1e-5)
            revin = (raw_days - float(np.mean(raw_days))) / float(np.std(raw_days))
            self.assertFalse(np.allclose(request.x[0, :, 0], revin, atol=1e-5))
            labels = {pd.Timestamp(day).strftime("%Y-%m-%d") for day in request.label_date}
            train_dates = {pd.Timestamp(day).strftime("%Y-%m-%d") for day in split.train_dates}
            validation_dates = {pd.Timestamp(day).strftime("%Y-%m-%d") for day in split.validation_dates}
            self.assertTrue(labels <= train_dates)
            self.assertTrue(labels.isdisjoint(validation_dates))
            return _tcn_weights()

        with tempfile.TemporaryDirectory() as temp:
            bundle = load_or_train_stage_a(key, frames, cache_root=Path(temp), trainer=trainer)
        self.assertEqual(bundle.status, "COMPLETED")
        self.assertEqual(len(calls), 3)
        self.assertEqual(bundle.source_validation_metric, "NOT_APPLICABLE")
        self.assertEqual(bundle.source_validation_value_space, "NOT_APPLICABLE")
        for source in bundle.sources:
            self.assertEqual(source.scaler.role, "source_model")
            self.assertEqual(source.scaler.columns, ("sales",))
            split = split_source_frame(clip_source_history(frames, source.source_key), ("sales",))
            self.assertEqual(
                set(source.scaler.fit_dates),
                {pd.Timestamp(day).strftime("%Y-%m-%d") for day in split.train_dates},
            )
            self.assertTrue(
                set(source.scaler.fit_dates).isdisjoint(
                    {pd.Timestamp(day).strftime("%Y-%m-%d") for day in split.validation_dates}
                )
            )

    def test_fit_keeps_validation_out_and_caches_last_epoch_raw_weights(self):
        seen = {}
        order = []

        def fake_seed(seed):
            order.append(("seed", int(seed)))

        def fake_fit(model, *args, **kwargs):
            order.append("fit")
            seen["args"] = args
            seen["kwargs"] = kwargs
            seen["loss"] = model.loss
            seen["clipnorm"] = model.optimizer.clipnorm
            seen["lr"] = float(model.optimizer.learning_rate)
            seen["names"] = [layer.name for layer in model.layers]
            layer = model.get_layer("blocks.0.conv1")
            kernel, bias = layer.get_weights()
            layer.set_weights([np.full_like(kernel, 0.25), bias])
            model._ema_shadow = np.full_like(kernel, 0.99)
            return None

        frames = _frames((1, 2, 3))
        key = _tcn_key(frames)
        import tensorflow as tf

        with tempfile.TemporaryDirectory() as temp, patch.object(tf.keras.Model, "fit", fake_fit), patch(
            "tensorflow.keras.utils.set_random_seed", fake_seed
        ):
            bundle = load_or_train_stage_a(key, frames, cache_root=Path(temp))
            cached = np.load(bundle.cache_dir / "source_0.npz")
            cached_kernel = np.asarray(cached["blocks.0.conv1__0"])
            cached.close()
        self.assertEqual(order[0], ("seed", 42))
        self.assertEqual(order[1:], ["fit", "fit", "fit"])
        self.assertEqual(seen["args"][0].shape[-1], 1)
        self.assertEqual(seen["args"][0].shape[0], seen["args"][1].shape[0])
        self.assertNotIn("validation_data", seen["kwargs"])
        self.assertNotIn("callbacks", seen["kwargs"])
        self.assertEqual(seen["kwargs"]["epochs"], 50)
        self.assertEqual(seen["kwargs"]["batch_size"], 16)
        self.assertIs(seen["kwargs"]["shuffle"], True)
        self.assertEqual(seen["loss"], "mse")
        self.assertIsNone(seen["clipnorm"])
        self.assertAlmostEqual(seen["lr"], 1e-4)
        self.assertTrue(all("revin" not in name for name in seen["names"]))
        self.assertTrue(np.all(cached_kernel == np.float32(0.25)))
        self.assertFalse(np.all(cached_kernel == np.float32(0.99)))

    def test_cnn_and_tcn_caches_do_not_read_each_other(self):
        frames = _frames((1, 2, 3))
        cnn_key = make_stage_a_key(
            frames, horizon=1, seed=42, backbone="cnn", selection_rule="top", trained_k=3, comparison_days=30
        )
        tcn_key = _tcn_key(frames)
        self.assertNotEqual(stage_a_key_sha256(cnn_key), stage_a_key_sha256(tcn_key))
        cnn_calls = []
        tcn_calls = []
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            load_or_train_stage_a(
                cnn_key,
                frames,
                cache_root=root,
                trainer=lambda request: cnn_calls.append(request.backbone) or _cnn_weights(),
            )
            self.assertEqual(inspect_stage_a(stage_a_cache_dir(cnn_key, root), tcn_key), "HOLD_HASH_MISMATCH")
            load_or_train_stage_a(
                tcn_key,
                frames,
                cache_root=root,
                trainer=lambda request: tcn_calls.append(request.backbone) or _tcn_weights(),
            )
            self.assertNotEqual(stage_a_cache_dir(cnn_key, root), stage_a_cache_dir(tcn_key, root))
        self.assertEqual(cnn_calls, ["cnn", "cnn", "cnn"])
        self.assertEqual(tcn_calls, ["tcn", "tcn", "tcn"])

    def test_incomplete_contract_blocks_only_sita_tcn(self):
        incomplete = assess_tcn_source_fit(_old_section())
        with patch("sita_ablation.stage_a.load_tcn_source_fit", return_value=incomplete), patch(
            "sita_ablation.data.load_condition_frames", side_effect=AssertionError("sealed")
        ), patch("pandas.read_parquet", side_effect=AssertionError("parquet")):
            preview = dry_run(3, workers=1, only_cell="D1:1:42", workspace_dirty=False)
        by_arm = {cell.spec.arm: cell for cell in preview.plan}
        self.assertEqual(by_arm["SITA_TCN"].status, BLOCKED_GAP_TCN_SOURCE_FIT)
        self.assertEqual(by_arm["TCN_REVIN"].status, "planned")
        self.assertIsNone(by_arm["TCN_REVIN"].stage_a_key_sha256)
        self.assertIn("GAP-TCN-SOURCE-FIT", preview.preflight.reasons)
        self.assertEqual(preview.preflight.gap_tcn_source_fit, "UNRESOLVED")

    def _assert_blocked(self, section, label):
        contract = assess_tcn_source_fit(section)
        self.assertFalse(contract.complete, label)
        frames = _frames((1, 2, 3))
        with patch("sita_ablation.stage_a.load_tcn_source_fit", return_value=contract):
            prepared = prepare_arm(_sita_spec(), frames, {}, bundle=None)
        self.assertEqual(prepared.status, BLOCKED_GAP_TCN_SOURCE_FIT, label)
        self.assertIsNone(prepared.fused, label)


def _sita_spec():
    return next(
        item
        for item in expand_runs(3)
        if item.arm == "SITA_TCN" and item.dataset == "D1" and item.horizon == 1 and item.seed == 42
    )


def _tcn_key(frames):
    return make_stage_a_key(
        frames, horizon=1, seed=42, backbone="tcn", selection_rule="top", trained_k=3, comparison_days=30
    )


def _cnn_weights():
    zero = np.float32(0)
    return {
        "conv1": [np.full((3, 5, 32), zero), np.full((32,), zero)],
        "conv2": [np.full((3, 32, 64), zero), np.full((64,), zero)],
        "conv3": [np.full((3, 64, 128), zero), np.full((128,), zero)],
    }


if __name__ == "__main__":
    unittest.main()
