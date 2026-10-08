"""Synthetic Stage A cache tests. The trainer is mocked and no sealed data is read."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import tensorflow as tf

from sita_ablation.data import (
    ConditionFrames,
    ContractError,
    DateSpan,
    clip_source_history,
    load_dataset_contract,
    split_source_frame,
)
from sita_ablation.source_selection import inverse_distance_weights, uniform_weights
from sita_ablation.stage_a import (
    fit_source_cnn,
    fuse_init,
    load_or_train_stage_a,
    make_stage_a_key,
    stage_a_key_sha256,
)


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


def _weights(rank):
    value = np.float32(rank + 1)
    return {
        "conv1": [np.full((3, 5, 32), value), np.full((32,), value)],
        "conv2": [np.full((3, 32, 64), value), np.full((64,), value)],
        "conv3": [np.full((3, 64, 128), value), np.full((128,), value)],
        "dense_out": [np.full((256, 1), np.float32(50 + rank)), np.ones((1,), dtype=np.float32)],
    }


class StageACacheTests(unittest.TestCase):
    def test_shared_cache_reload_and_p2_reuse(self):
        frames = _frames((1, 2, 3))
        key = make_stage_a_key(
            frames,
            horizon=1,
            seed=42,
            backbone="cnn",
            selection_rule="top",
            trained_k=3,
            comparison_days=30,
        )
        other_keys = [
            make_stage_a_key(frames, horizon=1, seed=42, backbone="cnn", selection_rule="bottom", trained_k=3, comparison_days=30),
            make_stage_a_key(frames, horizon=1, seed=42, backbone="cnn", selection_rule="top", trained_k=5, comparison_days=30),
            make_stage_a_key(frames, horizon=1, seed=42, backbone="cnn", selection_rule="top", trained_k=3, comparison_days=15),
            make_stage_a_key(frames, horizon=1, seed=42, backbone="tcn", selection_rule="top", trained_k=3, comparison_days=30),
            make_stage_a_key(frames, horizon=1, seed=42, backbone="cnn", selection_rule="random3", trained_k=3, comparison_days=30),
        ]
        digests = {stage_a_key_sha256(key)} | {stage_a_key_sha256(item) for item in other_keys}
        self.assertEqual(len(digests), 6)
        self.assertEqual(other_keys[-1].draw_rule, "sorted_permutation_v1")
        self.assertIsNone(key.draw_rule)
        calls = []

        def trainer(request):
            self.assertEqual(request.x.shape[-1], 5)
            self.assertFalse(hasattr(request, "y_validation"))
            split = split_source_frame(clip_source_history(frames, request.source_key), key.model_features)
            train_dates = {pd.Timestamp(day).strftime("%Y-%m-%d") for day in split.train_dates}
            validation_dates = {pd.Timestamp(day).strftime("%Y-%m-%d") for day in split.validation_dates}
            labels = {pd.Timestamp(day).strftime("%Y-%m-%d") for day in request.label_date}
            self.assertTrue(labels <= train_dates)
            self.assertTrue(labels.isdisjoint(validation_dates))
            calls.append(request.rank)
            return _weights(request.rank)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = load_or_train_stage_a(key, frames, cache_root=root, trainer=trainer)
            second = load_or_train_stage_a(key, frames, cache_root=root, trainer=trainer)
            self.assertEqual(calls, [0, 1, 2])
            self.assertEqual(len(list(root.iterdir())), 1)
            self.assertEqual(first.cache_dir, second.cache_dir)
            self.assertEqual(first.source_validation_metric, "NOT_APPLICABLE")
            self.assertEqual(first.source_validation_value_space, "NOT_APPLICABLE")
            names = {path.name for path in first.cache_dir.iterdir()}
            self.assertIn("DONE.json", names)
            self.assertNotIn("fused.npz", names)
            selection_text = (first.cache_dir / "selection.json").read_text(encoding="utf-8")
            self.assertNotIn("inverse_distance", selection_text)
            self.assertNotIn("uniform_weights", selection_text)
            for rank, source in enumerate(first.sources):
                self.assertEqual(source.scaler.role, "source_model")
                np.testing.assert_allclose(second.sources[rank].weights["conv1"][0], source.weights["conv1"][0])
                split = split_source_frame(clip_source_history(frames, source.source_key), key.model_features)
                self.assertEqual(
                    set(source.scaler.fit_dates),
                    {pd.Timestamp(day).strftime("%Y-%m-%d") for day in split.train_dates},
                )
            single = fuse_init(first, "single")
            inverse = fuse_init(first, "inverse_distance")
            uniform = fuse_init(first, "uniform")
            np.testing.assert_allclose(single["conv1"][0], first.sources[0].weights["conv1"][0])
            self.assertNotIn("dense_out", inverse)
            expected_inverse = sum(
                float(inverse_distance_weights(first.distances)[rank]) * first.sources[rank].weights["conv1"][0]
                for rank in range(3)
            )
            expected_uniform = sum(
                float(uniform_weights(3)[rank]) * first.sources[rank].weights["conv1"][0]
                for rank in range(3)
            )
            np.testing.assert_allclose(inverse["conv1"][0], expected_inverse)
            np.testing.assert_allclose(uniform["conv1"][0], expected_uniform)
            self.assertFalse(np.allclose(inverse["conv1"][0], uniform["conv1"][0]))
            self.assertEqual(len(list(root.iterdir())), 1)

    def test_skipped_when_fewer_than_k(self):
        frames = _frames((1, 2))
        key = make_stage_a_key(
            frames, horizon=1, seed=42, backbone="cnn", selection_rule="top", trained_k=3, comparison_days=30
        )
        calls = []
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = load_or_train_stage_a(key, frames, cache_root=root, trainer=lambda request: calls.append(request))
            again = load_or_train_stage_a(key, frames, cache_root=root, trainer=lambda request: calls.append(request))
            self.assertEqual(calls, [])
            self.assertEqual(bundle.status, "SKIPPED_INSUFFICIENT_CANDIDATES")
            self.assertEqual(again.status, bundle.status)
            self.assertEqual(bundle.sources, ())
            names = {path.name for path in bundle.cache_dir.iterdir()}
            self.assertEqual(names, {"key.json", "selection.json", "DONE.json"})
            done = json.loads((bundle.cache_dir / "DONE.json").read_text(encoding="utf-8"))
            self.assertEqual(done["status"], "SKIPPED_INSUFFICIENT_CANDIDATES")

    def test_hash_mismatch_keeps_the_directory(self):
        frames = _frames((1, 2, 3))
        key = make_stage_a_key(
            frames, horizon=1, seed=43, backbone="cnn", selection_rule="top", trained_k=3, comparison_days=30
        )
        calls = []
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = load_or_train_stage_a(
                key, frames, cache_root=root, trainer=lambda request: calls.append(request.rank) or _weights(request.rank)
            )
            weight_path = bundle.cache_dir / "source_0.npz"
            original = weight_path.read_bytes()
            weight_path.write_bytes(original + b"x")
            with self.assertRaises(ContractError) as caught:
                load_or_train_stage_a(key, frames, cache_root=root, trainer=lambda request: calls.append("again"))
            self.assertIn("HOLD_HASH_MISMATCH", str(caught.exception))
            self.assertEqual(calls, [0, 1, 2])
            self.assertEqual(weight_path.read_bytes(), original + b"x")
            self.assertTrue((bundle.cache_dir / "DONE.json").exists())

    def test_failure_retries_once_then_holds(self):
        frames = _frames((1, 2, 3))
        key = make_stage_a_key(
            frames, horizon=1, seed=44, backbone="cnn", selection_rule="top", trained_k=3, comparison_days=30
        )
        state = {"failures": 0}

        def trainer(request):
            state["failures"] += 1
            raise RuntimeError("synthetic fit failure")

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises(RuntimeError):
                load_or_train_stage_a(key, frames, cache_root=root, trainer=trainer)
            with self.assertRaises(RuntimeError):
                load_or_train_stage_a(key, frames, cache_root=root, trainer=trainer)
            with self.assertRaises(ContractError) as caught:
                load_or_train_stage_a(key, frames, cache_root=root, trainer=trainer)
            self.assertIn("HOLD_FAILED", str(caught.exception))
            self.assertEqual(state["failures"], 2)
            directory = root / stage_a_key_sha256(key)
            self.assertTrue((directory / "FAILED.attempt1.json").exists())
            failed = json.loads((directory / "FAILED.json").read_text(encoding="utf-8"))
            self.assertEqual(failed["attempt_count"], 2)
            self.assertFalse((directory / "DONE.json").exists())

    def test_tcn_fit_is_refused_when_the_contract_is_incomplete(self):
        from sita_ablation.stage_a import assess_tcn_source_fit

        frames = _frames((1, 2, 3))
        key = make_stage_a_key(
            frames, horizon=1, seed=42, backbone="tcn", selection_rule="top", trained_k=3, comparison_days=30
        )
        incomplete = assess_tcn_source_fit({"loss": None})
        with tempfile.TemporaryDirectory() as temp, patch(
            "sita_ablation.stage_a.load_tcn_source_fit", return_value=incomplete
        ):
            with self.assertRaises(ContractError) as caught:
                load_or_train_stage_a(key, frames, cache_root=Path(temp))
            self.assertIn("GAP-TCN-SOURCE-FIT", str(caught.exception))
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_source_cnn_fit_uses_the_frozen_settings(self):
        seen = {}

        def fake_fit(model, *args, **kwargs):
            seen["kwargs"] = kwargs
            seen["loss"] = model.loss
            seen["clipnorm"] = model.optimizer.clipnorm
            seen["lr"] = float(model.optimizer.learning_rate)
            return None

        x = np.zeros((4, 10, 5), dtype=np.float32)
        y = np.zeros((4,), dtype=np.float32)
        with patch.object(tf.keras.Model, "fit", fake_fit):
            weights = fit_source_cnn(x, y)
        self.assertNotIn("validation_data", seen["kwargs"])
        self.assertEqual(seen["kwargs"]["epochs"], 50)
        self.assertEqual(seen["kwargs"]["batch_size"], 16)
        self.assertEqual(seen["loss"], "mse")
        self.assertIsNone(seen["clipnorm"])
        self.assertAlmostEqual(seen["lr"], 1e-4)
        self.assertEqual(weights["conv1"][0].shape, (3, 5, 32))
        self.assertEqual(weights["dense_out"][0].shape, (256, 1))


if __name__ == "__main__":
    unittest.main()
