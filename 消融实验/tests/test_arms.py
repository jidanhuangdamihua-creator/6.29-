"""Synthetic arm-identity tests. No sealed data, source fit, or run artifacts."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from sita_ablation.arms import (
    BLOCKED_GAP_TCN_SOURCE_FIT,
    expand_runs,
    frozen_tensors,
    prepare_arm,
    run_directory,
    stage_a_key_for,
    tensor_trainable,
    validate_fusible_weights,
)
from sita_ablation.data import ConditionFrames, ContractError, DateSpan, load_dataset_contract
from sita_ablation.stage_a import (
    NOT_APPLICABLE,
    SourceRecord,
    StageABundle,
    assess_tcn_source_fit,
    stage_a_cache_dir,
    stage_a_key_sha256,
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


def _spec(arm, tag="main", phase=1):
    return next(
        item
        for item in expand_runs(phase)
        if item.arm == arm and item.protocol_tag == tag and item.dataset == "D1" and item.horizon == 1 and item.seed == 42
    )


def _arrays(rank):
    value = np.float32(rank + 1)
    return {
        "conv1": [np.full((3, 5, 32), value), np.full((32,), value)],
        "conv2": [np.full((3, 32, 64), value), np.full((64,), value)],
        "conv3": [np.full((3, 64, 128), value), np.full((128,), value)],
        "head_target": [np.full((256, 1), np.float32(9)), np.ones((1,), dtype=np.float32)],
        "affine_weight": [np.ones((5,), dtype=np.float32)],
        "affine_bias": [np.zeros((5,), dtype=np.float32)],
    }


def _bundle(key):
    sources = tuple(
        SourceRecord(rank=rank, source_key=("1", str(rank + 1)), weights=_arrays(rank), scaler=None, sample_count=1)  # type: ignore[arg-type]
        for rank in range(3)
    )
    return StageABundle(
        key=key,
        status="COMPLETED",
        distances=np.array([1.0, 3.0, 6.0], dtype=np.float64),
        selected=(("1", "1"), ("1", "2"), ("1", "3")),
        sources=sources,
        source_validation_metric=NOT_APPLICABLE,
        source_validation_value_space=NOT_APPLICABLE,
        cache_dir=Path("synthetic-not-written"),
    )


_GRID = {
    (dataset, horizon, seed)
    for dataset in ("D1", "D4", "D5", "D6")
    for horizon in (1, 3, 5)
    for seed in (42, 43, 44, 45, 46)
}


def _cells(specs, arm, tag):
    return {
        (item.dataset, item.horizon, item.seed)
        for item in specs
        if item.arm == arm and item.protocol_tag == tag
    }


class ArmIdentityTests(unittest.TestCase):
    def test_phase_counts_follow_the_frozen_arm_table(self):
        phase1 = expand_runs(1)
        phase2 = expand_runs(2)
        phase3 = expand_runs(3)
        self.assertEqual(len(phase1), 420)
        self.assertEqual(len(phase2), 540)
        self.assertEqual(len(phase3), 120)
        for arm in ("F1", "F2", "F3", "F4", "P1", "P2", "P3"):
            self.assertEqual(_cells(phase1, arm, "main"), _GRID, arm)
        # alt_lr is four arms, each one swapped learning rate, not one shared condition.
        for arm in ("F1", "F2", "F3", "F4"):
            self.assertEqual(_cells(phase2, arm, "alt_lr"), _GRID, arm)
            for dataset, rate in (("D1", 3e-4), ("D4", 3e-4), ("D5", 3e-5), ("D6", 3e-4)):
                rates = {
                    item.target_learning_rate
                    for item in phase2
                    if item.arm == arm and item.protocol_tag == "alt_lr" and item.dataset == dataset
                }
                self.assertEqual(rates, {rate})
        for tag in ("random3", "bottom3", "k5", "k10", "ell15"):
            self.assertEqual(_cells(phase2, "F4", tag), _GRID, tag)
        self.assertEqual(
            {(item.arm, item.protocol_tag) for item in phase2},
            {
                ("F1", "alt_lr"),
                ("F2", "alt_lr"),
                ("F3", "alt_lr"),
                ("F4", "alt_lr"),
                ("F4", "random3"),
                ("F4", "bottom3"),
                ("F4", "k5"),
                ("F4", "k10"),
                ("F4", "ell15"),
            },
        )
        self.assertEqual(_cells(phase3, "SITA_TCN", "main"), _GRID)
        self.assertEqual(_cells(phase3, "TCN_REVIN", "main"), _GRID)
        phase4 = expand_runs(4)
        ell15 = tuple(item for item in phase2 if item.protocol_tag == "ell15")
        self.assertEqual(len(phase4), 60)
        self.assertEqual(phase4, ell15)
        self.assertTrue(all(item.arm == "F4" and item.comparison_days == 15 and item.requested_k == 3 for item in phase4))
        phase5 = expand_runs(5)
        k5 = tuple(item for item in phase2 if item.protocol_tag == "k5")
        self.assertEqual(len(phase5), 60)
        self.assertEqual(phase5, k5)
        self.assertTrue(all(item.arm == "F4" and item.comparison_days == 30 and item.requested_k == 5 for item in phase5))
        names = {item.arm for item in phase1 + phase2 + phase3}
        self.assertTrue({"X1", "X2", "X3"}.isdisjoint(names))
        extra = expand_runs(1, include_no_revin_variants=True)
        self.assertEqual(len(extra), 600)
        self.assertTrue({"X1", "X2", "X3"}.isdisjoint({item.arm for item in extra}))
        for arm in ("P1", "P2", "P3"):
            bare = [item for item in extra if item.arm == arm and item.revin is False]
            self.assertEqual(len(bare), 60)
            self.assertEqual({(item.dataset, item.horizon, item.seed) for item in bare}, _GRID)

    def test_phase_grid_and_factor_switches(self):
        phase1 = expand_runs(1)
        phase2 = expand_runs(2)
        phase3 = expand_runs(3)
        self.assertEqual(len(phase1), 420)
        self.assertEqual(len(phase2), 540)
        self.assertEqual(len(phase3), 120)
        self.assertTrue(all(item.arm not in {"X1", "X2", "X3"} for item in phase1 + phase2 + phase3))
        f1, f2, f3, f4 = (_spec(name) for name in ("F1", "F2", "F3", "F4"))
        p1, p2, p3 = (_spec(name) for name in ("P1", "P2", "P3"))
        self.assertFalse(f1.revin)
        self.assertTrue(f2.revin)
        self.assertFalse(f3.revin)
        self.assertTrue(f4.revin and p1.revin and p2.revin and p3.revin)
        self.assertTrue(p1.freeze_conv)
        self.assertFalse(any(item.freeze_conv for item in (f1, f2, f3, f4, p2, p3)))
        self.assertEqual(frozen_tensors(p1), (
            "conv1.kernel", "conv1.bias", "conv2.kernel", "conv2.bias", "conv3.kernel", "conv3.bias",
        ))
        self.assertFalse(tensor_trainable(p1, "conv1.kernel"))
        self.assertTrue(tensor_trainable(p1, "head_target.kernel"))
        self.assertTrue(tensor_trainable(p1, "affine_weight"))
        self.assertTrue(tensor_trainable(f4, "conv1.kernel"))
        self.assertEqual(f1.target_learning_rate, 3e-5)
        self.assertEqual(_spec("F1", "alt_lr", phase=2).target_learning_rate, 3e-4)
        d5 = next(item for item in phase1 if item.arm == "F4" and item.dataset == "D5")
        d5_alt = next(item for item in phase2 if item.arm == "F4" and item.protocol_tag == "alt_lr" and item.dataset == "D5")
        self.assertEqual(d5.target_learning_rate, 3e-4)
        self.assertEqual(d5_alt.target_learning_rate, 3e-5)
        self.assertEqual(run_directory(f4, Path("/tmp/runs")), Path("/tmp/runs/F4/main/D1_h1/seed42"))
        self.assertEqual(run_directory(p1, Path("/tmp/runs")), Path("/tmp/runs/P1/main/D1_h1/seed42"))
        extra = expand_runs(1, include_no_revin_variants=True)
        self.assertEqual(len(extra), 600)
        bare_p1 = next(item for item in extra if item.arm == "P1" and item.revin is False)
        self.assertEqual(
            run_directory(bare_p1, Path("/tmp/runs")),
            Path("/tmp/runs/P1/main/D1_h1/seed42/revin0"),
        )
        self.assertNotEqual(run_directory(p1, Path("/tmp/runs")), run_directory(bare_p1, Path("/tmp/runs")))

    def test_run_directories_are_unique_and_not_created(self):
        frames = _frames()
        specs = expand_runs(1) + expand_runs(2) + expand_runs(3)
        extra = [item for item in expand_runs(1, include_no_revin_variants=True) if item.revin is False and item.arm in {"P1", "P2", "P3"}]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "runs"
            paths = [run_directory(item, root) for item in (*specs, *extra)]
            self.assertEqual(len(paths), len(set(paths)))
            self.assertTrue(all(not path.exists() for path in paths))
            self.assertFalse(root.exists())
            repeated = run_directory(specs[0], root)
            self.assertEqual(repeated, run_directory(specs[0], root))
            self.assertFalse(repeated.exists())
        f1 = _spec("F1")
        f1_alt = _spec("F1", "alt_lr", phase=2)
        f4 = _spec("F4")
        self.assertNotEqual(run_directory(f1, Path("/tmp/runs")), run_directory(f1_alt, Path("/tmp/runs")))
        self.assertNotEqual(run_directory(f4, Path("/tmp/runs")), run_directory(_spec("F4", "random3", phase=2), Path("/tmp/runs")))
        self.assertNotEqual(run_directory(f4, Path("/tmp/runs")), run_directory(_spec("F4", "k5", phase=2), Path("/tmp/runs")))
        self.assertNotEqual(run_directory(f4, Path("/tmp/runs")), run_directory(_spec("F4", "ell15", phase=2), Path("/tmp/runs")))
        self.assertNotEqual(run_directory(f4, Path("/tmp/runs")), run_directory(_spec("P3"), Path("/tmp/runs")))
        bare_p1 = next(item for item in extra if item.arm == "P1" and item.dataset == "D1" and item.horizon == 1 and item.seed == 42)
        self.assertEqual(stage_a_key_for(_spec("P1"), frames), stage_a_key_for(bare_p1, frames))

    def test_shared_stage_a_key_and_distinct_variants(self):
        frames = _frames()
        shared = [
            _spec("F3"),
            _spec("F4"),
            _spec("P1"),
            _spec("P3"),
            _spec("P2"),
            _spec("F3", "alt_lr", phase=2),
            _spec("F4", "alt_lr", phase=2),
        ]
        keys = [stage_a_key_for(spec, frames) for spec in shared]
        self.assertTrue(all(key is not None and key.trained_k == 3 and key.selection_rule == "top" for key in keys))
        self.assertEqual(len({stage_a_key_sha256(key) for key in keys}), 1)
        self.assertEqual(
            stage_a_cache_dir(keys[0], Path("/tmp/stage-a")),
            stage_a_cache_dir(keys[4], Path("/tmp/stage-a")),
        )
        f4 = keys[1]
        variants = {
            "random3": stage_a_key_for(_spec("F4", "random3", phase=2), frames),
            "bottom3": stage_a_key_for(_spec("F4", "bottom3", phase=2), frames),
            "k5": stage_a_key_for(_spec("F4", "k5", phase=2), frames),
            "k10": stage_a_key_for(_spec("F4", "k10", phase=2), frames),
            "ell15": stage_a_key_for(_spec("F4", "ell15", phase=2), frames),
            "tcn": stage_a_key_for(_spec("SITA_TCN", phase=3), frames),
        }
        digests = {stage_a_key_sha256(f4)} | {stage_a_key_sha256(key) for key in variants.values()}
        self.assertEqual(len(digests), 7)
        self.assertEqual(variants["random3"].selection_rule, "random3")
        self.assertEqual(variants["bottom3"].selection_rule, "bottom")
        self.assertEqual(variants["k5"].trained_k, 5)
        self.assertEqual(variants["k10"].trained_k, 10)
        self.assertEqual(variants["ell15"].comparison_start, f4.comparison_start)
        self.assertNotEqual(variants["ell15"].comparison_end, f4.comparison_end)
        self.assertEqual(variants["ell15"].source_history_end, f4.source_history_end)
        self.assertEqual(variants["tcn"].backbone, "tcn")
        self.assertEqual(variants["tcn"].model_features, ("sales",))

    def test_inverse_uniform_and_single_inits(self):
        frames = _frames()
        f4 = _spec("F4")
        bundle = _bundle(stage_a_key_for(f4, frames))
        template = _arrays(8)
        prepared = {
            spec.arm: prepare_arm(spec, frames, template, bundle)
            for spec in (_spec("F3"), f4, _spec("P1"), _spec("P2"), _spec("P3"))
        }
        inverse = prepared["F4"].fused["conv1"][0]
        self.assertTrue(np.allclose(prepared["F3"].fused["conv1"][0], inverse))
        self.assertTrue(np.allclose(prepared["P1"].fused["conv1"][0], inverse))
        self.assertFalse(np.allclose(prepared["P3"].fused["conv1"][0], inverse))
        self.assertTrue(np.allclose(prepared["P3"].fused["conv1"][0], np.full((3, 5, 32), 2.0)))
        self.assertTrue(np.allclose(prepared["P2"].fused["conv1"][0], bundle.sources[0].weights["conv1"][0]))
        self.assertEqual(prepared["F4"].spec.weight_mode, "inverse_distance")
        self.assertEqual(prepared["P3"].spec.weight_mode, "uniform")
        self.assertEqual(prepared["P2"].spec.weight_mode, "single")
        self.assertTrue(np.allclose(prepared["F4"].initialized["head_target"][0], template["head_target"][0]))
        f2 = prepare_arm(_spec("F2"), frames, template)
        f1 = prepare_arm(_spec("F1"), frames, template)
        self.assertIsNone(f1.stage_a_key)
        self.assertFalse(f1.spec.revin)
        self.assertTrue(f2.spec.revin)
        self.assertTrue(np.array_equal(f1.initialized["conv1"][0], template["conv1"][0]))
        self.assertTrue(np.array_equal(f2.initialized["affine_weight"][0], np.ones((5,), dtype=np.float32)))
        self.assertEqual(prepared["F3"].spec.revin, False)
        self.assertEqual(prepared["P1"].frozen_tensors, frozen_tensors(_spec("P1")))
        self.assertEqual(prepared["F4"].source_validation_metric, "NOT_APPLICABLE")
        other = _bundle(stage_a_key_for(_spec("F4", "bottom3", phase=2), frames))
        with self.assertRaises(ContractError):
            prepare_arm(f4, frames, template, other)

    def test_fusible_shape_and_order_errors(self):
        good = {name: _arrays(0)[name] for name in ("conv1", "conv2", "conv3")}
        validate_fusible_weights("cnn", good, channels=5)
        reversed_layers = {name: good[name] for name in ("conv3", "conv2", "conv1")}
        with self.assertRaises(ContractError):
            validate_fusible_weights("cnn", reversed_layers, channels=5)
        missing = {name: good[name] for name in ("conv1", "conv3")}
        with self.assertRaises(ContractError):
            validate_fusible_weights("cnn", missing, channels=5)
        bad_shape = {name: [array.copy() for array in tensors] for name, tensors in good.items()}
        bad_shape["conv1"][0] = np.full((3, 4, 32), 1, dtype=np.float32)
        with self.assertRaises(ContractError):
            validate_fusible_weights("cnn", bad_shape, channels=5)
        tcn = {
            "blocks.0.conv1": [np.zeros((3, 1, 16), dtype=np.float32), np.zeros((16,), dtype=np.float32)],
            "blocks.0.conv2": [np.zeros((3, 16, 16), dtype=np.float32), np.zeros((16,), dtype=np.float32)],
            "blocks.0.residual": [np.zeros((1, 1, 16), dtype=np.float32), np.zeros((16,), dtype=np.float32)],
            "blocks.1.conv1": [np.zeros((3, 16, 16), dtype=np.float32), np.zeros((16,), dtype=np.float32)],
            "blocks.1.conv2": [np.zeros((3, 16, 16), dtype=np.float32), np.zeros((16,), dtype=np.float32)],
        }
        validate_fusible_weights("tcn", tcn, channels=1)
        del tcn["blocks.0.residual"]
        with self.assertRaises(ContractError):
            validate_fusible_weights("tcn", tcn, channels=1)

    def test_sourced_tcn_is_blocked_while_the_contract_is_incomplete(self):
        frames = _frames()
        spec = _spec("SITA_TCN", phase=3)
        incomplete = assess_tcn_source_fit({"loss": None})
        with patch("sita_ablation.stage_a.load_tcn_source_fit", return_value=incomplete):
            prepared = prepare_arm(spec, frames, _arrays(0), bundle=None)
        self.assertEqual(prepared.status, BLOCKED_GAP_TCN_SOURCE_FIT)
        self.assertIsNone(prepared.fused)
        self.assertIsNone(prepared.initialized)
        self.assertEqual(prepared.stage_a_key.backbone, "tcn")
        self.assertTrue(spec.revin)
        self.assertFalse(spec.freeze_conv)
        plain = prepare_arm(_spec("TCN_REVIN", phase=3), frames, _arrays(0))
        self.assertEqual(plain.status, "READY")
        self.assertIsNone(plain.stage_a_key)
        self.assertTrue(plain.spec.revin)


if __name__ == "__main__":
    unittest.main()
