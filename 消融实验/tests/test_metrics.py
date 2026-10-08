"""Hand-checked original-sales metrics.

Synthetic vectors only. These tests do not read sealed tables, call a model,
train, start a run, or write a result file.
"""

from __future__ import annotations

import ast
import math
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from sita_ablation.adaptation import (
    ABSENT,
    COMPLETED,
    PRESENT,
    AdaptationResult,
)
from sita_ablation.arms import expand_runs
from sita_ablation.data import DATASETS, ContractError, load_dataset_contract
from sita_ablation.metrics import (
    ORIGINAL_SALES,
    TestMetrics,
    alpha_entropy,
    effective_source_count,
    evaluate_original_sales,
    evaluate_test,
    signed_error,
)
from sita_ablation.stage_a import NOT_APPLICABLE

_METRICS_PATH = Path(__file__).resolve().parents[1] / "sita_ablation" / "metrics.py"


class OriginalSalesFormulaTests(unittest.TestCase):
    def test_rmse_smape_and_counts_match_the_hand_formula(self):
        true = np.array([-1.0, 0.0, 4.0], dtype=np.float64)
        pred = np.array([-2.0, 2.0, 4.0], dtype=np.float64)
        pair = evaluate_original_sales(true, pred)
        # Clip first: true [0, 0, 4], pred [0, 2, 4].
        self.assertEqual(pair.value_space, ORIGINAL_SALES)
        self.assertEqual(pair.rmse, math.sqrt((0.0 + 4.0 + 0.0) / 3.0))
        self.assertNotEqual(pair.rmse, math.sqrt((1.0 + 4.0 + 0.0) / 3.0))
        eps = 1e-8
        term_zero = 2.0 * 0.0 / (0.0 + 0.0 + eps)
        term_two = 2.0 * 2.0 / (0.0 + 2.0 + eps)
        term_match = 0.0
        self.assertEqual(pair.smape, 100.0 * ((term_zero + term_two + term_match) / 3.0))
        self.assertEqual(pair.zero_sales_count, 1)
        self.assertEqual(pair.smape_zero_denominator_count, 1)
        self.assertEqual(pair.clipped_true_negative_count, 1)
        self.assertEqual(pair.clipped_pred_negative_count, 1)
        self.assertIsInstance(pair.rmse, float)
        self.assertIsInstance(pair.smape, float)

    def test_zero_denominator_samples_stay_in_the_mean(self):
        true = np.array([0.0, 10.0], dtype=np.float64)
        pred = np.array([0.0, 0.0], dtype=np.float64)
        pair = evaluate_original_sales(true, pred)
        eps = 1e-8
        term_zero = 2.0 * 0.0 / (0.0 + 0.0 + eps)
        term_ten = 2.0 * 10.0 / (10.0 + 0.0 + eps)
        kept = 100.0 * ((term_zero + term_ten) / 2.0)
        dropped = 100.0 * term_ten
        self.assertEqual(pair.smape, kept)
        self.assertNotEqual(pair.smape, dropped)
        self.assertEqual(pair.smape_zero_denominator_count, 1)
        self.assertEqual(pair.zero_sales_count, 1)
        self.assertEqual(pair.rmse, math.sqrt((0.0 + 100.0) / 2.0))
        self.assertTrue(math.isfinite(pair.smape))

    def test_smape_uses_the_frozen_denominator_term_and_full_precision(self):
        pair = evaluate_original_sales(np.array([1.0]), np.array([0.0]))
        eps = 1e-8
        expected = 100.0 * (2.0 * 1.0 / (1.0 + 0.0 + eps))
        self.assertEqual(pair.smape, expected)
        self.assertNotEqual(pair.smape, 200.0)
        self.assertNotEqual(pair.smape, round(pair.smape, 4))
        self.assertEqual(pair.rmse, 1.0)
        self.assertEqual(pair.smape_zero_denominator_count, 0)

    def test_original_scale_does_not_add_an_upper_clip_or_rmse_epsilon(self):
        pair = evaluate_original_sales(np.array([0.0]), np.array([10000.0]))
        self.assertEqual(pair.rmse, 10000.0)
        self.assertNotEqual(pair.rmse, 10000.0 + 1e-8)
        both_zero = evaluate_original_sales(np.array([0.0]), np.array([0.0]))
        self.assertEqual(both_zero.rmse, 0.0)
        self.assertEqual(both_zero.smape, 0.0)
        self.assertEqual(both_zero.zero_sales_count, 1)
        self.assertEqual(both_zero.smape_zero_denominator_count, 1)

    def test_pre_clip_zero_sales_are_not_the_post_clip_zeros(self):
        pair = evaluate_original_sales(np.array([-4.0]), np.array([-3.0]))
        self.assertEqual(pair.rmse, 0.0)
        self.assertEqual(pair.smape, 0.0)
        self.assertEqual(pair.zero_sales_count, 0)
        self.assertEqual(pair.smape_zero_denominator_count, 1)
        self.assertEqual(pair.clipped_true_negative_count, 1)
        self.assertEqual(pair.clipped_pred_negative_count, 1)

    def test_float64_values_are_not_collapsed(self):
        true = np.array([float(2**24 + 1), 0.0], dtype=np.float64)
        pred = np.array([float(2**24), 0.0], dtype=np.float64)
        pair = evaluate_original_sales(true, pred)
        self.assertEqual(pair.rmse, math.sqrt(0.5))
        self.assertNotEqual(pair.rmse, 0.0)

    def test_formula_rejects_misaligned_or_non_finite_vectors(self):
        with self.assertRaises(ContractError) as mismatch:
            evaluate_original_sales(np.array([1.0, 2.0]), np.array([1.0]))
        self.assertIn("lengths differ", str(mismatch.exception))
        with self.assertRaises(ContractError) as nan:
            evaluate_original_sales(np.array([1.0, np.nan]), np.array([1.0, 1.0]))
        self.assertIn("non-finite", str(nan.exception))
        with self.assertRaises(ContractError) as infinite:
            evaluate_original_sales(np.array([1.0]), np.array([np.inf]))
        self.assertIn("non-finite", str(infinite.exception))
        with self.assertRaises(ContractError) as shaped:
            evaluate_original_sales(np.ones((3, 1)), np.ones((3, 1)))
        self.assertIn("one-dimensional", str(shaped.exception))
        with self.assertRaises(ContractError):
            evaluate_original_sales(np.array([]), np.array([]))


class EffectiveSourceCountTests(unittest.TestCase):
    def test_formula_is_reciprocal_sum_of_squares(self):
        self.assertEqual(effective_source_count(np.array([0.5, 0.5, 0.0])), 2.0)
        self.assertEqual(effective_source_count(np.array([1.0])), 1.0)
        self.assertEqual(effective_source_count(np.array([2.0, 0.0])), 0.25)
        weights = np.array([1.0, 1.0, 1.0], dtype=np.float64) / np.float64(3.0)
        hand = 1.0 / (float(weights[0]) ** 2 + float(weights[1]) ** 2 + float(weights[2]) ** 2)
        self.assertEqual(effective_source_count(weights), hand)

    def test_undefined_or_non_finite_weights_fail(self):
        with self.assertRaises(ContractError):
            effective_source_count(np.array([0.0, 0.0]))
        with self.assertRaises(ContractError):
            effective_source_count(np.array([1.0, np.nan]))
        with self.assertRaises(ContractError):
            effective_source_count(np.array([]))


class FrozenTestWindowTests(unittest.TestCase):
    def test_every_dataset_covers_the_frozen_180_days(self):
        for dataset in DATASETS:
            with self.subTest(dataset=dataset):
                dates = _dates(dataset)
                self.assertEqual(len(dates), 180)
                self.assertEqual(len(set(dates)), 180)
                true = np.zeros(180, dtype=np.float64)
                pred = np.zeros(180, dtype=np.float64)
                pred[0] = 1.0
                scored = evaluate_test(_result(_spec("F1", dataset), dates, pred), true)
                self.assertIsInstance(scored, TestMetrics)
                self.assertEqual(scored.dataset, dataset)
                self.assertEqual(scored.n_test, 180)
                self.assertEqual(scored.unscored_test_dates, ())
                self.assertEqual(scored.pair.value_space, ORIGINAL_SALES)
                self.assertEqual(scored.pair.rmse, math.sqrt(1.0 / 180.0))
                eps = 1e-8
                one_day = 2.0 * abs(1.0 - 0.0) / (abs(0.0) + abs(1.0) + eps)
                self.assertEqual(scored.pair.smape, 100.0 * (one_day / 180.0))
                self.assertNotEqual(scored.pair.smape, 100.0 * one_day)
                self.assertEqual(scored.pair.zero_sales_count, 180)
                self.assertEqual(scored.pair.smape_zero_denominator_count, 179)
                self.assertEqual(scored.pair.clipped_true_negative_count, 0)
                self.assertEqual(scored.pair.clipped_pred_negative_count, 0)
                self.assertTrue(np.array_equal(scored.label_date, dates.to_numpy(dtype="datetime64[D]")))
                self.assertEqual(scored.pair.rmse, evaluate_original_sales(true, pred).rmse)

    def test_clip_counts_on_the_frozen_window_keep_original_values(self):
        dates = _dates("D6")
        true = np.ones(180, dtype=np.float64)
        pred = np.ones(180, dtype=np.float64)
        true[0] = -2.0
        pred[0] = -5.0
        true[1] = 0.0
        pred[1] = 0.0
        scored = evaluate_test(_result(_spec("F1", "D6"), dates, pred), true)
        self.assertEqual(scored.y_true[0], -2.0)
        self.assertEqual(scored.y_pred[0], -5.0)
        self.assertEqual(scored.pair.rmse, 0.0)
        self.assertNotEqual(scored.pair.rmse, math.sqrt(9.0 / 180.0))
        self.assertEqual(scored.pair.smape, 0.0)
        self.assertEqual(scored.pair.zero_sales_count, 1)
        self.assertEqual(scored.pair.smape_zero_denominator_count, 2)
        self.assertEqual(scored.pair.clipped_true_negative_count, 1)
        self.assertEqual(scored.pair.clipped_pred_negative_count, 1)

    def test_original_units_are_not_rescaled(self):
        dates = _dates("D5")
        true = np.zeros(180, dtype=np.float64)
        pred = np.full(180, 50.0, dtype=np.float64)
        scored = evaluate_test(_result(_spec("F1", "D5"), dates, pred), true)
        self.assertEqual(scored.y_pred[0], 50.0)
        self.assertEqual(scored.pair.rmse, 50.0)

    def test_string_dates_align_pointwise(self):
        dates = _dates("D4")
        labels = tuple(day.strftime("%Y-%m-%d") for day in dates)
        true = np.arange(180, dtype=np.float64)
        pred = true + 3.0
        scored = evaluate_test(_result(_spec("F1", "D4"), np.array(labels, dtype=object), pred), true)
        self.assertEqual(scored.pair.rmse, 3.0)
        self.assertEqual(str(scored.label_date[0]), "2025-01-15")
        self.assertEqual(str(scored.label_date[-1]), "2025-07-13")
        self.assertEqual(scored.y_true[7], 7.0)
        self.assertEqual(scored.y_pred[7], 10.0)

    def test_signed_error_and_alpha_entropy_use_the_closed_formulas(self):
        dates = _dates("D1")
        alpha = np.array([0.5, 0.5, 0.0], dtype=np.float64)
        pred = np.full(180, 4.0, dtype=np.float64)
        true = np.full(180, 1.0, dtype=np.float64)
        pred[0] = -2.0
        true[1] = -3.0
        scored = evaluate_test(
            _result(
                _spec("F4"),
                dates,
                pred,
                source_info=PRESENT,
                source_metric=NOT_APPLICABLE,
                source_space=NOT_APPLICABLE,
                effective=2.0,
                train_loss=(4.5, 3.5),
            ),
            true,
            alpha=alpha,
        )
        self.assertEqual(scored.effective_source_count, 2.0)
        self.assertGreater(scored.signed_error, 0.0)
        self.assertEqual(scored.signed_error, signed_error(true, pred))
        self.assertEqual(scored.alpha_entropy, alpha_entropy(alpha))
        self.assertNotEqual(scored.signed_error, scored.effective_source_count)
        self.assertNotEqual(scored.alpha_entropy, scored.effective_source_count)
        self.assertEqual(scored.source_validation_metric, NOT_APPLICABLE)
        self.assertEqual(scored.source_validation_value_space, NOT_APPLICABLE)
        self.assertNotEqual(scored.source_validation_metric, 4.5)
        self.assertNotIn("4.5", scored.source_validation_metric)
        self.assertNotEqual(scored.source_validation_value_space, "mse")

    def test_source_free_arm_leaves_source_validation_absent(self):
        dates = _dates("D1")
        scored = evaluate_test(
            _result(_spec("F1"), dates, np.zeros(180), train_loss=(4.5,)),
            np.zeros(180),
        )
        self.assertEqual(scored.source_info, ABSENT)
        self.assertIsNone(scored.source_validation_metric)
        self.assertIsNone(scored.source_validation_value_space)
        self.assertIsNone(scored.effective_source_count)
        self.assertEqual(scored.signed_error, 0.0)
        self.assertIsNone(scored.alpha_entropy)

    def test_uniform_weights_use_only_the_sum_of_squares(self):
        dates = _dates("D1")
        alpha = np.array([1.0, 1.0, 1.0], dtype=np.float64) / np.float64(3.0)
        hand = 1.0 / (float(alpha[0]) ** 2 + float(alpha[1]) ** 2 + float(alpha[2]) ** 2)
        scored = evaluate_test(
            _result(
                _spec("P3"),
                dates,
                np.ones(180),
                source_info=PRESENT,
                source_metric=NOT_APPLICABLE,
                source_space=NOT_APPLICABLE,
                effective=hand,
            ),
            np.zeros(180),
            alpha=alpha,
        )
        self.assertEqual(scored.effective_source_count, hand)
        self.assertEqual(scored.alpha_entropy, alpha_entropy(alpha))

    def test_date_prediction_and_true_value_failures(self):
        dates = _dates("D1")
        pred = np.ones(180, dtype=np.float64)
        true = np.zeros(180, dtype=np.float64)
        base = _result(_spec("F1"), dates, pred)
        cases = {
            "length": (base, true[:-1], "lengths differ"),
            "column": (
                _result(_spec("F1"), dates, pred.reshape(180, 1)),
                true.reshape(180, 1),
                "one-dimensional",
            ),
            "nan": (_result(_spec("F1"), dates, _with(pred, 5, np.nan)), true, "non-finite"),
            "inf": (_result(_spec("F1"), dates, pred), _with(true, 2, np.inf), "non-finite"),
            "missing": (_result(_spec("F1"), _with_object(dates, -1, None), pred), true, "missing"),
            "blank": (_result(_spec("F1"), _with_object(dates, 0, "  "), pred), true, "missing"),
            "duplicate": (_result(_spec("F1"), _duplicate_last(dates), pred), true, "duplicate"),
            "short": (
                _result(_spec("F1"), dates[:-1], pred[:-1]),
                true[:-1],
                "180",
            ),
            "extra": (
                _result(_spec("F1"), _extend(dates), np.ones(181)),
                np.zeros(181),
                "180",
            ),
            "shifted": (_result(_spec("F1"), dates + pd.Timedelta(days=1), pred), true, "do not cover"),
            "reversed": (_result(_spec("F1"), dates[::-1], pred), true, "misaligned"),
            "swapped": (_result(_spec("F1"), _swap_adjacent(dates), pred), true, "misaligned"),
            "other_dataset": (_result(_spec("F1", "D1"), _dates("D4"), pred), true, "do not cover"),
            "raw_weights": (replace(base, test_weight_variant="RAW"), true, "EMA"),
            "unscored": (replace(base, unscored_test_dates=("2017-07-01",)), true, "unscored"),
            "incomplete_status": (replace(base, status="BLOCKED"), true, "completed"),
        }
        for name, (result, observed, text) in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(ContractError) as raised:
                    evaluate_test(result, observed)
                self.assertIn(text, str(raised.exception))

    def test_pointwise_date_vector_must_match_both_sales_vectors(self):
        dates = _dates("D1")
        pred = np.ones(180, dtype=np.float64)
        result = _result(_spec("F1"), dates[:-1], pred)
        with self.assertRaises(ContractError) as raised:
            evaluate_test(result, np.zeros(180))
        self.assertIn("pointwise", str(raised.exception))

    def test_resolved_guesses_and_substituted_source_error_fail(self):
        dates = _dates("D1")
        pred = np.zeros(180, dtype=np.float64)
        true = np.zeros(180, dtype=np.float64)
        sourced = _result(
            _spec("F4"),
            dates,
            pred,
            source_info=PRESENT,
            source_metric=NOT_APPLICABLE,
            source_space=NOT_APPLICABLE,
            effective=2.0,
            train_loss=(4.5,),
        )
        alpha = np.array([0.5, 0.5, 0.0], dtype=np.float64)
        guesses = [
            replace(sourced, signed_error="0.5"),
            replace(sourced, signed_error=0.5),
            replace(sourced, alpha_entropy="1.0"),
            replace(sourced, alpha_entropy=1.0),
            replace(sourced, source_validation_metric="mse"),
            replace(sourced, source_validation_metric="4.5"),
            replace(sourced, source_validation_value_space="scaled_huber"),
            replace(sourced, effective_source_count=9.0),
        ]
        for guess in guesses:
            with self.subTest(guess=guess.signed_error, metric=guess.source_validation_metric):
                with self.assertRaises(ContractError):
                    evaluate_test(guess, true, alpha=alpha)
        with self.assertRaises(ContractError):
            evaluate_test(sourced, true, alpha=np.array([0.5, 0.5]))
        with self.assertRaises(ContractError):
            evaluate_test(sourced, true)
        free = _result(_spec("F1"), dates, pred)
        with self.assertRaises(ContractError):
            evaluate_test(free, true, alpha=alpha)

    def test_metrics_stay_in_memory_and_do_not_import_src(self):
        source = _METRICS_PATH.read_text(encoding="utf-8")
        for token in ("open(", "to_parquet", "to_csv", "json.dump", "wilcoxon", "bootstrap", "holm", "import src"):
            self.assertNotIn(token, source)
        self.assertIn("np.log(", source)
        self.assertIn("np.clip(", source)
        self.assertNotIn("np.log2", source)
        self.assertNotIn("np.log10", source)
        self.assertNotIn("math.log2", source)
        self.assertNotIn("math.log10", source)
        tree = ast.parse(source)
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
        self.assertFalse(any(name == "src" or name.startswith("src.") for name in modules))
        self.assertIn("evaluate_original_sales", source)
        dates = _dates("D1")
        scored = evaluate_test(_result(_spec("F1"), dates, np.zeros(180)), np.zeros(180))
        self.assertIsInstance(scored, TestMetrics)
        self.assertNotIsInstance(scored, (str, Path))


def _spec(arm: str, dataset: str = "D1"):
    return next(
        item
        for item in expand_runs(1)
        if item.arm == arm
        and item.protocol_tag == "main"
        and item.dataset == dataset
        and item.horizon == 1
        and item.seed == 42
    )


def _dates(dataset: str) -> pd.DatetimeIndex:
    contract = load_dataset_contract(dataset)
    return pd.date_range(contract.target_test.start, contract.target_test.end, freq="D")


def _result(
    spec,
    dates,
    pred,
    *,
    source_info=ABSENT,
    source_metric=None,
    source_space=None,
    effective=None,
    train_loss=(4.5,),
):
    return AdaptationResult(
        status=COMPLETED,
        spec=spec,
        train_loss=tuple(train_loss),
        val_loss=(0.2,),
        val_rmse=(0.3,),
        val_smape=(0.4,),
        epoch0_val_rmse=1.25,
        validation_rmse=0.3,
        validation_smape=0.4,
        validation_weight_variant="RAW",
        gradient_steps=50,
        raw_shift={},
        ema_shift={},
        test_predictions=np.array(pred, copy=True),
        label_date=np.array(dates, copy=True),
        unscored_test_dates=(),
        test_weight_variant="EMA",
        test_weight_sha256_before=None,
        test_weight_sha256_after=None,
        signed_error=None,
        alpha_entropy=None,
        effective_source_count=effective,
        source_info=source_info,
        source_validation_metric=source_metric,
        source_validation_value_space=source_space,
        trainable_names=(),
        frozen_names=(),
        revin_enabled=spec.revin,
        learning_rate=float(spec.target_learning_rate),
        checkpoint_selection=False,
    )


def _with(values: np.ndarray, index: int, replacement: float) -> np.ndarray:
    copied = np.array(values, dtype=np.float64, copy=True)
    copied[index] = replacement
    return copied


def _with_object(dates: pd.DatetimeIndex, index: int, replacement) -> np.ndarray:
    copied = np.array(dates.to_list(), dtype=object)
    copied[index] = replacement
    return copied


def _duplicate_last(dates: pd.DatetimeIndex) -> np.ndarray:
    copied = dates.to_numpy(dtype="datetime64[D]").copy()
    copied[-1] = copied[0]
    return copied


def _extend(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.date_range(dates[0], dates[-1] + pd.Timedelta(days=1), freq="D")


def _swap_adjacent(dates: pd.DatetimeIndex) -> np.ndarray:
    copied = dates.to_numpy(dtype="datetime64[ns]").copy()
    copied[0], copied[1] = copied[1], copied[0]
    return copied


if __name__ == "__main__":
    unittest.main()
