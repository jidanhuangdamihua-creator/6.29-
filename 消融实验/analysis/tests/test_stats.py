"""Hand-checked tests for RI, pairing, Wilcoxon, bootstrap, and Holm."""

from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from analysis.comparisons import CONDITIONS, assign_holm, condition_vector
from analysis.formulas import original_sales_scores
from analysis.stats import (
    StatisticsError,
    condition_mean,
    five_number,
    holm,
    paired_wilcoxon,
    percentile_bootstrap,
    relative_improvement,
    win_loss_tie,
)


class RelativeImprovementTests(unittest.TestCase):
    def test_hand_calculation_and_zero_control(self):
        # (4-2)/4*100 = 50; (2-4)/2*100 = -100.
        ri = relative_improvement(np.array([2.0, 4.0]), np.array([4.0, 2.0]))
        self.assertTrue(np.array_equal(ri, np.array([50.0, -100.0])))
        with self.assertRaises(StatisticsError):
            relative_improvement(np.array([1.0]), np.array([0.0]))

    def test_win_loss_tie_uses_exact_zero(self):
        counts = win_loss_tie(np.array([1.0, -2.0, 0.0, 3.0]))
        self.assertEqual((counts.wins, counts.losses, counts.ties), (2, 1, 1))


class PairingTests(unittest.TestCase):
    def test_conditions_pair_by_dataset_and_horizon_not_by_row_order(self):
        rows = []
        for dataset, horizon in reversed(CONDITIONS):
            for seed in (46, 45, 44, 43, 42):
                rows.append(
                    {
                        "分支": "F4",
                        "协议": "main",
                        "数据集": dataset,
                        "预测步长": horizon,
                        "种子": seed,
                        "测试RMSE": float(horizon + seed / 100.0 + (dataset == "D6")),
                    }
                )
        shuffled = pd.DataFrame(rows).sample(frac=1.0, random_state=0)
        observed = condition_vector(shuffled, "F4", "main", "测试RMSE")
        expected = []
        for dataset, horizon in CONDITIONS:
            seed_errors = [horizon + seed / 100.0 + (dataset == "D6") for seed in (42, 43, 44, 45, 46)]
            expected.append(sum(seed_errors) / 5.0)
        np.testing.assert_allclose(observed, np.array(expected))

    def test_condition_mean_averages_five_seeds(self):
        means = condition_mean(np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 10.0, 10.0, 10.0, 10.0]]))
        np.testing.assert_allclose(means, np.array([3.0, 10.0]))
        with self.assertRaises(StatisticsError):
            condition_mean(np.array([[1.0, 2.0, 3.0]]))


class WilcoxonTests(unittest.TestCase):
    def test_all_positive_ranks_match_the_exact_two_sided_count(self):
        # Absolute ranks are 1, 2, 3, 4. Every sign is positive, so the smaller
        # rank sum is 0. Two of the 16 sign assignments are at least this extreme.
        result = paired_wilcoxon(np.array([1.0, 2.0, 3.0, 4.0]))
        self.assertEqual(result.statistic, 0.0)
        self.assertEqual(result.pvalue, 0.125)
        self.assertEqual(result.method, "exact")
        self.assertEqual(result.n, 4)
        self.assertEqual(result.n_zero_discarded, 0)

    def test_a_zero_is_discarded_and_selects_the_approximation(self):
        result = paired_wilcoxon(np.array([1.0, 0.0, -2.0, 3.0]))
        self.assertEqual(result.method, "approx")
        self.assertEqual(result.n, 3)
        self.assertEqual(result.n_zero_discarded, 1)
        self.assertTrue(math.isfinite(result.pvalue))


class BootstrapTests(unittest.TestCase):
    def test_small_percentile_interval_matches_the_recorded_draws(self):
        # default_rng(20261001).integers(0, 4, size=(4, 4)) equals
        # [[0, 2, 1, 3], [0, 0, 1, 2], [2, 1, 1, 2], [1, 0, 2, 1]].
        # Means of [10, -5, 0, 20] under those draws are 6.25, 3.75, -2.5, 0.
        # Linear 2.5 and 97.5 percentiles of that sample are -2.3125 and 6.0625.
        interval = percentile_bootstrap(np.array([10.0, -5.0, 0.0, 20.0]), n_resamples=4, seed=20261001)
        self.assertEqual(interval.low, -2.3125)
        self.assertEqual(interval.high, 6.0625)
        self.assertEqual(interval.seed, 20261001)
        again = percentile_bootstrap(np.array([10.0, -5.0, 0.0, 20.0]), n_resamples=4, seed=20261001)
        self.assertEqual((interval.low, interval.high), (again.low, again.high))


class HolmTests(unittest.TestCase):
    def test_step_down_adjustment_and_the_unit_cap(self):
        # Sorted p-values 0.01, 0.03, 0.04 become 0.03, 0.06, 0.06.
        adjusted = holm(np.array([0.04, 0.01, 0.03]))
        np.testing.assert_allclose(adjusted, np.array([0.06, 0.03, 0.06]))
        # 2*0.7 = 1.4 is capped at 1, then monotonicity keeps the larger raw p at 1.
        capped = holm(np.array([0.8, 0.7]))
        np.testing.assert_allclose(capped, np.array([1.0, 1.0]))

    def test_tied_p_values_stay_in_input_order_before_monotonicity(self):
        adjusted = holm(np.array([0.02, 0.02]))
        np.testing.assert_allclose(adjusted, np.array([0.04, 0.04]))

    def test_holm_does_not_cross_families_or_unadjusted_rows(self):
        frame = pd.DataFrame(
            {
                "Holm族": ["A", "A", None, "B"],
                "指标": ["RMSE", "RMSE", "RMSE", "RMSE"],
                "Wilcoxon_p": [0.01, 0.04, 0.01, 0.01],
            }
        )
        adjusted = assign_holm(frame)
        # Family A has two tests: 2*0.01=0.02 and 1*0.04=0.04.
        # Family B has one test, so its adjusted p stays 0.01.
        # The unassigned row is not multiplied by either family.
        np.testing.assert_allclose(adjusted.loc[0, "Holm_p"], 0.02)
        np.testing.assert_allclose(adjusted.loc[1, "Holm_p"], 0.04)
        self.assertTrue(math.isnan(adjusted.loc[2, "Holm_p"]))
        np.testing.assert_allclose(adjusted.loc[3, "Holm_p"], 0.01)


class FormulaTests(unittest.TestCase):
    def test_clip_and_smape_match_the_hand_formula(self):
        rmse, smape = original_sales_scores(np.array([0.0, 2.0, -1.0]), np.array([0.0, 0.0, 1.0]))
        self.assertEqual(rmse, math.sqrt(5.0 / 3.0))
        term_zero = 0.0
        term_two = 4.0 / (2.0 + 1e-8)
        term_one = 2.0 / (1.0 + 1e-8)
        self.assertEqual(smape, 100.0 * (term_zero + term_two + term_one) / 3.0)

    def test_five_number_summary(self):
        summary = five_number(np.array([1.0, 2.0, 3.0, 4.0]))
        self.assertEqual(summary["min"], 1.0)
        self.assertEqual(summary["max"], 4.0)
        self.assertEqual(summary["median"], 2.5)


if __name__ == "__main__":
    unittest.main()
