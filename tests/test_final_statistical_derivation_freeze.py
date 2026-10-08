from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from tools.build_final_statistical_derivation_freeze import (
    ABLATION_CONTRASTS,
    ABLATION_ARMS,
    BOOTSTRAP_CI_LEVEL,
    BOOTSTRAP_RANDOM_SEED,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_TYPE,
    FORMAL_CONDITIONS,
    FORMAL_HORIZONS,
    FORMAL_METHODS,
    FORMAL_SEEDS,
    HOLM_ALPHA,
    HOLM_FAMILY_SIZE,
    LEARNING_POOL,
    STATISTICAL_UNIT,
    WILCOXON_ALTERNATIVE,
    WILCOXON_METHOD,
    best_learning_method,
    bootstrap_mean_ci,
    derive_condition_means,
    holm_adjust,
    rank_methods,
    relative_improvement_pct,
    wilcoxon_exact,
)


def test_protocol_constants_lock_the_final_contract():
    assert BOOTSTRAP_RESAMPLES == 100000
    assert BOOTSTRAP_RANDOM_SEED == 20260903
    assert BOOTSTRAP_CI_LEVEL == 0.95
    assert BOOTSTRAP_TYPE == "percentile"
    assert WILCOXON_ALTERNATIVE == "two-sided"
    assert WILCOXON_METHOD == "exact"
    assert HOLM_ALPHA == 0.05
    assert FORMAL_SEEDS == (42, 43, 44, 45, 46)
    assert FORMAL_HORIZONS == (1, 3, 5)
    assert len(FORMAL_CONDITIONS) == 12
    assert STATISTICAL_UNIT == "12_DATASET_HORIZON_CONDITION_MEANS"
    assert LEARNING_POOL == ("LSTM", "LightGBM", "DLinear", "TCN")
    assert FORMAL_METHODS == (
        "SITA",
        "LSTM",
        "LightGBM",
        "DLinear",
        "TCN",
        "Seasonal Naive",
    )
    assert ABLATION_ARMS == ("A", "B", "C", "D", "E")
    assert ABLATION_CONTRASTS == (("C", "A"), ("C", "B"), ("C", "E"), ("D", "C"))
    assert HOLM_FAMILY_SIZE == 4


def test_condition_means_are_five_seed_arithmetic_means():
    rows = [
        {"dataset": "D1", "target": "1/10", "horizon": 1, "seed": seed, "method": "SITA", "rmse": float(seed)}
        for seed in FORMAL_SEEDS
    ]
    result = derive_condition_means(rows, value_columns=("rmse",))
    assert len(result) == 1
    assert result[0]["n_seeds"] == 5
    assert result[0]["rmse"] == pytest.approx(44.0)


def test_relative_improvement_is_lower_is_better_reference_denominator():
    assert relative_improvement_pct(reference=20.0, treatment=15.0) == pytest.approx(25.0)
    assert relative_improvement_pct(reference=20.0, treatment=25.0) == pytest.approx(-25.0)
    with pytest.raises(ValueError, match="reference must be positive"):
        relative_improvement_pct(reference=0.0, treatment=1.0)


def test_wilcoxon_is_explicit_two_sided_exact_and_reports_zero_tie_counts():
    result = wilcoxon_exact([1.0, 2.0, 3.0, 4.0])
    assert result["method"] == "exact"
    assert result["alternative"] == "two-sided"
    assert result["zero_difference_count"] == 0
    assert result["tied_absolute_difference_count"] == 0
    assert result["statistic"] == pytest.approx(0.0)
    assert result["p_value"] == pytest.approx(0.125)

    with pytest.raises(ValueError, match="zero or tied absolute differences"):
        wilcoxon_exact([1.0, 1.0, 2.0, 3.0])


def test_bootstrap_is_reproducible_and_uses_condition_vector_length():
    values = np.arange(1.0, 13.0)
    first = bootstrap_mean_ci(values, comparison_key="main|RMSE|SITA_vs_LSTM")
    second = bootstrap_mean_ci(values, comparison_key="main|RMSE|SITA_vs_LSTM")
    assert first == second
    assert first[0] < 6.5 < first[1]
    with pytest.raises(ValueError, match="exactly 12"):
        bootstrap_mean_ci([1.0, 2.0], comparison_key="bad")


def test_holm_is_step_down_deterministic_and_separate_family():
    result = holm_adjust(
        {
            "C_vs_A": 0.001,
            "C_vs_B": 0.04,
            "C_vs_E": 0.02,
            "D_vs_C": 0.8,
        }
    )
    assert list(result) == ["C_vs_A", "C_vs_B", "C_vs_E", "D_vs_C"]
    assert result["C_vs_A"]["rank"] == 1
    assert result["C_vs_A"]["multiplier"] == 4
    assert result["C_vs_E"]["rank"] == 2
    assert result["D_vs_C"]["adjusted_p"] == pytest.approx(0.8)


def test_best_learning_is_selected_separately_by_metric():
    values = {
        "LSTM": {"RMSE": 2.0, "sMAPE": 5.0},
        "LightGBM": {"RMSE": 3.0, "sMAPE": 1.0},
        "DLinear": {"RMSE": 4.0, "sMAPE": 4.0},
        "TCN": {"RMSE": 5.0, "sMAPE": 3.0},
        "Seasonal Naive": {"RMSE": 0.1, "sMAPE": 0.1},
    }
    assert best_learning_method(values, metric="RMSE") == "LSTM"
    assert best_learning_method(values, metric="sMAPE") == "LightGBM"


def test_ranking_uses_exact_values_and_reports_strict_rank_one_counts():
    values = {
        "SITA": 1.0,
        "LSTM": 2.0,
        "LightGBM": 3.0,
        "DLinear": 4.0,
        "TCN": 5.0,
        "Seasonal Naive": 6.0,
    }
    result = rank_methods(values)
    assert result["tie_rule"] == "exact_numeric_no_tolerance"
    assert result["ties_observed"] == 0
    assert result["ranks"]["SITA"] == 1
    assert result["strict_rank_one_method"] == "SITA"
