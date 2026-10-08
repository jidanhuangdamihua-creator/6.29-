"""Unit tests for BL8 formal 60-cell audit helpers. No model training."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.bl8_msml_tl_formal_protocol import (
    BL8FormalError,
    EXPECTED_CELLS,
    MODE,
    METHOD,
    TARGETS,
    assert_allowed_matrix_cell,
    completeness_audit,
    expected_cell_keys,
    mechanism_failures,
    recompute_original_metrics,
    relative_improvement_pct,
    winner_label,
)
from src.evaluation.metrics import compute_metrics_with_protocol, smape


def test_expected_matrix_is_exactly_60_paper_cells():
    keys = expected_cell_keys()
    assert len(keys) == EXPECTED_CELLS
    assert len(set(keys)) == EXPECTED_CELLS
    assert {key[0] for key in keys} == {"D1", "D4", "D5", "D6"}
    assert {key[1] for key in keys if key[0] == "D4"} == {"166/258"}


def test_forbidden_d4_targets_are_rejected():
    with pytest.raises(BL8FormalError, match="166/311"):
        assert_allowed_matrix_cell("D4", "166/311", 1, 42, MODE, METHOD)


def test_relative_improvement_sign():
    assert relative_improvement_pct(10.0, 8.0) == pytest.approx(20.0)
    assert relative_improvement_pct(10.0, 12.0) == pytest.approx(-20.0)
    assert winner_label(10.0, 8.0) == "SITA"
    assert winner_label(8.0, 10.0) == "MSML-TL"
    assert winner_label(10.0, 10.0) == "TIE"


def test_completeness_counts_pass_fail_missing():
    rows = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "target": TARGETS[dataset],
                "horizon": horizon,
                "seed": seed,
                "mode": MODE,
                "method": METHOD,
                "cell_status": "PASS" if dataset != "D6" else "FAILED",
            }
            for dataset in ("D1", "D4", "D5", "D6")
            for horizon in (1, 3, 5)
            for seed in (42, 43, 44, 45, 46)
        ]
    )
    audit = completeness_audit(rows)
    assert audit["actual_cells"] == 45
    assert audit["failed_cells"] == 15
    assert audit["missing_cells"] == 0
    assert audit["per_dataset"]["D1"] == 15
    assert audit["per_dataset"]["D6"] == 0


def test_mechanism_failures_detect_conv3_fusion():
    failures = mechanism_failures(
        {
            "formal_bl8": True,
            "source_count_requested": 3,
            "source_count_selected": 3,
            "source_count_successful": 3,
            "source_failure_count": 0,
            "weight_mode": "inverse_distance",
            "selection_authority": "shared_protocol",
            "fused_layers": ["conv1", "conv2", "conv3"],
            "frozen_layers": ["conv1", "pool1", "conv2", "pool2"],
            "target_trainable_parameter_layers": ["conv3", "dense_out"],
            "rfe": False,
            "weight_sum": 1.0,
            "target_test_used_for_fit": False,
            "future_leakage_detected": False,
        }
    )
    assert any("conv3" in item or "fused_layers" in item for item in failures)


def test_original_metric_arrays_match_recompute():
    protocol = {
        "current_metric_space": "normalized_minmax_space",
        "paper_metric_space": "original_sales_space",
        "strict_paper_metrics": True,
        "current_accuracy_definition": "1/(RMSE+1e-8)",
        "paper_accuracy_definition": "1/(RMSE+1e-8)",
    }

    class Scaler:
        data_min_ = np.asarray([0.0])
        data_max_ = np.asarray([10.0])

    result = compute_metrics_with_protocol(
        y_true=np.asarray([0.1, 0.2, 0.3]),
        y_pred=np.asarray([0.15, 0.2, 0.25]),
        metric_protocol=protocol,
        sales_scaler=Scaler(),
        feature_columns=["sales"],
    )
    assert result["y_true_original"] is not None
    recomputed = recompute_original_metrics(result["y_true_original"], result["y_pred_original"])
    assert abs(recomputed["rmse"] - float(result["original_scale_rmse"])) <= 1e-12
    assert abs(recomputed["smape"] - float(result["original_scale_smape"])) <= 1e-12
    assert recomputed["smape"] == pytest.approx(smape(result["y_true_original"], result["y_pred_original"]))
