from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from scripts.baselines.bl4_lstm import _direct_h_arrays, _scale_partitions
from scripts.baselines.bl9_patchtst import (
    PATCHTST_CONFIG,
    assert_patchtst_config,
    compute_patch_num,
    fit_bl9,
    patchtst_forward_probe,
    predict_bl9,
    target_channel_index,
)
from scripts.baselines.bl10_timesnet import (
    TIMESNET_CONFIG,
    assert_timesnet_config,
    effective_top_k,
    fit_bl10,
    predict_bl10,
    rfft_nonzero_frequency_bins,
    timesnet_forward_probe,
)
from scripts.baselines.bl4_lstm import LEARNING_RATE, MAX_EPOCHS, PATIENCE
from scripts.bl9_bl10_formal_protocol import (
    FEATURE_METADATA,
    FORMAL_BASELINE_TRAINING_POLICY,
    HOLM_FAMILY,
    LOOKBACK,
    TARGETS,
    TARGET_COLUMN,
    expected_cell_keys,
    recompute_original_metrics,
    relative_improvement_pct,
    training_policy_report,
)
from src.evaluation.metrics import compute_original_scale_metrics


def test_patchtst_short_window_config_is_legal() -> None:
    config = assert_patchtst_config()
    assert config["d_model"] % config["n_heads"] == 0
    assert compute_patch_num() == 5
    assert config["patch_num"] == 5
    assert config["target_window"] == 1
    assert config["context_window"] == LOOKBACK
    assert PATCHTST_CONFIG["patch_len"] == 4


def test_timesnet_fft_bins_support_top_k() -> None:
    config = assert_timesnet_config()
    assert rfft_nonzero_frequency_bins(10) >= 3
    assert rfft_nonzero_frequency_bins(11) >= 3
    assert effective_top_k(10, 3, extra_len=1) == 3
    assert config["effective_top_k"] == 3
    assert config["pred_len"] == 1


def test_direct_h_label_alignment_is_single_offset() -> None:
    values = np.arange(15.0)
    x, y = _direct_h_arrays(values, horizon=3, lookback=10, name="train")
    assert x.shape == (3, 10, 1)
    assert y.shape == (3, 1)
    np.testing.assert_allclose(y.reshape(-1), [12.0, 13.0, 14.0])
    x5, y5 = _direct_h_arrays(values, horizon=5, lookback=10, name="train")
    assert x5.shape == (1, 10, 1)
    np.testing.assert_allclose(y5.reshape(-1), [14.0])


def test_patchtst_and_timesnet_forward_are_scalar_direct_h() -> None:
    patch_out, patch_config = patchtst_forward_probe()
    times_out, times_config, top_k = timesnet_forward_probe()
    assert tuple(patch_out.shape) == (2, 1, 1)
    assert tuple(times_out.shape) == (2, 1, 1)
    assert int(patch_config["target_window"]) == 1
    assert int(times_config["pred_len"]) == 1
    assert int(top_k) == 3
    assert np.isfinite(patch_out.detach().cpu().numpy()).all()
    assert np.isfinite(times_out.detach().cpu().numpy()).all()


@pytest.mark.parametrize("fit,predict", [(fit_bl9, predict_bl9), (fit_bl10, predict_bl10)])
def test_learning_baseline_contract_scaler_and_finite_loss(fit, predict) -> None:
    train = np.linspace(1.0, 15.0, 15)
    validation = np.linspace(16.0, 30.0, 15)
    fitted = fit(train, validation, horizon=1, seed=42)
    prediction = predict(fitted, train[-10:])
    assert np.isfinite(prediction)
    assert fitted.scaler.n_samples_seen_ == 15
    assert float(fitted.scaler.data_min_[0]) == pytest.approx(1.0)
    assert float(fitted.scaler.data_max_[0]) == pytest.approx(15.0)
    assert fitted.training_history
    assert np.isfinite([row["train_loss"] for row in fitted.training_history]).all()
    assert fitted.lookback == 10
    assert fitted.target_channel_index == 0


@pytest.mark.parametrize("fit,predict", [(fit_bl9, predict_bl9), (fit_bl10, predict_bl10)])
def test_constant_series_stays_in_original_sales_space(fit, predict) -> None:
    fitted = fit([4.0] * 15, [9.0] * 15, horizon=5, seed=42)
    assert predict(fitted, [4.0] * 10) == pytest.approx(4.0)


def test_independent_h1_h3_h5_models_are_not_shared() -> None:
    train = np.linspace(2.0, 16.0, 15)
    validation = np.linspace(17.0, 31.0, 15)
    h1 = fit_bl9(train, validation, horizon=1, seed=42)
    h3 = fit_bl9(train, validation, horizon=3, seed=42)
    h5 = fit_bl9(train, validation, horizon=5, seed=42)
    assert h1.horizon != h3.horizon != h5.horizon
    assert h1.model is not h3.model


def test_source_and_future_covariates_are_absent_from_fit_signature() -> None:
    for fn in (fit_bl9, fit_bl10):
        names = set(inspect.signature(fn).parameters)
        forbidden = {
            "source",
            "knn",
            "sita",
            "onpromotion",
            "transactions",
            "source_weights",
            "future_sales",
        }
        assert names.isdisjoint(forbidden)
        assert "train_sales" in names
        assert "validation_sales" in names


def test_metric_authority_is_reused_and_already_percent() -> None:
    metrics = compute_original_scale_metrics([1.0, 3.0], [2.0, 1.0])
    recomputed = recompute_original_metrics([1.0, 3.0], [2.0, 1.0])
    assert metrics["smape"] == pytest.approx(recomputed["smape"])
    assert 0.0 <= metrics["smape"] <= 200.0
    assert recomputed["smape_unit"] == "percent"


def test_training_policy_matches_formal_learning_baselines() -> None:
    report = training_policy_report()
    assert report["BL9_POLICY_MATCH"] == "YES"
    assert report["BL10_POLICY_MATCH"] == "YES"
    assert FORMAL_BASELINE_TRAINING_POLICY["optimizer"] == "Adam"
    assert FORMAL_BASELINE_TRAINING_POLICY["learning_rate"] == LEARNING_RATE
    assert FORMAL_BASELINE_TRAINING_POLICY["epochs"] == MAX_EPOCHS
    assert FORMAL_BASELINE_TRAINING_POLICY["early_stopping_patience"] == PATIENCE
    assert FORMAL_BASELINE_TRAINING_POLICY["batch"] == "full_train_tensor"
    assert FORMAL_BASELINE_TRAINING_POLICY["loss"] == "MSELoss"
    assert target_channel_index(FEATURE_METADATA) == 0
    assert TARGET_COLUMN == "sales"
    assert HOLM_FAMILY == "BL9_BL10_SITA_PAIRWISE"
    assert TARGETS["D4"] == "166/258"
    assert "166/311" not in TARGETS.values()


def test_relative_improvement_definition() -> None:
    assert relative_improvement_pct(10.0, 8.0) == pytest.approx(20.0)


def test_expected_matrix_is_120_unique_cells() -> None:
    keys = expected_cell_keys()
    assert len(keys) == 120
    assert len(set(keys)) == 120


def test_scaler_fit_does_not_see_validation_range() -> None:
    train = np.arange(15.0)
    validation = np.arange(100.0, 115.0)
    train_scaled, validation_scaled, scaler = _scale_partitions(train, validation)
    assert scaler.n_samples_seen_ == 15
    assert float(scaler.data_max_[0]) == 14.0
    assert validation_scaled.max() > 1.0
