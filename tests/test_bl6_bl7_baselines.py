from __future__ import annotations

import json
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.baselines.bl6_dlinear import (
    DLINEAR_KERNEL_SIZE,
    _direct_h_arrays,
    fit_bl6,
    predict_bl6,
)
from scripts.baselines.bl7_tcn import (
    TCN_CHANNELS,
    TCN_KERNEL_SIZE,
    TCN_LEVELS,
    TCNResidualBlock,
    fit_bl7,
    predict_bl7,
)
from scripts.baselines.baseline_data_loader import _build_entity_slice
from scripts.baselines.run_baselines_multiseed import (
    METHODS,
    build_run_id,
    evaluate_entity_protocol,
    parse_selector,
    publish_baseline_run,
)
from scripts.aggregate_d1_d6_results import _read_source
from src.evaluation.metrics import compute_original_scale_metrics


def _data() -> dict:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=55, freq="D"),
            "sales": np.linspace(10.0, 550.0, 55),
            "store_id": "1",
            "item_id": "10",
        }
    )
    return _build_entity_slice(frame, "d1", "1_10", entity_values=("1", "10"))


def test_bl6_and_bl7_are_registered_with_fixed_architecture() -> None:
    assert METHODS[-2:] == ("BL6_DLinear", "BL7_TCN")
    assert DLINEAR_KERNEL_SIZE == 3
    assert TCN_CHANNELS == 16
    assert TCN_LEVELS == 2
    assert TCN_KERNEL_SIZE == 3


def test_bl6_direct_h_arrays_and_fit_use_expected_shape() -> None:
    x, y = _direct_h_arrays(np.arange(25.0), horizon=3, lookback=10, name="train")
    assert x.shape == (13, 10, 1)
    assert y.shape == (13, 1)
    fitted = fit_bl6(np.arange(15.0), np.arange(15.0, 30.0), horizon=1, seed=42)
    prediction = predict_bl6(fitted, np.arange(10.0))
    assert np.isfinite(prediction)


def test_bl7_has_causal_residual_blocks_and_expected_shape() -> None:
    block = TCNResidualBlock(in_channels=1, out_channels=TCN_CHANNELS, kernel_size=3, dilation=2)
    assert block.causal is True
    fitted = fit_bl7(np.arange(15.0), np.arange(15.0, 30.0), horizon=5, seed=42)
    prediction = predict_bl7(fitted, np.arange(10.0))
    assert np.isfinite(prediction)


def test_bl7_residual_block_cannot_propagate_future_to_past() -> None:
    torch = importlib.import_module("torch")
    block = TCNResidualBlock(in_channels=1, out_channels=TCN_CHANNELS, kernel_size=3, dilation=2)
    block.eval()
    first = torch.zeros((1, 1, 10), dtype=torch.float32)
    second = first.clone()
    second[:, :, 8:] = 100.0
    with torch.no_grad():
        first_output = block(first)
        second_output = block(second)
    torch.testing.assert_close(first_output[:, :, :8], second_output[:, :, :8])


@pytest.mark.parametrize("fit,predict", [(fit_bl6, predict_bl6), (fit_bl7, predict_bl7)])
def test_learning_baseline_same_seed_is_deterministic(fit, predict) -> None:
    first = fit(np.arange(15.0), np.arange(15.0, 30.0), horizon=1, seed=42)
    second = fit(np.arange(15.0), np.arange(15.0, 30.0), horizon=1, seed=42)
    assert predict(first, np.arange(10.0)) == pytest.approx(
        predict(second, np.arange(10.0)), abs=1e-8
    )


@pytest.mark.parametrize("fit", [fit_bl6, fit_bl7])
def test_learning_baseline_fits_scaler_on_train_only_and_rejects_nonfinite(fit) -> None:
    fitted = fit(np.arange(15.0), np.arange(100.0, 115.0), horizon=1, seed=42)
    assert fitted.scaler.n_samples_seen_ == 15
    assert float(fitted.scaler.data_min_[0]) == 0.0
    assert float(fitted.scaler.data_max_[0]) == 14.0
    with pytest.raises(ValueError, match="finite"):
        fit([1.0, np.nan, 3.0], [1.0] * 15, horizon=1, seed=42)


@pytest.mark.parametrize("fit,predict", [(fit_bl6, predict_bl6), (fit_bl7, predict_bl7)])
def test_constant_training_series_inverse_transform_is_original_sales_value(fit, predict) -> None:
    fitted = fit([5.0] * 15, [7.0] * 15, horizon=1, seed=42)
    assert predict(fitted, [5.0] * 10) == pytest.approx(5.0)


def test_runner_fits_each_target_entity_horizon_seed_method_independently(monkeypatch) -> None:
    data = _data()
    calls = []

    def fake_fit(train, validation, *, horizon, lookback, seed):
        calls.append((tuple(train), tuple(validation), horizon, lookback, seed))
        return SimpleNamespace(predict=lambda values: float(values[-1]))

    def fake_predict(model, input_sales):
        return model.predict(input_sales)

    monkeypatch.setattr("scripts.baselines.run_baselines_multiseed.fit_bl6", fake_fit)
    monkeypatch.setattr("scripts.baselines.run_baselines_multiseed.predict_bl6", fake_predict)
    rows, raw = evaluate_entity_protocol(
        data,
        methods=("BL6_DLinear",),
        horizons=(1, 3),
        seeds=(42, 43),
        return_raw=True,
        run_id="test-run",
    )
    assert len(calls) == 4
    assert len(rows) == 4
    assert len(raw) == sum(len(data["sample_manifest"].for_horizon(h)) for h in (1, 3)) * 2
    assert not raw.duplicated(
        ["target_entity_key", "method", "horizon", "seed", "sample_key"]
    ).any()
    assert set(raw["target_timestamp"]) == {
        sample.label_date
        for h in (1, 3)
        for sample in data["sample_manifest"].for_horizon(h)
    }


def test_raw_metrics_match_summary_and_selector_is_exact() -> None:
    assert parse_selector("1,3,5", allowed=(1, 2, 3, 4, 5), name="horizons") == (1, 3, 5)
    assert parse_selector("BL6_DLinear,BL7_TCN", allowed=METHODS, name="methods") == (
        "BL6_DLinear",
        "BL7_TCN",
    )
    true = np.asarray([1.0, 3.0])
    pred = np.asarray([2.0, 1.0])
    metric = compute_original_scale_metrics(true, pred)
    assert metric["rmse"] == pytest.approx(np.sqrt(2.5))
    assert metric["smape"] == pytest.approx(
        100.0 * np.mean(2.0 * np.abs(pred - true) / (np.abs(true) + np.abs(pred) + 1e-8))
    )


def test_raw_metrics_equal_summary_metrics_and_duplicate_cells_fail() -> None:
    data = _data()

    def predictor(method, record, seed):
        del method, seed
        return float(record.input_sales[-1])

    summary, raw = evaluate_entity_protocol(
        data,
        predictor=predictor,
        methods=("BL1_HistoricalMean", "BL6_DLinear", "BL7_TCN"),
        horizons=(1,),
        seeds=(42,),
        return_raw=True,
        run_id="metric-parity",
    )
    for _, row in summary.iterrows():
        group = raw[
            (raw["method"] == row["method"])
            & (raw["horizon"] == row["horizon"])
            & (raw["seed"] == row["seed"])
        ]
        metric = compute_original_scale_metrics(group["y_true"], group["y_pred"])
        assert float(row["rmse"]) == pytest.approx(metric["rmse"], abs=1e-10)
        assert float(row["smape"]) == pytest.approx(metric["smape"], abs=1e-10)
    with pytest.raises(Exception, match="duplicate baseline"):
        from scripts.baselines.run_baselines_multiseed import _assert_no_duplicate_baseline_cells

        _assert_no_duplicate_baseline_cells(pd.concat([summary, summary.iloc[[0]]]))


def test_downstream_csv_loader_accepts_bl6_and_bl7(tmp_path: Path) -> None:
    path = tmp_path / "dataset1_without.csv"
    pd.DataFrame(
        [
            {"dataset_id": "D1", "target_entity_key": "1/10", "scenario": "without",
             "method": "BL6_DLinear", "horizon": 1, "seed": 42, "rmse": 1.0, "smape": 2.0},
            {"dataset_id": "D1", "target_entity_key": "1/10", "scenario": "without",
             "method": "BL7_TCN", "horizon": 1, "seed": 42, "rmse": 1.0, "smape": 2.0},
        ]
    ).to_csv(path, index=False)
    rows = _read_source(path, 1)
    assert {row["method"] for row in rows} == {"BL6_DLinear", "BL7_TCN"}


def test_isolated_publication_rejects_existing_path_and_does_not_promote(tmp_path: Path) -> None:
    summary = pd.DataFrame(
        [{
            "dataset_id": "D1", "target_entity_key": "1/10", "scenario": "without",
            "method": "BL6_DLinear", "horizon": 1, "seed": 42,
            "rmse": 1.0, "smape": 2.0, "mae": 1.0, "accuracy": 1.0,
            "primary_metric_space": "original_sales",
            "sample_count": 1, "sample_manifest_digest": "digest",
        }]
    )
    raw = pd.DataFrame(
        [{
            "run_id": "isolated", "dataset_id": "D1", "target_entity_key": "1/10",
            "scenario": "without", "method": "BL6_DLinear", "horizon": 1,
            "seed": 42, "sample_key": "key", "forecast_origin": "2020-01-01",
            "input_start": "2019-12-23", "input_end": "2020-01-01",
            "target_timestamp": "2020-01-02", "y_true": 1.0, "y_pred": 1.0,
            "metric_space": "original_sales_space",
        }]
    )
    run_id = build_run_id("smoke", "D1", ("1/10",), ("BL6_DLinear",), (1,), (42,))
    output = tmp_path / run_id
    manifest = publish_baseline_run(summary, raw, output_dir=output, run_id=run_id, smoke=True)
    assert Path(manifest["summary_path"]).is_file()
    assert Path(manifest["raw_prediction_path"]).is_file()
    assert Path(manifest["manifest_path"]).is_file()
    with pytest.raises(FileExistsError):
        publish_baseline_run(summary, raw, output_dir=output, run_id=run_id, smoke=True)
    payload = json.loads(Path(manifest["manifest_path"]).read_text(encoding="utf-8"))
    assert payload["promotion"] == "none"
