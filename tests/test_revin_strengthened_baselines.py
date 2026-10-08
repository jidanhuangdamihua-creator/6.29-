from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.baselines.baseline_data_loader import _build_entity_slice
from scripts.baselines.bl6_dlinear import fit_bl6, predict_bl6
from scripts.baselines.bl6_dlinear_revin import fit_bl6_revin, predict_bl6_revin
from scripts.baselines.bl7_tcn import fit_bl7, predict_bl7
from scripts.baselines.bl7_tcn_revin import fit_bl7_revin, predict_bl7_revin
from scripts.baselines.revin_strengthened_protocol import (
    DEFAULT_OUTPUT_DIR,
    EXPECTED_CELLS,
    FORMAL_BASELINE_OUTPUT_DIR,
    PAPER_HORIZONS,
    PAPER_METHODS,
    PAPER_SEEDS,
    PAPER_TARGETS,
    ROBUSTNESS_METHOD_IDS,
    aggregate_five_seed,
    annotate_robustness_rows,
    assert_identity_isolation,
    assert_isolated_output_dir,
    cell_completeness,
    expected_cell_keys,
    paper_method_name,
    relative_error_change,
    relative_improvement,
    filter_paper_entities,
)
from scripts.baselines.run_baselines_multiseed import (
    METHODS,
    ROBUSTNESS_METHODS,
    evaluate_entity_protocol,
    run_dataset,
)
from src.evaluation.metrics import compute_original_scale_metrics
from src.transfer_methods.revin_torch import (
    DEFAULT_REVIN_AFFINE,
    DEFAULT_REVIN_EPS,
    DEFAULT_REVIN_TARGET_CHANNEL,
    TorchRevIN,
    assert_matches_official_revin_contract,
)
from tools.build_paper_results_master_v1_2 import norm_method, validate_external_baseline_rows
from tools.promote_bl6_bl7_formal_results import FORMAL_TARGETS, METHODS as PROMOTED_METHODS


def _series(n: int = 30, start: float = 40.0) -> np.ndarray:
    return start + np.linspace(0.0, n - 1.0, n)


def _entity() -> dict:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=55, freq="D"),
            "sales": np.linspace(80.0, 630.0, 55),
            "store_id": "1",
            "item_id": "10",
        }
    )
    return _build_entity_slice(frame, "d1", "1/10", entity_values=("1", "10"))


def test_paper_matrix_matches_freeze_authority_and_excludes_retired_d4_targets() -> None:
    assert PAPER_TARGETS == FORMAL_TARGETS
    assert PAPER_HORIZONS == (1, 3, 5)
    assert PAPER_SEEDS == (42, 43, 44, 45, 46)
    assert EXPECTED_CELLS == 120
    assert "166/311" not in PAPER_TARGETS.values()
    assert "166/313" not in PAPER_TARGETS.values()
    assert len(expected_cell_keys()) == 120


def test_formal_dlinear_and_tcn_signatures_were_not_extended() -> None:
    assert "revin_enabled" not in inspect.signature(fit_bl6).parameters
    assert "revin_enabled" not in inspect.signature(fit_bl7).parameters
    assert METHODS[-2:] == ("BL6_DLinear", "BL7_TCN")
    assert "BL6_DLinear_RevIN" not in METHODS
    assert "BL7_TCN_RevIN" not in METHODS
    assert ROBUSTNESS_METHODS == ("BL6_DLinear_RevIN", "BL7_TCN_RevIN")
    assert set(PROMOTED_METHODS) == {"BL6_DLinear", "BL7_TCN"}


def test_a_revin_roundtrip_matches_official_sita_layer() -> None:
    import tf_compat  # noqa: F401
    tf = pytest.importorskip("tensorflow")
    from src.transfer_methods.revin import RevIN

    official = assert_matches_official_revin_contract()
    assert official["eps"] == pytest.approx(DEFAULT_REVIN_EPS)
    assert official["affine"] is DEFAULT_REVIN_AFFINE
    assert official["target_channel"] == DEFAULT_REVIN_TARGET_CHANNEL
    rng = np.random.default_rng(0)
    x = rng.normal(loc=12.0, scale=3.0, size=(4, 10, 1)).astype(np.float32)
    torch_layer = TorchRevIN(num_features=1, affine=True)
    tf_layer = RevIN(num_features=1, affine=True)
    x_norm_t, mean_t, std_t = torch_layer.normalize_numpy(x)
    x_norm_f, mean_f, std_f = tf_layer.normalize_numpy(x)
    np.testing.assert_allclose(x_norm_t, x_norm_f, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(mean_t, mean_f, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(std_t, std_f, rtol=1e-5, atol=1e-5)
    recovered = torch_layer.denormalize_sequence(x_norm_t, mean_t, std_t).detach().cpu().numpy()
    np.testing.assert_allclose(recovered, x, rtol=1e-5, atol=1e-5)
    y = torch_layer.denormalize_numpy(np.zeros((4, 1), dtype=np.float32), mean_t, std_t)
    y_tf = tf_layer.denormalize_numpy(np.zeros((4, 1), dtype=np.float32), mean_f, std_f)
    np.testing.assert_allclose(y, y_tf, rtol=1e-5, atol=1e-5)
    assert tf is not None


def test_b_predictions_and_labels_stay_in_original_sales_space() -> None:
    train = _series(15, start=100.0)
    validation = _series(15, start=115.0)
    lookback = np.linspace(120.0, 129.0, 10)
    fitted = fit_bl6_revin(train, validation, horizon=1, seed=42, revin_enabled=True)
    prediction = predict_bl6_revin(fitted, lookback)
    assert any("affine_weight" in name for name in fitted.model.state_dict())
    assert np.isfinite(prediction)
    assert prediction > 50.0
    metrics = compute_original_scale_metrics(np.asarray([lookback[-1]]), np.asarray([prediction]))
    assert metrics["primary_metric_space"] == "original_sales"
    assert 0.0 <= metrics["smape"] <= 200.0


def test_c_revin_statistics_do_not_use_future_targets() -> None:
    lookback = np.linspace(10.0, 19.0, 10, dtype=np.float32).reshape(1, 10, 1)
    future = np.asarray([999.0, 1000.0], dtype=np.float32)
    layer = TorchRevIN(num_features=1, affine=False)
    _, mean, stdev = layer.normalize_numpy(lookback)
    mutated = lookback.copy()
    # Future labels sit outside the window on purpose; stats must ignore them.
    _, mean_future, stdev_future = layer.normalize_numpy(lookback)
    np.testing.assert_allclose(mean, mean_future)
    np.testing.assert_allclose(stdev, stdev_future)
    assert float(np.max(np.abs(mean - future.mean()))) > 1.0
    mutated[:, -1, 0] = 80.0
    _, mean_changed, _ = layer.normalize_numpy(mutated)
    assert abs(float(mean_changed.reshape(-1)[0]) - float(mean.reshape(-1)[0])) > 1e-6


def test_d_revin_disabled_matches_formal_parent_implementation() -> None:
    train = _series(15)
    validation = _series(15, start=15.0)
    lookback = np.arange(10.0)
    dlinear_off = fit_bl6_revin(train, validation, horizon=1, seed=42, revin_enabled=False)
    dlinear_formal = fit_bl6(train, validation, horizon=1, seed=42)
    tcn_off = fit_bl7_revin(train, validation, horizon=1, seed=42, revin_enabled=False)
    tcn_formal = fit_bl7(train, validation, horizon=1, seed=42)
    dlinear_delta = abs(predict_bl6_revin(dlinear_off, lookback) - predict_bl6(dlinear_formal, lookback))
    tcn_delta = abs(predict_bl7_revin(tcn_off, lookback) - predict_bl7(tcn_formal, lookback))
    assert dlinear_delta == pytest.approx(0.0, abs=1e-12)
    assert tcn_delta == pytest.approx(0.0, abs=1e-12)


def test_e_identity_isolation_from_formal_dlinear_and_tcn() -> None:
    assert paper_method_name("BL6_DLinear_RevIN") == "DLinear_RevIN"
    assert paper_method_name("BL7_TCN_RevIN") == "TCN_RevIN"
    assert_identity_isolation(PAPER_METHODS)
    assert_identity_isolation(ROBUSTNESS_METHOD_IDS)
    assert norm_method("BL6_DLinear_RevIN") != "DLinear"
    assert norm_method("BL7_TCN_RevIN") != "TCN"
    assert norm_method("DLinear_RevIN") == "DLinear_RevIN"
    with pytest.raises(ValueError, match="collided with formal identities"):
        assert_identity_isolation(["DLinear"])
    with pytest.raises(ValueError, match="formal baseline directory"):
        assert_isolated_output_dir(FORMAL_BASELINE_OUTPUT_DIR)
    assert DEFAULT_OUTPUT_DIR != FORMAL_BASELINE_OUTPUT_DIR
    fake = pd.DataFrame(
        [
            {
                "dataset_id": dataset,
                "target_entity_key": target,
                "method": "DLinear_RevIN",
                "horizon": 1,
                "seed": 42,
                "rmse": 1.0,
                "smape": 2.0,
                "result_status": "confirmed_baseline",
                "metric_space": "original_sales_space",
            }
            for dataset, target in PAPER_TARGETS.items()
        ]
    )
    with pytest.raises(ValueError, match="unsupported methods"):
        validate_external_baseline_rows(fake)


def test_f_cell_completeness_contract_is_120() -> None:
    rows = []
    for dataset, target in PAPER_TARGETS.items():
        for method in PAPER_METHODS:
            for horizon in PAPER_HORIZONS:
                for seed in PAPER_SEEDS:
                    rows.append(
                        {
                            "dataset": dataset,
                            "target": target,
                            "mode": "without",
                            "method": method,
                            "horizon": horizon,
                            "seed": seed,
                            "rmse": 1.0,
                            "smape": 2.0,
                        }
                    )
    complete = cell_completeness(pd.DataFrame(rows))
    assert complete["EXPECTED_CELLS"] == 120
    assert complete["ACTUAL_CELLS"] == 120
    assert complete["DUPLICATE_CELLS"] == 0
    assert complete["MISSING_CELLS"] == 0
    assert complete["complete"] is True
    incomplete = cell_completeness(pd.DataFrame(rows[:-1]))
    assert incomplete["EXPECTED_CELLS"] == 120
    assert incomplete["ACTUAL_CELLS"] == 119
    assert incomplete["MISSING_CELLS"] == 1
    assert incomplete["complete"] is False
    aggregated = aggregate_five_seed(pd.DataFrame(rows))
    assert len(aggregated) == 24
    assert set(aggregated["n_seeds"]) == {5}


def test_relative_improvement_uses_baseline_error_denominator() -> None:
    assert relative_improvement(20.0, 15.0) == pytest.approx(25.0)
    assert relative_improvement(20.0, 25.0) == pytest.approx(-25.0)


def test_evaluate_entity_protocol_accepts_revin_methods_without_renaming_formal_rows() -> None:
    data = _entity()
    summary, raw = evaluate_entity_protocol(
        data,
        methods=("BL6_DLinear_RevIN",),
        horizons=(1,),
        seeds=(42,),
        return_raw=True,
        run_id="revin-isolation",
    )
    assert set(summary["method"]) == {"BL6_DLinear_RevIN"}
    assert set(raw["method"]) == {"BL6_DLinear_RevIN"}
    assert set(summary["metric_space"]) == {"original_sales_space"}
    assert np.isfinite(raw[["y_true", "y_pred"]].to_numpy(dtype=float)).all()
    y_true = raw["y_true"].to_numpy(dtype=float)
    y_pred = raw["y_pred"].to_numpy(dtype=float)
    assert y_true.max() > 10.0
    assert y_pred.max() > 10.0
    annotated = annotate_robustness_rows(summary, prediction_file="unused.csv")
    assert set(annotated["method"]) == {"DLinear_RevIN"}
    assert set(annotated["parent_method"]) == {"DLinear"}
    assert set(annotated["robustness_variant"]) == {"RevIN"}
    payload = json.loads(annotated.iloc[0]["revin_configuration"])
    assert payload["enabled"] is True
    assert payload["affine"] is True
    assert payload["eps"] == pytest.approx(1e-5)


def test_run_dataset_rejects_formal_output_dir_for_revin_methods(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="formal baseline output directory"):
        run_dataset(
            "d1",
            seeds=(42,),
            methods=("BL6_DLinear_RevIN",),
            horizons=(1,),
            output_dir=FORMAL_BASELINE_OUTPUT_DIR,
        )
    with pytest.raises(ValueError, match="isolated output_dir"):
        run_dataset(
            "d1",
            seeds=(42,),
            methods=("BL6_DLinear_RevIN",),
            horizons=(1,),
        )
    assert tmp_path.exists()


def test_filter_excludes_retired_d4_targets() -> None:
    slices = [
        {"dataset_id": "d4", "entity_key": "166/311"},
        {"dataset_id": "d4", "entity_key": "166/313"},
        {"dataset_id": "d4", "entity_key": "166/258"},
        {"dataset_id": "d4", "entity_key": "166/432"},
    ]
    selected = filter_paper_entities(slices)
    assert [item["entity_key"] for item in selected] == ["166/258"]


def test_comparison_math_uses_error_denominator_not_best_seed() -> None:
    from tools.compare_revin_strengthened_baselines import build_comparison

    rows = []
    condition_index = 0
    values = {
        "SITA": (1.0, 10.0),
        "DLinear": (4.0, 40.0),
        "TCN": (5.0, 50.0),
        "DLinear_RevIN": (2.0, 20.0),
        "TCN_RevIN": (3.0, 30.0),
    }
    for dataset, target in PAPER_TARGETS.items():
        for horizon in PAPER_HORIZONS:
            bump = 0.01 * condition_index
            condition_index += 1
            for seed in PAPER_SEEDS:
                for method, (rmse, smape) in values.items():
                    rows.append(
                        {
                            "dataset": dataset,
                            "target": target,
                            "mode": "without",
                            "method": method,
                            "horizon": horizon,
                            "seed": seed,
                            "rmse": rmse + bump,
                            "smape": smape + bump,
                        }
                    )
    frame = pd.DataFrame(rows)
    frozen = frame[frame["method"].isin(("SITA", "DLinear", "TCN"))].copy()
    strengthened = frame[frame["method"].isin(("DLinear_RevIN", "TCN_RevIN"))].copy()
    effect, versus, stats = build_comparison(frozen, strengthened)
    dlinear_effect = effect.loc[effect["parent_method"] == "DLinear"].iloc[0]
    assert dlinear_effect["rmse_absolute_difference"] == pytest.approx(-2.0)
    assert dlinear_effect["rmse_relative_change_pct"] == pytest.approx(relative_error_change(4.0, 2.0))
    sita_vs_dlinear_revin = versus.loc[
        (versus["strengthened_method"] == "DLinear_RevIN") & (versus["metric"] == "RMSE")
    ].iloc[0]
    assert sita_vs_dlinear_revin["sita_vs_strengthened_relative_improvement_pct"] == pytest.approx(50.0)
    assert bool(sita_vs_dlinear_revin["sita_still_better_than_strengthened"])
    assert set(stats["sita_wins"]) == {12}
