"""Fast unit tests for the formal BL8 MSML-TL original-paper mechanism contract."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tf_compat  # noqa: F401

from src.models.cnn_model import build_base_cnn
from src.transfer_methods import msml_tl
from src.transfer_methods.msml_tl import (
    BL8_FUSED_PARAMETER_LAYERS,
    BL8_FROZEN_TARGET_LAYERS,
    BL8_TARGET_TRAINABLE_PARAMETER_LAYERS,
    BL8_WEIGHT_EPSILON,
    FormalBL8ContractError,
    apply_bl8_target_trainability,
    compute_bl8_inverse_distance_weights,
    evaluate_msml_model,
    fine_tune_fused_target_model,
    fuse_source_models_layerwise,
    get_bl8_fused_parameter_layer_names,
    get_bl8_frozen_target_layer_names,
    get_bl8_target_trainable_parameter_layer_names,
    get_trainable_parameter_layer_names,
    get_transferable_layer_names,
    load_fused_params_into_target_model,
    run_msml_tl,
)


INPUT_SHAPE = (8, 3)
FEATURE_COLS = ("sales",)


class DummyK3Context:
    lifecycle_identity = ("bl8-unit",)

    def __init__(self, sources: Sequence[Dict[str, object]]):
        self._sources = list(sources)

    def selection_for_method(self, **_kwargs):
        sources = list(self._sources)

        class Evidence:
            def method_wrapper(self, **__kwargs):
                return {"sources": sources, "meta": {}}

        return Evidence()


def _entity_frame(entity: int, item: int, periods: int = 36, scale: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entity_id": entity,
            "item_id": item,
            "date": pd.date_range("2021-01-01", periods=periods, freq="D"),
            "sales": (np.arange(periods, dtype=np.float64) + float(item)) * scale,
        }
    )


def _tiny_pool() -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.concat(
        [
            _entity_frame(1, 1),
            _entity_frame(1, 2),
            _entity_frame(1, 3),
        ],
        ignore_index=True,
    )
    target = _entity_frame(2, 9)
    return source, target


def _three_selected_sources() -> List[Dict[str, object]]:
    return [
        {"source_key": (1, 1), "distance": 1.0, "weight": 0.5},
        {"source_key": (1, 2), "distance": 2.0, "weight": 0.3},
        {"source_key": (1, 3), "distance": 4.0, "weight": 0.2},
    ]


def _fill_layer(model, name: str, kernel_value: float, bias_value: float) -> None:
    layer = model.get_layer(name)
    filled = []
    for index, array in enumerate(layer.get_weights()):
        value = kernel_value if index == 0 else bias_value
        filled.append(np.full_like(array, value, dtype=np.float32))
    layer.set_weights(filled)


def _snapshot(model, name: str) -> List[np.ndarray]:
    return [np.array(weight, copy=True) for weight in model.get_layer(name).get_weights()]


def _max_abs_delta(left: Sequence[np.ndarray], right: Sequence[np.ndarray]) -> float:
    return float(max(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))) for a, b in zip(left, right)))


def _constant_source_models(values: Sequence[tuple[float, float]]) -> list:
    models = []
    for kernel_value, bias_value in values:
        model = build_base_cnn(INPUT_SHAPE, learning_rate=1e-3)
        for layer_name in ("conv1", "conv2", "conv3", "dense_out"):
            _fill_layer(model, layer_name, kernel_value, bias_value)
        models.append(model)
    return models


def test_t01_fused_layer_contract() -> None:
    fused = get_bl8_fused_parameter_layer_names()
    assert fused == ["conv1", "conv2"]
    assert list(BL8_FUSED_PARAMETER_LAYERS) == ["conv1", "conv2"]
    assert "conv3" not in fused
    assert "dense_out" not in fused
    source = inspect.getsource(run_msml_tl)
    assert "get_bl8_fused_parameter_layer_names" in source
    assert "get_transferable_layer_names" not in source


def test_historical_transferable_helper_unchanged_for_rfe_and_katl() -> None:
    assert get_transferable_layer_names() == ["conv1", "conv2", "conv3"]


def test_t02_frozen_layer_contract() -> None:
    model = build_base_cnn(INPUT_SHAPE, learning_rate=1e-3)
    frozen = apply_bl8_target_trainability(model)
    assert frozen == ["conv1", "pool1", "conv2", "pool2"]
    assert get_bl8_frozen_target_layer_names() == ["conv1", "pool1", "conv2", "pool2"]
    assert list(BL8_FROZEN_TARGET_LAYERS) == frozen
    assert model.get_layer("conv1").trainable is False
    assert model.get_layer("pool1").trainable is False
    assert model.get_layer("conv2").trainable is False
    assert model.get_layer("pool2").trainable is False
    assert model.get_layer("conv3").trainable is True
    assert model.get_layer("dense_out").trainable is True


def test_t03_actual_trainable_parameter_layers() -> None:
    model = build_base_cnn(INPUT_SHAPE, learning_rate=1e-3)
    apply_bl8_target_trainability(model)
    trainable = get_trainable_parameter_layer_names(model)
    assert trainable == ["conv3", "dense_out"]
    assert get_bl8_target_trainable_parameter_layer_names() == ["conv3", "dense_out"]
    assert list(BL8_TARGET_TRAINABLE_PARAMETER_LAYERS) == trainable
    for name in ("pool1", "pool2", "flatten", "dropout"):
        assert name not in trainable


def test_t04_conv3_is_not_loaded_from_fused_source_params() -> None:
    weights = [(1.0, 10.0), (2.0, 20.0), (3.0, 30.0)]
    sources = _constant_source_models(weights)
    alphas = [0.5, 0.3, 0.2]
    fused_names = get_bl8_fused_parameter_layer_names(sources[0])
    fused = fuse_source_models_layerwise(sources, alphas, fused_names)
    target = build_base_cnn(INPUT_SHAPE, learning_rate=1e-3)
    _fill_layer(target, "conv3", 7.0, 70.0)
    _fill_layer(target, "dense_out", 8.0, 80.0)
    conv3_before = _snapshot(target, "conv3")
    dense_before = _snapshot(target, "dense_out")
    load_fused_params_into_target_model(target, fused)
    assert "conv3" not in fused
    assert _max_abs_delta(conv3_before, _snapshot(target, "conv3")) == 0.0
    assert _max_abs_delta(dense_before, _snapshot(target, "dense_out")) == 0.0


def test_t05_kernel_and_bias_both_fused() -> None:
    weights = [(1.0, 10.0), (3.0, 30.0), (5.0, 50.0)]
    sources = _constant_source_models(weights)
    alphas = np.array([0.2, 0.3, 0.5], dtype=np.float64)
    fused = fuse_source_models_layerwise(
        sources,
        alphas,
        get_bl8_fused_parameter_layer_names(sources[0]),
    )
    expected_kernel = 0.2 * 1.0 + 0.3 * 3.0 + 0.5 * 5.0
    expected_bias = 0.2 * 10.0 + 0.3 * 30.0 + 0.5 * 50.0
    for layer_name in ("conv1", "conv2"):
        kernel, bias = fused[layer_name]
        assert float(np.max(np.abs(kernel.astype(np.float64) - expected_kernel))) <= 1e-7
        assert float(np.max(np.abs(bias.astype(np.float64) - expected_bias))) <= 1e-7
    assert "conv3" not in fused


def test_t06_dense_out_not_fused() -> None:
    sources = _constant_source_models([(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)])
    fused = fuse_source_models_layerwise(
        sources,
        [0.2, 0.3, 0.5],
        get_bl8_fused_parameter_layer_names(sources[0]),
    )
    target = build_base_cnn(INPUT_SHAPE, learning_rate=1e-3)
    dense_before = _snapshot(target, "dense_out")
    load_fused_params_into_target_model(target, fused)
    assert "dense_out" not in fused
    assert _max_abs_delta(dense_before, _snapshot(target, "dense_out")) == 0.0


def test_t07_inverse_distance_weighting() -> None:
    distances = [1.0, 2.0, 4.0]
    weights = compute_bl8_inverse_distance_weights(distances)
    assert np.all(weights > 0)
    assert abs(float(np.sum(weights)) - 1.0) < 1e-12
    assert weights[0] > weights[1] > weights[2]
    scores = 1.0 / (np.asarray(distances, dtype=np.float64) + BL8_WEIGHT_EPSILON)
    expected = scores / np.sum(scores)
    np.testing.assert_allclose(weights, expected, rtol=0.0, atol=1e-12)


def test_t08_formal_bl8_rejects_raw_distance() -> None:
    source, target = _tiny_pool()
    with pytest.raises(FormalBL8ContractError, match="inverse_distance"):
        run_msml_tl(
            source_df=source,
            target_df=target,
            feature_cols=FEATURE_COLS,
            k=3,
            weight_mode="raw_distance",
            formal_bl8=True,
            k3_selection_context=DummyK3Context(_three_selected_sources()),
        )


def test_t09_formal_bl8_requires_k_equals_3() -> None:
    source, target = _tiny_pool()
    context = DummyK3Context(_three_selected_sources())
    with pytest.raises(FormalBL8ContractError, match="k==3"):
        run_msml_tl(
            source_df=source,
            target_df=target,
            feature_cols=FEATURE_COLS,
            k=2,
            formal_bl8=True,
            k3_selection_context=context,
        )
    with pytest.raises(FormalBL8ContractError, match="k==3"):
        run_msml_tl(
            source_df=source,
            target_df=target,
            feature_cols=FEATURE_COLS,
            k=4,
            formal_bl8=True,
            k3_selection_context=context,
        )


def test_t10_formal_bl8_requires_three_selected_sources() -> None:
    source, target = _tiny_pool()
    with pytest.raises(FormalBL8ContractError, match="exactly 3 selected sources"):
        run_msml_tl(
            source_df=source,
            target_df=target,
            feature_cols=FEATURE_COLS,
            k=3,
            formal_bl8=True,
            k3_selection_context=DummyK3Context(_three_selected_sources()[:2]),
        )


def test_t11_formal_bl8_requires_three_successful_source_models(monkeypatch: pytest.MonkeyPatch) -> None:
    source, target = _tiny_pool()
    normalize_calls: list[int] = []
    real_normalize = msml_tl.normalize_successful_source_weights

    def tracking_normalize(weights):
        normalize_calls.append(len(list(weights)))
        return real_normalize(weights)

    def failing_train(**kwargs):
        if kwargs["source_key"] == (1, 2):
            raise RuntimeError("model weights contain non-finite values")
        model = build_base_cnn(INPUT_SHAPE, learning_rate=1e-3)
        return {
            "model": model,
            "input_shape": INPUT_SHAPE,
            "num_samples": 8,
            "source_key": kwargs["source_key"],
        }

    monkeypatch.setattr(msml_tl, "normalize_successful_source_weights", tracking_normalize)
    monkeypatch.setattr(msml_tl, "train_source_cnn_for_msml", failing_train)
    with pytest.raises(FormalBL8ContractError, match="exactly 3 successful source models"):
        run_msml_tl(
            source_df=source,
            target_df=target,
            feature_cols=FEATURE_COLS,
            k=3,
            formal_bl8=True,
            k3_selection_context=DummyK3Context(_three_selected_sources()),
        )
    assert normalize_calls == []


def test_t12_formal_bl8_requires_sealed_knn_context() -> None:
    source, target = _tiny_pool()
    with pytest.raises(FormalBL8ContractError, match="sealed formal KNN"):
        run_msml_tl(
            source_df=source,
            target_df=target,
            feature_cols=FEATURE_COLS,
            k=3,
            weight_mode="inverse_distance",
            formal_bl8=True,
            k3_selection_context=None,
        )


def test_trainability_is_assigned_before_compile(monkeypatch: pytest.MonkeyPatch) -> None:
    model = build_base_cnn(INPUT_SHAPE, learning_rate=1e-3)
    apply_bl8_target_trainability(model)
    seen: Dict[str, object] = {}
    original_compile = model.compile

    def compile_spy(*args, **kwargs):
        seen["conv1"] = bool(model.get_layer("conv1").trainable)
        seen["pool1"] = bool(model.get_layer("pool1").trainable)
        seen["conv2"] = bool(model.get_layer("conv2").trainable)
        seen["pool2"] = bool(model.get_layer("pool2").trainable)
        seen["conv3"] = bool(model.get_layer("conv3").trainable)
        seen["dense_out"] = bool(model.get_layer("dense_out").trainable)
        return original_compile(*args, **kwargs)

    def fit_spy(*args, **kwargs):
        seen["compiled_before_fit"] = "conv3" in seen
        return SimpleNamespace(history={})

    x_windows = np.zeros((4, *INPUT_SHAPE), dtype=np.float32)
    y_windows = np.zeros((4,), dtype=np.float32)
    monkeypatch.setattr(msml_tl, "build_tabular_sequence", lambda *args, **kwargs: (x_windows, y_windows))
    monkeypatch.setattr(msml_tl, "to_cnn_tensor", lambda values: values)
    model.compile = compile_spy
    model.fit = fit_spy
    fine_tune_fused_target_model(
        target_model=model,
        target_train_df=_entity_frame(2, 9, periods=16),
        target_val_df=_entity_frame(2, 9, periods=12, scale=1.1),
        feature_cols=FEATURE_COLS,
        horizon=1,
        window_size=4,
        epochs=1,
        batch_size=4,
    )
    assert seen["conv1"] is False
    assert seen["pool1"] is False
    assert seen["conv2"] is False
    assert seen["pool2"] is False
    assert seen["conv3"] is True
    assert seen["dense_out"] is True
    assert seen["compiled_before_fit"] is True
    source = inspect.getsource(fine_tune_fused_target_model)
    assert source.index("frozen_layers") < source.index("target_model.compile")
    assert source.index("target_model.compile") < source.index("target_model.fit")


def _patch_tiny_formal_execution(monkeypatch: pytest.MonkeyPatch) -> Dict[str, object]:
    captured: Dict[str, object] = {}
    models = _constant_source_models([(1.0, 10.0), (2.0, 20.0), (3.0, 30.0)])
    queue = list(models)

    def fake_train(**kwargs):
        model = queue.pop(0)
        return {
            "model": model,
            "input_shape": INPUT_SHAPE,
            "num_samples": 8,
            "source_key": kwargs["source_key"],
        }

    def fake_fine_tune(target_model, target_train_df, target_val_df, **kwargs):
        captured["fine_tune_args"] = {
            "target_train_df": target_train_df,
            "target_val_df": target_val_df,
            "keys": sorted(kwargs),
        }
        captured["fit_receives_test"] = "target_test_df" in kwargs
        captured["trainable_at_fine_tune"] = get_trainable_parameter_layer_names(target_model)
        captured["frozen_at_fine_tune"] = [
            layer.name for layer in target_model.layers if not layer.trainable
        ]
        captured["conv3_at_fine_tune"] = _snapshot(target_model, "conv3")
        captured["dense_at_fine_tune"] = _snapshot(target_model, "dense_out")
        return {
            "model": target_model,
            "history": SimpleNamespace(),
            "frozen_layers": list(captured["frozen_at_fine_tune"]),
        }

    def fake_evaluate(target_model, target_test_df, **kwargs):
        captured["evaluate_args"] = {
            "target_test_df": target_test_df,
            "keys": sorted(kwargs),
        }
        y_true = np.array([1.0, 2.0], dtype=np.float64)
        y_pred = np.array([1.1, 1.9], dtype=np.float64)
        return {
            "rmse": 0.1,
            "accuracy": 0.9,
            "smape": 0.05,
            "y_true": y_true,
            "y_pred": y_pred,
            "prediction_shape": tuple(y_pred.shape),
        }

    monkeypatch.setattr(msml_tl, "train_source_cnn_for_msml", fake_train)
    monkeypatch.setattr(msml_tl, "fine_tune_fused_target_model", fake_fine_tune)
    monkeypatch.setattr(msml_tl, "evaluate_msml_model", fake_evaluate)
    monkeypatch.setattr(msml_tl, "temporal_split_by_ratio_or_dates", lambda df: (df.copy(), df.copy(), df.copy()))
    monkeypatch.setattr(
        msml_tl,
        "normalize_features",
        lambda train, val, test, feature_columns=None, reuse_context=None: (
            train,
            val,
            test,
            object(),
            list(feature_columns or FEATURE_COLS),
        ),
    )
    return captured


def test_t13_target_fit_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_tiny_formal_execution(monkeypatch)
    source, target = _tiny_pool()
    run_msml_tl(
        source_df=source,
        target_df=target,
        feature_cols=FEATURE_COLS,
        k=3,
        formal_bl8=True,
        k3_selection_context=DummyK3Context(_three_selected_sources()),
        seed=42,
    )
    fit_args = captured["fine_tune_args"]
    assert fit_args["target_train_df"] is not None
    assert fit_args["target_val_df"] is not None
    assert captured["fit_receives_test"] is False
    source_text = inspect.getsource(fine_tune_fused_target_model)
    assert "validation_data=(X_val, y_val)" in source_text
    assert "target_test" not in source_text


def test_t14_target_test_prediction_only(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_tiny_formal_execution(monkeypatch)
    source, target = _tiny_pool()
    run_msml_tl(
        source_df=source,
        target_df=target,
        feature_cols=FEATURE_COLS,
        k=3,
        formal_bl8=True,
        k3_selection_context=DummyK3Context(_three_selected_sources()),
    )
    assert "target_test_df" in captured["evaluate_args"]
    evaluate_source = inspect.getsource(evaluate_msml_model)
    run_source = inspect.getsource(run_msml_tl)
    assert "target_model.predict" in evaluate_source
    assert ".fit(" not in evaluate_source
    assert run_source.index("fine_tune_fused_target_model") < run_source.index("evaluate_msml_model")


def test_t15_rfe_absent() -> None:
    run_source = inspect.getsource(run_msml_tl)
    module_source = inspect.getsource(msml_tl)
    assert '"rfe": False' in run_source
    assert "run_msml_tl_rfe" not in module_source
    assert "Recursive Feature Elimination" not in module_source
    assert "keep_ratio" not in run_source
    assert "estimator_name" not in run_source


def test_t16_metadata_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_tiny_formal_execution(monkeypatch)
    source, target = _tiny_pool()
    result = run_msml_tl(
        source_df=source,
        target_df=target,
        feature_cols=FEATURE_COLS,
        k=3,
        formal_bl8=True,
        k3_selection_context=DummyK3Context(_three_selected_sources()),
        seed=42,
        weight_mode="inverse_distance",
    )
    meta = result["meta"]
    assert meta["method"] == "MSML-TL"
    assert meta["fused_layers"] == ["conv1", "conv2"]
    assert meta["frozen_layers"] == ["conv1", "pool1", "conv2", "pool2"]
    assert "conv3" not in meta["fused_layers"]
    assert "conv3" not in meta["frozen_layers"]
    assert meta["target_trainable_parameter_layers"] == ["conv3", "dense_out"]
    assert captured["trainable_at_fine_tune"] == ["conv3", "dense_out"]
    assert meta["requested_source_count"] == 3
    assert meta["effective_selected_source_count"] == 3
    assert meta["successful_source_model_count"] == 3
    assert meta["weight_mode"] == "inverse_distance"
    assert meta["weight_epsilon"] == BL8_WEIGHT_EPSILON
    assert meta["seed"] == 42
    assert meta["rfe"] is False
    assert [entry["source_key"] for entry in result["source_models_info"]] == [(1, 1), (1, 2), (1, 3)]
    assert [entry["distance"] for entry in result["source_models_info"]] == [1.0, 2.0, 4.0]
    assert result["fused_result"]["y_true"] is not None
    assert result["fused_result"]["y_pred"] is not None


def test_weight_invariance_conv3_and_dense_out_unchanged_after_fusion_load() -> None:
    sources = _constant_source_models([(1.0, 11.0), (2.0, 22.0), (3.0, 33.0)])
    alphas = [0.5, 0.3, 0.2]
    fused = fuse_source_models_layerwise(
        sources,
        alphas,
        get_bl8_fused_parameter_layer_names(sources[0]),
    )
    target = build_base_cnn(INPUT_SHAPE, learning_rate=1e-3)
    conv3_before = _snapshot(target, "conv3")
    dense_before = _snapshot(target, "dense_out")
    load_fused_params_into_target_model(target, fused)
    apply_bl8_target_trainability(target)
    assert _max_abs_delta(conv3_before, _snapshot(target, "conv3")) == 0.0
    assert _max_abs_delta(dense_before, _snapshot(target, "dense_out")) == 0.0
    conv1_kernel, conv1_bias = target.get_layer("conv1").get_weights()
    expected_k = np.asarray(fused["conv1"][0], dtype=np.float32)
    expected_b = np.asarray(fused["conv1"][1], dtype=np.float32)
    np.testing.assert_array_equal(conv1_kernel, expected_k)
    np.testing.assert_array_equal(conv1_bias, expected_b)
    reconstructed_k = 0.5 * 1.0 + 0.3 * 2.0 + 0.2 * 3.0
    reconstructed_b = 0.5 * 11.0 + 0.3 * 22.0 + 0.2 * 33.0
    assert float(np.max(np.abs(fused["conv1"][0].astype(np.float64) - reconstructed_k))) <= 1e-7
    assert float(np.max(np.abs(fused["conv1"][1].astype(np.float64) - reconstructed_b))) <= 1e-7


def test_source_fusion_correctness_matches_weighted_sum() -> None:
    kernel_vals = [1.0, 4.0, 7.0]
    bias_vals = [2.0, 5.0, 8.0]
    sources = _constant_source_models(list(zip(kernel_vals, bias_vals)))
    alphas = np.array([0.1, 0.2, 0.7], dtype=np.float64)
    fused = fuse_source_models_layerwise(
        sources,
        alphas,
        get_bl8_fused_parameter_layer_names(sources[0]),
    )
    expected_kernel = float(alphas @ np.asarray(kernel_vals, dtype=np.float64))
    expected_bias = float(alphas @ np.asarray(bias_vals, dtype=np.float64))
    for layer_name in ("conv1", "conv2"):
        kernel, bias = fused[layer_name]
        assert float(np.max(np.abs(kernel.astype(np.float64) - expected_kernel))) <= 1e-7
        assert float(np.max(np.abs(bias.astype(np.float64) - expected_bias))) <= 1e-7
        target = build_base_cnn(INPUT_SHAPE, learning_rate=1e-3)
        load_fused_params_into_target_model(target, fused)
        for layer_name in ("conv1", "conv2"):
            kernel, bias = target.get_layer(layer_name).get_weights()
            np.testing.assert_array_equal(kernel, np.asarray(fused[layer_name][0], dtype=np.float32))
            np.testing.assert_array_equal(bias, np.asarray(fused[layer_name][1], dtype=np.float32))
