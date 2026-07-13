"""
Module 8: MSML-TL (Multi-Source Multi-Layer Transfer Learning)

本模块实现多源多层参数融合迁移学习：
1. 从 source pool 中选出 top-k 相似源
2. 对每个 source 分别训练 CNN 模型
3. 提取多个 source 模型对应层的参数
4. 按 source weight 对层参数做加权平均
5. 构造 target model 并加载融合后的层参数
6. 冻结已融合层，在 target 上微调剩余层
7. 在 target test 上预测并评估

与 MSSB-TL（模型切换）和 MSWA-TL（预测加权）不同，
MSML-TL 融合的是"层参数"本身。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from src.utils.environment import setup_logging
except ImportError:
    setup_logging = None

from src.models.cnn_model import build_base_cnn
from src.data_processing.data_preprocessing import (
    build_tabular_sequence,
    normalize_features,
    temporal_split_by_ratio_or_dates,
    to_cnn_tensor,
)
from src.source_selection.source_selector import SourceSelector
from src.utils.runtime_control import keras_verbose
from src.evaluation.metrics import smape
from src.utils.finite_diagnostics import NonFiniteArrayError, summarize_model_weights, validate_finite_array
from src.utils.source_fillna import fill_source_numeric_na
from src.transfer_methods.source_failure_tolerance import (
    AllSourcesFailedError,
    SOURCE_LEVEL_EXCEPTIONS,
    make_failed_source,
    normalize_successful_source_weights,
    enforce_formal_source_success,
    runtime_selection_meta,
    should_skip_source_exception,
    source_failure_meta,
)
from src.protocols.runner_adapter import source_key_mask
from src.protocols.provenance import (
    assert_actual_cnn_training_validated,
    bind_actual_cnn_source_frame,
)


LOGGER_NAME = "experiment"

# 默认参与参数融合的层名
_DEFAULT_TRANSFERABLE_LAYERS = ["conv1", "conv2", "conv3"]


def _get_logger() -> logging.Logger:
    """获取项目统一日志器；若未初始化则按默认参数初始化。"""
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers and setup_logging is not None:
        setup_logging(log_level="INFO", log_file=None)
        logger = logging.getLogger(LOGGER_NAME)
    return logger


def _validate_feature_cols(df: pd.DataFrame, feature_cols: Sequence[str], where: str) -> List[str]:
    """校验特征列是否存在并返回列表。"""
    cols = list(feature_cols)
    if not cols:
        raise ValueError("feature_cols must not be empty")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in {where}: {missing}")
    return cols


def _prepare_source_split(source_sequence_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按 source 风格比例切分单个 source 序列。"""
    source_one = source_sequence_df.copy()
    source_one.attrs["split_role"] = "source"
    source_one.attrs["split_mode"] = "ratio"
    source_one.attrs["split_config"] = {
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
    }
    return temporal_split_by_ratio_or_dates(source_one)


# -------------------------------------------------------------------
# 1. train_source_cnn_for_msml
# -------------------------------------------------------------------

def train_source_cnn_for_msml(
    source_sequence_df: pd.DataFrame,
    feature_cols: Sequence[str],
    horizon: int = 1,
    window_size: int = 10,
    learning_rate: float = 0.001,
    source_epochs: int = 3,
    batch_size: int = 16,
    source_key: Optional[Tuple] = None,
) -> Dict[str, object]:
    """
    在单个 source 序列上训练 CNN 模型。

    Args:
        source_sequence_df: 单个 source 序列 DataFrame。
        feature_cols: 特征列名列表。
        horizon: 预测步长。
        window_size: 滑窗大小。
        learning_rate: Adam 学习率。
        source_epochs: 训练轮数。
        batch_size: 批大小。
        source_key: 可选的 source 标识（如 (entity_id, item_id)）。

    Returns:
        {
            "model": 训练好的 Keras 模型,
            "input_shape": (window_size, num_features),
            "num_samples": int,
            "source_key": tuple or None,
        }
    """
    logger = _get_logger()
    logger.info(
        "[train_source_cnn_for_msml] Start. source_key=%s epochs=%d batch_size=%d",
        source_key, source_epochs, batch_size,
    )

    _validate_feature_cols(source_sequence_df, feature_cols, where="source_sequence_df")

    source_sequence_df = fill_source_numeric_na(source_sequence_df, feature_columns=feature_cols)
    src_train, src_val, src_test = _prepare_source_split(source_sequence_df)
    src_train, src_val, src_test, _, _ = normalize_features(
        src_train, src_val, src_test, feature_columns=feature_cols
    )

    X_source, y_source = build_tabular_sequence(
        src_train, horizon=horizon, window_size=window_size, feature_columns=feature_cols
    )
    if src_train.attrs.get("protocol_actual_source_key") is not None:
        assert_actual_cnn_training_validated(
            src_train,
            source_key=src_train.attrs["protocol_actual_source_key"],
        )
    if len(y_source) == 0:
        raise ValueError(
            f"Source sequence (key={source_key}) produced zero training windows; "
            "adjust window_size/horizon."
        )

    X_source = to_cnn_tensor(X_source)
    input_shape = X_source.shape[1:]

    model = build_base_cnn(input_shape, learning_rate=learning_rate)
    model.fit(
        X_source,
        y_source,
        epochs=source_epochs,
        batch_size=batch_size,
        verbose=keras_verbose(),
    )

    logger.info(
        "[train_source_cnn_for_msml] Finished. source_key=%s input_shape=%s num_samples=%d",
        source_key, input_shape, len(y_source),
    )
    return {
        "model": model,
        "input_shape": tuple(input_shape),
        "num_samples": len(y_source),
        "source_key": source_key,
    }


# -------------------------------------------------------------------
# 2. get_transferable_layer_names
# -------------------------------------------------------------------

def get_transferable_layer_names(model=None) -> List[str]:
    """
    返回参与参数融合的层名列表。

    默认只返回卷积层 ['conv1', 'conv2', 'conv3']，
    不包含 flatten 和 dense_out。

    Args:
        model: 可选的 Keras 模型，用于校验层名是否存在。

    Returns:
        层名列表。
    """
    layer_names = list(_DEFAULT_TRANSFERABLE_LAYERS)
    if model is not None:
        model_layer_names = {l.name for l in model.layers}
        for name in layer_names:
            if name not in model_layer_names:
                raise ValueError(
                    f"Transferable layer '{name}' not found in model. "
                    f"Available: {sorted(model_layer_names)}"
                )
    return layer_names


# -------------------------------------------------------------------
# 3. extract_layer_params
# -------------------------------------------------------------------

def extract_layer_params(model, layer_names: Sequence[str]) -> Dict[str, List[np.ndarray]]:
    """
    从模型中按层名提取权重参数。

    Args:
        model: Keras 模型。
        layer_names: 要提取的层名列表。

    Returns:
        {
            'conv1': [kernel_array, bias_array],
            'conv2': [kernel_array, bias_array],
            ...
        }

    Raises:
        ValueError: 层不存在或层没有权重。
    """
    model_layer_dict = {l.name: l for l in model.layers}
    result: Dict[str, List[np.ndarray]] = {}

    for name in layer_names:
        if name not in model_layer_dict:
            raise ValueError(
                f"Layer '{name}' not found in model. "
                f"Available: {sorted(model_layer_dict.keys())}"
            )
        layer = model_layer_dict[name]
        weights = layer.get_weights()
        if not weights:
            raise ValueError(
                f"Layer '{name}' has no weights (trainable parameters). "
                "Cannot extract parameters for fusion."
            )
        result[name] = [np.array(w) for w in weights]

    return result


# -------------------------------------------------------------------
# 4. weighted_average_layer_params
# -------------------------------------------------------------------

def weighted_average_layer_params(
    layer_params_list: List[List[np.ndarray]],
    weights: Sequence[float],
) -> List[np.ndarray]:
    """
    对多个 source 模型同一层的参数做加权平均。

    Args:
        layer_params_list: 每个元素是一个 source 的 [kernel, bias, ...] 列表。
        weights: 对应的权重数组，长度与 layer_params_list 相同。

    Returns:
        加权平均后的参数列表 [avg_kernel, avg_bias, ...]。

    Raises:
        ValueError: 参数 shape 不一致或 weights 长度不匹配。
    """
    n_sources = len(layer_params_list)
    w = np.asarray(weights, dtype=np.float64)

    if w.shape[0] != n_sources:
        raise ValueError(
            f"weights length ({w.shape[0]}) != number of sources ({n_sources})"
        )

    n_params = len(layer_params_list[0])
    for i, params in enumerate(layer_params_list):
        if len(params) != n_params:
            raise ValueError(
                f"Source {i} has {len(params)} param tensors, "
                f"expected {n_params} (same as source 0)."
            )

    averaged: List[np.ndarray] = []
    for p_idx in range(n_params):
        ref_shape = layer_params_list[0][p_idx].shape
        for s_idx in range(n_sources):
            if layer_params_list[s_idx][p_idx].shape != ref_shape:
                raise ValueError(
                    f"Shape mismatch at param index {p_idx}: "
                    f"source 0 shape={ref_shape}, "
                    f"source {s_idx} shape={layer_params_list[s_idx][p_idx].shape}"
                )
        acc = np.zeros(ref_shape, dtype=np.float64)
        for s_idx in range(n_sources):
            acc += float(w[s_idx]) * layer_params_list[s_idx][p_idx].astype(np.float64)
        averaged.append(acc)

    return averaged


# -------------------------------------------------------------------
# 5. fuse_source_models_layerwise
# -------------------------------------------------------------------

def fuse_source_models_layerwise(
    source_models: List,
    weights: Sequence[float],
    layer_names: Sequence[str],
) -> Dict[str, List[np.ndarray]]:
    """
    对多个 source model 的指定层做加权参数融合。

    Args:
        source_models: Keras 模型列表。
        weights: 对应权重列表（与 source_models 长度一致）。
        layer_names: 要融合的层名列表。

    Returns:
        {
            'conv1': [avg_kernel, avg_bias],
            'conv2': [avg_kernel, avg_bias],
            'conv3': [avg_kernel, avg_bias],
        }
    """
    logger = _get_logger()
    logger.info(
        "[fuse_source_models_layerwise] Start. num_sources=%d layers=%s",
        len(source_models), list(layer_names),
    )

    if len(source_models) != len(weights):
        raise ValueError(
            f"Number of source models ({len(source_models)}) != "
            f"number of weights ({len(weights)})"
        )

    all_params: List[Dict[str, List[np.ndarray]]] = []
    for i, model in enumerate(source_models):
        params = extract_layer_params(model, layer_names)
        all_params.append(params)

    fused: Dict[str, List[np.ndarray]] = {}
    for name in layer_names:
        per_source = [all_params[i][name] for i in range(len(source_models))]
        fused[name] = weighted_average_layer_params(per_source, weights)

    logger.info("[fuse_source_models_layerwise] Finished. fused_layers=%s", list(fused.keys()))
    return fused


# -------------------------------------------------------------------
# 6. load_fused_params_into_target_model
# -------------------------------------------------------------------

def load_fused_params_into_target_model(
    target_model,
    fused_params: Dict[str, List[np.ndarray]],
) -> object:
    """
    将融合后的层参数加载到 target model 对应层。

    Args:
        target_model: Keras 模型。
        fused_params: {layer_name: [param_array, ...]} 字典。

    Returns:
        已加载参数的 target_model。

    Raises:
        ValueError: 层名不存在。
    """
    logger = _get_logger()
    model_layer_dict = {l.name: l for l in target_model.layers}

    for name, params in fused_params.items():
        if name not in model_layer_dict:
            raise ValueError(
                f"Layer '{name}' not found in target model. "
                f"Available: {sorted(model_layer_dict.keys())}"
            )
        model_layer_dict[name].set_weights(params)

    logger.info(
        "[load_fused_params_into_target_model] Loaded fused params into layers: %s",
        list(fused_params.keys()),
    )
    return target_model


# -------------------------------------------------------------------
# 7. freeze_fused_layers
# -------------------------------------------------------------------

def freeze_fused_layers(
    target_model,
    fused_layer_names: Sequence[str],
) -> List[str]:
    """
    冻结已融合层，其余层保持可训练。

    Args:
        target_model: Keras 模型。
        fused_layer_names: 要冻结的层名列表。

    Returns:
        被冻结的层名列表。
    """
    logger = _get_logger()
    fused_set = set(fused_layer_names)
    frozen: List[str] = []

    for layer in target_model.layers:
        if layer.name in fused_set:
            layer.trainable = False
            frozen.append(layer.name)
        else:
            layer.trainable = True

    logger.info("[freeze_fused_layers] Frozen layers: %s", frozen)
    return frozen


# -------------------------------------------------------------------
# 8. fine_tune_fused_target_model
# -------------------------------------------------------------------

def fine_tune_fused_target_model(
    target_model,
    target_train_df: pd.DataFrame,
    target_val_df: pd.DataFrame,
    feature_cols: Sequence[str],
    horizon: int = 1,
    window_size: int = 10,
    learning_rate: float = 0.001,
    epochs: int = 3,
    batch_size: int = 16,
) -> Dict[str, object]:
    """
    在 target 数据上微调融合后的 target model。

    Args:
        target_model: 已加载融合参数并冻结部分层的 Keras 模型。
        target_train_df: 归一化后的 target 训练集。
        target_val_df: 归一化后的 target 验证集。
        feature_cols: 特征列名列表。
        horizon: 预测步长。
        window_size: 滑窗大小。
        learning_rate: 微调学习率。
        epochs: 微调轮数。
        batch_size: 批大小。

    Returns:
        {
            "model": 微调后的 Keras 模型,
            "history": Keras History 对象,
            "frozen_layers": 被冻结的层名列表,
        }
    """
    import tensorflow as tf

    logger = _get_logger()
    logger.info(
        "[fine_tune_fused_target_model] Start. epochs=%d batch_size=%d",
        epochs, batch_size,
    )

    X_train, y_train = build_tabular_sequence(
        target_train_df, horizon=horizon, window_size=window_size, feature_columns=feature_cols
    )
    X_val, y_val = build_tabular_sequence(
        target_val_df, horizon=horizon, window_size=window_size, feature_columns=feature_cols
    )

    if len(y_train) == 0:
        raise ValueError("Target train split produced zero windows; adjust window_size/horizon.")
    if len(y_val) == 0:
        raise ValueError("Target val split produced zero windows; adjust window_size/horizon.")

    X_train = to_cnn_tensor(X_train)
    X_val = to_cnn_tensor(X_val)

    frozen_layers = [l.name for l in target_model.layers if not l.trainable]

    target_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )

    history = target_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=keras_verbose(),
    )

    logger.info("[fine_tune_fused_target_model] Fine-tuning completed.")
    return {
        "model": target_model,
        "history": history,
        "frozen_layers": frozen_layers,
    }


# -------------------------------------------------------------------
# 9. evaluate_msml_model
# -------------------------------------------------------------------

def evaluate_msml_model(
    target_model,
    target_test_df: pd.DataFrame,
    feature_cols: Sequence[str],
    horizon: int = 1,
    window_size: int = 10,
    eps: float = 1e-8,
) -> Dict[str, object]:
    """
    在 target test 上评估 MSML-TL 模型。

    Args:
        target_model: 微调后的 Keras 模型。
        target_test_df: 归一化后的 target 测试集。
        feature_cols: 特征列名列表。
        horizon: 预测步长。
        window_size: 滑窗大小。
        eps: 数值稳定项。

    Returns:
        {
            "rmse": float,
            "accuracy": float,
            "y_pred": np.ndarray,
            "y_true": np.ndarray,
            "prediction_shape": tuple,
        }
    """
    logger = _get_logger()
    logger.info("[evaluate_msml_model] Start.")

    X_test, y_test = build_tabular_sequence(
        target_test_df, horizon=horizon, window_size=window_size, feature_columns=feature_cols
    )
    if len(y_test) == 0:
        raise ValueError("Target test split produced zero windows; adjust window_size/horizon.")

    X_test = to_cnn_tensor(X_test)
    diagnostics = {}
    diagnostics.update(validate_finite_array(X_test, name="X_test"))
    diagnostics.update(validate_finite_array(y_test, name="y_true", context=diagnostics))
    weight_diagnostics = summarize_model_weights(target_model)
    diagnostics.update(weight_diagnostics)
    if weight_diagnostics["model_weight_nan_count"] or weight_diagnostics["model_weight_inf_count"]:
        raise NonFiniteArrayError(
            "model weights contain non-finite values: "
            f"nan_count={weight_diagnostics['model_weight_nan_count']} "
            f"inf_count={weight_diagnostics['model_weight_inf_count']}",
            diagnostics=diagnostics,
        )

    y_pred = target_model.predict(X_test, verbose=0)
    diagnostics.update(validate_finite_array(y_pred, name="y_pred", context=diagnostics))
    y_pred_flat = y_pred.flatten()
    y_true = y_test.flatten()

    rmse = float(np.sqrt(np.mean((y_pred_flat - y_true) ** 2)))
    accuracy = float(1.0 / (rmse + eps))
    smape_value = float(smape(y_true, y_pred_flat, epsilon=eps))

    logger.info("[evaluate_msml_model] Finished. RMSE=%.4f Accuracy=%.4f", rmse, accuracy)
    return {
        "rmse": rmse,
        "accuracy": accuracy,
        "smape": smape_value,
        "y_pred": y_pred,
        "y_true": y_true,
        "prediction_shape": tuple(y_pred.shape),
        **diagnostics,
    }


# -------------------------------------------------------------------
# 10. run_msml_tl
# -------------------------------------------------------------------

def run_msml_tl(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    feature_cols: Sequence[str],
    k: int = 3,
    horizon: int = 1,
    window_size: int = 10,
    weight_mode: str = "inverse_distance",
    learning_rate: float = 0.001,
    source_epochs: int = 3,
    target_epochs: int = 3,
    batch_size: int = 16,
    group_cols: Sequence[str] = ("entity_id", "item_id"),
    metric_identity: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """
    运行 MSML-TL 完整流程。

    步骤:
    1. select_top_k_sources → 选源 + 权重
    2. 逐源训练 CNN → 多个 source model
    3. fuse_source_models_layerwise → 对指定层加权参数融合
    4. 构建 target model → 加载融合参数 → 冻结融合层
    5. 切分 target → 微调 target model
    6. 在 target test 上评估

    Args:
        source_df: Source pool DataFrame。
        target_df: Target DataFrame。
        feature_cols: 特征列名列表。
        k: 选择的相似源数量。
        horizon: 预测步长。
        window_size: 滑窗大小。
        weight_mode: 权重模式。
        learning_rate: 学习率。
        source_epochs: Source 训练轮数。
        target_epochs: Target 微调轮数。
        batch_size: 批大小。

    Returns:
        {
            "meta": {
                "method": "MSML-TL",
                "k": int,
                "weight_mode": str,
                "feature_cols": list,
                "selected_sources": list,
                "fused_layers": list,
            },
            "source_models_info": [...],
            "fused_result": {
                "rmse": float,
                "accuracy": float,
                "prediction_shape": tuple,
            },
            "frozen_layers": [...],
        }
    """
    logger = _get_logger()
    logger.info("[run_msml_tl] Start. k=%d weight_mode=%s", k, weight_mode)

    _validate_feature_cols(source_df, feature_cols, where="source_df")
    _validate_feature_cols(target_df, feature_cols, where="target_df")
    resolved_group_cols = tuple(group_cols)

    # --- Step 1: 选源 ---
    selector = SourceSelector()
    selection_result = selector.select_top_k_sources(
        target_df=target_df,
        source_df=source_df,
        feature_cols=feature_cols,
        k=k,
        group_cols=resolved_group_cols,
        weight_mode=weight_mode,
    )
    selected_sources = selection_result.get("sources", []) if isinstance(selection_result, dict) else selection_result
    if not selected_sources:
        raise ValueError("No source selected from source pool.")

    # --- Step 2: 逐源训练 CNN ---
    source_models: List = []
    source_weights: List[float] = []
    source_models_info: List[Dict[str, object]] = []
    input_shape_ref: Optional[Tuple[int, ...]] = None
    failed_sources: List[Dict[str, object]] = []

    for selected in selected_sources:
        source_key = tuple(selected["source_key"]) if isinstance(selected["source_key"], (list, tuple)) else (selected["source_key"],)
        if len(source_key) != len(resolved_group_cols):
            raise ValueError(f"Invalid source_key format: {source_key}")

        source_mask = source_key_mask(source_df, resolved_group_cols, source_key)
        source_sequence_df = source_df[source_mask].copy()

        if source_sequence_df.empty:
            raise ValueError(f"Selected source_key not found in source_df: {source_key}")
        bind_actual_cnn_source_frame(
            source_sequence_df,
            source_key=source_key,
            group_cols=resolved_group_cols,
            feature_cols=feature_cols,
        )

        try:
            train_result = train_source_cnn_for_msml(
                source_sequence_df=source_sequence_df,
                feature_cols=feature_cols,
                horizon=horizon,
                window_size=window_size,
                learning_rate=learning_rate,
                source_epochs=source_epochs,
                batch_size=batch_size,
                source_key=source_key,
            )
            weight_diagnostics = summarize_model_weights(train_result["model"])
            if weight_diagnostics["model_weight_nan_count"] or weight_diagnostics["model_weight_inf_count"]:
                raise NonFiniteArrayError(
                    "source model weights contain non-finite values: "
                    f"nan_count={weight_diagnostics['model_weight_nan_count']} "
                    f"inf_count={weight_diagnostics['model_weight_inf_count']}",
                    diagnostics=weight_diagnostics,
                )
        except SOURCE_LEVEL_EXCEPTIONS as exc:
            enforce_formal_source_success(source_df, source_key, exc)
            if not should_skip_source_exception(exc):
                raise
            failed_source = make_failed_source(source_key, exc)
            failed_sources.append(failed_source)
            logger.warning(
                "[run_msml_tl] Skipping failed source_key=%s exception_type=%s message=%s",
                source_key,
                failed_source["exception_type"],
                failed_source["exception_message"],
            )
            continue

        if input_shape_ref is None:
            input_shape_ref = train_result["input_shape"]
        elif train_result["input_shape"] != input_shape_ref:
            raise ValueError(
                f"Input shape mismatch: source_key={source_key} "
                f"shape={train_result['input_shape']} expected={input_shape_ref}"
            )

        source_models.append(train_result["model"])
        source_weights.append(float(selected["weight"]))
        source_models_info.append({
            "source_key": source_key,
            "distance": float(selected["distance"]),
            "weight": float(selected["weight"]),
            "num_samples": int(train_result["num_samples"]),
        })

    if not source_models:
        raise AllSourcesFailedError(
            "MSML-TL",
            failed_sources,
            selected_sources=selected_sources,
            selection_meta=runtime_selection_meta(selection_result),
        )

    source_weights = normalize_successful_source_weights(source_weights)
    for info, normalized_weight in zip(source_models_info, source_weights):
        info["weight"] = float(normalized_weight)

    # --- Step 3: 确定可融合层并做加权参数融合 ---
    layer_names = get_transferable_layer_names(source_models[0])
    fused_params = fuse_source_models_layerwise(source_models, source_weights, layer_names)

    # --- Step 4: 构建 target model → 加载融合参数 → 冻结 ---
    target_model = build_base_cnn(input_shape_ref, learning_rate=learning_rate)
    target_model = load_fused_params_into_target_model(target_model, fused_params)
    frozen_layers = freeze_fused_layers(target_model, layer_names)

    # --- Step 5: 切分 target → 归一化 → 微调 ---
    target_train_df, target_val_df, target_test_df = temporal_split_by_ratio_or_dates(target_df)
    target_train_df, target_val_df, target_test_df, target_scaler, target_feature_columns = normalize_features(
        target_train_df, target_val_df, target_test_df, feature_columns=feature_cols,
    )

    ft_result = fine_tune_fused_target_model(
        target_model=target_model,
        target_train_df=target_train_df,
        target_val_df=target_val_df,
        feature_cols=feature_cols,
        horizon=horizon,
        window_size=window_size,
        learning_rate=learning_rate,
        epochs=target_epochs,
        batch_size=batch_size,
    )

    # --- Step 6: 评估 ---
    eval_result = evaluate_msml_model(
        target_model=ft_result["model"],
        target_test_df=target_test_df,
        feature_cols=feature_cols,
        horizon=horizon,
        window_size=window_size,
    )

    logger.info(
        "[run_msml_tl] Finished. RMSE=%.4f Accuracy=%.4f prediction_shape=%s",
        eval_result["rmse"], eval_result["accuracy"], eval_result["prediction_shape"],
    )

    return {
        "meta": {
            "method": "MSML-TL",
            "k": int(k),
            "weight_mode": weight_mode,
            "feature_cols": list(feature_cols),
            "knn_feature_mode": (selection_result.get("meta", {}) or {}).get("knn_feature_mode", ""),
            "selected_sources": selected_sources,
            **runtime_selection_meta(selection_result),
            "fused_layers": list(layer_names),
            **source_failure_meta(
                requested_k=k,
                selected_sources=selected_sources,
                valid_source_count=len(source_models_info),
                failed_sources=failed_sources,
            ),
        },
        "source_models_info": source_models_info,
        "fused_result": {
            "rmse": eval_result["rmse"],
            "accuracy": eval_result["accuracy"],
            "smape": eval_result["smape"],
            "y_true": eval_result["y_true"],
            "y_pred": eval_result["y_pred"],
            "prediction_shape": eval_result["prediction_shape"],
            "sales_scaler": target_scaler,
            "feature_columns": target_feature_columns,
            **dict(metric_identity or {}),
            **{
                key: value
                for key, value in eval_result.items()
                if key.endswith("_nan_count")
                or key.endswith("_inf_count")
                or key in {"X_test_shape", "y_true_shape", "y_pred_shape"}
            },
        },
        "frozen_layers": frozen_layers,
    }
