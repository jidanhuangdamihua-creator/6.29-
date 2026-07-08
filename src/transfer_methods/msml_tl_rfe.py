"""
Module 9: MSML-TL-RFE (Multi-Source Multi-Layer Transfer Learning with Recursive Feature Elimination)

本模块在 MSML-TL 的基础上增加了 RFE（递归特征消除）步骤：
1. 从 source pool 中选出 top-k 相似源
2. 对 target_train 和所有 selected sources 的训练部分联合执行 RFE
3. 得到统一的特征子集
4. 对所有 target 和 selected sources 应用同一特征子集
5. 分别训练各 source CNN（使用 RFE 后的特征）
6. 对各层参数做加权平均融合
7. 构建 target model 并加载融合参数
8. 冻结已融合层，微调其余层
9. 评估模型

关键区别：
- RFE 必须在 source CNN 训练前进行
- source 和 target 必须使用同一组最终特征
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
from src.transfer_methods.msml_tl import (
    get_transferable_layer_names,
    extract_layer_params,
    weighted_average_layer_params,
    fuse_source_models_layerwise,
    load_fused_params_into_target_model,
    freeze_fused_layers,
)
from src.utils.runtime_control import keras_verbose
from src.evaluation.metrics import smape
from src.utils.finite_diagnostics import NonFiniteArrayError, summarize_model_weights, validate_finite_array
from src.transfer_methods.source_failure_tolerance import (
    AllSourcesFailedError,
    SOURCE_LEVEL_EXCEPTIONS,
    make_failed_source,
    normalize_successful_source_weights,
    should_skip_source_exception,
    source_failure_meta,
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
# 1. run_rfe_feature_selection
# -------------------------------------------------------------------

def run_rfe_feature_selection(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str = "sales",
    estimator_name: str = "random_forest",
    keep_ratio: float = 0.5,
    random_state: int = 42,
) -> Dict[str, object]:
    """
    对输入 train_df 执行 RFE，选出保留特征。

    Args:
        train_df: 用于 RFE 拟合的训练 DataFrame
        feature_cols: 候选特征列名
        target_col: 目标列名
        estimator_name: 评估器名称，支持 'random_forest', 'linear_regression' 等
        keep_ratio: 保留特征的比例 (0, 1]
        random_state: 随机种子

    Returns:
        {
            "selected_feature_cols": [...],
            "num_selected_features": int,
            "num_original_features": int,
            "keep_ratio": float,
        }

    Raises:
        ValueError: 若特征不足或 estimator_name 不支持
    """
    from sklearn.feature_selection import RFE

    logger = _get_logger()
    logger.info(
        "[run_rfe_feature_selection] Start. estimator=%s keep_ratio=%.2f",
        estimator_name, keep_ratio,
    )

    requested_cols = _validate_feature_cols(train_df, feature_cols, where="train_df")
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in train_df")
    cols = [c for c in requested_cols if c != target_col]
    if not cols:
        raise ValueError(
            "RFE candidate features must include at least one non-target column; "
            f"target_col={target_col!r}"
        )

    X = train_df[cols].to_numpy()
    y = train_df[target_col].to_numpy()

    # 创建评估器
    if estimator_name.lower() == "random_forest":
        try:
            from sklearn.ensemble import RandomForestRegressor
            estimator = RandomForestRegressor(random_state=random_state, n_estimators=10)
        except ImportError:
            raise ValueError("sklearn RandomForestRegressor not available")
    elif estimator_name.lower() == "linear_regression":
        try:
            from sklearn.linear_model import LinearRegression
            estimator = LinearRegression()
        except ImportError:
            raise ValueError("sklearn LinearRegression not available")
    else:
        raise ValueError(
            f"Unsupported estimator_name: {estimator_name}. "
            "Use 'random_forest' or 'linear_regression'."
        )

    # 计算保留特征数
    num_original = len(cols)
    num_to_select = max(1, int(np.ceil(num_original * keep_ratio)))

    logger.info(
        "[run_rfe_feature_selection] Feature count. original=%d to_select=%d",
        num_original, num_to_select,
    )

    # 执行 RFE
    rfe = RFE(estimator=estimator, n_features_to_select=num_to_select, step=1)
    rfe.fit(X, y)

    # 提取已选特征
    selected_indices = np.where(rfe.support_)[0]
    selected_cols = [cols[i] for i in selected_indices]

    logger.info(
        "[run_rfe_feature_selection] Finished. "
        "selected=%d selected_cols=%s",
        len(selected_cols), selected_cols,
    )

    return {
        "selected_feature_cols": selected_cols,
        "num_selected_features": len(selected_cols),
        "num_original_features": num_original,
        "keep_ratio": float(keep_ratio),
    }


# -------------------------------------------------------------------
# 2. build_joint_rfe_training_dataframe
# -------------------------------------------------------------------

def build_joint_rfe_training_dataframe(
    target_train_df: pd.DataFrame,
    selected_source_dfs: List[pd.DataFrame],
    feature_cols: Sequence[str],
    target_col: str = "sales",
) -> pd.DataFrame:
    """
    将 target_train_df 与多个 selected_source 的训练部分拼接，用于 RFE 拟合。

    Args:
        target_train_df: Target 训练集 DataFrame
        selected_source_dfs: 多个 source 训练集 DataFrame 列表
        feature_cols: 用于 RFE 的特征列
        target_col: 目标列名

    Returns:
        联合 DataFrame，包含 feature_cols + [target_col]

    要求：
        - 只用于 RFE 拟合，不用于直接训练 CNN
        - 不引入 target test/val 泄露
    """
    logger = _get_logger()
    logger.info(
        "[build_joint_rfe_training_dataframe] Start. "
        "target_rows=%d num_sources=%d",
        len(target_train_df), len(selected_source_dfs),
    )

    cols_to_keep = list(dict.fromkeys(list(feature_cols) + [target_col]))
    dfs_to_concat = [target_train_df[cols_to_keep].copy()]

    for i, src_df in enumerate(selected_source_dfs):
        if src_df.empty:
            logger.warning("[build_joint_rfe_training_dataframe] Source %d is empty, skipping", i)
            continue
        dfs_to_concat.append(src_df[cols_to_keep].copy())

    joint_df = pd.concat(dfs_to_concat, axis=0, ignore_index=True)

    logger.info(
        "[build_joint_rfe_training_dataframe] Finished. total_rows=%d",
        len(joint_df),
    )
    return joint_df


# -------------------------------------------------------------------
# 3. apply_selected_features_to_df
# -------------------------------------------------------------------

def apply_selected_features_to_df(
    df: pd.DataFrame,
    selected_feature_cols: Sequence[str],
    required_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    将 DataFrame 限制到 selected_feature_cols，并保留必要字段。

    Args:
        df: 输入 DataFrame
        selected_feature_cols: 已选特征列
        required_cols: 需要保留的额外列（如 date/entity_id/item_id/sales）

    Returns:
        裁剪后的 DataFrame

    要求：
        - 保证后续 build_tabular_sequence 仍可运行
        - sales 作为标签列必须保留
        - 如果缺失列，报明确错误
    """
    logger = _get_logger()

    selected_features = list(selected_feature_cols)
    extra_cols = list(required_cols) if required_cols else []

    # 确保 sales 在 required_cols 中
    if "sales" not in extra_cols:
        extra_cols = ["sales"] + extra_cols

    # 移除重复
    all_cols = list(dict.fromkeys(selected_features + extra_cols))

    # 检查列是否存在
    missing = [c for c in all_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in DataFrame: {missing}")

    result_df = df[all_cols].copy()

    logger.info(
        "[apply_selected_features_to_df] Finished. "
        "input_cols=%d output_cols=%d output_shape=%s",
        len(df.columns), len(all_cols), result_df.shape,
    )
    return result_df


# -------------------------------------------------------------------
# 4. train_source_cnn_for_msml_rfe
# -------------------------------------------------------------------

def train_source_cnn_for_msml_rfe(
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
    与模块8中的 train_source_cnn_for_msml 类似，但使用 RFE 后的特征。

    Args:
        source_sequence_df: 单个 source 序列 DataFrame（已应用 RFE 特征）
        feature_cols: 特征列名列表（已是 RFE 选出的）
        horizon: 预测步长
        window_size: 滑窗大小
        learning_rate: Adam 学习率
        source_epochs: 训练轮数
        batch_size: 批大小
        source_key: 可选的 source 标识

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
        "[train_source_cnn_for_msml_rfe] Start. source_key=%s epochs=%d batch_size=%d",
        source_key, source_epochs, batch_size,
    )

    _validate_feature_cols(source_sequence_df, feature_cols, where="source_sequence_df")

    src_train, src_val, src_test = _prepare_source_split(source_sequence_df)
    src_train, src_val, src_test, _, _ = normalize_features(src_train, src_val, src_test)

    X_source, y_source = build_tabular_sequence(src_train, horizon=horizon, window_size=window_size)
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
        "[train_source_cnn_for_msml_rfe] Finished. source_key=%s input_shape=%s num_samples=%d",
        source_key, input_shape, len(y_source),
    )
    return {
        "model": model,
        "input_shape": tuple(input_shape),
        "num_samples": len(y_source),
        "source_key": source_key,
    }


# -------------------------------------------------------------------
# 5. fine_tune_fused_target_model_rfe
# -------------------------------------------------------------------

def fine_tune_fused_target_model_rfe(
    target_model,
    target_train_df: pd.DataFrame,
    target_val_df: pd.DataFrame,
    feature_cols: Sequence[str],
    horizon: int = 1,
    window_size: int = 10,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 0.001,
) -> Dict[str, object]:
    """
    使用 RFE 后的 target train/val 数据微调 target model。

    Args:
        target_model: 已加载融合参数并冻结部分层的 Keras 模型
        target_train_df: 归一化后的 target 训练集（RFE 特征版本）
        target_val_df: 归一化后的 target 验证集（RFE 特征版本）
        feature_cols: 特征列名列表（RFE 选出的）
        horizon: 预测步长
        window_size: 滑窗大小
        epochs: 微调轮数
        batch_size: 批大小
        learning_rate: 微调学习率

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
        "[fine_tune_fused_target_model_rfe] Start. epochs=%d batch_size=%d",
        epochs, batch_size,
    )

    X_train, y_train = build_tabular_sequence(target_train_df, horizon=horizon, window_size=window_size)
    X_val, y_val = build_tabular_sequence(target_val_df, horizon=horizon, window_size=window_size)

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

    logger.info("[fine_tune_fused_target_model_rfe] Fine-tuning completed.")
    return {
        "model": target_model,
        "history": history,
        "frozen_layers": frozen_layers,
    }


# -------------------------------------------------------------------
# 6. evaluate_msml_rfe_model
# -------------------------------------------------------------------

def evaluate_msml_rfe_model(
    target_model,
    target_test_df: pd.DataFrame,
    feature_cols: Sequence[str],
    horizon: int = 1,
    window_size: int = 10,
    eps: float = 1e-8,
) -> Dict[str, object]:
    """
    使用 RFE 后的 target test 数据评估模型。

    Args:
        target_model: 微调后的 Keras 模型
        target_test_df: 归一化后的 target 测试集（RFE 特征版本）
        feature_cols: 特征列名列表（RFE 选出的）
        horizon: 预测步长
        window_size: 滑窗大小
        eps: 数值稳定项

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
    logger.info("[evaluate_msml_rfe_model] Start.")

    X_test, y_test = build_tabular_sequence(target_test_df, horizon=horizon, window_size=window_size)
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

    logger.info("[evaluate_msml_rfe_model] Finished. RMSE=%.4f Accuracy=%.4f", rmse, accuracy)
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
# 7. run_msml_tl_rfe (主函数)
# -------------------------------------------------------------------

def run_msml_tl_rfe(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    feature_cols: Sequence[str],
    k: int = 3,
    horizon: int = 1,
    window_size: int = 10,
    weight_mode: str = "inverse_distance",
    estimator_name: str = "random_forest",
    keep_ratio: float = 0.5,
    learning_rate: float = 0.001,
    source_epochs: int = 3,
    target_epochs: int = 3,
    batch_size: int = 16,
    random_state: int = 42,
) -> Dict[str, object]:
    """
    运行 MSML-TL-RFE 完整流程。

    步骤：
    1. 选出 top-k 相似源
    2. 切分 target_df 为 train/val/test
    3. 对各 selected source 提取对应的 source sequence
    4. 构建联合 RFE 训练数据（target_train + selected_sources_train）
    5. 执行 RFE，选出特征子集
    6. 将 selected_feature_cols 应用到 target/selected_sources 的所有部分
    7. 对每个 selected source 训练 CNN（使用 RFE 特征）
    8. 提取 source 权重
    9. 对指定层做加权参数融合
    10. 构建 target model，加载融合参数，冻结融合层
    11. 微调 target model
    12. 在 target test 上评估

    Args:
        source_df: Source pool DataFrame
        target_df: Target DataFrame
        feature_cols: 候选特征列名列表
        k: 选择的相似源数量
        horizon: 预测步长
        window_size: 滑窗大小
        weight_mode: 权重模式
        estimator_name: RFE 评估器名称
        keep_ratio: RFE 保留特征的比例
        learning_rate: 学习率
        source_epochs: Source 训练轮数
        target_epochs: Target 微调轮数
        batch_size: 批大小
        random_state: 随机种子

    Returns:
        {
            "meta": {
                "method": "MSML-TL-RFE",
                "k": int,
                "weight_mode": str,
                "feature_cols": list,
                "selected_feature_cols": list,
                "keep_ratio": float,
                "selected_sources": list,
                "fused_layers": list,
            },
            "rfe_info": {
                "selected_feature_cols": list,
                "num_selected_features": int,
                "num_original_features": int,
                "keep_ratio": float,
            },
            "source_models_info": [
                {
                    "source_key": tuple,
                    "distance": float,
                    "weight": float,
                },
                ...
            ],
            "fused_result": {
                "rmse": float,
                "accuracy": float,
                "prediction_shape": tuple,
            },
            "frozen_layers": [...],
        }
    """
    logger = _get_logger()
    logger.info(
        "[run_msml_tl_rfe] Start. k=%d weight_mode=%s keep_ratio=%.2f estimator=%s",
        k, weight_mode, keep_ratio, estimator_name,
    )

    _validate_feature_cols(source_df, feature_cols, where="source_df")
    _validate_feature_cols(target_df, feature_cols, where="target_df")

    # --- Step 1: 选源 ---
    selector = SourceSelector()
    selection_result = selector.select_top_k_sources(
        target_df=target_df,
        source_df=source_df,
        feature_cols=feature_cols,
        k=k,
        weight_mode=weight_mode,
    )
    selected_sources = selection_result.get("sources", []) if isinstance(selection_result, dict) else selection_result
    if not selected_sources:
        raise ValueError("No source selected from source pool.")

    logger.info("[run_msml_tl_rfe] Step 1: Selected %d sources", len(selected_sources))

    # --- Step 2: 切分 target ---
    target_train_df, target_val_df, target_test_df = temporal_split_by_ratio_or_dates(target_df)
    logger.info(
        "[run_msml_tl_rfe] Step 2: Split target. train=%d val=%d test=%d",
        len(target_train_df), len(target_val_df), len(target_test_df),
    )

    # --- Step 3: 提取 selected source sequences ---
    selected_source_train_dfs: List[pd.DataFrame] = []
    selected_source_sequences: Dict[Tuple, pd.DataFrame] = {}
    selected_source_keys: List[Tuple] = []

    for selected in selected_sources:
        source_key = tuple(selected["source_key"]) if isinstance(selected["source_key"], (list, tuple)) else (selected["source_key"],)
        if len(source_key) < 2:
            raise ValueError(f"Invalid source_key format: {source_key}")

        entity_id, item_id = source_key[0], source_key[1]
        source_sequence_df = source_df[
            (source_df["entity_id"] == entity_id) & (source_df["item_id"] == item_id)
        ].copy()

        if source_sequence_df.empty:
            raise ValueError(f"Selected source_key not found in source_df: {source_key}")

        selected_source_sequences[source_key] = source_sequence_df
        selected_source_keys.append(source_key)

        # 切分 source，提取 train 用于 RFE
        src_train, _, _ = _prepare_source_split(source_sequence_df)
        selected_source_train_dfs.append(src_train)

    logger.info("[run_msml_tl_rfe] Step 3: Extracted %d selected source sequences", len(selected_source_sequences))

    # --- Step 4: 构建联合 RFE 训练数据 ---
    joint_train_df = build_joint_rfe_training_dataframe(
        target_train_df=target_train_df,
        selected_source_dfs=selected_source_train_dfs,
        feature_cols=feature_cols,
        target_col="sales",
    )
    logger.info("[run_msml_tl_rfe] Step 4: Built joint RFE training data. rows=%d", len(joint_train_df))

    # --- Step 5: 执行 RFE ---
    rfe_result = run_rfe_feature_selection(
        train_df=joint_train_df,
        feature_cols=feature_cols,
        target_col="sales",
        estimator_name=estimator_name,
        keep_ratio=keep_ratio,
        random_state=random_state,
    )
    selected_feature_cols = rfe_result["selected_feature_cols"]
    logger.info(
        "[run_msml_tl_rfe] Step 5: RFE completed. "
        "original=%d selected=%d selected_cols=%s",
        rfe_result["num_original_features"],
        rfe_result["num_selected_features"],
        selected_feature_cols,
    )

    # --- Step 6: 应用 RFE 特征到 target 和 selected sources ---
    # 必要的非特征列
    required_cols = ["date", "entity_id", "item_id", "sales"]
    required_cols = [c for c in required_cols if c in target_train_df.columns]

    target_train_df_rfe = apply_selected_features_to_df(
        target_train_df, selected_feature_cols, required_cols=required_cols
    )
    target_val_df_rfe = apply_selected_features_to_df(
        target_val_df, selected_feature_cols, required_cols=required_cols
    )
    target_test_df_rfe = apply_selected_features_to_df(
        target_test_df, selected_feature_cols, required_cols=required_cols
    )

    selected_source_sequences_rfe: Dict[Tuple, pd.DataFrame] = {}
    for source_key, source_seq_df in selected_source_sequences.items():
        selected_source_sequences_rfe[source_key] = apply_selected_features_to_df(
            source_seq_df, selected_feature_cols, required_cols=required_cols
        )

    logger.info("[run_msml_tl_rfe] Step 6: Applied RFE features to all targets and sources")

    # --- Step 7: 对每个 selected source 训练 CNN（使用 RFE 特征）---
    source_models: List = []
    source_weights: List[float] = []
    source_models_info: List[Dict[str, object]] = []
    input_shape_ref: Optional[Tuple[int, ...]] = None
    failed_sources: List[Dict[str, object]] = []

    for source_key in selected_source_keys:
        source_sequence_df_rfe = selected_source_sequences_rfe[source_key]

        try:
            train_result = train_source_cnn_for_msml_rfe(
                source_sequence_df=source_sequence_df_rfe,
                feature_cols=selected_feature_cols,
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
            if not should_skip_source_exception(exc):
                raise
            failed_source = make_failed_source(source_key, exc)
            failed_sources.append(failed_source)
            logger.warning(
                "[run_msml_tl_rfe] Skipping failed source_key=%s exception_type=%s message=%s",
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

        # 从 selected_sources 中找对应权重
        weight_for_this_source = None
        for selected in selected_sources:
            selected_key = tuple(selected["source_key"]) if isinstance(selected["source_key"], (list, tuple)) else (selected["source_key"],)
            if selected_key == source_key:
                weight_for_this_source = float(selected["weight"])
                break

        if weight_for_this_source is None:
            raise ValueError(f"Could not find weight for source_key={source_key}")

        source_weights.append(weight_for_this_source)
        source_models_info.append({
            "source_key": source_key,
            "distance": float(next(s["distance"] for s in selected_sources if tuple(s["source_key"]) == source_key)),
            "weight": weight_for_this_source,
            "num_samples": int(train_result["num_samples"]),
        })

    if not source_models:
        raise AllSourcesFailedError("MSML-TL-RFE", failed_sources, selected_sources=selected_sources)

    source_weights = normalize_successful_source_weights(source_weights)
    for info, normalized_weight in zip(source_models_info, source_weights):
        info["weight"] = float(normalized_weight)

    logger.info("[run_msml_tl_rfe] Step 7: Trained %d source CNN models", len(source_models))

    # --- Step 8: 确定可融合层并做加权参数融合 ---
    layer_names = get_transferable_layer_names(source_models[0])
    fused_params = fuse_source_models_layerwise(source_models, source_weights, layer_names)
    logger.info(
        "[run_msml_tl_rfe] Step 8: Fused source models. layers=%s",
        list(layer_names),
    )

    # --- Step 9: 构建 target model → 加载融合参数 → 冻结 ---
    target_model = build_base_cnn(input_shape_ref, learning_rate=learning_rate)
    target_model = load_fused_params_into_target_model(target_model, fused_params)
    frozen_layers = freeze_fused_layers(target_model, layer_names)
    logger.info("[run_msml_tl_rfe] Step 9: Loaded fused params and froze layers")

    # --- Step 10: 归一化 target 数据 ---
    target_train_df_rfe, target_val_df_rfe, target_test_df_rfe, target_scaler, target_feature_columns = normalize_features(
        target_train_df_rfe, target_val_df_rfe, target_test_df_rfe,
    )
    logger.info("[run_msml_tl_rfe] Step 10: Normalized target features")

    # --- Step 11: 微调 target model ---
    ft_result = fine_tune_fused_target_model_rfe(
        target_model=target_model,
        target_train_df=target_train_df_rfe,
        target_val_df=target_val_df_rfe,
        feature_cols=selected_feature_cols,
        horizon=horizon,
        window_size=window_size,
        learning_rate=learning_rate,
        epochs=target_epochs,
        batch_size=batch_size,
    )
    logger.info("[run_msml_tl_rfe] Step 11: Fine-tuned target model")

    # --- Step 12: 在 target test 上评估 ---
    eval_result = evaluate_msml_rfe_model(
        target_model=ft_result["model"],
        target_test_df=target_test_df_rfe,
        feature_cols=selected_feature_cols,
        horizon=horizon,
        window_size=window_size,
    )
    logger.info(
        "[run_msml_tl_rfe] Step 12: Evaluated model. "
        "RMSE=%.4f Accuracy=%.4f",
        eval_result["rmse"], eval_result["accuracy"],
    )

    logger.info("[run_msml_tl_rfe] Finished successfully")

    return {
        "meta": {
            "method": "MSML-TL-RFE",
            "k": int(k),
            "weight_mode": weight_mode,
            "feature_cols": list(feature_cols),
            "selected_feature_cols": selected_feature_cols,
            "keep_ratio": float(keep_ratio),
            "selected_sources": selected_sources,
            "fused_layers": list(layer_names),
            **source_failure_meta(
                requested_k=k,
                selected_sources=selected_sources,
                valid_source_count=len(source_models_info),
                failed_sources=failed_sources,
            ),
        },
        "rfe_info": {
            "selected_feature_cols": selected_feature_cols,
            "num_selected_features": rfe_result["num_selected_features"],
            "num_original_features": rfe_result["num_original_features"],
            "keep_ratio": float(keep_ratio),
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
