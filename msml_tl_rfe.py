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
    from environment import setup_logging
except ImportError:
    setup_logging = None

from cnn_model import (
    build_base_cnn,
    EarlyStoppingConfig,
    create_training_callbacks,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_EARLY_STOPPING_MIN_DELTA,
)
from data_preprocessing import (
    build_tabular_sequence,
    is_identifier_like_feature_column,
    normalize_features,
    temporal_split_by_ratio_or_dates,
    to_cnn_tensor,
)
from src.evaluation.metrics import compute_metrics_with_protocol
from src.utils.experiment_hyperparams import FIXED_EPOCHS, FIXED_LEARNING_RATE, fixed_hyperparams_summary
from src.utils.source_fillna import fill_source_numeric_na
from source_selector import SourceSelector
from msml_tl import (
    get_transferable_layer_names,
    extract_layer_params,
    weighted_average_layer_params,
    fuse_source_models_layerwise,
    load_fused_params_into_target_model,
    freeze_fused_layers,
)
from src.utils.runtime_control import keras_verbose

LOGGER_NAME = "experiment"

# 默认需要迁移冻结的前 4 层；pool 层无权重，只参与冻结，不参与参数融合。
_DEFAULT_TRANSFERABLE_LAYERS = ["conv1", "pool1", "conv2", "pool2"]
_DEFAULT_FUSION_LAYERS = ["conv1", "conv2"]


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


def dedupe_preserve_order(cols: Sequence[str]) -> List[str]:
    """Remove duplicate column labels while preserving first-seen order."""
    seen = set()
    out: List[str] = []
    for col in cols:
        if col not in seen:
            out.append(col)
            seen.add(col)
    return out


def filter_model_input_feature_cols(feature_cols: Sequence[str]) -> List[str]:
    """Remove ID/code fields from modeling and RFE feature lists."""
    return [str(col) for col in dedupe_preserve_order(feature_cols) if not is_identifier_like_feature_column(str(col))]


def _duplicate_labels(cols: Sequence[str]) -> List[str]:
    """Return duplicate labels after their first occurrence."""
    return pd.Index(list(cols))[pd.Index(list(cols)).duplicated()].tolist()


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


def _normalize_source_key(source_key_raw: object) -> Tuple[object, object]:
    """Normalize source key to a strict (entity_id, item_id) tuple."""
    if isinstance(source_key_raw, tuple):
        key = source_key_raw
    elif isinstance(source_key_raw, list):
        key = tuple(source_key_raw)
    else:
        key = (source_key_raw,)

    if len(key) < 2:
        raise ValueError(f"Invalid source_key format (expected len>=2): {source_key_raw}")
    return key[0], key[1]


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
    use_sales_as_history_input: bool = True,
) -> Dict[str, object]:
    """
    对输入 train_df 执行 RFE，选出保留特征。

    Args:
        train_df: 用于 RFE 拟合的训练 DataFrame
        feature_cols: 原始候选特征列名
        target_col: 目标列名
        estimator_name: 评估器名称，支持 'random_forest', 'linear_regression' 等
        keep_ratio: 保留特征的比例 (0, 1]
        random_state: 随机种子
        use_sales_as_history_input: 若为 True，在 RFE 后将 sales 作为历史窗口输入加回

    Returns:
        {
            "rfe_selected_features": [...],
            "final_selected_features": [...],
            "selected_feature_cols": [...],  # backward-compatible alias for final_selected_features
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

    original_feature_cols = list(feature_cols)
    duplicate_feature_cols_before = _duplicate_labels(original_feature_cols)
    cols = dedupe_preserve_order(_validate_feature_cols(train_df, feature_cols, where="train_df"))
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in train_df")

    id_code_removed_from_rfe = [c for c in cols if c != target_col and is_identifier_like_feature_column(c)]
    rfe_candidate_cols = [
        c
        for c in cols
        if c != target_col and not is_identifier_like_feature_column(c)
    ]
    target_removed_from_rfe = target_col in cols and target_col not in rfe_candidate_cols
    if not rfe_candidate_cols:
        raise ValueError(
            "No RFE candidate features remain after removing the target and ID/code columns. "
            f"target_col={target_col!r} original_feature_cols={original_feature_cols}"
        )

    assert target_col not in rfe_candidate_cols

    X = train_df[rfe_candidate_cols].copy()
    duplicate_x_cols = X.columns[X.columns.duplicated()].tolist()
    if duplicate_x_cols:
        raise ValueError(f"Duplicate columns in RFE X: {duplicate_x_cols}")

    contains_target_in_rfe_X = target_col in X.columns
    assert target_col not in X.columns

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

    # 计算保留特征数：必须基于不含当前目标列的 RFE 候选特征。
    num_original = len(cols)
    num_candidates = len(rfe_candidate_cols)
    num_to_select_requested = max(1, int(np.ceil(num_candidates * keep_ratio)))
    selection_adjusted_warning = ""
    if num_to_select_requested > num_candidates:
        selection_adjusted_warning = (
            f"n_features_to_select adjusted from {num_to_select_requested} "
            f"to {num_candidates} because only {num_candidates} RFE candidates remain."
        )
        logger.warning("[run_rfe_feature_selection] %s", selection_adjusted_warning)
    num_to_select = min(num_to_select_requested, num_candidates)

    logger.info(
        "[run_rfe_feature_selection] Feature count. original=%d candidates=%d to_select=%d target_removed=%s",
        num_original, num_candidates, num_to_select, target_removed_from_rfe,
    )

    # 执行 RFE
    rfe = RFE(estimator=estimator, n_features_to_select=num_to_select, step=1)
    try:
        rfe.fit(X, y)
        selected_indices = np.where(rfe.support_)[0]
    except IndexError as exc:
        # 最小兜底：当 sklearn RFE 在较大样本组合下触发越界时，
        # 使用同一估计器的特征重要性做等价的前 num_to_select 选择。
        logger.warning(
            "[run_rfe_feature_selection] RFE failed with IndexError, fallback to estimator importance. error=%s",
            exc,
        )
        estimator.fit(X, y)
        if hasattr(estimator, "feature_importances_"):
            importance = np.asarray(estimator.feature_importances_, dtype=np.float64)
            selected_indices = np.argsort(importance)[::-1][:num_to_select]
        elif hasattr(estimator, "coef_"):
            coef = np.asarray(estimator.coef_, dtype=np.float64).reshape(-1)
            selected_indices = np.argsort(np.abs(coef))[::-1][:num_to_select]
        else:
            selected_indices = np.arange(num_to_select, dtype=np.int64)

    # 提取已选特征（做越界保护，防止第三方库异常返回非法下标）
    selected_indices = np.asarray(selected_indices, dtype=np.int64).reshape(-1)
    selected_indices = selected_indices[(selected_indices >= 0) & (selected_indices < num_candidates)]

    if selected_indices.size == 0:
        raise ValueError("RFE selected no valid feature indices.")

    # 去重并尽量保持原顺序
    selected_indices = np.asarray(list(dict.fromkeys(selected_indices.tolist())), dtype=np.int64)

    # 若不足目标数量，按原始列顺序补齐（最小兜底，避免后续索引错误）
    if selected_indices.size < num_to_select:
        remaining = [i for i in range(num_candidates) if i not in set(selected_indices.tolist())]
        needed = num_to_select - int(selected_indices.size)
        selected_indices = np.concatenate([selected_indices, np.asarray(remaining[:needed], dtype=np.int64)])

    # 若超出目标数量，截断到目标数量
    selected_indices = selected_indices[:num_to_select]
    rfe_selected_features = [rfe_candidate_cols[int(i)] for i in selected_indices]
    if target_col in rfe_selected_features:
        raise ValueError(f"RFE selected the target column unexpectedly: {target_col}")

    sales_added_back_as_history_input = (
        bool(use_sales_as_history_input)
        and target_col == "sales"
        and "sales" in cols
        and "sales" in train_df.columns
    )
    final_selected_features = list(rfe_selected_features)
    if sales_added_back_as_history_input:
        final_selected_features = ["sales"] + final_selected_features
    final_selected_features = dedupe_preserve_order(final_selected_features)

    logger.info(
        "[run_rfe_feature_selection] Finished. "
        "rfe_selected=%d rfe_selected_cols=%s final_selected_cols=%s",
        len(rfe_selected_features), rfe_selected_features, final_selected_features,
    )

    rfe_y_shape = tuple(np.asarray(y).shape)
    return {
        "original_feature_cols": original_feature_cols,
        "feature_cols": original_feature_cols,
        "rfe_candidate_cols": rfe_candidate_cols,
        "rfe_candidate_features": rfe_candidate_cols,
        "rfe_selected_features": rfe_selected_features,
        "rfe_selected_feature_cols": rfe_selected_features,
        "final_selected_features": final_selected_features,
        "final_model_feature_cols": final_selected_features,
        "selected_feature_cols": final_selected_features,
        "selected_features": final_selected_features,
        "num_selected_features": len(final_selected_features),
        "num_rfe_selected_features": len(rfe_selected_features),
        "num_original_features": num_original,
        "num_rfe_candidate_features": num_candidates,
        "n_features_to_select": int(num_to_select),
        "n_features_to_select_requested": int(num_to_select_requested),
        "keep_ratio": float(keep_ratio),
        "target_col": target_col,
        "target_removed_from_rfe": bool(target_removed_from_rfe),
        "id_code_removed_from_rfe": id_code_removed_from_rfe,
        "sales_excluded_from_rfe": bool(target_col == "sales" and target_removed_from_rfe),
        "sales_added_back_as_history_input": bool(sales_added_back_as_history_input),
        "duplicate_columns_before": duplicate_feature_cols_before,
        "duplicate_columns_after": duplicate_x_cols,
        "duplicate_sales_before": int(max(0, original_feature_cols.count(target_col) - 1)),
        "duplicate_sales_after": 0,
        "contains_target_in_rfe_X": bool(contains_target_in_rfe_X),
        "contains_duplicate_sales_in_joint_df": False,
        "rfe_input_shape": tuple(X.shape),
        "rfe_y_shape": rfe_y_shape,
        "rfe_estimator": estimator.__class__.__name__,
        "random_state": int(random_state),
        "selection_adjusted_warning": selection_adjusted_warning,
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

    cols_to_keep_raw = list(feature_cols) + [target_col]
    duplicate_columns_before = _duplicate_labels(cols_to_keep_raw)
    duplicate_sales_before = max(0, cols_to_keep_raw.count(target_col) - 1)
    cols_to_keep = dedupe_preserve_order(cols_to_keep_raw)

    missing_target = [c for c in cols_to_keep if c not in target_train_df.columns]
    if missing_target:
        raise ValueError(f"Missing columns in target_train_df for joint RFE dataframe: {missing_target}")

    dfs_to_concat = [target_train_df[cols_to_keep].copy()]

    for i, src_df in enumerate(selected_source_dfs):
        if src_df.empty:
            logger.warning("[build_joint_rfe_training_dataframe] Source %d is empty, skipping", i)
            continue
        missing_source = [c for c in cols_to_keep if c not in src_df.columns]
        if missing_source:
            raise ValueError(
                f"Missing columns in source {i} for joint RFE dataframe: {missing_source}"
            )
        dfs_to_concat.append(src_df[cols_to_keep].copy())

    joint_df = pd.concat(dfs_to_concat, axis=0, ignore_index=True)
    duplicate_columns_after = joint_df.columns[joint_df.columns.duplicated()].tolist()
    if duplicate_columns_after:
        raise ValueError(f"Duplicate columns in joint RFE dataframe: {duplicate_columns_after}")

    duplicate_sales_after = max(0, list(joint_df.columns).count(target_col) - 1)
    contains_duplicate_sales_in_joint_df = duplicate_sales_after > 0
    joint_df.attrs["rfe_audit"] = {
        "duplicate_columns_before": duplicate_columns_before,
        "duplicate_columns_after": duplicate_columns_after,
        "duplicate_sales_before": int(duplicate_sales_before),
        "duplicate_sales_after": int(duplicate_sales_after),
        "contains_duplicate_sales_in_joint_df": bool(contains_duplicate_sales_in_joint_df),
        "cols_to_keep": cols_to_keep,
        "target_col": target_col,
    }

    logger.info(
        "[build_joint_rfe_training_dataframe] Finished. total_rows=%d duplicate_sales_before=%d duplicate_sales_after=%d",
        len(joint_df), duplicate_sales_before, duplicate_sales_after,
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

    selected_features = dedupe_preserve_order(list(selected_feature_cols))
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
    result_df.attrs = df.attrs.copy()
    result_df.attrs["model_feature_cols"] = list(selected_features)

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
    learning_rate: float = FIXED_LEARNING_RATE,
    source_epochs: int = FIXED_EPOCHS,
    batch_size: int = 16,
    source_key: Optional[Tuple] = None,
    early_stopping_enabled: bool = True,
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    early_stopping_min_delta: float = DEFAULT_EARLY_STOPPING_MIN_DELTA,
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
        early_stopping_enabled: 是否启用早停
        early_stopping_patience: 早停耐心值
        early_stopping_min_delta: 早停最小改进阈值

    Returns:
        {
            "model": 训练好的 Keras 模型,
            "input_shape": (window_size, num_features),
            "num_samples": int,
            "source_key": tuple or None,
            "early_stopped": bool,
            "stopped_epoch": int,
            "best_val_loss": float,
        }
    """
    logger = _get_logger()
    logger.info(
        "[train_source_cnn_for_msml_rfe] Start. source_key=%s batch_size=%d early_stopping=%s hyperparams=%s. clipnorm=None means gradient clipping is disabled.",
        source_key, batch_size, early_stopping_enabled, fixed_hyperparams_summary(),
    )

    feature_cols = filter_model_input_feature_cols(feature_cols)
    _validate_feature_cols(source_sequence_df, feature_cols, where="source_sequence_df")

    src_train, src_val, src_test = _prepare_source_split(source_sequence_df)
    src_train = fill_source_numeric_na(src_train)
    src_val = fill_source_numeric_na(src_val)
    src_test = fill_source_numeric_na(src_test)
    src_train, src_val, src_test, _, _ = normalize_features(
        src_train,
        src_val,
        src_test,
        feature_cols=feature_cols,
    )

    X_train, y_train = build_tabular_sequence(
        src_train,
        horizon=horizon,
        window_size=window_size,
        feature_cols=feature_cols,
    )
    if len(y_train) == 0:
        raise ValueError(
            f"Source sequence (key={source_key}) produced zero training windows; "
            "adjust window_size/horizon."
        )

    # 构建验证数据用于早停
    X_val, y_val = build_tabular_sequence(
        src_val,
        horizon=horizon,
        window_size=window_size,
        feature_cols=feature_cols,
    )

    X_train = to_cnn_tensor(X_train)
    input_shape = X_train.shape[1:]

    model = build_base_cnn(input_shape, learning_rate=learning_rate)

    # 配置早停回调
    callbacks = []
    early_stopping_config = None
    if early_stopping_enabled and len(y_val) > 0:
        early_stopping_config = EarlyStoppingConfig(
            enabled=True,
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
            monitor="val_loss",
            restore_best_weights=True,
        )
        callbacks = create_training_callbacks(early_stopping_config=early_stopping_config)

    # 训练模型
    validation_data = None
    if len(y_val) > 0:
        X_val = to_cnn_tensor(X_val)
        validation_data = (X_val, y_val)

    history = model.fit(
        X_train,
        y_train,
        epochs=source_epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        callbacks=callbacks if callbacks else None,
        verbose=keras_verbose(),
    )

    # 检查是否早停
    early_stopped = False
    stopped_epoch = source_epochs
    best_val_loss = float("nan")

    if callbacks and len(callbacks) > 0:
        for cb in callbacks:
            if hasattr(cb, "stopped_epoch") and cb.stopped_epoch > 0:
                early_stopped = True
                stopped_epoch = cb.stopped_epoch + 1
                break

    if validation_data is not None and "val_loss" in history.history:
        best_val_loss = float(min(history.history["val_loss"]))

    logger.info(
        "[train_source_cnn_for_msml_rfe] Finished. source_key=%s input_shape=%s num_samples=%d early_stopped=%s stopped_epoch=%d",
        source_key, input_shape, len(y_train), early_stopped, stopped_epoch,
    )
    return {
        "model": model,
        "input_shape": tuple(input_shape),
        "num_samples": len(y_train),
        "source_key": source_key,
        "early_stopped": early_stopped,
        "stopped_epoch": stopped_epoch,
        "best_val_loss": best_val_loss,
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
    epochs: int = FIXED_EPOCHS,
    batch_size: int = 16,
    learning_rate: float = FIXED_LEARNING_RATE,
    early_stopping_enabled: bool = True,
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    early_stopping_min_delta: float = DEFAULT_EARLY_STOPPING_MIN_DELTA,
) -> Dict[str, object]:
    """
    使用 RFE 后的 target train/val 数据微调 target model。

    Args:
        target_model: 已加载融合参数的 Keras 模型
        target_train_df: 归一化后的训练数据
        target_val_df: 归一化后的验证数据
        feature_cols: 特征列
        horizon: 预测步长
        window_size: 滑窗大小
        epochs: 最大训练轮数
        batch_size: 批大小
        learning_rate: 学习率
        early_stopping_enabled: 是否启用早停
        early_stopping_patience: 早停耐心值
        early_stopping_min_delta: 早停最小改进阈值

    Returns:
        包含模型、历史记录、早停信息的字典
    """
    import tensorflow as tf

    logger = _get_logger()
    logger.info(
        "[fine_tune_fused_target_model_rfe] Start. batch_size=%d early_stopping=%s hyperparams=%s. clipnorm=None means gradient clipping is disabled.",
        batch_size, early_stopping_enabled, fixed_hyperparams_summary(),
    )

    X_train, y_train = build_tabular_sequence(
        target_train_df,
        horizon=horizon,
        window_size=window_size,
        feature_cols=feature_cols,
    )
    X_val, y_val = build_tabular_sequence(
        target_val_df,
        horizon=horizon,
        window_size=window_size,
        feature_cols=feature_cols,
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

    # 配置早停回调
    callbacks = []
    if early_stopping_enabled:
        early_stopping_config = EarlyStoppingConfig(
            enabled=True,
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
            monitor="val_loss",
            restore_best_weights=True,
        )
        callbacks = create_training_callbacks(early_stopping_config=early_stopping_config)

    history = target_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks if callbacks else None,
        verbose=keras_verbose(),
    )

    # 检查是否早停
    early_stopped = False
    stopped_epoch = epochs
    best_val_loss = float("nan")

    if callbacks and len(callbacks) > 0:
        for cb in callbacks:
            if hasattr(cb, "stopped_epoch") and cb.stopped_epoch > 0:
                early_stopped = True
                stopped_epoch = cb.stopped_epoch + 1
                break

    if "val_loss" in history.history:
        best_val_loss = float(min(history.history["val_loss"]))

    logger.info(
        "[fine_tune_fused_target_model_rfe] Fine-tuning completed. early_stopped=%s stopped_epoch=%d best_val_loss=%.6f",
        early_stopped, stopped_epoch, best_val_loss,
    )
    return {
        "model": target_model,
        "history": history,
        "frozen_layers": frozen_layers,
        "early_stopped": early_stopped,
        "stopped_epoch": stopped_epoch,
        "best_val_loss": best_val_loss,
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
    metric_protocol: dict | None = None,
    sales_scaler: object | None = None,
    feature_columns_for_scaler: object | None = None,
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

    X_test, y_test = build_tabular_sequence(
        target_test_df,
        horizon=horizon,
        window_size=window_size,
        feature_cols=feature_cols,
    )
    if len(y_test) == 0:
        raise ValueError("Target test split produced zero windows; adjust window_size/horizon.")

    X_test = to_cnn_tensor(X_test)

    y_pred = target_model.predict(X_test, verbose=0)
    y_true = y_test.flatten()
    metric_result = compute_metrics_with_protocol(
        y_true=y_true,
        y_pred=y_pred,
        metric_protocol=metric_protocol,
        sales_scaler=sales_scaler,
        feature_columns=feature_columns_for_scaler,
        eps=eps,
    )

    logger.info(
        "[evaluate_msml_rfe_model] Finished. RMSE=%.4f Accuracy=%.4f",
        float(metric_result["rmse"]),
        float(metric_result["accuracy"]),
    )
    return {
        "rmse": float(metric_result["rmse"]),
        "accuracy": float(metric_result["accuracy"]),
        "mae": float(metric_result.get("mae", float("nan"))),
        "mape": float(metric_result.get("mape", float("nan"))),
        "smape": float(metric_result.get("smape", float("nan"))),
        "rmse_current": float(metric_result.get("rmse_current", float("nan"))),
        "accuracy_current": float(metric_result.get("accuracy_current", float("nan"))),
        "mae_current": float(metric_result.get("mae_current", float("nan"))),
        "mape_current": float(metric_result.get("mape_current", float("nan"))),
        "smape_current": float(metric_result.get("smape_current", float("nan"))),
        "rmse_paper": float(metric_result.get("rmse_paper", float("nan"))),
        "accuracy_paper": float(metric_result.get("accuracy_paper", float("nan"))),
        "mae_paper": float(metric_result.get("mae_paper", float("nan"))),
        "mape_paper": float(metric_result.get("mape_paper", float("nan"))),
        "smape_paper": float(metric_result.get("smape_paper", float("nan"))),
        "normalized_rmse": float(metric_result.get("normalized_rmse", metric_result.get("rmse", float("nan")))),
        "normalized_accuracy": float(metric_result.get("normalized_accuracy", metric_result.get("accuracy", float("nan")))),
        "normalized_mae": float(metric_result.get("normalized_mae", metric_result.get("mae", float("nan")))),
        "normalized_mape": metric_result.get("normalized_mape"),
        "normalized_smape": metric_result.get("normalized_smape"),
        "original_scale_rmse": metric_result.get("original_scale_rmse"),
        "original_scale_accuracy": metric_result.get("original_scale_accuracy"),
        "original_scale_mae": metric_result.get("original_scale_mae"),
        "original_scale_mape": metric_result.get("original_scale_mape"),
        "original_scale_smape": metric_result.get("original_scale_smape"),
        "metric_space": str(metric_result.get("metric_space", metric_result.get("metric_space_current", "normalized_minmax_space"))),
        "metric_space_used": str(metric_result.get("metric_space_used", metric_result.get("metric_space", "normalized_minmax_space"))),
        "y_pred": y_pred,
        "y_true": y_true,
        "prediction_shape": tuple(y_pred.shape),
        "metric_space_current": str(metric_result["metric_space_current"]),
        "metric_space_paper": str(metric_result["metric_space_paper"]),
        "paper_metric_aligned": bool(metric_result["paper_metric_aligned"]),
        "inverse_transform_applied": bool(metric_result["inverse_transform_applied"]),
        "inverse_transform_available": bool(metric_result.get("inverse_transform_available", False)),
        "metric_notes": str(metric_result["metric_notes"]),
    }


def evaluate_msml_rfe_split(
    target_model,
    target_split_df: pd.DataFrame,
    feature_cols: Sequence[str],
    horizon: int = 1,
    window_size: int = 10,
    eps: float = 1e-8,
    metric_protocol: dict | None = None,
    sales_scaler: object | None = None,
    feature_columns_for_scaler: object | None = None,
    split_name: str = "split",
) -> Dict[str, object]:
    """Evaluate MSML-TL-RFE on a given split (val/test) with full metric breakdown."""
    logger = _get_logger()
    logger.info("[evaluate_msml_rfe_split] Start. split=%s", split_name)

    X_split, y_split = build_tabular_sequence(
        target_split_df,
        horizon=horizon,
        window_size=window_size,
        feature_cols=feature_cols,
    )
    if len(y_split) == 0:
        raise ValueError(f"Target {split_name} split produced zero windows; adjust window_size/horizon.")

    X_split = to_cnn_tensor(X_split)
    y_pred = target_model.predict(X_split, verbose=0)
    y_true = y_split.flatten()

    metric_result = compute_metrics_with_protocol(
        y_true=y_true,
        y_pred=y_pred,
        metric_protocol=metric_protocol,
        sales_scaler=sales_scaler,
        feature_columns=feature_columns_for_scaler,
        eps=eps,
    )

    logger.info(
        "[evaluate_msml_rfe_split] Finished. split=%s RMSE=%.4f Accuracy=%.4f",
        split_name,
        float(metric_result["rmse"]),
        float(metric_result["accuracy"]),
    )
    return {
        "rmse": float(metric_result["rmse"]),
        "accuracy": float(metric_result["accuracy"]),
        "mae": float(metric_result.get("mae", float("nan"))),
        "mape": float(metric_result.get("mape", float("nan"))),
        "smape": float(metric_result.get("smape", float("nan"))),
        "rmse_current": float(metric_result.get("rmse_current", float("nan"))),
        "accuracy_current": float(metric_result.get("accuracy_current", float("nan"))),
        "mae_current": float(metric_result.get("mae_current", float("nan"))),
        "mape_current": float(metric_result.get("mape_current", float("nan"))),
        "smape_current": float(metric_result.get("smape_current", float("nan"))),
        "rmse_paper": float(metric_result.get("rmse_paper", float("nan"))),
        "accuracy_paper": float(metric_result.get("accuracy_paper", float("nan"))),
        "mae_paper": float(metric_result.get("mae_paper", float("nan"))),
        "mape_paper": float(metric_result.get("mape_paper", float("nan"))),
        "smape_paper": float(metric_result.get("smape_paper", float("nan"))),
        "normalized_rmse": float(metric_result.get("normalized_rmse", metric_result.get("rmse", float("nan")))),
        "normalized_accuracy": float(metric_result.get("normalized_accuracy", metric_result.get("accuracy", float("nan")))),
        "normalized_mae": float(metric_result.get("normalized_mae", metric_result.get("mae", float("nan")))),
        "normalized_mape": metric_result.get("normalized_mape"),
        "normalized_smape": metric_result.get("normalized_smape"),
        "original_scale_rmse": metric_result.get("original_scale_rmse"),
        "original_scale_accuracy": metric_result.get("original_scale_accuracy"),
        "original_scale_mae": metric_result.get("original_scale_mae"),
        "original_scale_mape": metric_result.get("original_scale_mape"),
        "original_scale_smape": metric_result.get("original_scale_smape"),
        "metric_space": str(metric_result.get("metric_space", metric_result.get("metric_space_current", "normalized_minmax_space"))),
        "metric_space_used": str(metric_result.get("metric_space_used", metric_result.get("metric_space", "normalized_minmax_space"))),
        "prediction_shape": tuple(np.asarray(y_pred).shape),
        "metric_space_current": str(metric_result["metric_space_current"]),
        "metric_space_paper": str(metric_result["metric_space_paper"]),
        "paper_metric_aligned": bool(metric_result["paper_metric_aligned"]),
        "inverse_transform_applied": bool(metric_result["inverse_transform_applied"]),
        "inverse_transform_available": bool(metric_result.get("inverse_transform_available", False)),
        "metric_notes": str(metric_result["metric_notes"]),
    }


# -------------------------------------------------------------------
# 7. run_msml_tl_rfe (主函数)
# -------------------------------------------------------------------

def run_msml_tl_rfe(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    feature_cols: Sequence[str],
    k: int = 3,
    number_of_sources: int | None = None,
    horizon: int = 1,
    window_size: int = 10,
    weight_mode: str = "inverse_distance",
    estimator_name: str = "random_forest",
    keep_ratio: float = 0.5,
    learning_rate: float = FIXED_LEARNING_RATE,
    source_epochs: int = FIXED_EPOCHS,
    target_epochs: int = FIXED_EPOCHS,
    batch_size: int = 16,
    random_state: int = 42,
    include_sales_in_knn: bool = True,
    use_sales_as_history_input: bool = True,
    metric_protocol: dict | None = None,
    source_selection_window: str = "target_observed_window",
    full_target_df: pd.DataFrame | None = None,
    # 早停参数
    early_stopping_enabled: bool = True,
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    early_stopping_min_delta: float = DEFAULT_EARLY_STOPPING_MIN_DELTA,
    # 自适应源选择参数
    adaptive_source_selection: bool = False,
    min_sources: int = 1,
    max_sources: int | None = None,
    distance_jump_threshold: float = 0.5,
    distance_ratio_threshold: float | None = None,
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
        use_sales_as_history_input: RFE 后是否将 sales 作为历史窗口输入加回

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
    effective_number_of_sources = int(k if number_of_sources is None else number_of_sources)
    logger = _get_logger()
    logger.info(
        "[run_msml_tl_rfe] Start. number_of_sources=%d weight_mode=%s keep_ratio=%.2f estimator=%s",
        effective_number_of_sources,
        weight_mode,
        keep_ratio,
        estimator_name,
    )
    logger.info(
        "[run_msml_tl_rfe] Training hyperparameters: %s. clipnorm=None means gradient clipping is disabled.",
        fixed_hyperparams_summary(),
    )

    _validate_feature_cols(source_df, feature_cols, where="source_df")
    _validate_feature_cols(target_df, feature_cols, where="target_df")

    requested_k = int(effective_number_of_sources)
    selected_source_sequences: Dict[Tuple[object, object], pd.DataFrame] = {}
    selected_source_sequences_rfe: Dict[Tuple[object, object], pd.DataFrame] = {}
    selected_source_train_dfs: List[pd.DataFrame] = []
    source_models: List = []

    current_stage = "init"
    try:
        source_selection_window_norm = str(source_selection_window or "target_observed_window").strip().lower()
        selection_target_df = target_df
        target_train_df: Optional[pd.DataFrame] = None
        target_val_df: Optional[pd.DataFrame] = None
        target_test_df: Optional[pd.DataFrame] = None

        if source_selection_window_norm in {
            "train_window",
            "target_train_window",
            "observed_window",
            "target_observed_window",
            "train_val_window",
        }:
            target_train_df, target_val_df, target_test_df = temporal_split_by_ratio_or_dates(target_df)
            for split_df in (target_train_df, target_val_df, target_test_df):
                split_df.attrs = target_df.attrs.copy()

            if source_selection_window_norm in {"train_window", "target_train_window"}:
                selection_target_df = target_train_df
            else:
                selection_target_df = pd.concat([target_train_df, target_val_df], axis=0, ignore_index=True)
                selection_target_df.attrs = target_df.attrs.copy()

        uses_full_target_window = selection_target_df is target_df
        source_selection_audit = {
            "source_selection_window": source_selection_window_norm,
            "source_selection_uses_full_target_window": bool(uses_full_target_window),
            "source_selection_risk_recorded": bool(uses_full_target_window),
            "source_selection_risk_note": (
                "Source selection uses the full target window before temporal split."
                if uses_full_target_window
                else "Source selection uses the target observed window only."
            ),
            "source_selection_target_rows": int(len(selection_target_df)),
        }
        if uses_full_target_window:
            logger.warning("[run_msml_tl_rfe] %s", source_selection_audit["source_selection_risk_note"])

        # --- Step 1: 选源 ---
        current_stage = "step1_source_selection"
        selector = SourceSelector()
        effective_max_sources = max_sources if max_sources is not None else requested_k
        selector_target_df = full_target_df if full_target_df is not None else selection_target_df
        selection_result = selector.select_top_k_sources(
            target_df=selector_target_df,
            source_df=source_df,
            feature_cols=feature_cols,
            k=requested_k,
            weight_mode=weight_mode,
            include_sales_in_knn=include_sales_in_knn,
            # 自适应源选择参数
            adaptive_source_selection=adaptive_source_selection,
            min_sources=min_sources,
            max_sources=effective_max_sources,
            distance_jump_threshold=distance_jump_threshold,
            distance_ratio_threshold=distance_ratio_threshold,
        )
        selected_sources = selection_result.get("sources", []) if isinstance(selection_result, dict) else selection_result
        selection_meta = selection_result.get("meta", {}) if isinstance(selection_result, dict) else {}
        actual_selected_sources = len(selected_sources)
        if actual_selected_sources == 0:
            raise ValueError("No source selected from source pool.")

        logger.info(
            "[run_msml_tl_rfe] Selection stats: requested_k=%d actual_selected_sources=%d",
            requested_k,
            actual_selected_sources,
        )

        # --- Step 2: 切分 target ---
        current_stage = "step2_target_split"
        if target_train_df is None or target_val_df is None or target_test_df is None:
            target_train_df, target_val_df, target_test_df = temporal_split_by_ratio_or_dates(target_df)
        logger.info(
            "[run_msml_tl_rfe] Step 2: Split target. train=%d val=%d test=%d",
            len(target_train_df), len(target_val_df), len(target_test_df),
        )

        # --- Step 3: 提取 selected source sequences ---
        current_stage = "step3_extract_source_sequences"
        selected_source_keys: List[Tuple[object, object]] = []
        for selected in selected_sources:
            source_key = _normalize_source_key(selected.get("source_key"))
            entity_id, item_id = source_key
            source_sequence_df = source_df[
                (source_df["entity_id"] == entity_id) & (source_df["item_id"] == item_id)
            ].copy()

            if source_sequence_df.empty:
                raise ValueError(f"Selected source_key not found in source_df: {source_key}")

            if source_key in selected_source_sequences:
                raise ValueError(f"Duplicate source_key in selected_sources: {source_key}")

            selected_source_sequences[source_key] = source_sequence_df
            selected_source_keys.append(source_key)

            # 切分 source，提取 train 用于 RFE
            src_train, _, _ = _prepare_source_split(source_sequence_df)
            src_train = fill_source_numeric_na(src_train)
            selected_source_train_dfs.append(src_train)

        if len(selected_source_train_dfs) != actual_selected_sources:
            raise ValueError(
                "Length mismatch after source extraction: "
                f"actual_selected_sources={actual_selected_sources}, "
                f"len(source_dfs)={len(selected_source_train_dfs)}"
            )
        if len(selected_source_sequences) != actual_selected_sources:
            raise ValueError(
                "Length mismatch after source extraction: "
                f"actual_selected_sources={actual_selected_sources}, "
                f"len(source_sequences)={len(selected_source_sequences)}"
            )

        logger.info(
            "[run_msml_tl_rfe] Step 3 stats: requested_k=%d actual_selected_sources=%d len(source_dfs)=%d len(source_sequences)=%d",
            requested_k,
            actual_selected_sources,
            len(selected_source_train_dfs),
            len(selected_source_sequences),
        )

        # --- Step 4: 构建联合 RFE 训练数据 ---
        current_stage = "step4_build_joint_rfe_train_df"
        modeling_feature_cols = filter_model_input_feature_cols(feature_cols)
        if not modeling_feature_cols:
            raise ValueError("No safe modeling feature columns remain for MSML-TL-RFE after ID/code filtering.")
        joint_train_df = build_joint_rfe_training_dataframe(
            target_train_df=target_train_df,
            selected_source_dfs=selected_source_train_dfs,
            feature_cols=modeling_feature_cols,
            target_col="sales",
        )
        joint_rfe_audit = dict(joint_train_df.attrs.get("rfe_audit", {}))
        logger.info("[run_msml_tl_rfe] Step 4: Built joint RFE training data. rows=%d", len(joint_train_df))

        # --- Step 5: 执行 RFE ---
        current_stage = "step5_run_rfe"
        rfe_result = run_rfe_feature_selection(
            train_df=joint_train_df,
            feature_cols=modeling_feature_cols,
            target_col="sales",
            estimator_name=estimator_name,
            keep_ratio=keep_ratio,
            random_state=random_state,
            use_sales_as_history_input=use_sales_as_history_input,
        )
        rfe_audit_info = dict(rfe_result)
        rfe_audit_info.update(
            {
                "duplicate_columns_before": list(joint_rfe_audit.get("duplicate_columns_before", [])),
                "duplicate_columns_after": list(joint_rfe_audit.get("duplicate_columns_after", [])),
                "duplicate_sales_before": int(joint_rfe_audit.get("duplicate_sales_before", 0)),
                "duplicate_sales_after": int(joint_rfe_audit.get("duplicate_sales_after", 0)),
                "contains_duplicate_sales_in_joint_df": bool(
                    joint_rfe_audit.get("contains_duplicate_sales_in_joint_df", False)
                ),
            }
        )
        rfe_selected_feature_cols = list(rfe_result["rfe_selected_features"])
        selected_feature_cols = list(rfe_result["final_selected_features"])
        logger.info(
            "[run_msml_tl_rfe] Step 5: RFE completed. "
            "original=%d rfe_selected=%d final_selected=%d rfe_selected_cols=%s final_selected_cols=%s",
            rfe_result["num_original_features"],
            rfe_result["num_rfe_selected_features"],
            rfe_result["num_selected_features"],
            rfe_selected_feature_cols,
            selected_feature_cols,
        )

        # --- Step 6: 应用 RFE 特征到 target 和 selected sources ---
        current_stage = "step6_apply_selected_features"
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

        for source_key, source_seq_df in selected_source_sequences.items():
            selected_source_sequences_rfe[source_key] = apply_selected_features_to_df(
                source_seq_df, selected_feature_cols, required_cols=required_cols
            )

        if len(selected_source_sequences_rfe) != len(selected_source_sequences):
            raise ValueError(
                "Length mismatch after feature projection: "
                f"len(source_feature_dfs)={len(selected_source_sequences_rfe)} "
                f"len(source_sequences)={len(selected_source_sequences)}"
            )

        logger.info(
            "[run_msml_tl_rfe] Step 6 stats: requested_k=%d actual_selected_sources=%d len(source_feature_dfs)=%d len(source_sequences)=%d",
            requested_k,
            actual_selected_sources,
            len(selected_source_sequences_rfe),
            len(selected_source_sequences),
        )

        # --- Step 7: 对每个 selected source 训练 CNN（使用 RFE 特征）---
        current_stage = "step7_train_source_cnns"
        source_weights: List[float] = []
        source_models_info: List[Dict[str, object]] = []
        input_shape_ref: Optional[Tuple[int, ...]] = None

        selected_meta_by_key: Dict[Tuple[object, object], Dict[str, float]] = {}
        for selected in selected_sources:
            selected_key = _normalize_source_key(selected.get("source_key"))
            if selected_key in selected_meta_by_key:
                raise ValueError(f"Duplicate source_key in selected_sources meta: {selected_key}")
            selected_meta_by_key[selected_key] = {
                "distance": float(selected["distance"]),
                "weight": float(selected["weight"]),
            }

        for source_key in selected_source_keys:
            if source_key not in selected_source_sequences_rfe:
                raise ValueError(f"source_key missing in source_feature_dfs: {source_key}")
            if source_key not in selected_meta_by_key:
                raise ValueError(f"source_key missing in selected_sources meta: {source_key}")

            source_sequence_df_rfe = selected_source_sequences_rfe[source_key]

            train_result = train_source_cnn_for_msml_rfe(
                source_sequence_df=source_sequence_df_rfe,
                feature_cols=selected_feature_cols,
                horizon=horizon,
                window_size=window_size,
                learning_rate=learning_rate,
                source_epochs=source_epochs,
                batch_size=batch_size,
                source_key=source_key,
                early_stopping_enabled=early_stopping_enabled,
                early_stopping_patience=early_stopping_patience,
                early_stopping_min_delta=early_stopping_min_delta,
            )

            if input_shape_ref is None:
                input_shape_ref = train_result["input_shape"]
            elif train_result["input_shape"] != input_shape_ref:
                raise ValueError(
                    f"Input shape mismatch: source_key={source_key} "
                    f"shape={train_result['input_shape']} expected={input_shape_ref}"
                )

            source_models.append(train_result["model"])

            source_meta = selected_meta_by_key[source_key]
            source_weights.append(source_meta["weight"])
            source_models_info.append({
                "source_key": source_key,
                "distance": source_meta["distance"],
                "weight": source_meta["weight"],
                "num_samples": int(train_result["num_samples"]),
            })

        if len(source_models) == 0:
            raise ValueError("No source models were trained before fusion.")
        if len(source_models) != actual_selected_sources:
            raise ValueError(
                "Length mismatch before fusion: "
                f"actual_selected_sources={actual_selected_sources}, len(source_models)={len(source_models)}"
            )
        if len(source_weights) != len(source_models):
            raise ValueError(
                "Length mismatch before fusion: "
                f"len(source_weights)={len(source_weights)} len(source_models)={len(source_models)}"
            )

        logger.info(
            "[run_msml_tl_rfe] Step 7 stats: requested_k=%d actual_selected_sources=%d len(source_models)=%d len(source_feature_dfs)=%d len(source_sequences)=%d",
            requested_k,
            actual_selected_sources,
            len(source_models),
            len(selected_source_sequences_rfe),
            len(selected_source_sequences),
        )

        # --- Step 8: 确定可融合层并做加权参数融合 ---
        current_stage = "step8_fuse_source_models"
        fusion_layer_names = list(_DEFAULT_FUSION_LAYERS)
        freeze_layer_names = get_transferable_layer_names(source_models[0])
        fused_params = fuse_source_models_layerwise(source_models, source_weights, fusion_layer_names)
        logger.info(
            "[run_msml_tl_rfe] Step 8: Fused source models. layers=%s",
            list(fusion_layer_names),
        )

        # --- Step 9: 构建 target model → 加载融合参数 → 冻结 ---
        current_stage = "step9_build_target_model"
        target_model = build_base_cnn(input_shape_ref, learning_rate=learning_rate)
        target_model = load_fused_params_into_target_model(target_model, fused_params)
        frozen_layers = freeze_fused_layers(target_model, freeze_layer_names)
        logger.info("[run_msml_tl_rfe] Step 9: Loaded fused params and froze layers")

        # --- Step 10: 归一化 target 数据 ---
        current_stage = "step10_normalize_target"
        target_train_df_rfe, target_val_df_rfe, target_test_df_rfe, target_scaler, target_feature_columns = normalize_features(
            target_train_df_rfe,
            target_val_df_rfe,
            target_test_df_rfe,
            feature_cols=selected_feature_cols,
        )
        logger.info("[run_msml_tl_rfe] Step 10: Normalized target features")

        # --- Step 11: 微调 target model ---
        current_stage = "step11_finetune_target"
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
            early_stopping_enabled=early_stopping_enabled,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
        )
        logger.info("[run_msml_tl_rfe] Step 11: Fine-tuned target model")

        # --- Step 12: 在 target test/val 上评估 ---
        current_stage = "step12_evaluate"
        val_result = evaluate_msml_rfe_split(
            target_model=ft_result["model"],
            target_split_df=target_val_df_rfe,
            feature_cols=selected_feature_cols,
            horizon=horizon,
            window_size=window_size,
            metric_protocol=metric_protocol,
            sales_scaler=target_scaler,
            feature_columns_for_scaler=target_feature_columns,
            split_name="val",
        )

        eval_result = evaluate_msml_rfe_model(
            target_model=ft_result["model"],
            target_test_df=target_test_df_rfe,
            feature_cols=selected_feature_cols,
            horizon=horizon,
            window_size=window_size,
            metric_protocol=metric_protocol,
            sales_scaler=target_scaler,
            feature_columns_for_scaler=target_feature_columns,
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
                "k": int(requested_k),
                "horizon": int(horizon),
                "number_of_sources": int(requested_k),
                "number_of_pretrained_models": int(len(source_models_info)),
                "number_of_methods": 1,
                "requested_k": requested_k,
                "actual_selected_sources": actual_selected_sources,
                "weight_mode": weight_mode,
                "feature_cols": list(feature_cols),
                "modeling_feature_cols": list(modeling_feature_cols),
                "selected_feature_cols": selected_feature_cols,
                "selected_features": selected_feature_cols,
                "rfe_selected_features": rfe_selected_feature_cols,
                "rfe_candidate_features": list(rfe_result["rfe_candidate_features"]),
                "final_selected_features": selected_feature_cols,
                "keep_ratio": float(keep_ratio),
                "random_state": int(random_state),
                "target_col": str(rfe_result["target_col"]),
                "target_removed_from_rfe": bool(rfe_result["target_removed_from_rfe"]),
                "sales_added_back_as_history_input": bool(rfe_result["sales_added_back_as_history_input"]),
                "duplicate_sales_after": int(rfe_audit_info.get("duplicate_sales_after", 0)),
                "selected_sources": selected_sources,
                "source_selection_info": selection_meta,
                "source_selection_audit": source_selection_audit,
                "fused_layers": list(fusion_layer_names),
            },
            "rfe_info": rfe_audit_info,
            "source_models_info": source_models_info,
            "fused_result": {
                "rmse": eval_result["rmse"],
                "accuracy": eval_result["accuracy"],
                "mae": eval_result.get("mae", float("nan")),
                "mape": eval_result.get("mape", float("nan")),
                "smape": eval_result.get("smape", float("nan")),
                "rmse_current": eval_result.get("rmse_current", float("nan")),
                "accuracy_current": eval_result.get("accuracy_current", float("nan")),
                "mae_current": eval_result.get("mae_current", float("nan")),
                "mape_current": eval_result.get("mape_current", float("nan")),
                "smape_current": eval_result.get("smape_current", float("nan")),
                "rmse_paper": eval_result.get("rmse_paper", float("nan")),
                "accuracy_paper": eval_result.get("accuracy_paper", float("nan")),
                "mae_paper": eval_result.get("mae_paper", float("nan")),
                "mape_paper": eval_result.get("mape_paper", float("nan")),
                "smape_paper": eval_result.get("smape_paper", float("nan")),
                "normalized_rmse": eval_result.get("normalized_rmse", eval_result.get("rmse", float("nan"))),
                "normalized_accuracy": eval_result.get("normalized_accuracy", eval_result.get("accuracy", float("nan"))),
                "normalized_mae": eval_result.get("normalized_mae", eval_result.get("mae", float("nan"))),
                "normalized_mape": eval_result.get("normalized_mape"),
                "normalized_smape": eval_result.get("normalized_smape"),
                "original_scale_rmse": eval_result.get("original_scale_rmse"),
                "original_scale_accuracy": eval_result.get("original_scale_accuracy"),
                "original_scale_mae": eval_result.get("original_scale_mae"),
                "original_scale_mape": eval_result.get("original_scale_mape"),
                "original_scale_smape": eval_result.get("original_scale_smape"),
                "metric_space": eval_result.get("metric_space", "normalized_minmax_space"),
                "metric_space_used": eval_result.get("metric_space_used", eval_result.get("metric_space", "normalized_minmax_space")),
                "prediction_shape": eval_result["prediction_shape"],
                "metric_space_current": eval_result.get("metric_space_current", "normalized_minmax_space"),
                "metric_space_paper": eval_result.get("metric_space_paper", "original_sales_space"),
                "paper_metric_aligned": eval_result.get("paper_metric_aligned", False),
                "inverse_transform_applied": eval_result.get("inverse_transform_applied", False),
                "inverse_transform_available": eval_result.get("inverse_transform_available", False),
                "metric_notes": eval_result.get("metric_notes", ""),
                "val_rmse": val_result.get("rmse", float("nan")),
                "val_accuracy": val_result.get("accuracy", float("nan")),
                "val_mae": val_result.get("mae", float("nan")),
                "val_mape": val_result.get("mape", float("nan")),
                "val_smape": val_result.get("smape", float("nan")),
                "val_rmse_current": val_result.get("rmse_current", float("nan")),
                "val_accuracy_current": val_result.get("accuracy_current", float("nan")),
                "val_mae_current": val_result.get("mae_current", float("nan")),
                "val_mape_current": val_result.get("mape_current", float("nan")),
                "val_smape_current": val_result.get("smape_current", float("nan")),
                "val_rmse_paper": val_result.get("rmse_paper", float("nan")),
                "val_accuracy_paper": val_result.get("accuracy_paper", float("nan")),
                "val_mae_paper": val_result.get("mae_paper", float("nan")),
                "val_mape_paper": val_result.get("mape_paper", float("nan")),
                "val_smape_paper": val_result.get("smape_paper", float("nan")),
                "val_normalized_rmse": val_result.get("normalized_rmse", val_result.get("rmse", float("nan"))),
                "val_normalized_accuracy": val_result.get("normalized_accuracy", val_result.get("accuracy", float("nan"))),
                "val_normalized_mae": val_result.get("normalized_mae", val_result.get("mae", float("nan"))),
                "val_normalized_smape": val_result.get("normalized_smape"),
                "val_original_scale_rmse": val_result.get("original_scale_rmse"),
                "val_original_scale_accuracy": val_result.get("original_scale_accuracy"),
                "val_original_scale_mae": val_result.get("original_scale_mae"),
                "val_original_scale_smape": val_result.get("original_scale_smape"),
                "val_metric_space": val_result.get("metric_space", "normalized_minmax_space"),
                "val_metric_space_used": val_result.get("metric_space_used", val_result.get("metric_space", "normalized_minmax_space")),
                "val_metric_space_current": val_result.get("metric_space_current", "normalized_minmax_space"),
                "val_metric_space_paper": val_result.get("metric_space_paper", "original_sales_space"),
                "val_paper_metric_aligned": val_result.get("paper_metric_aligned", False),
                "val_inverse_transform_applied": val_result.get("inverse_transform_applied", False),
                "val_inverse_transform_available": val_result.get("inverse_transform_available", False),
                "val_metric_notes": val_result.get("metric_notes", ""),
            },
            "frozen_layers": frozen_layers,
        }
    except IndexError as exc:
        raise ValueError(
            "MSML-TL-RFE index boundary error. "
            f"stage={current_stage}, "
            f"requested_k={requested_k}, "
            f"len(source_sequences)={len(selected_source_sequences)}, "
            f"len(source_feature_dfs)={len(selected_source_sequences_rfe)}, "
            f"len(source_dfs)={len(selected_source_train_dfs)}, "
            f"len(source_models)={len(source_models)}"
        ) from exc
