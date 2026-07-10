"""
模块10：论文实验总运行器

职责：
1. 统一准备 source/target 数据
2. 统一运行 SS-TL / MSWA-TL / MSSB-TL / MSML-TL / MSML-TL-RFE
3. 统一抽取 rmse/accuracy/prediction_shape
4. 汇总结果、输出 DataFrame、保存 CSV、打印简洁表格
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.constants import MIXED_METRIC_SPACE
from src.utils.config import Config
from src.data_processing.data_preprocessing import (
    build_source_target_split,
    build_tabular_sequence,
    extract_datetime_features,
    load_dataset,
    normalize_features,
    temporal_split_by_ratio_or_dates,
    to_cnn_tensor,
)
try:
    from src.utils.environment import setup_logging
except ImportError:
    setup_logging = None

from src.utils.console_reporter import print_method_result, print_method_start
from src.utils.runtime_control import apply_logging_level, log_level_name, set_verbose_mode
from paper_reproduction_protocol import (
    MULTI_SOURCE_TL_METHODS,
    STATUS_PARTIAL,
    build_alignment_fields,
    ensure_paper_track_allowed,
    load_paper_protocol,
    resolve_strict_paper_mode,
    validate_paper_protocol_config,
)
from src.source_selection.source_selector import SourceSelector
from src.transfer_methods.source_failure_tolerance import runtime_selection_meta
from src.utils.source_fillna import fill_source_numeric_na


LOGGER_NAME = "experiment"
DEFAULT_METHODS = ["No-TL", "SS-TL", "MSWA-TL", "MSSB-TL", "MSML-TL", "MSML-TL-RFE"]


def _get_nested(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    current = config
    for part in key.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        else:
            current = getattr(current, part, None)
        if current is None:
            return default
    return current


def _date_range_summary(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty:
        return {
            "start_date": "N/A",
            "end_date": "N/A",
            "unique_days": 0,
            "range_days": 0,
            "rows": 0,
        }
    min_date = pd.Timestamp(df["date"].min())
    max_date = pd.Timestamp(df["date"].max())
    return {
        "start_date": min_date.strftime("%Y-%m-%d"),
        "end_date": max_date.strftime("%Y-%m-%d"),
        "unique_days": int(df["date"].nunique()),
        "range_days": int((max_date - min_date).days + 1),
        "rows": int(len(df)),
    }


def _resolve_paper_split_protocol_from_config(config: Any) -> Dict[str, Any]:
    protocol = _get_nested(config, "paper_reproduction.paper_split_protocol", None)
    if isinstance(protocol, dict) and protocol:
        return dict(protocol)

    observed = int(_get_nested(config, "paper_reproduction.split_protocol.target_window.train_val_days", 30))
    forecast = int(_get_nested(config, "paper_reproduction.split_protocol.target_window.test_days", 180))
    strategy = str(_get_nested(config, "paper_reproduction.split_protocol.target_eval_split.mode", "ratio"))
    split_kind = str(_get_nested(config, "paper_reproduction.split_protocol.target_window.kind", "rolling_recent_days"))
    return {
        "target_observed_window_days": observed,
        "target_forecast_window_days": forecast,
        "validation_strategy": strategy,
        "rolling_or_fixed_split": split_kind,
        "source_selection_window": "full_history",
        "source_pool_scope": "all_source_items",
        "paper_reference_note": "按论文相对窗口复刻",
    }


def _save_split_protocol_summary(
    dataset_name: str,
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    config: Any,
) -> None:
    source_train, source_val, source_test = temporal_split_by_ratio_or_dates(source_df)
    target_train, target_val, target_test = temporal_split_by_ratio_or_dates(target_df)

    protocol = _resolve_paper_split_protocol_from_config(config)
    horizon = int(_get_nested(config, "single_experiment.horizon", 1))

    summary = {
        "dataset": str(dataset_name),
        "paper_split_protocol": protocol,
        "strict_paper_split": bool(target_df.attrs.get("strict_paper_split", False)),
        "source_domain_time_range": _date_range_summary(source_df),
        "source_train_time_range": _date_range_summary(source_train),
        "source_validation_time_range": _date_range_summary(source_val),
        "source_test_time_range": _date_range_summary(source_test),
        "target_train_time_range": _date_range_summary(target_train),
        "target_validation_time_range": _date_range_summary(target_val),
        "target_test_forecast_time_range": _date_range_summary(target_test),
        "horizon_definition": {
            "horizon": horizon,
            "note": "horizon is forecast steps ahead in sliding-window supervision.",
        },
    }

    output_dir = Path("outputs") / "paper_alignment"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"split_protocol_{str(dataset_name).lower()}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)


def run_no_tl_experiment(
    target_df: pd.DataFrame,
    horizon: int = 1,
    window_size: int = 10,
    learning_rate: float = 0.001,
    target_epochs: int = 3,
    batch_size: int = 16,
    metric_protocol: Optional[Dict[str, Any]] = None,
    feature_cols: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """运行 No-TL，并返回统一结构。"""
    try:
        from src.experiment.run_no_tl_experiment import run_no_tl_experiment as _run_no_tl
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"No-TL dependency missing: {exc}") from exc

    raw = _run_no_tl(
        target_df=target_df,
        horizon=horizon,
        window_size=window_size,
        learning_rate=learning_rate,
        target_epochs=target_epochs,
        batch_size=batch_size,
        metric_protocol=metric_protocol,
        feature_cols=feature_cols,
    )
    return _extract_method_metrics(raw, method_name="No-TL")


def _get_logger() -> logging.Logger:
    """获取统一日志器；若尚未初始化，则按默认参数初始化。"""
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers and setup_logging is not None:
        setup_logging(log_level=log_level_name(), log_file=None)
        logger = logging.getLogger(LOGGER_NAME)
    apply_logging_level()
    return logger


def _ensure_feature_cols(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]],
) -> List[str]:
    """校验并返回可用特征列。"""
    if feature_cols is None:
        feature_cols = ["sales", "year", "month", "week", "day"]

    cols = [str(c) for c in feature_cols]
    missing_in_source = [c for c in cols if c not in source_df.columns]
    missing_in_target = [c for c in cols if c not in target_df.columns]

    if missing_in_source or missing_in_target:
        raise ValueError(
            "Invalid feature_cols. "
            f"missing_in_source={missing_in_source} "
            f"missing_in_target={missing_in_target}"
        )

    if "sales" not in cols:
        raise ValueError("feature_cols must include 'sales' for regression target construction.")

    object_in_source = [
        c
        for c in cols
        if pd.api.types.is_object_dtype(source_df[c]) or pd.api.types.is_string_dtype(source_df[c])
    ]
    object_in_target = [
        c
        for c in cols
        if pd.api.types.is_object_dtype(target_df[c]) or pd.api.types.is_string_dtype(target_df[c])
    ]
    if object_in_source or object_in_target:
        source_dtype_map = {c: str(source_df[c].dtype) for c in object_in_source}
        target_dtype_map = {c: str(target_df[c].dtype) for c in object_in_target}
        raise ValueError(
            "feature_cols contains object/string dtype columns before normalization. "
            f"source_bad={source_dtype_map} target_bad={target_dtype_map}"
        )

    return cols


def _shape_to_tuple(shape_value: Any) -> Any:
    """将形状信息统一为 tuple（若不可解析则原样返回字符串）。"""
    if shape_value is None:
        return "N/A"
    if isinstance(shape_value, tuple):
        return shape_value
    if isinstance(shape_value, list):
        return tuple(shape_value)
    if hasattr(shape_value, "shape"):
        return tuple(shape_value.shape)
    text = str(shape_value).strip()
    return text if text else "N/A"


def _coalesce_metric(*values: Any) -> Any:
    """Return the first numeric/reporting value that is not None/NaN."""
    for value in values:
        if value is None:
            continue
        if isinstance(value, (float, np.floating)) and np.isnan(value):
            continue
        return value
    return None


def _summarize_metric_space(rmse_metric_space: Any, smape_metric_space: Any, fallback: Any) -> str:
    rmse_space = str(rmse_metric_space or "").strip()
    smape_space = str(smape_metric_space or "").strip()
    if rmse_space and smape_space:
        return rmse_space if rmse_space == smape_space else MIXED_METRIC_SPACE
    return str(fallback)


def _extract_method_metrics(
    raw_result: Dict[str, Any],
    method_name: str,
    metric_protocol: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    从不同模块返回结构中统一抽取结果字段。

    抽取优先级：
    1) raw_result['fused_result']
    2) raw_result['final_result']
    3) raw_result 本身
    """
    if not isinstance(raw_result, dict):
        raise ValueError(f"{method_name} raw_result must be a dict, got {type(raw_result)}")

    candidates: List[Dict[str, Any]] = []
    for key in ("fused_result", "final_result"):
        value = raw_result.get(key)
        if isinstance(value, dict):
            candidates.append(value)
    candidates.append(raw_result)

    selected: Optional[Dict[str, Any]] = None
    for c in candidates:
        if "rmse" in c and "accuracy" in c:
            selected = c
            break

    if selected is None:
        raise ValueError(f"Cannot extract rmse/accuracy from method={method_name} result.")

    if (
        metric_protocol is not None
        and "y_true" in selected
        and "y_pred" in selected
        and selected.get("sales_scaler") is not None
        and selected.get("feature_columns") is not None
    ):
        from src.evaluation.metrics import compute_metrics_with_protocol

        metric_result = compute_metrics_with_protocol(
            y_true=np.asarray(selected["y_true"]).reshape(-1),
            y_pred=np.asarray(selected["y_pred"]).reshape(-1),
            metric_protocol=metric_protocol,
            sales_scaler=selected.get("sales_scaler"),
            feature_columns=selected.get("feature_columns"),
        )
        selected = {
            **selected,
            "rmse_current": selected.get("rmse_current", selected.get("rmse")),
            "accuracy_current": selected.get("accuracy_current", selected.get("accuracy")),
            "smape_current": selected.get("smape_current", selected.get("smape")),
            "normalized_rmse": selected.get("normalized_rmse", selected.get("rmse")),
            "normalized_accuracy": selected.get("normalized_accuracy", selected.get("accuracy")),
            "normalized_smape": selected.get("normalized_smape", selected.get("smape")),
            **metric_result,
        }

    prediction_shape = selected.get("prediction_shape")
    if prediction_shape is None:
        prediction_shape = selected.get("y_pred_shape")
    if prediction_shape is None and "y_pred" in selected:
        prediction_shape = np.asarray(selected["y_pred"]).shape

    method_meta = {}
    if isinstance(raw_result.get("meta"), dict):
        method_meta.update(raw_result["meta"])

    for optional_key in (
        "individual_results",
        "best_source_result",
        "source_models_info",
        "rfe_info",
        "frozen_layers",
    ):
        if optional_key in raw_result:
            method_meta[optional_key] = raw_result[optional_key]

    result = {
        "method": method_name,
        "training_time": float(selected.get("training_time", selected.get("training_time_seconds", float("nan")))),
        "rmse": float(selected["rmse"]),
        "accuracy": float(selected["accuracy"]),
        "mae": float(selected.get("mae", float("nan"))),
        "mape": float(selected.get("mape", float("nan"))),
        "smape": float(selected.get("smape", float("nan"))),
        "rmse_current": float(selected.get("rmse_current", float("nan"))),
        "accuracy_current": float(selected.get("accuracy_current", float("nan"))),
        "mae_current": float(selected.get("mae_current", float("nan"))),
        "mape_current": float(selected.get("mape_current", float("nan"))),
        "smape_current": float(selected.get("smape_current", float("nan"))),
        "rmse_paper": float(selected.get("rmse_paper", float("nan"))),
        "accuracy_paper": float(selected.get("accuracy_paper", float("nan"))),
        "mae_paper": float(selected.get("mae_paper", float("nan"))),
        "mape_paper": float(selected.get("mape_paper", float("nan"))),
        "smape_paper": float(selected.get("smape_paper", float("nan"))),
        "normalized_rmse": selected.get("normalized_rmse"),
        "normalized_accuracy": selected.get("normalized_accuracy"),
        "normalized_mae": selected.get("normalized_mae"),
        "normalized_mape": selected.get("normalized_mape"),
        "normalized_smape": selected.get("normalized_smape"),
        "original_scale_rmse": selected.get("original_scale_rmse"),
        "original_scale_accuracy": selected.get("original_scale_accuracy"),
        "original_scale_mae": selected.get("original_scale_mae"),
        "original_scale_mape": selected.get("original_scale_mape"),
        "original_scale_smape": selected.get("original_scale_smape"),
        "prediction_shape": _shape_to_tuple(prediction_shape),
        "metric_space": str(selected.get("metric_space", selected.get("metric_space_current", "normalized_minmax_space"))),
        "metric_space_used": str(selected.get("metric_space_used", selected.get("metric_space", "normalized_minmax_space"))),
        "rmse_metric_space": str(
            selected.get(
                "rmse_metric_space",
                selected.get("metric_space_used", selected.get("metric_space", "normalized_minmax_space")),
            )
        ),
        "smape_metric_space": str(
            selected.get(
                "smape_metric_space",
                selected.get("metric_space_used", selected.get("metric_space", "normalized_minmax_space")),
            )
        ),
        "metric_space_current": str(selected.get("metric_space_current", "normalized_minmax_space")),
        "metric_space_paper": str(selected.get("metric_space_paper", "original_sales_space")),
        "paper_metric_aligned": bool(selected.get("paper_metric_aligned", False)),
        "inverse_transform_applied": bool(selected.get("inverse_transform_applied", False)),
        "inverse_transform_available": bool(selected.get("inverse_transform_available", False)),
        "metric_protocol_note": str(selected.get("metric_protocol_note", "")),
        "metric_notes": str(selected.get("metric_notes", "")),
        "meta": method_meta,
    }
    for source_diagnostic_key in (
        "failed_source_count",
        "failed_source_keys",
        "skipped_source_count",
        "skipped_nonfinite_source_count",
        "selected_source_count",
        "valid_source_count",
        "failed_sources",
        "source_failure_messages",
    ):
        if source_diagnostic_key in method_meta:
            result[source_diagnostic_key] = method_meta[source_diagnostic_key]

    for feature_diagnostic_key in (
        "feature_source",
        "knn_feature_mode",
        "source_selection_feature_cols",
        "model_feature_cols",
        "feature_consistency_status",
        "json_only_features",
        "runtime_only_features",
        "source_numeric_na_repaired",
        "repaired_columns",
    ):
        if feature_diagnostic_key in method_meta:
            result[feature_diagnostic_key] = method_meta[feature_diagnostic_key]

    for diagnostic_key in (
        "y_pred_nan_count",
        "y_pred_inf_count",
        "y_true_nan_count",
        "y_true_inf_count",
        "X_test_nan_count",
        "X_test_inf_count",
        "model_weight_nan_count",
        "model_weight_inf_count",
    ):
        if diagnostic_key in selected:
            result[diagnostic_key] = selected[diagnostic_key]

    result["original_scale_rmse"] = _coalesce_metric(result.get("original_scale_rmse"), result.get("rmse_paper"))
    result["original_scale_accuracy"] = _coalesce_metric(
        result.get("original_scale_accuracy"),
        result.get("accuracy_paper"),
    )
    result["original_scale_mae"] = _coalesce_metric(result.get("original_scale_mae"), result.get("mae_paper"))
    result["original_scale_smape"] = _coalesce_metric(result.get("original_scale_smape"), result.get("smape_paper"))
    result["normalized_rmse"] = _coalesce_metric(result.get("normalized_rmse"), result.get("rmse_current"))
    result["normalized_accuracy"] = _coalesce_metric(result.get("normalized_accuracy"), result.get("accuracy_current"))
    result["normalized_mae"] = _coalesce_metric(result.get("normalized_mae"), result.get("mae_current"))
    result["normalized_smape"] = _coalesce_metric(result.get("normalized_smape"), result.get("smape_current"))
    smape_value = _coalesce_metric(
        result.get("original_scale_smape"),
        result.get("smape"),
        result.get("normalized_smape"),
    )
    result["smape"] = float(smape_value) if smape_value is not None else float("nan")
    fallback_metric_space = result.get("metric_space_used", result.get("metric_space", "normalized_minmax_space"))
    if "rmse_metric_space" in selected or "smape_metric_space" in selected:
        result["metric_space_used"] = _summarize_metric_space(
            result.get("rmse_metric_space"),
            result.get("smape_metric_space"),
            fallback_metric_space,
        )
        result["metric_space"] = result["metric_space_used"]
    elif _coalesce_metric(result.get("original_scale_smape")) is not None:
        result["metric_space_used"] = str(result.get("metric_space_paper", "original_sales_space"))
    elif _coalesce_metric(result.get("normalized_smape")) is not None:
        result["metric_space_used"] = str(result.get("metric_space_current", "normalized_minmax_space"))

    return result


def _prepare_path_from_config(
    dataset_name: str,
    config: Optional[Any],
    verbose_mode: str = "summary",
) -> str:
    """当用户未显式给 data_path 时，从配置推断 CSV 路径。"""
    if config is None:
        cfg = Config(
            config_file="config.yaml",
            supply_chain_file="supply_chain.yaml",
            verbose=(str(verbose_mode).lower() == "full"),
        )
    else:
        cfg = config

    cfg_dataset_path = None
    if hasattr(cfg, "get"):
        cfg_dataset_path = cfg.get("dataset.path", None)
    if cfg_dataset_path is None and hasattr(cfg, "dataset") and getattr(cfg.dataset, "path", None):
        cfg_dataset_path = str(cfg.dataset.path)

    if not cfg_dataset_path:
        raise ValueError(
            f"data_path is None and cannot infer path from config for dataset={dataset_name}."
        )

    return str(Path(str(cfg_dataset_path)) / "train.csv")


def prepare_base_data_for_experiments(
    dataset_name: str,
    data_path: Optional[str],
    config: Any = None,
    verbose_mode: str = "summary",
) -> Dict[str, pd.DataFrame]:
    """
    统一实验数据准备（仅模块2处理，不做训练）。

    Returns:
        {
          "raw_df": ...,
          "processed_df": ...,
          "source_df": ...,
          "target_df": ...,
        }
    """
    logger = _get_logger()
    logger.info("[prepare_base_data_for_experiments] Start. dataset=%s", dataset_name)

    if data_path is None:
        data_path = _prepare_path_from_config(
            dataset_name=dataset_name,
            config=config,
            verbose_mode=verbose_mode,
        )

    run_config = config
    if run_config is None:
        run_config = Config(
            config_file="config.yaml",
            supply_chain_file="supply_chain.yaml",
            verbose=(str(verbose_mode).lower() == "full"),
        )

    raw_df = load_dataset(dataset_name=dataset_name, data_path=str(data_path))
    processed_df = extract_datetime_features(raw_df)
    source_df, target_df = build_source_target_split(processed_df, run_config)

    source_df.attrs["dataset_name"] = str(dataset_name)
    target_df.attrs["dataset_name"] = str(dataset_name)
    _save_split_protocol_summary(
        dataset_name=str(dataset_name),
        source_df=source_df,
        target_df=target_df,
        config=run_config,
    )

    logger.info(
        "[prepare_base_data_for_experiments] Finished. raw=%d processed=%d source=%d target=%d",
        len(raw_df),
        len(processed_df),
        len(source_df),
        len(target_df),
    )

    return {
        "raw_df": raw_df,
        "processed_df": processed_df,
        "source_df": source_df,
        "target_df": target_df,
    }


def run_ss_tl_experiment(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    feature_cols: Sequence[str],
    horizon: int = 1,
    window_size: int = 10,
    learning_rate: float = 0.001,
    source_epochs: int = 3,
    target_epochs: int = 3,
    batch_size: int = 16,
    metric_protocol: Optional[Dict[str, Any]] = None,
    group_cols: Sequence[str] = ("entity_id", "item_id"),
) -> Dict[str, Any]:
    """运行 SS-TL，并返回统一结构。"""
    try:
        from src.transfer_methods.single_source_tl import (
            build_target_model_from_source,
            evaluate_regression_model,
            fine_tune_target_model,
            train_source_model,
        )
        from src.evaluation.metrics import compute_metrics_with_protocol
        from src.utils.finite_diagnostics import validate_finite_array
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"SS-TL dependency missing: {exc}") from exc

    logger = _get_logger()
    logger.info("[run_ss_tl_experiment] Start.")

    cols = _ensure_feature_cols(source_df, target_df, feature_cols)

    # 严格论文口径：SS-TL 使用 KNN 选最近单源（k=1），不再取排序后第一个 source。
    resolved_group_cols = tuple(group_cols)
    if len(resolved_group_cols) != 2:
        raise ValueError(f"group_cols must contain exactly two columns: {resolved_group_cols}")
    sort_cols = [
        col
        for col in dict.fromkeys(("entity_id", "item_id", *resolved_group_cols, "date"))
        if col in source_df.columns
    ]
    sorted_source = source_df.sort_values(sort_cols).copy()
    selector = SourceSelector()
    selection = selector.select_top_k_sources(
        target_df=target_df,
        source_df=sorted_source,
        feature_cols=cols,
        k=1,
        group_cols=resolved_group_cols,
        weight_mode="inverse_distance",
        debug_mode=False,
        include_sales_in_knn=True,
    )
    selected = selection.get("sources", []) if isinstance(selection, dict) else []
    if not selected:
        raise ValueError("SS-TL KNN source selection returned empty sources.")

    source_key = selected[0].get("source_key")
    if isinstance(source_key, list):
        source_key = tuple(source_key)
    if not isinstance(source_key, tuple) or len(source_key) != len(resolved_group_cols):
        raise ValueError(f"Invalid SS-TL source_key from selector: {source_key}")

    source_mask = pd.Series(True, index=sorted_source.index)
    for col, value in zip(resolved_group_cols, source_key):
        source_mask &= sorted_source[col] == value
    single_source_df = sorted_source[source_mask].copy()

    keep_cols = ["date", "entity_id", "item_id", *resolved_group_cols, *cols]
    keep_cols = [c for c in keep_cols if c in single_source_df.columns]
    single_source_df = single_source_df[keep_cols].copy()

    single_source_df.attrs["split_role"] = "source"
    single_source_df.attrs["split_mode"] = "ratio"
    single_source_df.attrs["split_config"] = {
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
    }
    single_source_df = fill_source_numeric_na(single_source_df, feature_columns=cols)

    target_min_df = target_df[[c for c in keep_cols if c in target_df.columns]].copy()
    target_min_df.attrs = target_df.attrs.copy()

    src_train, src_val, src_test = temporal_split_by_ratio_or_dates(single_source_df)
    src_train, src_val, src_test, _, _ = normalize_features(
        src_train, src_val, src_test, feature_columns=cols
    )

    X_source, y_source = build_tabular_sequence(
        src_train, horizon=horizon, window_size=window_size, feature_columns=cols
    )
    if len(y_source) == 0:
        raise ValueError("SS-TL source windows are empty; adjust window_size/horizon.")
    X_source = to_cnn_tensor(X_source)

    tgt_train, tgt_val, tgt_test = temporal_split_by_ratio_or_dates(target_min_df)
    tgt_train, tgt_val, tgt_test, tgt_scaler, tgt_feature_columns = normalize_features(
        tgt_train, tgt_val, tgt_test, feature_columns=cols
    )

    X_target_train, y_target_train = build_tabular_sequence(
        tgt_train, horizon=horizon, window_size=window_size, feature_columns=tgt_feature_columns
    )
    X_target_val, y_target_val = build_tabular_sequence(
        tgt_val, horizon=horizon, window_size=window_size, feature_columns=tgt_feature_columns
    )
    X_target_test, y_target_test = build_tabular_sequence(
        tgt_test, horizon=horizon, window_size=window_size, feature_columns=tgt_feature_columns
    )

    if len(y_target_train) == 0 or len(y_target_test) == 0:
        raise ValueError("SS-TL target windows are empty; adjust window_size/horizon.")

    X_target_train = to_cnn_tensor(X_target_train)
    X_target_val = to_cnn_tensor(X_target_val)
    X_target_test = to_cnn_tensor(X_target_test)

    if X_source.shape[1:] != X_target_train.shape[1:]:
        raise ValueError(
            f"SS-TL shape mismatch: source={X_source.shape[1:]} target={X_target_train.shape[1:]}"
        )

    input_shape = X_source.shape[1:]

    source_model = train_source_model(
        X_source=X_source,
        y_source=y_source,
        input_shape=input_shape,
        learning_rate=learning_rate,
        epochs=source_epochs,
        batch_size=batch_size,
    )

    target_model, frozen_names = build_target_model_from_source(
        source_model=source_model,
        input_shape=input_shape,
        learning_rate=learning_rate,
        freeze_first_n_layers=4,
    )

    target_model = fine_tune_target_model(
        target_model=target_model,
        X_target_train=X_target_train,
        y_target_train=y_target_train,
        X_target_val=X_target_val,
        y_target_val=y_target_val,
        epochs=target_epochs,
        batch_size=batch_size,
    )

    ss_raw = evaluate_regression_model(
        model=target_model,
        X_test=X_target_test,
        y_test=y_target_test,
    )
    y_pred = target_model.predict(X_target_test, verbose=0)
    ss_raw.update(
        validate_finite_array(
            y_pred,
            name="y_pred",
            context={
                key: value
                for key, value in ss_raw.items()
                if key.endswith("_nan_count")
                or key.endswith("_inf_count")
                or key in {"X_test_shape", "y_true_shape", "y_pred_shape"}
            },
        )
    )
    ss_raw.update(
        compute_metrics_with_protocol(
            y_true=y_target_test,
            y_pred=y_pred.flatten(),
            metric_protocol=metric_protocol,
            sales_scaler=tgt_scaler,
            feature_columns=tgt_feature_columns,
        )
    )
    ss_raw.setdefault(
        "y_pred_shape",
        _shape_to_tuple(getattr(y_pred, "shape", None)),
    )

    result = {
        "method": "SS-TL",
        "rmse": float(ss_raw["rmse"]),
        "accuracy": float(ss_raw["accuracy"]),
        "mae": float(ss_raw.get("mae", float("nan"))),
        "mape": float(ss_raw.get("mape", float("nan"))),
        "smape": float(ss_raw.get("smape", float("nan"))),
        "rmse_current": float(ss_raw.get("rmse_current", float("nan"))),
        "accuracy_current": float(ss_raw.get("accuracy_current", float("nan"))),
        "mae_current": float(ss_raw.get("mae_current", float("nan"))),
        "mape_current": float(ss_raw.get("mape_current", float("nan"))),
        "smape_current": float(ss_raw.get("smape_current", float("nan"))),
        "rmse_paper": float(ss_raw.get("rmse_paper", float("nan"))),
        "accuracy_paper": float(ss_raw.get("accuracy_paper", float("nan"))),
        "mae_paper": float(ss_raw.get("mae_paper", float("nan"))),
        "mape_paper": float(ss_raw.get("mape_paper", float("nan"))),
        "smape_paper": float(ss_raw.get("smape_paper", float("nan"))),
        "normalized_rmse": ss_raw.get("normalized_rmse"),
        "normalized_accuracy": ss_raw.get("normalized_accuracy"),
        "normalized_mae": ss_raw.get("normalized_mae"),
        "normalized_mape": ss_raw.get("normalized_mape"),
        "normalized_smape": ss_raw.get("normalized_smape"),
        "original_scale_rmse": ss_raw.get("original_scale_rmse"),
        "original_scale_accuracy": ss_raw.get("original_scale_accuracy"),
        "original_scale_mae": ss_raw.get("original_scale_mae"),
        "original_scale_mape": ss_raw.get("original_scale_mape"),
        "original_scale_smape": ss_raw.get("original_scale_smape"),
        "prediction_shape": _shape_to_tuple(ss_raw.get("y_pred_shape")),
        "metric_space": str(ss_raw.get("metric_space", ss_raw.get("metric_space_current", "normalized_minmax_space"))),
        "metric_space_used": str(ss_raw.get("metric_space_used", ss_raw.get("metric_space", "normalized_minmax_space"))),
        "rmse_metric_space": str(
            ss_raw.get(
                "rmse_metric_space",
                ss_raw.get("metric_space_used", ss_raw.get("metric_space", "normalized_minmax_space")),
            )
        ),
        "smape_metric_space": str(
            ss_raw.get(
                "smape_metric_space",
                ss_raw.get("metric_space_used", ss_raw.get("metric_space", "normalized_minmax_space")),
            )
        ),
        "metric_space_current": str(ss_raw.get("metric_space_current", "normalized_minmax_space")),
        "metric_space_paper": str(ss_raw.get("metric_space_paper", "original_sales_space")),
        "paper_metric_aligned": bool(ss_raw.get("paper_metric_aligned", False)),
        "inverse_transform_applied": bool(ss_raw.get("inverse_transform_applied", False)),
        "inverse_transform_available": bool(ss_raw.get("inverse_transform_available", False)),
        "metric_protocol_note": str(ss_raw.get("metric_protocol_note", "")),
        "metric_notes": str(ss_raw.get("metric_notes", "")),
        "y_pred_nan_count": ss_raw.get("y_pred_nan_count"),
        "y_pred_inf_count": ss_raw.get("y_pred_inf_count"),
        "y_true_nan_count": ss_raw.get("y_true_nan_count"),
        "y_true_inf_count": ss_raw.get("y_true_inf_count"),
        "X_test_nan_count": ss_raw.get("X_test_nan_count"),
        "X_test_inf_count": ss_raw.get("X_test_inf_count"),
        "model_weight_nan_count": ss_raw.get("model_weight_nan_count"),
        "model_weight_inf_count": ss_raw.get("model_weight_inf_count"),
        "meta": {
            "source_key": tuple(first_key),
            "source_selection_policy": "knn_top1",
            "source_distance": float(selected[0].get("distance", 0.0)),
            "source_weight": float(selected[0].get("weight", 1.0)),
            "selected_sources": selected,
            **runtime_selection_meta(selection),
            "feature_cols": list(cols),
            "input_shape": tuple(input_shape),
            "frozen_layers": list(frozen_names),
            "horizon": int(horizon),
            "window_size": int(window_size),
        },
    }

    logger.info(
        "[run_ss_tl_experiment] Finished. rmse=%.4f accuracy=%.4f",
        result["rmse"],
        result["accuracy"],
    )
    return result


def run_mswa_experiment(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    feature_cols: Sequence[str],
    k: int = 3,
    number_of_sources: Optional[int] = None,
    horizon: int = 1,
    window_size: int = 10,
    weight_mode: str = "inverse_distance",
    include_sales_in_knn: bool = True,
    learning_rate: float = 0.001,
    source_epochs: int = 3,
    target_epochs: int = 3,
    batch_size: int = 16,
    metric_protocol: Optional[Dict[str, Any]] = None,
    group_cols: Sequence[str] = ("entity_id", "item_id"),
) -> Dict[str, Any]:
    """运行 MSWA-TL，并返回统一结构。"""
    try:
        from src.transfer_methods.mswa_tl import run_mswa_tl
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"MSWA-TL dependency missing: {exc}") from exc

    k = int(number_of_sources) if number_of_sources is not None else int(k)
    raw = run_mswa_tl(
        source_df=source_df,
        target_df=target_df,
        feature_cols=feature_cols,
        k=k,
        group_cols=tuple(group_cols),
        horizon=horizon,
        window_size=window_size,
        weight_mode=weight_mode,
        learning_rate=learning_rate,
        source_epochs=source_epochs,
        target_epochs=target_epochs,
        batch_size=batch_size,
    )
    return _extract_method_metrics(raw, method_name="MSWA-TL", metric_protocol=metric_protocol)


def run_mssb_experiment(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    feature_cols: Sequence[str],
    k: int = 3,
    number_of_sources: Optional[int] = None,
    horizon: int = 1,
    window_size: int = 10,
    weight_mode: str = "inverse_distance",
    include_sales_in_knn: bool = True,
    learning_rate: float = 0.001,
    source_epochs: int = 3,
    target_epochs: int = 3,
    batch_size: int = 16,
    metric_protocol: Optional[Dict[str, Any]] = None,
    group_cols: Sequence[str] = ("entity_id", "item_id"),
) -> Dict[str, Any]:
    """运行 MSSB-TL，并返回统一结构。"""
    try:
        from src.transfer_methods.mssb_tl import run_mssb_tl
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"MSSB-TL dependency missing: {exc}") from exc

    k = int(number_of_sources) if number_of_sources is not None else int(k)
    raw = run_mssb_tl(
        source_df=source_df,
        target_df=target_df,
        feature_cols=feature_cols,
        k=k,
        group_cols=tuple(group_cols),
        horizon=horizon,
        window_size=window_size,
        weight_mode=weight_mode,
        learning_rate=learning_rate,
        source_epochs=source_epochs,
        target_epochs=target_epochs,
        batch_size=batch_size,
    )
    return _extract_method_metrics(raw, method_name="MSSB-TL", metric_protocol=metric_protocol)


def run_msml_experiment(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    feature_cols: Sequence[str],
    k: int = 3,
    number_of_sources: Optional[int] = None,
    horizon: int = 1,
    window_size: int = 10,
    weight_mode: str = "inverse_distance",
    include_sales_in_knn: bool = True,
    learning_rate: float = 0.001,
    source_epochs: int = 3,
    target_epochs: int = 3,
    batch_size: int = 16,
    metric_protocol: Optional[Dict[str, Any]] = None,
    group_cols: Sequence[str] = ("entity_id", "item_id"),
) -> Dict[str, Any]:
    """运行 MSML-TL，并返回统一结构。"""
    try:
        from src.transfer_methods.msml_tl import run_msml_tl
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"MSML-TL dependency missing: {exc}") from exc

    k = int(number_of_sources) if number_of_sources is not None else int(k)
    raw = run_msml_tl(
        source_df=source_df,
        target_df=target_df,
        feature_cols=feature_cols,
        k=k,
        group_cols=tuple(group_cols),
        horizon=horizon,
        window_size=window_size,
        weight_mode=weight_mode,
        learning_rate=learning_rate,
        source_epochs=source_epochs,
        target_epochs=target_epochs,
        batch_size=batch_size,
    )
    return _extract_method_metrics(raw, method_name="MSML-TL", metric_protocol=metric_protocol)


def run_msml_rfe_experiment(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    feature_cols: Sequence[str],
    k: int = 3,
    number_of_sources: Optional[int] = None,
    horizon: int = 1,
    window_size: int = 10,
    weight_mode: str = "inverse_distance",
    estimator_name: str = "random_forest",
    keep_ratio: float = 0.5,
    include_sales_in_knn: bool = True,
    learning_rate: float = 0.001,
    source_epochs: int = 3,
    target_epochs: int = 3,
    batch_size: int = 16,
    metric_protocol: Optional[Dict[str, Any]] = None,
    group_cols: Sequence[str] = ("entity_id", "item_id"),
) -> Dict[str, Any]:
    """运行 MSML-TL-RFE，并返回统一结构。"""
    try:
        from src.transfer_methods.msml_tl_rfe import run_msml_tl_rfe
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"MSML-TL-RFE dependency missing: {exc}") from exc

    k = int(number_of_sources) if number_of_sources is not None else int(k)
    raw = run_msml_tl_rfe(
        source_df=source_df,
        target_df=target_df,
        feature_cols=feature_cols,
        k=k,
        group_cols=tuple(group_cols),
        horizon=horizon,
        window_size=window_size,
        weight_mode=weight_mode,
        estimator_name=estimator_name,
        keep_ratio=keep_ratio,
        learning_rate=learning_rate,
        source_epochs=source_epochs,
        target_epochs=target_epochs,
        batch_size=batch_size,
    )
    return _extract_method_metrics(raw, method_name="MSML-TL-RFE", metric_protocol=metric_protocol)


def run_all_experiments(
    dataset_name: str = "Dataset1",
    data_path: Optional[str] = None,
    config: Optional[Any] = None,
    feature_cols: Optional[Sequence[str]] = None,
    k: int = 3,
    number_of_sources: Optional[int] = None,
    horizon: int = 1,
    window_size: int = 10,
    weight_mode: str = "inverse_distance",
    estimator_name: str = "random_forest",
    keep_ratio: float = 0.5,
    include_sales_in_knn: bool = True,
    learning_rate: float = 0.001,
    source_epochs: int = 2,
    target_epochs: int = 2,
    batch_size: int = 16,
    enabled_methods: Optional[Iterable[str]] = None,
    verbose_mode: str = "summary",
    show_method_progress: bool = True,
    strict_paper_mode: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    统一运行全部实验方法，返回统一结构结果。

    失败策略：失败即抛错。
    """
    set_verbose_mode(verbose_mode)
    logger = _get_logger()
    logger.info(
        "[run_all_experiments] Start. dataset=%s include_sales_in_knn=%s",
        dataset_name,
        bool(include_sales_in_knn),
    )

    protocol = load_paper_protocol(config)
    selected_number_of_sources = int(k if number_of_sources is None else number_of_sources)
    strict_mode = resolve_strict_paper_mode(config=config, explicit=strict_paper_mode)
    strict_multi_source_count = int(
        _get_nested(config, "paper_reproduction.strict_source_selection.multi_source_top_k", 3)
    )
    protocol["strict_paper_mode"] = strict_mode
    protocol["paper_strict_mode"] = strict_mode
    protocol.setdefault("metric_protocol", {})["strict_paper_metrics"] = bool(strict_mode)
    validate_report = validate_paper_protocol_config(protocol=protocol, strict_paper_mode=strict_mode)
    logger.info(
        "[run_all_experiments] paper_protocol_validation status=%s failures=%d warnings=%d",
        validate_report["status"],
        len(validate_report["failures"]),
        len(validate_report["warnings"]),
    )

    if isinstance(config, dict):
        config.setdefault("paper_reproduction", {})["strict_paper_mode"] = strict_mode
        config["paper_reproduction"]["paper_strict_mode"] = strict_mode
        config["paper_reproduction"].setdefault("metric_protocol", {})["strict_paper_metrics"] = bool(strict_mode)
    elif strict_mode:
        logger.warning(
            "[run_all_experiments] strict_paper_mode is enabled, but config is not a dict; "
            "TODO: ensure custom config object exposes paper_reproduction.strict_paper_mode "
            "for split-layer strict assertions."
        )

    data_bundle = prepare_base_data_for_experiments(
        dataset_name=dataset_name,
        data_path=data_path,
        config=config,
        verbose_mode=verbose_mode,
    )
    source_df = data_bundle["source_df"]
    target_df = data_bundle["target_df"]

    target_domains = []
    if {"entity_id", "item_id"}.issubset(set(target_df.columns)) and not target_df.empty:
        pairs = (
            target_df[["entity_id", "item_id"]]
            .drop_duplicates()
            .head(20)
            .to_numpy()
            .tolist()
        )
        target_domains = [f"entity={p[0]}|item={p[1]}" for p in pairs]
    logger.info(
        "[run_all_experiments] Context. dataset=%s target_domains=%s",
        dataset_name,
        target_domains,
    )
    if str(verbose_mode).lower() == "summary":
        print(
            f"[experiment_start] dataset={dataset_name} "
            f"target_domains={target_domains} "
            f"include_sales_in_knn={bool(include_sales_in_knn)}"
        )

    cols = _ensure_feature_cols(source_df, target_df, feature_cols)

    methods = list(enabled_methods) if enabled_methods is not None else list(DEFAULT_METHODS)
    method_set = set(methods)
    unknown = sorted(method_set.difference(DEFAULT_METHODS))
    if unknown:
        raise ValueError(f"Unknown methods in enabled_methods: {unknown}")

    results: List[Dict[str, Any]] = []

    method_total = len(methods)
    for method_idx, method in enumerate(methods, start=1):
        effective_source_count = selected_number_of_sources
        if strict_mode and method in MULTI_SOURCE_TL_METHODS:
            effective_source_count = strict_multi_source_count

        requested_source_count = effective_source_count if method in MULTI_SOURCE_TL_METHODS else (1 if method == "SS-TL" else 0)
        ensure_paper_track_allowed(
            method_name=method,
            requested_source_count=requested_source_count,
            protocol=protocol,
            strict_paper_mode=strict_mode,
        )
        if str(verbose_mode).lower() == "summary" and show_method_progress:
            print_method_start(method_idx, method_total, method)
        logger.info("[run_all_experiments] Running method=%s", method)
        method_start_time = time.perf_counter()
        try:
            if method == "No-TL":
                one = run_no_tl_experiment(
                    target_df=target_df,
                    horizon=horizon,
                    window_size=window_size,
                    learning_rate=learning_rate,
                    target_epochs=target_epochs,
                    batch_size=batch_size,
                    metric_protocol=protocol.get("metric_protocol", {}),
                    feature_cols=cols,
                )
            elif method == "SS-TL":
                one = run_ss_tl_experiment(
                    source_df=source_df,
                    target_df=target_df,
                    feature_cols=cols,
                    horizon=horizon,
                    window_size=window_size,
                    learning_rate=learning_rate,
                    source_epochs=source_epochs,
                    target_epochs=target_epochs,
                    batch_size=batch_size,
                    metric_protocol=protocol.get("metric_protocol", {}),
                )
            elif method == "MSWA-TL":
                one = run_mswa_experiment(
                    source_df=source_df,
                    target_df=target_df,
                    feature_cols=cols,
                    number_of_sources=effective_source_count,
                    horizon=horizon,
                    window_size=window_size,
                    weight_mode=weight_mode,
                    include_sales_in_knn=include_sales_in_knn,
                    learning_rate=learning_rate,
                    source_epochs=source_epochs,
                    target_epochs=target_epochs,
                    batch_size=batch_size,
                    metric_protocol=protocol.get("metric_protocol", {}),
                )
            elif method == "MSSB-TL":
                one = run_mssb_experiment(
                    source_df=source_df,
                    target_df=target_df,
                    feature_cols=cols,
                    number_of_sources=effective_source_count,
                    horizon=horizon,
                    window_size=window_size,
                    weight_mode=weight_mode,
                    include_sales_in_knn=include_sales_in_knn,
                    learning_rate=learning_rate,
                    source_epochs=source_epochs,
                    target_epochs=target_epochs,
                    batch_size=batch_size,
                    metric_protocol=protocol.get("metric_protocol", {}),
                )
            elif method == "MSML-TL":
                one = run_msml_experiment(
                    source_df=source_df,
                    target_df=target_df,
                    feature_cols=cols,
                    number_of_sources=effective_source_count,
                    horizon=horizon,
                    window_size=window_size,
                    weight_mode=weight_mode,
                    include_sales_in_knn=include_sales_in_knn,
                    learning_rate=learning_rate,
                    source_epochs=source_epochs,
                    target_epochs=target_epochs,
                    batch_size=batch_size,
                    metric_protocol=protocol.get("metric_protocol", {}),
                )
            elif method == "MSML-TL-RFE":
                one = run_msml_rfe_experiment(
                    source_df=source_df,
                    target_df=target_df,
                    feature_cols=cols,
                    number_of_sources=effective_source_count,
                    horizon=horizon,
                    window_size=window_size,
                    weight_mode=weight_mode,
                    estimator_name=estimator_name,
                    keep_ratio=keep_ratio,
                    include_sales_in_knn=include_sales_in_knn,
                    learning_rate=learning_rate,
                    source_epochs=source_epochs,
                    target_epochs=target_epochs,
                    batch_size=batch_size,
                    metric_protocol=protocol.get("metric_protocol", {}),
                )
            else:
                raise ValueError(f"Unsupported method={method}")
        except Exception as exc:
            raise RuntimeError(f"Method {method} failed: {exc}") from exc

        elapsed_time = float(time.perf_counter() - method_start_time)
        existing_training_time = one.get("training_time")
        if existing_training_time is None or (
            isinstance(existing_training_time, (float, np.floating)) and np.isnan(existing_training_time)
        ):
            one["training_time"] = elapsed_time

        if "rmse_metric_space" in one or "smape_metric_space" in one:
            one["metric_space_used"] = _summarize_metric_space(
                one.get("rmse_metric_space"),
                one.get("smape_metric_space"),
                one.get("metric_space_used", one.get("metric_space", "normalized_minmax_space")),
            )
            one["metric_space"] = one["metric_space_used"]
        elif _coalesce_metric(one.get("original_scale_smape"), one.get("smape_paper")) is not None:
            one["metric_space_used"] = str(one.get("metric_space_paper", "original_sales_space"))
        elif _coalesce_metric(one.get("normalized_smape"), one.get("smape_current")) is not None:
            one["metric_space_used"] = str(one.get("metric_space_current", "normalized_minmax_space"))

        one["protocol"] = build_alignment_fields(
            method_name=str(one.get("method", method)),
            requested_source_count=requested_source_count,
            method_meta=one.get("meta", {}),
            base_data=data_bundle,
            protocol=protocol,
        )

        results.append(one)
        if str(verbose_mode).lower() == "summary" and show_method_progress:
            print_method_result(
                one["method"],
                float(one["rmse"]),
                float(one["accuracy"]),
                smape=float(one.get("smape", np.nan)),
                original_scale_smape=one.get("original_scale_smape"),
            )
        logger.info(
            "[run_all_experiments] Completed method=%s smape=%.4f original_scale_smape=%s rmse=%.4f accuracy=%.4f",
            one["method"],
            float(one.get("smape", np.nan)),
            str(one.get("original_scale_smape", np.nan)),
            float(one["rmse"]),
            float(one["accuracy"]),
        )

    experiment_results = {
        "meta": {
            "dataset_name": dataset_name,
            "target_domains": target_domains,
            "feature_cols": list(cols),
            "k": int(k),
            "number_of_sources": int(selected_number_of_sources),
            "number_of_methods": int(len(methods)),
            "horizon": int(horizon),
            "window_size": int(window_size),
            "weight_mode": weight_mode,
            "include_sales_in_knn": bool(include_sales_in_knn),
            "enabled_methods": methods,
            "strict_paper_mode": strict_mode,
        },
        "results": results,
    }

    logger.info("[run_all_experiments] Finished. methods=%d", len(results))
    return experiment_results


def results_to_dataframe(experiment_results: Dict[str, Any]) -> pd.DataFrame:
    """将统一实验结果转换为 DataFrame。"""
    if not isinstance(experiment_results, dict) or "results" not in experiment_results:
        raise ValueError("experiment_results must be a dict containing 'results'.")

    meta = experiment_results.get("meta", {}) if isinstance(experiment_results, dict) else {}
    dataset_name = meta.get("dataset_name", None) if isinstance(meta, dict) else None
    include_sales_in_knn = meta.get("include_sales_in_knn", None) if isinstance(meta, dict) else None
    strict_paper_mode = meta.get("strict_paper_mode", None) if isinstance(meta, dict) else None

    rows: List[Dict[str, Any]] = []
    for one in experiment_results.get("results", []):
        protocol = one.get("protocol", {}) if isinstance(one, dict) else {}
        rows.append(
            {
                "dataset": dataset_name,
                "method": one.get("method", "N/A"),
                "strict_paper_mode": strict_paper_mode,
                "training_time": float(one.get("training_time", np.nan)),
                "experiment_scope": protocol.get("experiment_scope", protocol.get("experiment_track", "paper")),
                "experiment_track": protocol.get("experiment_track", "paper"),
                "source_protocol_aligned": bool(protocol.get("source_protocol_aligned", False)),
                "alignment_status": protocol.get("alignment_status", STATUS_PARTIAL),
                "metric_alignment_status": protocol.get("metric_alignment_status", STATUS_PARTIAL),
                "split_alignment_status": protocol.get("split_alignment_status", STATUS_PARTIAL),
                "source_pretrained_alignment_status": protocol.get(
                    "source_pretrained_alignment_status", STATUS_PARTIAL
                ),
                "paper_metric_space": protocol.get("paper_metric_space", "TODO"),
                "current_metric_space": protocol.get("current_metric_space", "normalized_minmax_space"),
                "metric_space_current": one.get(
                    "metric_space_current",
                    protocol.get("current_metric_space", "normalized_minmax_space"),
                ),
                "metric_space_paper": one.get(
                    "metric_space_paper",
                    protocol.get("paper_metric_space", "original_sales_space"),
                ),
                "paper_metric_aligned": bool(one.get("paper_metric_aligned", False)),
                "inverse_transform_applied": bool(one.get("inverse_transform_applied", False)),
                "metric_notes": one.get("metric_notes", ""),
                "paper_split_reference": protocol.get("paper_split_reference", "TODO"),
                "source_count": protocol.get("source_count", protocol.get("requested_source_count", np.nan)),
                "pretrained_model_count": protocol.get(
                    "pretrained_model_count", protocol.get("actual_pretrained_model_count", np.nan)
                ),
                "requested_source_count": protocol.get("requested_source_count", np.nan),
                "actual_pretrained_model_count": protocol.get("actual_pretrained_model_count", np.nan),
                "target_window_days": protocol.get("target_window_days", np.nan),
                "target_window_expected_days": protocol.get("target_window_expected_days", np.nan),
                "target_window_range_days": protocol.get("target_window_range_days", np.nan),
                "target_window_unique_days": protocol.get("target_window_unique_days", np.nan),
                "target_strict_paper_mode": protocol.get("target_strict_paper_mode", "False"),
                "metric_space": one.get("metric_space", one.get("metric_space_current", "normalized_minmax_space")),
                "metric_space_used": one.get("metric_space_used", one.get("metric_space", "normalized_minmax_space")),
                "rmse_metric_space": one.get(
                    "rmse_metric_space",
                    one.get("metric_space_used", one.get("metric_space", "normalized_minmax_space")),
                ),
                "smape_metric_space": one.get(
                    "smape_metric_space",
                    one.get("metric_space_used", one.get("metric_space", "normalized_minmax_space")),
                ),
                "rmse": float(one.get("rmse", np.nan)),
                "accuracy": float(one.get("accuracy", np.nan)),
                "mae": float(one.get("mae", np.nan)),
                "mape": float(one.get("mape", np.nan)),
                "smape": float(one.get("smape", np.nan)),
                "rmse_current": float(one.get("rmse_current", np.nan)),
                "accuracy_current": float(one.get("accuracy_current", np.nan)),
                "mae_current": float(one.get("mae_current", np.nan)),
                "mape_current": float(one.get("mape_current", np.nan)),
                "smape_current": float(one.get("smape_current", np.nan)),
                "rmse_paper": float(one.get("rmse_paper", np.nan)),
                "accuracy_paper": float(one.get("accuracy_paper", np.nan)),
                "mae_paper": float(one.get("mae_paper", np.nan)),
                "mape_paper": float(one.get("mape_paper", np.nan)),
                "smape_paper": float(one.get("smape_paper", np.nan)),
                "normalized_rmse": one.get("normalized_rmse", np.nan),
                "normalized_accuracy": one.get("normalized_accuracy", np.nan),
                "normalized_mae": one.get("normalized_mae", np.nan),
                "normalized_mape": one.get("normalized_mape", np.nan),
                "normalized_smape": one.get("normalized_smape", np.nan),
                "original_scale_rmse": one.get("original_scale_rmse", np.nan),
                "original_scale_accuracy": one.get("original_scale_accuracy", np.nan),
                "original_scale_mae": one.get("original_scale_mae", np.nan),
                "original_scale_mape": one.get("original_scale_mape", np.nan),
                "original_scale_smape": one.get("original_scale_smape", np.nan),
                "prediction_shape": str(one.get("prediction_shape", "N/A")),
                "include_sales_in_knn": include_sales_in_knn,
                "alignment_notes": protocol.get("alignment_notes", ""),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "dataset",
            "method",
            "strict_paper_mode",
            "training_time",
            "experiment_scope",
            "experiment_track",
            "source_protocol_aligned",
            "alignment_status",
            "metric_alignment_status",
            "split_alignment_status",
            "source_pretrained_alignment_status",
            "paper_metric_space",
            "current_metric_space",
            "metric_space_current",
            "metric_space_paper",
            "paper_metric_aligned",
            "inverse_transform_applied",
            "metric_notes",
            "paper_split_reference",
            "source_count",
            "pretrained_model_count",
            "requested_source_count",
            "actual_pretrained_model_count",
            "target_window_days",
            "target_window_expected_days",
            "target_window_range_days",
            "target_window_unique_days",
            "target_strict_paper_mode",
            "include_sales_in_knn",
            "metric_space",
            "metric_space_used",
            "rmse_metric_space",
            "smape_metric_space",
            "rmse",
            "accuracy",
            "mae",
            "mape",
            "smape",
            "rmse_current",
            "accuracy_current",
            "mae_current",
            "mape_current",
            "smape_current",
            "rmse_paper",
            "accuracy_paper",
            "mae_paper",
            "mape_paper",
            "smape_paper",
            "normalized_rmse",
            "normalized_accuracy",
            "normalized_mae",
            "normalized_mape",
            "normalized_smape",
            "original_scale_rmse",
            "original_scale_accuracy",
            "original_scale_mae",
            "original_scale_mape",
            "original_scale_smape",
            "prediction_shape",
            "alignment_notes",
        ],
    )


def save_results_to_csv(results_df: pd.DataFrame, output_path: str) -> None:
    """保存实验结果到 CSV，必要时自动创建目录。"""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False, encoding="utf-8")

    has_flag_col = "include_sales_in_knn" in results_df.columns
    flag_values = None
    if has_flag_col:
        try:
            flag_values = sorted(
                set(
                    results_df["include_sales_in_knn"]
                    .dropna()
                    .astype(object)
                    .tolist()
                )
            )
        except Exception:
            flag_values = "unavailable"

    msg = (
        f"[results_saved] path={str(out_path)} "
        f"include_sales_in_knn_column={has_flag_col} "
        f"include_sales_in_knn_values={flag_values}"
    )
    print(msg)
    try:
        _get_logger().info(msg)
    except Exception:
        pass


def print_results_table(results_df: pd.DataFrame) -> None:
    """按 sMAPE 升序打印简洁结果表。"""
    if results_df.empty:
        print("Results Table: (empty)")
        return

    if "smape" not in results_df.columns:
        print("No sMAPE column found. Please rerun experiments after metric update.")
        return

    table = results_df.sort_values(by="smape", ascending=True).reset_index(drop=True)
    print("Results Table:")
    print(table.to_string(index=False))
