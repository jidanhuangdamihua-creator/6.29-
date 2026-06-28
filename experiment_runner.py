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

from config import Config
from data_preprocessing import (
    build_source_target_split,
    build_tabular_sequence,
    extract_datetime_features,
    load_dataset,
    normalize_features,
    temporal_split_by_ratio_or_dates,
    to_cnn_tensor,
)
try:
    from environment import setup_logging
except ImportError:
    setup_logging = None

from src.utils.console_reporter import print_method_result, print_method_start
from src.utils.experiment_hyperparams import (
    FIXED_CLIPNORM,
    FIXED_DROPOUT,
    FIXED_EPOCHS,
    FIXED_LEARNING_RATE,
    fixed_hyperparams_summary,
)
from cnn_model import (
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_EARLY_STOPPING_MIN_DELTA,
)
from src.utils.runtime_control import apply_logging_level, log_level_name, set_verbose_mode
from src.utils.source_fillna import fill_source_numeric_na
from paper_reproduction_protocol import (
    MULTI_SOURCE_TL_METHODS,
    STATUS_PARTIAL,
    build_alignment_fields,
    ensure_paper_track_allowed,
    load_paper_protocol,
    resolve_strict_paper_mode,
    validate_paper_protocol_config,
)
from source_selector import SourceSelector


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
        "source_selection_window": "target_observed_window",
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


def _build_observed_target_window(target_df: pd.DataFrame) -> pd.DataFrame:
    logger = _get_logger()
    if target_df is None or target_df.empty:
        logger.info(
            "[source_selection] Using observed window only: %d rows (train+val), excluding test period",
            0,
        )
        return target_df

    target_train, target_val, _ = temporal_split_by_ratio_or_dates(target_df)
    observed = pd.concat([target_train, target_val], axis=0, ignore_index=True)
    observed = observed.sort_values(["date", "entity_id", "item_id"]).reset_index(drop=True)
    observed.attrs = target_df.attrs.copy()
    logger.info(
        "[source_selection] Using observed window only: %d rows (train+val), excluding test period",
        len(observed),
    )
    return observed


def run_no_tl_experiment(
    target_df: pd.DataFrame,
    horizon: int = 1,
    window_size: int = 10,
    learning_rate: float = FIXED_LEARNING_RATE,
    target_epochs: int = FIXED_EPOCHS,
    batch_size: int = 16,
    metric_protocol: Optional[Dict[str, Any]] = None,
    cnn_ablation_variant: str = "original",
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


NOT_RECORDED = "NOT_RECORDED_BY_CODE"


def _coalesce(*values: Any) -> Any:
    """Return the first non-None value from the arguments."""
    for v in values:
        if v is None:
            continue
        return v
    return None


def _coalesce_metric(*values: Any) -> Any:
    """Return the first numeric/reporting value that is not None/NaN."""
    for value in values:
        if value is None:
            continue
        if isinstance(value, (float, np.floating)) and np.isnan(value):
            continue
        return value
    return None


def _safe_float(value: Any, default: float = float("nan")) -> float:
    """Convert a reporting value to float, falling back for missing/invalid values."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _serialize_result_field(value: Any) -> Any:
    """Serialize nested result metadata for CSV output without inventing values."""
    if value is None:
        return NOT_RECORDED
    if isinstance(value, float) and np.isnan(value):
        return NOT_RECORDED
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=True, default=str)
    return value


# ---------------------------------------------------------------------------
# KNN 距离与权重审计字段工具函数（仅用于 CSV 输出，不参与训练/预测）
# ---------------------------------------------------------------------------

import math


def _serialize_pipe(values: Any) -> str:
    """将可迭代值用 | 拼接为字符串；None / 空返回空字符串。"""
    if values is None:
        return ""
    if isinstance(values, (list, tuple)):
        return "|".join(str(v) for v in values)
    if hasattr(values, "tolist") and not isinstance(values, (str, bytes, dict)):
        return _serialize_pipe(values.tolist())
    if isinstance(values, (np.ndarray,)):
        return _serialize_pipe(values.tolist())
    return str(values)


def _safe_float_knn(value: Any) -> float:
    """安全转换为 float；无法转换或为 None/NaN 时返回 NaN。"""
    if value is None:
        return float("nan")
    try:
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return float("nan")
        return v
    except (ValueError, TypeError):
        return float("nan")


def _compute_inverse_distance_weights(distances: Sequence[float]) -> List[float]:
    """按 1/distance 计算原始权重（不做归一化）。
    仅用于 CSV 审计字段，不参与训练。
    """
    raw: List[float] = []
    for d in distances:
        d_val = _safe_float_knn(d)
        if math.isnan(d_val) or d_val <= 0.0:
            raw.append(float("nan"))
        else:
            raw.append(1.0 / d_val)
    return raw


def _normalize_weights(raw_weights: Sequence[float]) -> List[float]:
    """对原始权重做归一化：w_i = raw_i / sum(raw_i)。
    仅用于 CSV 审计字段，不参与训练。
    """
    n = len(raw_weights)
    if n == 0:
        return []
    valid = [w for w in raw_weights if not math.isnan(w)]
    if not valid:
        return [float("nan")] * n
    total = sum(valid)
    if total <= 0.0:
        return [float("nan")] * n
    return [w / total if not math.isnan(w) else float("nan") for w in raw_weights]


def _attach_knn_audit_fields(selected_sources_raw: Any) -> Dict[str, str]:
    """从 selected_sources 列表中提取 KNN distance / weight 审计字段。"""
    if not isinstance(selected_sources_raw, list) or not selected_sources_raw:
        return {
            "selected_source_distances": "",
            "selected_source_weights_raw": "",
            "selected_source_weights_normalized": "",
        }
    distances: List[float] = []
    for s in selected_sources_raw:
        if isinstance(s, dict):
            distances.append(_safe_float_knn(s.get("distance")))
        else:
            distances.append(float("nan"))
    raw_weights = _compute_inverse_distance_weights(distances)
    normalized_weights = _normalize_weights(raw_weights)
    return {
        "selected_source_distances": _serialize_pipe(distances),
        "selected_source_weights_raw": _serialize_pipe(raw_weights),
        "selected_source_weights_normalized": _serialize_pipe(normalized_weights),
    }


def _compute_removed_features(
    candidate_features: Sequence[str] | None,
    selected_features: Sequence[str] | None,
) -> List[str]:
    """计算 RFE 从候选池中删除的特征：candidate - selected，保持候选顺序。

    仅用于 CSV 审计字段，不参与训练。
    """
    if not candidate_features:
        return []
    if not selected_features:
        return list(candidate_features)
    selected_set = set(selected_features)
    return [f for f in candidate_features if f not in selected_set]


# ---------------------------------------------------------------------------


def _extract_method_metrics(raw_result: Dict[str, Any], method_name: str) -> Dict[str, Any]:
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
        "horizon": int(method_meta["horizon"]) if "horizon" in method_meta else np.nan,
        "learning_rate": float(selected.get("learning_rate", FIXED_LEARNING_RATE)),
        "source_epochs": int(selected.get("source_epochs", FIXED_EPOCHS)),
        "target_epochs": int(selected.get("target_epochs", selected.get("epochs", FIXED_EPOCHS))),
        "epochs": int(selected.get("epochs", selected.get("target_epochs", FIXED_EPOCHS))),
        "clipnorm": selected.get("clipnorm", FIXED_CLIPNORM),
        "dropout": float(selected.get("dropout", FIXED_DROPOUT)),
        "training_time": float(selected.get("training_time", selected.get("training_time_seconds", float("nan")))),
        "rmse": _safe_float(selected.get("rmse"), float("nan")),
        "accuracy": _safe_float(selected.get("accuracy"), float("nan")),
        "mae": _safe_float(selected.get("mae"), float("nan")),
        "mape": _safe_float(selected.get("mape"), float("nan")),
        "smape": _safe_float(selected.get("smape"), float("nan")),
        "rmse_current": _safe_float(selected.get("rmse_current"), float("nan")),
        "accuracy_current": _safe_float(selected.get("accuracy_current"), float("nan")),
        "mae_current": _safe_float(selected.get("mae_current"), float("nan")),
        "mape_current": _safe_float(selected.get("mape_current"), float("nan")),
        "smape_current": _safe_float(selected.get("smape_current"), float("nan")),
        "rmse_paper": _safe_float(selected.get("rmse_paper"), float("nan")),
        "accuracy_paper": _safe_float(selected.get("accuracy_paper"), float("nan")),
        "mae_paper": _safe_float(selected.get("mae_paper"), float("nan")),
        "mape_paper": _safe_float(selected.get("mape_paper"), float("nan")),
        "smape_paper": _safe_float(selected.get("smape_paper"), float("nan")),
        "normalized_rmse": _safe_float(
            selected.get("normalized_rmse", selected.get("rmse")),
            float("nan"),
        ),
        "normalized_accuracy": _safe_float(
            selected.get("normalized_accuracy", selected.get("accuracy")),
            float("nan"),
        ),
        "normalized_mae": _safe_float(
            selected.get("normalized_mae", selected.get("mae")),
            float("nan"),
        ),
        "normalized_mape": selected.get("normalized_mape"),
        "normalized_smape": selected.get("normalized_smape"),
        "original_scale_rmse": selected.get("original_scale_rmse"),
        "original_scale_accuracy": selected.get("original_scale_accuracy"),
        "original_scale_mae": selected.get("original_scale_mae"),
        "original_scale_mape": selected.get("original_scale_mape"),
        "original_scale_smape": selected.get("original_scale_smape"),
        "metric_space": str(selected.get("metric_space", selected.get("metric_space_current", "normalized_minmax_space"))),
        "metric_space_used": str(
            selected.get(
                "metric_space_used",
                selected.get("metric_space", selected.get("metric_space_current", "normalized_minmax_space")),
            )
        ),
        "prediction_shape": _shape_to_tuple(prediction_shape),
        "metric_space_current": str(selected.get("metric_space_current", "normalized_minmax_space")),
        "metric_space_paper": str(selected.get("metric_space_paper", "original_sales_space")),
        "paper_metric_aligned": bool(selected.get("paper_metric_aligned", False)),
        "inverse_transform_applied": bool(selected.get("inverse_transform_applied", False)),
        "inverse_transform_available": bool(selected.get("inverse_transform_available", False)),
        "metric_notes": str(selected.get("metric_notes", "")),
        "val_rmse": _safe_float(selected.get("val_rmse"), float("nan")),
        "val_accuracy": _safe_float(selected.get("val_accuracy"), float("nan")),
        "val_mae": _safe_float(selected.get("val_mae"), float("nan")),
        "val_rmse_current": _safe_float(selected.get("val_rmse_current"), float("nan")),
        "val_accuracy_current": _safe_float(selected.get("val_accuracy_current"), float("nan")),
        "val_mae_current": _safe_float(selected.get("val_mae_current"), float("nan")),
        "val_rmse_paper": _safe_float(selected.get("val_rmse_paper"), float("nan")),
        "val_accuracy_paper": _safe_float(selected.get("val_accuracy_paper"), float("nan")),
        "val_mae_paper": _safe_float(selected.get("val_mae_paper"), float("nan")),
        "val_metric_space": str(selected.get("val_metric_space", "normalized_minmax_space")),
        "val_metric_space_used": str(selected.get("val_metric_space_used", selected.get("val_metric_space", "normalized_minmax_space"))),
        "val_metric_space_current": str(selected.get("val_metric_space_current", "normalized_minmax_space")),
        "val_metric_space_paper": str(selected.get("val_metric_space_paper", "original_sales_space")),
        "val_paper_metric_aligned": bool(selected.get("val_paper_metric_aligned", False)),
        "val_inverse_transform_applied": bool(selected.get("val_inverse_transform_applied", False)),
        "val_inverse_transform_available": bool(selected.get("val_inverse_transform_available", False)),
        "val_metric_notes": str(selected.get("val_metric_notes", "")),
        "meta": method_meta,
    }

    # --- Foolproof fallback: ensure new alias fields mirror old fields ---
    # normalized_* must equal primary rmse/accuracy/mae
    _normalized_rmse = _coalesce(result.get("normalized_rmse"), result.get("rmse"), float("nan"))
    result["normalized_rmse"] = _safe_float(_normalized_rmse, float("nan"))
    _normalized_accuracy = _coalesce(result.get("normalized_accuracy"), result.get("accuracy"), float("nan"))
    result["normalized_accuracy"] = _safe_float(_normalized_accuracy, float("nan"))
    _normalized_mae = _coalesce(result.get("normalized_mae"), result.get("mae"), float("nan"))
    result["normalized_mae"] = _safe_float(_normalized_mae, float("nan"))
    # original_scale_* must equal rmse_paper/accuracy_paper/mae_paper when those exist
    result["original_scale_rmse"] = _coalesce(result.get("original_scale_rmse"), result.get("rmse_paper"))
    result["original_scale_accuracy"] = _coalesce(result.get("original_scale_accuracy"), result.get("accuracy_paper"))
    result["original_scale_mae"] = _coalesce(result.get("original_scale_mae"), result.get("mae_paper"))
    result["original_scale_smape"] = _coalesce_metric(result.get("original_scale_smape"), result.get("smape_paper"))
    result["normalized_smape"] = _coalesce_metric(result.get("normalized_smape"), result.get("smape_current"))
    _smape_value = _coalesce_metric(
        result.get("original_scale_smape"),
        result.get("smape"),
        result.get("normalized_smape"),
    )
    result["smape"] = _safe_float(_smape_value, float("nan"))

    if _smape_value is None:
        existing_notes = str(result.get("metric_notes", "") or "")
        missing_note = "smape_missing_after_metric_extraction"
        result["metric_notes"] = (
            f"{existing_notes}; {missing_note}" if existing_notes else missing_note
        )
        result["metric_missing_reason"] = missing_note

    if _coalesce_metric(result.get("original_scale_smape")) is not None:
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
    learning_rate: float = FIXED_LEARNING_RATE,
    source_epochs: int = FIXED_EPOCHS,
    target_epochs: int = FIXED_EPOCHS,
    batch_size: int = 16,
    metric_protocol: Optional[Dict[str, Any]] = None,
    target_df_for_selection: Optional[pd.DataFrame] = None,
    early_stopping_enabled: bool = True,
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    early_stopping_min_delta: float = DEFAULT_EARLY_STOPPING_MIN_DELTA,
) -> Dict[str, Any]:
    """运行 SS-TL，并返回统一结构。"""
    try:
        from single_source_tl import (
            build_target_model_from_source,
            evaluate_regression_model,
            fine_tune_target_model,
            train_source_model,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"SS-TL dependency missing: {exc}") from exc

    logger = _get_logger()
    logger.info("[run_ss_tl_experiment] Start. hyperparams=%s", fixed_hyperparams_summary())

    cols = _ensure_feature_cols(source_df, target_df, feature_cols)

    # 严格论文口径：SS-TL 使用 KNN 选最近单源（k=1），不再取排序后第一个 source。
    sorted_source = source_df.sort_values(["entity_id", "item_id", "date"]).copy()
    selector = SourceSelector()
    selection_target_df = target_df if target_df_for_selection is None else target_df_for_selection
    selection = selector.select_top_k_sources(
        target_df=selection_target_df,
        source_df=sorted_source,
        feature_cols=cols,
        k=1,
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
    if not isinstance(source_key, tuple) or len(source_key) < 2:
        raise ValueError(f"Invalid SS-TL source_key from selector: {source_key}")

    first_key = [source_key[0], source_key[1]]
    single_source_df = sorted_source[
        (sorted_source["entity_id"] == first_key[0]) & (sorted_source["item_id"] == first_key[1])
    ].copy()

    keep_cols = ["date", "entity_id", "item_id"] + list(cols)
    # Preserve order while removing duplicated labels (e.g., Dataset2 feature_cols includes item_id).
    keep_cols = [c for c in dict.fromkeys(keep_cols) if c in single_source_df.columns]
    single_source_df = single_source_df[keep_cols].copy()

    single_source_df.attrs["split_role"] = "source"
    single_source_df.attrs["split_mode"] = "ratio"
    single_source_df.attrs["split_config"] = {
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
    }

    target_min_df = target_df[[c for c in keep_cols if c in target_df.columns]].copy()
    target_min_df.attrs = target_df.attrs.copy()

    src_train, src_val, src_test = temporal_split_by_ratio_or_dates(single_source_df)
    src_train = fill_source_numeric_na(src_train)
    src_val = fill_source_numeric_na(src_val)
    src_test = fill_source_numeric_na(src_test)
    src_train, src_val, src_test, _, _ = normalize_features(src_train, src_val, src_test)

    X_source, y_source = build_tabular_sequence(src_train, horizon=horizon, window_size=window_size)
    if len(y_source) == 0:
        raise ValueError("SS-TL source windows are empty; adjust window_size/horizon.")
    X_source = to_cnn_tensor(X_source)

    tgt_train, tgt_val, tgt_test = temporal_split_by_ratio_or_dates(target_min_df)
    tgt_train, tgt_val, tgt_test, tgt_scaler, tgt_feature_columns = normalize_features(tgt_train, tgt_val, tgt_test)

    X_target_train, y_target_train = build_tabular_sequence(
        tgt_train, horizon=horizon, window_size=window_size
    )
    X_target_val, y_target_val = build_tabular_sequence(
        tgt_val, horizon=horizon, window_size=window_size
    )
    X_target_test, y_target_test = build_tabular_sequence(
        tgt_test, horizon=horizon, window_size=window_size
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
        early_stopping_enabled=early_stopping_enabled,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
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
        early_stopping_enabled=early_stopping_enabled,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
    )

    ss_raw = evaluate_regression_model(
        model=target_model,
        X_test=X_target_test,
        y_test=y_target_test,
        metric_protocol=metric_protocol,
        sales_scaler=tgt_scaler,
        feature_columns=tgt_feature_columns,
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
        "metric_space_current": str(ss_raw.get("metric_space_current", "normalized_minmax_space")),
        "metric_space_paper": str(ss_raw.get("metric_space_paper", "original_sales_space")),
        "paper_metric_aligned": bool(ss_raw.get("paper_metric_aligned", False)),
        "inverse_transform_applied": bool(ss_raw.get("inverse_transform_applied", False)),
        "inverse_transform_available": bool(ss_raw.get("inverse_transform_available", False)),
        "metric_notes": str(ss_raw.get("metric_notes", "")),
        "meta": {
            "source_key": tuple(first_key),
            "source_selection_policy": "knn_top1",
            "source_distance": float(selected[0].get("distance", 0.0)),
            "source_weight": float(selected[0].get("weight", 1.0)),
            "selected_sources": selected,
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
    learning_rate: float = FIXED_LEARNING_RATE,
    source_epochs: int = FIXED_EPOCHS,
    target_epochs: int = FIXED_EPOCHS,
    batch_size: int = 16,
    metric_protocol: Optional[Dict[str, Any]] = None,
    target_df_for_selection: Optional[pd.DataFrame] = None,
    # 早停参数
    early_stopping_enabled: bool = True,
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    early_stopping_min_delta: float = DEFAULT_EARLY_STOPPING_MIN_DELTA,
    # 自适应源选择参数
    adaptive_source_selection: bool = False,
    min_sources: int = 1,
    max_sources: Optional[int] = None,
    distance_jump_threshold: float = 0.5,
) -> Dict[str, Any]:
    """运行 MSWA-TL，并返回统一结构。"""
    try:
        from mswa_tl import run_mswa_tl
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"MSWA-TL dependency missing: {exc}") from exc

    raw = run_mswa_tl(
        source_df=source_df,
        target_df=target_df,
        target_df_for_selection=target_df_for_selection,
        feature_cols=feature_cols,
        k=k,
        number_of_sources=number_of_sources,
        horizon=horizon,
        window_size=window_size,
        weight_mode=weight_mode,
        include_sales_in_knn=include_sales_in_knn,
        learning_rate=learning_rate,
        source_epochs=source_epochs,
        target_epochs=target_epochs,
        batch_size=batch_size,
        metric_protocol=metric_protocol,
        early_stopping_enabled=early_stopping_enabled,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        adaptive_source_selection=adaptive_source_selection,
        min_sources=min_sources,
        max_sources=max_sources,
        distance_jump_threshold=distance_jump_threshold,
    )
    return _extract_method_metrics(raw, method_name="MSWA-TL")


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
    learning_rate: float = FIXED_LEARNING_RATE,
    source_epochs: int = FIXED_EPOCHS,
    target_epochs: int = FIXED_EPOCHS,
    batch_size: int = 16,
    metric_protocol: Optional[Dict[str, Any]] = None,
    target_df_for_selection: Optional[pd.DataFrame] = None,
    # 早停参数
    early_stopping_enabled: bool = True,
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    early_stopping_min_delta: float = DEFAULT_EARLY_STOPPING_MIN_DELTA,
    # 自适应源选择参数
    adaptive_source_selection: bool = False,
    min_sources: int = 1,
    max_sources: Optional[int] = None,
    distance_jump_threshold: float = 0.5,
) -> Dict[str, Any]:
    """运行 MSSB-TL，并返回统一结构。"""
    try:
        from mssb_tl import run_mssb_tl
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"MSSB-TL dependency missing: {exc}") from exc

    raw = run_mssb_tl(
        source_df=source_df,
        target_df=target_df,
        target_df_for_selection=target_df_for_selection,
        feature_cols=feature_cols,
        k=k,
        number_of_sources=number_of_sources,
        horizon=horizon,
        window_size=window_size,
        weight_mode=weight_mode,
        include_sales_in_knn=include_sales_in_knn,
        learning_rate=learning_rate,
        source_epochs=source_epochs,
        target_epochs=target_epochs,
        batch_size=batch_size,
        metric_protocol=metric_protocol,
        early_stopping_enabled=early_stopping_enabled,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        adaptive_source_selection=adaptive_source_selection,
        min_sources=min_sources,
        max_sources=max_sources,
        distance_jump_threshold=distance_jump_threshold,
    )
    return _extract_method_metrics(raw, method_name="MSSB-TL")


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
    learning_rate: float = FIXED_LEARNING_RATE,
    source_epochs: int = FIXED_EPOCHS,
    target_epochs: int = FIXED_EPOCHS,
    batch_size: int = 16,
    metric_protocol: Optional[Dict[str, Any]] = None,
    target_df_for_selection: Optional[pd.DataFrame] = None,
    # 早停参数
    early_stopping_enabled: bool = True,
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    early_stopping_min_delta: float = DEFAULT_EARLY_STOPPING_MIN_DELTA,
    # 自适应源选择参数
    adaptive_source_selection: bool = False,
    min_sources: int = 1,
    max_sources: Optional[int] = None,
    distance_jump_threshold: float = 0.5,
) -> Dict[str, Any]:
    """运行 MSML-TL，并返回统一结构。"""
    try:
        from msml_tl import run_msml_tl
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"MSML-TL dependency missing: {exc}") from exc

    raw = run_msml_tl(
        source_df=source_df,
        target_df=target_df,
        target_df_for_selection=target_df_for_selection,
        feature_cols=feature_cols,
        k=k,
        number_of_sources=number_of_sources,
        horizon=horizon,
        window_size=window_size,
        weight_mode=weight_mode,
        include_sales_in_knn=include_sales_in_knn,
        learning_rate=learning_rate,
        source_epochs=source_epochs,
        target_epochs=target_epochs,
        batch_size=batch_size,
        metric_protocol=metric_protocol,
        early_stopping_enabled=early_stopping_enabled,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        adaptive_source_selection=adaptive_source_selection,
        min_sources=min_sources,
        max_sources=max_sources,
        distance_jump_threshold=distance_jump_threshold,
    )
    return _extract_method_metrics(raw, method_name="MSML-TL")


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
    learning_rate: float = FIXED_LEARNING_RATE,
    source_epochs: int = FIXED_EPOCHS,
    target_epochs: int = FIXED_EPOCHS,
    batch_size: int = 16,
    random_state: int = 42,
    metric_protocol: Optional[Dict[str, Any]] = None,
    source_selection_window: Optional[str] = None,
    full_target_df: Optional[pd.DataFrame] = None,
    # 早停参数
    early_stopping_enabled: bool = True,
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    early_stopping_min_delta: float = DEFAULT_EARLY_STOPPING_MIN_DELTA,
    # 自适应源选择参数
    adaptive_source_selection: bool = False,
    min_sources: int = 1,
    max_sources: Optional[int] = None,
    distance_jump_threshold: float = 0.5,
) -> Dict[str, Any]:
    """运行 MSML-TL-RFE，并返回统一结构。"""
    try:
        from msml_tl_rfe import run_msml_tl_rfe
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"MSML-TL-RFE dependency missing: {exc}") from exc

    raw = run_msml_tl_rfe(
        source_df=source_df,
        target_df=target_df,
        feature_cols=feature_cols,
        k=k,
        number_of_sources=number_of_sources,
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
        random_state=random_state,
        metric_protocol=metric_protocol,
        source_selection_window=source_selection_window or "target_observed_window",
        full_target_df=full_target_df,
        early_stopping_enabled=early_stopping_enabled,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        adaptive_source_selection=adaptive_source_selection,
        min_sources=min_sources,
        max_sources=max_sources,
        distance_jump_threshold=distance_jump_threshold,
    )
    return _extract_method_metrics(raw, method_name="MSML-TL-RFE")


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
    learning_rate: float = FIXED_LEARNING_RATE,
    source_epochs: int = FIXED_EPOCHS,
    target_epochs: int = FIXED_EPOCHS,
    batch_size: int = 16,
    random_state: int = 42,
    enabled_methods: Optional[Iterable[str]] = None,
    verbose_mode: str = "summary",
    show_method_progress: bool = True,
    strict_paper_mode: Optional[bool] = None,
    early_stopping_enabled: bool = True,
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    early_stopping_min_delta: float = DEFAULT_EARLY_STOPPING_MIN_DELTA,
    adaptive_source_selection: bool = False,
    min_sources: int = 1,
    max_sources: Optional[int] = None,
    distance_jump_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    统一运行全部实验方法，返回统一结构结果。

    失败策略：失败即抛错。
    """
    set_verbose_mode(verbose_mode)
    logger = _get_logger()
    logger.info(
        "[run_all_experiments] Start. dataset=%s include_sales_in_knn=%s hyperparams=%s",
        dataset_name,
        bool(include_sales_in_knn),
        fixed_hyperparams_summary(),
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
    target_df_for_selection = _build_observed_target_window(target_df)

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
                )
            elif method == "SS-TL":
                one = run_ss_tl_experiment(
                    source_df=source_df,
                    target_df=target_df,
                    target_df_for_selection=target_df_for_selection,
                    feature_cols=cols,
                    horizon=horizon,
                    window_size=window_size,
                    learning_rate=learning_rate,
                    source_epochs=source_epochs,
                    target_epochs=target_epochs,
                    batch_size=batch_size,
                    metric_protocol=protocol.get("metric_protocol", {}),
                    early_stopping_enabled=early_stopping_enabled,
                    early_stopping_patience=early_stopping_patience,
                    early_stopping_min_delta=early_stopping_min_delta,
                )
            elif method == "MSWA-TL":
                one = run_mswa_experiment(
                    source_df=source_df,
                    target_df=target_df,
                    target_df_for_selection=target_df_for_selection,
                    feature_cols=cols,
                    k=effective_source_count,
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
                    early_stopping_enabled=early_stopping_enabled,
                    early_stopping_patience=early_stopping_patience,
                    early_stopping_min_delta=early_stopping_min_delta,
                    adaptive_source_selection=adaptive_source_selection,
                    min_sources=min_sources,
                    max_sources=max_sources,
                    distance_jump_threshold=distance_jump_threshold,
                )
            elif method == "MSSB-TL":
                one = run_mssb_experiment(
                    source_df=source_df,
                    target_df=target_df,
                    target_df_for_selection=target_df_for_selection,
                    feature_cols=cols,
                    k=effective_source_count,
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
                    early_stopping_enabled=early_stopping_enabled,
                    early_stopping_patience=early_stopping_patience,
                    early_stopping_min_delta=early_stopping_min_delta,
                    adaptive_source_selection=adaptive_source_selection,
                    min_sources=min_sources,
                    max_sources=max_sources,
                    distance_jump_threshold=distance_jump_threshold,
                )
            elif method == "MSML-TL":
                one = run_msml_experiment(
                    source_df=source_df,
                    target_df=target_df,
                    target_df_for_selection=target_df_for_selection,
                    feature_cols=cols,
                    k=effective_source_count,
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
                    early_stopping_enabled=early_stopping_enabled,
                    early_stopping_patience=early_stopping_patience,
                    early_stopping_min_delta=early_stopping_min_delta,
                    adaptive_source_selection=adaptive_source_selection,
                    min_sources=min_sources,
                    max_sources=max_sources,
                    distance_jump_threshold=distance_jump_threshold,
                )
            elif method == "MSML-TL-RFE":
                one = run_msml_rfe_experiment(
                    source_df=source_df,
                    target_df=target_df,
                    feature_cols=cols,
                    k=effective_source_count,
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
                    random_state=random_state,
                    metric_protocol=protocol.get("metric_protocol", {}),
                    early_stopping_enabled=early_stopping_enabled,
                    early_stopping_patience=early_stopping_patience,
                    early_stopping_min_delta=early_stopping_min_delta,
                    adaptive_source_selection=adaptive_source_selection,
                    min_sources=min_sources,
                    max_sources=max_sources,
                    distance_jump_threshold=distance_jump_threshold,
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

        if _coalesce_metric(one.get("original_scale_smape"), one.get("smape_paper")) is not None:
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
            "random_state": int(random_state),
            "enabled_methods": methods,
            "strict_paper_mode": strict_mode,
            "learning_rate": float(learning_rate),
            "source_epochs": int(source_epochs),
            "target_epochs": int(target_epochs),
            "epochs": int(target_epochs),
            "clipnorm": FIXED_CLIPNORM,
            "dropout": float(FIXED_DROPOUT),
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
    learning_rate = meta.get("learning_rate", FIXED_LEARNING_RATE) if isinstance(meta, dict) else FIXED_LEARNING_RATE
    source_epochs = meta.get("source_epochs", FIXED_EPOCHS) if isinstance(meta, dict) else FIXED_EPOCHS
    target_epochs = meta.get("target_epochs", FIXED_EPOCHS) if isinstance(meta, dict) else FIXED_EPOCHS
    epochs = meta.get("epochs", target_epochs) if isinstance(meta, dict) else FIXED_EPOCHS
    clipnorm = meta.get("clipnorm", FIXED_CLIPNORM) if isinstance(meta, dict) else FIXED_CLIPNORM
    dropout = meta.get("dropout", FIXED_DROPOUT) if isinstance(meta, dict) else FIXED_DROPOUT

    rows: List[Dict[str, Any]] = []
    for one in experiment_results.get("results", []):
        protocol = one.get("protocol", {}) if isinstance(one, dict) else {}
        one_meta = one.get("meta", {}) if isinstance(one, dict) and isinstance(one.get("meta"), dict) else {}
        rfe_info = one_meta.get("rfe_info", {}) if isinstance(one_meta.get("rfe_info"), dict) else {}
        selected_features = (
            one_meta.get("selected_features")
            or one_meta.get("selected_feature_cols")
            or rfe_info.get("final_selected_features")
        )
        rfe_selected_features = (
            one_meta.get("rfe_selected_features")
            or rfe_info.get("rfe_selected_features")
            or rfe_info.get("rfe_selected_feature_cols")
        )
        rfe_candidate_features = (
            one_meta.get("rfe_candidate_features")
            or rfe_info.get("rfe_candidate_features")
            or rfe_info.get("rfe_candidate_cols")
        )
        rfe_removed_features = _compute_removed_features(rfe_candidate_features, rfe_selected_features)
        selected_sources = one_meta.get("selected_sources")
        # --- Attach KNN distance / weight audit fields (CSV-only, no training impact) ---
        knn_audit = _attach_knn_audit_fields(selected_sources)
        random_seed = one_meta.get("random_state", meta.get("random_state", None))
        target_removed_from_rfe = rfe_info.get(
            "target_removed_from_rfe", one_meta.get("target_removed_from_rfe")
        )
        sales_added_back_as_history_input = rfe_info.get(
            "sales_added_back_as_history_input",
            one_meta.get("sales_added_back_as_history_input"),
        )
        duplicate_sales_after = rfe_info.get(
            "duplicate_sales_after", one_meta.get("duplicate_sales_after")
        )
        metric_space = one.get(
            "metric_space_current",
            protocol.get("current_metric_space", "normalized_minmax_space"),
        )
        rows.append(
            {
                "dataset": dataset_name,
                "method": one.get("method", "N/A"),
                "horizon": one.get("horizon", one_meta.get("horizon", np.nan)),
                "learning_rate": float(one.get("learning_rate", learning_rate)),
                "source_epochs": int(one.get("source_epochs", source_epochs)),
                "target_epochs": int(one.get("target_epochs", target_epochs)),
                "epochs": int(one.get("epochs", epochs)),
                "clipnorm": one.get("clipnorm", clipnorm),
                "dropout": float(one.get("dropout", dropout)),
                "training_time": float(one.get("training_time", np.nan)),
                "strict_paper_mode": strict_paper_mode,
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
                "metric_space_used": one.get("metric_space_used", one.get("metric_space", metric_space)),
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
                "inverse_transform_available": bool(one.get("inverse_transform_available", False)),
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
                "selected_features": _serialize_result_field(selected_features),
                "rfe_selected_features": _serialize_result_field(rfe_selected_features),
                "rfe_candidate_features": _serialize_result_field(rfe_candidate_features),
                "rfe_removed_features": _serialize_result_field(rfe_removed_features),
                "selected_sources": _serialize_result_field(selected_sources),
                "selected_source_distances": knn_audit["selected_source_distances"],
                "selected_source_weights_raw": knn_audit["selected_source_weights_raw"],
                "selected_source_weights_normalized": knn_audit["selected_source_weights_normalized"],
                "random_seed": _serialize_result_field(random_seed),
                "target_removed_from_rfe": _serialize_result_field(target_removed_from_rfe),
                "sales_added_back_as_history_input": _serialize_result_field(sales_added_back_as_history_input),
                "duplicate_sales_after": _serialize_result_field(duplicate_sales_after),
                "metric_space": _serialize_result_field(metric_space),
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
            "horizon",
            "learning_rate",
            "source_epochs",
            "target_epochs",
            "epochs",
            "clipnorm",
            "dropout",
            "training_time",
            "strict_paper_mode",
            "experiment_scope",
            "experiment_track",
            "source_protocol_aligned",
            "alignment_status",
            "metric_alignment_status",
            "split_alignment_status",
            "source_pretrained_alignment_status",
            "paper_metric_space",
            "current_metric_space",
            "metric_space_used",
            "metric_space_current",
            "metric_space_paper",
            "paper_metric_aligned",
            "inverse_transform_applied",
            "inverse_transform_available",
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
            "selected_features",
            "rfe_selected_features",
            "rfe_candidate_features",
            "rfe_removed_features",
            "selected_sources",
            "selected_source_distances",
            "selected_source_weights_raw",
            "selected_source_weights_normalized",
            "random_seed",
            "target_removed_from_rfe",
            "sales_added_back_as_history_input",
            "duplicate_sales_after",
            "metric_space",
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
