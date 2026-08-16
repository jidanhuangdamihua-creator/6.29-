"""
数据预处理与滑窗构建模块

本模块只负责数据处理流程，不包含任何模型与训练逻辑。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from dataset_registry import get_dataset_profile, normalize_dataset_name
from src.protocols.experiment_protocol import ProtocolViolation
from src.protocols.transformation_identity import (
    NORMALIZATION_EVIDENCE_ATTR,
    SEQUENCE_EVIDENCE_ATTR,
    build_normalization_evidence,
    build_normalization_identity,
    build_sequence_evidence,
    build_sequence_identity,
    make_sample_boundary,
    transformation_identity_requested,
)

if TYPE_CHECKING:
    from src.protocols.transformation_reuse import TargetTransformationReuseContext
from src.utils.finite_diagnostics import validate_feature_frame_finite, validate_finite_array
from src.utils.dataframe_attrs import (
    context_with,
    copy_frame_with_lightweight_attrs,
    get_protocol_frame_context,
    lightweight_frame_attrs,
    set_protocol_frame_context,
)

try:
    from src.utils.environment import setup_logging
except ImportError:
    setup_logging = None


LOGGER_NAME = "experiment"
_NON_FEATURE_COLUMNS = {"date", "entity_id", "item_id", "qty_key", "promo_key", "region_id"}
_SOURCE_SELECTION_EXCLUDE_EXACT = {"entity_id", "item_id", "date"}
_SOURCE_SELECTION_LEAKAGE_KEYWORDS = (
    "future",
    "label",
    "target",
    "leak",
    "pred",
    "prediction",
    "y_true",
    "y_pred",
    "next",
)
KNN_FEATURE_MODE_NO_IDS_V2 = "paper_available_features_no_ids_v2"

DEFAULT_PAPER_SPLIT_PROTOCOL = {
    "target_observed_window_days": 30,
    "target_forecast_window_days": 180,
    "validation_strategy": "time_holdout",
    "rolling_or_fixed_split": "rolling_recent_days",
    "source_selection_window": "full_history",
    "source_pool_scope": "all_source_items",
}

STRICT_DATASET_PROTOCOL = {
    "Dataset1": {
        "target_entity_id": 1,
        "target_item_id": 10,
        "allowed_entities": [1, 2, 3],
        "source_item_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "target_split_days": {"train_days": 15, "val_days": 15, "test_days": 180},
        "source_split_ratio": {"train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1},
        "without_information_sharing_scope": "same_store",
    },
    "Dataset2": {
        "target_entity_id": "B1",
        "target_item_id": 10,
        "source_item_policy": "all_except_target_item",
        "target_split_days": {"train_days": 15, "val_days": 15, "test_days": 180},
        "source_split_ratio": {"train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1},
        "without_information_sharing_scope": "same_brand",
    },
    "Dataset3": {
        "target_store_id": 10,
        "target_split_days": {"train_days": 15, "val_days": 15, "test_days": 180},
        "source_split_ratio": {"train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1},
        "without_information_sharing_scope": "same_region",
        "region_field": "region_id",
    },
}


def _get_logger() -> logging.Logger:
    """获取项目统一日志器；若未初始化则按默认参数初始化。"""
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers and setup_logging is not None:
        setup_logging(log_level="INFO", log_file=None)
        logger = logging.getLogger(LOGGER_NAME)
    return logger


def _log_raw_schema(logger: logging.Logger, dataset_name: str, raw_df: pd.DataFrame) -> List[str]:
    """记录输入列名、dtype，并返回 object/string 列。"""
    logger.info("[load_dataset] Raw columns for %s: %s", dataset_name, list(raw_df.columns))
    dtypes_map = {col: str(dtype) for col, dtype in raw_df.dtypes.items()}
    logger.info("[load_dataset] Raw dtypes for %s: %s", dataset_name, dtypes_map)

    object_cols = [
        col
        for col in raw_df.columns
        if pd.api.types.is_object_dtype(raw_df[col]) or pd.api.types.is_string_dtype(raw_df[col])
    ]
    if object_cols:
        logger.warning("Detected mixed/object dtype columns: %s", object_cols)
    return object_cols


def _coerce_numeric_column(df: pd.DataFrame, col: str, logger: logging.Logger, dataset_name: str) -> None:
    """将列强制清洗为数值并记录清洗前后信息。"""
    before_dtype = str(df[col].dtype)
    raw_series = df[col]

    cleaned = raw_series.astype("string").str.strip()
    cleaned = cleaned.replace(r"^\s*$", pd.NA, regex=True)
    cleaned = cleaned.replace(r"(?i)^(NA|N/A|-|NULL|NONE|NAN)$", pd.NA, regex=True)
    cleaned = cleaned.str.replace(",", "", regex=False)
    cleaned = cleaned.str.replace(r"[^\d\.+\-eE]", "", regex=True)
    cleaned = cleaned.replace(r"^\s*$", pd.NA, regex=True)

    numeric = pd.to_numeric(cleaned, errors="coerce")
    coercion_nan_count = int((raw_series.notna() & numeric.isna()).sum())
    df[col] = numeric

    logger.info(
        "[load_dataset] %s column cleaned: column=%s before_dtype=%s after_dtype=%s coercion_nan_count=%d",
        dataset_name,
        col,
        before_dtype,
        str(df[col].dtype),
        coercion_nan_count,
    )


def _clean_rossmann_raw_columns(raw_df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """最小清洗 Rossmann 原始列，避免 object/mixed 污染后续流程。"""
    cleaned_df = raw_df.copy()
    object_cols = _log_raw_schema(logger, "Dataset3", cleaned_df)
    if not object_cols:
        return cleaned_df

    # Rossmann 数据中的这些字段应为数值（含可转为数值的假日编码），统一按数值清洗。
    expected_numeric_cols = {
        "Store",
        "DayOfWeek",
        "Sales",
        "Customers",
        "Open",
        "Promo",
        "StateHoliday",
        "SchoolHoliday",
    }

    for col in object_cols:
        if col in expected_numeric_cols:
            _coerce_numeric_column(cleaned_df, col, logger, dataset_name="Dataset3")
        else:
            before_dtype = str(cleaned_df[col].dtype)
            cleaned_df[col] = cleaned_df[col].astype("string")
            logger.info(
                "[load_dataset] Dataset3 column cleaned: column=%s before_dtype=%s after_dtype=%s coercion_nan_count=%d",
                col,
                before_dtype,
                str(cleaned_df[col].dtype),
                0,
            )

    return cleaned_df


def _ensure_base_columns(df: pd.DataFrame) -> pd.DataFrame:
    """校验标准字段是否齐全。"""
    required = {"date", "entity_id", "item_id", "sales"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df


def _safe_int(value: Any, default: int) -> int:
    """将值安全转换为整数。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _get_cfg(config: Any, key: str, default: Any = None) -> Any:
    """统一读取 dict 或 Config 对象中的配置值。"""
    if config is None:
        return default

    if hasattr(config, "get"):
        try:
            value = config.get(key, default)
            return default if value is None else value
        except TypeError:
            pass

    current = config
    for part in key.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        else:
            current = getattr(current, part, None)
        if current is None:
            return default
    return current


def _is_strict_paper_mode(config: Any) -> bool:
    """Resolve strict paper mode from config with backward-compatible aliases."""
    strict_value = _get_cfg(config, "paper_reproduction.strict_paper_mode", None)
    if strict_value is None:
        strict_value = _get_cfg(config, "paper_reproduction.paper_strict_mode", False)
    return bool(strict_value)


def _resolve_paper_split_protocol(config: Any) -> Dict[str, Any]:
    """Resolve unified paper split protocol with backward-compatible fallbacks."""
    direct_protocol = _get_cfg(config, "paper_reproduction.paper_split_protocol", None)
    if isinstance(direct_protocol, dict) and direct_protocol:
        protocol = dict(DEFAULT_PAPER_SPLIT_PROTOCOL)
        protocol.update(direct_protocol)
        return protocol

    # Backward-compatible mapping from legacy split_protocol structure.
    observed_days = _safe_int(
        _get_cfg(config, "paper_reproduction.split_protocol.target_window.train_val_days", 30),
        30,
    )
    forecast_days = _safe_int(
        _get_cfg(config, "paper_reproduction.split_protocol.target_window.test_days", 180),
        180,
    )
    validation_strategy = _get_cfg(
        config,
        "paper_reproduction.split_protocol.target_eval_split.mode",
        DEFAULT_PAPER_SPLIT_PROTOCOL["validation_strategy"],
    )
    rolling_or_fixed = _get_cfg(
        config,
        "paper_reproduction.split_protocol.target_window.kind",
        DEFAULT_PAPER_SPLIT_PROTOCOL["rolling_or_fixed_split"],
    )
    return {
        "target_observed_window_days": observed_days,
        "target_forecast_window_days": forecast_days,
        "validation_strategy": str(validation_strategy),
        "rolling_or_fixed_split": str(rolling_or_fixed),
        "source_selection_window": DEFAULT_PAPER_SPLIT_PROTOCOL["source_selection_window"],
        "source_pool_scope": DEFAULT_PAPER_SPLIT_PROTOCOL["source_pool_scope"],
    }


def _resolve_strict_dataset_protocol(config: Any, dataset_name: str) -> Dict[str, Any]:
    cfg_protocol = _get_cfg(config, "paper_reproduction.strict_dataset_protocol", {})
    if not isinstance(cfg_protocol, dict):
        cfg_protocol = {}
    defaults = dict(STRICT_DATASET_PROTOCOL.get(dataset_name, {}))
    override = cfg_protocol.get(dataset_name, {})
    if isinstance(override, dict):
        defaults.update(override)
    return defaults


def _is_strict_paper_split(config: Any) -> bool:
    """Resolve strict split mode. strict_paper_mode implies strict_paper_split."""
    explicit_strict_split = _get_cfg(config, "paper_reproduction.strict_paper_split", None)
    if explicit_strict_split is None:
        explicit_strict_split = _get_cfg(config, "paper_reproduction.paper_strict_split", None)
    if explicit_strict_split is None:
        explicit_strict_split = False
    return bool(explicit_strict_split) or _is_strict_paper_mode(config)


def _standardize_dataset1(df: pd.DataFrame) -> pd.DataFrame:
    """标准化 Dataset1（需求预测挑战赛）字段。"""
    col_map = {
        "date": "date",
        "store": "entity_id",
        "item": "item_id",
        "sales": "sales",
    }
    renamed = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    return renamed


def _standardize_rossmann_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """标准化 Dataset3（Rossmann 门店）字段。"""
    col_map = {
        "Date": "date",
        "Store": "item_id",
        "Store": "store_id",
        "DayOfWeek": "day_of_week",
        "Sales": "sales",
        "Promo": "promo",
        "Open": "open",
        "Customers": "customers",
        "StateHoliday": "state_holiday",
        "SchoolHoliday": "school_holiday",
    }
    renamed = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    if "store_id" not in renamed.columns and "item_id" in renamed.columns:
        renamed["store_id"] = renamed["item_id"]
    if "item_id" not in renamed.columns and "store_id" in renamed.columns:
        renamed["item_id"] = renamed["store_id"]

    # Dataset3 是 store-level 任务，region 字段在原始 CSV 不可见时只能保持 TODO 占位。
    if "region_id" not in renamed.columns:
        renamed["region_id"] = "TODO_REGION_UNAVAILABLE"

    renamed["entity_id"] = renamed["region_id"]

    if "state_holiday" in renamed.columns:
        renamed["state_holiday"] = pd.to_numeric(renamed["state_holiday"], errors="coerce").fillna(0)
    if "school_holiday" in renamed.columns:
        renamed["school_holiday"] = pd.to_numeric(renamed["school_holiday"], errors="coerce").fillna(0)

    if "state_holiday" in renamed.columns or "school_holiday" in renamed.columns:
        renamed["holiday"] = (
            pd.to_numeric(renamed.get("state_holiday", 0), errors="coerce").fillna(0)
            + pd.to_numeric(renamed.get("school_holiday", 0), errors="coerce").fillna(0)
        ).clip(upper=1)
    return renamed


def _standardize_pasta_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """将 Dataset2（意大利面需求）宽表转换为长表并标准化字段。"""
    date_col = "DATE" if "DATE" in df.columns else "date"
    qty_cols = [c for c in df.columns if c.upper().startswith("QTY_")]
    promo_cols = [c for c in df.columns if c.upper().startswith("PROMO_")]

    if not qty_cols:
        raise ValueError("Dataset2 requires QTY_* columns")

    qty_long = df[[date_col] + qty_cols].melt(
        id_vars=[date_col], var_name="qty_key", value_name="sales"
    )
    qty_meta = qty_long["qty_key"].str.extract(r"QTY_(B\d+)_(\d+)")
    qty_long["entity_id"] = qty_meta[0]
    qty_long["item_id"] = pd.to_numeric(qty_meta[1], errors="coerce")

    if promo_cols:
        promo_long = df[[date_col] + promo_cols].melt(
            id_vars=[date_col], var_name="promo_key", value_name="promo"
        )
        promo_meta = promo_long["promo_key"].str.extract(r"PROMO_(B\d+)_(\d+)")
        promo_long["entity_id"] = promo_meta[0]
        promo_long["item_id"] = pd.to_numeric(promo_meta[1], errors="coerce")
        merged = qty_long.merge(
            promo_long[[date_col, "entity_id", "item_id", "promo"]],
            on=[date_col, "entity_id", "item_id"],
            how="left",
        )
    else:
        merged = qty_long

    merged = merged.rename(columns={date_col: "date"})
    merged = merged.drop(columns=[col for col in ["qty_key", "promo_key"] if col in merged.columns])
    return merged


def load_dataset(dataset_name: str, data_path: str) -> pd.DataFrame:
    """
    读取并标准化数据集。

    Args:
        dataset_name: 数据集名称，支持 Dataset1 / Dataset2 / Dataset3 及兼容别名
        data_path: CSV 文件路径

    Returns:
        标准字段DataFrame，至少包含 date/entity_id/item_id/sales
    """
    logger = _get_logger()
    canonical_name = normalize_dataset_name(dataset_name)
    dataset_profile = get_dataset_profile(canonical_name)
    logger.info("[load_dataset] Start loading dataset=%s from %s", canonical_name, data_path)

    raw_df = pd.read_csv(data_path, low_memory=False)

    if dataset_profile != "rossmann":
        _log_raw_schema(logger, canonical_name, raw_df)

    if dataset_profile == "challenge":
        df = _standardize_dataset1(raw_df)
    elif dataset_profile == "pasta":
        df = _standardize_pasta_dataset(raw_df)
    elif dataset_profile == "rossmann":
        raw_df = _clean_rossmann_raw_columns(raw_df, logger)
        df = _standardize_rossmann_dataset(raw_df)
    else:
        raise ValueError("Unsupported dataset_name. Use Dataset1, Dataset2, or Dataset3.")

    df = _ensure_base_columns(df).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    before_drop = len(df)
    df = df.dropna().sort_values(["date", "entity_id", "item_id"]).reset_index(drop=True)
    logger.info(
        "[load_dataset] Finished. rows_before_dropna=%d rows_after_dropna=%d columns=%s",
        before_drop,
        len(df),
        list(df.columns),
    )
    return df


def extract_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    提取时间字段 year/month/week/day。

    Args:
        df: 含 date 列的DataFrame

    Returns:
        增强后的DataFrame
    """
    logger = _get_logger()
    logger.info("[extract_datetime_features] Start.")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["week"] = out["date"].dt.isocalendar().week.astype(int)
    out["day"] = out["date"].dt.day

    # 安全编码可共享的静态元数据，不改变既有缺失值处理路径。
    if "entity_id" in out.columns and (
        pd.api.types.is_object_dtype(out["entity_id"]) or pd.api.types.is_string_dtype(out["entity_id"])
    ):
        entity_codes, _ = pd.factorize(out["entity_id"].astype("string"), sort=True)
        out["entity_id_code"] = pd.Series(entity_codes, index=out.index, dtype="int64")
        out["brand_code"] = out["entity_id_code"].astype("int64")

    if "region_id" in out.columns:
        region_codes, _ = pd.factorize(out["region_id"].astype("string"), sort=True)
        out["region_code"] = pd.Series(region_codes, index=out.index, dtype="int64")

    if "promo" in out.columns and ("state_holiday" in out.columns or "school_holiday" in out.columns):
        promo_flag = pd.to_numeric(out["promo"], errors="coerce") > 0
        state_flag = pd.to_numeric(out.get("state_holiday", 0), errors="coerce") > 0
        school_flag = pd.to_numeric(out.get("school_holiday", 0), errors="coerce") > 0
        out["holiday_promo_profile"] = (promo_flag & (state_flag | school_flag)).astype("int64")

    out = out.dropna().sort_values(["date", "entity_id", "item_id"]).reset_index(drop=True)

    logger.info("[extract_datetime_features] Finished. columns=%s", list(out.columns))
    return out


def _infer_source_target_items(df: pd.DataFrame, config: Any) -> Tuple[List[int], List[int]]:
    """根据配置或默认规则推断 source/target 的 item_id 列表。"""
    unique_items = sorted(pd.Series(df["item_id"]).dropna().unique().tolist())
    if not unique_items:
        raise ValueError("No item_id values found.")

    source_items = _get_cfg(config, "preprocessing.source_item_ids", None)
    target_items = _get_cfg(config, "preprocessing.target_item_ids", None)

    if source_items and target_items:
        return [int(x) for x in source_items], [int(x) for x in target_items]

    target_item_from_cfg = _get_cfg(config, "dataset.target_product_id", None)
    if target_item_from_cfg is not None:
        target_item = _safe_int(target_item_from_cfg, default=int(unique_items[-1]))
    elif 10 in unique_items:
        target_item = 10
    else:
        target_item = int(unique_items[-1])

    target = [target_item]

    # 兼容当前 Dataset1（需求预测挑战赛）示例：当 target=10 且 item 1-9 全存在时，默认 source=1..9。
    default_source_block = list(range(1, 10))
    if target_item == 10 and all(item in unique_items for item in default_source_block):
        source = default_source_block
    else:
        source = [int(x) for x in unique_items if int(x) not in target]

    return source, target


def build_source_target_split(df: pd.DataFrame, config: Any) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    构建 Source/Target 子集。

    规则:
    - Source: 使用完整历史
    - Target: 仅保留最近约7个月，其中前约1个月用于 train+val，后约6个月用于 test

    Args:
        df: 标准化后的DataFrame
        config: 配置对象或字典

    Returns:
        (source_df, target_df)
    """
    logger = _get_logger()
    logger.info("[build_source_target_split] Start.")

    sorted_df = df.sort_values(["date", "entity_id", "item_id"]).reset_index(drop=True)
    dataset_name = str(_get_cfg(config, "dataset_name", "")).strip() or str(sorted_df.attrs.get("dataset_name", ""))
    if not dataset_name:
        dataset_name = "Dataset1"

    strict_paper_mode = _is_strict_paper_mode(config)
    strict_paper_split = _is_strict_paper_split(config)

    if strict_paper_mode and dataset_name in STRICT_DATASET_PROTOCOL:
        strict_spec = _resolve_strict_dataset_protocol(config=config, dataset_name=dataset_name)

        if dataset_name == "Dataset1":
            allowed_entities = set(int(v) for v in strict_spec.get("allowed_entities", [1, 2, 3]))
            target_entity_id = int(strict_spec.get("target_entity_id", 1))
            target_item_id = int(strict_spec.get("target_item_id", 10))
            source_item_ids = set(int(v) for v in strict_spec.get("source_item_ids", [1, 2, 3, 4, 5, 6, 7, 8, 9]))

            narrowed = sorted_df[sorted_df["entity_id"].isin(allowed_entities)].copy()
            target_df = narrowed[
                (narrowed["entity_id"].astype(int) == target_entity_id)
                & (narrowed["item_id"].astype(int) == target_item_id)
            ].copy()
            source_df = narrowed[
                narrowed["item_id"].astype(int).isin(source_item_ids)
            ].copy()
        elif dataset_name == "Dataset2":
            target_entity_id = str(strict_spec.get("target_entity_id", "B1"))
            target_item_id = int(strict_spec.get("target_item_id", 10))
            target_df = sorted_df[
                (sorted_df["entity_id"].astype(str) == target_entity_id)
                & (pd.to_numeric(sorted_df["item_id"], errors="coerce") == target_item_id)
            ].copy()
            source_df = sorted_df[
                pd.to_numeric(sorted_df["item_id"], errors="coerce") != target_item_id
            ].copy()
        elif dataset_name == "Dataset3":
            target_store_id = int(strict_spec.get("target_store_id", 10))
            source_df = sorted_df[pd.to_numeric(sorted_df["item_id"], errors="coerce") != target_store_id].copy()
            target_df = sorted_df[pd.to_numeric(sorted_df["item_id"], errors="coerce") == target_store_id].copy()
        else:
            source_items, target_items = _infer_source_target_items(sorted_df, config)
            source_df = sorted_df[sorted_df["item_id"].isin(source_items)].copy()
            target_df = sorted_df[sorted_df["item_id"].isin(target_items)].copy()
    else:
        source_items, target_items = _infer_source_target_items(sorted_df, config)
        source_df = sorted_df[sorted_df["item_id"].isin(source_items)].copy()
        target_df = sorted_df[sorted_df["item_id"].isin(target_items)].copy()

    paper_split_protocol = _resolve_paper_split_protocol(config)
    target_train_val_days = _safe_int(paper_split_protocol.get("target_observed_window_days", 30), 30)
    target_test_days = _safe_int(paper_split_protocol.get("target_forecast_window_days", 180), 180)
    split_days_cfg = {
        "train_days": int(round(target_train_val_days / 2)),
        "val_days": int(target_train_val_days - int(round(target_train_val_days / 2))),
        "test_days": int(target_test_days),
    }
    if strict_paper_mode and dataset_name in STRICT_DATASET_PROTOCOL:
        strict_spec = _resolve_strict_dataset_protocol(config=config, dataset_name=dataset_name)
        strict_split_days = strict_spec.get("target_split_days", {})
        if isinstance(strict_split_days, dict) and strict_split_days:
            split_days_cfg = {
                "train_days": _safe_int(strict_split_days.get("train_days", split_days_cfg["train_days"]), split_days_cfg["train_days"]),
                "val_days": _safe_int(strict_split_days.get("val_days", split_days_cfg["val_days"]), split_days_cfg["val_days"]),
                "test_days": _safe_int(strict_split_days.get("test_days", split_days_cfg["test_days"]), split_days_cfg["test_days"]),
            }

    total_days = int(split_days_cfg["train_days"] + split_days_cfg["val_days"] + split_days_cfg["test_days"])

    # Keep backward-compatible preprocessing fields, but strict mode enforces protocol values.
    preprocessing_train_val_days = _safe_int(
        _get_cfg(config, "preprocessing.target_train_val_days", target_train_val_days),
        target_train_val_days,
    )
    preprocessing_test_days = _safe_int(
        _get_cfg(config, "preprocessing.target_test_days", target_test_days),
        target_test_days,
    )
    if strict_paper_split and (
        preprocessing_train_val_days != target_train_val_days
        or preprocessing_test_days != target_test_days
    ):
        raise ValueError(
            "strict_paper_split requires preprocessing target window fields to match "
            "paper_reproduction.paper_split_protocol. "
            f"preprocessing=({preprocessing_train_val_days},{preprocessing_test_days}) "
            f"paper_split_protocol=({target_train_val_days},{target_test_days})"
        )

    if not target_df.empty:
        available_unique_days = int(target_df["date"].nunique())
        if strict_paper_split and available_unique_days < total_days:
            raise ValueError(
                "strict_paper_split requires full observed+forecast window without fallback. "
                f"required_days={total_days} available_unique_days={available_unique_days}."
            )

        target_max_date = target_df["date"].max()
        target_min_date = target_max_date - pd.Timedelta(days=total_days - 1)
        target_df = target_df[target_df["date"] >= target_min_date].copy()

        target_window_range_days = int((target_df["date"].max() - target_df["date"].min()).days + 1)
        target_unique_days = int(target_df["date"].nunique())
        if strict_paper_split and target_window_range_days != total_days:
            raise ValueError(
                "strict_paper_split requires exact target calendar window replication. "
                f"expected_days={total_days} actual_range_days={target_window_range_days}. "
                "TODO: verify whether source data itself misses the paper window boundary dates."
            )
        if strict_paper_split and target_unique_days != total_days:
            raise ValueError(
                "strict_paper_split requires exact daily granularity without missing days. "
                f"expected_unique_days={total_days} actual_unique_days={target_unique_days}."
            )

        target_df.attrs["split_role"] = "target"
        if strict_paper_mode:
            target_df.attrs["split_mode"] = "days"
        else:
            target_df.attrs["split_mode"] = _get_cfg(config, "preprocessing.target_split_mode", "ratio")
        target_df.attrs["split_config"] = {
            "train_ratio": _get_cfg(config, "preprocessing.target_train_ratio", 0.067),
            "val_ratio": _get_cfg(config, "preprocessing.target_val_ratio", 0.067),
            "test_ratio": _get_cfg(config, "preprocessing.target_test_ratio", 0.866),
            "date_boundaries": _get_cfg(config, "preprocessing.target_date_boundaries", {}),
            "train_days": int(split_days_cfg["train_days"]),
            "val_days": int(split_days_cfg["val_days"]),
            "test_days": int(split_days_cfg["test_days"]),
        }
        target_df.attrs["strict_paper_mode"] = strict_paper_mode
        target_df.attrs["strict_paper_split"] = strict_paper_split
        target_df.attrs["paper_split_protocol"] = dict(paper_split_protocol)
        target_df.attrs["strict_dataset_name"] = dataset_name
        target_df.attrs["target_window_expected_days"] = int(total_days)
        target_df.attrs["target_window_range_days"] = int(target_window_range_days)
        target_df.attrs["target_window_unique_days"] = int(target_unique_days)

    source_df.attrs["split_role"] = "source"
    source_df.attrs["split_mode"] = _get_cfg(config, "preprocessing.source_split_mode", "ratio")
    source_df.attrs["split_config"] = {
        "train_ratio": _get_cfg(config, "preprocessing.source_train_ratio", 0.8),
        "val_ratio": _get_cfg(config, "preprocessing.source_val_ratio", 0.1),
        "test_ratio": _get_cfg(config, "preprocessing.source_test_ratio", 0.1),
        "date_boundaries": _get_cfg(config, "preprocessing.source_date_boundaries", {}),
    }
    if strict_paper_mode and dataset_name in STRICT_DATASET_PROTOCOL:
        strict_spec = _resolve_strict_dataset_protocol(config=config, dataset_name=dataset_name)
        source_split_ratio = strict_spec.get("source_split_ratio", {})
        if isinstance(source_split_ratio, dict) and source_split_ratio:
            source_df.attrs["split_config"].update(
                {
                    "train_ratio": float(source_split_ratio.get("train_ratio", source_df.attrs["split_config"]["train_ratio"])),
                    "val_ratio": float(source_split_ratio.get("val_ratio", source_df.attrs["split_config"]["val_ratio"])),
                    "test_ratio": float(source_split_ratio.get("test_ratio", source_df.attrs["split_config"]["test_ratio"])),
                }
            )
    source_df.attrs["strict_paper_mode"] = strict_paper_mode
    source_df.attrs["strict_paper_split"] = strict_paper_split
    source_df.attrs["paper_split_protocol"] = dict(paper_split_protocol)
    source_df.attrs["strict_dataset_name"] = dataset_name

    logger.info(
        "[build_source_target_split] source_items=%s target_items=%s source_rows=%d target_rows=%d strict_paper_mode=%s strict_paper_split=%s target_window_days=%d",
        source_items,
        target_items,
        len(source_df),
        len(target_df),
        strict_paper_mode,
        strict_paper_split,
        total_days,
    )
    return source_df, target_df


def _split_by_ratio(df: pd.DataFrame, train_ratio: float, val_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按时间比例划分数据集。"""
    ordered = df.sort_values(["date", "entity_id", "item_id"]).reset_index(drop=True)
    unique_dates = ordered["date"].drop_duplicates().sort_values().to_numpy()
    n_dates = len(unique_dates)
    if n_dates < 3:
        raise ValueError("At least 3 unique dates are required for temporal split.")

    train_end = max(1, int(n_dates * train_ratio))
    val_end = max(train_end + 1, int(n_dates * (train_ratio + val_ratio)))
    val_end = min(val_end, n_dates - 1)

    train_dates = unique_dates[:train_end]
    val_dates = unique_dates[train_end:val_end]
    test_dates = unique_dates[val_end:]

    train_df = ordered[ordered["date"].isin(train_dates)].copy()
    val_df = ordered[ordered["date"].isin(val_dates)].copy()
    test_df = ordered[ordered["date"].isin(test_dates)].copy()

    return train_df, val_df, test_df


def _split_by_dates(df: pd.DataFrame, date_boundaries: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按明确日期边界划分数据集。"""
    ordered = df.sort_values(["date", "entity_id", "item_id"]).reset_index(drop=True)

    train_end = pd.to_datetime(date_boundaries.get("train_end"), errors="coerce")
    val_end = pd.to_datetime(date_boundaries.get("val_end"), errors="coerce")

    if pd.isna(train_end) or pd.isna(val_end):
        raise ValueError("For date split, split_config.date_boundaries must include train_end and val_end.")

    train_df = ordered[ordered["date"] <= train_end].copy()
    val_df = ordered[(ordered["date"] > train_end) & (ordered["date"] <= val_end)].copy()
    test_df = ordered[ordered["date"] > val_end].copy()
    return train_df, val_df, test_df


def temporal_split_by_ratio_or_dates(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    时间序列切分，支持日期边界模式和比例模式。

    行为说明:
    - 当 df.attrs["split_mode"] == "dates" 时按日期切分
    - 否则按比例切分
    - 若无显式配置，根据 split_role 使用默认比例
      source: 0.8/0.1/0.1
      target: 0.067/0.067/0.866

    Args:
        df: 输入DataFrame

    Returns:
        (train_df, val_df, test_df)
    """
    logger = _get_logger()
    role = str(df.attrs.get("split_role", "source")).lower()
    mode = str(df.attrs.get("split_mode", "ratio")).lower()
    split_config = df.attrs.get("split_config", {}) or {}
    if mode == "paper_split_protocol":
        mode = "days"
        split_config = {"train_days": 15, "val_days": 15, "test_days": 180}
    df = copy_frame_with_lightweight_attrs(df, deep=False)

    logger.info("[temporal_split_by_ratio_or_dates] Start. role=%s mode=%s", role, mode)

    if mode == "dates":
        boundaries = split_config.get("date_boundaries", {}) if isinstance(split_config, dict) else {}
        train_df, val_df, test_df = _split_by_dates(df, boundaries)
    elif mode == "days":
        ordered = df.sort_values(["date", "entity_id", "item_id"]).reset_index(drop=True)
        unique_dates = ordered["date"].drop_duplicates().sort_values().to_numpy()
        train_days = int(split_config.get("train_days", 0))
        val_days = int(split_config.get("val_days", 0))
        test_days = int(split_config.get("test_days", 0))
        total_days = train_days + val_days + test_days
        if total_days <= 0:
            raise ValueError("days split requires positive train_days/val_days/test_days")
        if len(unique_dates) < total_days:
            raise ValueError(
                f"days split requires {total_days} unique days, got {len(unique_dates)}"
            )

        eval_dates = unique_dates[-total_days:]
        train_dates = eval_dates[:train_days]
        val_dates = eval_dates[train_days:train_days + val_days]
        test_dates = eval_dates[train_days + val_days:]

        context = get_protocol_frame_context(df)
        if role == "target" and context is not None and context.sample_manifest is not None:
            observed_end = df.attrs.get("knn_observed_end")
            if observed_end is not None and len(test_dates) > 0:
                context_start = pd.Timestamp(observed_end).normalize() + pd.Timedelta(days=1)
                test_start = pd.Timestamp(test_dates[0]).normalize()
                if context_start < test_start:
                    available_dates = pd.DatetimeIndex(pd.to_datetime(unique_dates)).normalize()
                    leading_context = available_dates[
                        (available_dates >= context_start) & (available_dates < test_start)
                    ].to_numpy()
                    if len(leading_context) > 0:
                        test_dates = np.concatenate((leading_context, test_dates))

        train_df = ordered[ordered["date"].isin(train_dates)].copy()
        val_df = ordered[ordered["date"].isin(val_dates)].copy()
        test_df = ordered[ordered["date"].isin(test_dates)].copy()
    else:
        if role == "target":
            train_ratio = float(split_config.get("train_ratio", 0.067))
            val_ratio = float(split_config.get("val_ratio", 0.067))
        else:
            train_ratio = float(split_config.get("train_ratio", 0.8))
            val_ratio = float(split_config.get("val_ratio", 0.1))
        train_df, val_df, test_df = _split_by_ratio(df, train_ratio, val_ratio)

    inherited_attrs = lightweight_frame_attrs(df.attrs)
    for partition, split_frame in (
        ("train", train_df),
        ("validation", val_df),
        ("test", test_df),
    ):
        split_frame.attrs = dict(inherited_attrs)
        split_frame.attrs["temporal_partition"] = partition

    logger.info(
        "[temporal_split_by_ratio_or_dates] Finished. train=%d val=%d test=%d",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    return train_df, val_df, test_df


def _infer_feature_columns(df: pd.DataFrame) -> List[str]:
    """推断用于归一化与建模的数值特征列。"""
    return infer_modeling_feature_columns(df)


def _is_identifier_like_column(col_name: str) -> bool:
    """Return True for identifier-like columns that must not drive KNN distance."""
    name = str(col_name).strip().lower()
    if not name:
        return False
    if name in {
        "id",
        "store_nbr",
        "item_nbr",
        "store_id",
        "product_id",
        "entity_id",
        "item_id",
    }:
        return True
    return name.endswith("_id") or name.endswith("_nbr")


def infer_modeling_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    推断“可用于建模”的数值特征列（与 build_tabular_sequence 口径保持一致）。

    规则：
    - 排除 date
    - 保留其余数值列
    - 若存在 sales，则将其排在首位
    """
    exclude = {"date"}
    num_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    if "sales" in num_cols:
        ordered = ["sales"] + [c for c in num_cols if c != "sales"]
        return ordered
    return num_cols


def infer_source_selection_feature_columns(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    candidate_cols: Optional[List[str]] = None,
    include_sales_in_knn: bool = True,
) -> Dict[str, object]:
    """
    推断 source selection 用特征列，尽量与可建模特征一致并过滤明显泄漏字段。

    返回信息用于日志追踪：
    - selected_features: 最终用于 KNN 的特征
    - modeling_source_features: source 侧可建模特征
    - modeling_target_features: target 侧可建模特征
    - candidate_pool: 候选特征全集
    - missing_in_source: 候选中 source 缺失列
    - missing_in_target: 候选中 target 缺失列
    - excluded_by_rule: 因规则过滤掉的列
    """
    source_modeling = infer_modeling_feature_columns(source_df)
    target_modeling = infer_modeling_feature_columns(target_df)

    source_set = set(source_modeling)
    target_set = set(target_modeling)

    ordered_pool: List[str] = []
    for col in source_modeling + target_modeling:
        if col not in ordered_pool:
            ordered_pool.append(col)
    if candidate_cols:
        for col in candidate_cols:
            if col not in ordered_pool:
                ordered_pool.append(col)

    missing_in_source = [c for c in ordered_pool if c not in source_set]
    missing_in_target = [c for c in ordered_pool if c not in target_set]

    common_cols = [c for c in ordered_pool if c in source_set and c in target_set]

    def _should_exclude_source_selection_col(col: str) -> bool:
        lower = str(col).lower()
        should_exclude = (
            col in _SOURCE_SELECTION_EXCLUDE_EXACT
            or any(token in lower for token in _SOURCE_SELECTION_LEAKAGE_KEYWORDS)
            or _is_identifier_like_column(col)
        )
        if col == "sales" and not include_sales_in_knn:
            should_exclude = True
        return should_exclude

    excluded_by_rule: List[str] = []
    selected_features: List[str] = []
    for col in common_cols:
        should_exclude = _should_exclude_source_selection_col(col)
        if should_exclude:
            excluded_by_rule.append(col)
            continue
        selected_features.append(col)

    if not selected_features:
        selected_features = [c for c in common_cols if not _should_exclude_source_selection_col(c)]

    if not selected_features:
        raise ValueError(
            "No valid source-selection features after filtering. "
            f"common_cols={common_cols} excluded={excluded_by_rule}"
        )

    return {
        "selected_features": selected_features,
        "modeling_source_features": source_modeling,
        "modeling_target_features": target_modeling,
        "candidate_pool": ordered_pool,
        "missing_in_source": missing_in_source,
        "missing_in_target": missing_in_target,
        "excluded_by_rule": excluded_by_rule,
        "include_sales_in_knn": bool(include_sales_in_knn),
        "knn_feature_mode": KNN_FEATURE_MODE_NO_IDS_V2,
    }


def _assert_no_object_feature_columns(
    df: pd.DataFrame,
    where: str,
    feature_columns: Optional[Sequence[str]] = None,
) -> None:
    """严格校验建模候选列，禁止 object/string 混入。"""
    if feature_columns is None:
        candidate_cols = [c for c in df.columns if c not in _NON_FEATURE_COLUMNS]
    else:
        candidate_cols = [str(c) for c in feature_columns]
        missing = [c for c in candidate_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing explicit feature columns before {where}: {missing}")
    bad_cols = [
        c
        for c in candidate_cols
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])
    ]
    if bad_cols:
        dtype_map = {c: str(df[c].dtype) for c in bad_cols}
        raise ValueError(
            f"Non-numeric feature candidates detected before {where}: {dtype_map}. "
            "Please clean or drop these columns from feature_cols."
        )


def normalize_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: Optional[Sequence[str]] = None,
    validate_finite: bool = True,
    reuse_context: "TargetTransformationReuseContext | None" = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, MinMaxScaler, List[str]]:
    """
    对数值特征进行 MinMax 归一化。

    约束:
    - 只在 train 上 fit
    - val/test 使用同一 scaler transform

    Args:
        train_df: 训练集
        val_df: 验证集
        test_df: 测试集

    Returns:
        (train_scaled, val_scaled, test_scaled, scaler, feature_columns)
    """
    logger = _get_logger()
    logger.info("[normalize_features] Start.")

    explicit_feature_columns = feature_columns is not None
    if explicit_feature_columns:
        resolved_feature_columns = [str(col) for col in feature_columns or []]
    else:
        logger.warning(
            "[normalize_features] 未提供显式特征列表，已回退到自动推断，结果可能与 KNN 选源特征不一致"
        )
        resolved_feature_columns = _infer_feature_columns(train_df)

    _assert_no_object_feature_columns(
        train_df, where="normalize_features(train_df)", feature_columns=resolved_feature_columns
    )
    _assert_no_object_feature_columns(
        val_df, where="normalize_features(val_df)", feature_columns=resolved_feature_columns
    )
    _assert_no_object_feature_columns(
        test_df, where="normalize_features(test_df)", feature_columns=resolved_feature_columns
    )

    if not resolved_feature_columns:
        raise ValueError("No numeric feature columns found for normalization.")
    if "sales" not in resolved_feature_columns:
        raise ValueError("sales column is required in feature_columns for normalization and inverse metrics.")
    if validate_finite:
        validate_feature_frame_finite(
            train_df, resolved_feature_columns, context="pre_normalize_train", stage="pre_normalize_train"
        )
        validate_feature_frame_finite(
            val_df, resolved_feature_columns, context="pre_normalize_val", stage="pre_normalize_val"
        )
        validate_feature_frame_finite(
            test_df, resolved_feature_columns, context="pre_normalize_test", stage="pre_normalize_test"
        )

    identity_requested = tuple(
        transformation_identity_requested(frame) for frame in (train_df, val_df, test_df)
    )
    if any(identity_requested) and not all(identity_requested):
        raise ProtocolViolation("normalization partitions have inconsistent identity lifecycle markers")
    if reuse_context is not None:
        if not all(identity_requested):
            raise ProtocolViolation("normalization reuse requires formal identity-aware partitions")
        lookup_identity = build_normalization_identity(
            train_df,
            val_df,
            test_df,
            feature_cols=resolved_feature_columns,
            scaler=MinMaxScaler(),
            target_column="sales",
        )
        return reuse_context.normalize(
            identity=lookup_identity,
            raw_frames=(train_df, val_df, test_df),
            heavy_builder=lambda: normalize_features(
                train_df,
                val_df,
                test_df,
                feature_columns=resolved_feature_columns,
                validate_finite=validate_finite,
                reuse_context=None,
            ),
        )

    scaler = MinMaxScaler()
    train_scaled = copy_frame_with_lightweight_attrs(train_df)
    val_scaled = copy_frame_with_lightweight_attrs(val_df)
    test_scaled = copy_frame_with_lightweight_attrs(test_df)

    # 先显式转为float，避免将浮点归一化结果写回整型列触发兼容性告警。
    for frame in (train_scaled, val_scaled, test_scaled):
        for col in resolved_feature_columns:
            frame[col] = frame[col].astype(np.float64)

    scaler.fit(train_df[resolved_feature_columns])

    train_values = scaler.transform(train_df[resolved_feature_columns])
    val_values = scaler.transform(val_df[resolved_feature_columns])
    test_values = scaler.transform(test_df[resolved_feature_columns])

    for idx, col in enumerate(resolved_feature_columns):
        train_scaled[col] = train_values[:, idx]
        val_scaled[col] = val_values[:, idx]
        test_scaled[col] = test_values[:, idx]

    for raw_frame, scaled_frame in (
        (train_df, train_scaled),
        (val_df, val_scaled),
        (test_df, test_scaled),
    ):
        raw_context = get_protocol_frame_context(raw_frame)
        actual_source_key = (
            raw_context.actual_source_key
            if raw_context is not None
            else raw_frame.attrs.get("protocol_actual_source_key")
        )
        if actual_source_key is not None:
            scaled_frame.attrs = lightweight_frame_attrs(raw_frame.attrs)
            set_protocol_frame_context(
                scaled_frame,
                context_with(
                    raw_context,
                    actual_source_key=tuple(actual_source_key),
                    raw_partition=raw_frame,
                    fitted_scaler=scaler,
                    scaler_feature_cols=tuple(resolved_feature_columns),
                ),
            )

    if validate_finite:
        validate_feature_frame_finite(
            train_scaled, resolved_feature_columns, context="post_normalize_train", stage="post_normalize_train"
        )
        validate_feature_frame_finite(
            val_scaled, resolved_feature_columns, context="post_normalize_val", stage="post_normalize_val"
        )
        validate_feature_frame_finite(
            test_scaled, resolved_feature_columns, context="post_normalize_test", stage="post_normalize_test"
        )

    if any(identity_requested):
        evidence = build_normalization_evidence(
            train_df,
            val_df,
            test_df,
            train_scaled,
            val_scaled,
            test_scaled,
            feature_cols=resolved_feature_columns,
            scaler=scaler,
            target_column="sales",
        )
        for scaled_frame in (train_scaled, val_scaled, test_scaled):
            scaled_frame.attrs[NORMALIZATION_EVIDENCE_ATTR] = evidence

    logger.info("[normalize_features] Finished. features=%s", resolved_feature_columns)
    return train_scaled, val_scaled, test_scaled, scaler, resolved_feature_columns


def build_tabular_sequence(
    df: pd.DataFrame,
    horizon: int,
    window_size: int = 10,
    feature_columns: Optional[Sequence[str]] = None,
    validate_finite: bool = True,
    reuse_context: "TargetTransformationReuseContext | None" = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    构建滑窗序列数据。

    定义:
    - X: (samples, window_size, num_features)
    - y: (samples,)
    - y_t = t + horizon 的 sales

    Args:
        df: 输入DataFrame
        horizon: 预测步长
        window_size: 历史窗口长度

    Returns:
        (X, y)
    """
    logger = _get_logger()
    logger.info("[build_tabular_sequence] Start. horizon=%d window_size=%d", horizon, window_size)

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if window_size <= 0:
        raise ValueError("window_size must be positive")

    if feature_columns is None:
        resolved_for_identity = _infer_feature_columns(df)
    else:
        resolved_for_identity = [str(col) for col in feature_columns]
    if reuse_context is not None:
        lookup_identity = build_sequence_identity(
            df,
            feature_cols=resolved_for_identity,
            horizon=horizon,
            window_size=window_size,
            target_column="sales",
            group_cols=("entity_id", "item_id"),
            date_col="date",
            x_dtype="float32",
            y_dtype="float32",
        )
        return reuse_context.sequence(
            identity=lookup_identity,
            frame=df,
            heavy_builder=lambda: build_tabular_sequence(
                df,
                horizon=horizon,
                window_size=window_size,
                feature_columns=resolved_for_identity,
                validate_finite=validate_finite,
                reuse_context=None,
            ),
        )

    working_df = copy_frame_with_lightweight_attrs(df, deep=False)
    ordered = working_df.sort_values(
        ["entity_id", "item_id", "date"], kind="mergesort"
    ).reset_index(drop=True)
    if feature_columns is None:
        logger.warning(
            "[build_tabular_sequence] 未提供显式特征列表，已回退到自动推断，结果可能与 KNN 选源特征不一致"
        )
        resolved_feature_columns = _infer_feature_columns(ordered)
    else:
        resolved_feature_columns = [str(col) for col in feature_columns]

    _assert_no_object_feature_columns(
        ordered, where="build_tabular_sequence", feature_columns=resolved_feature_columns
    )
    if "sales" not in resolved_feature_columns:
        raise ValueError("sales column is required to build targets")
    if validate_finite:
        validate_feature_frame_finite(
            ordered, resolved_feature_columns, context="pre_sequence", stage="pre_sequence"
        )

    group_cols = ["entity_id", "item_id"]
    x_list: List[np.ndarray] = []
    y_list: List[float] = []
    sample_evidence = []

    for group_key, group in ordered.groupby(group_cols, sort=False):
        g = group.sort_values("date", kind="mergesort").reset_index(drop=True)
        values = g[resolved_feature_columns].to_numpy(dtype=np.float32)
        sales_values = g["sales"].to_numpy(dtype=np.float32)
        group_dates = pd.to_datetime(g["date"], errors="raise", utc=True)
        normalized_group_key = group_key if isinstance(group_key, tuple) else (group_key,)
        n = len(g)
        max_end = n - horizon

        for end_idx in range(window_size - 1, max_end):
            start_idx = end_idx - window_size + 1
            target_idx = end_idx + horizon
            if target_idx >= n:
                continue
            x_list.append(values[start_idx : end_idx + 1])
            y_list.append(float(sales_values[target_idx]))
            sample_evidence.append(
                make_sample_boundary(
                    group_key=normalized_group_key,
                    window_start=group_dates.iloc[start_idx],
                    window_end=group_dates.iloc[end_idx],
                    label_date=group_dates.iloc[target_idx],
                    horizon=horizon,
                    partition_role=str(df.attrs.get("temporal_partition", "unsplit")),
                )
            )

    if x_list:
        X = np.asarray(x_list, dtype=np.float32)
        y = np.asarray(y_list, dtype=np.float32)
    else:
        X = np.empty((0, window_size, len(resolved_feature_columns)), dtype=np.float32)
        y = np.empty((0,), dtype=np.float32)

    context = get_protocol_frame_context(working_df)
    manifest = context.sample_manifest if context is not None else None
    if manifest is not None and df.attrs.get("temporal_partition") == "test":
        records = tuple(manifest.for_horizon(int(horizon)))
        actual_date_pairs = []
        for _, group in ordered.groupby(group_cols, sort=False):
            g = group.sort_values("date", kind="mergesort").reset_index(drop=True)
            dates = pd.to_datetime(g["date"], errors="raise").dt.strftime("%Y-%m-%d")
            max_end = len(g) - horizon
            for end_idx in range(window_size - 1, max_end):
                start_idx = end_idx - window_size + 1
                target_idx = end_idx + horizon
                if target_idx < len(g):
                    actual_date_pairs.append(
                        (tuple(dates.iloc[start_idx : end_idx + 1]), dates.iloc[target_idx])
                    )
        expected_date_pairs = [
            (tuple(record.input_dates), str(record.label_date)) for record in records
        ]
        if actual_date_pairs != expected_date_pairs:
            raise ValueError(
                "CNN target sequence does not consume the shared protocol sample manifest: "
                f"actual={len(actual_date_pairs)} expected={len(expected_date_pairs)}"
            )

    if context is not None and context.actual_source_key is not None:
        from src.protocols.provenance import validate_actual_cnn_arrays_against_raw

        validate_actual_cnn_arrays_against_raw(
            working_df,
            input_tensor=X,
            labels=y,
            feature_cols=resolved_feature_columns,
            window_size=window_size,
            horizon=horizon,
        )
        validated_context = get_protocol_frame_context(working_df)
        if validated_context is not None:
            set_protocol_frame_context(df, validated_context)

    if validate_finite:
        validate_finite_array(X, name="X")
        validate_finite_array(y, name="y")

    if NORMALIZATION_EVIDENCE_ATTR in df.attrs:
        sequence_evidence = build_sequence_evidence(
            df,
            X,
            y,
            feature_cols=resolved_feature_columns,
            horizon=horizon,
            window_size=window_size,
            samples=sample_evidence,
            target_column="sales",
            group_cols=group_cols,
            date_col="date",
        )
        df.attrs[SEQUENCE_EVIDENCE_ATTR] = sequence_evidence

    logger.info("[build_tabular_sequence] Finished. X_shape=%s y_shape=%s", X.shape, y.shape)
    return X, y


def to_cnn_tensor(X: np.ndarray) -> np.ndarray:
    """
    将输入转换为 CNN 所需 3D 张量。

    Args:
        X: 输入数组

    Returns:
        形状为 (samples, window_size, num_features) 的 float32 数组
    """
    logger = _get_logger()
    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got shape {arr.shape}")
    logger.info("[to_cnn_tensor] Tensor ready. shape=%s", arr.shape)
    return arr
