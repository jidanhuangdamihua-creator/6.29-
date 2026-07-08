from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from src.constants import (
    MIXED_METRIC_PROTOCOL_NOTE,
    MIXED_METRIC_SPACE,
    NO_PAPER_REFERENCE,
    NOT_APPLICABLE,
    RESULT_CONTRACT_VERSION,
    SCHEMA_FAMILY_D4_D6,
    UNKNOWN,
)
from src.experiment.experiment_runner import (
    run_msml_experiment,
    run_msml_rfe_experiment,
    run_mssb_experiment,
    run_mswa_experiment,
    run_no_tl_experiment,
    run_ss_tl_experiment,
)
from src.transfer_methods.source_failure_tolerance import (
    AllSourcesFailedError,
    error_row_from_all_sources_failed,
)
from src.utils.finite_diagnostics import NonFiniteArrayError, validate_feature_frame_finite
from src.utils.result_validation import annotate_silent_metric_failure
from src.utils.source_fillna import fill_source_numeric_na


LOGGER = logging.getLogger("experiment")
MODEL_METADATA_COLUMNS = ("date", "entity_id", "item_id")
DIAGNOSTIC_COLUMNS = (
    "y_pred_nan_count",
    "y_pred_inf_count",
    "y_true_nan_count",
    "y_true_inf_count",
    "X_test_nan_count",
    "X_test_inf_count",
    "model_weight_nan_count",
    "model_weight_inf_count",
)
METRIC_STATUS_COLUMNS = (
    "metric_space",
    "metric_space_current",
    "metric_space_paper",
    "metric_space_used",
    "rmse_metric_space",
    "smape_metric_space",
    "paper_metric_aligned",
    "inverse_transform_applied",
    "inverse_transform_available",
    "metric_notes",
    "metric_protocol_note",
    "metric_protocol_error",
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
)
SOURCE_DOMAIN_DIAGNOSTIC_COLUMNS = (
    "knn_json_domain_filter",
    "source_domain_filter",
    "source_domain_filter_applied",
    "source_domain_filter_reason",
    "source_pool_size_before_filter",
    "source_pool_size_after_filter",
    "source_domain_filter_error",
)


def _scenario_name(config: Dict[str, Any]) -> str:
    return f"{config.get('info_sharing', 'without')}_information_sharing"


def _source_count_for_method(method: str, config: Dict[str, Any]) -> int:
    if method == "No-TL":
        return 0
    if method == "SS-TL":
        return 1
    return int(config.get("source_count", config.get("k", 3)))


def _stable_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _blank_if_missing(row: Dict[str, Any], keys: Sequence[str]) -> None:
    for key in keys:
        if key not in row or row[key] is None:
            row[key] = ""


def _d4_d6_has_no_paper_reference(config: Dict[str, Any]) -> bool:
    return int(config.get("dataset_id", 0)) in {4, 5, 6}


def _target_split_days(config: Dict[str, Any]) -> Dict[str, Any]:
    split = dict(config.get("target_split_config", {}) or config.get("split_config", {}) or {})
    return {
        "train_days": split.get("train_days", config.get("train_days", "")),
        "val_days": split.get("val_days", config.get("val_days", "")),
        "test_days": split.get("test_days", config.get("test_days", "")),
    }


def _metric_protocol_note(raw: Dict[str, Any], config: Dict[str, Any]) -> str:
    if raw.get("metric_protocol_note"):
        return str(raw["metric_protocol_note"])
    rmse_metric_space = str(raw.get("rmse_metric_space", "") or "").strip()
    smape_metric_space = str(raw.get("smape_metric_space", "") or "").strip()
    if raw.get("metric_space_used") == MIXED_METRIC_SPACE or (
        rmse_metric_space and smape_metric_space and rmse_metric_space != smape_metric_space
    ):
        return MIXED_METRIC_PROTOCOL_NOTE
    if raw.get("metric_notes"):
        return str(raw["metric_notes"])
    protocol = dict(config.get("metric_protocol", {}) or {})
    if (
        int(config.get("dataset_id", 0)) in {4, 5, 6}
        and str(protocol.get("current_metric_space", "normalized_minmax_space")) == "normalized_minmax_space"
        and str(protocol.get("paper_metric_space", "original_sales_space")) == "original_sales_space"
        and not bool(raw.get("paper_metric_aligned", False))
        and not bool(raw.get("inverse_transform_applied", False))
    ):
        return "inverse transform not available for solidified parquet path"
    return ""


def _method_runner(method: str):
    return {
        "No-TL": run_no_tl_experiment,
        "SS-TL": run_ss_tl_experiment,
        "MSWA-TL": run_mswa_experiment,
        "MSSB-TL": run_mssb_experiment,
        "MSML-TL": run_msml_experiment,
        "MSML-TL-RFE": run_msml_rfe_experiment,
    }[method]


def _is_numeric_or_bool(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series)


def _resolve_model_feature_cols(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    feature_cols: Sequence[str],
    config: Dict[str, Any],
) -> List[str]:
    """Keep only feature columns that can safely reach normalize_features."""
    requested_cols = list(dict.fromkeys(str(col) for col in feature_cols))
    model_cols: List[str] = []
    excluded: Dict[str, str] = {}

    for col in requested_cols:
        if col in MODEL_METADATA_COLUMNS:
            excluded[col] = "metadata identifier"
            continue
        if col not in source_df.columns or col not in target_df.columns:
            excluded[col] = "missing in source or target"
            continue
        if _is_numeric_or_bool(source_df[col]) and _is_numeric_or_bool(target_df[col]):
            model_cols.append(col)
            continue
        excluded[col] = f"non-numeric dtype source={source_df[col].dtype} target={target_df[col].dtype}"

    if excluded:
        LOGGER.warning(
            "[run_single_entity_experiment] D%s excluded non-model feature columns before normalization: %s",
            config.get("dataset_id", ""),
            excluded,
        )

    if "sales" not in model_cols:
        raise ValueError(
            "sales must remain in model_feature_cols for sequence target construction. "
            f"requested={requested_cols} selected={model_cols} excluded={excluded}"
        )
    return model_cols


def _build_model_dataframe(df: pd.DataFrame, model_feature_cols: Sequence[str]) -> pd.DataFrame:
    """Return a dataframe containing only identifiers plus numeric/bool model features."""
    keep_cols = [col for col in MODEL_METADATA_COLUMNS if col in df.columns]
    for col in model_feature_cols:
        if col in df.columns and col not in keep_cols:
            keep_cols.append(col)

    model_df = df[keep_cols].copy()
    for col in model_feature_cols:
        if col in model_df.columns and pd.api.types.is_bool_dtype(model_df[col]):
            model_df[col] = model_df[col].astype("int64")
    model_df.attrs = df.attrs.copy()
    attrs = model_df.attrs
    if attrs.get("role") == "target" and attrs.get("split_config", {}).get("mode") == "days":
        split_config = attrs["split_config"]
        train_days = int(split_config.get("train_days", 15))
        val_days = int(split_config.get("val_days", 15))
        n = int(model_df["date"].nunique())
        entity_values = model_df["entity_id"].dropna().astype(str).unique().tolist()
        entity = ", ".join(entity_values) if entity_values else "<missing>"
        min_required = train_days + val_days + 1
        if n < min_required:
            raise ValueError(
                f"target day split is too short: entity={entity} n={n} "
                f"train={train_days} val={val_days} min_required={min_required}"
            )
        attrs["split_config"] = {
            "mode": "days",
            "train_days": train_days,
            "val_days": val_days,
            "test_days": n - train_days - val_days,
        }
    model_only_cols = set(model_df.columns) - set(MODEL_METADATA_COLUMNS)
    expected_cols = set(str(col) for col in model_feature_cols)
    assert model_only_cols == expected_cols, (
        f"CNN 输入列与 model_feature_cols 不一致: "
        f"多余列={model_only_cols - expected_cols}, "
        f"缺失列={expected_cols - model_only_cols}"
    )
    return model_df


def _sanitize_source_model_dataframe(
    source_model_df: pd.DataFrame,
    model_feature_cols: Sequence[str],
) -> pd.DataFrame:
    """Return finite source model features without touching target/schema paths."""
    attrs = source_model_df.attrs.copy()
    sanitized = fill_source_numeric_na(source_model_df, feature_columns=model_feature_cols)
    for col in dict.fromkeys(str(col) for col in model_feature_cols):
        if col not in sanitized.columns:
            continue
        if pd.api.types.is_numeric_dtype(sanitized[col]):
            sanitized[col] = (
                pd.to_numeric(sanitized[col], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0)
            )
    sanitized.attrs.update(attrs)
    return sanitized


def _selection_meta(raw: Dict[str, Any], method: str, requested_k: int) -> Dict[str, Any]:
    meta = raw.get("meta", {}) if isinstance(raw.get("meta"), dict) else {}
    selected = meta.get("selected_sources", [])
    if not isinstance(selected, list):
        selected = []
    return {
        "requested_k": int(meta.get("requested_k", requested_k)),
        "effective_k": int(meta.get("effective_k", len(selected) if method != "No-TL" else 0)),
        "selected_source_count": int(meta.get("selected_source_count", len(selected) if method != "No-TL" else 0)),
        "valid_source_count": int(meta.get("valid_source_count", len(selected) if method != "No-TL" else 0)),
        "skipped_source_count": int(meta.get("skipped_source_count", 0)),
        "failed_source_count": int(meta.get("failed_source_count", meta.get("skipped_source_count", 0))),
        "failed_source_keys": meta.get("failed_source_keys", []),
        "skipped_nonfinite_source_count": int(meta.get("skipped_nonfinite_source_count", 0)),
        "failed_sources": meta.get("failed_sources", []),
        "source_failure_messages": meta.get("source_failure_messages", []),
        "date_alignment_mode": str(meta.get("date_alignment_mode", "")),
        "selected_sources": selected,
        "source_key": str(meta.get("source_key", "")),
    }


def _row_from_result(
    raw: Dict[str, Any],
    method: str,
    entity_key: str,
    config: Dict[str, Any],
    elapsed: float,
) -> Dict[str, Any]:
    requested_k = _source_count_for_method(method, config)
    source_meta = _selection_meta(raw, method, requested_k)
    split_days = _target_split_days(config)
    row = {
        "result_contract_version": RESULT_CONTRACT_VERSION,
        "schema_family": SCHEMA_FAMILY_D4_D6,
        "dataset": str(config.get("dataset_name", f"Dataset{config.get('dataset_id', '')}")),
        "dataset_id": int(config.get("dataset_id", 0)),
        "scenario": _scenario_name(config),
        "information_sharing": str(config.get("info_sharing", "without")),
        "target_entity_key": str(entity_key),
        "source_identifier": source_meta["source_key"],
        "method": method,
        "requested_k": source_meta["requested_k"],
        "effective_k": source_meta["effective_k"],
        "selected_source_count": source_meta["selected_source_count"],
        "valid_source_count": source_meta["valid_source_count"],
        "skipped_source_count": source_meta["skipped_source_count"],
        "failed_source_count": source_meta["failed_source_count"],
        "failed_source_keys": source_meta["failed_source_keys"],
        "skipped_nonfinite_source_count": source_meta["skipped_nonfinite_source_count"],
        "failed_sources": _stable_json_dumps(source_meta["failed_sources"]),
        "source_failure_messages": _stable_json_dumps(source_meta["source_failure_messages"]),
        "date_alignment_mode": source_meta["date_alignment_mode"],
        "rmse": float(raw.get("rmse", np.nan)),
        "accuracy": float(raw.get("accuracy", np.nan)),
        "mae": float(raw.get("mae", np.nan)),
        "mape": float(raw.get("mape", np.nan)),
        "smape": float(raw.get("smape", np.nan)),
        "training_time": float(raw.get("training_time", elapsed)),
        "prediction_shape": str(raw.get("prediction_shape", "N/A")),
        "selected_sources": _stable_json_dumps(source_meta["selected_sources"]),
        "error": str(raw.get("error", "")),
    }
    metric_protocol = dict(config.get("metric_protocol", {}) or {})
    row["metric_protocol"] = _stable_json_dumps(metric_protocol) if metric_protocol else ""
    row["metric_space_current"] = str(
        raw.get("metric_space_current", metric_protocol.get("current_metric_space", "normalized_minmax_space"))
    )
    row["metric_space_paper"] = str(
        raw.get("metric_space_paper", metric_protocol.get("paper_metric_space", "original_sales_space"))
    )
    row["metric_space_used"] = str(
        raw.get("metric_space_used", raw.get("metric_space", row["metric_space_current"]))
    )
    row["rmse_metric_space"] = str(raw.get("rmse_metric_space", row["metric_space_used"]))
    row["smape_metric_space"] = str(raw.get("smape_metric_space", row["metric_space_used"]))
    row["paper_metric_aligned"] = (
        NO_PAPER_REFERENCE if _d4_d6_has_no_paper_reference(config) else bool(raw.get("paper_metric_aligned", False))
    )
    row["inverse_transform_applied"] = bool(raw.get("inverse_transform_applied", False))
    row["inverse_transform_available"] = bool(raw.get("inverse_transform_available", False))
    row["metric_protocol_note"] = _metric_protocol_note(raw, config)
    row["metric_protocol_error"] = str(raw.get("metric_protocol_error", ""))
    row["alignment_status"] = str(raw.get("alignment_status", NO_PAPER_REFERENCE))
    row["metric_alignment_status"] = str(raw.get("metric_alignment_status", NO_PAPER_REFERENCE))
    row["split_alignment_status"] = str(raw.get("split_alignment_status", NO_PAPER_REFERENCE))
    row["source_protocol_aligned"] = str(raw.get("source_protocol_aligned", NO_PAPER_REFERENCE))
    row["target_split_mode"] = str(raw.get("target_split_mode", config.get("target_split_mode", "days")))
    row["source_split_mode"] = str(raw.get("source_split_mode", config.get("source_split_mode", "ratio")))
    row["target_window_days"] = raw.get(
        "target_window_days",
        config.get(
            "target_window_days",
            int(split_days["train_days"] or 0) + int(split_days["val_days"] or 0) + int(split_days["test_days"] or 0)
            if all(str(split_days[key]).strip() for key in ("train_days", "val_days", "test_days"))
            else "",
        ),
    )
    row.update(split_days)
    row["requested_source_count"] = source_meta["requested_k"]
    row["actual_pretrained_model_count"] = source_meta["valid_source_count"]
    row["pretrained_model_count"] = source_meta["valid_source_count"]
    for key in ("target_entity_id", "target_store_id", "target_item_id"):
        if key in raw:
            row[key] = raw[key]
        elif key in config:
            row[key] = config[key]
    for key in DIAGNOSTIC_COLUMNS:
        if key in raw:
            row[key] = raw[key]
    for key in METRIC_STATUS_COLUMNS:
        if key in raw:
            row[key] = raw[key]
    if _d4_d6_has_no_paper_reference(config):
        row["paper_metric_aligned"] = NO_PAPER_REFERENCE
        for key in (
            "rmse_paper",
            "accuracy_paper",
            "mae_paper",
            "mape_paper",
            "smape_paper",
        ):
            row[key] = ""
        for key in (
            "original_scale_rmse",
            "original_scale_accuracy",
            "original_scale_mae",
            "original_scale_mape",
            "original_scale_smape",
        ):
            if key not in raw:
                row[key] = ""
    _blank_if_missing(
        row,
        (
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
            "rmse_paper",
            "accuracy_paper",
            "mae_paper",
            "mape_paper",
            "smape_paper",
        ),
    )
    if "metric_protocol_note" not in row or not row["metric_protocol_note"]:
        row["metric_protocol_note"] = _metric_protocol_note(raw, config)
    for key in SOURCE_DOMAIN_DIAGNOSTIC_COLUMNS:
        if key in raw:
            row[key] = raw[key]
        elif key in config:
            value = config[key]
            if key in {"knn_json_domain_filter", "source_domain_filter"}:
                row[key] = "" if value is None else _stable_json_dumps(value)
            else:
                row[key] = value
    for key in (
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
        if key in raw:
            row[key] = raw[key]
        elif key in config:
            row[key] = config[key]
        elif key in {"source_numeric_na_repaired", "repaired_columns"}:
            row[key] = NOT_APPLICABLE
        elif key in {"source_selection_feature_cols", "model_feature_cols"}:
            row[key] = UNKNOWN
    if "source_domain_filter_name" not in row:
        row["source_domain_filter_name"] = str(config.get("source_domain_filter_name", ""))
    return annotate_silent_metric_failure(row)


def _row_from_nonfinite_error(
    exc: NonFiniteArrayError,
    method: str,
    entity_key: str,
    config: Dict[str, Any],
    elapsed: float,
) -> Dict[str, Any]:
    diagnostics = dict(exc.diagnostics)
    prediction_shape = diagnostics.get("y_pred_shape", diagnostics.get("prediction_shape", "N/A"))
    raw = {
        "rmse": np.nan,
        "accuracy": np.nan,
        "mae": np.nan,
        "mape": np.nan,
        "smape": np.nan,
        "training_time": elapsed,
        "prediction_shape": prediction_shape,
        "error": f"non_finite_prediction: {exc}",
        **diagnostics,
    }
    return _row_from_result(raw, method, entity_key, config, elapsed)


def run_single_entity_experiment(
    entity_key: str,
    source_df: pd.DataFrame,
    target_entity_df: pd.DataFrame,
    feature_cols: Sequence[str],
    config: Dict[str, Any],
    enabled_methods: Sequence[str],
) -> List[Dict[str, Any]]:
    """Run configured methods for one target entity and return CSV-ready rows."""
    rows: List[Dict[str, Any]] = []
    scenario = _scenario_name(config)
    source_df = source_df.copy()
    target_entity_df = target_entity_df.copy()
    source_df.attrs["information_sharing_scenario"] = scenario
    target_entity_df.attrs["information_sharing_scenario"] = scenario
    metric_protocol = dict(config.get("metric_protocol", {}) or {})
    model_feature_cols = _resolve_model_feature_cols(source_df, target_entity_df, feature_cols, config)
    source_model_df = _build_model_dataframe(source_df, model_feature_cols)
    source_model_df = _sanitize_source_model_dataframe(source_model_df, model_feature_cols)
    validate_feature_frame_finite(
        source_model_df,
        model_feature_cols,
        context="post_build_model_dataframe_source",
        dataset_id=config.get("dataset_id"),
        role="source",
        entity_id="source_pool",
        stage="post_build_model_dataframe",
    )
    target_model_df = _build_model_dataframe(target_entity_df, model_feature_cols)
    config = dict(config)
    config["target_split_config"] = dict(target_model_df.attrs.get("split_config", {}) or {})
    validate_feature_frame_finite(
        target_model_df,
        model_feature_cols,
        context="post_build_model_dataframe_target",
        dataset_id=config.get("dataset_id"),
        role="target",
        entity_id=str(entity_key),
        stage="post_build_model_dataframe",
    )
    source_model_df.attrs["information_sharing_scenario"] = scenario
    target_model_df.attrs["information_sharing_scenario"] = scenario

    for method in enabled_methods:
        if method not in {"No-TL", "SS-TL", "MSWA-TL", "MSSB-TL", "MSML-TL", "MSML-TL-RFE"}:
            raise ValueError(f"Unsupported method: {method}")
        source_model_df.attrs["method"] = method
        target_model_df.attrs["method"] = method
        t0 = time.perf_counter()
        runner = _method_runner(method)
        if method == "No-TL":
            raw = runner(
                target_df=target_model_df,
                horizon=int(config["horizon"]),
                window_size=int(config["window_size"]),
                learning_rate=float(config["learning_rate"]),
                target_epochs=int(config["target_epochs"]),
                batch_size=int(config["batch_size"]),
                metric_protocol=metric_protocol,
                feature_cols=list(model_feature_cols),
            )
        else:
            kwargs = {
                "source_df": source_model_df,
                "target_df": target_model_df,
                "feature_cols": list(model_feature_cols),
                "horizon": int(config["horizon"]),
                "window_size": int(config["window_size"]),
                "learning_rate": float(config["learning_rate"]),
                "source_epochs": int(config["source_epochs"]),
                "target_epochs": int(config["target_epochs"]),
                "batch_size": int(config["batch_size"]),
                "metric_protocol": metric_protocol,
            }
            if method not in {"SS-TL"}:
                kwargs["number_of_sources"] = _source_count_for_method(method, config)
            try:
                raw = runner(**kwargs)
            except AllSourcesFailedError as exc:
                raw = error_row_from_all_sources_failed(
                    exc,
                    requested_k=_source_count_for_method(method, config),
                    elapsed=time.perf_counter() - t0,
                )
                rows.append(_row_from_result(raw, method, entity_key, config, time.perf_counter() - t0))
                continue
            except NonFiniteArrayError as exc:
                rows.append(
                    _row_from_nonfinite_error(
                        exc,
                        method,
                        entity_key,
                        config,
                        time.perf_counter() - t0,
                    )
                )
                continue
        rows.append(_row_from_result(raw, method, entity_key, config, time.perf_counter() - t0))
    return rows
