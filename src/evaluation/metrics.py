"""Basic regression evaluation helpers used by experiment auditing and reporting."""

from __future__ import annotations

import logging

import numpy as np

from src.constants import MIXED_METRIC_PROTOCOL_NOTE, MIXED_METRIC_SPACE
from src.evaluation.metric_contract import (
    ORIGINAL_SALES_SPACE,
    SMAPE_CONTRACT_FIELDS,
    SMAPE_EPSILON,
    SMAPE_RANGE,
    MetricProtocolError,
)


SUPPORTED_METRIC_SPACES = {"normalized_minmax_space", "original_sales_space"}


def compute_original_scale_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-8,
) -> dict:
    """Compute the four primary metrics directly in original sales units."""
    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    predicted = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if true.size == 0 or true.shape != predicted.shape:
        raise ValueError(
            f"original-scale metric arrays must have the same non-empty shape: {true.shape}, {predicted.shape}"
        )
    if not np.isfinite(true).all() or not np.isfinite(predicted).all():
        raise ValueError("original-scale metric arrays must be finite")
    rmse = _compute_rmse(true, predicted)
    return {
        "rmse": rmse,
        "mae": _compute_mae(true, predicted),
        "smape": smape(true, predicted, epsilon=eps),
        "accuracy": float(1.0 / (rmse + eps)),
        "primary_metric_space": "original_sales",
    }


def _compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def _compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def _compute_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100.0)


def smape(y_true, y_pred, epsilon: float = 1e-8) -> float:
    """
    Symmetric Mean Absolute Percentage Error.
    返回百分比形式，范围通常为 0 到 200，越低越好。
    公式：
    sMAPE = mean( 2 * abs(y_pred - y_true) / (abs(y_true) + abs(y_pred) + epsilon) ) * 100
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true_arr.shape[0] != y_pred_arr.shape[0]:
        raise ValueError(
            "y_true and y_pred size mismatch: "
            f"y_true={y_true_arr.shape[0]} y_pred={y_pred_arr.shape[0]}"
        )
    return float(
        100.0
        * np.mean(
            2.0
            * np.abs(y_pred_arr - y_true_arr)
            / (np.abs(y_true_arr) + np.abs(y_pred_arr) + float(epsilon))
        )
    )


def _compute_accuracy_from_rmse(rmse: float, eps: float, definition: str) -> float:
    normalized = str(definition or "").strip().lower().replace(" ", "")
    if normalized in {"1/(rmse+1e-8)", "1/(rmse+eps)", "1/rmse"}:
        local_eps = 0.0 if normalized == "1/rmse" else eps
        return float(1.0 / (rmse + local_eps))
    raise ValueError(
        "Unsupported accuracy definition. "
        f"got={definition!r}. supported=['1/(RMSE+1e-8)','1/(RMSE+eps)','1/RMSE']"
    )


def _extract_sales_inverse_params(sales_scaler: object, feature_columns: object) -> tuple[float, float] | None:
    if sales_scaler is None or feature_columns is None:
        return None
    feature_list = [str(col) for col in list(feature_columns)]
    if "sales" not in feature_list:
        return None
    idx = int(feature_list.index("sales"))
    if not hasattr(sales_scaler, "data_min_") or not hasattr(sales_scaler, "data_max_"):
        return None
    data_min = np.asarray(sales_scaler.data_min_, dtype=np.float64)
    data_max = np.asarray(sales_scaler.data_max_, dtype=np.float64)
    if idx >= data_min.shape[0] or idx >= data_max.shape[0]:
        return None
    return float(data_min[idx]), float(data_max[idx])


def _inverse_minmax(values: np.ndarray, sales_min: float, sales_max: float) -> np.ndarray:
    return values * (sales_max - sales_min) + sales_min


def validate_metric_space(metric_space: str) -> str:
    normalized = str(metric_space).strip()
    if normalized not in SUPPORTED_METRIC_SPACES:
        raise ValueError(
            "Unsupported metric_space. "
            f"got={normalized!r} supported={sorted(SUPPORTED_METRIC_SPACES)}. "
            "TODO: add explicit paper-confirmed metric space mapping before extending this list."
        )
    return normalized


def compute_metrics_with_protocol(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_protocol: dict | None = None,
    sales_scaler: object | None = None,
    feature_columns: object | None = None,
    eps: float = 1e-8,
) -> dict:
    """Compute metrics while preserving a fail-closed strict paper contract."""
    protocol = dict(metric_protocol or {})
    current_space = validate_metric_space(protocol.get("current_metric_space", "normalized_minmax_space"))
    paper_space = validate_metric_space(protocol.get("paper_metric_space", "original_sales_space"))
    paper_accuracy_definition = str(protocol.get("paper_accuracy_definition", "1/(RMSE+1e-8)"))
    current_accuracy_definition = str(protocol.get("current_accuracy_definition", "1/(RMSE+1e-8)"))
    strict_paper_metrics = bool(protocol.get("strict_paper_metrics", False))
    metric_protocol_note = str(protocol.get("metric_protocol_note", "") or "")

    if y_true is None:
        raise MetricProtocolError("missing_y_true", missing_fields=("y_true",))
    if y_pred is None:
        raise MetricProtocolError("missing_y_pred", missing_fields=("y_pred",))
    y_true_arr = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true_arr.size == 0 or y_pred_arr.size == 0:
        raise MetricProtocolError("empty_input", detail=f"shapes={y_true_arr.shape},{y_pred_arr.shape}")
    if y_true_arr.shape[0] != y_pred_arr.shape[0]:
        raise MetricProtocolError(
            "length_mismatch",
            detail=f"y_true={y_true_arr.shape[0]} y_pred={y_pred_arr.shape[0]}",
        )
    if not np.isfinite(y_true_arr).all() or not np.isfinite(y_pred_arr).all():
        raise MetricProtocolError("nonfinite_input")

    rmse_current = _compute_rmse(y_true_arr, y_pred_arr)
    mae_current = _compute_mae(y_true_arr, y_pred_arr)
    mape_current = _compute_mape(y_true_arr, y_pred_arr, eps=eps)
    smape_current = smape(y_true_arr, y_pred_arr, epsilon=eps)
    accuracy_current = _compute_accuracy_from_rmse(
        rmse=rmse_current,
        eps=eps,
        definition=current_accuracy_definition,
    )

    y_true_paper = None
    y_pred_paper = None
    inverse_transform_status = "not_required" if current_space == paper_space else "unavailable"
    inverse_transform_attempted = False
    paper_metric_status = "valid" if current_space == paper_space else "unavailable"
    paper_metric_error = ""
    notes = []

    if current_space == "normalized_minmax_space" and paper_space == "original_sales_space":
        if sales_scaler is None:
            paper_metric_status = "missing_scaler"
            paper_metric_error = "sales_scaler is required for original-sales metrics"
        elif feature_columns is None:
            paper_metric_status = "missing_feature_columns"
            paper_metric_error = "feature_columns is required for original-sales metrics"
        elif "sales" not in [str(column) for column in list(feature_columns)]:
            paper_metric_status = "missing_sales_feature"
            paper_metric_error = "feature_columns must include sales"
        else:
            inverse_params = _extract_sales_inverse_params(
                sales_scaler=sales_scaler,
                feature_columns=feature_columns,
            )
            if inverse_params is None:
                paper_metric_status = "inverse_transform_failed"
                paper_metric_error = "sales scaler does not expose valid sales min/max parameters"
            else:
                inverse_transform_attempted = True
                try:
                    sales_min, sales_max = inverse_params
                    y_true_paper = _inverse_minmax(y_true_arr, sales_min, sales_max)
                    y_pred_paper = _inverse_minmax(y_pred_arr, sales_min, sales_max)
                    if not np.isfinite(y_true_paper).all() or not np.isfinite(y_pred_paper).all():
                        raise ValueError("inverse transform produced non-finite values")
                    inverse_transform_status = "applied"
                    paper_metric_status = "valid"
                except Exception as exc:  # the typed boundary records the original cause
                    inverse_transform_status = "failed"
                    paper_metric_status = "inverse_transform_failed"
                    paper_metric_error = str(exc)
    elif current_space == paper_space:
        y_true_paper = y_true_arr
        y_pred_paper = y_pred_arr
    else:
        paper_metric_status = "inverse_transform_failed"
        paper_metric_error = f"unsupported metric-space conversion: {current_space} -> {paper_space}"
        inverse_transform_status = "failed"

    if paper_metric_status != "valid":
        if strict_paper_metrics:
            missing = {
                "missing_scaler": ("sales_scaler",),
                "missing_feature_columns": ("feature_columns",),
                "missing_sales_feature": ("sales",),
            }.get(paper_metric_status, ())
            raise MetricProtocolError(
                paper_metric_status,
                missing_fields=missing,
                detail=paper_metric_error,
            )
        message = f"FALLBACK_CURRENT_SPACE: {paper_metric_error}"
        notes.append(message)
        logging.getLogger("experiment").warning(message)

    paper_metric_computed_valid = y_true_paper is not None and y_pred_paper is not None
    if paper_metric_computed_valid:
        target_negative_count = int(np.count_nonzero(y_true_paper < 0))
        if strict_paper_metrics and paper_space == ORIGINAL_SALES_SPACE and target_negative_count:
            raise MetricProtocolError(
                "negative_target",
                detail=f"target_negative_count={target_negative_count}",
            )
        rmse_paper = _compute_rmse(y_true_paper, y_pred_paper)
        mae_paper = _compute_mae(y_true_paper, y_pred_paper)
        mape_paper = _compute_mape(y_true_paper, y_pred_paper, eps=eps)
        smape_paper = smape(y_true_paper, y_pred_paper, epsilon=eps)
        accuracy_paper = _compute_accuracy_from_rmse(
            rmse=rmse_paper,
            eps=eps,
            definition=paper_accuracy_definition,
        )
        if strict_paper_metrics and (
            not np.isfinite(smape_paper) or not SMAPE_RANGE[0] <= smape_paper <= SMAPE_RANGE[1]
        ):
            raise MetricProtocolError("metric_computation_failed", detail=f"smape={smape_paper}")
    else:
        target_negative_count = int(np.count_nonzero(y_true_arr < 0))
        rmse_paper = accuracy_paper = mae_paper = mape_paper = smape_paper = None

    use_paper_metric = strict_paper_metrics and paper_metric_computed_valid
    paper_actual_space = paper_space if paper_metric_computed_valid else "unavailable"
    original_scale_available = paper_metric_computed_valid and paper_actual_space == ORIGINAL_SALES_SPACE
    original_scale_smape = float(smape_paper) if original_scale_available else None
    if use_paper_metric:
        rmse_final = float(rmse_paper)
        accuracy_final = float(accuracy_paper)
        mae_final = float(mae_paper)
        mape_final = float(mape_paper)
        smape_final = float(smape_paper)
        rmse_metric_space = paper_actual_space
        smape_metric_space = paper_actual_space
    else:
        rmse_final = rmse_current
        accuracy_final = accuracy_current
        mae_final = mae_current
        mape_final = mape_current
        if original_scale_available:
            smape_final = float(smape_paper)
            smape_metric_space = paper_actual_space
        else:
            smape_final = float(smape_current)
            smape_metric_space = current_space
        rmse_metric_space = current_space

    metric_space_used = rmse_metric_space if rmse_metric_space == smape_metric_space else MIXED_METRIC_SPACE
    if metric_space_used == MIXED_METRIC_SPACE and not metric_protocol_note:
        metric_protocol_note = MIXED_METRIC_PROTOCOL_NOTE

    audit_true = y_true_paper if paper_metric_computed_valid else y_true_arr
    audit_pred = y_pred_paper if paper_metric_computed_valid else y_pred_arr
    sample_count = int(audit_true.size)
    target_zero_count = int(np.count_nonzero(audit_true == 0))
    prediction_zero_count = int(np.count_nonzero(audit_pred == 0))
    prediction_negative_count = int(np.count_nonzero(audit_pred < 0))

    return {
        "rmse": float(rmse_final),
        "accuracy": float(accuracy_final),
        "mae": float(mae_final),
        "mape": float(mape_final),
        "smape": float(smape_final),
        "metric_space": metric_space_used,
        "metric_space_used": metric_space_used,
        "rmse_metric_space": rmse_metric_space,
        "smape_metric_space": smape_metric_space,
        "rmse_current": float(rmse_current),
        "accuracy_current": float(accuracy_current),
        "mae_current": float(mae_current),
        "mape_current": float(mape_current),
        "smape_current": float(smape_current),
        "rmse_paper": float(rmse_paper) if rmse_paper is not None else None,
        "accuracy_paper": float(accuracy_paper) if accuracy_paper is not None else None,
        "mae_paper": float(mae_paper) if mae_paper is not None else None,
        "mape_paper": float(mape_paper) if mape_paper is not None else None,
        "smape_paper": float(smape_paper) if smape_paper is not None else None,
        "normalized_rmse": float(rmse_current) if current_space == "normalized_minmax_space" else None,
        "normalized_accuracy": float(accuracy_current) if current_space == "normalized_minmax_space" else None,
        "normalized_mae": float(mae_current) if current_space == "normalized_minmax_space" else None,
        "normalized_mape": float(mape_current) if current_space == "normalized_minmax_space" else None,
        "normalized_smape": float(smape_current) if current_space == "normalized_minmax_space" else None,
        "original_scale_rmse": float(rmse_paper) if original_scale_available else None,
        "original_scale_accuracy": float(accuracy_paper) if original_scale_available else None,
        "original_scale_mae": float(mae_paper) if original_scale_available else None,
        "original_scale_mape": float(mape_paper) if original_scale_available else None,
        "original_scale_smape": original_scale_smape,
        "metric_space_current": current_space,
        "metric_space_paper": paper_space,
        "current_metric_space_actual": current_space,
        "paper_metric_space_requested": paper_space,
        "paper_metric_space_actual": paper_actual_space,
        "primary_metric_space_actual": smape_metric_space,
        "paper_metric_aligned": bool(use_paper_metric),
        "inverse_transform_attempted": bool(inverse_transform_attempted),
        "inverse_transform_status": inverse_transform_status,
        "inverse_transform_applied": inverse_transform_status == "applied",
        "inverse_transform_available": inverse_transform_status in {"applied", "not_required"},
        "strict_paper_metrics": bool(strict_paper_metrics),
        "paper_metric_computed_valid": bool(paper_metric_computed_valid),
        "paper_metric_status": paper_metric_status,
        "paper_metric_error": paper_metric_error,
        "metric_sample_count": sample_count,
        "target_zero_count": target_zero_count,
        "target_zero_rate": float(target_zero_count / sample_count),
        "target_negative_count": int(np.count_nonzero(audit_true < 0)),
        "target_negative_rate": float(np.count_nonzero(audit_true < 0) / sample_count),
        "prediction_zero_count": prediction_zero_count,
        "prediction_zero_rate": float(prediction_zero_count / sample_count),
        "prediction_negative_count": prediction_negative_count,
        "prediction_negative_rate": float(prediction_negative_count / sample_count),
        **SMAPE_CONTRACT_FIELDS,
        "metric_protocol_note": metric_protocol_note,
        "metric_notes": " | ".join(notes),
    }


def compute_rmse_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-8,
    metric_space: str = "normalized_minmax_space",
) -> dict:
    """Compute RMSE and derived accuracy metric 1 / (RMSE + eps)."""
    checked_metric_space = validate_metric_space(metric_space)
    result = compute_metrics_with_protocol(
        y_true=y_true,
        y_pred=y_pred,
        metric_protocol={
            "current_metric_space": checked_metric_space,
            "paper_metric_space": checked_metric_space,
            "current_accuracy_definition": "1/(RMSE+1e-8)",
            "paper_accuracy_definition": "1/(RMSE+1e-8)",
            "strict_paper_metrics": False,
        },
        sales_scaler=None,
        feature_columns=None,
        eps=eps,
    )
    return {
        "rmse": result["rmse"],
        "accuracy": result["accuracy"],
        "smape": result["smape"],
        "metric_space": result["metric_space"],
    }
