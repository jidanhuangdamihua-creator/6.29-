"""Basic regression evaluation helpers used by experiment auditing and reporting."""

from __future__ import annotations

import logging

import numpy as np


SUPPORTED_METRIC_SPACES = {"normalized_minmax_space", "original_sales_space"}


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
    """Compute current-space and paper-space metrics in one place.

    The input arrays are assumed to be in current metric space.
    """
    protocol = dict(metric_protocol or {})
    current_space = validate_metric_space(protocol.get("current_metric_space", "normalized_minmax_space"))
    paper_space = validate_metric_space(protocol.get("paper_metric_space", "original_sales_space"))
    paper_accuracy_definition = str(protocol.get("paper_accuracy_definition", "1/(RMSE+1e-8)"))
    current_accuracy_definition = str(protocol.get("current_accuracy_definition", "1/(RMSE+1e-8)"))
    strict_paper_metrics = bool(protocol.get("strict_paper_metrics", False))

    y_true_arr = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true_arr.shape[0] != y_pred_arr.shape[0]:
        raise ValueError(
            "y_true and y_pred size mismatch: "
            f"y_true={y_true_arr.shape[0]} y_pred={y_pred_arr.shape[0]}"
        )

    rmse_current = _compute_rmse(y_true_arr, y_pred_arr)
    mae_current = _compute_mae(y_true_arr, y_pred_arr)
    mape_current = _compute_mape(y_true_arr, y_pred_arr, eps=eps)
    smape_current = smape(y_true_arr, y_pred_arr, epsilon=eps)
    accuracy_current = _compute_accuracy_from_rmse(
        rmse=rmse_current,
        eps=eps,
        definition=current_accuracy_definition,
    )

    y_true_paper = y_true_arr
    y_pred_paper = y_pred_arr
    inverse_transform_applied = False
    inverse_transform_available = current_space == paper_space
    notes = []

    if current_space == "normalized_minmax_space" and paper_space == "original_sales_space":
        inverse_params = _extract_sales_inverse_params(sales_scaler=sales_scaler, feature_columns=feature_columns)
        if inverse_params is None:
            message = (
                "paper metric requires inverse-transform to original sales space, "
                "but scaler/feature_columns are missing."
            )
            if strict_paper_metrics:
                raise ValueError(message)
            notes.append(f"FALLBACK_CURRENT_SPACE: {message}")
            warning = "WARNING: sMAPE is computed on normalized scale because original-scale inverse transform is unavailable."
            notes.append(warning)
            logging.getLogger("experiment").warning(warning)
        else:
            sales_min, sales_max = inverse_params
            y_true_paper = _inverse_minmax(y_true_arr, sales_min, sales_max)
            y_pred_paper = _inverse_minmax(y_pred_arr, sales_min, sales_max)
            inverse_transform_applied = True
            inverse_transform_available = True

    rmse_paper = _compute_rmse(y_true_paper, y_pred_paper)
    mae_paper = _compute_mae(y_true_paper, y_pred_paper)
    mape_paper = _compute_mape(y_true_paper, y_pred_paper, eps=eps)
    smape_paper = smape(y_true_paper, y_pred_paper, epsilon=eps)
    accuracy_paper = _compute_accuracy_from_rmse(
        rmse=rmse_paper,
        eps=eps,
        definition=paper_accuracy_definition,
    )

    use_paper_metric = strict_paper_metrics
    metric_space_used = paper_space if use_paper_metric else current_space
    rmse_final = rmse_paper if use_paper_metric else rmse_current
    accuracy_final = accuracy_paper if use_paper_metric else accuracy_current
    mae_final = mae_paper if use_paper_metric else mae_current
    mape_final = mape_paper if use_paper_metric else mape_current
    original_scale_available = paper_space == "original_sales_space" and inverse_transform_available
    original_scale_smape = float(smape_paper) if original_scale_available else None
    smape_final = float(original_scale_smape) if original_scale_smape is not None else float(smape_current)

    return {
        "rmse": float(rmse_final),
        "accuracy": float(accuracy_final),
        "mae": float(mae_final),
        "mape": float(mape_final),
        "smape": float(smape_final),
        "metric_space": metric_space_used,
        "metric_space_used": metric_space_used,
        "rmse_current": float(rmse_current),
        "accuracy_current": float(accuracy_current),
        "mae_current": float(mae_current),
        "mape_current": float(mape_current),
        "smape_current": float(smape_current),
        "rmse_paper": float(rmse_paper),
        "accuracy_paper": float(accuracy_paper),
        "mae_paper": float(mae_paper),
        "mape_paper": float(mape_paper),
        "smape_paper": float(smape_paper),
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
        "paper_metric_aligned": bool(use_paper_metric),
        "inverse_transform_applied": bool(inverse_transform_applied),
        "inverse_transform_available": bool(inverse_transform_available),
        "strict_paper_metrics": bool(strict_paper_metrics),
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
