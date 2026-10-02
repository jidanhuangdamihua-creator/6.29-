"""Original-sales metrics for one completed target run.

The inputs are inverse-transformed sales, before the frozen negative clip.
Only formulas with a frozen value space are computed, in float64, at full
precision. Signed error is mean(pred - true) after the frozen negative clip.
Alpha entropy uses the natural logarithm. Source-side validation is not
defined. This module does not write files, pool seeds, or run significance tests.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import pandas as pd

from sita_ablation.adaptation import ABSENT, COMPLETED, PRESENT, AdaptationResult
from sita_ablation.data import DATASETS, ContractError, DatasetId, load_dataset_contract
from sita_ablation.stage_a import NOT_APPLICABLE

ORIGINAL_SALES = "original_sales"
TEST_DAYS = 180
_EMA = "EMA"
_ALPHA_CLIP_FLOOR = 1e-12
_ALPHA_CLIP_CEILING = 1.0


@dataclass(frozen=True)
class MetricPair:
    """One original-sales comparison after the frozen negative clip."""

    rmse: float
    smape: float
    zero_sales_count: int
    smape_zero_denominator_count: int
    clipped_true_negative_count: int
    clipped_pred_negative_count: int
    value_space: str

    def __post_init__(self) -> None:
        if self.value_space != ORIGINAL_SALES:
            raise ContractError(f"metrics stay in original sales, got {self.value_space!r}")
        if not math.isfinite(self.rmse) or not math.isfinite(self.smape):
            raise ContractError("original-sales metrics are non-finite")


@dataclass(frozen=True)
class TestMetrics:
    """In-memory test metrics for one completed cell. Nothing is written."""

    dataset: DatasetId
    n_test: int
    pair: MetricPair
    label_date: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    signed_error: float
    alpha_entropy: float | None
    effective_source_count: float | None
    source_info: str
    source_validation_metric: str | None
    source_validation_value_space: str | None
    unscored_test_dates: tuple[str, ...]
    test_weight_variant: str

    def __post_init__(self) -> None:
        if self.dataset not in DATASETS:
            raise ContractError(f"unknown dataset {self.dataset!r}")
        if self.n_test != TEST_DAYS or self.unscored_test_dates != ():
            raise ContractError("test metrics cover every frozen target day")
        if self.test_weight_variant != _EMA:
            raise ContractError("test metrics use EMA predictions")
        if self.pair.value_space != ORIGINAL_SALES:
            raise ContractError("test metrics are in original sales")
        if isinstance(self.signed_error, bool) or not isinstance(self.signed_error, (int, float)) or not math.isfinite(float(self.signed_error)):
            raise ContractError("signed error must be mean(pred - true)")
        if self.alpha_entropy is not None and (
            isinstance(self.alpha_entropy, bool)
            or not isinstance(self.alpha_entropy, (int, float))
            or not math.isfinite(float(self.alpha_entropy))
        ):
            raise ContractError("alpha entropy must be the natural-log formula or null")
        _require_source_fields(
            self.source_info,
            self.source_validation_metric,
            self.source_validation_value_space,
            self.effective_source_count,
        )


def evaluate_original_sales(y_true: np.ndarray, y_pred: np.ndarray) -> MetricPair:
    """Score one pair of original-sales vectors.

    ``y_true`` and ``y_pred`` are shape ``(N,)``, after the sales inverse
    transform and before clipping. Negatives are then set to 0. RMSE and
    sMAPE use the frozen formulas. Samples with a zero absolute denominator
    stay in the mean.
    """
    true_raw = _sales_vector(y_true, "true values")
    pred_raw = _sales_vector(y_pred, "predictions")
    if true_raw.shape != pred_raw.shape:
        raise ContractError(
            f"prediction and true-value lengths differ: pred={pred_raw.shape[0]} true={true_raw.shape[0]}"
        )
    if true_raw.size == 0:
        raise ContractError("original-sales metrics require at least one point")
    zero_sales_count = int(np.count_nonzero(true_raw == 0.0))
    clipped_true_negative_count = int(np.count_nonzero(true_raw < 0.0))
    clipped_pred_negative_count = int(np.count_nonzero(pred_raw < 0.0))
    true = np.maximum(true_raw, 0.0)
    pred = np.maximum(pred_raw, 0.0)
    smape_zero_denominator_count = int(np.count_nonzero(np.abs(true) + np.abs(pred) == 0.0))
    diff = pred - true
    rmse = float(np.sqrt(np.mean(np.square(diff), dtype=np.float64)))
    denom = np.abs(true) + np.abs(pred) + np.float64(1e-8)
    smape = float(np.float64(100.0) * np.mean(np.float64(2.0) * np.abs(diff) / denom, dtype=np.float64))
    return MetricPair(
        rmse=rmse,
        smape=smape,
        zero_sales_count=zero_sales_count,
        smape_zero_denominator_count=smape_zero_denominator_count,
        clipped_true_negative_count=clipped_true_negative_count,
        clipped_pred_negative_count=clipped_pred_negative_count,
        value_space=ORIGINAL_SALES,
    )


def signed_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return ``mean(pred - true)`` on the clipped original-sales vectors."""
    true, pred = _clipped_sales(y_true, y_pred)
    value = float(np.mean(pred - true, dtype=np.float64))
    if not math.isfinite(value):
        raise ContractError("signed error is non-finite")
    return value


def alpha_entropy(alpha: np.ndarray) -> float:
    """Natural-log entropy after clipping each weight to ``[1e-12, 1]``.

    A clipped term of 0 contributes 0. Weights are not renormalized.
    ``N_eff`` stays :func:`effective_source_count` on the unclipped weights.
    """
    values = _fusion_weights(alpha)
    clipped = np.clip(values, _ALPHA_CLIP_FLOOR, _ALPHA_CLIP_CEILING)
    terms = np.zeros(clipped.shape, dtype=np.float64)
    nonzero = clipped != 0.0
    terms[nonzero] = clipped[nonzero] * np.log(clipped[nonzero])
    value = float(-np.sum(terms, dtype=np.float64))
    if not math.isfinite(value):
        raise ContractError("alpha entropy is non-finite")
    return value


def effective_source_count(alpha: np.ndarray) -> float:
    """Return ``N_eff = 1/sum(alpha_i^2)`` on the unclipped fusion weights."""
    values = _fusion_weights(alpha)
    total = np.sum(np.square(values), dtype=np.float64)
    if not np.isfinite(total) or total == 0.0:
        raise ContractError("effective source count is undefined")
    count = np.float64(1.0) / np.float64(total)
    if not np.isfinite(count):
        raise ContractError("effective source count is non-finite")
    return float(count)


def evaluate_test(
    result: AdaptationResult,
    y_true: np.ndarray,
    *,
    alpha: np.ndarray | None = None,
) -> TestMetrics:
    """Score EMA test predictions on the frozen 180 target days.

    Dates, predictions, and true values must be finite and pointwise aligned.
    The date sequence must be the dataset's frozen test interval, with no
    missing day and no duplicate. The returned object stays in memory.
    """
    if result.status != COMPLETED:
        raise ContractError(f"test metrics require a completed adaptation, got {result.status!r}")
    if result.test_weight_variant != _EMA:
        raise ContractError("test metrics use EMA predictions")
    if result.unscored_test_dates:
        raise ContractError("test metrics require every frozen target day; unscored dates are present")
    _require_recorded_source_fields(result)
    dataset = result.spec.dataset
    pred = _sales_vector(result.test_predictions, "predictions")
    true = _sales_vector(y_true, "true values")
    if pred.shape != true.shape:
        raise ContractError(
            f"prediction and true-value lengths differ: pred={pred.shape[0]} true={true.shape[0]}"
        )
    dates = _label_dates(result.label_date)
    if not (true.shape[0] == pred.shape[0] == len(dates)):
        raise ContractError(
            "dates, predictions, and true values are not pointwise aligned: "
            f"dates={len(dates)} true={true.shape[0]} pred={pred.shape[0]}"
        )
    _require_frozen_test_dates(dataset, dates)
    count = _source_count(result, alpha)
    pair = evaluate_original_sales(true, pred)
    error = signed_error(true, pred)
    _require_optional_match(result.signed_error, error, "signed error")
    entropy = None if result.source_info == ABSENT else alpha_entropy(alpha)
    if result.source_info == ABSENT and result.alpha_entropy is not None:
        raise ContractError("a source-free arm has no alpha entropy")
    _require_optional_match(result.alpha_entropy, entropy, "alpha entropy")
    return TestMetrics(
        dataset=dataset,
        n_test=TEST_DAYS,
        pair=pair,
        label_date=np.array(dates, dtype="datetime64[D]"),
        y_true=true,
        y_pred=pred,
        signed_error=error,
        alpha_entropy=entropy,
        effective_source_count=count,
        source_info=result.source_info,
        source_validation_metric=result.source_validation_metric,
        source_validation_value_space=result.source_validation_value_space,
        unscored_test_dates=(),
        test_weight_variant=_EMA,
    )


def _source_count(result: AdaptationResult, alpha: np.ndarray | None) -> float | None:
    if result.source_info == ABSENT:
        if alpha is not None:
            raise ContractError("a source-free arm has no fusion weights")
        if result.effective_source_count is not None:
            raise ContractError("a source-free arm has no effective source count")
        return None
    if alpha is None:
        raise ContractError("a sourced arm requires fusion weights for the effective source count")
    weights = np.asarray(alpha)
    if weights.ndim != 1:
        raise ContractError(f"fusion weights must be a non-empty vector, got shape {weights.shape}")
    if result.spec.requested_k is not None and weights.shape[0] != int(result.spec.requested_k):
        raise ContractError("fusion weights are not aligned with the arm's source count")
    count = effective_source_count(alpha)
    recorded = result.effective_source_count
    if recorded is None or isinstance(recorded, (bool, str)) or not isinstance(recorded, (int, float)):
        raise ContractError("effective source count does not match 1/sum(alpha_i^2)")
    if not math.isfinite(float(recorded)) or np.float64(recorded) != np.float64(count):
        raise ContractError("effective source count does not match 1/sum(alpha_i^2)")
    return count


def _require_recorded_source_fields(result: AdaptationResult) -> None:
    metric = result.source_validation_metric
    space = result.source_validation_value_space
    if result.spec.selection_rule == "none":
        if result.source_info != ABSENT:
            raise ContractError("a source-free arm's source info is ABSENT")
    elif result.source_info != PRESENT:
        raise ContractError("a sourced arm's source info is PRESENT")
    if result.source_info == ABSENT:
        if metric is not None or space is not None:
            raise ContractError("a source-free arm does not record a source-validation metric")
        return
    if result.source_info != PRESENT:
        raise ContractError(f"source info must be ABSENT or PRESENT, got {result.source_info!r}")
    if metric != NOT_APPLICABLE or space != NOT_APPLICABLE:
        raise ContractError("source validation metric and value space are NOT_APPLICABLE")
    if isinstance(metric, (int, float)) or isinstance(space, (int, float)):
        raise ContractError("source validation cannot be filled with a numeric error")


def _require_source_fields(
    source_info: str,
    metric: str | None,
    space: str | None,
    count: float | None,
) -> None:
    if source_info == ABSENT:
        if metric is not None or space is not None or count is not None:
            raise ContractError("a source-free arm has no source-validation metric and no effective source count")
        return
    if source_info != PRESENT:
        raise ContractError(f"source info must be ABSENT or PRESENT, got {source_info!r}")
    if metric != NOT_APPLICABLE or space != NOT_APPLICABLE:
        raise ContractError("source validation metric and value space are NOT_APPLICABLE")
    if isinstance(count, bool) or not isinstance(count, (int, float)) or not math.isfinite(float(count)):
        raise ContractError("effective source count must be the frozen 1/sum(alpha_i^2)")


def _clipped_sales(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    true_raw = _sales_vector(y_true, "true values")
    pred_raw = _sales_vector(y_pred, "predictions")
    if true_raw.shape != pred_raw.shape:
        raise ContractError(
            f"prediction and true-value lengths differ: pred={pred_raw.shape[0]} true={true_raw.shape[0]}"
        )
    if true_raw.size == 0:
        raise ContractError("original-sales metrics require at least one point")
    return np.maximum(true_raw, 0.0), np.maximum(pred_raw, 0.0)


def _fusion_weights(alpha: np.ndarray) -> np.ndarray:
    weights = np.asarray(alpha)
    if weights.ndim != 1 or weights.size == 0:
        raise ContractError(f"fusion weights must be a non-empty vector, got shape {weights.shape}")
    if weights.dtype.kind not in {"f", "i", "u"}:
        raise ContractError("fusion weights must be numeric")
    values = np.array(weights, dtype=np.float64, copy=True)
    if not np.isfinite(values).all():
        raise ContractError("fusion weights contain a non-finite value")
    return values


def _require_optional_match(recorded: object, computed: float | None, name: str) -> None:
    if recorded is None:
        return
    if computed is None or isinstance(recorded, (bool, str)) or not isinstance(recorded, (int, float)):
        raise ContractError(f"{name} must match the frozen formula")
    if not math.isfinite(float(recorded)) or np.float64(recorded) != np.float64(computed):
        raise ContractError(f"{name} must match the frozen formula")


def _sales_vector(values: np.ndarray | Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ContractError(f"{name} must be a one-dimensional sales vector, got shape {array.shape}")
    if array.dtype.kind not in {"f", "i", "u"}:
        raise ContractError(f"{name} must be numeric sales")
    converted = np.array(array, dtype=np.float64, copy=True)
    if not np.isfinite(converted).all():
        raise ContractError(f"{name} contain a non-finite sales value")
    return converted


def _label_dates(values: np.ndarray | Sequence[object]) -> tuple[pd.Timestamp, ...]:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ContractError(f"label dates must be one-dimensional, got shape {raw.shape}")
    if raw.size and np.issubdtype(raw.dtype, np.datetime64):
        if np.isnat(raw).any():
            raise ContractError("test dates contain a missing date")
        return tuple(pd.Timestamp(item).normalize() for item in raw.tolist())
    return tuple(_one_date(item) for item in raw.tolist())


def _one_date(item: object) -> pd.Timestamp:
    if item is None or item is pd.NaT:
        raise ContractError("test dates contain a missing date")
    if isinstance(item, str) and not item.strip():
        raise ContractError("test dates contain a missing date")
    try:
        missing = pd.isna(item)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        raise ContractError("test dates contain a missing date")
    try:
        stamp = pd.Timestamp(item)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ContractError(f"test dates contain an unreadable date: {item!r}") from exc
    if pd.isna(stamp):
        raise ContractError("test dates contain a missing date")
    return stamp.normalize()


def _require_frozen_test_dates(dataset: DatasetId, dates: tuple[pd.Timestamp, ...]) -> None:
    contract = load_dataset_contract(dataset)
    expected = tuple(
        pd.Timestamp(day).normalize()
        for day in pd.date_range(contract.target_test.start, contract.target_test.end, freq="D")
    )
    if len(expected) != TEST_DAYS:
        raise ContractError(f"{dataset} frozen test interval is not {TEST_DAYS} days")
    if len(set(dates)) != len(dates):
        raise ContractError("test dates contain a duplicate date")
    if len(dates) != TEST_DAYS:
        raise ContractError(f"test metrics require the frozen {TEST_DAYS} target days, got {len(dates)} dates")
    missing = set(expected) - set(dates)
    extra = set(dates) - set(expected)
    if missing or extra:
        raise ContractError(
            "test dates do not cover the frozen 180-day target interval; "
            f"missing={len(missing)} extra={len(extra)}"
        )
    if dates != expected:
        raise ContractError("test dates are misaligned with the frozen 180-day target interval")
