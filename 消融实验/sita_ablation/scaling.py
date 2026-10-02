"""Three independent scalers: target model, source model, and source selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from sita_ablation.data import ContractError

_FLOAT_EPS = float(np.finfo(np.float64).eps)


@dataclass
class ModelScaler:
    role: str
    columns: tuple[str, ...]
    data_min: np.ndarray
    data_max: np.ndarray
    scale: np.ndarray
    min: np.ndarray
    fit_dates: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.role not in {"target_model", "source_model"}:
            raise ContractError(f"model scaler role must be target_model or source_model, got {self.role!r}")
        width = len(self.columns)
        for name in ("data_min", "data_max", "scale", "min"):
            value = getattr(self, name)
            if value.shape != (width,) or value.dtype != np.float64:
                raise ContractError(f"{self.role} {name} must have shape {(width,)} and float64")


@dataclass
class SelectionScaler:
    role: str
    variables: tuple[str, ...]
    minimum: np.ndarray
    maximum: np.ndarray
    range: np.ndarray

    def __post_init__(self) -> None:
        if self.role != "source_selection":
            raise ContractError(f"selection scaler role must be source_selection, got {self.role!r}")
        width = len(self.variables)
        for name in ("minimum", "maximum", "range"):
            value = getattr(self, name)
            if value.shape != (width,) or value.dtype != np.float64:
                raise ContractError(f"selection {name} must have shape {(width,)} and float64")


def fit_target_model_scaler(train_rows: pd.DataFrame, columns: tuple[str, ...]) -> ModelScaler:
    return _fit_model_scaler(train_rows, columns, role="target_model")


def fit_source_model_scaler(source_train_rows: pd.DataFrame, columns: tuple[str, ...]) -> ModelScaler:
    return _fit_model_scaler(source_train_rows, columns, role="source_model")


def transform_model_frame(scaler: ModelScaler, frame: pd.DataFrame) -> pd.DataFrame:
    values = _numeric_block(frame, scaler.columns, role=scaler.role)
    if not np.isfinite(values).all():
        raise ContractError(f"{scaler.role} transform received a non-finite model value; target rows are not imputed")
    transformed = _apply_model_scaler(scaler, values)
    out = frame.copy()
    if "sales_original" not in out.columns:
        out["sales_original"] = pd.to_numeric(out["sales"], errors="raise").to_numpy(dtype=np.float64)
    for index, column in enumerate(scaler.columns):
        out[column] = transformed[:, index]
    return out


def inverse_sales(scaler: ModelScaler, values: np.ndarray) -> np.ndarray:
    if "sales" not in scaler.columns:
        raise ContractError("sales inverse requires sales to be a model column")
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(flat).all():
        raise ContractError("sales inverse received a non-finite value")
    sales_index = scaler.columns.index("sales")
    scale = float(scaler.scale[sales_index])
    offset = float(scaler.min[sales_index])
    if scale == 0.0:
        raise ContractError("sales scale is 0; the frozen constant-column scale is 1")
    return (flat - offset) / scale


def fit_selection_scaler(
    target_window: np.ndarray,
    candidate_windows: np.ndarray,
    variables: tuple[str, ...],
) -> SelectionScaler:
    target = np.asarray(target_window, dtype=np.float64)
    candidates = np.asarray(candidate_windows, dtype=np.float64)
    if target.ndim != 2:
        raise ContractError(f"target comparison window must have shape (V, L), got {target.shape}")
    if candidates.ndim != 3:
        raise ContractError(f"candidate windows must have shape (N, V, L), got {candidates.shape}")
    if target.shape[0] != len(variables) or candidates.shape[1] != len(variables):
        raise ContractError("comparison variable count does not match the supplied windows")
    if candidates.shape[2] != target.shape[1]:
        raise ContractError("target and candidate comparison windows have different lengths")
    if candidates.shape[0] == 0:
        raise ContractError("selection scaler requires at least one legal candidate")
    if not np.isfinite(target).all() or not np.isfinite(candidates).all():
        raise ContractError("selection scaler received a non-finite comparison value")
    minimum = np.empty(len(variables), dtype=np.float64)
    maximum = np.empty(len(variables), dtype=np.float64)
    for index in range(len(variables)):
        values = np.concatenate([target[index], candidates[:, index, :].reshape(-1)])
        minimum[index] = float(np.min(values))
        maximum[index] = float(np.max(values))
    return SelectionScaler(
        role="source_selection",
        variables=tuple(variables),
        minimum=minimum,
        maximum=maximum,
        range=maximum - minimum,
    )


def transform_selection(scaler: SelectionScaler, windows: np.ndarray) -> np.ndarray:
    values = np.asarray(windows, dtype=np.float64)
    single = values.ndim == 2
    if single:
        values = values[np.newaxis, :, :]
    if values.ndim != 3 or values.shape[1] != len(scaler.variables):
        raise ContractError(f"selection transform expected (V, L) or (N, V, L), got {windows.shape}")
    scaled = np.empty_like(values)
    for index, width in enumerate(scaler.range):
        if width > 0.0:
            scaled[:, index, :] = (values[:, index, :] - scaler.minimum[index]) / width
        else:
            scaled[:, index, :] = 0.0
    flat = scaled.reshape(scaled.shape[0], -1)
    return flat[0] if single else flat


def _fit_model_scaler(rows: pd.DataFrame, columns: Sequence[str], *, role: str) -> ModelScaler:
    if "date" not in rows.columns:
        raise ContractError(f"{role} fit rows have no date column")
    if rows.empty:
        raise ContractError(f"{role} fit received no rows")
    if rows["date"].duplicated().any():
        raise ContractError(f"{role} fit rows contain duplicate dates")
    block = _numeric_block(rows, columns, role=role)
    if not np.isfinite(block).all():
        raise ContractError(f"{role} fit rows contain a non-finite value; missing model values are not imputed here")
    data_min = np.min(block, axis=0).astype(np.float64, copy=False)
    data_max = np.max(block, axis=0).astype(np.float64, copy=False)
    data_range = data_max - data_min
    scale = np.ones(len(columns), dtype=np.float64)
    varying = data_range >= 10.0 * _FLOAT_EPS
    scale[varying] = 1.0 / data_range[varying]
    offset = -data_min * scale
    fit_dates = tuple(pd.Timestamp(day).strftime("%Y-%m-%d") for day in rows["date"])
    return ModelScaler(
        role=role,
        columns=tuple(columns),
        data_min=np.array(data_min, dtype=np.float64, copy=True),
        data_max=np.array(data_max, dtype=np.float64, copy=True),
        scale=np.array(scale, dtype=np.float64, copy=True),
        min=np.array(offset, dtype=np.float64, copy=True),
        fit_dates=fit_dates,
    )


def _apply_model_scaler(scaler: ModelScaler, values: np.ndarray) -> np.ndarray:
    return values * scaler.scale + scaler.min


def _numeric_block(frame: pd.DataFrame, columns: Sequence[str], *, role: str) -> np.ndarray:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ContractError(f"{role} is missing columns {missing}")
    block = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    return block.to_numpy(dtype=np.float64)
