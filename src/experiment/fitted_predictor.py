"""Truth-free fitted predictor boundary for the formal D1-D6 worker.

The classes in this module deliberately expose prediction only.  Evaluator
truth, test frames, metric calculation, and model fitting data are not part of
the boundary.  Every formal method/seed fit produces one immutable bundle
containing exactly h1 through h5.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

from src.protocols.experiment_protocol import FORMAL_HORIZONS, FORMAL_METHODS
from src.protocols.feature_schema import PredictorFeatureMask, PredictorFeatureSchema


@runtime_checkable
class FittedPredictor(Protocol):
    """Smallest interface consumed by blind rollout."""

    def predict(self, transformed_tensor: object) -> np.ndarray:
        """Return one finite prediction per input sample."""


def _prediction_vector(values: object, *, expected_rows: int) -> np.ndarray:
    prediction = np.asarray(values, dtype=np.float64)
    if prediction.ndim == 2 and prediction.shape[1] == 1:
        prediction = prediction[:, 0]
    elif prediction.ndim != 1:
        raise ValueError(
            "fitted predictor output must have shape (samples,) or (samples, 1)"
        )
    if len(prediction) != expected_rows:
        raise ValueError(
            f"fitted predictor returned {len(prediction)} rows for {expected_rows} inputs"
        )
    if not np.isfinite(prediction).all():
        raise ValueError("fitted predictor returned non-finite values")
    return prediction


@dataclass(frozen=True)
class KerasPredictor:
    """Prediction-only Keras adapter, optionally bound to a frozen full mask."""

    model: Any
    feature_mask: Optional[PredictorFeatureMask] = None
    input_scaler: Optional[Any] = None

    def __post_init__(self) -> None:
        if not callable(getattr(self.model, "predict", None)):
            raise TypeError("KerasPredictor model must expose predict")

    def with_feature_mask(self, feature_mask: PredictorFeatureMask) -> "KerasPredictor":
        if self.feature_mask is not None and self.feature_mask.digest != feature_mask.digest:
            raise ValueError("predictor is already bound to a different feature mask")
        return replace(self, feature_mask=feature_mask)

    def predict(self, transformed_tensor: object) -> np.ndarray:
        values = np.asarray(transformed_tensor, dtype=np.float64)
        if values.ndim == 0:
            raise ValueError("predictor input must contain a sample dimension")
        if self.input_scaler is not None:
            if values.ndim < 2:
                raise ValueError("scaled predictor input must contain a feature dimension")
            original_shape = values.shape
            values = np.asarray(
                self.input_scaler.transform(values.reshape(-1, original_shape[-1])),
                dtype=np.float64,
            ).reshape(original_shape)
        if self.feature_mask is not None:
            values = self.feature_mask.apply(values)
        raw = self.model.predict(values, verbose=0)
        return _prediction_vector(raw, expected_rows=len(values))


@dataclass(frozen=True)
class WeightedPredictor:
    """Immutable output fusion over successful source-specific predictors."""

    predictors: Tuple[FittedPredictor, ...]
    weights: Tuple[float, ...]

    def __post_init__(self) -> None:
        predictors = tuple(self.predictors)
        raw_weights = np.asarray(tuple(self.weights), dtype=np.float64)
        if not predictors or len(predictors) != len(raw_weights):
            raise ValueError("predictors and weights must be non-empty and have equal length")
        if not all(isinstance(item, FittedPredictor) for item in predictors):
            raise TypeError("every weighted member must implement FittedPredictor")
        if not np.isfinite(raw_weights).all() or np.any(raw_weights < 0.0):
            raise ValueError("weights must be finite and nonnegative")
        total = float(raw_weights.sum())
        if total <= 0.0:
            raise ValueError("at least one successful predictor weight must be positive")
        normalized = tuple(float(value / total) for value in raw_weights)
        object.__setattr__(self, "predictors", predictors)
        object.__setattr__(self, "weights", normalized)

    def predict(self, transformed_tensor: object) -> np.ndarray:
        values = np.asarray(transformed_tensor)
        predictions = tuple(item.predict(values) for item in self.predictors)
        shapes = {item.shape for item in predictions}
        if len(shapes) != 1:
            raise ValueError("weighted predictors returned inconsistent shapes")
        return np.average(np.stack(predictions, axis=0), axis=0, weights=self.weights)


@dataclass(frozen=True)
class SwitchingPredictor:
    """Target-validation-only switching over source-specific predictors."""

    predictors: Tuple[FittedPredictor, ...]
    validation_rmse: Tuple[float, ...]
    source_keys: Tuple[Tuple[str, ...], ...]
    selected_index: int

    def __post_init__(self) -> None:
        predictors = tuple(self.predictors)
        scores = tuple(float(value) for value in self.validation_rmse)
        keys = tuple(tuple(str(part) for part in key) for key in self.source_keys)
        if not predictors or len(predictors) != len(scores) or len(keys) != len(scores):
            raise ValueError("switching predictors, validation RMSE, and source keys must align")
        if not np.isfinite(np.asarray(scores)).all() or any(value < 0.0 for value in scores):
            raise ValueError("validation RMSE must be finite and nonnegative")
        if not 0 <= int(self.selected_index) < len(predictors):
            raise ValueError("selected_index is outside the candidate predictors")
        object.__setattr__(self, "predictors", predictors)
        object.__setattr__(self, "validation_rmse", scores)
        object.__setattr__(self, "source_keys", keys)
        object.__setattr__(self, "selected_index", int(self.selected_index))

    @classmethod
    def from_validation_rmse(
        cls,
        predictors: Sequence[FittedPredictor],
        validation_rmse: Sequence[float],
        source_keys: Sequence[Sequence[object]],
    ) -> "SwitchingPredictor":
        scores = tuple(float(value) for value in validation_rmse)
        if not scores:
            raise ValueError("switching requires at least one validation RMSE")
        # Candidate order is frozen by source selection; np.argmin gives a
        # deterministic first-candidate tie break without consulting test data.
        selected_index = int(np.argmin(np.asarray(scores, dtype=np.float64)))
        return cls(
            tuple(predictors),
            scores,
            tuple(tuple(str(part) for part in key) for key in source_keys),
            selected_index,
        )

    @property
    def selected_source_key(self) -> Tuple[str, ...]:
        return self.source_keys[self.selected_index]

    def predict(self, transformed_tensor: object) -> np.ndarray:
        return self.predictors[self.selected_index].predict(transformed_tensor)


def _bind_mask(
    predictor: FittedPredictor,
    feature_mask: PredictorFeatureMask,
) -> FittedPredictor:
    if isinstance(predictor, KerasPredictor):
        return predictor.with_feature_mask(feature_mask)
    if isinstance(predictor, WeightedPredictor):
        return WeightedPredictor(
            predictors=tuple(_bind_mask(item, feature_mask) for item in predictor.predictors),
            weights=predictor.weights,
        )
    if isinstance(predictor, SwitchingPredictor):
        return SwitchingPredictor(
            predictors=tuple(_bind_mask(item, feature_mask) for item in predictor.predictors),
            validation_rmse=predictor.validation_rmse,
            source_keys=predictor.source_keys,
            selected_index=predictor.selected_index,
        )
    return predictor


@dataclass(frozen=True)
class FittedMethodHorizon:
    """One fitted method/horizon under exact predictor-schema identities."""

    method: str
    horizon: int
    predictor: FittedPredictor
    predictor_schema: PredictorFeatureSchema
    feature_mask: PredictorFeatureMask
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        method = str(self.method)
        horizon = int(self.horizon)
        if method not in FORMAL_METHODS:
            raise ValueError(f"unsupported formal method: {method!r}")
        if horizon not in FORMAL_HORIZONS:
            raise ValueError(f"unsupported formal horizon: {horizon!r}")
        if self.feature_mask.schema_digest != self.predictor_schema.digest:
            raise ValueError("feature mask schema does not match predictor schema")
        predictor = _bind_mask(self.predictor, self.feature_mask)
        if not isinstance(predictor, FittedPredictor):
            raise TypeError("predictor must implement FittedPredictor")
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "horizon", horizon)
        object.__setattr__(self, "predictor", predictor)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def predictor_feature_schema_digest(self) -> str:
        return self.predictor_schema.digest

    @property
    def feature_mask_digest(self) -> str:
        return self.feature_mask.digest


@dataclass(frozen=True)
class FittedMethodBundle:
    """Atomic method/seed fit containing exactly h1 through h5."""

    method: str
    seed: int
    fitted_horizons: Tuple[FittedMethodHorizon, ...]

    def __post_init__(self) -> None:
        fitted = tuple(self.fitted_horizons)
        if tuple(item.horizon for item in fitted) != FORMAL_HORIZONS:
            raise ValueError("method bundle must contain ordered h1 through h5 exactly once")
        if any(item.method != self.method for item in fitted):
            raise ValueError("method bundle contains a substituted method")
        if len({item.predictor_feature_schema_digest for item in fitted}) != 1:
            raise ValueError("predictor schema drift across horizons")
        if len({item.feature_mask_digest for item in fitted}) != 1:
            raise ValueError("feature mask drift across horizons")
        object.__setattr__(self, "fitted_horizons", fitted)
        object.__setattr__(self, "seed", int(self.seed))

    @property
    def horizons(self) -> Tuple[int, ...]:
        return tuple(item.horizon for item in self.fitted_horizons)

    def for_horizon(self, horizon: int) -> FittedMethodHorizon:
        normalized = int(horizon)
        if normalized not in FORMAL_HORIZONS:
            raise KeyError(horizon)
        return self.fitted_horizons[FORMAL_HORIZONS.index(normalized)]


def fit_joint_horizon_method_bundle(
    *,
    method: str,
    seed: int,
    predictor_schema: PredictorFeatureSchema,
    feature_mask: PredictorFeatureMask,
    fit_horizon: Callable[..., FittedPredictor],
    metadata: Optional[Mapping[str, object]] = None,
) -> FittedMethodBundle:
    """Fit each formal horizon once without accepting any evaluator object."""

    if str(method) not in FORMAL_METHODS:
        raise ValueError(f"unsupported formal method: {method!r}")
    if feature_mask.schema_digest != predictor_schema.digest:
        raise ValueError("feature mask schema does not match predictor schema")
    fitted = []
    for horizon in FORMAL_HORIZONS:
        predictor = fit_horizon(horizon=horizon)
        fitted.append(
            FittedMethodHorizon(
                method=str(method),
                horizon=horizon,
                predictor=predictor,
                predictor_schema=predictor_schema,
                feature_mask=feature_mask,
                metadata=dict(metadata or {}),
            )
        )
    return FittedMethodBundle(str(method), int(seed), tuple(fitted))


__all__ = [
    "FittedMethodBundle",
    "FittedMethodHorizon",
    "FittedPredictor",
    "KerasPredictor",
    "SwitchingPredictor",
    "WeightedPredictor",
    "fit_joint_horizon_method_bundle",
]
