"""Truth-free, joint-horizon rolling prediction for the sealed D1--D6 run."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
import re
from typing import Callable, Mapping, Optional

import numpy as np
import pandas as pd

from src.evaluation.metric_contract import PREDICTION_POLICY_ID
from src.experiment.fitted_predictor import (
    FittedMethodBundle,
    KerasPredictor,
    SwitchingPredictor,
    WeightedPredictor,
)
from src.protocols.artifact_schemas import get_worker_prediction_trace_schema
from src.protocols.experiment_protocol import (
    FORMAL_HORIZONS,
    FORMAL_METHODS,
    FORMAL_SEEDS,
    normalize_scenario,
)
from src.protocols.feature_schema import FeatureRole
from src.protocols.rolling_origin import (
    build_prediction_row_key,
    build_prediction_sample_key,
    build_truth_key,
    canonical_typed_sha256,
)
from src.protocols.sealing_protocol import (
    TARGET_BLIND_DAYS,
    TARGET_TRAIN_DAYS,
    TARGET_VALIDATION_DAYS,
    formal_sample_count,
    get_target_window,
    normalize_dataset_id,
)


_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


class RolloutProtocolError(ValueError):
    """The worker rollout cannot proceed without violating the sealed protocol."""


def _digest(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalize_digest(value: object, *, field_name: str) -> str:
    text = str(value).strip().lower()
    if text.startswith("sha256:"):
        text = text[7:]
    if not _HEX_RE.fullmatch(text):
        raise RolloutProtocolError(f"{field_name} must be a SHA-256 digest")
    return text


@dataclass(frozen=True)
class RolloutSchemaIdentities:
    """All schema identities consumed by a worker, with no evaluator identity."""

    predictor_feature_schema_digest: str
    feature_mask_digest: str
    observed_model_frame_schema_digest: str
    blind_covariate_frame_schema_digest: str

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            object.__setattr__(
                self,
                field_name,
                _normalize_digest(getattr(self, field_name), field_name=field_name),
            )


@dataclass(frozen=True)
class RolloutIdentity:
    """Immutable identity of one dataset/target/scenario/method/seed stream."""

    run_id: str
    cell_id: str
    attempt_id: str
    protocol_digest: str
    evaluation_contract_digest: str
    dataset_id: str
    target_entity_key: str
    scenario: str
    method: str
    seed: int
    prediction_policy_id: str = PREDICTION_POLICY_ID

    def __post_init__(self) -> None:
        for field_name in ("run_id", "cell_id", "attempt_id", "target_entity_key"):
            text = str(getattr(self, field_name))
            if not text or text != text.strip():
                raise RolloutProtocolError(f"{field_name} must be a non-empty canonical string")
            object.__setattr__(self, field_name, text)
        dataset_id = normalize_dataset_id(self.dataset_id)
        scenario = normalize_scenario(self.scenario)
        method = str(self.method)
        seed = int(self.seed)
        if method not in FORMAL_METHODS:
            raise RolloutProtocolError(f"unsupported formal method: {method!r}")
        if seed not in FORMAL_SEEDS:
            raise RolloutProtocolError(f"unsupported formal seed: {seed!r}")
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "scenario", scenario)
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(
            self,
            "protocol_digest",
            _normalize_digest(self.protocol_digest, field_name="protocol_digest"),
        )
        object.__setattr__(
            self,
            "evaluation_contract_digest",
            _normalize_digest(
                self.evaluation_contract_digest,
                field_name="evaluation_contract_digest",
            ),
        )
        if not str(self.prediction_policy_id).strip():
            raise RolloutProtocolError("prediction_policy_id must be non-empty")

    @property
    def rollout_stream_key(self) -> str:
        return canonical_typed_sha256(
            self.protocol_digest,
            self.dataset_id,
            self.target_entity_key,
            self.scenario,
            self.method,
            self.seed,
        )


def _validate_frame_contracts(
    observed: pd.DataFrame,
    blind: pd.DataFrame,
    *,
    bundle: FittedMethodBundle,
    identity: RolloutIdentity,
    schema_identities: RolloutSchemaIdentities,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    schema = bundle.fitted_horizons[0].predictor_schema
    observed_expected = ("target_entity_key", "date", *schema.ordered_names)
    blind_feature_names = tuple(
        field.name
        for field in schema.fields
        if field.role in {FeatureRole.FUTURE_KNOWN, FeatureRole.STATIC_KNOWN}
    )
    blind_expected = ("target_entity_key", "date", *blind_feature_names)
    if tuple(observed.columns) != observed_expected:
        raise RolloutProtocolError(
            f"observed_model_frame requires exact column order {observed_expected!r}"
        )
    if tuple(blind.columns) != blind_expected:
        raise RolloutProtocolError(
            f"blind_covariate_frame requires exact truth-free column order {blind_expected!r}"
        )
    for frame, field_name, expected_digest in (
        (
            observed,
            "observed_model_frame",
            schema_identities.observed_model_frame_schema_digest,
        ),
        (
            blind,
            "blind_covariate_frame",
            schema_identities.blind_covariate_frame_schema_digest,
        ),
    ):
        declared = frame.attrs.get("target_view_schema_digest")
        if declared is not None and _normalize_digest(
            declared, field_name=f"{field_name}_schema_digest"
        ) != expected_digest:
            raise RolloutProtocolError(f"{field_name} schema identity mismatch")

    prepared = []
    for frame, label in ((observed, "observed_model_frame"), (blind, "blind_covariate_frame")):
        copied = frame.copy()
        copied["date"] = pd.to_datetime(copied["date"], errors="coerce").dt.normalize()
        if copied["date"].isna().any() or copied["date"].duplicated().any():
            raise RolloutProtocolError(f"{label} contains invalid or duplicate dates")
        keys = tuple(copied["target_entity_key"].astype(str).drop_duplicates())
        if keys != (identity.target_entity_key,):
            raise RolloutProtocolError(f"{label} target entity does not match rollout identity")
        copied = copied.sort_values("date").reset_index(drop=True)
        prepared.append(copied)

    observed_copy, blind_copy = prepared
    window = get_target_window(identity.dataset_id)
    expected_observed = pd.date_range(window.observed_start, window.observed_end, freq="D")
    expected_blind = pd.date_range(window.blind_start, window.blind_end, freq="D")
    if not pd.DatetimeIndex(observed_copy["date"]).equals(expected_observed):
        raise RolloutProtocolError("observed_model_frame must contain the exact 30-day calendar")
    if not pd.DatetimeIndex(blind_copy["date"]).equals(expected_blind):
        raise RolloutProtocolError("blind_covariate_frame must contain the exact 180-day calendar")

    numeric_observed = observed_copy.loc[:, list(schema.ordered_names)].apply(
        pd.to_numeric, errors="coerce"
    )
    numeric_blind = blind_copy.loc[:, list(blind_feature_names)].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(numeric_observed.to_numpy(dtype=np.float64)).all():
        raise RolloutProtocolError("observed_model_frame predictor values must be finite")
    if numeric_blind.shape[1] and not np.isfinite(
        numeric_blind.to_numpy(dtype=np.float64)
    ).all():
        raise RolloutProtocolError("blind_covariate_frame values must be finite")
    if (numeric_observed["sales"].to_numpy(dtype=np.float64) < 0).any():
        raise RolloutProtocolError("observed sales must be nonnegative")
    observed_copy.loc[:, list(schema.ordered_names)] = numeric_observed
    if blind_feature_names:
        blind_copy.loc[:, list(blind_feature_names)] = numeric_blind
    return observed_copy, blind_copy


def _scaler_inverse_parameters(predictor: object) -> Optional[tuple[float, float]]:
    if isinstance(predictor, KerasPredictor):
        scaler = predictor.input_scaler
        if scaler is None:
            return None
        if not hasattr(scaler, "data_min_") or not hasattr(scaler, "data_max_"):
            raise RolloutProtocolError("predictor scaler cannot inverse-transform sales")
        minimum = np.asarray(scaler.data_min_, dtype=np.float64).reshape(-1)
        maximum = np.asarray(scaler.data_max_, dtype=np.float64).reshape(-1)
        if minimum.size == 0 or maximum.size == 0:
            raise RolloutProtocolError("predictor scaler has no sales parameters")
        return float(minimum[0]), float(maximum[0])
    if isinstance(predictor, SwitchingPredictor):
        return _scaler_inverse_parameters(predictor.predictors[predictor.selected_index])
    if isinstance(predictor, WeightedPredictor):
        parameters = tuple(_scaler_inverse_parameters(item) for item in predictor.predictors)
        available = tuple(item for item in parameters if item is not None)
        if not available:
            return None
        if len(available) != len(parameters) or len(set(available)) != 1:
            raise RolloutProtocolError("weighted predictor sales scalers do not match")
        return available[0]
    return None


def _inverse_prediction(
    value: float,
    *,
    horizon: int,
    predictor: object,
    callback: Optional[Callable[[float, int], float]],
) -> float:
    if callback is not None:
        converted = callback(float(value), int(horizon))
    else:
        parameters = _scaler_inverse_parameters(predictor)
        if parameters is None:
            converted = value
        else:
            minimum, maximum = parameters
            converted = float(value) * (maximum - minimum) + minimum
    result = float(np.asarray(converted, dtype=np.float64).reshape(()))
    if not np.isfinite(result):
        raise RolloutProtocolError("inverse-transformed prediction must be finite")
    return result


def _history_digest(stream_key: str, dates: list[pd.Timestamp], sales: list[float]) -> str:
    return _digest(
        {
            "policy": "rollout-history/v1",
            "rollout_stream_key": stream_key,
            "rows": [
                [timestamp.date().isoformat(), format(float(value), ".17g")]
                for timestamp, value in zip(dates, sales)
            ],
        }
    )


def _input_tensor(
    *,
    history_dates: list[pd.Timestamp],
    history_sales: list[float],
    observed_by_date: Mapping[pd.Timestamp, Mapping[str, object]],
    blind_by_date: Mapping[pd.Timestamp, Mapping[str, object]],
    bundle: FittedMethodBundle,
    input_window: int,
    recursive_derived_resolver: Optional[
        Callable[[str, date, tuple[tuple[date, float], ...]], float]
    ],
) -> np.ndarray:
    schema = bundle.fitted_horizons[0].predictor_schema
    selected_dates = history_dates[-input_window:]
    selected_sales = history_sales[-input_window:]
    history_view = tuple(
        (timestamp.date(), float(value))
        for timestamp, value in zip(history_dates, history_sales)
    )
    rows = []
    for timestamp, sales_value in zip(selected_dates, selected_sales):
        source = observed_by_date.get(timestamp) or blind_by_date.get(timestamp)
        if source is None:
            raise RolloutProtocolError("history date has no approved covariate row")
        values = []
        for field in schema.fields:
            if field.role is FeatureRole.TARGET_SIGNAL:
                value = sales_value
            elif field.role in {FeatureRole.FUTURE_KNOWN, FeatureRole.STATIC_KNOWN}:
                value = source[field.name]
            elif field.role is FeatureRole.RECURSIVE_DERIVED:
                if recursive_derived_resolver is None:
                    raise RolloutProtocolError(
                        f"recursive field {field.name!r} requires a sealed resolver"
                    )
                value = recursive_derived_resolver(field.name, timestamp.date(), history_view)
            else:
                raise RolloutProtocolError(f"forbidden rollout feature role: {field.role.value}")
            converted = float(value)
            if not np.isfinite(converted):
                raise RolloutProtocolError(f"rollout feature {field.name!r} must be finite")
            values.append(converted)
        rows.append(values)
    tensor = np.asarray(rows, dtype=np.float64)[np.newaxis, ...]
    if tensor.shape != (1, input_window, schema.dimension):
        raise RolloutProtocolError("rollout tensor shape does not match frozen schema")
    return tensor


def run_blind_rollout(
    *,
    fitted_predictors: FittedMethodBundle,
    observed_model_frame: pd.DataFrame,
    blind_covariate_frame: pd.DataFrame,
    schema_identities: RolloutSchemaIdentities,
    rollout_identity: RolloutIdentity,
    input_window: int = 10,
    inverse_transform_prediction: Optional[Callable[[float, int], float]] = None,
    recursive_derived_resolver: Optional[
        Callable[[str, date, tuple[tuple[date, float], ...]], float]
    ] = None,
) -> pd.DataFrame:
    """Predict h1--h5 from one origin snapshot and commit clipped h1 only.

    The signature intentionally contains no evaluator object or truth frame.
    Any horizon failure aborts before the origin's h1 is committed.
    """

    if not isinstance(fitted_predictors, FittedMethodBundle):
        raise TypeError("fitted_predictors must be a FittedMethodBundle")
    if not isinstance(schema_identities, RolloutSchemaIdentities):
        raise TypeError("schema_identities must be RolloutSchemaIdentities")
    if not isinstance(rollout_identity, RolloutIdentity):
        raise TypeError("rollout_identity must be RolloutIdentity")
    if isinstance(input_window, bool) or not isinstance(input_window, int) or input_window <= 0:
        raise RolloutProtocolError("input_window must be a positive integer")
    if input_window > TARGET_TRAIN_DAYS + TARGET_VALIDATION_DAYS:
        raise RolloutProtocolError("input_window exceeds the observed model history")
    if fitted_predictors.method != rollout_identity.method:
        raise RolloutProtocolError("fitted method does not match rollout identity")
    if fitted_predictors.seed != rollout_identity.seed:
        raise RolloutProtocolError("fitted seed does not match rollout identity")
    if fitted_predictors.horizons != FORMAL_HORIZONS:
        raise RolloutProtocolError("rollout requires fitted h1 through h5")

    first_fitted = fitted_predictors.fitted_horizons[0]
    if (
        first_fitted.predictor_feature_schema_digest
        != schema_identities.predictor_feature_schema_digest
    ):
        raise RolloutProtocolError("predictor feature schema identity mismatch")
    if first_fitted.feature_mask_digest != schema_identities.feature_mask_digest:
        raise RolloutProtocolError("feature mask identity mismatch")

    observed, blind = _validate_frame_contracts(
        observed_model_frame,
        blind_covariate_frame,
        bundle=fitted_predictors,
        identity=rollout_identity,
        schema_identities=schema_identities,
    )
    schema = first_fitted.predictor_schema
    stream_key = rollout_identity.rollout_stream_key
    history_dates = list(pd.DatetimeIndex(observed["date"]))
    history_sales = list(observed["sales"].to_numpy(dtype=np.float64))
    observed_by_date = {
        pd.Timestamp(row["date"]): row for row in observed.to_dict(orient="records")
    }
    blind_by_date = {
        pd.Timestamp(row["date"]): row for row in blind.to_dict(orient="records")
    }
    blind_dates = list(pd.DatetimeIndex(blind["date"]))
    records = []

    for origin_offset in range(TARGET_BLIND_DAYS):
        forecast_origin = (
            pd.Timestamp(get_target_window(rollout_identity.dataset_id).observed_end)
            + pd.Timedelta(days=origin_offset)
        )
        snapshot_digest = _history_digest(stream_key, history_dates, history_sales)
        tensor = _input_tensor(
            history_dates=history_dates,
            history_sales=history_sales,
            observed_by_date=observed_by_date,
            blind_by_date=blind_by_date,
            bundle=fitted_predictors,
            input_window=input_window,
            recursive_derived_resolver=recursive_derived_resolver,
        )
        input_digest = _digest(
            {
                "policy": "rollout-input/v1",
                "schema_digest": schema_identities.predictor_feature_schema_digest,
                "feature_mask_digest": schema_identities.feature_mask_digest,
                "shape": list(tensor.shape),
                "values": [format(float(value), ".17g") for value in tensor.reshape(-1)],
            }
        )
        valid_horizons = tuple(
            horizon
            for horizon in FORMAL_HORIZONS
            if origin_offset + horizon <= TARGET_BLIND_DAYS
        )
        pending = []
        for horizon in valid_horizons:
            fitted = fitted_predictors.for_horizon(horizon)
            prediction = np.asarray(fitted.predictor.predict(tensor), dtype=np.float64).reshape(-1)
            if prediction.shape != (1,) or not np.isfinite(prediction).all():
                raise RolloutProtocolError(
                    f"h{horizon} predictor must return one finite prediction"
                )
            raw_original = _inverse_prediction(
                float(prediction[0]),
                horizon=horizon,
                predictor=fitted.predictor,
                callback=inverse_transform_prediction,
            )
            clipped = max(0.0, raw_original)
            label_date = forecast_origin + pd.Timedelta(days=horizon)
            truth_key = build_truth_key(
                rollout_identity.evaluation_contract_digest,
                rollout_identity.dataset_id,
                rollout_identity.target_entity_key,
                label_date,
            )
            sample_key = build_prediction_sample_key(
                truth_key,
                forecast_origin,
                horizon,
            )
            pending.append(
                {
                    "horizon": horizon,
                    "label_date": label_date,
                    "truth_key": truth_key,
                    "sample_key": sample_key,
                    "prediction_row_key": build_prediction_row_key(
                        sample_key,
                        rollout_identity.scenario,
                        rollout_identity.method,
                        rollout_identity.seed,
                    ),
                    "y_pred_raw": raw_original,
                    "y_pred_clipped": clipped,
                    "was_clipped": raw_original < 0.0,
                }
            )

        # Origin barrier: mutation occurs only after every valid horizon above
        # has inverse-transformed and passed the finite check.
        h1 = next(item for item in pending if item["horizon"] == 1)
        history_dates.append(blind_dates[origin_offset])
        history_sales.append(float(h1["y_pred_clipped"]))
        after_digest = _history_digest(stream_key, history_dates, history_sales)
        for item in pending:
            records.append(
                {
                    "run_id": rollout_identity.run_id,
                    "cell_id": rollout_identity.cell_id,
                    "attempt_id": rollout_identity.attempt_id,
                    "dataset_id": rollout_identity.dataset_id,
                    "scenario": rollout_identity.scenario,
                    "target_entity_key": rollout_identity.target_entity_key,
                    "method": rollout_identity.method,
                    "seed": rollout_identity.seed,
                    "rollout_stream_key": stream_key,
                    "forecast_origin": forecast_origin.date(),
                    "label_date": item["label_date"].date(),
                    "horizon": item["horizon"],
                    "truth_key": item["truth_key"],
                    "sample_key": item["sample_key"],
                    "prediction_row_key": item["prediction_row_key"],
                    "y_pred_raw": item["y_pred_raw"],
                    "y_pred_clipped": item["y_pred_clipped"],
                    "was_clipped": item["was_clipped"],
                    "history_snapshot_digest": snapshot_digest,
                    "history_after_h1_commit_digest": after_digest,
                    "input_digest": input_digest,
                    "prediction_policy_id": rollout_identity.prediction_policy_id,
                    "predictor_feature_schema_digest": schema_identities.predictor_feature_schema_digest,
                    "feature_mask_digest": schema_identities.feature_mask_digest,
                }
            )

    counts = {horizon: 0 for horizon in FORMAL_HORIZONS}
    for record in records:
        counts[int(record["horizon"])] += 1
    expected_counts = {horizon: formal_sample_count(horizon) for horizon in FORMAL_HORIZONS}
    if counts != expected_counts:
        raise RolloutProtocolError(
            f"rollout sample counts drifted: expected={expected_counts} actual={counts}"
        )
    validated = get_worker_prediction_trace_schema().validate_records(records)
    return pd.DataFrame(validated, columns=get_worker_prediction_trace_schema().field_names)


__all__ = [
    "RolloutIdentity",
    "RolloutProtocolError",
    "RolloutSchemaIdentities",
    "run_blind_rollout",
]
