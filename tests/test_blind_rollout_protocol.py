from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from src.experiment.fitted_predictor import (
    FittedMethodBundle,
    FittedMethodHorizon,
    KerasPredictor,
)
from src.protocols.blind_rollout import (
    RolloutIdentity,
    RolloutProtocolError,
    RolloutSchemaIdentities,
    run_blind_rollout,
)
from src.protocols.feature_schema import PredictorFeatureMask, get_predictor_schema
from src.protocols.sealing_protocol import get_target_window


class LastSalesPredictor:
    def __init__(self, horizon: int, *, offset: float = 0.0) -> None:
        self.horizon = int(horizon)
        self.offset = float(offset)
        self.inputs: list[np.ndarray] = []

    def predict(self, values, verbose=0):
        array = np.asarray(values, dtype=np.float64)
        self.inputs.append(array.copy())
        return (array[:, -1, 0] + self.horizon + self.offset).reshape(-1, 1)


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    window = get_target_window("D1")
    observed_dates = pd.date_range(window.observed_start, window.observed_end, freq="D")
    blind_dates = pd.date_range(window.blind_start, window.blind_end, freq="D")

    def calendar(dates: pd.DatetimeIndex) -> dict[str, object]:
        return {
            "target_entity_key": "1/10",
            "date": dates,
            "year": dates.year.astype("int64"),
            "month": dates.month.astype("int64"),
            "week": dates.isocalendar().week.to_numpy(dtype="int64"),
            "day": dates.day.astype("int64"),
        }

    observed = pd.DataFrame(calendar(observed_dates))
    observed.insert(2, "sales", np.arange(len(observed_dates), dtype=np.float64))
    blind = pd.DataFrame(calendar(blind_dates))
    return observed, blind


def _bundle(*, seed: int = 42, offset_by_horizon: dict[int, float] | None = None):
    schema = get_predictor_schema("D1")
    mask = PredictorFeatureMask.full(schema)
    models = {
        horizon: LastSalesPredictor(
            horizon,
            offset=(offset_by_horizon or {}).get(horizon, 0.0),
        )
        for horizon in range(1, 6)
    }
    fitted = tuple(
        FittedMethodHorizon(
            method="No-TL",
            horizon=horizon,
            predictor=KerasPredictor(models[horizon], feature_mask=mask),
            predictor_schema=schema,
            feature_mask=mask,
        )
        for horizon in range(1, 6)
    )
    return FittedMethodBundle("No-TL", seed, fitted), models, schema, mask


def _identities(schema, mask):
    return RolloutSchemaIdentities(
        predictor_feature_schema_digest=schema.digest,
        feature_mask_digest=mask.digest,
        observed_model_frame_schema_digest="1" * 64,
        blind_covariate_frame_schema_digest="2" * 64,
    )


def _rollout_identity():
    return RolloutIdentity(
        run_id="run-1",
        cell_id="cell-1",
        attempt_id="attempt-1",
        protocol_digest="a" * 64,
        evaluation_contract_digest="b" * 64,
        dataset_id="D1",
        target_entity_key="1/10",
        scenario="without",
        method="No-TL",
        seed=42,
    )


def test_joint_rollout_has_exact_counts_origin_barrier_and_h1_chain() -> None:
    observed, blind = _frames()
    bundle, models, schema, mask = _bundle()

    trace = run_blind_rollout(
        fitted_predictors=bundle,
        observed_model_frame=observed,
        blind_covariate_frame=blind,
        schema_identities=_identities(schema, mask),
        rollout_identity=_rollout_identity(),
        input_window=10,
    )

    assert trace.groupby("horizon").size().to_dict() == {
        1: 180,
        2: 179,
        3: 178,
        4: 177,
        5: 176,
    }
    assert "y_true" not in trace.columns
    assert trace["y_pred_clipped"].ge(0).all()
    for _, rows in trace.groupby("forecast_origin", sort=False):
        assert rows["history_snapshot_digest"].nunique() == 1
        assert rows["history_after_h1_commit_digest"].nunique() == 1
        assert rows["input_digest"].nunique() == 1

    origins = list(trace.groupby("forecast_origin", sort=False))
    for (_, previous), (_, current) in zip(origins, origins[1:]):
        assert (
            previous["history_after_h1_commit_digest"].iloc[0]
            == current["history_snapshot_digest"].iloc[0]
        )

    # Every horizon at the first origin consumed exactly the same frozen tensor.
    first_inputs = [models[horizon].inputs[0] for horizon in range(1, 6)]
    for actual in first_inputs[1:]:
        np.testing.assert_array_equal(actual, first_inputs[0])
    # h2 never feeds back: the second origin ends in the prior clipped h1 (=30),
    # not the first-origin h2 (=31).
    assert models[1].inputs[1][0, -1, 0] == 30.0


def test_predictions_are_inverse_transformed_finite_checked_then_clipped() -> None:
    observed, blind = _frames()
    bundle, _, schema, mask = _bundle(offset_by_horizon={1: -100.0})

    trace = run_blind_rollout(
        fitted_predictors=bundle,
        observed_model_frame=observed,
        blind_covariate_frame=blind,
        schema_identities=_identities(schema, mask),
        rollout_identity=_rollout_identity(),
        input_window=10,
        inverse_transform_prediction=lambda value, horizon: value * 2.0,
    )

    first_h1 = trace[trace["horizon"] == 1].iloc[0]
    assert first_h1["y_pred_raw"] == -140.0
    assert first_h1["y_pred_clipped"] == 0.0
    assert bool(first_h1["was_clipped"]) is True


def test_rollout_rejects_truth_fields_and_has_no_truth_parameter() -> None:
    observed, blind = _frames()
    bundle, _, schema, mask = _bundle()
    forbidden = {"y_true", "truth", "evaluator_truth", "evaluator_truth_frame"}
    assert not forbidden.intersection(inspect.signature(run_blind_rollout).parameters)

    with pytest.raises(RolloutProtocolError, match="exact|truth|column"):
        run_blind_rollout(
            fitted_predictors=bundle,
            observed_model_frame=observed,
            blind_covariate_frame=blind.assign(y_true=999999.0),
            schema_identities=_identities(schema, mask),
            rollout_identity=_rollout_identity(),
            input_window=10,
        )


def test_stream_and_prediction_keys_change_at_the_declared_boundaries() -> None:
    observed, blind = _frames()
    bundle, _, schema, mask = _bundle()
    base = run_blind_rollout(
        fitted_predictors=bundle,
        observed_model_frame=observed,
        blind_covariate_frame=blind,
        schema_identities=_identities(schema, mask),
        rollout_identity=_rollout_identity(),
        input_window=10,
    )
    other_identity = RolloutIdentity(
        **{
            **_rollout_identity().__dict__,
            "seed": 43,
        }
    )
    other_bundle, _, _, _ = _bundle(seed=43)
    changed = run_blind_rollout(
        fitted_predictors=other_bundle,
        observed_model_frame=observed,
        blind_covariate_frame=blind,
        schema_identities=_identities(schema, mask),
        rollout_identity=other_identity,
        input_window=10,
    )

    assert base["rollout_stream_key"].nunique() == 1
    assert changed["rollout_stream_key"].nunique() == 1
    assert base["rollout_stream_key"].iloc[0] != changed["rollout_stream_key"].iloc[0]
    assert base["truth_key"].tolist() == changed["truth_key"].tolist()
    assert base["sample_key"].tolist() == changed["sample_key"].tolist()
    assert base["prediction_row_key"].tolist() != changed["prediction_row_key"].tolist()
