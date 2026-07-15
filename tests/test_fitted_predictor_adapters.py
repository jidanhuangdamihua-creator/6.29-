from __future__ import annotations

import inspect

import numpy as np
import pytest

from src.experiment.fitted_predictor import (
    FittedMethodHorizon,
    KerasPredictor,
    SwitchingPredictor,
    WeightedPredictor,
)
from src.protocols.feature_schema import PredictorFeatureMask, get_predictor_schema


class RecordingModel:
    def __init__(self, value: float) -> None:
        self.value = float(value)
        self.inputs: list[np.ndarray] = []

    def predict(self, values, verbose=0):
        assert verbose == 0
        array = np.asarray(values, dtype=np.float64)
        self.inputs.append(array.copy())
        return np.full((len(array), 1), self.value, dtype=np.float64)


class DoubleScaler:
    def transform(self, values):
        return np.asarray(values) * 2.0


def test_keras_predictor_keeps_full_schema_shape_and_zero_masks_rfe_fields() -> None:
    schema = get_predictor_schema("D1")
    mask = PredictorFeatureMask.from_selected_names(schema, ("sales",))
    model = RecordingModel(2.5)
    predictor = KerasPredictor(model=model, feature_mask=mask)
    tensor = np.ones((2, 10, schema.dimension), dtype=np.float64)

    actual = predictor.predict(tensor)

    assert actual.tolist() == [2.5, 2.5]
    assert model.inputs[0].shape == tensor.shape
    assert np.all(model.inputs[0][..., 0] == 1.0)
    assert np.all(model.inputs[0][..., 1:] == 0.0)


def test_keras_predictor_owns_its_train_fitted_input_scaler() -> None:
    model = RecordingModel(1.0)
    predictor = KerasPredictor(model=model, input_scaler=DoubleScaler())

    predictor.predict(np.ones((2, 3, 4), dtype=np.float64))

    assert model.inputs[0].shape == (2, 3, 4)
    assert np.all(model.inputs[0] == 2.0)


def test_weighted_predictor_freezes_normalized_successful_source_weights() -> None:
    left = KerasPredictor(RecordingModel(1.0))
    right = KerasPredictor(RecordingModel(4.0))
    predictor = WeightedPredictor((left, right), (1.0, 3.0))

    assert predictor.weights == (0.25, 0.75)
    assert predictor.predict(np.zeros((3, 2, 1))).tolist() == [3.25] * 3
    with pytest.raises((AttributeError, TypeError)):
        predictor.weights[0] = 1.0  # type: ignore[index]


def test_switching_predictor_uses_target_validation_rmse_only() -> None:
    predictors = (
        KerasPredictor(RecordingModel(10.0)),
        KerasPredictor(RecordingModel(20.0)),
    )
    predictor = SwitchingPredictor.from_validation_rmse(
        predictors,
        validation_rmse=(0.4, 0.2),
        source_keys=(("source-a",), ("source-b",)),
    )

    assert predictor.selected_index == 1
    assert predictor.selected_source_key == ("source-b",)
    assert predictor.predict(np.zeros((1, 2, 1))).tolist() == [20.0]


def test_fitted_horizon_rejects_schema_or_mask_drift() -> None:
    schema = get_predictor_schema("D1")
    full_mask = PredictorFeatureMask.full(schema)
    fitted = FittedMethodHorizon(
        method="No-TL",
        horizon=1,
        predictor=KerasPredictor(RecordingModel(1.0), feature_mask=full_mask),
        predictor_schema=schema,
        feature_mask=full_mask,
    )
    assert fitted.predictor_feature_schema_digest == schema.digest
    assert fitted.feature_mask_digest == full_mask.digest

    other_schema = get_predictor_schema("D2")
    with pytest.raises(ValueError, match="schema"):
        FittedMethodHorizon(
            method="No-TL",
            horizon=1,
            predictor=KerasPredictor(RecordingModel(1.0)),
            predictor_schema=schema,
            feature_mask=PredictorFeatureMask.full(other_schema),
        )


def test_formal_adapter_signatures_do_not_accept_evaluator_truth() -> None:
    from src.experiment.run_no_tl_experiment import fit_no_tl_predictor
    from src.experiment.experiment_runner import fit_ss_tl_predictor
    from src.transfer_methods.msml_tl import expose_fitted_msml_predictor
    from src.transfer_methods.msml_tl_rfe import expose_fitted_msml_rfe_predictor
    from src.transfer_methods.mssb_tl import expose_fitted_mssb_predictor
    from src.transfer_methods.mswa_tl import expose_fitted_mswa_predictor

    formal_fitters = (
        fit_no_tl_predictor,
        fit_ss_tl_predictor,
        expose_fitted_mswa_predictor,
        expose_fitted_mssb_predictor,
        expose_fitted_msml_predictor,
        expose_fitted_msml_rfe_predictor,
    )
    for fitter in formal_fitters:
        names = set(inspect.signature(fitter).parameters)
        assert not names.intersection(
            {"target_test_df", "evaluator_truth", "evaluator_truth_frame", "y_true"}
        )
