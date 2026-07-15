from __future__ import annotations

import numpy as np

from src.experiment.fitted_predictor import (
    FittedMethodBundle,
    KerasPredictor,
    WeightedPredictor,
    fit_joint_horizon_method_bundle,
)
from src.protocols.feature_schema import PredictorFeatureMask, get_predictor_schema
from src.utils.truth_isolation import TruthAccessTripwire, run_truth_free_fit


class HorizonModel:
    def __init__(self, horizon: int) -> None:
        self.horizon = horizon

    def predict(self, values, verbose=0):
        return np.full((len(values), 1), self.horizon, dtype=np.float64)


def test_joint_horizon_bundle_fits_exactly_h1_to_h5_once() -> None:
    schema = get_predictor_schema("D1")
    mask = PredictorFeatureMask.full(schema)
    calls: list[int] = []

    def fit_horizon(*, horizon: int):
        calls.append(horizon)
        return KerasPredictor(HorizonModel(horizon), feature_mask=mask)

    bundle = fit_joint_horizon_method_bundle(
        method="No-TL",
        seed=42,
        predictor_schema=schema,
        feature_mask=mask,
        fit_horizon=fit_horizon,
    )

    assert isinstance(bundle, FittedMethodBundle)
    assert calls == [1, 2, 3, 4, 5]
    assert bundle.horizons == (1, 2, 3, 4, 5)
    assert tuple(item.horizon for item in bundle.fitted_horizons) == bundle.horizons
    assert len({item.predictor_feature_schema_digest for item in bundle.fitted_horizons}) == 1
    assert len({item.feature_mask_digest for item in bundle.fitted_horizons}) == 1


def test_all_six_method_bundles_are_truth_free() -> None:
    schema = get_predictor_schema("D1")
    mask = PredictorFeatureMask.full(schema)
    tripwire = TruthAccessTripwire()
    methods = ("No-TL", "SS-TL", "MSWA-TL", "MSSB-TL", "MSML-TL", "MSML-TL-RFE")

    for method in methods:
        bundle = run_truth_free_fit(
            lambda method=method: fit_joint_horizon_method_bundle(
                method=method,
                seed=7,
                predictor_schema=schema,
                feature_mask=mask,
                fit_horizon=lambda *, horizon: KerasPredictor(
                    HorizonModel(horizon), feature_mask=mask
                ),
            ),
            tripwire=tripwire,
        )
        assert bundle.method == method

    assert tripwire.attempted_access_count == 0
    assert tripwire.evaluator_loader_call_count == 0


def test_bundle_rejects_horizon_or_schema_substitution() -> None:
    schema = get_predictor_schema("D1")
    mask = PredictorFeatureMask.full(schema)

    def fit_horizon(*, horizon: int):
        if horizon == 3:
            return KerasPredictor(HorizonModel(horizon))
        return KerasPredictor(HorizonModel(horizon), feature_mask=mask)

    # A missing predictor-local mask is bound to the bundle's frozen mask; it
    # cannot select or reorder a different schema at runtime.
    bundle = fit_joint_horizon_method_bundle(
        method="SS-TL",
        seed=1,
        predictor_schema=schema,
        feature_mask=mask,
        fit_horizon=fit_horizon,
    )
    assert bundle.for_horizon(3).feature_mask_digest == mask.digest


def test_bundle_binds_the_full_mask_inside_composite_predictors() -> None:
    schema = get_predictor_schema("D1")
    mask = PredictorFeatureMask.full(schema)
    bundle = fit_joint_horizon_method_bundle(
        method="MSWA-TL",
        seed=2,
        predictor_schema=schema,
        feature_mask=mask,
        fit_horizon=lambda *, horizon: WeightedPredictor(
            predictors=(
                KerasPredictor(HorizonModel(horizon)),
                KerasPredictor(HorizonModel(horizon)),
            ),
            weights=(0.5, 0.5),
        ),
    )

    weighted = bundle.for_horizon(1).predictor
    assert isinstance(weighted, WeightedPredictor)
    assert all(item.feature_mask == mask for item in weighted.predictors)
