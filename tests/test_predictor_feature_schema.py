from __future__ import annotations

import numpy as np
import pytest

from src.protocols.feature_schema import (
    FeatureRole,
    KnnObservedDispositionV1,
    PredictorFeatureMask,
    get_knn_schema,
    get_predictor_schema,
)
from src.protocols.sealing_protocol import FORMAL_HORIZONS, FORMAL_METHODS, FORMAL_SEEDS


def test_feature_role_and_knn_disposition_vocabularies_are_closed() -> None:
    assert tuple(role.value for role in FeatureRole) == (
        "target_signal",
        "future_known",
        "static_known",
        "observed_dynamic",
        "recursive_derived",
        "evaluation_only",
        "identifier_group_only",
    )
    assert tuple(value.value for value in KnnObservedDispositionV1) == (
        "knn_observed",
        "audit_only",
    )


@pytest.mark.parametrize("dataset_id", [f"D{i}" for i in range(1, 7)])
def test_one_exact_predictor_schema_is_shared_by_every_formal_context(dataset_id) -> None:
    expected = get_predictor_schema(dataset_id)
    assert expected.dimension == len(expected.ordered_names)
    assert len(expected.ordered_dtypes) == expected.dimension
    assert len(expected.ordered_roles) == expected.dimension
    assert len(expected.ordered_transforms) == expected.dimension
    assert len(expected.digest) == 64

    for method in FORMAL_METHODS:
        for scenario in ("without", "with"):
            for domain in ("source", "target"):
                for partition in ("train", "validation", "blind"):
                    for horizon in FORMAL_HORIZONS:
                        for seed in FORMAL_SEEDS:
                            actual = get_predictor_schema(
                                dataset_id,
                                method=method,
                                scenario=scenario,
                                domain=domain,
                                partition=partition,
                                horizon=horizon,
                                seed=seed,
                            )
                            assert actual is expected


def test_forbidden_dynamic_and_identifier_fields_never_enter_predictors() -> None:
    forbidden = {
        "D2": {"promo", "brand_id", "item_id", "entity_id"},
        "D3": {"Customers", "Open", "Promo", "store_id", "region"},
        "D4": {
            "stock_hour6_22_cnt",
            "activity_flag",
            "discount",
            "precpt",
            "avg_temperature",
            "avg_humidity",
            "avg_wind_level",
            "city_id",
            "store_id",
            "product_id",
            "first_category_id",
        },
        "D5": {
            "onpromotion",
            "transactions",
            "oil_price",
            "store_nbr",
            "item_nbr",
            "class",
            "cluster",
            "family",
        },
        "D6": {"sell_price", "item_id", "dept_id", "cat_id", "store_id", "state_id"},
    }
    for dataset_id, names in forbidden.items():
        assert names.isdisjoint(get_predictor_schema(dataset_id).ordered_names)

    d5 = get_predictor_schema("D5")
    assert "perishable" in d5.ordered_names
    assert d5.field("perishable").role is FeatureRole.STATIC_KNOWN


def test_knn_schemas_are_separate_ordered_and_fully_classified() -> None:
    assert get_knn_schema("D1").ordered_names == ("sales",)
    assert get_knn_schema("D2").ordered_names == ("sales", "promo")
    assert get_knn_schema("D3").ordered_names == ("sales", "Customers", "Open", "Promo")
    assert get_knn_schema("D4").ordered_names == ("sales",)
    assert get_knn_schema("D5").ordered_names == (
        "sales",
        "onpromotion",
        "transactions",
        "oil_price",
    )
    assert get_knn_schema("D6").ordered_names == ("sales", "sell_price")

    d4 = get_knn_schema("D4")
    assert all(
        field.disposition is KnnObservedDispositionV1.AUDIT_ONLY
        for field in d4.fields
        if field.name != "sales"
    )
    for dataset_id in ("D2", "D3", "D4", "D5", "D6"):
        schema = get_knn_schema(dataset_id)
        assert all(field.disposition in KnnObservedDispositionV1 for field in schema.fields)


def test_rfe_mask_preserves_full_schema_shape_and_sales() -> None:
    schema = get_predictor_schema("D5")
    full = PredictorFeatureMask.full(schema)
    rfe = PredictorFeatureMask.from_selected_names(schema, ("sales", "month", "perishable"))

    assert len(full.values) == schema.dimension
    assert len(rfe.values) == schema.dimension
    assert full.schema_digest == rfe.schema_digest == schema.digest
    assert full.digest != rfe.digest
    assert rfe.values[schema.index("sales")] is True

    transformed = np.arange(2 * 3 * schema.dimension, dtype=np.float64).reshape(
        2, 3, schema.dimension
    )
    masked = rfe.apply(transformed)
    assert masked.shape == transformed.shape
    assert np.array_equal(masked[..., rfe.values], transformed[..., rfe.values])
    assert np.count_nonzero(masked[..., np.logical_not(rfe.values)]) == 0

    with pytest.raises(ValueError, match="sales"):
        PredictorFeatureMask(schema.digest, (False,) * schema.dimension, schema.ordered_names)
