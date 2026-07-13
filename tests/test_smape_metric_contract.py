from __future__ import annotations

import numpy as np
import pytest

from src.evaluation import metric_contract
from src.evaluation.metric_contract import (
    METRIC_CONTRACT_VERSION,
    METRIC_IDENTITY_FIELDS,
    SMAPE_DEFINITION_ID,
    MetricProtocolError,
    compute_metric_index_digest,
    is_formally_comparable_smape_row,
)
from src.evaluation.metrics import compute_metrics_with_protocol


class DummyMinMaxScaler:
    data_min_ = np.array([10.0])
    data_max_ = np.array([20.0])


STRICT_PROTOCOL = {
    "current_metric_space": "normalized_minmax_space",
    "paper_metric_space": "original_sales_space",
    "strict_paper_metrics": True,
}

IDENTITY = {
    "metric_target_key": "target/item",
    "metric_horizon": 1,
    "metric_sample_count": 2,
    "metric_date_start": "2024-01-02",
    "metric_date_end": "2024-01-03",
    "metric_index_digest": "digest-123",
}


def _valid_row(**overrides):
    row = {
        "strict_paper_metrics": True,
        "paper_metric_space_requested": "original_sales_space",
        "paper_metric_space_actual": "original_sales_space",
        "primary_metric_space_actual": "original_sales_space",
        "smape_metric_space": "original_sales_space",
        "inverse_transform_status": "applied",
        "inverse_transform_applied": True,
        "paper_metric_computed_valid": True,
        "paper_metric_status": "valid",
        "paper_metric_error": "",
        "smape": 12.5,
        "metric_sample_count": 2,
        "metric_contract_version": METRIC_CONTRACT_VERSION,
        "smape_definition_id": SMAPE_DEFINITION_ID,
        "smape_unit": "percent",
        "smape_epsilon": 1e-8,
        "smape_range_min": 0.0,
        "smape_range_max": 200.0,
        "sales_value_policy": "clip_negative_to_zero_v1",
        "target_negative_count": 0,
        **IDENTITY,
    }
    row.update(overrides)
    return row


@pytest.mark.parametrize(
    ("kwargs", "status", "missing"),
    [
        ({"sales_scaler": None}, "missing_scaler", "sales_scaler"),
        ({"y_true": None}, "missing_y_true", "y_true"),
        ({"y_pred": None}, "missing_y_pred", "y_pred"),
        ({"feature_columns": None}, "missing_feature_columns", "feature_columns"),
        ({"feature_columns": ["year"]}, "missing_sales_feature", "sales"),
    ],
)
def test_strict_missing_input_raises_typed_error(kwargs, status, missing):
    call = {
        "y_true": np.array([0.0, 0.5]),
        "y_pred": np.array([0.25, 0.25]),
        "metric_protocol": STRICT_PROTOCOL,
        "sales_scaler": DummyMinMaxScaler(),
        "feature_columns": ["sales"],
    }
    call.update(kwargs)

    with pytest.raises(MetricProtocolError) as exc_info:
        compute_metrics_with_protocol(**call)

    assert exc_info.value.status == status
    assert missing in str(exc_info.value)


def test_strict_inverse_success_returns_original_sales_primary_metrics_and_contract():
    result = compute_metrics_with_protocol(
        y_true=np.array([0.0, 0.5]),
        y_pred=np.array([0.25, 0.25]),
        metric_protocol=STRICT_PROTOCOL,
        sales_scaler=DummyMinMaxScaler(),
        feature_columns=["sales"],
    )

    assert result["strict_paper_metrics"] is True
    assert result["paper_metric_space_requested"] == "original_sales_space"
    assert result["paper_metric_space_actual"] == "original_sales_space"
    assert result["primary_metric_space_actual"] == "original_sales_space"
    assert result["rmse_metric_space"] == "original_sales_space"
    assert result["smape_metric_space"] == "original_sales_space"
    assert result["inverse_transform_status"] == "applied"
    assert result["inverse_transform_applied"] is True
    assert result["paper_metric_computed_valid"] is True
    assert result["paper_metric_status"] == "valid"
    assert result["paper_metric_error"] == ""
    assert result["metric_contract_version"] == METRIC_CONTRACT_VERSION
    assert result["smape_definition_id"] == SMAPE_DEFINITION_ID
    assert result["smape_unit"] == "percent"
    assert result["smape_epsilon"] == 1e-8
    assert result["smape"] == result["smape_paper"] == result["original_scale_smape"]
    assert result["rmse"] == result["rmse_paper"] == result["original_scale_rmse"]


def test_strict_original_input_uses_not_required_and_remains_formally_comparable():
    result = compute_metrics_with_protocol(
        y_true=np.array([10.0, 15.0]),
        y_pred=np.array([12.5, 12.5]),
        metric_protocol={
            "current_metric_space": "original_sales_space",
            "paper_metric_space": "original_sales_space",
            "strict_paper_metrics": True,
        },
        sales_scaler=None,
        feature_columns=None,
    )
    result.update(IDENTITY)

    assert result["inverse_transform_status"] == "not_required"
    assert result["inverse_transform_applied"] is False
    assert is_formally_comparable_smape_row(result) == {
        "eligible": True,
        "failure_reasons": [],
    }


def test_legacy_row_without_contract_fields_is_excluded_without_key_error():
    result = is_formally_comparable_smape_row({"smape": 12.0})

    assert result["eligible"] is False
    assert "missing:metric_contract_version" in result["failure_reasons"]
    assert "missing:strict_paper_metrics" in result["failure_reasons"]


def test_numeric_non_strict_row_is_excluded_from_formal_comparison():
    result = is_formally_comparable_smape_row(
        _valid_row(strict_paper_metrics=False, smape=1.0)
    )

    assert result["eligible"] is False
    assert "invalid:strict_paper_metrics" in result["failure_reasons"]


def test_contract_rejects_negative_target_instead_of_clipping_inside_metric_layer():
    with pytest.raises(MetricProtocolError) as exc_info:
        compute_metrics_with_protocol(
            y_true=np.array([-1.0, 5.0]),
            y_pred=np.array([0.0, 5.0]),
            metric_protocol={
                "current_metric_space": "original_sales_space",
                "paper_metric_space": "original_sales_space",
                "strict_paper_metrics": True,
            },
            sales_scaler=None,
            feature_columns=None,
        )

    assert exc_info.value.status == "negative_target"


def test_metric_index_digest_is_order_sensitive_and_deterministic():
    first = compute_metric_index_digest(["a", "b"])
    second = compute_metric_index_digest(["a", "b"])
    reversed_digest = compute_metric_index_digest(["b", "a"])

    assert first == second
    assert first != reversed_digest


@pytest.mark.parametrize("field", METRIC_IDENTITY_FIELDS)
def test_formal_smape_rejects_missing_metric_identity_field(field):
    row = _valid_row()
    row.pop(field)

    decision = is_formally_comparable_smape_row(row)

    assert decision["eligible"] is False
    assert f"missing:{field}" in decision["failure_reasons"]


def test_shared_metric_identity_is_derived_from_orchestration_manifest():
    class Record:
        def __init__(self, sample_key, label_date, target_key=("target", "item")):
            self.sample_key = sample_key
            self.label_date = label_date
            self.target_key = target_key

    class Manifest:
        def for_horizon(self, horizon):
            assert horizon == 1
            return (
                Record("manifest-key-a", "2024-01-02"),
                Record("manifest-key-b", "2024-01-03"),
            )

    builder = getattr(metric_contract, "build_metric_identity_from_manifest", None)
    assert callable(builder)
    identity = builder(Manifest(), horizon=1)

    assert set(identity) == set(METRIC_IDENTITY_FIELDS)
    assert identity["metric_target_key"] == "target/item"
    assert identity["metric_horizon"] == 1
    assert identity["metric_sample_count"] == 2
    assert identity["metric_date_start"] == "2024-01-02"
    assert identity["metric_date_end"] == "2024-01-03"
    assert identity["metric_index_digest"]


@pytest.mark.parametrize("mode", ["empty", "multiple_targets"])
def test_shared_metric_identity_rejects_invalid_manifest(mode):
    class Record:
        def __init__(self, target_key):
            self.sample_key = str(target_key)
            self.label_date = "2024-01-02"
            self.target_key = target_key

    class Manifest:
        def for_horizon(self, horizon):
            assert horizon == 1
            if mode == "empty":
                return ()
            return (Record(("target-a",)), Record(("target-b",)))

    builder = getattr(metric_contract, "build_metric_identity_from_manifest", None)
    assert callable(builder)
    with pytest.raises(MetricProtocolError) as exc_info:
        builder(Manifest(), horizon=1)

    assert exc_info.value.status == "metric_identity_mismatch"


def test_not_required_legacy_inverse_boolean_does_not_control_eligibility():
    result = is_formally_comparable_smape_row(
        _valid_row(
            inverse_transform_status="not_required",
            inverse_transform_applied=False,
        )
    )

    assert result["eligible"] is True
