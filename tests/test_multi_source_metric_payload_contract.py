from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from scripts import run_full_paper_experiments as d1_d3_runner
from src.experiment import run_no_tl_experiment as no_tl_module
from src.evaluation.metric_contract import MetricProtocolError
from src.experiment import experiment_runner
from src.experiment.experiment_runner import _extract_method_metrics, results_to_dataframe
from src.transfer_methods import msml_tl, msml_tl_rfe, mssb_tl, mswa_tl
from src.utils.entity_experiment import _row_from_result
from src.utils.entity_experiment import (
    _metric_identity_from_manifest,
    _row_from_metric_protocol_error,
)


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


def _payload(container: str = "fused_result"):
    values = {
        "rmse": 0.01,
        "accuracy": 100.0,
        "smape": 0.01,
        "y_true": np.array([0.0, 0.5]),
        "y_pred": np.array([0.25, 0.25]),
        "sales_scaler": DummyMinMaxScaler(),
        "feature_columns": ["sales"],
        **IDENTITY,
    }
    return {container: values, "meta": {}}


@pytest.mark.parametrize(
    ("method", "container"),
    [
        ("MSWA-TL", "fused_result"),
        ("MSSB-TL", "final_result"),
        ("MSML-TL", "fused_result"),
        ("MSML-TL-RFE", "fused_result"),
    ],
)
def test_four_multisource_methods_cannot_passthrough_internal_strict_metrics(method, container):
    result = _extract_method_metrics(
        _payload(container),
        method_name=method,
        metric_protocol=STRICT_PROTOCOL,
        expected_metric_identity=IDENTITY,
    )

    assert result["smape"] != 0.01
    assert result["smape"] == result["smape_paper"] == result["original_scale_smape"]
    assert result["rmse"] == result["rmse_paper"] == result["original_scale_rmse"]
    assert result["paper_metric_space_actual"] == "original_sales_space"
    assert result["paper_metric_computed_valid"] is True
    assert result["paper_metric_status"] == "valid"
    for field, expected in IDENTITY.items():
        assert result[field] == expected


def test_no_tl_wrapper_uses_strict_extractor_and_expected_identity(monkeypatch):
    received = {}

    def fake_bottom_runner(**kwargs):
        received.update(kwargs["expected_metric_identity"])
        return {
            "method": "No-TL",
            **_payload()["fused_result"],
        }

    monkeypatch.setattr(no_tl_module, "run_no_tl_experiment", fake_bottom_runner)

    result = experiment_runner.run_no_tl_experiment(
        target_df=pd.DataFrame(),
        feature_cols=["sales"],
        metric_protocol=STRICT_PROTOCOL,
        expected_metric_identity=IDENTITY,
    )

    assert received == IDENTITY
    assert result["paper_metric_computed_valid"] is True
    assert result["smape_metric_space"] == "original_sales_space"
    assert {field: result[field] for field in IDENTITY} == IDENTITY


@pytest.mark.parametrize(
    ("field", "status"),
    [
        ("y_true", "missing_y_true"),
        ("y_pred", "missing_y_pred"),
        ("sales_scaler", "missing_scaler"),
        ("feature_columns", "missing_feature_columns"),
    ],
)
def test_strict_extractor_lists_missing_payload_field(field, status):
    raw = _payload()
    del raw["fused_result"][field]

    with pytest.raises(MetricProtocolError) as exc_info:
        _extract_method_metrics(
            raw,
            method_name="MSWA-TL",
            metric_protocol=STRICT_PROTOCOL,
            expected_metric_identity=IDENTITY,
        )

    assert exc_info.value.status == status
    assert field in str(exc_info.value)


def test_strict_extractor_rejects_payload_identity_mismatch():
    raw = _payload()
    raw["fused_result"]["metric_index_digest"] = "wrong"

    with pytest.raises(MetricProtocolError) as exc_info:
        _extract_method_metrics(
            raw,
            method_name="MSWA-TL",
            metric_protocol=STRICT_PROTOCOL,
            expected_metric_identity=IDENTITY,
        )

    assert exc_info.value.status == "metric_identity_mismatch"
    assert "metric_index_digest" in str(exc_info.value)


def test_strict_extractor_checks_manifest_sample_count_against_prediction_arrays():
    raw = _payload()
    expected = {**IDENTITY, "metric_sample_count": 3}
    raw["fused_result"].update(expected)

    with pytest.raises(MetricProtocolError) as exc_info:
        _extract_method_metrics(
            raw,
            method_name="MSWA-TL",
            metric_protocol=STRICT_PROTOCOL,
            expected_metric_identity=expected,
        )

    assert exc_info.value.status == "metric_identity_mismatch"
    assert "metric_sample_count" in str(exc_info.value)


def test_strict_extractor_does_not_require_or_trust_internal_rmse_and_accuracy():
    raw = _payload()
    del raw["fused_result"]["rmse"]
    del raw["fused_result"]["accuracy"]

    result = _extract_method_metrics(
        raw,
        method_name="MSWA-TL",
        metric_protocol=STRICT_PROTOCOL,
        expected_metric_identity=IDENTITY,
    )

    assert np.isfinite(result["rmse"])
    assert np.isfinite(result["smape"])
    assert result["paper_metric_computed_valid"] is True


def test_d4_d6_row_preserves_computed_paper_metrics_and_reference_semantics():
    extracted = _extract_method_metrics(
        _payload(),
        method_name="MSWA-TL",
        metric_protocol=STRICT_PROTOCOL,
        expected_metric_identity=IDENTITY,
    )
    row = _row_from_result(
        extracted,
        method="MSWA-TL",
        entity_key="target/item",
        config={
            "dataset_id": 5,
            "dataset_name": "Dataset5",
            "info_sharing": "without",
            "source_count": 3,
            "metric_protocol": STRICT_PROTOCOL,
        },
        elapsed=1.0,
    )

    assert row["rmse_paper"] == extracted["rmse_paper"]
    assert row["smape_paper"] == extracted["smape_paper"]
    assert row["original_scale_rmse"] == extracted["original_scale_rmse"]
    assert row["original_scale_smape"] == extracted["original_scale_smape"]
    assert row["paper_metric_aligned"] is True
    assert row["paper_reference_available"] is False
    assert row["paper_reference_status"] == "no_paper_reference"
    assert row["inverse_transform_status"] == "applied"


def test_d1_d3_dataframe_serialization_preserves_strict_metric_contract_fields():
    extracted = _extract_method_metrics(
        _payload(),
        method_name="MSWA-TL",
        metric_protocol=STRICT_PROTOCOL,
        expected_metric_identity=IDENTITY,
    )
    extracted["protocol"] = {}

    frame = results_to_dataframe(
        {
            "meta": {
                "dataset_name": "Dataset1",
                "strict_paper_mode": True,
            },
            "results": [extracted],
        }
    )

    for field in (
        "metric_contract_version",
        "smape_definition_id",
        "paper_metric_space_requested",
        "paper_metric_space_actual",
        "primary_metric_space_actual",
        "inverse_transform_status",
        "paper_metric_computed_valid",
        "paper_metric_status",
        "paper_metric_error",
        "metric_sample_count",
        "target_negative_count",
        "metric_index_digest",
    ):
        assert field in frame.columns
        assert frame.loc[0, field] == extracted[field]


def test_d1_d3_formal_runner_builds_and_forwards_expected_metric_identity():
    source = inspect.getsource(d1_d3_runner.run_experiment)

    assert "build_metric_identity_from_manifest(" in source
    assert '"expected_metric_identity": expected_metric_identity' in source
    assert "expected_metric_identity=expected_metric_identity" in source


def test_d1_d3_result_builder_copies_complete_metric_audit_and_identity():
    extracted = _extract_method_metrics(
        _payload(),
        method_name="MSWA-TL",
        metric_protocol=STRICT_PROTOCOL,
        expected_metric_identity=IDENTITY,
    )
    builder = getattr(d1_d3_runner, "_strict_metric_result_fields", None)

    assert callable(builder)
    fields = builder(extracted)
    for field in experiment_runner.METRIC_AUDIT_FIELDS:
        assert field in fields
        assert fields[field] == extracted[field]


@pytest.mark.parametrize(
    ("wrapper_name", "module", "implementation_name", "container"),
    [
        ("run_mswa_experiment", mswa_tl, "run_mswa_tl", "fused_result"),
        ("run_mssb_experiment", mssb_tl, "run_mssb_tl", "final_result"),
        ("run_msml_experiment", msml_tl, "run_msml_tl", "fused_result"),
        ("run_msml_rfe_experiment", msml_tl_rfe, "run_msml_tl_rfe", "fused_result"),
    ],
)
def test_four_multisource_wrappers_forward_orchestration_identity_into_payload(
    monkeypatch,
    wrapper_name,
    module,
    implementation_name,
    container,
):
    received = {}

    def fake_method(**kwargs):
        received.update(kwargs["metric_identity"])
        raw = _payload(container)
        raw[container].update(kwargs["metric_identity"])
        return raw

    monkeypatch.setattr(module, implementation_name, fake_method)
    wrapper = getattr(experiment_runner, wrapper_name)
    result = wrapper(
        source_df=None,
        target_df=None,
        feature_cols=["sales"],
        metric_protocol=STRICT_PROTOCOL,
        expected_metric_identity=IDENTITY,
    )

    assert received == IDENTITY
    assert result["metric_index_digest"] == IDENTITY["metric_index_digest"]


def test_expected_metric_identity_is_derived_from_orchestration_manifest():
    class Record:
        def __init__(self, sample_key, label_date):
            self.sample_key = sample_key
            self.label_date = label_date
            self.target_key = ("target", "item")

    class Manifest:
        def for_horizon(self, horizon):
            assert horizon == 1
            return (
                Record("manifest-key-a", "2024-01-02"),
                Record("manifest-key-b", "2024-01-03"),
            )

    identity = _metric_identity_from_manifest(Manifest(), horizon=1)

    assert identity["metric_target_key"] == "target/item"
    assert identity["metric_horizon"] == 1
    assert identity["metric_sample_count"] == 2
    assert identity["metric_date_start"] == "2024-01-02"
    assert identity["metric_date_end"] == "2024-01-03"
    assert identity["metric_index_digest"]


def test_metric_protocol_failure_serializes_invalid_row_without_finite_primary_metrics():
    exc = MetricProtocolError("missing_scaler", missing_fields=("sales_scaler",))
    row = _row_from_metric_protocol_error(
        exc,
        method="MSWA-TL",
        entity_key="target/item",
        config={
            "dataset_id": 5,
            "dataset_name": "Dataset5",
            "info_sharing": "without",
            "source_count": 3,
            "metric_protocol": STRICT_PROTOCOL,
        },
        elapsed=1.0,
    )

    assert np.isnan(row["rmse"])
    assert np.isnan(row["smape"])
    assert row["paper_metric_computed_valid"] is False
    assert row["paper_metric_status"] == "missing_scaler"
    assert "sales_scaler" in row["paper_metric_error"]
    assert row["result_status"] == "failed"
