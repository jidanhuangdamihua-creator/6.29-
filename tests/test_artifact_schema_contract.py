from __future__ import annotations

from datetime import date

import pytest

from src.protocols.artifact_schemas import (
    ARTIFACT_SCHEMA_REGISTRY_VERSION,
    PREDICTION_SEMANTIC_SORT_KEY,
    SchemaDefinitionDriftError,
    SchemaValidationError,
    UnknownSchemaError,
    get_artifact_schema_registry,
    get_artifact_schema,
    validate_schema_reference,
    validate_run_schema_versions,
)


EXPECTED_WORKER_FIELDS = [
    "run_id",
    "cell_id",
    "attempt_id",
    "dataset_id",
    "scenario",
    "target_entity_key",
    "method",
    "seed",
    "rollout_stream_key",
    "forecast_origin",
    "label_date",
    "horizon",
    "truth_key",
    "sample_key",
    "prediction_row_key",
    "y_pred_raw",
    "y_pred_clipped",
    "was_clipped",
    "history_snapshot_digest",
    "history_after_h1_commit_digest",
    "input_digest",
    "prediction_policy_id",
    "predictor_feature_schema_digest",
    "feature_mask_digest",
]


def test_registry_contains_only_the_registered_v1_artifact_types() -> None:
    registry = get_artifact_schema_registry()
    assert ARTIFACT_SCHEMA_REGISTRY_VERSION == "artifact_schema_registry_v1"
    assert registry.schema_names == (
        "CellResultManifestSchemaV1",
        "EvaluatedPredictionTraceSchemaV1",
        "FormalResultRowSchemaV1",
        "PreflightReportSchemaV1",
        "RunManifestSchemaV1",
        "SourceSelectionTraceSchemaV1",
        "WorkerManifestSchemaV1",
        "WorkerPredictionTraceSchemaV1",
    )
    assert all(registry.get(name).schema_version == "v1" for name in registry.schema_names)
    assert {
        name: registry.get(name).schema_digest
        for name in registry.schema_names
    } == {
        "CellResultManifestSchemaV1": "sha256:f2d156cce0eee3570da60f104e242a913536740b3e45e4f2e020ea9ea9405643",
        "EvaluatedPredictionTraceSchemaV1": "sha256:790ac478367e1f4670afde1758b4fc312743d4f1a22836332da86cc1386f3cd6",
        "FormalResultRowSchemaV1": "sha256:d8e3eadab5dd5e6ee9b696cc762e1c25cdb1fd27d8141be51e6b47e6f308907b",
        "PreflightReportSchemaV1": "sha256:1f7717c75d0237b5a15c870df8945ed25bfdf972927cf17c197324d61a98e7b0",
        "RunManifestSchemaV1": "sha256:8e360df45d5abdb05304099b3408a822d608cd1e88e55b6950e4894e733cfde1",
        "SourceSelectionTraceSchemaV1": "sha256:0befdfff05e1937aa7c3eed3d22cb692e6bdeab310e08718c0d61535cbf3e2fb",
        "WorkerManifestSchemaV1": "sha256:5bb08013d82c05fcd9670bd5d659c1edb5bf6a09458a7031518e69614433b689",
        "WorkerPredictionTraceSchemaV1": "sha256:34a7230d9f963f8ae13ea6021c07abad3ca8d92d28f11caea0cca3c4331e423c",
    }


def test_worker_trace_field_order_and_semantic_sort_key_are_frozen() -> None:
    schema = get_artifact_schema("WorkerPredictionTraceSchemaV1")
    assert list(schema.field_names) == EXPECTED_WORKER_FIELDS
    assert list(get_artifact_schema("EvaluatedPredictionTraceSchemaV1").field_names) == (
        EXPECTED_WORKER_FIELDS + ["y_true", "is_synthetic_date", "evaluator_join_status"]
    )
    assert schema.semantic_sort_key == PREDICTION_SEMANTIC_SORT_KEY
    assert schema.additional_properties is False
    assert schema.primary_key == ("prediction_row_key",)


def test_worker_schema_rejects_y_true_extra_fields_reordered_fields_and_bad_types() -> None:
    schema = get_artifact_schema("WorkerPredictionTraceSchemaV1")
    row = _worker_row()

    with pytest.raises(SchemaValidationError, match="unknown field|exact field order"):
        schema.validate_record({**row, "y_true": 1.0})

    reordered = dict(reversed(list(row.items())))
    with pytest.raises(SchemaValidationError, match="exact field order"):
        schema.validate_record(reordered)

    bad = dict(row)
    bad["horizon"] = "1"
    with pytest.raises(SchemaValidationError, match="horizon"):
        schema.validate_record(bad)


def test_exact_reader_tuple_and_same_version_digest_drift_fail_closed() -> None:
    schema = get_artifact_schema("WorkerPredictionTraceSchemaV1")
    assert validate_schema_reference(
        schema.schema_name, schema.schema_version, schema.schema_digest
    ) is schema

    with pytest.raises(UnknownSchemaError):
        validate_schema_reference(schema.schema_name, "v9", schema.schema_digest)
    with pytest.raises(SchemaDefinitionDriftError):
        validate_schema_reference(schema.schema_name, schema.schema_version, "sha256:" + "0" * 64)


def test_one_run_cannot_mix_schema_versions_for_one_artifact_type() -> None:
    schema = get_artifact_schema("WorkerPredictionTraceSchemaV1")
    references = [
        {
            "schema_name": schema.schema_name,
            "schema_version": schema.schema_version,
            "schema_digest": schema.schema_digest,
        },
        {
            "schema_name": schema.schema_name,
            "schema_version": "v2",
            "schema_digest": schema.schema_digest,
        },
    ]
    with pytest.raises(UnknownSchemaError):
        validate_run_schema_versions(references)


def _worker_row() -> dict[str, object]:
    digest = "a" * 64
    return {
        "run_id": "run-1",
        "cell_id": "D1/without/No-TL/42",
        "attempt_id": "attempt-1",
        "dataset_id": "D1",
        "scenario": "without",
        "target_entity_key": "1_10",
        "method": "No-TL",
        "seed": 42,
        "rollout_stream_key": digest,
        "forecast_origin": date(2017, 7, 1),
        "label_date": date(2017, 7, 2),
        "horizon": 1,
        "truth_key": digest,
        "sample_key": "b" * 64,
        "prediction_row_key": "c" * 64,
        "y_pred_raw": 1.25,
        "y_pred_clipped": 1.25,
        "was_clipped": False,
        "history_snapshot_digest": "d" * 64,
        "history_after_h1_commit_digest": "e" * 64,
        "input_digest": "f" * 64,
        "prediction_policy_id": "clipped_h1_v1",
        "predictor_feature_schema_digest": "1" * 64,
        "feature_mask_digest": "2" * 64,
    }
