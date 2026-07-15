from __future__ import annotations

import gzip
import json
from datetime import date

import pytest

from src.protocols.artifact_schemas import (
    SchemaValidationError,
    get_artifact_schema,
)
from src.utils.prediction_artifacts import (
    ArtifactPublicationError,
    canonical_content_sha256,
    join_worker_trace_with_truth,
    publish_prediction_artifact,
    read_prediction_artifact,
    semantic_prediction_sha256,
    validate_worker_trace,
)


def test_row_permutations_have_same_semantic_digest_but_distinct_physical_bytes(tmp_path) -> None:
    rows = [_worker_row(2), _worker_row(1)]
    reversed_rows = list(reversed(rows))
    schema = get_artifact_schema("WorkerPredictionTraceSchemaV1")

    assert semantic_prediction_sha256(rows) == semantic_prediction_sha256(reversed_rows)
    assert canonical_content_sha256(rows) == canonical_content_sha256(reversed_rows)

    first = publish_prediction_artifact(rows, tmp_path / "first.csv.gz", run_root=tmp_path)
    second = publish_prediction_artifact(reversed_rows, tmp_path / "second.csv.gz", run_root=tmp_path)
    assert first.artifact_sha256 == second.artifact_sha256
    assert first.semantic_prediction_digest == second.semantic_prediction_digest
    assert first.schema_digest == schema.schema_digest


def test_publication_uses_canonical_json_and_fixed_gzip_header(tmp_path) -> None:
    path = tmp_path / "trace.csv.gz"
    identity = publish_prediction_artifact([_worker_row(1)], path, run_root=tmp_path)
    raw = path.read_bytes()
    assert raw[:2] == b"\x1f\x8b"
    assert gzip.decompress(raw).decode("utf-8").endswith("\n")
    assert identity.artifact_sha256.startswith("sha256:")
    descriptor_path = tmp_path / "schemas" / (identity.schema_digest + ".json")
    assert descriptor_path.is_file()
    assert json.loads(descriptor_path.read_text(encoding="utf-8"))["schema_name"] == (
        "WorkerPredictionTraceSchemaV1"
    )


def test_read_revalidates_physical_and_logical_digests(tmp_path) -> None:
    path = tmp_path / "trace.csv.gz"
    identity = publish_prediction_artifact([_worker_row(1)], path)
    rows = read_prediction_artifact(path, expected=identity)
    assert rows[0]["horizon"] == 1

    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ArtifactPublicationError, match="artifact SHA-256"):
        read_prediction_artifact(path, expected=identity)


def test_evaluator_join_is_one_to_one_and_worker_validation_has_no_truth() -> None:
    worker = [_worker_row(1)]
    validate_worker_trace(worker)
    truth = [
        {
            "truth_key": "a" * 64,
            "label_date": date(2017, 7, 2),
            "target_entity_key": "1_10",
            "y_true": 2.0,
            "is_synthetic_date": False,
        }
    ]
    evaluated = join_worker_trace_with_truth(worker, truth)
    assert list(evaluated[0])[-3:] == ["y_true", "is_synthetic_date", "evaluator_join_status"]
    assert evaluated[0]["evaluator_join_status"] == "matched"

    with pytest.raises(SchemaValidationError, match="one-to-one|duplicate"):
        join_worker_trace_with_truth(worker, truth + truth)


def _worker_row(index: int) -> dict[str, object]:
    digest = "a" * 64
    row = {
        "run_id": "run-1",
        "cell_id": "D1/without/No-TL/42",
        "attempt_id": "attempt-1",
        "dataset_id": "D1",
        "scenario": "without",
        "target_entity_key": "1_10",
        "method": "No-TL",
        "seed": 42,
        "rollout_stream_key": digest,
        "forecast_origin": date(2017, 7, index),
        "label_date": date(2017, 7, index + 1),
        "horizon": 1,
        "truth_key": digest,
        "sample_key": ("b" * 63) + str(index),
        "prediction_row_key": ("c" * 63) + str(index),
        "y_pred_raw": float(index),
        "y_pred_clipped": float(index),
        "was_clipped": False,
        "history_snapshot_digest": "d" * 64,
        "history_after_h1_commit_digest": "e" * 64,
        "input_digest": "f" * 64,
        "prediction_policy_id": "clipped_h1_v1",
        "predictor_feature_schema_digest": "1" * 64,
        "feature_mask_digest": "2" * 64,
    }
    return row
