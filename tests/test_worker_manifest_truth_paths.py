from __future__ import annotations

import pytest

from src.protocols.artifact_schemas import SchemaValidationError
from src.utils.mode_cache import (
    WorkerRunLayout,
    build_worker_manifest,
)


def test_worker_manifest_rejects_truth_like_artifact_paths(tmp_path) -> None:
    layout = WorkerRunLayout(
        run_root=tmp_path,
        run_id="run-1",
        dataset_id="D1",
        scenario="without",
        cell_id="D1/without/No-TL/42",
        attempt_id="attempt-1",
    )

    with pytest.raises(SchemaValidationError, match="path"):
        build_worker_manifest(
            layout,
            method="No-TL",
            seed=42,
            schema_name="WorkerPredictionTraceSchemaV1",
            schema_version="v1",
            schema_digest="a" * 64,
            artifact_path="truth/worker_trace.csv.gz",
            row_count=0,
            canonical_content_sha256="b" * 64,
            artifact_sha256="c" * 64,
            semantic_prediction_digest="d" * 64,
            status="accepted",
        )
