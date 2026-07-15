from __future__ import annotations

import pandas as pd
import pytest

from src.utils.mode_cache import (
    CacheIsolationError,
    EvaluatorControlPlane,
    WorkerRunLayout,
    build_worker_manifest,
    create_evaluator_cache,
    create_worker_cache,
)


def test_worker_and_evaluator_caches_are_physically_separate(tmp_path) -> None:
    worker = create_worker_cache(tmp_path, {"observed_model_frame": pd.DataFrame({"sales": [1.0]})})
    evaluator, control = create_evaluator_cache(
        tmp_path, pd.DataFrame({"y_true": [1.0]}), return_control_plane=True
    )

    assert worker.root != evaluator.root
    assert worker.root.parent == tmp_path / "worker_cache"
    assert evaluator.root.parent == tmp_path / "evaluator_cache"
    assert worker.root.exists() and evaluator.root.exists()
    with pytest.raises(CacheIsolationError):
        worker.get_view("evaluator_truth_frame")
    assert control.resolve(evaluator.capability_id) is evaluator


def test_worker_layout_and_manifest_have_no_truth_reconstruction_fields(tmp_path) -> None:
    layout = WorkerRunLayout(
        run_root=tmp_path,
        run_id="run-1",
        dataset_id="D1",
        scenario="without",
        cell_id="D1/without/No-TL/42",
        attempt_id="attempt-1",
    )
    manifest = build_worker_manifest(
        layout,
        method="No-TL",
        seed=42,
        schema_name="WorkerPredictionTraceSchemaV1",
        schema_version="v1",
        schema_digest="a" * 64,
        artifact_path="worker_trace.csv.gz",
        row_count=0,
        canonical_content_sha256="b" * 64,
        artifact_sha256="c" * 64,
        semantic_prediction_digest="d" * 64,
        status="accepted",
    )

    lowered = {str(key).lower() for key in manifest}
    assert not any(
        token in key
        for key in lowered
        for token in ("truth", "evaluator", "capability", "template", "reconstruct")
    )
    assert "worker_cache" in layout.worker_cache_dir.parts
