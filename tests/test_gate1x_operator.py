from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.operations.materialize_d1_d6_sealed_authority import MaterializationConfig, MaterializationError, materialize


ROOT = Path(__file__).resolve().parents[1]


def test_dry_run_does_not_create_build_or_call_producer(tmp_path: Path) -> None:
    deployment = tmp_path / "deployment"
    result = materialize(MaterializationConfig(project_root=ROOT, deployment_root=deployment, dry_run=True))
    assert result["status"] == "dry_run"
    assert result["writes_performed"] is False
    assert result["producer_calls_performed"] == 0
    assert not deployment.exists()


def test_materialization_requires_passed_readiness_before_private_build(tmp_path: Path) -> None:
    report = tmp_path / "readiness.json"
    report.write_text(json.dumps({"status": "failed"}), encoding="utf-8")
    private = tmp_path / "private"
    with pytest.raises(MaterializationError, match="READINESS_NOT_PASSED"):
        materialize(MaterializationConfig(project_root=ROOT, deployment_root=tmp_path / "deployment", private_build_root=private, readiness_report=report))
    assert not private.exists()
