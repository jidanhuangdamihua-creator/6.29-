from __future__ import annotations

from pathlib import Path

import pytest

import scripts.run_unified_d1_d6 as unified
from scripts.validate_d1_d6_protocol_inputs import validate_formal_entry_preflight


def test_repository_sealed_root_passes_layered_preflight() -> None:
    report = validate_formal_entry_preflight(run_id="contract-test")

    assert report["status"] == "ready"
    assert len(report["dataset_states"]) == 6
    assert {item["state"] for item in report["dataset_states"]} == {"sealed"}
    assert report["failure_codes"] == []


def test_missing_dataset_blocks_globally_but_preserves_per_dataset_evidence(
    tmp_path: Path,
) -> None:
    report = validate_formal_entry_preflight(tmp_path, run_id="blocked")

    assert report["status"] == "blocked"
    assert len(report["dataset_states"]) == 6
    assert all(item["report_path"].endswith("validation_report.json") for item in report["dataset_states"])
    assert "D1:SEALED_ARTIFACT_MISSING" in report["failure_codes"]
    assert report["checks"]["six_dataset_seals"] is False


def test_blocked_preflight_creates_neither_run_directory_nor_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "formal-run"
    monkeypatch.setattr(
        unified,
        "validate_formal_entry_preflight",
        lambda **_kwargs: {
            "status": "blocked",
            "dataset_states": [{"dataset_id": "D2", "failure_codes": ["KNN_FINGERPRINT_MISMATCH"]}],
            "failure_codes": ["D2:KNN_FINGERPRINT_MISMATCH"],
        },
    )

    with pytest.raises(RuntimeError, match="blocked before attempt creation"):
        unified.prepare_formal_run(run_root, resume=False)

    assert not run_root.exists()
