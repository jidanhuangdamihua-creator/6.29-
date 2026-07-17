from __future__ import annotations

import json
from pathlib import Path

import tools.operations.gate1x_real_input_readiness as readiness


ROOT = Path(__file__).resolve().parents[1]


def test_readiness_is_hard_no_write_and_checks_all_six_datasets(monkeypatch, tmp_path: Path) -> None:
    calls: list[int] = []

    def fake_report(root, parent_root, old_root, dataset_id, identity=None):
        calls.append(dataset_id)
        return {"dataset": f"D{dataset_id}", "status": "passed", "failure_code": None}

    monkeypatch.setattr(readiness, "_dataset_report", fake_report)
    monkeypatch.setattr(readiness, "load_formal_identity", lambda root: {"combined_formal_identity_digest": readiness.COMBINED_FORMAL_IDENTITY_DIGEST})
    report = readiness.run_readiness(root=ROOT, parent_root=tmp_path / "parent", old_sealed_root=tmp_path / "old")
    assert calls == [1, 2, 3, 4, 5, 6]
    assert report["status"] == "passed"
    assert report["read_only"] is True
    assert report["writes_performed"] is False
    assert report["producer_calls_performed"] == 0
    assert report["private_build_created"] is False
    assert report["deployment_created"] is False
    assert not (tmp_path / "parent").exists()
    assert not (tmp_path / "old").exists()


def test_failed_or_unknown_readiness_is_nonzero_and_json_serializable(monkeypatch, capsys, tmp_path: Path) -> None:
    def fake_report(root, parent_root, old_root, dataset_id, identity=None):
        return {"dataset": f"D{dataset_id}", "status": "failed" if dataset_id == 5 else "passed", "failure_code": "UNKNOWN" if dataset_id == 5 else None}

    monkeypatch.setattr(readiness, "_dataset_report", fake_report)
    monkeypatch.setattr(readiness, "load_formal_identity", lambda root: {"combined_formal_identity_digest": readiness.COMBINED_FORMAL_IDENTITY_DIGEST})
    report = readiness.run_readiness(root=ROOT, parent_root=tmp_path / "parent", old_sealed_root=tmp_path / "old")
    encoded = json.dumps(report, ensure_ascii=False)
    assert json.loads(encoded)["status"] == "failed"
    assert report["failure_code"] == "UNKNOWN"
