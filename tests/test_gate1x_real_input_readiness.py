from __future__ import annotations

import json
from pathlib import Path

import tools.operations.gate1x_real_input_readiness as readiness
from src.protocols import formal_deployment_manifest as deployment


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


def test_runtime_readiness_uses_shared_digest_and_fails_closed_on_identity_change(
    monkeypatch, tmp_path: Path
) -> None:
    root = tmp_path / "machine-a" / "project"
    other_root = tmp_path / "machine-b" / "project"
    root.mkdir(parents=True)
    other_root.mkdir(parents=True)
    identity = {"combined_formal_identity_digest": "formal-identity"}
    mutate = False

    def fake_item(repository_root: Path, dataset_id: int) -> dict[str, object]:
        item: dict[str, object] = {
            "dataset": f"D{dataset_id}",
            "status": "passed",
            "failure_code": None,
            "formal_input": {
                "source_path": str(repository_root / "数据集" / "固化数据" / f"dataset{dataset_id}" / "source.parquet"),
                "target_path": str(repository_root / "数据集" / "固化数据" / f"dataset{dataset_id}" / "target.parquet"),
                "source_sha256": "a" * 64,
            },
            "raw_inputs": [{"path": str(repository_root / "数据集" / "原始数据" / "input.csv")}],
        }
        if mutate and dataset_id == 3:
            item["formal_input"]["source_sha256"] = "b" * 64  # type: ignore[index]
        return item

    def fake_report(root, parent_root, old_root, dataset_id, identity=None):
        return fake_item(Path(root), dataset_id)

    entries = {
        f"D{dataset_id}": {
            "source": {"path": f"dataset{dataset_id}/source.parquet", "sha256": "a" * 64, "size_bytes": 1},
            "target": {"path": f"dataset{dataset_id}/target.parquet", "sha256": "b" * 64, "size_bytes": 1},
            "source_schema_digest": "c" * 64,
            "target_schema_digest": "d" * 64,
            "consumer_fingerprint": "e" * 64,
        }
        for dataset_id in range(1, 7)
    }
    proofs = {
        f"D{dataset_id}": {
            "readiness_proof_digest": deployment.readiness_proof_digest(
                fake_item(other_root, dataset_id), repository_root=other_root
            ),
            "proof_identity_sha256": "f" * 64,
            "formal_identity": identity,
            "consumer_fingerprint": entries[f"D{dataset_id}"]["consumer_fingerprint"],
        }
        for dataset_id in range(1, 7)
    }

    monkeypatch.setattr(readiness, "_dataset_report", fake_report)
    monkeypatch.setattr(readiness, "_source_history_static_audit", lambda root: {"status": "passed"})
    monkeypatch.setattr(readiness, "load_formal_identity", lambda root: identity)
    monkeypatch.setattr(
        deployment,
        "validate_deployment_manifest",
        lambda repository_root: {
            "manifest": {"datasets": entries, "formal_identity": identity},
            "proofs": proofs,
        },
    )
    assert readiness.readiness_proof_digest is deployment.readiness_proof_digest

    accepted = readiness.run_readiness(root=root, require_deployment=True)
    assert accepted["status"] == "passed"
    assert accepted["preflight_status"] == "ready"

    mutate = True
    rejected = readiness.run_readiness(root=root, require_deployment=True)
    assert rejected["status"] == "failed"
    assert rejected["failure_code"] == "READINESS_PROOF_MISMATCH"
