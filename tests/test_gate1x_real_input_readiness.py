from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

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
    deployment_preflight = {
        "manifest": {"datasets": entries, "formal_identity": identity},
        "proofs": proofs,
    }

    def unexpected_manifest_revalidation(repository_root):
        raise AssertionError("deployment manifest was revalidated instead of reused")

    monkeypatch.setattr(
        deployment,
        "validate_deployment_manifest",
        unexpected_manifest_revalidation,
    )
    assert readiness.readiness_proof_digest is deployment.readiness_proof_digest

    accepted = readiness.run_readiness(
        root=root,
        require_deployment=True,
        deployment_preflight=deployment_preflight,
    )
    assert accepted["status"] == "passed"
    assert accepted["preflight_status"] == "ready"

    mutate = True
    rejected = readiness.run_readiness(
        root=root,
        require_deployment=True,
        deployment_preflight=deployment_preflight,
    )
    assert rejected["status"] == "failed"
    assert rejected["failure_code"] == "READINESS_PROOF_MISMATCH"


def test_d6_readiness_uses_sealed_target_without_raw_calendar_remerge(monkeypatch, tmp_path: Path) -> None:
    calendar_path = tmp_path / "数据集" / "原始数据" / "Dataset 6m5-forecasting-accuracy" / "calendar.csv"
    calendar_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": ["2015-01-01"],
            "weekday": ["Thursday"],
            "wday": [5],
            "wm_yr_wk": [1],
            "event_name_1": ["raw-event"],
            "event_type_1": ["Holiday"],
            "event_name_2": [""],
            "event_type_2": [""],
            "snap_CA": [1],
        }
    ).to_csv(calendar_path, index=False)

    spec = readiness.dataset_contract("D6")
    target = pd.DataFrame(
        {
            **{
                field: [value]
                for field, value in zip(spec.key_fields, spec.target_keys[0])
            },
            "date": [pd.Timestamp("2015-01-01")],
            "sales": [0.0],
            "wm_yr_wk": [1],
            "weekday": ["sealed-weekday"],
            "wday": [5],
            "event_name_1": ["sealed-event"],
            "event_type_1": ["Sealed"],
            "event_name_2": [""],
            "event_type_2": [""],
            "snap": [0],
        }
    )
    seen_columns: list[str] = []

    def capture_target(frame: pd.DataFrame, _spec: object) -> tuple[dict[str, int], list[dict[str, object]], int]:
        seen_columns.extend(frame.columns)
        return {}, [], spec.expected_blind_rows

    monkeypatch.setattr(
        readiness,
        "_base_report",
        lambda *args, **kwargs: {
            "dataset": "D6",
            "status": "failed",
            "failure_code": None,
            "worker_safe_fields": [],
            "proof_inputs_available": {},
            "after_slicing_rows": {},
            "pre_or_equal_origin_forecast_rows": 0,
            "cardinality": {},
            "evaluator_truth_fields": [],
            "audit_fields": [],
            "schema_fields": {},
            "field_exclusions": {},
            "source_entities": [],
            "post_origin_history_rows": 0,
        },
    )
    monkeypatch.setattr(readiness, "_target_frame", lambda root, dataset: target.copy())
    monkeypatch.setattr(
        readiness,
        "formal_input_paths",
        lambda root, dataset: {
            "source": tmp_path / "sealed" / "source.parquet",
            "target": tmp_path / "sealed" / "target.parquet",
        },
    )
    monkeypatch.setattr(readiness, "_calendarize_target_counts", capture_target)
    monkeypatch.setattr(
        readiness,
        "evaluate_formal_target_calendar",
        lambda frame, dataset_id: {
            "actual": 1,
            "expected": 1,
            "ready": True,
            "missing_exact_keys": [],
            "extra_dates": [],
            "unexpected_keys": [],
        },
    )
    monkeypatch.setattr(readiness, "_source_frame", lambda root, dataset: None)
    monkeypatch.setattr(
        readiness,
        "stream_source_history_candidates",
        lambda *args, **kwargs: {"complete_candidate_keys": [], "post_origin_history_rows": 0},
    )
    monkeypatch.setattr(readiness, "_verify_runtime_knn_authority", lambda **kwargs: {})

    readiness._dataset_report(tmp_path, tmp_path, tmp_path, 6, {})

    assert seen_columns == list(target.columns)
    assert not any(column.endswith(("_x", "_y")) for column in seen_columns)
