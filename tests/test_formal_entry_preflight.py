from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.run_unified_d1_d6 as unified
from scripts.validate_d1_d6_protocol_inputs import validate_formal_entry_preflight


def _compliant_preflight(**kwargs):
    report = validate_formal_entry_preflight(run_id=str(kwargs.get("run_id", "fixture")))
    report["status"] = "ready"
    report["failure_codes"] = []
    report["checks"] = {name: True for name in report["checks"]}
    for state in report["dataset_states"]:
        state["state"] = "sealed"
        state["failure_codes"] = []
    return report


def test_repository_old_sealed_root_fails_closed_on_frozen_schema() -> None:
    report = validate_formal_entry_preflight(run_id="contract-test")

    assert report["status"] == "blocked"
    assert len(report["dataset_states"]) == 6
    assert any(
        code.endswith("PREDICTOR_SCHEMA_MISMATCH")
        or code.endswith("KNN_SCHEMA_MISMATCH")
        for code in report["failure_codes"]
    )
    assert report["checks"]["six_dataset_seals"] is False


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


def test_formal_dry_run_resolves_complete_read_only_protocol_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "dry-run-must-not-exist"
    tasks = unified.build_tasks(None, smoke=False, run_dir=run_root)

    calls = []
    monkeypatch.setattr(
        unified,
        "validate_formal_entry_preflight",
        lambda **kwargs: calls.append(kwargs) or _compliant_preflight(**kwargs),
    )
    report = unified.build_formal_dry_run_report(tasks, run_root=run_root)

    assert report["preflight_status"] == "ready"
    assert report["run_root"] == str(run_root.resolve())
    assert report["formal_plan"] == {
        "bundle_count": 60,
        "unique_result_count": 60,
        "mode_count": 12,
        "horizons": [1, 2, 3, 4, 5],
        "seeds": [42, 43, 44, 45, 46],
    }
    assert report["scheduler_contract"] == {
        "d5_dependency": ["d5_without", "d5_with"],
        "d5_threads": 6,
        "ordinary_threads": 2,
        "total_thread_budget": 16,
    }
    datasets = report["datasets"]
    assert [item["dataset_id"] for item in datasets] == [f"D{i}" for i in range(1, 7)]
    assert [item["provenance_level"] for item in datasets[:2]] == [
        "raw_rebuilt",
        "raw_rebuilt",
    ]
    assert {item["provenance_level"] for item in datasets[2:]} == {
        "adopted_solidified"
    }
    for dataset in datasets:
        assert dataset["identity"]["source"]["sha256"]
        assert dataset["identity"]["target"]["sha256"]
        assert dataset["windows"]["target"]["blind_start"]
        assert dataset["windows"]["source"]["pretrain_start"]
        assert dataset["predictor_schema"]["fields"]
        assert dataset["knn_schema"]["fields"]
        assert "source_sales_repair" in dataset
    assert len(report["mode_cache_identities"]) == 12
    assert len(set(report["mode_cache_identities"].values())) == 12
    for dataset in datasets:
        dataset_id = int(dataset["dataset_id"][1:])
        target = dataset["identity"]["target"]
        relative_target = str(
            Path(target["path"]).relative_to(unified.PROJECT_ROOT)
        )
        input_identity = {
            relative_target: {
                "bytes": target["bytes"],
                "sha256": target["sha256"],
            }
        }
        for mode in unified.VALID_MODES:
            task = next(
                item
                for item in tasks
                if item.dataset_id == dataset_id and item.scenario == mode
            )
            pinned = unified._pin_task_identities(
                unified._task_plan_entry(task),
                input_identity=input_identity,
            )
            assert report["mode_cache_identities"][f"d{dataset_id}_{mode}"] == (
                pinned["mode_cache_identity"]["digest"]
            )
    assert report["schema_digests"]["artifact_registry"]
    assert report["schema_digests"]["result_registry"]
    assert not run_root.exists()
    assert calls == [{"run_id": run_root.name}]


def test_print_formal_dry_run_is_canonical_json_and_creates_no_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "dry-run-must-not-exist"
    tasks = unified.build_tasks(None, smoke=False, run_dir=run_root)

    monkeypatch.setattr(unified, "validate_formal_entry_preflight", _compliant_preflight)
    unified.print_dry_run(tasks, run_root=run_root)

    output = capsys.readouterr().out
    summary_line = next(
        line for line in output.splitlines() if line.startswith("[FORMAL DRY-RUN] ")
    )
    summary = json.loads(summary_line.removeprefix("[FORMAL DRY-RUN] "))
    assert summary["preflight_status"] == "ready"
    assert summary["formal_plan"]["bundle_count"] == 60
    assert not run_root.exists()


def test_readme_marks_authoritative_and_nonsealed_compatibility_paths() -> None:
    readme = (unified.PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "60 个 seed bundle" in readme
    assert "旧 300-cell run plan" in readme
    assert "append-only attempts" in readme
    assert "resume_lease_conflict" in readme
    assert "rehydrate" in readme
    assert "SEALED_SUCCESS" in readme
    assert "非封存兼容路径" in readme
