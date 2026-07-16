from __future__ import annotations

from copy import deepcopy
import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import pandas as pd

import scripts.adopt_and_seal_d3_d6 as adoption
from src.protocols.feature_schema import get_knn_schema, get_predictor_schema
from src.protocols.gate1_transformation import canonical_digest
from src.protocols.sealing_protocol import get_source_pretrain_window
from tools.operations import materialize_d1_d6_sealed_authority as operator


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "operations" / "materialize_d1_d6_sealed_authority.py"


def _proof() -> dict:
    return {
        "version": "source-sales-canonicalization/v1",
        "status": "canonicalized",
        "rows_examined": 2,
        "affected_rows": [],
        "repair_reason_counts": {
            "original_nan": 0,
            "original_negative": 0,
            "calendar_row_missing": 0,
        },
        "repair_mask_sha256": "a" * 64,
        "affected_date_digest": "sha256:" + "b" * 64,
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _frame(dataset_id: int, *, target: bool) -> pd.DataFrame:
    if target:
        window = get_source_pretrain_window(dataset_id)
        dates = pd.date_range(window.pretrain_end + pd.Timedelta(days=1), periods=3, freq="D")
    else:
        window = get_source_pretrain_window(dataset_id)
        dates = pd.date_range(window.pretrain_start, window.pretrain_end, freq="D")
    size = len(dates)
    frame = pd.DataFrame(
        {
            "entity_id": ["10" if target else "1"] * size,
            "store_id": [10 if target else 1] * size,
            "product_id": [1] * size,
            "store_nbr": [10 if target else 1] * size,
            "item_nbr": [1] * size,
            "item_id": [1] * size,
            "date": dates,
            "sales": [float(dataset_id)] * size,
        }
    )
    if dataset_id == 3:
        frame["SchoolHoliday"] = 0
    elif dataset_id == 4:
        frame["activity_flag"] = 1
        frame["discount"] = 0.0
        frame["holiday_flag"] = 0
        frame["precpt"] = 0.0
        frame["avg_temperature"] = 20.0
        frame["avg_humidity"] = 50.0
        frame["avg_wind_level"] = 1.0
    elif dataset_id == 5:
        frame["perishable"] = 1
        frame["onpromotion"] = 0
        frame["oil_price"] = 50.0
        frame["is_holiday"] = 0
    elif dataset_id == 6:
        frame["weekday"] = "Monday"
        frame["wday"] = 1
        frame["wm_yr_wk"] = 1
        frame["event_name_1"] = "none"
        frame["event_type_1"] = "none"
        frame["event_name_2"] = "none"
        frame["event_type_2"] = "none"
        frame["snap"] = 0
        frame["sell_price"] = 1.0
    return frame


def _make_dataset(root: Path, dataset_id: int) -> None:
    dataset = root / f"dataset{dataset_id}"
    dataset.mkdir()
    proof = _proof()
    manifest = {
        "dataset_id": f"D{dataset_id}",
        "source_sales_repair": proof,
        "source_sales_repair_mask_sha256": proof["repair_mask_sha256"],
        "source_sales_repair_reason_counts": proof["repair_reason_counts"],
        "artifacts": {
            "source": {"path": "source.parquet"},
            "target": {"path": "target.parquet"},
        },
    }
    for name in operator._artifact_names(dataset_id):
        path = dataset / name
        if name == "source.parquet":
            _frame(dataset_id, target=False).to_parquet(path, index=False)
        elif name == "target.parquet":
            _frame(dataset_id, target=True).to_parquet(path, index=False)
        elif name == "manifest.json":
            _write_json(path, manifest)
        elif name == "source_sales_canonicalization.json":
            _write_json(path, proof)
        elif name == "adopt_validation_report.json":
            _write_json(path, {"source_sales_repair": proof})
        elif name == "predictor_schema.json":
            _write_json(path, get_predictor_schema(f"D{dataset_id}").descriptor())
        elif name == "knn_schema.json":
            _write_json(path, get_knn_schema(f"D{dataset_id}").descriptor())
        else:
            _write_json(path, {"dataset_id": f"D{dataset_id}", "artifact": name})


def _fixture(tmp_path: Path) -> tuple[operator.MaterializationConfig, Path, Path]:
    old = tmp_path / "old"
    parent = tmp_path / "parents"
    deploy = tmp_path / "deployments"
    reports = tmp_path / "reports"
    outputs = tmp_path / "outputs-runs"
    for path in (old, parent, deploy, reports, outputs):
        path.mkdir()
    for dataset_id in range(1, 7):
        _make_dataset(old, dataset_id)
    for dataset_id in range(3, 7):
        shutil.copy2(old / f"dataset{dataset_id}" / "source.parquet", parent / f"dataset{dataset_id}-source.parquet")
        shutil.copy2(old / f"dataset{dataset_id}" / "target.parquet", parent / f"dataset{dataset_id}-target.parquet")
    run_id = "test-run-0001"
    config = operator.MaterializationConfig(
        old_sealed_root=old,
        parent_root=parent,
        private_build_root=deploy / f".private-build-{run_id}",
        final_deployment_parent=deploy,
        report_output=reports / "execution-report.json",
        manifest_candidate_output=reports / "manifest-candidate.json",
        run_id=run_id,
    )
    return config, old, outputs


def _copying_producer(old: Path, calls: list[int], *, fail_at: int | None = None):
    def run(dataset_id: int, parent_root: Path, output_root: Path):
        calls.append(dataset_id)
        if fail_at == dataset_id:
            return subprocess.CompletedProcess([], 7, "", "fixture producer failed")
        adoption.adopt_and_seal_dataset(
            dataset_id,
            parent_root=parent_root,
            output_dir=output_root,
        )
        return subprocess.CompletedProcess([], 0, f"sealed D{dataset_id}\n", "")

    return run


def _mutating_producer(
    old: Path,
    calls: list[int],
    *,
    dataset_to_mutate: int,
    mutate,
):
    def run(dataset_id: int, parent_root: Path, output_root: Path):
        completed = _copying_producer(old, calls)(dataset_id, parent_root, output_root)
        if dataset_id != dataset_to_mutate:
            return completed
        directory = output_root / f"dataset{dataset_id}"
        manifest_path = directory / "manifest.json"
        report_path = directory / "adopt_validation_report.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        proof = deepcopy(manifest["gate1_publication_proof"])
        mutate(proof, manifest, directory)
        if "proof_digest" in proof:
            proof["proof_digest"] = canonical_digest(
                {key: value for key, value in proof.items() if key != "proof_digest"}
            )
        manifest["gate1_publication_proof"] = proof
        report["gate1_publication_proof"] = proof
        _write_json(manifest_path, manifest)
        _write_json(report_path, report)
        return completed

    return run


@pytest.fixture(autouse=True)
def _stub_frozen_raw_authority(monkeypatch: pytest.MonkeyPatch) -> None:
    def verified(dataset_id: int) -> tuple[dict, ...]:
        return tuple(
            {"path": record["path"], "sha256": record["sha256"], "size_bytes": 1}
            for record in operator._frozen_raw_records(dataset_id)
        )

    def identity(dataset_id: int) -> dict:
        records = operator._frozen_raw_records(dataset_id)
        files = [
            {
                "name": f"D{dataset_id}:{record['path']}",
                "path": record["path"],
                "size_bytes": 1,
                "sha256": record["sha256"],
            }
            for record in records
        ]
        return {
            "dataset": f"D{dataset_id}",
            "files": files,
            "approved_input_set_digest": canonical_digest(files),
            "snapshot_identity": canonical_digest(
                [(item["path"], item["sha256"]) for item in files]
            ),
            "verified_from_bytes": True,
        }

    monkeypatch.setattr(adoption, "_raw_authority_identity", identity)
    monkeypatch.setattr(operator, "_verify_frozen_raw_authority_bytes", verified)


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _final_root(config: operator.MaterializationConfig) -> Path:
    report = json.loads(config.report_output.read_text(encoding="utf-8"))
    return Path(report["final_root"])


def test_cli_help_lists_only_fixed_arguments() -> None:
    completed = subprocess.run(
        [sys.executable, str(TOOL), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    help_text = completed.stdout
    for option in (
        "--old-sealed-root",
        "--parent-root",
        "--private-build-root",
        "--final-deployment-parent",
        "--report-output",
        "--manifest-candidate-output",
        "--run-id",
        "--dry-run",
    ):
        assert option in help_text
    for forbidden in (
        "--timeout",
        "--entity",
        "--batch",
        "--date",
        "--legacy",
        "--smoke",
        "--fallback",
    ):
        assert forbidden not in help_text


def test_success_preserves_old_root_copies_d1_d2_and_calls_each_dataset_once(
    tmp_path: Path,
) -> None:
    config, old, outputs = _fixture(tmp_path)
    before = _tree_bytes(old)
    (outputs / "sentinel.txt").write_bytes(b"unchanged-run-output")
    outputs_before = _tree_bytes(outputs)
    calls: list[int] = []

    def producer(dataset_id: int, parent_root: Path, output_root: Path):
        assert not list(config.final_deployment_parent.glob("d1_d6_sealed_v1_deploy_*"))
        return _copying_producer(old, calls)(dataset_id, parent_root, output_root)

    status = operator.materialize(
        config,
        producer_runner=producer,
        outputs_run_root=outputs,
    )

    assert status == 0
    assert calls == [3, 4, 5, 6]
    assert _tree_bytes(old) == before
    assert _tree_bytes(outputs) == outputs_before
    final = _final_root(config)
    assert final.is_dir()
    assert not config.private_build_root.exists()
    for dataset_id in (1, 2):
        for name in operator._artifact_names(dataset_id):
            assert (final / f"dataset{dataset_id}" / name).read_bytes() == (
                old / f"dataset{dataset_id}" / name
            ).read_bytes()
    manifest = json.loads(config.manifest_candidate_output.read_text(encoding="utf-8"))
    assert sum(len(items) for items in manifest["datasets"].values()) == 70
    assert set(manifest) == {
        "manifest_version",
        "sealed_root_version",
        "deployment_root",
        "run_id",
        "contract_digest",
        "formal_identity",
        "code_identity",
        "content_set_digest",
        "dataset_proofs",
        "publication_preflight",
        "datasets",
    }
    report = json.loads(config.report_output.read_text(encoding="utf-8"))
    assert [item["dataset"] for item in report["source_target_comparisons"]] == ["D1", "D2"]
    assert all(report["proof_identities"][f"D{i}"]["mode"] == "contract_transformed" for i in range(3, 7))
    assert config.manifest_candidate_output.read_bytes().endswith(b"\n")
    assert not config.manifest_candidate_output.read_bytes().endswith(b"\n\n")


def test_failure_keeps_old_root_unchanged_and_publishes_no_final_root(tmp_path: Path) -> None:
    config, old, outputs = _fixture(tmp_path)
    before = _tree_bytes(old)
    calls: list[int] = []

    status = operator.materialize(
        config,
        producer_runner=_copying_producer(old, calls, fail_at=4),
        outputs_run_root=outputs,
    )

    assert status != 0
    assert calls == [3, 4]
    assert _tree_bytes(old) == before
    assert not list(config.final_deployment_parent.glob("d1_d6_sealed_v1_deploy_*"))
    assert not config.manifest_candidate_output.exists()
    assert (config.private_build_root / "NON_AUTHORITATIVE.json").is_file()
    report = json.loads(config.report_output.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["failure_dataset"] == "D4"
    assert report["error_code"] == "PRODUCER_FAILED"


def test_transformed_artifact_tamper_blocks_publication_without_old_byte_gate(tmp_path: Path) -> None:
    config, old, outputs = _fixture(tmp_path)
    calls: list[int] = []

    def drift(dataset_id: int, parent_root: Path, output_root: Path):
        completed = _copying_producer(old, calls)(dataset_id, parent_root, output_root)
        if dataset_id == 3:
            (output_root / "dataset3" / "source.parquet").write_bytes(b"tampered")
        return completed

    assert operator.materialize(config, producer_runner=drift, outputs_run_root=outputs) != 0
    assert calls == [3]
    assert not config.manifest_candidate_output.exists()
    assert not list(config.final_deployment_parent.glob("d1_d6_sealed_v1_deploy_*"))
    report = json.loads(config.report_output.read_text())
    assert report["error_code"] in {"ARTIFACT_SIZE", "ARTIFACT_HASH"}
    assert report["datasets"][0]["status"] == "failed"


@pytest.mark.parametrize("dataset_id", [1, 2])
@pytest.mark.parametrize("role", ["source", "target"])
def test_d1_d2_pass_through_byte_drift_is_rejected(
    tmp_path: Path, dataset_id: int, role: str
) -> None:
    config, old, _outputs = _fixture(tmp_path)
    shutil.copytree(old, config.private_build_root)
    path = config.private_build_root / f"dataset{dataset_id}" / f"{role}.parquet"
    path.write_bytes(path.read_bytes() + b"drift")
    with pytest.raises(operator.MaterializationError) as failure:
        operator.validate_pass_through_dataset(old, config.private_build_root, dataset_id)
    assert failure.value.code == "SOURCE_TARGET_IDENTITY_DRIFT"


@pytest.mark.parametrize("dataset_id", [1, 2])
def test_d1_d2_pass_through_sidecar_drift_is_rejected(
    tmp_path: Path, dataset_id: int
) -> None:
    config, old, _outputs = _fixture(tmp_path)
    shutil.copytree(old, config.private_build_root)
    path = config.private_build_root / f"dataset{dataset_id}" / "provenance.json"
    path.write_bytes(path.read_bytes() + b"drift")
    with pytest.raises(operator.MaterializationError) as failure:
        operator.validate_pass_through_dataset(old, config.private_build_root, dataset_id)
    assert failure.value.code == "D1_D2_COPY_MISMATCH"


@pytest.mark.parametrize("dataset_id", [3, 4, 5, 6])
@pytest.mark.parametrize("role", ["source", "target"])
def test_d3_d6_legal_physical_byte_change_passes_contract_proof(
    tmp_path: Path, dataset_id: int, role: str
) -> None:
    config, old, outputs = _fixture(tmp_path)
    calls: list[int] = []

    def rewrite(proof, manifest, directory):
        path = directory / f"{role}.parquet"
        frame = pd.read_parquet(path)
        path.unlink()
        frame.to_parquet(path, index=False, compression="gzip")
        manifest["artifacts"][role]["size_bytes"] = path.stat().st_size
        manifest["artifacts"][role]["sha256"] = operator.sha256_file(path)

    status = operator.materialize(
        config,
        producer_runner=_mutating_producer(
            old, calls, dataset_to_mutate=dataset_id, mutate=rewrite
        ),
        outputs_run_root=outputs,
    )
    assert status == 0
    final = _final_root(config)
    assert (final / f"dataset{dataset_id}" / f"{role}.parquet").read_bytes() != (
        old / f"dataset{dataset_id}" / f"{role}.parquet"
    ).read_bytes()


@pytest.mark.parametrize("dataset_id", [3, 4, 5, 6])
@pytest.mark.parametrize("role", ["source", "target"])
def test_d3_d6_illegal_unbound_output_change_fails_closed(
    tmp_path: Path, dataset_id: int, role: str
) -> None:
    config, old, outputs = _fixture(tmp_path)
    calls: list[int] = []

    def corrupt(_proof, _manifest, directory):
        (directory / f"{role}.parquet").write_bytes(b"unbound-output")

    assert operator.materialize(
        config,
        producer_runner=_mutating_producer(
            old, calls, dataset_to_mutate=dataset_id, mutate=corrupt
        ),
        outputs_run_root=outputs,
    ) != 0
    assert not config.manifest_candidate_output.exists()


@pytest.mark.parametrize(
    ("dataset_id", "mutation", "expected_code"),
    [
        (3, "raw", "RAW_AUTHORITY"),
        (4, "formal", "FORMAL_IDENTITY"),
        (5, "code", "CODE_IDENTITY"),
        (6, "config", "PRODUCER_CONFIG"),
        (3, "approved", "APPROVED_INPUTS"),
        (4, "lineage", "PARENT_LINEAGE"),
        (5, "key", "KEY_PROOF"),
        (6, "window", "WINDOW_PROOF"),
        (3, "repair", "TRANSFORMATION_PROOF"),
        (4, "schema", "SCHEMA_PROOF"),
        (5, "availability", "NO_LEAKAGE_PROOF"),
        (6, "content", "CONTENT_PROOF"),
        (3, "preflight", "FORMAL_PREFLIGHT"),
        (4, "incomplete", "PUBLICATION_PROOF_DIGEST"),
    ],
)
def test_layered_publication_proof_mutations_fail_closed(
    tmp_path: Path, dataset_id: int, mutation: str, expected_code: str
) -> None:
    config, old, outputs = _fixture(tmp_path)
    calls: list[int] = []

    def mutate(proof, _manifest, _directory):
        if mutation == "raw":
            proof["raw_authority"]["files"][0]["sha256"] = "0" * 64
            proof["raw_authority"]["approved_input_set_digest"] = canonical_digest(
                proof["raw_authority"]["files"]
            )
        elif mutation == "formal":
            proof["formal_identity"]["files"]["implementation_scope"]["sha256"] = "0" * 64
        elif mutation == "code":
            proof["code_identity"]["files"]["producer"]["sha256"] = "0" * 64
        elif mutation == "config":
            proof["producer_configuration"]["dataset"] = "D0"
            proof["producer_configuration_digest"] = canonical_digest(
                proof["producer_configuration"]
            )
        elif mutation == "approved":
            proof["approved_inputs"]["artifacts"]["extra"] = {}
            proof["approved_inputs"]["digest"] = canonical_digest(
                proof["approved_inputs"]["artifacts"]
            )
        elif mutation == "lineage":
            proof["parent_lineage"]["raw_snapshot_identity"] = "0" * 64
        elif mutation == "key":
            proof["key_window"]["source_key"]["columns"] = ["wrong", "date"]
        elif mutation == "window":
            proof["key_window"]["horizon_days"] += 1
        elif mutation == "repair":
            proof["transformation"]["backward_fill"] = True
        elif mutation == "schema":
            proof["schemas"]["predictor"]["fields"][0]["dtype"] = "object"
        elif mutation == "availability":
            proof["availability_no_leakage"]["target_day_actual_isolated"] = False
        elif mutation == "content":
            proof["content"]["normalization"] = "unknown"
        elif mutation == "preflight":
            proof["formal_preflight"]["status"] = "failed"
        else:
            del proof["proof_digest"]

    assert operator.materialize(
        config,
        producer_runner=_mutating_producer(
            old, calls, dataset_to_mutate=dataset_id, mutate=mutate
        ),
        outputs_run_root=outputs,
    ) != 0
    report = json.loads(config.report_output.read_text(encoding="utf-8"))
    assert report["error_code"] == expected_code
    assert report["datasets"][-1]["status"] == "failed"
    assert not config.manifest_candidate_output.exists()


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("content_validation_level", "structural_only", "STRUCTURAL_ONLY"),
        ("adopted_content_validated", False, "CONTENT_UNVALIDATED"),
    ],
)
def test_structural_or_unvalidated_output_cannot_publish(
    tmp_path: Path, field: str, value: object, expected: str
) -> None:
    config, old, outputs = _fixture(tmp_path)

    def mutate(_proof, manifest, _directory):
        manifest[field] = value

    assert operator.materialize(
        config,
        producer_runner=_mutating_producer(
            old, [], dataset_to_mutate=3, mutate=mutate
        ),
        outputs_run_root=outputs,
    ) != 0
    assert json.loads(config.report_output.read_text())["error_code"] == expected


@pytest.mark.parametrize("mutation", ["extra", "missing", "replace"])
def test_approved_parent_input_set_mutations_fail_closed(
    tmp_path: Path, mutation: str
) -> None:
    config, old, outputs = _fixture(tmp_path)

    def mutate(proof, _manifest, _directory):
        artifacts = proof["approved_inputs"]["artifacts"]
        if mutation == "extra":
            artifacts["extra"] = {}
        elif mutation == "missing":
            del artifacts["target"]
        else:
            artifacts["source"]["path"] = "/tmp/substitute.parquet"
        proof["approved_inputs"]["digest"] = canonical_digest(artifacts)

    assert operator.materialize(
        config,
        producer_runner=_mutating_producer(old, [], dataset_to_mutate=3, mutate=mutate),
        outputs_run_root=outputs,
    ) != 0
    assert json.loads(config.report_output.read_text())["error_code"] in {
        "APPROVED_INPUTS",
        "PARENT_IDENTITY",
    }


@pytest.mark.parametrize(
    "mutation",
    ["field", "order", "dtype", "role", "digest", "required_deleted"],
)
def test_safe_schema_mutations_fail_closed(tmp_path: Path, mutation: str) -> None:
    config, old, outputs = _fixture(tmp_path)

    def mutate(proof, _manifest, _directory):
        fields = proof["schemas"]["predictor"]["fields"]
        if mutation == "field":
            fields.append({"name": "widened", "dtype": "float64", "role": "future_known", "transform": "identity"})
        elif mutation == "order":
            fields.reverse()
        elif mutation == "dtype":
            fields[0]["dtype"] = "object"
        elif mutation == "role":
            fields[0]["role"] = "audit_only"
        elif mutation == "digest":
            proof["schemas"]["predictor_digest"] = "0" * 64
        else:
            fields.pop()

    assert operator.materialize(
        config,
        producer_runner=_mutating_producer(old, [], dataset_to_mutate=3, mutate=mutate),
        outputs_run_root=outputs,
    ) != 0
    assert json.loads(config.report_output.read_text())["error_code"] == "SCHEMA_PROOF"


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("cardinality", "ROW_CARDINALITY"),
        ("generic_fill", "TRANSFORMATION_PROOF"),
        ("target_truth", "NO_LEAKAGE_PROOF"),
        ("target_day", "NO_LEAKAGE_PROOF"),
        ("audit", "NO_LEAKAGE_PROOF"),
        ("forbidden", "NO_LEAKAGE_PROOF"),
    ],
)
def test_cardinality_fill_and_leakage_mutations_fail_closed(
    tmp_path: Path, mutation: str, expected: str
) -> None:
    config, old, outputs = _fixture(tmp_path)

    def mutate(proof, _manifest, _directory):
        if mutation == "cardinality":
            proof["transformation"]["target_rows_after"] += 1
        elif mutation == "generic_fill":
            proof["transformation"]["generic_fill"] = True
        else:
            field = {
                "target_truth": "target_truth_isolated",
                "target_day": "target_day_actual_isolated",
                "audit": "audit_only_isolated",
                "forbidden": "forbidden_fields_isolated",
            }[mutation]
            proof["availability_no_leakage"][field] = False

    assert operator.materialize(
        config,
        producer_runner=_mutating_producer(old, [], dataset_to_mutate=3, mutate=mutate),
        outputs_run_root=outputs,
    ) != 0
    assert json.loads(config.report_output.read_text())["error_code"] == expected


def test_private_build_marker_must_belong_to_current_run(tmp_path: Path) -> None:
    config, old, outputs = _fixture(tmp_path)

    def change_marker_after_last(dataset_id: int, parent_root: Path, output_root: Path):
        completed = _copying_producer(old, [])(dataset_id, parent_root, output_root)
        if dataset_id == 6:
            _write_json(
                output_root / "NON_AUTHORITATIVE.json",
                operator._marker_payload("building", run_id="different-run"),
            )
        return completed

    assert operator.materialize(
        config, producer_runner=change_marker_after_last, outputs_run_root=outputs
    ) != 0
    report = json.loads(config.report_output.read_text())
    assert report["error_code"] == "PRIVATE_BUILD_OWNERSHIP"
    assert not config.manifest_candidate_output.exists()


def test_partial_external_publication_is_rolled_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, old, outputs = _fixture(tmp_path)
    real_publish = operator._publish_staged

    def fail_report(temporary: Path, destination: Path) -> None:
        if destination == config.report_output:
            raise OSError("injected report publication failure")
        real_publish(temporary, destination)

    monkeypatch.setattr(operator, "_publish_staged", fail_report)
    assert operator.materialize(
        config, producer_runner=_copying_producer(old, []), outputs_run_root=outputs
    ) != 0
    assert not config.manifest_candidate_output.exists()
    assert not list(config.final_deployment_parent.glob("d1_d6_sealed_v1_deploy_*"))
    marker = json.loads(
        (config.private_build_root / "NON_AUTHORITATIVE.json").read_text()
    )
    assert marker["materialization_status"] == "failed"


@pytest.mark.parametrize("damage", ["missing", "identity"])
def test_incomplete_or_mismatched_repair_proof_blocks_publication(
    tmp_path: Path, damage: str
) -> None:
    config, old, outputs = _fixture(tmp_path)
    calls: list[int] = []

    def damaged(dataset_id: int, parent_root: Path, output_root: Path):
        completed = _copying_producer(old, calls)(dataset_id, parent_root, output_root)
        if dataset_id == 3:
            sidecar_path = output_root / "dataset3" / "source_sales_canonicalization.json"
            sidecar = json.loads(sidecar_path.read_text())
            if damage == "missing":
                sidecar["status"] = None
                manifest_path = output_root / "dataset3" / "manifest.json"
                report_path = output_root / "dataset3" / "adopt_validation_report.json"
                manifest = json.loads(manifest_path.read_text())
                report = json.loads(report_path.read_text())
                manifest["source_sales_repair"] = sidecar
                report["source_sales_repair"] = sidecar
                _write_json(manifest_path, manifest)
                _write_json(report_path, report)
            else:
                sidecar["repair_mask_sha256"] = "c" * 64
            _write_json(sidecar_path, sidecar)
        return completed

    assert operator.materialize(config, producer_runner=damaged, outputs_run_root=outputs) != 0
    assert calls == [3]
    assert not config.manifest_candidate_output.exists()
    assert not list(config.final_deployment_parent.glob("d1_d6_sealed_v1_deploy_*"))


def test_cli_rejects_existing_private_or_final_root(tmp_path: Path) -> None:
    config, old, outputs = _fixture(tmp_path)
    config.private_build_root.mkdir()
    assert operator.materialize(config, producer_runner=_copying_producer(old, []), outputs_run_root=outputs) != 0

    second = tmp_path / "second"
    second.mkdir()
    config, old, outputs = _fixture(second)
    calls: list[int] = []

    def collide_after_last_dataset(dataset_id: int, parent_root: Path, output_root: Path):
        completed = _copying_producer(old, calls)(dataset_id, parent_root, output_root)
        if dataset_id == 6:
            entries = operator.inventory_artifacts(
                output_root, allow_non_authoritative_marker=True
            )
            digest = operator.content_set_digest(entries)
            (config.final_deployment_parent / f"d1_d6_sealed_v1_deploy_{digest[:16]}").mkdir()
        return completed

    assert operator.materialize(
        config, producer_runner=collide_after_last_dataset, outputs_run_root=outputs
    ) != 0
    assert calls == [3, 4, 5, 6]
    assert not config.manifest_candidate_output.exists()
    assert (config.private_build_root / "NON_AUTHORITATIVE.json").is_file()


def test_cli_rejects_cross_device_private_build_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, old, outputs = _fixture(tmp_path)
    devices = iter((1, 2))
    monkeypatch.setattr(operator, "_device_id", lambda _path: next(devices))
    calls: list[int] = []
    assert operator.materialize(
        config, producer_runner=_copying_producer(old, calls), outputs_run_root=outputs
    ) != 0
    assert calls == []
    assert not config.private_build_root.exists()


def test_manifest_candidate_rejects_extra_missing_absolute_parent_or_symlink_paths(
    tmp_path: Path,
) -> None:
    config, old, _outputs = _fixture(tmp_path)
    entries = operator.inventory_artifacts(old)
    digest = operator.content_set_digest(entries)
    candidate = operator.build_manifest_candidate(
        entries,
        digest,
        f"d1_d6_sealed_v1_deploy_{digest[:16]}",
        run_id="test-run-0001",
        dataset_proofs={f"D{i}": {"status": "passed"} for i in range(1, 7)},
    )
    operator.validate_manifest_candidate(candidate, old, entries)

    mutations = []
    extra = deepcopy(candidate)
    extra["datasets"]["D1"].append(
        {"logical_role": "extra", "path": "dataset1/extra", "size_bytes": 0, "sha256": "0" * 64}
    )
    mutations.append(extra)
    missing = deepcopy(candidate)
    missing["datasets"]["D1"].pop()
    mutations.append(missing)
    absolute = deepcopy(candidate)
    absolute["datasets"]["D1"][0]["path"] = "/tmp/escape"
    mutations.append(absolute)
    parent = deepcopy(candidate)
    parent["datasets"]["D1"][0]["path"] = "dataset1/../escape"
    mutations.append(parent)
    for mutation in mutations:
        with pytest.raises(operator.MaterializationError):
            operator.validate_manifest_candidate(mutation, old, entries)

    symlink_root = tmp_path / "symlink-root"
    shutil.copytree(old, symlink_root)
    victim = symlink_root / "dataset1" / "source.parquet"
    victim.unlink()
    victim.symlink_to(old / "dataset1" / "source.parquet")
    with pytest.raises(operator.MaterializationError):
        operator.validate_manifest_candidate(candidate, symlink_root, entries)


def test_content_set_digest_and_manifest_are_deterministic(tmp_path: Path) -> None:
    _config, old, _outputs = _fixture(tmp_path)
    first_entries = operator.inventory_artifacts(old)
    second_entries = operator.inventory_artifacts(old)
    first_digest = operator.content_set_digest(first_entries)
    second_digest = operator.content_set_digest(second_entries)
    assert first_digest == second_digest
    proofs = {f"D{i}": {"status": "passed"} for i in range(1, 7)}
    assert operator.build_manifest_candidate(first_entries, first_digest, "deploy", run_id="test-run-0001", dataset_proofs=proofs) == (
        operator.build_manifest_candidate(second_entries, second_digest, "deploy", run_id="test-run-0001", dataset_proofs=proofs)
    )


def test_dry_run_performs_no_io_or_producer_call(tmp_path: Path) -> None:
    base = tmp_path / "does-not-exist"
    paths = {
        "old": base / "old",
        "parents": base / "parents",
        "build": base / "build",
        "deployments": tmp_path / "deployments",
        "report": base / "report.json",
        "manifest": base / "manifest.json",
    }
    completed = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--dry-run",
            "--old-sealed-root",
            str(paths["old"]),
            "--parent-root",
            str(paths["parents"]),
            "--private-build-root",
            str(paths["build"]),
            "--final-deployment-parent",
            str(paths["deployments"]),
            "--report-output",
            str(paths["report"]),
            "--manifest-candidate-output",
            str(paths["manifest"]),
            "--run-id",
            "test-run-0001",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    plan = json.loads(completed.stdout)
    assert plan["producer_calls_performed"] == 0
    assert plan["writes_performed"] is False
    assert plan["data_traversal_performed"] is False
    assert not base.exists()
    assert not paths["deployments"].exists()


def test_dataset_is_never_split_below_dataset_level(tmp_path: Path) -> None:
    config, old, outputs = _fixture(tmp_path)
    calls: list[int] = []
    assert operator.materialize(
        config, producer_runner=_copying_producer(old, calls), outputs_run_root=outputs
    ) == 0
    assert calls == [3, 4, 5, 6]
    report = json.loads(config.report_output.read_text())
    assert [record["dataset"] for record in report["datasets"]] == ["D3", "D4", "D5", "D6"]
    assert all(record["returncode"] == 0 for record in report["datasets"])
    assert all(
        not ({"entity", "batch", "date_range", "file_fragment"} & set(record))
        for record in report["datasets"]
    )


def test_operator_has_no_training_or_formal_publication_calls() -> None:
    tree = ast.parse(TOOL.read_text(encoding="utf-8"))
    called_names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            called_names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            called_names.add(node.func.attr)
    assert not (
        {
            "fit",
            "fit_transform",
            "predict",
            "run_unified_d1_d6",
            "publish_run",
            "create_attempt",
        }
        & called_names
    )


def test_real_producer_command_is_one_whole_dataset_without_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured.update(kwargs)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(operator.subprocess, "run", fake_run)
    completed = operator._run_producer(5, Path("/tmp/parents"), Path("/tmp/build"))
    assert completed.returncode == 0
    assert captured["argv"] == [
        sys.executable,
        "scripts/adopt_and_seal_d3_d6.py",
        "--dataset",
        "d5",
        "--parent-root",
        "/tmp/parents",
        "--output-dir",
        "/tmp/build",
    ]
    assert captured["shell"] is False
    assert captured["timeout"] is None
    assert captured["check"] is False
