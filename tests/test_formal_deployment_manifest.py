from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.protocols import formal_deployment_manifest as deployment


ROOT = Path(__file__).resolve().parents[1]
SEALED_ROOT = ROOT / "数据集" / "固化数据" / "d1_d6_sealed_v1"


def _manifest_payload() -> dict[str, object]:
    path = SEALED_ROOT / "deployment-manifest.json"
    if not path.is_file():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def test_red_root_manifest_exists() -> None:
    assert (SEALED_ROOT / "deployment-manifest.json").is_file(), "ROOT_MANIFEST_MISSING"


def test_red_root_manifest_sidecar_exists() -> None:
    assert (SEALED_ROOT / "deployment-manifest.sha256").is_file(), "ROOT_MANIFEST_SHA_MISSING"


def test_red_all_formal_proofs_exist() -> None:
    missing = [
        dataset_id
        for dataset_id in range(1, 7)
        if not (SEALED_ROOT / f"dataset{dataset_id}" / "formal-proof.json").is_file()
    ]
    assert not missing, f"FORMAL_PROOF_MISSING datasets={missing}"


def test_red_root_identity_exists() -> None:
    value = _manifest_payload().get("root_identity_sha256")
    assert isinstance(value, str) and len(value) == 64, "ROOT_IDENTITY_MISSING"


def test_red_code_inventory_exists() -> None:
    assert (SEALED_ROOT / "code-inventory.json").is_file(), "CODE_INVENTORY_MISSING"


def _write_root_manifest(root: Path, payload: dict[str, object]) -> None:
    manifest = {
        **payload,
        "root_identity_sha256": deployment.sha256_bytes(
            deployment.canonical_json_bytes(payload)
        ),
    }
    data = deployment.pretty_json_bytes(manifest)
    (root / "deployment-manifest.json").write_bytes(data)
    (root / "deployment-manifest.sha256").write_text(
        f"{deployment.sha256_bytes(data)}  deployment-manifest.json\n",
        encoding="utf-8",
    )


def test_root_manifest_missing_and_sidecar_missing_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(deployment.DeploymentManifestError) as captured:
        deployment.load_deployment_manifest(tmp_path)
    assert captured.value.code == "ROOT_MANIFEST_MISSING"

    (tmp_path / "deployment-manifest.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(deployment.DeploymentManifestError) as captured:
        deployment.load_deployment_manifest(tmp_path)
    assert captured.value.code == "ROOT_MANIFEST_SHA_MISSING"


def test_manifest_bytes_and_root_identity_tampering_fail_closed(tmp_path: Path) -> None:
    _write_root_manifest(tmp_path, {"schema_version": "test"})
    manifest_path = tmp_path / "deployment-manifest.json"
    manifest_path.write_bytes(manifest_path.read_bytes() + b" ")
    with pytest.raises(deployment.DeploymentManifestError) as captured:
        deployment.load_deployment_manifest(tmp_path)
    assert captured.value.code == "ROOT_MANIFEST_SHA_MISMATCH"

    _write_root_manifest(tmp_path, {"schema_version": "test"})
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["root_identity_sha256"] = "0" * 64
    data = deployment.pretty_json_bytes(manifest)
    manifest_path.write_bytes(data)
    (tmp_path / "deployment-manifest.sha256").write_text(
        f"{deployment.sha256_bytes(data)}  deployment-manifest.json\n",
        encoding="utf-8",
    )
    with pytest.raises(deployment.DeploymentManifestError) as captured:
        deployment.load_deployment_manifest(tmp_path)
    assert captured.value.code == "ROOT_IDENTITY_MISMATCH"


@pytest.mark.parametrize("value", ("/absolute/file.json", "../escape.json", "a\\b.json"))
def test_manifest_bound_paths_reject_absolute_and_traversal(
    tmp_path: Path, value: str
) -> None:
    with pytest.raises(deployment.DeploymentManifestError):
        deployment._resolve_bound_path(tmp_path, value, code="PATH_INVALID")


def test_manifest_bound_paths_reject_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-manifest-fixture.json"
    outside.write_text("{}\n", encoding="utf-8")
    link = tmp_path / "escape.json"
    link.symlink_to(outside)
    with pytest.raises(deployment.DeploymentManifestError) as captured:
        deployment._resolve_bound_path(tmp_path, "escape.json", code="PATH_INVALID")
    assert captured.value.code == "PATH_INVALID"


def _write_parquet(path: Path, schema: pa.Schema) -> None:
    arrays = []
    for field in schema:
        if pa.types.is_integer(field.type):
            arrays.append(pa.array([1, 2], type=field.type))
        elif pa.types.is_timestamp(field.type):
            arrays.append(pa.array([0, 1], type=field.type))
        else:
            arrays.append(pa.array([1.0, 2.0], type=field.type))
    pq.write_table(pa.Table.from_arrays(arrays, schema=schema), path)


def test_schema_digest_is_order_dtype_nullable_role_and_key_sensitive(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.parquet"
    reordered = tmp_path / "reordered.parquet"
    changed = tmp_path / "changed.parquet"
    _write_parquet(
        first,
        pa.schema(
            [
                pa.field("id", pa.int64(), nullable=False),
                pa.field("date", pa.timestamp("ns"), nullable=True),
                pa.field("sales", pa.float64(), nullable=True),
            ]
        ),
    )
    _write_parquet(
        reordered,
        pa.schema(
            [
                pa.field("date", pa.timestamp("ns"), nullable=True),
                pa.field("id", pa.int64(), nullable=False),
                pa.field("sales", pa.float64(), nullable=True),
            ]
        ),
    )
    _write_parquet(
        changed,
        pa.schema(
            [
                pa.field("id", pa.int32(), nullable=True),
                pa.field("date", pa.timestamp("ns"), nullable=True),
                pa.field("sales", pa.float64(), nullable=True),
            ]
        ),
    )
    base = deployment.schema_descriptor(
        first, role="source", key_columns=("id",)
    )["schema_digest"]
    assert base != deployment.schema_descriptor(
        reordered, role="source", key_columns=("id",)
    )["schema_digest"]
    assert base != deployment.schema_descriptor(
        changed, role="source", key_columns=("id",)
    )["schema_digest"]
    assert base != deployment.schema_descriptor(
        first, role="target", key_columns=("id",)
    )["schema_digest"]
    assert base != deployment.schema_descriptor(
        first, role="source", key_columns=("sales",)
    )["schema_digest"]


def test_formal_proof_tampering_and_identity_mismatch_fail_closed() -> None:
    identity = {"combined_identity_sha256": "sha256:" + "1" * 64}
    payload: dict[str, object] = {
        "schema_version": deployment.PROOF_SCHEMA_VERSION,
        "dataset_id": "D1",
        "formal_identity": identity,
        "readiness_result": {"status": "passed", "failure_code": None},
        "key_date_uniqueness": {"status": "passed"},
    }
    proof = {
        **payload,
        "proof_identity_sha256": deployment.sha256_bytes(
            deployment.canonical_json_bytes(payload)
        ),
    }
    deployment.verify_formal_proof(proof, dataset_id=1, formal_identity=identity)
    proof["dataset_id"] = "D2"
    with pytest.raises(deployment.DeploymentManifestError):
        deployment.verify_formal_proof(proof, dataset_id=1, formal_identity=identity)


def test_code_inventory_is_deterministic_and_self_consistent() -> None:
    first = deployment.build_code_inventory(ROOT)
    second = deployment.build_code_inventory(ROOT)
    assert first == second
    assert first["file_count"] == len(first["files"])
    assert [item["path"] for item in first["files"]] == sorted(
        item["path"] for item in first["files"]
    )
    deployment.verify_code_inventory(ROOT, first)


def test_d1_d2_knn_authority_tracks_real_config_bytes() -> None:
    snapshot = deployment.frozen_artifact_snapshot(ROOT)
    authority = snapshot.get("d1_d2_knn")
    assert isinstance(authority, dict)
    for dataset_id in (1, 2):
        dataset_authority = deployment.D1_D2_KNN[dataset_id]
        actual_dataset = authority[f"D{dataset_id}"]
        for scenario in ("without", "with"):
            expected = dataset_authority[scenario]
            actual = actual_dataset[scenario]
            assert actual["path"] == expected["path"]
            assert actual["sha256"] == expected["sha256"]
            assert actual["selection_authority"] == "shared_protocol"
            assert actual["protocol_version"] == "d1_d6_protocol_v1"


def test_root_manifest_publishes_d1_d2_knn_authority() -> None:
    manifest = _manifest_payload()
    authority = manifest.get("d1_d2_knn_selection_authority")
    assert isinstance(authority, dict)
    assert set(authority) == {"D1", "D2"}
    for dataset_id in (1, 2):
        assert set(authority[f"D{dataset_id}"]) == {"without", "with"}


def test_generated_authority_json_contains_no_absolute_paths_or_timestamps() -> None:
    paths = [
        *(SEALED_ROOT / f"dataset{i}" / "formal-proof.json" for i in range(1, 7)),
        SEALED_ROOT / "code-inventory.json",
        SEALED_ROOT / "deployment-manifest.json",
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "/Users/" not in text, path
        assert "/private/" not in text, path
        assert '"timestamp"' not in text, path
        assert text.endswith("\n") and not text.endswith("\n\n"), path
