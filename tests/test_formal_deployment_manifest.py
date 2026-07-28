from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil

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


def _raw_formal_proof(dataset_id: int) -> dict[str, object]:
    return json.loads(
        (SEALED_ROOT / f"dataset{dataset_id}" / "formal-proof.json").read_text(
            encoding="utf-8"
        )
    )


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


def _parquet_identity_fixture(*, mtime_ns: int = 100) -> dict[str, object]:
    schema_payload = {
        "role": "source",
        "key_columns": ["id"],
        "date_column": "date",
        "columns": [
            {"name": "id", "dtype": "int64", "nullable": False},
            {"name": "date", "dtype": "timestamp[ns]", "nullable": True},
            {"name": "sales", "dtype": "double", "nullable": True},
        ],
    }
    schema_digest = deployment.sha256_bytes(
        deployment.canonical_json_bytes(schema_payload)
    )
    schema = {**schema_payload, "schema_digest": schema_digest}
    return {
        "path": "dataset1/source.parquet",
        "sha256": "a" * 64,
        "size_bytes": 123,
        "mtime_ns": mtime_ns,
        "rows": 2,
        "ordered_columns": ["id", "date", "sales"],
        "dtypes": {
            "id": "int64",
            "date": "timestamp[ns]",
            "sales": "double",
        },
        "nullable": {"id": False, "date": True, "sales": True},
        "schema_digest": schema_digest,
        "schema": schema,
    }


def _assert_structural_error(
    comparison: dict[str, object], *, field: str | None = None
) -> None:
    errors = comparison["structural_errors"]
    assert isinstance(errors, list) and errors
    if field is not None:
        assert any(error.get("field") == field for error in errors if isinstance(error, dict))


def test_parquet_identity_field_sets_are_explicit_and_disjoint() -> None:
    assert deployment.REQUIRED_BLOCKING_FIELDS
    assert deployment.OPTIONAL_DIAGNOSTIC_FIELDS == frozenset({"mtime_ns"})
    assert deployment.REQUIRED_BLOCKING_FIELDS.isdisjoint(
        deployment.OPTIONAL_DIAGNOSTIC_FIELDS
    )
    assert deployment.KNOWN_IDENTITY_FIELDS == (
        deployment.REQUIRED_BLOCKING_FIELDS
        | deployment.OPTIONAL_DIAGNOSTIC_FIELDS
    )


@pytest.mark.parametrize("side", ("expected", "actual"))
def test_unknown_identity_field_blocks_and_reports_its_name(side: str) -> None:
    expected = _parquet_identity_fixture()
    actual = _parquet_identity_fixture()
    identity = expected if side == "expected" else actual
    identity["future_content_identity"] = "same-looking-value"

    comparison = deployment.compare_parquet_identity(expected, actual)

    _assert_structural_error(comparison)
    assert "future_content_identity" in repr(comparison)


def test_same_unknown_identity_field_and_value_still_blocks() -> None:
    expected = _parquet_identity_fixture()
    actual = _parquet_identity_fixture()
    expected["future_content_identity"] = "same-looking-value"
    actual["future_content_identity"] = "same-looking-value"

    comparison = deployment.compare_parquet_identity(expected, actual)

    _assert_structural_error(comparison)
    assert "future_content_identity" in repr(comparison)


@pytest.mark.parametrize("field", sorted(deployment.REQUIRED_BLOCKING_FIELDS))
@pytest.mark.parametrize("side", ("expected", "actual", "both"))
def test_every_missing_blocking_field_fails_closed(field: str, side: str) -> None:
    expected = _parquet_identity_fixture()
    actual = _parquet_identity_fixture()
    if side in ("expected", "both"):
        del expected[field]
    if side in ("actual", "both"):
        del actual[field]

    comparison = deployment.compare_parquet_identity(expected, actual)

    _assert_structural_error(comparison, field=field)
    assert field in comparison["blocking_mismatches"]


@pytest.mark.parametrize("invalid", (None, [], "identity", 17))
def test_non_mapping_identity_is_a_controlled_structural_failure(invalid: object) -> None:
    comparison = deployment.compare_parquet_identity(
        invalid, _parquet_identity_fixture()
    )

    _assert_structural_error(comparison)


@pytest.mark.parametrize(
    ("field", "malformed"),
    (
        ("rows", "2"),
        ("size_bytes", True),
        ("ordered_columns", "id,date,sales"),
        ("dtypes", ["int64", "timestamp[ns]", "double"]),
        ("nullable", {"id": 0, "date": 1, "sales": 1}),
        ("schema", {"role": "source"}),
        ("sha256", "not-a-sha256"),
        ("schema_digest", "not-a-digest"),
        ("mtime_ns", True),
    ),
)
def test_same_malformed_identity_structure_fails_closed(
    field: str, malformed: object
) -> None:
    expected = _parquet_identity_fixture()
    actual = _parquet_identity_fixture()
    expected[field] = deepcopy(malformed)
    actual[field] = deepcopy(malformed)

    comparison = deployment.compare_parquet_identity(expected, actual)

    _assert_structural_error(comparison, field=field)


@pytest.mark.parametrize("side", ("expected", "actual", "both"))
def test_schema_dtype_must_match_top_level_dtype_on_each_side(side: str) -> None:
    expected = _parquet_identity_fixture()
    actual = _parquet_identity_fixture()
    identities = []
    if side in ("expected", "both"):
        identities.append(expected)
    if side in ("actual", "both"):
        identities.append(actual)
    for identity in identities:
        identity["dtypes"]["id"] = "int32"  # type: ignore[index]

    comparison = deployment.compare_parquet_identity(expected, actual)

    _assert_structural_error(comparison, field="dtypes")
    if side == "both":
        assert comparison["blocking_mismatches"] == {}
    else:
        assert "dtypes" in comparison["blocking_mismatches"]


@pytest.mark.parametrize("side", ("expected", "actual", "both"))
def test_schema_nullable_must_match_top_level_nullable_on_each_side(side: str) -> None:
    expected = _parquet_identity_fixture()
    actual = _parquet_identity_fixture()
    identities = []
    if side in ("expected", "both"):
        identities.append(expected)
    if side in ("actual", "both"):
        identities.append(actual)
    for identity in identities:
        identity["nullable"]["id"] = True  # type: ignore[index]

    comparison = deployment.compare_parquet_identity(expected, actual)

    _assert_structural_error(comparison, field="nullable")
    if side == "both":
        assert comparison["blocking_mismatches"] == {}
    else:
        assert "nullable" in comparison["blocking_mismatches"]


def test_schema_column_names_and_order_must_match_top_level_columns() -> None:
    expected = _parquet_identity_fixture()
    actual = _parquet_identity_fixture()
    actual["schema"]["columns"] = list(reversed(actual["schema"]["columns"]))  # type: ignore[index]

    comparison = deployment.compare_parquet_identity(expected, actual)

    _assert_structural_error(comparison, field="schema")


def test_schema_mutation_without_digest_update_fails_closed() -> None:
    expected = _parquet_identity_fixture()
    actual = _parquet_identity_fixture()
    for identity in (expected, actual):
        identity["dtypes"]["id"] = "int32"  # type: ignore[index]
        identity["schema"]["columns"][0]["dtype"] = "int32"  # type: ignore[index]

    comparison = deployment.compare_parquet_identity(expected, actual)

    _assert_structural_error(comparison, field="schema_digest")
    assert comparison["blocking_mismatches"] == {}


def test_changed_schema_digest_without_schema_change_fails_closed() -> None:
    expected = _parquet_identity_fixture()
    actual = _parquet_identity_fixture()
    for identity in (expected, actual):
        identity["schema_digest"] = "c" * 64
        identity["schema"]["schema_digest"] = "c" * 64  # type: ignore[index]

    comparison = deployment.compare_parquet_identity(expected, actual)

    _assert_structural_error(comparison, field="schema_digest")
    assert comparison["blocking_mismatches"] == {}


def test_schema_digest_uses_canonical_key_order() -> None:
    expected = _parquet_identity_fixture()
    actual = _parquet_identity_fixture()
    schema = actual["schema"]
    actual["schema"] = {
        "schema_digest": schema["schema_digest"],  # type: ignore[index]
        "columns": schema["columns"],  # type: ignore[index]
        "date_column": schema["date_column"],  # type: ignore[index]
        "key_columns": schema["key_columns"],  # type: ignore[index]
        "role": schema["role"],  # type: ignore[index]
    }

    comparison = deployment.compare_parquet_identity(expected, actual)

    assert comparison["structural_errors"] == []
    assert comparison["blocking_mismatches"] == {}


def test_content_identity_projection_is_strict_and_excludes_mtime() -> None:
    baseline = _parquet_identity_fixture(mtime_ns=100)
    changed_mtime = _parquet_identity_fixture(mtime_ns=200)

    projection = deployment.parquet_content_identity(baseline)

    assert set(projection) == deployment.REQUIRED_BLOCKING_FIELDS
    assert projection == deployment.parquet_content_identity(changed_mtime)
    malformed = deepcopy(baseline)
    malformed["mtime_ns"] = -1
    with pytest.raises(deployment.DeploymentManifestError) as captured:
        deployment.parquet_content_identity(malformed)
    assert captured.value.code == "PARQUET_IDENTITY_INVALID"


@pytest.mark.parametrize(
    "path",
    ("/absolute/file.parquet", "../escape.parquet", "dataset1/../escape.parquet", "a\\b.parquet"),
)
def test_parquet_identity_rejects_non_repository_relative_posix_paths(path: str) -> None:
    expected = _parquet_identity_fixture()
    actual = _parquet_identity_fixture()
    expected["path"] = path
    actual["path"] = path

    comparison = deployment.compare_parquet_identity(expected, actual)

    _assert_structural_error(comparison, field="path")


def test_parquet_identity_preserves_legal_repository_relative_posix_path() -> None:
    identity = _parquet_identity_fixture()

    comparison = deployment.compare_parquet_identity(identity, deepcopy(identity))

    assert comparison["structural_errors"] == []
    assert comparison["blocking_mismatches"] == {}


def test_content_identity_projection_digest_is_mtime_insensitive_and_content_sensitive() -> None:
    baseline = _parquet_identity_fixture(mtime_ns=100)
    changed_mtime = _parquet_identity_fixture(mtime_ns=200)
    changed_content = deepcopy(baseline)
    changed_content["sha256"] = "c" * 64

    def digest(identity: dict[str, object]) -> str:
        return deployment.sha256_bytes(
            deployment.canonical_json_bytes(deployment.parquet_content_identity(identity))
        )

    assert digest(baseline) == digest(changed_mtime)
    assert digest(baseline) != digest(changed_content)


def test_formal_proof_digest_uses_content_identity_projection() -> None:
    snapshot = deployment.frozen_artifact_snapshot(ROOT)
    changed_mtime = deepcopy(snapshot)
    changed_content = deepcopy(snapshot)
    changed_mtime["datasets"]["D3"]["source"]["mtime_ns"] += 1  # type: ignore[index]
    changed_content["datasets"]["D3"]["source"]["sha256"] = "c" * 64  # type: ignore[index]
    readiness = {
        "status": "passed",
        "failure_code": None,
        "duplicate_exact_keys": 0,
        "source_entities": [],
        "target_entities": [],
    }
    kwargs = {
        "snapshot": snapshot,
        "readiness": readiness,
        "formal_identity": {"combined_formal_identity_digest": "formal-identity"},
        "inventory_sha256": "a" * 64,
    }

    baseline = deployment.build_formal_proof(ROOT, 3, **kwargs)
    mtime_only = deployment.build_formal_proof(
        ROOT, 3, snapshot=changed_mtime, **{key: value for key, value in kwargs.items() if key != "snapshot"}
    )
    content_changed = deployment.build_formal_proof(
        ROOT, 3, snapshot=changed_content, **{key: value for key, value in kwargs.items() if key != "snapshot"}
    )

    assert "mtime_ns" not in baseline["source"]
    assert baseline == mtime_only
    assert baseline["proof_identity_sha256"] != content_changed["proof_identity_sha256"]


def test_formal_proof_authority_projection_is_raw_input_and_mtime_insensitive() -> None:
    proof = _raw_formal_proof(1)
    changed_mtime = deepcopy(proof)
    changed_mtime["source"]["mtime_ns"] += 1  # type: ignore[index]
    changed_mtime["target"]["mtime_ns"] += 1  # type: ignore[index]
    missing_mtime = deepcopy(proof)
    missing_mtime["source"].pop("mtime_ns")  # type: ignore[index]
    missing_mtime["target"].pop("mtime_ns")  # type: ignore[index]
    diagnostic_only = deepcopy(proof)
    diagnostic_only["parquet_identity_diagnostics"] = [{"code": "ignored"}]

    baseline_payload = deployment.formal_proof_authority_payload(proof)

    assert "proof_identity_sha256" not in baseline_payload
    assert "mtime_ns" not in baseline_payload["source"]  # type: ignore[index]
    assert baseline_payload == deployment.formal_proof_authority_payload(changed_mtime)
    assert baseline_payload == deployment.formal_proof_authority_payload(missing_mtime)
    assert baseline_payload == deployment.formal_proof_authority_payload(diagnostic_only)
    assert deployment.formal_proof_authority_digest(proof) == deployment.formal_proof_authority_digest(changed_mtime)
    unknown = deepcopy(proof)
    unknown["future_authority_field"] = "must-block"
    with pytest.raises(deployment.DeploymentManifestError) as captured:
        deployment.formal_proof_authority_payload(unknown)
    assert captured.value.code == "FORMAL_PROOF_STRUCTURE_INVALID"

    content_changed = deepcopy(proof)
    content_changed["source"]["sha256"] = "c" * 64  # type: ignore[index]
    assert deployment.formal_proof_authority_digest(proof) != deployment.formal_proof_authority_digest(content_changed)


def test_verify_formal_proof_uses_semantic_authority_digest_for_raw_and_legacy_identity() -> None:
    manifest = _manifest_payload()
    proof = _raw_formal_proof(1)
    authority_digest = deployment.formal_proof_authority_digest(proof)

    raw_with_mtime = deepcopy(proof)
    raw_with_mtime["source"]["mtime_ns"] += 1  # type: ignore[index]
    raw_with_mtime["proof_identity_sha256"] = authority_digest
    deployment.verify_formal_proof(
        raw_with_mtime,
        dataset_id=1,
        formal_identity=manifest["formal_identity"],
        authority_digest=authority_digest,
    )

    raw_without_mtime = deepcopy(proof)
    raw_without_mtime["source"].pop("mtime_ns")  # type: ignore[index]
    raw_without_mtime["target"].pop("mtime_ns")  # type: ignore[index]
    raw_without_mtime["proof_identity_sha256"] = authority_digest
    deployment.verify_formal_proof(
        raw_without_mtime,
        dataset_id=1,
        formal_identity=manifest["formal_identity"],
        authority_digest=authority_digest,
    )

    blocking_changed = deepcopy(proof)
    blocking_changed["source"]["sha256"] = "c" * 64  # type: ignore[index]
    with pytest.raises(deployment.DeploymentManifestError) as captured:
        deployment.verify_formal_proof(
            blocking_changed,
            dataset_id=1,
            formal_identity=manifest["formal_identity"],
            authority_digest=authority_digest,
        )
    assert captured.value.code == "FORMAL_PROOF_TAMPERED"


def test_legacy_raw_proof_digest_is_not_a_mtime_authority() -> None:
    manifest = _manifest_payload()
    proof = _raw_formal_proof(1)

    # This is the pre-repair proof shape: its embedded digest covered the raw
    # JSON, including mtime_ns.  The legacy root path still validates the
    # semantic proof fields without treating that historical digest as an
    # authority digest.
    deployment.verify_formal_proof(
        proof,
        dataset_id=1,
        formal_identity=manifest["formal_identity"],
        allow_legacy_raw_digest=True,
    )


def test_root_manifest_digest_uses_content_identity_projection() -> None:
    manifest = _manifest_payload()
    inventory_path = SEALED_ROOT / "code-inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    proofs = {
        f"D{dataset_id}": json.loads(
            (SEALED_ROOT / f"dataset{dataset_id}" / "formal-proof.json").read_text(
                encoding="utf-8"
            )
        )
        for dataset_id in range(1, 7)
    }

    def build(proof_set: dict[str, dict[str, object]]) -> dict[str, object]:
        return deployment.build_root_manifest(
            ROOT,
            proofs=proof_set,
            inventory=inventory,
            inventory_file_sha256=deployment.sha256_file(inventory_path),
            formal_identity=manifest["formal_identity"],
        )

    baseline = build(proofs)
    mtime_only_proofs = deepcopy(proofs)
    for proof in mtime_only_proofs.values():
        for role in ("source", "target"):
            proof[role]["mtime_ns"] = proof[role].get("mtime_ns", 0) + 1  # type: ignore[index]
    mtime_only = build(mtime_only_proofs)
    embedded_digest_only_proofs = deepcopy(proofs)
    for proof in embedded_digest_only_proofs.values():
        proof["proof_identity_sha256"] = "d" * 64
    embedded_digest_only = build(embedded_digest_only_proofs)
    content_changed_proofs = deepcopy(proofs)
    content_changed_proofs["D1"]["source"]["sha256"] = "c" * 64
    content_changed = build(content_changed_proofs)

    assert baseline["root_identity_sha256"] == mtime_only["root_identity_sha256"]
    assert baseline["root_identity_sha256"] == embedded_digest_only["root_identity_sha256"]
    assert baseline["root_identity_sha256"] != content_changed["root_identity_sha256"]
    assert "sha256" not in baseline["datasets"]["D1"]["formal_proof"]  # type: ignore[index]
    assert "authority_sha256" in baseline["datasets"]["D1"]["formal_proof"]  # type: ignore[index]


def test_mtime_only_difference_with_blocking_mismatch_has_no_accepted_diagnostic() -> None:
    expected = _parquet_identity_fixture(mtime_ns=100)
    actual = _parquet_identity_fixture(mtime_ns=200)
    actual["rows"] = 3

    comparison = deployment.compare_parquet_identity(expected, actual)

    assert "rows" in comparison["blocking_mismatches"]
    assert comparison["diagnostics"] == []


def test_parquet_identity_accepts_mtime_difference_without_identity_mismatch() -> None:
    authority = _parquet_identity_fixture(mtime_ns=100)
    current = _parquet_identity_fixture(mtime_ns=200)

    comparison = deployment.compare_parquet_identity(authority, current)

    assert comparison["blocking_mismatches"] == {}
    assert comparison["diagnostics"] == [
        {
            "code": "PARQUET_MTIME_DIFFERENCE_ACCEPTED",
            "field": "mtime_ns",
            "authority": 100,
            "current": 200,
        }
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("sha256", "b" * 64),
        ("size_bytes", 456),
        ("rows", 3),
        ("path", "dataset1/other.parquet"),
        ("ordered_columns", ["id", "sales", "date"]),
        ("dtypes", {"id": "int32", "date": "timestamp[ns]", "sales": "double"}),
        ("schema_digest", "different-schema-digest"),
        ("schema", {"changed": True}),
    ),
)
def test_parquet_content_identity_mismatch_blocks(
    field: str, value: object
) -> None:
    authority = _parquet_identity_fixture()
    current = deepcopy(authority)
    current[field] = value

    comparison = deployment.compare_parquet_identity(authority, current)

    assert field in comparison["blocking_mismatches"]
    assert comparison["diagnostics"] == []


def test_parquet_identity_accepts_legacy_manifest_without_mtime() -> None:
    authority = _parquet_identity_fixture()
    authority.pop("mtime_ns")
    current = _parquet_identity_fixture(mtime_ns=200)

    comparison = deployment.compare_parquet_identity(authority, current)

    assert comparison["blocking_mismatches"] == {}
    assert comparison["diagnostics"] == []


@pytest.mark.parametrize(
    ("expected_has_mtime", "actual_has_mtime"),
    ((True, True), (False, True), (True, False), (False, False)),
)
def test_all_mtime_presence_combinations_are_content_equivalent(
    expected_has_mtime: bool, actual_has_mtime: bool
) -> None:
    expected = _parquet_identity_fixture()
    actual = _parquet_identity_fixture()
    if not expected_has_mtime:
        expected.pop("mtime_ns")
    if not actual_has_mtime:
        actual.pop("mtime_ns")

    comparison = deployment.compare_parquet_identity(expected, actual)

    assert comparison["structural_errors"] == []
    assert comparison["blocking_mismatches"] == {}
    assert comparison["diagnostics"] == []


def test_missing_content_identity_field_blocks_fail_closed() -> None:
    authority = _parquet_identity_fixture()
    current = _parquet_identity_fixture()
    del current["schema_digest"]

    comparison = deployment.compare_parquet_identity(authority, current)

    assert comparison["blocking_mismatches"]["schema_digest"]


def test_validate_manifest_accepts_current_parquet_mtime_difference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = deployment.frozen_artifact_snapshot(ROOT)
    snapshot["datasets"]["D1"]["source"]["mtime_ns"] += 1  # type: ignore[index]
    monkeypatch.setattr(
        deployment, "frozen_artifact_snapshot", lambda _root: snapshot
    )
    monkeypatch.setattr(deployment, "verify_code_inventory", lambda *_args: None)

    report = deployment.validate_deployment_manifest(ROOT)

    assert report["preflight_status"] == "ready"
    diagnostic = report["parquet_identity_diagnostics"][0]  # type: ignore[index]
    assert diagnostic["dataset"] == "D1"
    assert diagnostic["role"] == "source"
    assert diagnostic["against"] == "current"
    assert diagnostic["code"] == "PARQUET_MTIME_DIFFERENCE_ACCEPTED"
    assert diagnostic["field"] == "mtime_ns"


def test_validate_manifest_accepts_formal_proof_parquet_mtime_difference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_json = deployment._json

    def load_modified_proof(path: Path, code: str) -> dict[str, object]:
        loaded = original_json(path, code)
        if path.name != "formal-proof.json" or path.parent.name != "dataset1":
            return loaded
        proof = deepcopy(loaded)
        proof["source"]["mtime_ns"] += 1  # type: ignore[index]
        payload = {
            key: value for key, value in proof.items() if key != "proof_identity_sha256"
        }
        proof["proof_identity_sha256"] = deployment.sha256_bytes(
            deployment.canonical_json_bytes(payload)
        )
        return proof

    monkeypatch.setattr(deployment, "_json", load_modified_proof)
    monkeypatch.setattr(deployment, "verify_code_inventory", lambda *_args: None)

    report = deployment.validate_deployment_manifest(ROOT)

    diagnostics = [
        item
        for item in report["parquet_identity_diagnostics"]  # type: ignore[index]
        if item["against"] == "formal_proof"
    ]
    assert diagnostics
    assert diagnostics[0]["dataset"] == "D1"
    assert diagnostics[0]["role"] == "source"
    assert diagnostics[0]["code"] == "PARQUET_MTIME_DIFFERENCE_ACCEPTED"


def test_validate_manifest_accepts_legacy_manifest_without_mtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = deepcopy(deployment.load_deployment_manifest(SEALED_ROOT))
    for role in ("source", "target"):
        manifest["datasets"]["D1"][role].pop("mtime_ns")  # type: ignore[index]
    monkeypatch.setattr(deployment, "load_deployment_manifest", lambda _root: manifest)
    monkeypatch.setattr(deployment, "verify_code_inventory", lambda *_args: None)

    report = deployment.validate_deployment_manifest(ROOT)

    assert report["preflight_status"] == "ready"


def test_formal_proof_tampering_and_identity_mismatch_fail_closed() -> None:
    manifest = _manifest_payload()
    identity = manifest["formal_identity"]
    proof = _raw_formal_proof(1)
    proof["proof_identity_sha256"] = deployment.formal_proof_authority_digest(proof)
    deployment.verify_formal_proof(proof, dataset_id=1, formal_identity=identity)
    proof["dataset_id"] = "D2"
    with pytest.raises(deployment.DeploymentManifestError):
        deployment.verify_formal_proof(proof, dataset_id=1, formal_identity=identity)


def test_builder_rolls_back_through_real_try_except_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.operations.build_d1_d6_root_manifest as builder

    sealed = tmp_path / "sealed"
    for dataset_id in range(1, 7):
        (sealed / f"dataset{dataset_id}").mkdir(parents=True)
    metadata_paths = [
        *(sealed / f"dataset{dataset_id}" / "formal-proof.json" for dataset_id in range(1, 7)),
        sealed / "code-inventory.json",
        sealed / "deployment-manifest.json",
        sealed / "deployment-manifest.sha256",
    ]
    original_bytes = {}
    for index, path in enumerate(metadata_paths):
        data = f"old-{index}".encode("ascii")
        path.write_bytes(data)
        original_bytes[path] = data

    monkeypatch.setattr(builder, "FORMAL_SEALED_ROOT_RELATIVE", Path("sealed"))
    monkeypatch.setattr(builder, "require_repository_identity", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(builder, "frozen_artifact_snapshot", lambda _root: {"snapshot": "fixed"})
    monkeypatch.setattr(builder, "verify_frozen_snapshot", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        builder,
        "formal_identity_payload",
        lambda _root: {"combined_identity_sha256": "formal"},
    )
    monkeypatch.setattr(
        builder,
        "build_code_inventory",
        lambda _root: {"inventory_sha256": "a" * 64, "file_count": 0, "files": []},
    )
    monkeypatch.setattr(
        builder,
        "run_readiness",
        lambda **_kwargs: {
            "status": "passed",
            "datasets": [
                {"status": "passed", "failure_code": None} for _ in range(6)
            ],
        },
    )
    monkeypatch.setattr(
        builder,
        "build_formal_proof",
        lambda _root, dataset_id, **_kwargs: {
            "dataset_id": f"D{dataset_id}",
            "proof_identity_sha256": "b" * 64,
        },
    )
    monkeypatch.setattr(builder, "verify_formal_proof", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        builder,
        "build_root_manifest",
        lambda *_args, **_kwargs: {
            "root_identity_sha256": deployment.sha256_bytes(
                deployment.canonical_json_bytes({})
            )
        },
    )

    original_atomic_write_json = builder.atomic_write_json
    write_calls: list[Path] = []

    def write_then_fail(path: Path, payload: object) -> None:
        original_atomic_write_json(path, payload)
        write_calls.append(path)
        if len(write_calls) == 2:
            raise RuntimeError("injected publication failure")

    monkeypatch.setattr(builder, "atomic_write_json", write_then_fail)

    with pytest.raises(RuntimeError, match="injected publication failure"):
        builder.build(
            tmp_path,
            expected_branch="unused",
            expected_head="unused",
            sealed_root=sealed,
        )

    assert len(write_calls) == 2
    assert {path: path.read_bytes() for path in metadata_paths} == original_bytes


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


def _readiness_proof_fixture(root: Path) -> dict[str, object]:
    formal_source = root / "数据集" / "固化数据" / "d1_d6_sealed_v1" / "dataset2" / "source.parquet"
    formal_target = root / "数据集" / "固化数据" / "d1_d6_sealed_v1" / "dataset2" / "target.parquet"
    knn_path = root / "configs" / "solidified" / "knn" / "Dataset5" / "knn_with_info_sharing.json"
    return {
        "dataset": "D2",
        "formal_identity": {"combined_formal_identity_digest": "formal-identity"},
        "formal_input": {
            "source_path": str(formal_source),
            "target_path": str(formal_target),
            "source_sha256": "a" * 64,
            "target_sha256": "b" * 64,
        },
        "raw_inputs": [
            {"path": str(root / "数据集" / "原始数据" / "input.csv"), "exists": True},
            {"path": str(root.parent / "temporary-input.csv"), "exists": False},
        ],
        "parent_inputs": {
            "source": {"path": str(root.parent / "parent-root" / formal_source.relative_to(root))},
            "target": {"path": str(root.parent / "parent-root" / formal_target.relative_to(root))},
        },
        "old_sealed_inputs": {
            "source": {"path": str(root.parent / "old-root" / formal_source.relative_to(root))},
            "target": {"path": str(root.parent / "old-root" / formal_target.relative_to(root))},
        },
        "sealed_identity": {
            "manifest_path": str(root / "数据集" / "固化数据" / "d1_d6_sealed_v1" / "dataset2" / "manifest.json"),
            "artifacts": {
                "source": {"path": str(formal_source), "sha256": "c" * 64},
                "target": {"path": str(formal_target), "sha256": "d" * 64},
            },
        },
        "selection_authority": {"scenarios": {"with": {"path": str(knn_path), "sha256": "e" * 64}}},
        "source_selection": {
            "scenarios": {"with": {"path": str(knn_path), "sha256": "f" * 64}},
            "stream_proof": {"authority_path": str(formal_source)},
        },
        "schema_fields": {"worker": ["date", "sales"]},
        "source_entities": [["source-1"]],
        "target_entities": [["target-1"]],
    }


def test_readiness_proof_digest_is_invariant_to_repository_root_and_normalizes_nested_paths(
    tmp_path: Path,
) -> None:
    root_a = tmp_path / "machine-a" / "project"
    root_b = tmp_path / "machine-b" / "project"
    root_a.mkdir(parents=True)
    (root_a / "README.md").write_text("same repository content\n", encoding="utf-8")
    shutil.copytree(root_a, root_b)

    payload_a = _readiness_proof_fixture(root_a)
    payload_b = _readiness_proof_fixture(root_b)
    assert deployment.readiness_proof_digest(payload_a, repository_root=root_a) == deployment.readiness_proof_digest(
        payload_b, repository_root=root_b
    )

    canonical = deployment.canonical_readiness_proof_payload(payload_a, repository_root=root_a)
    readiness = canonical["readiness"]
    assert readiness["formal_input"]["source_path"] == "数据集/固化数据/d1_d6_sealed_v1/dataset2/source.parquet"
    assert readiness["sealed_identity"]["manifest_path"] == "数据集/固化数据/d1_d6_sealed_v1/dataset2/manifest.json"
    assert readiness["sealed_identity"]["artifacts"]["source"]["path"] == "数据集/固化数据/d1_d6_sealed_v1/dataset2/source.parquet"
    assert readiness["raw_inputs"][0]["path"] == "数据集/原始数据/input.csv"
    assert readiness["source_selection"]["stream_proof"]["authority_path"] == "数据集/固化数据/d1_d6_sealed_v1/dataset2/source.parquet"
    assert readiness["selection_authority"]["scenarios"]["with"]["path"] == "configs/solidified/knn/Dataset5/knn_with_info_sharing.json"
    assert readiness["parent_inputs"] == {"source": {}, "target": {}}
    assert readiness["old_sealed_inputs"] == {"source": {}, "target": {}}
    assert readiness["raw_inputs"][1] == {"exists": False}


@pytest.mark.parametrize(
    ("section", "field", "value"),
    (
        ("formal_input", "source_sha256", "0" * 64),
        ("sealed_identity", "manifest_path", "数据集/固化数据/d1_d6_sealed_v1/dataset2/other-manifest.json"),
        ("selection_authority", "scenarios", {"with": {"path": "configs/changed.json", "sha256": "0" * 64}}),
        ("formal_identity", "combined_formal_identity_digest", "changed-formal-identity"),
        ("schema_fields", "worker", ["date", "changed"]),
        ("target_entities", "0", ["different-target"]),
    ),
)
def test_readiness_proof_digest_changes_for_sensitive_identity(
    tmp_path: Path, section: str, field: str, value: object
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    baseline = _readiness_proof_fixture(root)
    changed = deepcopy(baseline)
    if section == "target_entities":
        changed[section] = value
    else:
        changed[section][field] = value  # type: ignore[index]
    assert deployment.readiness_proof_digest(baseline, repository_root=root) != deployment.readiness_proof_digest(
        changed, repository_root=root
    )


@pytest.mark.parametrize(
    "field_path",
    (
        ("formal_input", "source_path"),
        ("sealed_identity", "manifest_path"),
        ("source_selection", "stream_proof", "authority_path"),
    ),
)
def test_formal_readiness_paths_outside_repository_fail_closed(
    tmp_path: Path, field_path: tuple[str, ...]
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    payload = _readiness_proof_fixture(root)
    cursor: dict[str, object] = payload
    for key in field_path[:-1]:
        cursor = cursor[key]  # type: ignore[assignment,index]
    cursor[field_path[-1]] = str(tmp_path / "outside" / "authority.json")
    with pytest.raises(deployment.DeploymentManifestError) as captured:
        deployment.canonical_readiness_proof_payload(payload, repository_root=root)
    assert captured.value.code == "READINESS_PROOF_PATH_INVALID"


def test_formal_readiness_path_structure_failure_is_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    payload = _readiness_proof_fixture(root)
    payload["formal_input"]["source_path"] = {"not": "a path"}  # type: ignore[index]
    with pytest.raises(deployment.DeploymentManifestError) as captured:
        deployment.canonical_readiness_proof_payload(payload, repository_root=root)
    assert captured.value.code == "READINESS_PROOF_PATH_INVALID"
