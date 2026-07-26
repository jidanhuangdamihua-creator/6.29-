#!/usr/bin/env python3
"""Deterministically build the D1-D6 formal proofs and root manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.protocols.formal_deployment_manifest import (  # noqa: E402
    EXPECTED_BRANCH,
    EXPECTED_HEAD,
    DeploymentManifestError,
    atomic_write_bytes,
    atomic_write_json,
    build_code_inventory,
    build_formal_proof,
    build_root_manifest,
    canonical_json_bytes,
    formal_identity_payload,
    frozen_artifact_snapshot,
    pretty_json_bytes,
    require_repository_identity,
    sha256_bytes,
    validate_deployment_manifest,
    verify_frozen_snapshot,
    verify_formal_proof,
)
from src.protocols.formal_input_paths import FORMAL_SEALED_ROOT_RELATIVE  # noqa: E402
from tools.operations.gate1x_real_input_readiness import run_readiness  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--expected-branch", default=EXPECTED_BRANCH)
    parser.add_argument("--expected-head", default=EXPECTED_HEAD)
    parser.add_argument("--sealed-root", type=Path, default=None)
    return parser


def _metadata_paths(sealed_root: Path) -> list[Path]:
    return [
        *(sealed_root / f"dataset{dataset_id}" / "formal-proof.json" for dataset_id in range(1, 7)),
        sealed_root / "code-inventory.json",
        sealed_root / "deployment-manifest.json",
        sealed_root / "deployment-manifest.sha256",
    ]


def _restore_metadata(previous: Mapping[Path, bytes | None]) -> None:
    for path, data in previous.items():
        if data is None:
            if path.exists():
                path.unlink()
        else:
            atomic_write_bytes(path, data)


def build(
    repository_root: Path,
    *,
    expected_branch: str,
    expected_head: str,
    sealed_root: Path | None = None,
) -> dict[str, object]:
    root = Path(repository_root).resolve(strict=True)
    sealed = Path(sealed_root or root / FORMAL_SEALED_ROOT_RELATIVE).resolve(strict=True)
    expected_sealed = (root / FORMAL_SEALED_ROOT_RELATIVE).resolve(strict=True)
    if sealed != expected_sealed:
        raise DeploymentManifestError("SEALED_ROOT_MISMATCH")
    if (sealed / "NON_AUTHORITATIVE").exists():
        raise DeploymentManifestError("NON_AUTHORITATIVE_ROOT")
    require_repository_identity(
        root, expected_branch=expected_branch, expected_head=expected_head
    )

    before = frozen_artifact_snapshot(root)
    verify_frozen_snapshot(before, repository_root=root)
    identity = formal_identity_payload(root)
    inventory = build_code_inventory(root)
    inventory_bytes = pretty_json_bytes(inventory)
    inventory_file_sha256 = sha256_bytes(inventory_bytes)

    readiness = run_readiness(
        root=root,
        parent_root=root,
        old_sealed_root=root,
        require_deployment=False,
    )
    datasets = readiness.get("datasets")
    if (
        readiness.get("status") != "passed"
        or not isinstance(datasets, list)
        or len(datasets) != 6
        or any(item.get("status") != "passed" for item in datasets)
    ):
        raise DeploymentManifestError(
            "FINAL_PREFLIGHT_NOT_READY", str(readiness.get("failure_code"))
        )

    proofs: dict[str, dict[str, object]] = {}
    for dataset_id, dataset_readiness in enumerate(datasets, start=1):
        proof = build_formal_proof(
            root,
            dataset_id,
            snapshot=before,
            readiness=dataset_readiness,
            formal_identity=identity,
            inventory_sha256=str(inventory["inventory_sha256"]),
        )
        verify_formal_proof(
            proof, dataset_id=dataset_id, formal_identity=identity
        )
        proofs[f"D{dataset_id}"] = proof

    manifest = build_root_manifest(
        root,
        proofs=proofs,
        inventory=inventory,
        inventory_file_sha256=inventory_file_sha256,
        formal_identity=identity,
    )
    manifest_bytes = pretty_json_bytes(manifest)
    manifest_sha256 = sha256_bytes(manifest_bytes)
    sidecar = f"{manifest_sha256}  deployment-manifest.json\n".encode("ascii")

    # Candidate validation is complete before any official metadata path changes.
    if manifest.get("root_identity_sha256") != sha256_bytes(
        canonical_json_bytes(
            {key: value for key, value in manifest.items() if key != "root_identity_sha256"}
        )
    ):
        raise DeploymentManifestError("ROOT_IDENTITY_MISMATCH")

    paths = _metadata_paths(sealed)
    previous = {path: path.read_bytes() if path.is_file() else None for path in paths}
    try:
        for dataset_id in range(1, 7):
            atomic_write_json(
                sealed / f"dataset{dataset_id}" / "formal-proof.json",
                proofs[f"D{dataset_id}"],
            )
        atomic_write_json(sealed / "code-inventory.json", inventory)
        # The root and its sidecar are deliberately published last.
        atomic_write_json(sealed / "deployment-manifest.json", manifest)
        atomic_write_bytes(sealed / "deployment-manifest.sha256", sidecar)
        result = validate_deployment_manifest(root, sealed_root=sealed)
        after = frozen_artifact_snapshot(root)
        if after != before:
            raise DeploymentManifestError(
                "FINAL_AUTHORITY_INVALID — PARQUET_MUTATED"
            )
        require_repository_identity(
            root, expected_branch=expected_branch, expected_head=expected_head
        )
    except BaseException:
        _restore_metadata(previous)
        raise
    return {
        "status": "passed",
        "preflight_status": result["preflight_status"],
        "datasets_ready": result["datasets_ready"],
        "datasets_total": result["datasets_total"],
        "manifest_path": str(sealed / "deployment-manifest.json"),
        "manifest_sha256": manifest_sha256,
        "root_identity_sha256": manifest["root_identity_sha256"],
        "code_inventory_sha256": inventory["inventory_sha256"],
        "formal_proofs": {
            key: sha256_bytes(pretty_json_bytes(value)) for key, value in proofs.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = build(
            args.repository_root,
            expected_branch=args.expected_branch,
            expected_head=args.expected_head,
            sealed_root=args.sealed_root,
        )
    except DeploymentManifestError as exc:
        print(json.dumps({"status": "failed", "error_code": exc.code, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
