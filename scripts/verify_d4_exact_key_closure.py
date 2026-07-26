#!/usr/bin/env python3
"""Verify the regenerated D4 exact composite-key selection authorities."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.protocols.candidate_pool import (  # noqa: E402
    SelectionEntry,
    build_candidate_pool_digest,
    build_consumer_fingerprint,
    build_selection_result_digest,
    build_source_pool_fingerprint,
)
from src.protocols.gate1_transformation import canonical_digest  # noqa: E402
from src.protocols.formal_input_paths import resolve_formal_dataset_paths  # noqa: E402


TARGETS = ("166_258", "166_432", "166_433", "166_313", "166_311")
EXPECTED_PARQUET_SHA256 = {
    "source": "17a1fa5bd1dddfd46bda2a6922ff7821aee2a7e79deca58a94ff7bf20821f7ef",
    "target": "f0b83798ea265c6b79f09487903404c7c75acfcac2657f53e989ef59588e5946",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _selection_entries(rows: list[Mapping[str, Any]]) -> tuple[SelectionEntry, ...]:
    return tuple(
        SelectionEntry(
            rank=int(row["source_rank"]),
            source_key=tuple(str(part) for part in row["source_key"]),
            distance=float(row["distance"]),
            weight=float(row["weight"]),
            tie_group=int(row["tie_group"]),
            observed_start="",
            observed_end="",
            raw_vector=(),
            scaled_vector=(),
        )
        for row in rows
    )


def _expected_exact_key_proof(
    *,
    target_key: tuple[str, str],
    candidate_keys: set[tuple[str, str]],
) -> dict[str, Any]:
    target_store, target_product = target_key
    cross_store_same_product = sum(
        key[0] != target_store and key[1] == target_product
        for key in candidate_keys
    )
    return {
        "entity_key_fields": ["store_id", "product_id"],
        "exact_target_tuple": list(target_key),
        "exact_target_tuple_excluded": target_key not in candidate_keys,
        "cross_store_same_product_available_count": cross_store_same_product,
        "cross_store_same_product_retained_count": cross_store_same_product,
        "same_store_other_product_retained_count": sum(
            key[0] == target_store and key[1] != target_product
            for key in candidate_keys
        ),
        "cross_store_other_product_retained_count": sum(
            key[0] != target_store and key[1] != target_product
            for key in candidate_keys
        ),
        "candidate_digest_verified": True,
        "consumer_fingerprint_verified": True,
    }


def _verify_scenario(scenario: str) -> dict[str, Any]:
    path = (
        ROOT
        / "configs/solidified/knn/Dataset4"
        / f"knn_{scenario}_info_sharing.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("info_sharing") != scenario:
        raise AssertionError(f"D4 scenario mismatch: {path}")
    if payload.get("group_cols") != ["store_id", "product_id"]:
        raise AssertionError("D4 authority does not use the full composite key")
    if tuple(payload.get("results", {})) != TARGETS:
        raise AssertionError("D4 target order or identity drifted")

    metadata_by_target = payload.get("selection_metadata", {})
    target_summary: dict[str, Any] = {}
    for target_id in TARGETS:
        metadata = metadata_by_target[target_id]
        digest_input = metadata["candidate_pool_digest_input"]
        target_key = tuple(str(part) for part in digest_input["target_key"])
        if target_key != tuple(target_id.split("_", 1)):
            raise AssertionError(f"D4 target digest identity mismatch: {target_id}")
        candidate_rows = [
            tuple(str(part) for part in key)
            for key in digest_input["candidate_keys"]
        ]
        if any(len(key) != 2 for key in candidate_rows):
            raise AssertionError(f"D4 candidate key arity drift: {scenario}/{target_id}")
        if len(candidate_rows) != len(set(candidate_rows)):
            raise AssertionError(f"D4 duplicate composite candidate: {scenario}/{target_id}")
        candidate_keys = set(candidate_rows)
        if target_key in candidate_keys:
            raise AssertionError(f"D4 target entered candidate pool: {scenario}/{target_id}")
        if scenario == "without" and any(
            key[0] != target_key[0] for key in candidate_keys
        ):
            raise AssertionError(f"D4 without-sharing left the legal store range: {target_id}")

        candidate_digest = build_candidate_pool_digest(**digest_input)
        if candidate_digest != metadata["candidate_pool_digest"]:
            raise AssertionError(f"D4 candidate digest mismatch: {scenario}/{target_id}")
        source_pool_fingerprint = build_source_pool_fingerprint(
            protocol_version=metadata["protocol_version"],
            dataset_id="D4",
            scenario=scenario,
            target_key=target_key,
            group_cols=("store_id", "product_id"),
            candidate_keys=candidate_rows,
        )
        if source_pool_fingerprint != metadata["source_pool_fingerprint"]:
            raise AssertionError(
                f"D4 source-pool fingerprint mismatch: {scenario}/{target_id}"
            )

        selected_sources = metadata["selected_sources_runtime"]
        selection_digest = build_selection_result_digest(
            protocol_version=metadata["protocol_version"],
            candidate_pool_digest=candidate_digest,
            k=int(metadata["requested_k"]),
            weight_mode=metadata["weight_mode"],
            weight_epsilon=1e-8,
            entries=_selection_entries(selected_sources),
        )
        if selection_digest != metadata["selection_result_digest"]:
            raise AssertionError(f"D4 selection digest mismatch: {scenario}/{target_id}")
        if any(
            tuple(str(part) for part in row["source_key"]) == target_key
            for row in selected_sources
        ):
            raise AssertionError(f"D4 Top-K contains target: {scenario}/{target_id}")

        consumer_fingerprint = build_consumer_fingerprint(
            protocol_version=metadata["protocol_version"],
            dataset_id="D4",
            scenario=scenario,
            target_key=target_key,
            source_pool_fingerprint=source_pool_fingerprint,
            candidate_pool_digest=candidate_digest,
            selection_result_digest=selection_digest,
            ordered_top_k=selected_sources,
        )
        if consumer_fingerprint != metadata["consumer_fingerprint"]:
            raise AssertionError(f"D4 consumer fingerprint mismatch: {scenario}/{target_id}")

        proof = _expected_exact_key_proof(
            target_key=target_key,
            candidate_keys=candidate_keys,
        )
        if metadata.get("d4_validation_proof") != proof:
            raise AssertionError(f"D4 validation proof mismatch: {scenario}/{target_id}")
        if metadata.get("d4_validation_proof_digest") != canonical_digest(proof):
            raise AssertionError(
                f"D4 validation proof digest mismatch: {scenario}/{target_id}"
            )
        if not proof["same_store_other_product_retained_count"]:
            raise AssertionError(f"D4 same-store other-product proof missing: {target_id}")
        if scenario == "with" and not (
            proof["cross_store_same_product_retained_count"] > 0
            and proof["cross_store_other_product_retained_count"] > 0
        ):
            raise AssertionError(f"D4 cross-store exact-key proof missing: {target_id}")

        target_summary[target_id] = {
            "candidate_count": len(candidate_keys),
            "cross_store_same_product_available_count": proof[
                "cross_store_same_product_available_count"
            ],
            "cross_store_same_product_retained_count": proof[
                "cross_store_same_product_retained_count"
            ],
            "candidate_pool_digest": candidate_digest,
            "selection_result_digest": selection_digest,
            "source_pool_fingerprint": source_pool_fingerprint,
            "consumer_fingerprint": consumer_fingerprint,
            "top_k": ["_".join(str(part) for part in row["source_key"]) for row in selected_sources],
        }

    manifest = payload.get("d4_manifest_identity")
    if not isinstance(manifest, dict):
        raise AssertionError(f"D4 manifest identity missing: {scenario}")
    manifest_payload = {
        key: value for key, value in manifest.items() if key != "manifest_identity_digest"
    }
    if manifest.get("manifest_identity_digest") != canonical_digest(manifest_payload):
        raise AssertionError(f"D4 manifest identity digest mismatch: {scenario}")
    for role, expected_hash in EXPECTED_PARQUET_SHA256.items():
        if manifest.get(f"{role}_parquet_sha256") != expected_hash:
            raise AssertionError(f"D4 manifest {role} parquet identity mismatch")
    for target_id, summary in target_summary.items():
        metadata = metadata_by_target[target_id]
        expected_consumer = {
            "candidate_pool_digest": summary["candidate_pool_digest"],
            "selection_result_digest": summary["selection_result_digest"],
            "source_pool_fingerprint": summary["source_pool_fingerprint"],
            "consumer_fingerprint": summary["consumer_fingerprint"],
            "validation_proof_digest": metadata["d4_validation_proof_digest"],
        }
        if manifest["target_consumers"].get(target_id) != expected_consumer:
            raise AssertionError(f"D4 manifest consumer mismatch: {scenario}/{target_id}")

    return {
        "authority_path": str(path),
        "authority_sha256": _sha256_file(path),
        "manifest_identity_digest": manifest["manifest_identity_digest"],
        "targets": target_summary,
    }


def main() -> int:
    formal_paths = resolve_formal_dataset_paths(4, repository_root=ROOT)
    source_path = formal_paths.source_path
    target_path = formal_paths.target_path
    if _sha256_file(source_path) != EXPECTED_PARQUET_SHA256["source"]:
        raise AssertionError("D4 source parquet changed")
    if _sha256_file(target_path) != EXPECTED_PARQUET_SHA256["target"]:
        raise AssertionError("D4 target parquet changed")

    report = {
        "status": "passed",
        "entity_key_fields": ["store_id", "product_id"],
        "scenarios": {
            scenario: _verify_scenario(scenario)
            for scenario in ("without", "with")
        },
    }
    if (
        report["scenarios"]["without"]["manifest_identity_digest"]
        == report["scenarios"]["with"]["manifest_identity_digest"]
    ):
        raise AssertionError("D4 with/without manifest identities collided")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("D4 EXACT-KEY VERIFICATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
