"""Fail-closed root identity for the sealed D1-D6 formal inputs.

This module owns the deterministic schemas and byte-level validation used by
the builder, readiness command, unified dry-run, and final acceptance.  It
never writes parquet, regenerates KNN data, trains, predicts, or publishes an
experiment result.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import date
from pathlib import Path, PurePosixPath, PureWindowsPath
import subprocess
from typing import Any, Mapping, Sequence

import pyarrow.parquet as pq

from src.constants import (
    D4_D6_RUNTIME_KNN_PROTOCOL_VERSION,
    SOURCE_HISTORY_CALENDAR,
    SOURCE_HISTORY_COMPLETENESS_POLICY,
    SOURCE_HISTORY_DAYS,
)

from src.protocols.formal_input_paths import (
    FORMAL_SEALED_ROOT_RELATIVE,
    resolve_all_formal_dataset_paths,
)
from src.protocols.gate1_transformation import (
    COMBINED_FORMAL_IDENTITY_DIGEST,
    CONTRACT_DIGEST,
    DECISION_BOOK_SHA256,
    FREEZE_COMMIT_SHA,
    MATRIX_SHA256,
    SCOPE_SHA256,
    dataset_contract,
    load_formal_identity,
)
from src.protocols.experiment_protocol import PROTOCOL_VERSION, get_experiment_protocol


MANIFEST_SCHEMA_VERSION = "d1_d6_deployment_manifest_v1"
PROOF_SCHEMA_VERSION = "d1_d6_formal_proof_v1"
INVENTORY_SCHEMA_VERSION = "d1_d6_code_inventory_v1"
EXPECTED_BRANCH = "codex/zuihou"
EXPECTED_HEAD = "3c4e06b17d660d344ed9a0a75d9489874e19d900"

FROZEN_PARQUETS: dict[int, dict[str, dict[str, object]]] = {
    1: {"source": {"rows": 49302, "size_bytes": 929130, "sha256": "18acf992b569f3e01999db2c94a3e85ca5e8fa0c7e760b111dd43825acd3fd08"}, "target": {"rows": 1826, "size_bytes": 56098, "sha256": "057e8dcef179adb4d81625476905bb6f5f4f41a2157e506155a16cf27aec39ec"}},
    2: {"source": {"rows": 48654, "size_bytes": 126090, "sha256": "63134a5900eb020c4b81489a881e2c78c9c21f18ce8ca9de16f0ca9e5af369a6"}, "target": {"rows": 1807, "size_bytes": 25053, "sha256": "d2bb78a71cccc0012f0f4f5175d80615565078b0cf7328d6741ab11063ec93c3"}},
    3: {"source": {"rows": 26766, "size_bytes": 137393, "sha256": "95d49f88390e415c172d482ffc2f1e9897af11bee88373db2cd87a224f0a79c3"}, "target": {"rows": 942, "size_bytes": 23827, "sha256": "9bd91ae611888315e4dffaff4680dae94bdf6c6b6740dda2aa0ed7be1a0496ce"}},
    4: {"source": {"rows": 7935702, "size_bytes": 117546804, "sha256": "17a1fa5bd1dddfd46bda2a6922ff7821aee2a7e79deca58a94ff7bf20821f7ef"}, "target": {"rows": 3847, "size_bytes": 83217, "sha256": "f0b83798ea265c6b79f09487903404c7c75acfcac2657f53e989ef59588e5946"}},
    5: {"source": {"rows": 45401018, "size_bytes": 282097362, "sha256": "368d896b0a7d2849ba2984e1bb3f4f07d36cf9175087ee755d6cd2a6c1c790b3"}, "target": {"rows": 7338, "size_bytes": 71791, "sha256": "e01cdb828b0669b6aab9cb7e92f60c2a6a283ad8cdb3f5ec9bbd6ad61d11f263"}},
    6: {"source": {"rows": 15964725, "size_bytes": 18392840, "sha256": "12297d9b00a8e40fd9f13966e41aa558fe15e00079f8d0b23477bf4e7c288458"}, "target": {"rows": 9705, "size_bytes": 53532, "sha256": "233540a998e980a5ecf1fe82d4e871637673fe5cf49d8717f9f64bee2aab04b1"}},
}

D4_KNN = {
    "with": {"path": "configs/solidified/knn/Dataset4/knn_with_info_sharing.json"},
    "without": {"path": "configs/solidified/knn/Dataset4/knn_without_info_sharing.json"},
}

D4_D6_KNN = {
    4: D4_KNN,
    5: {
        "with": {"path": "configs/solidified/knn/Dataset5/knn_with_info_sharing.json"},
        "without": {"path": "configs/solidified/knn/Dataset5/knn_without_info_sharing.json"},
    },
    6: {
        "with": {"path": "configs/solidified/knn/Dataset6/knn_with_info_sharing.json"},
        "without": {"path": "configs/solidified/knn/Dataset6/knn_without_info_sharing.json"},
    },
}

D1_D2_KNN = {
    1: {
        "with": {
            "path": "configs/solidified/knn/Dataset1/knn_with_info_sharing.json",
            "sha256": "03828cc63de28a329fdb32a7bc0f15bd201b962471197c1abe14575c1013666c",
        },
        "without": {
            "path": "configs/solidified/knn/Dataset1/knn_without_info_sharing.json",
            "sha256": "cb27e059711375ab4726a9ec1bb3ad1e3da32955e2cd49088183cc1547e68367",
        },
    },
    2: {
        "with": {
            "path": "configs/solidified/knn/Dataset2/knn_with_info_sharing.json",
            "sha256": "35bbbadc70db98781aa29829b567b5dc4887a83bfc5258214d429119380f652e",
        },
        "without": {
            "path": "configs/solidified/knn/Dataset2/knn_without_info_sharing.json",
            "sha256": "a2197ee8ceccd0382276bc27213934db2622305cef4485522129e45e77d43e1b",
        },
    },
}

D2_FROZEN_DATES = ("2018-04-01", "2018-04-25", "2018-05-01", "2018-06-02")

REQUIRED_BLOCKING_FIELDS = frozenset(
    {
        "path",
        "sha256",
        "size_bytes",
        "rows",
        "ordered_columns",
        "dtypes",
        "nullable",
        "schema_digest",
        "schema",
    }
)
OPTIONAL_DIAGNOSTIC_FIELDS = frozenset({"mtime_ns"})
KNOWN_IDENTITY_FIELDS = REQUIRED_BLOCKING_FIELDS | OPTIONAL_DIAGNOSTIC_FIELDS
_MISSING_IDENTITY_FIELD = object()

FORMAL_PROOF_AUTHORITY_FIELDS = frozenset(
    {
        "schema_version",
        "dataset_id",
        "dataset_root",
        "source",
        "target",
        "source_schema",
        "target_schema",
        "entity_key",
        "date_key",
        "row_count",
        "entity_count",
        "date_range",
        "key_date_uniqueness",
        "consumer_fingerprint",
        "consumer_fingerprint_input",
        "dataset_manifest_identity",
        "supporting_proof_files",
        "readiness_result",
        "readiness_proof_digest",
        "formal_identity",
        "dataset_specific",
    }
)
FORMAL_PROOF_DIGEST_FIELD = "proof_identity_sha256"
FORMAL_PROOF_NON_AUTHORITY_FIELDS = frozenset(
    {
        "diagnostic",
        "diagnostics",
        "parquet_identity_diagnostics",
        "warning",
        "warnings",
        "runtime_metadata",
        "runtime_meta",
        "runtime_info",
        "runtime_diagnostics",
    }
)
FORMAL_PROOF_KNOWN_FIELDS = (
    FORMAL_PROOF_AUTHORITY_FIELDS
    | {FORMAL_PROOF_DIGEST_FIELD}
    | FORMAL_PROOF_NON_AUTHORITY_FIELDS
)
_SEMANTIC_NON_AUTHORITY_KEYS = frozenset(
    {
        "mtime_ns",
        *FORMAL_PROOF_NON_AUTHORITY_FIELDS,
    }
)


class DeploymentManifestError(RuntimeError):
    """Stable final-preflight error with a machine-readable code."""

    def __init__(self, code: str, detail: str | None = None) -> None:
        self.code = str(code)
        self.detail = detail
        super().__init__(self.code if detail is None else f"{self.code}: {detail}")


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


READINESS_PROOF_PAYLOAD_SCHEMA_VERSION = "d1_d6_readiness_proof_payload_v1"
_READINESS_PATH_FIELDS = frozenset(
    {
        "authority_path",
        "code_inventory_path",
        "dataset_root",
        "formal_proof_path",
        "manifest_path",
        "path",
        "root_manifest_path",
        "sealed_root",
        "source_path",
        "target_path",
    }
)
_READINESS_FORMAL_PATH_ROOTS = frozenset(
    {"formal_identity", "formal_input", "sealed_identity", "selection_authority", "source_selection"}
)
_READINESS_DEPLOYMENT_DIAGNOSTIC_ROOTS = frozenset({"raw_inputs"})
_READINESS_PATH_OMITTED = object()


def _readiness_path_is_formal(path: tuple[str, ...]) -> bool:
    return bool(path and path[0] in _READINESS_FORMAL_PATH_ROOTS)


def _canonical_readiness_path(
    value: object,
    *,
    repository_root: Path,
    formal: bool,
) -> str | object:
    if not isinstance(value, str) or not value:
        raise DeploymentManifestError("READINESS_PROOF_PATH_INVALID", repr(value))
    normalized = value.replace("\\", "/")
    posix = PurePosixPath(normalized)
    windows = PureWindowsPath(value)
    if windows.is_absolute() and not posix.is_absolute():
        if formal:
            raise DeploymentManifestError("READINESS_PROOF_PATH_INVALID", value)
        return _READINESS_PATH_OMITTED
    if posix.is_absolute():
        try:
            relative = Path(normalized).resolve(strict=False).relative_to(repository_root)
        except (OSError, ValueError) as exc:
            if formal:
                raise DeploymentManifestError("READINESS_PROOF_PATH_INVALID", value) from exc
            return _READINESS_PATH_OMITTED
        return PurePosixPath(*relative.parts).as_posix()
    if ".." in posix.parts:
        if formal:
            raise DeploymentManifestError("READINESS_PROOF_PATH_INVALID", value)
        return _READINESS_PATH_OMITTED
    return posix.as_posix()


def canonical_readiness_proof_payload(
    readiness: Mapping[str, object],
    *,
    repository_root: Path,
) -> dict[str, object]:
    """Build the machine-independent, identity-preserving readiness payload."""

    if not isinstance(readiness, Mapping):
        raise DeploymentManifestError("READINESS_PROOF_PAYLOAD_INVALID", "mapping required")
    root = Path(repository_root).resolve(strict=True)

    def normalize(value: object, path: tuple[str, ...]) -> object:
        if isinstance(value, Mapping):
            result: dict[str, object] = {}
            for key, item in value.items():
                name = str(key)
                if not path and name in _READINESS_DEPLOYMENT_DIAGNOSTIC_ROOTS:
                    continue
                child_path = path + (name,)
                if name in _READINESS_PATH_FIELDS:
                    item = _canonical_readiness_path(
                        item,
                        repository_root=root,
                        formal=_readiness_path_is_formal(path),
                    )
                    if item is _READINESS_PATH_OMITTED:
                        continue
                else:
                    item = normalize(item, child_path)
                result[name] = item
            return result
        if isinstance(value, list):
            return [normalize(item, path + (str(index),)) for index, item in enumerate(value)]
        if isinstance(value, tuple):
            return [normalize(item, path + (str(index),)) for index, item in enumerate(value)]
        return value

    return {
        "schema_version": READINESS_PROOF_PAYLOAD_SCHEMA_VERSION,
        "readiness": normalize(readiness, ()),
    }


def readiness_proof_digest(
    readiness: Mapping[str, object],
    *,
    repository_root: Path,
) -> str:
    """Digest exactly the canonical readiness payload used by proof and preflight."""

    return sha256_bytes(
        canonical_json_bytes(
            canonical_readiness_proof_payload(
                readiness,
                repository_root=repository_root,
            )
        )
    )


def pretty_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, code: str) -> dict[str, object]:
    try:
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DeploymentManifestError(code, str(path)) from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DeploymentManifestError(code, f"{path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise DeploymentManifestError(code, f"JSON object required: {path}")
    return loaded


def _date_identity(value: object) -> str:
    """Normalize a persisted ISO date or fail closed at the manifest boundary."""
    try:
        return date.fromisoformat(str(value)).isoformat()
    except (TypeError, ValueError) as exc:
        raise DeploymentManifestError("D4_D6_KNN_METADATA_MISMATCH", f"invalid date: {value!r}") from exc


def _verify_d4_d6_knn_payload(path: Path, *, dataset_id: int) -> dict[str, object]:
    """Validate persisted D4-D6 selection identity without regenerating it."""
    payload = _json(path, "D4_D6_KNN_AUTHORITY_UNREADABLE")
    expected_top_level = {
        "source_history_days": SOURCE_HISTORY_DAYS,
        "source_history_expected_date_count": SOURCE_HISTORY_DAYS,
        "source_history_start": dataset_contract(dataset_id).source_history_start.isoformat(),
        "source_history_end": dataset_contract(dataset_id).source_history_end.isoformat(),
        "source_history_calendar": SOURCE_HISTORY_CALENDAR,
        "source_history_inclusive_end": True,
        "source_history_completeness_policy": SOURCE_HISTORY_COMPLETENESS_POLICY,
        "source_history_calendarization_rule": (
            "D5_APPROVED_SOURCE_HISTORY_CALENDARIZATION"
            if dataset_id == 5
            else "not_applicable"
        ),
    }
    if any(payload.get(field) != value for field, value in expected_top_level.items()):
        raise DeploymentManifestError("D4_D6_SOURCE_HISTORY_IDENTITY_MISMATCH", str(path))
    expected_authority = "shared_protocol" if dataset_id in {4, 5} else "runtime"
    expected_protocol = PROTOCOL_VERSION if dataset_id in {4, 5} else D4_D6_RUNTIME_KNN_PROTOCOL_VERSION
    expected_features = list(get_experiment_protocol(dataset_id).knn_feature_columns)
    metadata_map = payload.get("selection_metadata")
    if not isinstance(metadata_map, Mapping) or not metadata_map:
        raise DeploymentManifestError("D4_D6_KNN_METADATA_MISSING", str(path))
    targets: dict[str, object] = {}
    from src.protocols.candidate_pool import (
        build_candidate_pool_digest,
        build_consumer_fingerprint,
        build_source_pool_fingerprint,
    )

    spec = dataset_contract(dataset_id)
    for target_id, metadata in sorted(metadata_map.items()):
        if not isinstance(metadata, Mapping):
            raise DeploymentManifestError("D4_D6_KNN_METADATA_INVALID", str(path))
        if (
            metadata.get("selection_authority") != expected_authority
            or metadata.get("protocol_version") != expected_protocol
            or metadata.get("knn_feature_columns") != expected_features
            or metadata.get("historical_feature_columns") != expected_features
            or metadata.get("forecast_excluded_columns") != []
            or metadata.get("feature_scope") != "historical_observed"
            or metadata.get("max_allowed_date_relation") != "date<=origin"
        ):
            raise DeploymentManifestError("D4_D6_KNN_METADATA_MISMATCH", f"{path}:{target_id}")
        observed_start = metadata.get("knn_observed_start")
        observed_end = metadata.get("knn_observed_end")
        expected_window = get_experiment_protocol(dataset_id).observation_window(
            observed_start
        ) if observed_start is not None else None
        if (
            expected_window is None
            or _date_identity(observed_start) != expected_window.knn_observed_start.isoformat()
            or _date_identity(observed_end) != expected_window.knn_observed_end.isoformat()
        ):
            raise DeploymentManifestError("D4_D6_KNN_METADATA_MISMATCH", f"{path}:{target_id}")
        for field, expected in (
            ("source_history_days", SOURCE_HISTORY_DAYS),
            ("source_history_expected_date_count", SOURCE_HISTORY_DAYS),
            ("source_history_calendar", SOURCE_HISTORY_CALENDAR),
            ("source_history_completeness_policy", SOURCE_HISTORY_COMPLETENESS_POLICY),
            ("source_history_inclusive_end", True),
            (
                "source_history_calendarization_rule",
                "D5_APPROVED_SOURCE_HISTORY_CALENDARIZATION"
                if dataset_id == 5
                else "not_applicable",
            ),
        ):
            if metadata.get(field) != expected:
                raise DeploymentManifestError("D4_D6_SOURCE_HISTORY_IDENTITY_MISMATCH", f"{path}:{target_id}")
        digest_input = metadata.get("candidate_pool_digest_input")
        if not isinstance(digest_input, Mapping):
            raise DeploymentManifestError("D4_D6_CANDIDATE_DIGEST_INPUT_MISSING", f"{path}:{target_id}")
        if (
            digest_input.get("source_history_days") != SOURCE_HISTORY_DAYS
            or digest_input.get("source_history_completeness_policy") != SOURCE_HISTORY_COMPLETENESS_POLICY
            or digest_input.get("source_history_start") != spec.source_history_start.isoformat()
            or digest_input.get("source_history_end") != spec.source_history_end.isoformat()
            or digest_input.get("source_history_frame_digest") != metadata.get("source_history_frame_digest")
        ):
            raise DeploymentManifestError("D4_D6_CANDIDATE_HISTORY_IDENTITY_MISMATCH", f"{path}:{target_id}")
        candidate_keys = digest_input.get("candidate_keys")
        target_key = digest_input.get("target_key")
        if not isinstance(candidate_keys, list) or not isinstance(target_key, list):
            raise DeploymentManifestError("D4_D6_CANDIDATE_POOL_INVALID", f"{path}:{target_id}")
        normalized_candidates = {
            tuple(str(part) for part in key)
            for key in candidate_keys
            if isinstance(key, list)
        }
        normalized_target = tuple(str(part) for part in target_key)
        if len(normalized_candidates) != len(candidate_keys) or normalized_target in normalized_candidates:
            raise DeploymentManifestError("D4_D6_CANDIDATE_POOL_INVALID", f"{path}:{target_id}")
        try:
            candidate_digest = build_candidate_pool_digest(**dict(digest_input))
        except Exception as exc:
            raise DeploymentManifestError("D4_D6_CANDIDATE_DIGEST_INVALID", f"{path}:{target_id}: {exc}") from exc
        if candidate_digest != metadata.get("candidate_pool_digest"):
            raise DeploymentManifestError("D4_D6_CANDIDATE_DIGEST_INVALID", f"{path}:{target_id}")
        selected = metadata.get("selected_sources_runtime")
        if not isinstance(selected, list) or len(selected) != int(payload.get("k", 0)):
            raise DeploymentManifestError("D4_D6_SELECTED_SOURCE_INVALID", f"{path}:{target_id}")
        if metadata.get("consumer_frame_rows") != len(selected) * SOURCE_HISTORY_DAYS:
            raise DeploymentManifestError("D4_D6_CONSUMER_FRAME_CARDINALITY", f"{path}:{target_id}")
        source_pool_fingerprint = None
        if dataset_id == 5:
            source_pool_fingerprint = build_source_pool_fingerprint(
                protocol_version=metadata["protocol_version"],
                dataset_id=f"D{dataset_id}",
                scenario=metadata.get("protocol_scenario", payload.get("info_sharing", "")),
                target_key=target_key,
                group_cols=digest_input.get("group_cols", []),
                candidate_keys=candidate_keys,
            )
            if source_pool_fingerprint != metadata.get("source_pool_fingerprint"):
                raise DeploymentManifestError("D5_SOURCE_POOL_FINGERPRINT_INVALID", f"{path}:{target_id}")
        selected_keys = {
            tuple(str(part) for part in row.get("source_key", []))
            for row in selected
            if isinstance(row, Mapping)
        }
        if len(selected_keys) != len(selected) or not selected_keys.issubset(normalized_candidates):
            raise DeploymentManifestError("D4_D6_SELECTED_SOURCE_OUTSIDE_POOL", f"{path}:{target_id}")
        if dataset_id == 5:
            consumer_fingerprint = build_consumer_fingerprint(
                protocol_version=metadata["protocol_version"],
                dataset_id=f"D{dataset_id}",
                scenario=metadata.get("protocol_scenario", payload.get("info_sharing", "")),
                target_key=target_key,
                source_pool_fingerprint=source_pool_fingerprint,
                candidate_pool_digest=candidate_digest,
                selection_result_digest=metadata.get("selection_result_digest", ""),
                ordered_top_k=selected,
            )
            if consumer_fingerprint != metadata.get("consumer_fingerprint"):
                raise DeploymentManifestError("D5_CONSUMER_FINGERPRINT_INVALID", f"{path}:{target_id}")
        for field in ("source_history_frame_digest", "consumer_frame_digest", "candidate_pool_digest", "selection_result_digest"):
            value = metadata.get(field)
            if not isinstance(value, str) or len(value) != 64:
                raise DeploymentManifestError("D4_D6_DIGEST_MISSING", f"{path}:{target_id}/{field}")
        if dataset_id == 4 and ("729", "424") in normalized_candidates.union(selected_keys):
            raise DeploymentManifestError("D4_INCOMPLETE_CANDIDATE_PRESENT", f"{path}:{target_id}")
        targets[str(target_id)] = {
            "target_key": list(normalized_target),
            "candidate_pool_digest": candidate_digest,
            "selection_result_digest": metadata["selection_result_digest"],
            "source_history_frame_digest": metadata["source_history_frame_digest"],
            "consumer_frame_digest": metadata["consumer_frame_digest"],
            "consumer_frame_rows": metadata["consumer_frame_rows"],
        }
    if payload.get("selection_authority") != expected_authority:
        raise DeploymentManifestError("D4_D6_KNN_AUTHORITY_MISMATCH", str(path))
    if payload.get("protocol_version") != expected_protocol:
        raise DeploymentManifestError("D4_D6_KNN_PROTOCOL_MISMATCH", str(path))
    if dataset_id == 5:
        actual_features = payload.get("knn_feature_columns")
        if actual_features != expected_features:
            raise DeploymentManifestError(
                "D5_KNN_FEATURE_CONTRACT_MISMATCH",
                f"dataset=D5 expected features={expected_features!r} actual features={actual_features!r} "
                f"authority_path={path} consumer_fingerprint=<not-read>",
            )
        if payload.get("knn_frame_authority") != "configured_observed_frame":
            raise DeploymentManifestError(
                "D5_KNN_FRAME_AUTHORITY_MISMATCH",
                f"dataset=D5 authority_path={path} consumer_fingerprint=<not-read>",
            )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "source_history_days": SOURCE_HISTORY_DAYS,
        "source_history_start": spec.source_history_start.isoformat(),
        "source_history_end": spec.source_history_end.isoformat(),
        "source_history_expected_date_count": SOURCE_HISTORY_DAYS,
        "source_history_calendar": SOURCE_HISTORY_CALENDAR,
        "source_history_completeness_policy": SOURCE_HISTORY_COMPLETENESS_POLICY,
        "source_history_inclusive_end": True,
        "targets": targets,
    }


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve(strict=True).relative_to(root.resolve(strict=True)).as_posix()
    except (OSError, ValueError) as exc:
        raise DeploymentManifestError("FORMAL_PATH_OUTSIDE_REPOSITORY", str(path)) from exc


def _safe_relative(value: object, *, code: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise DeploymentManifestError(code, "relative POSIX path required")
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or ".." in candidate.parts or "\\" in value:
        raise DeploymentManifestError(code, value)
    return candidate


def _resolve_bound_path(root: Path, value: object, *, code: str) -> Path:
    relative = _safe_relative(value, code=code)
    try:
        resolved = (root / Path(*relative.parts)).resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise DeploymentManifestError(code, str(value)) from exc
    if resolved.is_symlink() or not resolved.is_file():
        raise DeploymentManifestError(code, str(value))
    return resolved


def formal_identity_payload(repository_root: Path) -> dict[str, str]:
    identity = load_formal_identity(Path(repository_root))
    actual = {
        "decision_book_sha256": str(identity.get("decision_book_sha256")),
        "contract_sha256": str(identity.get("contract_digest")),
        "scope_sha256": str(identity.get("scope_sha256")),
        "matrix_sha256": str(identity.get("matrix_sha256")),
        "combined_identity_sha256": str(identity.get("combined_formal_identity_digest")),
        "freeze_commit": str(identity.get("freeze_commit_sha")),
    }
    expected = {
        "decision_book_sha256": DECISION_BOOK_SHA256,
        "contract_sha256": CONTRACT_DIGEST,
        "scope_sha256": SCOPE_SHA256,
        "matrix_sha256": MATRIX_SHA256,
        "combined_identity_sha256": COMBINED_FORMAL_IDENTITY_DIGEST,
        "freeze_commit": FREEZE_COMMIT_SHA,
    }
    if actual != expected:
        raise DeploymentManifestError("FORMAL_IDENTITY_MISMATCH", f"actual={actual}")
    return actual


def repository_identity(repository_root: Path) -> dict[str, str]:
    root = Path(repository_root).resolve(strict=True)
    branch = subprocess.check_output(
        ["git", "-C", str(root), "branch", "--show-current"], text=True
    ).strip()
    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    return {"branch": branch, "head": head}


def require_repository_identity(
    repository_root: Path,
    *,
    expected_branch: str = EXPECTED_BRANCH,
    expected_head: str = EXPECTED_HEAD,
) -> dict[str, str]:
    identity = repository_identity(repository_root)
    if identity["branch"] != expected_branch:
        raise DeploymentManifestError("BRANCH_MISMATCH", identity["branch"])
    if identity["head"] != expected_head:
        raise DeploymentManifestError("HEAD_MISMATCH", identity["head"])
    return identity


def schema_descriptor(
    path: Path,
    *,
    role: str,
    key_columns: Sequence[str],
    date_column: str = "date",
) -> dict[str, object]:
    parquet = pq.ParquetFile(path)
    columns = [
        {"name": field.name, "dtype": str(field.type), "nullable": bool(field.nullable)}
        for field in parquet.schema_arrow
    ]
    payload = {
        "role": role,
        "key_columns": list(key_columns),
        "date_column": date_column,
        "columns": columns,
    }
    return {**payload, "schema_digest": _schema_digest(payload)}


def _schema_digest(payload: Mapping[str, object]) -> str:
    """Hash the exact schema payload emitted by :func:`schema_descriptor`."""

    return sha256_bytes(canonical_json_bytes(dict(payload)))


def _is_non_empty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value)


def _is_repository_relative_posix_path(value: object) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    if value.startswith("/"):
        return False
    windows = PureWindowsPath(value)
    if windows.is_absolute() or bool(windows.drive):
        return False
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return False
    candidate = PurePosixPath(value)
    return not candidate.is_absolute() and candidate.as_posix() == value


def _is_sha256_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _is_non_negative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_ordered_string_sequence(value: object) -> bool:
    if not isinstance(value, (list, tuple)):
        return False
    if not all(_is_non_empty_string(item) for item in value):
        return False
    return len(set(value)) == len(value)


def _identity_error(
    *, side: str, field: str, reason: str, value: object = _MISSING_IDENTITY_FIELD
) -> dict[str, object]:
    error: dict[str, object] = {"side": side, "field": field, "reason": reason}
    if value is not _MISSING_IDENTITY_FIELD:
        error["value_type"] = type(value).__name__
    return error


def _validate_schema_structure(
    schema: object,
    *,
    side: str,
) -> list[dict[str, object]]:
    errors: list[dict[str, object]] = []
    if not isinstance(schema, Mapping):
        return [
            _identity_error(
                side=side, field="schema", reason="must be a mapping", value=schema
            )
        ]
    schema_values = dict(schema)
    required = {"role", "key_columns", "date_column", "columns", "schema_digest"}
    for field in sorted(required - set(schema_values)):
        errors.append(
            _identity_error(
                side=side,
                field="schema",
                reason=f"missing nested field {field}",
            )
        )
    for field in sorted(set(schema_values) - required):
        errors.append(
            _identity_error(
                side=side,
                field="schema",
                reason=f"unknown nested field {field!r}",
            )
        )
    if not _is_non_empty_string(schema_values.get("role")):
        errors.append(
            _identity_error(
                side=side, field="schema", reason="role must be a non-empty string"
            )
        )
    if not _is_ordered_string_sequence(schema_values.get("key_columns")):
        errors.append(
            _identity_error(
                side=side,
                field="schema",
                reason="key_columns must be an ordered string sequence",
            )
        )
    if not _is_non_empty_string(schema_values.get("date_column")):
        errors.append(
            _identity_error(
                side=side,
                field="schema",
                reason="date_column must be a non-empty string",
            )
        )
    columns = schema_values.get("columns")
    if not isinstance(columns, (list, tuple)):
        errors.append(
            _identity_error(
                side=side,
                field="schema",
                reason="columns must be an ordered sequence",
                value=columns,
            )
        )
    else:
        column_names: list[str] = []
        for index, column in enumerate(columns):
            if not isinstance(column, Mapping):
                errors.append(
                    _identity_error(
                        side=side,
                        field="schema",
                        reason=f"columns[{index}] must be a mapping",
                        value=column,
                    )
                )
                continue
            column_values = dict(column)
            required_column_fields = {"name", "dtype", "nullable"}
            for field in sorted(required_column_fields - set(column_values)):
                errors.append(
                    _identity_error(
                        side=side,
                        field="schema",
                        reason=f"columns[{index}] missing nested field {field}",
                    )
                )
            for field in sorted(set(column_values) - required_column_fields):
                errors.append(
                    _identity_error(
                        side=side,
                        field="schema",
                        reason=f"columns[{index}] has unknown nested field {field!r}",
                    )
                )
            name = column_values.get("name")
            if not _is_non_empty_string(name):
                errors.append(
                    _identity_error(
                        side=side,
                        field="schema",
                        reason=f"columns[{index}].name must be a non-empty string",
                    )
                )
            else:
                column_names.append(name)
            if not _is_non_empty_string(column_values.get("dtype")):
                errors.append(
                    _identity_error(
                        side=side,
                        field="schema",
                        reason=f"columns[{index}].dtype must be a non-empty string",
                    )
                )
            if not isinstance(column_values.get("nullable"), bool):
                errors.append(
                    _identity_error(
                        side=side,
                        field="schema",
                        reason=f"columns[{index}].nullable must be bool",
                    )
                )
        if len(column_names) != len(set(column_names)):
            errors.append(
                _identity_error(
                    side=side,
                    field="schema",
                    reason="columns must not contain duplicate names",
                )
            )
    if not _is_sha256_digest(schema_values.get("schema_digest")):
        errors.append(
            _identity_error(
                side=side,
                field="schema_digest",
                reason="schema_digest must be a SHA-256 digest",
            )
        )
    if not errors:
        schema_payload = {
            field: schema_values[field]
            for field in ("role", "key_columns", "date_column", "columns")
        }
        if schema_values["schema_digest"] != _schema_digest(schema_payload):
            errors.append(
                _identity_error(
                    side=side,
                    field="schema_digest",
                    reason="schema_digest does not match canonical schema payload",
                )
            )
    return errors


def _validate_parquet_identity(
    identity: object,
    *,
    side: str,
) -> tuple[dict[str, object] | None, list[dict[str, object]]]:
    if not isinstance(identity, Mapping):
        return (
            None,
            [
                _identity_error(
                    side=side,
                    field="identity",
                    reason="identity must be a mapping",
                    value=identity,
                )
            ],
        )
    try:
        values = dict(identity)
    except Exception as exc:  # pragma: no cover - defensive for custom mappings
        return (
            None,
            [
                _identity_error(
                    side=side,
                    field="identity",
                    reason=f"identity mapping could not be read: {type(exc).__name__}",
                )
            ],
        )

    errors: list[dict[str, object]] = []
    missing = REQUIRED_BLOCKING_FIELDS - set(values)
    unknown = set(values) - KNOWN_IDENTITY_FIELDS
    for field in sorted(missing):
        errors.append(
            _identity_error(
                side=side, field=field, reason="required blocking field is missing"
            )
        )
    for field in sorted(unknown, key=repr):
        errors.append(
            _identity_error(
                side=side,
                field=f"<unknown:{field!r}>",
                reason="unknown identity field",
            )
        )

    for field in sorted(set(values) & KNOWN_IDENTITY_FIELDS):
        value = values[field]
        reason: str | None = None
        if field == "path" and not _is_repository_relative_posix_path(value):
            reason = "must be a repository-relative POSIX path"
        elif field in {"sha256", "schema_digest"} and not _is_sha256_digest(value):
            reason = "must be a SHA-256 digest"
        elif field in {"size_bytes", "rows", "mtime_ns"} and not _is_non_negative_int(value):
            reason = "must be a non-negative integer and must not be bool"
        elif field == "ordered_columns" and not _is_ordered_string_sequence(value):
            reason = "must be an ordered sequence of unique non-empty strings"
        elif field == "dtypes":
            if not isinstance(value, Mapping):
                reason = "must be a mapping of column names to dtype strings"
            elif not all(
                _is_non_empty_string(key) and _is_non_empty_string(dtype)
                for key, dtype in value.items()
            ):
                reason = "must map non-empty strings to non-empty strings"
        elif field == "nullable":
            if not isinstance(value, Mapping):
                reason = "must be a mapping of column names to bool"
            elif not all(
                _is_non_empty_string(key) and isinstance(nullable, bool)
                for key, nullable in value.items()
            ):
                reason = "must map non-empty strings to bool"
        elif field == "schema":
            errors.extend(_validate_schema_structure(value, side=side))
        if reason is not None:
            errors.append(_identity_error(side=side, field=field, reason=reason, value=value))

    ordered_columns = values.get("ordered_columns")
    dtypes = values.get("dtypes")
    nullable = values.get("nullable")
    schema = values.get("schema")
    if _is_ordered_string_sequence(ordered_columns):
        expected_columns = set(ordered_columns)
        if isinstance(dtypes, Mapping) and set(dtypes) != expected_columns:
            errors.append(
                _identity_error(
                    side=side,
                    field="dtypes",
                    reason="keys must match ordered_columns",
                )
            )
        if isinstance(nullable, Mapping) and set(nullable) != expected_columns:
            errors.append(
                _identity_error(
                    side=side,
                    field="nullable",
                    reason="keys must match ordered_columns",
                )
            )
        if isinstance(schema, Mapping):
            schema_columns = schema.get("columns")
            if isinstance(schema_columns, (list, tuple)):
                schema_names = [
                    column.get("name")
                    for column in schema_columns
                    if isinstance(column, Mapping)
                ]
                if schema_names != list(ordered_columns):
                    errors.append(
                        _identity_error(
                            side=side,
                            field="schema",
                            reason="columns must match ordered_columns",
                        )
                    )
                if (
                    schema_names == list(ordered_columns)
                    and isinstance(dtypes, Mapping)
                    and set(dtypes) == expected_columns
                    and isinstance(nullable, Mapping)
                    and set(nullable) == expected_columns
                    and len(schema_columns) == len(ordered_columns)
                    and all(isinstance(column, Mapping) for column in schema_columns)
                ):
                    for column_name, column in zip(ordered_columns, schema_columns):
                        if column.get("dtype") != dtypes[column_name]:  # type: ignore[index]
                            errors.append(
                                _identity_error(
                                    side=side,
                                    field="dtypes",
                                    reason=(
                                        "schema column dtype must match top-level "
                                        f"dtypes[{column_name!r}]"
                                    ),
                                )
                            )
                        if column.get("nullable") != nullable[column_name]:  # type: ignore[index]
                            errors.append(
                                _identity_error(
                                    side=side,
                                    field="nullable",
                                    reason=(
                                        "schema column nullable must match top-level "
                                        f"nullable[{column_name!r}]"
                                    ),
                                )
                            )
    if isinstance(schema, Mapping):
        if schema.get("schema_digest") != values.get("schema_digest"):
            errors.append(
                _identity_error(
                    side=side,
                    field="schema",
                    reason="nested schema_digest must match schema_digest",
                )
            )
    return values, errors


def parquet_content_identity(identity: object) -> dict[str, object]:
    """Return the validated parquet identity used by content-level digests.

    Filesystem diagnostics such as ``mtime_ns`` are intentionally absent.  The
    identity is validated before projection so unknown or malformed fields
    cannot be silently discarded.
    """

    values, errors = _validate_parquet_identity(identity, side="content_projection")
    if errors or values is None:
        raise DeploymentManifestError(
            "PARQUET_IDENTITY_INVALID", repr(errors)
        )
    return {
        field: values[field]
        for field in sorted(REQUIRED_BLOCKING_FIELDS)
    }


def _strip_semantic_non_authority_metadata(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            key: _strip_semantic_non_authority_metadata(item)
            for key, item in value.items()
            if key not in _SEMANTIC_NON_AUTHORITY_KEYS
        }
    if isinstance(value, list):
        return [_strip_semantic_non_authority_metadata(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_strip_semantic_non_authority_metadata(item) for item in value)
    return value


def formal_proof_authority_payload(
    proof: Mapping[str, object],
) -> dict[str, object]:
    """Project a raw formal proof onto its semantic authority fields.

    The raw proof is validated before projection.  This is deliberately a
    strict allow-list: unknown proof fields cannot become silently accepted
    authority inputs.  ``mtime_ns`` and explicitly named diagnostic/warning/
    runtime metadata are non-authoritative and are removed only after the
    source and target identities have passed their full structural checks.
    """

    if not isinstance(proof, Mapping):
        raise DeploymentManifestError(
            "FORMAL_PROOF_STRUCTURE_INVALID", "proof must be a mapping"
        )
    try:
        values = dict(proof)
    except Exception as exc:  # pragma: no cover - defensive for custom mappings
        raise DeploymentManifestError(
            "FORMAL_PROOF_STRUCTURE_INVALID", type(exc).__name__
        ) from exc

    unknown = set(values) - FORMAL_PROOF_KNOWN_FIELDS
    if unknown:
        raise DeploymentManifestError(
            "FORMAL_PROOF_STRUCTURE_INVALID",
            f"unknown fields={sorted(unknown, key=repr)!r}",
        )
    missing = FORMAL_PROOF_AUTHORITY_FIELDS - set(values)
    if missing:
        raise DeploymentManifestError(
            "FORMAL_PROOF_STRUCTURE_INVALID",
            f"missing fields={sorted(missing)!r}",
        )
    if FORMAL_PROOF_DIGEST_FIELD in values and not _is_sha256_digest(
        values[FORMAL_PROOF_DIGEST_FIELD]
    ):
        raise DeploymentManifestError(
            "FORMAL_PROOF_STRUCTURE_INVALID", "proof digest must be SHA-256"
        )

    projected: dict[str, object] = {
        field: values[field] for field in sorted(FORMAL_PROOF_AUTHORITY_FIELDS)
    }
    for role, schema_field in (("source", "source_schema"), ("target", "target_schema")):
        projected_identity = parquet_content_identity(values[role])
        if projected[schema_field] != projected_identity["schema"]:
            raise DeploymentManifestError(
                "FORMAL_PROOF_PARQUET_MISMATCH",
                f"{role}_schema does not match {role} identity",
            )
        projected[role] = projected_identity
    return _strip_semantic_non_authority_metadata(projected)  # type: ignore[return-value]


def formal_proof_authority_digest(proof: Mapping[str, object]) -> str:
    """Hash exactly the semantic projection used by formal authority."""

    return sha256_bytes(canonical_json_bytes(formal_proof_authority_payload(proof)))


def _identity_values_equal(left: object, right: object) -> bool:
    try:
        return bool(left == right)
    except Exception:
        return False


def compare_parquet_identity(
    authority: object,
    current: object,
) -> dict[str, object]:
    """Compare parquet identity without treating filesystem metadata as content.

    Missing blocking fields remain fail-closed.  ``mtime_ns`` is optional for
    compatibility with older manifests and is only reported diagnostically
    when both sides contain different values.
    """

    expected_values, expected_errors = _validate_parquet_identity(
        authority, side="expected"
    )
    actual_values, actual_errors = _validate_parquet_identity(current, side="actual")
    structural_errors = [*expected_errors, *actual_errors]
    blocking_mismatches: dict[str, object] = {}
    if expected_values is not None and actual_values is not None:
        for field in sorted(REQUIRED_BLOCKING_FIELDS):
            expected_missing = field not in expected_values
            actual_missing = field not in actual_values
            if expected_missing or actual_missing or not _identity_values_equal(
                expected_values.get(field, _MISSING_IDENTITY_FIELD),
                actual_values.get(field, _MISSING_IDENTITY_FIELD),
            ):
                blocking_mismatches[field] = {
                    "authority": (
                        "<missing>"
                        if expected_missing
                        else expected_values[field]
                    ),
                    "current": (
                        "<missing>" if actual_missing else actual_values[field]
                    ),
                }

    diagnostics: list[dict[str, object]] = []
    for field in sorted(OPTIONAL_DIAGNOSTIC_FIELDS):
        expected_value = (
            expected_values.get(field, _MISSING_IDENTITY_FIELD)
            if expected_values is not None
            else _MISSING_IDENTITY_FIELD
        )
        actual_value = (
            actual_values.get(field, _MISSING_IDENTITY_FIELD)
            if actual_values is not None
            else _MISSING_IDENTITY_FIELD
        )
        if (
            not structural_errors
            and not blocking_mismatches
            and expected_value is not _MISSING_IDENTITY_FIELD
            and actual_value is not _MISSING_IDENTITY_FIELD
            and not _identity_values_equal(expected_value, actual_value)
        ):
            diagnostics.append(
                {
                    "code": "PARQUET_MTIME_DIFFERENCE_ACCEPTED",
                    "field": field,
                    "authority": expected_value,
                    "current": actual_value,
                }
            )
    return {
        "structural_errors": structural_errors,
        "blocking_mismatches": blocking_mismatches,
        "diagnostics": diagnostics,
    }


def parquet_identity(
    path: Path,
    *,
    repository_root: Path,
    role: str,
    key_columns: Sequence[str],
) -> dict[str, object]:
    resolved = Path(path).resolve(strict=True)
    stat = resolved.stat()
    parquet = pq.ParquetFile(resolved)
    schema = schema_descriptor(
        resolved, role=role, key_columns=key_columns, date_column="date"
    )
    return {
        "path": _relative(resolved, Path(repository_root)),
        "sha256": sha256_file(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "rows": int(parquet.metadata.num_rows),
        "ordered_columns": [item["name"] for item in schema["columns"]],
        "dtypes": {item["name"]: item["dtype"] for item in schema["columns"]},
        "nullable": {item["name"]: item["nullable"] for item in schema["columns"]},
        "schema_digest": schema["schema_digest"],
        "schema": schema,
    }


def frozen_artifact_snapshot(repository_root: Path) -> dict[str, object]:
    root = Path(repository_root).resolve(strict=True)
    datasets: dict[str, object] = {}
    for paths in resolve_all_formal_dataset_paths(repository_root=root):
        spec = dataset_contract(paths.dataset_id)
        datasets[f"D{paths.dataset_id}"] = {
            "source": parquet_identity(paths.source_path, repository_root=root, role="source", key_columns=spec.key_fields),
            "target": parquet_identity(paths.target_path, repository_root=root, role="target", key_columns=spec.key_fields),
        }
    knn = {
        scenario: {
            "path": item["path"],
            "sha256": sha256_file(root / str(item["path"])),
            "size_bytes": int((root / str(item["path"])).stat().st_size),
            "mtime_ns": int((root / str(item["path"])).stat().st_mtime_ns),
        }
        for scenario, item in D4_KNN.items()
    }
    d4_d6_knn: dict[str, object] = {}
    for dataset_id, scenarios in D4_D6_KNN.items():
        d4_d6_knn[f"D{dataset_id}"] = {
            scenario: _verify_d4_d6_knn_payload(
                root / str(item["path"]), dataset_id=dataset_id
            )
            for scenario, item in scenarios.items()
        }
    strict_knn: dict[str, object] = {}
    for dataset_id, scenarios in D1_D2_KNN.items():
        dataset_payload: dict[str, object] = {}
        for scenario, expected in scenarios.items():
            path = root / str(expected["path"])
            payload = _json(path, "D1_D2_KNN_AUTHORITY_UNREADABLE")
            metadata = payload.get("selection_metadata", {}).get("1_10", {})
            if not isinstance(metadata, Mapping):
                raise DeploymentManifestError("D1_D2_KNN_METADATA_MISSING", f"D{dataset_id}/{scenario}")
            dataset_payload[scenario] = {
                "path": expected["path"],
                "sha256": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
                "mtime_ns": int(path.stat().st_mtime_ns),
                "selection_authority": payload.get("selection_authority"),
                "protocol_version": payload.get("protocol_version"),
                "feature_cols": payload.get("feature_cols"),
                "knn_feature_columns": payload.get(
                    "knn_feature_columns", payload.get("feature_cols")
                ),
                "knn_frame_authority": payload.get("knn_frame_authority"),
                "selection_metadata": dict(metadata),
            }
        strict_knn[f"D{dataset_id}"] = dataset_payload
    return {
        "datasets": datasets,
        "d1_d2_knn": strict_knn,
        "d4_knn": knn,
        "d4_d6_knn": d4_d6_knn,
    }


def verify_frozen_snapshot(
    snapshot: Mapping[str, object],
    *,
    repository_root: Path,
) -> None:
    root = Path(repository_root).resolve(strict=True)
    datasets = snapshot.get("datasets")
    if not isinstance(datasets, Mapping):
        raise DeploymentManifestError("FINAL_AUTHORITY_INVALID", "datasets missing")
    for dataset_id, roles in FROZEN_PARQUETS.items():
        actual_roles = datasets.get(f"D{dataset_id}")
        if not isinstance(actual_roles, Mapping):
            raise DeploymentManifestError("FINAL_AUTHORITY_INVALID", f"D{dataset_id} missing")
        for role, expected in roles.items():
            actual = actual_roles.get(role)
            if not isinstance(actual, Mapping) or any(
                actual.get(field) != expected[field]
                for field in ("rows", "size_bytes", "sha256")
            ):
                raise DeploymentManifestError(
                    "FINAL_AUTHORITY_INVALID — PARQUET_MUTATED",
                    f"D{dataset_id} {role}",
                )
    knn = snapshot.get("d4_knn")
    if not isinstance(knn, Mapping) or set(knn) != set(D4_KNN):
        raise DeploymentManifestError("D4_KNN_AUTHORITY_MISMATCH")
    d4_d6_knn = snapshot.get("d4_d6_knn")
    if not isinstance(d4_d6_knn, Mapping) or set(d4_d6_knn) != {f"D{i}" for i in (4, 5, 6)}:
        raise DeploymentManifestError("D4_D6_KNN_AUTHORITY_MISMATCH")
    for dataset_id, scenarios in D4_D6_KNN.items():
        actual_dataset = d4_d6_knn.get(f"D{dataset_id}")
        if not isinstance(actual_dataset, Mapping) or set(actual_dataset) != set(scenarios):
            raise DeploymentManifestError("D4_D6_KNN_AUTHORITY_MISMATCH", f"D{dataset_id}")
        for scenario in scenarios:
            actual = actual_dataset.get(scenario)
            if not isinstance(actual, Mapping) or actual.get("sha256") != sha256_file(
                root / str(scenarios[scenario]["path"])
            ):
                raise DeploymentManifestError("D4_D6_KNN_AUTHORITY_MISMATCH", f"D{dataset_id}/{scenario}")
    strict_knn = snapshot.get("d1_d2_knn")
    if not isinstance(strict_knn, Mapping):
        raise DeploymentManifestError("D1_D2_KNN_AUTHORITY_MISMATCH")
    for dataset_id, scenarios in D1_D2_KNN.items():
        actual_dataset = strict_knn.get(f"D{dataset_id}")
        if not isinstance(actual_dataset, Mapping):
            raise DeploymentManifestError("D1_D2_KNN_AUTHORITY_MISMATCH", f"D{dataset_id}")
        for scenario, expected in scenarios.items():
            actual = actual_dataset.get(scenario)
            if (
                not isinstance(actual, Mapping)
                or actual.get("sha256") != expected["sha256"]
                or actual.get("selection_authority") != "shared_protocol"
                or actual.get("protocol_version") != PROTOCOL_VERSION
                or actual.get("knn_feature_columns")
                != list(get_experiment_protocol(dataset_id).knn_feature_columns)
                or actual.get("knn_frame_authority") != "configured_observed_frame"
            ):
                raise DeploymentManifestError(
                    "D1_D2_KNN_AUTHORITY_MISMATCH", f"D{dataset_id}/{scenario}"
                )


def build_code_inventory(repository_root: Path) -> dict[str, object]:
    root = Path(repository_root).resolve(strict=True)
    output = subprocess.check_output(
        [
            "git", "-C", str(root), "ls-files", "-co", "--exclude-standard", "--",
            "src", "scripts", "tools/operations", "configs/solidified",
        ],
        text=True,
    )
    files: list[dict[str, str]] = []
    for relative in sorted(set(output.splitlines())):
        posix = PurePosixPath(relative)
        if (
            not relative
            or "__pycache__" in posix.parts
            or relative.endswith(".pyc")
            or "outputs" in posix.parts
            or "tests" in posix.parts
        ):
            continue
        path = (root / relative).resolve(strict=True)
        path.relative_to(root)
        if path.is_file() and not path.is_symlink():
            files.append({"path": posix.as_posix(), "sha256": sha256_file(path)})
    stream = b"".join(
        item["path"].encode("utf-8") + b"\0" + item["sha256"].encode("ascii") + b"\n"
        for item in files
    )
    return {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "file_count": len(files),
        "files": files,
        "inventory_sha256": sha256_bytes(stream),
    }


def verify_code_inventory(repository_root: Path, inventory: Mapping[str, object]) -> None:
    if dict(inventory) != build_code_inventory(repository_root):
        raise DeploymentManifestError("CODE_INVENTORY_MISMATCH")


def _supporting_files(dataset_root: Path, sealed_root: Path) -> list[dict[str, object]]:
    result = []
    for path in sorted(dataset_root.glob("*.json")):
        if path.name == "formal-proof.json" or path.is_symlink():
            continue
        result.append(
            {
                "path": path.resolve(strict=True).relative_to(sealed_root).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
            }
        )
    return result


def _d4_authority(repository_root: Path, readiness: Mapping[str, object]) -> dict[str, object]:
    root = Path(repository_root)
    source_selection = readiness.get("source_selection")
    if not isinstance(source_selection, Mapping):
        raise DeploymentManifestError("D4_EXACT_KEY_PROOF_MISSING")
    targets = source_selection.get("targets")
    if not isinstance(targets, Mapping) or sum(
        len(value) for value in targets.values() if isinstance(value, Mapping)
    ) != 10:
        raise DeploymentManifestError("D4_EXACT_KEY_MATRIX_INCOMPLETE")
    files: dict[str, object] = {}
    for scenario, expected in D4_KNN.items():
        path = root / str(expected["path"])
        payload = _json(path, "D4_KNN_AUTHORITY_UNREADABLE")
        identity = payload.get("d4_manifest_identity")
        if not isinstance(identity, Mapping):
            raise DeploymentManifestError("D4_KNN_MANIFEST_IDENTITY_MISSING", scenario)
        files[scenario] = {
            "path": expected["path"],
            "sha256": sha256_file(path),
            "manifest_identity": identity.get("manifest_identity_digest"),
            "target_consumers": identity.get("target_consumers"),
        }
    matrix = {
        target: {
            scenario: {
                key: report.get(key)
                for key in (
                    "candidate_pool_digest", "selection_result_digest",
                    "source_pool_fingerprint", "consumer_fingerprint",
                    "validation_proof_digest", "manifest_identity_digest",
                    "exact_key_proof",
                )
            }
            for scenario, report in scenarios.items()
            if isinstance(report, Mapping)
        }
        for target, scenarios in targets.items()
        if isinstance(scenarios, Mapping)
    }
    exact_key_proof_digest = sha256_bytes(canonical_json_bytes(matrix))
    return {
        "entity_key": ["store_id", "product_id"],
        "exclusion_scope": "current_exact_target_tuple_only",
        "source_history_days": SOURCE_HISTORY_DAYS,
        "source_history_start": dataset_contract(4).source_history_start.isoformat(),
        "source_history_end": dataset_contract(4).source_history_end.isoformat(),
        "source_history_expected_date_count": SOURCE_HISTORY_DAYS,
        "source_history_calendar": SOURCE_HISTORY_CALENDAR,
        "source_history_completeness_policy": SOURCE_HISTORY_COMPLETENESS_POLICY,
        "source_history_inclusive_end": True,
        "scenario_files": files,
        "target_scenario_matrix": matrix,
        "exact_key_proof_digest": exact_key_proof_digest,
    }


def _d4_d6_runtime_authority(
    repository_root: Path,
    dataset_id: int,
    readiness: Mapping[str, object],
) -> dict[str, object]:
    """Bind D5/D6 persisted runtime selections to readiness and file bytes."""
    root = Path(repository_root)
    source_selection = readiness.get("source_selection")
    scenarios = source_selection.get("scenarios") if isinstance(source_selection, Mapping) else None
    if not isinstance(scenarios, Mapping) or set(scenarios) != {"without", "with"}:
        raise DeploymentManifestError("D4_D6_SELECTION_AUTHORITY_MISSING", f"D{dataset_id}")
    files: dict[str, object] = {}
    for scenario, expected in D4_D6_KNN[dataset_id].items():
        path = root / str(expected["path"])
        payload = _verify_d4_d6_knn_payload(path, dataset_id=dataset_id)
        report = scenarios.get(scenario)
        if not isinstance(report, Mapping):
            raise DeploymentManifestError("D4_D6_SELECTION_READINESS_MISSING", f"D{dataset_id}/{scenario}")
        files[scenario] = {
            "path": expected["path"],
            "sha256": payload["sha256"],
            "source_history_days": SOURCE_HISTORY_DAYS,
            "source_history_start": payload["source_history_start"],
            "source_history_end": payload["source_history_end"],
            "source_history_expected_date_count": SOURCE_HISTORY_DAYS,
            "source_history_calendar": SOURCE_HISTORY_CALENDAR,
            "source_history_completeness_policy": SOURCE_HISTORY_COMPLETENESS_POLICY,
            "source_history_inclusive_end": True,
            "targets": payload["targets"],
            "readiness_digest": sha256_bytes(canonical_json_bytes(report)),
        }
    authority = {
        "dataset_id": f"D{dataset_id}",
        "selection_authority": "shared_protocol" if dataset_id == 5 else "runtime",
        "protocol_version": PROTOCOL_VERSION if dataset_id == 5 else D4_D6_RUNTIME_KNN_PROTOCOL_VERSION,
        "knn_feature_columns": list(get_experiment_protocol(dataset_id).knn_feature_columns),
        "source_history_days": SOURCE_HISTORY_DAYS,
        "source_history_start": dataset_contract(dataset_id).source_history_start.isoformat(),
        "source_history_end": dataset_contract(dataset_id).source_history_end.isoformat(),
        "source_history_expected_date_count": SOURCE_HISTORY_DAYS,
        "source_history_calendar": SOURCE_HISTORY_CALENDAR,
        "source_history_completeness_policy": SOURCE_HISTORY_COMPLETENESS_POLICY,
        "source_history_inclusive_end": True,
        "scenario_files": files,
    }
    return {**authority, "authority_digest": sha256_bytes(canonical_json_bytes(authority))}


def _d1_d2_authority(repository_root: Path, dataset_id: int) -> dict[str, object]:
    """Validate and expose the real D1/D2 shared-protocol KNN authorities."""
    if int(dataset_id) not in D1_D2_KNN:
        raise DeploymentManifestError("D1_D2_KNN_AUTHORITY_INVALID", str(dataset_id))
    window = get_experiment_protocol(dataset_id).observation_window()
    scenarios: dict[str, object] = {}
    matrix: dict[str, object] = {}
    root = Path(repository_root)
    for scenario, expected in D1_D2_KNN[int(dataset_id)].items():
        path = root / str(expected["path"])
        payload = _json(path, "D1_D2_KNN_AUTHORITY_UNREADABLE")
        if (
            payload.get("selection_authority") != "shared_protocol"
            or payload.get("protocol_version") != PROTOCOL_VERSION
            or payload.get("knn_feature_columns", payload.get("feature_cols"))
            != list(get_experiment_protocol(dataset_id).knn_feature_columns)
            or payload.get("knn_frame_authority") != "configured_observed_frame"
        ):
            raise DeploymentManifestError("D1_D2_KNN_AUTHORITY_MISMATCH", f"D{dataset_id}/{scenario}")
        metadata_map = payload.get("selection_metadata")
        metadata = metadata_map.get("1_10") if isinstance(metadata_map, Mapping) else None
        if not isinstance(metadata, Mapping):
            raise DeploymentManifestError("D1_D2_KNN_METADATA_MISSING", f"D{dataset_id}/{scenario}")
        expected_window = {
            "origin": window.origin.isoformat(),
            "observed_start": window.knn_observed_start.isoformat(),
            "observed_end": window.knn_observed_end.isoformat(),
            "observed_days": window.observed_days,
            "boundary": "inclusive",
        }
        actual_window = {
            "origin": metadata.get("origin"),
            "observed_start": metadata.get("knn_observed_start"),
            "observed_end": metadata.get("knn_observed_end"),
            "observed_days": metadata.get("observed_days"),
            "boundary": metadata.get("boundary"),
        }
        if actual_window != expected_window:
            raise DeploymentManifestError(
                "D1_D2_KNN_WINDOW_MISMATCH", f"D{dataset_id}/{scenario}"
            )
        expected_features = list(get_experiment_protocol(dataset_id).knn_feature_columns)
        if (
            metadata.get("knn_feature_columns") != expected_features
            or metadata.get("historical_feature_columns") != expected_features
            or metadata.get("forecast_excluded_columns")
            != (["promo"] if int(dataset_id) == 2 else [])
            or metadata.get("feature_scope") != "historical_observed"
            or metadata.get("max_allowed_date_relation") != "date<=origin"
        ):
            raise DeploymentManifestError(
                "D1_D2_KNN_SCHEMA_MISMATCH", f"D{dataset_id}/{scenario}"
            )
        digest_fields = (
            "source_frame_digest",
            "target_frame_digest",
            "candidate_pool_digest",
            "selection_digest",
            "selection_result_digest",
        )
        if any(not isinstance(metadata.get(field), str) or len(metadata[field]) != 64 for field in digest_fields):
            raise DeploymentManifestError(
                "D1_D2_KNN_DIGEST_MISSING", f"D{dataset_id}/{scenario}"
            )
        scenario_payload = {
            "path": expected["path"],
            "sha256": sha256_file(path),
            "selection_authority": payload["selection_authority"],
            "protocol_version": payload["protocol_version"],
            "feature_cols": payload["feature_cols"],
            "knn_feature_columns": payload.get(
                "knn_feature_columns", payload["feature_cols"]
            ),
            "knn_frame_authority": payload["knn_frame_authority"],
            "observed_window": expected_window,
            "source_frame_min_date": metadata.get("source_frame_min_date"),
            "source_frame_max_date": metadata.get("source_frame_max_date"),
            "target_frame_min_date": metadata.get("target_frame_min_date"),
            "target_frame_max_date": metadata.get("target_frame_max_date"),
            "source_frame_digest": metadata["source_frame_digest"],
            "target_frame_digest": metadata["target_frame_digest"],
            "candidate_pool_digest": metadata["candidate_pool_digest"],
            "selection_digest": metadata["selection_digest"],
            "selection_result_digest": metadata["selection_result_digest"],
        }
        scenarios[scenario] = scenario_payload
        matrix[scenario] = {
            key: scenario_payload[key]
            for key in (
                "source_frame_digest",
                "target_frame_digest",
                "candidate_pool_digest",
                "selection_digest",
                "selection_result_digest",
            )
        }
    authority_payload = {
        "selection_authority": "shared_protocol",
        "protocol_version": PROTOCOL_VERSION,
        "knn_frame_authority": "configured_observed_frame",
        "knn_feature_columns": list(get_experiment_protocol(dataset_id).knn_feature_columns),
        "observed_window": {
            "origin": window.origin.isoformat(),
            "observed_start": window.knn_observed_start.isoformat(),
            "observed_end": window.knn_observed_end.isoformat(),
            "observed_days": window.observed_days,
            "boundary": "inclusive",
        },
        "scenario_files": scenarios,
        "target_scenario_matrix": matrix,
    }
    return {
        **authority_payload,
        "authority_digest": sha256_bytes(canonical_json_bytes(authority_payload)),
    }


def build_formal_proof(
    repository_root: Path,
    dataset_id: int,
    *,
    snapshot: Mapping[str, object],
    readiness: Mapping[str, object],
    formal_identity: Mapping[str, str],
    inventory_sha256: str,
) -> dict[str, object]:
    root = Path(repository_root).resolve(strict=True)
    paths = resolve_all_formal_dataset_paths(repository_root=root)[dataset_id - 1]
    spec = dataset_contract(dataset_id)
    identities = snapshot["datasets"][f"D{dataset_id}"]  # type: ignore[index]
    source_identity = parquet_content_identity(identities["source"])  # type: ignore[index]
    target_identity = parquet_content_identity(identities["target"])  # type: ignore[index]
    readiness_digest = readiness_proof_digest(readiness, repository_root=root)
    manifest_path = paths.dataset_manifest_path
    consumer_payload = {
        "dataset_id": f"D{dataset_id}",
        "source_sha256": source_identity["sha256"],
        "target_sha256": target_identity["sha256"],
        "source_schema_digest": source_identity["schema_digest"],
        "target_schema_digest": target_identity["schema_digest"],
        "readiness_proof_digest": readiness_digest,
        "code_inventory_digest": inventory_sha256,
    }
    consumer_fingerprint = sha256_bytes(canonical_json_bytes(consumer_payload))
    source_entities = readiness.get("source_entities", [])
    target_entities = readiness.get("target_entities", [])
    dataset_specific: dict[str, object] = {}
    if dataset_id == 1:
        dataset_specific = {
            "legacy_fallback_used": False,
            "resolver": "formal_input_paths",
            "knn_selection_authority": _d1_d2_authority(root, 1),
        }
    elif dataset_id == 2:
        sealed = readiness.get("sealed_identity")
        sealed_summary = None
        if isinstance(sealed, Mapping):
            raw_artifacts = sealed.get("artifacts")
            artifacts: dict[str, object] = {}
            if isinstance(raw_artifacts, Mapping):
                for role in ("source", "target"):
                    raw = raw_artifacts.get(role)
                    if isinstance(raw, Mapping):
                        artifacts[role] = {
                            "path": f"dataset2/{role}.parquet",
                            **{
                                key: value
                                for key, value in raw.items()
                                if key != "path"
                            },
                        }
            sealed_summary = {
                "status": sealed.get("status"),
                "manifest_path": "dataset2/manifest.json",
                "manifest_version": sealed.get("manifest_version"),
                "artifacts": artifacts,
                "precalendarized": sealed.get("precalendarized"),
                "runtime_calendarization": sealed.get("runtime_calendarization"),
                "verified_zero_sales_dates": sealed.get("verified_zero_sales_dates"),
                "zero_date_evidence": sealed.get("zero_date_evidence"),
            }
        dataset_specific = {
            "runtime_calendarization": False,
            "frozen_dates": list(D2_FROZEN_DATES),
            "zero_date_evidence": sealed.get("zero_date_evidence") if isinstance(sealed, Mapping) else None,
            "source_entities_per_date": 27,
            "target_entities_per_date": 1,
            "sales_zero": True,
            "sealed_identity": sealed_summary,
            "knn_selection_authority": _d1_d2_authority(root, 2),
        }
    elif dataset_id == 3:
        dataset_specific = {
            "post_origin_source_rows": readiness.get("post_origin_history_rows"),
            "source_window": [spec.source_history_start.isoformat(), spec.source_history_end.isoformat()],
            "explanation": "The sealed source contains historical rows outside the consumer window; the formal consumer slices the frozen source window before selection.",
        }
    elif dataset_id == 4:
        dataset_specific = _d4_authority(root, readiness)
    elif dataset_id == 5:
        dataset_specific = _d4_d6_runtime_authority(root, 5, readiness)
    elif dataset_id == 6:
        dataset_specific = _d4_d6_runtime_authority(root, 6, readiness)
    payload: dict[str, object] = {
        "schema_version": PROOF_SCHEMA_VERSION,
        "dataset_id": f"D{dataset_id}",
        "dataset_root": f"dataset{dataset_id}",
        "source": source_identity,
        "target": target_identity,
        "source_schema": source_identity["schema"],
        "target_schema": target_identity["schema"],
        "entity_key": list(spec.key_fields),
        "date_key": "date",
        "row_count": {
            "source": source_identity["rows"],
            "target": target_identity["rows"],
        },
        "entity_count": {"source": len(source_entities), "target": len(target_entities)},
        "date_range": {
            "source": [spec.source_history_start.isoformat(), spec.source_history_end.isoformat()],
            "target": [spec.target_train_start.isoformat(), spec.blind_end.isoformat()],
        },
        "key_date_uniqueness": {
            "status": "passed" if readiness.get("duplicate_exact_keys") == 0 else "failed",
            "consumer_duplicate_exact_keys": readiness.get("duplicate_exact_keys"),
        },
        "consumer_fingerprint": consumer_fingerprint,
        "consumer_fingerprint_input": consumer_payload,
        "dataset_manifest_identity": {
            "path": manifest_path.relative_to(paths.sealed_root).as_posix(),
            "sha256": sha256_file(manifest_path),
        },
        "supporting_proof_files": _supporting_files(paths.dataset_root, paths.sealed_root),
        "readiness_result": {
            "status": readiness.get("status"),
            "failure_code": readiness.get("failure_code"),
        },
        "readiness_proof_digest": readiness_digest,
        "formal_identity": dict(formal_identity),
        "dataset_specific": dataset_specific,
    }
    return {
        **payload,
        FORMAL_PROOF_DIGEST_FIELD: formal_proof_authority_digest(payload),
    }


def verify_formal_proof(
    proof: Mapping[str, object],
    *,
    dataset_id: int,
    formal_identity: Mapping[str, str],
    authority_digest: str | None = None,
    allow_legacy_raw_digest: bool = False,
) -> None:
    projected = formal_proof_authority_payload(proof)
    if projected.get("schema_version") != PROOF_SCHEMA_VERSION or projected.get("dataset_id") != f"D{dataset_id}":
        raise DeploymentManifestError("FORMAL_PROOF_SCHEMA_MISMATCH", f"D{dataset_id}")
    expected_digest = proof.get(FORMAL_PROOF_DIGEST_FIELD)
    if not _is_sha256_digest(expected_digest):
        raise DeploymentManifestError("FORMAL_PROOF_TAMPERED", f"D{dataset_id}")
    semantic_digest = sha256_bytes(canonical_json_bytes(projected))
    if authority_digest is not None:
        if (
            not _is_sha256_digest(authority_digest)
            or authority_digest != semantic_digest
            or expected_digest != semantic_digest
        ):
            raise DeploymentManifestError("FORMAL_PROOF_TAMPERED", f"D{dataset_id}")
    elif not allow_legacy_raw_digest and expected_digest != semantic_digest:
        raise DeploymentManifestError("FORMAL_PROOF_TAMPERED", f"D{dataset_id}")
    if projected.get("formal_identity") != dict(formal_identity):
        raise DeploymentManifestError("FORMAL_PROOF_IDENTITY_MISMATCH", f"D{dataset_id}")
    if projected.get("readiness_result") != {"status": "passed", "failure_code": None}:
        raise DeploymentManifestError("FINAL_PREFLIGHT_NOT_READY", f"D{dataset_id}")
    key_date_uniqueness = projected.get("key_date_uniqueness")
    if not isinstance(key_date_uniqueness, Mapping) or key_date_uniqueness.get("status") != "passed":
        raise DeploymentManifestError("KEY_DATE_NOT_UNIQUE", f"D{dataset_id}")


def build_root_manifest(
    repository_root: Path,
    *,
    proofs: Mapping[str, Mapping[str, object]],
    inventory: Mapping[str, object],
    inventory_file_sha256: str,
    formal_identity: Mapping[str, str],
) -> dict[str, object]:
    root = Path(repository_root).resolve(strict=True)
    sealed = root / FORMAL_SEALED_ROOT_RELATIVE
    datasets: dict[str, object] = {}
    for dataset_id in range(1, 7):
        key = f"D{dataset_id}"
        proof = proofs[key]
        authority_proof = formal_proof_authority_payload(proof)
        datasets[key] = {
            "dataset_directory": f"dataset{dataset_id}",
            "source": authority_proof["source"],
            "target": authority_proof["target"],
            "dataset_manifest": authority_proof["dataset_manifest_identity"],
            "formal_proof": {
                "path": f"dataset{dataset_id}/formal-proof.json",
                "authority_sha256": formal_proof_authority_digest(proof),
            },
            "source_schema_digest": authority_proof["source"]["schema_digest"],  # type: ignore[index]
            "target_schema_digest": authority_proof["target"]["schema_digest"],  # type: ignore[index]
            "consumer_fingerprint": authority_proof["consumer_fingerprint"],
            "readiness_proof_digest": authority_proof["readiness_proof_digest"],
        }
    authority_proofs = {
        key: formal_proof_authority_payload(proof) for key, proof in proofs.items()
    }
    d1 = authority_proofs["D1"]["dataset_specific"]
    d2 = authority_proofs["D2"]["dataset_specific"]
    d4 = authority_proofs["D4"]["dataset_specific"]
    d5 = authority_proofs["D5"]["dataset_specific"]
    d6 = authority_proofs["D6"]["dataset_specific"]
    payload: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "publication_state": "authoritative",
        "sealed_root": FORMAL_SEALED_ROOT_RELATIVE.as_posix(),
        "formal_identity": dict(formal_identity),
        "code_inventory": {
            "path": "code-inventory.json",
            "sha256": inventory_file_sha256,
            "inventory_sha256": inventory["inventory_sha256"],
            "file_count": inventory["file_count"],
        },
        "datasets": datasets,
        "d4_selection_authority": {
            "with": d4["scenario_files"]["with"],  # type: ignore[index]
            "without": d4["scenario_files"]["without"],  # type: ignore[index]
            "exact_key_proof_digest": d4["exact_key_proof_digest"],  # type: ignore[index]
        },
        "d4_d6_selection_authority": {
            "D4": d4,
            "D5": d5,
            "D6": d6,
        },
        "d1_d2_knn_selection_authority": {
            "D1": d1["knn_selection_authority"]["scenario_files"],  # type: ignore[index]
            "D2": d2["knn_selection_authority"]["scenario_files"],  # type: ignore[index]
        },
    }
    return {**payload, "root_identity_sha256": sha256_bytes(canonical_json_bytes(payload))}


def load_deployment_manifest(sealed_root: Path) -> dict[str, object]:
    root = Path(sealed_root).resolve(strict=True)
    manifest_path = root / "deployment-manifest.json"
    if not manifest_path.is_file():
        raise DeploymentManifestError("ROOT_MANIFEST_MISSING")
    sidecar_path = root / "deployment-manifest.sha256"
    if not sidecar_path.is_file():
        raise DeploymentManifestError("ROOT_MANIFEST_SHA_MISSING")
    actual_file_sha = sha256_file(manifest_path)
    expected_sidecar = f"{actual_file_sha}  deployment-manifest.json\n"
    if sidecar_path.read_text(encoding="utf-8") != expected_sidecar:
        raise DeploymentManifestError("ROOT_MANIFEST_SHA_MISMATCH")
    manifest = _json(manifest_path, "ROOT_MANIFEST_UNREADABLE")
    expected_root = manifest.get("root_identity_sha256")
    payload = {key: value for key, value in manifest.items() if key != "root_identity_sha256"}
    if expected_root != sha256_bytes(canonical_json_bytes(payload)):
        raise DeploymentManifestError("ROOT_IDENTITY_MISMATCH")
    return manifest


def validate_deployment_manifest(
    repository_root: Path,
    *,
    sealed_root: Path | None = None,
) -> dict[str, object]:
    root = Path(repository_root).resolve(strict=True)
    sealed = Path(sealed_root or root / FORMAL_SEALED_ROOT_RELATIVE).resolve(strict=True)
    if sealed != (root / FORMAL_SEALED_ROOT_RELATIVE).resolve(strict=True):
        raise DeploymentManifestError("SEALED_ROOT_MISMATCH")
    if (sealed / "NON_AUTHORITATIVE").exists():
        raise DeploymentManifestError("NON_AUTHORITATIVE_ROOT")
    manifest = load_deployment_manifest(sealed)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise DeploymentManifestError("ROOT_MANIFEST_SCHEMA_MISMATCH")
    if manifest.get("publication_state") != "authoritative":
        raise DeploymentManifestError("PUBLICATION_STATE_MISMATCH")
    formal_identity = formal_identity_payload(root)
    if manifest.get("formal_identity") != formal_identity:
        raise DeploymentManifestError("FORMAL_IDENTITY_MISMATCH")
    inventory_entry = manifest.get("code_inventory")
    if not isinstance(inventory_entry, Mapping):
        raise DeploymentManifestError("CODE_INVENTORY_MISSING")
    inventory_path = _resolve_bound_path(sealed, inventory_entry.get("path"), code="CODE_INVENTORY_PATH_INVALID")
    if sha256_file(inventory_path) != inventory_entry.get("sha256"):
        raise DeploymentManifestError("CODE_INVENTORY_FILE_MISMATCH")
    inventory = _json(inventory_path, "CODE_INVENTORY_UNREADABLE")
    if (
        inventory.get("inventory_sha256") != inventory_entry.get("inventory_sha256")
        or inventory.get("file_count") != inventory_entry.get("file_count")
    ):
        raise DeploymentManifestError("CODE_INVENTORY_MISMATCH")
    verify_code_inventory(root, inventory)
    snapshot = frozen_artifact_snapshot(root)
    verify_frozen_snapshot(snapshot, repository_root=root)
    datasets = manifest.get("datasets")
    if not isinstance(datasets, Mapping) or set(datasets) != {f"D{i}" for i in range(1, 7)}:
        raise DeploymentManifestError("DATASET_ENTRY_MISSING")
    proofs: dict[str, object] = {}
    parquet_identity_diagnostics: list[dict[str, object]] = []
    resolved = resolve_all_formal_dataset_paths(repository_root=root)
    for paths in resolved:
        key = f"D{paths.dataset_id}"
        entry = datasets[key]
        if not isinstance(entry, Mapping):
            raise DeploymentManifestError("DATASET_ENTRY_INVALID", key)
        source = entry.get("source")
        target = entry.get("target")
        current = snapshot["datasets"][key]  # type: ignore[index]
        if not isinstance(current, Mapping):
            raise DeploymentManifestError("PARQUET_IDENTITY_MISMATCH", key)
        for role, authority_identity in (("source", source), ("target", target)):
            current_identity = current.get(role)
            if not isinstance(authority_identity, Mapping) or not isinstance(
                current_identity, Mapping
            ):
                raise DeploymentManifestError(
                    "PARQUET_IDENTITY_MISMATCH", f"{key}/{role}: identity object missing"
                )
            comparison = compare_parquet_identity(authority_identity, current_identity)
            structural_errors = comparison["structural_errors"]
            blocking_mismatches = comparison["blocking_mismatches"]
            if structural_errors or blocking_mismatches:
                raise DeploymentManifestError(
                    "PARQUET_IDENTITY_MISMATCH",
                    f"{key}/{role}: structural_errors={structural_errors!r}; "
                    f"fields={sorted(blocking_mismatches)}",
                )
            for diagnostic in comparison["diagnostics"]:
                parquet_identity_diagnostics.append(
                    {"dataset": key, "role": role, "against": "current", **diagnostic}
                )
        proof_entry = entry.get("formal_proof")
        if not isinstance(proof_entry, Mapping):
            raise DeploymentManifestError("FORMAL_PROOF_MISSING", key)
        proof_path = _resolve_bound_path(sealed, proof_entry.get("path"), code="FORMAL_PROOF_PATH_INVALID")
        proof = _json(proof_path, "FORMAL_PROOF_UNREADABLE")
        authority_digest = proof_entry.get("authority_sha256")
        if authority_digest is not None:
            if not _is_sha256_digest(authority_digest):
                raise DeploymentManifestError("FORMAL_PROOF_TAMPERED", key)
            verify_formal_proof(
                proof,
                dataset_id=paths.dataset_id,
                formal_identity=formal_identity,
                authority_digest=authority_digest,
            )
        else:
            legacy_raw_file_sha256 = proof_entry.get("sha256")
            if not _is_sha256_digest(legacy_raw_file_sha256):
                raise DeploymentManifestError("FORMAL_PROOF_TAMPERED", key)
            # ``sha256`` was the historical raw JSON-file hash.  It is kept
            # only so old manifests can be read; consulting it here would make
            # an mtime-only proof rewrite block semantic authority.
            verify_formal_proof(
                proof,
                dataset_id=paths.dataset_id,
                formal_identity=formal_identity,
                allow_legacy_raw_digest=True,
            )
        for role, authority_identity in (("source", source), ("target", target)):
            proof_identity = proof.get(role)
            if not isinstance(authority_identity, Mapping) or not isinstance(
                proof_identity, Mapping
            ):
                raise DeploymentManifestError(
                    "FORMAL_PROOF_PARQUET_MISMATCH",
                    f"{key}/{role}: identity object missing",
                )
            proof_comparison = compare_parquet_identity(authority_identity, proof_identity)
            proof_structural_errors = proof_comparison["structural_errors"]
            proof_mismatches = proof_comparison["blocking_mismatches"]
            if proof_structural_errors or proof_mismatches:
                raise DeploymentManifestError(
                    "FORMAL_PROOF_PARQUET_MISMATCH",
                    f"{key}/{role}: structural_errors={proof_structural_errors!r}; "
                    f"fields={sorted(proof_mismatches)}",
                )
            for diagnostic in proof_comparison["diagnostics"]:
                parquet_identity_diagnostics.append(
                    {"dataset": key, "role": role, "against": "formal_proof", **diagnostic}
                )
        if proof.get("consumer_fingerprint") != entry.get("consumer_fingerprint"):
            raise DeploymentManifestError("CONSUMER_FINGERPRINT_MISMATCH", key)
        proofs[key] = proof
    d2 = proofs["D2"]["dataset_specific"]  # type: ignore[index]
    if (
        d2.get("runtime_calendarization") is not False
        or d2.get("frozen_dates") != list(D2_FROZEN_DATES)
        or d2.get("sales_zero") is not True
    ):
        raise DeploymentManifestError("D2_FROZEN_CLOSURE_MISMATCH")
    strict_authority = manifest.get("d1_d2_knn_selection_authority")
    if not isinstance(strict_authority, Mapping):
        raise DeploymentManifestError("D1_D2_KNN_AUTHORITY_MISMATCH")
    for dataset_id in (1, 2):
        proof_authority = proofs[f"D{dataset_id}"]["dataset_specific"].get(  # type: ignore[index]
            "knn_selection_authority"
        )
        if (
            not isinstance(proof_authority, Mapping)
            or proof_authority.get("selection_authority") != "shared_protocol"
            or proof_authority.get("protocol_version") != PROTOCOL_VERSION
            or proof_authority.get("knn_frame_authority") != "configured_observed_frame"
            or strict_authority.get(f"D{dataset_id}") != proof_authority.get("scenario_files")
        ):
            raise DeploymentManifestError(
                "D1_D2_KNN_AUTHORITY_MISMATCH", f"D{dataset_id}"
            )
    d4 = proofs["D4"]["dataset_specific"]  # type: ignore[index]
    if (
        d4.get("entity_key") != ["store_id", "product_id"]
        or d4.get("exclusion_scope") != "current_exact_target_tuple_only"
        or d4.get("exact_key_proof_digest") != manifest["d4_selection_authority"]["exact_key_proof_digest"]  # type: ignore[index]
    ):
        raise DeploymentManifestError("D4_EXACT_KEY_AUTHORITY_MISMATCH")
    d4_d6_authority = manifest.get("d4_d6_selection_authority")
    expected_d4_d6_authority = {
        f"D{dataset_id}": proofs[f"D{dataset_id}"]["dataset_specific"]
        for dataset_id in (4, 5, 6)
    }
    if d4_d6_authority != expected_d4_d6_authority:
        raise DeploymentManifestError("D4_D6_SELECTION_AUTHORITY_MISMATCH")
    for dataset_id in (4, 5, 6):
        authority = expected_d4_d6_authority[f"D{dataset_id}"]
        scenario_files = authority.get("scenario_files") if isinstance(authority, Mapping) else None
        if not isinstance(scenario_files, Mapping):
            raise DeploymentManifestError("D4_D6_SELECTION_AUTHORITY_MISSING", f"D{dataset_id}")
        for scenario, entry in scenario_files.items():
            if not isinstance(entry, Mapping):
                raise DeploymentManifestError("D4_D6_SELECTION_AUTHORITY_INVALID", f"D{dataset_id}/{scenario}")
            path = root / str(entry.get("path"))
            if sha256_file(path) != entry.get("sha256"):
                raise DeploymentManifestError("D4_D6_KNN_BYTES_MISMATCH", f"D{dataset_id}/{scenario}")
    return {
        "preflight_status": "ready",
        "failure_code": None,
        "datasets_ready": 6,
        "datasets_total": 6,
        "manifest": manifest,
        "manifest_sha256": sha256_file(sealed / "deployment-manifest.json"),
        "root_identity_sha256": manifest["root_identity_sha256"],
        "code_inventory_sha256": inventory["inventory_sha256"],
        "parquet_identity_diagnostics": parquet_identity_diagnostics,
        "proofs": proofs,
    }


def atomic_write_json(path: Path, payload: object) -> None:
    """Write deterministic JSON through fsync, parse validation, and replace."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    data = pretty_json_bytes(payload)
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        parsed = json.loads(temporary.read_text(encoding="utf-8"))
        if pretty_json_bytes(parsed) != data:
            raise DeploymentManifestError("DETERMINISTIC_JSON_VALIDATION_FAILED", str(destination))
        os.replace(temporary, destination)
        directory_fd = os.open(str(destination.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(str(destination.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


__all__ = [
    "D1_D2_KNN", "D4_KNN", "D4_D6_KNN", "EXPECTED_BRANCH", "EXPECTED_HEAD", "FROZEN_PARQUETS",
    "DeploymentManifestError", "FORMAL_PROOF_AUTHORITY_FIELDS",
    "FORMAL_PROOF_DIGEST_FIELD", "KNOWN_IDENTITY_FIELDS", "OPTIONAL_DIAGNOSTIC_FIELDS",
    "REQUIRED_BLOCKING_FIELDS", "atomic_write_bytes", "atomic_write_json",
    "build_code_inventory", "build_formal_proof", "build_root_manifest",
    "canonical_json_bytes", "canonical_readiness_proof_payload", "compare_parquet_identity",
    "formal_identity_payload", "formal_proof_authority_digest",
    "formal_proof_authority_payload",
    "frozen_artifact_snapshot",
    "load_deployment_manifest", "parquet_identity", "pretty_json_bytes",
    "parquet_content_identity",
    "readiness_proof_digest", "repository_identity", "require_repository_identity", "sha256_bytes",
    "sha256_file", "validate_deployment_manifest", "verify_frozen_snapshot",
]
