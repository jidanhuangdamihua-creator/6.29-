"""Truth-free artifact rehydration and frozen logical-to-physical bindings."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Optional, Sequence
from uuid import uuid4

from src.utils.run_recovery import (
    ActorIdentity,
    RecoveryError,
    RunRecovery,
    RunState,
    StaleFencingTokenError,
)


_DIGEST = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")
_LOGICAL_ID = re.compile(r"^artifact-[0-9a-f]{64}$")


class ArtifactCondition(str, Enum):
    VALID = "VALID"
    ARTIFACT_MISSING = "ARTIFACT_MISSING"
    ARTIFACT_BYTES_MISMATCH = "ARTIFACT_BYTES_MISMATCH"


class ArtifactRehydrationError(RuntimeError):
    pass


def _require_digest(name: str, value: str) -> str:
    text = str(value)
    if not _DIGEST.fullmatch(text):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return text if text.startswith("sha256:") else "sha256:" + text


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def derive_logical_artifact_id(artifact_role: str, logical_key: str) -> str:
    """Derive a stable ID without incorporating attempt or physical identity."""
    role = str(artifact_role).strip()
    key = str(logical_key).strip()
    if not role or not key:
        raise ValueError("artifact role and logical key must be non-empty")
    payload = _canonical_bytes({"artifact_role": role, "logical_key": key})
    return "artifact-" + hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ArtifactBinding:
    logical_artifact_id: str
    schema_digest: str
    canonical_content_sha256: str
    semantic_prediction_digest: str
    artifact_sha256: str
    physical_path: str
    attempt_id: str
    fencing_token: int

    def __post_init__(self) -> None:
        if not _LOGICAL_ID.fullmatch(str(self.logical_artifact_id)):
            raise ValueError("logical_artifact_id must be a stable artifact SHA-256 ID")
        for field in (
            "schema_digest",
            "canonical_content_sha256",
            "semantic_prediction_digest",
            "artifact_sha256",
        ):
            object.__setattr__(self, field, _require_digest(field, getattr(self, field)))
        if not str(self.physical_path) or not str(self.attempt_id):
            raise ValueError("physical_path and attempt_id must be non-empty")
        if int(self.fencing_token) <= 0:
            raise ValueError("fencing_token must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrustedReplica:
    replica_id: str
    path: Path
    logical_artifact_id: str
    schema_digest: str
    canonical_content_sha256: str
    semantic_prediction_digest: str
    artifact_sha256: str
    trusted: bool = True

    def __post_init__(self) -> None:
        if not str(self.replica_id):
            raise ValueError("replica_id must be non-empty")
        object.__setattr__(self, "path", Path(self.path))
        if not _LOGICAL_ID.fullmatch(str(self.logical_artifact_id)):
            raise ValueError("replica logical_artifact_id is invalid")
        for field in (
            "schema_digest",
            "canonical_content_sha256",
            "semantic_prediction_digest",
            "artifact_sha256",
        ):
            object.__setattr__(self, field, _require_digest(field, getattr(self, field)))


class TrustedReplicaRegistry:
    """Closed registry; unregistered paths can never become rehydrate authority."""

    def __init__(self, replicas: Iterable[TrustedReplica] = ()) -> None:
        self._replicas: dict[str, TrustedReplica] = {}
        for replica in replicas:
            self.register(replica)

    def register(self, replica: TrustedReplica) -> None:
        if replica.replica_id in self._replicas:
            raise ValueError(f"duplicate trusted replica id: {replica.replica_id}")
        self._replicas[replica.replica_id] = replica

    def get(self, replica_id: str) -> TrustedReplica:
        try:
            replica = self._replicas[str(replica_id)]
        except KeyError as exc:
            raise ArtifactRehydrationError("rehydrate candidate is not registered") from exc
        if not replica.trusted:
            raise ArtifactRehydrationError("rehydrate candidate is not trusted")
        return replica

    def matching(self, binding: ArtifactBinding) -> tuple[TrustedReplica, ...]:
        return tuple(
            replica
            for replica in sorted(self._replicas.values(), key=lambda item: item.replica_id)
            if replica.trusted
            and replica.logical_artifact_id == binding.logical_artifact_id
            and replica.schema_digest == binding.schema_digest
            and replica.canonical_content_sha256 == binding.canonical_content_sha256
            and replica.semantic_prediction_digest == binding.semantic_prediction_digest
        )


@dataclass(frozen=True)
class FrozenBindingSet:
    path: Path
    digest: str
    attempt_id: str
    fencing_token: int


class ArtifactRehydrator:
    def __init__(self, recovery: RunRecovery, registry: TrustedReplicaRegistry) -> None:
        self.recovery = recovery
        self.registry = registry

    def classify(self, binding: ArtifactBinding) -> ArtifactCondition:
        path = Path(binding.physical_path)
        if not path.is_file():
            return ArtifactCondition.ARTIFACT_MISSING
        try:
            actual = _sha256_file(path)
        except OSError:
            return ArtifactCondition.ARTIFACT_BYTES_MISMATCH
        return (
            ArtifactCondition.VALID
            if actual == binding.artifact_sha256
            else ArtifactCondition.ARTIFACT_BYTES_MISMATCH
        )

    def _require_running_attempt(self, fencing_token: int) -> Mapping[str, Any]:
        state = self.recovery.load_state()
        if int(state.get("fencing_token", -1)) != int(fencing_token):
            raise StaleFencingTokenError("stale fencing token for rehydrate")
        run_state = RunState(state["run_state"])
        if run_state in {RunState.SEALED_SUCCESS, RunState.SEALED_FAILED}:
            raise ArtifactRehydrationError("rehydrate is forbidden in a sealed terminal state")
        if run_state is not RunState.RUNNING:
            raise ArtifactRehydrationError("rehydrate requires a running attempt")
        if state.get("downstream_scheduling_started"):
            raise ArtifactRehydrationError(
                "rehydrate after downstream scheduling requires a new attempt"
            )
        if state.get("artifact_binding_set") is not None:
            raise ArtifactRehydrationError("artifact binding set is already frozen")
        return state

    @staticmethod
    def _validate_replica(replica: TrustedReplica, binding: ArtifactBinding) -> None:
        if (
            replica.logical_artifact_id != binding.logical_artifact_id
            or replica.schema_digest != binding.schema_digest
            or replica.canonical_content_sha256 != binding.canonical_content_sha256
            or replica.semantic_prediction_digest != binding.semantic_prediction_digest
        ):
            raise ArtifactRehydrationError("trusted replica logical identity mismatch")
        if not replica.path.is_file():
            raise ArtifactRehydrationError("trusted replica bytes are missing")
        if _sha256_file(replica.path) != replica.artifact_sha256:
            raise ArtifactRehydrationError("trusted replica content-address mismatch")

    def rehydrate(
        self,
        binding: ArtifactBinding,
        *,
        actor: ActorIdentity,
        fencing_token: int,
        explicit: bool = False,
        replica_id: Optional[str] = None,
    ) -> ArtifactBinding:
        if not isinstance(actor, ActorIdentity):
            raise ArtifactRehydrationError("rehydrate requires authenticated actor identity")
        state = self._require_running_attempt(fencing_token)
        condition = self.classify(binding)
        if condition is ArtifactCondition.VALID:
            return binding
        attempt_id = str(state["current_attempt_id"])
        if binding.attempt_id == attempt_id:
            raise ArtifactRehydrationError("rehydrate must publish under a new attempt")
        matches = self.registry.matching(binding)
        if condition is ArtifactCondition.ARTIFACT_BYTES_MISMATCH and not explicit:
            raise ArtifactRehydrationError(
                "byte mismatch requires explicit authenticated rehydrate"
            )
        if explicit:
            if replica_id is None:
                raise ArtifactRehydrationError("explicit rehydrate requires replica_id")
            replica = self.registry.get(replica_id)
        else:
            if not matches:
                raise ArtifactRehydrationError(
                    "missing artifact has no registered trusted replica; recompute cell"
                )
            replica = matches[0]
        self._validate_replica(replica, binding)
        if replica not in matches:
            raise ArtifactRehydrationError("selected replica does not match logical identity")
        original = Path(binding.physical_path)
        if replica.path.resolve() == original.resolve():
            raise ArtifactRehydrationError("a mismatched file cannot sign itself")
        if (
            condition is ArtifactCondition.ARTIFACT_BYTES_MISMATCH
            and replica.artifact_sha256 == binding.artifact_sha256
        ):
            raise ArtifactRehydrationError(
                "byte mismatch requires a different trusted candidate"
            )

        parent = self.recovery.layout.attempt_rehydrated_artifacts_dir(attempt_id)
        logical_dir = parent / binding.logical_artifact_id
        logical_dir.mkdir(parents=True, exist_ok=True)
        suffix = "".join(replica.path.suffixes)
        destination = logical_dir / (replica.artifact_sha256.removeprefix("sha256:") + suffix)
        temporary = destination.with_name(f".{destination.name}.tmp.{uuid4().hex}")
        try:
            with replica.path.open("rb") as source, temporary.open("xb") as target:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    target.write(chunk)
                target.flush()
                os.fsync(target.fileno())
            if _sha256_file(temporary) != replica.artifact_sha256:
                raise ArtifactRehydrationError("rehydrated copy failed physical validation")
            if destination.exists():
                if _sha256_file(destination) != replica.artifact_sha256:
                    raise ArtifactRehydrationError("rehydrated destination digest conflict")
                temporary.unlink()
            else:
                os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        restored = ArtifactBinding(
            logical_artifact_id=binding.logical_artifact_id,
            schema_digest=binding.schema_digest,
            canonical_content_sha256=binding.canonical_content_sha256,
            semantic_prediction_digest=binding.semantic_prediction_digest,
            artifact_sha256=replica.artifact_sha256,
            physical_path=str(destination),
            attempt_id=attempt_id,
            fencing_token=int(fencing_token),
        )
        relative = destination.resolve().relative_to(
            self.recovery.layout.run_root.resolve()
        ).as_posix()
        self.recovery.record_artifact_rehydration(
            logical_artifact_id=binding.logical_artifact_id,
            old_artifact_sha256=binding.artifact_sha256,
            new_artifact_sha256=restored.artifact_sha256,
            physical_path=relative,
            condition=condition.value,
            actor=actor,
            fencing_token=fencing_token,
            reason=(
                "authenticated explicit trusted-replica rehydrate"
                if explicit
                else "automatic trusted-replica rehydrate for missing artifact"
            ),
        )
        return restored

    def freeze_binding_set(
        self,
        bindings: Sequence[ArtifactBinding],
        *,
        actor: ActorIdentity,
        fencing_token: int,
    ) -> FrozenBindingSet:
        if not isinstance(actor, ActorIdentity):
            raise ArtifactRehydrationError("binding freeze requires authenticated actor identity")
        current = self.recovery.load_state()
        if current.get("artifact_binding_set") is not None:
            raise FileExistsError("artifact binding set is already frozen for this attempt")
        state = self._require_running_attempt(fencing_token)
        attempt_id = str(state["current_attempt_id"])
        indexed: dict[str, dict[str, Any]] = {}
        root = self.recovery.layout.run_root.resolve()
        for binding in bindings:
            if binding.logical_artifact_id in indexed:
                raise ArtifactRehydrationError("duplicate logical_artifact_id in binding set")
            if binding.attempt_id != attempt_id or binding.fencing_token != int(fencing_token):
                raise ArtifactRehydrationError("binding is not owned by the current attempt")
            physical = Path(binding.physical_path).resolve()
            try:
                relative = physical.relative_to(root).as_posix()
            except ValueError as exc:
                raise ArtifactRehydrationError("bound physical path must be inside run root") from exc
            if not physical.is_file() or _sha256_file(physical) != binding.artifact_sha256:
                raise ArtifactRehydrationError("binding physical bytes failed validation")
            value = binding.to_dict()
            value["physical_path"] = relative
            indexed[binding.logical_artifact_id] = value
        payload: dict[str, Any] = {
            "binding_schema_version": "artifact-binding-set/v1",
            "attempt_id": attempt_id,
            "fencing_token": int(fencing_token),
            "fit_call_count": 0,
            "predict_call_count": 0,
            "bindings": dict(sorted(indexed.items())),
        }
        digest = "sha256:" + hashlib.sha256(_canonical_bytes(payload)).hexdigest()
        payload["binding_set_digest"] = digest
        path = self.recovery.publish_artifact_binding_set(
            payload,
            binding_set_digest=digest,
            actor=actor,
            fencing_token=fencing_token,
        )
        return FrozenBindingSet(path, digest, attempt_id, int(fencing_token))


def _load_frozen_binding_set(recovery: RunRecovery) -> tuple[dict[str, Any], str]:
    state = recovery.load_state()
    authority = state.get("artifact_binding_set")
    if not isinstance(authority, dict):
        raise ArtifactRehydrationError("artifact binding set is not frozen")
    if authority.get("attempt_id") != state.get("current_attempt_id"):
        raise ArtifactRehydrationError("artifact binding authority is not current")
    path = recovery.layout.run_root / str(authority.get("path", ""))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise ArtifactRehydrationError("artifact binding set is unreadable") from exc
    claimed = str(payload.get("binding_set_digest", ""))
    unsigned = dict(payload)
    unsigned.pop("binding_set_digest", None)
    actual = "sha256:" + hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()
    if claimed != actual or authority.get("digest") != actual:
        raise ArtifactRehydrationError("artifact binding set digest mismatch")
    if int(payload.get("fencing_token", -1)) != int(state.get("fencing_token", -2)):
        raise ArtifactRehydrationError("artifact binding set fencing mismatch")
    return payload, actual


def resolve_bound_artifact(
    recovery: RunRecovery,
    logical_artifact_id: str,
    *,
    expected_binding_set_digest: Optional[str] = None,
) -> Path:
    """Resolve only through the frozen authority; old manifest paths are ignored."""
    payload, digest = _load_frozen_binding_set(recovery)
    if expected_binding_set_digest is not None and digest != _require_digest(
        "expected_binding_set_digest", expected_binding_set_digest
    ):
        raise ArtifactRehydrationError("unexpected frozen binding set digest")
    try:
        raw = payload["bindings"][str(logical_artifact_id)]
    except (KeyError, TypeError) as exc:
        raise ArtifactRehydrationError("logical artifact is not in frozen binding set") from exc
    path = recovery.layout.run_root / str(raw["physical_path"])
    if not path.is_file() or _sha256_file(path) != _require_digest(
        "artifact_sha256", str(raw["artifact_sha256"])
    ):
        raise ArtifactRehydrationError("bound artifact bytes are missing or mismatched")
    return path
