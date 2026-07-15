from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

from src.utils.artifact_rehydration import (
    ArtifactBinding,
    ArtifactCondition,
    ArtifactRehydrationError,
    ArtifactRehydrator,
    TrustedReplica,
    TrustedReplicaRegistry,
    derive_logical_artifact_id,
    resolve_bound_artifact,
)
from src.utils.run_recovery import ActorIdentity, RunRecovery, RunState


def _sha(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _actor() -> ActorIdentity:
    return ActorIdentity("operator", "human", "authenticated-test", "a" * 64)


def _binding(path: Path, payload: bytes, *, attempt_id: str, token: int) -> ArtifactBinding:
    return ArtifactBinding(
        logical_artifact_id=derive_logical_artifact_id("worker_trace", "cell-1"),
        schema_digest="sha256:" + "b" * 64,
        canonical_content_sha256="sha256:" + "c" * 64,
        semantic_prediction_digest="sha256:" + "d" * 64,
        artifact_sha256=_sha(payload),
        physical_path=str(path),
        attempt_id=attempt_id,
        fencing_token=token,
    )


def _replica(replica_id: str, path: Path, binding: ArtifactBinding) -> TrustedReplica:
    return TrustedReplica(
        replica_id=replica_id,
        path=path,
        logical_artifact_id=binding.logical_artifact_id,
        schema_digest=binding.schema_digest,
        canonical_content_sha256=binding.canonical_content_sha256,
        semantic_prediction_digest=binding.semantic_prediction_digest,
        artifact_sha256=_sha(path.read_bytes()),
    )


def test_missing_trusted_replica_rehydrates_without_model_calls_and_freezes_binding(
    tmp_path: Path,
) -> None:
    recovery = RunRecovery(tmp_path / "run")
    lease = recovery.create(_actor(), run_identity="r" * 64)
    missing = tmp_path / "old-attempt" / "trace.csv.gz"
    binding = _binding(missing, b"old physical bytes", attempt_id="old", token=1)
    replica_path = tmp_path / "trusted" / "trace.csv.gz"
    replica_path.parent.mkdir()
    replica_path.write_bytes(b"new deterministic physical bytes")
    registry = TrustedReplicaRegistry([_replica("replica-a", replica_path, binding)])
    calls = {"fit": 0, "predict": 0}

    rehydrator = ArtifactRehydrator(recovery, registry)
    assert rehydrator.classify(binding) is ArtifactCondition.ARTIFACT_MISSING
    restored = rehydrator.rehydrate(
        binding,
        actor=_actor(),
        fencing_token=lease.fencing_token,
    )
    frozen = rehydrator.freeze_binding_set(
        [restored], actor=_actor(), fencing_token=lease.fencing_token
    )

    assert calls == {"fit": 0, "predict": 0}
    assert restored.schema_digest == binding.schema_digest
    assert restored.canonical_content_sha256 == binding.canonical_content_sha256
    assert restored.semantic_prediction_digest == binding.semantic_prediction_digest
    assert restored.artifact_sha256 != binding.artifact_sha256
    assert Path(restored.physical_path).read_bytes() == replica_path.read_bytes()
    assert recovery.read_events()[-2]["event_type"] == "artifact_rehydrated"
    assert recovery.read_events()[-1]["event_type"] == "artifact_binding_set_frozen"
    assert resolve_bound_artifact(
        recovery, binding.logical_artifact_id, expected_binding_set_digest=frozen.digest
    ) == Path(restored.physical_path)


def test_byte_mismatch_never_auto_rehydrates_or_signs_itself(tmp_path: Path) -> None:
    recovery = RunRecovery(tmp_path / "run")
    lease = recovery.create(_actor(), run_identity="r" * 64)
    corrupt = tmp_path / "corrupt.csv.gz"
    corrupt.write_bytes(b"tampered")
    binding = _binding(corrupt, b"expected", attempt_id="old", token=1)
    registry = TrustedReplicaRegistry([_replica("self", corrupt, binding)])
    rehydrator = ArtifactRehydrator(recovery, registry)

    assert rehydrator.classify(binding) is ArtifactCondition.ARTIFACT_BYTES_MISMATCH
    with pytest.raises(ArtifactRehydrationError, match="explicit authenticated"):
        rehydrator.rehydrate(binding, actor=_actor(), fencing_token=lease.fencing_token)
    with pytest.raises(ArtifactRehydrationError, match="cannot sign itself"):
        rehydrator.rehydrate(
            binding,
            actor=_actor(),
            fencing_token=lease.fencing_token,
            explicit=True,
            replica_id="self",
        )
    assert corrupt.read_bytes() == b"tampered"


def test_explicit_different_replica_preserves_old_manifest_and_switches_authority(
    tmp_path: Path,
) -> None:
    recovery = RunRecovery(tmp_path / "run")
    lease = recovery.create(_actor(), run_identity="r" * 64)
    corrupt = tmp_path / "old" / "trace.csv.gz"
    corrupt.parent.mkdir()
    corrupt.write_bytes(b"tampered")
    manifest = corrupt.with_suffix(".manifest.json")
    manifest.write_bytes(b'{"immutable":"old"}\n')
    old_manifest = manifest.read_bytes()
    binding = _binding(corrupt, b"expected", attempt_id="old", token=1)
    replica_path = tmp_path / "trusted" / "trace.csv.gz"
    replica_path.parent.mkdir()
    replica_path.write_bytes(b"valid alternate encoding")
    registry = TrustedReplicaRegistry([_replica("different", replica_path, binding)])
    rehydrator = ArtifactRehydrator(recovery, registry)

    restored = rehydrator.rehydrate(
        binding,
        actor=_actor(),
        fencing_token=lease.fencing_token,
        explicit=True,
        replica_id="different",
    )
    frozen = rehydrator.freeze_binding_set(
        [restored], actor=_actor(), fencing_token=lease.fencing_token
    )

    assert manifest.read_bytes() == old_manifest
    assert restored.attempt_id == lease.attempt_id
    assert restored.fencing_token == lease.fencing_token
    assert resolve_bound_artifact(recovery, binding.logical_artifact_id) == Path(
        restored.physical_path
    )
    with pytest.raises(FileExistsError, match="frozen"):
        rehydrator.freeze_binding_set(
            [replace(restored, physical_path=str(corrupt))],
            actor=_actor(),
            fencing_token=lease.fencing_token,
        )
    assert frozen.path.read_bytes() == frozen.path.read_bytes()
    expected_state = recovery.load_state()
    recovery.layout.state.write_text("{}\n", encoding="utf-8")
    assert recovery.rebuild_state() == expected_state


def test_rehydrate_is_forbidden_after_terminal_seal(tmp_path: Path) -> None:
    recovery = RunRecovery(tmp_path / "run")
    lease = recovery.create(_actor(), run_identity="r" * 64)
    missing = tmp_path / "missing"
    binding = _binding(missing, b"expected", attempt_id="old", token=1)
    replica = tmp_path / "replica"
    replica.write_bytes(b"alternate")
    rehydrator = ArtifactRehydrator(
        recovery, TrustedReplicaRegistry([_replica("replica", replica, binding)])
    )
    recovery.transition(
        RunState.COMPLETE_UNSEALED,
        actor=_actor(),
        fencing_token=lease.fencing_token,
        reason="complete",
    )
    recovery.transition(
        RunState.SEALED_SUCCESS,
        actor=_actor(),
        fencing_token=lease.fencing_token,
        reason="sealed",
    )

    with pytest.raises(ArtifactRehydrationError, match="sealed terminal"):
        rehydrator.rehydrate(binding, actor=_actor(), fencing_token=lease.fencing_token)
