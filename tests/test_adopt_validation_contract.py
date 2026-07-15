from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.protocols.adopt_validation import (
    AdoptValidationFailureReasonV1,
    VALIDATION_POLICY_VERSION,
    map_validator_exception,
    validate_manifest,
    validate_adopted_artifact,
)


def test_adopt_failure_reason_enum_is_closed_and_normative() -> None:
    assert AdoptValidationFailureReasonV1.PARENT_ARTIFACT_MISSING.value == "PARENT_ARTIFACT_MISSING"
    assert AdoptValidationFailureReasonV1.VALIDATOR_INTERNAL_ERROR.value == "VALIDATOR_INTERNAL_ERROR"
    assert "UNKNOWN" not in {reason.value for reason in AdoptValidationFailureReasonV1}


def test_unknown_validator_exception_maps_to_internal_error() -> None:
    reason = map_validator_exception(RuntimeError("unexpected validator failure"))
    assert reason is AdoptValidationFailureReasonV1.VALIDATOR_INTERNAL_ERROR


def test_adopt_manifest_missing_required_fields_is_a_closed_failure() -> None:
    result = validate_manifest({"provenance_level": "adopted_solidified"}, dataset_id="D3")

    assert result.status == "failed"
    assert AdoptValidationFailureReasonV1.MANIFEST_REQUIRED_FIELD_MISSING in result.failure_reasons


def test_adopt_validator_reports_hash_and_size_mismatch_without_warning_downgrade(tmp_path: Path) -> None:
    path = tmp_path / "parent.parquet"
    pd.DataFrame(
        {
            "entity_id": ["1"],
            "date": [pd.Timestamp("2024-01-01")],
            "sales": [1.0],
        }
    ).to_parquet(path, index=False)

    result = validate_adopted_artifact(
        path,
        expected_sha256="0" * 64,
        expected_size_bytes=path.stat().st_size + 1,
        validation_policy_version=VALIDATION_POLICY_VERSION,
    )

    assert result.status == "failed"
    assert result.failure_reasons == (
        AdoptValidationFailureReasonV1.PARENT_ARTIFACT_HASH_MISMATCH,
        AdoptValidationFailureReasonV1.PARENT_ARTIFACT_SIZE_MISMATCH,
    )
