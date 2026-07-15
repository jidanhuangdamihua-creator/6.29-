from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest

from src.protocols.feature_schema import (
    FutureKnownLineage,
    FutureKnownLineageViolation,
    audit_future_known_lineage,
    get_future_known_lineage,
    get_predictor_schema,
)
from src.protocols.sealing_protocol import get_target_window


@pytest.mark.parametrize("dataset_id", [f"D{i}" for i in range(1, 7)])
def test_every_future_known_field_has_complete_audited_lineage(dataset_id) -> None:
    schema = get_predictor_schema(dataset_id)
    lineage = get_future_known_lineage(dataset_id)
    report = audit_future_known_lineage(
        schema,
        lineage,
        cutoff=get_target_window(dataset_id).observed_end,
    )

    assert report.valid is True
    assert report.schema_digest == schema.digest
    assert report.audited_fields == tuple(item.feature_name for item in lineage)
    assert all(item.source_type for item in lineage)
    assert all(item.authority for item in lineage)
    assert all(item.generation_rule for item in lineage)
    assert all(len(item.code_digest) == 64 for item in lineage)


def test_lineage_rejects_sales_truth_prediction_and_sales_derived_dependencies() -> None:
    schema = get_predictor_schema("D1")
    lineage = list(get_future_known_lineage("D1"))
    base = lineage[0]

    for dependency in ("sales", "y_true", "prediction_h1", "rolling_sales_mean"):
        lineage[0] = replace(base, dependencies=(dependency,))
        with pytest.raises(FutureKnownLineageViolation, match="forbidden dependency"):
            audit_future_known_lineage(
                schema,
                lineage,
                cutoff=get_target_window("D1").observed_end,
            )


def test_lineage_fails_closed_on_missing_authority_or_post_cutoff_availability() -> None:
    schema = get_predictor_schema("D3")
    cutoff = get_target_window("D3").observed_end
    lineage = list(get_future_known_lineage("D3"))

    with pytest.raises(FutureKnownLineageViolation, match="authority"):
        replace(lineage[0], authority="")

    lineage = list(get_future_known_lineage("D3"))
    lineage[0] = replace(lineage[0], available_at=cutoff + timedelta(days=1))
    with pytest.raises(FutureKnownLineageViolation, match="after cutoff"):
        audit_future_known_lineage(schema, lineage, cutoff=cutoff)


def test_lineage_constructor_requires_a_sha256_code_digest() -> None:
    with pytest.raises(FutureKnownLineageViolation, match="code_digest"):
        FutureKnownLineage(
            feature_name="calendar_field",
            source_type="calendar",
            authority="sealed calendar",
            available_at=get_target_window("D1").observed_end,
            dependencies=("date",),
            generation_rule="extract calendar field",
            code_digest="not-a-digest",
        )
