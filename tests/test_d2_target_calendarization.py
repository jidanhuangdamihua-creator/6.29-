from __future__ import annotations

import json

import pandas as pd
import pytest


_REPAIR_DATES = (
    "2018-08-15",
    "2018-11-01",
    "2018-12-08",
    "2018-12-25",
    "2018-12-26",
)


def _target_frame(*, include_repair_dates: bool = False) -> pd.DataFrame:
    dates = pd.date_range("2014-01-02", "2018-12-31", freq="D")
    if not include_repair_dates:
        dates = dates.difference(
            pd.DatetimeIndex(
                [*pd.date_range("2014-02-01", periods=18, freq="D"), *_REPAIR_DATES]
            )
        )
    frame = pd.DataFrame(
        {
            "date": dates,
            "brand_id": pd.Series(1, index=range(len(dates)), dtype="int64"),
            "item_id": pd.Series(10, index=range(len(dates)), dtype="int64"),
            "sales": pd.Series(2.0, index=range(len(dates)), dtype="float64"),
            "promo": pd.Series(1.0, index=range(len(dates)), dtype="float64"),
            "entity_id": pd.Series(1, index=range(len(dates)), dtype="object"),
            "year": pd.Series(dates.year.to_numpy(), index=range(len(dates)), dtype="float64"),
            "month": pd.Series(dates.month.to_numpy(), index=range(len(dates)), dtype="float64"),
            "week": pd.Series(dates.isocalendar().week.to_numpy(), index=range(len(dates)), dtype="float64"),
            "day": pd.Series(dates.day.to_numpy(), index=range(len(dates)), dtype="float64"),
        }
    )
    frame.attrs["split_role"] = "target"
    return frame


def test_d2_target_producer_adds_only_authorized_zero_demand_rows() -> None:
    from src.protocols.d2_target_calendarization import calendarize_d2_target_frame

    original = _target_frame()
    repaired, evidence = calendarize_d2_target_frame(original)

    assert len(repaired) == len(original) + 5
    inserted = repaired[repaired["date"].isin(pd.to_datetime(_REPAIR_DATES))]
    assert len(inserted) == 5
    assert inserted[["brand_id", "item_id"]].drop_duplicates().to_records(index=False).tolist() == [(1, 10)]
    assert inserted["sales"].tolist() == [0.0] * 5
    assert inserted["promo"].tolist() == [0.0] * 5
    assert inserted["entity_id"].tolist() == [1] * 5
    assert inserted["year"].tolist() == [2018.0] * 5
    assert inserted["month"].tolist() == [8.0, 11.0, 12.0, 12.0, 12.0]
    assert inserted["day"].tolist() == [15.0, 1.0, 8.0, 25.0, 26.0]
    assert evidence["inserted_dates"] == list(_REPAIR_DATES)
    assert evidence["inserted_count"] == 5
    assert list(repaired.columns) == list(original.columns)
    assert repaired.dtypes.astype(str).to_dict() == original.dtypes.astype(str).to_dict()


def test_d2_target_producer_preserves_original_rows_and_is_idempotent() -> None:
    from src.protocols.d2_target_calendarization import (
        calendarize_d2_target_frame,
        target_semantic_digest,
    )

    original = _target_frame()
    repaired, first_evidence = calendarize_d2_target_frame(original)
    rerun, second_evidence = calendarize_d2_target_frame(repaired)

    assert target_semantic_digest(repaired) == target_semantic_digest(rerun)
    assert len(rerun) == 1807
    assert second_evidence["inserted_count"] == 0
    assert second_evidence["inserted_dates"] == []
    original_keys = set(zip(original["date"], original["brand_id"], original["item_id"]))
    rerun_original = rerun[
        rerun.apply(
            lambda row: (row["date"], row["brand_id"], row["item_id"]) in original_keys,
            axis=1,
        )
    ]
    assert target_semantic_digest(rerun_original) == target_semantic_digest(original)
    assert first_evidence["policy"] == "closed_day_zero_demand"


def test_d2_target_producer_repairs_only_authorized_existing_row() -> None:
    from src.protocols.d2_target_calendarization import calendarize_d2_target_frame

    original = _target_frame()
    repair_mask = original["date"].eq(pd.Timestamp("2018-06-02"))
    original.loc[repair_mask, ["entity_id", "year", "month", "week", "day"]] = None

    repaired, evidence = calendarize_d2_target_frame(original)
    repaired_row = repaired.loc[repaired["date"].eq(pd.Timestamp("2018-06-02"))].iloc[0]

    assert repaired_row[["entity_id", "year", "month", "week", "day"]].tolist() == [
        1,
        2018.0,
        6.0,
        22.0,
        2.0,
    ]
    assert evidence["existing_row_repair"]["changed_fields"] == [
        "entity_id",
        "year",
        "month",
        "week",
        "day",
    ]
    assert evidence["existing_row_repair"]["changed_cell_count"] == 5

    rerun, rerun_evidence = calendarize_d2_target_frame(repaired)
    assert rerun_evidence["existing_row_repair"]["changed_cell_count"] == 0
    pd.testing.assert_frame_equal(repaired, rerun)


def test_d2_target_producer_rejects_calendar_defect_outside_authorized_row() -> None:
    from src.protocols.d2_target_calendarization import calendarize_d2_target_frame
    from src.protocols.experiment_protocol import ProtocolViolation

    original = _target_frame()
    original.loc[original["date"].eq(pd.Timestamp("2018-06-03")), "week"] = None

    with pytest.raises(ProtocolViolation, match="outside the authorized"):
        calendarize_d2_target_frame(original)


def test_d2_target_rebuilder_recloses_schema_identity_and_is_idempotent(
    tmp_path,
) -> None:
    from scripts.rebuild_d2_target_authority import rebuild_d2_target_authority
    from src.protocols.d2_target_calendarization import calendarize_d2_target_frame

    dataset_root = tmp_path / "dataset2"
    dataset_root.mkdir()
    target, _ = calendarize_d2_target_frame(_target_frame())
    repair_mask = target["date"].eq(pd.Timestamp("2018-06-02"))
    target.loc[repair_mask, ["entity_id", "year", "month", "week", "day"]] = None
    target.to_parquet(dataset_root / "target.parquet", index=False)

    sidecars = {
        "calendarization_audit.json": {},
        "manifest.json": {
            "artifacts": {"target": {}},
            "parent_artifacts": {"target": {}},
            "dataset_canonicalization": {},
            "sealed_identity": {},
            "schema_fingerprints": {"target": "stale"},
        },
        "target_schema.json": {"schema_digest": "stale", "null_counts": {}},
        "provenance.json": {"formal_input_identity": {"target": {}}},
        "validation_report.json": {
            "artifact_identity": {"target": {}},
            "checks": [],
        },
    }
    for name, payload in sidecars.items():
        (dataset_root / name).write_text(json.dumps(payload), encoding="utf-8")

    first = rebuild_d2_target_authority(
        target_path=dataset_root / "target.parquet",
        audit_path=dataset_root / "calendarization_audit.json",
    )
    schema = json.loads((dataset_root / "target_schema.json").read_text(encoding="utf-8"))
    manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))

    assert first["existing_row_changed_cell_count"] == 5
    assert schema["schema_digest"] == manifest["schema_fingerprints"]["target"]
    assert all(count == 0 for count in schema["null_counts"].values())

    second = rebuild_d2_target_authority(
        target_path=dataset_root / "target.parquet",
        audit_path=dataset_root / "calendarization_audit.json",
    )
    assert second["status"] == "verified_idempotent"
    assert second["existing_row_changed_cell_count"] == 0
