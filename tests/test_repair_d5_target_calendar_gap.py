from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.operations.repair_d5_target_calendar_gap import (
    RepairBlocked,
    RepairSpec,
    _build_added_rows,
    build_candidate,
    inspect_authority,
    verify_candidate,
)


TARGETS = (
    (48, 364606),
    (48, 1159415),
    (48, 1159414),
    (48, 1349808),
    (48, 320682),
)
MISSING_DATES = tuple(
    pd.Timestamp(value)
    for value in (
        "2017-07-15",
        "2017-07-16",
        "2017-07-17",
        "2017-07-18",
        "2017-07-19",
        "2017-07-20",
        "2017-07-21",
        "2017-07-22",
        "2017-07-23",
        "2017-07-24",
        "2017-07-25",
        "2017-07-26",
        "2017-07-27",
        "2017-07-28",
        "2017-07-31",
    )
)
COLUMNS = (
    "date",
    "store_nbr",
    "item_nbr",
    "sales",
    "onpromotion",
    "family",
    "entity_id",
    "item_id",
    "year",
    "month",
    "week",
    "day",
    "class",
    "perishable",
    "city",
    "state",
    "type",
    "cluster",
    "transactions",
    "oil_price",
    "is_holiday",
)


def _frame(
    *,
    missing: set[tuple[int, int, pd.Timestamp]] | None = None,
    donor_mutation: tuple[pd.Timestamp, str, object] | None = None,
    static_mutation: tuple[str, object] | None = None,
    remove_donors_on: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if missing is None:
        missing = {(48, 1159415, date) for date in MISSING_DATES}
    rows: list[dict[str, object]] = []
    for date in pd.date_range("2017-01-17", "2017-08-14", freq="D"):
        for store_nbr, item_nbr in TARGETS:
            if (store_nbr, item_nbr, date) in missing:
                continue
            if remove_donors_on == date and item_nbr != 1159415:
                continue
            row = {
                "date": date,
                "store_nbr": store_nbr,
                "item_nbr": item_nbr,
                "sales": float(item_nbr % 10),
                "onpromotion": "False",
                "family": "GROCERY I",
                "entity_id": f"{store_nbr}_{item_nbr}",
                "item_id": str(item_nbr),
                "year": date.year,
                "month": date.month,
                "week": int(date.isocalendar().week),
                "day": date.day,
                "class": 1040,
                "perishable": 0,
                "city": "Quito",
                "state": "Pichincha",
                "type": "A",
                "cluster": 14,
                "transactions": 3000 + date.day,
                "oil_price": 45.0 + date.day / 100,
                "is_holiday": 0,
            }
            if donor_mutation and date == donor_mutation[0] and item_nbr == 364606:
                row[donor_mutation[1]] = donor_mutation[2]
            if static_mutation and row["item_nbr"] == 1159415 and date == pd.Timestamp("2017-01-17"):
                row[static_mutation[0]] = static_mutation[1]
            rows.append(row)
    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame["store_nbr"] = frame["store_nbr"].astype("int32")
    frame["item_nbr"] = frame["item_nbr"].astype("int32")
    return frame


def _write_fixture(tmp_path: Path, frame: pd.DataFrame) -> tuple[Path, RepairSpec, pa.Table]:
    path = tmp_path / "target.parquet"
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(table, path)
    source = pq.read_table(path)
    spec = RepairSpec(
        authority_path=path,
        expected_sha256=__import__("hashlib").sha256(path.read_bytes()).hexdigest(),
        expected_rows=len(source),
        target_keys=TARGETS,
        missing_dates=MISSING_DATES,
        formal_start=pd.Timestamp("2017-01-17"),
        formal_end=pd.Timestamp("2017-08-14"),
        blind_start=pd.Timestamp("2017-02-16"),
        blind_end=pd.Timestamp("2017-08-14"),
    )
    return path, spec, source


def test_inspect_authority_identifies_only_the_fifteen_requested_keys(tmp_path: Path) -> None:
    _, spec, source = _write_fixture(tmp_path, _frame())

    plan = inspect_authority(source, spec=spec)

    assert plan.mode == "repair"
    assert plan.missing_exact_keys == tuple(
        (48, 1159415, date) for date in MISSING_DATES
    )
    assert len(plan.added_rows) == 15


def test_extra_missing_date_fails_closed(tmp_path: Path) -> None:
    missing = {(48, 1159415, date) for date in MISSING_DATES}
    missing.add((48, 1159415, pd.Timestamp("2017-07-14")))
    _, spec, source = _write_fixture(tmp_path, _frame(missing=missing))

    with pytest.raises(RepairBlocked, match="missing exact key set"):
        inspect_authority(source, spec=spec)


def test_old_sha_mismatch_fails_closed(tmp_path: Path) -> None:
    _, spec, source = _write_fixture(tmp_path, _frame())
    wrong = RepairSpec(**{**spec.__dict__, "expected_sha256": "0" * 64})

    with pytest.raises(RepairBlocked, match="old SHA-256"):
        inspect_authority(source, spec=wrong)


def test_missing_donor_fails_closed(tmp_path: Path) -> None:
    date = pd.Timestamp("2017-07-15")
    _, spec, source = _write_fixture(tmp_path, _frame(remove_donors_on=date))
    frame = source.to_pandas()

    with pytest.raises(RepairBlocked, match="donor"):
        _build_added_rows(frame, source, spec)


def test_inconsistent_date_store_donor_fails_closed(tmp_path: Path) -> None:
    date = pd.Timestamp("2017-07-15")
    _, spec, source = _write_fixture(
        tmp_path,
        _frame(donor_mutation=(date, "transactions", 9999)),
    )

    with pytest.raises(RepairBlocked, match="transactions.*2017-07-15"):
        inspect_authority(source, spec=spec)


def test_inconsistent_entity_static_field_fails_closed(tmp_path: Path) -> None:
    _, spec, source = _write_fixture(
        tmp_path,
        _frame(static_mutation=("family", "OTHER")),
    )

    with pytest.raises(RepairBlocked, match="family.*1159415"):
        inspect_authority(source, spec=spec)


def test_candidate_has_expected_counts_schema_dtypes_and_unchanged_old_rows(
    tmp_path: Path,
) -> None:
    _, spec, source = _write_fixture(tmp_path, _frame())
    plan = inspect_authority(source, spec=spec)
    candidate = build_candidate(source, plan)

    report = verify_candidate(source, candidate, plan)

    assert report["new_total_rows"] == 1050
    assert report["new_formal_rows"] == 1050
    assert report["new_blind_rows"] == 900
    assert report["missing_exact_keys_after"] == 0
    assert report["duplicate_exact_keys_after"] == 0
    assert report["old_rows_changed"] == 0
    assert report["schema_preserved"] is True
    additions = candidate.slice(len(source)).to_pandas()
    assert additions["sales"].eq(0).all()
    assert additions["onpromotion"].eq("False").all()
    assert set(additions["date"]) == set(MISSING_DATES)


def test_already_complete_authority_is_a_noop_and_never_duplicates(tmp_path: Path) -> None:
    frame = _frame(missing=set())
    _, spec, source = _write_fixture(tmp_path, frame)

    plan = inspect_authority(source, spec=spec)
    candidate = build_candidate(source, plan)

    assert plan.mode == "already_complete"
    assert len(candidate) == len(source)
    assert candidate.equals(source)
    assert verify_candidate(source, candidate, plan)["added_rows"] == 0


def test_unknown_field_fails_closed(tmp_path: Path) -> None:
    frame = _frame()
    frame["unclassified_field"] = 1
    _, spec, source = _write_fixture(tmp_path, frame)

    with pytest.raises(RepairBlocked, match="unsupported target columns"):
        inspect_authority(source, spec=spec)


def test_formal_frozen_identity_tracks_repaired_d5_target() -> None:
    import hashlib

    from src.protocols.formal_deployment_manifest import FROZEN_PARQUETS

    path = Path("数据集/固化数据/d1_d6_sealed_v1/dataset5/target.parquet")
    expected = FROZEN_PARQUETS[5]["target"]

    assert expected == {
        "rows": 7338,
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
