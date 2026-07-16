from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pytest

import scripts.adopt_and_seal_d3_d6 as adoption
from scripts.adopt_and_seal_d3_d6 import adopt_and_seal_dataset
from scripts.regenerate_d1_d2_parquets import build_and_seal_dataset
from src.data_processing.sealed_daily import (
    calendarize_and_fill,
    canonicalize_source_sales,
)
from src.protocols.sealing_protocol import get_source_pretrain_window
from src.protocols.gate1_transformation import canonical_digest
from tools.operations import materialize_d1_d6_sealed_authority as gate1x_operator


def _d2_long_frame(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [date for date in dates for brand in range(1, 4) for item in range(1, 11)],
            "brand": [brand for _date in dates for brand in range(1, 4) for _item in range(1, 11)],
            "item": [item for _date in dates for _brand in range(1, 4) for item in range(1, 11)],
            "sales": [
                float(brand + item)
                for _date in dates
                for brand in range(1, 4)
                for item in range(1, 11)
            ],
            "promo": 0,
        }
    )


_REPAIR_REASONS = {
    "original_nan",
    "original_negative",
    "calendar_row_missing",
}


@pytest.fixture(autouse=True)
def _stub_frozen_raw_authority(monkeypatch: pytest.MonkeyPatch) -> None:
    def identity(dataset_id: int) -> dict:
        records = gate1x_operator._frozen_raw_records(dataset_id)
        files = [
            {
                "name": f"D{dataset_id}:{record['path']}",
                "path": record["path"],
                "size_bytes": 1,
                "sha256": record["sha256"],
            }
            for record in records
        ]
        return {
            "dataset": f"D{dataset_id}",
            "files": files,
            "approved_input_set_digest": canonical_digest(files),
            "snapshot_identity": canonical_digest(
                [(item["path"], item["sha256"]) for item in files]
            ),
            "verified_from_bytes": True,
        }

    monkeypatch.setattr(adoption, "_raw_authority_identity", identity)


def _write_d3_adopted_pair(
    root: Path,
    *,
    source_dates: pd.DatetimeIndex,
    source_sales: list[float],
    dataset_id: int = 3,
) -> tuple[Path, Path, pd.DataFrame]:
    source = root / "dataset3-source.parquet"
    target = root / "dataset3-target.parquet"
    source_frame = pd.DataFrame(
        {
            "entity_id": ["1"] * len(source_dates),
            "store_id": [1] * len(source_dates),
            "product_id": [1] * len(source_dates),
            "store_nbr": [1] * len(source_dates),
            "item_nbr": [1] * len(source_dates),
            "item_id": [1] * len(source_dates),
            "date": source_dates,
            "sales": source_sales,
        }
    )
    target_dates = pd.date_range(source_dates.max() + pd.Timedelta(days=1), periods=3, freq="D")
    target_frame = pd.DataFrame(
        {
            "entity_id": ["10"] * len(target_dates),
            "store_id": [10] * len(target_dates),
            "product_id": [1] * len(target_dates),
            "store_nbr": [10] * len(target_dates),
            "item_nbr": [1] * len(target_dates),
            "item_id": [1] * len(target_dates),
            "date": target_dates,
            "sales": [10.0, 11.0, 12.0],
        }
    )
    for frame in (source_frame, target_frame):
        if dataset_id == 3:
            frame["SchoolHoliday"] = 0
        elif dataset_id == 4:
            frame["activity_flag"] = 1
            frame["discount"] = 0.0
            frame["holiday_flag"] = 0
            frame["precpt"] = 0.0
            frame["avg_temperature"] = 20.0
            frame["avg_humidity"] = 50.0
            frame["avg_wind_level"] = 1.0
        elif dataset_id == 5:
            frame["perishable"] = 1
            frame["onpromotion"] = 0
            frame["oil_price"] = 50.0
            frame["is_holiday"] = 0
        elif dataset_id == 6:
            frame["weekday"] = "Monday"
            frame["wday"] = 1
            frame["wm_yr_wk"] = 1
            frame["event_name_1"] = "none"
            frame["event_type_1"] = "none"
            frame["event_name_2"] = "none"
            frame["event_type_2"] = "none"
            frame["snap"] = 0
            frame["sell_price"] = 1.0
    source_frame.to_parquet(source, index=False)
    target_frame.to_parquet(target, index=False)
    return source, target, source_frame


def _published_repair_proofs(dataset_dir: Path) -> tuple[dict, dict, dict]:
    sidecar = json.loads(
        (dataset_dir / "source_sales_canonicalization.json").read_text(encoding="utf-8")
    )
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads(
        (dataset_dir / "adopt_validation_report.json").read_text(encoding="utf-8")
    )
    return sidecar, manifest["source_sales_repair"], report["source_sales_repair"]


def _assert_complete_repair_proof(proof: dict) -> None:
    assert proof["status"] not in {
        "not_reconstructed_during_adoption",
        "unavailable",
    }
    assert set(proof["repair_reason_counts"]) == _REPAIR_REASONS
    assert sum(proof["repair_reason_counts"].values()) == len(proof["affected_rows"])
    assert len(proof["affected_rows"]) <= proof["rows_examined"]
    assert re.fullmatch(r"[0-9a-f]{64}", proof["repair_mask_sha256"])
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", proof["affected_date_digest"])


def _assert_public_repair_identity(dataset_dir: Path) -> None:
    sidecar, manifest_proof, report_proof = _published_repair_proofs(dataset_dir)
    _assert_complete_repair_proof(sidecar)
    assert sidecar == manifest_proof == report_proof


def test_calendarize_and_fill_adds_only_calendar_rows_and_preserves_unused_nulls() -> None:
    frame = pd.DataFrame(
        {
            "entity": ["A", "A"],
            "date": ["2024-01-01", "2024-01-03"],
            "sales": [3.0, 5.0],
            "unused_audit": [None, None],
        }
    )

    result = calendarize_and_fill(
        frame,
        group_cols=("entity",),
        start="2024-01-01",
        end="2024-01-03",
        fill_rules={"sales": "zero"},
    )

    assert result["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-01",
        "2024-01-02",
        "2024-01-03",
    ]
    assert result["sales"].tolist() == [3.0, 0.0, 5.0]
    assert result["unused_audit"].isna().all()
    assert result.attrs["fill_policy_engine_version"] == "calendarize-fill/v1"
    assert result.attrs["synthetic_date_count"] == 1


def test_source_sales_canonicalization_records_closed_repair_reasons() -> None:
    frame = pd.DataFrame(
        {
            "entity": ["A", "A", "A", "A"],
            "date": pd.date_range("2024-01-01", periods=4, freq="D"),
            "sales": [np.nan, -2.0, np.inf, np.nan],
        }
    )

    with pytest.raises(ValueError, match="infinity"):
        canonicalize_source_sales(frame, calendar_row_missing=[False, False, False, True])

    frame.loc[2, "sales"] = 4.0
    repaired, audit = canonicalize_source_sales(
        frame,
        calendar_row_missing=[False, False, False, True],
    )

    assert repaired["sales"].tolist() == [0.0, 0.0, 4.0, 0.0]
    assert audit["repair_reason_counts"] == {
        "original_nan": 1,
        "original_negative": 1,
        "calendar_row_missing": 1,
    }
    assert len(audit["repair_mask_sha256"]) == 64


def test_d2_raw_rebuild_publishes_approved_june_calendar_row(tmp_path: Path) -> None:
    dates = pd.date_range("2018-05-30", "2018-06-30", freq="D").difference(
        pd.DatetimeIndex([pd.Timestamp("2018-06-02")])
    )
    output = tmp_path / "d1_d6_sealed_v1"

    dataset_dir = build_and_seal_dataset(
        2,
        _d2_long_frame(dates),
        output_dir=output,
        raw_input_path=tmp_path / "d2-raw.csv",
    )

    assert dataset_dir == output / "dataset2"
    assert (dataset_dir / "source.parquet").is_file()
    assert (dataset_dir / "target.parquet").is_file()
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["provenance_level"] == "raw_rebuilt"
    assert manifest["fill_policy_engine_version"] == "calendarize-fill/v1"
    assert manifest["dataset_canonicalization"]["rule_id"] == "d2_june_absent_transaction_day_v1"

    target = pd.read_parquet(dataset_dir / "target.parquet")
    june_2 = target[target["date"] == pd.Timestamp("2018-06-02")]
    assert len(june_2) == 1
    assert june_2.iloc[0]["sales"] == 0.0
    assert june_2.iloc[0]["promo"] == 0


def test_adoption_records_parent_identity_and_contract_publication_proof(tmp_path: Path) -> None:
    source = tmp_path / "dataset3-source.parquet"
    target = tmp_path / "dataset3-target.parquet"
    source_frame = pd.DataFrame(
        {
            "entity_id": ["1", "1"],
            "store_id": [1, 1],
            "item_id": [1, 1],
            "date": pd.date_range("2015-01-01", periods=2, freq="D"),
            "sales": [1.0, 2.0],
            "SchoolHoliday": [0, 0],
        }
    )
    target_frame = source_frame.assign(
        entity_id="10",
        store_id=10,
        date=pd.date_range("2015-02-02", periods=len(source_frame), freq="D"),
    )
    source_frame.to_parquet(source, index=False)
    target_frame.to_parquet(target, index=False)

    dataset_dir = adopt_and_seal_dataset(
        3,
        source_path=source,
        target_path=target,
        output_dir=tmp_path / "sealed",
    )

    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["provenance_level"] == "gate1_contract_transformed"
    assert manifest["content_validation_level"] == "gate1_contract_validated"
    assert manifest["adopted_content_validated"] is True
    proof = manifest["gate1_publication_proof"]
    assert proof["status"] == "publication_ready"
    assert proof["formal_preflight"]["status"] == "passed"
    assert proof["availability_no_leakage"]["target_day_actual_isolated"] is True
    assert manifest["parent_artifacts"]["source"]["sha256"]
    assert manifest["parent_artifacts"]["source"]["size_bytes"] == source.stat().st_size
    assert manifest["parent_artifacts"]["source"]["first_seen_at"] is None
    assert manifest["parent_artifacts"]["source"]["first_seen_reliability"] == "unavailable"


def test_adoption_failure_does_not_publish_dataset_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        adopt_and_seal_dataset(
            3,
            source_path=tmp_path / "missing-source.parquet",
            target_path=tmp_path / "missing-target.parquet",
            output_dir=tmp_path / "sealed",
        )

    assert not (tmp_path / "sealed" / "dataset3").exists()


@pytest.mark.parametrize("dataset_id", [3, 4, 5, 6])
def test_d3_d6_adoption_calls_real_calendarization_and_canonicalization_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dataset_id: int,
) -> None:
    window = get_source_pretrain_window(dataset_id)
    full_dates = pd.date_range(window.pretrain_start, window.pretrain_end, freq="D")
    missing_date = full_dates[65]
    source_dates = full_dates.difference(pd.DatetimeIndex([missing_date]))
    source_sales = [5.0] * len(source_dates)
    source_sales[0] = np.nan
    source_sales[1] = -2.0
    source, target, _ = _write_d3_adopted_pair(
        tmp_path,
        source_dates=source_dates,
        source_sales=source_sales,
        dataset_id=dataset_id,
    )
    calls = {"calendarize": 0, "canonicalize": 0}
    call_order: list[str] = []
    observed_calendar_mask: list[bool] = []

    def calendarize_spy(*args, **kwargs):
        calls["calendarize"] += 1
        call_order.append("calendarize")
        result = calendarize_and_fill(*args, **kwargs)
        observed_calendar_mask[:] = result.attrs["calendar_row_missing_mask"]
        return result

    def canonicalize_spy(*args, **kwargs):
        calls["canonicalize"] += 1
        call_order.append("canonicalize")
        assert list(kwargs["calendar_row_missing"]) == observed_calendar_mask
        return canonicalize_source_sales(*args, **kwargs)

    monkeypatch.setattr(adoption, "calendarize_and_fill", calendarize_spy, raising=False)
    monkeypatch.setattr(
        adoption,
        "canonicalize_source_sales",
        canonicalize_spy,
        raising=False,
    )

    dataset_dir = adoption.adopt_and_seal_dataset(
        dataset_id,
        source_path=source,
        target_path=target,
        output_dir=tmp_path / "sealed",
    )

    assert calls == {"calendarize": 1, "canonicalize": 1}
    assert call_order == ["calendarize", "canonicalize"]
    assert sum(observed_calendar_mask) == 1
    sidecar, manifest_proof, report_proof = _published_repair_proofs(dataset_dir)
    assert sidecar["repair_reason_counts"] == {
        "original_nan": 1,
        "original_negative": 1,
        "calendar_row_missing": 1,
    }
    _assert_complete_repair_proof(sidecar)
    assert sidecar == manifest_proof == report_proof


def test_d3_zero_repair_proof_is_stable_while_history_view_is_reconstructed(
    tmp_path: Path,
) -> None:
    source_dates = pd.date_range("2014-08-06", "2015-02-01", freq="D")
    source, target, parent_frame = _write_d3_adopted_pair(
        tmp_path,
        source_dates=source_dates,
        source_sales=[5.0] * len(source_dates),
    )

    first = adopt_and_seal_dataset(
        3,
        source_path=source,
        target_path=target,
        output_dir=tmp_path / "sealed-first",
    )
    second = adopt_and_seal_dataset(
        3,
        source_path=source,
        target_path=target,
        output_dir=tmp_path / "sealed-second",
    )

    first_proof, first_manifest_proof, first_report_proof = _published_repair_proofs(first)
    second_proof, second_manifest_proof, second_report_proof = _published_repair_proofs(second)
    assert first_proof["repair_reason_counts"] == {
        "original_nan": 0,
        "original_negative": 0,
        "calendar_row_missing": 0,
    }
    assert first_proof["affected_rows"] == []
    _assert_complete_repair_proof(first_proof)
    assert first_proof == first_manifest_proof == first_report_proof
    assert first_proof == second_proof == second_manifest_proof == second_report_proof
    rebuilt = pd.read_parquet(first / "source.parquet")
    assert {"year", "month", "day"}.issubset(rebuilt.columns)
    assert rebuilt["sales"].tolist() == parent_frame["sales"].tolist()
    assert (first / "source.parquet").read_bytes() != source.read_bytes()
    assert pd.read_parquet(first / "source.parquet").equals(
        pd.read_parquet(second / "source.parquet")
    )


@pytest.mark.parametrize(
    ("artifact", "mutation"),
    [
        ("source_sales_canonicalization.json", "delete_rows_examined"),
        ("manifest.json", "null_repair_mask"),
        ("adopt_validation_report.json", "change_reason_count"),
    ],
)
def test_d3_adoption_public_proof_identity_detects_producer_tamper(
    tmp_path: Path,
    artifact: str,
    mutation: str,
) -> None:
    source_dates = pd.date_range("2014-08-06", "2015-02-01", freq="D")
    source, target, _ = _write_d3_adopted_pair(
        tmp_path,
        source_dates=source_dates,
        source_sales=[5.0] * len(source_dates),
    )
    dataset_dir = adopt_and_seal_dataset(
        3,
        source_path=source,
        target_path=target,
        output_dir=tmp_path / "sealed",
    )
    _assert_public_repair_identity(dataset_dir)

    path = dataset_dir / artifact
    payload = json.loads(path.read_text(encoding="utf-8"))
    proof = payload
    if artifact == "manifest.json" or artifact == "adopt_validation_report.json":
        proof = payload["source_sales_repair"]
    if mutation == "delete_rows_examined":
        del proof["rows_examined"]
    elif mutation == "null_repair_mask":
        proof["repair_mask_sha256"] = None
    else:
        proof["repair_reason_counts"]["original_nan"] += 1
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises((AssertionError, KeyError, TypeError)):
        _assert_public_repair_identity(dataset_dir)
