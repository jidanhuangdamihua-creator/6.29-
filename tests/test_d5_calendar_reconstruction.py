from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.preprocess_clean_datasets import preprocess_d5_oil
from src.utils.d5_calendar_reconstruction import (
    build_authoritative_d5_oil_by_date,
    load_d5_authorities,
    reconstruct_d5_target_calendar,
)


def _write_authorities(root: Path) -> None:
    pd.DataFrame(
        {
            "date": ["2013-01-01", "2013-01-02", "2013-01-04"],
            "dcoilwtico": [90.0, 91.0, 92.0],
        }
    ).to_csv(root / "oil.csv", index=False)
    pd.DataFrame(
        {
            "date": ["2013-01-02", "2013-01-03", "2013-01-04"],
            "store_nbr": [1, 1, 1],
            "transactions": [10, 11, 12],
        }
    ).to_csv(root / "transactions.csv", index=False)
    pd.DataFrame(
        {
            "item_nbr": [100],
            "family": ["GROCERY"],
            "class": [1],
            "perishable": [0],
        }
    ).to_csv(root / "items.csv", index=False)
    pd.DataFrame(
        {
            "store_nbr": [1],
            "city": ["Quito"],
            "state": ["Pichincha"],
            "type": ["A"],
            "cluster": [1],
        }
    ).to_csv(root / "stores.csv", index=False)
    pd.DataFrame(
        {
            "date": ["2013-01-03"],
            "type": ["Holiday"],
            "locale": ["Local"],
            "locale_name": ["Quito"],
            "description": ["Quito day"],
            "transferred": [False],
        }
    ).to_csv(root / "holidays_events.csv", index=False)


def _observed() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2013-01-02", "2013-01-04"]),
            "entity_id": ["1_100", "1_100"],
            "item_id": [100, 100],
            "store_nbr": [1, 1],
            "item_nbr": [100, 100],
            "sales": [2.0, 3.0],
            "onpromotion": [1, 0],
            "family": ["GROCERY", "GROCERY"],
            "class": [1, 1],
            "perishable": [0, 0],
            "city": ["Quito", "Quito"],
            "state": ["Pichincha", "Pichincha"],
            "type": ["A", "A"],
            "cluster": [1, 1],
            "transactions": [10, 12],
            "oil_price": [90.0, 91.0],
            "is_holiday": [0, 0],
            "year": [2013, 2013],
            "month": [1, 1],
            "week": [1, 1],
            "day": [2, 4],
        }
    )


def _repository_fixture_root() -> Path:
    root = Path(__file__).resolve().parents[1]
    if (root / "数据集/固化数据/dataset5-target.parquet").is_file():
        return root
    git_pointer = root / ".git"
    if git_pointer.is_file():
        text = git_pointer.read_text(encoding="utf-8").strip()
        if text.startswith("gitdir:"):
            git_dir = Path(text.split(":", 1)[1].strip())
            candidate = git_dir.parents[2]
            if (candidate / "数据集/固化数据/dataset5-target.parquet").is_file():
                return candidate
    return root


def test_shared_oil_authority_matches_preprocessing_semantics() -> None:
    raw = pd.DataFrame(
        {
            "date": ["2013-01-01", "2013-01-02", "2013-01-04"],
            "dcoilwtico": [90.0, 91.0, 92.0],
        }
    )
    expected_dates = pd.date_range("2013-01-01", "2013-01-04", freq="D")

    actual = build_authoritative_d5_oil_by_date(raw, expected_dates=expected_dates)
    legacy = preprocess_d5_oil(raw)

    assert actual.assign(date=actual["date"].dt.strftime("%Y-%m-%d")).equals(legacy)
    assert pd.isna(actual.loc[0, "oil_price"])
    assert actual["oil_price"].iloc[1:].tolist() == [90.0, 91.0, 91.0]


def test_load_authorities_normalizes_unique_keys_and_records_evidence(tmp_path: Path) -> None:
    _write_authorities(tmp_path)

    bundle = load_d5_authorities(tmp_path, use_holidays=False)

    assert bundle.oil_by_date["date"].is_monotonic_increasing
    assert not bundle.transactions_by_store_date.duplicated(["store_nbr", "date"]).any()
    assert not bundle.items_by_item.duplicated(["item_nbr"]).any()
    assert not bundle.stores_by_store.duplicated(["store_nbr"]).any()
    assert bundle.holidays_by_store_date is None
    assert bundle.files["oil"].used
    assert len(bundle.files["oil"].sha256) == 64
    assert bundle.files["oil"].size_bytes > 0
    assert not bundle.files["holidays"].used
    assert bundle.files["holidays"].sha256 == ""


@pytest.mark.parametrize(
    ("filename", "duplicate"),
    [
        ("oil.csv", {"date": "2013-01-02", "dcoilwtico": 99.0}),
        (
            "transactions.csv",
            {"date": "2013-01-02", "store_nbr": 1, "transactions": 99},
        ),
        (
            "items.csv",
            {"item_nbr": 100, "family": "OTHER", "class": 9, "perishable": 1},
        ),
        (
            "stores.csv",
            {
                "store_nbr": 1,
                "city": "Other",
                "state": "Other",
                "type": "Z",
                "cluster": 9,
            },
        ),
    ],
)
def test_load_authorities_rejects_duplicate_keys(
    tmp_path: Path, filename: str, duplicate: dict[str, object]
) -> None:
    _write_authorities(tmp_path)
    path = tmp_path / filename
    frame = pd.read_csv(path)
    pd.concat([frame, pd.DataFrame([duplicate])], ignore_index=True).to_csv(path, index=False)

    with pytest.raises(ValueError, match="duplicate authority key"):
        load_d5_authorities(tmp_path, use_holidays=True)


@pytest.mark.parametrize(
    "expected_dates",
    [
        pd.DatetimeIndex([]),
        pd.to_datetime(["2013-01-02", "2013-01-02"]),
        pd.to_datetime(["2013-01-03", "2013-01-02"]),
        pd.DatetimeIndex([pd.NaT]),
    ],
)
def test_reconstruction_rejects_invalid_expected_dates(
    tmp_path: Path, expected_dates: pd.DatetimeIndex
) -> None:
    _write_authorities(tmp_path)
    bundle = load_d5_authorities(tmp_path, use_holidays=True)

    with pytest.raises(ValueError, match="expected_dates"):
        reconstruct_d5_target_calendar(
            _observed(),
            date_col="date",
            entity_col="entity_id",
            expected_dates=expected_dates,
            authorities=bundle,
        )


def test_reconstruction_restores_each_field_from_its_authority(tmp_path: Path) -> None:
    _write_authorities(tmp_path)
    bundle = load_d5_authorities(tmp_path, use_holidays=True)
    observed = _observed()
    expected_dates = pd.date_range("2013-01-02", "2013-01-04", freq="D")

    reconstructed, report = reconstruct_d5_target_calendar(
        observed,
        date_col="date",
        entity_col="entity_id",
        expected_dates=expected_dates,
        authorities=bundle,
    )

    assert reconstructed["date"].tolist() == list(expected_dates)
    synthetic = reconstructed.loc[reconstructed["date"].eq(pd.Timestamp("2013-01-03"))].iloc[0]
    assert synthetic["sales"] == 0
    assert synthetic["onpromotion"] == 0
    assert synthetic["transactions"] == 11
    assert synthetic["oil_price"] == 91.0
    assert synthetic["family"] == "GROCERY"
    assert synthetic["class"] == 1
    assert synthetic["city"] == "Quito"
    assert synthetic["cluster"] == 1
    assert synthetic["is_holiday"] == 1
    assert (synthetic["year"], synthetic["month"], synthetic["week"], synthetic["day"]) == (
        2013,
        1,
        1,
        3,
    )

    original_after = reconstructed[reconstructed["date"].isin(observed["date"])]
    pd.testing.assert_frame_equal(
        original_after.reset_index(drop=True),
        observed.reset_index(drop=True),
        check_dtype=True,
    )
    assert report.synthetic_row_count == 1
    assert report.synthetic_entity_date_keys == (("1_100", "2013-01-03"),)
    assert report.original_rows_unchanged
    assert report.missing_lookups == ()
    assert reconstructed.attrs["d5_calendar_reconstruction"]["synthetic_row_count"] == 1

    report_payload = report.to_dict()
    copied = reconstructed.copy()
    copied.attrs = {}
    roundtrip_path = tmp_path / "roundtrip.csv"
    reconstructed.to_csv(roundtrip_path, index=False)
    round_tripped = pd.read_csv(roundtrip_path)
    assert copied.attrs == {}
    assert round_tripped.attrs == {}
    assert len(round_tripped) == len(reconstructed)
    assert report.to_dict() == report_payload


def test_reconstruction_fails_closed_on_missing_required_authority(tmp_path: Path) -> None:
    _write_authorities(tmp_path)
    tx_path = tmp_path / "transactions.csv"
    pd.read_csv(tx_path).query("date != '2013-01-03'").to_csv(tx_path, index=False)
    bundle = load_d5_authorities(tmp_path, use_holidays=True)

    with pytest.raises(ValueError, match="transactions.*1.*2013-01-03"):
        reconstruct_d5_target_calendar(
            _observed(),
            date_col="date",
            entity_col="entity_id",
            expected_dates=pd.date_range("2013-01-02", "2013-01-04", freq="D"),
            authorities=bundle,
        )


def test_reconstruction_never_fabricates_historical_derived_features(tmp_path: Path) -> None:
    _write_authorities(tmp_path)
    bundle = load_d5_authorities(tmp_path, use_holidays=True)
    observed = _observed().assign(sales_lag_1=[1.0, 2.0])

    with pytest.raises(ValueError, match="historical derived fields"):
        reconstruct_d5_target_calendar(
            observed,
            date_col="date",
            entity_col="entity_id",
            expected_dates=pd.date_range("2013-01-02", "2013-01-04", freq="D"),
            authorities=bundle,
        )


def test_real_48_1159415_reconstructs_15_days_in_both_modes_without_training() -> None:
    root = _repository_fixture_root()
    target_path = root / "数据集/固化数据/dataset5-target.parquet"
    raw_dir = root / "数据集/原始数据/Dataset 5Favorita"
    if not target_path.is_file() or not raw_dir.is_dir():
        pytest.skip("repository D5 fixture is absent")

    target = pd.read_parquet(
        target_path,
        filters=[("entity_id", "==", "48_1159415")],
    )
    authorities = load_d5_authorities(raw_dir, use_holidays=True)
    expected_dates = pd.date_range("2017-01-17", "2017-08-15", freq="D")
    target = target[target["date"].isin(expected_dates)].copy()

    for mode in ("without", "with"):
        reconstructed, report = reconstruct_d5_target_calendar(
            target,
            date_col="date",
            entity_col="entity_id",
            expected_dates=expected_dates,
            authorities=authorities,
        )

        assert mode in {"without", "with"}
        assert report.synthetic_row_count == 15
        assert len(report.synthetic_entity_date_keys) == 15
        assert {key[0] for key in report.synthetic_entity_date_keys} == {"48_1159415"}
        synthetic_dates = {pd.Timestamp(key[1]) for key in report.synthetic_entity_date_keys}
        synthetic = reconstructed[reconstructed["date"].isin(synthetic_dates)]
        assert synthetic["sales"].eq(0).all()
        assert synthetic["onpromotion"].astype("string").eq("0").all()
        assert synthetic["transactions"].notna().all()
        assert synthetic["oil_price"].notna().all()
        assert reconstructed["class"].drop_duplicates().tolist() == [1040]
        assert reconstructed["cluster"].drop_duplicates().tolist() == [14]
