from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pandas as pd

import scripts.run_d5_experiment as d5_runner
from src.utils.d5_calendar_reconstruction import (
    load_d5_authorities,
    reconstruct_d5_source_history_calendar,
)
from src.utils.parquet_data_loader import (
    ParquetSourceTargetLoad,
    expected_target_dates_from_windows,
    load_parquet_source_target,
    load_parquet_source_target_with_diagnostics,
)


def _write_runtime_fixture(root: Path) -> tuple[Path, Path, dict[str, object], pd.DatetimeIndex]:
    raw_dir = root / "raw"
    parquet_dir = root / "parquet"
    raw_dir.mkdir()
    parquet_dir.mkdir()
    dates = pd.date_range("2017-01-01", "2017-01-31", freq="D")
    source_history_dates = pd.date_range("2016-08-20", "2017-02-15", freq="D")
    oil_dates = pd.date_range("2016-08-19", "2017-02-15", freq="D")

    pd.DataFrame(
        {"date": oil_dates, "dcoilwtico": range(len(oil_dates))}
    ).to_csv(raw_dir / "oil.csv", index=False)
    pd.DataFrame(
        {"date": source_history_dates, "store_nbr": 48, "transactions": range(100, 280)}
    ).to_csv(raw_dir / "transactions.csv", index=False)
    pd.DataFrame(
        {"item_nbr": [1159415], "family": ["GROCERY I"], "class": [1042], "perishable": [0]}
    ).to_csv(raw_dir / "items.csv", index=False)
    pd.DataFrame(
        {
            "store_nbr": [48],
            "city": ["Quito"],
            "state": ["Pichincha"],
            "type": ["A"],
            "cluster": [14],
        }
    ).to_csv(raw_dir / "stores.csv", index=False)
    pd.DataFrame(
        {
            "date": ["2017-01-01"],
            "type": ["Work Day"],
            "locale": ["National"],
            "locale_name": ["Ecuador"],
            "transferred": [False],
        }
    ).to_csv(raw_dir / "holidays_events.csv", index=False)

    source = pd.DataFrame(
        {
            "date": [pd.Timestamp("2017-01-01")],
            "store_nbr": [48],
            "item_nbr": [1159415],
            "sales": [1.0],
            "onpromotion": [False],
            "family": ["GROCERY I"],
            "entity_id": ["49_1159415"],
            "item_id": ["1159415"],
            "year": [2017],
            "month": [1],
            "week": [52],
            "day": [1],
            "class": [1042],
            "perishable": [0],
            "city": ["Quito"],
            "state": ["Pichincha"],
            "type": ["A"],
            "cluster": [14],
            "transactions": [50],
            "oil_price": [0.0],
            "is_holiday": [0],
        }
    )
    target = pd.DataFrame(
        {
            "date": dates.delete(14),
            "store_nbr": 48,
            "item_nbr": 1159415,
            "sales": 2.0,
            "onpromotion": False,
            "family": "GROCERY I",
            "entity_id": "48_1159415",
            "item_id": "1159415",
            "year": 2017,
            "month": 1,
            "week": dates.delete(14).isocalendar().week.astype("int64").to_numpy(),
            "day": dates.delete(14).day,
            "class": 1042,
            "perishable": 0,
            "city": "Quito",
            "state": "Pichincha",
            "type": "A",
            "cluster": 14,
            "transactions": [value for index, value in enumerate(range(100, 131)) if index != 14],
            "oil_price": [float(index) for index in range(30)],
            "is_holiday": 0,
        }
    )
    source.to_parquet(parquet_dir / "dataset5-source.parquet", index=False)
    target.to_parquet(parquet_dir / "dataset5-target.parquet", index=False)
    windows: dict[str, object] = {
        "dataset_id": 5,
        "train_start": "2017-01-01",
        "test_end": "2017-01-31",
        "target_train_window": {},
    }
    return raw_dir, parquet_dir, windows, dates


def test_d5_loader_returns_explicit_report_and_preserves_tuple_api(tmp_path: Path) -> None:
    raw_dir, parquet_dir, windows, expected_dates = _write_runtime_fixture(tmp_path)
    authorities = load_d5_authorities(raw_dir, use_holidays=True)

    loaded = load_parquet_source_target_with_diagnostics(
        dataset_id=5,
        source_path=parquet_dir / "dataset5-source.parquet",
        target_path=parquet_dir / "dataset5-target.parquet",
        windows=windows,
        source_history_days=180,
        expected_dates=expected_dates,
        d5_authorities=authorities,
    )

    assert isinstance(loaded, ParquetSourceTargetLoad)
    assert loaded.calendar_reconstruction is not None
    assert loaded.calendar_reconstruction.synthetic_entity_date_keys == (
        ("48_1159415", "2017-01-15"),
    )
    assert len(loaded.source_df) == 180
    assert loaded.source_df["date"].nunique() == 180
    assert loaded.source_df["date"].min() == pd.Timestamp("2016-08-20")
    assert loaded.source_df["date"].max() == pd.Timestamp("2017-02-15")
    assert len(loaded.target_df) == 31
    synthetic = loaded.target_df.loc[loaded.target_df["date"].eq("2017-01-15")]
    assert len(synthetic) == 1
    assert synthetic.iloc[0]["transactions"] == 248
    assert synthetic.iloc[0]["family"] == "GROCERY I"

    source_df, target_df = load_parquet_source_target(
        dataset_id=5,
        source_path=parquet_dir / "dataset5-source.parquet",
        target_path=parquet_dir / "dataset5-target.parquet",
        windows=windows,
        source_history_days=180,
        expected_dates=expected_dates,
        d5_authorities=authorities,
    )
    assert len(source_df) == len(loaded.source_df)
    assert len(target_df) == len(loaded.target_df)


def test_d5_loader_pushes_frozen_source_window_into_parquet_read(
    tmp_path: Path, monkeypatch
) -> None:
    raw_dir, parquet_dir, windows, expected_dates = _write_runtime_fixture(tmp_path)
    authorities = load_d5_authorities(raw_dir, use_holidays=True)
    real_read_parquet = pd.read_parquet
    calls: list[tuple[str, object]] = []

    def tracked_read_parquet(path, *args, **kwargs):
        calls.append((Path(path).name, kwargs.get("filters")))
        return real_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", tracked_read_parquet)
    load_parquet_source_target_with_diagnostics(
        dataset_id=5,
        source_path=parquet_dir / "dataset5-source.parquet",
        target_path=parquet_dir / "dataset5-target.parquet",
        windows=windows,
        source_history_days=180,
        expected_dates=expected_dates,
        d5_authorities=authorities,
    )

    source_filters = next(filters for name, filters in calls if name == "dataset5-source.parquet")
    assert source_filters == [
        ("date", ">=", pd.Timestamp("2016-08-20")),
        ("date", "<=", pd.Timestamp("2017-02-15")),
    ]


def test_runner_loads_authorities_once_and_passes_window_dates(monkeypatch) -> None:
    expected_dates = pd.date_range("2017-01-17", "2017-08-15", freq="D")
    bundle = object()
    load_result = Mock(spec=ParquetSourceTargetLoad)
    authority_loader = Mock(return_value=bundle)
    parquet_loader = Mock(return_value=load_result)
    monkeypatch.setattr(d5_runner, "load_d5_authorities", authority_loader)
    monkeypatch.setattr(d5_runner, "load_parquet_source_target_with_diagnostics", parquet_loader)

    result = d5_runner.load_d5_runtime_inputs(
        raw_dir=Path("raw"),
        source_path=Path("parquet/source.parquet"),
        target_path=Path("parquet/target.parquet"),
        windows={"dataset_id": 5, "train_start": "2017-01-17", "test_end": "2017-08-15"},
        source_history_days=180,
    )

    assert result is load_result
    authority_loader.assert_called_once_with(Path("raw"), use_holidays=True)
    assert parquet_loader.call_count == 1
    kwargs = parquet_loader.call_args.kwargs
    assert kwargs["d5_authorities"] is bundle
    assert kwargs["expected_dates"].equals(expected_dates)


def test_d5_source_calendarization_zero_fills_missing_store_day_transactions(tmp_path: Path) -> None:
    raw_dir, parquet_dir, _, _ = _write_runtime_fixture(tmp_path)
    expected_dates = pd.date_range("2016-08-20", "2017-02-15", freq="D")
    missing_date = pd.Timestamp("2017-01-15")

    transactions = pd.read_csv(raw_dir / "transactions.csv")
    transactions = transactions.loc[pd.to_datetime(transactions["date"]) != missing_date]
    transactions.to_csv(raw_dir / "transactions.csv", index=False)
    authorities = load_d5_authorities(raw_dir, use_holidays=True)

    source = pd.read_parquet(parquet_dir / "dataset5-source.parquet")
    source = pd.concat(
        [source.assign(date=day) for day in expected_dates if day != missing_date],
        ignore_index=True,
    )
    source["date"] = pd.to_datetime(source["date"])
    source["year"] = source["date"].dt.year
    source["month"] = source["date"].dt.month
    source["week"] = source["date"].dt.isocalendar().week.astype("int64")
    source["day"] = source["date"].dt.day

    reconstructed, report = reconstruct_d5_source_history_calendar(
        source,
        expected_dates=expected_dates,
        authorities=authorities,
    )

    generated = reconstructed.loc[reconstructed["date"].eq(missing_date)].iloc[0]
    assert len(reconstructed) == len(expected_dates)
    assert generated["transactions"] == 0
    assert report.synthetic_row_count == 1
    tx_stats = next(item for item in report.field_stats if item.field == "transactions")
    assert tx_stats.restored_count == 0
    assert tx_stats.zero_fill_count == 1
    assert reconstructed.attrs["d5_calendar_reconstruction"]["strategy"] == "vectorized_entity_date_merge"


def test_expected_target_dates_come_from_fixed_window_authority() -> None:
    dates = expected_target_dates_from_windows(
        {"dataset_id": 5, "train_start": "2017-01-17", "test_end": "2017-08-15"}
    )

    assert len(dates) == 211
    assert dates[0] == pd.Timestamp("2017-01-17")
    assert dates[-1] == pd.Timestamp("2017-08-15")
