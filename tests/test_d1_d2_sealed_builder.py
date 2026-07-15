from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.regenerate_d1_d2_parquets import build_and_seal_dataset


def _raw_frame(dataset_id: int) -> pd.DataFrame:
    start = "2018-06-01" if dataset_id == 2 else "2017-06-01"
    dates = pd.date_range(start, periods=30, freq="D")
    if dataset_id == 2:
        dates = dates.difference(pd.DatetimeIndex([pd.Timestamp("2018-06-02")]))
        return pd.DataFrame(
            [
                {
                    "date": date,
                    "brand": brand,
                    "item": item,
                    "sales": float(brand + item),
                    "promo": int(item == 4),
                }
                for date in dates
                for brand in range(1, 4)
                for item in range(1, 11)
            ]
        )
    return pd.DataFrame(
        [
            {"date": date, "store": store, "item": item, "sales": float(store + item)}
            for date in dates
            for store in range(1, 4)
            for item in range(1, 11)
        ]
    )


def test_d1_builder_publishes_raw_rebuilt_manifest_and_exact_entities(tmp_path: Path) -> None:
    dataset_dir = build_and_seal_dataset(
        1,
        _raw_frame(1),
        output_dir=tmp_path / "sealed",
        raw_input_path=tmp_path / "raw-d1.csv",
    )

    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    source = pd.read_parquet(dataset_dir / "source.parquet")
    target = pd.read_parquet(dataset_dir / "target.parquet")
    assert manifest["provenance_level"] == "raw_rebuilt"
    assert source[["store_id", "item_id"]].drop_duplicates().shape[0] == 27
    assert target[["store_id", "item_id"]].drop_duplicates().to_records(index=False).tolist() == [(1, 10)]
    assert not list((tmp_path / "sealed").glob(".dataset*.tmp.*"))


def test_d2_builder_preserves_promo_and_adds_only_authorized_missing_day(tmp_path: Path) -> None:
    dataset_dir = build_and_seal_dataset(
        2,
        _raw_frame(2),
        output_dir=tmp_path / "sealed",
        raw_input_path=tmp_path / "raw-d2.csv",
    )

    source = pd.read_parquet(dataset_dir / "source.parquet")
    target = pd.read_parquet(dataset_dir / "target.parquet")
    for frame in (source, target):
        assert frame["date"].nunique() == 30
        june_2 = frame[frame["date"] == pd.Timestamp("2018-06-02")]
        assert len(june_2) > 0
        assert (june_2["sales"] == 0).all()
        assert "promo" in frame.columns
    assert source[(source["brand_id"] == 2) & (source["item_id"] == 4)]["promo"].max() == 1
