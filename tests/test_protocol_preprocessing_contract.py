from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.preprocess_d1_d3_offline import build_d3_protocol_frames
from scripts.preprocess_d4_d6_offline import calendarize_declared_zero_demand
from scripts.regenerate_d1_d2_parquets import (
    build_d1_protocol_frames,
    build_d2_protocol_frames,
)
from src.protocols.experiment_protocol import ProtocolViolation


class ProtocolPreprocessingContractTest(unittest.TestCase):
    def test_d1_generation_has_exact_27_sources_and_one_target(self) -> None:
        raw = pd.DataFrame(
            [
                {"date": date, "store": store, "item": item, "sales": store + item}
                for date in pd.date_range("2020-01-01", periods=30, freq="D")
                for store in range(1, 4)
                for item in range(1, 11)
            ]
        )
        source, target = build_d1_protocol_frames(raw)
        self.assertEqual(source[["store_id", "item_id"]].drop_duplicates().shape[0], 27)
        self.assertEqual(
            target[["store_id", "item_id"]].drop_duplicates().to_records(index=False).tolist(),
            [(1, 10)],
        )

    def test_d2_generation_has_exact_brand1_to3_item1_to9_sources(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "date": date,
                    "brand": brand,
                    "item": item,
                    "sales": brand + item,
                    "promo": 0,
                }
                for date in pd.date_range("2020-01-01", periods=30, freq="D")
                for brand in range(1, 4)
                for item in range(1, 11)
            ]
        )
        source, target = build_d2_protocol_frames(raw)
        self.assertEqual(source[["brand_id", "item_id"]].drop_duplicates().shape[0], 27)
        self.assertEqual(
            target[["brand_id", "item_id"]].drop_duplicates().to_records(index=False).tolist(),
            [(1, 10)],
        )

    def test_d2_wide_raw_generation_preserves_all_protocol_sources_and_promo(self) -> None:
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        raw = pd.DataFrame({"DATE": dates})
        for brand in range(1, 4):
            for item in range(1, 11):
                raw[f"QTY_B{brand}_{item}"] = brand * 100 + item
                raw[f"PROMO_B{brand}_{item}"] = int((brand, item) == (2, 3))

        source, target = build_d2_protocol_frames(raw)

        self.assertEqual(
            set(source[["brand_id", "item_id"]].drop_duplicates().itertuples(index=False, name=None)),
            {(brand, item) for brand in range(1, 4) for item in range(1, 10)},
        )
        self.assertNotIn((1, 10), set(source[["brand_id", "item_id"]].itertuples(index=False, name=None)))
        self.assertEqual(target["promo"].unique().tolist(), [0])
        promo_source = source[(source["brand_id"] == 2) & (source["item_id"] == 3)]
        self.assertEqual(promo_source["promo"].unique().tolist(), [1])

    def test_d2_wide_raw_missing_protocol_source_fails_explicitly(self) -> None:
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        raw = pd.DataFrame({"DATE": dates})
        for brand in range(1, 4):
            for item in range(1, 11):
                if (brand, item) != (3, 9):
                    raw[f"QTY_B{brand}_{item}"] = 1

        with self.assertRaisesRegex(ProtocolViolation, "QTY_B3_9"):
            build_d2_protocol_frames(raw)

    def test_d3_generation_has_store1_to30_excluding_store10(self) -> None:
        raw = pd.DataFrame(
            [
                {"Date": date, "Store": store, "Sales": store}
                for date in pd.date_range("2020-01-01", periods=30, freq="D")
                for store in range(1, 31)
            ]
        )
        source, target = build_d3_protocol_frames(raw)
        self.assertEqual(source["store_id"].nunique(), 29)
        self.assertNotIn(10, set(source["store_id"]))
        self.assertEqual(set(target["store_id"]), {10})

    def test_calendarization_requires_explicit_zero_demand_semantics(self) -> None:
        frame = pd.DataFrame(
            {
                "store_id": [1, 1],
                "item_id": [2, 2],
                "date": ["2020-01-01", "2020-01-03"],
                "sales": [3.0, 5.0],
            }
        )
        with self.assertRaisesRegex(ProtocolViolation, "zero-demand semantics"):
            calendarize_declared_zero_demand(
                frame,
                group_cols=("store_id", "item_id"),
                zero_demand_semantics=False,
            )
        completed = calendarize_declared_zero_demand(
            frame,
            group_cols=("store_id", "item_id"),
            zero_demand_semantics=True,
        )
        self.assertEqual(completed["sales"].tolist(), [3.0, 0.0, 5.0])
        self.assertTrue(completed.attrs["zero_demand_calendarization_declared"])


if __name__ == "__main__":
    unittest.main()
