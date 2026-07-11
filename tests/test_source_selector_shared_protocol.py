from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.protocols.experiment_protocol import ProtocolViolation
from src.protocols.runner_adapter import configure_protocol_frames
from src.source_selection.source_selector import SourceSelector


class SourceSelectorSharedProtocolTest(unittest.TestCase):
    def setUp(self) -> None:
        dates = pd.date_range("2020-01-01", periods=35, freq="D")
        self.target = pd.DataFrame(
            {
                "store_id": "T1",
                "item_id": "I0",
                "second_category_id": 20,
                "date": dates,
                "sales": np.r_[np.zeros(30), np.full(5, 9999.0)],
                "identifier_feature": 999,
            }
        )
        self.source = pd.concat(
            [
                pd.DataFrame(
                    {
                        "store_id": store,
                        "item_id": item,
                        "second_category_id": category,
                        "date": dates,
                        "sales": np.r_[np.full(30, value), np.full(5, future)],
                        "identifier_feature": identifier,
                    }
                )
                for store, item, category, value, future, identifier in (
                    ("T1", "I1", 20, 1.0, 1000.0, 1),
                    ("S2", "I2", 20, 2.0, 2000.0, 2),
                    ("S3", "I3", 30, 0.0, 3000.0, 3),
                )
            ],
            ignore_index=True,
        )

    def test_strict_frames_use_shared_daily_selection_and_ignore_model_ids(self) -> None:
        source, target = configure_protocol_frames(
            self.source,
            self.target,
            dataset_id="D4",
            scenario="with",
            group_cols=("store_id", "item_id"),
            grouping_col="second_category_id",
            observed_start="2020-01-01",
        )
        selection = SourceSelector().select_top_k_sources(
            target,
            source,
            feature_cols=("sales", "identifier_feature"),
            k=2,
            group_cols=("store_id", "item_id"),
            weight_mode="inverse_distance",
        )

        self.assertEqual(
            [tuple(row["source_key"]) for row in selection["sources"]],
            [("T1", "I1"), ("S2", "I2")],
        )
        self.assertEqual(selection["meta"]["protocol_version"], "d1_d6_protocol_v1")
        self.assertEqual(selection["meta"]["representation"], "daily_sales_flattened_30d")
        self.assertEqual(selection["meta"]["feature_cols"], ["sales"])
        self.assertEqual(selection["meta"]["target_observed_end"], "2020-01-30")
        self.assertTrue(selection["meta"]["target_test_excluded"])
        self.assertTrue(selection["meta"]["source_future_excluded"])

    def test_formal_selection_rejects_raw_distance_and_k_shrink(self) -> None:
        source, target = configure_protocol_frames(
            self.source,
            self.target,
            dataset_id="D4",
            scenario="with",
            group_cols=("store_id", "item_id"),
            grouping_col="second_category_id",
            observed_start="2020-01-01",
        )
        selector = SourceSelector()
        with self.assertRaisesRegex(ProtocolViolation, "inverse_distance"):
            selector.select_top_k_sources(
                target,
                source,
                feature_cols=("sales",),
                k=2,
                group_cols=("store_id", "item_id"),
                weight_mode="raw_distance",
            )
        with self.assertRaisesRegex(ProtocolViolation, "required K=3"):
            selector.select_top_k_sources(
                target,
                source,
                feature_cols=("sales",),
                k=3,
                group_cols=("store_id", "item_id"),
            )

    def test_d1_d6_named_frames_cannot_use_legacy_fallback(self) -> None:
        target = self.target.copy()
        source = self.source.copy()
        target.attrs["dataset_name"] = "Dataset4"
        source.attrs["dataset_name"] = "Dataset4"
        with self.assertRaisesRegex(ProtocolViolation, "shared protocol metadata"):
            SourceSelector().select_top_k_sources(
                target,
                source,
                feature_cols=("sales",),
                k=1,
                group_cols=("store_id", "item_id"),
            )


if __name__ == "__main__":
    unittest.main()
