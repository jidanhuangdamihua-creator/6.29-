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
        source_dates = pd.date_range(dates[0] - pd.Timedelta(days=150), periods=180, freq="D")
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
                        "date": source_dates,
                        "sales": np.full(180, value),
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

    def test_d4_formal_selection_uses_cross_group_candidates_and_ignores_model_ids(
        self,
    ) -> None:
        source, target = configure_protocol_frames(
            self.source,
            self.target,
            dataset_id="D4",
            scenario="with",
            group_cols=("store_id", "item_id"),
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
            [("S3", "I3"), ("T1", "I1")],
        )
        selected_keys = [tuple(row["source_key"]) for row in selection["sources"]]
        self.assertIn(("S3", "I3"), selected_keys)
        self.assertEqual(
            self.source.loc[
                (self.source["store_id"] == "S3") & (self.source["item_id"] == "I3"),
                "second_category_id",
            ].iloc[0],
            30,
        )
        self.assertEqual(self.target["second_category_id"].iloc[0], 20)
        self.assertEqual(selection["meta"]["protocol_version"], "d1_d6_protocol_v1")
        self.assertEqual(selection["meta"]["representation"], "daily_sales_flattened_30d")
        self.assertEqual(selection["meta"]["feature_cols"], ["sales"])
        self.assertEqual(selection["meta"]["target_observed_end"], "2020-01-30")
        self.assertTrue(selection["meta"]["target_test_excluded"])
        self.assertTrue(selection["meta"]["source_future_excluded"])
        self.assertTrue(selection["meta"]["cnn_provenance_validated"])
        self.assertEqual(
            selection["meta"]["cnn_provenance_source_keys"],
            [("S3", "I3"), ("T1", "I1")],
        )

    def test_formal_selection_rejects_raw_distance(self) -> None:
        source, target = configure_protocol_frames(
            self.source,
            self.target,
            dataset_id="D4",
            scenario="with",
            group_cols=("store_id", "item_id"),
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

    def test_d4_formal_selection_supports_required_k_three_across_groups(self) -> None:
        source, target = configure_protocol_frames(
            self.source,
            self.target,
            dataset_id="D4",
            scenario="with",
            group_cols=("store_id", "item_id"),
            observed_start="2020-01-01",
        )
        selection = SourceSelector().select_top_k_sources(
            target,
            source,
            feature_cols=("sales",),
            k=3,
            group_cols=("store_id", "item_id"),
        )

        selected_keys = [tuple(row["source_key"]) for row in selection["sources"]]
        self.assertEqual(len(selected_keys), 3)
        self.assertIn(("S3", "I3"), selected_keys)

    def test_d5_formal_selection_fails_fast_when_two_same_group_candidates_remain(
        self,
    ) -> None:
        dates = pd.date_range("2020-01-01", periods=35, freq="D")
        source_dates = pd.date_range(dates[0] - pd.Timedelta(days=150), periods=180, freq="D")
        target = pd.DataFrame(
            {
                "store_id": "T1",
                "item_id": "I0",
                "family": "F1",
                "date": dates,
                "sales": np.r_[np.zeros(30), np.full(5, 9999.0)],
            }
        )
        source = pd.concat(
            [
                pd.DataFrame(
                    {
                        "store_id": store,
                        "item_id": item,
                        "family": family,
                        "date": source_dates,
                        "sales": np.full(180, value),
                    }
                )
                for store, item, family, value in (
                    ("T1", "I1", "F1", 1.0),
                    ("S2", "I2", "F1", 2.0),
                    ("S3", "I3", "F2", 0.0),
                )
            ],
            ignore_index=True,
        )
        source["onpromotion"] = 0.0
        source["oil_price"] = 40.0
        target["onpromotion"] = 0.0
        target["oil_price"] = 40.0
        source, target = configure_protocol_frames(
            source,
            target,
            dataset_id="D5",
            scenario="with",
            group_cols=("store_id", "item_id"),
            grouping_col="family",
            observed_start="2020-01-01",
        )
        with self.assertRaisesRegex(ProtocolViolation, "required K=3"):
            SourceSelector().select_top_k_sources(
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
