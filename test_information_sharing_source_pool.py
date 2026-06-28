import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from data_preprocessing import (
    _standardize_rossmann_dataset,
    build_source_target_split,
    infer_source_selection_feature_columns,
    load_dataset,
    temporal_split_by_ratio_or_dates,
)
from scripts.run_full_paper_experiments import _apply_information_sharing_filter
from scripts.run_full_paper_experiments import _dataset1_channel_base_view


class TestInformationSharingSourcePool(unittest.TestCase):
    def _strict_dataset2_frame(self) -> pd.DataFrame:
        rows = []
        for day in range(220):
            for entity_id in ["B1", "B2", "B3", "B4"]:
                for item_id in range(1, 13):
                    rows.append(
                        {
                            "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day),
                            "entity_id": entity_id,
                            "item_id": item_id,
                            "sales": float((ord(entity_id[-1]) - ord("0")) * 100 + item_id + day),
                            "promo": 0,
                        }
                    )
        df = pd.DataFrame(rows)
        df.attrs["dataset_name"] = "Dataset2"
        return df

    def _strict_dataset1_frame(self) -> pd.DataFrame:
        rows = []
        for day in range(220):
            for entity_id in [1, 2, 3]:
                for item_id in [1, 2, 10]:
                    rows.append(
                        {
                            "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day),
                            "entity_id": entity_id,
                            "item_id": item_id,
                            "sales": float(entity_id * 100 + item_id + day),
                        }
                    )
        return pd.DataFrame(rows)

    def test_dataset1_channel_view_keeps_target_channel_separate_for_knn(self):
        source_df, target_df = build_source_target_split(
            self._strict_dataset1_frame(),
            {
                "dataset_name": "Dataset1",
                "paper_reproduction": {
                    "strict_paper_mode": True,
                    "strict_paper_split": False,
                    "strict_dataset_protocol": {
                        "Dataset1": {
                            "target_entity_ids": [1, 2, 3],
                            "target_item_id": 10,
                            "allowed_entities": [1, 2, 3],
                            "source_item_ids": [1, 2],
                            "target_split_days": {"train_days": 15, "val_days": 15, "test_days": 180},
                        }
                    },
                },
            },
        )
        base = {"source_df": source_df, "target_df": target_df}

        no_sharing = _dataset1_channel_base_view(base, channel_id=2, information_sharing_scenario="without_information_sharing")
        self.assertEqual({2}, set(no_sharing["target_df"]["entity_id"].unique()))
        self.assertEqual({10}, set(no_sharing["target_df"]["item_id"].unique()))
        self.assertEqual({2}, set(no_sharing["source_df"]["entity_id"].unique()))
        self.assertEqual({1, 2}, set(no_sharing["source_df"]["item_id"].unique()))
        self.assertEqual(2, no_sharing["target_df"].attrs["channel_id"])
        self.assertEqual(2, no_sharing["target_df"].attrs["target_entity_id"])
        self.assertEqual(10, no_sharing["target_df"].attrs["target_item_id"])

        with_sharing = _dataset1_channel_base_view(base, channel_id=2, information_sharing_scenario="with_information_sharing")
        self.assertEqual({2}, set(with_sharing["target_df"]["entity_id"].unique()))
        self.assertEqual({1, 2, 3}, set(with_sharing["source_df"]["entity_id"].unique()))
        self.assertEqual({1, 2}, set(with_sharing["source_df"]["item_id"].unique()))

    def test_dataset2_strict_split_uses_available_steps_and_records_missing_days(self):
        rows = []
        missing_offsets = {30, 50, 75, 120, 180}
        for day in range(208):
            if day in missing_offsets:
                continue
            for entity_id in ["B1", "B2", "B3"]:
                for item_id in range(1, 11):
                    rows.append(
                        {
                            "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day),
                            "entity_id": entity_id,
                            "item_id": item_id,
                            "sales": float(day + item_id),
                            "promo": 0,
                        }
                    )
        df = pd.DataFrame(rows)
        df.attrs["dataset_name"] = "Dataset2"

        _, target_df = build_source_target_split(
            df,
            {
                "dataset_name": "Dataset2",
                "paper_reproduction": {
                    "strict_paper_mode": True,
                    "strict_paper_split": True,
                    "strict_dataset_protocol": {
                        "Dataset2": {
                            "target_entity_id": "B1",
                            "target_item_id": 10,
                            "source_item_policy": "paper_first_9_items",
                            "source_entity_ids": ["B1", "B2", "B3"],
                            "target_split_days": {"train_days": 14, "val_days": 15, "test_days": 179},
                        }
                    },
                    "paper_split_protocol": {
                        "target_observed_window_days": 29,
                        "target_forecast_window_days": 179,
                    },
                },
            },
        )

        train_df, val_df, test_df = temporal_split_by_ratio_or_dates(target_df)
        self.assertEqual(14, train_df["date"].nunique())
        self.assertEqual(15, val_df["date"].nunique())
        self.assertEqual(174, test_df["date"].nunique())
        self.assertEqual(179, target_df.attrs["expected_test_steps"])
        self.assertEqual(174, target_df.attrs["actual_test_steps"])
        self.assertEqual(5, target_df.attrs["missing_calendar_days"])
        self.assertEqual("PARTIAL / DATASET_MISSING_DAYS", target_df.attrs["split_alignment_status"])

    def test_build_split_reads_dotted_dict_config_and_uses_dataset_protocol(self):
        cfg = {
            "dataset_name": "Dataset1",
            "paper_reproduction": {
                "strict_paper_mode": True,
                "strict_paper_split": False,
                "strict_dataset_protocol": {
                    "Dataset1": {
                        "target_entity_id": 1,
                        "target_item_id": 10,
                        "allowed_entities": [1, 2],
                        "source_item_ids": [1, 2],
                    }
                },
            },
        }

        source_df, target_df = build_source_target_split(self._strict_dataset1_frame(), cfg)

        self.assertEqual({1, 2}, set(source_df["entity_id"].unique()))
        self.assertEqual({1, 2}, set(source_df["item_id"].unique()))
        self.assertEqual({1}, set(target_df["entity_id"].unique()))
        self.assertEqual({10}, set(target_df["item_id"].unique()))
        self.assertTrue(source_df.attrs["strict_paper_mode"])

    def test_dataset2_strict_source_pool_uses_paper_first_9_items(self):
        cfg = {
            "dataset_name": "Dataset2",
            "paper_reproduction": {
                "strict_paper_mode": True,
                "strict_dataset_protocol": {
                    "Dataset2": {
                        "target_entity_id": "B1",
                        "target_item_id": 10,
                        "source_item_policy": "paper_first_9_items",
                        "source_entity_ids": ["B1", "B2", "B3"],
                    }
                },
            },
        }
        protocol = cfg["paper_reproduction"]

        source_df, target_df = build_source_target_split(self._strict_dataset2_frame(), cfg)

        self.assertEqual({"B1"}, set(target_df["entity_id"].unique()))
        self.assertEqual({10}, set(target_df["item_id"].unique()))
        self.assertEqual({"B1", "B2", "B3"}, set(source_df["entity_id"].unique()))
        self.assertEqual(set(range(1, 10)), set(source_df["item_id"].unique()))

        no_sharing = _apply_information_sharing_filter(
            dataset_name="Dataset2",
            source_df=source_df,
            target_df=target_df,
            use_information_sharing=False,
            strict_paper_mode=True,
            protocol=protocol,
            cfg={},
        )
        self.assertEqual({"B1"}, set(no_sharing["entity_id"].unique()))
        self.assertEqual(set(range(1, 10)), set(no_sharing["item_id"].unique()))

        with_sharing = _apply_information_sharing_filter(
            dataset_name="Dataset2",
            source_df=source_df,
            target_df=target_df,
            use_information_sharing=True,
            strict_paper_mode=True,
            protocol=protocol,
            cfg={},
        )
        self.assertEqual({"B1", "B2", "B3"}, set(with_sharing["entity_id"].unique()))
        self.assertEqual(set(range(1, 10)), set(with_sharing["item_id"].unique()))

    def test_source_selection_excludes_encoded_id_columns_but_keeps_sales_and_promo(self):
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=3),
                "entity_id": ["B1", "B2", "B3"],
                "item_id": [1, 2, 3],
                "store_id": [1, 2, 3],
                "sales": [10.0, 11.0, 12.0],
                "promo": [0, 1, 0],
                "brand_code": [0, 1, 2],
                "entity_id_code": [0, 1, 2],
                "year": [2024, 2024, 2024],
            }
        )

        info = infer_source_selection_feature_columns(
            source_df=frame,
            target_df=frame,
            candidate_cols=[
                "sales",
                "promo",
                "brand_code",
                "entity_id_code",
                "item_id",
                "entity_id",
                "store_id",
                "date",
            ],
            include_sales_in_knn=True,
        )

        self.assertIn("sales", info["selected_features"])
        self.assertIn("promo", info["selected_features"])
        self.assertNotIn("brand_code", info["selected_features"])
        self.assertNotIn("entity_id_code", info["selected_features"])
        self.assertNotIn("item_id", info["selected_features"])
        self.assertNotIn("entity_id", info["selected_features"])
        self.assertNotIn("store_id", info["selected_features"])
        self.assertNotIn("date", info["selected_features"])

    def test_dataset3_standardize_derives_paper_regions_from_store_id(self):
        raw_df = pd.DataFrame(
            {
                "Store": [1, 10, 11, 20, 21, 30],
                "Date": pd.date_range("2024-01-01", periods=6),
                "Sales": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "Customers": [10, 20, 30, 40, 50, 60],
                "Open": [1, 1, 1, 1, 1, 1],
                "Promo": [0, 0, 1, 1, 0, 0],
                "StateHoliday": [0, 0, 0, 0, 0, 0],
                "SchoolHoliday": [0, 0, 0, 0, 0, 0],
            }
        )

        standardized = _standardize_rossmann_dataset(raw_df)

        region_map = dict(zip(standardized["store_id"].tolist(), standardized["region_id"].tolist()))
        self.assertEqual("Region 1", region_map[1])
        self.assertEqual("Region 1", region_map[10])
        self.assertEqual("Region 2", region_map[11])
        self.assertEqual("Region 2", region_map[20])
        self.assertEqual("Region 3", region_map[21])
        self.assertEqual("Region 3", region_map[30])
        self.assertNotIn("TODO_REGION_UNAVAILABLE", standardized["region_id"].astype(str).tolist())

    def test_dataset3_information_sharing_filter_respects_paper_region_pool(self):
        source_df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=24),
                "entity_id": ["Region 1"] * 8 + ["Region 2"] * 8 + ["Region 3"] * 8,
                "item_id": [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 30, 29],
                "store_id": [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 30, 29],
                "region_id": ["Region 1"] * 8 + ["Region 2"] * 8 + ["Region 3"] * 8,
                "sales": [float(i) for i in range(24)],
            }
        )
        target_df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=2),
                "entity_id": ["Region 1", "Region 1"],
                "item_id": [10, 10],
                "store_id": [10, 10],
                "region_id": ["Region 1", "Region 1"],
                "sales": [10.0, 11.0],
            }
        )
        protocol = {
            "strict_dataset_protocol": {
                "Dataset3": {
                    "region_field": "region_id",
                }
            }
        }

        filtered = _apply_information_sharing_filter(
            dataset_name="Dataset3",
            source_df=source_df,
            target_df=target_df,
            use_information_sharing=False,
            strict_paper_mode=True,
            protocol=protocol,
            cfg={},
        )

        filtered_store_ids = set(pd.to_numeric(filtered["store_id"], errors="coerce").dropna().astype(int).unique())
        self.assertEqual({1, 2, 3, 4, 5, 6, 7, 8}, filtered_store_ids)
        self.assertEqual({"Region 1"}, set(filtered["region_id"].astype(str).unique()))
        self.assertEqual("without_information_sharing_same_region", filtered.attrs["source_pool_scope_mode"])
        self.assertNotIn(10, filtered_store_ids)

        with_sharing = _apply_information_sharing_filter(
            dataset_name="Dataset3",
            source_df=source_df,
            target_df=target_df,
            use_information_sharing=True,
            strict_paper_mode=True,
            protocol=protocol,
            cfg={},
        )

        with_store_ids = set(pd.to_numeric(with_sharing["store_id"], errors="coerce").dropna().astype(int).unique())
        self.assertTrue({11, 21}.issubset(with_store_ids))
        self.assertNotEqual(filtered_store_ids, with_store_ids)
        self.assertNotIn(10, with_store_ids)
        self.assertEqual("with_information_sharing_full_pool", with_sharing.attrs["source_pool_scope_mode"])

    def test_dataset3_store_type_merge_and_same_category_filter(self):
        with TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "Dataset3-Rossmann-mini.csv"
            pd.DataFrame(
                {
                    "Store": [2, 10],
                    "DayOfWeek": [1, 1],
                    "Date": pd.date_range("2024-01-01", periods=2),
                    "Sales": [20.0, 10.0],
                    "Customers": [200, 100],
                    "Open": [1, 1],
                    "Promo": [0, 0],
                    "StateHoliday": [0, 0],
                    "SchoolHoliday": [0, 0],
                }
            ).to_csv(data_path, index=False)

            standardized = load_dataset("Dataset3", str(data_path))

        self.assertIn("store_type", standardized.columns)
        store_10 = standardized.loc[standardized["store_id"] == 10, "store_type"]
        self.assertEqual("a", store_10.iloc[0])

        source_df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=4),
                "entity_id": ["Region 1"] * 4,
                "item_id": [1, 2, 3, 4],
                "store_id": [1, 2, 4, 8],
                "region_id": ["Region 1"] * 4,
                "store_type": ["c", "a", "c", "a"],
                "sales": [1.0, 2.0, 3.0, 4.0],
            }
        )
        target_df = pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-01-01")],
                "entity_id": ["Region 1"],
                "item_id": [10],
                "store_id": [10],
                "region_id": ["Region 1"],
                "store_type": ["a"],
                "sales": [10.0],
            }
        )
        protocol = {"strict_dataset_protocol": {"Dataset3": {"region_field": "region_id"}}}

        filtered = _apply_information_sharing_filter(
            dataset_name="Dataset3",
            source_df=source_df,
            target_df=target_df,
            use_information_sharing=False,
            strict_paper_mode=True,
            protocol=protocol,
            cfg={"same_category_mode": True},
        )

        self.assertEqual({"a"}, set(filtered["store_type"].astype(str).unique()))


if __name__ == "__main__":
    unittest.main()
