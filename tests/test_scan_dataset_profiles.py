import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import scan_dataset_profiles as scanner


class ScanDatasetProfilesTests(unittest.TestCase):
    def test_load_config_normalizes_dataset_entries_to_name_and_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "dataset_profile_scan_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "output_root": "outputs/dataset_profiles",
                        "datasets": [
                            {
                                "name": "Dataset1",
                                "path": "/tmp/dataset1.csv",
                                "date_col": "date",
                                "sales_col": "sales",
                                "entity_cols": ["store_id", "item_id"],
                                "source_entities": ["store_id=1|item_id=1"],
                                "target_entities": ["store_id=1|item_id=10"],
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            cfg = scanner.load_config(config_path)

        self.assertEqual(cfg["datasets"], [{"name": "Dataset1", "path": "/tmp/dataset1.csv"}])

    def test_infer_columns_keeps_scanning_with_low_confidence_fallback_entity(self):
        df = pd.DataFrame(
            {
                "note": ["a", "b", "c"],
                "flag": ["x", "x", "x"],
            }
        )

        result = scanner.infer_columns(df, "AmbiguousDataset")

        self.assertEqual(result["inferred_status"], "LOW_CONFIDENCE")
        self.assertIs(result["needs_manual_review"], True)
        self.assertIsNone(result["inferred_date_col"])
        self.assertIsNone(result["inferred_sales_col"])

    def test_search_strict_protocol_discovers_project_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scripts_dir = root / "scripts"
            scripts_dir.mkdir()
            (scripts_dir / "experiment_config.py").write_text(
                'STRICT_DATASET_PROTOCOL = {"DatasetX": {"target_entity_id": "S1", "target_item_id": 7}}\n',
                encoding="utf-8",
            )

            old_root = scanner.PROJECT_ROOT
            scanner.PROJECT_ROOT = root
            try:
                protocol = scanner._search_strict_protocol_in_project("DatasetX")
            finally:
                scanner.PROJECT_ROOT = old_root

        self.assertEqual(protocol, {"target_entity_id": "S1", "target_item_id": 7})

    def test_analyze_features_returns_empty_report_when_no_extra_features(self):
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "sales": [1, 2],
                "store": [1, 1],
                "_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "_sales": [1, 2],
                "_entity_id": ["store=1", "store=1"],
            }
        )
        inference = {
            "inferred_date_col": "date",
            "inferred_sales_col": "sales",
            "inferred_entity_cols": ["store"],
        }

        feature_df = scanner.analyze_features(df, inference)

        self.assertEqual(
            list(feature_df.columns),
            ["feature", "dtype", "missing_count", "missing_ratio", "unique_count"],
        )
        self.assertTrue(feature_df.empty)

    def test_resolve_data_file_prefers_nested_full_parquet_over_sample_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "data").mkdir()
            top_csv = root / "train_sample_100.csv"
            nested_parquet = root / "data" / "train.parquet"
            top_csv.write_text("date,sales\n2024-01-01,1\n", encoding="utf-8")
            nested_parquet.write_text("not a real parquet", encoding="utf-8")

            resolved = scanner._resolve_data_file(root, scanner.ScanLogger())

        self.assertEqual(resolved, nested_parquet)

    def test_resolve_data_file_rejects_sample_csv_without_sample_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sample_csv = root / "train_sample_100.csv"
            sample_csv.write_text("date,sales\n2024-01-01,1\n", encoding="utf-8")

            with self.assertRaisesRegex(FileNotFoundError, "sample"):
                scanner._resolve_data_file(root, scanner.ScanLogger())

    def test_apply_inferred_schema_builds_entity_id_without_changing_format(self):
        df = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02"],
                "sales": [3, 4],
                "store_id": [1, None],
                "item_id": [10, 11],
            }
        )
        inference = {
            "inferred_date_col": "date",
            "inferred_sales_col": "sales",
            "inferred_entity_cols": ["store_id", "item_id"],
        }

        work_df = scanner.apply_inferred_schema(df, inference, scanner.ScanLogger())

        self.assertEqual(
            work_df["_entity_id"].tolist(),
            ["store_id=1.0|item_id=10", "store_id=NA|item_id=11"],
        )

    def test_infer_date_col_prefers_dt_over_binary_holiday_flag(self):
        df = pd.DataFrame(
            {
                "dt": ["2025-04-30", "2025-05-01", "2025-05-02", "2025-05-03"],
                "holiday_flag": [0, 1, 1, 0],
                "sale_amount": [0.9, 2.0, 0.9, 1.0],
            }
        )

        result = scanner.infer_date_col(df)

        self.assertEqual(result["inferred_date_col"], "dt")

    def test_infer_columns_ignores_array_like_feature_columns(self):
        df = pd.DataFrame(
            {
                "dt": ["2025-04-30", "2025-05-01", "2025-05-02"],
                "sale_amount": [1.0, 2.0, 3.0],
                "store_id": [0, 0, 0],
                "product_id": [54, 54, 54],
                "hours_sale": [np.array([0.0, 1.0]), np.array([0.5, 0.0]), np.array([1.0, 0.0])],
            }
        )

        result = scanner.infer_columns(df, "Dataset4")

        self.assertEqual(result["inferred_date_col"], "dt")
        self.assertEqual(result["inferred_sales_col"], "sale_amount")


if __name__ == "__main__":
    unittest.main()
