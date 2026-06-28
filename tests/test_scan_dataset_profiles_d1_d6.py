import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts import scan_dataset_profiles_d1_d6 as scanner


class ScanDatasetProfilesD1D6DiscoveryTests(unittest.TestCase):
    def test_discover_datasets_prefers_extracted_folder_and_records_notebooks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "(Dataset 1.zip").write_text("zip placeholder", encoding="utf-8")
            ds1_dir = root / "Dataset 1"
            ds1_dir.mkdir()
            (ds1_dir / "train.csv").write_text("date,sales,store_id\n2024-01-01,1,A\n", encoding="utf-8")
            (root / "Dataset 2.csv").write_text("date,sales,store_id\n2024-01-01,2,B\n", encoding="utf-8")
            (root / "Dataset 3.ipynb").write_text("{}", encoding="utf-8")

            discovery = scanner.discover_dataset_paths(root)

        self.assertEqual(discovery.datasets["Dataset1"].path.resolve(), ds1_dir.resolve())
        self.assertEqual(discovery.datasets["Dataset1"].kind, "directory")
        self.assertEqual(discovery.datasets["Dataset2"].path.name, "Dataset 2.csv")
        self.assertIn("Dataset 3.ipynb", [p.name for p in discovery.ignored_files])

    def test_choose_main_table_prefers_file_with_date_sales_and_entity_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            aux = root / "items.csv"
            main = root / "train.csv"
            aux.write_text("item_id,family\n1,A\n2,B\n", encoding="utf-8")
            pd.DataFrame(
                {
                    "date": ["2024-01-01", "2024-01-02"],
                    "sales": [1, 2],
                    "store_id": ["S1", "S1"],
                    "item_id": [1, 1],
                }
            ).to_csv(main, index=False)

            files = scanner.collect_dataset_files(root)
            chosen, auxiliaries, profiles = scanner.choose_main_table(files, max_probe_rows=50)

        self.assertEqual(chosen, main)
        self.assertEqual(auxiliaries, [aux])
        self.assertGreater(profiles[str(main)]["main_table_score"], profiles[str(aux)]["main_table_score"])

    def test_chunked_long_table_scan_uses_all_csv_chunks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "train.csv"
            pd.DataFrame(
                {
                    "date": ["2024-01-01", "2024-01-02", "2024-02-01", "2024-02-02"],
                    "sales": [1, 0, 3, 4],
                    "store_id": ["A", "A", "B", "B"],
                    "item_id": [1, 1, 2, 2],
                }
            ).to_csv(path, index=False)
            inference = {
                "inferred_date_col": "date",
                "inferred_sales_col": "sales",
                "inferred_entity_cols": ["store_id", "item_id"],
            }

            profile = scanner.aggregate_long_table_full_scan(
                path,
                inference,
                chunk_size=2,
                file_format="csv",
            )

        self.assertEqual(profile.summary["scan_coverage"], "CHUNKED_FULL_SCAN")
        self.assertEqual(profile.summary["global_min_date"], "2024-01-01")
        self.assertEqual(profile.summary["global_max_date"], "2024-02-02")
        self.assertEqual(profile.summary["entity_count"], 2)
        self.assertEqual(int(profile.entity_time_span["meets_210_days"].sum()), 0)

    def test_m5_wide_scan_maps_calendar_and_keeps_all_entities(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sales_path = root / "sales_train_evaluation.csv"
            calendar_path = root / "calendar.csv"
            pd.DataFrame(
                {
                    "id": ["A_1", "B_1"],
                    "item_id": ["A", "B"],
                    "dept_id": ["D", "D"],
                    "cat_id": ["C", "C"],
                    "store_id": ["S1", "S2"],
                    "state_id": ["CA", "TX"],
                    "d_1": [0, 1],
                    "d_2": [2, 0],
                    "d_3": [3, 4],
                }
            ).to_csv(sales_path, index=False)
            pd.DataFrame(
                {
                    "d": ["d_1", "d_2", "d_3"],
                    "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
                }
            ).to_csv(calendar_path, index=False)

            profile = scanner.aggregate_m5_wide_full_scan(
                sales_path,
                calendar_path,
                chunk_size=1,
            )

        self.assertEqual(profile.summary["scan_coverage"], "CHUNKED_FULL_SCAN")
        self.assertEqual(profile.summary["global_min_date"], "2020-01-01")
        self.assertEqual(profile.summary["global_max_date"], "2020-01-03")
        self.assertEqual(profile.summary["entity_count"], 2)
        self.assertAlmostEqual(profile.summary["zero_sales_ratio"], 2 / 6)


if __name__ == "__main__":
    unittest.main()
