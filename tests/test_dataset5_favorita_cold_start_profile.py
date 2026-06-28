import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts import scan_dataset5_favorita_cold_start as favorita_scan


class Dataset5FavoritaColdStartProfileTests(unittest.TestCase):
    def test_identify_file_roles_marks_missing_m5_prices_and_calendar(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name in ["train.csv", "items.csv", "stores.csv", "holidays_events.csv", "oil.csv"]:
                (root / name).write_text("x\n", encoding="utf-8")

            roles = favorita_scan.identify_file_roles(root)

        self.assertEqual(roles["sales_file"], "train.csv")
        self.assertEqual(roles["calendar_files"], ["holidays_events.csv", "oil.csv"])
        self.assertEqual(roles["prices_file"], "")
        self.assertEqual(roles["dataset_family"], "Favorita")
        self.assertFalse(roles["m5_compatible_files"])

    def test_add_cold_start_flags_counts_observed_horizon_windows(self):
        df = pd.DataFrame(
            {
                "entity_id": ["a", "b", "c"],
                "span_days": [20, 50, 70],
                "valid_sales_days": [14, 44, 58],
                "date_density": [1.0, 0.9, 0.95],
            }
        )

        flagged, summary = favorita_scan.add_cold_start_flags(df, [7, 30], [7, 28])

        self.assertTrue(flagged.loc[0, "eligible_obs7_h7"])
        self.assertFalse(flagged.loc[0, "eligible_obs30_h28"])
        self.assertEqual(int(summary.loc[summary["task_id"].eq("obs7_h7"), "eligible_entity_count"].iloc[0]), 3)
        self.assertEqual(int(summary.loc[summary["task_id"].eq("obs30_h28"), "eligible_entity_count"].iloc[0]), 1)

    def test_source_pool_sizes_use_family_class_store_and_global(self):
        stats = pd.DataFrame(
            {
                "entity_id": ["store_id=1|item_id=10", "store_id=1|item_id=11", "store_id=2|item_id=12"],
                "store_id": [1, 1, 2],
                "item_id": [10, 11, 12],
                "family": ["A", "A", "B"],
                "class": [100, 200, 100],
                "eligible_obs7_h7": [True, True, True],
            }
        )

        pools = favorita_scan.build_source_target_candidates(stats, ["obs7_h7"], top_targets=1, top_sources=10)

        first = pools.iloc[0]
        self.assertEqual(first["same_category_source_count"], 1)
        self.assertEqual(first["same_department_source_count"], 1)
        self.assertEqual(first["same_store_source_count"], 1)
        self.assertEqual(first["global_source_count"], 2)

    def test_df_to_markdown_handles_list_values(self):
        text = favorita_scan.df_to_markdown(pd.DataFrame([{"calendar_files": ["a.csv", "b.csv"]}]))

        self.assertIn("a.csv", text)
        self.assertIn("b.csv", text)

    def test_build_summary_uses_favorita_label_and_protocol_fields(self):
        roles = {
            "dataset_family": "Favorita",
            "m5_compatible_files": False,
            "sales_file": "train.csv",
            "calendar_files": ["holidays_events.csv", "oil.csv"],
            "prices_file": "",
        }
        meta = {
            "total_rows": 3,
            "item_count": 2,
            "store_count": 1,
            "min_date": "2024-01-01",
            "max_date": "2024-01-03",
            "unique_dates": 3,
        }
        entity_stats = pd.DataFrame(
            {
                "span_days": [3],
                "valid_sales_days": [3],
                "sales_missing_count": [0],
                "row_count": [3],
                "zero_sales_count": [0],
                "date_density": [1.0],
            }
        )
        task_summary = pd.DataFrame(
            [{"task_id": "obs7_h7", "eligible_entity_count": 0, "eligible_entity_ratio": 0.0}]
        )

        summary = favorita_scan.build_summary(roles, meta, entity_stats, task_summary)

        self.assertEqual(summary.loc[0, "dataset_label_requested"], "Dataset5_Favorita")
        self.assertEqual(summary.loc[0, "detected_dataset_family"], "Favorita")
        self.assertEqual(summary.loc[0, "dataset_id"], "Dataset5")
        self.assertEqual(summary.loc[0, "dataset_name_final"], "Dataset5_Favorita")
        self.assertEqual(summary.loc[0, "standard_m5_structure"], False)
        self.assertEqual(summary.loc[0, "cold_start_construction"], "Yes")
        self.assertEqual(summary.loc[0, "cold_start_protocol_type"], "Favorita short-history cold-start")

    def test_write_reports_uses_favorita_filenames_and_no_m5style_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            roles = {
                "dataset_family": "Favorita",
                "m5_compatible_files": False,
                "sales_file": "train.csv",
                "calendar_files": ["holidays_events.csv"],
                "prices_file": "",
                "items_file": "items.csv",
                "stores_file": "stores.csv",
            }
            summary = pd.DataFrame([{"dataset_label_requested": "Dataset5_Favorita"}])
            entity_stats = pd.DataFrame([{"entity_id": "store_id=1|item_id=10"}])
            task_summary = pd.DataFrame([{"task_id": "obs7_h7", "eligible_entity_count": 1}])
            candidates = pd.DataFrame([{"task_id": "obs7_h7", "candidate": True}])

            favorita_scan.write_reports(out, roles, summary, entity_stats, task_summary, candidates)

            self.assertTrue((out / "dataset5_favorita_profile_summary.md").exists())
            self.assertTrue((out / "dataset5_favorita_entity_stats.csv").exists())
            self.assertTrue((out / "dataset5_favorita_cold_start_task_design.md").exists())
            self.assertTrue((out / "dataset5_favorita_source_target_candidates.csv").exists())
            old_name = "dataset5_" + "m5_profile_summary.md"
            self.assertFalse((out / old_name).exists())
            self.assertFalse((out / "dataset5_favorita_m5style_profile_summary.md").exists())

            rendered = "\n".join(path.read_text(encoding="utf-8") for path in out.glob("*.md"))
            forbidden_terms = [
                "Dataset5_" + "M5",
                "Dataset5_Favorita_" + "M5" + "style",
                "M5" + "style",
                "M5" + "-style",
                "M5" + "STYLE",
            ]
            for forbidden in forbidden_terms:
                self.assertNotIn(forbidden, rendered)
            self.assertIn("Favorita cold-start construction", rendered)
            self.assertIn("Dataset5 is identified as Favorita", rendered)


if __name__ == "__main__":
    unittest.main()
