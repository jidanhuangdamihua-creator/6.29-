import unittest

import pandas as pd

from scripts.run_full_paper_experiments import (
    build_dataset3_result_audit_fields,
    enrich_dataset3_source_audit_rows,
)


class Dataset3SourceAuditEnrichmentTest(unittest.TestCase):
    def test_adds_store_type_fields_from_runtime_dataframes(self):
        source_df = pd.DataFrame(
            {
                "source_key": ["unused"] * 3,
                "store_id": [2, 5, 6],
                "item_id": [2, 5, 6],
                "store_type": ["a", "a", "a"],
            }
        )
        target_df = pd.DataFrame(
            {
                "store_id": [10],
                "item_id": [10],
                "store_type": ["a"],
            }
        )
        rows = [
            {"dataset": "Dataset3", "source_rank": 1, "source_key": "('Region 1', 2)"},
            {"dataset": "Dataset3", "source_rank": 2, "source_key": "('Region 1', 6)"},
            {"dataset": "Dataset3", "source_rank": 3, "source_key": "('Region 1', 5)"},
        ]

        notes = []
        enriched = enrich_dataset3_source_audit_rows(
            rows=rows,
            source_df=source_df,
            target_df=target_df,
            notes=notes,
        )

        self.assertEqual([2, 6, 5], [row["source_store_id"] for row in enriched])
        self.assertEqual(["a", "a", "a"], [row["source_store_type"] for row in enriched])
        self.assertEqual([10, 10, 10], [row["target_store_id"] for row in enriched])
        self.assertEqual(["a", "a", "a"], [row["target_store_type"] for row in enriched])
        self.assertEqual([True, True, True], [row["same_category_pass"] for row in enriched])
        self.assertEqual([], notes)

    def test_parses_numpy_int64_source_key_string_from_validation_meta(self):
        source_df = pd.DataFrame(
            {
                "store_id": [2],
                "item_id": [2],
                "store_type": ["a"],
            }
        )
        target_df = pd.DataFrame(
            {
                "store_id": [10],
                "item_id": [10],
                "store_type": ["a"],
            }
        )
        rows = [
            {"dataset": "Dataset3", "source_rank": 1, "source_key": "('Region 1', np.int64(2))"},
        ]

        notes = []
        enriched = enrich_dataset3_source_audit_rows(
            rows=rows,
            source_df=source_df,
            target_df=target_df,
            notes=notes,
        )

        self.assertEqual(2, enriched[0]["source_store_id"])
        self.assertEqual("('Region 1', 2)", enriched[0]["source_key"])
        self.assertEqual("a", enriched[0]["source_store_type"])
        self.assertTrue(enriched[0]["same_category_pass"])
        self.assertEqual([], notes)

    def test_adds_dataset3_result_and_observed_window_audit_fields(self):
        source_df = pd.DataFrame(
            {
                "store_id": [2, 6],
                "item_id": [2, 6],
                "region_id": ["Region 1", "Region 1"],
                "entity_id": ["Region 1", "Region 1"],
                "store_type": ["a", "a"],
            }
        )
        target_df = pd.DataFrame(
            {
                "date": pd.date_range("2015-01-03", periods=210, freq="D"),
                "store_id": [10] * 210,
                "item_id": [10] * 210,
                "region_id": ["Region 1"] * 210,
                "entity_id": ["Region 1"] * 210,
                "store_type": ["a"] * 210,
            }
        )
        target_df.attrs["split_mode"] = "days"
        target_df.attrs["split_config"] = {"train_days": 15, "val_days": 15, "test_days": 180}
        rows = [
            {
                "dataset": "Dataset3",
                "source_rank": 1,
                "source_key": "('Region 1', 2)",
                "knn_features": "sales|customers",
                "target_test_data_excluded": True,
            },
            {
                "dataset": "Dataset3",
                "source_rank": 2,
                "source_key": "('Region 1', 6)",
                "knn_features": "sales|customers",
                "target_test_data_excluded": True,
            },
        ]

        enriched = enrich_dataset3_source_audit_rows(
            rows=rows,
            source_df=source_df,
            target_df=target_df,
            notes=[],
        )
        result_fields = build_dataset3_result_audit_fields(
            source_identification=enriched,
            source_df=source_df,
            target_df=target_df,
            feature_cols=["sales", "customers"],
            window_size=10,
            horizon=1,
        )

        self.assertEqual(["Region 1", "Region 1"], [row["source_region"] for row in enriched])
        self.assertEqual(["Region 1", "Region 1"], [row["target_region"] for row in enriched])
        self.assertEqual("2015-01-03", enriched[0]["observed_window_start_date"])
        self.assertEqual("2015-02-01", enriched[0]["observed_window_end_date"])
        self.assertEqual(30, enriched[0]["observed_window_days"])
        self.assertTrue(enriched[0]["target_test_data_excluded"])
        self.assertEqual("sales|customers", enriched[0]["knn_features"])
        self.assertEqual(10, result_fields["target_store_id"])
        self.assertEqual("a", result_fields["target_store_type"])
        self.assertEqual("Region 1", result_fields["target_region"])
        self.assertEqual("2|6", result_fields["selected_source_store_ids"])
        self.assertEqual("a|a", result_fields["selected_source_store_types"])
        self.assertEqual("Region 1|Region 1", result_fields["selected_source_regions"])
        self.assertEqual(15, result_fields["target_train_rows"])
        self.assertEqual(15, result_fields["target_val_rows"])
        self.assertEqual(180, result_fields["target_test_rows"])
        self.assertEqual(170, result_fields["sequence_test_samples"])


if __name__ == "__main__":
    unittest.main()
