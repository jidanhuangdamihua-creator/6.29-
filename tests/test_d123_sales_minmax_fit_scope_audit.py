import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd


class D123SalesMinMaxFitScopeAuditTest(unittest.TestCase):
    def test_audit_script_runs_and_writes_required_outputs(self):
        from scripts.audits import d123_sales_minmax_fit_scope_audit as audit

        tmp_dir = Path(tempfile.mkdtemp(prefix="d123_sales_minmax_audit_test_"))
        self.addCleanup(lambda: shutil.rmtree(tmp_dir, ignore_errors=True))

        exit_code = audit.main(["--output-dir", str(tmp_dir)])

        self.assertEqual(0, exit_code)

        detail_csv = tmp_dir / "d123_sales_minmax_fit_scope_audit.csv"
        summary_csv = tmp_dir / "d123_sales_minmax_fit_scope_summary.csv"
        report_md = tmp_dir / "d123_sales_minmax_fit_scope_audit.md"

        self.assertTrue(detail_csv.exists())
        self.assertTrue(summary_csv.exists())
        self.assertTrue(report_md.exists())

        details = pd.read_csv(detail_csv)
        summary = pd.read_csv(summary_csv)

        self.assertEqual({"Dataset1", "Dataset2", "Dataset3"}, set(details["dataset_name"]))
        self.assertEqual({"Dataset1", "Dataset2", "Dataset3"}, set(summary["dataset_name"]))

        required_detail_columns = {
            "dataset_id",
            "dataset_name",
            "fit_scope",
            "per_store_sales_min",
            "per_store_sales_max",
            "global_sales_min",
            "global_sales_max",
            "data_scope",
        }
        self.assertTrue(required_detail_columns.issubset(set(details.columns)))

        required_summary_columns = {
            "dataset_id",
            "dataset_name",
            "target_store",
            "per_store_sales_min",
            "per_store_sales_max",
            "global_sales_min",
            "global_sales_max",
            "min_difference",
            "max_difference",
            "range_ratio",
            "is_same_scope",
        }
        self.assertTrue(required_summary_columns.issubset(set(summary.columns)))

        for dataset_name in ["Dataset1", "Dataset2", "Dataset3"]:
            dataset_details = details[details["dataset_name"] == dataset_name]
            self.assertIn("per_store", set(dataset_details["fit_scope"]))
            self.assertIn("global", set(dataset_details["fit_scope"]))

            per_store = dataset_details[dataset_details["fit_scope"] == "per_store"].iloc[0]
            global_fit = dataset_details[dataset_details["fit_scope"] == "global"].iloc[0]

            self.assertGreaterEqual(per_store["per_store_sales_max"], per_store["per_store_sales_min"])
            self.assertGreaterEqual(global_fit["global_sales_max"], global_fit["global_sales_min"])

            row = summary[summary["dataset_name"] == dataset_name].iloc[0]
            self.assertGreater(row["global_sales_max"] - row["global_sales_min"], 0)
            self.assertFalse(pd.isna(row["range_ratio"]))

        report = report_md.read_text(encoding="utf-8")
        self.assertIn("只读", report)
        self.assertIn("Dataset1", report)
        self.assertIn("Dataset2", report)
        self.assertIn("Dataset3", report)


if __name__ == "__main__":
    unittest.main()
