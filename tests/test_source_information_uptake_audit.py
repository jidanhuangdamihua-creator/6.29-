import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd


class SourceInformationUptakeAuditTest(unittest.TestCase):
    def test_audit_script_imports_and_declares_required_outputs_and_columns(self):
        from scripts.audits import source_information_uptake_audit as audit

        self.assertEqual("source_information_uptake_details.csv", audit.DETAILS_CSV.name)
        self.assertEqual("source_information_uptake_summary.csv", audit.SUMMARY_CSV.name)
        self.assertEqual("source_information_uptake.md", audit.REPORT_MD.name)

        required = {
            "dataset_id",
            "dataset_name",
            "method_variant",
            "horizon",
            "random_seed",
            "selected_sources",
            "random_sources",
            "source_count",
            "target_window_days",
            "target_train_size",
            "target_val_size",
            "target_test_size",
            "source_train_size",
            "pretrain_train_loss_first_epoch",
            "pretrain_train_loss_final_epoch",
            "pretrain_val_rmse",
            "before_finetune_val_rmse",
            "before_finetune_test_rmse",
            "after_finetune_val_rmse",
            "after_finetune_test_rmse",
            "target_train_loss_first_epoch",
            "target_train_loss_final_epoch",
            "target_val_rmse_first_epoch",
            "target_val_rmse_final_epoch",
            "frozen_backbone",
            "shuffled_source",
            "shuffle_type",
            "random_source",
            "source_only",
            "rmse_delta_vs_notl",
            "rmse_pct_delta_vs_notl",
            "rmse_delta_vs_random_init",
            "rmse_pct_delta_vs_random_init",
            "rmse_delta_vs_knn_source",
            "rmse_pct_delta_vs_knn_source",
            "source_uptake_status",
            "forgetting_status",
            "notes",
            "run_time_seconds",
            "error_message",
        }
        self.assertTrue(required.issubset(set(audit.DETAIL_COLUMNS)))

    def test_method_variants_are_complete(self):
        from scripts.audits import source_information_uptake_audit as audit

        required_variants = {
            "No-TL",
            "Random-init-target-finetune",
            "KNN-source-pretrain-finetune",
            "Random-source-pretrain-finetune",
            "Shuffled-source-pretrain-finetune",
            "KNN-source-pretrain-frozen-backbone",
            "Source-only-prediction",
        }
        self.assertTrue(required_variants.issubset(set(audit.METHOD_VARIANTS)))

    def test_summary_and_markdown_generation_from_minimal_rows(self):
        from scripts.audits import source_information_uptake_audit as audit

        rows = []
        for variant, rmse in [
            ("No-TL", 1.0),
            ("Random-init-target-finetune", 1.0),
            ("KNN-source-pretrain-finetune", 0.98),
            ("Random-source-pretrain-finetune", 1.05),
            ("Shuffled-source-pretrain-finetune", 0.981),
            ("KNN-source-pretrain-frozen-backbone", 0.9),
            ("Source-only-prediction", 0.8),
        ]:
            row = {column: "" for column in audit.DETAIL_COLUMNS}
            row.update(
                {
                    "dataset_id": 1,
                    "dataset_name": "Dataset1",
                    "method_variant": variant,
                    "horizon": 1,
                    "random_seed": 42,
                    "after_finetune_test_rmse": rmse,
                    "before_finetune_test_rmse": rmse,
                    "rmse_delta_vs_notl": rmse - 1.0,
                    "rmse_pct_delta_vs_notl": (rmse - 1.0) / 1.0,
                    "rmse_delta_vs_random_init": rmse - 1.0,
                    "rmse_pct_delta_vs_random_init": (rmse - 1.0) / 1.0,
                    "source_uptake_status": "TEST",
                }
            )
            rows.append(row)

        summary = audit.build_summary(rows)
        self.assertFalse(summary.empty)
        for column in audit.SUMMARY_COLUMNS:
            self.assertIn(column, summary.columns)

        md = audit.build_markdown_report(rows, summary)
        self.assertIn("源信息吸收判断", md)
        self.assertIn("Dataset1", md)

    def test_quick_mode_creates_required_outputs(self):
        from scripts.audits import source_information_uptake_audit as audit

        tmp_dir = Path(tempfile.mkdtemp(prefix="source_uptake_audit_test_"))
        self.addCleanup(lambda: shutil.rmtree(tmp_dir, ignore_errors=True))

        exit_code = audit.main(
            [
                "--dataset-id",
                "1",
                "--seed",
                "42",
                "--quick",
                "--output-dir",
                str(tmp_dir),
            ]
        )
        self.assertEqual(0, exit_code)

        details_path = tmp_dir / "source_information_uptake_details.csv"
        summary_path = tmp_dir / "source_information_uptake_summary.csv"
        report_path = tmp_dir / "source_information_uptake.md"

        self.assertTrue(details_path.exists())
        self.assertTrue(summary_path.exists())
        self.assertTrue(report_path.exists())

        details = pd.read_csv(details_path)
        key_columns = {
            "dataset_id",
            "method_variant",
            "after_finetune_test_rmse",
            "rmse_delta_vs_notl",
            "rmse_delta_vs_random_init",
            "source_uptake_status",
            "notes",
        }
        self.assertTrue(key_columns.issubset(set(details.columns)))
        self.assertTrue(set(audit.METHOD_VARIANTS).issubset(set(details["method_variant"].astype(str))))

        summary = pd.read_csv(summary_path)
        self.assertFalse(summary.empty)

        report = report_path.read_text(encoding="utf-8")
        self.assertTrue(report.strip())
        self.assertIn("源信息吸收判断", report)


if __name__ == "__main__":
    unittest.main()
