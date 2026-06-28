import unittest


class NoTlTrainingLossCurveAuditTest(unittest.TestCase):
    def test_audit_script_imports_and_declares_required_outputs(self):
        from scripts.audits import notl_training_loss_curve_audit as audit

        self.assertEqual("notl_training_loss_curve_audit.csv", audit.LOSS_CURVE_CSV.name)
        self.assertEqual("notl_training_loss_curve_summary.csv", audit.SUMMARY_CSV.name)
        self.assertEqual("notl_training_loss_curve_audit.md", audit.REPORT_MD.name)

    def test_loss_curve_csv_required_columns(self):
        from scripts.audits import notl_training_loss_curve_audit as audit

        for column in ("dataset_id", "epoch", "train_loss"):
            self.assertIn(column, audit.LOSS_CURVE_COLUMNS)

    def test_summary_csv_required_columns(self):
        from scripts.audits import notl_training_loss_curve_audit as audit

        for column in ("dataset_id", "target_epochs", "early_stopping_enabled", "last_10_trend"):
            self.assertIn(column, audit.SUMMARY_COLUMNS)


if __name__ == "__main__":
    unittest.main()
