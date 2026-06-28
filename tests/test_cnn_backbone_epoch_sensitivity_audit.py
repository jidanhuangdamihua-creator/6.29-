import unittest


class CnnBackboneEpochSensitivityAuditTest(unittest.TestCase):
    def test_epoch_sensitivity_contract(self):
        from scripts.audits import cnn_backbone_epoch_sensitivity as audit

        self.assertEqual(42, audit.SEED)
        self.assertEqual([50], audit.TARGET_EPOCHS)
        self.assertEqual(["Dataset1", "Dataset2", "Dataset3"], audit.DATASETS)
        self.assertEqual(
            ["current_3layer_cnn", "conv1_gap_dense", "naive_persistence"],
            [spec.name for spec in audit.EPOCH_BACKBONE_SPECS],
        )
        self.assertEqual("cnn_backbone_epoch_sensitivity.csv", audit.EPOCH_SENSITIVITY_CSV.name)
        self.assertEqual("cnn_backbone_epoch_sensitivity.md", audit.EPOCH_SENSITIVITY_REPORT_MD.name)

        for column in (
            "dataset",
            "model_name",
            "random_seed",
            "target_epochs",
            "window_size",
            "batch_size",
            "train_windows",
            "val_windows",
            "test_windows",
            "normalized_rmse",
            "status",
        ):
            self.assertIn(column, audit.EPOCH_SENSITIVITY_COLUMNS)


if __name__ == "__main__":
    unittest.main()
