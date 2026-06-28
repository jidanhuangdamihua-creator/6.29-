import unittest


class CnnBackboneNoTlMultiseedAuditTest(unittest.TestCase):
    def test_multiseed_contract(self):
        from scripts.audits import cnn_backbone_notl_ablation_multiseed as audit

        self.assertEqual([42, 43, 44, 45, 46], audit.SEEDS)
        self.assertEqual(["Dataset1", "Dataset2", "Dataset3"], audit.DATASETS)
        self.assertEqual(
            ["current_3layer_cnn", "conv1_gap_dense", "naive_persistence"],
            [spec.name for spec in audit.MULTISEED_BACKBONE_SPECS],
        )
        self.assertEqual("cnn_backbone_notl_ablation_multiseed.csv", audit.MULTISEED_CSV.name)
        self.assertEqual("cnn_backbone_notl_ablation_multiseed.md", audit.MULTISEED_REPORT_MD.name)

        for column in (
            "dataset",
            "model_name",
            "random_seed",
            "window_size",
            "target_epochs",
            "batch_size",
            "train_windows",
            "val_windows",
            "test_windows",
            "normalized_rmse",
            "status",
        ):
            self.assertIn(column, audit.MULTISEED_COLUMNS)


if __name__ == "__main__":
    unittest.main()
