import unittest


class CnnBackboneNoTlAuditTest(unittest.TestCase):
    def test_audit_contract_names_outputs_and_required_columns(self):
        from scripts.audits import cnn_backbone_notl_audit as audit

        self.assertEqual(
            [
                "current_3layer_cnn",
                "conv1_gap_dense",
                "conv1_flatten_dense",
                "dense_only_mlp",
                "naive_persistence",
            ],
            [spec.name for spec in audit.BACKBONE_SPECS],
        )
        self.assertEqual("cnn_backbone_shape_audit.csv", audit.SHAPE_AUDIT_CSV.name)
        self.assertEqual("cnn_backbone_notl_ablation.csv", audit.ABLATION_CSV.name)
        self.assertEqual("cnn_backbone_notl_ablation.md", audit.REPORT_MD.name)

        for column in (
            "dataset",
            "input_shape",
            "conv1d_layer_count",
            "filters",
            "kernel_size",
            "padding",
            "pooling",
            "flatten_dense_params",
            "output_activation",
            "trainable_params",
            "train_windows",
            "val_windows",
            "test_windows",
            "window_size",
            "feature_dim",
            "layer_output_shapes",
            "time_dimension_risk",
        ):
            self.assertIn(column, audit.SHAPE_AUDIT_COLUMNS)

        for column in (
            "dataset",
            "model_name",
            "random_seed",
            "window_size",
            "train_windows",
            "val_windows",
            "test_windows",
            "rmse",
            "normalized_rmse",
            "original_scale_rmse",
            "status",
        ):
            self.assertIn(column, audit.ABLATION_COLUMNS)


if __name__ == "__main__":
    unittest.main()
