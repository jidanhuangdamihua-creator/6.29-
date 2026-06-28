import unittest


class CnnLrClipnormAblationContractTest(unittest.TestCase):
    def test_variant_contract_and_output_columns(self):
        from scripts.audits import cnn_lr_clipnorm_ablation as audit

        self.assertEqual(["fixed_optimizer"], audit.CNN_LR_ABLATION_VARIANTS)
        self.assertIn("lr-1e-4_epochs-50_clipnorm-None_dropout-0.1", audit.DETAIL_CSV.name)
        self.assertIn("lr-1e-4_epochs-50_clipnorm-None_dropout-0.1", audit.SUMMARY_CSV.name)
        self.assertIn("lr-1e-4_epochs-50_clipnorm-None_dropout-0.1", audit.COMPARISON_CSV.name)
        self.assertIn("lr-1e-4_epochs-50_clipnorm-None_dropout-0.1", audit.REPORT_MD.name)

        for column in (
            "dataset_id",
            "dataset",
            "seed",
            "method",
            "cnn_lr_ablation_variant",
            "optimizer_name",
            "learning_rate",
            "clipnorm",
            "optimizer_changed",
            "cnn_structure_changed",
            "batch_size_changed",
            "model_parameter_count",
            "original_model_parameter_count",
            "original_batch_size",
            "effective_batch_size",
            "train_windows",
            "epoch",
            "val_rmse",
            "test_rmse",
            "test_mae",
            "run_time_seconds",
            "error_message",
            "status",
            "notes",
        ):
            self.assertIn(column, audit.DETAIL_COLUMNS)

    def test_optimizer_variant_resolution(self):
        from scripts.audits.cnn_lr_clipnorm_ablation import resolve_lr_optimizer_variant

        original = resolve_lr_optimizer_variant("fixed_optimizer", original_learning_rate=1e-4)
        self.assertEqual("Adam", original.optimizer_name)
        self.assertEqual(1e-4, original.learning_rate)
        self.assertIsNone(original.clipnorm)
        self.assertFalse(original.optimizer_changed)

    def test_batch_size_and_parameter_count_are_unchanged(self):
        from scripts.audits.cnn_lr_clipnorm_ablation import (
            CNN_LR_ABLATION_VARIANTS,
            build_lr_ablation_model,
            resolve_lr_optimizer_variant,
        )

        input_shape = (10, 5)
        original_batch_size = 16
        original_model = build_lr_ablation_model(input_shape, "fixed_optimizer", original_learning_rate=1e-4)
        original_params = original_model.count_params()

        for variant in CNN_LR_ABLATION_VARIANTS:
            with self.subTest(variant=variant):
                meta = resolve_lr_optimizer_variant(
                    variant,
                    original_learning_rate=1e-4,
                    original_batch_size=original_batch_size,
                )
                model = build_lr_ablation_model(input_shape, variant, original_learning_rate=1e-4)
                self.assertEqual(original_batch_size, meta.effective_batch_size)
                self.assertFalse(meta.batch_size_changed)
                self.assertEqual(original_params, model.count_params())


if __name__ == "__main__":
    unittest.main()
