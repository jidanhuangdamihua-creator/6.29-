import unittest


class CnnStabilityFactorialAblationContractTest(unittest.TestCase):
    def test_variant_contract_and_output_columns(self):
        from scripts.audits import cnn_stability_factorial_ablation as audit

        self.assertEqual(
            [
                "original",
                "change1_batch_size_1",
                "change2_no_batch_norm",
                "change3_low_lr_clipnorm",
                "change123_all",
            ],
            audit.CNN_ABLATION_VARIANTS,
        )
        self.assertEqual("cnn_stability_factorial_ablation_details.csv", audit.DETAIL_CSV.name)
        self.assertEqual("cnn_stability_factorial_ablation_summary.csv", audit.SUMMARY_CSV.name)
        self.assertEqual("cnn_stability_factorial_ablation_comparison.csv", audit.COMPARISON_CSV.name)
        self.assertEqual("cnn_stability_factorial_ablation.md", audit.REPORT_MD.name)

        for column in (
            "dataset_id",
            "dataset",
            "seed",
            "method",
            "cnn_ablation_variant",
            "model_name",
            "change1_batch_size_1_enabled",
            "change2_no_batch_norm_enabled",
            "change3_low_lr_clipnorm_enabled",
            "batch_norm_enabled",
            "cnn_normalization",
            "original_batch_size",
            "effective_batch_size",
            "train_windows",
            "learning_rate",
            "clipnorm",
            "optimizer_name",
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

        for column in (
            "dataset_id",
            "dataset",
            "cnn_ablation_variant",
            "n_seeds",
            "mean_test_rmse",
            "std_test_rmse",
            "improvement_vs_original_mean_pct",
            "std_reduction_vs_original_pct",
            "rank_by_mean_rmse",
            "rank_by_std_rmse",
            "status",
            "conclusion",
        ):
            self.assertIn(column, audit.SUMMARY_COLUMNS)

        for column in (
            "dataset_id",
            "dataset",
            "original_mean_rmse",
            "change1_mean_rmse",
            "change2_mean_rmse",
            "change3_mean_rmse",
            "change123_mean_rmse",
            "best_single_change",
            "best_overall_variant",
            "does_combined_outperform_all_single_changes",
            "interpretation",
        ):
            self.assertIn(column, audit.COMPARISON_COLUMNS)

    def test_variant_hard_condition_resolution(self):
        from src.models.cnn_model import resolve_cnn_ablation_training_config

        original_batch_size = 16
        expectations = {
            "original": (False, False, False, 16, None),
            "change1_batch_size_1": (True, False, False, 1, None),
            "change2_no_batch_norm": (False, True, False, 16, None),
            "change3_low_lr_clipnorm": (False, False, True, 16, None),
            "change123_all": (True, True, True, 1, None),
        }
        for variant, expected in expectations.items():
            with self.subTest(variant=variant):
                meta = resolve_cnn_ablation_training_config(
                    cnn_ablation_variant=variant,
                    original_batch_size=original_batch_size,
                    original_learning_rate=1e-4,
                )
                change1, change2, change3, batch_size, clipnorm = expected
                self.assertEqual(change1, meta.change1_batch_size_1_enabled)
                self.assertEqual(change2, meta.change2_no_batch_norm_enabled)
                self.assertEqual(change3, meta.change3_low_lr_clipnorm_enabled)
                self.assertEqual(batch_size, meta.effective_batch_size)
                self.assertEqual(clipnorm, meta.clipnorm)
                if change3:
                    self.assertEqual(1e-4, meta.learning_rate)
                    self.assertEqual("Adam", meta.optimizer_name)
                else:
                    self.assertEqual(1e-4, meta.learning_rate)


if __name__ == "__main__":
    unittest.main()
