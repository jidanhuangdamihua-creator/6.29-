import unittest


class CnnLrEpochGridAblationContractTest(unittest.TestCase):
    def test_grid_contract_and_output_columns(self):
        from scripts.audits import cnn_lr_epoch_grid_ablation as audit

        self.assertEqual([1e-4], audit.LEARNING_RATES)
        self.assertEqual([50], audit.EPOCHS)
        self.assertEqual([42, 43, 44], audit.SEEDS)
        self.assertEqual(["Dataset1", "Dataset2", "Dataset3"], audit.DATASETS)
        self.assertEqual(9, audit.expected_detail_row_count())
        self.assertEqual(9, len(audit.iter_grid_combinations()))

        self.assertIn("lr-1e-4_epochs-50_clipnorm-None_dropout-0.1", audit.DETAIL_CSV.name)
        self.assertIn("lr-1e-4_epochs-50_clipnorm-None_dropout-0.1", audit.SUMMARY_CSV.name)
        self.assertIn("lr-1e-4_epochs-50_clipnorm-None_dropout-0.1", audit.COMPARISON_CSV.name)
        self.assertIn("lr-1e-4_epochs-50_clipnorm-None_dropout-0.1", audit.REPORT_MD.name)

        for column in (
            "dataset_id",
            "dataset",
            "seed",
            "method",
            "learning_rate",
            "clipnorm",
            "epoch",
            "optimizer_name",
            "optimizer_changed",
            "cnn_structure_changed",
            "batch_size_changed",
            "model_parameter_count",
            "original_model_parameter_count",
            "original_batch_size",
            "effective_batch_size",
            "train_windows",
            "val_rmse",
            "test_rmse",
            "test_mae",
            "run_time_seconds",
            "error_message",
            "status",
            "notes",
        ):
            self.assertIn(column, audit.DETAIL_COLUMNS)

    def test_batch_size_and_parameter_count_are_unchanged(self):
        from scripts.audits import cnn_lr_epoch_grid_ablation as audit

        input_shape = (10, 5)
        original_batch_size = 16
        original_model = audit.build_epoch_grid_model(input_shape, learning_rate=1e-4)
        original_params = original_model.count_params()

        for learning_rate in audit.LEARNING_RATES:
            with self.subTest(learning_rate=learning_rate):
                meta = audit.resolve_epoch_grid_variant(
                    learning_rate=learning_rate,
                    original_batch_size=original_batch_size,
                )
                model = audit.build_epoch_grid_model(input_shape, learning_rate=learning_rate)
                self.assertEqual(original_batch_size, meta.effective_batch_size)
                self.assertFalse(meta.batch_size_changed)
                self.assertFalse(meta.cnn_structure_changed)
                self.assertEqual(original_params, model.count_params())

    def test_hard_condition_check_flags_invalid_rows(self):
        from scripts.audits import cnn_lr_epoch_grid_ablation as audit

        valid_meta = audit.resolve_epoch_grid_variant(learning_rate=1e-4, original_batch_size=16)
        self.assertEqual([], audit.hard_condition_errors(valid_meta, 123, 123, expected_epoch=50))

        invalid_meta = audit.resolve_epoch_grid_variant(learning_rate=1e-4, original_batch_size=16)
        invalid_meta = invalid_meta.__class__(
            optimizer_name="SGD",
            learning_rate=2e-4,
            clipnorm=1.0,
            optimizer_changed=True,
            cnn_structure_changed=True,
            batch_size_changed=True,
            original_batch_size=16,
            effective_batch_size=1,
        )
        errors = audit.hard_condition_errors(invalid_meta, 124, 123, expected_epoch=7, observed_method="MSML-TL")
        joined = "; ".join(errors)
        self.assertIn("method must be No-TL", joined)
        self.assertIn("optimizer_name must be Adam", joined)
        self.assertIn("learning_rate must be 1e-4", joined)
        self.assertIn("epoch must be 50", joined)
        self.assertIn("clipnorm must be None", joined)
        self.assertIn("cnn_structure_changed must be False", joined)
        self.assertIn("batch_size_changed must be False", joined)
        self.assertIn("model_parameter_count must equal original_model_parameter_count", joined)
        self.assertIn("effective_batch_size must equal original_batch_size", joined)


if __name__ == "__main__":
    unittest.main()
