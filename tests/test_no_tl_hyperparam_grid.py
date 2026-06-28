import unittest


class NoTlHyperparamGridContractTest(unittest.TestCase):
    def test_grid_contract_and_output_columns(self):
        from scripts.audits import no_tl_hyperparam_grid as audit

        self.assertEqual(["Dataset1", "Dataset3"], audit.DATASETS)
        self.assertEqual([1e-4], audit.LEARNING_RATES)
        self.assertEqual([50], audit.EPOCHS)
        self.assertEqual([None], audit.CLIPNORMS)
        self.assertEqual([0.1], audit.DROPOUTS)
        self.assertEqual([42, 43, 44, 45, 46], audit.SEEDS)
        self.assertEqual(10, audit.expected_detail_row_count())
        self.assertEqual(10, len(audit.iter_grid_combinations()))

        self.assertIn("lr-1e-4_epochs-50_clipnorm-None_dropout-0.1", audit.DETAIL_CSV.name)
        self.assertIn("lr-1e-4_epochs-50_clipnorm-None_dropout-0.1", audit.AGG_CSV.name)
        self.assertIn("lr-1e-4_epochs-50_clipnorm-None_dropout-0.1", audit.REPORT_MD.name)
        self.assertIn("lr-1e-4_epochs-50_clipnorm-None_dropout-0.1", audit.BEST_BY_VAL_CSV.name)
        self.assertIn("lr-1e-4_epochs-50_clipnorm-None_dropout-0.1", audit.BEST_BY_TEST_RMSE_CSV.name)
        self.assertIn("lr-1e-4_epochs-50_clipnorm-None_dropout-0.1", audit.BEST_BY_NORMALIZED_RMSE_CSV.name)

        for column in (
            "dataset",
            "learning_rate",
            "epochs",
            "clipnorm",
            "dropout",
            "seed",
            "train_loss",
            "validation_loss",
            "test_mae",
            "test_rmse",
            "normalized_rmse",
            "original_scale_rmse",
            "mape_exclude_zero",
            "early_stopping_stopped_epoch",
            "best_validation_epoch",
            "training_time_seconds",
            "loss_anomaly",
            "gradient_explosion",
            "nan_detected",
            "overfitting_detected",
            "status",
            "log_file",
        ):
            self.assertIn(column, audit.DETAIL_COLUMNS)

        for column in (
            "dataset",
            "learning_rate",
            "epochs",
            "clipnorm",
            "dropout",
            "n_seeds",
            "train_loss_mean",
            "train_loss_std",
            "validation_loss_mean",
            "validation_loss_std",
            "test_mae_mean",
            "test_mae_std",
            "test_rmse_mean",
            "test_rmse_std",
            "normalized_rmse_mean",
            "normalized_rmse_std",
            "original_scale_rmse_mean",
            "original_scale_rmse_std",
            "mape_exclude_zero_mean",
            "mape_exclude_zero_std",
            "early_stopping_stopped_epoch_mean",
            "early_stopping_stopped_epoch_std",
            "best_validation_epoch_mean",
            "best_validation_epoch_std",
            "training_time_seconds_mean",
            "training_time_seconds_std",
        ):
            self.assertIn(column, audit.AGG_COLUMNS)

    def test_model_contract_preserves_batch_size_and_parameter_count(self):
        from scripts.audits import no_tl_hyperparam_grid as audit

        input_shape = (10, 5)
        original = audit.build_grid_model(input_shape, learning_rate=1e-4, clipnorm=None, dropout=0.1)
        original_params = original.count_params()

        for learning_rate in audit.LEARNING_RATES:
            for clipnorm in audit.CLIPNORMS:
                for dropout in audit.DROPOUTS:
                    with self.subTest(learning_rate=learning_rate, clipnorm=clipnorm, dropout=dropout):
                        meta = audit.resolve_grid_variant(
                            learning_rate=learning_rate,
                            epochs=50,
                            clipnorm=clipnorm,
                            dropout=dropout,
                            original_batch_size=16,
                        )
                        model = audit.build_grid_model(
                            input_shape,
                            learning_rate=learning_rate,
                            clipnorm=clipnorm,
                            dropout=dropout,
                        )
                        self.assertEqual(16, meta.effective_batch_size)
                        self.assertFalse(meta.cnn_structure_changed)
                        self.assertFalse(meta.batch_size_changed)
                        self.assertEqual(original_params, model.count_params())

    def test_build_seed_aggregate_computes_mean_and_std_per_hyperparam_group(self):
        import pandas as pd

        from scripts.audits import no_tl_hyperparam_grid as audit

        rows = []
        for seed, normalized_rmse in [(42, 0.40), (43, 0.44)]:
            rows.append(
                {
                    "dataset_id": 1,
                    "dataset": "Dataset1",
                    "learning_rate": 1e-4,
                    "epochs": 50,
                    "clipnorm": None,
                    "dropout": 0.1,
                    "seed": seed,
                    "status": "OK",
                    "train_loss": 0.1,
                    "validation_loss": 0.2,
                    "test_mae": 0.3,
                    "test_rmse": normalized_rmse,
                    "normalized_rmse": normalized_rmse,
                    "original_scale_rmse": normalized_rmse * 100,
                    "mape_exclude_zero": 10.0,
                    "early_stopping_stopped_epoch": 12,
                    "best_validation_epoch": 7,
                    "training_time_seconds": 1.0,
                }
            )

        agg = audit.build_seed_aggregate(pd.DataFrame(rows))
        self.assertEqual(1, len(agg))
        self.assertEqual(2, int(agg.iloc[0]["n_seeds"]))
        self.assertAlmostEqual(0.42, float(agg.iloc[0]["normalized_rmse_mean"]))
        self.assertAlmostEqual(0.028284271, float(agg.iloc[0]["normalized_rmse_std"]), places=8)

    def test_stability_flags_detect_nan_loss_and_overfit(self):
        from scripts.audits import no_tl_hyperparam_grid as audit

        flags = audit.analyze_training_history(
            history={
                "loss": [0.5, float("nan")],
                "val_loss": [0.4, 1.2],
            },
            train_loss=0.2,
            validation_loss=0.8,
        )

        self.assertTrue(flags["loss_anomaly"])
        self.assertTrue(flags["nan_detected"])
        self.assertTrue(flags["overfitting_detected"])
        self.assertTrue(flags["gradient_explosion"])

    def test_base_row_accepts_output_dir_outside_project_root(self):
        from pathlib import Path

        from scripts.audits import no_tl_hyperparam_grid as audit

        meta = audit.resolve_grid_variant(learning_rate=1e-4, epochs=50, clipnorm=None, dropout=0.1, original_batch_size=16)
        prepared = {"y_train": [1, 2], "y_val": [3], "y_test": [4]}
        log_file = Path("/private/tmp/no_tl_grid_test/Dataset1_lr-1e-4_epochs-50_clipnorm-None_dropout-0.1_seed-42.log")

        row = audit._base_row(
            dataset="Dataset1",
            seed=42,
            meta=meta,
            prepared=prepared,
            original_parameter_count=123,
            log_file=log_file,
        )

        self.assertEqual(str(log_file), row["log_file"])


if __name__ == "__main__":
    unittest.main()
