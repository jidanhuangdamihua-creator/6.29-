"""Self-tests for the small-sample training protocol audit script."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import small_sample_training_protocol_ablation as audit


class SmallSampleTrainingProtocolAuditTests(unittest.TestCase):
    def test_detail_columns_include_required_outputs(self) -> None:
        required = {
            "dataset_id",
            "dataset",
            "method",
            "protocol",
            "random_seed",
            "window_size",
            "horizon",
            "cnn_structure",
            "batch_size",
            "target_epochs",
            "early_stopping",
            "restore_best_weights",
            "actual_epochs_run",
            "target_train_windows",
            "target_val_windows",
            "target_test_windows",
            "rmse",
            "accuracy",
            "normalized_rmse",
            "original_scale_rmse",
            "val_rmse",
            "test_rmse",
            "run_time_seconds",
            "status",
            "error_message",
            "notes",
        }
        self.assertTrue(required.issubset(set(audit.DETAIL_COLUMNS)))

    def test_summary_calculates_diffs_and_improvement(self) -> None:
        detail_df = pd.DataFrame(
            [
                {
                    "dataset_id": 1,
                    "dataset": "Dataset1",
                    "method": "No-TL",
                    "protocol": "current_protocol",
                    "rmse": 0.5,
                    "accuracy": 2.0,
                    "status": "OK",
                    "notes": "",
                },
                {
                    "dataset_id": 1,
                    "dataset": "Dataset1",
                    "method": "No-TL",
                    "protocol": "small_sample_training_protocol",
                    "rmse": 0.4,
                    "accuracy": 2.5,
                    "status": "OK",
                    "notes": "",
                },
            ]
        )

        summary = audit._build_summary(detail_df)

        self.assertEqual(len(summary), 1)
        row = summary.iloc[0].to_dict()
        self.assertEqual(row["method"], "No-TL")
        self.assertAlmostEqual(row["rmse_diff"], -0.1)
        self.assertAlmostEqual(row["rmse_percent_change"], -20.0)
        self.assertAlmostEqual(row["accuracy_diff"], 0.5)
        self.assertAlmostEqual(row["accuracy_percent_change"], 25.0)
        self.assertTrue(row["improved"])

    def test_percent_change_handles_zero_denominator(self) -> None:
        self.assertTrue(math.isnan(audit._percent_change(1.0, 0.0)))


if __name__ == "__main__":
    unittest.main()
