import unittest

from scripts.run_full_paper_experiments import _clone_no_tl_record_for_scenario


class NoTlMinimalFixTest(unittest.TestCase):
    def test_clone_no_tl_record_reuses_primary_metrics_for_other_scenario(self):
        cached = {
            "dataset": "Dataset1",
            "method": "No-TL",
            "information_sharing": "without_information_sharing",
            "source_count": 0,
            "selected_source_count": 0,
            "selected_source_ids": "NOT_APPLICABLE",
            "selected_source_keys": "NOT_APPLICABLE",
            "selected_source_raw_columns": "NOT_APPLICABLE",
            "rmse": 0.123,
            "accuracy": 8.0,
            "mae": 0.04,
            "mape": 12.0,
            "prediction_shape": "(174, 1)",
            "metric_space_current": "normalized_minmax_space",
            "metric_space_used": "normalized_minmax_space",
            "paper_metric_aligned": False,
            "inverse_transform_applied": False,
            "inverse_transform_available": True,
            "rmse_paper": 12.3,
            "accuracy_paper": 0.08,
            "mae_paper": 4.0,
        }

        cloned = _clone_no_tl_record_for_scenario(cached, "with_information_sharing")

        self.assertEqual("with_information_sharing", cloned["information_sharing"])
        for field in (
            "rmse",
            "accuracy",
            "mae",
            "mape",
            "prediction_shape",
            "metric_space_current",
            "metric_space_used",
            "paper_metric_aligned",
            "inverse_transform_applied",
            "inverse_transform_available",
            "rmse_paper",
            "accuracy_paper",
            "mae_paper",
            "source_count",
            "selected_source_count",
            "selected_source_ids",
            "selected_source_keys",
            "selected_source_raw_columns",
        ):
            self.assertEqual(cached[field], cloned[field])


if __name__ == "__main__":
    unittest.main()
