import unittest

from scripts.run_full_paper_experiments import (
    _materialize_result_dataframes,
    normalize_information_sharing_contract,
)


class NoTlMinimalFixTest(unittest.TestCase):
    def test_no_tl_and_ss_tl_materialized_information_sharing_uses_contract_values(self):
        paper_df, extended_df = _materialize_result_dataframes(
            [
                {
                    "dataset": "Dataset1",
                    "method": "No-TL",
                    "information_sharing": "without_information_sharing",
                    "scenario": "without_information_sharing",
                    "source_domain_filter_reason": "without_information_sharing_same_store",
                    "source_pool_scope_mode": "without_information_sharing_same_store",
                    "signature_components": {"scenario": "without_information_sharing"},
                    "source_count": 0,
                    "selected_source_count": 0,
                    "rmse": 0.123,
                    "accuracy": 8.0,
                    "mae": 0.04,
                    "mape": 12.0,
                    "smape": 1.23,
                    "prediction_shape": "(174, 1)",
                    "metric_space_current": "normalized_minmax_space",
                    "metric_space_used": "normalized_minmax_space",
                    "paper_metric_aligned": False,
                    "inverse_transform_applied": False,
                    "inverse_transform_available": True,
                    "rmse_paper": 12.3,
                    "accuracy_paper": 0.08,
                    "mae_paper": 4.0,
                    "error": "",
                },
                {
                    "dataset": "Dataset1",
                    "method": "SS-TL",
                    "information_sharing": "with_information_sharing",
                    "scenario": "with_information_sharing",
                    "source_domain_filter_reason": "with_information_sharing_full_pool",
                    "source_pool_scope_mode": "with_information_sharing_full_pool",
                    "signature_components": {"scenario": "with_information_sharing"},
                    "source_count": 1,
                    "selected_source_count": 1,
                    "rmse": 0.111,
                    "accuracy": 9.0,
                    "mae": 0.03,
                    "mape": 11.0,
                    "smape": 1.11,
                    "prediction_shape": "(174, 1)",
                    "metric_space_current": "normalized_minmax_space",
                    "metric_space_used": "normalized_minmax_space",
                    "paper_metric_aligned": False,
                    "inverse_transform_applied": False,
                    "inverse_transform_available": True,
                    "rmse_paper": 11.1,
                    "accuracy_paper": 0.09,
                    "mae_paper": 3.0,
                    "error": "",
                },
            ],
            [],
        )

        self.assertTrue(extended_df.empty)
        self.assertEqual(["without", "with"], paper_df["information_sharing"].tolist())
        self.assertFalse(
            paper_df["information_sharing"].isin(
                {"without_information_sharing", "with_information_sharing"}
            ).any()
        )
        self.assertEqual(
            ["without_information_sharing", "with_information_sharing"],
            paper_df["scenario"].tolist(),
        )
        self.assertEqual(
            [
                "without_information_sharing_same_store",
                "with_information_sharing_full_pool",
            ],
            paper_df["source_domain_filter_reason"].tolist(),
        )
        self.assertEqual(
            [
                "without_information_sharing_same_store",
                "with_information_sharing_full_pool",
            ],
            paper_df["source_pool_scope_mode"].tolist(),
        )
        self.assertIn("without_information_sharing", str(paper_df.loc[0, "signature_components"]))
        self.assertIn("with_information_sharing", str(paper_df.loc[1, "signature_components"]))

    def test_contract_helper_accepts_only_current_and_legacy_aliases(self):
        self.assertEqual("without", normalize_information_sharing_contract("without"))
        self.assertEqual("without", normalize_information_sharing_contract("without_information_sharing"))
        self.assertEqual("with", normalize_information_sharing_contract("with"))
        self.assertEqual("with", normalize_information_sharing_contract("with_information_sharing"))

        with self.assertRaisesRegex(ValueError, "Unsupported information_sharing contract value"):
            normalize_information_sharing_contract("cross_store")


if __name__ == "__main__":
    unittest.main()
