import tempfile
import unittest
from pathlib import Path

import pandas as pd

from paper_reproduction_protocol import get_results_output_paths
from src.utils.scenario_reports import generate_scenario_separated_reports


class ScenarioSeparatedReportsTest(unittest.TestCase):
    def test_reports_keep_information_sharing_scenarios_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "results_reports"
            results_df = pd.DataFrame(
                [
                    {
                        "dataset": "Dataset1",
                        "method": "No-TL",
                        "scenario": "without_information_sharing",
                        "rmse": 100.0,
                        "accuracy": 0.10,
                    },
                    {
                        "dataset": "Dataset1",
                        "method": "MSML-TL-RFE",
                        "scenario": "without_information_sharing",
                        "rmse": 80.0,
                        "accuracy": 0.20,
                    },
                    {
                        "dataset": "Dataset1",
                        "method": "No-TL",
                        "scenario": "with_information_sharing",
                        "rmse": 50.0,
                        "accuracy": 0.50,
                    },
                    {
                        "dataset": "Dataset1",
                        "method": "MSML-TL-RFE",
                        "scenario": "with_information_sharing",
                        "rmse": 40.0,
                        "accuracy": 0.60,
                    },
                    {
                        "dataset": "Dataset2",
                        "method": "No-TL",
                        "scenario": "without_information_sharing",
                        "rmse": 200.0,
                        "accuracy": 0.30,
                    },
                    {
                        "dataset": "Dataset2",
                        "method": "No-TL",
                        "scenario": "without_information_sharing",
                        "rmse": 190.0,
                        "accuracy": 0.35,
                    },
                ]
            )

            report = generate_scenario_separated_reports(results_df, output_dir)

            without_rmse = pd.read_csv(
                output_dir / "rmse_comparison_without_information_sharing.csv",
                index_col=0,
            )
            with_rmse = pd.read_csv(
                output_dir / "rmse_comparison_with_information_sharing.csv",
                index_col=0,
            )
            without_improvement = pd.read_csv(
                output_dir / "improvement_vs_notl_without_information_sharing.csv"
            )
            duplicate_rows = pd.read_csv(
                output_dir / "duplicate_dataset_method_scenario_rows.csv"
            )

            self.assertEqual(100.0, without_rmse.loc["Dataset1", "No-TL"])
            self.assertEqual(50.0, with_rmse.loc["Dataset1", "No-TL"])
            self.assertIn("scenario", pd.read_csv(output_dir / "all_results_long_format.csv").columns)
            self.assertEqual(
                20.0,
                float(
                    without_improvement.loc[
                        without_improvement["method"] == "MSML-TL-RFE",
                        "rmse_improvement_percent",
                    ].iloc[0]
                ),
            )
            self.assertEqual(1, len(duplicate_rows))
            self.assertEqual("Dataset2", duplicate_rows.iloc[0]["dataset"])
            self.assertEqual(
                output_dir / "duplicate_dataset_method_scenario_rows.csv",
                report["duplicate_rows_path"],
            )
            self.assertTrue((output_dir / "rmse_comparison_without_information_sharing.png").exists())
            self.assertTrue((output_dir / "accuracy_comparison_with_information_sharing.png").exists())

    def test_results_output_paths_create_unique_run_directory_and_latest_pointer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            first_paths = get_results_output_paths(root, {"outputs": {}})
            second_paths = get_results_output_paths(root, {"outputs": {}})

            self.assertNotEqual(first_paths["run_dir"], second_paths["run_dir"])
            self.assertEqual(first_paths["run_dir"] / "results", first_paths["results_dir"])
            self.assertEqual(first_paths["run_dir"] / "results_reports", first_paths["reports_dir"])
            self.assertEqual(first_paths["run_dir"] / "paper_alignment", first_paths["alignment_dir"])
            self.assertEqual(first_paths["run_dir"] / "audits", first_paths["audits_dir"])
            self.assertTrue(first_paths["paper_csv"].match("*/outputs/runs/*/results/paper_results.csv"))
            self.assertEqual(str(second_paths["run_dir"]), (root / "outputs" / "latest_run.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
