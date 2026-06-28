from pathlib import Path
import unittest

import pandas as pd

from scripts import run_d4_experiment
from scripts.run_full_paper_experiments import ROOT, _resolve_output_paths
from scripts.run_unified_d1_d6 import build_tasks


class UnifiedD1D6OutputContractTest(unittest.TestCase):
    def test_unified_runner_passes_one_output_dir_to_d1_and_d4_tasks(self):
        run_dir = Path("outputs/runs/20990101_010203")

        tasks = build_tasks(["d1", "d4"], smoke=True, run_dir=run_dir)

        self.assertEqual(["D1", "D4-without", "D4-with"], [task.label for task in tasks])
        for task in tasks:
            self.assertIn("--output-dir", task.cmd)
            output_arg_index = task.cmd.index("--output-dir") + 1
            self.assertEqual(str(run_dir), task.cmd[output_arg_index])

        d4_without = tasks[1]
        d4_with = tasks[2]
        self.assertEqual("dataset4_without_results.csv", d4_without.result_filename)
        self.assertEqual("dataset4_with_results.csv", d4_with.result_filename)
        self.assertNotIn("_D4_300d", " ".join(d4_without.cmd + d4_with.cmd))


    def test_d4_alignment_uses_reference_dataset_to_error_columns_and_keeps_source_trace(self):
        raw = pd.DataFrame(
            [
                {
                    "dataset": "Dataset4",
                    "method": "MSWA-TL",
                    "rmse": 1.2,
                    "dataset_id": 4,
                    "scenario": "without_information_sharing",
                    "target_entity_key": "store-a",
                    "source_identifier": "store-b",
                    "selected_sources": [{"source_key": "store-b", "distance": 0.1}],
                }
            ]
        )

        aligned = run_d4_experiment._align_results_to_reference_schema(raw)
        reference_columns = run_d4_experiment._reference_result_columns()
        extra_columns = [
            "dataset_id",
            "scenario",
            "target_entity_key",
            "source_identifier",
            "selected_sources",
        ]

        self.assertEqual("dataset", reference_columns[0])
        self.assertEqual("error", reference_columns[-1])
        self.assertEqual(reference_columns + extra_columns, aligned.columns.tolist())
        self.assertEqual("Dataset4", aligned.loc[0, "dataset"])
        self.assertEqual("MSWA-TL", aligned.loc[0, "method"])
        self.assertIsNone(aligned.loc[0, "error"])
        self.assertEqual("store-b", aligned.loc[0, "source_identifier"])
        self.assertNotIn("unified_dataset", aligned.columns)


    def test_full_paper_runner_resolves_explicit_output_dir_without_new_timestamp(self):
        output_dir = Path("outputs/runs/20990101_010203")
        protocol = {"outputs": {"paper_results_csv": "paper_results.csv"}}

        paths = _resolve_output_paths(protocol=protocol, output_dir=output_dir)
        expected_run_dir = ROOT / output_dir

        self.assertEqual("20990101_010203", paths["run_id"])
        self.assertEqual(expected_run_dir, paths["run_dir"])
        self.assertEqual(expected_run_dir / "results", paths["results_dir"])
        self.assertEqual(expected_run_dir / "results" / "paper_results.csv", paths["paper_csv"])


if __name__ == "__main__":
    unittest.main()
