import argparse
import os
from pathlib import Path
import re
import subprocess
import unittest
from unittest.mock import patch

import pandas as pd

from scripts import run_d4_experiment, run_full_paper_experiments, run_unified_d1_d6
from scripts.run_full_paper_experiments import ROOT, _resolve_output_paths
from scripts.run_unified_d1_d6 import build_tasks


class UnifiedD1D6OutputContractTest(unittest.TestCase):
    def test_compat_results_copy_can_be_disabled_by_environment(self):
        self.assertTrue(
            hasattr(run_full_paper_experiments, "_should_sync_latest_results_copy")
        )

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RFE_DISABLE_COMPAT_RESULTS_COPY", None)
            self.assertTrue(
                run_full_paper_experiments._should_sync_latest_results_copy()
            )

        with patch.dict(
            os.environ,
            {"RFE_DISABLE_COMPAT_RESULTS_COPY": "1"},
            clear=False,
        ):
            self.assertFalse(
                run_full_paper_experiments._should_sync_latest_results_copy()
            )

    def test_parallel_runner_dry_run_prints_six_isolated_commands(self):
        runner = ROOT / "scripts" / "parallel_runner.sh"
        self.assertTrue(runner.is_file(), "parallel runner script is missing")

        completed = subprocess.run(
            ["bash", str(runner), "--dry-run"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(0, completed.returncode, completed.stderr)
        command_lines = [
            line for line in completed.stdout.splitlines() if line.startswith("[d")
        ]
        self.assertEqual(6, len(command_lines))
        for dataset_id, line in enumerate(command_lines, start=1):
            token = f"d{dataset_id}"
            self.assertIn(f"--only {token}", line)
            self.assertRegex(
                line,
                rf"--output-dir \S*/outputs/runs/\d{{8}}_\d{{6}}/{token}$",
            )
            self.assertNotRegex(line, rf"/{token}/{token}(?:/|$)")

        self.assertEqual(
            {"d1", "d2", "d3", "d4", "d5", "d6"},
            set(re.findall(r"--only (d[1-6])", completed.stdout)),
        )

    def test_parallel_runner_has_no_thread_limits_or_aggregation(self):
        runner = ROOT / "scripts" / "parallel_runner.sh"
        self.assertTrue(runner.is_file(), "parallel runner script is missing")
        text = runner.read_text(encoding="utf-8")

        for forbidden in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "TF_NUM_INTEROP_THREADS",
            "TF_NUM_INTRAOP_THREADS",
            "aggregate_d1_d6_results.py",
        ):
            self.assertNotIn(forbidden, text)

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

    def test_unified_runner_can_build_single_mode_tasks(self):
        run_dir = Path("outputs/runs/20990101_010203")

        without_tasks = build_tasks(
            ["d1", "d4"],
            smoke=True,
            run_dir=run_dir,
            info_sharing="without",
        )
        with_tasks = build_tasks(
            ["d1", "d4"],
            smoke=True,
            run_dir=run_dir,
            info_sharing="with",
        )

        self.assertEqual(["D1-without", "D4-without"], [task.label for task in without_tasks])
        self.assertEqual(["D1-with", "D4-with"], [task.label for task in with_tasks])
        self.assertEqual("dataset1_without_results.csv", without_tasks[0].result_filename)
        self.assertEqual("dataset1_with_results.csv", with_tasks[0].result_filename)
        self.assertIn("--info-sharing", without_tasks[0].cmd)
        self.assertIn("without", without_tasks[0].cmd)
        self.assertIn("--info-sharing", with_tasks[0].cmd)
        self.assertIn("with", with_tasks[0].cmd)
        self.assertEqual("dataset4_without_results.csv", without_tasks[1].result_filename)
        self.assertEqual("dataset4_with_results.csv", with_tasks[1].result_filename)

    def test_unified_runner_exits_nonzero_when_child_task_fails(self):
        failed_task = run_unified_d1_d6.Task(
            dataset_token="d5",
            dataset_id=5,
            label="D5-without",
            scenario="without",
            cmd=["python", "scripts/run_d5_experiment.py"],
            config_check="[CONFIG CHECK]",
            result_filename="dataset5_without_results.csv",
            returncode=1,
        )

        with patch.object(run_unified_d1_d6, "_parse_args") as parse_args, patch.object(
            run_unified_d1_d6, "build_tasks", return_value=[failed_task]
        ), patch.object(
            run_unified_d1_d6, "run_task", return_value=failed_task
        ), patch.object(
            run_unified_d1_d6, "print_result_summary"
        ):
            parse_args.return_value = argparse.Namespace(
                only=["d5"],
                smoke=False,
                dry_run=False,
                output_dir=None,
                info_sharing="without",
            )

            with self.assertRaises(SystemExit) as ctx:
                run_unified_d1_d6.main()

        self.assertEqual(1, ctx.exception.code)

    def test_parallel_mode_runner_dry_run_prints_twelve_isolated_mode_commands(self):
        runner = ROOT / "scripts" / "parallel_mode_runner.sh"
        self.assertTrue(runner.is_file(), "parallel mode runner script is missing")

        env = os.environ.copy()
        env["DRY_RUN"] = "1"
        env.pop("MAX_JOBS", None)
        completed = subprocess.run(
            ["bash", str(runner)],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertIn("[DRY-RUN] MAX_JOBS=10", completed.stdout)
        command_lines = [
            line for line in completed.stdout.splitlines() if line.startswith("[DRY-RUN] d")
        ]
        self.assertEqual(12, len(command_lines))
        expected_order = [
            "d5_without",
            "d5_with",
            "d1_without",
            "d1_with",
            "d2_without",
            "d2_with",
            "d3_without",
            "d3_with",
            "d4_without",
            "d6_without",
            "d4_with",
            "d6_with",
        ]
        self.assertEqual(expected_order, [line.split(":", 1)[0].split()[-1] for line in command_lines])
        self.assertRegex(
            completed.stdout,
            r"d5_without: .* --info-sharing without --output-dir \S*/outputs/runs/\d{8}_\d{6}/d5_without log=\S*/outputs/parallel_mode_runs/\d{8}_\d{6}/d5_without\.log",
        )
        self.assertRegex(
            completed.stdout,
            r"d5_with: .* --info-sharing with --output-dir \S*/outputs/runs/\d{8}_\d{6}/d5_with log=\S*/outputs/parallel_mode_runs/\d{8}_\d{6}/d5_with\.log",
        )
        self.assertNotRegex(completed.stdout, r"d5_without.*--output-dir \S*/d5(?:\s|$)")
        self.assertNotRegex(completed.stdout, r"d5_with.*--output-dir \S*/d5(?:\s|$)")

    def test_parallel_mode_runner_dry_run_honors_max_jobs_override(self):
        runner = ROOT / "scripts" / "parallel_mode_runner.sh"
        self.assertTrue(runner.is_file(), "parallel mode runner script is missing")

        env = os.environ.copy()
        env["DRY_RUN"] = "1"
        env["MAX_JOBS"] = "12"
        completed = subprocess.run(
            ["bash", str(runner)],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertIn("[DRY-RUN] MAX_JOBS=12", completed.stdout)

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
