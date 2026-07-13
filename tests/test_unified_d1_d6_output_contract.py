import argparse
import os
import tempfile
from pathlib import Path
import re
import subprocess
import unittest
from unittest.mock import patch

import pandas as pd

from scripts import run_d4_experiment, run_full_paper_experiments, run_unified_d1_d6
from scripts.run_full_paper_experiments import ROOT, _resolve_output_paths
from scripts.run_unified_d1_d6 import build_tasks
from src.constants import RESULT_CONTRACT_VERSION


class UnifiedD1D6OutputContractTest(unittest.TestCase):
    def test_d3_strict_without_uses_shared_store_pool_not_legacy_region_filter(self):
        source = pd.DataFrame(
            {"store_id": [1, 2, 9, 20], "region_id": [1, 2, 3, 4], "sales": [1, 2, 3, 4]}
        )
        filtered = run_full_paper_experiments._apply_information_sharing_filter(
            dataset_name="Dataset3",
            source_df=source,
            target_df=pd.DataFrame({"store_id": [10]}),
            use_information_sharing=False,
            strict_paper_mode=True,
            protocol={},
            cfg={},
        )
        self.assertEqual(filtered["store_id"].tolist(), [1, 2, 9, 20])
        self.assertEqual(
            filtered.attrs["domain_filter_used"],
            {"mode": "shared_protocol_exact_pool"},
        )

    def test_compat_results_copy_is_opt_in_by_environment(self):
        self.assertTrue(
            hasattr(run_full_paper_experiments, "_should_sync_latest_results_copy")
        )

        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(
                run_full_paper_experiments._should_sync_latest_results_copy()
            )

        with patch.dict(
            os.environ,
            {"RFE_ENABLE_COMPAT_RESULTS_COPY": "1"},
            clear=True,
        ):
            self.assertTrue(
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

    def test_unified_runner_passes_isolated_output_dirs_to_d1_and_d4_tasks(self):
        run_dir = Path("outputs/runs/20990101_010203")

        tasks = build_tasks(["d1", "d4"], smoke=True, run_dir=run_dir)

        self.assertEqual(
            ["D1-without", "D1-with", "D4-without", "D4-with"],
            [task.label for task in tasks],
        )

        expected_dirs = {
            "D1-without": run_dir / "d1_without",
            "D1-with": run_dir / "d1_with",
            "D4-without": run_dir / "d4_without",
            "D4-with": run_dir / "d4_with",
        }

        for task in tasks:
            self.assertIn("--output-dir", task.cmd)
            output_arg_index = task.cmd.index("--output-dir") + 1
            self.assertEqual(
                str(expected_dirs[task.label]),
                task.cmd[output_arg_index],
            )

        d4_without = tasks[2]
        d4_with = tasks[3]
        self.assertEqual(
            "dataset4_without_results.csv",
            d4_without.result_filename,
        )
        self.assertEqual(
            "dataset4_with_results.csv",
            d4_with.result_filename,
        )
        self.assertNotIn(
            "_D4_300d",
            " ".join(d4_without.cmd + d4_with.cmd),
        )

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

        self.assertEqual("result_contract_version", reference_columns[0])
        self.assertIn("dataset", reference_columns)
        self.assertIn("error", reference_columns)
        for column in reference_columns + extra_columns:
            self.assertIn(column, aligned.columns)
        self.assertEqual("Dataset4", aligned.loc[0, "dataset"])
        self.assertEqual("MSWA-TL", aligned.loc[0, "method"])
        self.assertEqual("", aligned.loc[0, "error"])
        self.assertEqual("store-b", aligned.loc[0, "source_identifier"])
        self.assertNotIn("unified_dataset", aligned.columns)
        self.assertEqual("d1_d6_superset_v1", aligned.loc[0, "result_contract_version"])

    def test_full_paper_materialize_uses_superset_order_without_cropping_extras(self):
        paper_df, _ = run_full_paper_experiments._materialize_result_dataframes(
            [
                {
                    "dataset": "Dataset1",
                    "method": "No-TL",
                    "information_sharing": "without_information_sharing",
                    "rmse": 1.0,
                    "smape": 2.0,
                    "prediction_shape": (2, 1),
                    "selected_sources": "not_applicable",
                    "legacy_metric_only": "keep-me",
                    "error": "",
                }
            ],
            [],
        )

        self.assertEqual(RESULT_CONTRACT_VERSION, paper_df.loc[0, "result_contract_version"])
        self.assertEqual("d1_d3_single_target_runtime_knn", paper_df.loc[0, "schema_family"])
        self.assertEqual("legacy_unverified", paper_df.loc[0, "result_status"])
        self.assertIn("selected_sources", paper_df.columns)
        self.assertIn("legacy_metric_only", paper_df.columns)
        self.assertLess(
            paper_df.columns.get_loc("result_contract_version"),
            paper_df.columns.get_loc("legacy_metric_only"),
        )
        self.assertEqual("without", paper_df.loc[0, "information_sharing"])
        self.assertNotIn(
            paper_df.loc[0, "information_sharing"],
            {"with_information_sharing", "without_information_sharing"},
        )

    def test_full_paper_materialize_normalizes_only_information_sharing_contract(self):
        paper_df, _ = run_full_paper_experiments._materialize_result_dataframes(
            [
                {
                    "dataset": "Dataset1",
                    "method": "MSWA-TL",
                    "information_sharing": "with_information_sharing",
                    "scenario": "with_information_sharing",
                    "source_domain_filter_reason": "with_information_sharing_full_pool",
                    "source_pool_scope_mode": "with_information_sharing_full_pool",
                    "signature_components": {"scenario": "with_information_sharing"},
                    "rmse": 1.0,
                    "smape": 2.0,
                    "prediction_shape": (2, 1),
                    "error": "",
                }
            ],
            [],
        )

        self.assertEqual("with", paper_df.loc[0, "information_sharing"])
        self.assertEqual("with_information_sharing", paper_df.loc[0, "scenario"])
        self.assertEqual(
            "with_information_sharing_full_pool",
            paper_df.loc[0, "source_domain_filter_reason"],
        )
        self.assertEqual(
            "with_information_sharing_full_pool",
            paper_df.loc[0, "source_pool_scope_mode"],
        )
        self.assertIn(
            "with_information_sharing",
            str(paper_df.loc[0, "signature_components"]),
        )

    def test_full_paper_materialize_rejects_unknown_information_sharing_contract(self):
        with self.assertRaisesRegex(ValueError, "Unsupported information_sharing contract value"):
            run_full_paper_experiments._materialize_result_dataframes(
                [
                    {
                        "dataset": "Dataset1",
                        "method": "MSWA-TL",
                        "information_sharing": "cross_store",
                        "rmse": 1.0,
                        "smape": 2.0,
                        "prediction_shape": (2, 1),
                        "error": "",
                    }
                ],
                [],
            )

    def test_error_row_normalizes_information_sharing_contract(self):
        row = run_full_paper_experiments._build_error_row(
            dataset_name="Dataset1",
            method_name="MSWA-TL",
            source_count=3,
            information_sharing_scenario="without_information_sharing",
            protocol={},
            strict_paper_mode=True,
            exc=RuntimeError("boom"),
        )

        self.assertEqual("without", row["information_sharing"])

    def test_latest_results_copy_reuses_dataset_frames_without_shape_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_root = Path(tmpdir)
            output_dir = fake_root / "run"
            paper_df = pd.DataFrame(
                [
                    {
                        "dataset": "Dataset1",
                        "method": "No-TL",
                        "information_sharing": "without",
                        "rmse": 1.0,
                        "accuracy": 0.5,
                        "smape": 2.0,
                        "error": "",
                        "legacy_metric_only": "keep-me",
                    },
                    {
                        "dataset": "Dataset1",
                        "method": "MSWA-TL",
                        "information_sharing": "with",
                        "rmse": 0.9,
                        "accuracy": 0.6,
                        "smape": 1.8,
                        "error": "",
                        "legacy_metric_only": "keep-too",
                    },
                ]
            )
            extended_df = pd.DataFrame(columns=paper_df.columns)

            with patch.object(run_full_paper_experiments, "ROOT", fake_root), patch.object(
                run_full_paper_experiments,
                "_should_sync_latest_results_copy",
                return_value=True,
            ), patch.object(
                run_full_paper_experiments,
                "add_rank_column",
                side_effect=lambda df, metric_col, ascending: df.assign(rank=range(1, len(df) + 1)),
                create=True,
            ):
                paths = run_full_paper_experiments._resolve_output_paths(
                    protocol={},
                    output_dir=output_dir,
                )
                saved = run_full_paper_experiments._save_run_results(
                    paper_results_df=paper_df,
                    extended_results_df=extended_df,
                    output_paths=paths,
                    datasets=["Dataset1"],
                )

            run_dataset_df = pd.read_csv(saved["Dataset1"], dtype=str, keep_default_na=False)
            compat_dataset_df = pd.read_csv(
                fake_root / "outputs" / "experiment_results" / "dataset1_results.csv",
                dtype=str,
                keep_default_na=False,
            )

        self.assertEqual(list(run_dataset_df.columns), list(compat_dataset_df.columns))
        self.assertEqual(len(run_dataset_df), len(compat_dataset_df))
        self.assertEqual(
            run_dataset_df["method"].value_counts().sort_index().to_dict(),
            compat_dataset_df["method"].value_counts().sort_index().to_dict(),
        )
        self.assertEqual(run_dataset_df.to_dict("records"), compat_dataset_df.to_dict("records"))


    def test_full_paper_runner_reserves_explicit_output_dir_without_new_timestamp(self):
        output_dir = Path("outputs/runs/20990101_010203")
        protocol = {"outputs": {"paper_results_csv": "paper_results.csv"}}

        with tempfile.TemporaryDirectory() as tmp:
            fake_root = Path(tmp)

            with patch.object(
                run_full_paper_experiments,
                "ROOT",
                fake_root,
            ):
                paths = _resolve_output_paths(
                    protocol=protocol,
                    output_dir=output_dir,
                )
                expected_run_dir = fake_root / output_dir

                self.assertEqual("20990101_010203", paths["run_id"])
                self.assertEqual(expected_run_dir, paths["run_dir"])
                self.assertEqual(
                    expected_run_dir / "results",
                    paths["results_dir"],
                )
                self.assertEqual(
                    expected_run_dir / "results" / "paper_results.csv",
                    paths["paper_csv"],
                )

                with self.assertRaisesRegex(
                    FileExistsError,
                    "will not be reused",
                ):
                    _resolve_output_paths(
                        protocol=protocol,
                        output_dir=output_dir,
                    )



if __name__ == "__main__":
    unittest.main()
