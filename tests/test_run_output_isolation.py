from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import aggregate_d1_d6_results
from scripts.run_unified_d1_d6 import build_tasks
from src.utils.run_utils import create_run_dir, reserve_new_output_dir


class RunOutputIsolationTest(unittest.TestCase):
    def test_two_consecutive_run_directories_are_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = create_run_dir(root, "formal")
            second = create_run_dir(root, "formal")

            self.assertNotEqual(first, second)
            self.assertEqual(first.parent, root / "outputs" / "runs")
            self.assertEqual(second.parent, root / "outputs" / "runs")


    def test_existing_output_directory_is_not_reused_or_modified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "existing-run"
            output_dir.mkdir()
            sentinel = output_dir / "sentinel.txt"
            sentinel.write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                reserve_new_output_dir(output_dir)

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_unified_tasks_share_one_parent_with_distinct_dataset_mode_directories(self) -> None:
        run_root = Path("/tmp") / "formal-run"
        tasks = build_tasks(
            only=["d1", "d2", "d3", "d4", "d5", "d6"],
            smoke=False,
            run_dir=run_root,
        )

        self.assertEqual(len(tasks), 60)
        result_paths = {task.expected_result_path for task in tasks if task.expected_result_path}
        self.assertEqual(len(result_paths), 60)
        mode_dirs = {path.parents[3] for path in result_paths}
        self.assertEqual(len(mode_dirs), 12)
        self.assertEqual({path.parent for path in mode_dirs}, {run_root})

    def test_aggregator_cli_requires_an_explicit_run_root(self) -> None:
        with patch("sys.argv", ["aggregate_d1_d6_results.py"]):
            with self.assertRaises(SystemExit):
                aggregate_d1_d6_results._parse_args()

    def test_unified_runner_dry_run_fails_closed_on_old_sealed_schema(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        runs_root = project_root / "outputs" / "runs"
        before = set(runs_root.iterdir()) if runs_root.is_dir() else set()
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/run_unified_d1_d6.py",
                "--only",
                "d1",
                "--dry-run",
            ],
            cwd=project_root,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertNotEqual(completed.returncode, 0)
        combined = completed.stdout + completed.stderr
        self.assertRegex(combined, "PREDICTOR_SCHEMA_MISMATCH|KNN_SCHEMA_MISMATCH")
        after = set(runs_root.iterdir()) if runs_root.is_dir() else set()
        self.assertEqual(before, after)
