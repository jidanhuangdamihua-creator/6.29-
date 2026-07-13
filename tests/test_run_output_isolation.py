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

        self.assertEqual(len(tasks), 12)
        task_dirs = {task.expected_result_path.parent.parent for task in tasks if task.expected_result_path}
        self.assertEqual(len(task_dirs), 12)
        self.assertEqual({path.parent for path in task_dirs}, {run_root})

    def test_aggregator_cli_requires_an_explicit_run_root(self) -> None:
        with patch("sys.argv", ["aggregate_d1_d6_results.py"]):
            with self.assertRaises(SystemExit):
                aggregate_d1_d6_results._parse_args()

    def test_unified_runner_dry_run_is_directly_executable(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
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

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("[D1-without]", completed.stdout)
        self.assertIn("[D1-with]", completed.stdout)
