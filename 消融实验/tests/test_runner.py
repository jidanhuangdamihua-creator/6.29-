"""Runner gates, dry-run plans, and resume handling.

Uses temporary directories, injected adaptation/metrics/IO, and a fake
workspace flag. Does not read sealed parquet or call model.fit.
"""

from __future__ import annotations

import json
import re
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from sita_ablation.adaptation import TargetWindows
from sita_ablation.arms import BLOCKED_GAP_TCN_SOURCE_FIT, expand_runs, run_directory
from sita_ablation.data import ContractError
from sita_ablation.formal_context import PreparedCondition, load_plan_frames, prepare_conditions
from sita_ablation.io import PHASE_AUDIT_CATEGORIES, phase_audit_path
from sita_ablation.runner import (
    PLANNED,
    PROTOCOL_VERSION,
    CellContext,
    ExecutionResult,
    PreflightResult,
    dispatch_cells,
    dry_run,
    execute,
    main,
)
from sita_ablation.source_selection import STATUS_SKIPPED
from sita_ablation.stage_a import assess_tcn_source_fit, frozen_config_sha256

_PACKAGE = Path(__file__).resolve().parents[1]
_REPO = _PACKAGE.parent
_RUNNER = _PACKAGE / "sita_ablation" / "runner.py"
_CELL_FILES = {
    "FAILED.json",
    "DONE.json",
    "selection.json",
    "cell_record.json",
    "predictions.parquet",
    "phase_1_audit.json",
    "phase_2_audit.json",
    "phase_3_audit.json",
    "phase_4_audit.json",
    "phase_5_audit.json",
}
_FORMAL_ARGV = ["--phase", "1", "--arms", "F4", "--workers", "1", "--only-cell", "D1:1:42"]
_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}\+00:00$")
_CELL_ID_RE = re.compile(
    r"^(?:F1|F2|F3|F4|P1|P2|P3|SITA_TCN|TCN_REVIN):"
    r"(?:main|alt_lr|random3|bottom3|k5|k10|ell15):"
    r"(?:D1|D4|D5|D6):h(?:1|3|5):seed(?:42|43|44|45|46)$"
)


def _spec(arm, *, phase=1, seed=42, dataset="D1", horizon=1, tag="main"):
    return next(
        item
        for item in expand_runs(phase)
        if item.arm == arm
        and item.protocol_tag == tag
        and item.dataset == dataset
        and item.horizon == horizon
        and item.seed == seed
    )


def _clock():
    state = {"n": 0}
    start = datetime(2026, 10, 2, 1, 2, 3, 123456, tzinfo=timezone.utc)

    def clock():
        value = start + timedelta(seconds=state["n"])
        state["n"] += 1
        return value

    return clock


def _raise(name):
    def _blocked(*_args, **_kwargs):
        raise AssertionError(name)

    return _blocked


def _context(root: Path, **overrides) -> CellContext:
    values = {
        "clock": _clock(),
        "inspect": _raise("inspect"),
        "write_run": _raise("write_run"),
        "write_failed": _raise("write_failed"),
        "logs_root": root / "logs",
        "adapt": _raise("adapt"),
        "evaluate": _raise("evaluate"),
        "select": _raise("select"),
    }
    values.update(overrides)
    return CellContext(**values)


def _files(root: Path) -> tuple[str, ...]:
    if not root.exists():
        return ()
    return tuple(sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()))


def _cell_files(package: Path = _PACKAGE) -> tuple[str, ...]:
    found = []
    for folder in ("runs", "audit", "logs", "cache"):
        root = package / folder
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.name in _CELL_FILES:
                found.append(str(path.relative_to(package)))
    return tuple(sorted(found))


def _accepted_result(audit: Path) -> ExecutionResult:
    gate = PreflightResult(
        accepted=True,
        workspace_dirty=False,
        open_1="CLOSED",
        open_2="CLOSED",
        gap_tcn_source_fit="CLOSED",
        frozen_config_sha256="a" * 64,
        protocol_version=PROTOCOL_VERSION,
        schema_version=1,
        reasons=(),
    )
    return ExecutionResult(gate, (), (), audit)


class RunnerGateTests(unittest.TestCase):
    def test_dirty_workspace_refuses_and_writes_nothing(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            context = _context(root)
            with patch("sita_ablation.runner.expand_runs", side_effect=AssertionError("plan")), \
                 patch("sita_ablation.data.load_condition_frames", side_effect=AssertionError("sealed")), \
                 patch("pandas.read_parquet", side_effect=AssertionError("parquet")), \
                 patch("sita_ablation.adaptation.adapt_target", side_effect=AssertionError("adapt")), \
                 patch("sita_ablation.runner.stage_a_mod.frozen_config_sha256", side_effect=AssertionError("hash")):
                result = execute(
                    1,
                    workers=1,
                    runs_root=root / "runs",
                    arms=("F1",),
                    only_cell="D1:1:42",
                    workspace_dirty=True,
                    context=context,
                )
            self.assertFalse(result.preflight.accepted)
            self.assertTrue(result.preflight.workspace_dirty)
            self.assertEqual(result.preflight.reasons, ("workspace_dirty",))
            self.assertEqual(result.outcomes, ())
            self.assertIsNone(result.audit_path)
            self.assertEqual(result.plan, ())
            self.assertEqual(_files(root), ())

    def test_only_horizon_splits_phase5_into_three_disjoint_groups(self):
        shards = []
        for horizon in (1, 3, 5):
            preview = dry_run(5, workers=1, only_horizon=horizon, workspace_dirty=False)
            self.assertTrue(preview.preflight.accepted)
            self.assertEqual(len(preview.plan), 20)
            self.assertTrue(all(cell.spec.horizon == horizon and cell.spec.protocol_tag == "k5" for cell in preview.plan))
            shards.append({cell.cell_id for cell in preview.plan})
        self.assertEqual(len(set().union(*shards)), 60)
        self.assertTrue(shards[0].isdisjoint(shards[1]) and shards[1].isdisjoint(shards[2]))

    def test_closed_gaps_do_not_refuse_a_clean_preflight(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            context = _context(root)
            with patch("sita_ablation.data.load_condition_frames", side_effect=AssertionError("sealed")), \
                 patch("pandas.read_parquet", side_effect=AssertionError("parquet")), \
                 patch("sita_ablation.adaptation.adapt_target", side_effect=AssertionError("adapt")):
                preview = dry_run(
                    1,
                    workers=1,
                    arms=("F1",),
                    only_cell="D1:1:42",
                    workspace_dirty=False,
                )
                with self.assertRaises(ContractError):
                    execute(
                        1,
                        workers=1,
                        runs_root=root / "runs",
                        arms=("F1",),
                        only_cell="D1:1:42",
                        workspace_dirty=False,
                        context=context,
                    )
            self.assertTrue(preview.preflight.accepted)
            self.assertFalse(preview.preflight.workspace_dirty)
            self.assertEqual(preview.preflight.open_1, "CLOSED")
            self.assertEqual(preview.preflight.open_2, "CLOSED")
            self.assertEqual(preview.preflight.gap_tcn_source_fit, "CLOSED")
            self.assertNotIn("OPEN-1", preview.preflight.reasons)
            self.assertNotIn("OPEN-2", preview.preflight.reasons)
            self.assertNotIn("GAP-TCN-SOURCE-FIT", preview.preflight.reasons)
            self.assertEqual(_files(root), ())

    def test_closed_tcn_contract_plans_sita_tcn_and_leaves_tcn_revin_sourceless(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            with patch("sita_ablation.data.load_condition_frames", side_effect=AssertionError("sealed")), \
                 patch("pandas.read_parquet", side_effect=AssertionError("parquet")), \
                 patch("sita_ablation.adaptation.adapt_target", side_effect=AssertionError("adapt")), \
                 patch("sita_ablation.metrics.evaluate_test", side_effect=AssertionError("metrics")):
                preview = dry_run(3, workers=1, only_cell="D1:1:42", workspace_dirty=False)
                with self.assertRaises(ContractError):
                    execute(
                        3,
                        workers=1,
                        runs_root=root / "runs",
                        only_cell="D1:1:42",
                        workspace_dirty=False,
                        context=_context(root),
                    )
            by_arm = {cell.spec.arm: cell for cell in preview.plan}
            self.assertEqual(set(by_arm), {"SITA_TCN", "TCN_REVIN"})
            self.assertEqual(by_arm["SITA_TCN"].status, PLANNED)
            self.assertEqual(by_arm["TCN_REVIN"].status, PLANNED)
            self.assertIsNone(by_arm["TCN_REVIN"].stage_a_key_sha256)
            self.assertEqual(by_arm["TCN_REVIN"].spec.selection_rule, "none")
            self.assertEqual(len(by_arm["SITA_TCN"].stage_a_key_sha256), 64)
            self.assertNotIn("GAP-TCN-SOURCE-FIT", preview.preflight.reasons)
            self.assertEqual(_files(root), ())
            source = _RUNNER.read_text(encoding="utf-8")
            self.assertNotIn("model.fit", source)
            self.assertNotIn("load_condition_frames", source)

    def test_frozen_config_hash_mismatch_refuses_start(self):
        real_hash = frozen_config_sha256()
        with TemporaryDirectory() as temp:
            root = Path(temp)
            with patch("sita_ablation.runner.stage_a_mod.frozen_config_sha256", return_value="0" * 64), \
                 patch("sita_ablation.data.load_condition_frames", side_effect=AssertionError("sealed")), \
                 patch("pandas.read_parquet", side_effect=AssertionError("parquet")):
                result = execute(
                    1,
                    workers=1,
                    runs_root=root / "runs",
                    arms=("F1",),
                    only_cell="D1:1:42",
                    workspace_dirty=False,
                    context=_context(root),
                )
            self.assertFalse(result.preflight.accepted)
            self.assertIn("frozen_config_sha256", result.preflight.reasons)
            self.assertEqual(result.preflight.frozen_config_sha256, real_hash)
            self.assertNotEqual(real_hash, "0" * 64)
            self.assertEqual(result.outcomes, ())
            self.assertEqual(_files(root), ())

    def test_dry_run_plans_without_fit_or_run_files(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            with patch("sita_ablation.data.load_condition_frames", side_effect=AssertionError("sealed")) as sealed, \
                 patch("pandas.read_parquet", side_effect=AssertionError("parquet")) as parquet, \
                 patch("sita_ablation.adaptation.adapt_target", side_effect=AssertionError("adapt")) as adapt, \
                 patch("sita_ablation.metrics.evaluate_test", side_effect=AssertionError("metrics")) as metrics, \
                 patch("sita_ablation.io.write_run_atomically", side_effect=AssertionError("write")) as write_run, \
                 patch("sita_ablation.io.write_cell_failed", side_effect=AssertionError("failed")) as write_failed, \
                 patch("sita_ablation.io.write_phase_audit", side_effect=AssertionError("audit")) as audit:
                result = dry_run(
                    1,
                    workers=1,
                    arms=("F1", "F4"),
                    only_cell="D1:1:42",
                    workspace_dirty=False,
                )
            sealed.assert_not_called()
            parquet.assert_not_called()
            adapt.assert_not_called()
            metrics.assert_not_called()
            write_run.assert_not_called()
            write_failed.assert_not_called()
            audit.assert_not_called()
            self.assertEqual(result.preflight.schema_version, 1)
            self.assertEqual(result.preflight.protocol_version, PROTOCOL_VERSION)
            self.assertEqual(result.preflight.frozen_config_sha256, frozen_config_sha256())
            self.assertTrue(result.preflight.accepted)
            self.assertEqual(result.preflight.gap_tcn_source_fit, "CLOSED")
            self.assertNotIn("GAP-TCN-SOURCE-FIT", result.preflight.reasons)
            by_arm = {cell.spec.arm: cell for cell in result.plan}
            self.assertEqual(by_arm["F1"].status, PLANNED)
            self.assertIsNone(by_arm["F1"].stage_a_key_sha256)
            self.assertEqual(by_arm["F1"].directory, "消融实验/runs/F1/main/D1_h1/seed42")
            self.assertEqual(by_arm["F4"].status, PLANNED)
            self.assertEqual(len(by_arm["F4"].stage_a_key_sha256), 64)
            self.assertIn("/F4/main/", by_arm["F4"].directory)
            source = _RUNNER.read_text(encoding="utf-8")
            for token in ("model.fit", "load_condition_frames", "import src", "DONE.json", "predictions.parquet"):
                self.assertNotIn(token, source)
            self.assertEqual(_files(root), ())


class RunnerScheduleTests(unittest.TestCase):
    def test_matching_done_is_skipped(self):
        spec = _spec("F1")
        with TemporaryDirectory() as temp:
            root = Path(temp)
            directory = run_directory(spec, root / "runs")
            directory.mkdir(parents=True)
            marker = directory / "DONE.json"
            marker.write_bytes(b"stable-done")
            extra = directory / "predictions.parquet"
            extra.write_bytes(b"stable-pred")
            seen = {"adapt": 0}

            def adapt(_spec):
                seen["adapt"] += 1
                raise AssertionError("adapt")

            context = _context(
                root,
                inspect=lambda _directory: "SKIP",
                adapt=adapt,
                write_run=_raise("write_run"),
                write_failed=_raise("write_failed"),
            )
            result = dispatch_cells(
                (spec,),
                phase=1,
                runs_root=root / "runs",
                context=context,
                audit_path=root / "audit" / "phase_1_audit.json",
            )
            self.assertEqual(result.outcomes[0].status, "SKIP")
            self.assertEqual(result.outcomes[0].attempts, ())
            self.assertEqual(seen["adapt"], 0)
            self.assertEqual(marker.read_bytes(), b"stable-done")
            self.assertEqual(extra.read_bytes(), b"stable-pred")

    def test_done_hash_mismatch_keeps_the_directory(self):
        spec = _spec("F4")
        with TemporaryDirectory() as temp:
            root = Path(temp)
            directory = run_directory(spec, root / "runs")
            directory.mkdir(parents=True)
            done = directory / "DONE.json"
            predictions = directory / "predictions.parquet"
            done.write_bytes(b"mismatched-done")
            predictions.write_bytes(b"mismatched-pred")
            before = _files(directory)

            def inspect(path: Path) -> str:
                self.assertEqual(path, directory)
                return "HOLD_HASH_MISMATCH"

            context = _context(
                root,
                inspect=inspect,
                write_run=_raise("write_run"),
                write_failed=_raise("write_failed"),
                adapt=_raise("adapt"),
            )
            result = dispatch_cells(
                (spec,),
                phase=1,
                runs_root=root / "runs",
                context=context,
                audit_path=root / "audit" / "phase_1_audit.json",
            )
            self.assertEqual(result.outcomes[0].status, "HOLD_HASH_MISMATCH")
            self.assertEqual(done.read_bytes(), b"mismatched-done")
            self.assertEqual(predictions.read_bytes(), b"mismatched-pred")
            self.assertEqual(_files(directory), before)
            self.assertFalse((directory / "FAILED.json").exists())

    def test_failure_retries_once_then_holds(self):
        spec = _spec("F1")
        with TemporaryDirectory() as temp:
            root = Path(temp)
            seeds = []
            failed_calls = []

            def adapt(item):
                seeds.append(item.seed)
                raise RuntimeError("synthetic fit failure")

            def write_failed(directory: Path, attempt_count: int) -> None:
                directory.mkdir(parents=True, exist_ok=True)
                body = {"attempt_count": attempt_count}
                self.assertEqual(set(body), {"attempt_count"})
                (directory / "FAILED.json").write_text(json.dumps(body, sort_keys=True), encoding="utf-8")
                failed_calls.append(attempt_count)

            context = _context(
                root,
                inspect=lambda _directory: "RUN",
                adapt=adapt,
                write_failed=write_failed,
                write_run=_raise("write_run"),
                evaluate=_raise("evaluate"),
            )
            result = dispatch_cells(
                (spec,),
                phase=1,
                runs_root=root / "runs",
                context=context,
                audit_path=root / "audit" / "phase_1_audit.json",
            )
            outcome = result.outcomes[0]
            self.assertEqual(outcome.status, "HOLD_FAILED")
            self.assertEqual(seeds, [42, 42])
            self.assertEqual(failed_calls, [1, 2])
            self.assertEqual(len(outcome.attempts), 2)
            for stamp in outcome.attempts:
                self.assertRegex(stamp.started_at, _TIMESTAMP_RE)
                self.assertRegex(stamp.finished_at, _TIMESTAMP_RE)
                self.assertEqual(stamp.seed, 42)
                self.assertLessEqual(stamp.started_at, stamp.finished_at)
            self.assertLessEqual(outcome.attempts[0].finished_at, outcome.attempts[1].started_at)
            directory = run_directory(spec, root / "runs")
            body = json.loads((directory / "FAILED.json").read_text(encoding="utf-8"))
            self.assertEqual(body, {"attempt_count": 2})
            self.assertFalse((directory / "DONE.json").exists())
            log_text = next((root / "logs").glob("*.log")).read_text(encoding="utf-8")
            self.assertEqual(log_text.count("RuntimeError: synthetic fit failure"), 2)
            held = _context(
                root,
                inspect=lambda _directory: "HOLD_FAILED",
                adapt=_raise("adapt"),
                write_failed=_raise("write_failed"),
                write_run=_raise("write_run"),
            )
            again = dispatch_cells(
                (spec,),
                phase=1,
                runs_root=root / "runs",
                context=held,
                audit_path=root / "audit" / "phase_1_audit.json",
            )
            self.assertEqual(again.outcomes[0].status, "HOLD_FAILED")
            self.assertEqual(again.outcomes[0].attempts, ())
            self.assertEqual(json.loads((directory / "FAILED.json").read_text(encoding="utf-8")), {"attempt_count": 2})

    def test_insufficient_candidates_keep_only_the_prescribed_files(self):
        spec = _spec("F3")
        with TemporaryDirectory() as temp:
            root = Path(temp)
            seen = {"adapt": 0, "evaluate": 0, "failed": 0}

            def write_run(directory: Path, payload) -> None:
                self.assertIsNone(payload.adaptation)
                self.assertIsNone(payload.metrics)
                self.assertEqual(payload.selection.status, STATUS_SKIPPED)
                self.assertEqual(payload.selection.selected, ())
                self.assertEqual(payload.spec.seed, spec.seed)
                self.assertIsNotNone(payload.started_at.tzinfo)
                self.assertIsNotNone(payload.finished_at.tzinfo)
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "selection.json").write_text("{}\n", encoding="utf-8")
                (directory / "DONE.json").write_text("{}\n", encoding="utf-8")

            context = _context(
                root,
                inspect=lambda _directory: "RUN",
                select=lambda _spec: SimpleNamespace(status=STATUS_SKIPPED, selected=()),
                write_run=write_run,
                adapt=lambda _spec: seen.__setitem__("adapt", seen["adapt"] + 1),
                evaluate=lambda _result: seen.__setitem__("evaluate", seen["evaluate"] + 1),
                write_failed=lambda *_args: seen.__setitem__("failed", seen["failed"] + 1),
            )
            result = dispatch_cells(
                (spec,),
                phase=1,
                runs_root=root / "runs",
                context=context,
                audit_path=root / "audit" / "phase_1_audit.json",
            )
            directory = run_directory(spec, root / "runs")
            self.assertEqual(result.outcomes[0].status, STATUS_SKIPPED)
            self.assertEqual(seen, {"adapt": 0, "evaluate": 0, "failed": 0})
            self.assertEqual(_files(directory), ("DONE.json", "selection.json"))
            self.assertRegex(result.outcomes[0].attempts[0].started_at, _TIMESTAMP_RE)
            self.assertRegex(result.outcomes[0].attempts[0].finished_at, _TIMESTAMP_RE)

    def test_phase_audit_statuses_match_schema_v1(self):
        skipped = _spec("F1", seed=42)
        missing = _spec("F1", seed=45)
        mismatched = _spec("F4", seed=42)
        failed = _spec("F2", seed=42)
        insufficient = _spec("F3", seed=42)
        blocked = _spec("SITA_TCN", phase=3, seed=42)
        incomplete = assess_tcn_source_fit({"loss": None})
        with TemporaryDirectory() as temp, patch(
            "sita_ablation.stage_a.load_tcn_source_fit", return_value=incomplete
        ):
            root = Path(temp)
            mismatch_dir = run_directory(mismatched, root / "runs")
            mismatch_dir.mkdir(parents=True)
            (mismatch_dir / "DONE.json").write_bytes(b"keep")

            def inspect(directory: Path) -> str:
                if directory == run_directory(mismatched, root / "runs"):
                    return "HOLD_HASH_MISMATCH"
                if directory == run_directory(skipped, root / "runs"):
                    return "SKIP"
                return "RUN"

            def adapt(spec):
                if spec.arm == "F2":
                    raise RuntimeError("synthetic fit failure")
                raise AssertionError(spec.arm)

            def write_failed(directory: Path, attempt_count: int) -> None:
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "FAILED.json").write_text(
                    json.dumps({"attempt_count": attempt_count}),
                    encoding="utf-8",
                )

            def write_run(directory: Path, payload) -> None:
                self.assertEqual(payload.selection.status, STATUS_SKIPPED)
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "selection.json").write_text("{}\n", encoding="utf-8")
                (directory / "DONE.json").write_text("{}\n", encoding="utf-8")

            context = _context(
                root,
                inspect=inspect,
                adapt=adapt,
                select=lambda spec: SimpleNamespace(status=STATUS_SKIPPED, selected=()) if spec.arm == "F3" else None,
                write_failed=write_failed,
                write_run=write_run,
                evaluate=_raise("evaluate"),
            )
            audit_path = root / "audit" / "phase_1_audit.json"
            result = dispatch_cells(
                (skipped, mismatched, failed, insufficient, blocked),
                phase=1,
                runs_root=root / "runs",
                context=context,
                audit_path=audit_path,
                planned=(skipped, mismatched, failed, insufficient, blocked, missing),
            )
            self.assertEqual(
                {item.status for item in result.outcomes},
                {"SKIP", "HOLD_HASH_MISMATCH", "HOLD_FAILED", STATUS_SKIPPED, BLOCKED_GAP_TCN_SOURCE_FIT},
            )
            body = json.loads(audit_path.read_text(encoding="utf-8"))
            self.assertEqual(set(body), set(PHASE_AUDIT_CATEGORIES))
            self.assertEqual(audit_path.name, "phase_1_audit.json")
            listed = [cell for name in PHASE_AUDIT_CATEGORIES for cell in body[name]]
            for name in PHASE_AUDIT_CATEGORIES:
                self.assertEqual(body[name], sorted(body[name]))
            self.assertEqual(len(listed), len(set(listed)))
            for cell in listed:
                self.assertRegex(cell, _CELL_ID_RE)
            self.assertEqual(body["missing"], ["F1:main:D1:h1:seed45"])
            self.assertEqual(body["hash_mismatch"], ["F4:main:D1:h1:seed42"])
            self.assertEqual(body["failed"], ["F2:main:D1:h1:seed42"])
            self.assertEqual(body["insufficient_candidates"], ["F3:main:D1:h1:seed42"])
            self.assertNotIn("SITA_TCN:main:D1:h1:seed42", listed)
            self.assertNotIn("F1:main:D1:h1:seed42", listed)
            self.assertNotIn("planned", body)
            self.assertNotIn("blocked", body)
            self.assertEqual((mismatch_dir / "DONE.json").read_bytes(), b"keep")


class FormalCliAssemblyTests(unittest.TestCase):
    def test_dry_run_does_not_load_context_or_create_an_audit_file(self):
        before = _cell_files()
        with patch("sita_ablation.runner.probe_workspace_dirty", return_value=False), \
             patch("sita_ablation.runner.assemble_formal_context", side_effect=AssertionError("context")) as loader, \
             patch("sita_ablation.runner.execute", side_effect=AssertionError("execute")) as execute_fn, \
             patch("sita_ablation.formal_context.load_condition_frames", side_effect=AssertionError("sealed")):
            code = main([*_FORMAL_ARGV, "--dry-run"])
        self.assertEqual(code, 0)
        loader.assert_not_called()
        execute_fn.assert_not_called()
        self.assertEqual(_cell_files(), before)

    def test_failed_preflight_does_not_load_context(self):
        before = _cell_files()
        with patch("sita_ablation.runner.probe_workspace_dirty", return_value=True), \
             patch("sita_ablation.runner.assemble_formal_context", side_effect=AssertionError("context")) as loader, \
             patch("sita_ablation.runner.execute", side_effect=AssertionError("execute")) as execute_fn, \
             patch("sita_ablation.formal_context.load_condition_frames", side_effect=AssertionError("sealed")):
            code = main(_FORMAL_ARGV)
        self.assertEqual(code, 2)
        loader.assert_not_called()
        execute_fn.assert_not_called()
        self.assertEqual(_cell_files(), before)

    def test_formal_cli_passes_context_and_audit_path_for_the_only_cell(self):
        sentinel = _context(_PACKAGE)
        loaded = []
        seen = {}

        def loader(specs):
            loaded.append(tuple(specs))
            return sentinel

        def fake_execute(*_args, **kwargs):
            seen.update(kwargs)
            return _accepted_result(kwargs["audit_path"])

        before = _cell_files()
        with patch("sita_ablation.runner.probe_workspace_dirty", return_value=False), \
             patch("sita_ablation.runner.assemble_formal_context", side_effect=loader), \
             patch("sita_ablation.runner.execute", side_effect=fake_execute), \
             patch("sita_ablation.formal_context.load_condition_frames", side_effect=AssertionError("sealed")), \
             patch("sita_ablation.adaptation.adapt_target", side_effect=AssertionError("adapt")):
            code = main(_FORMAL_ARGV)
        self.assertEqual(code, 0)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(
            [(spec.arm, spec.protocol_tag, spec.dataset, spec.horizon, spec.seed) for spec in loaded[0]],
            [("F4", "main", "D1", 1, 42)],
        )
        self.assertIs(seen["context"], sentinel)
        self.assertEqual(seen["audit_path"], _REPO / phase_audit_path(1))
        self.assertEqual(seen["only_cell"], "D1:1:42")
        self.assertIsNotNone(seen["context"])
        self.assertIsNotNone(seen["audit_path"])
        self.assertEqual(_cell_files(), before)

    def test_only_cell_loads_that_dataset_and_horizon(self):
        calls = []

        def fake_load(dataset):
            calls.append(dataset)
            return dataset

        with patch("sita_ablation.formal_context.load_condition_frames", side_effect=fake_load):
            one = load_plan_frames((_spec("F4"),))
            calls.clear()
            several = load_plan_frames((_spec("F4"), _spec("F4", horizon=3), _spec("F4", dataset="D4")))
        self.assertEqual(one, {"D1": "D1"})
        self.assertEqual(calls, ["D1", "D4"])
        self.assertEqual(several, {"D1": "D1", "D4": "D4"})

        frame = SimpleNamespace(
            model_features=("sales", "on_promo"),
            key_fields=("series_id",),
            target_key=("target",),
            target_test=SimpleNamespace(iso=lambda: ("2020-01-01", "2020-06-28")),
        )
        built = []

        def fake_windows(_rows, **kwargs):
            built.append((kwargs["horizon"], kwargs["partition"]))
            return SimpleNamespace()

        def fake_test(_rows, **kwargs):
            built.append((kwargs["horizon"], "test"))
            return SimpleNamespace(y_original=np.zeros(180, dtype=np.float64))

        with patch("sita_ablation.formal_context.partition_rows", return_value=frame), \
             patch("sita_ablation.formal_context.fit_target_model_scaler", return_value="scaler"), \
             patch("sita_ablation.formal_context.transform_model_frame", side_effect=lambda _scaler, rows: rows), \
             patch("sita_ablation.formal_context.observed_through_test", return_value=frame), \
             patch("sita_ablation.formal_context.build_target_partition_windows", side_effect=fake_windows), \
             patch("sita_ablation.formal_context.build_test_windows", side_effect=fake_test):
            prepared = prepare_conditions((_spec("F4"), _spec("F4")), {"D1": frame})
        self.assertEqual(set(prepared), {("D1", 1)})
        self.assertEqual(built, [(1, "train"), (1, "validation"), (1, "test")])

    def test_context_construction_failure_creates_no_cell_files(self):
        before = _cell_files()

        def loader(_specs):
            raise ContractError("formal context could not be assembled: sealed table missing")

        with patch("sita_ablation.runner.probe_workspace_dirty", return_value=False), \
             patch("sita_ablation.runner.assemble_formal_context", side_effect=loader), \
             patch("sita_ablation.runner.execute", side_effect=AssertionError("execute")) as execute_fn, \
             patch("sita_ablation.formal_context.load_condition_frames", side_effect=AssertionError("sealed")):
            with self.assertRaises(ContractError) as caught:
                main(_FORMAL_ARGV)
        self.assertIn("formal context could not be assembled", str(caught.exception))
        self.assertNotIn("sealed tables are not read", str(caught.exception))
        execute_fn.assert_not_called()
        self.assertEqual(_cell_files(), before)

    def test_assembled_context_keeps_the_loaded_frames_and_candidate_rule(self):
        spec = _spec("F4")
        frame = SimpleNamespace(dataset="D1", model_features=("sales",))
        windows = TargetWindows(train=SimpleNamespace(), validation=SimpleNamespace(), test=SimpleNamespace())
        condition = PreparedCondition(windows=windows, target_scaler="scaler", y_true=np.zeros(180))
        template = SimpleNamespace(tensors={})
        selection = SimpleNamespace(status="ok")
        with patch("sita_ablation.formal_context.load_plan_frames", return_value={"D1": frame}), \
             patch("sita_ablation.formal_context.prepare_conditions", return_value={("D1", 1): condition}), \
             patch("sita_ablation.formal_context._templates_for", return_value={("cnn", 1, True, 42): template}), \
             patch("sita_ablation.formal_context.load_or_train_stage_a", side_effect=AssertionError("stage-a")), \
             patch("sita_ablation.formal_context.adapt_target", side_effect=AssertionError("adapt")), \
             patch("sita_ablation.formal_context.rank_candidates", return_value=selection) as rank:
            from sita_ablation.formal_context import assemble_formal_context

            context = assemble_formal_context((spec,))
            chosen = context.select(spec)
        rank.assert_called_once_with(
            "D1",
            frame,
            comparison_days=30,
            rule="top",
            k=3,
            seed=42,
        )
        self.assertIs(chosen, selection)
        self.assertIs(context.frames["D1"], frame)
        self.assertIs(context.conditions[("D1", 1)], condition)
        self.assertIs(context.windows, windows)
        self.assertEqual(context.target_scaler, "scaler")
        self.assertEqual(context.y_true.shape, (180,))
        self.assertIs(context.init_template, template)
        self.assertIsNone(context.stage_a)


if __name__ == "__main__":
    unittest.main()
