"""Synthetic run-directory IO.

Uses TemporaryDirectory and in-memory result objects. Does not read sealed
tables, train a model, or start a run.
"""

from __future__ import annotations

import ast
import hashlib
import json
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from sita_ablation.adaptation import (
    ABSENT,
    COMPLETED,
    DEFINED,
    PRESENT,
    UNDEFINED_ZERO_INIT_NORM,
    UNRESOLVED,
    AdaptationResult,
    ParameterShift,
)
from sita_ablation.arms import expand_runs, run_directory
from sita_ablation.data import ContractError, load_dataset_contract
from sita_ablation.d5_source_history import inactive_reconstruction_summary
from sita_ablation.io import (
    CELL_RECORD_FILE,
    CELL_RECORD_KEYS,
    DONE_FILE,
    FAILED_FILE,
    PAIR_KEYS,
    PHASE_AUDIT_CATEGORIES,
    PREDICTION_PATH,
    PREDICTIONS_FILE,
    RUN_SPEC_KEYS,
    SCHEMA_VERSION,
    SELECTION_FILE,
    SHIFT_GROUP_KEYS,
    SOURCE_INFO_KEYS,
    canonical_json_bytes,
    cell_id,
    config_digest,
    inspect_done,
    log_path,
    phase_audit_path,
    read_cell_record,
    read_predictions,
    require_clean_workspace_for_formal_run,
    write_cell_failed,
    write_phase_audit,
    write_run_atomically,
)
from sita_ablation.metrics import evaluate_test, effective_source_count
from sita_ablation.scaling import SelectionScaler
from sita_ablation.source_selection import (
    STATUS_COMPLETED,
    STATUS_SKIPPED,
    GateCounts,
    SelectedSource,
    SelectionResult,
    inverse_distance_weights,
    top3_overlap_count,
    uniform_weights,
)
from sita_ablation.windows import (
    WindowBatch,
    build_target_partition_windows,
    expected_target_partition_count,
    validate_target_sample_counts,
)
from sita_ablation.adaptation import TargetWindows, _require_windows
from sita_ablation.stage_a import (
    NOT_APPLICABLE,
    StageAKey,
    frozen_config_sha256,
    stage_a_key_sha256,
)

_PACKAGE = Path(__file__).resolve().parents[1]
_IO_PATH = _PACKAGE / "sita_ablation" / "io.py"
_RUNS = _PACKAGE / "runs"
_CACHE = _PACKAGE / "cache"
_STARTED = datetime(2026, 10, 2, 1, 2, 3, 123456, tzinfo=timezone.utc)
_FINISHED = _STARTED + timedelta(seconds=3, microseconds=123456)
_PRECISE = float(np.sqrt(np.float64(2.0)) / np.float64(3.0))


def _tree(root: Path) -> tuple[str, ...] | None:
    if not root.exists():
        return None
    return tuple(sorted(str(path.relative_to(root)) for path in root.rglob("*")))


_RUNS_BEFORE = _tree(_RUNS)
_CACHE_BEFORE = _tree(_CACHE)


def _spec(arm: str, *, revin: bool, horizon: int = 1, phase: int = 1, protocol_tag: str = "main"):
    matches = [
        item
        for item in expand_runs(phase, include_no_revin_variants=True)
        if item.arm == arm
        and item.dataset == "D1"
        and item.horizon == horizon
        and item.seed == 42
        and item.protocol_tag == protocol_tag
        and item.revin is revin
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one {arm} revin={revin} spec, found {len(matches)}")
    return matches[0]


def _dates() -> pd.DatetimeIndex:
    contract = load_dataset_contract("D1")
    return pd.date_range(contract.target_test.start, contract.target_test.end, freq="D")


def _comparison() -> tuple[str, str, tuple[str, ...]]:
    contract = load_dataset_contract("D1")
    start = pd.Timestamp(contract.target_train.start).strftime("%Y-%m-%d")
    end = pd.Timestamp(contract.target_val.end).strftime("%Y-%m-%d")
    dates = tuple(day.strftime("%Y-%m-%d") for day in pd.date_range(start, end, freq="D"))
    return start, end, dates


def _history() -> tuple[str, str]:
    return load_dataset_contract("D1").source_history.iso()


def _stage_key(*, rule: str = "top", trained_k: int = 3, horizon: int = 1) -> StageAKey:
    contract = load_dataset_contract("D1")
    start, end, _dates_ = _comparison()
    history_start, history_end = _history()
    return StageAKey(
        dataset="D1",
        horizon=horizon,
        seed=42,
        backbone="cnn",
        selection_rule=rule,
        trained_k=trained_k,
        comparison_start=start,
        comparison_end=end,
        source_history_start=history_start,
        source_history_end=history_end,
        model_features=tuple(contract.model_features),
        similarity_variables=tuple(contract.similarity_variables),
        scaling_rule="per_similarity_variable_minmax",
        draw_rule="sorted_permutation_v1" if rule == "random3" else None,
        frozen_config_sha256=frozen_config_sha256(),
    )


def _selection(status: str, *, rule: str = "top", requested_k: int = 3) -> SelectionResult:
    contract = load_dataset_contract("D1")
    start, end, dates = _comparison()
    history_start, history_end = _history()
    keys = (("1", "1"), ("1", "2"), ("1", "3"))
    if status == STATUS_SKIPPED:
        return SelectionResult(
            dataset="D1",
            status=STATUS_SKIPPED,
            rule=rule,
            requested_k=requested_k,
            seed=42,
            comparison_days=30,
            comparison_start=start,
            comparison_end=end,
            comparison_dates=dates,
            similarity_variables=tuple(contract.similarity_variables),
            target_key=tuple(contract.target_key),
            selected=(),
            distances=np.empty(0, dtype=np.float64),
            inverse_distance_weights=np.empty(0, dtype=np.float64),
            uniform_weights=np.empty(0, dtype=np.float64),
            selection_scaler=None,
            exclusions=(),
            n_legal=2,
            legal_keys=keys[:2],
            source_history_start=history_start,
            source_history_end=history_end,
            gate_counts=GateCounts(identity=4, source_history=3, comparison_window=2, comparison_variables=2),
            reconstruction=inactive_reconstruction_summary(history_start, history_end),
            scaler_population_keys=(),
        )
    distances = np.array([np.sqrt(2.0), np.sqrt(3.0), np.sqrt(5.0)], dtype=np.float64)
    selected = tuple(
        SelectedSource(source_key=key, rank=rank, distance=float(distances[rank]), tie_group=0)
        for rank, key in enumerate(keys)
    )
    return SelectionResult(
        dataset="D1",
        status=STATUS_COMPLETED,
        rule=rule,
        requested_k=requested_k,
        seed=42,
        comparison_days=30,
        comparison_start=start,
        comparison_end=end,
        comparison_dates=dates,
        similarity_variables=tuple(contract.similarity_variables),
        target_key=tuple(contract.target_key),
        selected=selected,
        distances=distances,
        inverse_distance_weights=inverse_distance_weights(distances),
        uniform_weights=uniform_weights(3),
        selection_scaler=SelectionScaler(
            role="source_selection",
            variables=("sales",),
            minimum=np.array([0.0], dtype=np.float64),
            maximum=np.array([1.0], dtype=np.float64),
            range=np.array([1.0], dtype=np.float64),
        ),
        exclusions=(),
        n_legal=3,
        legal_keys=keys,
        source_history_start=history_start,
        source_history_end=history_end,
        gate_counts=GateCounts(identity=3, source_history=3, comparison_window=3, comparison_variables=3),
        reconstruction=inactive_reconstruction_summary(history_start, history_end),
        scaler_population_keys=keys,
    )


def _shifts(*, revin: bool) -> dict[str, ParameterShift]:
    defined = ParameterShift(absolute=_PRECISE, relative=_PRECISE, status=DEFINED)
    undefined = ParameterShift(absolute=0.0, relative=None, status=UNDEFINED_ZERO_INIT_NORM)
    groups = {"conv1": defined, "conv2": defined, "conv3": defined, "head": defined}
    if revin:
        groups["revin"] = undefined
    return groups


def _adaptation(spec, pred, *, sourced: bool):
    count = None
    samples = expected_target_partition_count(spec.horizon)
    if sourced:
        count = effective_source_count(_selection(STATUS_COMPLETED).inverse_distance_weights)
    epochs = ( _PRECISE,) * 50
    return AdaptationResult(
        status=COMPLETED,
        spec=spec,
        train_loss=epochs,
        val_loss=epochs,
        val_rmse=epochs,
        val_smape=epochs,
        epoch0_val_rmse=_PRECISE,
        validation_rmse=_PRECISE,
        validation_smape=_PRECISE,
        validation_weight_variant="RAW",
        gradient_steps=50,
        raw_shift=_shifts(revin=spec.revin),
        ema_shift=_shifts(revin=spec.revin),
        test_predictions=np.array(pred, dtype=np.float64, copy=True),
        label_date=np.array(_dates(), copy=True),
        unscored_test_dates=(),
        test_weight_variant="EMA",
        test_weight_sha256_before=None,
        test_weight_sha256_after=None,
        signed_error=None,
        alpha_entropy=None,
        effective_source_count=count,
        source_info=PRESENT if sourced else ABSENT,
        source_validation_metric=NOT_APPLICABLE if sourced else None,
        source_validation_value_space=NOT_APPLICABLE if sourced else None,
        trainable_names=(),
        frozen_names=(),
        revin_enabled=spec.revin,
        learning_rate=float(spec.target_learning_rate),
        checkpoint_selection=False,
        n_train=samples,
        n_validation=samples,
    )


def _sales():
    true = np.arange(180, dtype=np.float64)
    pred = true + _PRECISE
    pred[0] = -1.5
    true[1] = 0.0
    pred[1] = 0.0
    return true, pred


def _payload(spec, *, sourced: bool, selection=None, stage_key=None, top3_overlap_count=None):
    true, pred = _sales()
    adaptation = _adaptation(spec, pred, sourced=sourced)
    alpha = None
    if sourced:
        alpha = _selection(STATUS_COMPLETED).inverse_distance_weights
    metrics = evaluate_test(adaptation, true, alpha=alpha)
    adaptation = replace(adaptation, signed_error=metrics.signed_error, alpha_entropy=metrics.alpha_entropy)
    return (
        _payload_from(
            spec,
            adaptation,
            metrics,
            selection=selection,
            stage_key=stage_key,
            top3_overlap_count=top3_overlap_count,
        ),
        metrics,
    )


def _payload_from(spec, adaptation, metrics, *, selection=None, stage_key=None, top3_overlap_count=None):
    from sita_ablation.io import RunPayload

    return RunPayload(
        spec=spec,
        started_at=_STARTED,
        finished_at=_FINISHED,
        adaptation=adaptation,
        metrics=metrics,
        stage_a_key=stage_key,
        selection=selection,
        top3_overlap_count=top3_overlap_count,
    )


def _files(directory: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}


def _target_frame(days: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "store": ["1"] * days,
            "item": ["1"] * days,
            "date": pd.date_range("2016-08-01", periods=days, freq="D"),
            "sales": np.linspace(0.0, 1.0, days),
            "sales_original": np.linspace(0.0, 10.0, days),
        }
    )


def _count_batch(count: int, partition: str) -> WindowBatch:
    labels = np.array([np.datetime64("2016-01-01") + np.timedelta64(index, "D") for index in range(count)])
    zeros = np.zeros
    return WindowBatch(
        x=zeros((count, 10, 1), np.float32),
        x_sales=zeros((count, 10, 1), np.float32),
        y_scaled=zeros((count,), np.float32),
        y_original=zeros((count,), np.float64),
        input_start=labels,
        input_end=labels,
        label_date=labels,
        partition=partition,
    )


def _with_keys(selection: SelectionResult, keys: tuple[tuple[str, str], ...], rule: str) -> SelectionResult:
    distances = np.arange(1, len(keys) + 1, dtype=np.float64)
    selected = tuple(
        SelectedSource(source_key=key, rank=rank, distance=float(distances[rank]), tie_group=0)
        for rank, key in enumerate(keys)
    )
    return replace(
        selection,
        rule=rule,
        selected=selected,
        distances=distances,
        inverse_distance_weights=inverse_distance_weights(distances),
        uniform_weights=uniform_weights(len(keys)),
    )


class RunIoTests(unittest.TestCase):
    def test_io_does_not_train_or_import_src(self):
        source = _IO_PATH.read_text(encoding="utf-8")
        self.assertNotIn("import src", source)
        self.assertNotIn("load_condition_frames", source)
        self.assertNotIn("load_or_train_stage_a", source)
        tree = ast.parse(source)
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
        self.assertFalse(any(name == "src" or name.startswith("src.") for name in modules))
        self.assertNotIn("IO_SCHEMA_GAPS", source)
        self.assertNotIn("source_key_overlap", source)
        self.assertNotIn("rank_candidates", source)
        self.assertNotIn("top3_overlap_count(", source)
        self.assertEqual(SCHEMA_VERSION, 1)
        self.assertEqual(len(RUN_SPEC_KEYS), 13)
        self.assertIn("CELL_RECORD_KEYS", source)

    def test_completed_cell_round_trip_uses_run_directory(self):
        spec = _spec("F1", revin=False)
        bare = _spec("P1", revin=False)
        payload, metrics = _payload(spec, sourced=False)
        sourced, _sourced_metrics = _payload(
            bare,
            sourced=True,
            selection=_selection(STATUS_COMPLETED),
            stage_key=_stage_key(),
        )
        with TemporaryDirectory() as temp:
            root = Path(temp)
            directory = run_directory(spec, root)
            bare_directory = run_directory(bare, root)
            self.assertFalse(str(directory).endswith("revin0"))
            self.assertTrue(str(bare_directory).endswith("revin0"))
            self.assertFalse(str(run_directory(_spec("F3", revin=False), root)).endswith("revin0"))
            record = write_run_atomically(directory, payload)
            sourced_record = write_run_atomically(bare_directory, sourced)
            self.assertEqual(record.status, COMPLETED)
            self.assertEqual(set(_files(directory)), {PREDICTIONS_FILE, CELL_RECORD_FILE, DONE_FILE})
            self.assertEqual(sourced_record.stage_a_key_sha256, stage_a_key_sha256(_stage_key()))
            self.assertIsNone(record.stage_a_key_sha256)
            self.assertEqual(record.elapsed_seconds, (_FINISHED - _STARTED).total_seconds())
            body = read_cell_record(directory)
            self.assertEqual(set(body), set(CELL_RECORD_KEYS))
            self.assertEqual(body["n_train"], 5)
            self.assertEqual(body["n_validation"], 5)
            self.assertEqual(body["n_test"], 180)
            self.assertEqual(body["prediction_path"], PREDICTION_PATH)
            self.assertNotIn("top3_overlap_count", json.dumps(body))
            self.assertEqual(body["pair"]["rmse"], metrics.pair.rmse)
            self.assertEqual(np.float64(body["train_loss"][0]), np.float64(_PRECISE))
            self.assertNotEqual(round(body["train_loss"][0], 6), body["train_loss"][0])
            table = read_predictions(directory)
            pred = np.asarray(table.column("y_pred_original").to_numpy(zero_copy_only=False))
            self.assertEqual(pred.dtype, np.float64)
            self.assertTrue(np.array_equal(pred, metrics.y_pred))
            self.assertEqual(float(pred[0]), -1.5)
            done = json.loads((directory / DONE_FILE).read_text(encoding="utf-8"))
            self.assertEqual(set(done["artifact_sha256"]), {PREDICTIONS_FILE, CELL_RECORD_FILE})
            self.assertEqual(inspect_done(directory), "SKIP")
            self.assertEqual(bare_directory.name, "revin0")

    def test_matching_hash_resumes_without_rewriting(self):
        spec = _spec("F1", revin=False)
        payload, _metrics = _payload(spec, sourced=False)
        with TemporaryDirectory() as temp:
            directory = run_directory(spec, Path(temp))
            write_run_atomically(directory, payload)
            before = _files(directory)
            mtimes = {path.name: path.stat().st_mtime_ns for path in directory.iterdir()}
            self.assertEqual(inspect_done(directory), "SKIP")
            again = write_run_atomically(directory, payload)
            self.assertEqual(again.status, COMPLETED)
            self.assertEqual(_files(directory), before)
            self.assertEqual({path.name: path.stat().st_mtime_ns for path in directory.iterdir()}, mtimes)

    def test_hash_mismatch_keeps_the_directory(self):
        spec = _spec("F1", revin=False)
        payload, _metrics = _payload(spec, sourced=False)
        with TemporaryDirectory() as temp:
            directory = run_directory(spec, Path(temp))
            write_run_atomically(directory, payload)
            prediction = directory / PREDICTIONS_FILE
            prediction.write_bytes(prediction.read_bytes() + b"\x00")
            preserved = _files(directory)
            self.assertEqual(inspect_done(directory), "HOLD_HASH_MISMATCH")
            with self.assertRaises(ContractError) as caught:
                write_run_atomically(directory, payload)
            self.assertIn("HOLD_HASH_MISMATCH", str(caught.exception))
            self.assertEqual(_files(directory), preserved)
            self.assertTrue((directory / DONE_FILE).exists())

    def test_identity_mismatch_is_not_rewritten(self):
        spec = _spec("F1", revin=False)
        payload, _metrics = _payload(spec, sourced=False)
        with TemporaryDirectory() as temp:
            directory = run_directory(spec, Path(temp))
            write_run_atomically(directory, payload)
            cell = directory / CELL_RECORD_FILE
            body = json.loads(cell.read_text(encoding="utf-8"))
            body["signed_error"] = 0.5
            text = json.dumps(body, sort_keys=True, ensure_ascii=False)
            cell.write_text(text, encoding="utf-8")
            done_path = directory / DONE_FILE
            done = json.loads(done_path.read_text(encoding="utf-8"))
            done["artifact_sha256"][CELL_RECORD_FILE] = hashlib.sha256(text.encode("utf-8")).hexdigest()
            done_path.write_text(json.dumps(done, sort_keys=True, ensure_ascii=False), encoding="utf-8")
            preserved = _files(directory)
            self.assertEqual(inspect_done(directory), "HOLD_HASH_MISMATCH")
            with self.assertRaises(ContractError) as caught:
                write_run_atomically(directory, payload)
            self.assertIn("HOLD_HASH_MISMATCH", str(caught.exception))
            self.assertEqual(_files(directory), preserved)

    def test_skipped_cell_writes_only_selection_and_done(self):
        spec = _spec("F4", revin=True)
        from sita_ablation.io import RunPayload

        payload = RunPayload(
            spec=spec,
            started_at=_STARTED,
            finished_at=_FINISHED,
            selection=_selection(STATUS_SKIPPED),
            stage_a_key=_stage_key(),
        )
        with TemporaryDirectory() as temp:
            directory = run_directory(spec, Path(temp))
            record = write_run_atomically(directory, payload)
            self.assertEqual(record.status, STATUS_SKIPPED)
            self.assertEqual({path.name for path in directory.iterdir()}, {SELECTION_FILE, DONE_FILE})
            selection = json.loads((directory / SELECTION_FILE).read_text(encoding="utf-8"))
            self.assertEqual(
                selection,
                {"distances": [], "ranks": [], "selected_keys": [], "status": STATUS_SKIPPED},
            )
            done = json.loads((directory / DONE_FILE).read_text(encoding="utf-8"))
            self.assertEqual(set(done["artifact_sha256"]), {SELECTION_FILE})
            self.assertEqual(done["stage_a_key_sha256"], stage_a_key_sha256(_stage_key()))
            self.assertEqual(inspect_done(directory), "SKIP")
            before = _files(directory)
            self.assertEqual(write_run_atomically(directory, payload).status, STATUS_SKIPPED)
            self.assertEqual(_files(directory), before)
            (directory / SELECTION_FILE).write_text("{}\n", encoding="utf-8")
            preserved = _files(directory)
            self.assertEqual(inspect_done(directory), "HOLD_HASH_MISMATCH")
            with self.assertRaises(ContractError):
                write_run_atomically(directory, payload)
            self.assertEqual(_files(directory), preserved)
            self.assertFalse((directory / PREDICTIONS_FILE).exists())
            self.assertFalse(any(path.suffix == ".npz" for path in directory.iterdir()))

    def test_status_tokens_round_trip_without_substitution(self):
        free_spec = _spec("F1", revin=False)
        sourced_spec = _spec("P1", revin=False)
        free_payload, _free_metrics = _payload(free_spec, sourced=False)
        sourced_payload, _sourced_metrics = _payload(
            sourced_spec,
            sourced=True,
            selection=_selection(STATUS_COMPLETED),
            stage_key=_stage_key(),
        )
        with TemporaryDirectory() as temp:
            root = Path(temp)
            free_dir = run_directory(free_spec, root)
            sourced_dir = run_directory(sourced_spec, root)
            write_run_atomically(free_dir, free_payload)
            write_run_atomically(sourced_dir, sourced_payload)
            free = read_cell_record(free_dir)
            sourced = read_cell_record(sourced_dir)
            free_text = (free_dir / CELL_RECORD_FILE).read_text(encoding="utf-8")
            sourced_text = (sourced_dir / CELL_RECORD_FILE).read_text(encoding="utf-8")
            self.assertIsInstance(free["signed_error"], float)
            self.assertIsNone(free["alpha_entropy"])
            self.assertGreater(free["signed_error"], 0.0)
            self.assertEqual(free["source_info"], ABSENT)
            self.assertNotIn(NOT_APPLICABLE, free_text)
            self.assertNotIn('""', free_text)
            self.assertEqual(sourced["source_info"]["source_validation_metric"], NOT_APPLICABLE)
            self.assertEqual(sourced["source_info"]["source_validation_value_space"], NOT_APPLICABLE)
            self.assertIsInstance(sourced["source_info"]["alpha_entropy"], float)
            self.assertEqual(sourced["alpha_entropy"], sourced["source_info"]["alpha_entropy"])
            self.assertIsInstance(sourced["source_info"]["source_validation_metric"], str)
            self.assertIsInstance(sourced["source_info"]["effective_source_count"], float)
            self.assertNotEqual(sourced["source_info"]["effective_source_count"], UNRESOLVED)
            self.assertNotIn(f'"{ABSENT}"', sourced_text)
            for label, body in (("free", free), ("sourced", sourced)):
                with self.subTest(arm=label):
                    head = body["ema_shift"]["head"]
                    self.assertEqual(head["status"], DEFINED)
                    self.assertEqual(np.float64(head["relative"]), np.float64(_PRECISE))
                    self.assertIsInstance(head["relative"], float)
            self.assertNotIn("revin", free["raw_shift"])
        with TemporaryDirectory() as temp:
            spec = _spec("F4", revin=True)
            payload, _metrics = _payload(
                spec,
                sourced=True,
                selection=_selection(STATUS_COMPLETED),
                stage_key=_stage_key(),
            )
            directory = run_directory(spec, Path(temp))
            write_run_atomically(directory, payload)
            body = read_cell_record(directory)
            group = body["raw_shift"]["revin"]
            self.assertEqual(group["status"], UNDEFINED_ZERO_INIT_NORM)
            self.assertIsNone(group["relative"])
            self.assertNotEqual(group["relative"], 0)
            self.assertNotEqual(group["relative"], 0.0)
            self.assertEqual(group["absolute"], 0.0)
            text = (directory / CELL_RECORD_FILE).read_text(encoding="utf-8")
            self.assertNotIn('"UNRESOLVED"', text)
            self.assertIn(f'"{NOT_APPLICABLE}"', text)
            self.assertNotIn(f'"{ABSENT}"', text)

    def test_interrupted_write_does_not_create_done(self):
        spec = _spec("F1", revin=False)
        payload, _metrics = _payload(spec, sourced=False)
        with TemporaryDirectory() as temp:
            directory = run_directory(spec, Path(temp))
            real_replace = __import__("sita_ablation.io", fromlist=["_replace"])._replace

            def fail_cell(source, destination):
                if Path(destination).name == CELL_RECORD_FILE:
                    raise OSError("interrupted")
                return real_replace(source, destination)

            with patch("sita_ablation.io._replace", fail_cell):
                with self.assertRaises(OSError):
                    write_run_atomically(directory, payload)
            self.assertFalse((directory / DONE_FILE).exists())
            self.assertTrue((directory / PREDICTIONS_FILE).exists())
            self.assertFalse((directory / CELL_RECORD_FILE).exists())
            self.assertFalse(any(path.name.endswith(".tmp") for path in directory.iterdir()))

        with TemporaryDirectory() as temp:
            directory = run_directory(spec, Path(temp))

            def fail_done(source, destination):
                if Path(destination).name == DONE_FILE:
                    raise OSError("interrupted")
                return real_replace(source, destination)

            with patch("sita_ablation.io._replace", fail_done):
                with self.assertRaises(OSError):
                    write_run_atomically(directory, payload)
            self.assertFalse((directory / DONE_FILE).exists())
            self.assertTrue((directory / PREDICTIONS_FILE).exists())
            self.assertTrue((directory / CELL_RECORD_FILE).exists())
            self.assertEqual(inspect_done(directory), "RUN")

        with TemporaryDirectory() as temp:
            directory = run_directory(spec, Path(temp))
            broken = replace(payload.adaptation, signed_error="0")
            rejected = replace(payload, adaptation=broken)
            with self.assertRaises(ContractError):
                write_run_atomically(directory, rejected)
            self.assertFalse(directory.exists())

    def test_log_path_and_phase_audit_do_not_fill_missing_rows(self):
        plain = _spec("P1", revin=True)
        bare = _spec("P1", revin=False)
        self.assertNotEqual(log_path(plain).name, log_path(bare).name)
        self.assertTrue(log_path(bare).name.endswith("__revin0.log"))
        self.assertFalse(log_path(bare).exists())
        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertEqual(
                log_path(bare, root).name,
                f"{bare.arm}__{bare.protocol_tag}__{bare.dataset}_h{bare.horizon}_seed{bare.seed}__revin0.log",
            )
            self.assertFalse(log_path(bare, root).exists())
            audit = root / "phase_1_audit.json"
            write_phase_audit(
                audit,
                phase=1,
                missing=(),
                hash_mismatch=("F4:main:D1:h1:seed42",),
                failed=(),
                insufficient_candidates=(),
            )
            body = json.loads(audit.read_text(encoding="utf-8"))
            self.assertEqual(body["missing"], [])
            self.assertEqual(body["failed"], [])
            self.assertEqual(body["insufficient_candidates"], [])
            self.assertEqual(body["hash_mismatch"], ["F4:main:D1:h1:seed42"])
            self.assertNotIn("rmse", json.dumps(body))

    def test_failed_marker_is_not_retried_by_io(self):
        spec = _spec("F1", revin=False)
        payload, _metrics = _payload(spec, sourced=False)
        with TemporaryDirectory() as temp:
            directory = run_directory(spec, Path(temp))
            directory.mkdir(parents=True)
            (directory / FAILED_FILE).write_text(
                json.dumps({"attempt_count": 1}, sort_keys=True),
                encoding="utf-8",
            )
            self.assertEqual(inspect_done(directory), "RETRY_ONCE")
            (directory / FAILED_FILE).write_text(
                json.dumps({"attempt_count": 2}, sort_keys=True),
                encoding="utf-8",
            )
            preserved = _files(directory)
            self.assertEqual(inspect_done(directory), "HOLD_FAILED")
            with self.assertRaises(ContractError) as caught:
                write_run_atomically(directory, payload)
            self.assertIn("HOLD_FAILED", str(caught.exception))
            self.assertEqual(_files(directory), preserved)
            self.assertFalse((directory / DONE_FILE).exists())
            (directory / FAILED_FILE).write_text(
                json.dumps({"attempt_count": 1, "error": "RuntimeError: hidden"}, sort_keys=True),
                encoding="utf-8",
            )
            with self.assertRaises(ContractError) as extra:
                inspect_done(directory)
            self.assertIn("attempt_count", str(extra.exception))


class GapIoSchemaTests(unittest.TestCase):
    def test_schema_keys_and_config_digest(self):
        spec = _spec("F1", revin=False)
        other = _spec("F4", revin=True)
        frozen = frozen_config_sha256()
        first = config_digest(frozen, spec)
        self.assertEqual(first, config_digest(frozen, spec))
        self.assertNotEqual(first, config_digest(frozen, other))
        identity = {
            "frozen_config_sha256": frozen,
            "run_spec": {name: getattr(spec, name) for name in RUN_SPEC_KEYS},
        }
        self.assertEqual(set(identity), {"frozen_config_sha256", "run_spec"})
        self.assertEqual(set(identity["run_spec"]), set(RUN_SPEC_KEYS))
        self.assertNotIn("git_head", identity)
        encoded = canonical_json_bytes(identity)
        self.assertEqual(first, hashlib.sha256(encoded).hexdigest())
        altered = {
            "frozen_config_sha256": frozen,
            "git_head": "a" * 40,
            "run_spec": identity["run_spec"],
        }
        self.assertNotEqual(first, hashlib.sha256(canonical_json_bytes(altered)).hexdigest())
        payload, _metrics = _payload(spec, sourced=False)
        with TemporaryDirectory() as temp:
            directory = run_directory(spec, Path(temp))
            write_run_atomically(directory, payload)
            body = read_cell_record(directory)
            self.assertEqual(set(body), set(CELL_RECORD_KEYS))
            self.assertEqual(set(body["run_spec"]), set(RUN_SPEC_KEYS))
            self.assertEqual(set(body["pair"]), set(PAIR_KEYS))
            self.assertEqual(set(body["raw_shift"]["head"]), set(SHIFT_GROUP_KEYS))
            self.assertEqual(set(body["ema_shift"]["head"]), set(SHIFT_GROUP_KEYS))
            self.assertIn("raw_shift", body)
            self.assertIn("ema_shift", body)
            self.assertEqual(body["config_digest"], first)
            self.assertNotEqual(body["config_digest"], body["git_head"])
            self.assertEqual(body["source_info"], ABSENT)
            self.assertIsInstance(body["tensorflow_version"], str)
            self.assertNotIn("numpy_version", body)
            self.assertEqual(body["prediction_path"], "predictions.parquet")
            self.assertEqual(set(body["artifact_sha256"]), {PREDICTIONS_FILE})

    def test_timestamps_are_utc_microseconds(self):
        spec = _spec("F1", revin=False)
        payload, _metrics = _payload(spec, sourced=False)
        naive = replace(payload, started_at=datetime(2026, 10, 2, 1, 2, 3))
        with TemporaryDirectory() as temp:
            directory = run_directory(spec, Path(temp))
            with self.assertRaises(ContractError) as caught:
                write_run_atomically(directory, naive)
            self.assertIn("naive", str(caught.exception))
        plus8 = timezone(timedelta(hours=8))
        started = datetime(2026, 10, 2, 9, 2, 3, 123456, tzinfo=plus8)
        finished = started + timedelta(seconds=3, microseconds=1)
        shifted = replace(payload, started_at=started, finished_at=finished)
        zero = datetime(2026, 10, 2, 0, 0, 0, tzinfo=timezone.utc)
        exact = replace(payload, started_at=zero, finished_at=zero + timedelta(seconds=1))
        with TemporaryDirectory() as temp:
            root = Path(temp)
            shifted_dir = run_directory(spec, root / "shifted")
            exact_dir = run_directory(spec, root / "exact")
            write_run_atomically(shifted_dir, shifted)
            write_run_atomically(exact_dir, exact)
            shifted_body = read_cell_record(shifted_dir)
            exact_body = read_cell_record(exact_dir)
            self.assertEqual(shifted_body["started_at"], "2026-10-02T01:02:03.123456+00:00")
            self.assertEqual(shifted_body["finished_at"], "2026-10-02T01:02:06.123457+00:00")
            self.assertEqual(exact_body["started_at"], "2026-10-02T00:00:00.000000+00:00")
            self.assertEqual(exact_body["finished_at"], "2026-10-02T00:00:01.000000+00:00")
            self.assertEqual(shifted_body["elapsed_seconds"], (finished.astimezone(timezone.utc) - started.astimezone(timezone.utc)).total_seconds())

    def test_sample_counts_follow_the_target_window(self):
        self.assertEqual([expected_target_partition_count(horizon) for horizon in (1, 3, 5)], [5, 3, 1])
        frame = _target_frame(15)
        short = _target_frame(14)
        for horizon, expected in ((1, 5), (3, 3), (5, 1)):
            batch = build_target_partition_windows(
                frame,
                key_fields=("store", "item"),
                key=("1", "1"),
                feature_columns=("sales",),
                horizon=horizon,
                partition="train",
            )
            self.assertEqual(len(batch.label_date), expected)
            self.assertEqual(int(batch.y_scaled.shape[0]), expected)
            with self.assertRaises(ContractError):
                build_target_partition_windows(
                    short,
                    key_fields=("store", "item"),
                    key=("1", "1"),
                    feature_columns=("sales",),
                    horizon=horizon,
                    partition="train",
                )
            spec = _spec("F1", revin=False, horizon=horizon)
            payload, _metrics = _payload(spec, sourced=False)
            with TemporaryDirectory() as temp:
                directory = run_directory(spec, Path(temp))
                write_run_atomically(directory, payload)
                body = read_cell_record(directory)
                self.assertEqual(body["n_train"], expected)
                self.assertEqual(body["n_validation"], expected)
                self.assertEqual(body["n_test"], 180)
            broken = replace(payload.adaptation, n_train=expected - 1)
            rejected = replace(payload, adaptation=broken)
            with TemporaryDirectory() as temp:
                with self.assertRaises(ContractError):
                    write_run_atomically(run_directory(spec, Path(temp)), rejected)
        spec = _spec("F1", revin=False)
        windows = TargetWindows(
            train=_count_batch(4, "train"),
            validation=_count_batch(5, "validation"),
            test=_count_batch(180, "test"),
        )
        with self.assertRaises(ContractError) as counted:
            _require_windows(spec, windows)
        self.assertIn("WindowBatch", str(counted.exception))
        valid = validate_target_sample_counts(
            _count_batch(3, "train"),
            _count_batch(3, "validation"),
            _count_batch(180, "test"),
            3,
        )
        self.assertEqual(valid, (3, 3, 180))

    def test_overlap_prediction_path_and_skip_hash(self):
        top = _selection(STATUS_COMPLETED, rule="top")
        random_same = _with_keys(top, (("1", "1"), ("1", "2"), ("1", "3")), "random3")
        random_partial = _with_keys(top, (("1", "3"), ("9", "9"), ("8", "8")), "random3")
        bottom = _with_keys(top, (("1", "1"), ("4", "4"), ("5", "5")), "bottom")
        skipped = _selection(STATUS_SKIPPED, rule="random3")
        self.assertEqual(top3_overlap_count(random_same, top, horizon=1, reference_horizon=1), 3)
        self.assertEqual(top3_overlap_count(random_partial, top, horizon=3, reference_horizon=3), 1)
        self.assertEqual(top3_overlap_count(bottom, top, horizon=5, reference_horizon=5), 1)
        self.assertIsNone(top3_overlap_count(random_same, None, horizon=1, reference_horizon=1))
        self.assertIsNone(top3_overlap_count(random_same, skipped, horizon=1, reference_horizon=1))
        self.assertIsNone(top3_overlap_count(skipped, top, horizon=1, reference_horizon=1))
        self.assertNotEqual(top3_overlap_count(random_same, None, horizon=1, reference_horizon=1), 0)
        disjoint = _with_keys(top, (("7", "7"), ("8", "8"), ("9", "9")), "random3")
        self.assertEqual(top3_overlap_count(disjoint, top, horizon=1, reference_horizon=1), 0)
        for tag, rule, count in (("random3", "random3", 2), ("bottom3", "bottom", 0)):
            spec = _spec("F4", revin=True, phase=2, protocol_tag=tag)
            payload, _metrics = _payload(
                spec,
                sourced=True,
                selection=_selection(STATUS_COMPLETED, rule=rule),
                stage_key=_stage_key(rule=rule),
                top3_overlap_count=count,
            )
            with TemporaryDirectory() as temp:
                directory = run_directory(spec, Path(temp))
                write_run_atomically(directory, payload)
                body = read_cell_record(directory)
                self.assertEqual(set(SOURCE_INFO_KEYS) <= set(body["source_info"]), True)
                self.assertEqual(body["source_info"]["top3_overlap_count"], count)
                self.assertEqual(body["prediction_path"], "predictions.parquet")
                self.assertFalse(Path(body["prediction_path"]).is_absolute())
        omitted = _spec("F4", revin=True, phase=2, protocol_tag="random3")
        omitted_payload, _metrics = _payload(
            omitted,
            sourced=True,
            selection=_selection(STATUS_COMPLETED, rule="random3"),
            stage_key=_stage_key(rule="random3"),
        )
        with TemporaryDirectory() as temp:
            directory = run_directory(omitted, Path(temp))
            write_run_atomically(directory, omitted_payload)
            body = read_cell_record(directory)
            self.assertNotIn("top3_overlap_count", body["source_info"])
        illegal = replace(omitted_payload, top3_overlap_count=4)
        top_spec = _spec("F4", revin=True)
        top_payload, _metrics = _payload(
            top_spec,
            sourced=True,
            selection=_selection(STATUS_COMPLETED),
            stage_key=_stage_key(),
            top3_overlap_count=1,
        )
        with TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises(ContractError):
                write_run_atomically(run_directory(omitted, root / "illegal"), illegal)
            with self.assertRaises(ContractError):
                write_run_atomically(run_directory(top_spec, root / "top"), top_payload)
        skip_spec = _spec("F4", revin=True, phase=2, protocol_tag="random3")
        from sita_ablation.io import RunPayload

        skip_payload = RunPayload(
            spec=skip_spec,
            started_at=_STARTED,
            finished_at=_FINISHED,
            selection=_selection(STATUS_SKIPPED, rule="random3"),
            stage_a_key=_stage_key(rule="random3"),
        )
        with TemporaryDirectory() as temp:
            directory = run_directory(skip_spec, Path(temp))
            record = write_run_atomically(directory, skip_payload)
            names = {path.name for path in directory.iterdir()}
            self.assertEqual(names, {SELECTION_FILE, DONE_FILE})
            self.assertEqual(set(record.artifact_sha256), {SELECTION_FILE})
            self.assertFalse((directory / PREDICTIONS_FILE).exists())
            self.assertFalse((directory / CELL_RECORD_FILE).exists())
            done = json.loads((directory / DONE_FILE).read_text(encoding="utf-8"))
            self.assertEqual(set(done["artifact_sha256"]), {SELECTION_FILE})
            self.assertNotIn(PREDICTIONS_FILE, done["artifact_sha256"])
            self.assertNotIn("prediction_path", json.dumps(done))
        with TemporaryDirectory() as temp:
            with self.assertRaises(ContractError):
                write_run_atomically(
                    run_directory(skip_spec, Path(temp)),
                    replace(skip_payload, top3_overlap_count=0),
                )

    def test_revin0_logs_phase_audit_and_failed_bodies(self):
        for arm in ("P1", "P2", "P3"):
            bare = _spec(arm, revin=False)
            plain = _spec(arm, revin=True)
            self.assertTrue(log_path(bare).name.endswith("__revin0.log"))
            self.assertFalse(log_path(plain).name.endswith("__revin0.log"))
            self.assertEqual(run_directory(bare, Path("anchor")).name, "revin0")
        for arm in ("F1", "F3"):
            self.assertNotIn("revin0", log_path(_spec(arm, revin=False)).name)
        example = _spec("F4", revin=True, phase=2, protocol_tag="random3", horizon=3)
        self.assertEqual(cell_id(example), "F4:random3:D1:h3:seed42")
        self.assertEqual(phase_audit_path(2), Path("消融实验") / "audit" / "phase_2_audit.json")
        self.assertEqual(require_clean_workspace_for_formal_run(False), None)
        with self.assertRaises(ContractError) as dirty:
            require_clean_workspace_for_formal_run(True)
        self.assertIn("workspace_dirty", str(dirty.exception))
        with TemporaryDirectory() as temp:
            root = Path(temp)
            audit = root / "phase_2_audit.json"
            write_phase_audit(
                audit,
                phase=2,
                missing=("F4:bottom3:D1:h5:seed46", "F4:random3:D1:h3:seed42"),
                hash_mismatch=("F4:k5:D1:h1:seed43",),
                failed=(),
                insufficient_candidates=("F4:ell15:D1:h1:seed44",),
            )
            body = json.loads(audit.read_text(encoding="utf-8"))
            self.assertEqual(set(body), set(PHASE_AUDIT_CATEGORIES))
            self.assertEqual(body["missing"], ["F4:bottom3:D1:h5:seed46", "F4:random3:D1:h3:seed42"])
            self.assertEqual(body["failed"], [])
            write_phase_audit(root / "phase_5_h1_audit.json", phase=5)
            self.assertEqual(
                json.loads((root / "phase_5_h1_audit.json").read_text(encoding="utf-8"))["missing"],
                [],
            )
            with self.assertRaises(ContractError):
                write_phase_audit(root / "phase_5_h2_audit.json", phase=5)
            with self.assertRaises(ContractError):
                write_phase_audit(root / "phase_audit.json", phase=1, missing=("F4:main:D1:h1:seed42",))
            with self.assertRaises(ContractError):
                write_phase_audit(root / "phase_1_audit.json", phase=1, missing=("D1:1:42",))
            with self.assertRaises(ContractError):
                write_phase_audit(
                    root / "phase_1_audit.json",
                    phase=1,
                    missing=("F4:main:D1:h1:seed42",),
                    failed=("F4:main:D1:h1:seed42",),
                )
        spec = _spec("F1", revin=False)
        with TemporaryDirectory() as temp:
            directory = run_directory(spec, Path(temp) / "cell")
            directory.mkdir(parents=True)
            write_cell_failed(directory, 1)
            self.assertEqual(
                json.loads((directory / FAILED_FILE).read_text(encoding="utf-8")),
                {"attempt_count": 1},
            )
            write_cell_failed(directory, 2)
            self.assertEqual(inspect_done(directory), "HOLD_FAILED")
            with self.assertRaises(ContractError):
                write_cell_failed(directory, 3)
            from sita_ablation.stage_a import _write_failure

            cache = Path(temp) / "stage-a"
            _write_failure(cache, _stage_key(), RuntimeError("synthetic fit failure"))
            failed = json.loads((cache / FAILED_FILE).read_text(encoding="utf-8"))
            self.assertEqual(set(failed), {"attempt_count", "error", "stage_a_key_sha256"})
            self.assertEqual(failed["attempt_count"], 1)
            self.assertIn("RuntimeError", failed["error"])
            self.assertIn("synthetic fit failure", failed["error"])


class RunDirectoryGuardTests(unittest.TestCase):
    def test_real_run_and_cache_directories_stay_unchanged(self):
        self.assertEqual(_tree(_RUNS), _RUNS_BEFORE)
        self.assertEqual(_tree(_CACHE), _CACHE_BEFORE)


if __name__ == "__main__":
    unittest.main()
