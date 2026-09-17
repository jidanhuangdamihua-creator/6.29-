"""Fail-closed orchestration contract for the corrected formal SITA matrix.

This module owns only the corrected SITA target/LR authority, matrix planning,
provenance, digest, attempt/resume semantics, and the thin execution
orchestration needed to call existing ``src`` numerical authorities.  The
training-capable imports remain lazy and are reachable only when ``--execute``
is explicitly requested.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_RELPATH = Path("configs") / "ms_katl_corrected_sita_60cell_manifest.json"
FORMAL_RUNNER_LABEL = "CORRECTED_SITA_FORMAL_60CELL"
PAPER_METHOD_NAME = "SITA"
RUNTIME_METHOD_ID = "FUSED_TARGET_ONLY_V3"
SCENARIO = "without"
DATASETS = ("D1", "D4", "D5", "D6")
HORIZONS = (1, 3, 5)
SEEDS = (42, 43, 44, 45, 46)
TARGETS = {
    "D1": "1/10",
    "D4": "166/258",
    "D5": "48/1159414",
    "D6": "CA_1/FOODS_3_080",
}
TARGET_LR_LABELS = {
    "D1": "3e-5",
    "D4": "3e-5",
    "D5": "3e-4",
    "D6": "3e-5",
}
SOURCE_LR_LABEL = "1e-4"
EXPECTED_TARGET_ASSERTIONS = {
    "D1": "1/10",
    "D4": "166/258",
    "D5": "48/1159414",
    "D6": "CA_1/FOODS_3_080",
}
MATRIX_COLUMNS = (
    "cell_key",
    "dataset",
    "target",
    "horizon",
    "seed",
    "stage_a_seed",
    "target_lr",
    "paper_method_name",
    "runtime_method_id",
    "scenario",
)
CRITICAL_RUNTIME_FILES = (
    "configs/ms_katl_corrected_sita_60cell_manifest.json",
    "scripts/run_sita_corrected_60cell.py",
    "src/transfer_methods/ms_katl_corrected_sita_60cell.py",
    "src/transfer_methods/ms_katl_fto_three_track_robustness.py",
    "src/transfer_methods/ms_katl_lite.py",
    "src/transfer_methods/ms_katl_mechanism_ablation.py",
    "src/transfer_methods/ms_katl_primary_target_lr_screening.py",
    "src/transfer_methods/ms_katl_v3_frame_assembly.py",
    "src/protocols/reproducibility.py",
    "src/evaluation/metric_contract.py",
)
HISTORICAL_ROOT_MARKERS = (
    "20260818_190337_640037_MS_KATL_FTO_THREE_TRACK_ROBUSTNESS",
    "sita_all_targets_d4_d6_5seed",
    "20260818_180250_355272_MS_KATL_FTO_V3_SCREENING",
    "20260818_155005_113008_MS_KATL_PRIMARY_TARGET_LR_SCREENING",
    "final_statistical_freeze_20260910",
    "03_v3_raw_5seed",
    "ms_katl_ae_ablation",
    "mechanism_diagnostic",
)


class CorrectedSITAContractError(RuntimeError):
    """A corrected-SITA contract violation that must stop the run."""


@dataclass(frozen=True)
class CellPlan:
    dataset: str
    target: str
    horizon: int
    seed: int

    @property
    def cell_key(self) -> str:
        return f"{self.dataset}__{self.target.replace('/', '-')}__h{self.horizon}__s{self.seed}"

    @property
    def target_lr_label(self) -> str:
        return TARGET_LR_LABELS[self.dataset]

    def matrix_row(self) -> Dict[str, Any]:
        return {
            "cell_key": self.cell_key,
            "dataset": self.dataset,
            "target": self.target,
            "horizon": int(self.horizon),
            "seed": int(self.seed),
            "stage_a_seed": int(self.seed),
            "target_lr": self.target_lr_label,
            "paper_method_name": PAPER_METHOD_NAME,
            "runtime_method_id": RUNTIME_METHOD_ID,
            "scenario": SCENARIO,
        }


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_manifest(project_root: Path = PROJECT_ROOT) -> Dict[str, Any]:
    path = Path(project_root) / MANIFEST_RELPATH
    if not path.is_file():
        raise CorrectedSITAContractError(f"formal corrected-SITA manifest missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorrectedSITAContractError(f"cannot read corrected-SITA manifest: {path}") from exc
    return payload


def assert_formal_target_authority(manifest: Mapping[str, Any]) -> None:
    targets = dict(manifest.get("targets") or {})
    if targets != EXPECTED_TARGET_ASSERTIONS:
        raise CorrectedSITAContractError(
            f"FORMAL_TARGET_ASSERTIONS failed: targets={targets!r}; expected={EXPECTED_TARGET_ASSERTIONS!r}"
        )
    if targets.get("D4") != "166/258":
        raise CorrectedSITAContractError("D4 target must be 166/258; refusing 166/311 substitution")
    forbidden = {"166/311", "166-311", "166_311"}
    if forbidden.intersection(str(value) for value in targets.values()):
        raise CorrectedSITAContractError("historical D4 target 166/311 cannot enter corrected formal SITA")
    if str(manifest.get("paper_method_name")) != PAPER_METHOD_NAME:
        raise CorrectedSITAContractError("paper_method_name must be SITA")
    if str(manifest.get("runtime_method_id")) != RUNTIME_METHOD_ID:
        raise CorrectedSITAContractError("runtime_method_id must be FUSED_TARGET_ONLY_V3")
    if tuple(manifest.get("datasets") or ()) != DATASETS:
        raise CorrectedSITAContractError("corrected SITA datasets must be D1,D4,D5,D6 in order")
    if tuple(int(item) for item in manifest.get("horizons") or ()) != HORIZONS:
        raise CorrectedSITAContractError("corrected SITA horizons must be H1,H3,H5")
    if tuple(int(item) for item in manifest.get("seeds") or ()) != SEEDS:
        raise CorrectedSITAContractError("corrected SITA seeds must be 42,43,44,45,46")
    if str(manifest.get("source_learning_rate")) != SOURCE_LR_LABEL:
        raise CorrectedSITAContractError("source LR must be the formal authority value 1e-4")
    if dict(manifest.get("target_learning_rates") or {}) != TARGET_LR_LABELS:
        raise CorrectedSITAContractError("target LR mapping drifted from corrected formal SITA mapping")
    if int(manifest.get("source_count_k", 0)) != 3:
        raise CorrectedSITAContractError("corrected SITA K must be 3")
    if str(manifest.get("formal_weight_variant")) != "EMA":
        raise CorrectedSITAContractError("formal weight variant must be EMA")
    if str(manifest.get("metric_space")) != "original_sales_space":
        raise CorrectedSITAContractError("metric space must be original_sales_space")


def validate_authority_against_runtime(manifest: Mapping[str, Any]) -> None:
    """Check live non-training constants without loading model execution code."""
    try:
        from src.transfer_methods.ms_katl_primary_target_lr_screening import STAGE_A_LEARNING_RATE
        from src.transfer_methods.ms_katl_primary_target_lr_screening import SCREENING_EPOCHS
    except Exception as exc:  # pragma: no cover - environment-specific import failure
        raise CorrectedSITAContractError(f"formal runtime authority import failed: {exc}") from exc
    if float(STAGE_A_LEARNING_RATE) != 1e-4:
        raise CorrectedSITAContractError(f"active Stage-A LR drifted: {STAGE_A_LEARNING_RATE!r}")
    if float(manifest.get("source_learning_rate_value", 1e-4)) != float(STAGE_A_LEARNING_RATE):
        raise CorrectedSITAContractError("manifest source LR disagrees with active Stage-A authority")
    if int(manifest.get("source_epochs", 0)) != int(SCREENING_EPOCHS):
        raise CorrectedSITAContractError("manifest source epochs disagree with active formal authority")


def enumerate_cells(manifest: Optional[Mapping[str, Any]] = None) -> Tuple[CellPlan, ...]:
    payload = dict(manifest or load_manifest())
    assert_formal_target_authority(payload)
    cells = tuple(
        CellPlan(dataset, EXPECTED_TARGET_ASSERTIONS[dataset], horizon, seed)
        for dataset in DATASETS
        for horizon in HORIZONS
        for seed in SEEDS
    )
    if len(cells) != 60 or len({cell.cell_key for cell in cells}) != 60:
        raise CorrectedSITAContractError("corrected formal SITA matrix must contain 60 unique cells")
    return cells


def canonical_matrix_csv(cells: Sequence[CellPlan]) -> bytes:
    rows = [cell.matrix_row() for cell in cells]
    import io

    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=list(MATRIX_COLUMNS), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode("utf-8")


def matrix_digest(cells: Sequence[CellPlan]) -> str:
    return sha256_bytes(canonical_matrix_csv(cells))


def protocol_payload(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    keys = (
        "paper_method_name", "runtime_method_id", "scenario", "datasets", "targets",
        "horizons", "seeds", "source_count_k", "source_learning_rate", "source_epochs",
        "target_learning_rates", "input_window", "target_train_days", "target_val_days",
        "target_test_days", "target_epochs", "batch_size", "loss", "huber_delta",
        "optimizer", "schedule", "ema", "ema_momentum", "revin", "revin_affine",
        "formal_weight_variant", "metric_space", "test_refit", "early_stopping",
        "stage_a_seed_contract_version",
    )
    return {key: manifest.get(key) for key in keys}


def protocol_digest(manifest: Mapping[str, Any]) -> str:
    return sha256_bytes(_canonical_json(protocol_payload(manifest)))


def _file_digest(path: Path) -> str:
    if not path.is_file():
        raise CorrectedSITAContractError(f"critical runtime file missing: {path}")
    return sha256_bytes(path.read_bytes())


def critical_runtime_snapshot(project_root: Path = PROJECT_ROOT) -> Dict[str, str]:
    root = Path(project_root)
    return {relpath: _file_digest(root / relpath) for relpath in CRITICAL_RUNTIME_FILES}


def critical_runtime_digest(project_root: Path = PROJECT_ROOT) -> str:
    return sha256_bytes(_canonical_json(critical_runtime_snapshot(project_root)))


def _git_output(project_root: Path, args: Sequence[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=str(project_root), stderr=subprocess.STDOUT, text=True
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CorrectedSITAContractError(f"git snapshot failed: {' '.join(args)}") from exc


def working_tree_protocol_digest(project_root: Path = PROJECT_ROOT) -> str:
    root = Path(project_root)
    head = _git_output(root, ["rev-parse", "HEAD"]).strip()
    critical_diff = _git_output(root, ["diff", "--no-ext-diff", "--binary", "--", *CRITICAL_RUNTIME_FILES])
    untracked = _git_output(root, ["ls-files", "--others", "--exclude-standard", "--", *CRITICAL_RUNTIME_FILES])
    untracked_digests = {
        relpath: _file_digest(root / relpath)
        for relpath in untracked.splitlines()
        if relpath.strip()
    }
    payload = {
        "HEAD": head,
        "critical_file_bytes": critical_runtime_snapshot(root),
        "critical_git_diff": critical_diff,
        "critical_untracked_file_digests": untracked_digests,
    }
    return sha256_bytes(_canonical_json(payload))


def resolve_output_root(project_root: Path, requested: Optional[Path]) -> Path:
    if requested is None:
        return (Path(project_root) / "outputs" / f"{FORMAL_RUNNER_LABEL}_PLAN_ONLY").resolve()
    path = Path(requested)
    if not path.is_absolute():
        path = Path(project_root) / path
    return path.resolve()


def assert_output_isolation(output_root: Path, project_root: Path = PROJECT_ROOT) -> None:
    resolved = Path(output_root).resolve()
    project_outputs = (Path(project_root) / "outputs").resolve()
    text = str(resolved).lower()
    if any(marker.lower() in text for marker in HISTORICAL_ROOT_MARKERS):
        raise CorrectedSITAContractError(f"historical output root rejected: {resolved}")
    historical_runs_root = project_outputs / "runs"
    if resolved == historical_runs_root or historical_runs_root in resolved.parents:
        raise CorrectedSITAContractError(f"historical outputs/runs root rejected: {resolved}")
    if resolved == project_outputs or project_outputs not in resolved.parents:
        raise CorrectedSITAContractError(
            f"corrected SITA output must be an isolated child of {project_outputs}: {resolved}"
        )
    if resolved.name == "" or resolved == Path(project_root).resolve():
        raise CorrectedSITAContractError(f"unsafe output root: {resolved}")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorrectedSITAContractError(f"invalid JSON artifact: {path}") from exc


def attempt_root(run_root: Path, cell: CellPlan, attempt_number: int) -> Path:
    return Path(run_root) / "cells" / cell.cell_key / f"attempt_{int(attempt_number):03d}"


def _attempt_numbers(cell_root: Path) -> List[int]:
    numbers: List[int] = []
    for path in Path(cell_root).glob("attempt_*"):
        try:
            numbers.append(int(path.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return sorted(numbers)


def _pid_alive(pid: Any) -> bool:
    try:
        value = int(pid)
    except (TypeError, ValueError):
        return False
    if value <= 0:
        return False
    try:
        os.kill(value, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _latest_attempt(cell_root: Path) -> Optional[Path]:
    numbers = _attempt_numbers(cell_root)
    return cell_root / f"attempt_{numbers[-1]:03d}" if numbers else None


def _is_complete_attempt(path: Path, cell: CellPlan, digests: Mapping[str, str]) -> bool:
    status_path = path / "status.json"
    result_path = path / "result.json"
    if not status_path.is_file() or not result_path.is_file():
        return False
    status = _read_json(status_path)
    result = _read_json(result_path)
    return (
        status.get("status") == "COMPLETE"
        and "exception_type" not in status
        and "exception_message" not in status
        and result.get("status") == "COMPLETE"
        and "exception_type" not in result
        and "exception_message" not in result
        and result.get("cell_key") == cell.cell_key
        and result.get("dataset") == cell.dataset
        and result.get("target") == cell.target
        and int(result.get("horizon")) == int(cell.horizon)
        and int(result.get("seed")) == int(cell.seed)
        and result.get("stage_a_seed") == cell.seed
        and result.get("matrix_digest") == digests["matrix_digest"]
        and result.get("protocol_digest") == digests["protocol_digest"]
        and result.get("critical_runtime_digest") == digests["critical_runtime_digest"]
        and result.get("working_tree_protocol_digest") == digests["working_tree_protocol_digest"]
    )


def _assert_attempt_digests(path: Path, digests: Mapping[str, str]) -> None:
    """Refuse resume when any persisted attempt artifact belongs to another contract."""
    for artifact_name in ("status.json", "result.json", "provenance.json"):
        artifact_path = path / artifact_name
        if not artifact_path.is_file():
            continue
        payload = _read_json(artifact_path)
        for key, expected in digests.items():
            if payload.get(key) != expected:
                raise CorrectedSITAContractError(f"RESUME_REFUSED_DIGEST_MISMATCH: {key}")


def prepare_cell_attempt(
    run_root: Path,
    cell: CellPlan,
    *,
    resume: bool,
    digests: Mapping[str, str],
) -> Tuple[str, Optional[Path]]:
    cell_root = Path(run_root) / "cells" / cell.cell_key
    latest = _latest_attempt(cell_root)
    if latest is not None and resume:
        _assert_attempt_digests(latest, digests)
    if latest is not None and _is_complete_attempt(latest, cell, digests):
        return "SKIPPED_COMPLETE", latest
    if not resume and latest is not None:
        raise CorrectedSITAContractError(
            f"existing cell attempts require explicit --resume: {cell.cell_key}"
        )
    if latest is not None and (latest / "status.json").is_file():
        previous = _read_json(latest / "status.json")
        previous_status = str(previous.get("status") or "").upper()
        if previous_status == "RUNNING":
            if _pid_alive(previous.get("pid")):
                raise CorrectedSITAContractError(
                    f"RESUME_ACTIVE_ATTEMPT: {cell.cell_key} pid={previous.get('pid')}"
                )
            previous["status"] = "INTERRUPTED"
            previous["interrupted_at"] = _now()
            previous["interrupted_reason"] = "resume_observed_dead_process"
            _write_json(latest / "status.json", previous)
    next_number = (_attempt_numbers(cell_root)[-1] + 1) if _attempt_numbers(cell_root) else 1
    path = attempt_root(run_root, cell, next_number)
    path.mkdir(parents=True, exist_ok=False)
    _write_json(
        path / "status.json",
        {
            "status": "RUNNING",
            "cell_key": cell.cell_key,
            "attempt_id": f"attempt_{next_number:03d}",
            "pid": os.getpid(),
            "start_time": _now(),
            **dict(digests),
        },
    )
    return "RUN", path


def finalize_attempt(
    attempt_path: Path,
    *,
    status: str,
    result: Optional[Mapping[str, Any]] = None,
    error: Optional[BaseException] = None,
) -> None:
    status_payload = _read_json(attempt_path / "status.json")
    status_payload.update({"status": status, "end_time": _now()})
    if error is not None:
        status_payload["exception_type"] = type(error).__name__
        status_payload["exception_message"] = str(error)
    if result is not None:
        _write_json(attempt_path / "result.json", result)
    _write_json(attempt_path / "status.json", status_payload)


def _validate_digest_fields(payload: Mapping[str, Any], digests: Mapping[str, str]) -> None:
    for key, expected in digests.items():
        if payload.get(key) != expected:
            raise CorrectedSITAContractError(f"RESUME_REFUSED_DIGEST_MISMATCH: {key}")


def create_manifest(
    run_root: Path,
    *,
    manifest: Mapping[str, Any],
    cells: Sequence[CellPlan],
    digests: Mapping[str, str],
    branch: str,
    head: str,
) -> Dict[str, Any]:
    return {
        "manifest_version": "corrected_sita_runtime_v2",
        "runner_label": FORMAL_RUNNER_LABEL,
        "paper_method_name": PAPER_METHOD_NAME,
        "runtime_method_id": RUNTIME_METHOD_ID,
        "run_root": str(Path(run_root).resolve()),
        "created_at": _now(),
        "branch": branch,
        "HEAD": head,
        "matrix_digest": digests["matrix_digest"],
        "protocol_digest": digests["protocol_digest"],
        "critical_runtime_digest": digests["critical_runtime_digest"],
        "working_tree_protocol_digest": digests["working_tree_protocol_digest"],
        "matrix": [cell.matrix_row() for cell in cells],
        "protocol": protocol_payload(manifest),
        "critical_runtime_files": list(CRITICAL_RUNTIME_FILES),
        "output_layout": "cells/<cell_key>/attempt_<nnn>/{status,result,provenance}.json",
        "resume_policy": "fail_closed_digests_preserve_attempts",
    }


def read_git_identity(project_root: Path = PROJECT_ROOT) -> Tuple[str, str]:
    return (
        _git_output(Path(project_root), ["branch", "--show-current"]).strip(),
        _git_output(Path(project_root), ["rev-parse", "HEAD"]).strip(),
    )


def build_digests(manifest: Mapping[str, Any], cells: Sequence[CellPlan], project_root: Path = PROJECT_ROOT) -> Dict[str, str]:
    return {
        "matrix_digest": matrix_digest(cells),
        "protocol_digest": protocol_digest(manifest),
        "critical_runtime_digest": critical_runtime_digest(project_root),
        "working_tree_protocol_digest": working_tree_protocol_digest(project_root),
    }


def coverage_summary(cells: Sequence[CellPlan]) -> Dict[str, Any]:
    return {
        "TOTAL_CELLS": len(cells),
        "UNIQUE_CELL_KEYS": len({cell.cell_key for cell in cells}),
        **{f"{dataset}_CELLS": sum(cell.dataset == dataset for cell in cells) for dataset in DATASETS},
        **{f"H{horizon}_CELLS": sum(cell.horizon == horizon for cell in cells) for horizon in HORIZONS},
        **{f"SEED{seed}_CELLS": sum(cell.seed == seed for cell in cells) for seed in SEEDS},
    }


def validate_result_contract(result: Mapping[str, Any], cell: CellPlan, digests: Mapping[str, str]) -> None:
    required = {
        "cell_key", "dataset", "target", "horizon", "seed", "stage_a_seed",
        "paper_method_name", "runtime_method_id", "source_lr", "target_lr",
        "selected_source_ids", "source_distances", "source_fusion_weights",
        "source_weight_digests", "fused_target_digest", "rmse_original",
        "smape_original", "metric_space", "formal_weight_variant", "status",
        "attempt_id", "start_time", "end_time", "matrix_digest", "protocol_digest",
        "critical_runtime_digest", "working_tree_protocol_digest", "branch", "HEAD",
    }
    missing = sorted(required - set(result))
    if missing:
        raise CorrectedSITAContractError(f"result provenance missing fields: {missing}")
    if result["status"] != "COMPLETE" or result["cell_key"] != cell.cell_key:
        raise CorrectedSITAContractError("result status or cell identity is invalid")
    if "exception_type" in result or "exception_message" in result:
        raise CorrectedSITAContractError("COMPLETE result cannot contain an exception marker")
    if result["dataset"] != cell.dataset or result["target"] != cell.target:
        raise CorrectedSITAContractError("result target identity mismatch")
    if int(result["horizon"]) != cell.horizon or int(result["seed"]) != cell.seed:
        raise CorrectedSITAContractError("result horizon/seed mismatch")
    if int(result["stage_a_seed"]) != cell.seed:
        raise CorrectedSITAContractError("stage_a_seed must equal cell seed")
    if result["paper_method_name"] != PAPER_METHOD_NAME or result["runtime_method_id"] != RUNTIME_METHOD_ID:
        raise CorrectedSITAContractError("result method identity mismatch")
    if result["target_lr"] != cell.target_lr_label or result["source_lr"] != SOURCE_LR_LABEL:
        raise CorrectedSITAContractError("result LR identity mismatch")
    if result["metric_space"] != "original_sales_space" or result["formal_weight_variant"] != "EMA":
        raise CorrectedSITAContractError("result metric space or EMA identity mismatch")
    try:
        import math
        if not math.isfinite(float(result["rmse_original"])) or not math.isfinite(float(result["smape_original"])):
            raise CorrectedSITAContractError("result metrics must be finite")
    except (TypeError, ValueError) as exc:
        raise CorrectedSITAContractError("result metrics must be numeric") from exc
    _validate_digest_fields(result, digests)


def build_result_payload(
    *,
    cell: CellPlan,
    attempt_path: Path,
    authority_payload: Mapping[str, Any],
    digests: Mapping[str, str],
    branch: str,
    head: str,
    start_time: str,
    end_time: str,
) -> Dict[str, Any]:
    stage_a_seed = int(authority_payload.get("seed", cell.seed))
    if stage_a_seed != cell.seed:
        raise CorrectedSITAContractError("formal authority returned stage_a seed different from cell seed")
    result = {
        "status": "COMPLETE",
        "cell_key": cell.cell_key,
        "dataset": cell.dataset,
        "target": cell.target,
        "horizon": int(cell.horizon),
        "seed": int(cell.seed),
        "stage_a_seed": stage_a_seed,
        "paper_method_name": PAPER_METHOD_NAME,
        "runtime_method_id": RUNTIME_METHOD_ID,
        "scenario": SCENARIO,
        "source_lr": SOURCE_LR_LABEL,
        "target_lr": cell.target_lr_label,
        "selected_source_ids": authority_payload.get("source_identities", []),
        "source_distances": authority_payload.get("source_distances", []),
        "source_fusion_weights": authority_payload.get("source_weights", []),
        "source_weight_digests": authority_payload.get("source_model_digests", []),
        "fused_target_digest": authority_payload.get("fused_init_digest", ""),
        "rmse_original": authority_payload.get("rmse_original"),
        "smape_original": authority_payload.get("smape_original"),
        "metric_space": "original_sales_space",
        "formal_weight_variant": "EMA",
        "matrix_digest": digests["matrix_digest"],
        "protocol_digest": digests["protocol_digest"],
        "critical_runtime_digest": digests["critical_runtime_digest"],
        "working_tree_protocol_digest": digests["working_tree_protocol_digest"],
        "branch": branch,
        "HEAD": head,
        "attempt_id": attempt_path.name,
        "start_time": start_time,
        "end_time": end_time,
        "prediction_artifact_path": authority_payload.get("prediction_artifact_path"),
        "architecture_digest": authority_payload.get("architecture_digest"),
        "training_recipe": authority_payload.get("training_recipe", {}),
        "ema_applied": authority_payload.get("ema_applied"),
    }
    validate_result_contract(result, cell, digests)
    return result


def _run_stage_a(
    *,
    cell: Mapping[str, Any],
    horizon: int,
    seed: int,
    epochs: int,
    batch_size: int,
) -> Tuple[Any, Any]:
    """Assemble the existing Stage-A and shared-target authorities."""
    from src.protocols.reproducibility import set_protocol_seed
    from src.transfer_methods.ms_katl_lite import prepare_ms_katl_stage_a
    from src.transfer_methods.ms_katl_mechanism_ablation import prepare_shared_target_bundle
    from src.transfer_methods.ms_katl_primary_target_lr_screening import STAGE_A_LEARNING_RATE

    set_protocol_seed(int(seed), include_frameworks=True)
    stage_a = prepare_ms_katl_stage_a(
        source_df=cell["source_model_df"],
        target_df=cell["target_model_df"],
        feature_cols=cell["model_feature_cols"],
        k=3,
        horizon=int(horizon),
        window_size=10,
        learning_rate=STAGE_A_LEARNING_RATE,
        source_epochs=int(epochs),
        batch_size=int(batch_size),
        tau_k=1.0,
        group_cols=cell["group_cols"],
        k3_selection_context=cell["k3_selection_context"],
        k3_raw_source_context=cell["k3_raw_source_context"],
        transformation_reuse_context=cell["transformation_reuse_context"],
        seed=int(seed),
    )
    bundle = prepare_shared_target_bundle(
        stage_a,
        cell["target_model_df"],
        cell["model_feature_cols"],
        horizon=int(horizon),
        window_size=10,
        learning_rate=STAGE_A_LEARNING_RATE,
        seed=int(seed),
        transformation_reuse_context=cell["transformation_reuse_context"],
    )
    return stage_a, bundle


def _run_arm(
    *,
    identity: Any,
    spec: Any,
    stage_a: Any,
    bundle: Any,
    epochs: int,
    batch_size: int,
    metric_protocol: Optional[dict],
    pred_root: Path,
    parsed: Mapping[str, Any],
) -> dict:
    """Run one V3 arm through existing training and metric authorities."""
    import numpy as np
    import pandas as pd

    from src.evaluation.metric_contract import ORIGINAL_SALES_SPACE
    from src.protocols.reproducibility import set_protocol_seed
    from src.transfer_methods.ms_katl_fto_three_track_robustness import RobustnessContractError
    from src.transfer_methods.ms_katl_mechanism_ablation import (
        resolve_ablation_arm,
        run_ablation_arm,
    )
    from src.transfer_methods.ms_katl_primary_target_lr_screening import (
        compute_screening_metrics,
        sales_minmax,
    )

    set_protocol_seed(int(identity.seed), include_frameworks=True)
    ablation = resolve_ablation_arm(spec.ablation_arm)
    arm = run_ablation_arm(
        ablation,
        bundle,
        epochs=int(epochs),
        batch_size=int(batch_size),
        gamma=0.5,
        use_adversarial=False,
        metric_protocol=metric_protocol,
        fine_tune_learning_rate=float(identity.selected_lr),
        **spec.run_ablation_kwargs(),
    )
    metrics = compute_screening_metrics(
        arm["y_true"],
        arm["y_pred"],
        target_scaler=bundle.target_scaler,
        feature_columns=bundle.target_feature_columns,
    )
    if metrics["metric_space"] != ORIGINAL_SALES_SPACE:
        raise RobustnessContractError(
            f"metric space drifted to {metrics['metric_space']}; original_sales_space required"
        )
    sales_min, sales_max = sales_minmax(bundle.target_scaler, bundle.target_feature_columns)
    pred_dir = (
        pred_root
        / f"mode={identity.mode}"
        / identity.arm
        / identity.dataset
        / parsed["prediction_dirname"]
        / f"h{int(identity.horizon)}_s{int(identity.seed)}_{identity.selected_lr_label}"
    )
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / "predictions.csv"
    pd.DataFrame(
        {
            "y_true_normalized": np.asarray(arm["y_true"]).reshape(-1),
            "y_pred_normalized": np.asarray(arm["y_pred"]).reshape(-1),
            "y_true_original": metrics["y_true_original"],
            "y_pred_original": metrics["y_pred_original"],
        }
    ).to_csv(pred_path, index=False, encoding="utf-8")
    source_payload = {
        "source_identities": [tuple(item) for item in stage_a.source_identities],
        "source_distances": list(stage_a.source_distances),
        "source_model_digests": list(stage_a.source_weight_digests),
        "source_scaler_digests": list(stage_a.source_scaler_digests),
        "fused_init_digest": bundle.fused_target_digest,
        "target_scaler_digest": bundle.target_split_digest.get("target_scaler"),
        "target_split_digest": dict(bundle.target_split_digest),
    }
    return {
        "status": "OK",
        "arm": identity.arm,
        "dataset": identity.dataset,
        "target_identity": identity.target_identity,
        "mode": identity.mode,
        "horizon": int(identity.horizon),
        "seed": int(identity.seed),
        "epochs": int(epochs),
        "learning_rate": float(identity.selected_lr),
        "learning_rate_label": identity.selected_lr_label,
        "method": spec.method_display,
        "rmse_original": metrics["rmse_original"],
        "smape_original": metrics["smape_original"],
        "rmse_normalized": metrics["rmse_normalized"],
        "smape_normalized": metrics["smape_normalized"],
        "metric_space": metrics["metric_space"],
        "sales_min": sales_min if sales_min is not None else metrics["sales_min"],
        "sales_max": sales_max if sales_max is not None else metrics["sales_max"],
        "architecture_digest": spec.architecture_digest,
        "architecture": dict(spec.architecture),
        "training_recipe": dict(spec.training_recipe),
        "revin": spec.revin_provenance(),
        "revin_enabled": spec.revin_enabled,
        "pass_training_recipe": spec.pass_training_recipe,
        "prediction_artifact_path": str(pred_path),
        "stage_a_shared": "YES",
        "SHARED_STAGE_A_EQUIVALENCE": "PASS",
        "run_status": "OK",
        "failure_reason": "",
        **source_payload,
        "final_learning_rate": (arm.get("training_recipe") or {}).get("final_learning_rate"),
        "ema_applied": (arm.get("training_recipe") or {}).get("ema_applied"),
        "identity_tuple": list(identity.identity_tuple),
    }


def execute_authority_cell(
    cell: CellPlan,
    *,
    attempt_path: Path,
    project_root: Path,
    manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Run exactly one cell through the existing V3 formal numerical path.

    This function is never reached by plan-only mode.  It deliberately keeps
    all model/training imports local and does not implement a second algorithm.
    """
    from src.transfer_methods.ms_katl_v3_frame_assembly import assemble_v3_cell_frames
    from src.transfer_methods.ms_katl_fto_three_track_robustness import v3_arm_spec
    from src.utils.d4_d6_runtime import load_default_metric_protocol

    prepared = assemble_v3_cell_frames(
        cell.dataset,
        cell.target,
        horizon=cell.horizon,
        seed=cell.seed,
        project_root=project_root,
        mode=SCENARIO,
    )
    parsed = prepared["parsed"]
    batch_size = int(manifest["batch_size"])
    stage_a, bundle = _run_stage_a(
        cell=prepared,
        horizon=cell.horizon,
        seed=cell.seed,
        epochs=int(manifest["source_epochs"]),
        batch_size=batch_size,
    )
    if int(getattr(stage_a, "stage_a_seed", cell.seed)) != cell.seed:
        raise CorrectedSITAContractError("formal Stage-A authority did not preserve exact cell seed")
    identity = type("CorrectedSITAIdentity", (), {})()
    identity.arm = "v3"
    identity.dataset = cell.dataset
    identity.target_identity = cell.target
    identity.mode = SCENARIO
    identity.horizon = cell.horizon
    identity.seed = cell.seed
    identity.selected_lr = float(_parse_lr_label(cell.target_lr_label))
    identity.selected_lr_label = cell.target_lr_label
    identity.identity_tuple = ("v3", cell.dataset, cell.target, SCENARIO, cell.horizon, cell.seed, cell.target_lr_label)
    spec = v3_arm_spec()
    if spec.method_display != RUNTIME_METHOD_ID or spec.key != "v3":
        raise CorrectedSITAContractError("formal V3 authority identity drifted")
    payload = _run_arm(
        identity=identity,
        spec=spec,
        stage_a=stage_a,
        bundle=bundle,
        epochs=int(manifest["target_epochs"]),
        batch_size=batch_size,
        metric_protocol=load_default_metric_protocol(project_root),
        pred_root=attempt_path / "predictions",
        parsed=parsed,
    )
    payload["source_weights"] = list(getattr(stage_a, "source_weights", []))
    payload["stage_a_seed"] = cell.seed
    return payload


def _parse_lr_label(label: str) -> float:
    mapping = {"3e-5": 3e-5, "1e-4": 1e-4, "3e-4": 3e-4}
    try:
        return float(mapping[label])
    except KeyError as exc:
        raise CorrectedSITAContractError(f"unsupported formal LR label: {label}") from exc


def summarize_attempts(run_root: Path, cells: Sequence[CellPlan]) -> Dict[str, Any]:
    counts = {"COMPLETE": 0, "FAILED": 0, "INTERRUPTED": 0, "PENDING": 0, "SKIPPED_COMPLETE": 0}
    rows: List[Dict[str, Any]] = []
    for cell in cells:
        cell_root = Path(run_root) / "cells" / cell.cell_key
        latest = _latest_attempt(cell_root)
        if latest is None:
            counts["PENDING"] += 1
            rows.append({"cell_key": cell.cell_key, "dataset": cell.dataset, "target": cell.target, "horizon": cell.horizon, "seed": cell.seed, "status": "PENDING"})
            continue
        status = str(_read_json(latest / "status.json").get("status") or "PENDING").upper()
        if status not in counts:
            status = "PENDING"
        counts[status] += 1
        rows.append({"cell_key": cell.cell_key, "dataset": cell.dataset, "target": cell.target, "horizon": cell.horizon, "seed": cell.seed, "status": status, "attempt_id": latest.name})
    return {"TOTAL": len(cells), **counts, "coverage": coverage_summary(cells), "rows": rows}


def write_summary(run_root: Path, summary: Mapping[str, Any]) -> None:
    _write_json(Path(run_root) / "run_summary.json", summary)
    rows = summary.get("rows") or []
    fieldnames = ("cell_key", "dataset", "target", "horizon", "seed", "status", "attempt_id")
    with (Path(run_root) / "run_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fieldnames} for row in rows)


__all__ = [
    "CellPlan", "CorrectedSITAContractError", "CRITICAL_RUNTIME_FILES", "DATASETS",
    "HORIZONS", "SEEDS", "TARGETS", "TARGET_LR_LABELS", "SOURCE_LR_LABEL",
    "assert_formal_target_authority", "assert_output_isolation", "build_digests",
    "build_result_payload", "canonical_matrix_csv", "coverage_summary", "create_manifest",
    "critical_runtime_digest", "enumerate_cells", "execute_authority_cell", "finalize_attempt",
    "load_manifest", "matrix_digest", "prepare_cell_attempt", "protocol_digest", "protocol_payload",
    "read_git_identity", "resolve_output_root", "summarize_attempts", "validate_authority_against_runtime",
    "validate_result_contract", "write_summary", "working_tree_protocol_digest",
]
