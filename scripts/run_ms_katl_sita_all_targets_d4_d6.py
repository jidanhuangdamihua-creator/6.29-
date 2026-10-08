"""Enumerate D4–D6 canonical targets through frozen SITA / FUSED_TARGET_ONLY_V3.

This wrapper does not change the SITA model, training recipe, or data protocol.
It only expands the confirmatory V3-without matrix from frozen primary targets
to all canonical D4/D5/D6 candidates. Existing three-track / screening commands
are unchanged.

Usage:
    python scripts/run_ms_katl_sita_all_targets_d4_d6.py --dry-run
    python scripts/run_ms_katl_sita_all_targets_d4_d6.py --output-dir outputs/sita_all_targets_d4_d6_5seed
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tf_compat  # noqa: F401

from scripts.run_ms_katl_fto_three_track_robustness import (
    _append_jsonl,
    _failed_cell_payload,
    _run_arm,
    _run_stage_a,
)
from scripts.run_ms_katl_fto_v2_screening import _csv_ready, _dump_csv
from scripts.run_ms_katl_primary_target_lr_screening import (
    PROJECT_ROOT,
    _load_d46_pool,
    _prepare_d46_target_cell,
)
from src.transfer_methods.ms_katl_fto_three_track_robustness import (
    cell_path,
    classify_cell_artifact,
    jsonable,
    load_valid_ok_cells,
    robustness_arm_registry,
    write_cell_artifact,
)
from src.transfer_methods.ms_katl_primary_target_lr_screening import (
    SCREENING_EPOCHS,
    parse_target_identity,
)
from src.transfer_methods.ms_katl_sita_all_targets_d4_d6 import (
    ALL_TARGETS_DATASETS,
    ALL_TARGETS_HORIZONS,
    ALL_TARGETS_LABEL,
    ALL_TARGETS_SEEDS,
    D5_MAX_JOBS,
    EXPECTED_TOTAL_CELLS,
    LR_POLICY,
    MAX_JOBS,
    PAPER_SITA_MODE,
    SITA_FORMAL_CONFIG,
    SITA_FORMAL_MANIFEST_RELPATH,
    SITA_FORMAL_RUNNER,
    TARGET_POLICY,
    WORKERS,
    AllTargetsContractError,
    assert_output_does_not_clobber_frozen,
    assert_sita_identity_frozen,
    build_all_target_identities,
    cell_plan_rows,
    completion_guard_225,
    default_output_dir,
    format_dry_run,
    result_row_with_required_fields,
    source_authority_path,
    validate_all_target_matrix,
)
from src.utils.d4_d6_runtime import load_default_metric_protocol


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen SITA / FUSED_TARGET_ONLY_V3 all-target D4-D6 confirmatory matrix."
    )
    parser.add_argument("--dry-run", "--preflight", action="store_true", dest="dry_run")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Independent output root. Default: outputs/sita_all_targets_d4_d6_5seed",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=MAX_JOBS,
        help="Must remain 1. Inherited from the sequential three-track SITA runner.",
    )
    parser.add_argument(
        "--d5-max-jobs",
        type=int,
        default=D5_MAX_JOBS,
        help="Must remain 1. D5 sealed source loading is not parallel-safe here.",
    )
    return parser.parse_args()


def _resolve_output_dir(requested: Optional[Path]) -> Path:
    if requested is None:
        run_dir = default_output_dir(PROJECT_ROOT)
    else:
        run_dir = Path(requested)
        if not run_dir.is_absolute():
            run_dir = PROJECT_ROOT / run_dir
    return assert_output_does_not_clobber_frozen(run_dir, PROJECT_ROOT)


def _assert_serial_caps(max_jobs: int, d5_max_jobs: int) -> None:
    if int(max_jobs) != MAX_JOBS:
        raise AllTargetsContractError(
            f"MAX_JOBS must remain {MAX_JOBS} for this SITA wrapper, got {max_jobs}"
        )
    if int(d5_max_jobs) != D5_MAX_JOBS:
        raise AllTargetsContractError(
            f"D5_MAX_JOBS must remain {D5_MAX_JOBS}, got {d5_max_jobs}"
        )


def _write_dry_run_manifest(
    run_dir: Path,
    identities: Sequence[Any],
    report: Mapping[str, Any],
    identity_flags: Mapping[str, Any],
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "label": ALL_TARGETS_LABEL,
        "sita_formal_runner": SITA_FORMAL_RUNNER,
        "sita_formal_config": SITA_FORMAL_CONFIG,
        "sita_formal_manifest": str(PROJECT_ROOT / SITA_FORMAL_MANIFEST_RELPATH),
        "mode": PAPER_SITA_MODE,
        "method": SITA_FORMAL_CONFIG,
        "target_policy": TARGET_POLICY,
        "lr_policy": LR_POLICY,
        "sita_model_changed": False,
        "sita_training_protocol_changed": False,
        "data_protocol_changed": False,
        "frozen_results_overwritten": False,
        "max_jobs": MAX_JOBS,
        "d5_max_jobs": D5_MAX_JOBS,
        "workers": WORKERS,
        "identity_flags": dict(identity_flags),
        "matrix": dict(report),
        "cells": cell_plan_rows(identities, project_root=PROJECT_ROOT, run_dir=run_dir),
    }
    path = run_dir / "dry_run_manifest.json"
    path.write_text(json.dumps(jsonable(payload), indent=2) + "\n", encoding="utf-8")
    return path


def _write_aggregates(
    results_dir: Path,
    identities: Sequence[Any],
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    results_dir.mkdir(parents=True, exist_ok=True)
    ready_rows = [_csv_ready(result_row_with_required_fields(row)) for row in rows]
    all_runs_path = results_dir / "sita_all_targets_all_runs.csv"
    _dump_csv(all_runs_path, ready_rows)
    guard = completion_guard_225(identities, rows)
    guard_path = results_dir / "sita_all_targets_completion_guard.json"
    guard_path.write_text(json.dumps(guard, indent=2), encoding="utf-8")
    return {
        "guard": guard,
        "paths": {"all_runs": all_runs_path, "completion_guard": guard_path},
    }


def main() -> None:
    args = _parse_args()
    _assert_serial_caps(args.max_jobs, args.d5_max_jobs)
    identity_flags = assert_sita_identity_frozen(PROJECT_ROOT)
    identities = build_all_target_identities(project_root=PROJECT_ROOT)
    report = validate_all_target_matrix(identities)
    run_dir = _resolve_output_dir(args.output_dir)
    print(format_dry_run(
        identities=identities,
        report=report,
        run_dir=run_dir,
        identity_flags=identity_flags,
    ), end="")
    manifest_path = _write_dry_run_manifest(run_dir, identities, report, identity_flags)
    print(f"DRY_RUN_MANIFEST={manifest_path}")
    if args.dry_run:
        if int(report["DRY_RUN_CELL_COUNT"]) != EXPECTED_TOTAL_CELLS:
            raise SystemExit("DRY_RUN_CELL_COUNT != 225")
        return

    if int(report["DRY_RUN_CELL_COUNT"]) != EXPECTED_TOTAL_CELLS:
        raise AllTargetsContractError("refusing full run because dry-run cell count is not 225")

    registry = robustness_arm_registry()
    v3_spec = registry["v3"]
    results_dir = run_dir / "results"
    pred_root = run_dir / "predictions"
    cells_dir = run_dir / "cells"
    logs_dir = run_dir / "logs"
    for path in (results_dir, pred_root, cells_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "label": ALL_TARGETS_LABEL,
                "sita_formal_runner": SITA_FORMAL_RUNNER,
                "sita_formal_config": SITA_FORMAL_CONFIG,
                "method": SITA_FORMAL_CONFIG,
                "arm": "v3",
                "mode": PAPER_SITA_MODE,
                "source_mode": PAPER_SITA_MODE,
                "fused_init_source_mode": PAPER_SITA_MODE,
                "datasets": list(ALL_TARGETS_DATASETS),
                "horizons": list(ALL_TARGETS_HORIZONS),
                "seeds": list(ALL_TARGETS_SEEDS),
                "epochs": SCREENING_EPOCHS,
                "expected_cells": EXPECTED_TOTAL_CELLS,
                "target_policy": TARGET_POLICY,
                "lr_policy": LR_POLICY,
                "max_jobs": MAX_JOBS,
                "d5_max_jobs": D5_MAX_JOBS,
                "workers": WORKERS,
                "sita_model_changed": False,
                "sita_training_protocol_changed": False,
                "data_protocol_changed": False,
                "frozen_results_overwritten": False,
                "architecture_digest": identity_flags["architecture_digest"],
                "v3_frozen_manifest": str(PROJECT_ROOT / SITA_FORMAL_MANIFEST_RELPATH),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    completed = load_valid_ok_cells(cells_dir, identities, registry)
    pending = [item for item in identities if item.identity_tuple not in completed]
    print(f"RESUME_OK_CELLS={len(completed)}")
    print(f"PENDING_CELLS={len(pending)}")

    grouped: Dict[tuple, List[Any]] = defaultdict(list)
    for identity in pending:
        grouped[identity.paired_tuple].append(identity)

    d46_data_cache: Dict[str, dict] = {}
    metric_protocol = load_default_metric_protocol(PROJECT_ROOT)
    log_path = logs_dir / "cells.jsonl"

    for paired_key, arm_identities in grouped.items():
        dataset, target_identity, mode, horizon, seed = paired_key
        if dataset not in ALL_TARGETS_DATASETS:
            raise AllTargetsContractError(f"unexpected dataset in execute loop: {dataset}")
        if mode != PAPER_SITA_MODE:
            raise AllTargetsContractError(f"paper SITA mode must remain without, got {mode}")
        parsed = parse_target_identity(dataset, target_identity)
        print(
            f"\n=== SITA {dataset} {target_identity} MODE={mode} H={horizon} S={seed} ==="
        )
        try:
            if dataset not in d46_data_cache:
                d46_data_cache[dataset] = _load_d46_pool(dataset, SCREENING_EPOCHS)
            pool = d46_data_cache[dataset]
            cell_pool = {
                **pool,
                "config": {
                    **pool["config"],
                    "horizon": int(horizon),
                    "random_state": int(seed),
                },
            }
            cell = _prepare_d46_target_cell(dataset, cell_pool, parsed)
            batch_size = int(cell_pool["config"]["batch_size"])
            stage_a, bundle = _run_stage_a(
                cell=cell,
                horizon=int(horizon),
                seed=int(seed),
                epochs=SCREENING_EPOCHS,
                batch_size=batch_size,
            )
            for identity in arm_identities:
                existing = cell_path(cells_dir, identity)
                if classify_cell_artifact(existing, identity, v3_spec) == "OK":
                    print(f"SKIP_OK {identity.cell_id}")
                    continue
                try:
                    payload = _run_arm(
                        identity=identity,
                        spec=v3_spec,
                        stage_a=stage_a,
                        bundle=bundle,
                        epochs=SCREENING_EPOCHS,
                        batch_size=batch_size,
                        metric_protocol=metric_protocol,
                        pred_root=pred_root,
                        parsed=parsed,
                    )
                except Exception as exc:
                    payload = _failed_cell_payload(
                        identity, v3_spec, f"{type(exc).__name__}: {exc}"
                    )
                    print(f"CELL_FAILED {identity.cell_id} reason={exc}")
                payload.update(
                    {
                        "source_mode": identity.mode,
                        "fused_init_source_mode": identity.mode,
                        "source_authority_identity": str(
                            source_authority_path(identity.dataset, PROJECT_ROOT)
                        ),
                        "target": identity.target_identity,
                        "RMSE": payload.get("rmse_original"),
                        "sMAPE": payload.get("smape_original"),
                    }
                )
                write_cell_artifact(cell_path(cells_dir, identity), payload)
                _append_jsonl(
                    log_path,
                    {
                        "cell_id": identity.cell_id,
                        "status": payload["status"],
                        "dataset": identity.dataset,
                        "target_identity": identity.target_identity,
                        "mode": identity.mode,
                        "horizon": identity.horizon,
                        "seed": identity.seed,
                    },
                )
                if payload["status"] == "OK":
                    completed[identity.identity_tuple] = payload
                    print(
                        f"CELL_OK {identity.cell_id} RMSE={payload['rmse_original']:.6g} "
                        f"sMAPE={payload['smape_original']:.6g}"
                    )
        except Exception as exc:
            for identity in arm_identities:
                payload = _failed_cell_payload(identity, v3_spec, f"{type(exc).__name__}: {exc}")
                payload.update(
                    {
                        "source_mode": identity.mode,
                        "fused_init_source_mode": identity.mode,
                        "source_authority_identity": str(
                            source_authority_path(identity.dataset, PROJECT_ROOT)
                        ),
                        "target": identity.target_identity,
                        "RMSE": None,
                        "sMAPE": payload.get("smape_original"),
                    }
                )
                write_cell_artifact(cell_path(cells_dir, identity), payload)
                _append_jsonl(log_path, {"cell_id": identity.cell_id, "status": "FAILED"})
            print(f"CELL_GROUP_FAILED {paired_key} reason={exc}")

    all_rows: List[dict] = []
    for identity in identities:
        path = cell_path(cells_dir, identity)
        if path.is_file():
            all_rows.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            all_rows.append(_failed_cell_payload(identity, v3_spec, "MISSING"))
    aggregates = _write_aggregates(results_dir, identities, all_rows)
    guard = aggregates["guard"]
    print("\n============================================================")
    print(f"EXPECTED_TOTAL_CELLS={guard['EXPECTED_TOTAL_CELLS']}")
    print(f"OK_RUNS={guard['OK_RUNS']}")
    print(f"FAILED_RUNS={guard['FAILED_RUNS']}")
    print(f"DUPLICATE_RUNS={guard['DUPLICATE_RUNS']}")
    print(f"MISSING_RUNS={guard['MISSING_RUNS']}")
    print(f"SITA_ALL_TARGETS_COMPLETE={guard['SITA_ALL_TARGETS_COMPLETE']}")
    print(f"Run root: {run_dir}")
    print(f"All runs: {aggregates['paths']['all_runs']}")
    print("============================================================")
    if guard["SITA_ALL_TARGETS_COMPLETE"] != "YES":
        raise SystemExit("SITA_ALL_TARGETS_COMPLETE=NO")


if __name__ == "__main__":
    main()
