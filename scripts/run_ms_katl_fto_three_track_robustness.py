"""MS-KATL FUSED_TARGET_ONLY three-track paired robustness runner.

Confirmatory evaluation: frozen V1 / V2 / V3 on the V3 primary-target set.
Does not reselect targets or learning rates, does not modify FORMAL_METHODS,
and does not overwrite V1/V2/V3 screening outputs.

Usage:
    python scripts/run_ms_katl_fto_three_track_robustness.py --preflight
    python scripts/run_ms_katl_fto_three_track_robustness.py --smoke
    python scripts/run_ms_katl_fto_three_track_robustness.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tf_compat  # noqa: F401

import numpy as np
import pandas as pd

from paper_reproduction_protocol import load_paper_protocol, resolve_strict_paper_mode
from scripts.run_full_paper_experiments import _load_config
from scripts.run_ms_katl_fto_v2_screening import _csv_ready, _dump_csv
from scripts.run_ms_katl_primary_target_lr_screening import (
    PROJECT_ROOT,
    _load_d13_pool,
    _load_d46_pool,
    _prepare_d13_target_cell,
    _prepare_d46_target_cell,
)
from src.evaluation.metric_contract import ORIGINAL_SALES_SPACE
from src.protocols.reproducibility import set_protocol_seed
from src.transfer_methods.ms_katl_fto_three_track_robustness import (
    EXPECTED_RUNS,
    PAIRED_CELLS,
    ROBUSTNESS_ARMS,
    ROBUSTNESS_HORIZONS,
    ROBUSTNESS_LABEL,
    ROBUSTNESS_SEEDS,
    ROBUSTNESS_TRACK,
    RunIdentity,
    RobustnessArmSpec,
    RobustnessContractError,
    assert_formal_methods_unchanged,
    assert_not_screening_output_dir,
    assert_stage_a_is_arm_independent,
    build_run_identities,
    cell_path,
    classify_cell_artifact,
    common_targets_from_frozen_manifest,
    completion_guard,
    expected_full_matrix_counts,
    format_preflight,
    jsonable,
    load_target_lr_contract_table,
    load_valid_ok_cells,
    load_v3_frozen_robustness_manifest,
    optional_paired_wilcoxon,
    pair_completed_runs,
    parse_arm_keys,
    parse_dataset_ids,
    parse_int_tuple,
    robustness_arm_registry,
    smoke_guard,
    summarize_by_dataset,
    summarize_paired_rows,
    v3_frozen_manifest_path,
    write_cell_artifact,
    write_v3_frozen_robustness_manifest,
)
from src.transfer_methods.ms_katl_v3_frame_assembly import v3_knn_authority_path
from src.transfer_methods.ms_katl_lite import prepare_ms_katl_stage_a
from src.transfer_methods.ms_katl_mechanism_ablation import (
    prepare_shared_target_bundle,
    resolve_ablation_arm,
    run_ablation_arm,
)
from src.transfer_methods.ms_katl_primary_target_lr_screening import (
    SCREENING_EPOCHS,
    SCREENING_MODE,
    STAGE_A_LEARNING_RATE,
    compute_screening_metrics,
    parse_target_identity,
    sales_minmax,
)
from src.utils.d4_d6_runtime import load_default_metric_protocol
from src.utils.run_utils import create_run_dir


V3_COMPLETION_SEEDS = (45, 46)
EXPLICIT_SEED_ALLOWLIST = ROBUSTNESS_SEEDS + V3_COMPLETION_SEEDS
FROZEN_WITHOUT_RESULT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "runs"
    / "20260818_190337_640037_MS_KATL_FTO_THREE_TRACK_ROBUSTNESS"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FUSED_TARGET_ONLY V1/V2/V3 three-track paired robustness."
    )
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Bounded wiring check: D1 common target, H=1, seed=42, arms=v1,v2,v3.",
    )
    parser.add_argument("--arms", default=None, help="Comma-separated v1,v2,v3. Default all.")
    parser.add_argument("--datasets", default=None, help="Comma-separated D1-D6. Default all.")
    parser.add_argument("--horizons", default=None, help="Comma-separated 1,3,5. Default all.")
    parser.add_argument(
        "--seeds",
        default=None,
        help=(
            "Comma-separated seeds. Default remains the frozen robustness set 42,43,44; "
            "explicit V3-only completion may additionally use 45,46."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--freeze-v3",
        action="store_true",
        help="Write configs/ms_katl_fto_v3_robustness_manifest.json and exit.",
    )
    return parser.parse_args()


def _resolve_matrix(args: argparse.Namespace) -> Dict[str, Any]:
    if args.smoke:
        if args.datasets not in {None, "d1", "D1"}:
            raise RobustnessContractError("smoke is fixed to D1")
        if args.horizons not in {None, "1"}:
            raise RobustnessContractError("smoke is fixed to horizon=1")
        if args.seeds not in {None, "42"}:
            raise RobustnessContractError("smoke is fixed to seed=42")
        if args.arms not in {None, "v1,v2,v3", "V1,V2,V3"}:
            raise RobustnessContractError("smoke is fixed to arms=v1,v2,v3")
        return {
            "arms": ROBUSTNESS_ARMS,
            "datasets": ("D1",),
            "horizons": (1,),
            "seeds": (42,),
            "smoke": True,
        }
    arms = parse_arm_keys(args.arms)
    seeds = (
        ROBUSTNESS_SEEDS
        if args.seeds is None or str(args.seeds).strip().lower() in {"", "all"}
        else parse_int_tuple(args.seeds, EXPLICIT_SEED_ALLOWLIST, name="seed")
    )
    if any(seed in V3_COMPLETION_SEEDS for seed in seeds) and arms != ("v3",):
        raise RobustnessContractError(
            "seed45/46 completion is authorized only for the frozen V3 arm"
        )
    return {
        "arms": arms,
        "datasets": parse_dataset_ids(args.datasets),
        "horizons": parse_int_tuple(args.horizons, ROBUSTNESS_HORIZONS, name="horizon"),
        "seeds": seeds,
        "smoke": False,
    }


def _planned_output_dir(requested: Optional[Path]) -> Optional[Path]:
    if requested is None:
        return None
    run_dir = Path(requested)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    return run_dir.resolve()


def _assert_not_frozen_result_root(run_dir: Path) -> None:
    frozen = FROZEN_WITHOUT_RESULT_ROOT.resolve()
    resolved = Path(run_dir).resolve()
    if resolved == frozen or frozen in resolved.parents:
        raise RobustnessContractError(
            f"refusing to write seed completion inside frozen result root: {frozen}"
        )


def _source_authority(identity: RunIdentity) -> Path:
    if identity.mode != "without":
        raise RobustnessContractError("three-track completion source mode must be without")
    return v3_knn_authority_path(identity.dataset, identity.mode).resolve()


def _format_completion_cell_manifest(
    identities: Sequence[RunIdentity], planned_run_dir: Optional[Path]
) -> str:
    if planned_run_dir is None:
        return "PREFLIGHT_OUTPUT_PATHS=DEFERRED_UNTIL_RUN_ROOT_CREATION\n"
    _assert_not_frozen_result_root(planned_run_dir)
    lines = [
        "",
        "V3_COMPLETION_CELL_MANIFEST_BEGIN",
        "SOURCE_MODE=without",
        "FUSED_INIT_SOURCE_MODE=without",
    ]
    for index, identity in enumerate(identities, start=1):
        lines.extend(
            [
                f"CELL_{index}_DATASET={identity.dataset}",
                f"CELL_{index}_TARGET={identity.target_identity}",
                f"CELL_{index}_MODE={identity.mode}",
                f"CELL_{index}_HORIZON={identity.horizon}",
                f"CELL_{index}_SEED={identity.seed}",
                f"CELL_{index}_SELECTED_LR={identity.selected_lr_label}",
                f"CELL_{index}_SOURCE_AUTHORITY={_source_authority(identity)}",
                f"CELL_{index}_OUTPUT_PATH={cell_path(planned_run_dir / 'cells', identity)}",
            ]
        )
    lines.append("V3_COMPLETION_CELL_MANIFEST_END")
    return "\n".join(lines) + "\n"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frozen_v3_tree_sha256() -> Tuple[str, int]:
    root = FROZEN_WITHOUT_RESULT_ROOT.resolve()
    paths = sorted({*root.glob("cells/v3_*.json"), *root.glob("predictions/v3/**/*")})
    files = [path for path in paths if path.is_file()]
    if not files:
        raise RobustnessContractError(f"frozen V3 result tree is empty: {root}")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest(), len(files)


def _frozen_protection_payload() -> Dict[str, Any]:
    selection_manifest = (
        PROJECT_ROOT
        / "outputs"
        / "runs"
        / "20260818_180250_355272_MS_KATL_FTO_V3_SCREENING"
        / "results"
        / "v3_selection_manifest.json"
    ).resolve()
    robustness_manifest = v3_frozen_manifest_path(PROJECT_ROOT).resolve()
    for path in (selection_manifest, robustness_manifest):
        if not path.is_file():
            raise RobustnessContractError(f"frozen protection artifact missing: {path}")
    tree_sha, tree_files = _frozen_v3_tree_sha256()
    return {
        "selection_manifest": str(selection_manifest),
        "selection_manifest_sha256": _sha256_file(selection_manifest),
        "robustness_manifest": str(robustness_manifest),
        "robustness_manifest_sha256": _sha256_file(robustness_manifest),
        "existing_v3_result_root": str(FROZEN_WITHOUT_RESULT_ROOT.resolve()),
        "existing_v3_result_tree_sha256": tree_sha,
        "existing_v3_result_tree_file_count": tree_files,
        "tree_scope": "cells/v3_*.json plus predictions/v3 recursive",
    }


def _write_frozen_protection(run_dir: Path, payload: Mapping[str, Any]) -> Path:
    path = Path(run_dir) / "frozen_protection.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != dict(payload):
            raise RobustnessContractError(
                "frozen protection identity changed since this completion root was created"
            )
        return path
    path.write_text(json.dumps(dict(payload), indent=2) + "\n", encoding="utf-8")
    return path


def _ensure_frozen_v3(project_root: Path) -> Dict[str, Any]:
    path = v3_frozen_manifest_path(project_root)
    if not path.is_file():
        write_v3_frozen_robustness_manifest(project_root)
    return load_v3_frozen_robustness_manifest(project_root)


def _prepare_d13_cfg(horizon: int, seed: int, epochs: int) -> Tuple[dict, dict]:
    cfg = deepcopy(_load_config())
    cfg.setdefault("single_experiment", {})
    cfg["single_experiment"]["horizon"] = int(horizon)
    cfg["single_experiment"]["seed"] = int(seed)
    cfg["single_experiment"]["source_epochs"] = int(epochs)
    cfg["single_experiment"]["target_epochs"] = int(epochs)
    protocol = load_paper_protocol(cfg)
    protocol["strict_paper_mode"] = True
    protocol.setdefault("metric_protocol", {})["strict_paper_metrics"] = True
    resolve_strict_paper_mode(cfg, explicit=True)
    return cfg, protocol


def _resolve_output_dir(requested: Optional[Path], *, smoke: bool) -> Path:
    if requested is None:
        label = ROBUSTNESS_LABEL + ("_SMOKE" if smoke else "")
        run_dir = create_run_dir(PROJECT_ROOT, label=label)
        assert_not_screening_output_dir(run_dir)
        return run_dir
    run_dir = Path(requested)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    _assert_not_frozen_result_root(run_dir)
    assert_not_screening_output_dir(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(jsonable(dict(payload)), default=str) + "\n")


def _failed_cell_payload(identity: RunIdentity, spec: RobustnessArmSpec, reason: str) -> dict:
    return {
        "status": "FAILED",
        "arm": identity.arm,
        "dataset": identity.dataset,
        "target_identity": identity.target_identity,
        "mode": identity.mode,
        "horizon": int(identity.horizon),
        "seed": int(identity.seed),
        "learning_rate": float(identity.selected_lr),
        "learning_rate_label": identity.selected_lr_label,
        "method": spec.method_display,
        "rmse_original": None,
        "smape_original": None,
        "metric_space": ORIGINAL_SALES_SPACE,
        "architecture_digest": spec.architecture_digest,
        "training_recipe": dict(spec.training_recipe),
        "revin": spec.revin_provenance(),
        "failure_reason": str(reason),
        "run_status": "FAILED",
    }


def _run_stage_a(
    *,
    cell: dict,
    horizon: int,
    seed: int,
    epochs: int,
    batch_size: int,
) -> Tuple[Any, Any]:
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
    identity: RunIdentity,
    spec: RobustnessArmSpec,
    stage_a: Any,
    bundle: Any,
    epochs: int,
    batch_size: int,
    metric_protocol: Optional[dict],
    pred_root: Path,
    parsed: dict,
) -> dict:
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


def _write_aggregates(
    results_dir: Path,
    identities: Sequence[RunIdentity],
    rows: Sequence[Mapping[str, Any]],
    *,
    smoke: bool,
) -> Dict[str, Any]:
    results_dir.mkdir(parents=True, exist_ok=True)
    all_runs_path = results_dir / "three_track_robustness_all_runs.csv"
    paired_path = results_dir / "three_track_paired_by_cell.csv"
    dataset_path = results_dir / "three_track_summary_by_dataset.csv"
    global_path = results_dir / "three_track_summary_global.csv"
    _dump_csv(all_runs_path, [_csv_ready(dict(row)) for row in rows])
    paired = pair_completed_runs(rows)
    _dump_csv(paired_path, paired)
    dataset_summary = summarize_by_dataset(paired)
    _dump_csv(dataset_path, dataset_summary)
    global_summary = summarize_paired_rows(paired, scope="global")
    _dump_csv(global_path, [global_summary])
    wilcoxon = optional_paired_wilcoxon(paired)
    (results_dir / "three_track_wilcoxon.json").write_text(
        json.dumps(wilcoxon, indent=2, default=str), encoding="utf-8"
    )
    guard = completion_guard(identities, rows)
    (results_dir / "three_track_completion_guard.json").write_text(
        json.dumps(guard, indent=2), encoding="utf-8"
    )
    smoke_payload = smoke_guard(rows, paired) if smoke else None
    if smoke_payload is not None:
        (results_dir / "three_track_smoke_guard.json").write_text(
            json.dumps(smoke_payload, indent=2), encoding="utf-8"
        )
    return {
        "paired": paired,
        "dataset_summary": dataset_summary,
        "global_summary": global_summary,
        "wilcoxon": wilcoxon,
        "guard": guard,
        "smoke": smoke_payload,
        "paths": {
            "all_runs": all_runs_path,
            "paired": paired_path,
            "dataset_summary": dataset_path,
            "global_summary": global_path,
        },
    }


def main() -> None:
    args = _parse_args()
    assert_formal_methods_unchanged()
    if args.freeze_v3:
        path = write_v3_frozen_robustness_manifest(PROJECT_ROOT)
        print(f"V3_FROZEN_MANIFEST={path}")
        print("selection_frozen=true")
        print("READY_TO_FREEZE_V3_AND_ENTER_ROBUSTNESS=YES")
        return

    frozen = _ensure_frozen_v3(PROJECT_ROOT)
    common_targets = common_targets_from_frozen_manifest(frozen)
    matrix = _resolve_matrix(args)
    lr_table = load_target_lr_contract_table(PROJECT_ROOT, common_targets=common_targets)
    identities = build_run_identities(
        arms=matrix["arms"],
        datasets=matrix["datasets"],
        horizons=matrix["horizons"],
        seeds=matrix["seeds"],
        lr_table=lr_table,
    )
    protection = _frozen_protection_payload()
    counts = expected_full_matrix_counts(
        matrix["arms"], matrix["datasets"], matrix["horizons"], matrix["seeds"]
    )
    if not matrix["smoke"] and tuple(matrix["arms"]) == ROBUSTNESS_ARMS and tuple(
        matrix["datasets"]
    ) == tuple(frozen["datasets"]) and tuple(matrix["horizons"]) == ROBUSTNESS_HORIZONS and tuple(
        matrix["seeds"]
    ) == ROBUSTNESS_SEEDS:
        if counts["expected_runs"] != EXPECTED_RUNS or counts["paired_cells"] != PAIRED_CELLS:
            raise RobustnessContractError(
                f"full matrix must be {EXPECTED_RUNS} runs / {PAIRED_CELLS} paired cells, got {counts}"
            )
        if len(identities) != EXPECTED_RUNS:
            raise RobustnessContractError(f"UNIQUE_IDENTITIES={len(identities)} != {EXPECTED_RUNS}")
    if len(set(item.identity_tuple for item in identities)) != len(identities):
        raise RobustnessContractError("duplicate cell identity in plan")

    print(
        format_preflight(
            arms=matrix["arms"],
            datasets=matrix["datasets"],
            horizons=matrix["horizons"],
            seeds=matrix["seeds"],
            lr_table=lr_table,
            counts=counts,
        ),
        end="",
    )
    planned_run_dir = _planned_output_dir(args.output_dir)
    if any(seed in V3_COMPLETION_SEEDS for seed in matrix["seeds"]):
        if planned_run_dir is None:
            raise RobustnessContractError(
                "seed45/46 completion requires an explicit isolated --output-dir"
            )
        print(_format_completion_cell_manifest(identities, planned_run_dir), end="")
        print(f"V3_SELECTION_MANIFEST={protection['selection_manifest']}")
        print(f"V3_SELECTION_MANIFEST_SHA256={protection['selection_manifest_sha256']}")
        print(f"V3_ROBUSTNESS_MANIFEST={protection['robustness_manifest']}")
        print(f"V3_ROBUSTNESS_MANIFEST_SHA256={protection['robustness_manifest_sha256']}")
        print(f"EXISTING_V3_RESULT_ROOT={protection['existing_v3_result_root']}")
        print(f"EXISTING_RESULT_TREE_SHA256={protection['existing_v3_result_tree_sha256']}")
        print(f"EXISTING_RESULT_TREE_FILE_COUNT={protection['existing_v3_result_tree_file_count']}")
    if args.preflight:
        return

    registry = robustness_arm_registry()
    shared_flag = assert_stage_a_is_arm_independent()
    run_dir = _resolve_output_dir(args.output_dir, smoke=bool(matrix["smoke"]))
    if any(seed in V3_COMPLETION_SEEDS for seed in matrix["seeds"]):
        protection_path = _write_frozen_protection(run_dir, protection)
        print(f"FROZEN_PROTECTION_RECORD={protection_path}")
    results_dir = run_dir / "results"
    pred_root = run_dir / "predictions"
    cells_dir = run_dir / "cells"
    logs_dir = run_dir / "logs"
    for path in (results_dir, pred_root, cells_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "label": ROBUSTNESS_LABEL,
                "track": ROBUSTNESS_TRACK,
                "smoke": matrix["smoke"],
                "arms": list(matrix["arms"]),
                "datasets": list(matrix["datasets"]),
                "horizons": list(matrix["horizons"]),
                "seeds": list(matrix["seeds"]),
                "epochs": SCREENING_EPOCHS,
                "expected_runs": counts["expected_runs"],
                "paired_cells": counts["paired_cells"],
                "common_target_policy": frozen["common_target_policy"],
                "lr_policy": "ARM_SPECIFIC_FROZEN_TARGET_LR",
                "robustness_reselection": False,
                "target_lr_table": lr_table["rows"],
                "v3_frozen_manifest": str(v3_frozen_manifest_path(PROJECT_ROOT)),
                "SHARED_STAGE_A_EQUIVALENCE": shared_flag,
                "formal_methods_change": False,
                "mode": identities[0].mode if identities else SCREENING_MODE,
                "source_mode": identities[0].mode if identities else SCREENING_MODE,
                "fused_init_source_mode": identities[0].mode if identities else SCREENING_MODE,
                "frozen_without_result_root": str(FROZEN_WITHOUT_RESULT_ROOT.resolve()),
                "seed45_46_completion": any(
                    seed in V3_COMPLETION_SEEDS for seed in matrix["seeds"]
                ),
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

    grouped: Dict[Tuple[Any, ...], List[RunIdentity]] = defaultdict(list)
    for identity in pending:
        grouped[identity.paired_tuple].append(identity)

    d13_pool_cache: Dict[Tuple[str, int, int], dict] = {}
    d46_data_cache: Dict[str, dict] = {}
    metric_protocol = load_default_metric_protocol(PROJECT_ROOT)
    log_path = logs_dir / "cells.jsonl"

    for paired_key, arm_identities in grouped.items():
        dataset, target_identity, mode, horizon, seed = paired_key
        parsed = parse_target_identity(dataset, target_identity)
        print(
            f"\n=== PAIRED {dataset} {target_identity} MODE={mode} H={horizon} S={seed} "
            f"arms={[item.arm for item in arm_identities]} ==="
        )
        try:
            if dataset in {"D1", "D2", "D3"}:
                cache_key = (dataset, int(horizon), int(seed))
                if cache_key not in d13_pool_cache:
                    cfg, protocol = _prepare_d13_cfg(int(horizon), int(seed), SCREENING_EPOCHS)
                    d13_pool_cache[cache_key] = _load_d13_pool(dataset, cfg, protocol)
                cell = _prepare_d13_target_cell(dataset, d13_pool_cache[cache_key], parsed)
                batch_size = 16
            else:
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
                spec = registry[identity.arm]
                existing = cell_path(cells_dir, identity)
                if classify_cell_artifact(existing, identity, spec) == "OK":
                    print(f"SKIP_OK {identity.cell_id}")
                    continue
                try:
                    payload = _run_arm(
                        identity=identity,
                        spec=spec,
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
                        identity, spec, f"{type(exc).__name__}: {exc}"
                    )
                    print(f"CELL_FAILED {identity.cell_id} reason={exc}")
                if any(seed in V3_COMPLETION_SEEDS for seed in matrix["seeds"]):
                    payload.update(
                        {
                            "source_mode": identity.mode,
                            "fused_init_source_mode": identity.mode,
                            "source_authority_identity": str(_source_authority(identity)),
                        }
                    )
                write_cell_artifact(cell_path(cells_dir, identity), payload)
                _append_jsonl(
                    log_path,
                    {
                        "cell_id": identity.cell_id,
                        "status": payload["status"],
                        "arm": identity.arm,
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
                spec = registry[identity.arm]
                payload = _failed_cell_payload(identity, spec, f"{type(exc).__name__}: {exc}")
                if any(seed in V3_COMPLETION_SEEDS for seed in matrix["seeds"]):
                    payload.update(
                        {
                            "source_mode": identity.mode,
                            "fused_init_source_mode": identity.mode,
                            "source_authority_identity": str(_source_authority(identity)),
                        }
                    )
                write_cell_artifact(cell_path(cells_dir, identity), payload)
                _append_jsonl(log_path, {"cell_id": identity.cell_id, "status": "FAILED"})
            print(f"PAIRED_FAILED {paired_key} reason={exc}")

    all_rows: List[dict] = []
    for identity in identities:
        path = cell_path(cells_dir, identity)
        spec = registry[identity.arm]
        status = classify_cell_artifact(path, identity, spec)
        if status == "OK":
            all_rows.append(json.loads(path.read_text(encoding="utf-8")))
        elif path.is_file():
            all_rows.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            all_rows.append(_failed_cell_payload(identity, spec, "MISSING"))

    aggregates = _write_aggregates(results_dir, identities, all_rows, smoke=bool(matrix["smoke"]))
    guard = aggregates["guard"]
    print("\n============================================================")
    print(f"EXPECTED_RUNS={guard['EXPECTED_RUNS']}")
    print(f"OK_RUNS={guard['OK_RUNS']}")
    print(f"FAILED_RUNS={guard['FAILED_RUNS']}")
    print(f"DUPLICATE_RUNS={guard['DUPLICATE_RUNS']}")
    print(f"MISSING_RUNS={guard['MISSING_RUNS']}")
    print(f"ROBUSTNESS_COMPLETE={guard['ROBUSTNESS_COMPLETE']}")
    if aggregates["smoke"] is not None:
        for key, value in aggregates["smoke"].items():
            print(f"{key}={value}")
    print(f"Run root: {run_dir}")
    print(f"All runs: {aggregates['paths']['all_runs']}")
    print(f"Paired: {aggregates['paths']['paired']}")
    print("============================================================")
    if not matrix["smoke"] and counts["expected_runs"] == EXPECTED_RUNS and guard["ROBUSTNESS_COMPLETE"] != "YES":
        raise SystemExit("ROBUSTNESS_COMPLETE=NO")


if __name__ == "__main__":
    main()
