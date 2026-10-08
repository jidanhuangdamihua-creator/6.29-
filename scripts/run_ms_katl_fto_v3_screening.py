"""FUSED_TARGET_ONLY_V3 (V2 recipe + RevIN) screening runner.

Development / screening track. Same Stage-A pipeline and V2 training recipe as
FUSED_TARGET_ONLY_V2; adds RevIN around the target CNN. Does not modify
FORMAL_METHODS, MSML, No-TL, baselines, frozen v1/v2 outputs, or
configs/primary_target_lr_selection_manifest.json, and does not launch
robustness experiments.

Usage:
    python scripts/run_ms_katl_fto_v3_screening.py --preflight
    python scripts/run_ms_katl_fto_v3_screening.py --smoke
    python scripts/run_ms_katl_fto_v3_screening.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tf_compat  # noqa: F401

import numpy as np
import pandas as pd

from scripts.run_ms_katl_fto_v2_screening import (
    SMOKE_DEFAULT_EPOCHS,
    _csv_ready,
    _dump_csv,
    _requested_datasets,
    _requested_targets,
)
from scripts.run_ms_katl_primary_target_lr_screening import (
    PROJECT_ROOT,
    _load_d13_pool,
    _load_d46_pool,
    _prepare_d13_target_cell,
    _prepare_d46_target_cell,
)
from paper_reproduction_protocol import load_paper_protocol, resolve_strict_paper_mode
from scripts.run_full_paper_experiments import _load_config
from src.protocols.reproducibility import set_protocol_seed
from src.transfer_methods.ms_katl_fto_v2_screening import (
    load_v1_reference_runs,
    load_v2_reference_runs,
)
from src.transfer_methods.ms_katl_fto_v3_screening import (
    V3_METHOD_ARM,
    V3_METHOD_DISPLAY,
    V3_SCREENING_LABEL,
    V3_TRACK,
    assert_v3_lr_arms_share_stage_a,
    attach_v1_deltas,
    attach_v2_deltas,
    default_v1_all_runs_csv,
    default_v2_all_runs_csv,
    format_v3_vs_frozen_summary,
    v3_architecture,
    v3_architecture_digest,
    v3_training_recipe,
)
from src.transfer_methods.ms_katl_lite import prepare_ms_katl_stage_a
from src.transfer_methods.ms_katl_mechanism_ablation import (
    prepare_shared_target_bundle,
    resolve_ablation_arm,
    run_ablation_arm,
)
from src.transfer_methods.ms_katl_primary_target_lr_screening import (
    LR_GRID,
    SCREENING_DATASETS,
    SCREENING_EPOCHS,
    SCREENING_HORIZON,
    SCREENING_MODE,
    SCREENING_SEED,
    STAGE_A_LEARNING_RATE,
    assert_screening_does_not_change_formal_methods,
    attach_reference_deltas,
    compute_screening_metrics,
    default_baseline_csv,
    default_formal_csv,
    format_dataset_terminal,
    format_global_summary,
    format_lr,
    load_baseline_references,
    load_formal_references,
    parse_target_identity,
    sales_minmax,
    select_dataset_primary_target,
    select_target_best_lr,
)
from src.utils.d4_d6_runtime import load_default_metric_protocol
from src.utils.run_utils import create_run_dir, reserve_new_output_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FUSED_TARGET_ONLY_V3 (V2 recipe + RevIN) LR screening."
    )
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Lightweight check: subset datasets/targets, small epochs. Not for selection.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--datasets", default=None, help="Comma-separated D1-D6.")
    parser.add_argument(
        "--targets",
        default=None,
        help="Comma-separated target identities (smoke only), e.g. 166/258.",
    )
    parser.add_argument(
        "--v1-run",
        type=Path,
        default=None,
        help="Path to frozen v1 screening_all_runs.csv (default: frozen v1 run).",
    )
    parser.add_argument(
        "--v2-run",
        type=Path,
        default=None,
        help="Path to frozen v2 v2_screening_all_runs.csv (default: frozen v2 run).",
    )
    return parser.parse_args()


def _failed_row_v3(dataset: str, parsed: dict, lr: float, epochs: int, reason: str) -> dict:
    return {
        "dataset": dataset,
        "target_identity": parsed["target_identity"],
        "mode": SCREENING_MODE,
        "horizon": SCREENING_HORIZON,
        "seed": SCREENING_SEED,
        "epochs": int(epochs),
        "learning_rate": float(lr),
        "learning_rate_label": format_lr(lr),
        "method": V3_METHOD_DISPLAY,
        "rmse_original": None,
        "smape_original": None,
        "rmse_normalized": None,
        "smape_normalized": None,
        "metric_space": "original_sales_space",
        "run_status": "FAILED",
        "failure_reason": str(reason),
        "SCREENING_STATUS": "FAILED",
    }


def _run_target_lr_arms_v3(
    *,
    dataset: str,
    parsed: dict,
    cell: dict,
    epochs: int,
    batch_size: int,
    metric_protocol: Optional[dict],
    pred_root: Path,
) -> List[dict]:
    recipe = v3_training_recipe()
    arch = v3_architecture()
    set_protocol_seed(SCREENING_SEED, include_frameworks=True)
    stage_a = prepare_ms_katl_stage_a(
        source_df=cell["source_model_df"],
        target_df=cell["target_model_df"],
        feature_cols=cell["model_feature_cols"],
        k=3,
        horizon=SCREENING_HORIZON,
        window_size=10,
        learning_rate=STAGE_A_LEARNING_RATE,
        source_epochs=int(epochs),
        batch_size=int(batch_size),
        tau_k=1.0,
        group_cols=cell["group_cols"],
        k3_selection_context=cell["k3_selection_context"],
        k3_raw_source_context=cell["k3_raw_source_context"],
        transformation_reuse_context=cell["transformation_reuse_context"],
        seed=SCREENING_SEED,
    )
    bundle = prepare_shared_target_bundle(
        stage_a,
        cell["target_model_df"],
        cell["model_feature_cols"],
        horizon=SCREENING_HORIZON,
        window_size=10,
        learning_rate=STAGE_A_LEARNING_RATE,
        seed=SCREENING_SEED,
        transformation_reuse_context=cell["transformation_reuse_context"],
    )
    spec = resolve_ablation_arm(V3_METHOD_ARM)
    source_payload = {
        "source_identities": [tuple(item) for item in stage_a.source_identities],
        "source_distances": list(stage_a.source_distances),
        "source_model_digests": list(stage_a.source_weight_digests),
        "source_scaler_digests": list(stage_a.source_scaler_digests),
        "fused_init_digest": bundle.fused_target_digest,
        "target_scaler_digest": bundle.target_split_digest.get("target_scaler"),
        "target_split_digest": dict(bundle.target_split_digest),
        "architecture_digest": v3_architecture_digest(),
    }
    sales_min, sales_max = sales_minmax(bundle.target_scaler, bundle.target_feature_columns)
    rows: List[dict] = []
    for lr in LR_GRID:
        set_protocol_seed(SCREENING_SEED, include_frameworks=True)
        arm = run_ablation_arm(
            spec,
            bundle,
            epochs=int(epochs),
            batch_size=int(batch_size),
            gamma=0.5,
            use_adversarial=False,
            metric_protocol=metric_protocol,
            fine_tune_learning_rate=float(lr),
            training_recipe=recipe,
            revin_enabled=True,
            revin_affine=bool(arch["revin_affine"]),
            revin_eps=float(arch["revin_eps"]),
            revin_target_channel=int(arch["revin_target_channel"]),
        )
        metrics = compute_screening_metrics(
            arm["y_true"],
            arm["y_pred"],
            target_scaler=bundle.target_scaler,
            feature_columns=bundle.target_feature_columns,
        )
        drifts = arm.get("conv_drifts") or {}
        lr_label = format_lr(float(lr))
        row = {
            "dataset": dataset,
            "target_identity": parsed["target_identity"],
            "mode": SCREENING_MODE,
            "horizon": SCREENING_HORIZON,
            "seed": SCREENING_SEED,
            "epochs": int(epochs),
            "learning_rate": float(lr),
            "learning_rate_label": lr_label,
            "method": V3_METHOD_DISPLAY,
            "rmse_original": metrics["rmse_original"],
            "smape_original": metrics["smape_original"],
            "rmse_normalized": metrics["rmse_normalized"],
            "smape_normalized": metrics["smape_normalized"],
            "metric_space": metrics["metric_space"],
            "sales_min": sales_min if sales_min is not None else metrics["sales_min"],
            "sales_max": sales_max if sales_max is not None else metrics["sales_max"],
            "conv1_relative_drift": drifts.get("conv1_relative_weight_drift"),
            "conv2_relative_drift": drifts.get("conv2_relative_weight_drift"),
            "conv3_relative_drift": drifts.get("conv3_relative_weight_drift"),
            "total_backbone_relative_drift": drifts.get("total_backbone_relative_weight_drift"),
            "run_status": "OK",
            "failure_reason": "",
            "SCREENING_STATUS": "OK",
            "y_true_normalized": np.asarray(arm["y_true"]).reshape(-1),
            "y_pred_normalized": np.asarray(arm["y_pred"]).reshape(-1),
            "y_true_original": metrics["y_true_original"],
            "y_pred_original": metrics["y_pred_original"],
            **source_payload,
            **{f"arch_{key}": value for key, value in arch.items()},
            **{f"recipe_{key}": value for key, value in recipe.items()},
            "final_learning_rate": (arm.get("training_recipe") or {}).get("final_learning_rate"),
            "ema_applied": (arm.get("training_recipe") or {}).get("ema_applied"),
            "revin_enabled": arm.get("revin_enabled"),
            "revin_affine": arm.get("revin_affine"),
        }
        pred_dir = pred_root / dataset / parsed["prediction_dirname"] / lr_label
        pred_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "y_true_normalized": row["y_true_normalized"],
                "y_pred_normalized": row["y_pred_normalized"],
                "y_true_original": row["y_true_original"],
                "y_pred_original": row["y_pred_original"],
            }
        ).to_csv(pred_dir / "predictions.csv", index=False, encoding="utf-8")
        rows.append(row)
    flags = assert_v3_lr_arms_share_stage_a(rows)
    for row in rows:
        row["LR_ARMS_FUSED_INIT_MATCH"] = flags["LR_ARMS_FUSED_INIT_MATCH"]
        row["STAGE_A_SHARED"] = "YES"
    return rows


def main() -> None:
    args = _parse_args()
    smoke = bool(args.smoke)
    if args.targets is not None and not smoke:
        raise SystemExit("--targets is only allowed with --smoke")
    epochs = int(args.epochs) if args.epochs is not None else (
        SMOKE_DEFAULT_EPOCHS if smoke else SCREENING_EPOCHS
    )
    datasets = _requested_datasets(args.datasets, smoke)
    recipe = v3_training_recipe()
    arch = v3_architecture()

    print("================================================================================")
    print(f"{V3_METHOD_DISPLAY} SCREENING PREFLIGHT (track={V3_TRACK})")
    print("================================================================================")
    print(f"MODE={'SMOKE' if smoke else 'PRODUCTION'}")
    print(f"EPOCHS={epochs}  HORIZON={SCREENING_HORIZON}  SEED={SCREENING_SEED}")
    print(f"LR_GRID={[format_lr(lr) for lr in LR_GRID]}")
    print("CANONICAL_LR_GRID_CHANGE=NO")
    print(f"TRAINING_RECIPE={json.dumps(recipe)}")
    print(f"ARCHITECTURE={json.dumps(arch)}")
    print(f"ARCHITECTURE_DIGEST={v3_architecture_digest()}")
    plan_targets: Dict[str, tuple] = {}
    for dataset in datasets:
        plan_targets[dataset] = _requested_targets(dataset, args.targets, smoke)
        print(f"{dataset} TARGETS={list(plan_targets[dataset])}")
    total_runs = sum(len(v) for v in plan_targets.values()) * len(LR_GRID)
    print(f"TOTAL_SCREENING_RUNS={total_runs}")
    print("FORMAL_METHODS_CHANGE=NO  ARCHITECTURE_CHANGE=YES(RevIN)  AUTO_START_ROBUSTNESS=NO")
    print("SELECTION_FROZEN=NO")
    print("================================================================================")
    if args.preflight:
        return

    if not smoke:
        if tuple(datasets) != SCREENING_DATASETS:
            raise SystemExit("production v3 screening requires all D1-D6 datasets (or use --smoke)")
        if epochs != SCREENING_EPOCHS:
            raise SystemExit(
                f"production v3 screening requires epochs={SCREENING_EPOCHS} (or use --smoke)"
            )

    assert_screening_does_not_change_formal_methods()

    if args.output_dir is None:
        label = V3_SCREENING_LABEL + ("_SMOKE" if smoke else "")
        run_dir = create_run_dir(PROJECT_ROOT, label=label)
    else:
        run_dir = Path(args.output_dir)
        if not run_dir.is_absolute():
            run_dir = PROJECT_ROOT / run_dir
        reserve_new_output_dir(run_dir)
    pred_root = run_dir / "predictions"
    pred_root.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "method": V3_METHOD_DISPLAY,
                "track": V3_TRACK,
                "smoke": smoke,
                "horizon": SCREENING_HORIZON,
                "seed": SCREENING_SEED,
                "epochs": epochs,
                "lr_grid": list(LR_GRID),
                "stage_a_learning_rate": STAGE_A_LEARNING_RATE,
                "datasets": list(datasets),
                "targets": {key: list(value) for key, value in plan_targets.items()},
                "total_screening_runs": total_runs,
                "training_recipe": recipe,
                "architecture": arch,
                "architecture_digest": v3_architecture_digest(),
                "architecture_change": "RevIN wrap on FUSED_TARGET_ONLY",
                "auto_start_robustness": False,
                "selection_frozen": False,
                "canonical_lr_grid_changed": False,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    v1_csv = args.v1_run if args.v1_run is not None else default_v1_all_runs_csv(PROJECT_ROOT)
    if v1_csv is not None and v1_csv.is_dir():
        v1_csv = v1_csv / "results" / "screening_all_runs.csv"
    v2_csv = args.v2_run if args.v2_run is not None else default_v2_all_runs_csv(PROJECT_ROOT)
    if v2_csv is not None and v2_csv.is_dir():
        v2_csv = v2_csv / "results" / "v2_screening_all_runs.csv"
    v1_index = load_v1_reference_runs(v1_csv)
    v2_index = load_v2_reference_runs(v2_csv)
    print(f"V1_REFERENCE_ROWS={len(v1_index)} ({v1_csv})")
    print(f"V2_REFERENCE_ROWS={len(v2_index)} ({v2_csv})")

    d13_needed = any(dataset in {"D1", "D2", "D3"} for dataset in datasets)
    d13_cfg = None
    d13_protocol = None
    if d13_needed:
        d13_cfg = _load_config()
        d13_cfg.setdefault("single_experiment", {})["horizon"] = SCREENING_HORIZON
        d13_cfg["single_experiment"]["seed"] = SCREENING_SEED
        d13_cfg["single_experiment"]["source_epochs"] = epochs
        d13_cfg["single_experiment"]["target_epochs"] = epochs
        d13_protocol = load_paper_protocol(d13_cfg)
        d13_protocol["strict_paper_mode"] = True
        d13_protocol.setdefault("metric_protocol", {})["strict_paper_metrics"] = True
        resolve_strict_paper_mode(d13_cfg, explicit=True)

    metric_protocol = load_default_metric_protocol(PROJECT_ROOT)
    all_runs: List[dict] = []
    target_best_rows: List[dict] = []
    primary_rows: List[dict] = []
    dataset_pools: Dict[str, Any] = {}

    for dataset in datasets:
        print(f"\n=== LOAD {dataset} ===")
        set_protocol_seed(SCREENING_SEED, include_frameworks=True)
        if dataset in {"D1", "D2", "D3"}:
            dataset_pools[dataset] = _load_d13_pool(dataset, d13_cfg, d13_protocol)
        else:
            dataset_pools[dataset] = _load_d46_pool(dataset, epochs)

    for dataset in datasets:
        dataset_run_rows: List[dict] = []
        dataset_best_rows: List[dict] = []
        formal_csv = default_formal_csv(dataset, PROJECT_ROOT)
        baseline_csv = default_baseline_csv(dataset, PROJECT_ROOT)
        arms_label = ",".join(format_lr(lr) for lr in LR_GRID)
        for identity in plan_targets[dataset]:
            parsed = parse_target_identity(dataset, identity)
            print(
                f"\n=== {dataset} TARGET {parsed['target_identity']} | "
                f"V3 Stage-A once, then {arms_label} ==="
            )
            try:
                if dataset in {"D1", "D2", "D3"}:
                    cell = _prepare_d13_target_cell(dataset, dataset_pools[dataset], parsed)
                    batch_size = 16
                else:
                    cell = _prepare_d46_target_cell(dataset, dataset_pools[dataset], parsed)
                    batch_size = int(dataset_pools[dataset]["config"]["batch_size"])
                lr_rows = _run_target_lr_arms_v3(
                    dataset=dataset,
                    parsed=parsed,
                    cell=cell,
                    epochs=epochs,
                    batch_size=batch_size,
                    metric_protocol=metric_protocol,
                    pred_root=pred_root,
                )
            except Exception as exc:
                lr_rows = [
                    _failed_row_v3(dataset, parsed, lr, epochs, f"{type(exc).__name__}: {exc}")
                    for lr in LR_GRID
                ]
                print(f"SCREENING_STATUS=FAILED target={parsed['target_identity']} reason={exc}")
            formal_refs = load_formal_references(
                formal_csv,
                dataset_id=dataset,
                target_identity=parsed["target_identity"],
            )
            baseline_refs = load_baseline_references(
                baseline_csv,
                dataset_id=dataset,
                target_identity=parsed["target_identity"],
            )
            gated = [
                attach_reference_deltas(
                    row, formal_refs=formal_refs, baseline_refs=baseline_refs
                )
                for row in lr_rows
            ]
            compared = [attach_v2_deltas(attach_v1_deltas(row, v1_index), v2_index) for row in gated]
            dataset_run_rows.extend(compared)
            all_runs.extend(compared)
            best = select_target_best_lr(compared)
            dataset_best_rows.append(best)
            target_best_rows.append(best)

        selection = select_dataset_primary_target(dataset_best_rows)
        selected = selection.get("selected") or {}
        primary_rows.append(
            {
                "dataset": dataset,
                "selected_primary_target": selected.get("target_identity"),
                "selected_learning_rate": selected.get("best_lr"),
                "rmse_original": selected.get("best_rmse_original") or selected.get("rmse_original"),
                "smape_original": selected.get("best_smape_original") or selected.get("smape_original"),
                "no_tl_rmse": selected.get("no_tl_rmse"),
                "msml_rmse": selected.get("msml_rmse"),
                "bl4_rmse": selected.get("bl4_rmse"),
                "best_bl_method": selected.get("best_bl_method"),
                "best_bl_rmse": selected.get("best_bl_rmse"),
                "beats_no_tl": selected.get("beats_no_tl"),
                "beats_msml": selected.get("beats_msml"),
                "beats_bl4": selected.get("beats_bl4"),
                "beats_best_bl": selected.get("beats_best_bl"),
                "v1_rmse_original": selected.get("v1_rmse_original"),
                "v2_rmse_original": selected.get("v2_rmse_original"),
                "delta_rmse_vs_v1": selected.get("delta_rmse_vs_v1"),
                "delta_rmse_vs_v2": selected.get("delta_rmse_vs_v2"),
                "delta_smape_vs_v1": selected.get("delta_smape_vs_v1"),
                "delta_smape_vs_v2": selected.get("delta_smape_vs_v2"),
                "selection_role": "development_v3_candidate",
                "selection_frozen": "no",
                "target_selection_stage": V3_TRACK,
                "verdict": selection.get("verdict"),
                "PRIMARY_TARGET_EVIDENCE": selection.get("evidence"),
                "rmse_ranks": json.dumps(selection.get("rmse_ranks") or {}),
                "transfer_ranks": json.dumps(selection.get("transfer_ranks") or {}),
            }
        )
        print(format_dataset_terminal(dataset, dataset_run_rows, selection))

    assert_screening_does_not_change_formal_methods()
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    _dump_csv(results_dir / "v3_screening_all_runs.csv", [_csv_ready(r) for r in all_runs])
    _dump_csv(results_dir / "v3_screening_target_best_lr.csv", [_csv_ready(r) for r in target_best_rows])
    _dump_csv(results_dir / "v3_screening_dataset_primary_targets.csv", primary_rows)
    manifest = {
        "method": V3_METHOD_DISPLAY,
        "track": V3_TRACK,
        "smoke": smoke,
        "training_recipe": recipe,
        "architecture": arch,
        "architecture_digest": v3_architecture_digest(),
        "selection_frozen": False,
        "auto_start_robustness": False,
        "v1_reference_csv": str(v1_csv),
        "v2_reference_csv": str(v2_csv),
        "run_dir": str(run_dir),
        "datasets": {
            row["dataset"]: {
                "primary_target": row["selected_primary_target"],
                "learning_rate": row["selected_learning_rate"],
                "rmse_original": row["rmse_original"],
                "smape_original": row["smape_original"],
                "verdict": row["verdict"],
            }
            for row in primary_rows
        },
    }
    (results_dir / "v3_selection_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )

    if not smoke:
        print(format_global_summary(primary_rows))
    print(format_v3_vs_frozen_summary(all_runs))
    sentinel = "FTO_V3_SMOKE_COMPLETE" if smoke else "D1_D6_FTO_V3_SCREENING_COMPLETE"
    print(sentinel)
    print(f"Run root: {run_dir}")
    print(f"All runs: {results_dir / 'v3_screening_all_runs.csv'}")
    print(f"Target best LR: {results_dir / 'v3_screening_target_best_lr.csv'}")
    print(f"Primary targets: {results_dir / 'v3_screening_dataset_primary_targets.csv'}")
    print(f"V3 manifest: {results_dir / 'v3_selection_manifest.json'}")
    print(f"Predictions: {pred_root}")


if __name__ == "__main__":
    main()
