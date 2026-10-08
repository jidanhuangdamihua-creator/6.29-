"""MS-KATL FUSED_TARGET_ONLY_V3_2_SEASONAL_RESIDUAL robustness runner.

Parent V3 plus frozen lag-7 seasonal residual on Stage-B only.
Does not reselect targets or learning rates, does not modify FORMAL_METHODS,
does not inherit V3.1 calibration, and does not overwrite V3 outputs.

Usage:
    python scripts/run_ms_katl_fto_v3_2_seasonal_residual_robustness.py --preflight
    python scripts/run_ms_katl_fto_v3_2_seasonal_residual_robustness.py --smoke
    python scripts/run_ms_katl_fto_v3_2_seasonal_residual_robustness.py
    python scripts/run_ms_katl_fto_v3_2_seasonal_residual_robustness.py --output-dir <existing_v3_2_root>
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

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
from src.transfer_methods.ms_katl_fto_three_track_robustness import jsonable, relative_improvement, write_cell_artifact
from src.transfer_methods.ms_katl_fto_v3_2_seasonal_residual import (
    EXPECTED_COMMON_TARGETS,
    EXPECTED_RUNS,
    EXPECTED_V3_ARCHITECTURE_DIGEST,
    EXPECTED_V3_LR,
    LAG_DAYS,
    METHOD_DISPLAY,
    PARENT_METHOD,
    RESIDUALIZATION_SCOPE,
    REVIN_OUTPUT_MODE,
    ROBUSTNESS_LABEL,
    ROBUSTNESS_TRACK,
    SEASONAL_PRIOR_TYPE,
    V3_1_CALIBRATION_ENABLED,
    V3_2_HORIZONS,
    V3_2_SEEDS,
    WINDOW_SIZE,
    V32ContractError,
    V32RunIdentity,
    assert_formal_methods_unchanged_v32,
    assert_not_v3_parent_output_dir,
    assert_parent_fingerprints_unchanged,
    assert_residual_revin_semantics,
    assert_v3_1_not_inherited,
    assert_v3_parent_contract,
    build_dataset_horizon_tables,
    build_v3_2_identities,
    cell_path,
    classify_cell_artifact,
    completion_guard,
    expected_matrix_counts,
    fingerprint_parent_artifacts,
    format_headline_tables,
    format_preflight,
    frozen_v3_lr_map,
    inherited_v3_spec,
    load_frozen_v3_robustness_rows,
    load_valid_ok_cells,
    load_v3_2_manifest,
    lookup_parent_v3_metrics,
    parent_v3_prediction_path,
    parse_dataset_ids,
    parse_int_tuple,
    run_v3_2_stage_b,
    split_target_frames,
    v3_2_architecture,
    v3_2_manifest_path,
    v3_2_revin_config,
    v3_2_training_recipe,
    write_v3_2_manifest,
)
from src.transfer_methods.ms_katl_lite import prepare_ms_katl_stage_a
from src.transfer_methods.ms_katl_mechanism_ablation import prepare_shared_target_bundle
from src.transfer_methods.ms_katl_primary_target_lr_screening import (
    SCREENING_DATASETS,
    SCREENING_EPOCHS,
    SCREENING_MODE,
    STAGE_A_LEARNING_RATE,
    parse_target_identity,
    sales_minmax,
)
from src.utils.d4_d6_runtime import load_default_metric_protocol
from src.utils.run_utils import create_run_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FUSED_TARGET_ONLY_V3_2_SEASONAL_RESIDUAL confirmatory robustness."
    )
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Bounded wiring check: D1 common target, H=1, seed=42.",
    )
    parser.add_argument("--datasets", default=None, help="Comma-separated D1-D6. Default all.")
    parser.add_argument("--horizons", default=None, help="Comma-separated 1,3,5. Default all.")
    parser.add_argument("--seeds", default=None, help="Comma-separated 42,43,44. Default all.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Write configs/ms_katl_fto_v3_2_seasonal_residual_manifest.json and exit.",
    )
    return parser.parse_args()


def _resolve_matrix(args: argparse.Namespace) -> Dict[str, Any]:
    if args.smoke:
        if args.datasets not in {None, "d1", "D1"}:
            raise V32ContractError("smoke is fixed to D1")
        if args.horizons not in {None, "1"}:
            raise V32ContractError("smoke is fixed to horizon=1")
        if args.seeds not in {None, "42"}:
            raise V32ContractError("smoke is fixed to seed=42")
        return {
            "datasets": ("D1",),
            "horizons": (1,),
            "seeds": (42,),
            "smoke": True,
        }
    return {
        "datasets": parse_dataset_ids(args.datasets),
        "horizons": parse_int_tuple(args.horizons, V3_2_HORIZONS, name="horizon"),
        "seeds": parse_int_tuple(args.seeds, V3_2_SEEDS, name="seed"),
        "smoke": False,
    }


def _prepare_d13_cfg(horizon: int, seed: int, epochs: int):
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
        assert_not_v3_parent_output_dir(run_dir, PROJECT_ROOT)
        return run_dir
    run_dir = Path(requested)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    assert_not_v3_parent_output_dir(run_dir, PROJECT_ROOT)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(jsonable(dict(payload)), default=str) + "\n")


def _run_stage_a(
    *,
    cell: dict,
    horizon: int,
    seed: int,
    epochs: int,
    batch_size: int,
):
    set_protocol_seed(int(seed), include_frameworks=True)
    stage_a = prepare_ms_katl_stage_a(
        source_df=cell["source_model_df"],
        target_df=cell["target_model_df"],
        feature_cols=cell["model_feature_cols"],
        k=3,
        horizon=int(horizon),
        window_size=WINDOW_SIZE,
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
        window_size=WINDOW_SIZE,
        learning_rate=STAGE_A_LEARNING_RATE,
        seed=int(seed),
        transformation_reuse_context=cell["transformation_reuse_context"],
    )
    return stage_a, bundle


def _failed_cell_payload(identity: V32RunIdentity, reason: str) -> dict:
    return {
        "status": "FAILED",
        "dataset": identity.dataset,
        "target_identity": identity.target_identity,
        "horizon": int(identity.horizon),
        "seed": int(identity.seed),
        "learning_rate": float(identity.selected_lr),
        "learning_rate_label": identity.selected_lr_label,
        "parent_method": PARENT_METHOD,
        "seasonal_prior_type": SEASONAL_PRIOR_TYPE,
        "seasonal_lag_days": LAG_DAYS,
        "train_residual_sample_count": None,
        "validation_residual_sample_count": None,
        "test_prediction_count": None,
        "prior_maturity_guard": None,
        "rmse_original": None,
        "smape_original": None,
        "parent_v3_rmse_reference": None,
        "parent_v3_smape_reference": None,
        "relative_improvement_vs_v3_rmse": None,
        "relative_improvement_vs_v3_smape": None,
        "metric_space": ORIGINAL_SALES_SPACE,
        "failure_reason": str(reason),
        "run_status": "FAILED",
        "V3_1_CALIBRATION_ENABLED": V3_1_CALIBRATION_ENABLED,
    }


def _write_prediction_csv(
    path: Path,
    *,
    timestamps: Sequence[str],
    origins: Sequence[str],
    y_true: np.ndarray,
    seasonal_prior: np.ndarray,
    predicted_residual: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(np.asarray(y_true).reshape(-1).shape[0])
    if len(timestamps) != n:
        raise V32ContractError(
            f"test timestamp count {len(timestamps)} != prediction length {n}"
        )
    pd.DataFrame(
        {
            "forecast_origin": list(origins),
            "timestamp": list(timestamps),
            "y_true": np.asarray(y_true, dtype=np.float64).reshape(-1),
            "seasonal_prior": np.asarray(seasonal_prior, dtype=np.float64).reshape(-1),
            "predicted_residual": np.asarray(predicted_residual, dtype=np.float64).reshape(-1),
            "y_pred_v3_2": np.asarray(y_pred, dtype=np.float64).reshape(-1),
        }
    ).to_csv(path, index=False, encoding="utf-8")


def _write_audit_csv(path: Path, rows: Sequence[Mapping[str, Any]], *, predicted_residual: np.ndarray, reconstructed: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        frame = pd.DataFrame(
            columns=[
                "timestamp",
                "forecast_origin",
                "lag_timestamp",
                "lag_value",
                "actual_target",
                "actual_residual",
                "predicted_residual",
                "reconstructed_prediction",
            ]
        )
    else:
        frame = frame.copy()
        frame["predicted_residual"] = np.asarray(predicted_residual, dtype=np.float64).reshape(-1)
        frame["reconstructed_prediction"] = np.asarray(reconstructed, dtype=np.float64).reshape(-1)
    frame.to_csv(path, index=False, encoding="utf-8")


def _run_cell(
    *,
    identity: V32RunIdentity,
    spec,
    stage_a: Any,
    bundle: Any,
    cell: dict,
    metric_protocol: Optional[dict],
    pred_root: Path,
    audit_root: Path,
    parsed: dict,
    v3_rows: Sequence[Mapping[str, Any]],
    smoke: bool,
) -> dict:
    del smoke
    set_protocol_seed(int(identity.seed), include_frameworks=True)
    raw_train, raw_val, raw_test = split_target_frames(cell["target_model_df"])
    fitted = run_v3_2_stage_b(
        bundle=bundle,
        identity=identity,
        spec=spec,
        raw_train=raw_train,
        raw_val=raw_val,
        raw_test=raw_test,
        feature_columns=cell["model_feature_cols"],
        epochs=SCREENING_EPOCHS,
        batch_size=int(cell.get("batch_size") or 16),
        metric_protocol=metric_protocol,
    )
    parent_rmse, parent_smape = lookup_parent_v3_metrics(v3_rows, identity)
    parent_pred = parent_v3_prediction_path(PROJECT_ROOT, identity, parsed)
    if parent_rmse is None and parent_pred.is_file():
        parent_frame = pd.read_csv(parent_pred)
        parent_rmse = float(
            np.sqrt(
                np.mean(
                    (
                        np.asarray(parent_frame["y_true_original"], dtype=np.float64)
                        - np.asarray(parent_frame["y_pred_original"], dtype=np.float64)
                    )
                    ** 2
                )
            )
        )
        if "y_true_original" in parent_frame.columns:
            from src.evaluation.metrics import smape as smape_fn

            parent_smape = float(
                smape_fn(
                    np.asarray(parent_frame["y_true_original"], dtype=np.float64),
                    np.asarray(parent_frame["y_pred_original"], dtype=np.float64),
                )
            )
    timestamps = [item["label_date"] for item in fitted["residualized"]["test"].meta]
    origins = [item["window_end"] for item in fitted["residualized"]["test"].meta]
    pred_dir = (
        pred_root
        / identity.dataset
        / parsed["prediction_dirname"]
        / f"h{int(identity.horizon)}_s{int(identity.seed)}_{identity.selected_lr_label}"
    )
    pred_path = pred_dir / "predictions.csv"
    _write_prediction_csv(
        pred_path,
        timestamps=timestamps,
        origins=origins,
        y_true=fitted["y_true_original"],
        seasonal_prior=fitted["seasonal_prior"],
        predicted_residual=fitted["predicted_residual_original"],
        y_pred=fitted["y_pred_original"],
    )
    audit_dir = (
        audit_root
        / identity.dataset
        / parsed["prediction_dirname"]
        / f"h{int(identity.horizon)}_s{int(identity.seed)}_{identity.selected_lr_label}"
    )
    audit_path = audit_dir / "seasonal_residual_audit.csv"
    _write_audit_csv(
        audit_path,
        fitted["residualized"]["test"].audit_rows,
        predicted_residual=fitted["predicted_residual_original"],
        reconstructed=fitted["y_pred_original"],
    )
    sales_min, sales_max = sales_minmax(bundle.target_scaler, bundle.target_feature_columns)
    return {
        "status": "OK",
        "dataset": identity.dataset,
        "target_identity": identity.target_identity,
        "mode": SCREENING_MODE,
        "horizon": int(identity.horizon),
        "seed": int(identity.seed),
        "epochs": SCREENING_EPOCHS,
        "learning_rate": float(identity.selected_lr),
        "learning_rate_label": identity.selected_lr_label,
        "method": METHOD_DISPLAY,
        "parent_method": PARENT_METHOD,
        "parent_architecture_digest": spec.architecture_digest,
        "seasonal_prior_type": SEASONAL_PRIOR_TYPE,
        "seasonal_lag_days": LAG_DAYS,
        "train_residual_sample_count": int(fitted["train_residual_sample_count"]),
        "validation_residual_sample_count": int(fitted["validation_residual_sample_count"]),
        "test_prediction_count": int(fitted["test_prediction_count"]),
        "prior_maturity_guard": fitted["prior_maturity_guard"],
        "residual_validation_loss": fitted["residual_validation_loss"],
        "rmse_original": float(fitted["rmse_original"]),
        "smape_original": float(fitted["smape_original"]),
        "parent_v3_rmse_reference": parent_rmse,
        "parent_v3_smape_reference": parent_smape,
        "relative_improvement_vs_v3_rmse": relative_improvement(parent_rmse, float(fitted["rmse_original"]))
        if parent_rmse is not None
        else None,
        "relative_improvement_vs_v3_smape": relative_improvement(parent_smape, float(fitted["smape_original"]))
        if parent_smape is not None
        else None,
        "metric_space": ORIGINAL_SALES_SPACE,
        "architecture": dict(v3_2_architecture()),
        "architecture_digest": spec.architecture_digest,
        "training_recipe": dict(v3_2_training_recipe()),
        "revin": v3_2_revin_config(),
        "revin_output_mode": REVIN_OUTPUT_MODE,
        "residualization_scope": RESIDUALIZATION_SCOPE,
        "sales_min": sales_min,
        "sales_max": sales_max,
        "prediction_artifact_path": str(pred_path),
        "audit_artifact_path": str(audit_path),
        "PRIOR_ALIGNMENT": "PASS",
        "PRIOR_MATURITY": "PASS",
        "RESIDUAL_REVIN_SEMANTICS": assert_residual_revin_semantics(),
        "V3_1_CALIBRATION_ENABLED": V3_1_CALIBRATION_ENABLED,
        "TEST_FUTURE_TRUTH_USED": "NO",
        "source_identities": [tuple(item) for item in stage_a.source_identities],
        "run_status": "OK",
        "failure_reason": "",
        "identity_tuple": list(identity.identity_tuple),
    }


def _write_aggregates(
    results_dir: Path,
    identities: Sequence[V32RunIdentity],
    rows: Sequence[Mapping[str, Any]],
    *,
    smoke: bool,
) -> Dict[str, Any]:
    results_dir.mkdir(parents=True, exist_ok=True)
    all_runs_path = results_dir / "v3_2_all_runs.csv"
    _dump_csv(all_runs_path, [_csv_ready(dict(row)) for row in rows])
    guard = completion_guard(identities, rows)
    (results_dir / "v3_2_completion_guard.json").write_text(
        json.dumps(guard, indent=2), encoding="utf-8"
    )
    tables = None
    if not smoke and guard["ROBUSTNESS_COMPLETE"] == "YES":
        v3_rows = load_frozen_v3_robustness_rows(PROJECT_ROOT)
        tables = build_dataset_horizon_tables(rows, v3_rows, PROJECT_ROOT)
        _dump_csv(results_dir / "v3_2_summary_by_dataset_horizon.csv", tables["rmse"])
        vs_v3_rows = []
        for item in tables["rmse"]:
            vs_v3_rows.append(
                {
                    "DATASET": item["DATASET"],
                    "HORIZON": item["HORIZON"],
                    "V3_RMSE_MEAN": item["V3_RMSE_MEAN"],
                    "V3_RMSE_STD": item["V3_RMSE_STD"],
                    "V3_2_RMSE_MEAN": item["V3_2_RMSE_MEAN"],
                    "V3_2_RMSE_STD": item["V3_2_RMSE_STD"],
                    "V3_2_VS_V3_PCT": item["V3_2_VS_V3_PCT"],
                }
            )
        _dump_csv(results_dir / "v3_2_vs_v3.csv", vs_v3_rows)
        _dump_csv(results_dir / "v3_2_vs_baselines.csv", tables["rmse"])
        _dump_csv(results_dir / "v3_2_vs_baselines_smape.csv", tables["smape"])
        (results_dir / "v3_2_diagnostics.json").write_text(
            json.dumps(tables["diagnostics"], indent=2), encoding="utf-8"
        )
    return {
        "guard": guard,
        "tables": tables,
        "paths": {"all_runs": all_runs_path},
    }


def _print_smoke(payload: Mapping[str, Any]) -> None:
    print(f"SMOKE_CELL_OK={'YES' if payload.get('status') == 'OK' else 'NO'}")
    print("PRIOR=LAG7")
    print(f"PRIOR_ALIGNMENT={payload.get('PRIOR_ALIGNMENT')}")
    print(f"PRIOR_MATURITY={payload.get('PRIOR_MATURITY')}")
    print(f"RESIDUAL_REVIN_SEMANTICS={payload.get('RESIDUAL_REVIN_SEMANTICS')}")
    print(f"TRAIN_RESIDUAL_SAMPLES={payload.get('train_residual_sample_count')}")
    print(f"VAL_RESIDUAL_SAMPLES={payload.get('validation_residual_sample_count')}")
    print(f"TEST_PREDICTIONS={payload.get('test_prediction_count')}")
    print(f"V3_PARENT_RMSE={payload.get('parent_v3_rmse_reference')}")
    print(f"V3_2_RMSE={payload.get('rmse_original')}")
    print(f"V3_PARENT_SMAPE={payload.get('parent_v3_smape_reference')}")
    print(f"V3_2_SMAPE={payload.get('smape_original')}")
    print(f"V3_2_IMPROVEMENT_VS_V3_RMSE={payload.get('relative_improvement_vs_v3_rmse')}")


def main() -> None:
    args = _parse_args()
    assert_formal_methods_unchanged_v32()
    assert_v3_1_not_inherited()
    parent_fp_before = fingerprint_parent_artifacts(PROJECT_ROOT)
    if args.write_manifest:
        path = write_v3_2_manifest(PROJECT_ROOT)
        print(f"V3_2_MANIFEST={path}")
        return

    frozen = assert_v3_parent_contract(PROJECT_ROOT)
    if not v3_2_manifest_path(PROJECT_ROOT).is_file():
        write_v3_2_manifest(PROJECT_ROOT)
    load_v3_2_manifest(PROJECT_ROOT)
    lr_map = frozen_v3_lr_map(frozen)
    matrix = _resolve_matrix(args)
    identities = build_v3_2_identities(
        datasets=matrix["datasets"],
        horizons=matrix["horizons"],
        seeds=matrix["seeds"],
        lr_map=lr_map,
    )
    counts = expected_matrix_counts(matrix["datasets"], matrix["horizons"], matrix["seeds"])
    if (
        not matrix["smoke"]
        and tuple(matrix["datasets"]) == SCREENING_DATASETS
        and tuple(matrix["horizons"]) == V3_2_HORIZONS
        and tuple(matrix["seeds"]) == V3_2_SEEDS
    ):
        if counts["expected_runs"] != EXPECTED_RUNS:
            raise V32ContractError(f"full matrix must be {EXPECTED_RUNS} runs, got {counts}")
        if len(identities) != EXPECTED_RUNS:
            raise V32ContractError(f"UNIQUE_IDENTITIES={len(identities)} != {EXPECTED_RUNS}")
    print(
        format_preflight(
            datasets=matrix["datasets"],
            horizons=matrix["horizons"],
            seeds=matrix["seeds"],
            lr_map=lr_map,
            counts=counts,
        ),
        end="",
    )
    if args.preflight:
        assert_parent_fingerprints_unchanged(parent_fp_before, fingerprint_parent_artifacts(PROJECT_ROOT))
        return

    spec = inherited_v3_spec()
    run_dir = _resolve_output_dir(args.output_dir, smoke=bool(matrix["smoke"]))
    results_dir = run_dir / "results"
    pred_root = run_dir / "predictions"
    audit_root = run_dir / "audit"
    cells_dir = run_dir / "cells"
    logs_dir = run_dir / "logs"
    for path in (results_dir, pred_root, audit_root, cells_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "label": ROBUSTNESS_LABEL,
                "track": ROBUSTNESS_TRACK,
                "method": METHOD_DISPLAY,
                "parent_method": PARENT_METHOD,
                "smoke": matrix["smoke"],
                "datasets": list(matrix["datasets"]),
                "horizons": list(matrix["horizons"]),
                "seeds": list(matrix["seeds"]),
                "epochs": SCREENING_EPOCHS,
                "expected_runs": counts["expected_runs"],
                "common_target_policy": "V3_FROZEN_PRIMARY_TARGETS",
                "lr_policy": "V3_FROZEN_TARGET_LR",
                "seasonal_prior_type": SEASONAL_PRIOR_TYPE,
                "seasonal_lag_days": LAG_DAYS,
                "residualization_scope": RESIDUALIZATION_SCOPE,
                "frozen_lr_mapping": {f"{ds}|{tgt}": lr for (ds, tgt), lr in EXPECTED_V3_LR.items()},
                "formal_methods_change": False,
                "V3_1_calibration": False,
                "parent_architecture_digest": EXPECTED_V3_ARCHITECTURE_DIGEST,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    completed = load_valid_ok_cells(cells_dir, identities, spec.architecture_digest)
    pending = [item for item in identities if item.identity_tuple not in completed]
    print(f"RESUME_OK_CELLS={len(completed)}")
    print(f"PENDING_CELLS={len(pending)}")

    d13_pool_cache: Dict[tuple, dict] = {}
    d46_data_cache: Dict[str, dict] = {}
    metric_protocol = load_default_metric_protocol(PROJECT_ROOT)
    log_path = logs_dir / "cells.jsonl"
    smoke_payload = None
    v3_rows = load_frozen_v3_robustness_rows(PROJECT_ROOT)

    for identity in pending:
        parsed = parse_target_identity(identity.dataset, identity.target_identity)
        print(
            f"\n=== V3.2 {identity.dataset} {identity.target_identity} "
            f"H={identity.horizon} S={identity.seed} LR={identity.selected_lr_label} ==="
        )
        existing = cell_path(cells_dir, identity)
        if classify_cell_artifact(existing, identity, spec.architecture_digest) == "OK":
            print(f"SKIP_OK {identity.cell_id}")
            continue
        try:
            if identity.dataset in {"D1", "D2", "D3"}:
                cache_key = (identity.dataset, int(identity.horizon), int(identity.seed))
                if cache_key not in d13_pool_cache:
                    cfg, protocol = _prepare_d13_cfg(int(identity.horizon), int(identity.seed), SCREENING_EPOCHS)
                    d13_pool_cache[cache_key] = _load_d13_pool(identity.dataset, cfg, protocol)
                cell = _prepare_d13_target_cell(identity.dataset, d13_pool_cache[cache_key], parsed)
                cell["batch_size"] = 16
            else:
                if identity.dataset not in d46_data_cache:
                    d46_data_cache[identity.dataset] = _load_d46_pool(identity.dataset, SCREENING_EPOCHS)
                pool = d46_data_cache[identity.dataset]
                cell_pool = {
                    **pool,
                    "config": {
                        **pool["config"],
                        "horizon": int(identity.horizon),
                        "random_state": int(identity.seed),
                    },
                }
                cell = _prepare_d46_target_cell(identity.dataset, cell_pool, parsed)
                cell["batch_size"] = int(cell_pool["config"]["batch_size"])
            stage_a, bundle = _run_stage_a(
                cell=cell,
                horizon=int(identity.horizon),
                seed=int(identity.seed),
                epochs=SCREENING_EPOCHS,
                batch_size=int(cell["batch_size"]),
            )
            payload = _run_cell(
                identity=identity,
                spec=spec,
                stage_a=stage_a,
                bundle=bundle,
                cell=cell,
                metric_protocol=metric_protocol,
                pred_root=pred_root,
                audit_root=audit_root,
                parsed=parsed,
                v3_rows=v3_rows,
                smoke=bool(matrix["smoke"]),
            )
        except Exception as exc:
            payload = _failed_cell_payload(identity, f"{type(exc).__name__}: {exc}")
            print(f"CELL_FAILED {identity.cell_id} reason={exc}")
        write_cell_artifact(cell_path(cells_dir, identity), payload)
        _append_jsonl(
            log_path,
            {
                "cell_id": identity.cell_id,
                "status": payload["status"],
                "dataset": identity.dataset,
                "target_identity": identity.target_identity,
                "horizon": identity.horizon,
                "seed": identity.seed,
            },
        )
        if payload["status"] == "OK":
            completed[identity.identity_tuple] = payload
            print(
                f"CELL_OK {identity.cell_id} V3_2_RMSE={payload['rmse_original']:.6g} "
                f"V3_RMSE={payload['parent_v3_rmse_reference']} "
                f"N_TEST={payload['test_prediction_count']}"
            )
        if matrix["smoke"]:
            smoke_payload = payload

    all_rows: List[dict] = []
    for identity in identities:
        path = cell_path(cells_dir, identity)
        status = classify_cell_artifact(path, identity, spec.architecture_digest)
        if path.is_file():
            all_rows.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            all_rows.append(_failed_cell_payload(identity, "MISSING"))
        if status == "MALFORMED":
            raise V32ContractError(f"malformed cell fails closed: {path}")

    aggregates = _write_aggregates(
        results_dir, identities, all_rows, smoke=bool(matrix["smoke"])
    )
    guard = aggregates["guard"]
    print("\n============================================================")
    print(f"EXPECTED_RUNS={guard['EXPECTED_RUNS']}")
    print(f"OK_RUNS={guard['OK_RUNS']}")
    print(f"FAILED_RUNS={guard['FAILED_RUNS']}")
    print(f"DUPLICATE_RUNS={guard['DUPLICATE_RUNS']}")
    print(f"MISSING_RUNS={guard['MISSING_RUNS']}")
    print(f"ROBUSTNESS_COMPLETE={guard['ROBUSTNESS_COMPLETE']}")
    if smoke_payload is not None:
        _print_smoke(smoke_payload)
    if aggregates["tables"] is not None:
        print(format_headline_tables(aggregates["tables"]), end="")
    print(f"Run root: {run_dir}")
    print(f"All runs: {aggregates['paths']['all_runs']}")
    print("============================================================")
    assert_parent_fingerprints_unchanged(parent_fp_before, fingerprint_parent_artifacts(PROJECT_ROOT))
    if not matrix["smoke"] and counts["expected_runs"] == EXPECTED_RUNS and guard["ROBUSTNESS_COMPLETE"] != "YES":
        raise SystemExit("ROBUSTNESS_COMPLETE=NO")


if __name__ == "__main__":
    main()
