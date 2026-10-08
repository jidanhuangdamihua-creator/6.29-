"""MS-KATL FUSED_TARGET_ONLY_V3_1_CALIBRATED robustness runner.

Derived arm: frozen V3 plus validation-median additive bias calibration.
Does not reselect targets or learning rates, does not modify FORMAL_METHODS,
and does not overwrite V3 screening or three-track robustness outputs.

Usage:
    python scripts/run_ms_katl_fto_v3_1_calibrated_robustness.py --preflight
    python scripts/run_ms_katl_fto_v3_1_calibrated_robustness.py --smoke
    python scripts/run_ms_katl_fto_v3_1_calibrated_robustness.py
    python scripts/run_ms_katl_fto_v3_1_calibrated_robustness.py --output-dir <existing_v3_1_root>
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
from src.transfer_methods.ms_katl_fto_three_track_robustness import jsonable, write_cell_artifact
from src.transfer_methods.ms_katl_fto_v3_1_calibrated import (
    CALIBRATION_TYPE,
    EXPECTED_COMMON_TARGETS,
    EXPECTED_RUNS,
    EXPECTED_V3_ARCHITECTURE_DIGEST,
    EXPECTED_V3_LR,
    METHOD_DISPLAY,
    PARENT_METHOD,
    ROBUSTNESS_LABEL,
    ROBUSTNESS_TRACK,
    V3_1_HORIZONS,
    V3_1_SEEDS,
    V3_RAW_REPRODUCTION_ATOL,
    V3_RAW_REPRODUCTION_RTOL,
    WINDOW_SIZE,
    V31ContractError,
    V31RunIdentity,
    assert_calibration_rule_is_uniform,
    assert_formal_methods_unchanged_v31,
    assert_not_v3_parent_output_dir,
    assert_parent_fingerprints_unchanged,
    assert_v3_parent_contract,
    build_dataset_horizon_tables,
    build_v3_1_identities,
    calibrate_v3_raw,
    cell_path,
    classify_cell_artifact,
    compare_raw_v3_reproduction,
    completion_guard,
    expected_matrix_counts,
    fingerprint_parent_artifacts,
    format_preflight,
    frozen_v3_lr_map,
    inherited_v3_spec,
    load_frozen_v3_robustness_rows,
    load_valid_ok_cells,
    load_v3_1_manifest,
    parent_v3_prediction_path,
    parse_dataset_ids,
    parse_int_tuple,
    run_v3_raw,
    split_target_frames,
    test_label_dates,
    v3_1_architecture,
    v3_1_manifest_path,
    v3_1_revin_config,
    v3_1_training_recipe,
    write_v3_1_manifest,
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
        description="FUSED_TARGET_ONLY_V3_1_CALIBRATED confirmatory robustness."
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
        help="Write configs/ms_katl_fto_v3_1_calibrated_manifest.json and exit.",
    )
    return parser.parse_args()


def _resolve_matrix(args: argparse.Namespace) -> Dict[str, Any]:
    if args.smoke:
        if args.datasets not in {None, "d1", "D1"}:
            raise V31ContractError("smoke is fixed to D1")
        if args.horizons not in {None, "1"}:
            raise V31ContractError("smoke is fixed to horizon=1")
        if args.seeds not in {None, "42"}:
            raise V31ContractError("smoke is fixed to seed=42")
        return {
            "datasets": ("D1",),
            "horizons": (1,),
            "seeds": (42,),
            "smoke": True,
        }
    return {
        "datasets": parse_dataset_ids(args.datasets),
        "horizons": parse_int_tuple(args.horizons, V3_1_HORIZONS, name="horizon"),
        "seeds": parse_int_tuple(args.seeds, V3_1_SEEDS, name="seed"),
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


def _failed_cell_payload(identity: V31RunIdentity, reason: str) -> dict:
    return {
        "status": "FAILED",
        "dataset": identity.dataset,
        "target_identity": identity.target_identity,
        "horizon": int(identity.horizon),
        "seed": int(identity.seed),
        "learning_rate": float(identity.selected_lr),
        "learning_rate_label": identity.selected_lr_label,
        "parent_method": PARENT_METHOD,
        "parent_architecture_digest": EXPECTED_V3_ARCHITECTURE_DIGEST,
        "calibration_type": CALIBRATION_TYPE,
        "calibration_sample_count": None,
        "calibration_bias": None,
        "rmse_v3_raw": None,
        "smape_v3_raw": None,
        "rmse_v3_1_calibrated": None,
        "smape_v3_1_calibrated": None,
        "rmse_improvement_vs_raw_v3": None,
        "smape_improvement_vs_raw_v3": None,
        "metric_space": ORIGINAL_SALES_SPACE,
        "failure_reason": str(reason),
        "run_status": "FAILED",
    }


def _write_prediction_csv(
    path: Path,
    *,
    timestamps: Sequence[str],
    y_true: np.ndarray,
    y_pred_raw: np.ndarray,
    y_pred_cal: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(np.asarray(y_true).reshape(-1).shape[0])
    if len(timestamps) != n:
        raise V31ContractError(
            f"test timestamp count {len(timestamps)} != prediction length {n}"
        )
    pd.DataFrame(
        {
            "timestamp": list(timestamps),
            "y_true": np.asarray(y_true, dtype=np.float64).reshape(-1),
            "y_pred_v3_raw": np.asarray(y_pred_raw, dtype=np.float64).reshape(-1),
            "y_pred_v3_1_calibrated": np.asarray(y_pred_cal, dtype=np.float64).reshape(-1),
        }
    ).to_csv(path, index=False, encoding="utf-8")


def _write_calibration_csv(
    path: Path,
    *,
    meta: Sequence[Mapping[str, str]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    residual: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "validation_timestamp": [item["label_date"] for item in meta],
            "validation_y_true": np.asarray(y_true, dtype=np.float64).reshape(-1),
            "validation_y_pred": np.asarray(y_pred, dtype=np.float64).reshape(-1),
            "validation_residual": np.asarray(residual, dtype=np.float64).reshape(-1),
            "window_start": [item["window_start"] for item in meta],
            "window_end": [item["window_end"] for item in meta],
        }
    ).to_csv(path, index=False, encoding="utf-8")


def _run_cell(
    *,
    identity: V31RunIdentity,
    spec,
    stage_a: Any,
    bundle: Any,
    cell: dict,
    metric_protocol: Optional[dict],
    pred_root: Path,
    cal_root: Path,
    parsed: dict,
    smoke: bool,
) -> dict:
    set_protocol_seed(int(identity.seed), include_frameworks=True)
    raw_train, raw_val, raw_test = split_target_frames(cell["target_model_df"])
    raw = run_v3_raw(
        bundle=bundle,
        identity=identity,
        spec=spec,
        epochs=SCREENING_EPOCHS,
        batch_size=int(cell.get("batch_size") or 16),
        metric_protocol=metric_protocol,
    )
    calibrated = calibrate_v3_raw(
        raw=raw,
        bundle=bundle,
        raw_train=raw_train,
        raw_val=raw_val,
        raw_test=raw_test,
        identity=identity,
        feature_columns=cell["model_feature_cols"],
    )
    reproduction = None
    parent_pred = parent_v3_prediction_path(PROJECT_ROOT, identity, parsed)
    if smoke or parent_pred.is_file():
        reproduction = compare_raw_v3_reproduction(
            calibrated["y_pred_v3_raw"],
            parent_pred,
            y_true_original=calibrated["y_true"],
            rmse_raw=float(calibrated["rmse_v3_raw"]),
        )
    timestamps = test_label_dates(
        raw_test,
        horizon=int(identity.horizon),
        window_size=WINDOW_SIZE,
        feature_columns=cell["model_feature_cols"],
    )
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
        y_true=calibrated["y_true"],
        y_pred_raw=calibrated["y_pred_v3_raw"],
        y_pred_cal=calibrated["y_pred_v3_1_calibrated"],
    )
    cal_dir = (
        cal_root
        / identity.dataset
        / parsed["prediction_dirname"]
        / f"h{int(identity.horizon)}_s{int(identity.seed)}_{identity.selected_lr_label}"
    )
    cal_path = cal_dir / "validation_calibration.csv"
    _write_calibration_csv(
        cal_path,
        meta=calibrated["validation_meta"],
        y_true=calibrated["validation_y_true"],
        y_pred=calibrated["validation_y_pred"],
        residual=calibrated["validation_residual"],
    )
    sales_min, sales_max = sales_minmax(bundle.target_scaler, bundle.target_feature_columns)
    flags = assert_calibration_rule_is_uniform()
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
        "calibration_type": CALIBRATION_TYPE,
        "calibration_sample_count": int(calibrated["calibration_sample_count"]),
        "calibration_bias": float(calibrated["calibration_bias"]),
        "rmse_v3_raw": float(calibrated["rmse_v3_raw"]),
        "smape_v3_raw": float(calibrated["smape_v3_raw"]),
        "rmse_v3_1_calibrated": float(calibrated["rmse_v3_1_calibrated"]),
        "smape_v3_1_calibrated": float(calibrated["smape_v3_1_calibrated"]),
        "rmse_improvement_vs_raw_v3": calibrated["rmse_improvement_vs_raw_v3"],
        "smape_improvement_vs_raw_v3": calibrated["smape_improvement_vs_raw_v3"],
        "metric_space": ORIGINAL_SALES_SPACE,
        "architecture": dict(v3_1_architecture()),
        "architecture_digest": spec.architecture_digest,
        "training_recipe": dict(v3_1_training_recipe()),
        "revin": v3_1_revin_config(),
        "sales_min": sales_min,
        "sales_max": sales_max,
        "prediction_artifact_path": str(pred_path),
        "calibration_artifact_path": str(cal_path),
        "raw_residual_mean": calibrated["raw_residual_mean"],
        "raw_residual_median": calibrated["raw_residual_median"],
        "post_calibration_residual_mean": calibrated["post_calibration_residual_mean"],
        "post_calibration_residual_median": calibrated["post_calibration_residual_median"],
        "V3_RAW_REPRODUCTION": None if reproduction is None else reproduction["V3_RAW_REPRODUCTION"],
        "V3_RAW_REPRODUCTION_MAX_ABS": None if reproduction is None else reproduction["max_abs_diff"],
        "V3_RAW_REPRODUCTION_MODE": None if reproduction is None else reproduction.get("mode"),
        "V3_RAW_REPRODUCTION_RMSE_REL": None if reproduction is None else reproduction.get("rmse_rel"),
        "V3_RAW_REPRODUCTION_TOLERANCE": (
            None
            if reproduction is None
            else (
                f"same_process_pred_rtol={V3_RAW_REPRODUCTION_RTOL},"
                f"atol={V3_RAW_REPRODUCTION_ATOL};"
                f"parent_rmse_rtol={reproduction.get('parent_rmse_rtol')}"
            )
        ),
        "TEST_TRUTH_USED_FOR_CALIBRATION": "NO",
        **flags,
        "source_identities": [tuple(item) for item in stage_a.source_identities],
        "run_status": "OK",
        "failure_reason": "",
        "identity_tuple": list(identity.identity_tuple),
    }


def _write_aggregates(
    results_dir: Path,
    identities: Sequence[V31RunIdentity],
    rows: Sequence[Mapping[str, Any]],
    *,
    smoke: bool,
) -> Dict[str, Any]:
    results_dir.mkdir(parents=True, exist_ok=True)
    all_runs_path = results_dir / "v3_1_all_runs.csv"
    _dump_csv(all_runs_path, [_csv_ready(dict(row)) for row in rows])
    guard = completion_guard(identities, rows)
    (results_dir / "v3_1_completion_guard.json").write_text(
        json.dumps(guard, indent=2), encoding="utf-8"
    )
    tables = None
    if not smoke and guard["ROBUSTNESS_COMPLETE"] == "YES":
        v3_rows = load_frozen_v3_robustness_rows(PROJECT_ROOT)
        tables = build_dataset_horizon_tables(rows, v3_rows, PROJECT_ROOT)
        _dump_csv(results_dir / "v3_1_summary_by_dataset_horizon.csv", tables["rmse"])
        vs_v3_rows = []
        for item in tables["rmse"]:
            vs_v3_rows.append(
                {
                    "DATASET": item["DATASET"],
                    "HORIZON": item["HORIZON"],
                    "V3_RMSE_MEAN": item["V3_RMSE_MEAN"],
                    "V3_RMSE_STD": item["V3_RMSE_STD"],
                    "V3_1_RMSE_MEAN": item["V3_1_RMSE_MEAN"],
                    "V3_1_RMSE_STD": item["V3_1_RMSE_STD"],
                    "V3_1_VS_V3_PCT": item["V3_1_VS_V3_PCT"],
                }
            )
        _dump_csv(results_dir / "v3_1_vs_v3.csv", vs_v3_rows)
        _dump_csv(results_dir / "v3_1_vs_baselines.csv", tables["rmse"])
        _dump_csv(results_dir / "v3_1_vs_baselines_smape.csv", tables["smape"])
        (results_dir / "v3_1_diagnostics.json").write_text(
            json.dumps(tables["diagnostics"], indent=2), encoding="utf-8"
        )
    return {
        "guard": guard,
        "tables": tables,
        "paths": {"all_runs": all_runs_path},
    }


def _print_smoke(payload: Mapping[str, Any]) -> None:
    print(f"SMOKE_CELL_OK={'YES' if payload.get('status') == 'OK' else 'NO'}")
    print(f"V3_RAW_REPRODUCTION={payload.get('V3_RAW_REPRODUCTION')}")
    print(f"V3_RAW_REPRODUCTION_MODE={payload.get('V3_RAW_REPRODUCTION_MODE')}")
    print(f"V3_RAW_REPRODUCTION_TOLERANCE={payload.get('V3_RAW_REPRODUCTION_TOLERANCE')}")
    print(f"CALIBRATION_SAMPLE_COUNT={payload.get('calibration_sample_count')}")
    print(f"CALIBRATION_BIAS={payload.get('calibration_bias')}")
    print(f"RAW_V3_RMSE={payload.get('rmse_v3_raw')}")
    print(f"V3_1_RMSE={payload.get('rmse_v3_1_calibrated')}")
    print(f"RAW_V3_SMAPE={payload.get('smape_v3_raw')}")
    print(f"V3_1_SMAPE={payload.get('smape_v3_1_calibrated')}")
    print(f"TEST_TRUTH_USED_FOR_CALIBRATION={payload.get('TEST_TRUTH_USED_FOR_CALIBRATION')}")


def main() -> None:
    args = _parse_args()
    assert_formal_methods_unchanged_v31()
    parent_fp_before = fingerprint_parent_artifacts(PROJECT_ROOT)
    if args.write_manifest:
        path = write_v3_1_manifest(PROJECT_ROOT)
        print(f"V3_1_MANIFEST={path}")
        return

    frozen = assert_v3_parent_contract(PROJECT_ROOT)
    if not v3_1_manifest_path(PROJECT_ROOT).is_file():
        write_v3_1_manifest(PROJECT_ROOT)
    load_v3_1_manifest(PROJECT_ROOT)
    lr_map = frozen_v3_lr_map(frozen)
    matrix = _resolve_matrix(args)
    identities = build_v3_1_identities(
        datasets=matrix["datasets"],
        horizons=matrix["horizons"],
        seeds=matrix["seeds"],
        lr_map=lr_map,
    )
    counts = expected_matrix_counts(matrix["datasets"], matrix["horizons"], matrix["seeds"])
    if (
        not matrix["smoke"]
        and tuple(matrix["datasets"]) == SCREENING_DATASETS
        and tuple(matrix["horizons"]) == V3_1_HORIZONS
        and tuple(matrix["seeds"]) == V3_1_SEEDS
    ):
        if counts["expected_runs"] != EXPECTED_RUNS:
            raise V31ContractError(f"full matrix must be {EXPECTED_RUNS} runs, got {counts}")
        if len(identities) != EXPECTED_RUNS:
            raise V31ContractError(f"UNIQUE_IDENTITIES={len(identities)} != {EXPECTED_RUNS}")
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
    cal_root = run_dir / "calibration"
    cells_dir = run_dir / "cells"
    logs_dir = run_dir / "logs"
    for path in (results_dir, pred_root, cal_root, cells_dir, logs_dir):
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
                "calibration_type": CALIBRATION_TYPE,
                "frozen_lr_mapping": {f"{ds}|{tgt}": lr for (ds, tgt), lr in EXPECTED_V3_LR.items()},
                "formal_methods_change": False,
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

    for identity in pending:
        parsed = parse_target_identity(identity.dataset, identity.target_identity)
        print(
            f"\n=== V3.1 {identity.dataset} {identity.target_identity} "
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
                cal_root=cal_root,
                parsed=parsed,
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
                f"CELL_OK {identity.cell_id} RAW_RMSE={payload['rmse_v3_raw']:.6g} "
                f"V3_1_RMSE={payload['rmse_v3_1_calibrated']:.6g} "
                f"BIAS={payload['calibration_bias']:.6g} "
                f"N_CAL={payload['calibration_sample_count']}"
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
            raise V31ContractError(f"malformed cell fails closed: {path}")

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
        diag = aggregates["tables"]["diagnostics"]
        print(f"V3_1_BEATS_BEST_BL={diag['beats_best_bl']}/{diag['n']}")
        print(f"V3_1_TOP2={diag['top2']}/{diag['n']}")
        print(f"V3_1_TOP3={diag['top3']}/{diag['n']}")
    print(f"Run root: {run_dir}")
    print(f"All runs: {aggregates['paths']['all_runs']}")
    print("============================================================")
    assert_parent_fingerprints_unchanged(parent_fp_before, fingerprint_parent_artifacts(PROJECT_ROOT))
    if not matrix["smoke"] and counts["expected_runs"] == EXPECTED_RUNS and guard["ROBUSTNESS_COMPLETE"] != "YES":
        raise SystemExit("ROBUSTNESS_COMPLETE=NO")


if __name__ == "__main__":
    main()
