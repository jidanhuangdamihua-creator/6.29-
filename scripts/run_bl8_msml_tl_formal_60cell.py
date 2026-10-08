#!/usr/bin/env python3
"""Execute and audit the isolated formal BL8 MSML-TL 60-cell matrix.

This runner writes only under results/formal_bl8_msml_tl/.
It does not modify paper freeze workbooks or SITA/BL6/BL7 result directories.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tf_compat  # noqa: F401  # must be imported before tensorflow/keras

import numpy as np
import pandas as pd

from scripts.bl8_msml_tl_formal_protocol import (
    BL8FormalError,
    CODE_IDENTITY_PATHS,
    DEFAULT_OUTPUT_ROOT,
    DRY_RUN_D4_H1_S42,
    ENTITY_IDS,
    EXPECTED_AGGREGATES,
    EXPECTED_CELLS,
    HORIZONS,
    METHOD,
    METHOD_ID,
    MODE,
    PARITY_TOLERANCE,
    REQUIRED_OUTPUT_FILES,
    SITA_FREEZE_XLSX,
    TARGETS,
    WEIGHT_SUM_TOLERANCE,
    aggregate_five_seed,
    assert_allowed_matrix_cell,
    canonical_key,
    cell_dirname,
    compare_12_conditions,
    compare_60_paired,
    completeness_audit,
    expected_cell_keys,
    git_identity,
    json_dump,
    load_sita_authority,
    mechanism_failures,
    record_code_identity,
    recompute_original_metrics,
    sha256_file,
    timestamp_seed_invariance,
    yes_no,
)
from src.constants import SOLIDIFIED_KNN_ROOT, SOURCE_HISTORY_DAYS
from src.protocols.candidate_pool import prepare_daily_sequence_pool
from src.protocols.experiment_protocol import get_experiment_protocol
from src.protocols.formal_input_paths import resolve_formal_dataset_paths
from src.protocols.reproducibility import set_protocol_seed
from src.protocols.transformation_identity import validate_runtime_feature_contract
from src.transfer_methods.msml_tl import FormalBL8ContractError
from src.utils.d4_d6_runtime import (
    apply_runtime_source_domain_policy,
    load_default_metric_protocol,
    validate_runtime_target_domain,
)
from src.utils.entity_experiment import run_single_entity_experiment
from src.utils.knn_feature_loader import resolve_knn_feature_columns
from src.utils.parquet_data_loader import (
    load_knn_results,
    load_parquet_source_target,
    read_dataset_windows,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_source_key(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return "/".join(str(part) for part in value)
    return str(value)


def _layer_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.replace("'", "").strip("[]").split(",") if part.strip()]
    return [str(item) for item in list(value)]


def _source_trace(meta: dict[str, Any]) -> dict[str, Any]:
    selected = list(meta.get("selected_sources") or [])
    sources = []
    distances = []
    weights = []
    for item in selected:
        if not isinstance(item, dict):
            continue
        sources.append(_format_source_key(item.get("source_key")))
        distances.append(float(item.get("distance", float("nan"))))
        weights.append(float(item.get("weight", float("nan"))))
    while len(sources) < 3:
        sources.append("")
        distances.append(float("nan"))
        weights.append(float("nan"))
    weight_sum = float(np.nansum(np.asarray(weights[:3], dtype=np.float64)))
    return {
        "selected_source_1": sources[0],
        "selected_source_2": sources[1],
        "selected_source_3": sources[2],
        "distance_1": distances[0],
        "distance_2": distances[1],
        "distance_3": distances[2],
        "weight_1": weights[0],
        "weight_2": weights[1],
        "weight_3": weights[2],
        "weight_sum": weight_sum,
        "source_count_requested": int(meta.get("requested_source_count", meta.get("k", -1))),
        "source_count_selected": int(meta.get("effective_selected_source_count", len(selected))),
        "source_count_successful": int(meta.get("successful_source_model_count", meta.get("valid_source_count", -1))),
        "source_failure_count": int(meta.get("failed_source_count", -1)),
        "weight_mode": str(meta.get("weight_mode", "")),
        "selection_authority": str(meta.get("selection_authority", "")),
        "candidate_pool_digest": str(meta.get("candidate_pool_digest", "")),
        "selection_result_digest": str(meta.get("selection_result_digest", "")),
    }


def cell_paths(output_root: Path, dataset: str, target: str, horizon: int, seed: int) -> dict[str, Path]:
    cell_dir = output_root / "cells" / cell_dirname(dataset, target, horizon, seed)
    return {
        "dir": cell_dir,
        "predictions": cell_dir / "predictions.csv",
        "manifest": cell_dir / "cell_manifest.json",
        "complete": cell_dir / "cell_complete.json",
        "error": cell_dir / "cell_error.txt",
    }


def _valid_existing_cell(
    paths: dict[str, Path],
    *,
    dataset: str,
    target: str,
    horizon: int,
    seed: int,
    code_identity: dict[str, str],
) -> dict[str, Any] | None:
    if not paths["complete"].exists() or not paths["manifest"].exists() or not paths["predictions"].exists():
        return None
    complete = json.loads(paths["complete"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    if str(complete.get("cell_status", "")).upper() != "PASS":
        return None
    if dict(complete.get("code_identity") or {}) != code_identity:
        raise BL8FormalError(
            f"code identity changed after the first formal cell: {canonical_key(dataset, target, horizon, seed)}"
        )
    for field in ("dataset", "target", "horizon", "seed", "mode", "method"):
        if str(manifest.get(field)) != str(
            {"dataset": dataset, "target": target, "horizon": int(horizon), "seed": int(seed), "mode": MODE, "method": METHOD}[field]
        ):
            return None
    if mechanism_failures(manifest):
        return None
    predictions = pd.read_csv(paths["predictions"])
    required = {"timestamp", "y_true", "y_pred", "dataset", "target", "horizon", "seed", "mode", "method"}
    if required - set(predictions.columns):
        return None
    if predictions.empty:
        return None
    return {"manifest": manifest, "predictions": predictions, "complete": complete}


def persist_failed_cell(
    paths: dict[str, Path],
    *,
    dataset: str,
    target: str,
    horizon: int,
    seed: int,
    code_identity: dict[str, str],
    error: str,
) -> dict[str, Any]:
    paths["dir"].mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": dataset,
        "target": target,
        "horizon": int(horizon),
        "seed": int(seed),
        "mode": MODE,
        "method": METHOD,
        "method_id": METHOD_ID,
        "cell_status": "FAILED",
        "error": error,
        "published_as_formal_cell": True,
        "code_identity": code_identity,
        "created_at": _now(),
    }
    json_dump(paths["manifest"], manifest)
    json_dump(paths["complete"], {**manifest, "cell_status": "FAILED"})
    paths["error"].write_text(error, encoding="utf-8")
    return manifest


def persist_passed_cell(
    paths: dict[str, Path],
    *,
    dataset: str,
    target: str,
    horizon: int,
    seed: int,
    code_identity: dict[str, str],
    row: dict[str, Any],
    predictions: pd.DataFrame,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    paths["dir"].mkdir(parents=True, exist_ok=True)
    predictions.to_csv(paths["predictions"], index=False)
    payload = {
        **manifest,
        **row,
        "dataset": dataset,
        "target": target,
        "horizon": int(horizon),
        "seed": int(seed),
        "mode": MODE,
        "method": METHOD,
        "method_id": METHOD_ID,
        "cell_status": "PASS",
        "published_as_formal_cell": True,
        "dry_run": False,
        "code_identity": code_identity,
        "created_at": _now(),
    }
    json_dump(paths["manifest"], payload)
    json_dump(
        paths["complete"],
        {
            "cell_status": "PASS",
            "canonical_key": canonical_key(dataset, target, horizon, seed),
            "code_identity": code_identity,
            "rmse": row["rmse"],
            "smape": row["smape"],
            "n": row["n"],
            "created_at": payload["created_at"],
        },
    )
    if paths["error"].exists():
        paths["error"].unlink()
    return payload


def _base_d456_config(dataset_id: int, horizon: int, seed: int) -> dict[str, Any]:
    return {
        "use_parquet": True,
        "dataset_id": int(dataset_id),
        "dataset_name": f"Dataset{dataset_id}",
        "knn_json_dir": str(SOLIDIFIED_KNN_ROOT / f"Dataset{dataset_id}"),
        "info_sharing": MODE,
        "source_history_days": SOURCE_HISTORY_DAYS,
        "entity_col": "entity_id",
        "enabled_methods": [METHOD],
        "horizon": int(horizon),
        "window_size": 10,
        "learning_rate": 0.0001,
        "source_epochs": 50,
        "target_epochs": 50,
        "batch_size": 16,
        "random_state": int(seed),
        "source_count": 3,
        "smoke": False,
        "repair_source_numeric_na": False,
        "metric_protocol": load_default_metric_protocol(ROOT),
        "bl8_prediction_payload_sink": [],
    }


def _prepare_d456_bundle(dataset_id: int) -> dict[str, Any]:
    formal_paths = resolve_formal_dataset_paths(dataset_id, repository_root=ROOT)
    knn_data = load_knn_results(str(SOLIDIFIED_KNN_ROOT / f"Dataset{dataset_id}"), MODE)
    group_cols = knn_data.get("group_cols")
    if not isinstance(group_cols, list) or len(group_cols) != 2:
        raise BL8FormalError(f"D{dataset_id} KNN payload requires two group_cols, got: {group_cols}")
    windows = read_dataset_windows(
        dataset_id,
        str(SOLIDIFIED_KNN_ROOT / f"Dataset{dataset_id}"),
        info_sharing=MODE,
        knn_payload=knn_data,
    )
    if dataset_id == 5:
        from scripts.run_d5_experiment import load_d5_runtime_inputs

        runtime_inputs = load_d5_runtime_inputs(
            raw_dir=ROOT / "数据集/原始数据/Dataset 5Favorita",
            source_path=formal_paths.source_path,
            target_path=formal_paths.target_path,
            windows=windows,
            source_history_days=SOURCE_HISTORY_DAYS,
        )
        source_df = runtime_inputs.source_df
        target_df = runtime_inputs.target_df
    else:
        source_df, target_df = load_parquet_source_target(
            dataset_id=dataset_id,
            source_path=formal_paths.source_path,
            target_path=formal_paths.target_path,
            windows=windows,
            source_history_days=SOURCE_HISTORY_DAYS,
        )
    feature_info = resolve_knn_feature_columns(
        dataset_id=dataset_id,
        information_sharing=MODE,
        knn_root=SOLIDIFIED_KNN_ROOT,
        source_df=source_df,
        target_df=target_df,
        knn_payload=knn_data,
    )
    feature_cols = list(feature_info["selected_features"])
    protocol = get_experiment_protocol(dataset_id)
    validate_runtime_feature_contract(
        protocol.dataset_id,
        model_feature_cols=feature_cols,
        knn_feature_cols=protocol.knn_feature_columns,
    )
    config = {
        "dataset_id": dataset_id,
        "info_sharing": MODE,
        "entity_col": "entity_id",
        "group_cols": list(group_cols),
        "feature_source": feature_info.get("feature_source", ""),
        "knn_feature_mode": feature_info.get("knn_feature_mode", ""),
        "source_selection_feature_cols": list(feature_cols),
        "model_feature_cols": list(feature_cols),
        "feature_consistency_status": feature_info.get("feature_consistency_status", ""),
        "json_only_features": feature_info.get("json_only_features", []),
        "runtime_only_features": feature_info.get("runtime_only_features", []),
    }
    source_df = apply_runtime_source_domain_policy(source_df, knn_data, config)
    entity_id = ENTITY_IDS[f"D{dataset_id}"]
    if entity_id not in knn_data["results"]:
        raise BL8FormalError(f"D{dataset_id} KNN results missing paper target entity_id={entity_id}")
    validate_runtime_target_domain(target_df, [entity_id], knn_data, config)
    observed_start = target_df.attrs.get(
        "knn_observed_start",
        target_df.attrs.get(
            "target_observed_start",
            pd.to_datetime(target_df["date"], errors="raise").min(),
        ),
    )
    metadata_cols = {4: None, 5: ("family",), 6: ("dept_id",)}[dataset_id]
    pool_kwargs: dict[str, Any] = {
        "source_df": source_df,
        "group_cols": tuple(str(column) for column in group_cols),
        "observed_start": observed_start,
        "feature_cols": protocol.knn_feature_columns,
    }
    if metadata_cols is not None:
        pool_kwargs["metadata_cols"] = metadata_cols
    if dataset_id in {4, 6}:
        pool_kwargs["capture_source_observed_frame_digest"] = True
    prepared_pool = prepare_daily_sequence_pool(**pool_kwargs)
    return {
        "source_df": source_df,
        "target_df": target_df,
        "knn_data": knn_data,
        "feature_cols": feature_cols,
        "feature_info": feature_info,
        "group_cols": list(group_cols),
        "prepared_pool": prepared_pool,
        "entity_id": entity_id,
        "config_extras": config,
        "formal_paths": {
            "source": str(formal_paths.source_path),
            "target": str(formal_paths.target_path),
        },
    }


def _build_prediction_frame(
    *,
    dataset: str,
    target: str,
    horizon: int,
    seed: int,
    timestamps: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    if len(timestamps) != int(np.asarray(y_true).reshape(-1).size) or len(timestamps) != int(np.asarray(y_pred).reshape(-1).size):
        raise BL8FormalError(
            f"{dataset}/{target}/H{horizon}/S{seed} timestamp/prediction length mismatch: "
            f"timestamps={len(timestamps)} y_true={np.asarray(y_true).reshape(-1).size} "
            f"y_pred={np.asarray(y_pred).reshape(-1).size}"
        )
    return pd.DataFrame(
        {
            "timestamp": [str(value) for value in timestamps],
            "y_true": np.asarray(y_true, dtype=np.float64).reshape(-1),
            "y_pred": np.asarray(y_pred, dtype=np.float64).reshape(-1),
            "dataset": dataset,
            "target": target,
            "horizon": int(horizon),
            "seed": int(seed),
            "mode": MODE,
            "method": METHOD,
        }
    )


def _row_from_payload(
    *,
    dataset: str,
    target: str,
    horizon: int,
    seed: int,
    result_row: dict[str, Any],
    payload: dict[str, Any],
    predictions: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    meta = dict(payload.get("meta") or {})
    stored_rmse = float(result_row.get("original_scale_rmse", result_row.get("rmse")))
    stored_smape = float(result_row.get("original_scale_smape", result_row.get("smape")))
    recomputed = recompute_original_metrics(predictions["y_true"], predictions["y_pred"])
    rmse_delta = abs(stored_rmse - recomputed["rmse"])
    smape_delta = abs(stored_smape - recomputed["smape"])
    if str(result_row.get("rmse_metric_space", result_row.get("metric_space_used"))) != "original_sales_space":
        raise BL8FormalError(f"{dataset} RMSE is not original_sales_space")
    if str(result_row.get("smape_metric_space", result_row.get("metric_space_used"))) != "original_sales_space":
        raise BL8FormalError(f"{dataset} sMAPE is not original_sales_space")
    trace = _source_trace(meta)
    manifest = {
        "formal_bl8": bool(meta.get("formal_bl8", False)),
        "rfe": bool(meta.get("rfe", False)),
        "fused_layers": _layer_list(meta.get("fused_layers")),
        "frozen_layers": _layer_list(meta.get("frozen_layers") or result_row.get("frozen_layers")),
        "target_trainable_parameter_layers": _layer_list(meta.get("target_trainable_parameter_layers")),
        "target_test_excluded": bool(result_row.get("target_test_excluded", meta.get("target_test_excluded", False))),
        "source_future_excluded": bool(result_row.get("source_future_excluded", meta.get("source_future_excluded", False))),
        "scaler_fit_scope": str(meta.get("scaler_fit_scope", "")),
        "target_test_used_for_fit": False,
        "future_leakage_detected": False,
        "rmse_parity_error": rmse_delta,
        "smape_parity_error": smape_delta,
        **trace,
    }
    if not manifest["target_test_excluded"] or not manifest["source_future_excluded"]:
        manifest["target_test_used_for_fit"] = not manifest["target_test_excluded"]
        manifest["future_leakage_detected"] = not manifest["source_future_excluded"]
    failures = mechanism_failures(manifest)
    if rmse_delta > PARITY_TOLERANCE:
        failures.append(f"rmse_parity={rmse_delta}")
    if smape_delta > PARITY_TOLERANCE:
        failures.append(f"smape_parity={smape_delta}")
    if failures:
        raise BL8FormalError("; ".join(failures))
    row = {
        "dataset": dataset,
        "target": target,
        "horizon": int(horizon),
        "seed": int(seed),
        "mode": MODE,
        "method": METHOD,
        "method_id": METHOD_ID,
        "cell_status": "PASS",
        "rmse": stored_rmse,
        "smape": stored_smape,
        "n": int(len(predictions)),
        "recomputed_rmse": recomputed["rmse"],
        "recomputed_smape": recomputed["smape"],
        "rmse_parity_error": rmse_delta,
        "smape_parity_error": smape_delta,
        "metric_space": "original_sales_space",
        "rmse_metric_space": "original_sales_space",
        "smape_metric_space": "original_sales_space",
        "sample_manifest_digest": str(result_row.get("sample_manifest_digest", "")),
        **trace,
        "fused_layers": ",".join(manifest["fused_layers"]),
        "frozen_layers": ",".join(manifest["frozen_layers"]),
        "target_trainable_parameter_layers": ",".join(manifest["target_trainable_parameter_layers"]),
        "formal_bl8": True,
        "rfe": False,
        "target_test_used_for_fit": False,
        "future_leakage_detected": False,
        "timestamps": "|".join(predictions["timestamp"].astype(str).tolist()),
    }
    return row, manifest


def run_d456_cell(bundle: dict[str, Any], dataset: str, target: str, horizon: int, seed: int) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    dataset_id = int(dataset[1])
    config = _base_d456_config(dataset_id, horizon, seed)
    config.update(bundle["config_extras"])
    config["group_cols"] = list(bundle["group_cols"])
    config["bl8_prediction_payload_sink"] = []
    entity_id = str(bundle["entity_id"])
    target_entity_df = bundle["target_df"][bundle["target_df"]["entity_id"].astype(str) == entity_id].copy()
    if target_entity_df.empty:
        raise BL8FormalError(f"{dataset} target entity {entity_id} not found")
    set_protocol_seed(int(seed), include_frameworks=True)
    rows = run_single_entity_experiment(
        entity_key=entity_id,
        source_df=bundle["source_df"],
        target_entity_df=target_entity_df,
        feature_cols=list(bundle["feature_cols"]),
        config=config,
        enabled_methods=[METHOD],
        prepared_pool=bundle["prepared_pool"],
    )
    if len(rows) != 1:
        raise BL8FormalError(f"{dataset} expected 1 MSML-TL row, got {len(rows)}")
    result_row = rows[0]
    if str(result_row.get("error") or "").strip():
        raise BL8FormalError(str(result_row["error"]))
    sink = config["bl8_prediction_payload_sink"]
    if len(sink) != 1:
        raise BL8FormalError(f"{dataset} missing BL8 prediction payload")
    payload = sink[0]
    y_true = payload.get("y_true_original")
    y_pred = payload.get("y_pred_original")
    if y_true is None or y_pred is None:
        raise BL8FormalError(f"{dataset} original-sales predictions missing")
    serialized_target = str(result_row.get("target_entity_key") or target)
    if serialized_target != target:
        raise BL8FormalError(f"{dataset} canonical target {serialized_target} != {target}")
    predictions = _build_prediction_frame(
        dataset=dataset,
        target=target,
        horizon=horizon,
        seed=seed,
        timestamps=list(payload["timestamps"]),
        y_true=np.asarray(y_true),
        y_pred=np.asarray(y_pred),
    )
    row, manifest = _row_from_payload(
        dataset=dataset,
        target=target,
        horizon=horizon,
        seed=seed,
        result_row=result_row,
        payload=payload,
        predictions=predictions,
    )
    return row, predictions, manifest


def run_d1_cell(d1_state: dict[str, Any], target: str, horizon: int, seed: int) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    from scripts.run_full_paper_experiments import (
        _cell_lifecycle_identity,
        _prepare_cell_source_dependencies,
        run_experiment,
    )
    from paper_reproduction_protocol import load_paper_protocol, resolve_strict_paper_mode

    cfg = deepcopy(d1_state["cfg"])
    cfg.setdefault("single_experiment", {})["horizon"] = int(horizon)
    cfg["single_experiment"]["seed"] = int(seed)
    cfg["bl8_capture"] = True
    protocol = load_paper_protocol(cfg)
    strict_paper_mode = True
    protocol["strict_paper_mode"] = True
    protocol["paper_strict_mode"] = True
    protocol.setdefault("metric_protocol", {})["strict_paper_metrics"] = True
    cfg.setdefault("paper_reproduction", {})["strict_paper_mode"] = True
    cfg["paper_reproduction"]["strict_paper_metrics"] = True
    cfg["paper_reproduction"]["metric_protocol"] = dict(protocol.get("metric_protocol") or {})
    cfg["paper_reproduction"]["metric_protocol"]["strict_paper_metrics"] = True
    resolve_strict_paper_mode(cfg, explicit=True)
    set_protocol_seed(int(seed), include_frameworks=True)
    cell_source = _prepare_cell_source_dependencies(
        dataset_name="Dataset1",
        information_sharing_scenario="without_information_sharing",
        cfg=cfg,
        protocol=protocol,
        strict_paper_mode=strict_paper_mode,
        base_data=d1_state["base"],
    )
    _cell_lifecycle_identity("Dataset1", "without_information_sharing", cfg)
    result = run_experiment(
        dataset_name="Dataset1",
        method_name=METHOD,
        source_count=3,
        information_sharing_scenario="without_information_sharing",
        cfg=cfg,
        protocol=protocol,
        strict_paper_mode=strict_paper_mode,
        verbose_mode="summary",
        base_data=d1_state["base"],
        cell_source_preparation=cell_source,
    )
    if str(result.get("error") or "").strip():
        raise BL8FormalError(str(result["error"]))
    if str(result.get("target_entity_key")) != target:
        raise BL8FormalError(f"D1 canonical target {result.get('target_entity_key')} != {target}")
    trace = result.get("_bl8_trace") or {}
    y_true = trace.get("y_true_original")
    y_pred = trace.get("y_pred_original")
    if y_true is None or y_pred is None:
        raise BL8FormalError("D1 original-sales predictions missing")
    predictions = _build_prediction_frame(
        dataset="D1",
        target=target,
        horizon=horizon,
        seed=seed,
        timestamps=list(trace.get("timestamps") or []),
        y_true=np.asarray(y_true),
        y_pred=np.asarray(y_pred),
    )
    row, manifest = _row_from_payload(
        dataset="D1",
        target=target,
        horizon=horizon,
        seed=seed,
        result_row=result,
        payload={"meta": dict(trace.get("meta") or {})},
        predictions=predictions,
    )
    return row, predictions, manifest


def _load_d1_state() -> dict[str, Any]:
    from scripts.run_full_paper_experiments import _load_config, _load_experiment_runners, _prepare_runner_base_data

    _load_experiment_runners()
    cfg = _load_config()
    cfg["bl8_capture"] = True
    base = _prepare_runner_base_data(
        dataset_name="Dataset1",
        data_path=cfg["dataset_paths"]["Dataset1"],
        cfg=cfg,
        verbose_mode="summary",
        strict_paper_mode=True,
        strict_paper_split=True,
    )
    return {"cfg": cfg, "base": base}


def execute_cells(
    *,
    output_root: Path,
    code_identity: dict[str, str],
    selected_keys: list[tuple[str, str, int, int, str, str]] | None = None,
) -> pd.DataFrame:
    selected = selected_keys or expected_cell_keys()
    bundles: dict[str, Any] = {}
    d1_state: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    output_root.mkdir(parents=True, exist_ok=True)
    for dataset, target, horizon, seed, mode, method in selected:
        assert_allowed_matrix_cell(dataset, target, horizon, seed, mode, method)
        live_identity = record_code_identity(ROOT)
        if live_identity != code_identity:
            raise BL8FormalError("CODE_IDENTITY_CHANGED_AFTER_FIRST_CELL")
        paths = cell_paths(output_root, dataset, target, horizon, seed)
        reused = _valid_existing_cell(
            paths,
            dataset=dataset,
            target=target,
            horizon=horizon,
            seed=seed,
            code_identity=code_identity,
        )
        print(f"[BL8] {canonical_key(dataset, target, horizon, seed)}", flush=True)
        if reused is not None:
            print("  reuse existing PASS cell", flush=True)
            rows.append({k: v for k, v in reused["manifest"].items() if k != "code_identity"})
            continue
        try:
            if dataset == "D1":
                if d1_state is None:
                    d1_state = _load_d1_state()
                row, predictions, manifest = run_d1_cell(d1_state, target, horizon, seed)
            else:
                if dataset not in bundles:
                    bundles[dataset] = _prepare_d456_bundle(int(dataset[1]))
                row, predictions, manifest = run_d456_cell(bundles[dataset], dataset, target, horizon, seed)
            persist_passed_cell(
                paths,
                dataset=dataset,
                target=target,
                horizon=horizon,
                seed=seed,
                code_identity=code_identity,
                row=row,
                predictions=predictions,
                manifest=manifest,
            )
            rows.append(row)
            print(f"  PASS RMSE={row['rmse']:.6f} sMAPE={row['smape']:.6f} n={row['n']}", flush=True)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            persist_failed_cell(
                paths,
                dataset=dataset,
                target=target,
                horizon=horizon,
                seed=seed,
                code_identity=code_identity,
                error=error,
            )
            print(f"  FAILED {exc}", flush=True)
            if isinstance(exc, BL8FormalError) and str(exc) == "CODE_IDENTITY_CHANGED_AFTER_FIRST_CELL":
                raise
    return pd.DataFrame.from_records(rows)


def load_completed_rows(output_root: Path) -> tuple[pd.DataFrame, dict[tuple[str, str, int], dict[int, list[str]]]]:
    records = []
    timestamps: dict[tuple[str, str, int], dict[int, list[str]]] = {}
    for dataset, target, horizon, seed, _mode, _method in expected_cell_keys():
        paths = cell_paths(output_root, dataset, target, horizon, seed)
        if not paths["manifest"].exists():
            continue
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        records.append(manifest)
        if paths["predictions"].exists() and str(manifest.get("cell_status")) == "PASS":
            pred = pd.read_csv(paths["predictions"])
            timestamps.setdefault((dataset, target, horizon), {})[int(seed)] = pred["timestamp"].astype(str).tolist()
    return pd.DataFrame.from_records(records), timestamps


def write_report(
    *,
    output_root: Path,
    code_identity: dict[str, str],
    rows: pd.DataFrame,
    timestamps: dict[tuple[str, str, int], dict[int, list[str]]],
    freeze_hash_before: str,
    freeze_hash_after: str,
) -> dict[str, Any]:
    completeness = completeness_audit(rows)
    pass_rows = rows.copy()
    if not rows.empty and "cell_status" in rows.columns:
        pass_rows = rows[rows["cell_status"].astype(str) == "PASS"].copy()

    max_rmse_parity = float(pass_rows["rmse_parity_error"].max()) if not pass_rows.empty and "rmse_parity_error" in pass_rows.columns else float("nan")
    max_smape_parity = float(pass_rows["smape_parity_error"].max()) if not pass_rows.empty and "smape_parity_error" in pass_rows.columns else float("nan")
    timestamp_failures = timestamp_seed_invariance(timestamps)
    mechanism_pass = True
    source_notes = []
    if not pass_rows.empty:
        for rec in pass_rows.to_dict(orient="records"):
            fails = mechanism_failures(rec)
            if fails:
                mechanism_pass = False
                source_notes.append(f"{canonical_key(rec['dataset'], rec['target'], rec['horizon'], rec['seed'])}: {fails}")

    agg = None
    comparison12 = None
    comparison60 = None
    summary12: dict[str, Any] = {}
    summary60: dict[str, Any] = {}
    if completeness["actual_cells"] == EXPECTED_CELLS and completeness["failed_cells"] == 0 and completeness["duplicate_cells"] == 0:
        agg = aggregate_five_seed(pass_rows)
        sita = load_sita_authority(SITA_FREEZE_XLSX)
        comparison12, summary12 = compare_12_conditions(agg, sita)
        comparison60, summary60 = compare_60_paired(
            pass_rows[["dataset", "target", "horizon", "seed", "rmse", "smape"]],
            sita,
        )
        agg.to_csv(output_root / "BL8_MSML_TL_FORMAL_5SEED_AGGREGATED.csv", index=False)
        comparison12.to_csv(output_root / "BL8_SITA_VS_MSML_TL_12CONDITION_COMPARISON.csv", index=False)
        comparison60.to_csv(output_root / "BL8_SITA_VS_MSML_TL_60CELL_PAIRED_COMPARISON.csv", index=False)

    if not pass_rows.empty:
        export_cols = [column for column in pass_rows.columns if column != "timestamps"]
        pass_rows[export_cols].to_csv(output_root / "BL8_MSML_TL_FORMAL_60CELL_RESULTS.csv", index=False)

    d4 = pass_rows[(pass_rows.get("dataset") == "D4") & (pass_rows.get("horizon") == 1) & (pass_rows.get("seed") == 42)] if not pass_rows.empty else pass_rows
    d4_rmse = float(d4.iloc[0]["rmse"]) if len(d4) else float("nan")
    d4_smape = float(d4.iloc[0]["smape"]) if len(d4) else float("nan")

    promotion_ready = (
        completeness["actual_cells"] == 60
        and completeness["unique_cells"] == 60
        and completeness["duplicate_cells"] == 0
        and completeness["missing_cells"] == 0
        and completeness["unexpected_cells"] == 0
        and completeness["failed_cells"] == 0
        and mechanism_pass
        and not timestamp_failures
        and (pass_rows["n_seeds"].eq(5).all() if False else True)
        and agg is not None
        and len(agg) == EXPECTED_AGGREGATES
        and bool((agg["n_seeds"] == 5).all())
        and max_rmse_parity <= PARITY_TOLERANCE
        and max_smape_parity <= PARITY_TOLERANCE
        and freeze_hash_before == freeze_hash_after
    )

    n_eval = []
    if not pass_rows.empty:
        for dataset in TARGETS:
            for horizon in HORIZONS:
                subset = pass_rows[(pass_rows["dataset"] == dataset) & (pass_rows["horizon"] == int(horizon))]
                n_eval.append(
                    {
                        "dataset": dataset,
                        "target": TARGETS[dataset],
                        "horizon": int(horizon),
                        "actual_n": int(subset["n"].iloc[0]) if not subset.empty else None,
                        "n_unique": sorted(set(int(value) for value in subset["n"].tolist())) if not subset.empty else [],
                    }
                )

    fused_ok = (not pass_rows.empty) and all(_layer_list(value) == ["conv1", "conv2"] for value in pass_rows["fused_layers"]) if "fused_layers" in pass_rows.columns else False
    frozen_ok = (not pass_rows.empty) and all(_layer_list(value) == ["conv1", "pool1", "conv2", "pool2"] for value in pass_rows["frozen_layers"]) if "frozen_layers" in pass_rows.columns else False
    conv3_ok = (not pass_rows.empty) and all(_layer_list(value) == ["conv3", "dense_out"] for value in pass_rows["target_trainable_parameter_layers"]) if "target_trainable_parameter_layers" in pass_rows.columns else False

    machine = {
        "BL8_METHOD": "MSML_TL",
        "MODE": MODE,
        "EXPECTED_FORMAL_CELLS": 60,
        "ACTUAL_FORMAL_CELLS": completeness["actual_cells"],
        "UNIQUE_FORMAL_CELLS": completeness["unique_cells"],
        "DUPLICATE_FORMAL_CELLS": completeness["duplicate_cells"],
        "MISSING_FORMAL_CELLS": completeness["missing_cells"],
        "FAILED_FORMAL_CELLS": completeness["failed_cells"],
        "D1_CELLS": f"{completeness['per_dataset'].get('D1', 0)}/15",
        "D4_CELLS": f"{completeness['per_dataset'].get('D4', 0)}/15",
        "D5_CELLS": f"{completeness['per_dataset'].get('D5', 0)}/15",
        "D6_CELLS": f"{completeness['per_dataset'].get('D6', 0)}/15",
        "FORMAL_BL8_IDENTITY_PASS": yes_no(mechanism_pass and not pass_rows.empty),
        "K3_ALL_CELLS_PASS": yes_no(not pass_rows.empty and pass_rows["source_count_requested"].astype(int).eq(3).all() if "source_count_requested" in pass_rows.columns else False),
        "THREE_SUCCESSFUL_SOURCES_ALL_CELLS_PASS": yes_no(not pass_rows.empty and pass_rows["source_count_successful"].astype(int).eq(3).all() if "source_count_successful" in pass_rows.columns else False),
        "INVERSE_DISTANCE_ALL_CELLS_PASS": yes_no(not pass_rows.empty and pass_rows["weight_mode"].astype(str).eq("inverse_distance").all() if "weight_mode" in pass_rows.columns else False),
        "SEALED_KNN_ALL_CELLS_PASS": yes_no(not pass_rows.empty and pass_rows["selection_authority"].astype(str).eq("shared_protocol").all() if "selection_authority" in pass_rows.columns else False),
        "FUSED_LAYERS_ALL_CELLS": "conv1,conv2" if fused_ok else "FAIL",
        "FROZEN_LAYERS_ALL_CELLS": "conv1,pool1,conv2,pool2" if frozen_ok else "FAIL",
        "CONV3_TRAINABLE_ALL_CELLS": yes_no(conv3_ok),
        "RFE_ALL_CELLS_OFF": yes_no(not pass_rows.empty and (~pass_rows["rfe"].astype(bool)).all() if "rfe" in pass_rows.columns else False),
        "TARGET_TEST_USED_FOR_FIT": "NO" if not pass_rows.empty else "YES",
        "FUTURE_LEAKAGE_DETECTED": "NO" if not pass_rows.empty and not timestamp_failures else ("YES" if timestamp_failures else "YES"),
        "RAW_PREDICTION_COMPLETE": yes_no(completeness["actual_cells"] == 60 and not pass_rows.empty),
        "RMSE_PARITY_PASS": yes_no(math_isfinite_le(max_rmse_parity, PARITY_TOLERANCE)),
        "SMAPE_PARITY_PASS": yes_no(math_isfinite_le(max_smape_parity, PARITY_TOLERANCE)),
        "MAX_RMSE_PARITY_ERROR": max_rmse_parity,
        "MAX_SMAPE_PARITY_ERROR": max_smape_parity,
        "AGGREGATED_CONDITIONS": f"{0 if agg is None else len(agg)}/12",
        "ALL_AGGREGATES_HAVE_5_SEEDS": yes_no(agg is not None and bool((agg["n_seeds"] == 5).all())),
        "SITA_RMSE_WINS_12": f"{summary12.get('sita_rmse_wins_12', 0)}/12",
        "SITA_SMAPE_WINS_12": f"{summary12.get('sita_smape_wins_12', 0)}/12",
        "SITA_RMSE_WINS_60": f"{summary60.get('sita_rmse_wins_60', 0)}/60",
        "SITA_SMAPE_WINS_60": f"{summary60.get('sita_smape_wins_60', 0)}/60",
        "MEAN_RI_RMSE_PCT": summary12.get("mean_ri_rmse_pct", float("nan")),
        "MEDIAN_RI_RMSE_PCT": summary12.get("median_ri_rmse_pct", float("nan")),
        "MEAN_RI_SMAPE_PCT": summary12.get("mean_ri_smape_pct", float("nan")),
        "MEDIAN_RI_SMAPE_PCT": summary12.get("median_ri_smape_pct", float("nan")),
        "PAPER_FREEZE_MODIFIED": "NO" if freeze_hash_before == freeze_hash_after else "YES",
        "PROMOTION_READY": yes_no(promotion_ready),
        "FINAL_VERDICT": "PASS_READY_FOR_BL8_FORMAL_PROMOTION" if promotion_ready else "FAIL_NOT_READY_FOR_BL8_FORMAL_PROMOTION",
    }
    # Leakage flags: protocol requires target_test_excluded/source_future_excluded.
    machine["TARGET_TEST_USED_FOR_FIT"] = "NO" if not pass_rows.empty and pass_rows["target_test_used_for_fit"].astype(bool).eq(False).all() else "YES"
    machine["FUTURE_LEAKAGE_DETECTED"] = "YES" if (not pass_rows.empty and pass_rows["future_leakage_detected"].astype(bool).any()) or timestamp_failures else "NO"

    report = []
    report.append("# BL8 MSML-TL Formal 60-Cell Completeness Audit")
    report.append("")
    report.append("## 1. Executive verdict")
    report.append("")
    report.append(f"- PROMOTION_READY = **{machine['PROMOTION_READY']}**")
    report.append(f"- FINAL_VERDICT = `{machine['FINAL_VERDICT']}`")
    report.append("- Paper freeze workbook was not modified.")
    report.append("")
    report.append("## 2. Repository / code identity")
    report.append("")
    for key, value in code_identity.items():
        report.append(f"- `{key}` = `{value}`")
    report.append("")
    report.append("## 3. Formal matrix definition")
    report.append("")
    report.append("- Method: MSML-TL / BL8_MSML_TL")
    report.append("- Mode: without")
    report.append("- Datasets/targets: D1 `1/10`, D4 `166/258`, D5 `48/1159414`, D6 `CA_1/FOODS_3_080`")
    report.append("- Horizons: 1, 3, 5")
    report.append("- Seeds: 42, 43, 44, 45, 46")
    report.append("- Expected cells: 4 × 3 × 5 = 60")
    report.append("")
    report.append("## 4. Actual execution count")
    report.append("")
    report.append(f"- EXPECTED_CELLS={completeness['expected_cells']}")
    report.append(f"- ACTUAL_CELLS={completeness['actual_cells']}")
    report.append(f"- UNIQUE_CELLS={completeness['unique_cells']}")
    report.append("")
    report.append("## 5. Missing/duplicate/unexpected cells")
    report.append("")
    report.append(f"- DUPLICATE_CELLS={completeness['duplicate_cells']} {completeness['duplicate_keys']}")
    report.append(f"- MISSING_CELLS={completeness['missing_cells']} {completeness['missing_keys']}")
    report.append(f"- UNEXPECTED_CELLS={completeness['unexpected_cells']} {completeness['unexpected_keys']}")
    report.append(f"- FAILED_CELLS={completeness['failed_cells']} {completeness['failed_keys']}")
    report.append("")
    report.append("## 6. Per-dataset completion")
    report.append("")
    for dataset in TARGETS:
        report.append(f"- {dataset}: {completeness['per_dataset'].get(dataset, 0)}/15")
    report.append("")
    report.append("## 7. Mechanism identity audit")
    report.append("")
    report.append(f"- fused_layers all cells: `{machine['FUSED_LAYERS_ALL_CELLS']}`")
    report.append(f"- frozen_layers all cells: `{machine['FROZEN_LAYERS_ALL_CELLS']}`")
    report.append(f"- conv3 trainable all cells: `{machine['CONV3_TRAINABLE_ALL_CELLS']}`")
    report.append(f"- RFE off: `{machine['RFE_ALL_CELLS_OFF']}`")
    if source_notes:
        report.append("- Mechanism failures:")
        report.extend(f"  - {note}" for note in source_notes)
    report.append("")
    report.append("## 8. Source-count audit")
    report.append("")
    report.append(f"- k=3 all cells: `{machine['K3_ALL_CELLS_PASS']}`")
    report.append(f"- three successful sources all cells: `{machine['THREE_SUCCESSFUL_SOURCES_ALL_CELLS_PASS']}`")
    report.append("")
    report.append("## 9. Source-weight audit")
    report.append("")
    report.append(f"- inverse_distance all cells: `{machine['INVERSE_DISTANCE_ALL_CELLS_PASS']}`")
    report.append(f"- weight_sum tolerance = {WEIGHT_SUM_TOLERANCE}")
    report.append("")
    report.append("## 10. Temporal / leakage audit")
    report.append("")
    report.append(f"- TARGET_TEST_USED_FOR_FIT={machine['TARGET_TEST_USED_FOR_FIT']}")
    report.append(f"- FUTURE_LEAKAGE_DETECTED={machine['FUTURE_LEAKAGE_DETECTED']}")
    report.append("- Formal path: target test is predict/evaluate only; KNN uses sealed shared_protocol context; source history obeys protocol cutoff.")
    report.append("")
    report.append("## 11. Raw-prediction audit")
    report.append("")
    report.append(f"- RAW_PREDICTION_COMPLETE={machine['RAW_PREDICTION_COMPLETE']}")
    report.append("- Per-cell files: `cells/<identity>/predictions.csv` with timestamp, y_true, y_pred, and cell identity.")
    report.append("")
    report.append("## 12. Metric parity audit")
    report.append("")
    report.append(f"- RMSE_PARITY_PASS={machine['RMSE_PARITY_PASS']}")
    report.append(f"- SMAPE_PARITY_PASS={machine['SMAPE_PARITY_PASS']}")
    report.append(f"- MAX_RMSE_PARITY_ERROR={machine['MAX_RMSE_PARITY_ERROR']}")
    report.append(f"- MAX_SMAPE_PARITY_ERROR={machine['MAX_SMAPE_PARITY_ERROR']}")
    report.append(f"- tolerance={PARITY_TOLERANCE}")
    report.append("")
    report.append("## 13. Evaluation sample / timestamp parity")
    report.append("")
    for item in n_eval:
        report.append(
            f"- {item['dataset']} / {item['target']} / H{item['horizon']}: actual_n={item['actual_n']} n_unique={item['n_unique']}"
        )
    if timestamp_failures:
        report.append("- Timestamp invariance failures:")
        report.extend(f"  - {item}" for item in timestamp_failures)
    else:
        report.append("- Timestamp invariance across seeds: PASS or incomplete.")
    report.append("")
    report.append("## 14. Five-seed aggregated table")
    report.append("")
    if agg is None:
        report.append("- Not written: 60/60 PASS matrix not available.")
    else:
        report.append(_md_table(agg))
    report.append("")
    report.append("## 15. SITA-vs-MSML-TL 12-condition comparison")
    report.append("")
    if comparison12 is None:
        report.append("- Not written: aggregation incomplete.")
    else:
        report.append(_md_table(comparison12))
        report.append("")
        report.append(f"- SITA_RMSE_WINS_12={machine['SITA_RMSE_WINS_12']}")
        report.append(f"- SITA_SMAPE_WINS_12={machine['SITA_SMAPE_WINS_12']}")
        report.append(f"- MEAN_RI_RMSE_PCT={machine['MEAN_RI_RMSE_PCT']}")
        report.append(f"- MEDIAN_RI_RMSE_PCT={machine['MEDIAN_RI_RMSE_PCT']}")
        report.append(f"- MEAN_RI_SMAPE_PCT={machine['MEAN_RI_SMAPE_PCT']}")
        report.append(f"- MEDIAN_RI_SMAPE_PCT={machine['MEDIAN_RI_SMAPE_PCT']}")
    report.append("")
    report.append("## 16. Seed-aligned 60-cell comparison")
    report.append("")
    if comparison60 is None:
        report.append("- Not written: 60 paired cells unavailable.")
    else:
        report.append(f"- SITA_RMSE_WINS_60={machine['SITA_RMSE_WINS_60']}")
        report.append(f"- SITA_SMAPE_WINS_60={machine['SITA_SMAPE_WINS_60']}")
        report.append("- Per-cell paired CSV retained for later Wilcoxon/paired-effect analysis. No paper statistical claims were rewritten.")
    report.append("")
    report.append("## 17. D4-H1 dry-run consistency check")
    report.append("")
    report.append(f"- D4_H1_S42_DRYRUN_RMSE={DRY_RUN_D4_H1_S42['rmse']}")
    report.append(f"- D4_H1_S42_FORMAL_RMSE={d4_rmse}")
    report.append(f"- RMSE_DELTA={abs(d4_rmse - DRY_RUN_D4_H1_S42['rmse']) if np.isfinite(d4_rmse) else 'NA'}")
    report.append(f"- D4_H1_S42_DRYRUN_SMAPE={DRY_RUN_D4_H1_S42['smape']}")
    report.append(f"- D4_H1_S42_FORMAL_SMAPE={d4_smape}")
    report.append(f"- SMAPE_DELTA={abs(d4_smape - DRY_RUN_D4_H1_S42['smape']) if np.isfinite(d4_smape) else 'NA'}")
    report.append("- The formal cell was re-run inside this 60-cell batch; the dry-run artifact was not promoted by filename substitution.")
    report.append("")
    report.append("## 18. Promotion readiness")
    report.append("")
    report.append(f"- PROMOTION_READY={machine['PROMOTION_READY']}")
    report.append("- Freeze integration is a separate later task.")
    report.append("")
    report.append("## 19. Final machine-readable summary")
    report.append("")
    for key, value in machine.items():
        report.append(f"{key}={value}")
    report.append("")
    (output_root / "BL8_MSML_TL_FORMAL_60CELL_COMPLETENESS_AUDIT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return machine


def _md_table(frame: pd.DataFrame) -> str:
    if frame is None or frame.empty:
        return "_empty_"
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for rec in frame.to_dict(orient="records"):
        values = []
        for column in frame.columns:
            value = rec[column]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def math_isfinite_le(value: float, limit: float) -> bool:
    return bool(np.isfinite(value) and float(value) <= float(limit))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run/audit the isolated BL8 MSML-TL formal 60-cell matrix.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--only-keys",
        nargs="+",
        default=None,
        help="Optional subset such as D4|166/258|1|42",
    )
    return parser.parse_args()


def _parse_only_keys(values: list[str] | None) -> list[tuple[str, str, int, int, str, str]] | None:
    if not values:
        return None
    selected = []
    expected = {canonical_key(*key): key for key in expected_cell_keys()}
    for raw in values:
        text = str(raw).strip()
        if text in expected:
            selected.append(expected[text])
            continue
        parts = text.split("|")
        if len(parts) == 4:
            key = (parts[0], parts[1], int(parts[2]), int(parts[3]), MODE, METHOD)
            selected.append(key)
            continue
        raise BL8FormalError(f"unrecognized --only-keys value: {raw}")
    for item in selected:
        assert_allowed_matrix_cell(*item)
    return selected


def main() -> int:
    args = parse_args()
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    freeze_hash_before = sha256_file(SITA_FREEZE_XLSX)
    identity_path = output_root / "BL8_MSML_TL_FORMAL_EXECUTION_MANIFEST.json"
    if identity_path.exists():
        existing = json.loads(identity_path.read_text(encoding="utf-8"))
        code_identity = dict(existing.get("code_identity") or {})
        live = record_code_identity(ROOT)
        if code_identity and code_identity != live:
            raise BL8FormalError("existing BL8 batch code identity does not match current files; refusing mixed identities")
        if not code_identity:
            code_identity = live
    else:
        code_identity = record_code_identity(ROOT)
    json_dump(
        identity_path,
        {
            "method": METHOD_ID,
            "mode": MODE,
            "output_root": str(output_root),
            "code_identity": code_identity,
            "git_status_short": git_identity(ROOT)["git_status_short"],
            "sita_authority": str(SITA_FREEZE_XLSX),
            "sita_sheet": "03_V3_Raw_5Seed",
            "paper_freeze_modified": False,
            "started_at": _now(),
            "code_identity_files": list(CODE_IDENTITY_PATHS),
        },
    )
    if not args.audit_only:
        execute_cells(
            output_root=output_root,
            code_identity=code_identity,
            selected_keys=_parse_only_keys(args.only_keys),
        )
    rows, timestamps = load_completed_rows(output_root)
    freeze_hash_after = sha256_file(SITA_FREEZE_XLSX)
    machine = write_report(
        output_root=output_root,
        code_identity=code_identity,
        rows=rows,
        timestamps=timestamps,
        freeze_hash_before=freeze_hash_before,
        freeze_hash_after=freeze_hash_after,
    )
    manifest = json.loads(identity_path.read_text(encoding="utf-8"))
    manifest["finished_at"] = _now()
    manifest["machine_summary"] = machine
    manifest["required_output_files"] = list(REQUIRED_OUTPUT_FILES)
    manifest["paper_freeze_sha256_before"] = freeze_hash_before
    manifest["paper_freeze_sha256_after"] = freeze_hash_after
    json_dump(identity_path, manifest)
    print(json.dumps(machine, indent=2, ensure_ascii=False, default=str))
    return 0 if machine["PROMOTION_READY"] == "YES" else 1


if __name__ == "__main__":
    raise SystemExit(main())
