"""D1-D6 FUSED_TARGET_ONLY primary-target × learning-rate screening runner.

Development / screening track. Does not modify FORMAL_METHODS, architecture,
MSML, No-TL, or baselines, and does not launch robustness experiments.

Usage:
    python scripts/run_ms_katl_primary_target_lr_screening.py --preflight
    python scripts/run_ms_katl_primary_target_lr_screening.py
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.constants import SOLIDIFIED_KNN_ROOT, SOURCE_HISTORY_DAYS

import tf_compat  # noqa: F401

import numpy as np
import pandas as pd

from paper_reproduction_protocol import load_paper_protocol, resolve_strict_paper_mode
from scripts.run_full_paper_experiments import (
    _bind_protocol_cell_identity as _bind_d13_cell_identity,
    _cell_lifecycle_identity,
    _cell_owned_configured_source,
    _load_config,
    _load_solidified_base_data,
    _prepare_cell_source_dependencies,
    _project_modeling_frame,
    _require_cell_context_identity,
)
from src.protocols.candidate_pool import prepare_daily_sequence_pool
from src.protocols.experiment_protocol import (
    get_experiment_protocol,
    normalize_scenario,
    normalize_source_key,
)
from src.protocols.formal_input_paths import resolve_formal_dataset_paths
from src.protocols.raw_preprocessing import TargetK3RawSourceContext
from src.protocols.reproducibility import set_protocol_seed
from src.protocols.runner_adapter import configure_protocol_frames, source_key_mask
from src.protocols.transformation_identity import validate_runtime_feature_contract
from src.protocols.transformation_reuse import TargetTransformationReuseContext
from src.source_selection.source_selector import TargetK3SelectionContext
from src.transfer_methods.ms_katl_lite import prepare_ms_katl_stage_a
from src.transfer_methods.ms_katl_mechanism_ablation import (
    prepare_shared_target_bundle,
    resolve_ablation_arm,
    run_ablation_arm,
)
from src.transfer_methods.ms_katl_primary_target_lr_screening import (
    LR_GRID,
    METHOD_ARM,
    METHOD_DISPLAY,
    SCREENING_DATASETS,
    SCREENING_EPOCHS,
    SCREENING_HORIZON,
    SCREENING_LABEL,
    SCREENING_MODE,
    SCREENING_SEED,
    STAGE_A_LEARNING_RATE,
    TRACK,
    architecture_digest,
    assert_lr_arms_share_stage_a,
    assert_screening_does_not_change_formal_methods,
    attach_reference_deltas,
    build_frozen_manifest,
    canonical_candidate_targets,
    compute_screening_metrics,
    default_baseline_csv,
    default_formal_csv,
    format_dataset_terminal,
    format_global_summary,
    format_lr,
    fused_target_only_architecture,
    load_baseline_references,
    load_formal_references,
    parse_target_identity,
    plan_screening,
    print_screening_preflight,
    sales_minmax,
    select_dataset_primary_target,
    select_target_best_lr,
    write_frozen_config_manifest,
    write_screening_outputs,
)
from src.utils.d4_d6_runtime import (
    apply_runtime_source_domain_policy,
    load_default_metric_protocol,
    validate_runtime_target_domain,
)
from src.utils.d5_calendar_reconstruction import load_d5_authorities
from src.utils.entity_experiment import (
    _bind_protocol_cell_identity,
    _build_model_dataframe,
    _require_bound_protocol_cell_identity,
    _resolve_model_feature_cols,
    _sanitize_source_model_dataframe,
    _scenario_name,
)
from src.utils.finite_diagnostics import validate_feature_frame_finite
from src.utils.knn_feature_loader import resolve_knn_feature_columns
from src.utils.parquet_data_loader import (
    expected_target_dates_from_windows,
    load_knn_results,
    load_parquet_source_target,
    load_parquet_source_target_with_diagnostics,
    read_dataset_windows,
)
from src.utils.run_utils import create_run_dir, reserve_new_output_dir

PROJECT_ROOT = Path(__file__).resolve().parent.parent
D13_GROUP_COLS = {
    "D1": ("store_id", "item_id"),
    "D2": ("brand_id", "item_id"),
    "D3": ("store_id",),
}
D46_GROUPING_COL = {4: None, 5: "family", 6: "dept_id"}
D46_POOL_METADATA = {4: (), 5: ("family",), 6: ("dept_id",)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="D1-D6 FUSED_TARGET_ONLY primary target × LR screening."
    )
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=SCREENING_EPOCHS)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated D1-D6. Default all. Production screening uses all.",
    )
    return parser.parse_args()


def _requested_datasets(spec: object) -> tuple[str, ...]:
    if spec is None or str(spec).strip().lower() in {"", "all"}:
        return SCREENING_DATASETS
    parts = [part.strip().upper() for part in str(spec).replace(";", ",").split(",") if part.strip()]
    resolved = tuple(f"D{int(part.replace('D', ''))}" for part in parts)
    extra = [item for item in resolved if item not in SCREENING_DATASETS]
    if extra:
        raise ValueError(f"unknown datasets: {extra}")
    return resolved


def _jsonable(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _load_d13_pool(dataset: str, cfg: dict, protocol: dict) -> dict:
    dataset_name = f"Dataset{_dataset_number(dataset)}"
    base = _load_solidified_base_data(
        dataset_name=dataset_name,
        cfg=cfg,
        strict_paper_mode=True,
        strict_paper_split=True,
    )
    cell_source = _prepare_cell_source_dependencies(
        dataset_name=dataset_name,
        information_sharing_scenario="without_information_sharing",
        cfg=cfg,
        protocol=protocol,
        strict_paper_mode=True,
        base_data=base,
    )
    return {
        "dataset_name": dataset_name,
        "base": base,
        "cell_source": cell_source,
        "cfg": cfg,
        "protocol": protocol,
    }


def _prepare_d13_target_cell(dataset: str, pool: dict, parsed: dict) -> dict:
    dataset_name = pool["dataset_name"]
    cfg = pool["cfg"]
    cell_source = pool["cell_source"]
    target_df = pool["base"]["target_df"].copy()
    expected_lifecycle_identity = _cell_lifecycle_identity(
        dataset_name, "without_information_sharing", cfg
    )
    _require_cell_context_identity(cell_source, expected_lifecycle_identity)
    source_df = cell_source.source_frame.copy()
    source_df.attrs = deepcopy(cell_source.source_frame.attrs)
    modeling_feature_cols = list(cell_source.modeling_feature_cols)
    target_df.attrs["information_sharing_scenario"] = source_df.attrs.get(
        "information_sharing_scenario", ""
    )
    target_df.attrs["signature_static_feature_cols"] = list(
        source_df.attrs.get("signature_static_feature_cols", [])
    )
    target_df = _project_modeling_frame(target_df, tuple(source_df.columns))
    group_cols = D13_GROUP_COLS[dataset]
    protocol_key = parsed["target_identity"].split("/")
    mask = source_key_mask(target_df, group_cols, protocol_key)
    filtered = target_df.loc[mask].copy()
    if filtered.empty:
        raise ValueError(
            f"{dataset} target {parsed['target_identity']} not found in target_df "
            f"using protocol key columns {group_cols}"
        )
    filtered.attrs = deepcopy(target_df.attrs)
    target_df = filtered
    _bind_d13_cell_identity(source_df, expected_lifecycle_identity)
    _bind_d13_cell_identity(target_df, expected_lifecycle_identity)
    source_df, target_df = configure_protocol_frames(
        source_df,
        target_df,
        dataset_id=dataset_name,
        scenario="without_information_sharing",
        group_cols=group_cols,
        observed_start=None if dataset in {"D1", "D2"} else "2015-01-03",
        prepared_pool=cell_source.prepared_pool,
        retain_source_frame=True,
        enforce_formal_target=True,
    )
    source_df = _cell_owned_configured_source(
        cell_source,
        source_df,
        expected_lifecycle_identity,
    )
    return {
        "source_model_df": source_df,
        "target_model_df": target_df,
        "model_feature_cols": modeling_feature_cols,
        "group_cols": group_cols,
        "k3_selection_context": cell_source.k3_selection_context,
        "k3_raw_source_context": cell_source.k3_raw_source_context,
        "transformation_reuse_context": cell_source.transformation_reuse_context,
        "protocol_target_key": target_df.attrs["protocol_target_key"],
    }


def _dataset_number(dataset: str) -> int:
    return int(dataset[1:])


def _d46_config(dataset_id: int, epochs: int) -> dict:
    return {
        "use_parquet": True,
        "dataset_id": dataset_id,
        "dataset_name": f"Dataset{dataset_id}",
        "knn_json_dir": str(SOLIDIFIED_KNN_ROOT / f"Dataset{dataset_id}"),
        "info_sharing": SCREENING_MODE,
        "source_history_days": SOURCE_HISTORY_DAYS,
        "entity_col": "entity_id",
        "horizon": SCREENING_HORIZON,
        "window_size": 10,
        "learning_rate": STAGE_A_LEARNING_RATE,
        "source_epochs": int(epochs),
        "target_epochs": int(epochs),
        "batch_size": 16,
        "random_state": SCREENING_SEED,
        "source_count": 3,
        "smoke": False,
        "extended_methods_enabled": True,
        "ms_katl_gamma": 0.5,
        "ms_katl_tau_k": 1.0,
        "metric_protocol": load_default_metric_protocol(PROJECT_ROOT),
        "raw_dir": "数据集/原始数据/Dataset 5Favorita" if dataset_id == 5 else None,
    }


def _load_d46_pool(dataset: str, epochs: int) -> dict:
    dataset_id = _dataset_number(dataset)
    config = _d46_config(dataset_id, epochs)
    formal_paths = resolve_formal_dataset_paths(dataset_id, repository_root=PROJECT_ROOT)
    knn_data = load_knn_results(config["knn_json_dir"], config["info_sharing"])
    group_cols = knn_data.get("group_cols")
    if not isinstance(group_cols, list) or len(group_cols) != 2:
        raise ValueError(f"{dataset} KNN payload requires two group_cols, got: {group_cols}")
    config["group_cols"] = list(group_cols)
    windows = read_dataset_windows(
        config["dataset_id"],
        config["knn_json_dir"],
        info_sharing=config["info_sharing"],
        knn_payload=knn_data,
    )
    if dataset_id == 5:
        authorities = load_d5_authorities(PROJECT_ROOT / config["raw_dir"], use_holidays=True)
        runtime_inputs = load_parquet_source_target_with_diagnostics(
            dataset_id=dataset_id,
            source_path=formal_paths.source_path,
            target_path=formal_paths.target_path,
            windows=windows,
            source_history_days=config["source_history_days"],
            expected_dates=expected_target_dates_from_windows(windows),
            d5_authorities=authorities,
        )
        source_df = runtime_inputs.source_df
        target_df = runtime_inputs.target_df
    else:
        source_df, target_df = load_parquet_source_target(
            dataset_id=dataset_id,
            source_path=formal_paths.source_path,
            target_path=formal_paths.target_path,
            windows=windows,
            source_history_days=config["source_history_days"],
        )
    feature_info = resolve_knn_feature_columns(
        dataset_id=dataset_id,
        information_sharing=config["info_sharing"],
        knn_root=SOLIDIFIED_KNN_ROOT,
        source_df=source_df,
        target_df=target_df,
        knn_payload=knn_data,
    )
    feature_cols = list(feature_info["selected_features"])
    validate_runtime_feature_contract(
        dataset,
        model_feature_cols=feature_cols,
        knn_feature_cols=get_experiment_protocol(dataset_id).knn_feature_columns,
    )
    source_df = apply_runtime_source_domain_policy(source_df, knn_data, config)
    return {
        "config": config,
        "knn_data": knn_data,
        "source_df": source_df,
        "target_df": target_df,
        "feature_cols": feature_cols,
        "group_cols": tuple(str(col) for col in config["group_cols"]),
    }


def _prepare_d46_target_cell(dataset: str, pool: dict, parsed: dict) -> dict:
    config = pool["config"]
    dataset_id = int(config["dataset_id"])
    entity_key = parsed["entity_key"]
    knn_data = pool["knn_data"]
    source_df = pool["source_df"]
    target_df = pool["target_df"]
    feature_cols = pool["feature_cols"]
    group_cols = pool["group_cols"]
    if entity_key not in knn_data["results"]:
        raise ValueError(f"target {entity_key} missing from {dataset} KNN results")
    validate_runtime_target_domain(target_df, [entity_key], knn_data, config)
    target_entity_df = target_df[target_df[config["entity_col"]].astype(str) == entity_key].copy()
    if target_entity_df.empty:
        raise ValueError(f"target {entity_key} not found in {dataset} target_df")
    scenario = _scenario_name(config)
    observed_start = target_entity_df.attrs.get(
        "knn_observed_start",
        target_entity_df.attrs.get(
            "target_observed_start",
            pd.to_datetime(target_entity_df["date"], errors="raise").min(),
        ),
    )
    pool_kwargs = {
        "group_cols": group_cols,
        "observed_start": observed_start,
        "feature_cols": get_experiment_protocol(dataset_id).knn_feature_columns,
        "capture_source_observed_frame_digest": True,
    }
    metadata_cols = D46_POOL_METADATA[dataset_id]
    if metadata_cols:
        pool_kwargs["metadata_cols"] = metadata_cols
    prepared_pool = prepare_daily_sequence_pool(source_df, **pool_kwargs)
    source_bound, target_bound = configure_protocol_frames(
        source_df,
        target_entity_df,
        dataset_id=dataset_id,
        scenario=scenario,
        group_cols=group_cols,
        grouping_col=D46_GROUPING_COL[dataset_id],
        observed_start=observed_start,
        prepared_pool=prepared_pool,
        retain_source_frame=True,
        enforce_formal_target=True,
    )
    cell_lifecycle_identity = (
        get_experiment_protocol(dataset_id).dataset_id,
        normalize_scenario(scenario),
        int(config["horizon"]),
        int(config["random_state"]),
        normalize_source_key(target_bound.attrs["protocol_target_key"]),
    )
    _bind_protocol_cell_identity(source_bound, cell_lifecycle_identity)
    _bind_protocol_cell_identity(target_bound, cell_lifecycle_identity)
    model_feature_cols = _resolve_model_feature_cols(source_bound, target_bound, feature_cols, config)
    source_model_df = _sanitize_source_model_dataframe(
        _build_model_dataframe(
            source_bound,
            model_feature_cols,
            source_selection_group_cols=group_cols,
            required_passthrough_cols=get_experiment_protocol(dataset_id).knn_feature_columns,
        ),
        model_feature_cols,
    )
    target_model_df = _build_model_dataframe(
        target_bound,
        model_feature_cols,
        source_selection_group_cols=group_cols,
        required_passthrough_cols=get_experiment_protocol(dataset_id).knn_feature_columns,
    )
    _require_bound_protocol_cell_identity(source_model_df, cell_lifecycle_identity)
    _require_bound_protocol_cell_identity(target_model_df, cell_lifecycle_identity)
    validate_feature_frame_finite(
        source_model_df,
        model_feature_cols,
        context="screening_source",
        dataset_id=dataset_id,
        role="source",
        entity_id="source_pool",
        stage="ms_katl_primary_target_lr_screening",
    )
    validate_feature_frame_finite(
        target_model_df,
        model_feature_cols,
        context="screening_target",
        dataset_id=dataset_id,
        role="target",
        entity_id=entity_key,
        stage="ms_katl_primary_target_lr_screening",
    )
    source_model_df.attrs["information_sharing_scenario"] = cell_lifecycle_identity[1]
    target_model_df.attrs["information_sharing_scenario"] = cell_lifecycle_identity[1]
    k3_selection_context = TargetK3SelectionContext(lifecycle_identity=cell_lifecycle_identity)
    k3_raw_source_context = TargetK3RawSourceContext(
        lifecycle_identity=k3_selection_context.lifecycle_identity
    )
    transformation_reuse_context = TargetTransformationReuseContext(
        lifecycle_identity=k3_selection_context.lifecycle_identity
    )
    return {
        "source_model_df": source_model_df,
        "target_model_df": target_model_df,
        "model_feature_cols": model_feature_cols,
        "group_cols": group_cols,
        "k3_selection_context": k3_selection_context,
        "k3_raw_source_context": k3_raw_source_context,
        "transformation_reuse_context": transformation_reuse_context,
        "protocol_target_key": target_bound.attrs["protocol_target_key"],
    }


def _failed_row(dataset: str, parsed: dict, lr: float, reason: str) -> dict:
    return {
        "dataset": dataset,
        "target_identity": parsed["target_identity"],
        "mode": SCREENING_MODE,
        "horizon": SCREENING_HORIZON,
        "seed": SCREENING_SEED,
        "epochs": SCREENING_EPOCHS,
        "learning_rate": float(lr),
        "learning_rate_label": format_lr(lr),
        "method": METHOD_DISPLAY,
        "rmse_original": None,
        "smape_original": None,
        "rmse_normalized": None,
        "smape_normalized": None,
        "metric_space": "original_sales_space",
        "run_status": "FAILED",
        "failure_reason": str(reason),
        "SCREENING_STATUS": "FAILED",
    }


def _run_target_lr_arms(
    *,
    dataset: str,
    parsed: dict,
    cell: dict,
    epochs: int,
    batch_size: int,
    metric_protocol: Optional[dict],
    pred_root: Path,
) -> List[dict]:
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
    spec = resolve_ablation_arm(METHOD_ARM)
    arch = fused_target_only_architecture()
    source_identities = [tuple(item) for item in stage_a.source_identities]
    source_payload = {
        "source_identities": source_identities,
        "source_distances": list(stage_a.source_distances),
        "source_model_digests": list(stage_a.source_weight_digests),
        "source_scaler_digests": list(stage_a.source_scaler_digests),
        "fused_init_digest": bundle.fused_target_digest,
        "target_scaler_digest": bundle.target_split_digest.get("target_scaler"),
        "target_split_digest": dict(bundle.target_split_digest),
        "architecture_digest": architecture_digest(),
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
        )
        metrics = compute_screening_metrics(
            arm["y_true"],
            arm["y_pred"],
            target_scaler=bundle.target_scaler,
            feature_columns=bundle.target_feature_columns,
        )
        drifts = arm.get("conv_drifts") or {}
        row = {
            "dataset": dataset,
            "target_identity": parsed["target_identity"],
            "mode": SCREENING_MODE,
            "horizon": SCREENING_HORIZON,
            "seed": SCREENING_SEED,
            "epochs": int(epochs),
            "learning_rate": float(lr),
            "learning_rate_label": format_lr(lr),
            "method": METHOD_DISPLAY,
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
        }
        pred_dir = pred_root / dataset / parsed["prediction_dirname"] / format_lr(lr)
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
    flags = assert_lr_arms_share_stage_a(rows)
    for row in rows:
        row["LR_ARMS_FUSED_INIT_MATCH"] = flags["LR_ARMS_FUSED_INIT_MATCH"]
        row["STAGE_A_SHARED"] = "YES"
    return rows


def _csv_ready(row: dict) -> dict:
    skip = {
        "y_true_normalized",
        "y_pred_normalized",
        "y_true_original",
        "y_pred_original",
        "target_split_digest",
    }
    out = {}
    for key, value in row.items():
        if key in skip:
            continue
        if isinstance(value, (list, tuple)):
            out[key] = json.dumps(_jsonable(value), ensure_ascii=False)
        else:
            out[key] = value
    return out


def main() -> None:
    args = _parse_args()
    requested = _requested_datasets(args.datasets)
    plan = plan_screening(datasets=requested if requested == SCREENING_DATASETS else SCREENING_DATASETS)
    print_screening_preflight(plan)
    for dataset in requested:
        identities = canonical_candidate_targets(dataset)
        print(f"{dataset} candidate_target_count={len(identities)}")
        print(f"{dataset} candidate_target_identities={list(identities)}")
    if args.preflight or args.dry_run:
        return

    if requested != SCREENING_DATASETS:
        raise SystemExit("production screening requires all D1-D6 datasets")
    if int(args.epochs) != SCREENING_EPOCHS:
        print(
            f"WARNING: epochs override {args.epochs} is for lightweight checks only; "
            f"canonical screening uses {SCREENING_EPOCHS}"
        )

    assert_screening_does_not_change_formal_methods()
    fused_target_only_architecture()
    if args.output_dir is None:
        run_dir = create_run_dir(PROJECT_ROOT, label=SCREENING_LABEL)
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
                "method": METHOD_DISPLAY,
                "track": TRACK,
                "horizon": SCREENING_HORIZON,
                "seed": SCREENING_SEED,
                "epochs": int(args.epochs),
                "lr_grid": list(LR_GRID),
                "stage_a_learning_rate": STAGE_A_LEARNING_RATE,
                "datasets": list(requested),
                "inventory": {key: list(value) for key, value in plan["inventory"].items()},
                "total_targets": plan["total_targets"],
                "total_screening_runs": plan["total_screening_runs"],
                "auto_start_robustness": False,
                "architecture": fused_target_only_architecture(),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    d13_cfg = _load_config()
    d13_cfg.setdefault("single_experiment", {})["horizon"] = SCREENING_HORIZON
    d13_cfg["single_experiment"]["seed"] = SCREENING_SEED
    d13_cfg["single_experiment"]["source_epochs"] = int(args.epochs)
    d13_cfg["single_experiment"]["target_epochs"] = int(args.epochs)
    d13_protocol = load_paper_protocol(d13_cfg)
    d13_protocol["strict_paper_mode"] = True
    d13_protocol.setdefault("metric_protocol", {})["strict_paper_metrics"] = True
    resolve_strict_paper_mode(d13_cfg, explicit=True)

    metric_protocol = load_default_metric_protocol(PROJECT_ROOT)
    all_runs: List[dict] = []
    target_best_rows: List[dict] = []
    primary_rows: List[dict] = []
    dataset_pools: Dict[str, Any] = {}

    for dataset in SCREENING_DATASETS:
        print(f"\n=== LOAD {dataset} ===")
        set_protocol_seed(SCREENING_SEED, include_frameworks=True)
        if dataset in {"D1", "D2", "D3"}:
            dataset_pools[dataset] = _load_d13_pool(dataset, d13_cfg, d13_protocol)
        else:
            dataset_pools[dataset] = _load_d46_pool(dataset, int(args.epochs))

    for dataset in SCREENING_DATASETS:
        dataset_run_rows: List[dict] = []
        dataset_best_rows: List[dict] = []
        formal_csv = default_formal_csv(dataset, PROJECT_ROOT)
        baseline_csv = default_baseline_csv(dataset, PROJECT_ROOT)
        for identity in plan["inventory"][dataset]:
            parsed = parse_target_identity(dataset, identity)
            print(f"\n=== {dataset} TARGET {parsed['target_identity']} | Stage-A once, then LR_GRID ===")
            try:
                if dataset in {"D1", "D2", "D3"}:
                    cell = _prepare_d13_target_cell(dataset, dataset_pools[dataset], parsed)
                    batch_size = 16
                else:
                    cell = _prepare_d46_target_cell(dataset, dataset_pools[dataset], parsed)
                    batch_size = int(dataset_pools[dataset]["config"]["batch_size"])
                lr_rows = _run_target_lr_arms(
                    dataset=dataset,
                    parsed=parsed,
                    cell=cell,
                    epochs=int(args.epochs),
                    batch_size=batch_size,
                    metric_protocol=metric_protocol,
                    pred_root=pred_root,
                )
            except Exception as exc:
                lr_rows = [
                    _failed_row(dataset, parsed, lr, f"{type(exc).__name__}: {exc}")
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
            compared = [
                attach_reference_deltas(row, formal_refs=formal_refs, baseline_refs=baseline_refs)
                for row in lr_rows
            ]
            dataset_run_rows.extend(compared)
            all_runs.extend(compared)
            best = select_target_best_lr(compared)
            dataset_best_rows.append(best)
            target_best_rows.append(best)

        selection = select_dataset_primary_target(dataset_best_rows)
        selected = selection.get("selected") or {}
        primary_row = {
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
            "relative_improvement_vs_no_tl": selected.get("relative_improvement_vs_no_tl"),
            "relative_improvement_vs_msml": selected.get("relative_improvement_vs_msml"),
            "relative_improvement_vs_bl4": selected.get("relative_improvement_vs_bl4"),
            "relative_improvement_vs_best_bl": selected.get("relative_improvement_vs_best_bl"),
            "selection_role": "development_selected",
            "selection_frozen": "yes",
            "target_selection_stage": TRACK,
            "verdict": selection.get("verdict"),
            "PRIMARY_TARGET_EVIDENCE": selection.get("evidence"),
            "rmse_ranks": json.dumps(selection.get("rmse_ranks") or {}),
            "transfer_ranks": json.dumps(selection.get("transfer_ranks") or {}),
        }
        primary_rows.append(primary_row)
        print(format_dataset_terminal(dataset, dataset_run_rows, selection))

    assert_screening_does_not_change_formal_methods()
    manifest = build_frozen_manifest(primary_rows, run_dir=run_dir)
    provenance = {
        "method": METHOD_DISPLAY,
        "track": TRACK,
        "plan": {
            "total_targets": plan["total_targets"],
            "total_screening_runs": plan["total_screening_runs"],
            "inventory": {key: list(value) for key, value in plan["inventory"].items()},
        },
        "formal_methods": list(assert_screening_does_not_change_formal_methods()),
        "architecture": fused_target_only_architecture(),
        "auto_start_robustness": False,
        "metric_space": "original_sales_space",
    }
    output_paths = write_screening_outputs(
        run_dir,
        all_runs=[_csv_ready(row) for row in all_runs],
        target_best=[_csv_ready(row) for row in target_best_rows],
        primary_rows=primary_rows,
        manifest=manifest,
        provenance=provenance,
    )
    config_manifest = write_frozen_config_manifest(PROJECT_ROOT, manifest)
    print(format_global_summary(primary_rows))
    print("D1_D6_PRIMARY_TARGET_LR_SCREENING_COMPLETE")
    print(f"Run root: {run_dir}")
    print(f"All runs: {output_paths['all_runs_csv']}")
    print(f"Target best LR: {output_paths['target_best_csv']}")
    print(f"Primary targets: {output_paths['primary_csv']}")
    print(f"Reference comparison: {output_paths['comparison_csv']}")
    print(f"Provenance: {output_paths['provenance_json']}")
    print(f"Run manifest: {output_paths['manifest_json']}")
    print(f"Frozen config manifest: {config_manifest}")
    print(f"Predictions: {pred_root}")


if __name__ == "__main__":
    main()
