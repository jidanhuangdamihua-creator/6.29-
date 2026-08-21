"""Frozen V3 D1–D6 frame assembly reused by A–E live context.

This module does not train. It delegates to the same sealed-parquet and
protocol authorities used by V3 screening / three-track robustness.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from src.constants import SOLIDIFIED_KNN_ROOT, SOURCE_HISTORY_DAYS
from src.protocols.candidate_pool import prepare_daily_sequence_pool
from src.protocols.experiment_protocol import (
    get_experiment_protocol,
    normalize_scenario,
    normalize_source_key,
)
from src.protocols.formal_input_paths import resolve_formal_dataset_paths
from src.protocols.raw_preprocessing import TargetK3RawSourceContext
from src.protocols.runner_adapter import configure_protocol_frames, source_key_mask
from src.protocols.transformation_identity import validate_runtime_feature_contract
from src.protocols.transformation_reuse import TargetTransformationReuseContext
from src.source_selection.source_selector import TargetK3SelectionContext
from src.transfer_methods.ms_katl_ae_ablation import (
    STAGE_A_LEARNING_RATE,
    STAGE_B_BATCH_SIZE,
    STAGE_B_EPOCHS,
    AEContractError,
    project_root_from,
)
from src.transfer_methods.ms_katl_primary_target_lr_screening import SCREENING_MODE, parse_target_identity
from src.utils.d4_d6_runtime import apply_runtime_source_domain_policy, load_default_metric_protocol, validate_runtime_target_domain
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

D13_GROUP_COLS = {
    "D1": ("store_id", "item_id"),
    "D2": ("brand_id", "item_id"),
    "D3": ("store_id",),
}
D46_GROUPING_COL = {4: None, 5: "family", 6: "dept_id"}
D46_POOL_METADATA = {4: (), 5: ("family",), 6: ("dept_id",)}
WINDOW_SIZE = 10
REQUESTED_K = 3

_D13_POOL_CACHE: Dict[Tuple[str, int, int, str], dict] = {}
_D13_BASE_CACHE: Dict[Tuple[str, str], dict] = {}
_D46_POOL_CACHE: Dict[Tuple[str, str], dict] = {}
_D46_BOUND_CACHE: Dict[Tuple[str, str, str], dict] = {}


def reset_v3_frame_caches() -> None:
    _D13_POOL_CACHE.clear()
    _D13_BASE_CACHE.clear()
    _D46_POOL_CACHE.clear()
    _D46_BOUND_CACHE.clear()


def dataset_number(dataset: str) -> int:
    return int(str(dataset).strip().upper()[1:])


def _require_sealed_paths(dataset: str, project_root: Path):
    paths = resolve_formal_dataset_paths(dataset_number(dataset), repository_root=project_root)
    sealed = Path(paths.sealed_root).resolve()
    for label, path in (("source", paths.source_path), ("target", paths.target_path)):
        resolved = Path(path).resolve()
        try:
            resolved.relative_to(sealed)
        except ValueError as exc:
            raise AEContractError(f"path outside sealed root ({label}): {resolved}") from exc
        if not resolved.is_file():
            raise AEContractError(f"missing sealed parquet ({label}): {resolved}")
    return paths


def _prepare_d13_cfg(horizon: int, seed: int, epochs: int) -> Tuple[dict, dict]:
    from paper_reproduction_protocol import load_paper_protocol, resolve_strict_paper_mode
    from scripts.run_full_paper_experiments import _load_config

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


def _load_d13_base(dataset: str, *, project_root: Path, cfg: dict) -> dict:
    cache_key = (dataset, str(project_root))
    if cache_key in _D13_BASE_CACHE:
        return _D13_BASE_CACHE[cache_key]
    from scripts.run_full_paper_experiments import _load_solidified_base_data

    paths = _require_sealed_paths(dataset, project_root)
    dataset_name = f"Dataset{dataset_number(dataset)}"
    base = _load_solidified_base_data(
        dataset_name=dataset_name,
        cfg=cfg,
        strict_paper_mode=True,
        strict_paper_split=True,
        explicit_paths={"source": paths.source_path, "target": paths.target_path},
    )
    payload = {
        "dataset_name": dataset_name,
        "base": base,
        "sealed_source_path": str(paths.source_path),
        "sealed_target_path": str(paths.target_path),
        "frame_authority": "scripts.run_full_paper_experiments._load_solidified_base_data",
    }
    _D13_BASE_CACHE[cache_key] = payload
    return payload


def load_d13_pool(dataset: str, horizon: int, seed: int, *, project_root: Path) -> dict:
    cache_key = (dataset, int(horizon), int(seed), str(project_root))
    if cache_key in _D13_POOL_CACHE:
        return _D13_POOL_CACHE[cache_key]
    from scripts.run_full_paper_experiments import _prepare_cell_source_dependencies

    cfg, protocol = _prepare_d13_cfg(int(horizon), int(seed), STAGE_B_EPOCHS)
    loaded = _load_d13_base(dataset, project_root=project_root, cfg=cfg)
    dataset_name = loaded["dataset_name"]
    base = loaded["base"]
    cell_source = _prepare_cell_source_dependencies(
        dataset_name=dataset_name,
        information_sharing_scenario="without_information_sharing",
        cfg=cfg,
        protocol=protocol,
        strict_paper_mode=True,
        base_data=base,
    )
    payload = {
        "dataset_name": dataset_name,
        "base": base,
        "cell_source": cell_source,
        "cfg": cfg,
        "protocol": protocol,
        "sealed_source_path": loaded["sealed_source_path"],
        "sealed_target_path": loaded["sealed_target_path"],
        "frame_authority": loaded["frame_authority"],
    }
    _D13_POOL_CACHE[cache_key] = payload
    return payload


def prepare_d13_target_cell(dataset: str, pool: dict, parsed: dict) -> dict:
    from scripts.run_full_paper_experiments import (
        _bind_protocol_cell_identity as bind_d13,
        _cell_lifecycle_identity,
        _cell_owned_configured_source,
        _project_modeling_frame,
        _require_cell_context_identity,
    )

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
        raise AEContractError(
            f"{dataset} target {parsed['target_identity']} not found using {group_cols}"
        )
    filtered.attrs = deepcopy(target_df.attrs)
    target_df = filtered
    bind_d13(source_df, expected_lifecycle_identity)
    bind_d13(target_df, expected_lifecycle_identity)
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
    source_df = _cell_owned_configured_source(cell_source, source_df, expected_lifecycle_identity)
    return {
        "source_model_df": source_df,
        "target_model_df": target_df,
        "model_feature_cols": modeling_feature_cols,
        "group_cols": group_cols,
        "k3_selection_context": cell_source.k3_selection_context,
        "k3_raw_source_context": cell_source.k3_raw_source_context,
        "transformation_reuse_context": cell_source.transformation_reuse_context,
        "protocol_target_key": target_df.attrs["protocol_target_key"],
        "frame_authority": pool["frame_authority"],
        "sealed_source_path": pool["sealed_source_path"],
        "sealed_target_path": pool["sealed_target_path"],
    }


def _d46_config(dataset_id: int, project_root: Path) -> dict:
    return {
        "use_parquet": True,
        "dataset_id": dataset_id,
        "dataset_name": f"Dataset{dataset_id}",
        "knn_json_dir": str(Path(SOLIDIFIED_KNN_ROOT) / f"Dataset{dataset_id}"),
        "info_sharing": SCREENING_MODE,
        "source_history_days": SOURCE_HISTORY_DAYS,
        "entity_col": "entity_id",
        "horizon": 1,
        "window_size": WINDOW_SIZE,
        "learning_rate": STAGE_A_LEARNING_RATE,
        "source_epochs": int(STAGE_B_EPOCHS),
        "target_epochs": int(STAGE_B_EPOCHS),
        "batch_size": int(STAGE_B_BATCH_SIZE),
        "random_state": 42,
        "source_count": REQUESTED_K,
        "smoke": False,
        "extended_methods_enabled": True,
        "ms_katl_gamma": 0.5,
        "ms_katl_tau_k": 1.0,
        "metric_protocol": load_default_metric_protocol(project_root),
        "raw_dir": "数据集/原始数据/Dataset 5Favorita" if dataset_id == 5 else None,
    }


def load_d46_pool(dataset: str, *, project_root: Path) -> dict:
    cache_key = (dataset, str(project_root))
    if cache_key in _D46_POOL_CACHE:
        return _D46_POOL_CACHE[cache_key]
    dataset_id = dataset_number(dataset)
    config = _d46_config(dataset_id, project_root)
    paths = _require_sealed_paths(dataset, project_root)
    knn_dir = Path(SOLIDIFIED_KNN_ROOT) / f"Dataset{dataset_id}"
    if not knn_dir.is_dir():
        raise AEContractError(f"KNN authority missing: {knn_dir}")
    knn_data = load_knn_results(str(knn_dir), config["info_sharing"])
    group_cols = knn_data.get("group_cols")
    if not isinstance(group_cols, list) or len(group_cols) != 2:
        raise AEContractError(f"{dataset} KNN payload requires two group_cols, got: {group_cols}")
    config["group_cols"] = list(group_cols)
    config["knn_json_dir"] = str(knn_dir)
    windows = read_dataset_windows(
        config["dataset_id"],
        config["knn_json_dir"],
        info_sharing=config["info_sharing"],
        knn_payload=knn_data,
    )
    if dataset_id == 5:
        raw_dir = project_root / str(config["raw_dir"])
        if not raw_dir.is_dir():
            raise AEContractError(f"D5 V3 holiday authority missing: {raw_dir}")
        authorities = load_d5_authorities(raw_dir, use_holidays=True)
        loaded = load_parquet_source_target_with_diagnostics(
            dataset_id=dataset_id,
            source_path=paths.source_path,
            target_path=paths.target_path,
            windows=windows,
            source_history_days=config["source_history_days"],
            expected_dates=expected_target_dates_from_windows(windows),
            d5_authorities=authorities,
        )
        source_df = loaded.source_df
        target_df = loaded.target_df
    else:
        source_df, target_df = load_parquet_source_target(
            dataset_id=dataset_id,
            source_path=paths.source_path,
            target_path=paths.target_path,
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
    payload = {
        "config": config,
        "knn_data": knn_data,
        "source_df": source_df,
        "target_df": target_df,
        "feature_cols": feature_cols,
        "group_cols": tuple(str(col) for col in config["group_cols"]),
        "sealed_source_path": str(paths.source_path),
        "sealed_target_path": str(paths.target_path),
        "frame_authority": "src.utils.parquet_data_loader.load_parquet_source_target",
    }
    _D46_POOL_CACHE[cache_key] = payload
    return payload


def _bound_d46_frames(dataset: str, pool: dict, parsed: dict) -> dict:
    config = dict(pool["config"])
    dataset_id = int(config["dataset_id"])
    entity_key = parsed["entity_key"]
    cache_key = (dataset, str(entity_key), str(pool["sealed_source_path"]))
    if cache_key in _D46_BOUND_CACHE:
        return _D46_BOUND_CACHE[cache_key]
    knn_data = pool["knn_data"]
    source_df = pool["source_df"]
    target_df = pool["target_df"]
    feature_cols = pool["feature_cols"]
    group_cols = pool["group_cols"]
    if entity_key not in knn_data["results"]:
        raise AEContractError(f"target {entity_key} missing from {dataset} KNN results")
    validate_runtime_target_domain(target_df, [entity_key], knn_data, config)
    target_entity_df = target_df[target_df[config["entity_col"]].astype(str) == entity_key].copy()
    if target_entity_df.empty:
        raise AEContractError(f"target {entity_key} not found in {dataset} target_df")
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
    payload = {
        "config": config,
        "dataset_id": dataset_id,
        "feature_cols": feature_cols,
        "group_cols": group_cols,
        "scenario": scenario,
        "source_bound": source_bound,
        "target_bound": target_bound,
        "protocol_target_key": target_bound.attrs["protocol_target_key"],
        "frame_authority": pool["frame_authority"],
        "sealed_source_path": pool["sealed_source_path"],
        "sealed_target_path": pool["sealed_target_path"],
        "entity_key": entity_key,
    }
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
    validate_feature_frame_finite(
        source_model_df,
        model_feature_cols,
        context="ae_live_source",
        dataset_id=dataset_id,
        role="source",
        entity_id="source_pool",
        stage="ms_katl_ae_live_context",
    )
    validate_feature_frame_finite(
        target_model_df,
        model_feature_cols,
        context="ae_live_target",
        dataset_id=dataset_id,
        role="target",
        entity_id=entity_key,
        stage="ms_katl_ae_live_context",
    )
    payload["model_feature_cols"] = model_feature_cols
    payload["source_model_template"] = source_model_df
    payload["target_model_template"] = target_model_df
    _D46_BOUND_CACHE[cache_key] = payload
    return payload


def prepare_d46_target_cell(dataset: str, pool: dict, parsed: dict, *, horizon: int, seed: int) -> dict:
    bound = _bound_d46_frames(dataset, pool, parsed)
    config = dict(bound["config"])
    config["horizon"] = int(horizon)
    config["random_state"] = int(seed)
    dataset_id = int(bound["dataset_id"])
    source_model_df = bound["source_model_template"].copy(deep=False)
    target_model_df = bound["target_model_template"].copy(deep=False)
    source_model_df.attrs = deepcopy(bound["source_model_template"].attrs)
    target_model_df.attrs = deepcopy(bound["target_model_template"].attrs)
    group_cols = bound["group_cols"]
    scenario = bound["scenario"]
    model_feature_cols = bound["model_feature_cols"]
    cell_lifecycle_identity = (
        get_experiment_protocol(dataset_id).dataset_id,
        normalize_scenario(scenario),
        int(config["horizon"]),
        int(config["random_state"]),
        normalize_source_key(bound["protocol_target_key"]),
    )
    _bind_protocol_cell_identity(source_model_df, cell_lifecycle_identity)
    _bind_protocol_cell_identity(target_model_df, cell_lifecycle_identity)
    _require_bound_protocol_cell_identity(source_model_df, cell_lifecycle_identity)
    _require_bound_protocol_cell_identity(target_model_df, cell_lifecycle_identity)
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
        "protocol_target_key": bound["protocol_target_key"],
        "frame_authority": bound["frame_authority"],
        "sealed_source_path": bound["sealed_source_path"],
        "sealed_target_path": bound["sealed_target_path"],
    }


def assemble_v3_cell_frames(
    dataset: str,
    target_identity: str,
    *,
    horizon: int,
    seed: int,
    project_root: Optional[Path] = None,
) -> dict:
    """Return the frozen V3 source/target modeling frames for one AE cell. No training."""
    root = project_root_from(project_root)
    parsed = parse_target_identity(dataset, target_identity)
    if parsed["target_identity"] != target_identity:
        raise AEContractError(f"target identity drifted: {parsed['target_identity']}")
    if dataset in {"D1", "D2", "D3"}:
        pool = load_d13_pool(dataset, horizon, seed, project_root=root)
        cell = prepare_d13_target_cell(dataset, pool, parsed)
    elif dataset in {"D4", "D5", "D6"}:
        pool = load_d46_pool(dataset, project_root=root)
        cell = prepare_d46_target_cell(dataset, pool, parsed, horizon=horizon, seed=seed)
    else:
        raise AEContractError(f"unknown dataset {dataset}")
    if cell["source_model_df"].empty or cell["target_model_df"].empty:
        raise AEContractError(f"empty V3 frames for {dataset}/{target_identity}")
    cell["parsed"] = parsed
    cell["scenario"] = SCREENING_MODE
    return cell
