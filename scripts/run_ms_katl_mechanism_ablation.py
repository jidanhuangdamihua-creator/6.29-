"""MS-KATL-Lite single-cell mechanism ablation runner (extended / diagnostic).

Fixed cell:
    D4 | store_id=166, product_id=258 | without | H=1 | seed=42

Does not modify FORMAL_METHODS, MSML-TL, No-TL, or existing viability results.

Usage:
    python scripts/run_ms_katl_mechanism_ablation.py --preflight
    python scripts/run_ms_katl_mechanism_ablation.py
    python scripts/run_ms_katl_mechanism_ablation.py --epochs 1   # targeted dry training only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.constants import SOLIDIFIED_KNN_ROOT, SOURCE_HISTORY_DAYS

import tf_compat  # noqa: F401

import pandas as pd

from src.protocols.candidate_pool import prepare_daily_sequence_pool
from src.protocols.experiment_protocol import (
    get_experiment_protocol,
    normalize_scenario,
    normalize_source_key,
)
from src.protocols.formal_input_paths import resolve_formal_dataset_paths
from src.protocols.raw_preprocessing import TargetK3RawSourceContext
from src.protocols.reproducibility import set_protocol_seed
from src.protocols.runner_adapter import configure_protocol_frames
from src.protocols.transformation_identity import validate_runtime_feature_contract
from src.protocols.transformation_reuse import TargetTransformationReuseContext
from src.source_selection.source_selector import TargetK3SelectionContext
from src.transfer_methods.ms_katl_lite import prepare_ms_katl_stage_a
from src.transfer_methods.ms_katl_mechanism_ablation import (
    FIXED_CELL,
    assert_formal_methods_untouched,
    prepare_shared_target_bundle,
    print_preflight,
    run_six_arm_mechanism_ablation,
)
from src.utils.d4_d6_runtime import (
    apply_runtime_source_domain_policy,
    load_default_metric_protocol,
    validate_runtime_target_domain,
)
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
    load_knn_results,
    load_parquet_source_target,
    read_dataset_windows,
)
from src.utils.run_utils import create_run_dir, reserve_new_output_dir

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MS-KATL-Lite single-cell mechanism ablation (extended track)."
    )
    parser.add_argument("--preflight", action="store_true", help="Print the fixed cell contract and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Alias for --preflight.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _prepare_fixed_cell_frames(config: dict):
    formal_paths = resolve_formal_dataset_paths(4, repository_root=PROJECT_ROOT)
    knn_data = load_knn_results(config["knn_json_dir"], config["info_sharing"])
    group_cols = knn_data.get("group_cols")
    if not isinstance(group_cols, list) or len(group_cols) != 2:
        raise ValueError(f"D4 KNN payload requires two group_cols, got: {group_cols}")
    config["group_cols"] = list(group_cols)

    windows = read_dataset_windows(
        config["dataset_id"],
        config["knn_json_dir"],
        info_sharing=config["info_sharing"],
        knn_payload=knn_data,
    )
    source_df, target_df = load_parquet_source_target(
        dataset_id=config["dataset_id"],
        source_path=formal_paths.source_path,
        target_path=formal_paths.target_path,
        windows=windows,
        source_history_days=config["source_history_days"],
    )
    feature_info = resolve_knn_feature_columns(
        dataset_id=config["dataset_id"],
        information_sharing=config["info_sharing"],
        knn_root=SOLIDIFIED_KNN_ROOT,
        source_df=source_df,
        target_df=target_df,
        knn_payload=knn_data,
    )
    feature_cols = list(feature_info["selected_features"])
    validate_runtime_feature_contract(
        "D4",
        model_feature_cols=feature_cols,
        knn_feature_cols=get_experiment_protocol("D4").knn_feature_columns,
    )
    source_df = apply_runtime_source_domain_policy(source_df, knn_data, config)
    entity_key = FIXED_CELL["entity_key"]
    if entity_key not in knn_data["results"]:
        raise ValueError(f"fixed target {entity_key} missing from D4 KNN results")
    validate_runtime_target_domain(target_df, [entity_key], knn_data, config)
    selected_source_entities = {
        str(item.get("source_entity"))
        for item in knn_data["results"][entity_key][: int(config["smoke_source_limit"])]
        if isinstance(item, dict) and item.get("source_entity") is not None
    }
    if selected_source_entities:
        source_df = source_df[
            source_df[config["entity_col"]].astype(str).isin(selected_source_entities)
        ].copy()
    target_entity_df = target_df[target_df[config["entity_col"]] == entity_key].copy()
    if target_entity_df.empty:
        raise ValueError(f"fixed target {entity_key} not found in target_df")

    scenario = _scenario_name(config)
    source_selection_group_cols = tuple(str(col) for col in config["group_cols"])
    observed_start = target_entity_df.attrs.get(
        "knn_observed_start",
        target_entity_df.attrs.get(
            "target_observed_start",
            pd.to_datetime(target_entity_df["date"], errors="raise").min(),
        ),
    )
    prepared_pool = prepare_daily_sequence_pool(
        source_df,
        group_cols=source_selection_group_cols,
        observed_start=observed_start,
        feature_cols=get_experiment_protocol(4).knn_feature_columns,
        capture_source_observed_frame_digest=True,
    )
    source_df, target_entity_df = configure_protocol_frames(
        source_df,
        target_entity_df,
        dataset_id=4,
        scenario=scenario,
        group_cols=source_selection_group_cols,
        grouping_col=None,
        observed_start=observed_start,
        prepared_pool=prepared_pool,
        retain_source_frame=True,
        enforce_formal_target=True,
    )
    cell_lifecycle_identity = (
        get_experiment_protocol(4).dataset_id,
        normalize_scenario(scenario),
        int(config["horizon"]),
        int(config["random_state"]),
        normalize_source_key(target_entity_df.attrs["protocol_target_key"]),
    )
    _bind_protocol_cell_identity(source_df, cell_lifecycle_identity)
    _bind_protocol_cell_identity(target_entity_df, cell_lifecycle_identity)
    model_feature_cols = _resolve_model_feature_cols(source_df, target_entity_df, feature_cols, config)
    source_model_df = _sanitize_source_model_dataframe(
        _build_model_dataframe(
            source_df,
            model_feature_cols,
            source_selection_group_cols=source_selection_group_cols,
            required_passthrough_cols=get_experiment_protocol(4).knn_feature_columns,
        ),
        model_feature_cols,
    )
    target_model_df = _build_model_dataframe(
        target_entity_df,
        model_feature_cols,
        source_selection_group_cols=source_selection_group_cols,
        required_passthrough_cols=get_experiment_protocol(4).knn_feature_columns,
    )
    _require_bound_protocol_cell_identity(source_model_df, cell_lifecycle_identity)
    _require_bound_protocol_cell_identity(target_model_df, cell_lifecycle_identity)
    validate_feature_frame_finite(
        source_model_df, model_feature_cols, context="mechanism_ablation_source",
        dataset_id=4, role="source", entity_id="source_pool", stage="mechanism_ablation",
    )
    validate_feature_frame_finite(
        target_model_df, model_feature_cols, context="mechanism_ablation_target",
        dataset_id=4, role="target", entity_id=entity_key, stage="mechanism_ablation",
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
        "group_cols": source_selection_group_cols,
        "k3_selection_context": k3_selection_context,
        "k3_raw_source_context": k3_raw_source_context,
        "transformation_reuse_context": transformation_reuse_context,
        "protocol_target_key": target_entity_df.attrs["protocol_target_key"],
    }


def main() -> None:
    args = _parse_args()
    print_preflight()
    if args.preflight or args.dry_run:
        return

    assert_formal_methods_untouched()
    set_protocol_seed(int(FIXED_CELL["seed"]), include_frameworks=True)
    config = {
        "use_parquet": True,
        "dataset_id": 4,
        "dataset_name": "Dataset4",
        "knn_json_dir": str(SOLIDIFIED_KNN_ROOT / "Dataset4"),
        "info_sharing": FIXED_CELL["mode"],
        "source_history_days": SOURCE_HISTORY_DAYS,
        "entity_col": "entity_id",
        "horizon": int(FIXED_CELL["horizon"]),
        "window_size": 10,
        "learning_rate": 0.0001,
        "source_epochs": int(args.epochs),
        "target_epochs": int(args.epochs),
        "batch_size": 16,
        "random_state": int(FIXED_CELL["seed"]),
        "source_count": 3,
        "smoke": True,
        "extended_methods_enabled": True,
        "smoke_source_limit": 3,
        "ms_katl_gamma": float(FIXED_CELL["gamma"]),
        "ms_katl_tau_k": 1.0,
        "metric_protocol": load_default_metric_protocol(PROJECT_ROOT),
    }
    cell = _prepare_fixed_cell_frames(config)
    if args.output_dir is None:
        run_dir = create_run_dir(PROJECT_ROOT, label="MS_KATL_MECHANISM_D4_h1_s42")
    else:
        run_dir = Path(args.output_dir)
        if not run_dir.is_absolute():
            run_dir = PROJECT_ROOT / run_dir
        reserve_new_output_dir(run_dir)
    (run_dir / "run_config.json").write_text(
        json.dumps({**config, "ablation_arms": 6, "shared_stage_a": True}, indent=2, default=str),
        encoding="utf-8",
    )

    print("Preparing shared Stage-A artifacts once...")
    stage_a = prepare_ms_katl_stage_a(
        source_df=cell["source_model_df"],
        target_df=cell["target_model_df"],
        feature_cols=cell["model_feature_cols"],
        k=3,
        horizon=int(config["horizon"]),
        window_size=int(config["window_size"]),
        learning_rate=float(config["learning_rate"]),
        source_epochs=int(config["source_epochs"]),
        batch_size=int(config["batch_size"]),
        tau_k=float(config["ms_katl_tau_k"]),
        group_cols=cell["group_cols"],
        k3_selection_context=cell["k3_selection_context"],
        k3_raw_source_context=cell["k3_raw_source_context"],
        transformation_reuse_context=cell["transformation_reuse_context"],
        seed=int(config["random_state"]),
    )
    bundle = prepare_shared_target_bundle(
        stage_a,
        cell["target_model_df"],
        cell["model_feature_cols"],
        horizon=int(config["horizon"]),
        window_size=int(config["window_size"]),
        learning_rate=float(config["learning_rate"]),
        seed=int(config["random_state"]),
        transformation_reuse_context=cell["transformation_reuse_context"],
    )
    print(
        "STAGE_A ready | identities="
        f"{stage_a.source_identities} | digests={stage_a.source_weight_digests}"
    )
    result = run_six_arm_mechanism_ablation(
        bundle,
        epochs=int(args.epochs),
        batch_size=int(config["batch_size"]),
        gamma=float(config["ms_katl_gamma"]),
        lambda_adv=0.0,
        metric_protocol=config["metric_protocol"],
        run_dir=run_dir,
        cell_meta={
            **FIXED_CELL,
            "epochs": int(args.epochs),
            "protocol_target_key": cell["protocol_target_key"],
        },
    )
    print(result["summary"])
    if result["output_paths"]:
        print(f"Results CSV: {result['output_paths']['results_csv']}")
        print(f"Diagnostics: {result['output_paths']['diagnostics_json']}")
        print(f"Predictions: {result['output_paths']['predictions_dir']}")


if __name__ == "__main__":
    main()
