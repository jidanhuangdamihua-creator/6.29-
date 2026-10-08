"""MS-KATL-Lite D4 cross-target validation runner (extended / diagnostic).

Fixed matrix:
    D4 | 5 targets | without | H=1 | seed=42 | arms={A,B,C}

Does not modify FORMAL_METHODS, MSML-TL, No-TL, baselines, or architecture.

Usage:
    python scripts/run_ms_katl_d4_cross_target.py --preflight
    python scripts/run_ms_katl_d4_cross_target.py
    python scripts/run_ms_katl_d4_cross_target.py --epochs 1
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
from src.transfer_methods.ms_katl_cross_target import (
    CROSS_TARGET_LABEL,
    DEFAULT_BASELINE_RELATIVE,
    DEFAULT_FORMAL_RELATIVE,
    aggregate_cross_target,
    assert_new_run_does_not_clobber_historical,
    assert_protected_artifacts_unchanged,
    build_target_comparison,
    decide_cross_target_verdict,
    format_cross_target_terminal_summary,
    format_target_method_table,
    load_baseline_references,
    load_formal_references,
    overall_pairing_flags,
    parse_cross_target_arm_spec,
    parse_cross_target_targets,
    parse_d4_target_identity,
    plan_d4_cross_target,
    print_d4_cross_target_preflight,
    snapshot_protected_artifacts,
    write_cross_target_outputs,
)
from src.transfer_methods.ms_katl_lite import prepare_ms_katl_stage_a
from src.transfer_methods.ms_katl_mechanism_ablation import (
    assert_formal_methods_untouched,
    prepare_shared_target_bundle,
    run_abc_mechanism_ablation,
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
        description="MS-KATL-Lite D4 H=1 seed=42 A/B/C cross-target validation."
    )
    parser.add_argument("--preflight", action="store_true", help="Print the cross-target contract and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Alias for --preflight.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--targets", default="all")
    parser.add_argument(
        "--arms",
        default="full_current,fused_target_only,raw_target_only",
        help="Must be exactly A/B/C. Unknown or D/E/F arms fail closed.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--formal-csv", type=Path, default=None)
    parser.add_argument("--baseline-csv", type=Path, default=None)
    return parser.parse_args()


def _load_d4_without_pool(config: dict):
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
    return {
        "knn_data": knn_data,
        "source_df": source_df,
        "target_df": target_df,
        "feature_cols": feature_cols,
        "group_cols": tuple(str(col) for col in config["group_cols"]),
    }


def prepare_d4_without_target_cell(
    config: dict,
    *,
    entity_key: str,
    knn_data: dict,
    source_df,
    target_df,
    feature_cols,
    group_cols,
):
    if entity_key not in knn_data["results"]:
        raise ValueError(f"target {entity_key} missing from D4 KNN results")
    validate_runtime_target_domain(target_df, [entity_key], knn_data, config)
    selected_source_entities = {
        str(item.get("source_entity"))
        for item in knn_data["results"][entity_key][: int(config["smoke_source_limit"])]
        if isinstance(item, dict) and item.get("source_entity") is not None
    }
    source_slice = source_df
    if selected_source_entities:
        source_slice = source_df[
            source_df[config["entity_col"]].astype(str).isin(selected_source_entities)
        ].copy()
    target_entity_df = target_df[target_df[config["entity_col"]] == entity_key].copy()
    if target_entity_df.empty:
        raise ValueError(f"target {entity_key} not found in target_df")

    scenario = _scenario_name(config)
    observed_start = target_entity_df.attrs.get(
        "knn_observed_start",
        target_entity_df.attrs.get(
            "target_observed_start",
            pd.to_datetime(target_entity_df["date"], errors="raise").min(),
        ),
    )
    prepared_pool = prepare_daily_sequence_pool(
        source_slice,
        group_cols=group_cols,
        observed_start=observed_start,
        feature_cols=get_experiment_protocol(4).knn_feature_columns,
        capture_source_observed_frame_digest=True,
    )
    source_bound, target_bound = configure_protocol_frames(
        source_slice,
        target_entity_df,
        dataset_id=4,
        scenario=scenario,
        group_cols=group_cols,
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
            required_passthrough_cols=get_experiment_protocol(4).knn_feature_columns,
        ),
        model_feature_cols,
    )
    target_model_df = _build_model_dataframe(
        target_bound,
        model_feature_cols,
        source_selection_group_cols=group_cols,
        required_passthrough_cols=get_experiment_protocol(4).knn_feature_columns,
    )
    _require_bound_protocol_cell_identity(source_model_df, cell_lifecycle_identity)
    _require_bound_protocol_cell_identity(target_model_df, cell_lifecycle_identity)
    validate_feature_frame_finite(
        source_model_df, model_feature_cols, context="cross_target_source",
        dataset_id=4, role="source", entity_id="source_pool", stage="ms_katl_cross_target",
    )
    validate_feature_frame_finite(
        target_model_df, model_feature_cols, context="cross_target_target",
        dataset_id=4, role="target", entity_id=entity_key, stage="ms_katl_cross_target",
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


def _build_config(epochs: int) -> dict:
    return {
        "use_parquet": True,
        "dataset_id": 4,
        "dataset_name": "Dataset4",
        "knn_json_dir": str(SOLIDIFIED_KNN_ROOT / "Dataset4"),
        "info_sharing": "without",
        "source_history_days": SOURCE_HISTORY_DAYS,
        "entity_col": "entity_id",
        "horizon": 1,
        "window_size": 10,
        "learning_rate": 0.0001,
        "source_epochs": int(epochs),
        "target_epochs": int(epochs),
        "batch_size": 16,
        "random_state": 42,
        "source_count": 3,
        "smoke": True,
        "extended_methods_enabled": True,
        "smoke_source_limit": 3,
        "ms_katl_gamma": 0.5,
        "ms_katl_tau_k": 1.0,
        "metric_protocol": load_default_metric_protocol(PROJECT_ROOT),
    }


def main() -> None:
    args = _parse_args()
    plan = plan_d4_cross_target(targets=args.targets, arms=args.arms)
    parse_cross_target_arm_spec(args.arms)
    parse_cross_target_targets(args.targets)
    print_d4_cross_target_preflight()
    if args.preflight or args.dry_run:
        return

    assert_formal_methods_untouched()
    protected_before = snapshot_protected_artifacts(PROJECT_ROOT)
    config = _build_config(args.epochs)
    if args.output_dir is None:
        run_dir = create_run_dir(PROJECT_ROOT, label=CROSS_TARGET_LABEL)
    else:
        run_dir = Path(args.output_dir)
        if not run_dir.is_absolute():
            run_dir = PROJECT_ROOT / run_dir
        reserve_new_output_dir(run_dir)
    assert_new_run_does_not_clobber_historical(PROJECT_ROOT, run_dir)
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                **{k: v for k, v in config.items() if k != "metric_protocol"},
                "targets": list(plan["targets"]),
                "arms": list(plan["arms"]),
                "total_arm_runs": 15,
                "shared_stage_a_within_target": True,
                "formal_methods_rerun": False,
                "baselines_rerun": False,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    pool = _load_d4_without_pool(config)
    formal_csv = args.formal_csv or (PROJECT_ROOT / DEFAULT_FORMAL_RELATIVE)
    baseline_csv = args.baseline_csv or (PROJECT_ROOT / DEFAULT_BASELINE_RELATIVE)
    comparisons = []
    for identity in plan["targets"]:
        parsed = parse_d4_target_identity(identity)
        print(f"\n=== TARGET {parsed['target_identity']} | Stage-A once, then A/B/C ===")
        set_protocol_seed(int(config["random_state"]), include_frameworks=True)
        cell = prepare_d4_without_target_cell(
            config,
            entity_key=parsed["entity_key"],
            knn_data=pool["knn_data"],
            source_df=pool["source_df"],
            target_df=pool["target_df"],
            feature_cols=pool["feature_cols"],
            group_cols=pool["group_cols"],
        )
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
        cell_meta = {
            "dataset": "D4",
            "target_identity": parsed["target_identity"],
            "target_store_id": parsed["store_id"],
            "target_product_id": parsed["product_id"],
            "entity_key": parsed["entity_key"],
            "mode": "without",
            "horizon": 1,
            "seed": 42,
            "epochs": int(args.epochs),
            "gamma": 0.5,
            "lambda_adv": 0.0,
            "protocol_target_key": cell["protocol_target_key"],
        }
        abc = run_abc_mechanism_ablation(
            bundle,
            epochs=int(args.epochs),
            batch_size=int(config["batch_size"]),
            gamma=float(config["ms_katl_gamma"]),
            lambda_adv=0.0,
            metric_protocol=config["metric_protocol"],
            cell_meta=cell_meta,
        )
        formal_refs = load_formal_references(
            formal_csv,
            target_identity=parsed["target_identity"],
            horizon=1,
            seed=42,
            mode="without",
        )
        baseline_refs = load_baseline_references(
            baseline_csv,
            target_identity=parsed["target_identity"],
            horizon=1,
            seed=42,
            mode="without",
        )
        comparison = build_target_comparison(
            target_identity=parsed["target_identity"],
            arm_rows=abc["arm_rows"],
            flags=abc["flags"],
            formal_refs=formal_refs,
            baseline_refs=baseline_refs,
            cell_meta=cell_meta,
        )
        print(format_target_method_table(comparison))
        comparisons.append(comparison)

    aggregates = aggregate_cross_target(comparisons)
    pairing_flags = overall_pairing_flags(comparisons)
    protected_after = snapshot_protected_artifacts(PROJECT_ROOT)
    assert_protected_artifacts_unchanged(protected_before, protected_after)
    pairing_flags["HISTORICAL_RESULTS_UNTOUCHED"] = "YES"
    pairing_flags["FORMAL_METHODS_UNCHANGED"] = "YES"
    verdict = decide_cross_target_verdict(aggregates, comparisons)
    output_paths = write_cross_target_outputs(
        run_dir,
        comparisons,
        aggregates,
        cell_contract={
            "dataset": "D4",
            "mode": "without",
            "horizon": 1,
            "seed": 42,
            "epochs": int(args.epochs),
            "gamma": 0.5,
            "lambda_adv": 0.0,
            "targets": list(plan["targets"]),
            "arms": list(plan["arms"]),
            "total_arm_runs": 15,
        },
        pairing_flags=pairing_flags,
        protected_before=protected_before,
        protected_after=protected_after,
        verdict=verdict,
    )
    summary = format_cross_target_terminal_summary(
        comparisons,
        aggregates,
        pairing_flags,
        verdict=verdict,
    )
    print(summary)
    print(f"Results CSV: {output_paths['results_csv']}")
    print(f"Summary CSV: {output_paths['summary_csv']}")
    print(f"Diagnostics: {output_paths['diagnostics_json']}")
    print(f"Predictions: {output_paths['predictions_dir']}")
    print(f"Run root: {run_dir}")


if __name__ == "__main__":
    main()
