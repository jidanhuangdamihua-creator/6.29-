"""Validate MSML-TL-RFE RFE leakage fixes (RFE-only and full runs)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_preprocessing import temporal_split_by_ratio_or_dates
from experiment_runner import prepare_base_data_for_experiments, run_msml_rfe_experiment
from msml_tl_rfe import (
    _normalize_source_key,
    _prepare_source_split,
    build_joint_rfe_training_dataframe,
    run_rfe_feature_selection,
)
from paper_reproduction_protocol import load_paper_protocol, resolve_strict_paper_mode
from source_selector import SourceSelector
from scripts.run_full_paper_experiments import (
    _apply_information_sharing_filter,
    _load_config,
    _resolve_dataset_feature_cols,
    _scenario_to_bool,
    enrich_dataset3_source_audit_rows,
)


def _parse_seed_list(seed_text: str) -> List[int]:
    seeds: List[int] = []
    for part in seed_text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            seeds.extend(list(range(start, end + 1)))
        else:
            seeds.append(int(part))
    return seeds


def _parse_int_list(value_text: str) -> List[int]:
    return [int(part.strip()) for part in str(value_text).split(",") if part.strip()]


def _run_rfe_only(
    dataset_name: str,
    seed: int,
    source_count: int,
    information_sharing: str,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    start = time.time()
    record: Dict[str, Any] = {
        "dataset_id": dataset_name,
        "seed": int(seed),
        "status": "FAIL",
        "error_message": "",
    }

    try:
        protocol = load_paper_protocol(cfg)
        strict_paper_mode = resolve_strict_paper_mode(cfg, explicit=None)

        base = prepare_base_data_for_experiments(
            dataset_name=dataset_name,
            data_path=cfg["dataset_paths"][dataset_name],
            config=cfg,
            verbose_mode="summary",
        )
        source_df = base["source_df"]
        target_df = base["target_df"].copy()

        feature_cols = _resolve_dataset_feature_cols(
            dataset_name=dataset_name,
            source_df=source_df,
            target_df=target_df,
            cfg=cfg,
        )

        use_information_sharing = _scenario_to_bool(information_sharing)
        source_df = _apply_information_sharing_filter(
            dataset_name=dataset_name,
            source_df=source_df,
            target_df=target_df,
            use_information_sharing=use_information_sharing,
            strict_paper_mode=bool(strict_paper_mode),
            protocol=protocol,
            cfg=cfg,
        )
        target_df.attrs["information_sharing_scenario"] = source_df.attrs.get(
            "information_sharing_scenario", ""
        )
        source_df_rows = int(len(source_df))
        source_df_entity_id_unique = int(source_df["entity_id"].nunique(dropna=True)) if "entity_id" in source_df.columns else 0

        # Respect configured source_selection_window if present in cfg
        source_selection_window = (
            cfg.get("paper_reproduction", {})
            .get("paper_split_protocol", {})
            .get("source_selection_window")
        )
        source_selection_window_norm = str(source_selection_window or "target_observed_window").strip().lower()
        selection_target_df = target_df
        if source_selection_window_norm in {
            "train_window",
            "target_train_window",
            "observed_window",
            "target_observed_window",
            "train_val_window",
        }:
            target_train_df, target_val_df, _ = temporal_split_by_ratio_or_dates(target_df)
            for split_df in (target_train_df, target_val_df):
                split_df.attrs = target_df.attrs.copy()
            if source_selection_window_norm in {"train_window", "target_train_window"}:
                selection_target_df = target_train_df
            else:
                selection_target_df = pd.concat([target_train_df, target_val_df], axis=0, ignore_index=True)
                selection_target_df.attrs = target_df.attrs.copy()

        np.random.seed(seed)
        selector = SourceSelector()
        selection_result = selector.select_top_k_sources(
            target_df=selection_target_df,
            source_df=source_df,
            feature_cols=feature_cols,
            k=int(source_count),
            weight_mode=str(cfg["single_experiment"]["weight_mode"]),
            include_sales_in_knn=True,
        )
        selected_sources = (
            selection_result.get("sources", [])
            if isinstance(selection_result, dict)
            else selection_result
        )
        if not selected_sources:
            raise ValueError("No sources selected by SourceSelector")

        target_train_df, _, _ = temporal_split_by_ratio_or_dates(target_df)

        selected_source_train_dfs: List[pd.DataFrame] = []
        for selected in selected_sources:
            source_key = _normalize_source_key(selected.get("source_key"))
            entity_id, item_id = source_key
            source_sequence_df = source_df[
                (source_df["entity_id"] == entity_id) & (source_df["item_id"] == item_id)
            ].copy()
            if source_sequence_df.empty:
                raise ValueError(f"Selected source_key not found: {source_key}")
            src_train, _, _ = _prepare_source_split(source_sequence_df)
            selected_source_train_dfs.append(src_train)

        joint_train_df = build_joint_rfe_training_dataframe(
            target_train_df=target_train_df,
            selected_source_dfs=selected_source_train_dfs,
            feature_cols=feature_cols,
            target_col="sales",
        )
        joint_audit = dict(joint_train_df.attrs.get("rfe_audit", {}))

        rfe_result = run_rfe_feature_selection(
            train_df=joint_train_df,
            feature_cols=feature_cols,
            target_col="sales",
            estimator_name=str(cfg["single_experiment"]["estimator_name"]),
            keep_ratio=float(cfg["single_experiment"]["keep_ratio"]),
            random_state=int(seed),
            use_sales_as_history_input=True,
        )

        rfe_candidate_cols = list(rfe_result.get("rfe_candidate_cols", []))
        rfe_selected_features = list(rfe_result.get("rfe_selected_features", []))
        final_selected_features = list(rfe_result.get("final_selected_features", []))
        target_col = str(rfe_result.get("target_col", "sales"))
        duplicate_sales_count = int(joint_audit.get("duplicate_sales_after", 0))
        x_duplicate_columns = list(rfe_result.get("duplicate_columns_after", []))

        contains_target_in_rfe_X = bool(rfe_result.get("contains_target_in_rfe_X", False))
        contains_target_in_rfe_candidates = target_col in rfe_candidate_cols
        contains_target_in_rfe_selected = target_col in rfe_selected_features
        contains_target_in_final_features = target_col in final_selected_features
        sales_added_back_as_history_input = bool(rfe_result.get("sales_added_back_as_history_input", False))

        status = "PASS"
        if contains_target_in_rfe_X:
            status = "FAIL"
        if contains_target_in_rfe_candidates:
            status = "FAIL"
        if contains_target_in_rfe_selected:
            status = "FAIL"
        if duplicate_sales_count != 0:
            status = "FAIL"
        if x_duplicate_columns:
            status = "FAIL"
        if contains_target_in_final_features and not sales_added_back_as_history_input:
            status = "FAIL"

        record.update(
            {
                "source_df_rows": source_df_rows,
                "source_df_entity_id_unique": source_df_entity_id_unique,
                "original_feature_cols": list(rfe_result.get("original_feature_cols", [])),
                "rfe_candidate_cols": rfe_candidate_cols,
                "rfe_selected_features": rfe_selected_features,
                "final_selected_features": final_selected_features,
                "target_col": target_col,
                "contains_target_in_rfe_X": contains_target_in_rfe_X,
                "contains_target_in_rfe_candidates": contains_target_in_rfe_candidates,
                "contains_target_in_rfe_selected": contains_target_in_rfe_selected,
                "contains_target_in_final_features": contains_target_in_final_features,
                "sales_added_back_as_history_input": sales_added_back_as_history_input,
                "joint_df_duplicate_columns": list(joint_audit.get("duplicate_columns_after", [])),
                "joint_df_duplicate_sales_count": duplicate_sales_count,
                "X_duplicate_columns": x_duplicate_columns,
                "X_shape": rfe_result.get("rfe_input_shape"),
                "y_shape": rfe_result.get("rfe_y_shape"),
                "n_features_to_select": int(rfe_result.get("n_features_to_select", 0)),
                "keep_ratio": float(rfe_result.get("keep_ratio", 0.0)),
                "status": status,
            }
        )
    except Exception as exc:
        record["status"] = "FAIL"
        record["error_message"] = str(exc)

    record["run_time_seconds"] = float(time.time() - start)
    return record


def _run_full_training(
    dataset_name: str,
    seed: int,
    source_count: int,
    information_sharing: str,
    horizon: int,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    start = time.time()
    record: Dict[str, Any] = {
        "dataset_id": dataset_name,
        "seed": int(seed),
        "scenario": information_sharing,
        "horizon": int(horizon),
        "status": "FAIL",
        "error_message": "",
    }

    try:
        protocol = load_paper_protocol(cfg)
        strict_paper_mode = resolve_strict_paper_mode(cfg, explicit=None)

        base = prepare_base_data_for_experiments(
            dataset_name=dataset_name,
            data_path=cfg["dataset_paths"][dataset_name],
            config=cfg,
            verbose_mode="summary",
        )
        source_df = base["source_df"]
        target_df = base["target_df"].copy()

        feature_cols = _resolve_dataset_feature_cols(
            dataset_name=dataset_name,
            source_df=source_df,
            target_df=target_df,
            cfg=cfg,
        )

        use_information_sharing = _scenario_to_bool(information_sharing)
        source_df = _apply_information_sharing_filter(
            dataset_name=dataset_name,
            source_df=source_df,
            target_df=target_df,
            use_information_sharing=use_information_sharing,
            strict_paper_mode=bool(strict_paper_mode),
            protocol=protocol,
            cfg=cfg,
        )
        target_df.attrs["information_sharing_scenario"] = source_df.attrs.get(
            "information_sharing_scenario", ""
        )
        source_df_rows = int(len(source_df))
        source_df_entity_id_unique = int(source_df["entity_id"].nunique(dropna=True)) if "entity_id" in source_df.columns else 0

        np.random.seed(seed)
        source_selection_window_cfg = (
            cfg.get("paper_reproduction", {})
            .get("paper_split_protocol", {})
            .get("source_selection_window")
        )
        if not source_selection_window_cfg:
            source_selection_window_cfg = "target_observed_window"
        record["source_selection_window"] = source_selection_window_cfg

        raw = run_msml_rfe_experiment(
            source_df=source_df,
            target_df=target_df,
            feature_cols=feature_cols,
            k=int(source_count),
            number_of_sources=int(source_count),
            horizon=int(horizon),
            window_size=int(cfg["single_experiment"]["window_size"]),
            weight_mode=str(cfg["single_experiment"]["weight_mode"]),
            estimator_name=str(cfg["single_experiment"]["estimator_name"]),
            keep_ratio=float(cfg["single_experiment"]["keep_ratio"]),
            include_sales_in_knn=True,
            learning_rate=float(cfg["single_experiment"]["learning_rate"]),
            source_epochs=int(cfg["single_experiment"]["source_epochs"]),
            target_epochs=int(cfg["single_experiment"]["target_epochs"]),
            batch_size=int(cfg["single_experiment"]["batch_size"]),
            random_state=int(seed),
            metric_protocol=protocol.get("metric_protocol", {}),
            source_selection_window=source_selection_window_cfg,
        )

        meta = raw.get("meta", {}) if isinstance(raw, dict) else {}
        rfe_info = meta.get("rfe_info", {}) if isinstance(meta.get("rfe_info"), dict) else {}

        rfe_candidate_features = list(meta.get("rfe_candidate_features", rfe_info.get("rfe_candidate_features", [])))
        rfe_selected_features = list(meta.get("rfe_selected_features", rfe_info.get("rfe_selected_features", [])))
        final_selected_features = list(
            meta.get("final_selected_features", meta.get("selected_feature_cols", []))
        )
        selected_sources = meta.get("selected_sources", [])
        source_identification: List[Dict[str, Any]] = []
        source_audit_notes: List[str] = []
        if isinstance(selected_sources, list):
            for idx, source_meta in enumerate(selected_sources, start=1):
                source_identification.append(
                    {
                        "dataset": dataset_name,
                        "method": "MSML-TL-RFE",
                        "information_sharing": information_sharing,
                        "requested_source_count": int(source_count),
                        "effective_source_count": int(source_count),
                        "source_rank": int(idx),
                        "source_key": str(source_meta.get("source_key")),
                        "distance": float(source_meta.get("distance", 0.0)),
                        "weight": float(source_meta.get("weight", 0.0)),
                        "source_pool_scope_mode": str(source_df.attrs.get("source_pool_scope_mode", "")),
                        "source_pool_scope_note": str(source_df.attrs.get("source_pool_scope_note", "")),
                        "signature_base_features": "|".join(feature_cols),
                        "signature_static_features": "|".join(source_df.attrs.get("signature_static_feature_cols", [])),
                    }
                )
        if dataset_name == "Dataset3" and source_identification:
            source_identification = enrich_dataset3_source_audit_rows(
                rows=source_identification,
                source_df=source_df,
                target_df=target_df,
                notes=source_audit_notes,
            )
        metric_space_used = raw.get("metric_space", raw.get("metric_space_current", "normalized_minmax_space"))
        metric_space_current = raw.get("metric_space_current", "normalized_minmax_space")
        metric_space_paper = raw.get("metric_space_paper", "original_sales_space")
        paper_metric_aligned = bool(raw.get("paper_metric_aligned", False))
        inverse_transform_applied = bool(raw.get("inverse_transform_applied", False))
        metric_notes = str(raw.get("metric_notes", ""))
        val_metric_notes = str(raw.get("val_metric_notes", ""))

        rmse = float(raw.get("rmse", float("nan")))
        accuracy = float(raw.get("accuracy", float("nan")))
        mae = float(raw.get("mae", float("nan")))
        rmse_normalized = float(raw.get("rmse_current", rmse))
        accuracy_normalized = float(raw.get("accuracy_current", accuracy))
        mae_normalized = float(raw.get("mae_current", mae))

        val_rmse_normalized = float(raw.get("val_rmse_current", float("nan")))
        test_rmse_normalized = rmse_normalized
        test_mae_normalized = mae_normalized

        val_original_ok = bool(raw.get("val_inverse_transform_applied", inverse_transform_applied))
        test_original_ok = inverse_transform_applied
        val_rmse_original = float(raw.get("val_rmse_paper", float("nan"))) if val_original_ok else float("nan")
        test_rmse_original = float(raw.get("rmse_paper", float("nan"))) if test_original_ok else float("nan")
        test_mae_original = float(raw.get("mae_paper", float("nan"))) if test_original_ok else float("nan")

        status = "PASS"
        if not np.isfinite(rmse):
            status = "FAIL"

        record.update(
            {
                "source_df_rows": source_df_rows,
                "source_df_entity_id_unique": source_df_entity_id_unique,
                "scenario": information_sharing,
                "horizon": int(raw.get("horizon", horizon)),
                "source_selection_window": source_selection_window_cfg,
                "selected_sources": selected_sources,
                "source_identification": source_identification,
                "source_audit_notes": source_audit_notes,
                "rfe_candidate_features": rfe_candidate_features,
                "rfe_selected_features": rfe_selected_features,
                "final_selected_features": final_selected_features,
                "test_rmse": rmse,
                "accuracy": accuracy,
                "mae": mae,
                "metric_space": metric_space_used,
                "metric_space_used": metric_space_used,
                "metric_space_current": metric_space_current,
                "metric_space_paper": metric_space_paper,
                "paper_metric_aligned": paper_metric_aligned,
                "inverse_transform_applied": inverse_transform_applied,
                "metric_notes": metric_notes,
                "val_metric_notes": val_metric_notes,
                "rmse_normalized": rmse_normalized,
                "accuracy_normalized": accuracy_normalized,
                "mae_normalized": mae_normalized,
                "rmse_original": test_rmse_original,
                "accuracy_original": float(raw.get("accuracy_paper", float("nan"))) if test_original_ok else float("nan"),
                "mae_original": test_mae_original,
                "val_rmse_normalized": val_rmse_normalized,
                "test_rmse_normalized": test_rmse_normalized,
                "test_mae_normalized": test_mae_normalized,
                "val_rmse_original": val_rmse_original,
                "test_rmse_original": test_rmse_original,
                "test_mae_original": test_mae_original,
                "random_seed": int(seed),
                "status": status,
            }
        )
    except Exception as exc:
        record["status"] = "FAIL"
        record["error_message"] = str(exc)

    record["run_time_seconds"] = float(time.time() - start)
    return record


def _write_horizon_summary(full_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    summary_path = output_dir / "final_msml_tl_rfe_observed_k3_horizon_summary.csv"
    if full_df.empty:
        summary_df = pd.DataFrame()
        summary_df.to_csv(summary_path, index=False)
        return summary_df

    grouped = (
        full_df.groupby(["dataset_id", "scenario", "horizon"], dropna=False)
        .agg(
            n_runs=("status", "size"),
            n_pass=("status", lambda s: int((s == "PASS").sum())),
            mean_rmse_normalized=("rmse_normalized", "mean"),
            std_rmse_normalized=("rmse_normalized", "std"),
            mean_accuracy_normalized=("accuracy_normalized", "mean"),
            mean_rmse_original=("rmse_original", "mean"),
        )
        .reset_index()
    )
    grouped["summary_level"] = "dataset_scenario_horizon"
    grouped["mean_rmse_normalized_over_horizons_unconfirmed_protocol"] = np.nan

    over_horizons = (
        full_df.groupby(["dataset_id", "scenario"], dropna=False)
        .agg(
            n_runs=("status", "size"),
            n_pass=("status", lambda s: int((s == "PASS").sum())),
            mean_rmse_normalized=("rmse_normalized", "mean"),
            std_rmse_normalized=("rmse_normalized", "std"),
            mean_accuracy_normalized=("accuracy_normalized", "mean"),
            mean_rmse_original=("rmse_original", "mean"),
        )
        .reset_index()
    )
    over_horizons["horizon"] = "ALL"
    over_horizons["summary_level"] = "dataset_scenario_over_horizons_unconfirmed_protocol"
    over_horizons["mean_rmse_normalized_over_horizons_unconfirmed_protocol"] = over_horizons[
        "mean_rmse_normalized"
    ]

    summary_df = pd.concat([grouped, over_horizons], axis=0, ignore_index=True)
    summary_df = summary_df[
        [
            "dataset_id",
            "scenario",
            "horizon",
            "summary_level",
            "n_runs",
            "n_pass",
            "mean_rmse_normalized",
            "std_rmse_normalized",
            "mean_accuracy_normalized",
            "mean_rmse_original",
            "mean_rmse_normalized_over_horizons_unconfirmed_protocol",
        ]
    ]
    summary_df.to_csv(summary_path, index=False)
    return summary_df


def _write_horizon_audit(full_df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path) -> Path:
    audit_path = output_dir / "final_msml_tl_rfe_observed_k3_horizon_audit.md"
    expected_horizons = {1, 2, 3, 4, 5}
    observed_horizons = set(int(h) for h in full_df["horizon"].dropna().unique()) if "horizon" in full_df else set()
    total_rows = int(len(full_df))
    all_pass = bool(total_rows > 0 and (full_df["status"] == "PASS").all())
    has_todo_region = bool(
        full_df.astype(str).apply(lambda col: col.str.contains("TODO_REGION_UNAVAILABLE", na=False)).any().any()
    ) if not full_df.empty else False

    per_horizon = full_df.groupby("horizon", dropna=False).size().to_dict() if "horizon" in full_df else {}
    scenario_pool_diff = "UNKNOWN"
    if {"dataset_id", "scenario", "source_df_rows", "source_df_entity_id_unique"}.issubset(full_df.columns):
        pool_cols = ["source_df_rows", "source_df_entity_id_unique"]
        diffs: List[str] = []
        for dataset_id, dataset_df in full_df.groupby("dataset_id"):
            scenario_stats = dataset_df.groupby("scenario")[pool_cols].first()
            if {"with_information_sharing", "without_information_sharing"}.issubset(set(scenario_stats.index)):
                changed = not scenario_stats.loc["with_information_sharing"].equals(
                    scenario_stats.loc["without_information_sharing"]
                )
                diffs.append(f"{dataset_id}: {'different' if changed else 'same'}")
        if diffs:
            scenario_pool_diff = "; ".join(diffs)

    mean_accuracy = float(full_df["accuracy_normalized"].mean()) if "accuracy_normalized" in full_df else float("nan")
    mean_rmse = float(full_df["rmse_normalized"].mean()) if "rmse_normalized" in full_df else float("nan")
    close_to_paper = bool(
        np.isfinite(mean_accuracy)
        and np.isfinite(mean_rmse)
        and abs(mean_accuracy - 5.67) <= 0.5
        and 0.18 <= mean_rmse <= 0.19
    )

    final_status = "PARTIAL"
    if total_rows != 150 or observed_horizons != expected_horizons or not all_pass or has_todo_region:
        final_status = "FAIL"

    rows = [
        "# MSML-TL-RFE Observed k=3 Horizon Audit",
        "",
        f"- Horizon coverage: {sorted(observed_horizons)}",
        f"- Covers horizon=1..5: {observed_horizons == expected_horizons}",
        f"- Rows per horizon: {dict(sorted(per_horizon.items()))}",
        f"- Total rows: {total_rows}",
        f"- Expected 150 rows: {total_rows == 150}",
        f"- All PASS: {all_pass}",
        f"- Source pool with/without still different: {scenario_pool_diff}",
        f"- Contains TODO_REGION_UNAVAILABLE: {has_todo_region}",
        f"- Mean normalized accuracy vs paper 5.67: {mean_accuracy:.6g}",
        f"- Mean normalized RMSE vs paper 0.18-0.19: {mean_rmse:.6g}",
        f"- Close to paper 5.67 accuracy / 0.18-0.19 RMSE: {close_to_paper}",
        "- Horizon aggregation formula: UNCERTAIN; cross-horizon means are marked unconfirmed protocol in summary.",
        f"- Final judgment: {final_status} / UNCERTAIN",
    ]
    audit_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return audit_path


def _write_source_identification_outputs(
    source_rows: Sequence[Dict[str, Any]],
    output_dir: Path,
    source_count: int,
) -> Dict[str, Path | None]:
    if not source_rows:
        return {"csv": None, "audit_md": None}

    source_df = pd.DataFrame(source_rows)
    paths: Dict[str, Path | None] = {"csv": None, "audit_md": None}
    grouped_cols = ["dataset", "method", "information_sharing"]

    for (dataset_name, method_name, scenario), group_df in source_df.groupby(grouped_cols, dropna=False):
        if dataset_name == "Dataset3" and method_name == "MSML-TL-RFE":
            dataset_slug = str(dataset_name).lower()
            method_slug = str(method_name).lower().replace("-", "_")
            csv_path = output_dir / f"{dataset_slug}_{method_slug}_k{int(source_count)}_{scenario}_sources.csv"
            group_df.to_csv(csv_path, index=False)
            paths["csv"] = csv_path

            missing_notes = []
            if "same_category_audit_note" in group_df.columns:
                missing_notes = [
                    str(v)
                    for v in group_df["same_category_audit_note"].dropna().tolist()
                    if str(v).strip()
                ]
            all_same_category = (
                bool(group_df["same_category_pass"].fillna(False).all())
                if "same_category_pass" in group_df.columns
                else False
            )
            target_store_type = (
                str(group_df["target_store_type"].dropna().iloc[0])
                if "target_store_type" in group_df.columns and not group_df["target_store_type"].dropna().empty
                else ""
            )
            source_store_types = (
                group_df["source_store_type"].dropna().astype(str).tolist()
                if "source_store_type" in group_df.columns
                else []
            )
            audit_rows = [
                "# Dataset3 Same-Category Source Audit",
                "",
                f"- dataset: {dataset_name}",
                f"- method: {method_name}",
                f"- information_sharing: {scenario}",
                f"- source_count: {int(source_count)}",
                f"- target_store_id: {group_df['target_store_id'].dropna().iloc[0] if 'target_store_id' in group_df.columns and not group_df['target_store_id'].dropna().empty else ''}",
                f"- target_store_type: {target_store_type}",
                f"- top_source_store_types: {source_store_types}",
                f"- same_category_pass_all: {all_same_category}",
                f"- missing_or_parse_notes_count: {len(missing_notes)}",
            ]
            if missing_notes:
                audit_rows.append("")
                audit_rows.append("## Missing Or Parse Notes")
                for note in missing_notes:
                    audit_rows.append(f"- {note}")
            audit_rows.append("")
            audit_rows.append("## Selected Sources")
            audit_rows.append("")
            display_cols = [
                "source_rank",
                "source_key",
                "source_store_id",
                "source_store_type",
                "target_store_id",
                "target_store_type",
                "same_category_pass",
            ]
            existing_display_cols = [c for c in display_cols if c in group_df.columns]
            audit_rows.append("| " + " | ".join(existing_display_cols) + " |")
            audit_rows.append("| " + " | ".join(["---"] * len(existing_display_cols)) + " |")
            for _, display_row in group_df[existing_display_cols].iterrows():
                audit_rows.append(
                    "| "
                    + " | ".join("" if pd.isna(display_row[col]) else str(display_row[col]) for col in existing_display_cols)
                    + " |"
                )
            audit_path = output_dir / f"{dataset_slug}_{method_slug}_k{int(source_count)}_{scenario}_same_category_audit.md"
            audit_path.write_text("\n".join(audit_rows) + "\n", encoding="utf-8")
            paths["audit_md"] = audit_path

    return paths


def _format_source_selection(selected_sources: Sequence[Dict[str, Any]]) -> str:
    if not selected_sources:
        return "[]"
    return json.dumps(selected_sources, ensure_ascii=True, indent=2)


def _audit_source_selection_split_safety(
    datasets: Sequence[str],
    source_count: int,
    information_sharing: str,
    cfg: Dict[str, Any],
    output_dir: Path,
) -> Path:
    rows: List[str] = []
    protocol = load_paper_protocol(cfg)
    strict_paper_mode = resolve_strict_paper_mode(cfg, explicit=None)

    for dataset_name in datasets:
        base = prepare_base_data_for_experiments(
            dataset_name=dataset_name,
            data_path=cfg["dataset_paths"][dataset_name],
            config=cfg,
            verbose_mode="summary",
        )
        source_df = base["source_df"]
        target_df = base["target_df"].copy()

        feature_cols = _resolve_dataset_feature_cols(
            dataset_name=dataset_name,
            source_df=source_df,
            target_df=target_df,
            cfg=cfg,
        )

        use_information_sharing = _scenario_to_bool(information_sharing)
        source_df = _apply_information_sharing_filter(
            dataset_name=dataset_name,
            source_df=source_df,
            target_df=target_df,
            use_information_sharing=use_information_sharing,
            strict_paper_mode=bool(strict_paper_mode),
            protocol=protocol,
            cfg=cfg,
        )

        selector = SourceSelector()
        selection_full = selector.select_top_k_sources(
            target_df=target_df,
            source_df=source_df,
            feature_cols=feature_cols,
            k=int(source_count),
            weight_mode=str(cfg["single_experiment"]["weight_mode"]),
            include_sales_in_knn=True,
        )
        selected_full = selection_full.get("sources", []) if isinstance(selection_full, dict) else []

        target_train_df, target_val_df, _ = temporal_split_by_ratio_or_dates(target_df)
        observed_df = pd.concat([target_train_df, target_val_df], axis=0, ignore_index=True)
        observed_df.attrs = target_df.attrs.copy()

        selection_observed = selector.select_top_k_sources(
            target_df=observed_df,
            source_df=source_df,
            feature_cols=feature_cols,
            k=int(source_count),
            weight_mode=str(cfg["single_experiment"]["weight_mode"]),
            include_sales_in_knn=True,
        )
        selected_observed = selection_observed.get("sources", []) if isinstance(selection_observed, dict) else []

        changed = selected_full != selected_observed
        rows.append(f"## {dataset_name}")
        rows.append("")
        rows.append(f"- scenario: {information_sharing}")
        rows.append(f"- source_count: {int(source_count)}")
        rows.append(f"- selection_changed: {str(changed)}")
        rows.append("")
        rows.append("Full target window selected_sources:")
        rows.append("```json")
        rows.append(_format_source_selection(selected_full))
        rows.append("```")
        rows.append("")
        rows.append("Observed (train+val) window selected_sources:")
        rows.append("```json")
        rows.append(_format_source_selection(selected_observed))
        rows.append("```")
        rows.append("")

    audit_path = output_dir / "source_selection_split_safety_audit.md"
    audit_path.write_text("\n".join(rows), encoding="utf-8")
    return audit_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate MSML-TL-RFE leakage fix.")
    parser.add_argument(
        "--mode",
        choices=["rfe", "full", "both"],
        default="rfe",
        help="Validation mode: rfe-only, full training, or both.",
    )
    parser.add_argument(
        "--datasets",
        default="Dataset1,Dataset2,Dataset3",
        help="Comma-separated dataset names.",
    )
    parser.add_argument(
        "--seeds",
        default="42-46",
        help="Seed list like 42-46 or 42,43,44.",
    )
    parser.add_argument(
        "--horizons",
        default="1",
        help="Comma-separated forecast horizons, for example 1,2,3,4,5.",
    )
    parser.add_argument(
        "--scenario",
        choices=["without_information_sharing", "with_information_sharing"],
        default="without_information_sharing",
        help="Information sharing scenario.",
    )
    parser.add_argument(
        "--scenarios",
        help="Comma-separated scenarios. Overrides --scenario when provided.",
    )
    parser.add_argument(
        "--source-count", type=int, default=3, help="Number of sources (k)."
    )
    parser.add_argument(
        "--source-selection-window",
        choices=[
            "full_history",
            "train_window",
            "target_train_window",
            "observed_window",
            "target_observed_window",
            "train_val_window",
        ],
        help="Override source_selection_window for source selection (overrides config).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "paper_alignment"),
        help="Output directory for validation artifacts.",
    )
    parser.add_argument(
        "--strict-paper-mode",
        action="store_true",
        help="Enable strict paper mode (split + metric strictness).",
    )
    parser.add_argument(
        "--audit-source-selection",
        action="store_true",
        help="Generate a split-safety audit comparing full vs observed target windows.",
    )
    args = parser.parse_args()

    cfg = _load_config()
    # Allow CLI to override configured source_selection_window
    if getattr(args, "source_selection_window", None):
        cfg.setdefault("paper_reproduction", {}).setdefault("paper_split_protocol", {})[
            "source_selection_window"
        ] = args.source_selection_window
    if args.strict_paper_mode:
        cfg.setdefault("paper_reproduction", {})["strict_paper_mode"] = True
        cfg["paper_reproduction"]["paper_strict_mode"] = True
        cfg["paper_reproduction"].setdefault("metric_protocol", {})["strict_paper_metrics"] = True
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    seeds = _parse_seed_list(args.seeds)
    horizons = _parse_int_list(args.horizons)
    if args.scenarios:
        scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    else:
        scenarios = [args.scenario]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.audit_source_selection:
        _audit_source_selection_split_safety(
            datasets=datasets,
            source_count=int(args.source_count),
            information_sharing=args.scenario,
            cfg=cfg,
            output_dir=output_dir,
        )

    if args.mode in {"rfe", "both"}:
        rfe_rows: List[Dict[str, Any]] = []
        for dataset_name in datasets:
            for seed in seeds:
                rfe_rows.append(
                    _run_rfe_only(
                        dataset_name=dataset_name,
                        seed=seed,
                        source_count=int(args.source_count),
                        information_sharing=args.scenario,
                        cfg=cfg,
                    )
                )
        rfe_df = pd.DataFrame(rfe_rows)
        rfe_df.to_csv(output_dir / "msml_tl_rfe_rfe_only_validation.csv", index=False)
        rfe_df.to_json(output_dir / "msml_tl_rfe_rfe_only_validation.json", orient="records", indent=2)

    if args.mode in {"full", "both"}:
        full_rows: List[Dict[str, Any]] = []
        source_rows: List[Dict[str, Any]] = []
        for dataset_name in datasets:
            for scenario in scenarios:
                for seed in seeds:
                    for horizon in horizons:
                        full_record = _run_full_training(
                            dataset_name=dataset_name,
                            seed=seed,
                            source_count=int(args.source_count),
                            information_sharing=scenario,
                            horizon=int(horizon),
                            cfg=cfg,
                        )
                        full_rows.append(full_record)
                        for source_row in full_record.get("source_identification", []) or []:
                            source_rows.append(
                                {
                                    **source_row,
                                    "seed": int(seed),
                                    "horizon": int(horizon),
                                }
                            )
        full_df = pd.DataFrame(full_rows)
        full_df.to_csv(output_dir / "msml_tl_rfe_full_validation.csv", index=False)
        full_df.to_json(output_dir / "msml_tl_rfe_full_validation.json", orient="records", indent=2)
        full_df.to_csv(
            output_dir / "final_msml_tl_rfe_observed_k3_horizon_results.csv",
            index=False,
        )
        summary_df = _write_horizon_summary(full_df, output_dir)
        _write_horizon_audit(full_df, summary_df, output_dir)
        _write_source_identification_outputs(
            source_rows=source_rows,
            output_dir=output_dir,
            source_count=int(args.source_count),
        )


if __name__ == "__main__":
    main()
