"""
Dataset2 source-pool x information-sharing x seed matrix audit.

Matrix:
- source_pool: B123, B1234
- information_sharing: no-sharing, with-sharing
- seed: 42, 43, 44, 45, 46
- method: No-TL, SS-TL, MSWA-TL, MSSB-TL, MSML-TL, MSML-TL-RFE

The script is standalone and reuses the existing preprocessing, KNN, RFE,
training, and metric code. For no-sharing, it applies the project's current
Dataset2 no-sharing rule after constructing the requested source pool: only
same-brand B1 Item1-9 sources remain for target B1 Item10.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_preprocessing import build_source_target_split, extract_datetime_features, load_dataset
from environment import setup_reproducibility
from experiment_runner import (
    run_msml_experiment,
    run_msml_rfe_experiment,
    run_mssb_experiment,
    run_mswa_experiment,
    run_no_tl_experiment,
    run_ss_tl_experiment,
)
from scripts.run_full_paper_experiments import (
    _build_observed_target_window,
    _load_config,
    _resolve_dataset_feature_cols,
)
from source_selector import SourceSelector


DATASET = "Dataset2"
TARGET_ENTITY = "B1"
TARGET_ITEM = 10
SOURCE_ITEMS = list(range(1, 10))
SOURCE_POOLS = {
    "B123": ["B1", "B2", "B3"],
    "B1234": ["B1", "B2", "B3", "B4"],
}
INFORMATION_SHARING_SCENARIOS = ["no-sharing", "with-sharing"]
SEEDS = [42, 43, 44, 45, 46]
METHODS = ["No-TL", "SS-TL", "MSWA-TL", "MSSB-TL", "MSML-TL", "MSML-TL-RFE"]
PAPER_TABLE6_TOP3 = [("B1", 4), ("B2", 3), ("B3", 2)]
EPS = 1e-8

OUT_DIR = ROOT / "outputs" / "audits"
ALL_METHODS_CSV = OUT_DIR / "dataset2_pool_sharing_seed_matrix_all_methods.csv"
KNN_TOP10_CSV = OUT_DIR / "dataset2_pool_sharing_seed_matrix_knn_top10.csv"
METHOD_COMPARE_CSV = OUT_DIR / "dataset2_pool_sharing_seed_matrix_compare.csv"
SUMMARY_CSV = OUT_DIR / "dataset2_pool_sharing_seed_matrix_summary.csv"
REPORT_MD = OUT_DIR / "dataset2_pool_sharing_seed_matrix.md"


def configure_logging() -> None:
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger("experiment").setLevel(logging.ERROR)
    logging.disable(logging.WARNING)


def key_tuple(raw: Any) -> Tuple[str, int]:
    if isinstance(raw, (tuple, list)) and len(raw) >= 2:
        return str(raw[0]), int(raw[1])
    if isinstance(raw, str):
        cleaned = raw.strip().strip("()")
        parts = [p.strip().strip("'\"") for p in cleaned.split(",")]
        if len(parts) >= 2:
            return str(parts[0]), int(float(parts[1]))
    raise ValueError(f"Invalid source key: {raw!r}")


def key_label(key: Tuple[str, int]) -> str:
    return f"{key[0]} Item{int(key[1])}"


def join_keys(keys: Sequence[Tuple[str, int]]) -> str:
    return "|".join(key_label(k) for k in keys)


def pipe(values: Iterable[Any]) -> str:
    return "|".join("" if v is None else str(v) for v in values)


def as_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def inverse_weights(distances: Sequence[float]) -> Tuple[List[float], List[float]]:
    raw = [1.0 / (float(d) + EPS) if math.isfinite(float(d)) else float("nan") for d in distances]
    valid = [w for w in raw if math.isfinite(w)]
    total = sum(valid)
    norm = [w / total if math.isfinite(w) and total > 0 else float("nan") for w in raw]
    return raw, norm


def markdown_table(df: pd.DataFrame, cols: Sequence[str] | None = None, max_rows: int | None = None) -> str:
    if cols is not None:
        df = df[list(cols)].copy()
    if max_rows is not None:
        df = df.head(max_rows).copy()
    if df.empty:
        return "_No rows._"
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.6f}")
        else:
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else str(x))
    lines = [
        "| " + " | ".join(display.columns) + " |",
        "| " + " | ".join(["---"] * len(display.columns)) + " |",
    ]
    for _, row in display.iterrows():
        vals = [str(row[col]).replace("|", "\\|") for col in display.columns]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def load_context() -> Dict[str, Any]:
    cfg = copy.deepcopy(_load_config())
    cfg.setdefault("paper_reproduction", {})
    cfg["paper_reproduction"]["strict_paper_mode"] = True
    cfg["paper_reproduction"]["paper_strict_mode"] = True
    protocol = cfg["paper_reproduction"]
    protocol.setdefault("metric_protocol", {})["strict_paper_metrics"] = bool(
        protocol.get("metric_protocol", {}).get("strict_paper_metrics", False)
    )

    raw_df = load_dataset(DATASET, cfg["dataset_paths"][DATASET])
    processed_df = extract_datetime_features(raw_df)
    source_df, target_df = build_source_target_split(processed_df, cfg)
    observed_target_df = _build_observed_target_window(target_df)
    feature_cols = _resolve_dataset_feature_cols(DATASET, source_df, target_df, cfg)
    return {
        "cfg": cfg,
        "protocol": protocol,
        "processed_df": processed_df,
        "target_df": target_df,
        "observed_target_df": observed_target_df,
        "feature_cols": feature_cols,
    }


def build_source_pool(processed_df: pd.DataFrame, source_pool: str, information_sharing: str) -> pd.DataFrame:
    entities = SOURCE_POOLS[source_pool]
    item_values = pd.to_numeric(processed_df["item_id"], errors="coerce")
    mask = processed_df["entity_id"].astype(str).isin(entities) & item_values.isin(SOURCE_ITEMS)
    source_df = processed_df.loc[mask].copy()

    scenario = "with_information_sharing" if information_sharing == "with-sharing" else "without_information_sharing"
    source_df.attrs["dataset_name"] = DATASET
    source_df.attrs["information_sharing_scenario"] = scenario
    source_df.attrs["signature_static_feature_cols"] = ["promo"] if information_sharing == "with-sharing" else []
    source_df.attrs["requested_source_pool"] = source_pool

    if information_sharing == "no-sharing":
        source_df = source_df[source_df["entity_id"].astype(str) == TARGET_ENTITY].copy()
        source_df.attrs["source_pool_scope_mode"] = f"{source_pool}_no_sharing_same_brand_B1"
        source_df.attrs["source_pool_scope_note"] = (
            f"Requested {source_pool}, then current Dataset2 no-sharing rule keeps same-brand B1 only."
        )
    else:
        source_df.attrs["source_pool_scope_mode"] = f"{source_pool}_with_sharing"
        source_df.attrs["source_pool_scope_note"] = f"Requested {source_pool} Item1-9 with cross-brand sharing."

    return source_df.sort_values(["entity_id", "item_id", "date"]).reset_index(drop=True)


def source_group_count(source_df: pd.DataFrame) -> int:
    return int(source_df.groupby(["entity_id", "item_id"], sort=False).ngroups)


def select_knn(source_df: pd.DataFrame, target_df: pd.DataFrame, feature_cols: Sequence[str], k: int) -> Dict[str, Any]:
    return SourceSelector().select_top_k_sources(
        target_df=target_df,
        source_df=source_df,
        feature_cols=list(feature_cols),
        k=k,
        group_cols=("entity_id", "item_id"),
        weight_mode="inverse_distance",
        debug_mode=False,
        include_sales_in_knn=True,
        knn_representation="paper_observed_sequence",
    )


def build_knn_top10_rows(
    source_pool: str,
    information_sharing: str,
    source_df: pd.DataFrame,
    observed_target_df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    result = select_knn(source_df, observed_target_df, feature_cols, k=10)
    sources = result.get("sources", [])
    keys = [key_tuple(row["source_key"]) for row in sources]
    distances = [float(row["distance"]) for row in sources]
    raw_weights, normalized = inverse_weights(distances)
    top3 = keys[:3]
    matches_ordered = top3 == PAPER_TABLE6_TOP3
    matches_set = set(top3) == set(PAPER_TABLE6_TOP3)

    rows = []
    for idx, row in enumerate(sources):
        key = keys[idx]
        rows.append(
            {
                "dataset": DATASET,
                "source_pool": source_pool,
                "information_sharing": information_sharing,
                "effective_information_sharing": source_df.attrs.get("information_sharing_scenario", ""),
                "rank": int(row.get("source_rank", idx + 1)),
                "source": key_label(key),
                "source_entity": key[0],
                "source_item": key[1],
                "distance": distances[idx],
                "inverse_distance_raw_weight": raw_weights[idx],
                "inverse_distance_normalized_weight": normalized[idx],
                "is_top3": idx < 3,
                "candidate_pool_size": source_group_count(source_df),
                "knn_representation": result.get("meta", {}).get("knn_representation", ""),
                "knn_feature_cols": pipe(result.get("meta", {}).get("feature_cols", [])),
                "b4_in_top1": bool(keys and keys[0][0] == "B4"),
                "b4_in_top3": any(k[0] == "B4" for k in top3),
                "b4_in_top10": any(k[0] == "B4" for k in keys),
                "matches_paper_table6_ordered": matches_ordered,
                "matches_paper_table6_set": matches_set,
                "source_pool_scope_mode": source_df.attrs.get("source_pool_scope_mode", ""),
                "source_pool_scope_note": source_df.attrs.get("source_pool_scope_note", ""),
            }
        )

    summary = {
        "source_pool": source_pool,
        "information_sharing": information_sharing,
        "top10_keys": keys,
        "top3_keys": top3,
        "top10": join_keys(keys),
        "top3": join_keys(top3),
        "candidate_pool_size": source_group_count(source_df),
        "knn_representation": result.get("meta", {}).get("knn_representation", ""),
        "knn_feature_cols": pipe(result.get("meta", {}).get("feature_cols", [])),
        "b4_in_top1": bool(keys and keys[0][0] == "B4"),
        "b4_in_top3": any(k[0] == "B4" for k in top3),
        "b4_in_top10": any(k[0] == "B4" for k in keys),
        "matches_paper_table6_ordered": matches_ordered,
        "matches_paper_table6_set": matches_set,
        "source_pool_scope_mode": source_df.attrs.get("source_pool_scope_mode", ""),
        "source_pool_scope_note": source_df.attrs.get("source_pool_scope_note", ""),
    }
    return pd.DataFrame(rows), summary


def serialize_selected_sources(selected: Any) -> Tuple[List[Tuple[str, int]], List[float], List[float], List[float]]:
    if not isinstance(selected, list):
        return [], [], [], []
    keys = [key_tuple(row.get("source_key")) for row in selected if isinstance(row, dict)]
    distances = [as_float(row.get("distance")) for row in selected if isinstance(row, dict)]
    raw, norm = inverse_weights(distances) if distances else ([], [])
    weights = [as_float(row.get("weight")) for row in selected if isinstance(row, dict)]
    return keys, distances, raw, weights or norm


def run_one_method(
    method: str,
    source_pool: str,
    information_sharing: str,
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    observed_target_df: pd.DataFrame,
    feature_cols: Sequence[str],
    cfg: Dict[str, Any],
    protocol: Dict[str, Any],
    seed: int,
    knn_summary: Dict[str, Any],
) -> Dict[str, Any]:
    exp_cfg = cfg["single_experiment"]
    setup_reproducibility(seed)
    common = {
        "source_df": source_df,
        "target_df": target_df,
        "feature_cols": list(feature_cols),
        "horizon": int(exp_cfg["horizon"]),
        "window_size": int(exp_cfg["window_size"]),
        "learning_rate": float(exp_cfg.get("learning_rate", 1e-4)),
        "source_epochs": int(exp_cfg["source_epochs"]),
        "target_epochs": int(exp_cfg["target_epochs"]),
        "batch_size": int(exp_cfg["batch_size"]),
        "metric_protocol": protocol.get("metric_protocol", {}),
    }

    if method == "No-TL":
        raw = run_no_tl_experiment(
            target_df=target_df,
            horizon=int(exp_cfg["horizon"]),
            window_size=int(exp_cfg["window_size"]),
            learning_rate=float(exp_cfg.get("learning_rate", 1e-4)),
            target_epochs=int(exp_cfg["target_epochs"]),
            batch_size=int(exp_cfg["batch_size"]),
            metric_protocol=protocol.get("metric_protocol", {}),
        )
    elif method == "SS-TL":
        raw = run_ss_tl_experiment(**common, target_df_for_selection=observed_target_df)
    elif method == "MSWA-TL":
        raw = run_mswa_experiment(**common, target_df_for_selection=observed_target_df, k=3, number_of_sources=3, weight_mode=str(exp_cfg["weight_mode"]))
    elif method == "MSSB-TL":
        raw = run_mssb_experiment(**common, target_df_for_selection=observed_target_df, k=3, number_of_sources=3, weight_mode=str(exp_cfg["weight_mode"]))
    elif method == "MSML-TL":
        raw = run_msml_experiment(**common, target_df_for_selection=observed_target_df, k=3, number_of_sources=3, weight_mode=str(exp_cfg["weight_mode"]))
    elif method == "MSML-TL-RFE":
        raw = run_msml_rfe_experiment(
            **common,
            k=3,
            number_of_sources=3,
            weight_mode=str(exp_cfg["weight_mode"]),
            estimator_name=str(exp_cfg.get("estimator_name", "random_forest")),
            keep_ratio=float(exp_cfg["keep_ratio"]),
            random_state=seed,
            source_selection_window="target_observed_window",
        )
    else:
        raise ValueError(f"Unsupported method: {method}")

    meta = raw.get("meta", {}) if isinstance(raw.get("meta"), dict) else {}
    rfe_info = meta.get("rfe_info", {}) if isinstance(meta.get("rfe_info"), dict) else {}
    selected_raw = meta.get("selected_sources", [])
    selected_keys, selected_distances, selected_raw_weights, selected_norm_weights = serialize_selected_sources(selected_raw)
    selected_features = (
        meta.get("selected_features")
        or meta.get("selected_feature_cols")
        or meta.get("final_selected_features")
        or rfe_info.get("final_selected_features")
        or []
    )
    rfe_selected = (
        meta.get("rfe_selected_features")
        or meta.get("rfe_selected_feature_cols")
        or rfe_info.get("rfe_selected_features")
        or []
    )
    rfe_candidates = (
        meta.get("rfe_candidate_features")
        or rfe_info.get("rfe_candidate_features")
        or rfe_info.get("rfe_candidate_cols")
        or []
    )
    source_selection_info = meta.get("source_selection_info", {}) if isinstance(meta.get("source_selection_info"), dict) else {}
    notes = []
    if method == "No-TL":
        notes.append("No-TL does not use source pool; row duplicated for matrix comparison.")
    if information_sharing == "no-sharing":
        notes.append("No-sharing applies current Dataset2 same-brand rule after requested pool construction.")
    if method == "MSML-TL":
        notes.append("Project method name used directly: MSML-TL.")
    if method == "MSML-TL-RFE":
        notes.append("RFE uses source_selection_window=target_observed_window.")

    return {
        "dataset": DATASET,
        "source_pool": source_pool,
        "information_sharing": information_sharing,
        "effective_information_sharing": source_df.attrs.get("information_sharing_scenario", ""),
        "source_pool_entities": "|".join(SOURCE_POOLS[source_pool]),
        "source_pool_items": "Item1-9",
        "source_pool_scope_mode": source_df.attrs.get("source_pool_scope_mode", ""),
        "source_pool_scope_note": source_df.attrs.get("source_pool_scope_note", ""),
        "candidate_pool_size": source_group_count(source_df),
        "target_entity": TARGET_ENTITY,
        "target_item": f"Item{TARGET_ITEM}",
        "method": method,
        "seed": seed,
        "knn_representation": source_selection_info.get("knn_representation", knn_summary.get("knn_representation", "paper_observed_sequence")),
        "knn_feature_cols": pipe(source_selection_info.get("feature_cols", [])) or knn_summary.get("knn_feature_cols", ""),
        "selected_sources": join_keys(selected_keys) if selected_keys else "NOT_APPLICABLE",
        "selected_source_rankings": pipe(range(1, len(selected_keys) + 1)) if selected_keys else "NOT_APPLICABLE",
        "selected_distances": pipe(selected_distances) if selected_distances else "NOT_APPLICABLE",
        "inverse_distance_raw_weights": pipe(selected_raw_weights) if selected_raw_weights else "NOT_APPLICABLE",
        "inverse_distance_normalized_weights": pipe(selected_norm_weights) if selected_norm_weights else "NOT_APPLICABLE",
        "b4_in_top1": bool(selected_keys and selected_keys[0][0] == "B4"),
        "b4_in_top3": any(k[0] == "B4" for k in selected_keys[:3]),
        "b4_in_top10": knn_summary.get("b4_in_top10", False),
        "rfe_candidate_features": pipe(rfe_candidates),
        "rfe_selected_features": pipe(rfe_selected),
        "final_selected_features": pipe(selected_features),
        "rmse": float(raw.get("rmse", np.nan)),
        "accuracy": float(raw.get("accuracy", np.nan)),
        "normalized_rmse": float(raw.get("normalized_rmse", raw.get("rmse", np.nan))),
        "original_scale_rmse": raw.get("original_scale_rmse", raw.get("rmse_paper")),
        "normalized_accuracy": float(raw.get("normalized_accuracy", raw.get("accuracy", np.nan))),
        "original_scale_accuracy": raw.get("original_scale_accuracy", raw.get("accuracy_paper")),
        "rmse_paper": float(raw.get("rmse_paper", np.nan)),
        "accuracy_paper": float(raw.get("accuracy_paper", np.nan)),
        "matches_paper_table6_ordered": knn_summary.get("matches_paper_table6_ordered", False),
        "matches_paper_table6_set": knn_summary.get("matches_paper_table6_set", False),
        "notes": " ".join(notes),
    }


def build_compare(all_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["information_sharing", "seed", "method"]
    for keys, group in all_df.groupby(group_cols, sort=False):
        information_sharing, seed, method = keys
        if set(group["source_pool"]) != {"B123", "B1234"}:
            continue
        left = group[group["source_pool"] == "B123"].iloc[0]
        right = group[group["source_pool"] == "B1234"].iloc[0]
        rmse_delta = float(right["rmse"]) - float(left["rmse"])
        acc_delta = float(right["accuracy"]) - float(left["accuracy"])
        norm_delta = float(right["normalized_rmse"]) - float(left["normalized_rmse"])
        orig_left = as_float(left["original_scale_rmse"])
        orig_right = as_float(right["original_scale_rmse"])
        orig_delta = orig_right - orig_left if math.isfinite(orig_left) and math.isfinite(orig_right) else float("nan")
        rows.append(
            {
                "information_sharing": information_sharing,
                "seed": int(seed),
                "method": method,
                "rmse_B123": float(left["rmse"]),
                "rmse_B1234": float(right["rmse"]),
                "rmse_delta_B1234_minus_B123": rmse_delta,
                "accuracy_B123": float(left["accuracy"]),
                "accuracy_B1234": float(right["accuracy"]),
                "accuracy_delta_B1234_minus_B123": acc_delta,
                "normalized_rmse_delta_B1234_minus_B123": norm_delta,
                "original_scale_rmse_delta_B1234_minus_B123": orig_delta,
                "effect_after_adding_B4": "unchanged" if abs(rmse_delta) <= 1e-10 else ("improved" if rmse_delta < 0 else "worsened"),
                "selected_sources_B123": left["selected_sources"],
                "selected_sources_B1234": right["selected_sources"],
            }
        )
    return pd.DataFrame(rows)


def build_summary(all_df: pd.DataFrame, compare_df: pd.DataFrame, knn_df: pd.DataFrame) -> pd.DataFrame:
    metric_summary = (
        all_df.groupby(["source_pool", "information_sharing", "method"], as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            normalized_rmse_mean=("normalized_rmse", "mean"),
            original_scale_rmse_mean=("original_scale_rmse", "mean"),
        )
    )
    top3 = (
        knn_df[knn_df["is_top3"]]
        .groupby(["source_pool", "information_sharing"])["source"]
        .apply(lambda s: "|".join(s.astype(str).tolist()))
        .reset_index(name="knn_top3")
    )
    return metric_summary.merge(top3, on=["source_pool", "information_sharing"], how="left")


def write_report(all_df: pd.DataFrame, knn_df: pd.DataFrame, compare_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    best_rows = []
    for (information_sharing, method), group in summary_df.groupby(["information_sharing", "method"], sort=False):
        b123 = group[group["source_pool"] == "B123"].iloc[0]
        b1234 = group[group["source_pool"] == "B1234"].iloc[0]
        delta = float(b1234["rmse_mean"]) - float(b123["rmse_mean"])
        if abs(delta) <= 1e-10:
            best_pool = "tie"
            best_rmse = float(b123["rmse_mean"])
            top3 = str(b123["knn_top3"])
        elif delta < 0:
            best_pool = "B1234"
            best_rmse = float(b1234["rmse_mean"])
            top3 = str(b1234["knn_top3"])
        else:
            best_pool = "B123"
            best_rmse = float(b123["rmse_mean"])
            top3 = str(b123["knn_top3"])
        best_rows.append(
            {
                "information_sharing": information_sharing,
                "method": method,
                "best_source_pool_by_mean_rmse": best_pool,
                "rmse_mean": best_rmse,
                "B1234_minus_B123_rmse_mean": delta,
                "knn_top3": top3,
            }
        )
    best_by_method = pd.DataFrame(best_rows)
    compare_seed_mean = (
        compare_df.groupby(["information_sharing", "method"], as_index=False)
        .agg(
            mean_rmse_delta=("rmse_delta_B1234_minus_B123", "mean"),
            mean_accuracy_delta=("accuracy_delta_B1234_minus_B123", "mean"),
        )
    )
    knn_top3 = (
        knn_df[knn_df["is_top3"]]
        .groupby(["source_pool", "information_sharing"])["source"]
        .apply(lambda s: "|".join(s.astype(str).tolist()))
        .reset_index(name="top3")
    )
    lines = [
        "# Dataset2 Pool x Sharing x Seed Matrix Audit",
        "",
        "## Scope",
        "- Dataset2 only.",
        "- source_pool: B123, B1234.",
        "- information_sharing: no-sharing, with-sharing.",
        "- seed: 42, 43, 44, 45, 46.",
        "- method: No-TL, SS-TL, MSWA-TL, MSSB-TL, MSML-TL, MSML-TL-RFE.",
        "- Target fixed at B1 Item10; source items fixed at Item1-9.",
        "- KNN uses current paper_observed_sequence behavior.",
        "",
        "## KNN Top3",
        markdown_table(knn_top3),
        "",
        "## Mean Results By Seed",
        markdown_table(summary_df, ["source_pool", "information_sharing", "method", "rmse_mean", "rmse_std", "accuracy_mean", "accuracy_std", "knn_top3"]),
        "",
        "## B1234 Minus B123 Mean Delta",
        markdown_table(compare_seed_mean),
        "",
        "## Best Source Pool By Mean RMSE",
        markdown_table(best_by_method),
        "",
        "## Notes",
        "- In no-sharing, the current Dataset2 rule keeps same-brand B1 sources after the requested pool is constructed, so B123 and B1234 have the same effective source candidates.",
        "- No-TL does not use a source pool, but rows are emitted for every matrix cell for comparison.",
        "- MSML-TL is the project method name used directly; any MSADW-TL label is only a display alias elsewhere.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    configure_logging()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ctx = load_context()

    pools: Dict[Tuple[str, str], pd.DataFrame] = {}
    summaries: Dict[Tuple[str, str], Dict[str, Any]] = {}
    knn_frames: List[pd.DataFrame] = []
    for source_pool in SOURCE_POOLS:
        for information_sharing in INFORMATION_SHARING_SCENARIOS:
            source_df = build_source_pool(ctx["processed_df"], source_pool, information_sharing)
            pools[(source_pool, information_sharing)] = source_df
            knn_df, summary = build_knn_top10_rows(
                source_pool,
                information_sharing,
                source_df,
                ctx["observed_target_df"],
                ctx["feature_cols"],
            )
            knn_frames.append(knn_df)
            summaries[(source_pool, information_sharing)] = summary

    no_tl_cache: Dict[int, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    total = len(SOURCE_POOLS) * len(INFORMATION_SHARING_SCENARIOS) * len(SEEDS) * len(METHODS)
    done = 0
    for source_pool in SOURCE_POOLS:
        for information_sharing in INFORMATION_SHARING_SCENARIOS:
            source_df = pools[(source_pool, information_sharing)]
            knn_summary = summaries[(source_pool, information_sharing)]
            for seed in SEEDS:
                for method in METHODS:
                    done += 1
                    print(f"[run {done}/{total}] pool={source_pool} sharing={information_sharing} seed={seed} method={method}")
                    if method == "No-TL" and seed in no_tl_cache:
                        cached = copy.deepcopy(no_tl_cache[seed])
                        cached.update(
                            {
                                "source_pool": source_pool,
                                "information_sharing": information_sharing,
                                "effective_information_sharing": source_df.attrs.get("information_sharing_scenario", ""),
                                "source_pool_entities": "|".join(SOURCE_POOLS[source_pool]),
                                "source_pool_scope_mode": source_df.attrs.get("source_pool_scope_mode", ""),
                                "source_pool_scope_note": source_df.attrs.get("source_pool_scope_note", ""),
                                "candidate_pool_size": source_group_count(source_df),
                                "b4_in_top10": knn_summary.get("b4_in_top10", False),
                                "matches_paper_table6_ordered": knn_summary.get("matches_paper_table6_ordered", False),
                                "matches_paper_table6_set": knn_summary.get("matches_paper_table6_set", False),
                            }
                        )
                        rows.append(cached)
                        continue
                    row = run_one_method(
                        method,
                        source_pool,
                        information_sharing,
                        source_df,
                        ctx["target_df"],
                        ctx["observed_target_df"],
                        ctx["feature_cols"],
                        ctx["cfg"],
                        ctx["protocol"],
                        seed,
                        knn_summary,
                    )
                    if method == "No-TL":
                        no_tl_cache[seed] = copy.deepcopy(row)
                    rows.append(row)

    all_df = pd.DataFrame(rows)
    knn_df = pd.concat(knn_frames, ignore_index=True)
    compare_df = build_compare(all_df)
    summary_df = build_summary(all_df, compare_df, knn_df)

    all_df.to_csv(ALL_METHODS_CSV, index=False, encoding="utf-8")
    knn_df.to_csv(KNN_TOP10_CSV, index=False, encoding="utf-8")
    compare_df.to_csv(METHOD_COMPARE_CSV, index=False, encoding="utf-8")
    summary_df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8")
    write_report(all_df, knn_df, compare_df, summary_df)

    print("\nOutput files:")
    for path in [ALL_METHODS_CSV, KNN_TOP10_CSV, METHOD_COMPARE_CSV, SUMMARY_CSV, REPORT_MD]:
        print(f"- {path}")
    print("KNN Top3:")
    for _, row in knn_df[knn_df["is_top3"]].groupby(["source_pool", "information_sharing"])["source"].apply(lambda s: "|".join(s.tolist())).reset_index(name="top3").iterrows():
        print(f"- {row['source_pool']} {row['information_sharing']}: {row['top3']}")
    print("Mean RMSE by source_pool/information_sharing/method:")
    for _, row in summary_df.iterrows():
        print(f"- {row['source_pool']} {row['information_sharing']} {row['method']}: mean_rmse={row['rmse_mean']:.6f}")


if __name__ == "__main__":
    main()
