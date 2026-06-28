"""
Dataset2 with-sharing B4 source-pool sensitivity audit.

This script is intentionally standalone: it does not modify the main
experiment defaults, cleaning, KNN distance formula, RFE logic, or metrics.
It reuses the current Dataset2 preprocessing path, paper_observed_sequence KNN,
and existing method runners, while swapping only the audited source pool.
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
    _apply_information_sharing_filter,
    _build_observed_target_window,
    _load_config,
    _resolve_dataset_feature_cols,
    _resolve_experiment_seed,
)
from source_selector import SourceSelector


DATASET = "Dataset2"
INFORMATION_SHARING = "with_information_sharing"
TARGET_ENTITY = "B1"
TARGET_ITEM = 10
SOURCE_ITEMS = list(range(1, 10))
METHODS = ["No-TL", "SS-TL", "MSWA-TL", "MSSB-TL", "MSML-TL", "MSML-TL-RFE"]
SOURCE_POOLS = {
    "d2_b123": ["B1", "B2", "B3"],
    "d2_b1234": ["B1", "B2", "B3", "B4"],
}
PAPER_TABLE6_TOP3 = [("B1", 4), ("B2", 3), ("B3", 2)]
OUT_DIR = ROOT / "outputs" / "audits"
ALL_METHODS_CSV = OUT_DIR / "dataset2_b123_vs_b1234_withsharing_all_methods.csv"
KNN_TOP10_CSV = OUT_DIR / "dataset2_b123_vs_b1234_withsharing_knn_top10.csv"
METHOD_COMPARE_CSV = OUT_DIR / "dataset2_b123_vs_b1234_withsharing_method_compare.csv"
REPORT_MD = OUT_DIR / "dataset2_b123_vs_b1234_withsharing_all_methods.md"
EPS = 1e-8


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


def jsonish(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def as_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def markdown_table(df: pd.DataFrame, cols: Sequence[str] | None = None) -> str:
    if cols is not None:
        df = df[list(cols)].copy()
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
    base_source_df = _apply_information_sharing_filter(
        dataset_name=DATASET,
        source_df=source_df,
        target_df=target_df,
        use_information_sharing=True,
        strict_paper_mode=True,
        protocol=protocol,
        cfg=cfg,
    )
    observed_target_df = _build_observed_target_window(target_df)
    feature_cols = _resolve_dataset_feature_cols(DATASET, base_source_df, target_df, cfg)
    seed = _resolve_experiment_seed(cfg)
    exp_cfg = cfg["single_experiment"]

    return {
        "cfg": cfg,
        "protocol": protocol,
        "processed_df": processed_df,
        "base_source_df": base_source_df,
        "target_df": target_df,
        "observed_target_df": observed_target_df,
        "feature_cols": feature_cols,
        "seed": seed,
        "exp_cfg": exp_cfg,
    }


def build_source_pool(processed_df: pd.DataFrame, base_source_df: pd.DataFrame, setting: str) -> pd.DataFrame:
    entities = SOURCE_POOLS[setting]
    item_values = pd.to_numeric(processed_df["item_id"], errors="coerce")
    mask = processed_df["entity_id"].astype(str).isin(entities) & item_values.isin(SOURCE_ITEMS)
    source_df = processed_df.loc[mask].copy()
    source_df.attrs = base_source_df.attrs.copy()
    source_df.attrs["information_sharing_scenario"] = INFORMATION_SHARING
    source_df.attrs["source_pool_scope_mode"] = setting
    source_df.attrs["source_pool_scope_note"] = f"{'/'.join(entities)} Item1-9 sensitivity source pool"
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


def inverse_weights(distances: Sequence[float]) -> Tuple[List[float], List[float]]:
    raw = [1.0 / (float(d) + EPS) if math.isfinite(float(d)) else float("nan") for d in distances]
    valid = [w for w in raw if math.isfinite(w)]
    total = sum(valid)
    norm = [w / total if math.isfinite(w) and total > 0 else float("nan") for w in raw]
    return raw, norm


def build_knn_top10_rows(setting: str, source_df: pd.DataFrame, observed_target_df: pd.DataFrame, feature_cols: Sequence[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
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
                "information_sharing": INFORMATION_SHARING,
                "source_pool_setting": setting,
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
                "matches_paper_table6_ordered": matches_ordered,
                "matches_paper_table6_set": matches_set,
            }
        )

    summary = {
        "setting": setting,
        "top10_keys": keys,
        "top3_keys": top3,
        "top10": join_keys(keys),
        "top3": join_keys(top3),
        "candidate_pool_size": source_group_count(source_df),
        "b4_in_top1": bool(keys and keys[0][0] == "B4"),
        "b4_in_top3": any(k[0] == "B4" for k in top3),
        "b4_in_top10": any(k[0] == "B4" for k in keys),
        "matches_paper_table6_ordered": matches_ordered,
        "matches_paper_table6_set": matches_set,
        "knn_representation": result.get("meta", {}).get("knn_representation", ""),
        "knn_feature_cols": pipe(result.get("meta", {}).get("feature_cols", [])),
        "top10_sources": sources,
        "top10_distances": distances,
        "top10_raw_weights": raw_weights,
        "top10_normalized_weights": normalized,
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


def method_notes(method: str) -> str:
    if method == "MSML-TL":
        return "Project method name used directly: MSML-TL."
    if method == "MSML-TL-RFE":
        return "RFE runner uses source_selection_window=target_observed_window."
    if method == "No-TL":
        return "No-TL does not use source pool; duplicated under both pool settings for comparison."
    return ""


def run_one_method(
    method: str,
    setting: str,
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    observed_target_df: pd.DataFrame,
    feature_cols: Sequence[str],
    cfg: Dict[str, Any],
    protocol: Dict[str, Any],
    seed: int,
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
        raw = run_mswa_experiment(
            **common,
            target_df_for_selection=observed_target_df,
            k=3,
            number_of_sources=3,
            weight_mode=str(exp_cfg["weight_mode"]),
        )
    elif method == "MSSB-TL":
        raw = run_mssb_experiment(
            **common,
            target_df_for_selection=observed_target_df,
            k=3,
            number_of_sources=3,
            weight_mode=str(exp_cfg["weight_mode"]),
        )
    elif method == "MSML-TL":
        raw = run_msml_experiment(
            **common,
            target_df_for_selection=observed_target_df,
            k=3,
            number_of_sources=3,
            weight_mode=str(exp_cfg["weight_mode"]),
        )
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

    return {
        "dataset": DATASET,
        "information_sharing": INFORMATION_SHARING,
        "source_pool_setting": setting,
        "source_pool_entities": "|".join(SOURCE_POOLS[setting]),
        "source_pool_items": "Item1-9",
        "candidate_pool_size": source_group_count(source_df),
        "target_entity": TARGET_ENTITY,
        "target_item": f"Item{TARGET_ITEM}",
        "method": method,
        "seed": seed,
        "knn_representation": source_selection_info.get("knn_representation", "paper_observed_sequence"),
        "knn_feature_cols": pipe(source_selection_info.get("feature_cols", feature_cols)),
        "selected_sources": join_keys(selected_keys) if selected_keys else "NOT_APPLICABLE",
        "selected_source_rankings": pipe(range(1, len(selected_keys) + 1)) if selected_keys else "NOT_APPLICABLE",
        "selected_distances": pipe(selected_distances) if selected_distances else "NOT_APPLICABLE",
        "inverse_distance_raw_weights": pipe(selected_raw_weights) if selected_raw_weights else "NOT_APPLICABLE",
        "inverse_distance_normalized_weights": pipe(selected_norm_weights) if selected_norm_weights else "NOT_APPLICABLE",
        "b4_in_top1": bool(selected_keys and selected_keys[0][0] == "B4"),
        "b4_in_top3": any(k[0] == "B4" for k in selected_keys[:3]),
        "b4_in_top10": "",
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
        "matches_paper_table6_ordered": "",
        "matches_paper_table6_set": "",
        "notes": method_notes(method),
    }


def attach_knn_summary_to_method_rows(rows_df: pd.DataFrame, summaries: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    out = rows_df.copy()
    for idx, row in out.iterrows():
        summary = summaries[str(row["source_pool_setting"])]
        out.at[idx, "b4_in_top10"] = summary["b4_in_top10"]
        out.at[idx, "matches_paper_table6_ordered"] = summary["matches_paper_table6_ordered"]
        out.at[idx, "matches_paper_table6_set"] = summary["matches_paper_table6_set"]
        if row["method"] == "No-TL":
            out.at[idx, "knn_feature_cols"] = summary["knn_feature_cols"]
            out.at[idx, "knn_representation"] = summary["knn_representation"]
    return out


def build_method_compare(all_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        left = all_df[(all_df["method"] == method) & (all_df["source_pool_setting"] == "d2_b123")].iloc[0]
        right = all_df[(all_df["method"] == method) & (all_df["source_pool_setting"] == "d2_b1234")].iloc[0]
        rmse_delta = float(right["rmse"]) - float(left["rmse"])
        acc_delta = float(right["accuracy"]) - float(left["accuracy"])
        norm_rmse_delta = float(right["normalized_rmse"]) - float(left["normalized_rmse"])
        orig_left = as_float(left["original_scale_rmse"])
        orig_right = as_float(right["original_scale_rmse"])
        orig_delta = orig_right - orig_left if math.isfinite(orig_left) and math.isfinite(orig_right) else float("nan")
        if abs(rmse_delta) <= 1e-10:
            effect = "unchanged"
        elif rmse_delta < 0:
            effect = "improved"
        else:
            effect = "worsened"
        rows.append(
            {
                "method": method,
                "rmse_b123": float(left["rmse"]),
                "rmse_b1234": float(right["rmse"]),
                "rmse_delta": rmse_delta,
                "accuracy_b123": float(left["accuracy"]),
                "accuracy_b1234": float(right["accuracy"]),
                "accuracy_delta": acc_delta,
                "normalized_rmse_delta": norm_rmse_delta,
                "original_scale_rmse_delta": orig_delta,
                "effect_after_adding_b4": effect,
                "selected_sources_b123": left["selected_sources"],
                "selected_sources_b1234": right["selected_sources"],
            }
        )
    return pd.DataFrame(rows)


def write_report(all_df: pd.DataFrame, knn_df: pd.DataFrame, compare_df: pd.DataFrame, summaries: Dict[str, Dict[str, Any]]) -> None:
    top3_changed = summaries["d2_b123"]["top3"] != summaries["d2_b1234"]["top3"]
    b4_enters_top3 = summaries["d2_b1234"]["b4_in_top3"]
    b4_enters_top10 = summaries["d2_b1234"]["b4_in_top10"]
    b123_matches = summaries["d2_b123"]["matches_paper_table6_ordered"]
    b1234_matches = summaries["d2_b1234"]["matches_paper_table6_ordered"]

    compare_abs = compare_df.copy()
    compare_abs["abs_rmse_delta"] = compare_abs["rmse_delta"].abs()
    largest = compare_abs.sort_values("abs_rmse_delta", ascending=False).iloc[0]
    better_pool = (
        "d2_b1234"
        if float(compare_df["rmse_b1234"].mean()) < float(compare_df["rmse_b123"].mean())
        else "d2_b123"
        if float(compare_df["rmse_b123"].mean()) < float(compare_df["rmse_b1234"].mean())
        else "tie"
    )
    overlap_b123 = len(set(summaries["d2_b123"]["top3_keys"]) & set(PAPER_TABLE6_TOP3))
    overlap_b1234 = len(set(summaries["d2_b1234"]["top3_keys"]) & set(PAPER_TABLE6_TOP3))
    if b123_matches and not b1234_matches:
        closer_pool = "d2_b123"
    elif b1234_matches and not b123_matches:
        closer_pool = "d2_b1234"
    elif overlap_b123 > overlap_b1234:
        closer_pool = "d2_b123"
    elif overlap_b1234 > overlap_b123:
        closer_pool = "d2_b1234"
    else:
        closer_pool = "neither/tie"

    lines = [
        "# Dataset2 B123 vs B1234 With-Sharing Full-Method Audit",
        "",
        "## 1. 实验目的",
        "比较 Dataset2 在 with_information_sharing=True、target=B1 Item10、observed-window KNN 下，source pool 排除或包含 B4 时对 KNN 选源和六种模型结果的影响。",
        "",
        "## 2. 实验设置",
        f"- Dataset: {DATASET}",
        f"- Information sharing: {INFORMATION_SHARING}",
        f"- Target: {TARGET_ENTITY} Item{TARGET_ITEM}",
        f"- Source pools: d2_b123 = B1/B2/B3 Item1-9; d2_b1234 = B1/B2/B3/B4 Item1-9",
        f"- Seed: {all_df['seed'].iloc[0]}",
        f"- KNN representation: paper_observed_sequence",
        f"- KNN feature columns: {summaries['d2_b123']['knn_feature_cols']}",
        "- Methods: No-TL, SS-TL, MSWA-TL, MSSB-TL, MSML-TL, MSML-TL-RFE",
        "- No-TL 不使用 source pool，但在两个 setting 下各输出一行。",
        "",
        "## 3. KNN源池对比",
        markdown_table(
            knn_df,
            [
                "source_pool_setting",
                "rank",
                "source",
                "distance",
                "inverse_distance_raw_weight",
                "inverse_distance_normalized_weight",
                "is_top3",
            ],
        ),
        "",
        f"- d2_b123 Top3: {summaries['d2_b123']['top3']}",
        f"- d2_b1234 Top3: {summaries['d2_b1234']['top3']}",
        f"- Top3 是否变化: {top3_changed}",
        f"- B4 是否进入 Top3: {b4_enters_top3}",
        f"- B4 是否进入 Top10: {b4_enters_top10}",
        "",
        "## 4. 六种方法结果对比",
        markdown_table(
            compare_df,
            [
                "method",
                "rmse_b123",
                "rmse_b1234",
                "rmse_delta",
                "accuracy_b123",
                "accuracy_b1234",
                "accuracy_delta",
                "normalized_rmse_delta",
                "original_scale_rmse_delta",
                "effect_after_adding_b4",
            ],
        ),
        "",
        "## 5. 与论文 Table 6 的一致性",
        "- 论文 Table 6 Dataset2 with information sharing 选源: B1 Item4 / B2 Item3 / B3 Item2。",
        f"- d2_b123 ordered match: {summaries['d2_b123']['matches_paper_table6_ordered']}; set match: {summaries['d2_b123']['matches_paper_table6_set']}",
        f"- d2_b1234 ordered match: {summaries['d2_b1234']['matches_paper_table6_ordered']}; set match: {summaries['d2_b1234']['matches_paper_table6_set']}",
        f"- d2_b123 Top3 overlap with Table 6: {overlap_b123}/3",
        f"- d2_b1234 Top3 overlap with Table 6: {overlap_b1234}/3",
        f"- 更接近论文 Table 6 的 source pool: {closer_pool}",
        "",
        "## 6. B4 对选源和结果的影响",
        f"- 包含 B4 后 KNN Top3 是否变化: {top3_changed}",
        f"- 包含 B4 后结果变化最大的 RMSE 方法: {largest['method']} (rmse_delta={float(largest['rmse_delta']):.6f})",
        f"- 按六种方法平均 RMSE 判断，当前完整 Dataset2 下效果更好的 source pool: {better_pool}",
        f"- 当前结果是否支持 B4 会实质影响 Dataset2 迁移学习结果: {'是' if top3_changed or any(compare_df['rmse_delta'].abs() > 1e-10) else '否'}",
        "",
        "## 7. 最终结论",
        f"- 若目标是复现论文表，应采用: {closer_pool}。",
        f"- 若目标是完整 Dataset2 客观评估，应采用: d2_b1234，因为它包含物理数据中可用的 B4 Item1-9 候选源。",
        f"- 包含 B4 后是否优于排除 B4: {'是' if better_pool == 'd2_b1234' else '否' if better_pool == 'd2_b123' else '两者持平'}。",
        "",
        "## 8. 需要谨慎说明的限制",
        "- 该审计只改变 source pool；没有改变数据清洗、KNN 距离公式、RFE 逻辑或 RMSE/accuracy 计算。",
        "- 神经网络训练即使设置 seed，在不同 TensorFlow/硬件后端下仍可能存在极小数值差异。",
        "- MSML-TL 为项目内部已有实现名；启动器中的 MSADW-TL 是展示别名，不是本脚本调用的方法名。",
        "- Dataset2 论文绝对时间边界证据在项目文档中仍标记为缺失/部分对齐，因此本结果应按当前代码的严格相对窗口协议解释。",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    configure_logging()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ctx = load_context()
    seed = int(ctx["seed"])
    setup_reproducibility(seed)

    all_rows: List[Dict[str, Any]] = []
    knn_frames: List[pd.DataFrame] = []
    summaries: Dict[str, Dict[str, Any]] = {}
    pools: Dict[str, pd.DataFrame] = {}

    for setting in SOURCE_POOLS:
        source_df = build_source_pool(ctx["processed_df"], ctx["base_source_df"], setting)
        pools[setting] = source_df
        knn_df, summary = build_knn_top10_rows(
            setting=setting,
            source_df=source_df,
            observed_target_df=ctx["observed_target_df"],
            feature_cols=ctx["feature_cols"],
        )
        knn_frames.append(knn_df)
        summaries[setting] = summary

    for setting, source_df in pools.items():
        for method in METHODS:
            print(f"[run] setting={setting} method={method}")
            row = run_one_method(
                method=method,
                setting=setting,
                source_df=source_df,
                target_df=ctx["target_df"],
                observed_target_df=ctx["observed_target_df"],
                feature_cols=ctx["feature_cols"],
                cfg=ctx["cfg"],
                protocol=ctx["protocol"],
                seed=seed,
            )
            all_rows.append(row)

    knn_df = pd.concat(knn_frames, ignore_index=True)
    all_df = attach_knn_summary_to_method_rows(pd.DataFrame(all_rows), summaries)
    compare_df = build_method_compare(all_df)

    all_df.to_csv(ALL_METHODS_CSV, index=False, encoding="utf-8")
    knn_df.to_csv(KNN_TOP10_CSV, index=False, encoding="utf-8")
    compare_df.to_csv(METHOD_COMPARE_CSV, index=False, encoding="utf-8")
    write_report(all_df, knn_df, compare_df, summaries)

    overlap_b123 = len(set(summaries["d2_b123"]["top3_keys"]) & set(PAPER_TABLE6_TOP3))
    overlap_b1234 = len(set(summaries["d2_b1234"]["top3_keys"]) & set(PAPER_TABLE6_TOP3))
    if summaries["d2_b123"]["matches_paper_table6_ordered"] and not summaries["d2_b1234"]["matches_paper_table6_ordered"]:
        closer_pool = "d2_b123"
    elif summaries["d2_b1234"]["matches_paper_table6_ordered"] and not summaries["d2_b123"]["matches_paper_table6_ordered"]:
        closer_pool = "d2_b1234"
    elif overlap_b123 > overlap_b1234:
        closer_pool = "d2_b123"
    elif overlap_b1234 > overlap_b123:
        closer_pool = "d2_b1234"
    else:
        closer_pool = "neither/tie"
    better_pool = (
        "d2_b1234"
        if float(compare_df["rmse_b1234"].mean()) < float(compare_df["rmse_b123"].mean())
        else "d2_b123"
        if float(compare_df["rmse_b123"].mean()) < float(compare_df["rmse_b1234"].mean())
        else "tie"
    )

    print("\nOutput files:")
    print(f"- {ALL_METHODS_CSV}")
    print(f"- {KNN_TOP10_CSV}")
    print(f"- {METHOD_COMPARE_CSV}")
    print(f"- {REPORT_MD}")
    print(f"d2_b123 Top3: {summaries['d2_b123']['top3']}")
    print(f"d2_b1234 Top3: {summaries['d2_b1234']['top3']}")
    print("RMSE by method:")
    for _, row in compare_df.iterrows():
        print(f"- {row['method']}: d2_b123={row['rmse_b123']:.6f}, d2_b1234={row['rmse_b1234']:.6f}")
    print("RMSE delta by method:")
    for _, row in compare_df.iterrows():
        print(f"- {row['method']}: rmse_delta={row['rmse_delta']:.6f}")
    print(f"Source pool closer to paper Table 6: {closer_pool}")
    print(f"Source pool better under current full Dataset2 evaluation: {better_pool}")


if __name__ == "__main__":
    main()
