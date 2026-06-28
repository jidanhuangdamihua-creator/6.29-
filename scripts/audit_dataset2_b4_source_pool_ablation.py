"""
Read-only Dataset2 KNN source-pool ablation for B4 inclusion.

This script does not modify model code, training logic, default config, or the
project's source-selection implementation. It reuses the current Dataset2
cleaning path, feature filtering, paper_observed_sequence KNN representation,
and target observed window, then writes a CSV and Markdown audit report.
"""

from __future__ import annotations

import copy
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_preprocessing import build_source_target_split, extract_datetime_features, load_dataset
from scripts.run_full_paper_experiments import (
    _apply_information_sharing_filter,
    _build_observed_target_window,
    _load_config,
    _resolve_dataset_feature_cols,
)
from source_selector import SourceSelector


DATASET_NAME = "Dataset2"
TARGET_KEY = ("B1", 10)
SOURCE_ITEM_IDS = list(range(1, 10))
EXPERIMENTS = [
    ("A_exclude_B4", ["B1", "B2", "B3"], "B1/B2/B3 Item 1-9"),
    ("B_include_B4", ["B1", "B2", "B3", "B4"], "B1/B2/B3/B4 Item 1-9"),
]
PAPER_TABLE6_TOP3 = [("B1", 4), ("B2", 3), ("B3", 2)]
CSV_PATH = ROOT / "outputs" / "audits" / "dataset2_b4_source_pool_ablation.csv"
MD_PATH = ROOT / "outputs" / "audits" / "dataset2_b4_source_pool_ablation.md"
EPS = 1e-8


def configure_logging() -> None:
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("experiment").setLevel(logging.WARNING)
    logging.disable(logging.WARNING)


def key_tuple(raw_key: Any) -> Tuple[str, int]:
    if isinstance(raw_key, (tuple, list)) and len(raw_key) >= 2:
        return str(raw_key[0]), int(raw_key[1])
    raise ValueError(f"Invalid source key: {raw_key!r}")


def key_label(key: Tuple[str, int]) -> str:
    return f"{key[0]} Item {int(key[1])}"


def compact_key_label(key: Tuple[str, int]) -> str:
    return f"{key[0]} Item{int(key[1])}"


def join_keys(keys: Sequence[Tuple[str, int]]) -> str:
    return " | ".join(key_label(key) for key in keys)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda value: f"{float(value):.6f}")
        else:
            display[col] = display[col].astype(str)
    lines = [
        "| " + " | ".join(str(col) for col in display.columns) + " |",
        "| " + " | ".join(["---"] * len(display.columns)) + " |",
    ]
    for _, row in display.iterrows():
        values = [str(row[col]).replace("|", "\\|") for col in display.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def load_context() -> Dict[str, Any]:
    cfg = copy.deepcopy(_load_config())
    cfg.setdefault("paper_reproduction", {})
    cfg["paper_reproduction"]["strict_paper_mode"] = True
    cfg["paper_reproduction"]["paper_strict_mode"] = True
    cfg["paper_reproduction"]["strict_paper_split"] = True
    cfg["paper_reproduction"]["paper_strict_split"] = True

    raw_df = load_dataset(DATASET_NAME, cfg["dataset_paths"][DATASET_NAME])
    processed_df = extract_datetime_features(raw_df)
    source_df, target_df = build_source_target_split(processed_df, cfg)
    observed_target_df = _build_observed_target_window(target_df)
    feature_cols = _resolve_dataset_feature_cols(DATASET_NAME, source_df, target_df, cfg)
    base_source_df = _apply_information_sharing_filter(
        dataset_name=DATASET_NAME,
        source_df=source_df,
        target_df=target_df,
        use_information_sharing=True,
        strict_paper_mode=True,
        protocol=cfg["paper_reproduction"],
        cfg=cfg,
    )

    return {
        "cfg": cfg,
        "processed_df": processed_df,
        "base_source_df": base_source_df,
        "target_df": target_df,
        "observed_target_df": observed_target_df,
        "feature_cols": feature_cols,
    }


def build_source_pool(
    processed_df: pd.DataFrame,
    base_source_df: pd.DataFrame,
    entity_ids: Sequence[str],
) -> pd.DataFrame:
    mask = (
        processed_df["entity_id"].astype(str).isin([str(v) for v in entity_ids])
        & pd.to_numeric(processed_df["item_id"], errors="coerce").isin(SOURCE_ITEM_IDS)
    )
    source_df = processed_df.loc[mask].copy()
    source_df.attrs = base_source_df.attrs.copy()
    source_df.attrs["source_pool_scope_mode"] = (
        "dataset2_ablation_" + ("include_B4" if "B4" in entity_ids else "exclude_B4")
    )
    return source_df


def select_sources(
    target_df: pd.DataFrame,
    source_df: pd.DataFrame,
    feature_cols: Sequence[str],
    k: int,
) -> Dict[str, Any]:
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


def source_group_count(source_df: pd.DataFrame) -> int:
    return int(source_df.groupby(["entity_id", "item_id"], sort=False).ngroups)


def summarize_experiment(
    experiment_id: str,
    entity_ids: Sequence[str],
    pool_label: str,
    target_df: pd.DataFrame,
    source_df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    top10_result = select_sources(target_df, source_df, feature_cols, k=10)
    top3_result = select_sources(target_df, source_df, feature_cols, k=3)

    top10_sources = top10_result["sources"]
    top3_sources = top3_result["sources"]
    top3_keys = [key_tuple(row["source_key"]) for row in top3_sources]
    top10_keys = [key_tuple(row["source_key"]) for row in top10_sources]
    top3_weight_by_key = {key_tuple(row["source_key"]): float(row["weight"]) for row in top3_sources}
    matches_paper_ordered = top3_keys == PAPER_TABLE6_TOP3
    matches_paper_set = set(top3_keys) == set(PAPER_TABLE6_TOP3)
    contains_b4_pool = "B4" in {str(v) for v in entity_ids}
    contains_b4_top10 = any(key[0] == "B4" for key in top10_keys)
    contains_b4_top3 = any(key[0] == "B4" for key in top3_keys)
    candidate_size = source_group_count(source_df)

    rows: List[Dict[str, Any]] = []
    for row in top10_sources:
        key = key_tuple(row["source_key"])
        distance = float(row["distance"])
        raw_weight = float(1.0 / (distance + EPS))
        rows.append(
            {
                "experiment": experiment_id,
                "source_pool": pool_label,
                "candidate_pool_size": candidate_size,
                "rank": int(row["source_rank"]),
                "source_brand": key[0],
                "source_item": key[1],
                "source_label": key_label(key),
                "is_top3_selected": key in top3_keys,
                "euclidean_distance": distance,
                "inverse_distance_raw_weight": raw_weight,
                "normalized_weight_top10": float(row["weight"]),
                "normalized_weight_top3_if_selected": top3_weight_by_key.get(key, ""),
                "contains_B4_in_pool": contains_b4_pool,
                "contains_B4_in_top10": contains_b4_top10,
                "contains_B4_in_top3": contains_b4_top3,
                "top3_selected_sources": join_keys(top3_keys),
                "paper_table6_expected_sources": join_keys(PAPER_TABLE6_TOP3),
                "matches_paper_table6_ordered": matches_paper_ordered,
                "matches_paper_table6_set": matches_paper_set,
                "knn_representation": top10_result["meta"].get("knn_representation", ""),
                "knn_feature_mode": top10_result["meta"].get("knn_feature_mode", ""),
                "knn_features": " | ".join(top10_result["meta"].get("feature_cols", [])),
                "observed_window_rows": top10_result["meta"].get("observed_window_rows", ""),
                "observed_window_unique_dates": top10_result["meta"].get("observed_window_unique_dates", ""),
            }
        )

    summary = {
        "experiment": experiment_id,
        "source_pool": pool_label,
        "candidate_pool_size": candidate_size,
        "top10": join_keys(top10_keys),
        "top3": join_keys(top3_keys),
        "contains_B4_in_pool": contains_b4_pool,
        "contains_B4_in_top10": contains_b4_top10,
        "contains_B4_in_top3": contains_b4_top3,
        "matches_paper_table6_ordered": matches_paper_ordered,
        "matches_paper_table6_set": matches_paper_set,
        "knn_features": " | ".join(top10_result["meta"].get("feature_cols", [])),
        "observed_window_rows": top10_result["meta"].get("observed_window_rows", ""),
        "observed_window_unique_dates": top10_result["meta"].get("observed_window_unique_dates", ""),
    }
    return rows, summary


def write_report(rows_df: pd.DataFrame, summaries: Sequence[Dict[str, Any]]) -> None:
    summary_df = pd.DataFrame(summaries)
    top10_view = rows_df[
        [
            "experiment",
            "rank",
            "source_label",
            "euclidean_distance",
            "inverse_distance_raw_weight",
            "normalized_weight_top10",
            "normalized_weight_top3_if_selected",
            "is_top3_selected",
        ]
    ].copy()

    summary_by_exp = {row["experiment"]: row for row in summaries}
    a_top3 = summary_by_exp["A_exclude_B4"]["top3"]
    b_top3 = summary_by_exp["B_include_B4"]["top3"]
    top3_changed = a_top3 != b_top3
    b4_in_top3 = bool(summary_by_exp["B_include_B4"]["contains_B4_in_top3"])
    b4_in_top10 = bool(summary_by_exp["B_include_B4"]["contains_B4_in_top10"])
    exclude_b4_matches = bool(summary_by_exp["A_exclude_B4"]["matches_paper_table6_ordered"])

    if top3_changed:
        top3_change_text = f"Yes. A selects {a_top3}; B selects {b_top3}."
    else:
        top3_change_text = f"No. Both settings select {a_top3}."

    b4_text = (
        f"B4 enters Top-3: {b4_in_top3}. B4 enters Top-10: {b4_in_top10}."
    )
    exclude_text = (
        "No. In this audit the A setting already matches the Dataset2 Table 6 source set, and adding B4 is the setting that changes it."
        if exclude_b4_matches
        else "Excluding B4 avoids B4 entering the selected sources, but it is not sufficient to reproduce the Dataset2 Table 6 source set under the current KNN implementation; A still selects B2 Item 2 instead of B2 Item 3."
    )

    lines = [
        "# Dataset2 B4 Source Pool Ablation",
        "",
        "## Scope",
        "",
        "- Read-only audit only: no main program, training logic, KNN implementation, or model state was modified.",
        f"- Target domain: {key_label(TARGET_KEY)}.",
        "- Reused current Dataset2 cleaning, current KNN feature filtering, current `paper_observed_sequence` representation, and current target observed window.",
        f"- Paper Table 6 Dataset2 with information sharing expected sources: {join_keys(PAPER_TABLE6_TOP3)}.",
        "",
        "## Summary",
        "",
        markdown_table(summary_df),
        "",
        "## Top-10 Ranking",
        "",
        markdown_table(top10_view),
        "",
        "## Conclusions",
        "",
        f"- Including B4 changes Top-3: {top3_change_text}",
        f"- B4 inclusion result: {b4_text}",
        f"- Does excluding B4 affect paper-alignment result: {exclude_text}",
        "- The current B1/B2/B3 source-pool setting should be described as a constraint used to align the reproduction experiment table with paper Table 6, not as a paper-explicit rule, unless the paper text is separately cited as explicitly excluding B4.",
    ]
    MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_logging()
    context = load_context()
    rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    for experiment_id, entity_ids, pool_label in EXPERIMENTS:
        source_df = build_source_pool(
            processed_df=context["processed_df"],
            base_source_df=context["base_source_df"],
            entity_ids=entity_ids,
        )
        experiment_rows, summary = summarize_experiment(
            experiment_id=experiment_id,
            entity_ids=entity_ids,
            pool_label=pool_label,
            target_df=context["observed_target_df"],
            source_df=source_df,
            feature_cols=context["feature_cols"],
        )
        rows.extend(experiment_rows)
        summaries.append(summary)

    rows_df = pd.DataFrame(rows)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows_df.to_csv(CSV_PATH, index=False, encoding="utf-8")
    write_report(rows_df, summaries)

    print("Dataset2 B4 source-pool ablation complete")
    print(f"CSV: {CSV_PATH}")
    print(f"Markdown: {MD_PATH}")
    for summary in summaries:
        print(
            f"{summary['experiment']} Top-3: {summary['top3']} | "
            f"B4 in pool/top10/top3: {summary['contains_B4_in_pool']}/"
            f"{summary['contains_B4_in_top10']}/{summary['contains_B4_in_top3']} | "
            f"matches Table 6 ordered/set: {summary['matches_paper_table6_ordered']}/"
            f"{summary['matches_paper_table6_set']}"
        )


if __name__ == "__main__":
    main()
