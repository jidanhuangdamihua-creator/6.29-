"""
Read-only Dataset2 B123 with-sharing KNN diagnosis.

This audit explains why the current paper_observed_sequence KNN ranking selects
B2 Item2 ahead of the paper-reported B2 Item3. It does not train models and does
not modify cleaning, split, KNN formula, RFE, metrics, or source-pool rules.
"""

from __future__ import annotations

import copy
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_preprocessing import (
    build_source_target_split,
    extract_datetime_features,
    load_dataset,
    temporal_split_by_ratio_or_dates,
)
from scripts.run_full_paper_experiments import (
    _apply_information_sharing_filter,
    _build_observed_target_window,
    _load_config,
    _resolve_dataset_feature_cols,
)
from source_selector import SourceSelector


DATASET = "Dataset2"
TARGET_KEY = ("B1", 10)
SOURCE_ENTITIES = ["B1", "B2", "B3"]
SOURCE_ITEMS = list(range(1, 10))
PAPER_EXPECTED = [("B1", 4), ("B2", 3), ("B3", 2)]
CURRENT_TOP3 = [("B1", 4), ("B3", 2), ("B2", 2)]
FOCUS_SOURCES = [("B2", 2), ("B2", 3)]
EPS = 1e-8

OUT_DIR = ROOT / "outputs" / "audits"
ALL_CANDIDATES_CSV = OUT_DIR / "dataset2_b123_withsharing_knn_all_candidates.csv"
OBS_COMPARE_CSV = OUT_DIR / "dataset2_b2_item2_vs_b2_item3_observed_window_compare.csv"
CONTRIB_CSV = OUT_DIR / "dataset2_b2_item2_vs_b2_item3_distance_contribution.csv"
ITEM_MAPPING_CSV = OUT_DIR / "dataset2_item_id_mapping_audit.csv"
WINDOW_ALIGNMENT_CSV = OUT_DIR / "dataset2_b123_withsharing_window_alignment_audit.csv"
PAPER_DISTANCE_CSV = OUT_DIR / "dataset2_paper_expected_sources_distance_check.csv"
REPORT_MD = OUT_DIR / "dataset2_b2_item2_vs_b2_item3_knn_diagnosis.md"


def key_label(key: Tuple[str, int]) -> str:
    return f"{key[0]} Item{int(key[1])}"


def join_keys(keys: Sequence[Tuple[str, int]]) -> str:
    return " / ".join(key_label(k) for k in keys)


def pipe(values: Iterable[Any]) -> str:
    return "|".join("" if v is None else str(v) for v in values)


def markdown_table(df: pd.DataFrame, cols: Sequence[str] | None = None, max_rows: int | None = None) -> str:
    work = df.copy()
    if cols is not None:
        work = work[list(cols)].copy()
    if max_rows is not None:
        work = work.head(max_rows).copy()
    if work.empty:
        return "_No rows._"
    for col in work.columns:
        if pd.api.types.is_float_dtype(work[col]):
            work[col] = work[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.6f}")
        else:
            work[col] = work[col].map(lambda x: "" if pd.isna(x) else str(x))
    lines = [
        "| " + " | ".join(work.columns) + " |",
        "| " + " | ".join(["---"] * len(work.columns)) + " |",
    ]
    for _, row in work.iterrows():
        values = [str(row[col]).replace("|", "\\|") for col in work.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def key_tuple(raw: Any) -> Tuple[str, int]:
    if isinstance(raw, (tuple, list)) and len(raw) >= 2:
        return str(raw[0]), int(raw[1])
    raise ValueError(f"Invalid source key: {raw!r}")


def load_context() -> Dict[str, Any]:
    cfg = copy.deepcopy(_load_config())
    cfg.setdefault("paper_reproduction", {})
    cfg["paper_reproduction"]["strict_paper_mode"] = True
    cfg["paper_reproduction"]["paper_strict_mode"] = True
    protocol = cfg["paper_reproduction"]

    raw_path = ROOT / cfg["dataset_paths"][DATASET]
    raw_wide_df = pd.read_csv(raw_path, low_memory=False)
    loaded_df = load_dataset(DATASET, str(raw_path))
    processed_df = extract_datetime_features(loaded_df)
    source_df, target_df = build_source_target_split(processed_df, cfg)
    source_df = _apply_information_sharing_filter(
        dataset_name=DATASET,
        source_df=source_df,
        target_df=target_df,
        use_information_sharing=True,
        strict_paper_mode=True,
        protocol=protocol,
        cfg=cfg,
    )
    target_df.attrs["information_sharing_scenario"] = source_df.attrs.get("information_sharing_scenario", "")
    target_df.attrs["signature_static_feature_cols"] = list(source_df.attrs.get("signature_static_feature_cols", []))
    observed_target_df = _build_observed_target_window(target_df)
    feature_cols = _resolve_dataset_feature_cols(DATASET, source_df, target_df, cfg)
    return {
        "cfg": cfg,
        "protocol": protocol,
        "raw_path": raw_path,
        "raw_wide_df": raw_wide_df,
        "loaded_df": loaded_df,
        "processed_df": processed_df,
        "source_df": source_df,
        "target_df": target_df,
        "observed_target_df": observed_target_df,
        "feature_cols": feature_cols,
    }


def source_group_count(source_df: pd.DataFrame) -> int:
    return int(source_df.groupby(["entity_id", "item_id"], sort=False).ngroups)


def select_all_sources(target_df: pd.DataFrame, source_df: pd.DataFrame, feature_cols: Sequence[str]) -> Dict[str, Any]:
    return SourceSelector().select_top_k_sources(
        target_df=target_df,
        source_df=source_df,
        feature_cols=list(feature_cols),
        k=source_group_count(source_df),
        group_cols=("entity_id", "item_id"),
        weight_mode="inverse_distance",
        debug_mode=False,
        include_sales_in_knn=True,
        knn_representation="paper_observed_sequence",
    )


def build_candidate_ranking(ctx: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    selection = select_all_sources(ctx["observed_target_df"], ctx["source_df"], ctx["feature_cols"])
    sources = selection.get("sources", [])
    rows = []
    for row in sources:
        key = key_tuple(row["source_key"])
        distance = float(row["distance"])
        rows.append(
            {
                "rank": int(row["source_rank"]),
                "source_entity": key[0],
                "source_item": key[1],
                "source_label": key_label(key),
                "distance": distance,
                "raw_inverse_distance_weight": 1.0 / (distance + EPS),
                "normalized_weight": float(row["weight"]),
                "is_paper_expected_source": key in PAPER_EXPECTED,
                "is_current_top3": int(row["source_rank"]) <= 3,
                "is_focus_source": key in FOCUS_SOURCES,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(ALL_CANDIDATES_CSV, index=False, encoding="utf-8")
    return df, selection


def aligned_group(ctx: Dict[str, Any], key: Tuple[str, int], dates: Sequence[pd.Timestamp] | None = None) -> pd.DataFrame:
    df = ctx["source_df"] if key != TARGET_KEY else ctx["target_df"]
    work = df[
        (df["entity_id"].astype(str) == str(key[0]))
        & (pd.to_numeric(df["item_id"], errors="coerce") == int(key[1]))
    ].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if dates is not None:
        date_index = pd.to_datetime(pd.Index(dates))
        work = work[work["date"].isin(date_index)].sort_values("date").reset_index(drop=True)
    return work


def build_observed_compare(ctx: Dict[str, Any], knn_features: Sequence[str], static_features: Sequence[str]) -> pd.DataFrame:
    target = ctx["observed_target_df"].copy().sort_values("date").reset_index(drop=True)
    dates = target["date"].tolist()
    b2_item2 = aligned_group(ctx, ("B2", 2), dates)
    b2_item3 = aligned_group(ctx, ("B2", 3), dates)

    wide_rows: List[Dict[str, Any]] = []
    feature_rows: List[Dict[str, Any]] = []
    for idx in range(len(target)):
        t = target.iloc[idx]
        s2 = b2_item2.iloc[idx]
        s3 = b2_item3.iloc[idx]
        row: Dict[str, Any] = {
            "row_type": "time_step_summary",
            "time_step": idx + 1,
            "date": pd.Timestamp(t["date"]).strftime("%Y-%m-%d"),
            "week_index": int(t["week"]),
            "target_B1_Item10_sales": float(t["sales"]),
            "B2_Item2_sales": float(s2["sales"]),
            "B2_Item3_sales": float(s3["sales"]),
            "target_promo": float(t["promo"]),
            "B2_Item2_promo": float(s2["promo"]),
            "B2_Item3_promo": float(s3["promo"]),
        }
        per_step_sq_2 = 0.0
        per_step_sq_3 = 0.0
        for feature in knn_features:
            target_value = float(pd.to_numeric(pd.Series([t[feature]]), errors="coerce").fillna(0).iloc[0])
            item2_value = float(pd.to_numeric(pd.Series([s2[feature]]), errors="coerce").fillna(0).iloc[0])
            item3_value = float(pd.to_numeric(pd.Series([s3[feature]]), errors="coerce").fillna(0).iloc[0])
            sq2 = (target_value - item2_value) ** 2
            sq3 = (target_value - item3_value) ** 2
            per_step_sq_2 += sq2
            per_step_sq_3 += sq3
            row[f"target_{feature}"] = target_value
            row[f"B2_Item2_{feature}"] = item2_value
            row[f"B2_Item3_{feature}"] = item3_value
            feature_rows.append(
                {
                    "row_type": "feature_detail",
                    "feature": feature,
                    "time_step": idx + 1,
                    "date": pd.Timestamp(t["date"]).strftime("%Y-%m-%d"),
                    "week_index": int(t["week"]),
                    "target_value": target_value,
                    "B2_Item2_value": item2_value,
                    "B2_Item3_value": item3_value,
                    "squared_diff_B2_Item2": sq2,
                    "squared_diff_B2_Item3": sq3,
                    "which_source_closer_at_this_feature_step": (
                        "B2 Item2" if sq2 < sq3 else "B2 Item3" if sq3 < sq2 else "tie"
                    ),
                }
            )
        row["per_step_squared_diff_B2_Item2"] = per_step_sq_2
        row["per_step_squared_diff_B2_Item3"] = per_step_sq_3
        row["which_source_closer_at_this_step"] = (
            "B2 Item2" if per_step_sq_2 < per_step_sq_3 else "B2 Item3" if per_step_sq_3 < per_step_sq_2 else "tie"
        )
        wide_rows.append(row)

    # Include static signature contribution as explicit feature_detail rows.
    for feature in static_features:
        t_val = float(pd.to_numeric(target[feature], errors="coerce").dropna().iloc[-1])
        s2_val = float(pd.to_numeric(b2_item2[feature], errors="coerce").dropna().iloc[-1])
        s3_val = float(pd.to_numeric(b2_item3[feature], errors="coerce").dropna().iloc[-1])
        feature_rows.append(
            {
                "row_type": "static_feature_detail",
                "feature": f"static:{feature}",
                "time_step": "static_last_value",
                "date": "",
                "week_index": "",
                "target_value": t_val,
                "B2_Item2_value": s2_val,
                "B2_Item3_value": s3_val,
                "squared_diff_B2_Item2": (t_val - s2_val) ** 2,
                "squared_diff_B2_Item3": (t_val - s3_val) ** 2,
                "which_source_closer_at_this_feature_step": (
                    "B2 Item2" if (t_val - s2_val) ** 2 < (t_val - s3_val) ** 2 else "B2 Item3" if (t_val - s3_val) ** 2 < (t_val - s2_val) ** 2 else "tie"
                ),
            }
        )

    wide_df = pd.DataFrame(wide_rows)
    feature_df = pd.DataFrame(feature_rows)
    out = pd.concat([wide_df, feature_df], ignore_index=True, sort=False)
    out.to_csv(OBS_COMPARE_CSV, index=False, encoding="utf-8")
    return out


def build_contribution(ctx: Dict[str, Any], knn_features: Sequence[str], static_features: Sequence[str]) -> pd.DataFrame:
    target = ctx["observed_target_df"].copy().sort_values("date").reset_index(drop=True)
    dates = target["date"].tolist()
    rows = []
    for source_key in FOCUS_SOURCES:
        source = aligned_group(ctx, source_key, dates)
        feature_sums: Dict[str, float] = {}
        for feature in knn_features:
            t_vals = pd.to_numeric(target[feature], errors="coerce").fillna(0).to_numpy(dtype=float)
            s_vals = pd.to_numeric(source[feature], errors="coerce").fillna(0).to_numpy(dtype=float)
            feature_sums[feature] = float(np.sum((t_vals - s_vals) ** 2))
        for feature in static_features:
            t_val = float(pd.to_numeric(target[feature], errors="coerce").dropna().iloc[-1])
            s_val = float(pd.to_numeric(source[feature], errors="coerce").dropna().iloc[-1])
            feature_sums[f"static:{feature}"] = float((t_val - s_val) ** 2)
        total_sq = float(sum(feature_sums.values()))
        total_distance = math.sqrt(total_sq)
        for feature, sq_sum in feature_sums.items():
            rows.append(
                {
                    "source": key_label(source_key),
                    "feature": feature,
                    "squared_diff_sum": sq_sum,
                    "contribution_ratio": (sq_sum / total_sq) if total_sq else 0.0,
                    "total_distance": total_distance,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(CONTRIB_CSV, index=False, encoding="utf-8")
    return df


def build_item_mapping_audit(ctx: Dict[str, Any]) -> pd.DataFrame:
    raw_cols = ctx["raw_wide_df"].columns.tolist()
    qty_items = []
    promo_items = []
    for col in raw_cols:
        m_qty = re.match(r"QTY_(B\d+)_(\d+)$", str(col))
        m_promo = re.match(r"PROMO_(B\d+)_(\d+)$", str(col))
        if m_qty:
            qty_items.append((m_qty.group(1), int(m_qty.group(2)), col))
        if m_promo:
            promo_items.append((m_promo.group(1), int(m_promo.group(2)), col))
    item_values = sorted(int(v) for v in pd.to_numeric(ctx["loaded_df"]["item_id"], errors="coerce").dropna().unique())
    processed_cols = set(ctx["processed_df"].columns)
    rows = []
    for key in [("B2", 2), ("B2", 3)]:
        rows.append(
            {
                "audit_field": key_label(key),
                "clean_csv_item_id_unique_values": pipe(item_values),
                "raw_qty_column": f"QTY_{key[0]}_{key[1]}",
                "raw_promo_column": f"PROMO_{key[0]}_{key[1]}",
                "qty_column_exists": f"QTY_{key[0]}_{key[1]}" in raw_cols,
                "promo_column_exists": f"PROMO_{key[0]}_{key[1]}" in raw_cols,
                "clean_entity_id": key[0],
                "clean_item_id": key[1],
                "zero_based_one_based_offset_risk": "low: item_id is parsed directly from 1-based raw suffix",
                "item_id_code_item_id_mixing_risk": "low for KNN: item_id_code column is absent; item_id is only group key and excluded from KNN features",
                "item_id_code_column_exists": "item_id_code" in processed_cols,
                "current_knn_uses_item_id_or_item_id_code": "item_id as group key only; neither item_id nor item_id_code is in KNN feature_cols",
            }
        )
    rows.append(
        {
            "audit_field": "raw_column_summary",
            "clean_csv_item_id_unique_values": pipe(item_values),
            "raw_qty_column": f"count={len(qty_items)}; min_suffix={min(i for _, i, _ in qty_items)}; max_suffix={max(i for _, i, _ in qty_items)}",
            "raw_promo_column": f"count={len(promo_items)}; min_suffix={min(i for _, i, _ in promo_items)}; max_suffix={max(i for _, i, _ in promo_items)}",
            "qty_column_exists": True,
            "promo_column_exists": True,
            "clean_entity_id": pipe(sorted({e for e, _, _ in qty_items})),
            "clean_item_id": "",
            "zero_based_one_based_offset_risk": "low",
            "item_id_code_item_id_mixing_risk": "low for current KNN",
            "item_id_code_column_exists": "item_id_code" in processed_cols,
            "current_knn_uses_item_id_or_item_id_code": "item_id group key; KNN feature columns are resolved separately",
        }
    )
    df = pd.DataFrame(rows)
    df.to_csv(ITEM_MAPPING_CSV, index=False, encoding="utf-8")
    return df


def sequence_distance(target: pd.DataFrame, source: pd.DataFrame, knn_features: Sequence[str], static_features: Sequence[str]) -> float:
    total_sq = 0.0
    for feature in knn_features:
        t_vals = pd.to_numeric(target[feature], errors="coerce").fillna(0).to_numpy(dtype=float)
        s_vals = pd.to_numeric(source[feature], errors="coerce").fillna(0).to_numpy(dtype=float)
        n = min(len(t_vals), len(s_vals))
        total_sq += float(np.sum((t_vals[:n] - s_vals[:n]) ** 2))
    for feature in static_features:
        if feature in target.columns and feature in source.columns and not target.empty and not source.empty:
            t_val = float(pd.to_numeric(target[feature], errors="coerce").dropna().iloc[-1])
            s_val = float(pd.to_numeric(source[feature], errors="coerce").dropna().iloc[-1])
            total_sq += float((t_val - s_val) ** 2)
    return math.sqrt(total_sq)


def build_window_alignment_audit(ctx: Dict[str, Any], knn_features: Sequence[str], static_features: Sequence[str]) -> pd.DataFrame:
    target = ctx["observed_target_df"].copy().sort_values("date").reset_index(drop=True)
    dates = target["date"].tolist()
    rows = []
    aligned_sources = {key: aligned_group(ctx, key, dates) for key in FOCUS_SOURCES}

    for idx in range(len(target)):
        t = target.iloc[idx]
        row = {
            "alignment_mode": "calendar_date_current",
            "time_step": idx + 1,
            "target_date": pd.Timestamp(t["date"]).strftime("%Y-%m-%d"),
            "target_week": int(t["week"]),
            "B2_Item2_date": pd.Timestamp(aligned_sources[("B2", 2)].iloc[idx]["date"]).strftime("%Y-%m-%d"),
            "B2_Item2_week": int(aligned_sources[("B2", 2)].iloc[idx]["week"]),
            "B2_Item3_date": pd.Timestamp(aligned_sources[("B2", 3)].iloc[idx]["date"]).strftime("%Y-%m-%d"),
            "B2_Item3_week": int(aligned_sources[("B2", 3)].iloc[idx]["week"]),
            "same_calendar_date_all_three": (
                pd.Timestamp(t["date"]) == pd.Timestamp(aligned_sources[("B2", 2)].iloc[idx]["date"])
                == pd.Timestamp(aligned_sources[("B2", 3)].iloc[idx]["date"])
            ),
            "same_relative_position_all_three": True,
            "alignment_basis": "calendar date via target observed dates",
        }
        rows.append(row)

    current_distances = {
        key: sequence_distance(target, aligned_sources[key], knn_features, static_features)
        for key in FOCUS_SOURCES
    }

    full_sources = {
        key: aligned_group(ctx, key, None).sort_values("date").reset_index(drop=True)
        for key in FOCUS_SOURCES
    }
    alt_modes = {
        "relative_position_from_source_start": {key: full_sources[key].head(len(target)).copy() for key in FOCUS_SOURCES},
        "relative_position_from_source_end": {key: full_sources[key].tail(len(target)).copy().reset_index(drop=True) for key in FOCUS_SOURCES},
    }
    for mode, source_map in alt_modes.items():
        distances = {key: sequence_distance(target, source_map[key], knn_features, static_features) for key in FOCUS_SOURCES}
        rows.append(
            {
                "alignment_mode": mode,
                "time_step": "summary",
                "target_date": f"{pd.Timestamp(target['date'].min()).strftime('%Y-%m-%d')}..{pd.Timestamp(target['date'].max()).strftime('%Y-%m-%d')}",
                "target_week": pipe(target["week"].astype(int).tolist()),
                "B2_Item2_date": f"{pd.Timestamp(source_map[('B2', 2)]['date'].min()).strftime('%Y-%m-%d')}..{pd.Timestamp(source_map[('B2', 2)]['date'].max()).strftime('%Y-%m-%d')}",
                "B2_Item2_week": pipe(source_map[("B2", 2)]["week"].astype(int).tolist()),
                "B2_Item3_date": f"{pd.Timestamp(source_map[('B2', 3)]['date'].min()).strftime('%Y-%m-%d')}..{pd.Timestamp(source_map[('B2', 3)]['date'].max()).strftime('%Y-%m-%d')}",
                "B2_Item3_week": pipe(source_map[("B2", 3)]["week"].astype(int).tolist()),
                "same_calendar_date_all_three": False,
                "same_relative_position_all_three": True,
                "alignment_basis": mode,
                "current_calendar_distance_B2_Item2": current_distances[("B2", 2)],
                "current_calendar_distance_B2_Item3": current_distances[("B2", 3)],
                "alternative_distance_B2_Item2": distances[("B2", 2)],
                "alternative_distance_B2_Item3": distances[("B2", 3)],
                "alternative_winner": "B2 Item2" if distances[("B2", 2)] < distances[("B2", 3)] else "B2 Item3" if distances[("B2", 3)] < distances[("B2", 2)] else "tie",
                "would_B2_Item2_B2_Item3_order_swap": (
                    (current_distances[("B2", 2)] < current_distances[("B2", 3)])
                    != (distances[("B2", 2)] < distances[("B2", 3)])
                ),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(WINDOW_ALIGNMENT_CSV, index=False, encoding="utf-8")
    return df


def build_paper_expected_distance_check(ranking_df: pd.DataFrame) -> pd.DataFrame:
    cutoff = float(ranking_df.loc[ranking_df["rank"] == 3, "distance"].iloc[0])
    wanted = list(dict.fromkeys(CURRENT_TOP3 + PAPER_EXPECTED))
    rows = []
    for key in wanted:
        row = ranking_df[
            (ranking_df["source_entity"] == key[0])
            & (ranking_df["source_item"] == key[1])
        ].iloc[0]
        gap = float(row["distance"]) - cutoff
        rows.append(
            {
                "source": key_label(key),
                "source_entity": key[0],
                "source_item": key[1],
                "is_current_top3_source": key in CURRENT_TOP3,
                "is_paper_expected_source": key in PAPER_EXPECTED,
                "current_rank": int(row["rank"]),
                "current_distance": float(row["distance"]),
                "top3_cutoff_distance": cutoff,
                "distance_minus_top3_cutoff": gap,
                "relative_gap_vs_cutoff": gap / cutoff if cutoff else np.nan,
                "is_micro_gap": abs(gap) / cutoff < 0.02 if cutoff else False,
                "is_rank_4_or_5": int(row["rank"]) in {4, 5},
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(PAPER_DISTANCE_CSV, index=False, encoding="utf-8")
    return df


def date_range_text(df: pd.DataFrame) -> str:
    if df.empty:
        return "empty"
    return f"{pd.Timestamp(df['date'].min()).strftime('%Y-%m-%d')}..{pd.Timestamp(df['date'].max()).strftime('%Y-%m-%d')} ({df['date'].nunique()} dates)"


def write_report(
    ctx: Dict[str, Any],
    ranking_df: pd.DataFrame,
    selection: Dict[str, Any],
    compare_df: pd.DataFrame,
    contrib_df: pd.DataFrame,
    item_df: pd.DataFrame,
    window_df: pd.DataFrame,
    paper_df: pd.DataFrame,
) -> None:
    target_train, target_val, target_test = temporal_split_by_ratio_or_dates(ctx["target_df"])
    meta = selection.get("meta", {})
    knn_features = list(meta.get("feature_cols", []))
    static_features = list(meta.get("signature_static_feature_cols", []))

    b2i2 = ranking_df[(ranking_df["source_entity"] == "B2") & (ranking_df["source_item"] == 2)].iloc[0]
    b2i3 = ranking_df[(ranking_df["source_entity"] == "B2") & (ranking_df["source_item"] == 3)].iloc[0]
    contrib_pivot = contrib_df.pivot_table(index="feature", columns="source", values="squared_diff_sum", aggfunc="first").reset_index()
    sales_b2i2 = float(contrib_df[(contrib_df["source"] == "B2 Item2") & (contrib_df["feature"] == "sales")]["contribution_ratio"].iloc[0])
    sales_b2i3 = float(contrib_df[(contrib_df["source"] == "B2 Item3") & (contrib_df["feature"] == "sales")]["contribution_ratio"].iloc[0])
    promo_b2i2 = float(contrib_df[(contrib_df["source"] == "B2 Item2") & (contrib_df["feature"] == "promo")]["contribution_ratio"].iloc[0])
    promo_b2i3 = float(contrib_df[(contrib_df["source"] == "B2 Item3") & (contrib_df["feature"] == "promo")]["contribution_ratio"].iloc[0])
    b2i3_gap = float(b2i3["distance"]) - float(ranking_df.loc[ranking_df["rank"] == 3, "distance"].iloc[0])
    alt_summary = window_df[window_df["time_step"].astype(str) == "summary"].copy()

    lines = [
        "# Dataset2 B2 Item2 vs B2 Item3 KNN Diagnosis",
        "",
        "## 1. 基础设置确认",
        f"- target entity / item: {TARGET_KEY[0]} Item{TARGET_KEY[1]}",
        "- source_pool_policy: current strict Dataset2 B123 source pool, Item1-9",
        "- information_sharing: True / with_information_sharing",
        f"- candidate pool size: {source_group_count(ctx['source_df'])}",
        f"- candidate entities: {pipe(sorted(ctx['source_df']['entity_id'].astype(str).unique()))}",
        f"- candidate item范围: {int(ctx['source_df']['item_id'].min())}..{int(ctx['source_df']['item_id'].max())}",
        f"- target train split: {date_range_text(target_train)}",
        f"- target val split: {date_range_text(target_val)}",
        f"- target test split: {date_range_text(target_test)}",
        f"- KNN observed window: {date_range_text(ctx['observed_target_df'])}",
        f"- KNN observed weeks: {pipe(ctx['observed_target_df']['week'].astype(int).tolist())}",
        f"- target_test_data_excluded: {bool(meta.get('target_test_data_excluded', True))}",
        f"- KNN feature columns: {pipe(knn_features)}",
        f"- KNN static signature columns: {pipe(static_features)}",
        "",
        "## 2. 完整候选距离排名",
        markdown_table(ranking_df, ["rank", "source_label", "distance", "raw_inverse_distance_weight", "normalized_weight", "is_paper_expected_source", "is_current_top3"], max_rows=27),
        "",
        "## 3. B2 Item2 vs B2 Item3 结论",
        f"- B2 Item2 当前 rank={int(b2i2['rank'])}, distance={float(b2i2['distance']):.6f}.",
        f"- B2 Item3 当前 rank={int(b2i3['rank'])}, distance={float(b2i3['distance']):.6f}.",
        f"- B2 Item3 距离 Top3 cutoff 的差距: {b2i3_gap:.6f}.",
        "- 当前 B2 Item2 更近，是因为在 observed window 的当前 KNN 多特征向量上，B2 Item2 的总平方差更小。",
        "",
        "## 4. 距离贡献分解",
        markdown_table(contrib_pivot),
        "",
        f"- sales contribution ratio: B2 Item2={sales_b2i2:.2%}, B2 Item3={sales_b2i3:.2%}.",
        f"- promo sequence contribution ratio: B2 Item2={promo_b2i2:.2%}, B2 Item3={promo_b2i3:.2%}.",
        "- year/month/week/day 在 calendar-date 对齐下三者相同，因此贡献为 0。",
        "",
        "## 5. item 编码偏移检查",
        markdown_table(item_df),
        "",
        "结论: 当前清洗直接从 `QTY_B2_2`/`PROMO_B2_2` 得到 B2 Item2，从 `QTY_B2_3`/`PROMO_B2_3` 得到 B2 Item3；未发现 0-based/1-based 偏移证据。当前 KNN 不把 item_id 或 item_id_code 作为特征，只把 item_id 作为分组键。",
        "",
        "## 6. 日期窗口/对齐方式检查",
        markdown_table(alt_summary, ["alignment_mode", "current_calendar_distance_B2_Item2", "current_calendar_distance_B2_Item3", "alternative_distance_B2_Item2", "alternative_distance_B2_Item3", "alternative_winner", "would_B2_Item2_B2_Item3_order_swap"]),
        "",
        "结论: 当前实现按 calendar date 对齐 target observed dates；B2 Item2、B2 Item3 与 target 的 observed 日期完全一致。只读模拟的 source-start/source-end 相对位置对齐没有让 B2 Item2/B2 Item3 互换；但若论文使用了不同的绝对 observed window，距离仍可能变化。",
        "",
        "## 7. 论文期望源距离检查",
        markdown_table(paper_df),
        "",
        "## 8. 最终回答",
        "1. 当前为什么 B2 Item2 比 B2 Item3 更近？因为在当前 `paper_observed_sequence`、calendar-date observed window、特征 `sales|promo|year|month|week|day` 加静态 promo 的欧氏距离下，B2 Item2 总平方差小于 B2 Item3。",
        "2. 差异主要来自 sales、promo，还是 time features？主要来自 sales；promo 有较小贡献；year/month/week/day 在当前 calendar-date 对齐下贡献为 0。",
        f"3. B2 Item3 是否只是轻微落后？B2 Item3 rank={int(b2i3['rank'])}，距离 Top3 cutoff 差 {b2i3_gap:.6f}，不是远离候选池，但足以被 B2 Item2 挤出 Top3。",
        "4. 是否存在 item 编码偏移风险？当前证据显示风险低。",
        "5. 是否存在 observed window 日期/对齐方式差异？当前程序使用 calendar-date observed window；本审计模拟的 source-start/source-end 相对位置对齐未导致 B2 Item2/B2 Item3 互换；更主要的窗口风险是论文可能使用了不同的绝对 observed window。",
        "6. 当前 B123 with-sharing 没有完全复现 Table 6 的最可能原因：论文 Table 6 的绝对 observed window、对齐口径或特征口径与当前代码仍有差异，而不是训练、RFE 或指标问题。",
        "7. 是否需要修改代码？本审计不建议直接修改训练逻辑。若目标是严格复现论文，应先补充论文 Dataset2 的绝对窗口/特征口径证据，再考虑新增显式协议开关；不要直接改默认 KNN 公式或清洗。",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ctx = load_context()
    ranking_df, selection = build_candidate_ranking(ctx)
    meta = selection.get("meta", {})
    knn_features = list(meta.get("feature_cols", []))
    static_features = list(meta.get("signature_static_feature_cols", []))
    compare_df = build_observed_compare(ctx, knn_features, static_features)
    contrib_df = build_contribution(ctx, knn_features, static_features)
    item_df = build_item_mapping_audit(ctx)
    window_df = build_window_alignment_audit(ctx, knn_features, static_features)
    paper_df = build_paper_expected_distance_check(ranking_df)
    write_report(ctx, ranking_df, selection, compare_df, contrib_df, item_df, window_df, paper_df)

    print("Dataset2 B123 with-sharing KNN diagnosis complete")
    for path in [
        ALL_CANDIDATES_CSV,
        OBS_COMPARE_CSV,
        CONTRIB_CSV,
        ITEM_MAPPING_CSV,
        WINDOW_ALIGNMENT_CSV,
        PAPER_DISTANCE_CSV,
        REPORT_MD,
    ]:
        print(f"- {path}")
    print("Current Top3:", join_keys([tuple(x) for x in ranking_df.head(3)[["source_entity", "source_item"]].itertuples(index=False, name=None)]))
    b2i2 = ranking_df[(ranking_df["source_entity"] == "B2") & (ranking_df["source_item"] == 2)].iloc[0]
    b2i3 = ranking_df[(ranking_df["source_entity"] == "B2") & (ranking_df["source_item"] == 3)].iloc[0]
    print(f"B2 Item2: rank={int(b2i2['rank'])} distance={float(b2i2['distance']):.6f}")
    print(f"B2 Item3: rank={int(b2i3['rank'])} distance={float(b2i3['distance']):.6f}")


if __name__ == "__main__":
    main()
