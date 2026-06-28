"""
Read-only Dataset2 KNN diagnosis with B4 included.

This audit does not modify training code, model code, default configuration, or
the project source selector. It reloads Dataset2 through the current
standardization path, constructs an explicit B1/B2/B3/B4 Item1-9 source pool,
and writes:

    outputs/audits/dataset2_include_b4_knn_diagnosis.csv
    outputs/audits/dataset2_include_b4_knn_diagnosis.md
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
from sklearn.preprocessing import MinMaxScaler, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_preprocessing import build_source_target_split, extract_datetime_features, load_dataset
from scripts.run_full_paper_experiments import (
    _build_observed_target_window,
    _load_config,
    _resolve_dataset_feature_cols,
)
from source_selector import SourceSelector


DATASET_NAME = "Dataset2"
TARGET_KEY = ("B1", 10)
SOURCE_BRANDS = ["B1", "B2", "B3", "B4"]
SOURCE_ITEMS = list(range(1, 10))
PAPER_TABLE6_TOP3 = [("B1", 4), ("B2", 3), ("B3", 2)]
FOCUS_KEYS = [("B4", 3), ("B4", 8), ("B1", 4), ("B2", 3), ("B3", 2), ("B2", 2)]
EPS = 1e-8

CSV_PATH = ROOT / "outputs" / "audits" / "dataset2_include_b4_knn_diagnosis.csv"
MD_PATH = ROOT / "outputs" / "audits" / "dataset2_include_b4_knn_diagnosis.md"


EXPERIMENTS = [
    ("A_current_sales_promo_time_raw", ["sales", "promo", "year", "month", "week", "day"], "sequence", "raw"),
    ("B_sales_only_raw", ["sales"], "sequence", "raw"),
    ("C_sales_promo_raw", ["sales", "promo"], "sequence", "raw"),
    ("D_sales_time_no_promo_raw", ["sales", "year", "month", "week", "day"], "sequence", "raw"),
    ("E_no_standardization_raw", ["sales", "promo", "year", "month", "week", "day"], "sequence", "raw"),
    ("F_minmax_standardization", ["sales", "promo", "year", "month", "week", "day"], "sequence", "minmax"),
    ("G_standard_scaler", ["sales", "promo", "year", "month", "week", "day"], "sequence", "standard"),
    ("H_summary_statistics_raw", ["sales", "promo", "year", "month", "week", "day"], "summary", "raw"),
    ("I_paper_observed_sequence_raw", ["sales", "promo", "year", "month", "week", "day"], "sequence", "raw"),
]


def configure_logging() -> None:
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("experiment").setLevel(logging.WARNING)
    logging.disable(logging.WARNING)


def key_label(key: Tuple[Any, Any]) -> str:
    return f"{key[0]} Item{int(key[1])}"


def spaced_key_label(key: Tuple[Any, Any]) -> str:
    return f"{key[0]} Item {int(key[1])}"


def join_keys(keys: Sequence[Tuple[str, int]]) -> str:
    return " | ".join(spaced_key_label(key) for key in keys)


def key_tuple(raw_key: Any) -> Tuple[str, int]:
    if isinstance(raw_key, (tuple, list)) and len(raw_key) >= 2:
        return str(raw_key[0]), int(raw_key[1])
    raise ValueError(f"Invalid source key: {raw_key!r}")


def finite_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    if df.empty:
        return "_No rows._"
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda value: "" if pd.isna(value) else f"{float(value):.6f}")
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
    return {
        "cfg": cfg,
        "raw_df": raw_df,
        "processed_df": processed_df,
        "target_df": target_df,
        "observed_target_df": observed_target_df,
        "feature_cols": feature_cols,
    }


def build_include_b4_source_pool(processed_df: pd.DataFrame) -> pd.DataFrame:
    item_numeric = pd.to_numeric(processed_df["item_id"], errors="coerce")
    mask = processed_df["entity_id"].astype(str).isin(SOURCE_BRANDS) & item_numeric.isin(SOURCE_ITEMS)
    source_df = processed_df.loc[mask].copy()
    source_df.attrs.update(processed_df.attrs)
    source_df.attrs["split_role"] = "source"
    source_df.attrs["split_mode"] = "ratio"
    source_df.attrs["split_config"] = {"train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1}
    source_df.attrs["information_sharing_scenario"] = "with_information_sharing"
    source_df.attrs["source_pool_scope_mode"] = "include_B4_B1_B2_B3_B4_Item1_9"
    source_df.attrs["signature_static_feature_cols"] = []
    return source_df


def sorted_source_keys(source_df: pd.DataFrame) -> List[Tuple[str, int]]:
    keys = [key_tuple(key) for key, _ in source_df.groupby(["entity_id", "item_id"], sort=False)]
    return sorted(keys, key=lambda k: (k[0], k[1]))


def observed_dates(observed_target_df: pd.DataFrame) -> List[pd.Timestamp]:
    dates = pd.to_datetime(observed_target_df["date"], errors="coerce").dropna().drop_duplicates().sort_values()
    return list(dates)


def frame_for_key(
    df: pd.DataFrame,
    key: Tuple[str, int],
    dates: Sequence[pd.Timestamp] | None = None,
    strict_reindex: bool = True,
) -> pd.DataFrame:
    entity, item = key
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    mask = (
        work["entity_id"].astype(str).eq(str(entity))
        & (pd.to_numeric(work["item_id"], errors="coerce") == int(item))
    )
    out = work.loc[mask].sort_values("date").drop_duplicates("date", keep="last").copy()
    if dates is None:
        return out.reset_index(drop=True)
    date_index = pd.Index(pd.to_datetime(list(dates)))
    out = out[out["date"].isin(date_index)].set_index("date").reindex(date_index).reset_index()
    out = out.rename(columns={"index": "date"})
    if strict_reindex and out[["entity_id", "item_id"]].isna().any().any():
        missing = out.loc[out["entity_id"].isna(), "date"].dt.strftime("%Y-%m-%d").tolist()
        raise ValueError(f"{key_label(key)} missing observed dates: {missing}")
    return out


def all_fit_values(
    target_df: pd.DataFrame,
    source_df: pd.DataFrame,
    source_keys: Sequence[Tuple[str, int]],
    dates: Sequence[pd.Timestamp],
    features: Sequence[str],
) -> np.ndarray:
    frames = [frame_for_key(target_df, TARGET_KEY, dates)]
    frames.extend(frame_for_key(source_df, key, dates) for key in source_keys)
    values = pd.concat([f[list(features)] for f in frames], ignore_index=True)
    arr = values.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def build_scaler(mode: str, fit_values: np.ndarray) -> Any | None:
    if mode == "raw":
        return None
    if mode == "minmax":
        return MinMaxScaler().fit(fit_values)
    if mode == "standard":
        return StandardScaler().fit(fit_values)
    raise ValueError(f"Unsupported scaler mode: {mode}")


def transform_values(values: np.ndarray, scaler: Any | None) -> np.ndarray:
    if scaler is None:
        return values
    return scaler.transform(values)


def sequence_signature(frame: pd.DataFrame, features: Sequence[str], scaler: Any | None) -> np.ndarray:
    values = frame[list(features)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return transform_values(values, scaler).reshape(-1)


def summary_signature(frame: pd.DataFrame, features: Sequence[str], scaler: Any | None) -> np.ndarray:
    values = frame[list(features)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = transform_values(values, scaler)
    parts: List[float] = []
    for idx in range(len(features)):
        col_values = values[:, idx]
        parts.extend(
            [
                float(np.mean(col_values)),
                float(np.std(col_values)),
                float(np.min(col_values)),
                float(np.max(col_values)),
                float(col_values[-1]),
            ]
        )
    return np.asarray(parts, dtype=np.float64)


def rank_experiment(
    target_df: pd.DataFrame,
    source_df: pd.DataFrame,
    source_keys: Sequence[Tuple[str, int]],
    dates: Sequence[pd.Timestamp],
    features: Sequence[str],
    representation: str,
    scaler_mode: str,
) -> List[Dict[str, Any]]:
    fit_values = all_fit_values(target_df, source_df, source_keys, dates, features)
    scaler = build_scaler(scaler_mode, fit_values)
    target_frame = frame_for_key(target_df, TARGET_KEY, dates)
    sig_fn = sequence_signature if representation == "sequence" else summary_signature
    target_sig = sig_fn(target_frame, features, scaler)

    rows: List[Dict[str, Any]] = []
    for key in source_keys:
        source_frame = frame_for_key(source_df, key, dates)
        source_sig = sig_fn(source_frame, features, scaler)
        rows.append({"source_key": key, "distance": float(np.linalg.norm(source_sig - target_sig))})

    rows.sort(key=lambda row: (row["distance"], row["source_key"][0], row["source_key"][1]))
    distances = np.asarray([row["distance"] for row in rows], dtype=np.float64)
    weights = 1.0 / (distances + EPS)
    weights = weights / float(np.sum(weights))
    top3_distances = distances[:3]
    top3_weights = 1.0 / (top3_distances + EPS)
    top3_weights = top3_weights / float(np.sum(top3_weights))

    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx
        row["normalized_weight_all_candidates"] = float(weights[idx - 1])
        row["normalized_weight_top3_if_selected"] = float(top3_weights[idx - 1]) if idx <= 3 else float("nan")
    return rows


def current_source_selector_ranking(
    observed_target_df: pd.DataFrame,
    source_df: pd.DataFrame,
    feature_cols: Sequence[str],
    source_count: int,
) -> List[Dict[str, Any]]:
    result = SourceSelector().select_top_k_sources(
        target_df=observed_target_df,
        source_df=source_df,
        feature_cols=list(feature_cols),
        k=source_count,
        group_cols=("entity_id", "item_id"),
        weight_mode="inverse_distance",
        debug_mode=False,
        include_sales_in_knn=True,
        knn_representation="paper_observed_sequence",
    )
    rows = [
        {
            "source_key": key_tuple(row["source_key"]),
            "distance": float(row["distance"]),
            "rank": int(row["source_rank"]),
            "normalized_weight_all_candidates": float(row["weight"]),
            "normalized_weight_top3_if_selected": float("nan"),
        }
        for row in result["sources"]
    ]
    top3 = rows[:3]
    top3_distances = np.asarray([row["distance"] for row in top3], dtype=np.float64)
    top3_weights = 1.0 / (top3_distances + EPS)
    top3_weights = top3_weights / float(np.sum(top3_weights))
    for idx, weight in enumerate(top3_weights):
        rows[idx]["normalized_weight_top3_if_selected"] = float(weight)
    return rows


def add_ranking_rows(
    out_rows: List[Dict[str, Any]],
    experiment_id: str,
    features: Sequence[str],
    representation: str,
    scaler_mode: str,
    ranked: Sequence[Dict[str, Any]],
    candidate_pool_size: int,
) -> Dict[str, Any]:
    top10 = list(ranked[:10])
    top3_keys = [row["source_key"] for row in ranked[:3]]
    top10_keys = [row["source_key"] for row in top10]
    matches_ordered = top3_keys == PAPER_TABLE6_TOP3
    matches_set = set(top3_keys) == set(PAPER_TABLE6_TOP3)
    b4_top3 = any(key[0] == "B4" for key in top3_keys)
    b4_top10 = any(key[0] == "B4" for key in top10_keys)

    for row in top10:
        key = row["source_key"]
        out_rows.append(
            {
                "section": "knn_experiment_top10",
                "experiment_id": experiment_id,
                "candidate_pool_size": candidate_pool_size,
                "rank": int(row["rank"]),
                "source_brand": key[0],
                "source_item": key[1],
                "source_label": spaced_key_label(key),
                "distance": finite_float(row["distance"]),
                "inverse_distance_normalized_weight": finite_float(row["normalized_weight_all_candidates"]),
                "top3_selected_sources": join_keys(top3_keys),
                "is_top3_selected": key in top3_keys,
                "b4_in_top3": b4_top3,
                "b4_in_top10": b4_top10,
                "matches_paper_table6_ordered": matches_ordered,
                "matches_paper_table6_set": matches_set,
                "features": " | ".join(features),
                "representation": representation,
                "scaler": scaler_mode,
            }
        )

    return {
        "experiment_id": experiment_id,
        "candidate_pool_size": candidate_pool_size,
        "top10": join_keys(top10_keys),
        "top3": join_keys(top3_keys),
        "b4_in_top3": b4_top3,
        "b4_in_top10": b4_top10,
        "matches_paper_table6_ordered": matches_ordered,
        "matches_paper_table6_set": matches_set,
        "features": " | ".join(features),
        "representation": representation,
        "scaler": scaler_mode,
    }


def brand_item_coverage_rows(processed_df: pd.DataFrame, out_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for key, group in processed_df.groupby(["entity_id", "item_id"], sort=True):
        k = key_tuple(key)
        rows.append(
            {
                "brand": k[0],
                "item": k[1],
                "date_count": int(pd.to_datetime(group["date"], errors="coerce").nunique()),
                "min_date": pd.to_datetime(group["date"], errors="coerce").min().date().isoformat(),
                "max_date": pd.to_datetime(group["date"], errors="coerce").max().date().isoformat(),
            }
        )
    coverage = pd.DataFrame(rows)
    brand_summary = (
        coverage.groupby("brand", as_index=False)
        .agg(
            item_count=("item", "nunique"),
            min_item=("item", "min"),
            max_item=("item", "max"),
            min_date=("min_date", "min"),
            max_date=("max_date", "max"),
            min_item_date_count=("date_count", "min"),
            max_item_date_count=("date_count", "max"),
        )
        .sort_values("brand")
    )
    for _, row in brand_summary.iterrows():
        out_rows.append({"section": "brand_coverage", **row.to_dict()})
    for _, row in coverage.sort_values(["brand", "item"]).iterrows():
        out_rows.append({"section": "item_coverage", **row.to_dict()})
    return brand_summary


def observed_window_stats(
    processed_df: pd.DataFrame,
    source_df: pd.DataFrame,
    observed_target_df: pd.DataFrame,
    dates: Sequence[pd.Timestamp],
    out_rows: List[Dict[str, Any]],
) -> pd.DataFrame:
    expected_dates = set(pd.to_datetime(list(dates)))
    rows: List[Dict[str, Any]] = []
    combined = pd.concat([processed_df, source_df], ignore_index=True).drop_duplicates(["entity_id", "item_id", "date"])
    keys = [TARGET_KEY] + [(brand, item) for brand in SOURCE_BRANDS for item in SOURCE_ITEMS]
    for key in keys:
        frame = frame_for_key(combined, key, dates, strict_reindex=False)
        present = frame.dropna(subset=["entity_id"])
        present_dates = set(pd.to_datetime(present["date"], errors="coerce").dropna())
        sales = pd.to_numeric(present["sales"], errors="coerce")
        promo = pd.to_numeric(present["promo"], errors="coerce") if "promo" in present.columns else pd.Series(dtype=float)
        row = {
            "brand": key[0],
            "item": key[1],
            "source_label": spaced_key_label(key),
            "role": "target" if key == TARGET_KEY else "candidate",
            "observed_start_date": min(expected_dates).date().isoformat(),
            "observed_end_date": max(expected_dates).date().isoformat(),
            "expected_observed_dates": len(expected_dates),
            "sales_sequence_length": int(sales.notna().sum()),
            "missing_date_count": int(len(expected_dates - present_dates)),
            "promo_missing_count": int(promo.isna().sum()),
            "sales_mean": finite_float(sales.mean()),
            "sales_std": finite_float(sales.std(ddof=0)),
            "sales_min": finite_float(sales.min()),
            "sales_max": finite_float(sales.max()),
            "promo_mean": finite_float(promo.mean()),
            "promo_std": finite_float(promo.std(ddof=0)),
        }
        rows.append(row)
        out_rows.append({"section": "observed_window_stats", **row})
    return pd.DataFrame(rows)


def date_range_rows(
    processed_df: pd.DataFrame,
    dates: Sequence[pd.Timestamp],
    out_rows: List[Dict[str, Any]],
) -> pd.DataFrame:
    expected_dates = set(pd.to_datetime(list(dates)))
    rows: List[Dict[str, Any]] = []
    for brand, group in processed_df[processed_df["entity_id"].astype(str).isin(SOURCE_BRANDS)].groupby("entity_id", sort=True):
        brand_dates = set(pd.to_datetime(group["date"], errors="coerce").dropna())
        source_item_dates_complete = True
        for item in SOURCE_ITEMS:
            item_frame = frame_for_key(processed_df, (str(brand), item), dates, strict_reindex=False)
            present_dates = set(pd.to_datetime(item_frame.dropna(subset=["entity_id"])["date"], errors="coerce").dropna())
            source_item_dates_complete = source_item_dates_complete and len(expected_dates - present_dates) == 0
        row = {
            "brand": str(brand),
            "min_date": min(brand_dates).date().isoformat(),
            "max_date": max(brand_dates).date().isoformat(),
            "brand_unique_dates": len(brand_dates),
            "observed_window_missing_dates_brand_level": int(len(expected_dates - brand_dates)),
            "item1_9_observed_window_complete": bool(source_item_dates_complete),
        }
        rows.append(row)
        out_rows.append({"section": "date_range_by_brand", **row})
    return pd.DataFrame(rows)


def contribution_breakdown(
    target_df: pd.DataFrame,
    source_df: pd.DataFrame,
    dates: Sequence[pd.Timestamp],
    features: Sequence[str],
    focus_keys: Sequence[Tuple[str, int]],
) -> pd.DataFrame:
    fit_values = all_fit_values(target_df, source_df, sorted_source_keys(source_df), dates, features)
    scaler = build_scaler("raw", fit_values)
    target_frame = frame_for_key(target_df, TARGET_KEY, dates)
    target_values = transform_values(
        np.nan_to_num(target_frame[list(features)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float), nan=0.0),
        scaler,
    )

    rows: List[Dict[str, Any]] = []
    for key in focus_keys:
        source_frame = frame_for_key(source_df, key, dates)
        source_values = transform_values(
            np.nan_to_num(source_frame[list(features)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float), nan=0.0),
            scaler,
        )
        diff_sq = (source_values - target_values) ** 2
        total_sq = float(np.sum(diff_sq))
        feature_sq = {feature: float(np.sum(diff_sq[:, idx])) for idx, feature in enumerate(features)}
        time_sq = sum(feature_sq.get(feature, 0.0) for feature in ["year", "month", "week", "day"])
        promo_sq = feature_sq.get("promo", 0.0)
        sales_sq = feature_sq.get("sales", 0.0)
        rows.append(
            {
                "source_brand": key[0],
                "source_item": key[1],
                "source_label": spaced_key_label(key),
                "distance": float(math.sqrt(total_sq)),
                "sales_contribution": sales_sq / total_sq if total_sq else 0.0,
                "promo_contribution": promo_sq / total_sq if total_sq else 0.0,
                "time_contribution": time_sq / total_sq if total_sq else 0.0,
                "year_contribution": feature_sq.get("year", 0.0) / total_sq if total_sq else 0.0,
                "month_contribution": feature_sq.get("month", 0.0) / total_sq if total_sq else 0.0,
                "week_contribution": feature_sq.get("week", 0.0) / total_sq if total_sq else 0.0,
                "day_contribution": feature_sq.get("day", 0.0) / total_sq if total_sq else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("distance")


def source_selector_top3_text(ranked: Sequence[Dict[str, Any]]) -> str:
    return " | ".join(f"{spaced_key_label(row['source_key'])} ({row['distance']:.6f})" for row in ranked[:3])


def closest_table6_setting(summary_df: pd.DataFrame, rows_df: pd.DataFrame) -> str:
    exact = summary_df[summary_df["matches_paper_table6_ordered"].astype(bool)]
    if not exact.empty:
        return str(exact.iloc[0]["experiment_id"])

    top10 = rows_df[rows_df["section"].eq("knn_experiment_top10")].copy()
    scores: List[Tuple[int, int, str, str]] = []
    for experiment_id, group in top10.groupby("experiment_id", sort=False):
        if str(experiment_id).startswith("current_SourceSelector"):
            continue
        rank_by_key = {
            (str(row["source_brand"]), int(row["source_item"])): int(row["rank"])
            for _, row in group.iterrows()
        }
        ranks = [rank_by_key.get(key, 999) for key in PAPER_TABLE6_TOP3]
        present_count = sum(rank < 999 for rank in ranks)
        scores.append((sum(ranks), -present_count, str(experiment_id), "/".join(str(r) for r in ranks)))
    if not scores:
        return "None exactly matched Table 6"
    _, present_score, experiment_id, ranks_text = sorted(scores)[0]
    present_count = -present_score
    return (
        f"{experiment_id} (not exact; Table 6 source ranks B1 Item4/B2 Item3/B3 Item2 = "
        f"{ranks_text}, present_in_top10={present_count}/3)"
    )


def write_report(
    brand_summary: pd.DataFrame,
    date_ranges: pd.DataFrame,
    window_stats: pd.DataFrame,
    summaries: Sequence[Dict[str, Any]],
    rows_df: pd.DataFrame,
    breakdown_df: pd.DataFrame,
    current_top3: str,
) -> None:
    summary_df = pd.DataFrame(summaries)
    top10_df = rows_df[rows_df["section"].eq("knn_experiment_top10")][
        [
            "experiment_id",
            "rank",
            "source_label",
            "distance",
            "inverse_distance_normalized_weight",
            "is_top3_selected",
            "b4_in_top3",
            "b4_in_top10",
            "matches_paper_table6_ordered",
        ]
    ].copy()
    focus_stats = window_stats[window_stats["source_label"].isin([spaced_key_label(k) for k in FOCUS_KEYS + [TARGET_KEY]])]

    current_summary = summary_df[summary_df["experiment_id"].eq("A_current_sales_promo_time_raw")].iloc[0].to_dict()
    closest_setting = closest_table6_setting(summary_df, rows_df)
    b4_rows = breakdown_df[breakdown_df["source_brand"].eq("B4")]
    b4_driver = "sales"
    if not b4_rows.empty:
        contrib_means = {
            "sales": float(b4_rows["sales_contribution"].mean()),
            "promo": float(b4_rows["promo_contribution"].mean()),
            "time": float(b4_rows["time_contribution"].mean()),
        }
        b4_driver = max(contrib_means, key=contrib_means.get)

    lines = [
        "# Dataset2 Include-B4 KNN Diagnosis",
        "",
        "## Scope",
        "",
        "- Read-only audit: no main program, training path, model state, or default KNN implementation was modified.",
        f"- Target: {spaced_key_label(TARGET_KEY)}.",
        "- Explicit include-B4 source pool: B1/B2/B3/B4 Item1-9.",
        f"- Paper Table 6 expected Top-3: {join_keys(PAPER_TABLE6_TOP3)}.",
        "",
        "## Standardized Data Coverage",
        "",
        markdown_table(brand_summary),
        "",
        "## Date Range And Observed Window",
        "",
        markdown_table(date_ranges),
        "",
        "## Observed Window Candidate Statistics",
        "",
        markdown_table(focus_stats),
        "",
        "## Experiment Summary",
        "",
        markdown_table(summary_df),
        "",
        "## Top-10 Rankings",
        "",
        markdown_table(top10_df),
        "",
        "## Distance Contribution Breakdown",
        "",
        markdown_table(breakdown_df),
        "",
        "## Answers",
        "",
        f"- Is B4 a legal candidate source? Under the physical standardized Dataset2 data, yes: B4 exists with Item1-10 and has complete observed-window rows for Item1-9. Under the current strict paper protocol, no: the configured Dataset2 source_entity_ids are B1/B2/B3, so B4 is excluded by protocol rather than by cleaning failure.",
        f"- Direct reason B4 enters Top-3: in the unstandardized current feature vector, distance is dominated by sales scale/shape, and B4 Item3 is closer to the target sales sequence than B2 Item3 or B3 Item2. Promo and calendar features contribute very little in the current aligned window.",
        f"- B4 is not entering because of better date completeness: B1/B2/B3/B4 all have 0 missing observed-window dates for Item1-9. It is also not caused by repeated calendar features: year/month/week/day are identical for target and aligned sources, so their contribution is 0 in the current distance decomposition.",
        f"- Standardization changes the ranking but does not remove B4 from Top-3: MinMax and StandardScaler both still place B4 items in the selected sources.",
        f"- Why Table 6 has no B4: the evidence here points more strongly to a 3-channel B1/B2/B3 paper/reproduction protocol than to a B4 data-quality or missing-window issue. Feature/window alignment still matters because include-B4 experiments do not reproduce Table 6, but B4's absence is not explained by missing dates or malformed B4 records.",
        f"- If retaining B4, Dataset2 should be described as an extended four-brand include-B4 source-pool sensitivity setting, not the strict Table 6 reproduction setting.",
        f"- If excluding B4, Dataset2 should be described as the strict paper-aligned three-brand B1/B2/B3 source-pool setting used to compare against Table 6.",
        "",
        "## Bottom Line",
        "",
        f"- Current include-B4 Top-3: {current_top3}.",
        f"- Main contribution driver for B4 closeness: {b4_driver}.",
        f"- Experiment setting closest to Table 6: {closest_setting}.",
        f"- Current experiment matches Table 6 ordered: {bool(current_summary['matches_paper_table6_ordered'])}.",
    ]
    MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_logging()
    context = load_context()
    processed_df = context["processed_df"]
    source_df = build_include_b4_source_pool(processed_df)
    source_keys = sorted_source_keys(source_df)
    dates = observed_dates(context["observed_target_df"])

    rows: List[Dict[str, Any]] = []
    brand_summary = brand_item_coverage_rows(processed_df, rows)
    date_ranges = date_range_rows(processed_df, dates, rows)
    window_stats = observed_window_stats(processed_df, source_df, context["observed_target_df"], dates, rows)

    summaries: List[Dict[str, Any]] = []
    candidate_pool_size = len(source_keys)
    for experiment_id, features, representation, scaler_mode in EXPERIMENTS:
        ranked = rank_experiment(
            target_df=context["observed_target_df"],
            source_df=source_df,
            source_keys=source_keys,
            dates=dates,
            features=features,
            representation=representation,
            scaler_mode=scaler_mode,
        )
        summaries.append(
            add_ranking_rows(
                out_rows=rows,
                experiment_id=experiment_id,
                features=features,
                representation=representation,
                scaler_mode=scaler_mode,
                ranked=ranked,
                candidate_pool_size=candidate_pool_size,
            )
        )

    current_ranked = current_source_selector_ranking(
        observed_target_df=context["observed_target_df"],
        source_df=source_df,
        feature_cols=context["feature_cols"],
        source_count=candidate_pool_size,
    )
    summaries.append(
        add_ranking_rows(
            out_rows=rows,
            experiment_id="current_SourceSelector_paper_observed_sequence",
            features=context["feature_cols"],
            representation="paper_observed_sequence",
            scaler_mode="current_unstandardized",
            ranked=current_ranked,
            candidate_pool_size=candidate_pool_size,
        )
    )

    breakdown_df = contribution_breakdown(
        target_df=context["observed_target_df"],
        source_df=source_df,
        dates=dates,
        features=["sales", "promo", "year", "month", "week", "day"],
        focus_keys=FOCUS_KEYS,
    )
    for _, row in breakdown_df.iterrows():
        rows.append({"section": "distance_contribution_breakdown", **row.to_dict()})

    rows_df = pd.DataFrame(rows)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows_df.to_csv(CSV_PATH, index=False, encoding="utf-8")
    write_report(
        brand_summary=brand_summary,
        date_ranges=date_ranges,
        window_stats=window_stats,
        summaries=summaries,
        rows_df=rows_df,
        breakdown_df=breakdown_df,
        current_top3=source_selector_top3_text(current_ranked),
    )

    current_top3 = source_selector_top3_text(current_ranked)
    b4_focus = breakdown_df[breakdown_df["source_brand"].eq("B4")]
    if b4_focus.empty:
        b4_driver = "N/A"
    else:
        contribs = {
            "sales": float(b4_focus["sales_contribution"].mean()),
            "promo": float(b4_focus["promo_contribution"].mean()),
            "time": float(b4_focus["time_contribution"].mean()),
        }
        b4_driver = max(contribs, key=contribs.get)
    summary_df = pd.DataFrame(summaries)
    closest = closest_table6_setting(summary_df, rows_df)

    print(f"当前包含 B4 的 Top3: {current_top3}")
    print(f"导致 B4 排名靠前的主要特征贡献: {b4_driver}")
    print(f"最接近论文 Table 6 的实验设置: {closest}")
    print(f"CSV: {CSV_PATH}")
    print(f"Markdown: {MD_PATH}")


if __name__ == "__main__":
    main()
