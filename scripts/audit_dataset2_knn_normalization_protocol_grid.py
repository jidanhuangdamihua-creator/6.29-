"""
Dataset2 KNN normalization protocol grid audit.

Pure read-only KNN audit. No model training and no changes to cleaning, split,
source-pool rules, KNN formula, RFE, or metrics.
"""

from __future__ import annotations

import copy
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
)


DATASET = "Dataset2"
TARGET_KEY = ("B1", 10)
SOURCE_POOLS = {
    "B123": ["B1", "B2", "B3"],
    "B1234": ["B1", "B2", "B3", "B4"],
}
SHARING = ["no-sharing", "with-sharing"]
SOURCE_ITEMS = list(range(1, 10))
FEATURE_SETS = {
    "sales_only": ["sales"],
    "sales_promo": ["sales", "promo"],
    "sales_promo_time": ["sales", "promo", "year", "month", "week", "day"],
    "sales_promo_exclude_time": ["sales", "promo"],
}
NORMALIZATIONS = [
    "raw",
    "global_minmax",
    "per_series_minmax",
    "per_brand_minmax",
    "observed_window_minmax",
    "zscore_per_series",
]
STATIC_PROMO = [True, False]
PAPER_TABLE6_TOP3 = [("B1", 4), ("B2", 3), ("B3", 2)]
EPS = 1e-8

OUT_DIR = ROOT / "outputs" / "audits"
SUMMARY_CSV = OUT_DIR / "dataset2_knn_normalization_protocol_grid_summary.csv"
TOP10_CSV = OUT_DIR / "dataset2_knn_normalization_protocol_grid_top10.csv"
FOCUS_CSV = OUT_DIR / "dataset2_knn_normalization_protocol_grid_b2_item2_vs_item3.csv"
REPORT_MD = OUT_DIR / "dataset2_knn_normalization_protocol_grid.md"


def key_label(key: Tuple[str, int]) -> str:
    return f"{key[0]} Item{int(key[1])}"


def join_keys(keys: Sequence[Tuple[str, int]]) -> str:
    return "|".join(key_label(k) for k in keys)


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


def load_context() -> Dict[str, Any]:
    cfg = copy.deepcopy(_load_config())
    cfg.setdefault("paper_reproduction", {})
    cfg["paper_reproduction"]["strict_paper_mode"] = True
    cfg["paper_reproduction"]["paper_strict_mode"] = True
    raw_df = load_dataset(DATASET, cfg["dataset_paths"][DATASET])
    processed_df = extract_datetime_features(raw_df)
    source_df, target_df = build_source_target_split(processed_df, cfg)
    observed_target_df = _build_observed_target_window(target_df)
    return {
        "cfg": cfg,
        "processed_df": processed_df,
        "base_source_df": source_df,
        "target_df": target_df,
        "observed_target_df": observed_target_df,
    }


def build_effective_source_pool(processed_df: pd.DataFrame, source_pool: str, sharing: str) -> pd.DataFrame:
    entities = SOURCE_POOLS[source_pool]
    item_values = pd.to_numeric(processed_df["item_id"], errors="coerce")
    mask = processed_df["entity_id"].astype(str).isin(entities) & item_values.isin(SOURCE_ITEMS)
    if sharing == "no-sharing":
        mask = mask & (processed_df["entity_id"].astype(str) == TARGET_KEY[0])
    source_df = processed_df.loc[mask].copy()
    return source_df.sort_values(["entity_id", "item_id", "date"]).reset_index(drop=True)


def aligned_series(df: pd.DataFrame, key: Tuple[str, int], dates: Sequence[pd.Timestamp]) -> pd.DataFrame:
    work = df[
        (df["entity_id"].astype(str) == str(key[0]))
        & (pd.to_numeric(df["item_id"], errors="coerce") == int(key[1]))
    ].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work[work["date"].isin(pd.to_datetime(pd.Index(dates)))].sort_values("date").reset_index(drop=True)
    return work


def full_series(df: pd.DataFrame, key: Tuple[str, int]) -> pd.DataFrame:
    work = df[
        (df["entity_id"].astype(str) == str(key[0]))
        & (pd.to_numeric(df["item_id"], errors="coerce") == int(key[1]))
    ].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    return work.sort_values("date").reset_index(drop=True)


def candidate_keys(source_df: pd.DataFrame) -> List[Tuple[str, int]]:
    keys: List[Tuple[str, int]] = []
    for key, _ in source_df.groupby(["entity_id", "item_id"], sort=False):
        keys.append((str(key[0]), int(key[1])))
    return keys


def fit_transform_frame(
    frame: pd.DataFrame,
    key: Tuple[str, int],
    feature_cols: Sequence[str],
    static_promo: bool,
    normalization: str,
    context: Dict[str, Any],
) -> pd.DataFrame:
    out = frame.copy()
    scale_cols = list(dict.fromkeys(list(feature_cols) + (["promo"] if static_promo else [])))
    if normalization == "raw":
        return out

    if normalization == "global_minmax":
        scaler = context["global_minmax"]
        out[scale_cols] = scaler.transform(out[scale_cols])
    elif normalization == "observed_window_minmax":
        scaler = context["observed_window_minmax"]
        out[scale_cols] = scaler.transform(out[scale_cols])
    elif normalization == "per_series_minmax":
        scaler = context["per_series_minmax"][key]
        out[scale_cols] = scaler.transform(out[scale_cols])
    elif normalization == "per_brand_minmax":
        scaler = context["per_brand_minmax"][str(key[0])]
        out[scale_cols] = scaler.transform(out[scale_cols])
    elif normalization == "zscore_per_series":
        scaler = context["zscore_per_series"][key]
        out[scale_cols] = scaler.transform(out[scale_cols])
    else:
        raise ValueError(f"Unknown normalization: {normalization}")
    return out


def vector_from_frame(frame: pd.DataFrame, feature_cols: Sequence[str], static_promo: bool) -> np.ndarray:
    values = frame[list(feature_cols)].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=float).reshape(-1)
    if static_promo:
        static_value = float(pd.to_numeric(frame["promo"], errors="coerce").dropna().iloc[-1])
        values = np.concatenate([values, np.asarray([static_value], dtype=float)])
    return values


def make_scaler(df: pd.DataFrame, cols: Sequence[str], kind: str):
    scaler = MinMaxScaler() if kind == "minmax" else StandardScaler()
    scaler.fit(df[list(cols)])
    return scaler


def build_normalization_context(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    observed_target_df: pd.DataFrame,
    keys: Sequence[Tuple[str, int]],
    feature_cols: Sequence[str],
    static_promo: bool,
) -> Dict[str, Any]:
    dates = observed_target_df["date"].tolist()
    scale_cols = list(dict.fromkeys(list(feature_cols) + (["promo"] if static_promo else [])))
    target_key = TARGET_KEY
    all_keys = [target_key] + list(keys)

    full_by_key: Dict[Tuple[str, int], pd.DataFrame] = {target_key: full_series(target_df, target_key)}
    observed_by_key: Dict[Tuple[str, int], pd.DataFrame] = {target_key: aligned_series(observed_target_df, target_key, dates)}
    for key in keys:
        full_by_key[key] = full_series(source_df, key)
        observed_by_key[key] = aligned_series(source_df, key, dates)

    global_fit = pd.concat([full_by_key[key] for key in all_keys], ignore_index=True)
    observed_fit = pd.concat([observed_by_key[key] for key in all_keys], ignore_index=True)

    per_series_minmax = {key: make_scaler(full_by_key[key], scale_cols, "minmax") for key in all_keys}
    zscore_per_series = {key: make_scaler(full_by_key[key], scale_cols, "zscore") for key in all_keys}

    brand_frames: Dict[str, List[pd.DataFrame]] = {}
    for key in all_keys:
        brand_frames.setdefault(str(key[0]), []).append(full_by_key[key])
    per_brand_minmax = {
        brand: make_scaler(pd.concat(frames, ignore_index=True), scale_cols, "minmax")
        for brand, frames in brand_frames.items()
    }

    return {
        "scale_cols": scale_cols,
        "full_by_key": full_by_key,
        "observed_by_key": observed_by_key,
        "global_minmax": make_scaler(global_fit, scale_cols, "minmax"),
        "observed_window_minmax": make_scaler(observed_fit, scale_cols, "minmax"),
        "per_series_minmax": per_series_minmax,
        "per_brand_minmax": per_brand_minmax,
        "zscore_per_series": zscore_per_series,
    }


def compute_protocol(
    source_pool: str,
    sharing: str,
    feature_set_name: str,
    feature_cols: Sequence[str],
    normalization: str,
    static_promo: bool,
    ctx: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    source_df = build_effective_source_pool(ctx["processed_df"], source_pool, sharing)
    observed_target_df = ctx["observed_target_df"]
    dates = observed_target_df["date"].tolist()
    keys = candidate_keys(source_df)
    norm_ctx = build_normalization_context(source_df, ctx["target_df"], observed_target_df, keys, feature_cols, static_promo)

    target_obs = aligned_series(observed_target_df, TARGET_KEY, dates)
    target_norm = fit_transform_frame(target_obs, TARGET_KEY, feature_cols, static_promo, normalization, norm_ctx)
    target_vec = vector_from_frame(target_norm, feature_cols, static_promo)

    rows = []
    for key in keys:
        source_obs = aligned_series(source_df, key, dates)
        source_norm = fit_transform_frame(source_obs, key, feature_cols, static_promo, normalization, norm_ctx)
        source_vec = vector_from_frame(source_norm, feature_cols, static_promo)
        distance = float(np.linalg.norm(source_vec - target_vec))
        rows.append(
            {
                "source_pool": source_pool,
                "sharing": sharing,
                "feature_set": feature_set_name,
                "feature_cols": pipe(feature_cols),
                "normalization": normalization,
                "static_promo": bool(static_promo),
                "source_entity": key[0],
                "source_item": key[1],
                "source_label": key_label(key),
                "distance": distance,
            }
        )
    rows = sorted(rows, key=lambda r: r["distance"])
    weights_raw = [1.0 / (row["distance"] + EPS) for row in rows]
    denom = sum(weights_raw)
    for idx, row in enumerate(rows):
        row["rank"] = idx + 1
        row["raw_inverse_distance_weight"] = weights_raw[idx]
        row["normalized_weight"] = weights_raw[idx] / denom if denom else np.nan
        row["is_top3"] = idx < 3
        row["is_top10"] = idx < 10
        key = (row["source_entity"], row["source_item"])
        row["is_paper_expected"] = key in PAPER_TABLE6_TOP3
        row["is_B2_Item2"] = key == ("B2", 2)
        row["is_B2_Item3"] = key == ("B2", 3)

    top3_keys = [(row["source_entity"], int(row["source_item"])) for row in rows[:3]]
    b2_item2 = next((row for row in rows if row["source_entity"] == "B2" and row["source_item"] == 2), None)
    b2_item3 = next((row for row in rows if row["source_entity"] == "B2" and row["source_item"] == 3), None)
    summary = {
        "source_pool": source_pool,
        "sharing": sharing,
        "feature_set": feature_set_name,
        "feature_cols": pipe(feature_cols),
        "normalization": normalization,
        "static_promo": bool(static_promo),
        "candidate_pool_size": len(keys),
        "top1": key_label(top3_keys[0]) if top3_keys else "",
        "top3": join_keys(top3_keys),
        "top10": join_keys([(row["source_entity"], int(row["source_item"])) for row in rows[:10]]),
        "matches_paper_table6_ordered": top3_keys == PAPER_TABLE6_TOP3,
        "matches_paper_table6_set": set(top3_keys) == set(PAPER_TABLE6_TOP3),
        "paper_expected_overlap_top3": len(set(top3_keys) & set(PAPER_TABLE6_TOP3)),
        "b4_in_top3": any(key[0] == "B4" for key in top3_keys),
        "b2_item2_rank": int(b2_item2["rank"]) if b2_item2 else np.nan,
        "b2_item2_distance": float(b2_item2["distance"]) if b2_item2 else np.nan,
        "b2_item3_rank": int(b2_item3["rank"]) if b2_item3 else np.nan,
        "b2_item3_distance": float(b2_item3["distance"]) if b2_item3 else np.nan,
        "b2_item3_beats_b2_item2": bool(b2_item3 and b2_item2 and b2_item3["distance"] < b2_item2["distance"]),
    }
    focus = [dict(row) for row in rows if row["source_label"] in {"B2 Item2", "B2 Item3"}]
    return summary, rows[:10], focus


def run_grid(ctx: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries: List[Dict[str, Any]] = []
    top10_rows: List[Dict[str, Any]] = []
    focus_rows: List[Dict[str, Any]] = []
    for source_pool in SOURCE_POOLS:
        for sharing in SHARING:
            for feature_set_name, feature_cols in FEATURE_SETS.items():
                for normalization in NORMALIZATIONS:
                    for static_promo in STATIC_PROMO:
                        summary, top10, focus = compute_protocol(
                            source_pool=source_pool,
                            sharing=sharing,
                            feature_set_name=feature_set_name,
                            feature_cols=feature_cols,
                            normalization=normalization,
                            static_promo=static_promo,
                            ctx=ctx,
                        )
                        summaries.append(summary)
                        top10_rows.extend(top10)
                        focus_rows.extend(focus)
    return pd.DataFrame(summaries), pd.DataFrame(top10_rows), pd.DataFrame(focus_rows)


def write_report(summary_df: pd.DataFrame, top10_df: pd.DataFrame, focus_df: pd.DataFrame) -> None:
    paper_matches = summary_df[summary_df["matches_paper_table6_ordered"]].copy()
    best_overlap = summary_df.sort_values(
        ["paper_expected_overlap_top3", "matches_paper_table6_ordered", "matches_paper_table6_set"],
        ascending=[False, False, False],
    ).head(20)
    b2_beats = summary_df[summary_df["b2_item3_beats_b2_item2"]].copy()
    top3_counts = (
        summary_df.groupby(["top3"], as_index=False)
        .size()
        .rename(columns={"size": "protocol_count"})
        .sort_values("protocol_count", ascending=False)
        .head(20)
    )
    lines = [
        "# Dataset2 KNN Normalization Protocol Grid",
        "",
        "## Scope",
        "- Dataset2 only; target fixed at B1 Item10.",
        "- source_pool: B123 / B1234.",
        "- sharing: no-sharing / with-sharing. For no-sharing, current Dataset2 same-brand rule leaves B1 Item1-9 as the effective pool.",
        "- feature_set grid: sales_only, sales_promo, sales_promo_time, sales_promo_exclude_time.",
        "- normalization grid: raw, global_minmax, per_series_minmax, per_brand_minmax, observed_window_minmax, zscore_per_series.",
        "- static_promo grid: include / exclude.",
        "",
        "## Normalization Definitions",
        "- raw: no transform.",
        "- global_minmax: fit MinMax on current effective source pool full history + target full target window.",
        "- per_series_minmax: fit MinMax separately for each (entity,item) full history.",
        "- per_brand_minmax: fit MinMax by brand/entity full history.",
        "- observed_window_minmax: fit MinMax on target observed window + candidate observed-window rows.",
        "- zscore_per_series: fit StandardScaler separately for each (entity,item) full history.",
        "- static_promo include adds one extra `static:promo` component using the last observed-window promo value after the selected normalization.",
        "",
        "## Ordered Paper Table 6 Matches",
        markdown_table(paper_matches, max_rows=50),
        "",
        "## Best Paper-Overlap Protocols",
        markdown_table(best_overlap, ["source_pool", "sharing", "feature_set", "normalization", "static_promo", "top3", "paper_expected_overlap_top3", "matches_paper_table6_ordered", "matches_paper_table6_set", "b2_item2_rank", "b2_item3_rank"], max_rows=20),
        "",
        "## Protocols Where B2 Item3 Beats B2 Item2",
        markdown_table(b2_beats, ["source_pool", "sharing", "feature_set", "normalization", "static_promo", "top3", "b2_item2_rank", "b2_item2_distance", "b2_item3_rank", "b2_item3_distance"], max_rows=80),
        "",
        "## Most Frequent Top3 Outcomes",
        markdown_table(top3_counts, max_rows=20),
        "",
        "## Output Files",
        f"- `{SUMMARY_CSV}`",
        f"- `{TOP10_CSV}`",
        f"- `{FOCUS_CSV}`",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ctx = load_context()
    summary_df, top10_df, focus_df = run_grid(ctx)
    summary_df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8")
    top10_df.to_csv(TOP10_CSV, index=False, encoding="utf-8")
    focus_df.to_csv(FOCUS_CSV, index=False, encoding="utf-8")
    write_report(summary_df, top10_df, focus_df)
    print("Dataset2 KNN normalization protocol grid complete")
    for path in [SUMMARY_CSV, TOP10_CSV, FOCUS_CSV, REPORT_MD]:
        print(f"- {path}")
    print(f"protocols={len(summary_df)} ordered_paper_matches={int(summary_df['matches_paper_table6_ordered'].sum())}")
    print(f"B2 Item3 beats B2 Item2 protocols={int(summary_df['b2_item3_beats_b2_item2'].sum())}")


if __name__ == "__main__":
    main()
