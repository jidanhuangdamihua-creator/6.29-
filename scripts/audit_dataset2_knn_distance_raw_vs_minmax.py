"""
Read-only Dataset2 KNN raw-vs-MinMax distance audit.

This audit verifies what values the current SourceSelector uses for Dataset2
B123 with-sharing KNN and compares that raw ranking with explicit audit-only
MinMax variants. It does not modify KNN, cleaning, split, RFE, training, or
metrics.
"""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

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


DATASET = "Dataset2"
TARGET_KEY = ("B1", 10)
SOURCE_ENTITIES = ["B1", "B2", "B3"]
SOURCE_ITEMS = list(range(1, 10))
FOCUS_KEYS = [("B1", 10), ("B2", 2), ("B2", 3)]
OUT_DIR = ROOT / "outputs" / "audits"
VECTOR_CSV = OUT_DIR / "dataset2_knn_actual_input_vectors_b1item10_b2item2_b2item3.csv"
RANGE_CSV = OUT_DIR / "dataset2_knn_input_value_ranges.csv"
RANK_COMPARE_CSV = OUT_DIR / "dataset2_knn_raw_vs_minmax_distance_ranking.csv"
SCALER_CSV = OUT_DIR / "dataset2_knn_minmax_scaler_audit.csv"
REPORT_MD = OUT_DIR / "dataset2_knn_distance_raw_vs_minmax_audit.md"
EPS = 1e-8


def key_label(key: Tuple[str, int]) -> str:
    return f"{key[0]} Item{int(key[1])}"


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
        vals = [str(row[col]).replace("|", "\\|") for col in work.columns]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def load_context() -> Dict[str, Any]:
    cfg = copy.deepcopy(_load_config())
    cfg.setdefault("paper_reproduction", {})
    cfg["paper_reproduction"]["strict_paper_mode"] = True
    cfg["paper_reproduction"]["paper_strict_mode"] = True
    protocol = cfg["paper_reproduction"]

    raw_df = load_dataset(DATASET, cfg["dataset_paths"][DATASET])
    processed_df = extract_datetime_features(raw_df)
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
    selector = SourceSelector()
    selection = selector.select_top_k_sources(
        target_df=observed_target_df,
        source_df=source_df,
        feature_cols=feature_cols,
        k=int(source_df.groupby(["entity_id", "item_id"], sort=False).ngroups),
        group_cols=("entity_id", "item_id"),
        weight_mode="inverse_distance",
        debug_mode=False,
        include_sales_in_knn=True,
        knn_representation="paper_observed_sequence",
    )
    return {
        "cfg": cfg,
        "processed_df": processed_df,
        "source_df": source_df,
        "target_df": target_df,
        "observed_target_df": observed_target_df,
        "feature_cols": feature_cols,
        "selection": selection,
    }


def aligned_group(ctx: Dict[str, Any], key: Tuple[str, int]) -> pd.DataFrame:
    df = ctx["observed_target_df"] if key == TARGET_KEY else ctx["source_df"]
    dates = pd.to_datetime(ctx["observed_target_df"]["date"])
    work = df[
        (df["entity_id"].astype(str) == str(key[0]))
        & (pd.to_numeric(df["item_id"], errors="coerce") == int(key[1]))
    ].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work[work["date"].isin(dates)].sort_values("date").reset_index(drop=True)
    return work


def vector_rows_for_group(
    group: pd.DataFrame,
    key: Tuple[str, int],
    feature_cols: Sequence[str],
    static_cols: Sequence[str],
    value_mode: str,
    scaler: MinMaxScaler | None = None,
    scaler_features: Sequence[str] | None = None,
) -> List[Dict[str, Any]]:
    work = group.copy().sort_values("date").reset_index(drop=True)
    transform_features = list(scaler_features or feature_cols)
    if scaler is not None:
        transformed = scaler.transform(work[transform_features])
        for idx, feature in enumerate(transform_features):
            work[feature] = transformed[:, idx]
    rows: List[Dict[str, Any]] = []
    pos = 0
    for t_idx, row in work.iterrows():
        for feature in feature_cols:
            rows.append(
                {
                    "value_mode": value_mode,
                    "series": key_label(key),
                    "entity": key[0],
                    "item": key[1],
                    "vector_position": pos,
                    "component_type": "time_series",
                    "time_step": t_idx + 1,
                    "date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
                    "feature": feature,
                    "value": float(row[feature]),
                }
            )
            pos += 1
    for feature in static_cols:
        value = float(pd.to_numeric(work[feature], errors="coerce").dropna().iloc[-1])
        rows.append(
            {
                "value_mode": value_mode,
                "series": key_label(key),
                "entity": key[0],
                "item": key[1],
                "vector_position": pos,
                "component_type": "static_last_value",
                "time_step": "static",
                "date": "",
                "feature": f"static:{feature}",
                "value": value,
            }
        )
        pos += 1
    return rows


def fit_scaler(frame: pd.DataFrame, feature_cols: Sequence[str]) -> MinMaxScaler:
    scaler = MinMaxScaler()
    scaler.fit(frame[list(feature_cols)])
    return scaler


def build_scalers(ctx: Dict[str, Any], knn_features: Sequence[str]) -> Tuple[Dict[str, MinMaxScaler], pd.DataFrame]:
    observed_target = ctx["observed_target_df"].copy()
    source = ctx["source_df"].copy()
    observed_dates = pd.to_datetime(observed_target["date"])
    source_observed = source[source["date"].isin(observed_dates)].copy()
    full_history = pd.concat([source, ctx["target_df"]], ignore_index=True)
    all_observed = pd.concat([source_observed, observed_target], ignore_index=True)

    scopes = {
        "audit_minmax_fit_target_observed_only": observed_target,
        "audit_minmax_fit_all_candidates_observed_plus_target": all_observed,
        "audit_minmax_fit_full_history_source_plus_target": full_history,
    }
    scalers = {name: fit_scaler(df, knn_features) for name, df in scopes.items()}
    rows = []
    for name, df in scopes.items():
        scaler = scalers[name]
        for idx, feature in enumerate(knn_features):
            rows.append(
                {
                    "value_mode": name,
                    "scaler_used_by_current_knn": False,
                    "fit_scope": name.replace("audit_minmax_fit_", ""),
                    "fit_rows": int(len(df)),
                    "fit_unique_series": int(df.groupby(["entity_id", "item_id"], sort=False).ngroups),
                    "fit_min_date": pd.Timestamp(df["date"].min()).strftime("%Y-%m-%d"),
                    "fit_max_date": pd.Timestamp(df["date"].max()).strftime("%Y-%m-%d"),
                    "feature": feature,
                    "data_min": float(scaler.data_min_[idx]),
                    "data_max": float(scaler.data_max_[idx]),
                    "data_range": float(scaler.data_range_[idx]),
                }
            )
    rows.insert(
        0,
        {
            "value_mode": "current_raw_no_minmax",
            "scaler_used_by_current_knn": False,
            "fit_scope": "none: SourceSelector does not call normalize_features or MinMaxScaler",
            "fit_rows": 0,
            "fit_unique_series": 0,
            "fit_min_date": "",
            "fit_max_date": "",
            "feature": "",
            "data_min": np.nan,
            "data_max": np.nan,
            "data_range": np.nan,
        },
    )
    return scalers, pd.DataFrame(rows)


def vector_distance(
    target_group: pd.DataFrame,
    source_group: pd.DataFrame,
    feature_cols: Sequence[str],
    static_cols: Sequence[str],
    scaler: MinMaxScaler | None = None,
) -> float:
    tgt = target_group.copy().sort_values("date").reset_index(drop=True)
    src = source_group.copy().sort_values("date").reset_index(drop=True)
    if scaler is not None:
        tgt_vals = scaler.transform(tgt[list(feature_cols)])
        src_vals = scaler.transform(src[list(feature_cols)])
        for idx, feature in enumerate(feature_cols):
            tgt[feature] = tgt_vals[:, idx]
            src[feature] = src_vals[:, idx]
    parts = []
    for feature in feature_cols:
        parts.extend((pd.to_numeric(src[feature], errors="coerce").fillna(0).to_numpy(dtype=float) - pd.to_numeric(tgt[feature], errors="coerce").fillna(0).to_numpy(dtype=float)).tolist())
    for feature in static_cols:
        parts.append(float(pd.to_numeric(src[feature], errors="coerce").dropna().iloc[-1]) - float(pd.to_numeric(tgt[feature], errors="coerce").dropna().iloc[-1]))
    return float(np.linalg.norm(np.asarray(parts, dtype=float)))


def build_outputs(ctx: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meta = ctx["selection"]["meta"]
    knn_features = list(meta["feature_cols"])
    static_cols = list(meta.get("signature_static_feature_cols", []))
    scalers, scaler_df = build_scalers(ctx, knn_features)

    groups = {key: aligned_group(ctx, key) for key in FOCUS_KEYS}
    vector_rows: List[Dict[str, Any]] = []
    for key in FOCUS_KEYS:
        vector_rows.extend(vector_rows_for_group(groups[key], key, knn_features, static_cols, "current_raw_no_minmax"))
        for mode, scaler in scalers.items():
            vector_rows.extend(vector_rows_for_group(groups[key], key, knn_features, static_cols, mode, scaler=scaler, scaler_features=knn_features))
    vector_df = pd.DataFrame(vector_rows)

    range_rows = []
    for mode, sub in vector_df.groupby("value_mode", sort=False):
        for feature, g in sub.groupby("feature", sort=False):
            range_rows.append(
                {
                    "value_mode": mode,
                    "feature": feature,
                    "min_value": float(g["value"].min()),
                    "max_value": float(g["value"].max()),
                    "mean_value": float(g["value"].mean()),
                    "std_value": float(g["value"].std(ddof=0)),
                    "row_count": int(len(g)),
                }
            )
    range_df = pd.DataFrame(range_rows)

    source_keys = []
    for _, group in ctx["source_df"].groupby(["entity_id", "item_id"], sort=False):
        key = (str(group["entity_id"].iloc[0]), int(group["item_id"].iloc[0]))
        source_keys.append(key)
    rows = []
    raw_rank_by_key = {tuple(row["source_key"]): int(row["source_rank"]) for row in ctx["selection"]["sources"]}
    raw_distance_by_key = {tuple(row["source_key"]): float(row["distance"]) for row in ctx["selection"]["sources"]}
    all_modes: Dict[str, MinMaxScaler | None] = {"current_raw_no_minmax": None, **scalers}
    for mode, scaler in all_modes.items():
        dist_rows = []
        for key in source_keys:
            src_group = aligned_group(ctx, key)
            dist_rows.append(
                {
                    "value_mode": mode,
                    "source_entity": key[0],
                    "source_item": key[1],
                    "source_label": key_label(key),
                    "distance": vector_distance(groups[TARGET_KEY], src_group, knn_features, static_cols, scaler=scaler),
                }
            )
        dist_rows = sorted(dist_rows, key=lambda r: r["distance"])
        raw_weights = [1.0 / (r["distance"] + EPS) for r in dist_rows]
        denom = sum(raw_weights)
        for idx, row in enumerate(dist_rows):
            key = (row["source_entity"], row["source_item"])
            row["rank"] = idx + 1
            row["raw_inverse_distance_weight"] = raw_weights[idx]
            row["normalized_weight"] = raw_weights[idx] / denom
            row["current_raw_rank"] = raw_rank_by_key.get(key)
            row["current_raw_distance"] = raw_distance_by_key.get(key)
            row["rank_changed_vs_current_raw"] = row["rank"] != row["current_raw_rank"]
            rows.append(row)
    ranking_df = pd.DataFrame(rows)

    vector_df.to_csv(VECTOR_CSV, index=False, encoding="utf-8")
    range_df.to_csv(RANGE_CSV, index=False, encoding="utf-8")
    ranking_df.to_csv(RANK_COMPARE_CSV, index=False, encoding="utf-8")
    scaler_df.to_csv(SCALER_CSV, index=False, encoding="utf-8")
    return vector_df, range_df, ranking_df, scaler_df


def write_report(ctx: Dict[str, Any], vector_df: pd.DataFrame, range_df: pd.DataFrame, ranking_df: pd.DataFrame, scaler_df: pd.DataFrame) -> None:
    meta = ctx["selection"]["meta"]
    raw_top = ranking_df[ranking_df["value_mode"] == "current_raw_no_minmax"].sort_values("rank").head(5)
    mode_top = (
        ranking_df.sort_values(["value_mode", "rank"])
        .groupby("value_mode")
        .head(5)[["value_mode", "rank", "source_label", "distance", "rank_changed_vs_current_raw"]]
    )
    focus = ranking_df[
        ranking_df["source_label"].isin(["B2 Item2", "B2 Item3"])
    ][["value_mode", "rank", "source_label", "distance", "current_raw_rank", "rank_changed_vs_current_raw"]].sort_values(["value_mode", "rank"])
    raw_ranges = range_df[range_df["value_mode"] == "current_raw_no_minmax"]
    normalized_ranges = range_df[range_df["value_mode"] != "current_raw_no_minmax"]
    any_changed = bool(ranking_df[ranking_df["value_mode"] != "current_raw_no_minmax"]["rank_changed_vs_current_raw"].any())

    lines = [
        "# Dataset2 KNN Distance Raw vs MinMax Audit",
        "",
        "## Direct Answer",
        "- 当前 KNN 使用原始 `sales/promo/year/month/week/day` 数值，不使用 MinMax 后的值。",
        "- 证据：`SourceSelector.select_top_k_sources` 只调用 `infer_source_selection_feature_columns`、构造 observed sequence signature、计算欧氏距离；没有调用 `normalize_features` 或 `MinMaxScaler`。",
        "- 因此当前 KNN 没有 scaler fit scope；per-item/per-entity/whole dataset/train-only/train+val/full-history 都不适用于当前 KNN。",
        "- 本报告中的 MinMax 结果只是只读审计模拟，用于回答“如果 MinMax 会怎样”。",
        "",
        "## Current KNN Metadata",
        f"- representation: {meta.get('knn_representation')}",
        f"- feature_cols: {pipe(meta.get('feature_cols', []))}",
        f"- static signature cols: {pipe(meta.get('signature_static_feature_cols', []))}",
        f"- observed_window_rows: {meta.get('observed_window_rows')}",
        f"- target_test_data_excluded: {meta.get('target_test_data_excluded')}",
        "",
        "## Actual Current Raw Input Value Ranges",
        markdown_table(raw_ranges),
        "",
        "## Audit MinMax Scaler Fit Scopes",
        markdown_table(scaler_df, max_rows=30),
        "",
        "## Normalized Input Value Ranges",
        markdown_table(normalized_ranges, max_rows=40),
        "",
        "## Top5 Ranking By Mode",
        markdown_table(mode_top),
        "",
        "## B2 Item2 vs B2 Item3 By Mode",
        markdown_table(focus),
        "",
        f"Ranking changed under at least one MinMax simulation: {any_changed}.",
        "",
        "## Output Files",
        f"- `{VECTOR_CSV}`",
        f"- `{RANGE_CSV}`",
        f"- `{RANK_COMPARE_CSV}`",
        f"- `{SCALER_CSV}`",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ctx = load_context()
    vector_df, range_df, ranking_df, scaler_df = build_outputs(ctx)
    write_report(ctx, vector_df, range_df, ranking_df, scaler_df)
    print("Dataset2 KNN raw-vs-MinMax audit complete")
    for path in [VECTOR_CSV, RANGE_CSV, RANK_COMPARE_CSV, SCALER_CSV, REPORT_MD]:
        print(f"- {path}")
    for mode, group in ranking_df.groupby("value_mode", sort=False):
        top3 = " / ".join(group.sort_values("rank").head(3)["source_label"].tolist())
        print(f"{mode} Top3: {top3}")


if __name__ == "__main__":
    main()
