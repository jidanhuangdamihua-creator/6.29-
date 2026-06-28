#!/usr/bin/env python3
"""Dataset5 Favorita cold-start profile scanner."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


OBSERVED_WINDOWS = [7, 14, 30]
HORIZON_WINDOWS = [7, 14, 28]


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def df_to_markdown(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df.empty:
        return "_无数据_"
    view = df.head(max_rows)
    cols = list(view.columns)
    lines = [
        "| " + " | ".join(str(c) for c in cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in view.iterrows():
        vals = [_format_markdown_value(row[c]) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _format_markdown_value(value: object) -> str:
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(v) for v in value)
    if isinstance(value, dict):
        return str(value)
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def identify_file_roles(dataset_root: Path) -> Dict[str, object]:
    names = {p.name: p for p in dataset_root.iterdir() if p.is_file()}
    m5_sales = [n for n in names if n.startswith("sales_") and n.endswith(".csv")]
    m5_calendar = "calendar.csv" in names
    m5_prices = "sell_prices.csv" in names

    if m5_sales:
        sales_file = sorted(m5_sales)[0]
    elif "train.csv" in names:
        sales_file = "train.csv"
    else:
        sales_file = ""

    calendar_files: List[str] = []
    if m5_calendar:
        calendar_files.append("calendar.csv")
    else:
        for name in ["holidays_events.csv", "oil.csv", "transactions.csv"]:
            if name in names:
                calendar_files.append(name)

    prices_file = "sell_prices.csv" if m5_prices else ""
    dataset_family = "M5" if (m5_sales and m5_calendar and m5_prices) else "Favorita"
    return {
        "dataset_family": dataset_family,
        "m5_compatible_files": bool(m5_sales and m5_calendar and m5_prices),
        "sales_file": sales_file,
        "calendar_files": calendar_files,
        "prices_file": prices_file,
        "items_file": "items.csv" if "items.csv" in names else "",
        "stores_file": "stores.csv" if "stores.csv" in names else "",
    }


def _task_id(observed: int, horizon: int) -> str:
    return f"obs{observed}_h{horizon}"


def add_cold_start_flags(
    entity_stats: pd.DataFrame,
    observed_windows: Iterable[int] = OBSERVED_WINDOWS,
    horizon_windows: Iterable[int] = HORIZON_WINDOWS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    stats = entity_stats.copy()
    rows = []
    for observed in observed_windows:
        for horizon in horizon_windows:
            task_id = _task_id(observed, horizon)
            required = observed + horizon
            col = f"eligible_{task_id}"
            stats[col] = (
                (stats["span_days"] >= required)
                & (stats["valid_sales_days"] >= required)
                & (stats["date_density"] > 0)
            )
            count = int(stats[col].sum())
            rows.append({
                "task_id": task_id,
                "observed_days": observed,
                "horizon_days": horizon,
                "required_days": required,
                "eligible_entity_count": count,
                "eligible_entity_ratio": float(count / max(len(stats), 1)),
            })
    return stats, pd.DataFrame(rows)


def build_source_target_candidates(
    entity_stats: pd.DataFrame,
    task_ids: Iterable[str],
    top_targets: int = 50,
    top_sources: int = 20,
) -> pd.DataFrame:
    rows = []
    for task_id in task_ids:
        eligible_col = f"eligible_{task_id}"
        if eligible_col not in entity_stats.columns:
            continue
        eligible = entity_stats[entity_stats[eligible_col]].copy()
        if eligible.empty:
            continue
        for col, default in {
            "quality_score": 0.0,
            "valid_sales_days": 0,
            "date_density": 0.0,
            "zero_sales_ratio": 0.0,
            "span_days": 0,
        }.items():
            if col not in eligible.columns:
                eligible[col] = default
        eligible = eligible.sort_values(
            ["quality_score", "valid_sales_days", "date_density", "zero_sales_ratio"],
            ascending=[False, False, False, True],
        )
        for _, target in eligible.head(top_targets).iterrows():
            others = eligible[eligible["entity_id"] != target["entity_id"]]
            same_category = others[others["family"] == target["family"]]
            same_department = others[others["class"] == target["class"]]
            same_store = others[others["store_id"] == target["store_id"]]
            pools = {
                "same_category": same_category,
                "same_department": same_department,
                "same_store": same_store,
                "global": others,
            }
            row = {
                "task_id": task_id,
                "candidate_target_entity": target["entity_id"],
                "candidate": True,
                "target_store_id": int(target["store_id"]),
                "target_item_id": int(target["item_id"]),
                "target_family": target.get("family", ""),
                "target_class": target.get("class", ""),
                "target_span_days": int(target["span_days"]),
                "target_valid_sales_days": int(target["valid_sales_days"]),
                "target_date_density": float(target["date_density"]),
                "target_zero_sales_ratio": float(target["zero_sales_ratio"]),
            }
            for pool_name, pool_df in pools.items():
                row[f"{pool_name}_source_count"] = int(len(pool_df))
                row[f"{pool_name}_source_entities"] = "|".join(pool_df["entity_id"].head(top_sources).astype(str))
            row["candidate_reason"] = (
                "Cold-start/short-history candidate generated from full sales profile; "
                "source pools are candidates and require manual confirmation."
            )
            rows.append(row)
    return pd.DataFrame(rows)


def scan_sales_train(train_path: Path, chunksize: int = 5_000_000) -> Tuple[pd.DataFrame, Dict[str, object]]:
    parts = []
    all_dates = set()
    item_ids = set()
    store_ids = set()
    total_rows = 0

    usecols = ["date", "store_nbr", "item_nbr", "unit_sales"]
    dtypes = {"store_nbr": "int32", "item_nbr": "int32", "unit_sales": "float32"}
    for chunk in pd.read_csv(train_path, usecols=usecols, dtype=dtypes, chunksize=chunksize):
        total_rows += len(chunk)
        all_dates.update(chunk["date"].dropna().unique().tolist())
        item_ids.update(chunk["item_nbr"].dropna().unique().tolist())
        store_ids.update(chunk["store_nbr"].dropna().unique().tolist())
        chunk["_valid_sales"] = chunk["unit_sales"].notna().astype("int32")
        chunk["_zero_sales"] = ((chunk["unit_sales"] == 0) & chunk["unit_sales"].notna()).astype("int32")
        chunk["_missing_sales"] = chunk["unit_sales"].isna().astype("int32")
        grouped = chunk.groupby(["store_nbr", "item_nbr"], sort=False).agg(
            row_count=("date", "size"),
            min_date=("date", "min"),
            max_date=("date", "max"),
            valid_sales_days=("_valid_sales", "sum"),
            zero_sales_count=("_zero_sales", "sum"),
            sales_missing_count=("_missing_sales", "sum"),
        ).reset_index()
        parts.append(grouped)

    combined = pd.concat(parts, ignore_index=True)
    stats = combined.groupby(["store_nbr", "item_nbr"], sort=False).agg(
        row_count=("row_count", "sum"),
        min_date=("min_date", "min"),
        max_date=("max_date", "max"),
        valid_sales_days=("valid_sales_days", "sum"),
        zero_sales_count=("zero_sales_count", "sum"),
        sales_missing_count=("sales_missing_count", "sum"),
    ).reset_index()
    meta = {
        "total_rows": int(total_rows),
        "item_count": int(len(item_ids)),
        "store_count": int(len(store_ids)),
        "min_date": min(all_dates) if all_dates else "",
        "max_date": max(all_dates) if all_dates else "",
        "unique_dates": int(len(all_dates)),
    }
    return stats, meta


def finalize_entity_stats(stats: pd.DataFrame, items: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    out = stats.rename(columns={"store_nbr": "store_id", "item_nbr": "item_id"}).copy()
    out["min_date"] = pd.to_datetime(out["min_date"])
    out["max_date"] = pd.to_datetime(out["max_date"])
    out["span_days"] = (out["max_date"] - out["min_date"]).dt.days + 1
    out["unique_dates"] = out["row_count"].astype(int)
    out["sales_missing_rate"] = out["sales_missing_count"] / out["row_count"].replace(0, np.nan)
    out["zero_sales_ratio"] = out["zero_sales_count"] / out["valid_sales_days"].replace(0, np.nan)
    out["zero_sales_ratio"] = out["zero_sales_ratio"].fillna(0.0)
    out["date_density"] = out["unique_dates"] / out["span_days"].replace(0, np.nan)
    out["entity_id"] = "store_id=" + out["store_id"].astype(str) + "|item_id=" + out["item_id"].astype(str)

    item_meta = items.rename(columns={"item_nbr": "item_id"})
    store_meta = stores.rename(columns={"store_nbr": "store_id"})
    out = out.merge(item_meta, on="item_id", how="left")
    out = out.merge(store_meta, on="store_id", how="left")
    out["quality_score"] = (
        np.minimum(out["valid_sales_days"] / 58.0, 1.0) * 0.40
        + out["date_density"].fillna(0) * 0.35
        + (1.0 - out["sales_missing_rate"].fillna(1.0)) * 0.15
        + (1.0 - out["zero_sales_ratio"].clip(upper=1).fillna(1.0)) * 0.10
    )
    ordered_cols = [
        "entity_id", "store_id", "item_id", "family", "class", "perishable",
        "city", "state", "type", "cluster", "row_count", "min_date", "max_date",
        "span_days", "unique_dates", "valid_sales_days", "sales_missing_count",
        "sales_missing_rate", "zero_sales_count", "zero_sales_ratio", "date_density",
        "quality_score",
    ]
    remaining = [c for c in out.columns if c not in ordered_cols]
    return out[ordered_cols + remaining].sort_values(
        ["quality_score", "valid_sales_days", "date_density"],
        ascending=[False, False, False],
    )


def build_summary(
    roles: Dict[str, object],
    meta: Dict[str, object],
    entity_stats: pd.DataFrame,
    task_summary: pd.DataFrame,
) -> pd.DataFrame:
    summary = dict(meta)
    summary.update({
        "dataset_id": "Dataset5",
        "dataset_label_requested": "Dataset5_Favorita",
        "dataset_name_final": "Dataset5_Favorita",
        "detected_dataset_family": roles["dataset_family"],
        "standard_m5_structure": bool(roles["m5_compatible_files"]),
        "m5_compatible_files": roles["m5_compatible_files"],
        "cold_start_construction": "Yes",
        "cold_start_protocol_type": "Favorita short-history cold-start",
        "sales_file": roles["sales_file"],
        "calendar_files": "|".join(roles["calendar_files"]),
        "prices_file": roles["prices_file"],
        "entity_unit": "item_id + store_id",
        "entity_count": int(len(entity_stats)),
        "span_days_min": float(entity_stats["span_days"].min()),
        "span_days_median": float(entity_stats["span_days"].median()),
        "span_days_max": float(entity_stats["span_days"].max()),
        "valid_sales_days_min": float(entity_stats["valid_sales_days"].min()),
        "valid_sales_days_median": float(entity_stats["valid_sales_days"].median()),
        "valid_sales_days_max": float(entity_stats["valid_sales_days"].max()),
        "sales_missing_rate": float(entity_stats["sales_missing_count"].sum() / max(entity_stats["row_count"].sum(), 1)),
        "zero_sales_ratio": float(entity_stats["zero_sales_count"].sum() / max(entity_stats["valid_sales_days"].sum(), 1)),
        "date_density_median": float(entity_stats["date_density"].median()),
    })
    for _, row in task_summary.iterrows():
        summary[f"{row['task_id']}_eligible_count"] = int(row["eligible_entity_count"])
        summary[f"{row['task_id']}_eligible_ratio"] = float(row["eligible_entity_ratio"])
    return pd.DataFrame([summary])


def write_reports(
    output_dir: Path,
    roles: Dict[str, object],
    summary_df: pd.DataFrame,
    entity_stats: pd.DataFrame,
    task_summary: pd.DataFrame,
    candidates: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(summary_df, output_dir / "dataset5_favorita_profile_summary.csv")
    write_csv(entity_stats, output_dir / "dataset5_favorita_entity_stats.csv")
    write_csv(candidates, output_dir / "dataset5_favorita_source_target_candidates.csv")

    warnings = []
    if not roles["m5_compatible_files"]:
        warnings.append(
            "给定路径识别为 Favorita；本报告按 Favorita 全量数据构造 Favorita cold-start 任务。"
        )
    if not roles["prices_file"]:
        warnings.append("未发现 prices 文件；source/target 构造未使用价格特征。")

    md = [
        "# Dataset5_Favorita Full Profile Summary",
        "",
        "Dataset5 is identified as Favorita. Historical M5-related labels are deprecated naming errors and must not be used as formal evidence.",
        "",
        "## 文件角色识别",
        "",
        df_to_markdown(pd.DataFrame([roles])),
        "",
        "## 关键警告",
    ]
    md.extend([f"- {w}" for w in warnings] or ["- 无"])
    md.extend([
        "",
        "## 全局统计",
        "",
        df_to_markdown(summary_df.T.reset_index().rename(columns={"index": "metric", 0: "value"}), max_rows=80),
        "",
        "## Cold-start 窗口统计",
        "",
        df_to_markdown(task_summary),
        "",
        "## 实体统计样例",
        "",
        df_to_markdown(entity_stats.head(20)),
        "",
        "## Source/Target 候选样例",
        "",
        df_to_markdown(candidates.head(30)),
        "",
        "完整实体明细见 `dataset5_favorita_entity_stats.csv`；完整候选划分见 `dataset5_favorita_source_target_candidates.csv`。",
    ])
    (output_dir / "dataset5_favorita_profile_summary.md").write_text("\n".join(md), encoding="utf-8")

    design = [
        "# Dataset5_Favorita Cold-start Task Design",
        "",
        "本研究需要构造 cold-start / short-history 任务，而不是普通时间序列预测任务。",
        "",
        "## 数据集判定",
        "",
        "- Dataset5 is identified as Favorita. Historical M5-related labels are deprecated naming errors and must not be used as formal evidence.",
        "- Favorita cold-start construction uses short-history item-store windows.",
        "- 实体单位：`item_id + store_id`，对应 Favorita 的 `item_nbr + store_nbr`。",
        "",
        "## 任务窗口",
        "",
        df_to_markdown(task_summary),
        "",
        "## Source pool 设计",
        "",
        "- same category：同 `family` 的其他实体。",
        "- same department：同 `class` 的其他实体。",
        "- same store：同 `store_id` 的其他实体。",
        "- global：所有满足对应 cold-start 窗口的其他实体。",
        "",
        "## 候选划分原则",
        "",
        "- target entity 必须满足 observed + horizon 的最短历史要求。",
        "- source entity 必须满足同一 task 的最短历史要求。",
        "- 候选划分只作为 candidate，需要人工确认业务含义和论文设定一致性。",
    ]
    (output_dir / "dataset5_favorita_cold_start_task_design.md").write_text("\n".join(design), encoding="utf-8")


def run(dataset_root: Path, output_root: Path, chunksize: int) -> Path:
    roles = identify_file_roles(dataset_root)
    if not roles["sales_file"]:
        raise FileNotFoundError(f"No sales/train file found under {dataset_root}")
    train_path = dataset_root / str(roles["sales_file"])
    items_path = dataset_root / str(roles["items_file"])
    stores_path = dataset_root / str(roles["stores_file"])
    if not items_path.is_file() or not stores_path.is_file():
        raise FileNotFoundError("Dataset5 requires items.csv and stores.csv for source pool design.")

    items = pd.read_csv(items_path)
    stores = pd.read_csv(stores_path)
    raw_stats, meta = scan_sales_train(train_path, chunksize=chunksize)
    entity_stats = finalize_entity_stats(raw_stats, items, stores)
    entity_stats, task_summary = add_cold_start_flags(entity_stats, OBSERVED_WINDOWS, HORIZON_WINDOWS)
    task_ids = task_summary["task_id"].tolist()
    candidates = build_source_target_candidates(entity_stats, task_ids)
    summary_df = build_summary(roles, meta, entity_stats, task_summary)

    run_dir = output_root / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S") / "Dataset5"
    write_reports(run_dir, roles, summary_df, entity_stats, task_summary, candidates)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan Dataset5 Favorita as a Favorita cold-start profile.")
    parser.add_argument("--dataset-root", default="/Users/ming/Desktop/复现实验/Favorita")
    parser.add_argument("--output-root", default="outputs/dataset_profiles")
    parser.add_argument("--chunksize", type=int, default=5_000_000)
    args = parser.parse_args()

    run_dir = run(Path(args.dataset_root), Path(args.output_root), args.chunksize)
    print(f"Dataset5 Favorita cold-start profile saved to: {run_dir}")


if __name__ == "__main__":
    main()
