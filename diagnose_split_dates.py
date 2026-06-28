"""诊断脚本：输出 Dataset1 的 raw split dates 和 effective sample dates。

用法:
    python diagnose_split_dates.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_preprocessing import (
    build_tabular_sequence,
    temporal_split_by_ratio_or_dates,
    _resolve_strict_dataset_protocol,
    _resolve_paper_split_protocol,
    _safe_int,
    _get_cfg,
    STRICT_DATASET_PROTOCOL,
)
from experiment_runner import prepare_base_data_for_experiments
from paper_reproduction_protocol import load_paper_protocol, resolve_strict_paper_mode
from scripts.run_full_paper_experiments import (
    _apply_information_sharing_filter,
    _load_config,
    _resolve_dataset_feature_cols,
    _scenario_to_bool,
)


def describe_split(
    label: str,
    df: pd.DataFrame,
) -> dict:
    """返回 split 的描述统计。"""
    if df.empty:
        return {"label": label, "row_count": 0, "min_date": None, "max_date": None}
    dates = pd.to_datetime(df["date"])
    return {
        "label": label,
        "row_count": int(len(df)),
        "min_date": str(dates.min().date()),
        "max_date": str(dates.max().date()),
        "unique_dates": int(dates.nunique()),
    }


def describe_effective_samples(
    label: str,
    df: pd.DataFrame,
    x_dates: List[str],
    y_dates: List[str],
) -> dict:
    """返回 effective sample dates 的描述统计。"""
    if not x_dates:
        return {"label": label, "row_count": 0, "x_min_date": None, "x_max_date": None, "y_min_date": None, "y_max_date": None}
    return {
        "label": label,
        "row_count": len(x_dates),
        "x_min_date": min(x_dates),
        "x_max_date": max(x_dates),
        "y_min_date": min(y_dates),
        "y_max_date": max(y_dates),
    }


def build_sequence_with_dates(
    df: pd.DataFrame,
    horizon: int = 1,
    window_size: int = 10,
    feature_cols: List[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    构建滑窗序列，同时记录每个样本的 X 最后日期和 y 日期。
    
    Returns:
        (X, y, x_end_dates, y_dates)
        x_end_dates[i] = X[i] 最后一个时间步对应的日期
        y_dates[i] = 目标日期
    """
    if feature_cols is None:
        from data_preprocessing import _infer_feature_columns
        feature_columns = [c for c in _infer_feature_columns(df) if c != "sales"]
    else:
        feature_columns = list(feature_cols)

    ordered = df.sort_values(["entity_id", "item_id", "date"]).reset_index(drop=True)
    
    x_list = []
    y_list = []
    x_end_date_list = []
    y_date_list = []

    for _, group in ordered.groupby(["entity_id", "item_id"], sort=False):
        g = group.sort_values("date").reset_index(drop=True)
        values = g[feature_columns].to_numpy(dtype=np.float32)
        sales_values = g["sales"].to_numpy(dtype=np.float32)
        dates = g["date"].to_numpy()
        n = len(g)
        max_end = n - horizon

        for end_idx in range(window_size - 1, max_end):
            start_idx = end_idx - window_size + 1
            target_idx = end_idx + horizon
            if target_idx >= n:
                continue
            x_list.append(values[start_idx : end_idx + 1])
            y_list.append(float(sales_values[target_idx]))
            x_end_date_list.append(str(pd.Timestamp(dates[end_idx]).date()))
            y_date_list.append(str(pd.Timestamp(dates[target_idx]).date()))

    if x_list:
        X = np.asarray(x_list, dtype=np.float32)
        y = np.asarray(y_list, dtype=np.float32)
    else:
        X = np.empty((0, window_size, len(feature_columns)), dtype=np.float32)
        y = np.empty((0,), dtype=np.float32)

    return X, y, x_end_date_list, y_date_list


def main():
    cfg = _load_config()
    dataset_name = "Dataset1"
    scenario = "without_information_sharing"

    print("=" * 80)
    print("Dataset1 日期切分诊断报告")
    print("=" * 80)

    # Step 1: 加载数据
    protocol = load_paper_protocol(cfg)
    strict_paper_mode = resolve_strict_paper_mode(cfg, explicit=None)

    base = prepare_base_data_for_experiments(
        dataset_name=dataset_name,
        data_path=cfg["dataset_paths"][dataset_name],
        config=cfg,
        verbose_mode="summary",
    )
    source_df_raw = base["source_df"].copy()
    target_df_raw = base["target_df"].copy()

    print(f"\n📦 原始数据概览:")
    print(f"   source_df: {len(source_df_raw)} rows, date range {source_df_raw['date'].min().date()} ~ {source_df_raw['date'].max().date()}")
    print(f"   target_df (after windowing): {len(target_df_raw)} rows, date range {target_df_raw['date'].min().date()} ~ {target_df_raw['date'].max().date()}")

    # Step 2: 应用信息共享过滤
    use_information_sharing = _scenario_to_bool(scenario)
    source_df = _apply_information_sharing_filter(
        dataset_name=dataset_name,
        source_df=source_df_raw,
        target_df=target_df_raw,
        use_information_sharing=use_information_sharing,
        strict_paper_mode=bool(strict_paper_mode),
        protocol=protocol,
        cfg=cfg,
    )
    target_df = target_df_raw

    print(f"\n📦 信息共享过滤后:")
    print(f"   source_df: {len(source_df)} rows, date range {source_df['date'].min().date()} ~ {source_df['date'].max().date()}")

    # =========================================================================
    # 任务 1: 输出 raw split dates
    # =========================================================================
    print("\n" + "=" * 80)
    print("任务 1: Raw Split Dates (进入 sliding window 前)")
    print("=" * 80)

    # Source split (ratio mode: 0.8/0.1/0.1)
    source_train_df, source_val_df, source_test_df = temporal_split_by_ratio_or_dates(source_df)

    # Target split (days mode: 15/15/180)
    target_train_df, target_val_df, target_test_df = temporal_split_by_ratio_or_dates(target_df)

    raw_splits = [
        describe_split("source_train", source_train_df),
        describe_split("source_val", source_val_df),
        describe_split("source_test", source_test_df),
        describe_split("target_train", target_train_df),
        describe_split("target_val", target_val_df),
        describe_split("target_test", target_test_df),
    ]

    print(f"\n{'Split':<20} {'Rows':>8} {'Min Date':>14} {'Max Date':>14} {'Unique':>8}")
    print("-" * 70)
    for s in raw_splits:
        print(f"{s['label']:<20} {s['row_count']:>8} {s['min_date']:>14} {s['max_date']:>14} {s['unique_dates']:>8}")

    # =========================================================================
    # 任务 1 (续): 输出 effective sample dates (sliding window 后)
    # =========================================================================
    print("\n" + "=" * 80)
    print("任务 1 (续): Effective Sample Dates (window_size=10, horizon=1 后)")
    print("=" * 80)

    feature_cols = _resolve_dataset_feature_cols(
        dataset_name=dataset_name,
        source_df=source_df,
        target_df=target_df,
        cfg=cfg,
    )

    effective_splits = []
    for label, split_df in [
        ("source_train", source_train_df),
        ("source_val", source_val_df),
        ("source_test", source_test_df),
        ("target_train", target_train_df),
        ("target_val", target_val_df),
        ("target_test", target_test_df),
    ]:
        X, y, x_dates, y_dates = build_sequence_with_dates(
            split_df, horizon=1, window_size=10, feature_cols=feature_cols
        )
        info = describe_effective_samples(label, split_df, x_dates, y_dates)
        info["x_min_first"] = str(pd.Timestamp(split_df["date"].min()).date()) if not split_df.empty else "N/A"
        info["x_max_last"] = str(pd.Timestamp(split_df["date"].max()).date()) if not split_df.empty else "N/A"
        effective_splits.append(info)

    print(f"\n{'Split':<20} {'Samples':>8} {'X Min Date':>14} {'X Max Date':>14} {'Y Min Date':>14} {'Y Max Date':>14}")
    print("-" * 90)
    for s in effective_splits:
        print(f"{s['label']:<20} {s['row_count']:>8} {s.get('x_min_date','N/A'):>14} {s.get('x_max_date','N/A'):>14} {s.get('y_min_date','N/A'):>14} {s.get('y_max_date','N/A'):>14}")

    # =========================================================================
    # 任务 2: 与论文 Table 2 对比
    # =========================================================================
    print("\n" + "=" * 80)
    print("任务 2: 与论文 Table 2 对比")
    print("=" * 80)

    paper_raw = {
        "source_train": ("2013-01-01", "2016-12-31"),
        "source_val":   ("2017-01-01", "2017-06-30"),
        "source_test":  ("2017-07-01", "2017-12-31"),
        "target_train": ("2017-06-01", "2017-06-15"),
        "target_val":   ("2017-06-16", "2017-06-30"),
        "target_test":  ("2017-07-01", "2017-12-31"),
    }

    print(f"\n{'Split':<20} {'Current Min':>14} {'Current Max':>14} {'Paper Min':>14} {'Paper Max':>14} {'Match?':>8}")
    print("-" * 85)
    all_raw_match = True
    for s in raw_splits:
        paper_range = paper_raw.get(s["label"], ("?", "?"))
        match = (s["min_date"] == paper_range[0] and s["max_date"] == paper_range[1])
        if not match:
            all_raw_match = False
        status = "✅" if match else "❌"
        print(f"{s['label']:<20} {s['min_date']:>14} {s['max_date']:>14} {paper_range[0]:>14} {paper_range[1]:>14} {status:>8}")

    # =========================================================================
    # 任务 3/4: 诊断结论
    # =========================================================================
    print("\n" + "=" * 80)
    print("任务 3/4: 诊断结论")
    print("=" * 80)

    if all_raw_match:
        print("\n✅ Raw split dates 完全符合论文 Table 2。")
        print("   偏移来源于 sliding-window 后的 effective sample dates（正常行为）。")
        print("   不需要修改训练切分逻辑。")
    else:
        print("\n❌ Raw split dates 不符合论文 Table 2。")
        print("   需要在 strict paper mode 下为 Dataset1 增加绝对日期切分。")
        
        # 详细分析每个不匹配的 split
        print("\n详细差异分析:")
        for s in raw_splits:
            paper_range = paper_raw.get(s["label"], ("?", "?"))
            match = (s["min_date"] == paper_range[0] and s["max_date"] == paper_range[1])
            if not match:
                print(f"\n  {s['label']}:")
                print(f"    当前: {s['min_date']} ~ {s['max_date']}")
                print(f"    论文: {paper_range[0]} ~ {paper_range[1]}")
                # 对于 source，ratio 模式可能产生不同结果
                if "source" in s["label"]:
                    print(f"    原因: source 使用 ratio 切分 (0.8/0.1/0.1)，受数据日期范围影响")
                if "target" in s["label"]:
                    print(f"    原因: target 使用相对窗口倒推 (max_date - 210 days)，非绝对日期")

    # 额外信息：target 窗口是如何确定的
    print(f"\n📋 Target 窗口计算细节:")
    print(f"   target_df max date: {target_df_raw['date'].max().date()}")
    total_days = 15 + 15 + 180  # 210
    print(f"   total_days = train(15) + val(15) + test(180) = {total_days}")
    print(f"   target_min_date = max_date - {total_days - 1} days = {(target_df_raw['date'].max() - pd.Timedelta(days=total_days-1)).date()}")
    
    print(f"\n📋 Source 切分细节:")
    print(f"   source 全量日期范围: {source_df_raw['date'].min().date()} ~ {source_df_raw['date'].max().date()}")
    unique_source_dates = sorted(source_df_raw["date"].drop_duplicates())
    n_dates = len(unique_source_dates)
    train_end_idx = max(1, int(n_dates * 0.8))
    val_end_idx = max(train_end_idx + 1, int(n_dates * (0.8 + 0.1)))
    print(f"   source unique dates: {n_dates}")
    print(f"   ratio split: 0.8/0.1/0.1 → train_end_idx={train_end_idx}, val_end_idx={val_end_idx}")
    print(f"   source_train last date: {unique_source_dates[train_end_idx - 1].date()}")
    print(f"   source_val last date: {unique_source_dates[val_end_idx - 1].date() if val_end_idx <= n_dates else unique_source_dates[-1].date()}")

    print("\n" + "=" * 80)
    print("诊断完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
